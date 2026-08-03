"""
Shared overhead tracking for benchmark runs.

Tracks per-batch forward/backward latency, peak CUDA memory, parameter counts,
and a lightweight FLOPs estimate derived from registered module hooks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def _as_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _safe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _is_activation_module(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return any(keyword in name for keyword in ("alpha", "golu", "gelu", "swish", "prelu", "relu")) and any(
        parameter.requires_grad for parameter in module.parameters(recurse=False)
    )


def _activation_flop_cost(module: nn.Module) -> int:
    name = module.__class__.__name__.lower()
    if any(keyword in name for keyword in ("golu", "gelu", "swish")):
        return 8
    if "prelu" in name:
        return 2
    if "relu" in name:
        return 1
    return 4


@dataclass
class OverheadTracker:
    task_name: str
    activation_name: str
    warmup_steps: int = 10
    model: Optional[nn.Module] = None
    device: Optional[torch.device | str] = None
    output_root: str | Path = "outputs/overhead"
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    forward_times: list[float] = field(default_factory=list)
    backward_times: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    _forward_start: Optional[float] = None
    _backward_start: Optional[float] = None
    _current_batch_is_warmup: bool = False
    _completed_batches: int = 0
    _peak_memory_reset_after_warmup: bool = False
    _track_flops: bool = False
    _current_forward_flops: int = 0
    _total_forward_flops: int = 0
    _hook_handles: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.warmup_steps = max(int(self.warmup_steps), 0)
        self.device = _as_device(self.device)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.model is not None:
            self.attach_model(self.model)

    @property
    def forward_ms(self) -> list[float]:
        return self.forward_times

    @property
    def backward_ms(self) -> list[float]:
        return self.backward_times

    def attach_model(self, model: nn.Module) -> "OverheadTracker":
        self.model = model
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        if self.model is None:
            return self

        for module in self.model.modules():
            if len(list(module.children())) > 0:
                continue
            self._hook_handles.append(module.register_forward_hook(self._flop_hook))

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def _flop_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not self._track_flops:
            return

        output_tensor = _first_tensor(output)
        if output_tensor is None:
            return

        output_elements = int(output_tensor.numel())

        if isinstance(module, nn.Conv2d):
            batch_size, out_channels, out_h, out_w = output_tensor.shape
            kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            groups = max(int(module.groups), 1)
            kernel_ops = (module.in_channels // groups) * kernel_h * kernel_w * 2
            bias_ops = 1 if module.bias is not None else 0
            self._current_forward_flops += batch_size * out_channels * out_h * out_w * (kernel_ops + bias_ops)
            return

        if isinstance(module, nn.Linear):
            leading_elements = max(output_elements // max(module.out_features, 1), 1)
            bias_ops = 1 if module.bias is not None else 0
            self._current_forward_flops += leading_elements * module.out_features * (module.in_features * 2 + bias_ops)
            return

        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self._current_forward_flops += output_elements * 4
            return

        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU)):
            self._current_forward_flops += output_elements * _activation_flop_cost(module)
            return

        if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            self._current_forward_flops += output_elements * 2
            return

        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            self._current_forward_flops += output_elements * 4
            return

        if _is_activation_module(module):
            self._current_forward_flops += output_elements * _activation_flop_cost(module)

    def start_forward(self) -> "OverheadTracker":
        _safe_sync(self.device)
        self._current_batch_is_warmup = self._completed_batches < self.warmup_steps
        self._current_forward_flops = 0
        self._track_flops = True
        self._forward_start = time.perf_counter()
        return self

    def end_forward(self, batch_size: Optional[int] = None) -> float:
        _safe_sync(self.device)
        if self._forward_start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._forward_start) * 1000.0
        self._total_forward_flops += int(self._current_forward_flops)
        self._track_flops = False
        self._forward_start = None
        if not self._current_batch_is_warmup:
            self.forward_times.append(float(elapsed_ms))
            if batch_size is not None:
                self.batch_sizes.append(int(batch_size))
        return float(elapsed_ms)

    def start_backward(self) -> "OverheadTracker":
        _safe_sync(self.device)
        self._backward_start = time.perf_counter()
        return self

    def end_backward(self) -> float:
        _safe_sync(self.device)
        if self._backward_start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._backward_start) * 1000.0
        if not self._current_batch_is_warmup:
            self.backward_times.append(float(elapsed_ms))
        self._backward_start = None
        self._completed_batches += 1
        if (
            not self._peak_memory_reset_after_warmup
            and torch.cuda.is_available()
            and self.device.type == "cuda"
            and self._completed_batches == self.warmup_steps
        ):
            torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_memory_reset_after_warmup = True
        return float(elapsed_ms)

    def _peak_memory_mb(self) -> Optional[float]:
        if torch.cuda.is_available() and self.device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        return None

    def _parameter_stats(self) -> Dict[str, int]:
        total_params = 0
        trainable_params = 0
        activation_params = 0

        if self.model is not None:
            total_params = sum(parameter.numel() for parameter in self.model.parameters())
            trainable_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            activation_params = sum(
                parameter.numel()
                for module in self.model.modules()
                if _is_activation_module(module)
                for parameter in module.parameters(recurse=False)
            )

        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "activation_params": int(activation_params),
        }

    def summary(self) -> Dict[str, Any]:
        parameter_stats = self._parameter_stats()
        forward_mean = mean(self.forward_times) if self.forward_times else 0.0
        backward_mean = mean(self.backward_times) if self.backward_times else 0.0
        forward_std = pstdev(self.forward_times) if len(self.forward_times) > 1 else 0.0
        backward_std = pstdev(self.backward_times) if len(self.backward_times) > 1 else 0.0
        peak_memory_mb = self._peak_memory_mb()
        batch_count = max(len(self.forward_times), len(self.backward_times))
        total_forward_flops = int(self._total_forward_flops)
        estimated_backward_flops = int(total_forward_flops * 2)

        return {
            "task_name": self.task_name,
            "activation_name": self.activation_name,
            "warmup_steps": int(self.warmup_steps),
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "batch_count": int(batch_count),
            "forward_ms": {
                "mean": float(forward_mean),
                "std": float(forward_std),
                "values": [float(value) for value in self.forward_times],
            },
            "backward_ms": {
                "mean": float(backward_mean),
                "std": float(backward_std),
                "values": [float(value) for value in self.backward_times],
            },
            "forward_latency_ms": float(forward_mean),
            "backward_latency_ms": float(backward_mean),
            "forward_times": [float(value) for value in self.forward_times],
            "backward_times": [float(value) for value in self.backward_times],
            "peak_cuda_memory_mb": peak_memory_mb,
            "estimated_forward_flops": total_forward_flops,
            "estimated_backward_flops": estimated_backward_flops,
            "estimated_total_flops": int(total_forward_flops + estimated_backward_flops),
            **parameter_stats,
            "batch_sizes": [int(size) for size in self.batch_sizes],
            "notes": {
                "flop_method": "module-hook estimate; backward FLOPs approximated as 2x forward FLOPs",
                "memory_metric": "torch.cuda.max_memory_allocated" if peak_memory_mb is not None else "cpu",
            },
            **self.metadata,
        }

    def save(self, output_root: str | Path | None = None) -> Dict[str, Any]:
        payload = self.summary()
        target_root = Path(output_root or self.output_root)
        target_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        seed_part = f"_seed-{payload.get('seed')}" if payload.get("seed") is not None else ""
        run_id_part = f"_{self.run_id}" if self.run_id else ""
        filename = f"{timestamp}_{self.task_name}_{self.activation_name}{seed_part}{run_id_part}.json"
        path = target_root / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, sort_keys=True)
        payload["file_path"] = str(path)
        return payload