"""Standalone complexity profiler for the benchmark models.

This script measures, for each core benchmark architecture and activation function:
- exact parameter counts
- parameter deltas versus the ReLU baseline
- forward-pass FLOPs for a single representative batch
- peak CUDA memory from a dummy forward + backward pass

The output is written to artifacts/complexity_analysis.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_classification import ResNet18 as ClassificationResNet18
from experiments.run_detection import VOC_CLASSES, build_detection_model
from experiments.run_language_model import MiniGPT, build_language_model_dataloaders
from experiments.run_segmentation import UNet, VOC_SEG_CLASSES
from utils.overhead_tracker import OverheadTracker

try:
    from torch.utils.flop_counter import FlopCounterMode
except Exception:  # pragma: no cover - fallback for older runtimes
    FlopCounterMode = None


ACTIVATIONS = [
    "relu",
    "gelu",
    "swish",
    "prelu",
    "pgelu",
    "adaptive_swish",
    "golu_static",
    "alpha_golu",
]

TASKS = [
    "classification",
    "detection",
    "segmentation",
    "language_model",
]

MB = 1024.0 ** 2


@dataclass(frozen=True)
class TaskSpec:
    name: str
    batch_size: int
    batch_shape: dict[str, Any]


TASK_SPECS: dict[str, TaskSpec] = {
    "classification": TaskSpec(
        name="classification",
        batch_size=1,
        batch_shape={"images": [1, 3, 32, 32], "labels": [1], "num_classes": 10},
    ),
    "detection": TaskSpec(
        name="detection",
        batch_size=1,
        batch_shape={"images": [1, 3, 320, 320], "boxes": [[1, 4]], "num_classes": len(VOC_CLASSES) + 1},
    ),
    "segmentation": TaskSpec(
        name="segmentation",
        batch_size=1,
        batch_shape={"images": [1, 3, 256, 256], "masks": [1, 256, 256], "num_classes": VOC_SEG_CLASSES},
    ),
    "language_model": TaskSpec(
        name="language_model",
        batch_size=1,
        batch_shape={"tokens": [1, 64]},
    ),
}


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


@lru_cache(maxsize=1)
def resolve_language_model_vocab_size(data_root: str) -> tuple[int, str]:
    try:
        _, _, vocab = build_language_model_dataloaders(
            dataset_name="wikitext2",
            root=data_root,
            block_size=64,
            batch_size=1,
            seed=0,
        )
        return len(vocab), "dataset_vocab"
    except Exception as exc:  # pragma: no cover - network/corpus fallback
        return 1000, f"fallback_fixed_1000 ({type(exc).__name__}: {exc})"


def build_model(task_name: str, activation: str, device: torch.device, lm_vocab_size: int) -> nn.Module:
    if task_name == "classification":
        return ClassificationResNet18(num_classes=10, act_type=activation, alpha_layout="channel").to(device)
    if task_name == "detection":
        return build_detection_model(act_type=activation, num_classes=len(VOC_CLASSES) + 1).to(device)
    if task_name == "segmentation":
        return UNet(act_type=activation, num_classes=VOC_SEG_CLASSES).to(device)
    if task_name == "language_model":
        return MiniGPT(vocab_size=lm_vocab_size, act_type=activation, max_seq_len=64).to(device)
    raise ValueError(f"Unknown task: {task_name}")


def build_sample(task_name: str, device: torch.device, lm_vocab_size: int) -> dict[str, Any]:
    if task_name == "classification":
        return {
            "images": torch.randn(1, 3, 32, 32, device=device),
            "labels": torch.randint(0, 10, (1,), device=device),
        }
    if task_name == "detection":
        return {
            "images": [torch.randn(3, 320, 320, device=device)],
            "targets": [
                {
                    "boxes": torch.tensor([[40.0, 40.0, 200.0, 200.0]], device=device),
                    "labels": torch.tensor([1], device=device),
                }
            ],
        }
    if task_name == "segmentation":
        return {
            "images": torch.randn(1, 3, 256, 256, device=device),
            "masks": torch.randint(0, VOC_SEG_CLASSES, (1, 256, 256), device=device),
        }
    if task_name == "language_model":
        return {
            "tokens": torch.randint(0, lm_vocab_size, (1, 64), device=device),
        }
    raise ValueError(f"Unknown task: {task_name}")


def forward_only(task_name: str, model: nn.Module, sample: dict[str, Any]) -> torch.Tensor | list[Any] | dict[str, Any]:
    if task_name == "classification":
        return model(sample["images"])
    if task_name == "detection":
        return model(sample["images"])
    if task_name == "segmentation":
        return model(sample["images"])
    if task_name == "language_model":
        return model(sample["tokens"])
    raise ValueError(f"Unknown task: {task_name}")


def training_loss(task_name: str, model: nn.Module, sample: dict[str, Any], lm_vocab_size: int) -> torch.Tensor:
    if task_name == "classification":
        logits = model(sample["images"])
        return nn.CrossEntropyLoss()(logits.float(), sample["labels"])
    if task_name == "detection":
        loss_dict = model(sample["images"], sample["targets"])
        return sum(loss_value.float() for loss_value in loss_dict.values())
    if task_name == "segmentation":
        logits = model(sample["images"])
        return nn.CrossEntropyLoss(ignore_index=255)(logits.float(), sample["masks"])
    if task_name == "language_model":
        tokens = sample["tokens"]
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(inputs)
        return nn.CrossEntropyLoss()(logits.float().reshape(-1, lm_vocab_size), targets.reshape(-1))
    raise ValueError(f"Unknown task: {task_name}")


def measure_forward_flops(task_name: str, activation: str, device: torch.device, sample: dict[str, Any], lm_vocab_size: int) -> tuple[int, str]:
    model = build_model(task_name, activation, device, lm_vocab_size)
    model.eval()

    try:
        if FlopCounterMode is None:
            raise RuntimeError("torch.utils.flop_counter.FlopCounterMode is unavailable")

        with torch.no_grad():
            with FlopCounterMode(display=False) as mode:
                _ = forward_only(task_name, model, sample)
        return int(mode.get_total_flops()), "torch.utils.flop_counter.FlopCounterMode"
    except Exception as exc:  # pragma: no cover - fallback path
        tracker = OverheadTracker(
            task_name=task_name,
            activation_name=activation,
            model=model,
            device=device,
            enabled=True,
        )
        with torch.no_grad():
            tracker.start_forward()
            _ = forward_only(task_name, model, sample)
            tracker.end_forward()
        summary = tracker.summary()
        return int(summary.get("estimated_forward_flops", 0)), f"overhead_tracker_fallback ({type(exc).__name__})"


def measure_peak_cuda_memory(task_name: str, activation: str, device: torch.device, sample: dict[str, Any], lm_vocab_size: int) -> float | None:
    if device.type != "cuda":
        return None

    model = build_model(task_name, activation, device, lm_vocab_size)
    model.train()
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    loss = training_loss(task_name, model, sample, lm_vocab_size)
    loss.backward()
    torch.cuda.synchronize(device)

    return float(torch.cuda.max_memory_allocated(device) / MB)


def profile_task_activation(task_name: str, activation: str, device: torch.device, lm_vocab_size: int, sample: dict[str, Any]) -> dict[str, Any]:
    params_model = build_model(task_name, activation, device, lm_vocab_size)
    param_stats = count_parameters(params_model)
    del params_model
    gc.collect()

    forward_flops, flop_backend = measure_forward_flops(task_name, activation, device, sample, lm_vocab_size)
    del sample
    gc.collect()

    memory_sample = build_sample(task_name, device, lm_vocab_size)
    peak_cuda_memory_mb = measure_peak_cuda_memory(task_name, activation, device, memory_sample, lm_vocab_size)
    del memory_sample
    gc.collect()

    return {
        "task": task_name,
        "activation": activation,
        **param_stats,
        "forward_flops": int(forward_flops),
        "flops_backend": flop_backend,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "batch_spec": TASK_SPECS[task_name].batch_shape,
    }


def build_complexity_analysis(tasks: list[str], activations: list[str], device: torch.device, data_root: str) -> dict[str, Any]:
    if "language_model" in tasks:
        lm_vocab_size, lm_vocab_source = resolve_language_model_vocab_size(data_root)
    else:
        lm_vocab_size, lm_vocab_source = 1000, "not_requested"
    analysis: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "flop_counter_backend": "torch.utils.flop_counter.FlopCounterMode" if FlopCounterMode is not None else "overhead_tracker_fallback",
        "language_model_vocab": {
            "size": lm_vocab_size,
            "source": lm_vocab_source,
        },
        "tasks": {},
        "table": [],
    }

    for task_name in tasks:
        if task_name not in TASK_SPECS:
            raise ValueError(f"Unknown task: {task_name}")

        task_rows: list[dict[str, Any]] = []
        baseline_row: dict[str, Any] | None = None
        shared_sample = build_sample(task_name, device, lm_vocab_size)

        for activation in activations:
            row = profile_task_activation(task_name, activation, device, lm_vocab_size, shared_sample)
            if activation == "relu":
                baseline_row = row
            task_rows.append(row)

        if baseline_row is None:
            raise RuntimeError(f"Task '{task_name}' did not include the ReLU baseline")

        for row in task_rows:
            row["delta_params_vs_relu"] = int(row["total_params"] - baseline_row["total_params"])
            row["delta_forward_flops_vs_relu"] = int(row["forward_flops"] - baseline_row["forward_flops"])
            if row["peak_cuda_memory_mb"] is not None and baseline_row["peak_cuda_memory_mb"] is not None:
                row["delta_peak_cuda_memory_mb_vs_relu"] = float(row["peak_cuda_memory_mb"] - baseline_row["peak_cuda_memory_mb"])
            else:
                row["delta_peak_cuda_memory_mb_vs_relu"] = None

        analysis["tasks"][task_name] = {
            "batch_spec": TASK_SPECS[task_name].batch_shape,
            "baseline_activation": "relu",
            "rows": task_rows,
        }
        analysis["table"].extend(task_rows)

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile parameter count, FLOPs, and peak CUDA memory for the benchmark models")
    parser.add_argument("--output", type=str, default="artifacts/complexity_analysis.json", help="Path to write the complexity JSON table")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, or cpu")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root used to resolve the WikiText-2 vocabulary")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASKS, choices=TASKS, help="Tasks to profile")
    parser.add_argument("--activations", type=str, nargs="+", default=ACTIVATIONS, choices=ACTIVATIONS, help="Activations to profile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"[Complexity] Device: {device}")
    if device.type != "cuda":
        print("[Complexity] CUDA unavailable; peak_cuda_memory_mb will be null")

    analysis = build_complexity_analysis(
        tasks=list(args.tasks),
        activations=list(args.activations),
        device=device,
        data_root=args.data_root,
    )

    output_path = Path(args.output)
    write_json(output_path, analysis)
    print(f"[IO] Saved complexity analysis to {output_path}")


if __name__ == "__main__":
    main()