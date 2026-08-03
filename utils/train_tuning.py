"""
Shared training-time tuning helpers for benchmark runners.

Provides:
- heuristic DataLoader defaults tuned for GPU-backed benchmark runs
- activation-parameter extraction for AdamW parameter groups
- optional warmup for activation learning rates
- gradient clipping restricted to activation parameters
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn

from utils.experiment_config import load_benchmark_config


_ACTIVATION_PARAM_NAMES = {"raw_alpha", "beta"}


def _is_activation_module(module: nn.Module) -> bool:
    if isinstance(module, nn.PReLU):
        return True

    direct_param_names = {name for name, _ in module.named_parameters(recurse=False)}
    if direct_param_names & _ACTIVATION_PARAM_NAMES:
        return True

    module_name = module.__class__.__name__.lower()
    return any(keyword in module_name for keyword in ("alphagolu", "golu", "pgelu", "swishadaptive", "adaptive_swish"))


def collect_activation_parameters(model: nn.Module) -> list[nn.Parameter]:
    activation_params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        if not _is_activation_module(module):
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad and id(parameter) not in seen:
                activation_params.append(parameter)
                seen.add(id(parameter))

    return activation_params


def split_model_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    activation_params = collect_activation_parameters(model)
    activation_ids = {id(parameter) for parameter in activation_params}

    base_params: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in activation_ids:
            base_params.append(parameter)

    return base_params, activation_params


def build_adamw_with_activation_groups(
    model: nn.Module,
    *,
    base_lr: float,
    base_weight_decay: float = 1e-4,
    activation_lr: float | None = None,
    activation_weight_decay: float = 0.0,
    warmup_epochs: int = 1,
) -> tuple[torch.optim.Optimizer, Callable[[int], float], list[nn.Parameter]]:
    base_params, activation_params = split_model_parameters(model)
    target_activation_lr = float(activation_lr if activation_lr is not None else base_lr)

    parameter_groups: list[dict[str, object]] = [
        {"params": base_params, "lr": float(base_lr), "weight_decay": float(base_weight_decay)},
    ]
    if activation_params:
        parameter_groups.append(
            {
                "params": activation_params,
                "lr": target_activation_lr,
                "weight_decay": float(activation_weight_decay),
            }
        )

    optimizer = torch.optim.AdamW(parameter_groups, lr=base_lr)
    warmup_epochs = max(int(warmup_epochs), 0)

    def set_activation_lr(epoch_index: int) -> float:
        if not activation_params:
            return 0.0
        current_lr = target_activation_lr
        if warmup_epochs > 0:
            scale = min(1.0, float(epoch_index + 1) / float(warmup_epochs))
            current_lr = target_activation_lr * scale
        optimizer.param_groups[-1]["lr"] = current_lr
        return float(current_lr)

    return optimizer, set_activation_lr, activation_params


def resolve_task_alpha_hparams(
    task_name: str,
    alpha_lr: float | None = None,
    *,
    config_path: str | None = "configs/paper_benchmark.json",
    default_alpha_lr: float = 1e-3,
    default_warmup_epochs: int = 1,
    default_grad_clip_norm: float = 1.0,
) -> tuple[float, int, float]:
    config = load_benchmark_config(config_path) if config_path else {}
    task_key = task_name.lower().strip()

    alpha_lr_map = config.get("alpha_lr_by_task", {})
    warmup_map = config.get("alpha_lr_warmup_epochs", {})
    clip_map = config.get("alpha_grad_clip_norm", {})

    resolved_alpha_lr = float(alpha_lr if alpha_lr is not None else alpha_lr_map.get(task_key, default_alpha_lr))
    resolved_warmup_epochs = int(warmup_map.get(task_key, default_warmup_epochs))
    resolved_clip_norm = float(clip_map.get(task_key, default_grad_clip_norm))
    return resolved_alpha_lr, max(resolved_warmup_epochs, 0), resolved_clip_norm


def clip_activation_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    activation_params = collect_activation_parameters(model)
    if not activation_params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(activation_params, max_norm=max_norm))


def default_loader_kwargs(num_workers: int | None = None) -> dict[str, object]:
    cpu_count = os.cpu_count() or 4
    if num_workers is None:
        if not torch.cuda.is_available() and os.name == "nt":
            num_workers = 0
        else:
            num_workers = 4 if torch.cuda.is_available() else 2
            num_workers = min(num_workers, max(1, cpu_count // 2))

    kwargs: dict[str, object] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs
