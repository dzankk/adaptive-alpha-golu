"""
Phase 2 Scale-Up Benchmark: CIFAR-100 Image Classification
============================================================
Dedicated, standalone scaled classification runner. Reuses the ResNet-18 backbone,
activation factory, and evaluation helpers from experiments/run_classification.py
(imported, not modified) but adds CIFAR-100 (100-class) data loading and a
correspondingly adjusted classifier head via ResNet18(num_classes=100, ...).

Outputs are written under outputs/runs_scale/classification/ -- strictly separate
from Phase 1's outputs/runs/classification/ artifacts.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse Phase 1's model/activation/eval code as-is (no modifications to that file).
from experiments.run_classification import (
    ResNet18,
    evaluate_model,
    reset_all_seeds,
    seed_worker,
    summarize_model_overhead,
)
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.experiment_config import load_benchmark_config
from utils.overhead_tracker import OverheadTracker
from utils.run_artifacts import build_run_manifest, create_run_directory, stable_seed_directory, write_json
from utils.train_tuning import (
    bf16_autocast,
    clamp_alpha_golu_modules,
    clear_training_checkpoints,
    clip_activation_gradients,
    compute_model_grad_norm,
    default_loader_kwargs,
    load_training_checkpoint,
    overhead_tracking_enabled,
    resolve_task_alpha_hparams,
    save_training_checkpoint,
    split_model_parameters,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs_scale"
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_dataloaders(
    batch_size: int = 128,
    seed: int = 42,
    root: str = "./data",
    val_split: float = 0.1,
):
    """CIFAR-100 train/val/test loaders; not supported by Phase 1's get_dataloaders()."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    generator = torch.Generator()
    generator.manual_seed(seed)

    val_size = int(len(trainset) * val_split)
    train_size = len(trainset) - val_size
    trainset, valset = (random_split(trainset, [train_size, val_size], generator=generator) if val_size > 0 else (trainset, None))

    train_loader_kwargs = default_loader_kwargs()
    eval_loader_kwargs = default_loader_kwargs(num_workers=1)

    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator, **train_loader_kwargs
    )
    val_loader = None
    if valset is not None:
        val_loader = DataLoader(
            valset, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=generator, **eval_loader_kwargs
        )
    test_loader = DataLoader(
        testset, batch_size=256, shuffle=False, worker_init_fn=seed_worker, generator=generator, **eval_loader_kwargs
    )
    return train_loader, val_loader, test_loader


def _resolve_scale_recipe(config_path: str | None, base_lr: float | None) -> dict:
    config = load_benchmark_config(config_path) if config_path else {}
    scale_cfg = config.get("classification_scale_up", {}) if isinstance(config, dict) else {}
    base_lr_by_task = config.get("base_lr_by_task", {}) if isinstance(config, dict) else {}
    return {
        "base_lr": float(base_lr if base_lr is not None else base_lr_by_task.get("classification", 0.1)),
        "num_classes": int(scale_cfg.get("num_classes", 100)),
        "weight_decay": float(scale_cfg.get("weight_decay", 5e-4)),
        "momentum": float(scale_cfg.get("momentum", 0.9)),
        "scheduler_step_size": int(scale_cfg.get("scheduler_step_size", 40)),
        "scheduler_gamma": float(scale_cfg.get("scheduler_gamma", 0.1)),
    }


def train_single_seed_classification_scale(
    act_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    data_root: str = "./data",
    base_lr: float | None = None,
    alpha_lr: float | None = None,
    val_split: float = 0.1,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    reset_all_seeds(seed)
    recipe = _resolve_scale_recipe(config_path, base_lr)

    train_loader, val_loader, test_loader = get_cifar100_dataloaders(seed=seed, root=data_root, val_split=val_split)
    eval_loader = val_loader if val_loader is not None else test_loader

    model = ResNet18(num_classes=recipe["num_classes"], act_type=act_type).to(device)
    overhead_tracker = OverheadTracker(task_name="classification_scale", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "classification", alpha_lr, config_path=config_path
    )

    base_params, act_params = split_model_parameters(model)
    parameter_groups = [{"params": base_params, "lr": recipe["base_lr"], "weight_decay": recipe["weight_decay"]}]
    if act_params:
        parameter_groups.append({"params": act_params, "lr": float(alpha_lr), "weight_decay": 0.0})
    optimizer = torch.optim.SGD(parameter_groups, lr=recipe["base_lr"], momentum=recipe["momentum"], nesterov=True)
    scheduler = StepLR(optimizer, step_size=recipe["scheduler_step_size"], gamma=recipe["scheduler_gamma"])

    warmup_epochs = max(int(alpha_warmup_epochs), 0)

    def set_alpha_lr(epoch_index: int) -> float:
        if not act_params:
            return 0.0
        current_lr = float(alpha_lr)
        if warmup_epochs > 0:
            current_lr *= min(1.0, float(epoch_index + 1) / float(warmup_epochs))
        optimizer.param_groups[-1]["lr"] = current_lr
        return float(current_lr)

    alpha_logger = AlphaTrajectoryLogger(model)
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"

    epoch_seconds: list[float] = []
    epoch_losses: list[float] = []
    lr_history: list[float] = []
    grad_norm_history: list[float] = []
    alpha_clamp_events = 0
    alpha_clamp_checks = 0

    seed_dir = None
    start_epoch = 0
    run_dir = None
    progress_path = None
    if save_artifacts:
        seed_dir = stable_seed_directory(str(OUTPUT_ROOT), "classification", act_type, seed)
        if resume:
            checkpoint = load_training_checkpoint(seed_dir, map_location=device)
            if checkpoint is not None:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                if checkpoint.get("scheduler_state") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                extra = checkpoint.get("extra", {})
                epoch_seconds = list(extra.get("epoch_seconds", []))
                epoch_losses = list(extra.get("epoch_losses", []))
                lr_history = list(extra.get("lr_history", []))
                grad_norm_history = list(extra.get("grad_norm_history", []))
                alpha_clamp_events = int(extra.get("alpha_clamp_events", 0))
                alpha_clamp_checks = int(extra.get("alpha_clamp_checks", 0))
                alpha_logger.alpha_history = extra.get("alpha_history", alpha_logger.alpha_history)
                start_epoch = int(checkpoint["epoch"])
                print(f"[CLASSIFICATION_SCALE] Resuming activation={act_type} seed={seed} from epoch {start_epoch + 1}/{epochs}", flush=True)
        else:
            clear_training_checkpoints(seed_dir)

        run_dir = create_run_directory(str(OUTPUT_ROOT / "classification"), "classification", act_type, [seed])
        progress_path = run_dir / "progress.json"
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="classification_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "dataset_name": "cifar100",
                    "num_classes": recipe["num_classes"],
                    "data_root": data_root,
                    "epochs": epochs,
                    "base_lr": recipe["base_lr"],
                    "alpha_lr": alpha_lr,
                    "weight_decay": recipe["weight_decay"],
                    "momentum": recipe["momentum"],
                },
            ),
        )

    current_alpha_lr = set_alpha_lr(start_epoch)
    train_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        current_alpha_lr = set_alpha_lr(epoch)
        model.train()
        epoch_loss_total = 0.0
        epoch_batches = 0
        epoch_grad_norm_total = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with bf16_autocast(amp_enabled):
                outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.float(), labels)
            loss.backward()
            grad_norm = compute_model_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            epoch_grad_norm_total += grad_norm
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks

        scheduler.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()
        mean_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        mean_epoch_grad_norm = epoch_grad_norm_total / max(epoch_batches, 1)
        epoch_losses.append(mean_epoch_loss)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        grad_norm_history.append(mean_epoch_grad_norm)

        if save_artifacts and progress_path is not None:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "task": "classification_scale",
                    "dataset_name": "cifar100",
                    "activation": act_type,
                    "seed": seed,
                    "epochs": epochs,
                    "epoch": epoch + 1,
                    "progress_pct": float(((epoch + 1) / max(epochs, 1)) * 100.0),
                    "epoch_loss": mean_epoch_loss,
                    "epoch_loss_history": epoch_losses,
                    "epoch_seconds": epoch_seconds,
                    "lr_history": lr_history,
                    "grad_norm_history": grad_norm_history,
                    "alpha_lr_final": current_alpha_lr,
                    "alpha_clamp_events": alpha_clamp_events,
                    "alpha_clamp_checks": alpha_clamp_checks,
                    "alpha_history": alpha_logger.alpha_history,
                },
            )
            save_training_checkpoint(
                seed_dir,
                epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                extra={
                    "epoch_seconds": epoch_seconds,
                    "epoch_losses": epoch_losses,
                    "lr_history": lr_history,
                    "grad_norm_history": grad_norm_history,
                    "alpha_clamp_events": alpha_clamp_events,
                    "alpha_clamp_checks": alpha_clamp_checks,
                    "alpha_history": alpha_logger.alpha_history,
                },
            )
        if not save_artifacts:
            print(f"[CLASSIFICATION_SCALE] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.4f} | alpha_lr={current_alpha_lr:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_start
    eval_loss, accuracy = evaluate_model(model, eval_loader, device, amp_enabled=amp_enabled)
    alphas = model.extract_alphas()
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}

    if save_artifacts:
        payload = {
            "task": "classification_scale",
            "dataset_name": "cifar100",
            "num_classes": recipe["num_classes"],
            "data_root": data_root,
            "activation": act_type,
            "alpha_lr": alpha_lr,
            "alpha_lr_final": current_alpha_lr,
            "seed": seed,
            "epochs": epochs,
            "base_lr": recipe["base_lr"],
            "weight_decay": recipe["weight_decay"],
            "momentum": recipe["momentum"],
            "accuracy": accuracy,
            "eval_loss": eval_loss,
            "alpha_values": alphas,
            "alpha_history": alpha_logger.alpha_history,
            "train_seconds": train_seconds,
            "epoch_seconds": epoch_seconds,
            "epoch_loss_history": epoch_losses,
            "lr_history": lr_history,
            "grad_norm_history": grad_norm_history,
            "alpha_clamp_events": alpha_clamp_events,
            "alpha_clamp_checks": alpha_clamp_checks,
            **overhead,
        }
        write_json(run_dir / "results.json", payload)
        write_json(progress_path, {**payload, "status": "completed", "progress_pct": 100.0})
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="classification_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={"dataset_name": "cifar100", "num_classes": recipe["num_classes"], "epochs": epochs},
            ),
        )
        clear_training_checkpoints(seed_dir)

    return float(accuracy)


def run_classification_scale_benchmark(
    seeds: list[int] | None = None,
    epochs: int = 120,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    activations: list[str] | None = None,
    amp: bool = False,
    resume: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 999]
    activations = activations or ["golu_static", "alpha_golu"]
    print(f"Running CIFAR-100 Scale-Up Classification Benchmark on {device} (N={len(seeds)})")

    for act_type in activations:
        scores = []
        for seed in seeds:
            accuracy = train_single_seed_classification_scale(
                act_type=act_type,
                seed=seed,
                epochs=epochs,
                device=device,
                data_root=data_root,
                alpha_lr=alpha_lr,
                config_path=config_path,
                save_artifacts=True,
                amp=amp,
                resume=resume,
            )
            scores.append(accuracy)
            print(f"Activation: {act_type.ljust(15)} | Seed {seed} | CIFAR-100 Accuracy: {accuracy:.2f}%")
        mean_score = sum(scores) / len(scores)
        print(f"--> {act_type.upper()} Mean Accuracy: {mean_score:.2f}%\n")


def train_and_eval(
    activation: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 120,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    """Returns CIFAR-100 top-1 accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_single_seed_classification_scale(
        act_type=activation,
        seed=seed,
        epochs=epochs,
        device=device,
        data_root=data_root,
        alpha_lr=alpha_lr,
        config_path=config_path,
        save_artifacts=save_artifacts,
        amp=amp,
        resume=resume,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 scale-up: CIFAR-100 classification benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--config", type=str, default="configs/phase2_complex_scale.json", help="Benchmark config file")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    parser.add_argument("--fresh", action="store_true", help="Ignore any saved epoch checkpoint and restart this seed from epoch 0")
    args = parser.parse_args()

    if args.benchmark:
        run_classification_scale_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp, resume=not args.fresh)
    else:
        for seed in args.seeds:
            accuracy = train_and_eval(activation=args.activation, seed=seed, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, save_artifacts=True, amp=args.amp, resume=not args.fresh)
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | CIFAR-100 Accuracy: {accuracy:.2f}%")
