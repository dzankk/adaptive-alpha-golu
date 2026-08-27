"""
Phase 2 Scale-Up Benchmark: Higher-Resolution Diffusion
==========================================================
Dedicated, standalone scaled diffusion runner. Reuses the DDPM noise schedule,
activation factory, and LR-schedule resolution helpers from
experiments/run_diffusion.py (imported, not modified) but adds a
ScaledDiffusionUNet with a configurable channel width and trains on CIFAR-10
resized to a configurable (higher) resolution.

Outputs are written under outputs/runs_scale/diffusion/ -- strictly separate
from Phase 1's outputs/runs/diffusion/ artifacts.
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse Phase 1's activation factory, position embeddings, seeding, and LR-schedule
# resolution as-is (no modifications to that file).
from experiments.run_diffusion import (
    SinusoidalPositionEmbeddings,
    _resolve_diffusion_schedule_hparams,
    get_activation,
    reset_seeds,
    seed_worker,
)
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.experiment_config import load_benchmark_config
from utils.overhead_tracker import OverheadTracker
from utils.run_artifacts import build_run_manifest, create_run_directory, stable_seed_directory, write_json
from utils.train_tuning import (
    bf16_autocast,
    build_adamw_with_activation_groups,
    clamp_alpha_golu_modules,
    clear_training_checkpoints,
    clip_activation_gradients,
    compute_model_grad_norm,
    default_loader_kwargs,
    load_training_checkpoint,
    overhead_tracking_enabled,
    resolve_task_alpha_hparams,
    save_training_checkpoint,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs_scale"


class ScaledDiffusionUNet(nn.Module):
    """Same shallow conv-UNet shape as Phase 1's DiffusionUNet, but with a configurable channel width."""

    def __init__(self, in_channels: int = 3, act_type: str = "alpha_golu", base_channels: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels),
            get_activation(act_type),
        )
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.act1 = get_activation(act_type)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.act2 = get_activation(act_type)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(time)[:, :, None, None]
        h = self.act1(self.conv1(x)) + t_emb
        h = self.act2(self.conv2(h))
        return self.out_conv(h)


def _resolve_scale_recipe(config_path: str | None) -> dict:
    config = load_benchmark_config(config_path) if config_path else {}
    scale_cfg = config.get("diffusion_scale_up", {}) if isinstance(config, dict) else {}
    return {
        "image_size": int(scale_cfg.get("image_size", 64)),
        "base_channels": int(scale_cfg.get("base_channels", 128)),
    }


def train_single_seed_diffusion_scale(
    act_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    data_root: str = "./data",
    base_lr: float | None = None,
    alpha_lr: float | None = None,
    image_size: int | None = None,
    base_channels: int | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    reset_seeds(seed)
    recipe = _resolve_scale_recipe(config_path)
    resolved_image_size = int(image_size) if image_size is not None else recipe["image_size"]
    resolved_base_channels = int(base_channels) if base_channels is not None else recipe["base_channels"]

    transform = transforms.Compose(
        [
            transforms.Resize((resolved_image_size, resolved_image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    loader_g = torch.Generator().manual_seed(seed)
    eval_loader_g = torch.Generator().manual_seed(seed + 999)
    eval_sample_g = torch.Generator(device=device.type).manual_seed(seed + 999)

    train_loader_kwargs = default_loader_kwargs()
    test_loader_kwargs = default_loader_kwargs(num_workers=1)
    trainloader = DataLoader(
        full_dataset, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=loader_g, **train_loader_kwargs
    )
    testloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, worker_init_fn=seed_worker, generator=eval_loader_g, **test_loader_kwargs
    )

    timesteps = 1000
    beta = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    model = ScaledDiffusionUNet(in_channels=3, act_type=act_type, base_channels=resolved_base_channels).to(device)
    overhead_tracker = OverheadTracker(task_name="diffusion_scale", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    alpha_logger = AlphaTrajectoryLogger(model)
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams("diffusion", alpha_lr, config_path=config_path)
    schedule_hparams = _resolve_diffusion_schedule_hparams(config_path=config_path, base_lr=base_lr)
    resolved_base_lr = float(schedule_hparams["base_lr"])
    scheduler_name = str(schedule_hparams["scheduler_name"])
    scheduler_warmup_steps = int(schedule_hparams["warmup_steps"])
    scheduler_min_lr = float(schedule_hparams["min_lr"])

    optimizer, set_alpha_lr, act_params = build_adamw_with_activation_groups(
        model, base_lr=resolved_base_lr, base_weight_decay=1e-4, activation_lr=alpha_lr, activation_weight_decay=0.0, warmup_epochs=alpha_warmup_epochs
    )

    total_train_steps = max(int(epochs) * max(len(trainloader), 1), 1)
    warmup_steps = min(max(scheduler_warmup_steps, 0), total_train_steps - 1) if total_train_steps > 1 else 0
    min_lr_ratio = min(1.0, scheduler_min_lr / max(resolved_base_lr, 1e-12))

    def set_base_lr_for_step(step_index: int) -> float:
        if scheduler_name == "none":
            current_lr = resolved_base_lr
        else:
            clamped_step = min(max(step_index, 0), total_train_steps - 1)
            if scheduler_name == "cosine_warmup" and warmup_steps > 0 and clamped_step < warmup_steps:
                current_lr = resolved_base_lr * (float(clamped_step + 1) / float(warmup_steps))
            else:
                progress_denom = max(total_train_steps - warmup_steps, 1)
                progress = min(max(float(clamped_step - warmup_steps) / float(progress_denom), 0.0), 1.0)
                cosine_scale = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
                current_lr = resolved_base_lr * cosine_scale
        optimizer.param_groups[0]["lr"] = float(current_lr)
        return float(current_lr)

    criterion = nn.MSELoss()
    epoch_seconds: list[float] = []
    epoch_losses: list[float] = []
    lr_history: list[float] = []
    grad_norm_history: list[float] = []
    amp_enabled = bool(amp) and torch.cuda.is_available() and device.type == "cuda"
    alpha_clamp_events = 0
    alpha_clamp_checks = 0

    seed_dir = None
    start_epoch = 0
    run_dir = None
    progress_path = None
    if save_artifacts:
        seed_dir = stable_seed_directory(str(OUTPUT_ROOT), "diffusion", act_type, seed)
        if resume:
            checkpoint = load_training_checkpoint(seed_dir, map_location=device)
            if checkpoint is not None:
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                extra = checkpoint.get("extra", {})
                epoch_seconds = list(extra.get("epoch_seconds", []))
                epoch_losses = list(extra.get("epoch_losses", []))
                lr_history = list(extra.get("lr_history", []))
                grad_norm_history = list(extra.get("grad_norm_history", []))
                alpha_clamp_events = int(extra.get("alpha_clamp_events", 0))
                alpha_clamp_checks = int(extra.get("alpha_clamp_checks", 0))
                alpha_logger.alpha_history = extra.get("alpha_history", alpha_logger.alpha_history)
                start_epoch = int(checkpoint["epoch"])
                print(f"[DIFFUSION_SCALE] Resuming activation={act_type} seed={seed} from epoch {start_epoch + 1}/{epochs}", flush=True)
        else:
            clear_training_checkpoints(seed_dir)

        run_dir = create_run_directory(str(OUTPUT_ROOT / "diffusion"), "diffusion", act_type, [seed])
        progress_path = run_dir / "progress.json"
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="diffusion_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "data_root": data_root,
                    "epochs": epochs,
                    "image_size": resolved_image_size,
                    "base_channels": resolved_base_channels,
                    "base_lr": resolved_base_lr,
                    "scheduler": scheduler_name,
                    "alpha_lr": alpha_lr,
                },
            ),
        )

    global_step = start_epoch * max(len(trainloader), 1)
    current_base_lr = set_base_lr_for_step(global_step)
    current_alpha_lr = set_alpha_lr(start_epoch)
    train_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        model.train()
        current_alpha_lr = set_alpha_lr(epoch)
        epoch_loss_total = 0.0
        epoch_batches = 0
        epoch_grad_norm_total = 0.0
        for x0, _ in trainloader:
            x0 = x0.to(device, non_blocking=True)
            current_base_lr = set_base_lr_for_step(global_step)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device)
            noise = torch.randn(x0.shape, device=device)

            a_hat_t = alpha_hat[t][:, None, None, None]
            xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise

            optimizer.zero_grad()
            with bf16_autocast(amp_enabled):
                pred_noise = model(xt, t)
            loss = criterion(pred_noise.float(), noise)
            loss.backward()
            grad_norm = compute_model_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            epoch_grad_norm_total += grad_norm
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            global_step += 1
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=0.2, max_alpha=3.0)
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks

        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()
        mean_epoch_loss = epoch_loss_total / max(epoch_batches, 1)
        mean_epoch_grad_norm = epoch_grad_norm_total / max(epoch_batches, 1)
        epoch_losses.append(mean_epoch_loss)
        lr_history.append(float(current_base_lr))
        grad_norm_history.append(mean_epoch_grad_norm)

        if save_artifacts and progress_path is not None:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "task": "diffusion_scale",
                    "data_root": data_root,
                    "activation": act_type,
                    "image_size": resolved_image_size,
                    "base_channels": resolved_base_channels,
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
            print(f"[DIFFUSION_SCALE] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.6f} | alpha_lr={current_alpha_lr:.6f}", flush=True)

    model.eval()
    val_losses = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for x0, _ in testloader:
                x0 = x0.to(device, non_blocking=True)
                t = torch.randint(0, timesteps, (x0.size(0),), device=device, generator=eval_sample_g)
                noise = torch.randn(x0.shape, device=device, generator=eval_sample_g)
                a_hat_t = alpha_hat[t][:, None, None, None]
                xt = torch.sqrt(a_hat_t) * x0 + torch.sqrt(1 - a_hat_t) * noise
                pred_noise = model(xt, t)
                batch_loss = criterion(pred_noise.float(), noise).item()
                batch_size = x0.size(0)
                val_losses.append(batch_loss)
                total_loss += batch_loss * batch_size
                total_examples += batch_size

    train_seconds = time.perf_counter() - train_start
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}
    loss = float(total_loss / total_examples) if total_examples > 0 else (float(sum(val_losses) / len(val_losses)) if val_losses else 0.0)

    if save_artifacts:
        payload = {
            "task": "diffusion_scale",
            "data_root": data_root,
            "activation": act_type,
            "seed": seed,
            "epochs": epochs,
            "image_size": resolved_image_size,
            "base_channels": resolved_base_channels,
            "base_lr": resolved_base_lr,
            "scheduler": scheduler_name,
            "alpha_lr": alpha_lr,
            "alpha_lr_final": current_alpha_lr if act_params else None,
            "mse": float(loss),
            "train_seconds": train_seconds,
            "epoch_seconds": epoch_seconds,
            "epoch_loss_history": epoch_losses,
            "lr_history": lr_history,
            "grad_norm_history": grad_norm_history,
            "alpha_clamp_events": alpha_clamp_events,
            "alpha_clamp_checks": alpha_clamp_checks,
            "alpha_history": alpha_logger.alpha_history,
            **overhead,
        }
        write_json(run_dir / "results.json", payload)
        write_json(progress_path, {**payload, "status": "completed", "progress_pct": 100.0})
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="diffusion_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={"image_size": resolved_image_size, "base_channels": resolved_base_channels, "epochs": epochs},
            ),
        )
        clear_training_checkpoints(seed_dir)

    return float(loss)


def run_diffusion_scale_benchmark(
    seeds: list[int] | None = None,
    epochs: int = 60,
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
    print(f"Running Scale-Up Diffusion Benchmark on {device} (N={len(seeds)})")

    for act_type in activations:
        scores = []
        for seed in seeds:
            mse = train_single_seed_diffusion_scale(
                act_type=act_type, seed=seed, epochs=epochs, device=device, data_root=data_root,
                alpha_lr=alpha_lr, config_path=config_path, save_artifacts=True, amp=amp, resume=resume,
            )
            scores.append(mse)
            print(f"Activation: {act_type.ljust(15)} | Seed {seed} | MSE: {mse:.6f}")
        mean_score = sum(scores) / len(scores)
        print(f"--> {act_type.upper()} Mean MSE: {mean_score:.6f}\n")


def train_and_eval(
    activation: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 60,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    """Returns validation MSE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_single_seed_diffusion_scale(
        act_type=activation, seed=seed, epochs=epochs, device=device, data_root=data_root,
        alpha_lr=alpha_lr, config_path=config_path, save_artifacts=save_artifacts, amp=amp, resume=resume,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 scale-up: higher-resolution diffusion benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters")
    parser.add_argument("--image-size", type=int, default=None, help="Resized image resolution override (e.g. 64)")
    parser.add_argument("--base-channels", type=int, default=None, help="UNet channel-width override")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--config", type=str, default="configs/phase2_complex_scale.json", help="Benchmark config file")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    parser.add_argument("--fresh", action="store_true", help="Ignore any saved epoch checkpoint and restart this seed from epoch 0")
    args = parser.parse_args()

    if args.benchmark:
        run_diffusion_scale_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp, resume=not args.fresh)
    else:
        for seed in args.seeds:
            mse = train_single_seed_diffusion_scale(
                act_type=args.activation,
                seed=seed,
                epochs=args.epochs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                data_root=args.data_root,
                alpha_lr=args.alpha_lr,
                image_size=args.image_size,
                base_channels=args.base_channels,
                config_path=args.config,
                save_artifacts=True,
                amp=args.amp,
                resume=not args.fresh,
            )
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | MSE: {mse:.6f}")
