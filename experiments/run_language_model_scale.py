"""
Phase 2 Scale-Up Benchmark: Larger-Context Language Modeling
==============================================================
Dedicated, standalone scaled language-model runner. Reuses the WikiText-2 data
pipeline, tokenizer, and MiniGPT architecture from experiments/run_language_model.py
(imported, not modified) but exposes n_layers, d_model, and context length
(block_size) as configurable knobs -- these are already constructor parameters
of MiniGPT, just not threaded through Phase 1's train_single_seed_lm().

Outputs are written under outputs/runs_scale/language_model/ -- strictly separate
from Phase 1's outputs/runs/language_model/ artifacts.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse Phase 1's data pipeline and model architecture as-is (no modifications to that file).
from experiments.run_language_model import (
    MiniGPT,
    build_language_model_dataloaders,
    reset_all_seeds,
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
    load_training_checkpoint,
    overhead_tracking_enabled,
    save_training_checkpoint,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs_scale"


def _resolve_scale_recipe(config_path: str | None) -> dict:
    config = load_benchmark_config(config_path) if config_path else {}
    scale_cfg = config.get("language_model_scale_up", {}) if isinstance(config, dict) else {}
    return {
        "block_size": int(scale_cfg.get("block_size", scale_cfg.get("context_len", 512))),
        "d_model": int(scale_cfg.get("d_model", 256)),
        "n_layers": int(scale_cfg.get("n_layers", 6)),
        "n_heads": int(scale_cfg.get("n_heads", 8)),
        # Deliberately distinct from Phase 1's shared alpha_lr_by_task.language_model (0.001, == base_lr):
        # at 6 layers/d_model=256 that ratio measurably destabilized alpha. 1e-4 (10x separation)
        # recovered most of the gap but still trailed golu_static; 2e-5 (50x separation) with a
        # longer warmup is the current iteration (see configs/phase2_complex_scale.json notes).
        "alpha_lr": float(scale_cfg.get("alpha_lr", 2e-5)),
        "alpha_lr_warmup_epochs": int(scale_cfg.get("alpha_lr_warmup_epochs", 4)),
        "alpha_grad_clip_norm": float(scale_cfg.get("alpha_grad_clip_norm", 0.5)),
        "alpha_min": float(scale_cfg.get("alpha_min", 0.1)),
        "alpha_max": float(scale_cfg.get("alpha_max", 5.0)),
    }


def train_single_seed_lm_scale(
    act_type: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 20,
    device: torch.device | str = "cuda",
    dataset_name: str = "wikitext2",
    data_root: str = "./data",
    alpha_lr: float | None = None,
    n_layers: int | None = None,
    d_model: int | None = None,
    context_len: int | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    """Returns validation perplexity for the scaled MiniGPT configuration."""
    reset_all_seeds(seed)
    recipe = _resolve_scale_recipe(config_path)
    block_size = int(context_len) if context_len is not None else recipe["block_size"]
    d_model_value = int(d_model) if d_model is not None else recipe["d_model"]
    n_layers_value = int(n_layers) if n_layers is not None else recipe["n_layers"]
    n_heads_value = recipe["n_heads"]

    train_loader, valid_loader, vocab = build_language_model_dataloaders(
        dataset_name=dataset_name,
        root=data_root,
        block_size=block_size,
        batch_size=16,
        seed=seed,
    )

    model = MiniGPT(
        vocab_size=len(vocab),
        d_model=d_model_value,
        n_heads=n_heads_value,
        n_layers=n_layers_value,
        max_seq_len=block_size,
        act_type=act_type,
    ).to(device)
    overhead_tracker = OverheadTracker(task_name="language_model_scale", activation_name=act_type, model=model, device=device) if overhead_tracking_enabled() else None
    # Deliberately bypasses resolve_task_alpha_hparams()'s alpha_lr_by_task["language_model"] lookup:
    # that shared Phase 1 value (0.001, == base_lr) is what destabilized alpha at this scale, so the
    # scale recipe (language_model_scale_up) is the sole source of truth here, not the shared map.
    alpha_lr = float(alpha_lr) if alpha_lr is not None else recipe["alpha_lr"]
    alpha_warmup_epochs = recipe["alpha_lr_warmup_epochs"]
    alpha_grad_clip_norm = recipe["alpha_grad_clip_norm"]
    optimizer, set_alpha_lr, act_params = build_adamw_with_activation_groups(
        model, base_lr=1e-3, base_weight_decay=1e-4, activation_lr=alpha_lr, activation_weight_decay=0.0, warmup_epochs=alpha_warmup_epochs
    )
    criterion = nn.CrossEntropyLoss()
    alpha_logger = AlphaTrajectoryLogger(model)
    amp_enabled = bool(amp) and torch.cuda.is_available() and "cuda" in str(device)

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
        seed_dir = stable_seed_directory(str(OUTPUT_ROOT), "language_model", act_type, seed)
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
                print(f"[LANGUAGE_MODEL_SCALE] Resuming activation={act_type} seed={seed} from epoch {start_epoch + 1}/{epochs}", flush=True)
        else:
            clear_training_checkpoints(seed_dir)

        run_dir = create_run_directory(str(OUTPUT_ROOT / "language_model"), "language_model", act_type, [seed])
        progress_path = run_dir / "progress.json"
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="language_model_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "dataset_name": dataset_name,
                    "data_root": data_root,
                    "epochs": epochs,
                    "block_size": block_size,
                    "d_model": d_model_value,
                    "n_layers": n_layers_value,
                    "n_heads": n_heads_value,
                    "alpha_lr": alpha_lr,
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
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            with bf16_autocast(amp_enabled):
                logits = model(inputs)
            loss = criterion(logits.float().reshape(-1, len(vocab)), targets.reshape(-1))
            loss.backward()
            grad_norm = compute_model_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm for activation={act_type}, seed={seed}, epoch={epoch + 1}")
            epoch_grad_norm_total += grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
            epoch_loss_total += float(loss.item())
            epoch_batches += 1
            clamp_events, clamp_checks = clamp_alpha_golu_modules(model, min_alpha=recipe["alpha_min"], max_alpha=recipe["alpha_max"])
            alpha_clamp_events += clamp_events
            alpha_clamp_checks += clamp_checks

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
                    "task": "language_model_scale",
                    "dataset_name": dataset_name,
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
            print(f"[LANGUAGE_MODEL_SCALE] Epoch {epoch + 1}/{epochs} - Loss: {mean_epoch_loss:.4f} | alpha_lr={current_alpha_lr:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_start
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for batch in valid_loader:
                batch = batch.to(device, non_blocking=True)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.float().reshape(-1, len(vocab)), targets.reshape(-1))
                total_loss += float(loss.item()) * targets.numel()
                total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    overhead = overhead_tracker.save() if overhead_tracker is not None else {}

    if save_artifacts:
        payload = {
            "task": "language_model_scale",
            "dataset_name": dataset_name,
            "data_root": data_root,
            "activation": act_type,
            "alpha_lr": alpha_lr,
            "alpha_lr_final": current_alpha_lr if act_params else None,
            "seed": seed,
            "epochs": epochs,
            "block_size": block_size,
            "d_model": d_model_value,
            "n_layers": n_layers_value,
            "n_heads": n_heads_value,
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
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
                task="language_model_scale",
                seeds=[seed],
                activations=[act_type],
                extra_config={"dataset_name": dataset_name, "block_size": block_size, "d_model": d_model_value, "n_layers": n_layers_value},
            ),
        )
        clear_training_checkpoints(seed_dir)

    return float(perplexity)


def run_language_model_scale_benchmark(
    seeds: list[int] | None = None,
    epochs: int = 20,
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
    print(f"Running Scale-Up Language Model Benchmark on {device} (N={len(seeds)})")

    for act_type in activations:
        scores = []
        for seed in seeds:
            perplexity = train_single_seed_lm_scale(
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
            scores.append(perplexity)
            print(f"Activation: {act_type.ljust(15)} | Seed {seed} | Perplexity: {perplexity:.2f}")
        mean_score = sum(scores) / len(scores)
        print(f"--> {act_type.upper()} Mean Perplexity: {mean_score:.2f}\n")


def train_and_eval(
    activation: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 20,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/phase2_complex_scale.json",
    save_artifacts: bool = False,
    amp: bool = False,
    resume: bool = True,
) -> float:
    """Returns validation perplexity."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_single_seed_lm_scale(
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

    parser = argparse.ArgumentParser(description="Phase 2 scale-up: larger-context language model benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for activation parameters")
    parser.add_argument("--n-layers", type=int, default=None, help="Transformer layer count override")
    parser.add_argument("--d-model", type=int, default=None, help="Model width override")
    parser.add_argument("--context-len", type=int, default=None, help="Context window (block_size) override")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--config", type=str, default="configs/phase2_complex_scale.json", help="Benchmark config file")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    parser.add_argument("--fresh", action="store_true", help="Ignore any saved epoch checkpoint and restart this seed from epoch 0")
    args = parser.parse_args()

    if args.benchmark:
        run_language_model_scale_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp, resume=not args.fresh)
    else:
        for seed in args.seeds:
            perplexity = train_single_seed_lm_scale(
                act_type=args.activation,
                seed=seed,
                epochs=args.epochs,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                data_root=args.data_root,
                alpha_lr=args.alpha_lr,
                n_layers=args.n_layers,
                d_model=args.d_model,
                context_len=args.context_len,
                config_path=args.config,
                save_artifacts=True,
                amp=args.amp,
                resume=not args.fresh,
            )
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | Perplexity: {perplexity:.2f}")
