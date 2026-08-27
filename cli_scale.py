"""
Phase 2 Scale-Up CLI (standalone)
==================================
Separate, dedicated entrypoint for the Phase 2 scaled-up benchmark suite.
Fully independent of cli.py -- it only imports the new experiments/run_*_scale.py
runners and shared utils/ helpers, so nothing about Phase 1's cli.py or its
experiment runners is touched.

Usage:
    python cli_scale.py run --task classification --activation alpha_golu --seeds 42 123 999
    python cli_scale.py run_all --tasks classification language_model diffusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_classification_scale import train_and_eval as run_classification_scale
from experiments.run_language_model_scale import train_and_eval as run_language_model_scale
from experiments.run_diffusion_scale import train_and_eval as run_diffusion_scale

from utils.experiment_config import load_benchmark_config
from utils.run_artifacts import write_json
from utils.stats import calculate_p_value, compute_summary_statistics

TASK_MAP = {
    "classification": run_classification_scale,
    "language_model": run_language_model_scale,
    "diffusion": run_diffusion_scale,
}

DEFAULT_CONFIG = "configs/phase2_complex_scale.json"
DEFAULT_EPOCHS_BY_TASK = {"classification": 120, "language_model": 20, "diffusion": 60}


def _resolve_epochs(config: dict, task_name: str) -> int:
    epochs_by_task = config.get("epochs_by_task", {}) if isinstance(config, dict) else {}
    value = epochs_by_task.get(task_name)
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_EPOCHS_BY_TASK.get(task_name, 20)


def handle_run(args) -> None:
    config = load_benchmark_config(args.config) if args.config else {}
    if args.task not in TASK_MAP:
        print(f"[Error] Unknown Phase 2 task '{args.task}'. Choose from: {list(TASK_MAP.keys())}")
        sys.exit(1)

    runner_fn = TASK_MAP[args.task]
    epochs = args.epochs if args.epochs is not None else _resolve_epochs(config, args.task)

    scores = []
    for seed in args.seeds:
        metric = runner_fn(
            activation=args.activation,
            seed=seed,
            epochs=epochs,
            data_root=args.data_root,
            config_path=args.config,
            save_artifacts=not args.no_save_artifacts,
            amp=args.amp,
            resume=not args.fresh,
        )
        print(f"[PHASE2][{args.task.upper()}][{args.activation}] seed={seed} -> {metric:.4f}")
        scores.append(metric)

    stats = compute_summary_statistics(scores)
    print(f"[PHASE2][{args.task.upper()}][{args.activation}] mean={stats['mean']:.4f} std={stats['std']:.4f} sem={stats['sem']:.4f}")


def handle_run_all(args) -> None:
    config = load_benchmark_config(args.config) if args.config else {}
    tasks = args.tasks or config.get("tasks", list(TASK_MAP.keys()))
    activations = args.activations or config.get("activations", ["golu_static", "alpha_golu"])
    seeds = args.seeds or config.get("seeds", [42, 123, 999])

    all_results: dict = {}
    for task_name in tasks:
        if task_name not in TASK_MAP:
            print(f"[Warning] Skipping unknown Phase 2 task '{task_name}'")
            continue

        runner_fn = TASK_MAP[task_name]
        epochs = _resolve_epochs(config, task_name)
        task_results: dict = {}
        print(f"\n################ Phase 2 Task: {task_name.upper()} ################")

        for act in activations:
            print(f"\n--- Activation: {act.upper()} ---")
            scores = []
            for seed in seeds:
                metric = runner_fn(
                    activation=act,
                    seed=seed,
                    epochs=epochs,
                    data_root=args.data_root,
                    config_path=args.config,
                    save_artifacts=not args.no_save_artifacts,
                    amp=args.amp,
                    resume=not args.fresh,
                )
                print(f"Seed {seed} -> Score: {metric:.4f}")
                scores.append(metric)
            stats = compute_summary_statistics(scores)
            task_results[act] = {"scores": scores, **stats}

        if "golu_static" in task_results and "alpha_golu" in task_results:
            task_results["p_value_welch_alpha_vs_static"] = calculate_p_value(
                task_results["golu_static"]["scores"], task_results["alpha_golu"]["scores"]
            )

        all_results[task_name] = task_results

    summary_path = Path(config.get("summary_path", "outputs/benchmark_results_phase2.json"))
    write_json(summary_path, all_results)
    print(f"\n[IO] Phase 2 scale-up summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 scale-up benchmark CLI (standalone, separate from cli.py)")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a single Phase 2 scale-up task/activation")
    run_parser.add_argument("--task", type=str, required=True, choices=list(TASK_MAP.keys()), help="Phase 2 task to run")
    run_parser.add_argument("--activation", type=str, default="alpha_golu", help="Activation function")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds")
    run_parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: from config or task default)")
    run_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Phase 2 benchmark config file")
    run_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    run_parser.add_argument("--no-save-artifacts", action="store_true", help="Disable saving per-seed run JSONs/manifests")
    run_parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    run_parser.add_argument("--fresh", action="store_true", help="Ignore any saved epoch checkpoint and restart this seed from epoch 0")

    run_all_parser = subparsers.add_parser("run_all", help="Run the full Phase 2 sweep across tasks/activations")
    run_all_parser.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASK_MAP.keys()), help="Subset of Phase 2 tasks (default: config's tasks list)")
    run_all_parser.add_argument("--activations", type=str, nargs="+", default=None, help="Activations to evaluate (default: config's activations list)")
    run_all_parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds (default: config's seeds list)")
    run_all_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Phase 2 benchmark config file")
    run_all_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    run_all_parser.add_argument("--no-save-artifacts", action="store_true", help="Disable saving per-seed run JSONs/manifests")
    run_all_parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    run_all_parser.add_argument("--fresh", action="store_true", help="Ignore any saved epoch checkpoint and restart every seed from epoch 0")

    args = parser.parse_args()
    if args.command == "run":
        handle_run(args)
    elif args.command == "run_all":
        handle_run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
