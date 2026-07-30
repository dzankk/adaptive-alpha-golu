"""
Adaptive Alpha-GoLU Benchmark Suite CLI
========================================
Unified Command Line Interface for reproducing paper experiments, running 
multitask evaluations, and extracting statistical diagnostics

usage examples:
    python cli.py run --task classification --activation alpha_golu --seeds 42 123 999
    python cli.py run_all --seeds 42 123 999
    python cli.py analyze --results_dir outputs/
"""

import argparse
import sys
import os
from typing import List

# Import benchmark modules
from experiments.run_classification import train_and_eval as run_classification
from experiments.run_detection import train_and_eval as run_detection
from experiments.run_segmentation import train_and_eval as run_segmentation
from experiments.run_diffusion import train_and_eval as run_diffusion
from experiments.run_language_model import train_and_eval as run_language_model
from experiments.run_adversarial_robustness import train_and_eval as run_robustness

from utils.stats import compute_summary_statistics, calculate_p_value

TASK_MAP = {
    "classification": run_classification,
    "detection": run_detection,
    "segmentation": run_segmentation,
    "diffusion": run_diffusion,
    "language_model": run_language_model,
    "robustness": run_robustness,
}

SUPPORTED_ACTIVATIONS = ["gelu", "swish", "prelu", "pgelu", "golu_static", "alpha_golu"]


def handle_run(args):
    """Executes a specific benchmark task across specified random seeds."""
    task = args.task.lower()
    act = args.activation.lower()
    seeds = args.seeds

    if task not in TASK_MAP:
        print(f"[Error] Task '{task}' not recognized. Choose from: {list(TASK_MAP.keys())}")
        sys.exit(1)

    print(f"\n================ Running {task.upper()} | Activation: {act.upper()} ================")
    results = []
    for seed in seeds:
        print(f"\n---> Running Seed: {seed}")
        metric = TASK_MAP[task](act, seed=seed)
        results.append(metric)
        print(f"Seed {seed} Output Metric: {metric:.4f}")

    stats = compute_summary_statistics(results)
    print(f"\n[Summary - {task.upper()} - {act.upper()}]")
    print(f"Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | SEM: {stats['sem']:.4f}")


def handle_run_all(args):
    """Runs all benchmark tasks across selected activations and seeds."""
    seeds = args.seeds
    # Fix: Default to ALL supported activations if not provided
    activations = args.activations if args.activations else SUPPORTED_ACTIVATIONS
    
    print("\n================ Launching Full Paper Benchmark Suite ================")
    print(f"Activations to test ({len(activations)}): {', '.join(activations)}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    
    all_task_results = {}

    for task_name, runner_fn in TASK_MAP.items():
        print(f"\n\n################ Task: {task_name.upper()} ################")
        all_task_results[task_name] = {}
        
        for act in activations:
            print(f"\n--- Activation: {act.upper()} ---")
            accs = []
            for seed in seeds:
                acc = runner_fn(act, seed=seed)
                accs.append(acc)
                print(f"Seed {seed} -> Score: {acc:.4f}")
            
            stats = compute_summary_statistics(accs)
            all_task_results[task_name][act] = {
                "scores": accs,
                "mean": stats["mean"],
                "std": stats["std"],
                "sem": stats["sem"]
            }

        # Calculate p-value comparing Alpha-GoLU against Static GoLU if both present
        if "golu_static" in activations and "alpha_golu" in activations:
            p_val = calculate_p_value(
                all_task_results[task_name]["golu_static"]["scores"],
                all_task_results[task_name]["alpha_golu"]["scores"]
            )
            print(f"\n[Statistical Significance] Paired t-test (Alpha-GoLU vs Static): p = {p_val:.4f}")

    print("\n================ All Experiments Completed Successfully! ================")


def main():
    parser = argparse.ArgumentParser(
        description="CLI for Adaptive Alpha-GoLU Benchmark Suite & Paper Reproducibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a single task benchmark")
    run_parser.add_argument("--task", type=str, required=True, choices=list(TASK_MAP.keys()), help="Benchmark task")
    run_parser.add_argument("--activation", type=str, default="alpha_golu", choices=SUPPORTED_ACTIVATIONS, help="Activation function")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds for statistical testing")

    # Command: run_all
    run_all_parser = subparsers.add_parser("run_all", help="Reproduce all paper tables across all tasks")
    run_all_parser.add_argument(
        "--activations", 
        type=str, 
        nargs="+", 
        default=SUPPORTED_ACTIVATIONS,  # Fix: defaults to all 6 activations
        choices=SUPPORTED_ACTIVATIONS, 
        help="Activations to evaluate"
    )
    run_all_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999], help="Random seeds for statistical testing")

    args = parser.parse_args()

    if args.command == "run":
        handle_run(args)
    elif args.command == "run_all":
        handle_run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
