"""
Adaptive Alpha-GoLU Benchmark Suite CLI
========================================
Unified Command Line Interface for reproducing paper experiments, running 
multitask evaluations, and extracting statistical diagnostics

usage examples:
    python cli.py run --task classification --activation alpha_golu --seeds 42 123 999 2024 2025
    python cli.py run_all --seeds 42 123 999 2024 2025
    python cli.py generate_table --results_path outputs/benchmark_results.json
"""

import argparse
import json
import os
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Import benchmark modules
from experiments.run_classification import train_and_eval as run_classification
from experiments.run_detection import train_and_eval as run_detection
from experiments.run_segmentation import train_and_eval as run_segmentation
from experiments.run_diffusion import train_and_eval as run_diffusion
from experiments.run_language_model import train_and_eval as run_language_model
from experiments.run_adversarial_robustness import train_and_eval as run_robustness

from utils.experiment_config import load_benchmark_config
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.preflight import run_preflight_checks, save_preflight_report
from utils.stats import compute_summary_statistics, calculate_p_value

TASK_MAP = {
    "classification": run_classification,
    "detection": run_detection,
    "segmentation": run_segmentation,
    "diffusion": run_diffusion,
    "language_model": run_language_model,
    "robustness": run_robustness,
}

# Updated canonical set: includes standard static baselines and their parametric counterparts
SUPPORTED_ACTIVATIONS = [
    "relu",
    "gelu",
    "swish",
    "prelu",
    "pgelu",
    "adaptive_swish",
    "golu_static",
    "alpha_golu"
]

DEFAULT_SEEDS = [42, 123, 999, 2024, 2025]


def _stable_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _resume_state_path(output_root: str, scope: str, payload: dict) -> Path:
    return Path(output_root) / ".resume" / scope / f"{_stable_signature(payload)}.json"


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _save_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def _run_preflight_or_abort(data_root: str, output_root: str, min_free_gb: float = 20.0):
    project_root = Path(__file__).resolve().parents[0]
    report = run_preflight_checks(
        project_root=project_root,
        data_root=Path(data_root),
        output_root=Path(output_root),
        min_free_gb=min_free_gb,
    )
    report_dir = Path(output_root) / "preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = save_preflight_report(report, report_dir)

    print("\n================ Preflight Check ================")
    for item in report["checks"]:
        print(f"[{item['status'].upper():<5}] {item['check']}: {item['details']}")
    print(f"[IO] Preflight report saved to {report_path}")

    if report["errors"]:
        print("[Error] Preflight failed:")
        for message in report["errors"]:
            print(f"  - {message}")
        sys.exit(1)

    if report["warnings"]:
        print("[Warning] Preflight warnings:")
        for message in report["warnings"]:
            print(f"  - {message}")


def handle_run(args):
    """Executes a specific benchmark task across specified random seeds."""
    task = args.task.lower()
    act = args.activation.lower()
    seeds = args.seeds
    _run_preflight_or_abort(args.data_root, args.output_root, args.min_free_gb)
    resume_path = _resume_state_path(args.output_root, "single_task", {"task": task, "activation": act, "seeds": seeds})
    resume_state = _load_json_file(resume_path)
    saved_scores = resume_state.get("scores", {}) if isinstance(resume_state.get("scores", {}), dict) else {}

    if task not in TASK_MAP:
        print(f"[Error] Task '{task}' not recognized. Choose from: {list(TASK_MAP.keys())}")
        sys.exit(1)

    print(f"\n================ Running {task.upper()} | Activation: {act.upper()} ================")
    results = []
    for seed in seeds:
        seed_key = str(seed)
        if seed_key in saved_scores:
            metric = float(saved_scores[seed_key])
            results.append(metric)
            print(f"---> Skipping Seed: {seed} (already completed)")
            print(f"Seed {seed} Output Metric: {metric:.4f}")
            continue

        print(f"\n---> Running Seed: {seed}")
        metric = TASK_MAP[task](act, seed=seed)
        results.append(metric)
        print(f"Seed {seed} Output Metric: {metric:.4f}")
        saved_scores[seed_key] = metric
        _save_json_file(
            resume_path,
            {
                "task": task,
                "activation": act,
                "seeds": seeds,
                "scores": saved_scores,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    stats = compute_summary_statistics(results)
    run_dir = create_run_directory(args.output_root, task=task, activation=act, seeds=seeds)
    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command="run",
            task=task,
            seeds=seeds,
            activations=[act],
            extra_config={"activation": act},
        ),
    )
    write_json(
        run_dir / "results.json",
        {
            "task": task,
            "activation": act,
            "seeds": seeds,
            "scores": results,
            "summary": stats,
        },
    )
    print(f"\n[Summary - {task.upper()} - {act.upper()}]")
    print(f"Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | SEM: {stats['sem']:.4f}")
    print(f"[IO] Saved run artifacts to {run_dir}")


def handle_run_all(args):
    """Runs all benchmark tasks across selected activations and seeds."""
    config = load_benchmark_config(args.config) if args.config else {}
    seeds = args.seeds if args.seeds != DEFAULT_SEEDS else config.get("seeds", DEFAULT_SEEDS)
    activations = args.activations if args.activations != SUPPORTED_ACTIVATIONS else config.get("activations", SUPPORTED_ACTIVATIONS)
    task_names = config.get("tasks", list(TASK_MAP.keys()))
    output_root = config.get("output_root", "outputs/runs")
    summary_path = config.get("summary_path", "outputs/benchmark_results.json")
    _run_preflight_or_abort(config.get("data_root", "./data"), output_root, args.min_free_gb)

    selected_tasks = []
    for task_name in task_names:
        normalized_task = task_name.lower()
        if normalized_task not in TASK_MAP:
            print(f"[Error] Task '{normalized_task}' not recognized in config. Choose from: {list(TASK_MAP.keys())}")
            sys.exit(1)
        selected_tasks.append(normalized_task)
    
    print("\n================ Launching Full Paper Benchmark Suite ================")
    print(f"Activations to test ({len(activations)}): {', '.join(activations)}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    if args.config:
        print(f"[Config] Loaded benchmark config from {args.config}")

    resume_path = _resume_state_path(
        output_root,
        "full_suite",
        {"tasks": selected_tasks, "activations": activations, "seeds": seeds, "config": args.config},
    )
    resume_state = _load_json_file(resume_path)
    all_task_results = resume_state.get("all_task_results", {}) if isinstance(resume_state.get("all_task_results", {}), dict) else {}

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)

    for task_name in selected_tasks:
        runner_fn = TASK_MAP[task_name]
        print(f"\n\n################ Task: {task_name.upper()} ################")
        task_results = all_task_results.get(task_name, {}) if isinstance(all_task_results.get(task_name, {}), dict) else {}
        
        for act in activations:
            print(f"\n--- Activation: {act.upper()} ---")
            existing_entry = task_results.get(act, {}) if isinstance(task_results.get(act, {}), dict) else {}
            completed_scores = existing_entry.get("completed_scores", {}) if isinstance(existing_entry.get("completed_scores", {}), dict) else {}
            accs = [float(completed_scores[str(seed)]) for seed in seeds if str(seed) in completed_scores]

            for seed in seeds:
                seed_key = str(seed)
                if seed_key in completed_scores:
                    print(f"Seed {seed} -> Score: {float(completed_scores[seed_key]):.4f} (already completed)")
                    continue

                acc = runner_fn(act, seed=seed)
                accs.append(acc)
                print(f"Seed {seed} -> Score: {acc:.4f}")

                completed_scores[seed_key] = acc
                stats = compute_summary_statistics(accs)
                task_results[act] = {
                    "scores": accs,
                    "completed_scores": completed_scores,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "sem": stats["sem"],
                }
                all_task_results[task_name] = task_results
                _save_json_file(
                    resume_path,
                    {
                        "tasks": selected_tasks,
                        "activations": activations,
                        "seeds": seeds,
                        "all_task_results": all_task_results,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            
            stats = compute_summary_statistics(accs)
            task_results[act] = {
                "scores": accs,
                "completed_scores": completed_scores,
                "mean": stats["mean"],
                "std": stats["std"],
                "sem": stats["sem"]
            }
            all_task_results[task_name] = task_results

        if "golu_static" in activations and "alpha_golu" in activations:
            p_val = calculate_p_value(
                task_results["golu_static"]["scores"],
                task_results["alpha_golu"]["scores"]
            )
            all_task_results[task_name]["p_value_welch_alpha_vs_static"] = p_val
            print(f"\n[Statistical Significance] Welch t-test (Alpha-GoLU vs Static): p = {p_val:.4f}")

    json_path = summary_path
    with open(json_path, "w") as f:
        json.dump(all_task_results, f, indent=4)
    run_dir = create_run_directory(output_root, task="full_suite", activation="all", seeds=seeds)
    write_json(run_dir / "benchmark_results.json", all_task_results)
    if args.config:
        write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            command="run_all",
            task="full_suite",
            seeds=seeds,
            activations=activations,
            extra_config={
                "activations": activations,
                "tasks": selected_tasks,
                "summary_path": summary_path,
                "output_root": output_root,
                **({"config_path": args.config} if args.config else {}),
            },
        ),
    )
    if resume_path.exists():
        try:
            resume_path.unlink()
        except OSError:
            pass
    print(f"\n[IO] Saved benchmark JSON summary to {json_path}")
    print(f"[IO] Saved full-suite run artifacts to {run_dir}")
    print("\n================ All Experiments Completed Successfully! ================")


def handle_generate_table(args):
    """Generates a publication-ready LaTeX benchmark table from JSON results with bold/underline highlighting."""
    json_path = args.results_path
    if not os.path.exists(json_path):
        print(f"[Error] Benchmark results file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    tasks = list(data.keys())
    if not tasks:
        print("[Error] Results file contains no tasks.")
        return

    # Identify all activation keys present in data
    first_task = tasks[0]
    acts = [
        k for k in data[first_task].keys()
        if not k.startswith("p_value_") and k != "completed_scores"
    ]

    print("\n% ===== Auto-Generated Publication LaTeX Benchmark Table =====")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\begin{tabular}{l" + "c" * len(tasks) + "}")
    print("\\toprule")
    
    header = "Activation & " + " & ".join([f"\\textbf{{{t.replace('_', ' ').title()}}}" for t in tasks]) + " \\\\"
    print(header)
    print("\\midrule")

    # Find best (max/min) per task for formatting highlights
    # Note: For LM and Diffusion lower score is better; for Classification, Detection, Segmentation, Robustness higher is better.
    lower_is_better_tasks = {"language_model", "diffusion"}

    best_per_task = {}
    second_best_per_task = {}

    for t in tasks:
        means = {a: data[t][a]["mean"] for a in acts if a in data[t]}
        if not means:
            continue
        
        sorted_acts = sorted(means.keys(), key=lambda a: means[a], reverse=(t not in lower_is_better_tasks))
        best_per_task[t] = sorted_acts[0]
        second_best_per_task[t] = sorted_acts[1] if len(sorted_acts) > 1 else None

    # Render Activation Rows
    for act in acts:
        formatted_act_name = act.replace("_", "-").upper()
        row = [f"\\texttt{{{formatted_act_name}}}"]
        
        for t in tasks:
            if act in data[t]:
                m = data[t][act]["mean"]
                s = data[t][act]["std"]
                cell_str = f"{m:.2f} $\\pm$ {s:.2f}"
                
                if act == best_per_task.get(t):
                    cell_str = f"\\textbf{{{cell_str}}}"
                elif act == second_best_per_task.get(t):
                    cell_str = f"\\underline{{{cell_str}}}"
                
                row.append(cell_str)
            else:
                row.append("N/A")
        print(" & ".join(row) + " \\\\")

    print("\\midrule")
    
    # Render Welch t-test p-value row
    p_row = ["\\textit{$p$-value (Welch; $\\alpha$ vs Static)}"]
    for t in tasks:
        p_val = data[t].get("p_value_welch_alpha_vs_static", None)
        if p_val is not None:
            p_row.append(f"\\textit{{p = {p_val:.4f}}}")
        else:
            p_row.append("N/A")
    print(" & ".join(p_row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Empirical benchmark comparison across tasks. Best performance is in \\textbf{bold}; second best is \\underline{underlined}. Statistical significance is computed via Welch's $t$-test between Alpha-GoLU and GoLU Static.}")
    print("\\label{tab:benchmark_results}")
    print("\\end{table*}\n")


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
    run_parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds for statistical testing")
    run_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root used for preflight checks")
    run_parser.add_argument("--output-root", type=str, default="outputs/runs", help="Output root used for preflight checks")
    run_parser.add_argument("--min-free-gb", type=float, default=20.0, help="Minimum free disk space required before launching")

    # Command: run_all
    run_all_parser = subparsers.add_parser("run_all", help="Reproduce all paper tables across all tasks")
    run_all_parser.add_argument(
        "--activations", 
        type=str, 
        nargs="+", 
        default=SUPPORTED_ACTIVATIONS, 
        choices=SUPPORTED_ACTIVATIONS, 
        help="Activations to evaluate"
    )
    run_all_parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds for statistical testing")
    run_all_parser.add_argument("--config", type=str, default=None, help="Path to a JSON benchmark config file")
    run_all_parser.add_argument("--min-free-gb", type=float, default=20.0, help="Minimum free disk space required before launching")

    # Command: preflight
    preflight_parser = subparsers.add_parser("preflight", help="Run environment and storage checks before a long benchmark run")
    preflight_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root to check")
    preflight_parser.add_argument("--output-root", type=str, default="outputs/preflight", help="Directory where the preflight report will be saved")
    preflight_parser.add_argument("--min-free-gb", type=float, default=20.0, help="Minimum free disk space required")

    # Command: generate_table
    table_parser = subparsers.add_parser("generate_table", help="Generate LaTeX table from saved JSON results")
    table_parser.add_argument("--results_path", type=str, default="outputs/benchmark_results.json", help="Path to JSON results file")

    args = parser.parse_args()

    if args.command == "run":
        handle_run(args)
    elif args.command == "run_all":
        handle_run_all(args)
    elif args.command == "generate_table":
        handle_generate_table(args)
    elif args.command == "preflight":
        _run_preflight_or_abort(args.data_root, args.output_root, args.min_free_gb)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
