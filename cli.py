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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
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
from utils.data_prep import prepare_all_datasets
from utils.stats import compute_summary_statistics, calculate_p_value
from utils.visualization import plot_paper_alpha_trajectories, plot_paper_benchmark_summary, plot_paper_overhead_summary, plot_parametric_comparison

TASK_MAP = {
    "classification": run_classification,
    "detection": run_detection,
    "segmentation": run_segmentation,
    "diffusion": run_diffusion,
    "language_model": run_language_model,
    "robustness": run_robustness,
}

# Updated canonical set: includes standard static baselines and their parametric counterparts.
CANONICAL_ACTIVATIONS = [
    "relu",
    "gelu",
    "swish",
    "prelu",
    "pgelu",
    "adaptive_swish",
    "golu_static",
    "alpha_golu"
]

SUPPORTED_ACTIVATIONS = CANONICAL_ACTIVATIONS + ["swish_adaptive"]

PARAMETRIC_ACTIVATIONS = ["alpha_golu", "prelu", "pgelu", "adaptive_swish", "swish_adaptive"]

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


def _load_overhead_records(overhead_root: str) -> list[dict]:
    root_path = Path(overhead_root)
    if not root_path.exists():
        return []

    records = []
    for json_path in sorted(root_path.rglob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(payload, dict):
            payload.setdefault("file_path", str(json_path))
            records.append(payload)
    return records


def _aggregate_overhead_records(records: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        task_name = str(record.get("task_name") or record.get("task") or "").strip()
        activation_name = str(record.get("activation_name") or record.get("activation") or "").strip()
        if not task_name or not activation_name:
            continue
        grouped[activation_name][task_name].append(record)
    return grouped


def _metric_mean(records: list[dict], key: str, nested_key: str | None = None) -> float | None:
    values = []
    for record in records:
        value: float | None
        if nested_key is None:
            raw_value = record.get(key)
            value = float(raw_value) if isinstance(raw_value, (int, float)) else None
        else:
            nested = record.get(key, {})
            if not isinstance(nested, dict):
                value = None
            else:
                raw_value = nested.get(nested_key)
                value = float(raw_value) if isinstance(raw_value, (int, float)) else None
        if value is not None:
            values.append(value)
    return float(mean(values)) if values else None


def _metric_mean_with_aliases(records: list[dict], *, primary_key: str, nested_key: str | None = None, aliases: list[str] | None = None) -> float | None:
    for key in [primary_key, *(aliases or [])]:
        value = _metric_mean(records, key, nested_key)
        if value is not None:
            return value
    return None


def _format_metric(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def handle_generate_overhead_table(args):
    """Generates a publication-ready LaTeX overhead table from saved overhead JSON records."""
    records = _load_overhead_records(args.overhead_root)
    if not records:
        print(f"[Error] No overhead JSON records found under {args.overhead_root}")
        return

    grouped = _aggregate_overhead_records(records)
    if not grouped:
        print(f"[Error] No usable overhead records found under {args.overhead_root}")
        return

    task_order = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]
    task_labels = {
        "classification": "Classification",
        "detection": "Detection",
        "segmentation": "Segmentation",
        "diffusion": "Diffusion",
        "language_model": "Language Modeling",
        "robustness": "Robustness",
    }

    activations = [act for act in CANONICAL_ACTIVATIONS if act in grouped]
    for act in sorted(grouped.keys()):
        if act not in activations:
            activations.append(act)

    lines = [
        "% ===== Auto-Generated Publication LaTeX Overhead Table =====",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l" + "c" * len(task_order) + r"}",
        r"\toprule",
            "Activation & " + " & ".join([f"\\textbf{{{task_labels[task]}}}" for task in task_order]) + r" \\",
        r"\midrule",
    ]

    for activation in activations:
        row = [f"\\texttt{{{activation.replace('_', '-').upper()}}}"]
        for task in task_order:
            task_records = grouped.get(activation, {}).get(task, [])
            if not task_records:
                row.append("N/A")
                continue

            forward_ms = _metric_mean_with_aliases(
                task_records,
                primary_key="forward_latency_ms",
                aliases=["forward_ms"],
            )
            backward_ms = _metric_mean_with_aliases(
                task_records,
                primary_key="backward_latency_ms",
                aliases=["backward_ms"],
            )
            peak_mb = _metric_mean(task_records, "peak_cuda_memory_mb")
            total_flops = _metric_mean(task_records, "estimated_total_flops")
            cell = (
                r"\shortstack{"
                + f"Fwd {_format_metric(forward_ms)} ms" + r"\\"
                + f"Bwd {_format_metric(backward_ms)} ms" + r"\\"
                + f"Mem {_format_metric(peak_mb, digits=1)} MB" + r"\\"
                + f"FLOPs {_format_metric(None if total_flops is None else total_flops / 1e6, digits=1, suffix='M')}"
                + r"}"
            )
            row.append(cell)
        lines.append(" & ".join(row) + " " + chr(92) + chr(92))

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Mean overhead summary aggregated from saved benchmark runs. Each cell reports forward latency, backward latency, peak CUDA memory, and estimated total FLOPs for the corresponding activation-task pair.}",
        r"\label{tab:overhead_results}",
        r"\end{table*}",
    ])

    latex_table = "\n".join(lines) + "\n"
    print(latex_table)

    output_path = Path(args.output_path) if args.output_path else Path(args.overhead_root) / "overhead_table.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(latex_table)
    print(f"[IO] Saved overhead LaTeX table to {output_path}")


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
    config = load_benchmark_config(args.config) if args.config else {}
    task_alpha_lr = config.get("alpha_lr_by_task", {}).get(task)
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
        metric = TASK_MAP[task](act, seed=seed, data_root=args.data_root, alpha_lr=task_alpha_lr, save_artifacts=args.save_artifacts, amp=args.amp)
        results.append(metric)
        print(f"Seed {seed} Output Metric: {metric:.4f}")
        saved_scores[seed_key] = metric
        _save_json_file(
            resume_path,
            {
                "task": task,
                "activation": act,
                "seeds": seeds,
                "amp": bool(args.amp),
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
            extra_config={"activation": act, "amp": bool(args.amp)},
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
    activations = args.activations if args.activations != CANONICAL_ACTIVATIONS else config.get("activations", CANONICAL_ACTIVATIONS)
    task_names = args.tasks if args.tasks else config.get("tasks", list(TASK_MAP.keys()))
    task_alpha_lrs = config.get("alpha_lr_by_task", {})
    output_root = config.get("output_root", "outputs/runs")
    summary_path = config.get("summary_path", "outputs/benchmark_results.json")
    data_root = config.get("data_root", args.data_root)
    _run_preflight_or_abort(data_root, output_root, args.min_free_gb)

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
        {"tasks": selected_tasks, "activations": activations, "seeds": seeds, "config": args.config, "amp": bool(args.amp)},
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

                alpha_lr = task_alpha_lrs.get(task_name)
                acc = runner_fn(act, seed=seed, data_root=data_root, alpha_lr=alpha_lr, save_artifacts=args.save_artifacts, amp=args.amp)
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
                    "amp": bool(args.amp),
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


def _build_benchmark_latex_table(data: dict) -> str:
    tasks = list(data.keys())
    if not tasks:
        return "% ===== Auto-Generated Publication LaTeX Benchmark Table =====\n% Empty benchmark results.\n"

    first_task = tasks[0]
    acts = [k for k in data[first_task].keys() if not k.startswith("p_value_") and k != "completed_scores"]
    lower_is_better_tasks = {"language_model", "diffusion"}

    best_per_task = {}
    second_best_per_task = {}
    for task_name in tasks:
        means = {activation: data[task_name][activation]["mean"] for activation in acts if activation in data[task_name]}
        if not means:
            continue
        sorted_acts = sorted(means.keys(), key=lambda activation: means[activation], reverse=(task_name not in lower_is_better_tasks))
        best_per_task[task_name] = sorted_acts[0]
        second_best_per_task[task_name] = sorted_acts[1] if len(sorted_acts) > 1 else None

    lines = [
        r"% ===== Auto-Generated Publication LaTeX Benchmark Table =====",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(tasks) + r"}",
        r"\toprule",
        "Activation & " + " & ".join([f"\\textbf{{{task.replace('_', ' ').title()}}}" for task in tasks]) + r" \\",
        r"\midrule",
    ]

    for activation in acts:
        formatted_act_name = activation.replace("_", "-").upper()
        row = [f"\\texttt{{{formatted_act_name}}}"]
        for task_name in tasks:
            if activation in data[task_name]:
                mean_value = data[task_name][activation]["mean"]
                std_value = data[task_name][activation]["std"]
                cell_str = f"{mean_value:.2f} $\\pm$ {std_value:.2f}"
                if activation == best_per_task.get(task_name):
                    cell_str = f"\\textbf{{{cell_str}}}"
                elif activation == second_best_per_task.get(task_name):
                    cell_str = f"\\underline{{{cell_str}}}"
                row.append(cell_str)
            else:
                row.append("N/A")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")
    p_row = [r"\textit{$p$-value (Welch; $\alpha$ vs Static)}"]
    for task_name in tasks:
        p_val = data[task_name].get("p_value_welch_alpha_vs_static", None)
        if p_val is not None:
            p_row.append(f"\\textit{{p = {p_val:.4f}}}")
        else:
            p_row.append("N/A")
    lines.append(" & ".join(p_row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Empirical benchmark comparison across tasks. Best performance is in \textbf{bold}; second best is \underline{underlined}. Statistical significance is computed via Welch's $t$-test between Alpha-GoLU and GoLU Static.}",
        r"\label{tab:benchmark_results}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def handle_generate_paper_assets(args):
    """Generates paper-ready tables and figures from saved benchmark artifacts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(args.results_path):
        with open(args.results_path, "r", encoding="utf-8") as handle:
            benchmark_data = json.load(handle)
        if isinstance(benchmark_data, dict) and benchmark_data:
            table_path = output_dir / "benchmark_table.tex"
            table_path.write_text(_build_benchmark_latex_table(benchmark_data), encoding="utf-8")
            print(f"[IO] Saved benchmark LaTeX table to {table_path}")
            plot_paper_benchmark_summary(args.results_path, str(output_dir))
    else:
        print(f"[Warning] Benchmark results not found at {args.results_path}")

    if os.path.exists(args.overhead_root):
        overhead_args = argparse.Namespace(overhead_root=args.overhead_root, output_path=str(output_dir / "overhead_table.tex"))
        handle_generate_overhead_table(overhead_args)
        plot_paper_overhead_summary(args.overhead_root, str(output_dir))
    else:
        print(f"[Warning] Overhead root not found at {args.overhead_root}")

    plot_paper_alpha_trajectories(args.runs_root, str(output_dir), activation_name=args.activation)
    print(f"[IO] Paper assets written to {output_dir}")


def handle_generate_parametric_comparison_plot(args):
    """Plots layer-averaged trajectories for parametric activations from saved run JSON files."""
    if args.run_jsons:
        run_jsons = [str(Path(path)) for path in args.run_jsons]
    else:
        runs_root = Path(args.runs_root)
        run_jsons = []
        if runs_root.exists():
            for result_path in runs_root.rglob("results.json"):
                payload = _load_json_file(result_path)
                if not payload:
                    continue
                activation_name = str(payload.get("activation", payload.get("activation_name", ""))).lower().strip()
                task_name = str(payload.get("task", "")).lower().strip()
                if args.task and task_name != args.task.lower().strip():
                    continue
                if activation_name in PARAMETRIC_ACTIVATIONS:
                    run_jsons.append(str(result_path))

    if not run_jsons:
        print("[Error] No parametric run JSONs found to plot")
        return

    plot_parametric_comparison(run_jsons, save_path=args.output_path, task_name=args.task)


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
    run_parser.add_argument("--save-artifacts", action="store_true", help="Save per-seed run JSONs, manifests, and trajectory plots")
    run_parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")

    # Command: run_all
    run_all_parser = subparsers.add_parser("run_all", help="Reproduce all paper tables across all tasks")
    run_all_parser.add_argument(
        "--activations", 
        type=str, 
        nargs="+", 
        default=CANONICAL_ACTIVATIONS, 
        choices=SUPPORTED_ACTIVATIONS, 
        help="Activations to evaluate"
    )
    run_all_parser.add_argument("--tasks", type=str, nargs="+", default=None, choices=list(TASK_MAP.keys()), help="Subset of tasks to run instead of the config task list")
    run_all_parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds for statistical testing")
    run_all_parser.add_argument("--config", type=str, default=None, help="Path to a JSON benchmark config file")
    run_all_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root used when config does not specify one")
    run_all_parser.add_argument("--min-free-gb", type=float, default=20.0, help="Minimum free disk space required before launching")
    run_all_parser.add_argument("--save-artifacts", action="store_true", help="Save per-seed run JSONs, manifests, and trajectory plots")
    run_all_parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")

    # Command: preflight
    preflight_parser = subparsers.add_parser("preflight", help="Run environment and storage checks before a long benchmark run")
    preflight_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root to check")
    preflight_parser.add_argument("--output-root", type=str, default="outputs/preflight", help="Directory where the preflight report will be saved")
    preflight_parser.add_argument("--min-free-gb", type=float, default=20.0, help="Minimum free disk space required")

    # Command: prepare_data
    prep_parser = subparsers.add_parser("prepare_data", help="Download and cache all benchmark datasets without training")
    prep_parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")

    # Command: generate_table
    table_parser = subparsers.add_parser("generate_table", help="Generate LaTeX table from saved JSON results")
    table_parser.add_argument("--results_path", type=str, default="outputs/benchmark_results.json", help="Path to JSON results file")

    # Command: generate_overhead_table
    overhead_parser = subparsers.add_parser("generate_overhead_table", help="Generate LaTeX overhead table from saved overhead JSON records")
    overhead_parser.add_argument("--overhead-root", type=str, default="outputs/overhead", help="Directory containing overhead JSON records")
    overhead_parser.add_argument("--output-path", type=str, default=None, help="Optional path where the LaTeX table will be written")

    # Command: generate_paper_assets
    paper_parser = subparsers.add_parser("generate_paper_assets", help="Generate paper-ready tables and figures from saved artifacts")
    paper_parser.add_argument("--results-path", type=str, default="outputs/benchmark_results.json", help="Path to benchmark summary JSON")
    paper_parser.add_argument("--overhead-root", type=str, default="outputs/overhead", help="Directory containing overhead JSON records")
    paper_parser.add_argument("--runs-root", type=str, default="outputs/runs", help="Directory containing benchmark run artifacts")
    paper_parser.add_argument("--output-dir", type=str, default="outputs/paper_assets", help="Directory where paper assets will be written")
    paper_parser.add_argument("--activation", type=str, default="alpha_golu", choices=SUPPORTED_ACTIVATIONS, help="Activation to use when scanning trajectory plots")

    # Command: generate_parametric_comparison_plot
    parametric_parser = subparsers.add_parser("generate_parametric_comparison_plot", help="Plot layer-averaged parameter adaptation curves for parametric activations")
    parametric_parser.add_argument("--run-jsons", type=str, nargs="*", default=None, help="Explicit run JSON paths to compare")
    parametric_parser.add_argument("--runs-root", type=str, default="outputs/runs", help="Directory to scan for run JSONs if no explicit paths are provided")
    parametric_parser.add_argument("--task", type=str, default=None, choices=list(TASK_MAP.keys()), help="Optional task filter when scanning output runs")
    parametric_parser.add_argument("--output-path", type=str, default="outputs/paper_assets/parametric_comparison.png", help="Where to save the comparison figure")

    args = parser.parse_args()

    if args.command == "run":
        handle_run(args)
    elif args.command == "run_all":
        handle_run_all(args)
    elif args.command == "generate_table":
        handle_generate_table(args)
    elif args.command == "generate_overhead_table":
        handle_generate_overhead_table(args)
    elif args.command == "generate_paper_assets":
        handle_generate_paper_assets(args)
    elif args.command == "generate_parametric_comparison_plot":
        handle_generate_parametric_comparison_plot(args)
    elif args.command == "preflight":
        _run_preflight_or_abort(args.data_root, args.output_root, args.min_free_gb)
    elif args.command == "prepare_data":
        report = prepare_all_datasets(args.data_root)
        print(f"\n[IO] Prepared datasets at {report['root']}: {', '.join(report['datasets'])}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
