"""
Scaled Benchmark Significance & Timing Logger
==============================================
Standalone reporting utility: scans saved per-seed run artifacts
(outputs/runs/<task>/**/results.json), computes paired-seed statistical
significance (p-value, Cohen's d) between activations and a baseline, and
compares mean epoch/train wall-clock time and inference throughput across
activations. Writes a human-readable report to
outputs/reports/significance_and_scale_analysis.log.

Usage:
    python -m utils.scaled_benchmark_logger \
        --tasks language_model detection \
        --activations golu_static alpha_golu \
        --baseline golu_static
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.stats import (
    calculate_cohens_d,
    calculate_p_value,
    calculate_paired_cohens_d,
    compute_summary_statistics,
    format_timing_table,
    summarize_timing_by_activation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mirrors cli.py's task->run-folder mapping and task->metric-key mapping (duplicated here,
# rather than imported, to avoid this lightweight reporting module depending on the heavy
# top-level cli.py entry point).
TASK_RUN_FOLDER = {
    "classification": "classification",
    "detection": "detection",
    "segmentation": "segmentation",
    "diffusion": "diffusion",
    "language_model": "language_model",
    "robustness": "corruption_robustness",
}

TASK_METRIC_KEYS = {
    "classification": "accuracy",
    "detection": "map50",
    "segmentation": "miou",
    "diffusion": "mse",
    "language_model": "perplexity",
    "robustness": "corruption_acc",
}

LOWER_IS_BETTER_TASKS = {"diffusion", "language_model"}


def _task_run_root(task_name: str, output_root: str | Path = PROJECT_ROOT / "outputs" / "runs") -> Path:
    folder = TASK_RUN_FOLDER.get(task_name, task_name)
    return Path(output_root) / folder


_SEED_SUFFIX_RE = re.compile(r"_seeds-(\d+)$")


def load_results_by_activation(
    task_name: str,
    activations: List[str],
    *,
    output_root: str | Path = PROJECT_ROOT / "outputs" / "runs",
) -> Dict[str, List[Dict[str, Any]]]:
    """Loads the most recent saved results.json per seed for `task_name`, for each activation.

    Results are ordered by ascending seed number (not by directory timestamp), since two
    activations' run directories are created at unrelated times -- sorting by timestamp would
    silently misalign index i between activations and corrupt any paired-seed comparison
    (e.g. calculate_p_value(..., paired=True)). If a seed has multiple historical run
    directories (re-runs), only the most recently modified one is used.
    """
    run_root = _task_run_root(task_name, output_root)
    results_by_activation: Dict[str, List[Dict[str, Any]]] = {activation: [] for activation in activations}
    if not run_root.exists():
        return results_by_activation

    for activation in activations:
        latest_by_seed: Dict[int, tuple[float, Dict[str, Any]]] = {}
        for run_dir in run_root.glob(f"*_{task_name}_{activation}_seeds-*"):
            match = _SEED_SUFFIX_RE.search(run_dir.name)
            if match is None:
                continue
            seed = int(match.group(1))
            result_path = run_dir / "results.json"
            if not result_path.exists():
                continue
            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue

            mtime = run_dir.stat().st_mtime
            existing = latest_by_seed.get(seed)
            if existing is None or mtime >= existing[0]:
                latest_by_seed[seed] = (mtime, payload)

        results_by_activation[activation] = [payload for _, payload in (latest_by_seed[seed] for seed in sorted(latest_by_seed))]

    return results_by_activation



def _metric_scores(task_name: str, payloads: List[Dict[str, Any]]) -> List[float]:
    metric_key = TASK_METRIC_KEYS.get(task_name)
    if not metric_key:
        return []
    scores = []
    for payload in payloads:
        value = payload.get(metric_key)
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def build_significance_report(
    tasks: List[str],
    activations: List[str],
    *,
    baseline_activation: str = "golu_static",
    proposed_activation: str = "alpha_golu",
    output_root: str | Path = PROJECT_ROOT / "outputs" / "runs",
) -> Dict[str, Any]:
    """Computes significance and timing summaries for each task, returned as a plain dict."""
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_activation": baseline_activation,
        "proposed_activation": proposed_activation,
        "tasks": {},
    }

    for task_name in tasks:
        results_by_activation = load_results_by_activation(task_name, activations, output_root=output_root)
        timing_summary = summarize_timing_by_activation(results_by_activation, baseline_activation=baseline_activation)

        task_report: Dict[str, Any] = {
            "n_runs_by_activation": {activation: len(payloads) for activation, payloads in results_by_activation.items()},
            "timing": timing_summary,
        }

        baseline_scores = _metric_scores(task_name, results_by_activation.get(baseline_activation, []))
        proposed_scores = _metric_scores(task_name, results_by_activation.get(proposed_activation, []))
        task_report["metric_key"] = TASK_METRIC_KEYS.get(task_name)
        task_report["lower_is_better"] = task_name in LOWER_IS_BETTER_TASKS
        task_report["baseline_scores"] = baseline_scores
        task_report["proposed_scores"] = proposed_scores
        task_report["baseline_stats"] = compute_summary_statistics(baseline_scores)
        task_report["proposed_stats"] = compute_summary_statistics(proposed_scores)

        if len(baseline_scores) >= 2 and len(proposed_scores) >= 2:
            task_report["p_value_paired"] = calculate_p_value(baseline_scores, proposed_scores, paired=True)
            task_report["p_value_welch"] = calculate_p_value(baseline_scores, proposed_scores, paired=False)
            task_report["cohens_d"] = calculate_cohens_d(baseline_scores, proposed_scores)
            task_report["cohens_d_paired"] = calculate_paired_cohens_d(baseline_scores, proposed_scores)
        else:
            task_report["p_value_paired"] = None
            task_report["p_value_welch"] = None
            task_report["cohens_d"] = None
            task_report["cohens_d_paired"] = None

        report["tasks"][task_name] = task_report

    return report


def format_significance_report(report: Dict[str, Any]) -> str:
    """Renders `build_significance_report`'s output as a human-readable log."""
    baseline_activation = report["baseline_activation"]
    proposed_activation = report["proposed_activation"]
    lines = [
        "=" * 78,
        "Significance & Scale-Up Analysis",
        f"Generated (UTC): {report['generated_utc']}",
        f"Baseline: {baseline_activation}   Proposed: {proposed_activation}",
        "=" * 78,
    ]

    for task_name, task_report in report["tasks"].items():
        lines.append("")
        lines.append(f"--- Task: {task_name.upper()} ---")
        lines.append(f"metric: {task_report['metric_key']}  (lower_is_better={task_report['lower_is_better']})")
        lines.append(f"n_runs_by_activation: {task_report['n_runs_by_activation']}")

        b_stats, p_stats = task_report["baseline_stats"], task_report["proposed_stats"]
        lines.append(
            f"{baseline_activation}: mean={b_stats['mean']:.4f} std={b_stats['std']:.4f} sem={b_stats['sem']:.4f}  "
            f"scores={task_report['baseline_scores']}"
        )
        lines.append(
            f"{proposed_activation}: mean={p_stats['mean']:.4f} std={p_stats['std']:.4f} sem={p_stats['sem']:.4f}  "
            f"scores={task_report['proposed_scores']}"
        )

        p_paired = task_report["p_value_paired"]
        p_welch = task_report["p_value_welch"]
        d = task_report["cohens_d"]
        d_paired = task_report["cohens_d_paired"]
        if p_paired is None:
            lines.append("significance: not enough seeds (need >= 2 per activation)")
        else:
            lines.append(
                f"significance: p_paired={p_paired:.4f}  p_welch={p_welch:.4f}  "
                f"cohens_d={d:.3f}  cohens_d_paired={d_paired:.3f}"
            )

        lines.append("")
        lines.append("timing (mean per run, relative to baseline):")
        lines.append(format_timing_table(task_report["timing"], baseline_activation=baseline_activation))

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def write_significance_report(
    tasks: List[str],
    activations: List[str],
    *,
    baseline_activation: str = "golu_static",
    proposed_activation: str = "alpha_golu",
    output_root: str | Path = PROJECT_ROOT / "outputs" / "runs",
    log_path: str | Path = PROJECT_ROOT / "outputs" / "reports" / "significance_and_scale_analysis.log",
) -> Path:
    """Builds the report and appends it to the dedicated significance/scale-analysis log file."""
    report = build_significance_report(
        tasks,
        activations,
        baseline_activation=baseline_activation,
        proposed_activation=proposed_activation,
        output_root=output_root,
    )
    rendered = format_significance_report(report)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n\n")

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Significance and scale-up timing analysis report")
    parser.add_argument("--tasks", type=str, nargs="+", required=True, help="Tasks to analyze (e.g. language_model detection)")
    parser.add_argument("--activations", type=str, nargs="+", default=["golu_static", "alpha_golu"], help="Activations to include in the timing table")
    parser.add_argument("--baseline", type=str, default="golu_static", help="Baseline activation for comparison")
    parser.add_argument("--proposed", type=str, default="alpha_golu", help="Proposed activation for comparison")
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "outputs" / "runs"), help="Root directory containing outputs/runs/<task>/** run artifacts")
    parser.add_argument("--log-path", type=str, default=str(PROJECT_ROOT / "outputs" / "reports" / "significance_and_scale_analysis.log"), help="Destination log file")
    args = parser.parse_args()

    activations = list(dict.fromkeys([*args.activations, args.baseline, args.proposed]))
    log_path = write_significance_report(
        args.tasks,
        activations,
        baseline_activation=args.baseline,
        proposed_activation=args.proposed,
        output_root=args.output_root,
        log_path=args.log_path,
    )
    print(f"[IO] Significance and scale-up analysis appended to {log_path}")


if __name__ == "__main__":
    main()
