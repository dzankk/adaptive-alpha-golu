"""
 Visualization Suite
========================================
Generates vector-grade figures depicting trajectory dynamics, variance reduction,
and accuracy comparisons for final reports and paper submissions.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from typing import Dict, Any


TASK_LABELS = {
    "classification": "Classification",
    "detection": "Detection",
    "segmentation": "Segmentation",
    "diffusion": "Diffusion",
    "language_model": "Language Modeling",
    "robustness": "Corruption Robustness",
    "classification_scale": "Classification (Scale-Up)",
    "diffusion_scale": "Diffusion (Scale-Up)",
    "language_model_scale": "Language Modeling (Scale-Up)",
}

LOWER_IS_BETTER = {"diffusion", "language_model", "diffusion_scale", "language_model_scale"}

TASK_ORDER = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]

PHASE2_TASK_ORDER = ["classification_scale", "diffusion_scale", "language_model_scale"]

# Legacy/alternate top-level task keys to also check when a benchmark summary or overhead
# record was produced before the "robustness" run-folder/task-key naming was standardized.
TASK_ALIASES = {
    "robustness": ["corruption_robustness", "adversarial_robustness"],
}

DEFAULT_OVERHEAD_ACTIVATIONS = ["relu", "gelu", "swish", "prelu", "pgelu", "golu_static", "alpha_golu", "adaptive_swish", "swish_adaptive"]

PARAMETRIC_ACTIVATION_ORDER = ["alpha_golu", "prelu", "pgelu", "adaptive_swish", "swish_adaptive"]

PARAMETRIC_ACTIVATION_LABELS = {
    "alpha_golu": r"Alpha-GoLU ($\alpha$)",
    "adaptive_alpha_golu": r"Alpha-GoLU ($\alpha$)",
    "prelu": r"PReLU ($a$)",
    "pgelu": r"PGELU ($\alpha$)",
    "adaptive_swish": r"Parametric Swish ($\beta$)",
    "swish_adaptive": r"Parametric Swish ($\beta$)",
}


def _grid_shape(num_panels: int, max_cols: int = 3) -> tuple[int, int]:
    """Computes a (rows, cols) subplot grid that tightly fits num_panels with minimal empty
    cells, capped at max_cols columns. E.g. 4 panels -> 2x2 (not 2x3 with 2 dangling empty
    slots), while 6 panels still -> 2x3 as before."""
    import math

    n = max(num_panels, 1)
    if n <= max_cols:
        return 1, n
    cols = min(max_cols, math.ceil(math.sqrt(n)))
    rows = -(-n // cols)
    return rows, cols


def _load_json(path: str | Path) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_latest_task_result(
    runs_root: str | Path,
    task_name: str,
    activation_name: str = "alpha_golu",
    required_field: str | None = None,
) -> dict:
    """Finds the most recent results.json for (task_name, activation_name). If `required_field`
    is given, prefers the most recent run that actually has a non-empty value for that field --
    otherwise a single stale/partial run (e.g. an interrupted re-run, or one saved by an older
    code version that predates a given field) being the newest by mtime would silently blank out
    a plot even though older runs for the same task/activation have perfectly good data."""
    root_path = Path(runs_root)
    if not root_path.exists():
        return {}

    candidate_task_names = {task_name, *TASK_ALIASES.get(task_name, [])}
    candidates = []
    for result_path in root_path.rglob("results.json"):
        payload = _load_json(result_path)
        if not payload:
            continue
        if str(payload.get("task", "")).lower() not in candidate_task_names:
            continue
        if str(payload.get("activation", payload.get("activation_name", ""))).lower() != activation_name:
            continue
        candidates.append((result_path.stat().st_mtime, payload))

    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    if required_field:
        for _, payload in candidates:
            if payload.get(required_field):
                return payload
    return candidates[0][1]


def _layer_average_history(alpha_history: dict) -> np.ndarray | None:
    if not isinstance(alpha_history, dict) or not alpha_history:
        return None

    arrays = []
    for history in alpha_history.values():
        if not history:
            continue
        arrays.append(np.asarray(history, dtype=np.float64))

    if not arrays:
        return None

    min_len = min(array.shape[0] for array in arrays if array.ndim == 1 and array.size > 0)
    if min_len <= 0:
        return None

    stacked = np.stack([array[:min_len] for array in arrays], axis=0)
    return np.mean(stacked, axis=0)


def _parametric_comparison_label(task_name: str, activation_name: str, include_task: bool) -> str:
    activation_label = PARAMETRIC_ACTIVATION_LABELS.get(activation_name, activation_name.replace("_", " ").title())
    if not include_task:
        return activation_label
    task_label = TASK_LABELS.get(task_name, task_name.replace("_", " ").title())
    return f"{task_label} · {activation_label}"


def _select_latest_parametric_runs(run_json_paths: list[str], task_name: str | None = None) -> list[tuple[str, dict]]:
    latest: dict[tuple[str, str], tuple[float, str, dict]] = {}
    task_filter = task_name.lower().strip() if task_name else None

    for path in run_json_paths:
        payload = _load_json(path)
        if not payload:
            continue

        current_task = str(payload.get("task", "")).lower().strip()
        activation_name = str(payload.get("activation", payload.get("activation_name", ""))).lower().strip()
        if not current_task or not activation_name:
            continue
        if task_filter and current_task != task_filter:
            continue

        key = (current_task, activation_name)
        stat_result = Path(path)
        try:
            mtime = stat_result.stat().st_mtime
        except OSError:
            mtime = 0.0

        previous = latest.get(key)
        if previous is None or mtime >= previous[0]:
            latest[key] = (mtime, str(path), payload)

    ordered_keys = sorted(
        latest.keys(),
        key=lambda item: (
            TASK_ORDER.index(item[0]) if item[0] in TASK_ORDER else len(TASK_ORDER),
            PARAMETRIC_ACTIVATION_ORDER.index(item[1]) if item[1] in PARAMETRIC_ACTIVATION_ORDER else len(PARAMETRIC_ACTIVATION_ORDER),
        ),
    )
    return [(latest[key][1], latest[key][2]) for key in ordered_keys]


def _resolve_task_data(results: dict, task: str) -> dict:
    """Looks up `task` in a benchmark summary dict, falling back to legacy alias keys (see
    TASK_ALIASES) if the canonical key is missing/incomplete -- e.g. older summaries that
    split robustness runs across "corruption_robustness"/"adversarial_robustness" keys."""
    for candidate in [task, *TASK_ALIASES.get(task, [])]:
        task_data = results.get(candidate)
        if isinstance(task_data, dict) and task_data.get("alpha_golu") and task_data.get("golu_static"):
            return task_data
    task_data = results.get(task, {})
    return task_data if isinstance(task_data, dict) else {}


def _significance_stars(p_value) -> str:
    """Standard significance-star convention: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise."""
    if not isinstance(p_value, (int, float)) or not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def plot_parametric_comparison(
    run_json_paths: list[str],
    save_path: str = "outputs/paper_assets/parametric_comparison.png",
    title: str = "Layer-Averaged Parametric Activation Trajectories",
    task_name: str | None = None,
):
    """Plots layer-averaged parameter trajectories for multiple parametric activation runs."""
    if not run_json_paths:
        print("[Visualizer] No run JSONs provided for parametric comparison plot")
        return None

    series = []
    selected_runs = _select_latest_parametric_runs(run_json_paths, task_name=task_name)
    include_task_in_label = len({str(_load_json(path).get("task", "")).lower().strip() for path, _ in selected_runs if _load_json(path)}) > 1

    for path, payload in selected_runs:
        activation_name = str(payload.get("activation", payload.get("activation_name", "unknown"))).lower().strip()
        current_task = str(payload.get("task", "")).lower().strip()
        history = _layer_average_history(payload.get("alpha_history", {}))
        if history is None or history.size == 0:
            continue

        display_label = _parametric_comparison_label(current_task, activation_name, include_task_in_label)
        series.append((display_label, history))

    if not series:
        print("[Visualizer] No usable parametric trajectories found in the provided run JSONs")
        return None

    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for index, (label, history) in enumerate(series):
        steps = np.arange(1, len(history) + 1)
        color = color_cycle[index % len(color_cycle)] if color_cycle else None
        ax.plot(steps, history, label=label, linewidth=2.2, color=color)

    ax.set_title(title)
    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel(r"Layer-Averaged Parameter")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Parametric comparison plot saved to: {save_path}")
    return save_path


def plot_paper_benchmark_summary(
    results_path: str = "outputs/benchmark_results.json",
    save_dir: str = "outputs/paper_assets",
    task_order: list[str] | None = None,
):
    """Plots a paper-style summary of Alpha-GoLU versus Static GoLU, normalized to each task's
    Static GoLU baseline (=100%). Raw task metrics use incompatible units/scales (accuracy %,
    mIoU, MSE, perplexity, ...), so a shared absolute-value axis hides small-scale tasks; the
    relative scale keeps every task's bars visible and comparable on one axis."""
    results = _load_json(results_path)
    if not results:
        print(f"[Visualizer] No benchmark summary found at {results_path}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = task_order or TASK_ORDER
    alpha_values = []
    static_values = []
    task_labels = []
    raw_pairs = []
    p_values = []

    for task in task_order:
        task_data = _resolve_task_data(results, task)
        if not isinstance(task_data, dict):
            continue
        alpha_entry = task_data.get("alpha_golu", {})
        static_entry = task_data.get("golu_static", {})
        if not alpha_entry or not static_entry:
            continue

        alpha_mean = float(alpha_entry.get("mean", np.nan))
        static_mean = float(static_entry.get("mean", np.nan))
        if not np.isfinite(alpha_mean) or not np.isfinite(static_mean) or static_mean == 0 or alpha_mean == 0:
            continue

        if task in LOWER_IS_BETTER:
            # Lower raw values are better, so invert the ratio to keep "higher % = better" consistent.
            alpha_relative = 100.0 * static_mean / alpha_mean
        else:
            alpha_relative = 100.0 * alpha_mean / static_mean

        static_values.append(100.0)
        alpha_values.append(alpha_relative)
        task_labels.append(TASK_LABELS.get(task, task.title()))
        raw_pairs.append((static_mean, alpha_mean))
        p_values.append(task_data.get("p_value_welch_alpha_vs_static"))

    if not task_labels:
        print(f"[Visualizer] No usable task entries found in {results_path}")
        return None

    x = np.arange(len(task_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_static = ax.bar(x - width / 2, static_values, width, label="Static GoLU", color="#8da0cb")
    bars_alpha = ax.bar(x + width / 2, alpha_values, width, label="Alpha-GoLU", color="#fc8d62")
    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.0)

    for bar_static, bar_alpha, (static_raw, alpha_raw), p_value in zip(bars_static, bars_alpha, raw_pairs, p_values):
        top = max(bar_static.get_height(), bar_alpha.get_height())
        star = _significance_stars(p_value)
        label = f"{static_raw:.3g} / {alpha_raw:.3g}"
        if star:
            label = f"{label}\n{star}"
        ax.annotate(
            label,
            xy=((bar_static.get_x() + bar_alpha.get_x() + bar_alpha.get_width()) / 2, top),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.set_ylabel("Relative Performance (% of Static GoLU baseline, higher = better)")
    ax.set_title("Alpha-GoLU vs Static GoLU Across Benchmarks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.margins(y=0.12)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_benchmark_summary.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Paper benchmark summary saved to: {save_path}")
    return save_path


def plot_paper_overhead_summary(
    overhead_root: str = "outputs/overhead",
    save_dir: str = "outputs/paper_assets",
    task_order: list[str] | None = None,
    activations: list[str] | None = None,
):
    """Plots mean forward/backward latency per activation, grouped by task, from overhead JSON
    records. Baselines (e.g. ReLU/GELU/Static GoLU) are plotted alongside Alpha-GoLU whenever
    their overhead was tracked, so reviewers can judge relative cost; tasks/activations with no
    tracked records are simply omitted rather than left as empty gaps."""
    root_path = Path(overhead_root)
    if not root_path.exists():
        print(f"[Visualizer] No overhead directory found at {overhead_root}")
        return None

    records = []
    for json_path in sorted(root_path.rglob("*.json")):
        payload = _load_json(json_path)
        if payload:
            records.append(payload)

    if not records:
        print(f"[Visualizer] No overhead records found under {overhead_root}")
        return None

    task_order = task_order or TASK_ORDER
    if activations is None:
        present = {str(record.get("activation_name", record.get("activation", ""))).lower() for record in records}
        activations = [act for act in DEFAULT_OVERHEAD_ACTIVATIONS if act in present]
        activations += sorted(present - set(activations))

    def _task_candidates(task: str) -> list[str]:
        return [task, *TASK_ALIASES.get(task, [])]

    # task -> activation -> (mean_forward_ms, mean_backward_ms)
    task_activation_latency: dict[str, dict[str, tuple[float, float]]] = {}
    for task in task_order:
        candidates = _task_candidates(task)
        per_activation: dict[str, tuple[float, float]] = {}
        for activation in activations:
            forward_vals, backward_vals = [], []
            for record in records:
                record_task = str(record.get("task_name", record.get("task", ""))).lower()
                record_act = str(record.get("activation_name", record.get("activation", ""))).lower()
                if record_task not in candidates or record_act != activation:
                    continue
                forward = record.get("forward_ms", {}).get("mean", record.get("forward_latency_ms"))
                backward = record.get("backward_ms", {}).get("mean", record.get("backward_latency_ms"))
                if isinstance(forward, (int, float)) and np.isfinite(forward):
                    forward_vals.append(float(forward))
                if isinstance(backward, (int, float)) and np.isfinite(backward):
                    backward_vals.append(float(backward))
            if forward_vals and backward_vals:
                per_activation[activation] = (float(np.mean(forward_vals)), float(np.mean(backward_vals)))
        if per_activation:
            task_activation_latency[task] = per_activation

    rendered_tasks = [task for task in task_order if task in task_activation_latency]
    if not rendered_tasks:
        print(f"[Visualizer] No usable overhead entries found under {overhead_root}")
        return None

    # One subplot per task with its own y-axis, instead of a single shared axis -- latency
    # scales differ by 1-2 orders of magnitude across tasks (e.g. ~2ms language modeling vs
    # ~130ms segmentation), which squashed the cheaper tasks into invisible stubs on one axis.
    forward_color, backward_color = "#66c2a5", "#fc8d62"
    bar_width = 0.32

    rows, cols = _grid_shape(len(rendered_tasks))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, task in zip(axes, rendered_tasks):
        acts_present = [act for act in activations if act in task_activation_latency[task]]
        x = np.arange(len(acts_present))
        fwd_heights = [task_activation_latency[task][act][0] for act in acts_present]
        bwd_heights = [task_activation_latency[task][act][1] for act in acts_present]

        # Two colors encode Forward vs Backward only -- coloring bars by activation AND listing
        # activations on the x-axis was redundant and cluttered; the activation identity is
        # already fully conveyed by the tick labels.
        fwd_bars = ax.bar(x - bar_width / 2, fwd_heights, width=bar_width, color=forward_color, edgecolor="black", linewidth=0.5)
        bwd_bars = ax.bar(x + bar_width / 2, bwd_heights, width=bar_width, color=backward_color, edgecolor="black", linewidth=0.5)
        ax.bar_label(fwd_bars, fmt="%.1f", padding=2, fontsize=8)
        ax.bar_label(bwd_bars, fmt="%.1f", padding=2, fontsize=8)

        ax.set_xticks(x)
        # Only rotate labels when there are enough of them to actually risk overlapping --
        # a single centered "ALPHA GOLU" label doesn't need to be rotated at an angle.
        if len(acts_present) <= 2:
            ax.set_xticklabels([act.replace("_", " ").upper() for act in acts_present], rotation=0, ha="center", fontsize=9)
        else:
            ax.set_xticklabels([act.replace("_", " ").upper() for act in acts_present], rotation=30, ha="right", fontsize=9)
        ax.set_title(TASK_LABELS.get(task, task.title()), fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(y=0.15)

    for ax in axes[len(rendered_tasks):]:
        ax.set_axis_off()

    legend_handles = [
        Patch(facecolor=forward_color, edgecolor="black", label="Forward"),
        Patch(facecolor=backward_color, edgecolor="black", label="Backward"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=10)

    if len(rendered_tasks) == 1:
        fig.suptitle(f"Runtime Overhead: {TASK_LABELS.get(rendered_tasks[0], rendered_tasks[0].title())}", fontsize=14, y=1.03)
    else:
        fig.suptitle(f"Runtime Overhead Across {len(rendered_tasks)} Benchmarks", fontsize=14, y=1.03)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "paper_overhead_summary.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Paper overhead summary saved to: {save_path}")
    return save_path


def plot_paper_alpha_trajectories(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    activation_name: str = "alpha_golu",
    task_order: list[str] | None = None,
):
    """Plots alpha trajectories from the latest Alpha-GoLU run for each task."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = task_order or TASK_ORDER
    rows, cols = _grid_shape(len(task_order))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()
    used_axes = 0

    for task in task_order:
        result = _find_latest_task_result(root_path, task, activation_name=activation_name, required_field="alpha_history")
        ax = axes[used_axes]
        used_axes += 1
        if not result:
            ax.text(0.5, 0.5, "No run found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TASK_LABELS.get(task, task.title()))
            ax.set_axis_off()
            continue

        alpha_history = result.get("alpha_history", {})
        if not isinstance(alpha_history, dict) or not alpha_history:
            ax.text(0.5, 0.5, "No alpha history", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TASK_LABELS.get(task, task.title()))
            ax.set_axis_off()
            continue

        for layer_name, history in alpha_history.items():
            if not history:
                continue
            progress = np.linspace(0, 100, len(history))
            ax.plot(progress, history, linewidth=1.6, alpha=0.85, label=layer_name)

        max_epochs = max((len(history) for history in alpha_history.values() if history), default=0)
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="alpha = 1.0")
        ax.set_title(f"{TASK_LABELS.get(task, task.title())} ({max_epochs} Epochs)")
        ax.set_xlabel("Training Progress (%)")
        ax.set_ylabel(r"$\alpha$")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(True, alpha=0.25)

    for ax in axes[used_axes:]:
        ax.set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    fig.suptitle("Alpha-GoLU Trajectory Dashboard", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_alpha_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Paper alpha trajectory dashboard saved to: {save_path}")
    return save_path


def _select_representative_layers(alpha_history: dict, num_layers: int = 3) -> list[tuple[str, str, list]]:
    """Picks up to num_layers representative layers (Early/.../Late) from an alpha_history dict,
    spread evenly across insertion order (a proxy for network depth)."""
    layer_names = [name for name, history in alpha_history.items() if history]
    if not layer_names:
        return []

    if len(layer_names) <= num_layers:
        indices = list(range(len(layer_names)))
    else:
        indices = sorted({round(i * (len(layer_names) - 1) / (num_layers - 1)) for i in range(num_layers)})

    stage_labels = ["Early", "Mid", "Late"] if num_layers == 3 else [f"Layer {rank + 1}" for rank in range(num_layers)]
    n_picked = len(indices)
    picks = []
    for rank, idx in enumerate(indices):
        name = layer_names[idx]
        if n_picked == 1:
            stage = stage_labels[0]
        else:
            # Map proportionally across the full stage-label range (e.g. only 2 layers found
            # should be labeled "Early"/"Late", not "Early"/"Mid" -- picking stage_labels[rank]
            # directly would mislabel the last available layer as "Mid" and silently drop "Late".
            stage_idx = round(rank * (len(stage_labels) - 1) / (n_picked - 1))
            stage = stage_labels[stage_idx] if stage_idx < len(stage_labels) else f"Layer {rank + 1}"
        picks.append((name, stage, alpha_history[name]))
    return picks


def plot_curated_alpha_trajectories(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    activation_name: str = "alpha_golu",
    num_layers: int = 3,
    task_order: list[str] | None = None,
):
    """Plots Early/Mid/Late representative layer alpha trajectories per task, one subplot per
    task, instead of every layer -- a cleaner alternative to plot_paper_alpha_trajectories."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    stage_colors = {"Early": "#1b9e77", "Mid": "#d95f02", "Late": "#7570b3"}
    task_order = task_order or TASK_ORDER

    rows, cols = _grid_shape(len(task_order))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, task in zip(axes, task_order):
        ax.set_title(TASK_LABELS.get(task, task.title()))

        result = _find_latest_task_result(root_path, task, activation_name=activation_name, required_field="alpha_history")
        alpha_history = result.get("alpha_history") if result else None
        if not isinstance(alpha_history, dict) or not alpha_history:
            ax.text(0.5, 0.5, "No alpha history", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        picks = _select_representative_layers(alpha_history, num_layers=num_layers)
        if not picks:
            ax.text(0.5, 0.5, "No alpha history", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        max_epochs = max((len(history) for _, _, history in picks if history), default=0)
        ax.set_title(f"{TASK_LABELS.get(task, task.title())} ({max_epochs} Epochs)")

        for layer_name, stage, history in picks:
            progress = np.linspace(0, 100, len(history))
            ax.plot(progress, history, linewidth=1.8, alpha=0.9, label=stage, color=stage_colors.get(stage))

        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label=r"Baseline Static ($\alpha=1.0$)")
        ax.set_xlabel("Training Progress (%)")
        ax.set_ylabel(r"$\alpha$")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.grid(True, alpha=0.25)

    for ax in axes[len(task_order):]:
        ax.set_axis_off()

    # One shared legend (stage colors are identical across every subplot) instead of a
    # per-subplot legend with full module paths, which was tiny and collided with the curves.
    legend_handles = [Patch(facecolor=color, label=stage) for stage, color in stage_colors.items()]
    legend_handles.append(
        plt.Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.2, label=r"Baseline Static ($\alpha=1.0$)")
    )
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(legend_handles), frameon=False, fontsize=10)

    fig.suptitle("Alpha-GoLU Curated Trajectories (Early / Mid / Late Layers)", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_alpha_trajectories_curated.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Curated alpha trajectory dashboard saved to: {save_path}")
    return save_path


def _aggregate_seed_histories(
    task: str, activation: str, field_name: str, output_root: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int] | None:
    """Averages a per-epoch history field (e.g. epoch_loss_history, grad_norm_history) across
    all available seeds for (task, activation). Shorter/interrupted seeds are NaN-padded rather
    than truncating every seed to the shortest one, so an incomplete run doesn't silently cut
    off an otherwise-complete curve. Returns (epochs, mean, std_or_None, n_seeds), or None if no
    usable history exists."""
    from utils.scaled_benchmark_logger import load_results_by_activation

    payloads = load_results_by_activation(task, [activation], output_root=output_root).get(activation, [])
    histories = [
        payload[field_name]
        for payload in payloads
        if isinstance(payload.get(field_name), list) and payload[field_name]
    ]
    if not histories:
        return None

    lengths = [len(history) for history in histories]
    max_len = max(lengths)
    if len(set(lengths)) > 1:
        print(
            f"[Visualizer] Warning: {task}/{activation} seeds have inconsistent "
            f"{field_name} lengths {sorted(lengths)} -- likely an interrupted/incomplete run; "
            "each seed is only averaged over the epochs it actually has."
        )

    padded = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    for row, history in enumerate(histories):
        padded[row, : len(history)] = history
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0) if len(histories) > 1 else None
    epochs = np.arange(1, max_len + 1)
    return epochs, mean, std, len(histories)


def plot_paper_convergence_curves(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    task_order: list[str] | None = None,
    baseline_activation: str = "golu_static",
    proposed_activation: str = "alpha_golu",
):
    """Plots per-task training-loss convergence curves comparing baseline vs proposed activation
    (default: Static GoLU vs Alpha-GoLU), one subplot per task, averaged across all available
    seeds per activation with a shaded +/-1 std band (rather than a single arbitrary seed's
    curve). Complements the final-metric summary bar chart by showing *how* training progressed,
    not just the end result; the seed-count is shown in the legend so a single-seed activation
    (e.g. a baseline that's only been run once) is visibly distinguishable from an averaged one."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = task_order or TASK_ORDER
    rows, cols = _grid_shape(len(task_order))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    rendered = 0
    for ax, task in zip(axes, task_order):
        ax.set_title(TASK_LABELS.get(task, task.title()))
        plotted = False
        for activation, label, color in (
            (baseline_activation, "Static GoLU", "#8da0cb"),
            (proposed_activation, "Alpha-GoLU", "#fc8d62"),
        ):
            aggregated = _aggregate_seed_histories(task, activation, "epoch_loss_history", root_path)
            if aggregated is None:
                continue
            epochs, mean_loss, std_loss, n_seeds = aggregated
            ax.plot(epochs, mean_loss, label=f"{label} (n={n_seeds})", linewidth=1.8, color=color)
            if std_loss is not None:
                ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15, linewidth=0)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No loss history", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        rendered += 1
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[len(task_order):]:
        ax.set_axis_off()

    if rendered == 0:
        plt.close(fig)
        print(f"[Visualizer] No usable loss-history entries found under {runs_root}")
        return None

    fig.suptitle("Training Loss Convergence: Static GoLU vs Alpha-GoLU", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_convergence_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Convergence curves saved to: {save_path}")
    return save_path


def plot_gradient_stability(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    task_order: list[str] | None = None,
    baseline_activation: str = "golu_static",
    proposed_activation: str = "alpha_golu",
):
    """Plots per-task gradient-norm trajectories comparing baseline vs proposed activation,
    averaged across all available seeds with a shaded +/-1 std band. Lower/smoother gradient
    norms during training support an optimization-stability claim even when raw training-loss
    convergence looks similar between activations (see plot_paper_convergence_curves)."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = task_order or TASK_ORDER
    rows, cols = _grid_shape(len(task_order))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    rendered = 0
    for ax, task in zip(axes, task_order):
        ax.set_title(TASK_LABELS.get(task, task.title()))
        plotted = False
        for activation, label, color in (
            (baseline_activation, "Static GoLU", "#8da0cb"),
            (proposed_activation, "Alpha-GoLU", "#fc8d62"),
        ):
            aggregated = _aggregate_seed_histories(task, activation, "grad_norm_history", root_path)
            if aggregated is None:
                continue
            epochs, mean_norm, std_norm, n_seeds = aggregated
            ax.plot(epochs, mean_norm, label=f"{label} (n={n_seeds})", linewidth=1.8, color=color)
            if std_norm is not None:
                ax.fill_between(epochs, mean_norm - std_norm, mean_norm + std_norm, color=color, alpha=0.15, linewidth=0)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No grad-norm history", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        rendered += 1
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for ax in axes[len(task_order):]:
        ax.set_axis_off()

    if rendered == 0:
        plt.close(fig)
        print(f"[Visualizer] No usable grad-norm-history entries found under {runs_root}")
        return None

    fig.suptitle("Gradient-Norm Stability: Static GoLU vs Alpha-GoLU", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_gradient_stability.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Gradient stability plot saved to: {save_path}")
    return save_path


def plot_alpha_distribution(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    activation_name: str = "alpha_golu",
    task_order: list[str] | None = None,
):
    """Plots a histogram of each layer's FINAL (converged) alpha value per task, one subplot per
    task. Complements the trajectory dashboards by turning "vision tasks cluster above 1.0,
    language modeling clusters below 1.0" from an eyeballed trajectory observation into a single
    citable distribution figure."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = task_order or TASK_ORDER
    rows, cols = _grid_shape(len(task_order))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    rendered = 0
    for ax, task in zip(axes, task_order):
        result = _find_latest_task_result(root_path, task, activation_name=activation_name, required_field="alpha_history")
        alpha_history = result.get("alpha_history") if result else None
        if not isinstance(alpha_history, dict) or not alpha_history:
            ax.text(0.5, 0.5, "No alpha history", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TASK_LABELS.get(task, task.title()))
            ax.set_axis_off()
            continue

        final_values = [history[-1] for history in alpha_history.values() if history]
        if not final_values:
            ax.text(0.5, 0.5, "No alpha history", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TASK_LABELS.get(task, task.title()))
            ax.set_axis_off()
            continue

        mean_final = float(np.mean(final_values))
        n_layers = len(final_values)
        # A histogram looks blocky/misleading with very few points (thin unfilled-looking bars,
        # a mean line stranded in empty space between spikes) -- show individual points as a
        # strip plot instead once there are too few layers for bins to be meaningful.
        if n_layers <= 5:
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=n_layers)
            ax.scatter(final_values, 0.5 + jitter, s=90, color="#7570b3", edgecolor="black", linewidth=0.6, zorder=3)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel("Individual Layers")
        else:
            ax.hist(final_values, bins=min(15, max(5, n_layers)), color="#7570b3", edgecolor="black", alpha=0.85)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("Layer Count")

        ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label=r"Static ($\alpha=1.0$)")
        ax.axvline(mean_final, color="#1b9e77", linestyle="-", linewidth=1.5, label=f"Mean={mean_final:.3f}")
        ax.set_title(f"{TASK_LABELS.get(task, task.title())} (n={n_layers} layers)")
        ax.set_xlabel(r"Final $\alpha$")
        ax.ticklabel_format(axis="x", useOffset=False, style="plain")
        ax.margins(y=0.2)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.25)
        rendered += 1

    for ax in axes[len(task_order):]:
        ax.set_axis_off()

    if rendered == 0:
        plt.close(fig)
        print(f"[Visualizer] No usable alpha_history entries found under {runs_root}")
        return None

    fig.suptitle(f"Converged Alpha Distribution by Task ({activation_name})", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_alpha_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Alpha distribution plot saved to: {save_path}")
    return save_path


def plot_corruption_breakdown(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    activations: list[str] | None = None,
    include_clean: bool = True,
):
    """Grouped bar chart of mean per-corruption-type robustness accuracy across activations,
    averaged over all available seeds per activation. A single aggregate "robustness" score
    hides which corruption types actually drive the gap between activations, so this breaks it
    down by corruption type (whatever was tracked -- see CORRUPTION_SUITE in
    experiments/run_adversarial_robustness.py) plus clean accuracy for reference. Uses
    load_results_by_activation so the legacy robustness folder/task-key split
    (corruption_robustness/adversarial_robustness) is merged automatically."""
    from utils.scaled_benchmark_logger import load_results_by_activation

    activations = activations or DEFAULT_OVERHEAD_ACTIVATIONS
    results_by_activation = load_results_by_activation("robustness", activations, output_root=runs_root)

    corruption_names: list[str] = []
    activation_means: dict[str, dict[str, float]] = {}
    for act, payloads in results_by_activation.items():
        if not payloads:
            continue
        per_corruption_values: dict[str, list[float]] = defaultdict(list)
        for payload in payloads:
            if include_clean and isinstance(payload.get("clean_acc"), (int, float)):
                per_corruption_values["clean"].append(float(payload["clean_acc"]))
            for key, value in payload.items():
                if key in ("corruption_acc", "clean_acc"):
                    continue
                if key.endswith("_acc") and isinstance(value, (int, float)):
                    per_corruption_values[key[: -len("_acc")]].append(float(value))

        means = {name: float(np.mean(values)) for name, values in per_corruption_values.items() if values}
        if not means:
            continue
        activation_means[act] = means
        for name in means:
            if name not in corruption_names:
                corruption_names.append(name)

    if not activation_means:
        print(f"[Visualizer] No robustness per-corruption entries found under {runs_root}")
        return None

    corruption_names = sorted(corruption_names, key=lambda name: (name != "clean", name))
    rendered_activations = [act for act in activations if act in activation_means]
    corruption_types = [name for name in corruption_names if name != "clean"]
    has_clean = "clean" in corruption_names

    n_acts = max(len(rendered_activations), 1)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    act_color_map = {act: color_cycle[i % len(color_cycle)] for i, act in enumerate(rendered_activations)}
    width = 0.8 / n_acts

    # Clean accuracy sits on a much higher scale than corrupted accuracy, so it gets its own
    # panel instead of a bar cluster competing with the corruption-type clusters for the same
    # y-axis (which used to squash all corruption bars into the bottom quarter of the plot).
    fig_width = min(max(9.0, 1.3 * len(corruption_types) * n_acts), 16.0)
    if has_clean:
        fig, (ax_clean, ax_corrupt) = plt.subplots(
            1, 2, figsize=(fig_width, 5.5), gridspec_kw={"width_ratios": [1, max(2.0, len(corruption_types))]}
        )
    else:
        fig, ax_corrupt = plt.subplots(figsize=(fig_width, 5.5))
        ax_clean = None

    if ax_clean is not None:
        clean_x = np.arange(n_acts)
        clean_heights = [activation_means[act].get("clean", 0.0) for act in rendered_activations]
        clean_bars = ax_clean.bar(clean_x, clean_heights, color=[act_color_map[act] for act in rendered_activations])
        ax_clean.bar_label(clean_bars, fmt="%.1f", padding=2, fontsize=9)
        ax_clean.set_xticks(clean_x)
        ax_clean.set_xticklabels([act.replace("_", " ").upper() for act in rendered_activations], rotation=45, ha="right", fontsize=9)
        ax_clean.set_ylabel("Accuracy (%)", fontsize=12)
        ax_clean.set_title("Clean", fontsize=13)
        ax_clean.grid(True, axis="y", alpha=0.25)

    x = np.arange(len(corruption_types))
    for i, act in enumerate(rendered_activations):
        raw_values = [activation_means[act].get(name) for name in corruption_types]
        heights = [value if value is not None else 0.0 for value in raw_values]
        labels = [f"{value:.1f}" if value is not None else "" for value in raw_values]
        offsets = x - 0.4 + width * (i + 0.5)
        bars = ax_corrupt.bar(offsets, heights, width=width, label=act.replace("_", " ").upper(), color=act_color_map[act])
        ax_corrupt.bar_label(bars, labels=labels, padding=2, fontsize=9, rotation=90)

    ax_corrupt.set_xticks(x)
    ax_corrupt.set_xticklabels([name.replace("_", " ").title() for name in corruption_types], rotation=0, ha="center", fontsize=11)
    ax_corrupt.set_ylabel("Accuracy (%)", fontsize=12)
    ax_corrupt.set_title("Under Corruption", fontsize=13)
    ax_corrupt.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Robustness Breakdown by Corruption Type", fontsize=15, y=1.04)
    handles, labels = ax_corrupt.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=min(4, n_acts), frameon=False, fontsize=10)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "paper_corruption_breakdown.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Corruption breakdown chart saved to: {save_path}")
    return save_path


def plot_robustness_retention(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    activations: list[str] | None = None,
):
    """Bar chart of each activation's Robustness Retention Ratio (mean corruption accuracy /
    mean clean accuracy x 100), averaged across seeds via load_results_by_activation. A single
    aggregate robustness score conflates "starts higher" with "degrades less" -- this isolates
    the degradation rate so a higher retention ratio means an activation keeps more of its clean
    performance once corrupted, regardless of its raw clean-accuracy level."""
    from utils.scaled_benchmark_logger import load_results_by_activation

    activations = activations or DEFAULT_OVERHEAD_ACTIVATIONS
    results_by_activation = load_results_by_activation("robustness", activations, output_root=runs_root)

    retention_by_activation: dict[str, float] = {}
    for act, payloads in results_by_activation.items():
        clean_values = [float(p["clean_acc"]) for p in payloads if isinstance(p.get("clean_acc"), (int, float))]
        corruption_values = [float(p["corruption_acc"]) for p in payloads if isinstance(p.get("corruption_acc"), (int, float))]
        if not clean_values or not corruption_values:
            continue
        mean_clean = float(np.mean(clean_values))
        mean_corruption = float(np.mean(corruption_values))
        if mean_clean <= 0:
            continue
        retention_by_activation[act] = 100.0 * mean_corruption / mean_clean

    if not retention_by_activation:
        print(f"[Visualizer] No robustness clean/corruption accuracy pairs found under {runs_root}")
        return None

    rendered_activations = [act for act in activations if act in retention_by_activation]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    fig_width = min(max(7.0, 1.1 * len(rendered_activations)), 12.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    heights = [retention_by_activation[act] for act in rendered_activations]
    colors = [color_cycle[i % len(color_cycle)] for i in range(len(rendered_activations))]
    bars = ax.bar(range(len(rendered_activations)), heights, color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)

    ax.set_xticks(range(len(rendered_activations)))
    ax.set_xticklabels([act.replace("_", " ").upper() for act in rendered_activations], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Retention Ratio (%) = Corruption Acc / Clean Acc", fontsize=12)
    ax.set_title("Robustness Retention Ratio by Activation", fontsize=14)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(y=0.12)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "paper_robustness_retention.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Robustness retention ratio chart saved to: {save_path}")
    return save_path


def plot_experiment_dashboard(results_dict: Dict[str, Any], save_dir: str = "outputs"):
    """
    Generates a 4-panel empirical evaluation dashboard.
    
    Panels:
    1. Validation Accuracy Convergence
    2. Training Loss Trajectory
    3. Alpha Parameter Evolution Across Layers
    4. Latent Space Activation Variance (Sigma^2)
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Alpha-GoLU: Empirical Evaluation Dashboard", fontsize=16, fontweight='bold')

    # Panel 1: Validation Accuracy
    ax1 = axs[0, 0]
    for act_name, metrics in results_dict.items():
        if 'val_acc' in metrics:
            ax1.plot(metrics['val_acc'], label=f"{act_name.upper()}", linewidth=2)
    ax1.set_title("Validation Accuracy Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Training Loss
    ax2 = axs[0, 1]
    for act_name, metrics in results_dict.items():
        if 'train_loss' in metrics:
            ax2.plot(metrics['train_loss'], label=f"{act_name.upper()}", linewidth=2)
    ax2.set_title("Cross-Entropy Loss Trajectory")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Alpha Trajectories
    ax3 = axs[1, 0]
    if 'alpha_golu' in results_dict and 'alpha_history' in results_dict['alpha_golu']:
        alpha_hist = np.array(results_dict['alpha_golu']['alpha_history'])  # Shape: (epochs, num_layers)
        num_layers = alpha_hist.shape[1] if alpha_hist.ndim > 1 else 1
        
        if alpha_hist.ndim == 1:
            ax3.plot(alpha_hist, marker='o', label="Layer Alpha")
        else:
            for layer_idx in range(num_layers):
                ax3.plot(alpha_hist[:, layer_idx], marker='o', label=f"Layer {layer_idx+1} Alpha")
                
        ax3.axhline(1.0, color='red', linestyle='--', label='Static Baseline (1.0)')
        ax3.set_title("Alpha Evolution Across Network Depth")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel(r"Learned $\alpha$ Value")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Alpha Tracking Inactive", ha='center', va='center', transform=ax3.transAxes)

    # Panel 4: Latent Variance Comparison
    ax4 = axs[1, 1]
    acts = [act for act in results_dict.keys() if 'latent_var' in results_dict[act]]
    
    if acts:
        final_vars = [np.mean(results_dict[act]['latent_var'][-1]) for act in acts]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(acts)]
        
        bars = ax4.bar([a.upper() for a in acts], final_vars, color=colors, alpha=0.85)
        ax4.set_title("Final Layer Latent Variance (Lower = Squeezed)")
        ax4.set_ylabel(r"Activation Variance ($\sigma^2$)")
        ax4.grid(True, axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, "Variance Metrics Unavailable", ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "experiment_dashboard.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualizer] Research dashboard saved to: {save_path}")
