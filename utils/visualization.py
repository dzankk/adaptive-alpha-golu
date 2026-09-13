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
    """Computes a (rows, cols) subplot grid that fits num_panels, capped at max_cols columns."""
    cols = min(max_cols, max(num_panels, 1))
    rows = -(-max(num_panels, 1) // cols)
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

    if not task_labels:
        print(f"[Visualizer] No usable task entries found in {results_path}")
        return None

    x = np.arange(len(task_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_static = ax.bar(x - width / 2, static_values, width, label="Static GoLU", color="#8da0cb")
    bars_alpha = ax.bar(x + width / 2, alpha_values, width, label="Alpha-GoLU", color="#fc8d62")
    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.0)

    for bar_static, bar_alpha, (static_raw, alpha_raw) in zip(bars_static, bars_alpha, raw_pairs):
        top = max(bar_static.get_height(), bar_alpha.get_height())
        ax.annotate(
            f"{static_raw:.3g} / {alpha_raw:.3g}",
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

    bar_width = 0.32
    intra_gap = 0.12
    inter_task_gap = 0.9
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    act_color_map = {act: color_cycle[i % len(color_cycle)] for i, act in enumerate(activations)}

    fwd_xs, fwd_heights, bwd_xs, bwd_heights, bar_colors = [], [], [], [], []
    tick_positions, task_labels = [], []
    cursor = 0.0
    for task in rendered_tasks:
        acts_present = [act for act in activations if act in task_activation_latency[task]]
        group_start = cursor
        for act in acts_present:
            forward_ms, backward_ms = task_activation_latency[task][act]
            fwd_xs.append(cursor)
            fwd_heights.append(forward_ms)
            bwd_xs.append(cursor + bar_width)
            bwd_heights.append(backward_ms)
            bar_colors.append(act_color_map[act])
            cursor += 2 * bar_width + intra_gap
        group_end = cursor - intra_gap
        tick_positions.append((group_start + group_end) / 2.0)
        task_labels.append(TASK_LABELS.get(task, task.title()))
        cursor = group_end + inter_task_gap

    fig, ax = plt.subplots(figsize=(max(12, 2.2 * len(tick_positions)), 5.5))
    fwd_bars = ax.bar(fwd_xs, fwd_heights, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5)
    bwd_bars = ax.bar(bwd_xs, bwd_heights, width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.5, hatch="//")
    ax.bar_label(fwd_bars, fmt="%.1f ms", padding=2, fontsize=7, rotation=90)
    ax.bar_label(bwd_bars, fmt="%.1f ms", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(task_labels, rotation=0, ha="center")
    ax.set_ylabel("Latency (ms)")
    if len(rendered_tasks) == 1:
        ax.set_title(f"Runtime Overhead: {task_labels[0]}")
    else:
        ax.set_title(f"Runtime Overhead Across {len(rendered_tasks)} Benchmarks")
    ax.grid(True, axis="y", alpha=0.25)

    rendered_activations = [act for act in activations if any(act in task_activation_latency[task] for task in rendered_tasks)]
    legend_handles = [Patch(facecolor=act_color_map[act], edgecolor="black", label=act.replace("_", " ").upper()) for act in rendered_activations]
    legend_handles.append(Patch(facecolor="white", edgecolor="black", label="Forward"))
    legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="Backward"))
    ax.legend(handles=legend_handles, frameon=False, ncol=min(4, len(legend_handles)), fontsize=8)

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
            ax.plot(history, linewidth=1.6, alpha=0.85, label=layer_name)

        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label="alpha = 1.0")
        ax.set_title(TASK_LABELS.get(task, task.title()))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\alpha$")
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
    """Picks up to num_layers representative layers (Early/Mid/Late) from an alpha_history dict,
    spread evenly across insertion order (a proxy for network depth)."""
    layer_names = [name for name, history in alpha_history.items() if history]
    if not layer_names:
        return []

    if len(layer_names) <= num_layers:
        indices = list(range(len(layer_names)))
    else:
        indices = sorted({round(i * (len(layer_names) - 1) / (num_layers - 1)) for i in range(num_layers)})

    stage_labels = ["Early", "Mid", "Late"] if num_layers == 3 else [f"Layer {rank + 1}" for rank in range(num_layers)]
    picks = []
    for rank, idx in enumerate(indices):
        name = layer_names[idx]
        stage = stage_labels[rank] if rank < len(stage_labels) else f"Layer {rank + 1}"
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

        for layer_name, stage, history in picks:
            ax.plot(history, linewidth=1.8, alpha=0.9, label=f"{stage} ({layer_name})", color=stage_colors.get(stage))

        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, label=r"Baseline Static ($\alpha=1.0$)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\alpha$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, frameon=False)

    for ax in axes[len(task_order):]:
        ax.set_axis_off()

    fig.suptitle("Alpha-GoLU Curated Trajectories (Early / Mid / Late Layers)", y=1.02, fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_alpha_trajectories_curated.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Curated alpha trajectory dashboard saved to: {save_path}")
    return save_path


def plot_paper_convergence_curves(
    runs_root: str = "outputs/runs",
    save_dir: str = "outputs/paper_assets",
    task_order: list[str] | None = None,
    baseline_activation: str = "golu_static",
    proposed_activation: str = "alpha_golu",
):
    """Plots per-task training-loss convergence curves comparing baseline vs proposed activation
    (default: Static GoLU vs Alpha-GoLU), one subplot per task, using each activation's most
    recent saved `epoch_loss_history`. Complements the final-metric summary bar chart by showing
    *how* training progressed, not just the end result."""
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
            result = _find_latest_task_result(root_path, task, activation_name=activation, required_field="epoch_loss_history")
            history = result.get("epoch_loss_history") if result else None
            if not isinstance(history, list) or not history:
                continue
            epochs = np.arange(1, len(history) + 1)
            ax.plot(epochs, history, label=label, linewidth=1.8, color=color)
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

    x = np.arange(len(corruption_names))
    n_acts = max(len(rendered_activations), 1)
    width = 0.8 / n_acts
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(corruption_names) * n_acts), 5.5))
    for i, act in enumerate(rendered_activations):
        raw_values = [activation_means[act].get(name) for name in corruption_names]
        heights = [value if value is not None else 0.0 for value in raw_values]
        labels = [f"{value:.1f}" if value is not None else "" for value in raw_values]
        offsets = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(offsets, heights, width=width, label=act.replace("_", " ").upper(), color=color_cycle[i % len(color_cycle)])
        ax.bar_label(bars, labels=labels, padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ").title() for name in corruption_names], rotation=0, ha="center")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness Breakdown by Corruption Type")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(4, n_acts), fontsize=8)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "paper_corruption_breakdown.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Corruption breakdown chart saved to: {save_path}")
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
