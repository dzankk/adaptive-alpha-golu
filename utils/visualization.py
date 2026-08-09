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
from typing import Dict, Any


TASK_LABELS = {
    "classification": "Classification",
    "detection": "Detection",
    "segmentation": "Segmentation",
    "diffusion": "Diffusion",
    "language_model": "Language Modeling",
    "robustness": "Corruption Robustness",
}

LOWER_IS_BETTER = {"diffusion", "language_model"}

TASK_ORDER = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]

PARAMETRIC_ACTIVATION_ORDER = ["alpha_golu", "prelu", "pgelu", "adaptive_swish", "swish_adaptive"]

PARAMETRIC_ACTIVATION_LABELS = {
    "alpha_golu": r"Alpha-GoLU ($\alpha$)",
    "adaptive_alpha_golu": r"Alpha-GoLU ($\alpha$)",
    "prelu": r"PReLU ($a$)",
    "pgelu": r"PGELU ($\alpha$)",
    "adaptive_swish": r"Parametric Swish ($\beta$)",
    "swish_adaptive": r"Parametric Swish ($\beta$)",
}


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


def _find_latest_task_result(runs_root: str | Path, task_name: str, activation_name: str = "alpha_golu") -> dict:
    root_path = Path(runs_root)
    if not root_path.exists():
        return {}

    candidates = []
    for result_path in root_path.rglob("results.json"):
        payload = _load_json(result_path)
        if not payload:
            continue
        if str(payload.get("task", "")).lower() != task_name:
            continue
        if str(payload.get("activation", payload.get("activation_name", ""))).lower() != activation_name:
            continue
        candidates.append((result_path.stat().st_mtime, payload))

    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0], reverse=True)
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


def plot_paper_benchmark_summary(results_path: str = "outputs/benchmark_results.json", save_dir: str = "outputs/paper_assets"):
    """Plots a paper-style summary of mean performance for Alpha-GoLU versus Static GoLU."""
    results = _load_json(results_path)
    if not results:
        print(f"[Visualizer] No benchmark summary found at {results_path}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]
    alpha_values = []
    static_values = []
    task_labels = []

    for task in task_order:
        task_data = results.get(task, {})
        if not isinstance(task_data, dict):
            continue
        alpha_entry = task_data.get("alpha_golu", {})
        static_entry = task_data.get("golu_static", {})
        if not alpha_entry or not static_entry:
            continue

        alpha_mean = float(alpha_entry.get("mean", np.nan))
        static_mean = float(static_entry.get("mean", np.nan))
        if not np.isfinite(alpha_mean) or not np.isfinite(static_mean):
            continue

        alpha_values.append(alpha_mean)
        static_values.append(static_mean)
        task_labels.append(TASK_LABELS.get(task, task.title()))

    if not task_labels:
        print(f"[Visualizer] No usable task entries found in {results_path}")
        return None

    x = np.arange(len(task_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, static_values, width, label="Static GoLU", color="#8da0cb")
    ax.bar(x + width / 2, alpha_values, width, label="Alpha-GoLU", color="#fc8d62")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.set_ylabel("Task Metric")
    ax.set_title("Alpha-GoLU vs Static GoLU Across Benchmarks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "paper_benchmark_summary.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Paper benchmark summary saved to: {save_path}")
    return save_path


def plot_paper_overhead_summary(overhead_root: str = "outputs/overhead", save_dir: str = "outputs/paper_assets"):
    """Plots mean forward/backward latency for Alpha-GoLU across tasks from overhead JSON records."""
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

    task_order = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]
    task_labels = []
    forward_vals = []
    backward_vals = []

    for task in task_order:
        task_records = [record for record in records if str(record.get("task_name", record.get("task", ""))).lower() == task and str(record.get("activation_name", record.get("activation", ""))).lower() == "alpha_golu"]
        if not task_records:
            continue
        forward = [float(record.get("forward_ms", {}).get("mean", record.get("forward_latency_ms", np.nan))) for record in task_records if isinstance(record, dict)]
        backward = [float(record.get("backward_ms", {}).get("mean", record.get("backward_latency_ms", np.nan))) for record in task_records if isinstance(record, dict)]
        forward = [value for value in forward if np.isfinite(value)]
        backward = [value for value in backward if np.isfinite(value)]
        if not forward or not backward:
            continue
        task_labels.append(TASK_LABELS.get(task, task.title()))
        forward_vals.append(float(np.mean(forward)))
        backward_vals.append(float(np.mean(backward)))

    if not task_labels:
        print(f"[Visualizer] No usable overhead entries found under {overhead_root}")
        return None

    x = np.arange(len(task_labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, forward_vals, width, label="Forward", color="#66c2a5")
    ax.bar(x + width / 2, backward_vals, width, label="Backward", color="#fc8d62")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Alpha-GoLU Runtime Overhead Across Benchmarks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "paper_overhead_summary.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualizer] Paper overhead summary saved to: {save_path}")
    return save_path


def plot_paper_alpha_trajectories(runs_root: str = "outputs/runs", save_dir: str = "outputs/paper_assets", activation_name: str = "alpha_golu"):
    """Plots alpha trajectories from the latest Alpha-GoLU run for each task."""
    root_path = Path(runs_root)
    if not root_path.exists():
        print(f"[Visualizer] No runs directory found at {runs_root}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    task_order = ["classification", "detection", "segmentation", "diffusion", "language_model", "robustness"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    axes = axes.flatten()
    used_axes = 0

    for task in task_order:
        result = _find_latest_task_result(root_path, task, activation_name=activation_name)
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
