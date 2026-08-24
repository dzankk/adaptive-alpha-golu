"""
Statistical Rigor & Metrics Suite
=================================
Computes descriptive statistics (mean, std, SEM) and paired/Welch t-tests
across random seeds for paper reporting, plus wall-clock timing comparisons.
"""

import math

import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional


def compute_summary_statistics(acc_list: List[float]) -> Dict[str, float]:
    """Calculates mean, std dev, and standard error of the mean (SEM)."""
    arr = np.array(acc_list)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "sem": float(stats.sem(arr, ddof=1)) if len(arr) > 1 else 0.0
    }


def _welch_p_value(baseline: np.ndarray, proposed: np.ndarray) -> float:
    baseline_mean = float(np.mean(baseline))
    proposed_mean = float(np.mean(proposed))
    baseline_var = float(np.var(baseline, ddof=1))
    proposed_var = float(np.var(proposed, ddof=1))
    se_sq = (baseline_var / baseline.size) + (proposed_var / proposed.size)
    if se_sq <= 1e-12:
        return 1.0 if abs(baseline_mean - proposed_mean) <= 1e-12 else 0.0

    _, p_val = stats.ttest_ind(baseline, proposed, equal_var=False, nan_policy="omit")
    if np.isfinite(p_val):
        return float(p_val)

    baseline_n = baseline.size
    proposed_n = proposed.size
    t_stat = abs(baseline_mean - proposed_mean) / np.sqrt(se_sq)
    denom = ((baseline_var / baseline_n) ** 2) / max(baseline_n - 1, 1) + ((proposed_var / proposed_n) ** 2) / max(proposed_n - 1, 1)
    df = (se_sq ** 2) / denom if denom > 0 else max(baseline_n + proposed_n - 2, 1)
    p_val = 2.0 * stats.t.sf(t_stat, df)
    return float(np.clip(p_val, 0.0, 1.0))


def calculate_p_value(baseline_accs: List[float], proposed_accs: List[float], *, paired: bool = True) -> float:
    """
    Checks whether Alpha-GoLU's metric difference over a baseline is statistically significant.

    Defaults to a paired t-test (scipy.stats.ttest_rel), since benchmark runs evaluate every
    activation on the *same* seed list in the *same* order -- pairing on seed removes the
    between-seed variance component and gives materially more power than an independent test
    for the same sample size. Falls back to Welch's unequal-variance t-test when the two lists
    don't have matching lengths (i.e. they can't actually be paired) or `paired=False`.
    """
    if len(baseline_accs) < 2 or len(proposed_accs) < 2:
        return 1.0  # Requires at least 2 seeds for variance estimation

    baseline = np.asarray(baseline_accs, dtype=np.float64)
    proposed = np.asarray(proposed_accs, dtype=np.float64)

    if paired and baseline.size == proposed.size:
        diffs = proposed - baseline
        diff_var = float(np.var(diffs, ddof=1))
        if diff_var <= 1e-12:
            return 1.0 if np.allclose(diffs, 0.0, atol=1e-12) else 0.0
        _, p_val = stats.ttest_rel(proposed, baseline, nan_policy="omit")
        if np.isfinite(p_val):
            return float(np.clip(p_val, 0.0, 1.0))
        # Fall through to Welch's below if the paired test produced a non-finite result.

    return _welch_p_value(baseline, proposed)


def calculate_cohens_d(baseline_accs: List[float], proposed_accs: List[float]) -> float:
    """Independent-samples Cohen's d (pooled std) effect size of proposed relative to baseline."""
    baseline = np.asarray(baseline_accs, dtype=np.float64)
    proposed = np.asarray(proposed_accs, dtype=np.float64)
    if baseline.size < 2 or proposed.size < 2:
        return float("nan")

    pooled_var = (
        (baseline.size - 1) * np.var(baseline, ddof=1) + (proposed.size - 1) * np.var(proposed, ddof=1)
    ) / (baseline.size + proposed.size - 2)
    pooled_std = float(np.sqrt(pooled_var))
    if pooled_std <= 1e-12:
        return float("nan")
    return float((np.mean(proposed) - np.mean(baseline)) / pooled_std)


def calculate_paired_cohens_d(baseline_accs: List[float], proposed_accs: List[float]) -> float:
    """Paired Cohen's d_z (mean difference / std of per-seed differences); requires matching lengths."""
    baseline = np.asarray(baseline_accs, dtype=np.float64)
    proposed = np.asarray(proposed_accs, dtype=np.float64)
    if baseline.size != proposed.size or baseline.size < 2:
        return float("nan")

    diffs = proposed - baseline
    diff_std = float(np.std(diffs, ddof=1))
    if diff_std <= 1e-12:
        return float("nan")
    return float(np.mean(diffs) / diff_std)


def _mean_epoch_seconds(payload: Dict[str, Any]) -> Optional[float]:
    values = payload.get("epoch_seconds")
    if not isinstance(values, list) or not values:
        return None
    finite_values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return float(np.mean(finite_values)) if finite_values else None


def _samples_per_second(payload: Dict[str, Any]) -> Optional[float]:
    forward_ms = payload.get("forward_latency_ms")
    batch_sizes = payload.get("batch_sizes")
    if not isinstance(forward_ms, (int, float)) or forward_ms <= 0:
        return None
    if not isinstance(batch_sizes, list) or not batch_sizes:
        return None
    mean_batch_size = float(np.mean(batch_sizes))
    return mean_batch_size / (float(forward_ms) / 1000.0)


def summarize_timing_by_activation(
    results_by_activation: Dict[str, List[Dict[str, Any]]],
    baseline_activation: str = "golu_static",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compares mean epoch/train wall-clock time and inference throughput across activations.

    `results_by_activation` maps activation name to a list of per-seed results.json payloads
    (as loaded from outputs/runs/<task>/**/results.json). Each payload may optionally contain
    `epoch_seconds` (list[float]), `train_seconds` (float), and overhead-tracker throughput
    fields (`forward_latency_ms`, `batch_sizes`) -- all missing fields degrade to None rather
    than raising, since overhead tracking is opt-in.
    """
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for activation, payloads in results_by_activation.items():
        epoch_seconds_values = [value for value in (_mean_epoch_seconds(payload) for payload in payloads) if value is not None]
        train_seconds_values = [
            float(payload["train_seconds"])
            for payload in payloads
            if isinstance(payload.get("train_seconds"), (int, float)) and math.isfinite(float(payload["train_seconds"]))
        ]
        throughput_values = [value for value in (_samples_per_second(payload) for payload in payloads) if value is not None]
        summary[activation] = {
            "mean_epoch_seconds": float(np.mean(epoch_seconds_values)) if epoch_seconds_values else None,
            "mean_train_seconds": float(np.mean(train_seconds_values)) if train_seconds_values else None,
            "mean_samples_per_second": float(np.mean(throughput_values)) if throughput_values else None,
            "n_runs": len(payloads),
        }

    baseline = summary.get(baseline_activation)
    for row in summary.values():
        row["epoch_seconds_vs_baseline"] = (
            row["mean_epoch_seconds"] / baseline["mean_epoch_seconds"]
            if baseline and baseline.get("mean_epoch_seconds") and row["mean_epoch_seconds"] is not None
            else None
        )
        row["train_seconds_vs_baseline"] = (
            row["mean_train_seconds"] / baseline["mean_train_seconds"]
            if baseline and baseline.get("mean_train_seconds") and row["mean_train_seconds"] is not None
            else None
        )
    return summary


def format_timing_table(
    timing_summary: Dict[str, Dict[str, Optional[float]]],
    baseline_activation: str = "golu_static",
) -> str:
    """Renders `summarize_timing_by_activation`'s output as an aligned plain-text table."""
    headers = [
        "activation",
        "mean_epoch_s",
        "mean_train_s",
        "samples/s",
        f"epoch_s vs {baseline_activation}",
        f"train_s vs {baseline_activation}",
        "n_runs",
    ]

    def _fmt(value: Optional[float], suffix: str = "") -> str:
        return f"{value:.3f}{suffix}" if isinstance(value, (int, float)) else "N/A"

    rows = []
    for activation in sorted(timing_summary.keys()):
        row = timing_summary[activation]
        rows.append(
            [
                activation,
                _fmt(row.get("mean_epoch_seconds"), "s"),
                _fmt(row.get("mean_train_seconds"), "s"),
                _fmt(row.get("mean_samples_per_second")),
                _fmt(row.get("epoch_seconds_vs_baseline"), "x"),
                _fmt(row.get("train_seconds_vs_baseline"), "x"),
                str(row.get("n_runs", 0)),
            ]
        )

    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def _fmt_row(cells: List[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, col_widths))

    lines = [_fmt_row(headers), _fmt_row(["-" * width for width in col_widths])]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)

