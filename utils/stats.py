"""
Statistical Rigor & Metrics Suite
=================================
Computes descriptive statistics (mean, std, SEM) and Welch t-tests 
across random seeds for paper reporting.
"""

import numpy as np
from scipy import stats
from typing import List, Dict


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


def calculate_p_value(baseline_accs: List[float], proposed_accs: List[float]) -> float:
    """
    Performs Welch's two-tailed t-test to check if Alpha-GoLU's accuracy 
    improvement is statistically significant over the baseline.
    """
    if len(baseline_accs) < 2 or len(proposed_accs) < 2:
        return 1.0  # Requires at least 2 seeds for variance estimation

    baseline = np.asarray(baseline_accs, dtype=np.float64)
    proposed = np.asarray(proposed_accs, dtype=np.float64)

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
