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
    
    _, p_val = stats.ttest_ind(baseline_accs, proposed_accs, equal_var=False)
    return float(p_val)
