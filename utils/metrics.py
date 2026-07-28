"""
Statistical Rigor & Metrics Suite
=================================
Computes mean, standard deviation, and paired t-tests across random seeds.
"""

import numpy as np
from scipy import stats

def compute_summary_statistics(acc_list: list) -> dict:
    """Calculates mean, std dev, and standard error for accuracy across runs."""
    arr = np.array(acc_list)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(stats.sem(arr)) if len(arr) > 1 else 0.0
    }

def calculate_p_value(baseline_accs: list, proposed_accs: list) -> float:
    """Performs a paired t-test to check if alpha-GoLU is statistically significant."""
    if len(baseline_accs) < 2 or len(proposed_accs) < 2:
        return 1.0  # Need at least 2 runs for significance testing
    t_stat, p_val = stats.ttest_rel(baseline_accs, proposed_accs)
    return float(p_val)
