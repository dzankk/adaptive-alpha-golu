"""
Estimate Total Wall-Clock GPU-Hours From Saved Run Timestamps
===============================================================
Estimates the approximate total GPU-hours consumed across all saved training runs,
for the "scale gap" disclosure in the paper's Limitations section -- without
fabricating a number.

Method: progress.json/results.json (and any checkpoints) are rewritten repeatedly
throughout training via atomic write-then-replace, so their file-modification times
track real wall-clock progress. For each per-seed run directory, the span between the
earliest and latest file mtime approximates that run's training duration. Summing
this across every run directory gives a rough total.

IMPORTANT: run this directly on the machine that did the training (e.g. the GPU
server), BEFORE these files are touched by git or copied elsewhere -- a git checkout
resets file mtimes to the checkout time, which would silently corrupt this estimate.
outputs/runs/ and outputs/runs_scale/ are gitignored, so on the original training
server their mtimes should still be authentic.

Usage:
    python -m diagnostics.estimate_compute_usage
    python -m diagnostics.estimate_compute_usage --runs-root outputs/runs outputs/runs_scale
"""

import argparse
from pathlib import Path


def _run_directories(runs_root: Path) -> list[Path]:
    """Every task/activation_seed directory that contains at least one file."""
    if not runs_root.exists():
        return []
    directories = []
    for task_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if any(f.is_file() for f in seed_dir.rglob("*")):
                directories.append(seed_dir)
    return directories


def _duration_seconds(run_dir: Path) -> float:
    mtimes = [f.stat().st_mtime for f in run_dir.rglob("*") if f.is_file()]
    if len(mtimes) < 2:
        return 0.0
    return max(mtimes) - min(mtimes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate total wall-clock GPU-hours from saved run file timestamps")
    parser.add_argument("--runs-root", type=str, nargs="+", default=["outputs/runs", "outputs/runs_scale"])
    args = parser.parse_args()

    total_seconds = 0.0
    print(f"{'Task':<24} {'Run dir':<28} {'Hours':>8}")
    for root in args.runs_root:
        root_path = Path(root)
        run_dirs = _run_directories(root_path)
        if not run_dirs:
            print(f"[Warning] No run directories found under {root_path}")
            continue
        for run_dir in run_dirs:
            duration = _duration_seconds(run_dir)
            total_seconds += duration
            print(f"{run_dir.parent.name:<24} {run_dir.name:<28} {duration / 3600:>8.2f}")

    print(f"\nTotal estimated GPU-hours (wall clock, summed across all saved runs): {total_seconds / 3600:.1f}")
    print("NOTE: this is a rough estimate from file-modification timestamps, not a rigorous")
    print("profiler measurement -- suitable for an approximate 'compute scale' disclosure only.")


if __name__ == "__main__":
    main()
