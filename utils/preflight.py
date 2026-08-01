"""
Preflight checks for long benchmark runs.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_git_command(args: list[str], cwd: Optional[str] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _disk_free_gb(path: Path) -> Optional[float]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.free / (1024 ** 3)


def _can_write(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            return True
    except OSError:
        return False


def run_preflight_checks(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path,
    min_free_gb: float = 20.0,
) -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    def add_check(name: str, status: str, details: str) -> None:
        checks.append({"check": name, "status": status, "details": details})

    python_version = sys.version.split()[0]
    add_check("python", "ok", python_version)

    torch_version = getattr(torch, "__version__", None)
    add_check("torch", "ok" if torch_version else "warn", torch_version or "torch unavailable")

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        add_check("cuda", "ok", device_name)
    else:
        add_check("cuda", "warn", "CUDA not available; runs will use CPU")
        warnings.append("CUDA is not available")

    free_gb = _disk_free_gb(project_root)
    if free_gb is None:
        add_check("disk", "warn", f"Could not determine free space for {project_root}")
    elif free_gb < min_free_gb:
        message = f"Only {free_gb:.1f} GB free on {project_root}; minimum recommended is {min_free_gb:.1f} GB"
        add_check("disk", "error", message)
        errors.append(message)
    else:
        add_check("disk", "ok", f"{free_gb:.1f} GB free")

    if data_root.exists():
        add_check("data_root", "ok", str(data_root))
    else:
        add_check("data_root", "warn", f"{data_root} does not exist yet; datasets may still download")
        warnings.append(f"Data root {data_root} does not exist yet")

    if _can_write(output_root):
        add_check("output_root", "ok", str(output_root))
    else:
        message = f"Cannot write to output directory: {output_root}"
        add_check("output_root", "error", message)
        errors.append(message)

    git_commit = _run_git_command(["rev-parse", "HEAD"], cwd=str(project_root))
    git_branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=str(project_root))
    git_dirty = bool(_run_git_command(["status", "--porcelain"], cwd=str(project_root)))

    if git_commit is None:
        add_check("git", "warn", "Git metadata unavailable")
        warnings.append("Git metadata unavailable")
    else:
        dirty_suffix = "; dirty" if git_dirty else ""
        add_check("git", "ok", f"{git_branch or 'unknown'} @ {git_commit[:8]}{dirty_suffix}")

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": python_version,
        "platform": platform.platform(),
        "project_root": str(project_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "min_free_gb": min_free_gb,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
    }


def save_preflight_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preflight_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4, sort_keys=True)
    return report_path