"""
Structured run artifact helpers for benchmark reproducibility.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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


def collect_environment_metadata() -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        torch = None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "git_commit": _run_git_command(["rev-parse", "HEAD"]),
        "git_branch": _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_run_git_command(["status", "--porcelain"])),
    }


def build_run_name(task: str, activation: Optional[str], seeds: Iterable[int]) -> str:
    seed_part = "-".join(str(seed) for seed in seeds)
    activation_part = activation or "all"
    return f"{task}_{activation_part}_seeds-{seed_part}"


def create_run_directory(base_dir: str, task: str, activation: Optional[str], seeds: Iterable[int]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{timestamp}_{build_run_name(task, activation, seeds)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def stable_seed_directory(base_dir: str, task: str, activation: str, seed: int) -> Path:
    """Non-timestamped per-seed directory, so a resumed process can find a prior checkpoint."""
    seed_dir = Path(base_dir) / task / f"{activation}_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    return seed_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def build_run_manifest(
    *,
    command: str,
    task: str,
    seeds: list[int],
    activations: list[str],
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    manifest = {
        "command": command,
        "task": task,
        "seeds": list(seeds),
        "activations": list(activations),
        "environment": collect_environment_metadata(),
    }
    if extra_config:
        manifest["config"] = extra_config
    return manifest