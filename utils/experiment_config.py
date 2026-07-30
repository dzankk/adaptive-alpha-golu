"""
Helpers for loading benchmark experiment configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


REQUIRED_BENCHMARK_KEYS = {"tasks", "activations", "seeds", "output_root", "summary_path"}


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    raw_path = Path(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    candidate_paths = [raw_path]
    if not raw_path.is_absolute():
        candidate_paths.extend([
            repo_root / raw_path,
            repo_root / "configs" / raw_path,
            repo_root / "configs" / raw_path.name,
        ])

    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Benchmark config not found at {config_path}. Tried: {', '.join(str(candidate) for candidate in candidate_paths)}"
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a JSON object")

    missing = sorted(REQUIRED_BENCHMARK_KEYS - set(data.keys()))
    if missing:
        raise ValueError(
            f"Benchmark config is missing required keys: {', '.join(missing)}"
        )

    return data