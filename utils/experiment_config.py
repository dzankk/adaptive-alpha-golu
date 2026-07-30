"""
Helpers for loading benchmark experiment configuration files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_benchmark_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found at {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a JSON object")

    return data