"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml


_REQUIRED_TOP_LEVEL = {
    "seed",
    "system",
    "true_prior",
    "learned_initial",
    "target",
    "training",
    "sampler",
    "calibration",
    "evaluation",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    missing = sorted(_REQUIRED_TOP_LEVEL - set(data))
    if missing:
        raise ValueError(f"configuration missing keys: {missing}")
    return data


def with_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    copied = deepcopy(config)
    copied["seed"] = int(seed)
    return copied
