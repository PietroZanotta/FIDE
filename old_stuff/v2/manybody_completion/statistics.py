"""Seed-level aggregation and lightweight uncertainty intervals."""

from __future__ import annotations

import numpy as np


def percentile_interval(values: list[float] | np.ndarray, level: float = 0.95) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty values")
    alpha = (1.0 - level) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1.0 - alpha)),
        "count": int(arr.size),
    }


def paired_effect(left: list[float], right: list[float], level: float = 0.95) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have identical shapes")
    return percentile_interval((a - b).tolist(), level=level)
