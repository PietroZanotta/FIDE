"""Cluster-aware uncertainty summaries for matched stochastic comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper}


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    num_resamples: int,
    confidence: float = 0.95,
) -> Interval:
    """Bootstrap a mean by resampling independent ensemble clusters."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("values must be a one-dimensional array with at least two clusters")
    if num_resamples < 100:
        raise ValueError("num_resamples must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(num_resamples, array.size))
    samples = np.mean(array[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return Interval(
        estimate=float(np.mean(array)),
        lower=float(np.quantile(samples, alpha)),
        upper=float(np.quantile(samples, 1.0 - alpha)),
    )


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    num_resamples: int,
    confidence: float = 0.95,
) -> Interval:
    """Bootstrap ``mean(left - right)`` using matched ensemble clusters."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("paired arrays must have identical shapes")
    return bootstrap_mean_interval(
        left_array - right_array,
        seed=seed,
        num_resamples=num_resamples,
        confidence=confidence,
    )
