"""Seed-level aggregation and paired bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def percentile_interval(
    values: list[float] | np.ndarray,
    level: float = 0.95,
) -> dict[str, float]:
    """Summarize an observed sample using empirical quantiles.

    This is appropriate for descriptive summaries of the observed values
    themselves. It is not the confidence interval used for paired method
    comparisons.
    """
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


def paired_effect(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
    level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260806,
) -> dict[str, float]:
    """Estimate a paired mean effect with a percentile bootstrap CI.

    The inferential unit is the independent training seed.

    For seed i, define

        delta_i = left_i - right_i.

    We bootstrap the *paired seed effects* delta_i, not the two methods
    independently. Each bootstrap replicate resamples the seed indices with
    replacement and computes the mean paired effect.

    Negative effects therefore mean that `left` has a smaller metric than
    `right`. Whether that is desirable depends on the metric.
    """
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError("paired arrays must have identical shapes")

    if a.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional")

    if a.size == 0:
        raise ValueError("paired arrays must not be empty")

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired arrays must contain only finite values")

    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")

    effects = a - b
    n = effects.size

    rng = np.random.default_rng(bootstrap_seed)

    # Shape:
    #   (bootstrap_resamples, n)
    #
    # Every row is one bootstrap resample of the independent training seeds.
    sample_indices = rng.integers(
        low=0,
        high=n,
        size=(bootstrap_resamples, n),
    )

    bootstrap_means = effects[sample_indices].mean(axis=1)

    alpha = (1.0 - level) / 2.0

    return {
        "mean": float(np.mean(effects)),
        "lower": float(np.quantile(bootstrap_means, alpha)),
        "upper": float(np.quantile(bootstrap_means, 1.0 - alpha)),
        "count": int(n),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_seed": int(bootstrap_seed),
    }