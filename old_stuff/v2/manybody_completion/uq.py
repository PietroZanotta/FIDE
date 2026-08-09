"""Held-out higher-order uncertainty summaries."""

from __future__ import annotations

import numpy as np

from .metrics import binary_entropy, normalized, weighted_mean, weighted_variance


def weighted_quantile(values: np.ndarray, probabilities: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    p = normalized(probabilities)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(p[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile, side="left")])


def summarize_higher_order(
    triplet_values: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    p = normalized(probabilities)
    mean = weighted_mean(triplet_values, p)
    variance = weighted_variance(triplet_values, p)
    intervals = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        alpha = (1.0 - level) / 2.0
        intervals[f"{int(level * 100)}"] = {
            "lower": weighted_quantile(triplet_values, p, alpha),
            "upper": weighted_quantile(triplet_values, p, 1.0 - alpha),
        }
    mode_plus = float(np.sum(p[np.asarray(labels) > 0]))
    return {
        "triplet_mean": mean,
        "triplet_variance": variance,
        "triplet_standard_deviation": float(np.sqrt(max(variance, 0.0))),
        "predictive_intervals": intervals,
        "mode_plus_probability": mode_plus,
        "mode_entropy_bits": binary_entropy(mode_plus),
    }


def aggregate_seed_higher_order_uq(seed_summaries: list[dict]) -> dict:
    if not seed_summaries:
        raise ValueError("at least one seed summary is required")
    means = np.asarray([x["triplet_mean"] for x in seed_summaries], dtype=np.float64)
    variances = np.asarray([x["triplet_variance"] for x in seed_summaries], dtype=np.float64)
    within = float(np.mean(variances))
    between = float(np.var(means, ddof=1)) if means.size > 1 else 0.0
    return {
        "mean_within_seed_variance": within,
        "between_seed_mean_variance": between,
        "total_predictive_variance": within + between,
        "seed_count": int(means.size),
    }
