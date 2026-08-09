"""Held-out higher-order uncertainty summaries.

The finite benchmark lets us evaluate predictive intervals exactly against the
reference law.  Monte Carlo uncertainty for a returned ensemble is summarized
with effective-sample-size approximations; multi-seed aggregation separates
within-seed predictive variance from between-seed model variance.
"""

from __future__ import annotations

import math
import numpy as np

from .metrics import binary_entropy, normalized, weighted_mean, weighted_variance


def weighted_quantile(values: np.ndarray, probabilities: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    p = normalized(probabilities)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(p[order])
    return float(sorted_values[np.searchsorted(cumulative, quantile, side="left")])


def _normal_interval(mean: float, standard_error: float) -> dict[str, float]:
    radius = 1.959963984540054 * max(float(standard_error), 0.0)
    return {"lower": float(mean - radius), "upper": float(mean + radius)}


def _probability_interval(probability: float, effective_sample_size: float) -> dict[str, float]:
    p = min(max(float(probability), 0.0), 1.0)
    n = max(float(effective_sample_size), 1.0)
    standard_error = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    interval = _normal_interval(p, standard_error)
    interval["lower"] = max(0.0, interval["lower"])
    interval["upper"] = min(1.0, interval["upper"])
    return interval


def _expected_interval_score(
    values: np.ndarray,
    reference_probabilities: np.ndarray,
    lower: float,
    upper: float,
    alpha: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    reference = normalized(reference_probabilities)
    score = np.full(values.shape, upper - lower, dtype=np.float64)
    below = values < lower
    above = values > upper
    score[below] += (2.0 / alpha) * (lower - values[below])
    score[above] += (2.0 / alpha) * (values[above] - upper)
    return float(np.sum(reference * score))


def summarize_higher_order(
    triplet_values: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    reference_probabilities: np.ndarray | None = None,
    effective_sample_size: float | None = None,
) -> dict:
    p = normalized(probabilities)
    values = np.asarray(triplet_values, dtype=np.float64)
    labels_array = np.asarray(labels)
    mean = weighted_mean(values, p)
    variance = weighted_variance(values, p)
    exact_evaluation = effective_sample_size is None
    effective_n = None if exact_evaluation else max(float(effective_sample_size), 1.0)
    mean_standard_error = (
        0.0 if exact_evaluation else math.sqrt(max(variance, 0.0) / effective_n)
    )

    intervals: dict[str, dict[str, float]] = {}
    reference = normalized(reference_probabilities) if reference_probabilities is not None else None
    for level in (0.50, 0.80, 0.90, 0.95):
        alpha = (1.0 - level) / 2.0
        lower = weighted_quantile(values, p, alpha)
        upper = weighted_quantile(values, p, 1.0 - alpha)
        entry: dict[str, float] = {"lower": lower, "upper": upper}
        if reference is not None:
            entry["reference_coverage"] = float(
                np.sum(reference[(values >= lower) & (values <= upper)])
            )
            entry["expected_interval_score"] = _expected_interval_score(
                values, reference, lower, upper, 1.0 - level
            )
        intervals[f"{int(level * 100)}"] = entry

    mode_plus = float(np.sum(p[labels_array > 0]))
    mode_standard_error = (
        0.0
        if exact_evaluation
        else math.sqrt(max(mode_plus * (1.0 - mode_plus), 0.0) / effective_n)
    )
    result = {
        "triplet_mean": mean,
        "triplet_variance": variance,
        "triplet_standard_deviation": float(np.sqrt(max(variance, 0.0))),
        "triplet_mean_standard_error": mean_standard_error,
        "triplet_mean_95_interval": _normal_interval(mean, mean_standard_error),
        "predictive_intervals": intervals,
        "mode_plus_probability": mode_plus,
        "mode_probability_standard_error": mode_standard_error,
        "mode_probability_95_interval": (
            {"lower": mode_plus, "upper": mode_plus}
            if exact_evaluation
            else _probability_interval(mode_plus, effective_n)
        ),
        "mode_entropy_bits": binary_entropy(mode_plus),
        "effective_sample_size_for_uq": effective_n,
        "monte_carlo_uq_method": (
            "exact finite-support evaluation"
            if exact_evaluation
            else "ESS normal approximation"
        ),
    }
    if reference is not None:
        reference_mode = float(np.sum(reference[labels_array > 0]))
        clipped = min(max(mode_plus, 1e-15), 1.0 - 1e-15)
        result.update(
            {
                "reference_mode_plus_probability": reference_mode,
                "mode_probability_error": abs(mode_plus - reference_mode),
                "expected_mode_brier_score": (
                    reference_mode * (1.0 - mode_plus) ** 2
                    + (1.0 - reference_mode) * mode_plus**2
                ),
                "expected_mode_log_score": -(
                    reference_mode * math.log(clipped)
                    + (1.0 - reference_mode) * math.log(1.0 - clipped)
                ),
            }
        )
    return result


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
