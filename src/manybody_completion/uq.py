"""Higher-order conditional uncertainty quantification.

All summaries condition on the same observed pair vector ``c``.  Predictive
uncertainty is calculated from held-out angular descriptors and never feeds
back into training, RMC, IBI, relaxation, or moment projection.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .statistics import bootstrap_mean_interval


def _validate_descriptor_arrays(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if predicted_array.ndim != 3:
        raise ValueError("predicted must have shape (E, M, K)")
    if reference_array.ndim != 2:
        raise ValueError("reference must have shape (R, K)")
    if predicted_array.shape[-1] != reference_array.shape[-1]:
        raise ValueError("predicted and reference descriptor dimensions differ")
    if predicted_array.shape[0] < 2 or reference_array.shape[0] < 2:
        raise ValueError("at least two predictive clusters and references are required")
    return predicted_array, reference_array


def _bootstrap_quantile_endpoints(
    predicted: np.ndarray,
    lower_probability: float,
    upper_probability: float,
    *,
    seed: int,
    num_resamples: int,
    confidence: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_clusters = predicted.shape[0]
    estimate_flat = predicted.reshape((-1, predicted.shape[-1]))
    estimate_lower = np.quantile(estimate_flat, lower_probability, axis=0)
    estimate_upper = np.quantile(estimate_flat, upper_probability, axis=0)
    lower_samples = np.empty((num_resamples, predicted.shape[-1]))
    upper_samples = np.empty_like(lower_samples)
    for index in range(num_resamples):
        selected = rng.integers(0, num_clusters, size=num_clusters)
        flattened = predicted[selected].reshape((-1, predicted.shape[-1]))
        lower_samples[index] = np.quantile(flattened, lower_probability, axis=0)
        upper_samples[index] = np.quantile(flattened, upper_probability, axis=0)
    alpha = 0.5 * (1.0 - confidence)
    return {
        "lower": {
            "estimate": estimate_lower,
            "confidence_lower": np.quantile(lower_samples, alpha, axis=0),
            "confidence_upper": np.quantile(lower_samples, 1.0 - alpha, axis=0),
        },
        "upper": {
            "estimate": estimate_upper,
            "confidence_lower": np.quantile(upper_samples, alpha, axis=0),
            "confidence_upper": np.quantile(upper_samples, 1.0 - alpha, axis=0),
        },
    }


def _interval_score(
    observations: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    width = upper - lower
    below = (2.0 / alpha) * np.maximum(lower - observations, 0.0)
    above = (2.0 / alpha) * np.maximum(observations - upper, 0.0)
    return width + below + above


def multivariate_energy_score(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 256,
    seed: int = 0,
) -> float:
    """Sample energy score averaged over held-out reference descriptors."""
    predicted_array = np.asarray(predicted, dtype=np.float64)
    predicted_array = predicted_array.reshape((-1, predicted_array.shape[-1]))
    reference_array = np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if predicted_array.shape[0] > max_samples:
        predicted_array = predicted_array[
            rng.choice(predicted_array.shape[0], max_samples, replace=False)
        ]
    if reference_array.shape[0] > max_samples:
        reference_array = reference_array[
            rng.choice(reference_array.shape[0], max_samples, replace=False)
        ]
    cross = np.linalg.norm(
        predicted_array[:, None, :] - reference_array[None, :, :], axis=-1
    )
    within = np.linalg.norm(
        predicted_array[:, None, :] - predicted_array[None, :, :], axis=-1
    )
    return float(np.mean(cross) - 0.5 * np.mean(within))


def higher_order_conditional_uq(
    predicted_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    labels: np.ndarray,
    *,
    interval_levels: Iterable[float] = (0.5, 0.8, 0.9),
    seed: int,
    num_resamples: int,
    confidence: float = 0.95,
    target_mode_probabilities: tuple[float, float, float] = (0.5, 0.5, 0.0),
) -> dict[str, Any]:
    """Summarize conditional higher-order predictive uncertainty.

    ``labels`` has shape ``(E, M)`` and contains 0=A, 1=B, and 2=far.  Cluster
    bootstraps resample the independent ensemble axis, preserving the replicas
    within each ensemble.
    """
    predicted, reference = _validate_descriptor_arrays(
        predicted_descriptors,
        reference_descriptors,
    )
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.shape != predicted.shape[:2]:
        raise ValueError("labels must have shape (E, M)")
    if num_resamples < 100:
        raise ValueError("num_resamples must be at least 100")

    flattened = predicted.reshape((-1, predicted.shape[-1]))
    predictive_mean = np.mean(flattened, axis=0)
    predictive_variance = np.var(flattened, axis=0)
    ensemble_means = np.mean(predicted, axis=1)
    within_ensemble_variance = np.mean(np.var(predicted, axis=1), axis=0)
    between_ensemble_variance = np.var(ensemble_means, axis=0)

    intervals: dict[str, Any] = {}
    for offset, level in enumerate(interval_levels):
        level_value = float(level)
        if not 0.0 < level_value < 1.0:
            raise ValueError("interval levels must lie in (0, 1)")
        alpha = 1.0 - level_value
        endpoints = _bootstrap_quantile_endpoints(
            predicted,
            0.5 * alpha,
            1.0 - 0.5 * alpha,
            seed=seed + 101 * offset,
            num_resamples=num_resamples,
            confidence=confidence,
        )
        lower = endpoints["lower"]["estimate"]
        upper = endpoints["upper"]["estimate"]
        covered = (reference >= lower) & (reference <= upper)
        score = _interval_score(reference, lower, upper, alpha)
        intervals[f"{level_value:.2f}"] = {
            **endpoints,
            "width": upper - lower,
            "reference_coverage_per_dimension": np.mean(covered, axis=0),
            "reference_coverage_mean": float(np.mean(covered)),
            "interval_score_per_dimension": np.mean(score, axis=0),
            "interval_score_mean": float(np.mean(score)),
        }

    per_ensemble_probabilities = np.stack(
        [np.mean(label_array == value, axis=1) for value in (0, 1, 2)], axis=-1
    )
    mode_intervals = {
        name: bootstrap_mean_interval(
            per_ensemble_probabilities[:, index],
            seed=seed + 1000 + index,
            num_resamples=num_resamples,
            confidence=confidence,
        ).as_dict()
        for index, name in enumerate(("A", "B", "far"))
    }
    estimated_probabilities = np.mean(per_ensemble_probabilities, axis=0)
    target = np.asarray(target_mode_probabilities, dtype=np.float64)
    normalized = estimated_probabilities[:2] / max(
        np.sum(estimated_probabilities[:2]), 1e-15
    )
    positive_probabilities = normalized[normalized > 0.0]
    mode_entropy = -np.sum(
        positive_probabilities * np.log(positive_probabilities)
    ) / np.log(2.0)

    return {
        "condition": "shared exact pair-statistic vector c",
        "descriptor_dimension": int(predicted.shape[-1]),
        "num_predictive_ensembles": int(predicted.shape[0]),
        "replicas_per_ensemble": int(predicted.shape[1]),
        "num_reference_configurations": int(reference.shape[0]),
        "predictive_mean": predictive_mean,
        "predictive_variance": predictive_variance,
        "sampling_variance_within_ensemble": within_ensemble_variance,
        "between_ensemble_variance": between_ensemble_variance,
        "predictive_intervals": intervals,
        "multivariate_energy_score": multivariate_energy_score(
            predicted,
            reference,
            seed=seed + 2000,
        ),
        "mode_probability_intervals": mode_intervals,
        "mode_probability_estimate": {
            "A": float(estimated_probabilities[0]),
            "B": float(estimated_probabilities[1]),
            "far": float(estimated_probabilities[2]),
        },
        "mode_probability_target": {
            "A": float(target[0]),
            "B": float(target[1]),
            "far": float(target[2]),
        },
        "mode_probability_l1_error": float(np.sum(np.abs(estimated_probabilities - target))),
        "mode_probability_total_variation": float(
            0.5 * np.sum(np.abs(estimated_probabilities - target))
        ),
        "normalized_mode_entropy": float(mode_entropy),
        "epistemic_status": "requires multiple independently trained seeds",
    }


def aggregate_seed_higher_order_uq(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Decompose predictive variance across independently trained model seeds."""
    if len(seed_summaries) < 2:
        raise ValueError("at least two seed summaries are required")
    means = np.asarray([summary["predictive_mean"] for summary in seed_summaries])
    variances = np.asarray(
        [summary["predictive_variance"] for summary in seed_summaries]
    )
    aleatoric = np.mean(variances, axis=0)
    epistemic = np.var(means, axis=0, ddof=1)
    return {
        "num_training_seeds": len(seed_summaries),
        "conditional_mean_across_seeds": np.mean(means, axis=0),
        "aleatoric_variance": aleatoric,
        "epistemic_variance": epistemic,
        "total_predictive_variance": aleatoric + epistemic,
        "epistemic_fraction": epistemic / np.maximum(aleatoric + epistemic, 1e-15),
    }
