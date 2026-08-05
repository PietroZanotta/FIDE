"""Held-out higher-order diagnostics for generated many-body ensembles.

These functions are evaluation-only.  They operate on angular descriptors that
are intentionally absent from the generator condition vector and projection
constraints.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .observables import ensemble_angular_cosine_moments


def batch_ensemble_angular_moments(
    coordinates: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
) -> Array:
    """Compute one angular descriptor vector per sampled ensemble."""
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError(
            f"coordinates must have shape (B, M, N, 2); got {coordinates.shape}"
        )
    return jax.vmap(
        lambda sample: ensemble_angular_cosine_moments(
            sample, box, orders, neighbor_scale
        )
    )(coordinates)


def _pairwise_squared_distances(left: Array, right: Array) -> Array:
    delta = left[:, None, :] - right[None, :, :]
    return jnp.sum(delta * delta, axis=-1)


def biased_rbf_mmd_squared(left: Array, right: Array, bandwidth: Array | float) -> Array:
    """Nonnegative biased MMD estimate with a Gaussian kernel."""
    left = jnp.asarray(left)
    right = jnp.asarray(right, dtype=left.dtype)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right must have shapes (A, K) and (B, K)")
    if left.shape[0] < 1 or right.shape[0] < 1:
        raise ValueError("MMD inputs must be nonempty")
    bandwidth = jnp.asarray(bandwidth, dtype=left.dtype)
    denominator = 2.0 * jnp.maximum(bandwidth * bandwidth, 1e-15)
    kernel_xx = jnp.exp(-_pairwise_squared_distances(left, left) / denominator)
    kernel_yy = jnp.exp(-_pairwise_squared_distances(right, right) / denominator)
    kernel_xy = jnp.exp(-_pairwise_squared_distances(left, right) / denominator)
    estimate = jnp.mean(kernel_xx) + jnp.mean(kernel_yy) - 2.0 * jnp.mean(kernel_xy)
    return jnp.maximum(estimate, 0.0)


def median_reference_bandwidth(reference_whitened: Array) -> Array:
    """Median nonzero pairwise distance with a stable positive fallback."""
    reference_whitened = jnp.asarray(reference_whitened)
    if reference_whitened.ndim != 2 or reference_whitened.shape[0] < 2:
        raise ValueError("reference_whitened must contain at least two vectors")
    distances = jnp.sqrt(
        jnp.maximum(_pairwise_squared_distances(reference_whitened, reference_whitened), 0.0)
    )
    mask = ~jnp.eye(reference_whitened.shape[0], dtype=bool)
    values = jnp.where(mask, distances, jnp.nan)
    median = jnp.nanmedian(values)
    return jnp.maximum(median, jnp.asarray(1e-3, dtype=reference_whitened.dtype))


def angular_distribution_diagnostics(
    predicted: Array,
    reference: Array,
    labels: Array,
    angular_scale: Array,
    *,
    bandwidth: Array | float,
) -> dict[str, Array]:
    """Compare held-out angular vectors without feeding them to training."""
    predicted = jnp.asarray(predicted)
    reference = jnp.asarray(reference, dtype=predicted.dtype)
    labels = jnp.asarray(labels)
    angular_scale = jnp.asarray(angular_scale, dtype=predicted.dtype)
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("predicted and reference must have the same shape (B, K)")
    if labels.shape != (predicted.shape[0],):
        raise ValueError("labels must have shape (B,)")
    if angular_scale.shape != (predicted.shape[1],):
        raise ValueError("angular_scale must have shape (K,)")

    scale = jnp.maximum(angular_scale, 1e-15)
    predicted_white = predicted / scale
    reference_white = reference / scale
    difference = predicted - reference
    difference_white = difference / scale
    raw_rmse = jnp.sqrt(jnp.mean(difference * difference))
    whitened_rmse = jnp.sqrt(jnp.mean(difference_white * difference_white))
    pooled_mmd = biased_rbf_mmd_squared(
        predicted_white, reference_white, bandwidth
    )

    unique_labels = jnp.unique(labels, size=labels.shape[0], fill_value=-1)
    valid_labels = unique_labels[unique_labels >= 0]
    # The calibration benchmark currently has two regimes.  Requiring exactly
    # two makes the separation metric unambiguous rather than averaging an
    # arbitrary collection of pairwise distances.
    if int(valid_labels.shape[0]) != 2:
        raise ValueError("angular regime diagnostics require exactly two labels")
    first, second = valid_labels[0], valid_labels[1]
    first_mask = labels == first
    second_mask = labels == second
    reference_first = jnp.mean(reference_white[first_mask], axis=0)
    reference_second = jnp.mean(reference_white[second_mask], axis=0)
    predicted_first = jnp.mean(predicted_white[first_mask], axis=0)
    predicted_second = jnp.mean(predicted_white[second_mask], axis=0)
    reference_delta = reference_second - reference_first
    predicted_delta = predicted_second - predicted_first
    reference_separation = jnp.linalg.norm(reference_delta)
    predicted_separation = jnp.linalg.norm(predicted_delta)
    alignment = jnp.vdot(reference_delta, predicted_delta) / jnp.maximum(
        reference_separation * predicted_separation, 1e-15
    )
    mmd_first = biased_rbf_mmd_squared(
        predicted_white[first_mask], reference_white[first_mask], bandwidth
    )
    mmd_second = biased_rbf_mmd_squared(
        predicted_white[second_mask], reference_white[second_mask], bandwidth
    )
    return {
        "angular_rmse": raw_rmse,
        "angular_whitened_rmse": whitened_rmse,
        "angular_mmd2": pooled_mmd,
        "angular_regime_mmd2_mean": 0.5 * (mmd_first + mmd_second),
        "reference_regime_centroid_separation": reference_separation,
        "predicted_regime_centroid_separation": predicted_separation,
        "regime_separation_ratio": predicted_separation
        / jnp.maximum(reference_separation, 1e-15),
        "regime_separation_alignment": alignment,
    }


def evaluate_angular_stages(
    coordinates_by_stage: Mapping[str, Array],
    reference: Array,
    labels: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
    angular_scale: Array,
    *,
    bandwidth: Array | float,
) -> dict[str, dict[str, Array]]:
    """Compute descriptor vectors and diagnostics for named coordinate stages."""
    output: dict[str, dict[str, Array]] = {}
    for stage, coordinates in coordinates_by_stage.items():
        predicted = batch_ensemble_angular_moments(
            coordinates, box, orders, neighbor_scale
        )
        output[stage] = {
            "moments": predicted,
            "diagnostics": angular_distribution_diagnostics(
                predicted,
                reference,
                labels,
                angular_scale,
                bandwidth=bandwidth,
            ),
        }
    return output
