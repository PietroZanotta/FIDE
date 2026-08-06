"""Stagewise feasibility, correction, and unresolved-mode diagnostics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .energy import PhysicalParameters, mean_repulsive_energy, overlap_fraction
from .geometry import periodic_rms_displacement
from .observables import (
    PairBasis,
    angular_cosine_moments,
    ensemble_pair_moments,
)
from .statistics import bootstrap_mean_interval


def _pairwise_squared(left: Array, right: Array) -> Array:
    delta = left[:, None, :] - right[None, :, :]
    return jnp.sum(delta * delta, axis=-1)


def rbf_mmd_squared(left: Array, right: Array, bandwidth: Array) -> Array:
    """Nonnegative biased Gaussian-kernel MMD estimate."""
    denominator = 2.0 * jnp.maximum(bandwidth * bandwidth, 1e-12)
    xx = jnp.exp(-_pairwise_squared(left, left) / denominator)
    yy = jnp.exp(-_pairwise_squared(right, right) / denominator)
    xy = jnp.exp(-_pairwise_squared(left, right) / denominator)
    return jnp.maximum(jnp.mean(xx) + jnp.mean(yy) - 2.0 * jnp.mean(xy), 0.0)


def median_bandwidth(reference: Array) -> Array:
    """Median nonzero pairwise distance with a stable fallback."""
    distances = jnp.sqrt(jnp.maximum(_pairwise_squared(reference, reference), 0.0))
    mask = ~jnp.eye(reference.shape[0], dtype=bool)
    values = jnp.where(mask, distances, jnp.nan)
    return jnp.maximum(jnp.nanmedian(values), 1e-3)


def reference_angular_descriptors(
    motif_a: Array,
    motif_b: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
) -> tuple[Array, Array]:
    """Evaluation-only angular fingerprints of the two exact modes."""
    descriptor_a = angular_cosine_moments(
        motif_a[None], box, orders, neighbor_scale
    )[0]
    descriptor_b = angular_cosine_moments(
        motif_b[None], box, orders, neighbor_scale
    )[0]
    return descriptor_a, descriptor_b


def classify_modes(
    configurations: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
    reference_a: Array,
    reference_b: Array,
    angular_scale: Array,
    far_threshold: float,
) -> dict[str, Array]:
    """Classify each configuration as A, B, or far from both references."""
    leading = configurations.shape[:-2]
    flattened = configurations.reshape((-1,) + configurations.shape[-2:])
    descriptor = angular_cosine_moments(
        flattened, box, orders, neighbor_scale
    )
    scale = jnp.maximum(angular_scale, 1e-12)
    distance_a = jnp.linalg.norm((descriptor - reference_a) / scale, axis=-1)
    distance_b = jnp.linalg.norm((descriptor - reference_b) / scale, axis=-1)
    minimum = jnp.minimum(distance_a, distance_b)
    nearest = (distance_b < distance_a).astype(jnp.int32)
    label = jnp.where(minimum > far_threshold, 2, nearest)
    return {
        "descriptor": descriptor.reshape(leading + descriptor.shape[-1:]),
        "distance_a": distance_a.reshape(leading),
        "distance_b": distance_b.reshape(leading),
        "minimum_distance": minimum.reshape(leading),
        "label": label.reshape(leading),
    }


def transition_matrix(left_labels: Array, right_labels: Array) -> Array:
    """Normalized 3x3 transition matrix for A, B, and far states."""
    left = left_labels.reshape(-1)
    right = right_labels.reshape(-1)
    counts = jnp.zeros((3, 3), dtype=jnp.float64)
    counts = counts.at[left, right].add(1.0)
    return counts / jnp.maximum(jnp.sum(counts), 1.0)


def stage_arrays(
    coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    physical: PhysicalParameters,
    overlap_threshold: float,
    angular_orders: Array,
    angular_neighbor_scale: float,
    reference_a: Array,
    reference_b: Array,
    angular_scale: Array,
    far_threshold: float,
) -> dict[str, Array]:
    """Return per-ensemble arrays before uncertainty aggregation."""
    moments = jax.vmap(
        lambda ensemble: ensemble_pair_moments(ensemble, box, basis)
    )(coordinates)
    pair_error = jnp.linalg.norm(
        (moments - target_moments) / jnp.maximum(moment_scales, 1e-12), axis=-1
    )
    energy = jax.vmap(
        lambda ensemble: mean_repulsive_energy(ensemble, box, physical)
    )(coordinates)
    overlaps = jax.vmap(
        lambda ensemble: overlap_fraction(ensemble, box, overlap_threshold)
    )(coordinates)
    classification = classify_modes(
        coordinates,
        box,
        angular_orders,
        angular_neighbor_scale,
        reference_a,
        reference_b,
        angular_scale,
        far_threshold,
    )
    labels = classification["label"]
    mode_a = jnp.mean((labels == 0).astype(coordinates.dtype), axis=-1)
    mode_b = jnp.mean((labels == 1).astype(coordinates.dtype), axis=-1)
    far = jnp.mean((labels == 2).astype(coordinates.dtype), axis=-1)
    probabilities = jnp.stack((mode_a, mode_b), axis=-1)
    entropy = -jnp.sum(
        jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0.0),
        axis=-1,
    ) / jnp.log(jnp.asarray(2.0, coordinates.dtype))
    return {
        "pair_error": pair_error,
        "energy": energy,
        "overlap_fraction": overlaps,
        "mode_a_fraction": mode_a,
        "mode_b_fraction": mode_b,
        "far_fraction": far,
        "mode_entropy": entropy,
        "mean_reference_distance": jnp.mean(
            classification["minimum_distance"], axis=-1
        ),
        "labels": labels,
        "angular_descriptors": classification["descriptor"],
    }


def correction_arrays(left: Array, right: Array, box: Array) -> Array:
    """Per-ensemble periodic RMS corrections."""
    return jax.vmap(lambda a, b: periodic_rms_displacement(a, b, box))(left, right)


def summarize_stage(
    arrays: dict[str, Array],
    balanced_reference_descriptors: Array,
    angular_scale: Array,
    *,
    bootstrap_seed: int,
    num_bootstrap: int,
) -> dict[str, Any]:
    """Aggregate stage arrays with cluster-bootstrap intervals."""
    result: dict[str, Any] = {}
    scalar_names = (
        "pair_error",
        "energy",
        "overlap_fraction",
        "mode_a_fraction",
        "mode_b_fraction",
        "far_fraction",
        "mode_entropy",
        "mean_reference_distance",
    )
    for offset, name in enumerate(scalar_names):
        values = np.asarray(jax.device_get(arrays[name]))
        result[name] = bootstrap_mean_interval(
            values,
            seed=bootstrap_seed + offset,
            num_resamples=num_bootstrap,
        ).as_dict()
    predicted = arrays["angular_descriptors"].reshape(
        (-1, arrays["angular_descriptors"].shape[-1])
    )
    scale = jnp.maximum(angular_scale, 1e-12)
    predicted_white = predicted / scale
    reference_white = balanced_reference_descriptors / scale
    bandwidth = median_bandwidth(reference_white)
    result["angular_mmd2"] = float(
        rbf_mmd_squared(predicted_white, reference_white, bandwidth)
    )
    result["angular_mmd_bandwidth"] = float(bandwidth)
    return result
