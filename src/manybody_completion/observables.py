"""Observed pair statistics and evaluation-only angular descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances, periodic_direction_displacements


@dataclass(frozen=True)
class PairBasis:
    """Gaussian radial basis for reduced pair observations."""

    centers: Array
    widths: Array

    @classmethod
    def uniform(
        cls,
        num_basis: int,
        r_min: float,
        r_max: float,
        width: float,
        *,
        dtype: jnp.dtype,
    ) -> "PairBasis":
        centers = jnp.linspace(r_min, r_max, num_basis, dtype=dtype)
        widths = jnp.full((num_basis,), width, dtype=dtype)
        return cls(centers=centers, widths=widths)


def pair_moments(coordinates: Array, box: Array, basis: PairBasis) -> Array:
    """Mean ordered-pair Gaussian radial coefficients for each configuration."""
    coordinates = jnp.asarray(coordinates)
    distances = chord_distances(coordinates, box)
    num_particles = coordinates.shape[-2]
    mask = 1.0 - jnp.eye(num_particles, dtype=coordinates.dtype)
    standardized = (
        distances[..., None] - jnp.asarray(basis.centers, dtype=coordinates.dtype)
    ) / jnp.asarray(basis.widths, dtype=coordinates.dtype)
    values = jnp.exp(-0.5 * standardized * standardized) * mask[..., None]
    denominator = jnp.asarray(num_particles * (num_particles - 1), coordinates.dtype)
    return jnp.sum(values, axis=(-3, -2)) / denominator


def ensemble_pair_moments(ensemble: Array, box: Array, basis: PairBasis) -> Array:
    """Distribution-level pair coefficients averaged over replicas."""
    return jnp.mean(pair_moments(ensemble, box, basis), axis=-2)


def angular_cosine_moments(
    coordinates: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
) -> Array:
    """Smooth weighted bond-angle cosine moments per configuration.

    These descriptors are invariant to translation and particle permutation,
    and equivariant under neither hidden labels nor conditions.  They are used
    only for evaluation in the homometric benchmark.
    """
    coordinates = jnp.asarray(coordinates)
    dtype = coordinates.dtype
    directions = periodic_direction_displacements(coordinates, box)
    radial = chord_distances(coordinates, box)
    num_particles = coordinates.shape[-2]
    eye = jnp.eye(num_particles, dtype=dtype)
    neighbor_weight = jnp.exp(-((radial / neighbor_scale) ** 2)) * (1.0 - eye)
    norm = jnp.sqrt(jnp.sum(directions * directions, axis=-1) + 1e-18)
    unit = directions / norm[..., None]

    # At center i, vectors point from i to j after a harmless sign reversal.
    cosine = jnp.einsum("...ijd,...ikd->...ijk", unit, unit)
    triplet_weight = neighbor_weight[..., :, :, None] * neighbor_weight[..., :, None, :]
    distinct_neighbors = 1.0 - jnp.eye(num_particles, dtype=dtype)
    triplet_weight = triplet_weight * distinct_neighbors[None, ...]
    denominator = jnp.maximum(jnp.sum(triplet_weight, axis=(-3, -2, -1)), 1e-18)
    orders = jnp.asarray(orders, dtype=dtype)
    powered = cosine[..., None] ** orders
    numerator = jnp.sum(
        triplet_weight[..., None] * powered,
        axis=(-4, -3, -2),
    )
    return numerator / denominator[..., None]


def ensemble_angular_moments(
    ensemble: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
) -> Array:
    """Average angular descriptors over replicas."""
    return jnp.mean(
        angular_cosine_moments(ensemble, box, orders, neighbor_scale), axis=-2
    )
