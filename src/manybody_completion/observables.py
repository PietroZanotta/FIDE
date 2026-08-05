"""Reduced pair statistics and held-out angular descriptors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances, off_diagonal_mask, periodic_direction_displacements


@dataclass(frozen=True)
class PairBasis:
    """Gaussian radial basis used for smooth pair coefficients."""

    centers: Array
    widths: Array

    @classmethod
    def uniform(
        cls,
        num_basis: int,
        r_min: float,
        r_max: float,
        width: float | None = None,
        dtype: jnp.dtype = jnp.float64,
    ) -> "PairBasis":
        if num_basis < 1:
            raise ValueError("num_basis must be positive")
        centers = jnp.linspace(r_min, r_max, num_basis, dtype=dtype)
        if width is None:
            width = (r_max - r_min) / max(num_basis - 1, 1)
        widths = jnp.full((num_basis,), width, dtype=dtype)
        return cls(centers=centers, widths=widths)

    def validate(self) -> None:
        if self.centers.ndim != 1 or self.widths.ndim != 1:
            raise ValueError("centers and widths must both be one-dimensional")
        if self.centers.shape != self.widths.shape:
            raise ValueError("centers and widths must have the same shape")


def radial_basis_values(distances: Array, basis: PairBasis) -> Array:
    """Evaluate normalized Gaussian RBFs at distances.

    The final axis indexes basis functions.
    """
    basis.validate()
    distances = jnp.asarray(distances)
    centers = jnp.asarray(basis.centers, dtype=distances.dtype)
    widths = jnp.asarray(basis.widths, dtype=distances.dtype)
    scaled = (distances[..., None] - centers) / widths
    return jnp.exp(-0.5 * scaled * scaled)


def per_configuration_pair_moments(
    coordinates: Array,
    box: Array,
    basis: PairBasis,
) -> Array:
    """Compute Eq. (3) for every configuration.

    Args:
        coordinates: ``(..., N, 2)``.

    Returns:
        ``(..., R)`` ordered-pair averages normalized by ``N(N-1)``.
    """
    coordinates = jnp.asarray(coordinates)
    n = coordinates.shape[-2]
    if n < 2:
        raise ValueError("at least two particles are required")
    distances = chord_distances(coordinates, box)
    values = radial_basis_values(distances, basis)
    mask = off_diagonal_mask(n, dtype=coordinates.dtype)
    masked = values * mask[..., None]
    return jnp.sum(masked, axis=(-3, -2)) / (n * (n - 1))


def ensemble_pair_moments(coordinates: Array, box: Array, basis: PairBasis) -> Array:
    """Compute Eq. (4) for an ensemble with shape ``(M, N, 2)``."""
    if coordinates.ndim != 3:
        raise ValueError(f"coordinates must have shape (M, N, 2); got {coordinates.shape}")
    return jnp.mean(per_configuration_pair_moments(coordinates, box, basis), axis=0)


def pair_diagnostics(
    coordinates: Array,
    box: Array,
    overlap_threshold: float,
) -> dict[str, Array]:
    """Return minimum pair distance and overlap fraction."""
    coordinates = jnp.asarray(coordinates)
    n = coordinates.shape[-2]
    distances = chord_distances(coordinates, box)
    mask = off_diagonal_mask(n, dtype=bool)
    broadcast_mask = jnp.broadcast_to(mask, distances.shape)
    minimum = jnp.min(jnp.where(broadcast_mask, distances, jnp.inf), axis=(-2, -1))
    overlaps = jnp.sum((distances < overlap_threshold) & broadcast_mask, axis=(-2, -1))
    overlap_fraction = overlaps / (n * (n - 1))
    return {
        "minimum_pair_distance": minimum,
        "overlap_fraction": overlap_fraction,
    }


def angular_cosine_moments(
    coordinates: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
    epsilon: float = 1e-8,
) -> Array:
    """Held-out smooth angular descriptor for each configuration.

    For each center particle ``i`` and ordered distinct neighbors ``j,k``, the
    descriptor averages ``cos(order * theta_jik)`` with smooth radial weights.
    This is permutation- and translation-invariant and is not used as a pair
    projection constraint.
    """
    coordinates = jnp.asarray(coordinates)
    orders = jnp.asarray(orders, dtype=coordinates.dtype)
    n = coordinates.shape[-2]
    direction = periodic_direction_displacements(coordinates, box)  # (..., i, j, q)
    direction_norm = jnp.sqrt(jnp.sum(direction * direction, axis=-1) + epsilon)
    unit = direction / direction_norm[..., None]
    radii = chord_distances(coordinates, box)

    # (..., i, j, k): angle between vectors i<-j and i<-k.
    cosine = jnp.einsum("...ijq,...ikq->...ijk", unit, unit)
    cosine = jnp.clip(cosine, -1.0, 1.0)
    theta = jnp.arccos(cosine)

    neighbor_weight = jnp.exp(-0.5 * (radii / neighbor_scale) ** 2)
    weights = neighbor_weight[..., :, :, None] * neighbor_weight[..., :, None, :]

    particle_mask = 1.0 - jnp.eye(n, dtype=coordinates.dtype)
    valid = (
        particle_mask[None, :, :, None] * particle_mask[None, :, None, :]
        if coordinates.ndim == 3
        else particle_mask[:, :, None] * particle_mask[:, None, :]
    )
    # Remove j == k. Broadcasting covers arbitrary leading dimensions.
    valid_jk = 1.0 - jnp.eye(n, dtype=coordinates.dtype)
    valid = valid * valid_jk
    while valid.ndim < weights.ndim:
        valid = valid[None, ...]
    weights = weights * valid

    features = jnp.cos(theta[..., None] * orders)
    numerator = jnp.sum(weights[..., None] * features, axis=(-4, -3, -2))
    denominator = jnp.sum(weights, axis=(-3, -2, -1))
    return numerator / jnp.maximum(denominator[..., None], epsilon)


def ensemble_angular_cosine_moments(
    coordinates: Array,
    box: Array,
    orders: Array,
    neighbor_scale: float,
    epsilon: float = 1e-8,
) -> Array:
    """Average held-out angular descriptors across an ensemble.

    Args:
        coordinates: Periodic ensemble with shape ``(M, N, 2)``.

    Returns:
        One descriptor vector with shape ``(K,)``, where ``K`` is the number
        of requested angular orders.  Keeping this ensemble reduction explicit
        mirrors :func:`ensemble_pair_moments` and prevents accidental use of
        per-replica held-out statistics in evaluation code.
    """
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError(
            f"coordinates must have shape (M, N, 2); got {coordinates.shape}"
        )
    return jnp.mean(
        angular_cosine_moments(
            coordinates,
            box,
            orders,
            neighbor_scale,
            epsilon,
        ),
        axis=0,
    )
