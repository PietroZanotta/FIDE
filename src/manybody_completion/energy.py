"""Smooth physical plausibility diagnostics and proximal energy."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances


@dataclass(frozen=True)
class PhysicalParameters:
    r0: float = 0.24
    kappa: float = 30.0


def repulsive_energy_per_configuration(
    coordinates: Array,
    box: Array,
    parameters: PhysicalParameters,
) -> Array:
    """Mean ordered-pair smooth repulsive energy."""
    coordinates = jnp.asarray(coordinates)
    distances = chord_distances(coordinates, box)
    num_particles = coordinates.shape[-2]
    mask = 1.0 - jnp.eye(num_particles, dtype=coordinates.dtype)
    penalty = jax.nn.softplus(parameters.kappa * (parameters.r0 - distances))
    penalty = (penalty / parameters.kappa) ** 2
    denominator = jnp.asarray(num_particles * (num_particles - 1), coordinates.dtype)
    return jnp.sum(mask * penalty, axis=(-2, -1)) / denominator


def mean_repulsive_energy(
    ensemble: Array,
    box: Array,
    parameters: PhysicalParameters,
) -> Array:
    """Average repulsive energy over replicas."""
    return jnp.mean(repulsive_energy_per_configuration(ensemble, box, parameters))


def overlap_fraction(ensemble: Array, box: Array, threshold: float) -> Array:
    """Fraction of unordered pairs below a smooth chord-distance threshold."""
    distances = chord_distances(ensemble, box)
    num_particles = ensemble.shape[-2]
    rows, columns = jnp.triu_indices(num_particles, k=1)
    return jnp.mean((distances[..., rows, columns] < threshold).astype(ensemble.dtype))
