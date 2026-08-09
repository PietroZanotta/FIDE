"""Smooth pair and three-body energies for synthetic data families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances, off_diagonal_mask, periodic_direction_displacements

EnergyFamily = Literal["pair", "angular"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EnergyParameters:
    """Parameters shared by the pair-only and angular data families."""

    r0: float = 0.16
    kappa: float = 40.0
    pair_strength: float = 1.0
    angular_strength: float = 0.0
    angular_target_cosine: float = 0.0
    angular_neighbor_scale: float = 0.24

    def tree_flatten(self):
        children = (
            self.r0,
            self.kappa,
            self.pair_strength,
            self.angular_strength,
            self.angular_target_cosine,
            self.angular_neighbor_scale,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(*children)


def soft_repulsive_energy_per_configuration(
    coordinates: Array,
    box: Array,
    r0: float,
    kappa: float,
) -> Array:
    """Smooth repulsive energy from Eq. (10), one value per configuration."""
    coordinates = jnp.asarray(coordinates)
    n = coordinates.shape[-2]
    distances = chord_distances(coordinates, box)
    penalty = jax.nn.softplus(kappa * (r0 - distances)) ** 2
    mask = off_diagonal_mask(n, dtype=coordinates.dtype)
    return jnp.sum(penalty * mask, axis=(-2, -1)) / (n * (n - 1))


def angular_energy_per_configuration(
    coordinates: Array,
    box: Array,
    target_cosine: float,
    neighbor_scale: float,
    epsilon: float = 1e-8,
) -> Array:
    """Smooth local three-body energy that favors a target bond angle.

    This family creates hidden angular structure while remaining fully
    exchangeable.  It is intended for the ambiguity benchmark, not as a claim
    of a complete physical model.
    """
    coordinates = jnp.asarray(coordinates)
    n = coordinates.shape[-2]
    direction = periodic_direction_displacements(coordinates, box)
    direction_norm = jnp.sqrt(jnp.sum(direction * direction, axis=-1) + epsilon)
    unit = direction / direction_norm[..., None]
    radii = chord_distances(coordinates, box)
    cosine = jnp.einsum("...ijq,...ikq->...ijk", unit, unit)
    cosine = jnp.clip(cosine, -1.0, 1.0)

    radial_weight = jnp.exp(-0.5 * (radii / neighbor_scale) ** 2)
    weights = radial_weight[..., :, :, None] * radial_weight[..., :, None, :]

    off_diag = 1.0 - jnp.eye(n, dtype=coordinates.dtype)
    valid = off_diag[:, :, None] * off_diag[:, None, :] * off_diag[None, :, :]
    while valid.ndim < weights.ndim:
        valid = valid[None, ...]
    weights = weights * valid

    mismatch = (cosine - target_cosine) ** 2
    denominator = jnp.maximum(jnp.sum(weights, axis=(-3, -2, -1)), epsilon)
    return jnp.sum(weights * mismatch, axis=(-3, -2, -1)) / denominator


def total_energy_per_configuration(
    coordinates: Array,
    box: Array,
    params: EnergyParameters,
    family: EnergyFamily,
) -> Array:
    pair = params.pair_strength * soft_repulsive_energy_per_configuration(
        coordinates, box, params.r0, params.kappa
    )
    if family == "pair":
        return pair
    if family == "angular":
        angular = angular_energy_per_configuration(
            coordinates,
            box,
            params.angular_target_cosine,
            params.angular_neighbor_scale,
        )
        return pair + params.angular_strength * angular
    raise ValueError(f"unknown energy family: {family}")


def total_energy(
    coordinates: Array,
    box: Array,
    params: EnergyParameters,
    family: EnergyFamily,
) -> Array:
    """Sum energy over all leading configurations for independent dynamics."""
    return jnp.sum(total_energy_per_configuration(coordinates, box, params, family))
