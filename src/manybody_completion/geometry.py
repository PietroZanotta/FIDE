"""Smooth periodic geometry used consistently by models, solvers, and metrics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def wrap_positions(coordinates: Array, box: Array) -> Array:
    """Wrap physical coordinates into ``[0, box)``."""
    coordinates = jnp.asarray(coordinates)
    box = jnp.asarray(box, dtype=coordinates.dtype)
    return jnp.mod(coordinates, box)


def minimum_image_displacement(left: Array, right: Array, box: Array) -> Array:
    """Return the branch-valued shortest displacement ``right - left``."""
    left = jnp.asarray(left)
    right = jnp.asarray(right, dtype=left.dtype)
    box = jnp.asarray(box, dtype=left.dtype)
    delta = right - left
    return jnp.mod(delta + 0.5 * box, box) - 0.5 * box


def gauge_fixed_displacement(source: Array, target: Array, box: Array) -> Array:
    """Minimum-image displacement with global translation removed per replica."""
    delta = minimum_image_displacement(source, target, box)
    return delta - jnp.mean(delta, axis=-2, keepdims=True)


def chord_displacements(coordinates: Array, box: Array) -> Array:
    """Pairwise smooth half-period chord displacements from the methodology."""
    coordinates = jnp.asarray(coordinates)
    box = jnp.asarray(box, dtype=coordinates.dtype)
    delta = coordinates[..., :, None, :] - coordinates[..., None, :, :]
    return (box / jnp.pi) * jnp.sin(jnp.pi * delta / box)


def periodic_direction_displacements(coordinates: Array, box: Array) -> Array:
    """Oriented, smooth, fully periodic pair directions.

    The full-period sine is required for orientation-sensitive features.  The
    half-period chord is used only for the radial norm.
    """
    coordinates = jnp.asarray(coordinates)
    box = jnp.asarray(box, dtype=coordinates.dtype)
    delta = coordinates[..., :, None, :] - coordinates[..., None, :, :]
    return (box / (2.0 * jnp.pi)) * jnp.sin(2.0 * jnp.pi * delta / box)


def chord_distances(coordinates: Array, box: Array) -> Array:
    """Pairwise smooth periodic distances."""
    displacement = chord_displacements(coordinates, box)
    return jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + 1e-24)



def periodic_mean_squared_displacement(left: Array, right: Array, box: Array) -> Array:
    """Mean squared minimum-image displacement per particle."""
    delta = minimum_image_displacement(left, right, box)
    return jnp.mean(jnp.sum(delta * delta, axis=-1))


def periodic_rms_displacement(left: Array, right: Array, box: Array) -> Array:
    """RMS minimum-image displacement per particle."""
    delta = minimum_image_displacement(left, right, box)
    return jnp.sqrt(jnp.mean(jnp.sum(delta * delta, axis=-1)))
