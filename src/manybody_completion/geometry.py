"""Smooth periodic geometry shared by simulators, observables, and solvers.

Coordinates are physical coordinates in a rectangular periodic box.  The
chord displacement follows Eq. (2) of the methodology document componentwise:

    d_q(i,j) = L_q / pi * sin(pi * (x_iq - x_jq) / L_q).

Its norm is smooth across periodic branch cuts.  It is intentionally not the
Euclidean minimum-image displacement far from the origin.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _validate_last_dimension(x: Array, expected: int, name: str) -> None:
    if x.ndim < 1 or x.shape[-1] != expected:
        raise ValueError(f"{name} must have last dimension {expected}; got shape {x.shape}")


def wrap_positions(positions: Array, box: Array) -> Array:
    """Wrap physical coordinates into ``[0, box_q)`` componentwise."""
    positions = jnp.asarray(positions)
    box = jnp.asarray(box, dtype=positions.dtype)
    _validate_last_dimension(positions, 2, "positions")
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    return jnp.mod(positions, box)


def chord_displacements(positions: Array, box: Array) -> Array:
    """Return all ordered smooth periodic pair displacements.

    Args:
        positions: Array with shape ``(..., N, 2)``.
        box: Array with shape ``(2,)``.

    Returns:
        Array with shape ``(..., N, N, 2)`` where entry ``[..., i, j, :]``
        is the displacement from particle ``j`` to particle ``i``.
    """
    positions = jnp.asarray(positions)
    box = jnp.asarray(box, dtype=positions.dtype)
    _validate_last_dimension(positions, 2, "positions")
    if positions.ndim < 2:
        raise ValueError(f"positions must have shape (..., N, 2); got {positions.shape}")
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    delta = positions[..., :, None, :] - positions[..., None, :, :]
    return (box / jnp.pi) * jnp.sin(jnp.pi * delta / box)


def periodic_direction_displacements(positions: Array, box: Array) -> Array:
    """Return periodic oriented bond features for angular calculations.

    Unlike :func:`chord_displacements`, this uses a full-period sine, so the
    oriented feature is unchanged when either coordinate is shifted by an
    integer box vector.  It is locally proportional to the physical
    displacement and is intended only for smoothly weighted local angular
    descriptors.
    """
    positions = jnp.asarray(positions)
    box = jnp.asarray(box, dtype=positions.dtype)
    _validate_last_dimension(positions, 2, "positions")
    if positions.ndim < 2:
        raise ValueError(f"positions must have shape (..., N, 2); got {positions.shape}")
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    delta = positions[..., :, None, :] - positions[..., None, :, :]
    return (box / (2.0 * jnp.pi)) * jnp.sin(2.0 * jnp.pi * delta / box)


def chord_squared_distances(positions: Array, box: Array) -> Array:
    """Return squared chord distances with shape ``(..., N, N)``."""
    displacement = chord_displacements(positions, box)
    return jnp.sum(displacement * displacement, axis=-1)


def chord_distances(positions: Array, box: Array, *, epsilon: float = 1e-12) -> Array:
    """Return differentiable chord distances with shape ``(..., N, N)``."""
    r2 = chord_squared_distances(positions, box)
    return jnp.sqrt(r2 + jnp.asarray(epsilon, dtype=r2.dtype))


def off_diagonal_mask(num_particles: int, dtype: jnp.dtype = jnp.float32) -> Array:
    """Return an ``(N, N)`` mask that is one exactly off the diagonal."""
    return (1.0 - jnp.eye(num_particles, dtype=dtype)).astype(dtype)


def translate(positions: Array, shift: Array, box: Array) -> Array:
    """Translate all particles and wrap the result."""
    return wrap_positions(jnp.asarray(positions) + jnp.asarray(shift), box)
