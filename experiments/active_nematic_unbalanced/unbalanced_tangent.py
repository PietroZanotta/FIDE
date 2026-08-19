"""Finite-dimensional unbalanced tangent lower model for raw observables."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


class UnbalancedTangentResult(NamedTuple):
    total_action: Array
    transport_action: Array
    reaction_action: Array
    coefficients: Array
    residual: Array
    gram: Array


def unbalanced_tangent_action(
    *,
    phi: Array,
    grad_phi: Array,
    velocity: Array,
    normalized_weights: Array,
    mass: Array | float,
    target_raw_moments: Array,
    target_raw_moment_dot: Array,
    reference_source_rate: Array | float,
    reaction_kappa: float,
    ridge: float = 1.0e-9,
) -> UnbalancedTangentResult:
    """Minimum correction cost for finite-measure moment-rate constraints.

    For raw observables Phi, the correction map is

        (delta, alpha) -> integral (J Phi delta + Phi alpha) d mu.

    Under the cost ``integral (|delta|^2+kappa alpha^2) d mu``, its Gram matrix
    is ``integral [JPhi JPhi^T + Phi Phi^T/kappa] d mu``.  The returned transport
    and reaction terms are evaluated from the same minimizing coefficients.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64)
    grad_phi = jnp.asarray(grad_phi, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    weights = jnp.asarray(normalized_weights, dtype=jnp.float64)
    mass = jnp.asarray(mass, dtype=jnp.float64)
    target = jnp.asarray(target_raw_moments, dtype=jnp.float64)
    target_dot = jnp.asarray(target_raw_moment_dot, dtype=jnp.float64)
    if phi.ndim != 2 or grad_phi.shape[:2] != phi.shape:
        raise ValueError("phi and grad_phi shapes are incompatible")
    if velocity.shape != (phi.shape[0], grad_phi.shape[-1]):
        raise ValueError("velocity shape is incompatible with feature gradients")
    if weights.shape != (phi.shape[0],):
        raise ValueError("normalized_weights must align with particles")
    if target.shape != (phi.shape[1],) or target_dot.shape != target.shape:
        raise ValueError("raw target moments must align with observables")
    if reaction_kappa <= 0.0:
        raise ValueError("reaction_kappa must be positive")

    advective = jnp.einsum("nmd,nd->nm", grad_phi, velocity)
    base_rate = mass * jnp.einsum("n,nm->m", weights, advective)
    base_rate = base_rate + jnp.asarray(reference_source_rate) * target
    residual = target_dot - base_rate
    transport_gram = mass * jnp.einsum(
        "n,nmd,nkd->mk", weights, grad_phi, grad_phi
    )
    reaction_gram = (mass / float(reaction_kappa)) * jnp.einsum(
        "n,nm,nk->mk", weights, phi, phi
    )
    gram = transport_gram + reaction_gram
    coefficients = jnp.linalg.solve(
        gram + float(ridge) * jnp.eye(phi.shape[1], dtype=phi.dtype), residual
    )
    transport = coefficients @ transport_gram @ coefficients
    reaction = coefficients @ reaction_gram @ coefficients
    return UnbalancedTangentResult(
        total_action=transport + reaction,
        transport_action=transport,
        reaction_action=reaction,
        coefficients=coefficients,
        residual=residual,
        gram=gram,
    )


def append_global_mass_observable(
    phi: Array, grad_phi: Array
) -> tuple[Array, Array]:
    """Append Phi=1, JPhi=0 without consuming a movable sensor."""
    phi = jnp.asarray(phi, dtype=jnp.float64)
    grad_phi = jnp.asarray(grad_phi, dtype=jnp.float64)
    ones = jnp.ones(phi.shape[:-1] + (1,), dtype=phi.dtype)
    zeros = jnp.zeros(grad_phi.shape[:-2] + (1, grad_phi.shape[-1]), dtype=phi.dtype)
    return jnp.concatenate([phi, ones], axis=-1), jnp.concatenate(
        [grad_phi, zeros], axis=-2
    )
