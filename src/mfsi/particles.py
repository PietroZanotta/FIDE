from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .projection import EmpiricalIProjector, IProjectionState

Array = jax.Array


@dataclass(frozen=True)
class ParticleMFSIConfig:
    """Small linear-system regularization for moment-rate calculations."""

    covariance_ridge: float = 1.0e-7
    tangent_ridge: float = 1.0e-7


class ParticleMFSIState(NamedTuple):
    projection: IProjectionState
    lambda_dot: Array
    forcing: Array
    advective_moments: Array
    tangent_gram: Array
    tangent_residual: Array
    tangent_action: Array


def particle_mfsi_state(
    *,
    phi: Array,
    grad_phi: Array,
    velocity: Array,
    base_weights: Array,
    target: Array,
    target_dot: Array,
    projector: EmpiricalIProjector,
    cfg: ParticleMFSIConfig = ParticleMFSIConfig(),
) -> ParticleMFSIState:
    """Compute the empirical MFSI forcing and tangent action on particles.

    Shapes:
        phi:         [n_particles, n_observables]
        grad_phi:    [n_particles, n_observables, state_dim]
        velocity:    [n_particles, state_dim]
        base_weights:[n_particles]
        target:      [n_observables]
        target_dot:  [n_observables]

    The I-projection derivative is implicit via ``EmpiricalIProjector``. The
    remaining solves are only in observable dimension and are differentiated
    directly.
    """
    phi = jnp.asarray(phi, dtype=jnp.float64)
    grad_phi = jnp.asarray(grad_phi, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    base_weights = jnp.asarray(base_weights, dtype=jnp.float64)
    target = jnp.asarray(target, dtype=jnp.float64)
    target_dot = jnp.asarray(target_dot, dtype=jnp.float64)

    projection = projector.project(phi, base_weights, target)
    w = projection.weights
    lam = projection.lam
    moment = projection.moments
    covariance = projection.covariance

    # m_i = J Phi(x_i) u(x_i)
    advective_moments = jnp.einsum("nmd,nd->nm", grad_phi, velocity)
    mean_advective = jnp.einsum("n,nm->m", w, advective_moments)

    # g_i = lambda^T m_i and Cov(Phi, g).
    g = advective_moments @ lam
    mean_g = w @ g
    centered_phi = phi - moment[None, :]
    cov_phi_g = jnp.einsum("n,nm,n->m", w, centered_phi, g - mean_g)

    m = phi.shape[-1]
    eye = jnp.eye(m, dtype=phi.dtype)
    lambda_dot_rhs = target_dot - mean_advective - cov_phi_g
    lambda_dot = jnp.linalg.solve(
        covariance + cfg.covariance_ridge * eye,
        lambda_dot_rhs,
    )

    forcing = centered_phi @ lambda_dot + g - mean_g
    forcing = forcing - w @ forcing

    tangent_gram = jnp.einsum("nmd,nkd,n->mk", grad_phi, grad_phi, w)
    tangent_residual = mean_advective - target_dot
    tangent_action = tangent_residual @ jnp.linalg.solve(
        tangent_gram + cfg.tangent_ridge * eye,
        tangent_residual,
    )

    return ParticleMFSIState(
        projection=projection,
        lambda_dot=lambda_dot,
        forcing=forcing,
        advective_moments=advective_moments,
        tangent_gram=tangent_gram,
        tangent_residual=tangent_residual,
        tangent_action=tangent_action,
    )
