from __future__ import annotations

import jax
import jax.numpy as jnp

from .reference import Params, velocity_mlp

Array = jax.Array


def latent_velocity_and_divergence(params: Params, t: Array, z: Array) -> tuple[Array, Array]:
    """Evaluate latent velocity and its exact divergence in two dimensions."""
    z = jnp.asarray(z, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    velocity = velocity_mlp(params, t, z)

    def single_divergence(point: Array) -> Array:
        jacobian = jax.jacfwd(lambda value: velocity_mlp(params, t, value))(point)
        return jnp.trace(jacobian)

    flat = z.reshape((-1, 2))
    divergence = jax.vmap(single_divergence)(flat).reshape(z.shape[:-1])
    return velocity, divergence


def backward_latent_with_log_density_correction(
    params: Params,
    z_t: Array,
    time: Array,
    *,
    steps: int,
) -> tuple[Array, Array]:
    """Integrate latent states from ``time`` to zero with CNF correction.

    Returns ``(z0, correction)`` where

    ``log q_t(z_t) = log q_0(z0) + correction``

    and ``correction = -integral_0^t div f(s,z_s) ds`` along the trajectory.
    """
    if steps < 1:
        raise ValueError("steps must be at least one")
    z_t = jnp.asarray(z_t, dtype=jnp.float64)
    time = jnp.asarray(time, dtype=jnp.float64)
    dt = -time / float(steps)
    correction0 = jnp.zeros(z_t.shape[:-1], dtype=jnp.float64)

    def derivative(t: Array, z: Array) -> tuple[Array, Array]:
        return latent_velocity_and_divergence(params, t, z)

    def step(index: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        z, correction = state
        t = time + index.astype(jnp.float64) * dt
        k1z, k1c = derivative(t, z)
        k2z, k2c = derivative(t + 0.5 * dt, z + 0.5 * dt * k1z)
        k3z, k3c = derivative(t + 0.5 * dt, z + 0.5 * dt * k2z)
        k4z, k4c = derivative(t + dt, z + dt * k3z)
        z_next = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        correction_next = correction + (dt / 6.0) * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        return z_next, correction_next

    return jax.lax.fori_loop(0, steps, step, (z_t, correction0))


def logistic_log_abs_det_jacobian(z: Array, bounds: Array) -> Array:
    """Log absolute Jacobian determinant for rectangle logistic coordinates."""
    z = jnp.asarray(z, dtype=jnp.float64)
    bounds = jnp.asarray(bounds, dtype=jnp.float64)
    width = jnp.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2]])
    # log(sigmoid(z)) + log(1-sigmoid(z)), evaluated stably.
    log_diagonal = jnp.log(width) - jax.nn.softplus(-z) - jax.nn.softplus(z)
    return jnp.sum(log_diagonal, axis=-1)
