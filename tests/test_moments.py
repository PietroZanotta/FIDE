import jax
import jax.numpy as jnp

from mfsi.moments import (
    QuadraticBridgeConfig,
    bridge_halfspace_constraints,
    fit_quadratic_bridge_gls,
    max_constraint_violation,
)


def test_quadratic_bridge_recovers_exact_beta():
    t_obs = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    c0 = jnp.asarray([0.2, 0.4])
    c1 = jnp.asarray([0.8, 0.1])
    beta_true = jnp.asarray([0.35, -0.2])
    z = t_obs * (1.0 - t_obs)
    y = (1.0 - t_obs[:, None]) * c0 + t_obs[:, None] * c1 + z[:, None] * beta_true
    covariance = jnp.broadcast_to(jnp.eye(2)[None, :, :] * 0.03, (len(t_obs), 2, 2))

    out = fit_quadratic_bridge_gls(
        t_obs,
        y,
        covariance,
        c0,
        c1,
        t_obs,
        QuadraticBridgeConfig(ridge_rel=0.0, variance_floor=1e-12),
    )
    assert jnp.allclose(out.beta, beta_true, rtol=1e-9, atol=1e-9)
    assert jnp.allclose(out.c[0], c0)
    assert jnp.allclose(out.c[-1], c1)


def test_quadratic_bridge_gradient_matches_finite_difference():
    t_obs = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float64)
    c0 = jnp.asarray([0.1, 0.2], dtype=jnp.float64)
    c1 = jnp.asarray([0.7, 0.5], dtype=jnp.float64)
    covariance = jnp.broadcast_to(jnp.eye(2)[None, :, :] * 0.04, (3, 2, 2))
    y0 = jnp.asarray([[0.22, 0.29], [0.49, 0.31], [0.62, 0.44]], dtype=jnp.float64)

    def loss(y):
        out = fit_quadratic_bridge_gls(t_obs, y, covariance, c0, c1, t_obs)
        return jnp.sum(out.beta**2)

    grad = jax.grad(loss)(y0)
    eps = 1e-5
    direction = jnp.asarray([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]])
    fd = (loss(y0 + eps * direction) - loss(y0 - eps * direction)) / (2 * eps)
    ad = jnp.sum(grad * direction)
    assert jnp.allclose(ad, fd, rtol=2e-5, atol=2e-7)


def test_bridge_halfspace_constraints_match_direct_check():
    # Square moment hull: 0 <= c_i <= 1.
    equations = jnp.asarray([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ])
    c0 = jnp.asarray([0.2, 0.3])
    c1 = jnp.asarray([0.8, 0.7])
    times = jnp.asarray([0.0, 0.5, 1.0])
    A, b = bridge_halfspace_constraints(c0, c1, times, equations)
    assert float(max_constraint_violation(jnp.zeros(2), A, b)) == 0.0
