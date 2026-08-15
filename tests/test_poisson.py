import jax
import jax.numpy as jnp

from mfsi.poisson import PoissonConfig, solve_weighted_poisson


def test_weighted_poisson_gradient_through_q_operator_and_rhs():
    cfg = PoissonConfig(dx=0.25, cg_tol=1e-12, cg_maxiter=300)
    q0 = jnp.asarray([
        [1.0, 0.9, 0.8, 0.7],
        [0.95, 0.85, 0.75, 0.65],
        [0.9, 0.8, 0.7, 0.6],
        [0.85, 0.75, 0.65, 0.55],
    ], dtype=jnp.float64)
    h = jnp.asarray([
        [0.3, -0.2, 0.1, -0.1],
        [-0.1, 0.2, -0.2, 0.1],
        [0.15, -0.05, 0.08, -0.12],
        [-0.2, 0.1, -0.05, 0.1],
    ], dtype=jnp.float64)
    # Center h under q so the physical RHS has zero total mass.
    h = h - jnp.sum(q0 * h) / jnp.sum(q0)
    direction = jnp.linspace(-0.15, 0.12, q0.size).reshape(q0.shape)

    def objective(theta):
        q = q0 * jnp.exp(theta * direction)
        # q changes both K(q) and b(q,h).
        return solve_weighted_poisson(q, h, cfg).action

    theta = jnp.asarray(0.0, dtype=jnp.float64)
    grad = jax.grad(objective)(theta)
    eps = 1e-5
    fd = (objective(theta + eps) - objective(theta - eps)) / (2 * eps)
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=2e-4, atol=2e-6)
