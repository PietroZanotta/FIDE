import jax
import jax.numpy as jnp

from mfsi.linear import implicit_cg


def test_implicit_cg_gradient_through_operator_and_rhs():
    def objective(theta):
        matrix = jnp.asarray([[2.0 + theta, 0.2], [0.2, 1.5]], dtype=jnp.float64)
        rhs = jnp.asarray([1.0, 0.5 + theta], dtype=jnp.float64)
        x = implicit_cg(lambda v: matrix @ v, rhs, tol=1e-12, maxiter=50)
        return jnp.sum(x**2)

    theta = jnp.asarray(0.3, dtype=jnp.float64)
    grad = jax.grad(objective)(theta)
    eps = 1e-5
    fd = (objective(theta + eps) - objective(theta - eps)) / (2 * eps)
    assert jnp.allclose(grad, fd, rtol=2e-5, atol=2e-6)
