import jax
import jax.numpy as jnp

from mfsi.measurements import GaussianSensor2D


def test_eta_gradient_is_finite():
    family = GaussianSensor2D()
    x = jnp.asarray([[0.0, 0.0], [1.0, 0.5]], dtype=jnp.float64)
    eta = jnp.asarray([0.31, 1.27], dtype=jnp.float64)
    grad = jax.grad(lambda e: jnp.sum(family.features(x, e)))(eta)
    assert jnp.all(jnp.isfinite(grad))
