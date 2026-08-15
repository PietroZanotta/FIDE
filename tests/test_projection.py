import jax
import jax.numpy as jnp

from mfsi.projection import EmpiricalIProjector, IProjectionConfig


def test_i_projection_implicit_gradient_matches_finite_difference():
    phi = jnp.asarray(
        [[0.1, 0.7], [0.8, 0.2], [0.3, 0.4], [0.9, 0.9]],
        dtype=jnp.float64,
    )
    base = jnp.ones(4, dtype=jnp.float64) / 4.0
    projector = EmpiricalIProjector(IProjectionConfig(max_steps=40, newton_ridge=1e-10))

    def loss(target):
        state = projector.project(phi, base, target)
        return jnp.sum(state.lam**2)

    target = jnp.asarray([0.52, 0.55], dtype=jnp.float64)
    grad = jax.grad(loss)(target)

    eps = 1e-5
    eye = jnp.eye(2, dtype=jnp.float64)
    fd = jnp.asarray([
        (loss(target + eps * eye[i]) - loss(target - eps * eye[i])) / (2 * eps)
        for i in range(2)
    ])
    assert jnp.allclose(grad, fd, rtol=2e-4, atol=2e-5)
