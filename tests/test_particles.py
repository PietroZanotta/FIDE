import jax
import jax.numpy as jnp

from mfsi.particles import particle_mfsi_state
from mfsi.projection import EmpiricalIProjector, IProjectionConfig


def test_particle_mfsi_forcing_is_centered_and_differentiable():
    x = jnp.asarray([
        [-1.0, -0.5],
        [-0.2, 0.7],
        [0.4, -0.3],
        [0.9, 0.8],
        [1.2, -0.7],
    ], dtype=jnp.float64)
    base = jnp.ones(x.shape[0], dtype=jnp.float64) / x.shape[0]
    velocity = jnp.asarray([[0.2, -0.1]] * x.shape[0], dtype=jnp.float64)
    projector = EmpiricalIProjector(IProjectionConfig(max_steps=60, newton_ridge=1e-10))

    def objective(eta):
        # Two smooth observables with explicit eta dependence.
        phi = jnp.stack([
            jax.nn.sigmoid(x[:, 0] + eta[0] * x[:, 1]),
            jax.nn.sigmoid(x[:, 1] - eta[1] * x[:, 0]),
        ], axis=-1)
        grad_phi = jax.vmap(jax.jacfwd(lambda xx: jnp.stack([
            jax.nn.sigmoid(xx[0] + eta[0] * xx[1]),
            jax.nn.sigmoid(xx[1] - eta[1] * xx[0]),
        ])))(x)
        target = jnp.asarray([0.52, 0.48])
        target_dot = jnp.asarray([0.02, -0.01])
        state = particle_mfsi_state(
            phi=phi,
            grad_phi=grad_phi,
            velocity=velocity,
            base_weights=base,
            target=target,
            target_dot=target_dot,
            projector=projector,
        )
        centered = state.projection.weights @ state.forcing
        return state.tangent_action + centered**2, (centered, state.tangent_action)

    eta = jnp.asarray([0.3, 0.6])
    (value, (centered, tangent)), grad = jax.value_and_grad(objective, has_aux=True)(eta)
    assert jnp.isfinite(value)
    assert jnp.abs(centered) < 1e-12
    assert tangent >= 0.0
    assert jnp.all(jnp.isfinite(grad))
