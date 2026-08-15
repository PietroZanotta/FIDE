import jax.numpy as jnp

from mfsi.projection import EmpiricalIProjector, IProjectionConfig


def test_relative_ess_is_one_at_base_distribution():
    phi = jnp.array([[0.0, 0.0], [1.0, 0.2], [0.2, 1.0], [0.8, 0.7]])
    base = jnp.array([0.6, 0.2, 0.15, 0.05])
    target = base @ phi
    state = EmpiricalIProjector(IProjectionConfig(max_steps=30)).project(phi, base, target)
    assert jnp.allclose(state.ess_fraction, 1.0, atol=1e-7)
