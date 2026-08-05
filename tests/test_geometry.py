import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.geometry import chord_distances, translate, wrap_positions


def test_u1_periodic_wrapping_invariance():
    box = jnp.array([1.0, 1.3])
    x = jnp.array([[[0.1, 0.2], [0.8, 1.1], [0.4, 0.7]]])
    shifted = x + jnp.array([2.0, -2.6])
    np.testing.assert_allclose(chord_distances(x, box), chord_distances(shifted, box), atol=1e-12)
    wrapped = wrap_positions(shifted, box)
    assert bool(jnp.all((wrapped >= 0) & (wrapped < box)))


def test_u3_translation_invariance_of_scalar_geometry():
    box = jnp.array([1.0, 1.0])
    x = jax.random.uniform(jax.random.PRNGKey(0), (3, 5, 2), dtype=jnp.float64)
    shifted = translate(x, jnp.array([0.23, -0.41]), box)
    np.testing.assert_allclose(chord_distances(x, box), chord_distances(shifted, box), atol=1e-12)
