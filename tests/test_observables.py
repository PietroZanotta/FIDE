import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.observables import PairBasis, ensemble_pair_moments


def test_u2_pair_moments_permutation_invariance():
    key = jax.random.PRNGKey(1)
    box = jnp.array([1.0, 1.0])
    x = jax.random.uniform(key, (4, 7, 2), dtype=jnp.float64)
    permutation = jnp.array([4, 0, 6, 2, 1, 5, 3])
    basis = PairBasis.uniform(8, 0.03, 0.68, 0.08)
    expected = ensemble_pair_moments(x, box, basis)
    actual = ensemble_pair_moments(x[:, permutation], box, basis)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_pair_moments_are_ensemble_level_average():
    box = jnp.array([1.0, 1.0])
    basis = PairBasis.uniform(4, 0.05, 0.5, 0.1)
    x = jnp.array(
        [
            [[0.1, 0.1], [0.2, 0.1]],
            [[0.1, 0.1], [0.5, 0.1]],
        ],
        dtype=jnp.float64,
    )
    both = ensemble_pair_moments(x, box, basis)
    separately = 0.5 * (
        ensemble_pair_moments(x[:1], box, basis) + ensemble_pair_moments(x[1:], box, basis)
    )
    np.testing.assert_allclose(both, separately, atol=1e-12)
