import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.energies import EnergyParameters, angular_energy_per_configuration
from manybody_completion.geometry import translate
from manybody_completion.observables import angular_cosine_moments

jax.config.update("jax_enable_x64", True)


def test_angular_features_are_translation_and_permutation_invariant():
    coordinates = jnp.asarray(
        [
            [
                [0.93, 0.08],
                [0.04, 0.11],
                [0.89, 0.95],
                [0.14, 0.88],
            ]
        ],
        dtype=jnp.float64,
    )
    box = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    shifted = translate(coordinates, jnp.asarray([0.23, -0.31]), box)
    permutation = jnp.asarray([2, 0, 3, 1])
    permuted = coordinates[:, permutation, :]
    orders = jnp.asarray([1.0, 2.0, 4.0, 6.0])

    reference = angular_cosine_moments(coordinates, box, orders, 0.24)
    shifted_value = angular_cosine_moments(shifted, box, orders, 0.24)
    permuted_value = angular_cosine_moments(permuted, box, orders, 0.24)
    np.testing.assert_allclose(reference, shifted_value, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(reference, permuted_value, atol=2e-12, rtol=2e-12)

    reference_energy = angular_energy_per_configuration(coordinates, box, 0.0, 0.24)
    shifted_energy = angular_energy_per_configuration(shifted, box, 0.0, 0.24)
    permuted_energy = angular_energy_per_configuration(permuted, box, 0.0, 0.24)
    np.testing.assert_allclose(reference_energy, shifted_energy, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(reference_energy, permuted_energy, atol=2e-12, rtol=2e-12)
