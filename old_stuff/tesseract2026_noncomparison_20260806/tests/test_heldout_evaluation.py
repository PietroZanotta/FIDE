import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.heldout_evaluation import (
    angular_distribution_diagnostics,
    batch_ensemble_angular_moments,
    biased_rbf_mmd_squared,
    median_reference_bandwidth,
)

jax.config.update("jax_enable_x64", True)


def test_identical_angular_distributions_have_zero_error_and_mmd():
    reference = jnp.asarray(
        [
            [-0.2, 0.1],
            [-0.1, 0.2],
            [0.3, -0.2],
            [0.4, -0.1],
        ],
        dtype=jnp.float64,
    )
    labels = jnp.asarray([0, 0, 1, 1])
    scale = jnp.asarray([0.2, 0.15])
    bandwidth = median_reference_bandwidth(reference / scale)
    diagnostics = angular_distribution_diagnostics(
        reference, reference, labels, scale, bandwidth=bandwidth
    )
    assert float(diagnostics["angular_rmse"]) == 0.0
    assert float(diagnostics["angular_whitened_rmse"]) == 0.0
    np.testing.assert_allclose(diagnostics["angular_mmd2"], 0.0, atol=1e-15)
    np.testing.assert_allclose(
        diagnostics["regime_separation_ratio"], 1.0, atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostics["regime_separation_alignment"], 1.0, atol=1e-14
    )


def test_rbf_mmd_is_symmetric_and_detects_shift():
    left = jnp.asarray([[0.0], [0.2], [0.4]], dtype=jnp.float64)
    right = left + 1.0
    forward = biased_rbf_mmd_squared(left, right, 0.5)
    reverse = biased_rbf_mmd_squared(right, left, 0.5)
    np.testing.assert_allclose(forward, reverse, atol=1e-15)
    assert float(forward) > 0.1


def test_batch_angular_moments_preserves_batch_and_feature_axes():
    coordinates = jnp.asarray(
        [
            [
                [[0.1, 0.1], [0.3, 0.1], [0.1, 0.3]],
                [[0.2, 0.2], [0.4, 0.2], [0.2, 0.4]],
            ],
            [
                [[0.6, 0.6], [0.8, 0.6], [0.6, 0.8]],
                [[0.7, 0.7], [0.9, 0.7], [0.7, 0.9]],
            ],
        ],
        dtype=jnp.float64,
    )
    values = batch_ensemble_angular_moments(
        coordinates,
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([1.0, 2.0, 4.0]),
        0.3,
    )
    assert values.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(values)))
