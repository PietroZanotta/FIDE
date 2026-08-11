"""Validation for the Experiment-E strength/reference controls."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import example_b as exb
import observable_design_toy as od
import strength_matched_observables as sm

jax.config.update("jax_enable_x64", True)


def _standardization() -> od.Standardization:
    k0, k1 = jax.random.split(jax.random.PRNGKey(301))
    return od.fit_standardization(exb.sample_ring(k0, 4096), exb.sample_four_lobes(k1, 4096))


def test_reference_warps_fix_endpoints_and_are_monotone():
    grid = jnp.linspace(0.0, 1.0, 101)
    for geometry in sm.GEOMETRIES:
        values, derivatives = sm.warp_time(grid, geometry)
        np.testing.assert_allclose(float(values[0]), 0.0, atol=1e-14)
        np.testing.assert_allclose(float(values[-1]), 1.0, atol=1e-14)
        assert np.all(np.diff(np.asarray(values)) >= -1e-14)
        assert np.all(np.asarray(derivatives) >= -1e-14)


def test_reference_warp_derivatives_match_finite_difference():
    eps = 1e-6
    for geometry in sm.GEOMETRIES:
        for t in (0.13, 0.37, 0.71, 0.89):
            _, derivative = sm.warp_time(jnp.asarray(t), geometry)
            plus = sm.warp_time(jnp.asarray(t + eps), geometry)[0]
            minus = sm.warp_time(jnp.asarray(t - eps), geometry)[0]
            np.testing.assert_allclose(float(derivative), float((plus - minus) / (2 * eps)),
                                       atol=2e-10, rtol=2e-9)


def test_default_geometry_is_exact_original_reference():
    key = jax.random.PRNGKey(302)
    t = jnp.asarray(0.43)
    x, dx = exb.sample_bridge(key, t, 256)
    other_x, other_dx = sm.sample_bridge_geometry(key, t, 256, "default")
    np.testing.assert_array_equal(np.asarray(x), np.asarray(other_x))
    np.testing.assert_array_equal(np.asarray(dx), np.asarray(other_dx))
    reference = exb.load_model()[0]
    np.testing.assert_array_equal(
        np.asarray(exb.reference_velocity(reference, t, x)),
        np.asarray(sm.reference_velocity_geometry(reference, t, x, "default")),
    )


def test_strength_matrix_matches_direct_definition():
    standardization = _standardization()
    times = jnp.asarray([0.15, 0.5, 0.85])
    matrix, means, _ = sm.strength_matrix(jax.random.PRNGKey(303), times, 2048,
                                          standardization)
    _, A = od.initialize_stiefel(jax.random.PRNGKey(304), 3)
    direct = jnp.mean(jnp.sum((means @ A.T) ** 2, axis=-1))
    np.testing.assert_allclose(float(sm.constraint_strength(A, matrix)), float(direct),
                               atol=2e-13, rtol=2e-13)


def test_random_pool_is_row_orthonormal():
    pool = sm.random_stiefel_pool(jax.random.PRNGKey(305), 32)
    products = jnp.einsum("nri,nsi->nrs", pool, pool)
    np.testing.assert_allclose(np.asarray(products), np.broadcast_to(np.eye(3), (32, 3, 3)),
                               atol=2e-12, rtol=2e-12)
