"""Numerical validation for Experiment-D learned observables."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import example_b as exb
import observable_design_toy as od

jax.config.update("jax_enable_x64", True)


def _standardization(seed: int = 10, n: int = 4096) -> od.Standardization:
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
    return od.fit_standardization(exb.sample_ring(k0, n), exb.sample_four_lobes(k1, n))


def test_stiefel_constraint_and_shared_capacity():
    for R in (2, 3, 4):
        _, A = od.initialize_stiefel(jax.random.PRNGKey(100 + R), R)
        np.testing.assert_allclose(np.asarray(A @ A.T), np.eye(R), atol=2e-12, rtol=2e-12)


def test_phi_gradient_matches_finite_difference():
    standardization = _standardization()
    _, A = od.initialize_stiefel(jax.random.PRNGKey(11), 3)
    x = jnp.asarray([0.37, -0.61], dtype=jnp.float64)
    analytic = od.observable_jacobian(A, standardization, x)
    autodiff = jax.jacfwd(lambda xx: od.observable_values(A, standardization, xx))(x)
    eps = 1e-6
    fd = jnp.stack([
        (od.observable_values(A, standardization, x + eps * jnp.eye(2)[d])
         - od.observable_values(A, standardization, x - eps * jnp.eye(2)[d])) / (2 * eps)
        for d in range(2)
    ], axis=-1)
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(autodiff), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(np.asarray(analytic), np.asarray(fd), atol=2e-8, rtol=2e-8)


def test_calibration_implicit_unrolled_and_finite_difference_gradients_agree():
    standardization = _standardization(12)
    key = jax.random.PRNGKey(13)
    kx, kb, kd = jax.random.split(key, 3)
    x, _ = exb.sample_bridge(kx, jnp.asarray(0.43), 256)
    B = jax.random.normal(kb, (2, 5), dtype=jnp.float64)
    direction = jax.random.normal(kd, B.shape, dtype=jnp.float64)
    logw = jnp.zeros(x.shape[0], dtype=x.dtype)
    target = jnp.zeros(2, dtype=x.dtype)
    probe = jnp.asarray([0.4, -0.7])

    def scalar_implicit(s):
        A = od.stiefel_rows(B + s * direction)
        ph = od.observable_values(A, standardization, x)
        lam = exb.core.calibrate_empirical_implicit(logw, ph, target)
        return probe @ lam

    def scalar_unrolled(s):
        A = od.stiefel_rows(B + s * direction)
        ph = od.observable_values(A, standardization, x)
        lam = exb.core._calibrate_empirical_primal(logw, ph, target, iterations=40)
        return probe @ lam

    implicit = float(jax.grad(scalar_implicit)(jnp.asarray(0.0)))
    unrolled = float(jax.grad(scalar_unrolled)(jnp.asarray(0.0)))
    eps = 2e-5
    finite = float((scalar_implicit(eps) - scalar_implicit(-eps)) / (2 * eps))
    np.testing.assert_allclose(implicit, unrolled, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(implicit, finite, atol=2e-5, rtol=2e-4)


def test_fiber_objective_directional_gradient_matches_finite_difference():
    standardization = _standardization(14, 2048)
    kr, kb, kd, kbank = jax.random.split(jax.random.PRNGKey(15), 4)
    input_dim = exb.STATE_DIM + 1 + 2 * exb.TIME_FREQ
    reference = exb.core.init_mlp(kr, input_dim, (8,), exb.STATE_DIM)
    B = jax.random.normal(kb, (2, 5), dtype=jnp.float64)
    direction = jax.random.normal(kd, B.shape, dtype=jnp.float64)
    banks = od.make_fiber_banks(kbank, jnp.asarray([0.25, 0.65]), 0.04, 40, reference)

    def loss(s):
        A = od.stiefel_rows(B + s * direction)
        return od.fiber_objective_from_A(A, standardization, banks, 0.04)[0]

    analytic = float(jax.grad(loss)(jnp.asarray(0.0)))
    eps = 2e-4
    finite = float((loss(eps) - loss(-eps)) / (2 * eps))
    np.testing.assert_allclose(analytic, finite, atol=3e-3, rtol=8e-3)


def test_R5_invariance_of_constraint_family_and_projected_weights():
    standardization = _standardization(16)
    kx, k1, k2 = jax.random.split(jax.random.PRNGKey(17), 3)
    x, _ = exb.sample_bridge(kx, jnp.asarray(0.57), 512)
    z = od.standardized_dictionary(x, standardization)
    _, A1 = od.initialize_stiefel(k1, 5)
    _, A2 = od.initialize_stiefel(k2, 5)
    target1 = jnp.zeros(5, dtype=x.dtype)
    # Consistent target-coordinate transformation; zero remains zero.
    transform = A2 @ A1.T
    target2 = transform @ target1
    logw = jnp.zeros(x.shape[0], dtype=x.dtype)
    lam1 = exb.core.calibrate_empirical_implicit(logw, z @ A1.T, target1)
    lam2 = exb.core.calibrate_empirical_implicit(logw, z @ A2.T, target2)
    w1, m1, _ = exb.core.empirical_tilt_from_lambda(lam1, logw, z @ A1.T)
    w2, m2, _ = exb.core.empirical_tilt_from_lambda(lam2, logw, z @ A2.T)
    np.testing.assert_allclose(np.asarray(w1), np.asarray(w2), atol=2e-10, rtol=2e-9)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(target1), atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.asarray(m2), np.asarray(target2), atol=1e-8, rtol=0)


def test_endpoint_equivalence_before_and_after_calibration():
    standardization = _standardization(18, 12000)
    _, A = od.initialize_stiefel(jax.random.PRNGKey(19), 3)
    model = od.ObservableModel(A, standardization)
    result = od.endpoint_equivalence(jax.random.PRNGKey(20), model, 30000)
    # Independent finite banks need not match exactly, but population equality
    # implies a small raw discrepancy and I-projection removes it numerically.
    assert result["max_abs_expectation_gap"] < 0.06
    assert result["calibrated_max_abs_gap"] < 2e-8


def test_design_whitening_is_invertible_and_endpoint_means_are_centered():
    key = jax.random.PRNGKey(21)
    k0, k1 = jax.random.split(key)
    x0, x1 = exb.sample_ring(k0, 30000), exb.sample_four_lobes(k1, 30000)
    standardization = od.fit_standardization(x0, x1)
    assert np.linalg.matrix_rank(np.asarray(standardization.whitening)) == 5
    z0 = od.standardized_dictionary(x0, standardization).mean(axis=0)
    z1 = od.standardized_dictionary(x1, standardization).mean(axis=0)
    np.testing.assert_allclose(np.asarray(0.5 * (z0 + z1)), np.zeros(5), atol=2e-12, rtol=0)

