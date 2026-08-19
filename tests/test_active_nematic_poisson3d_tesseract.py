from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

NATIVE_BUILD = (
    Path(__file__).parents[1]
    / "native"
    / "active_nematic_poisson3d_tesseract"
    / "build"
)
if str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.active_nematic.periodic_numerics import (
    PeriodicGrid3D,
    PeriodicPoissonConfig,
    periodic_weighted_laplacian3d,
    periodic_weighted_laplacian_diag3d,
    solve_periodic_weighted_poisson3d_batch_jax,
)
from experiments.active_nematic.poisson3d_tesseract import (
    is_active_nematic_poisson3d_available,
    solve_periodic_weighted_poisson3d_batch_tesseract,
)

pytestmark = pytest.mark.skipif(
    not is_active_nematic_poisson3d_available(),
    reason="optional active-nematic 3D native Tesseract is not built",
)


def _native():
    return pytest.importorskip("_active_nematic_poisson3d_native")


def _problem(seed: int = 8):
    rng = np.random.default_rng(seed)
    shape = (2, 7, 6, 5)
    grid = PeriodicGrid3D(4.0, shape[1:], polarity_metric_radius=0.7)
    q = 0.4 + rng.random(shape)
    potential = rng.normal(size=shape)
    gauge = q.reshape((shape[0], -1))
    gauge /= np.linalg.norm(gauge, axis=-1, keepdims=True)
    gauge = np.ascontiguousarray(gauge.reshape(shape))
    return grid, np.ascontiguousarray(q), np.ascontiguousarray(potential), gauge


def test_native_stencil_and_diagonal_match_jax_reference() -> None:
    native = _native()
    grid, q, potential, gauge = _problem()
    actual = native.weighted_laplacian_batch(
        potential, q, *grid.spacings
    )
    expected = periodic_weighted_laplacian3d(
        jnp.asarray(potential), jnp.asarray(q), grid.spacings
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-12)

    gauge_strength = 0.8
    actual_diagonal = native.diagonal_batch(
        q, gauge, *grid.spacings, gauge_strength
    )
    expected_diagonal = periodic_weighted_laplacian_diag3d(
        jnp.asarray(q), grid.spacings
    ) + gauge_strength * jnp.asarray(gauge) ** 2
    np.testing.assert_allclose(
        actual_diagonal, expected_diagonal, rtol=2.0e-14, atol=2.0e-12
    )


def test_native_manufactured_solution_and_true_residual() -> None:
    native = _native()
    grid, q, expected, gauge = _problem(9)
    gauge_strength = 0.6
    physical = np.asarray(
        periodic_weighted_laplacian3d(expected, q, grid.spacings)
    )
    gauge_dot = np.sum(gauge * expected, axis=(-3, -2, -1), keepdims=True)
    rhs = np.ascontiguousarray(
        physical + gauge_strength * gauge * gauge_dot
    )
    result = native.solve_batch(
        q,
        rhs,
        gauge,
        *grid.spacings,
        gauge_strength,
        1.0e-11,
        800,
        None,
    )
    assert np.all(result["converged"])
    assert np.max(result["relative_residual"]) < 1.1e-11
    np.testing.assert_allclose(
        result["potential"], expected, rtol=3.0e-10, atol=3.0e-10
    )


def test_native_implicit_vjp_matches_centered_finite_difference() -> None:
    native = _native()
    grid, q, expected, gauge = _problem(10)
    rng = np.random.default_rng(11)
    gauge_strength = 0.9
    rhs = np.ascontiguousarray(
        periodic_weighted_laplacian3d(expected, q, grid.spacings)
        + gauge_strength
        * gauge
        * np.sum(gauge * expected, axis=(-3, -2, -1), keepdims=True)
    )
    cotangent = np.ascontiguousarray(rng.normal(size=q.shape))
    tangent = np.ascontiguousarray(rng.normal(size=q.shape))
    potential = native.solve_batch(
        q, rhs, gauge, *grid.spacings, gauge_strength, 1.0e-11, 800, None
    )["potential"]
    adjoint = native.solve_batch(
        q, cotangent, gauge, *grid.spacings, gauge_strength, 1.0e-11, 800, None
    )["potential"]
    q_bar = native.weighted_operator_vjp(
        np.ascontiguousarray(potential),
        np.ascontiguousarray(adjoint),
        *grid.spacings,
    )
    epsilon = 1.0e-6
    objectives = []
    for sign in (-1.0, 1.0):
        solution = native.solve_batch(
            np.ascontiguousarray(q + sign * epsilon * tangent),
            rhs,
            gauge,
            *grid.spacings,
            gauge_strength,
            1.0e-11,
            800,
            None,
        )["potential"]
        objectives.append(np.sum(cotangent * solution))
    finite_difference = (objectives[1] - objectives[0]) / (2.0 * epsilon)
    analytic = np.sum(q_bar * tangent)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-7, atol=2.0e-7)


def test_tesseract_action_and_gradient_match_jax_reference() -> None:
    grid, q_numpy, _, _ = _problem(12)
    q = jnp.asarray(q_numpy)
    raw_h = jax.random.normal(jax.random.PRNGKey(12), q.shape, dtype=jnp.float64)
    h = raw_h - jnp.sum(q * raw_h, axis=(-3, -2, -1), keepdims=True) / jnp.sum(
        q, axis=(-3, -2, -1), keepdims=True
    )
    cfg = PeriodicPoissonConfig(
        operator_floor_rel=2.0e-4,
        cg_tol=1.0e-10,
        cg_maxiter=800,
        gauge_strength=1.0,
    )

    def native_objective(scale):
        return jnp.sum(
            solve_periodic_weighted_poisson3d_batch_tesseract(
                scale * q, h, grid, cfg
            ).action
        )

    def jax_objective(scale):
        return jnp.sum(
            solve_periodic_weighted_poisson3d_batch_jax(
                scale * q, h, grid, cfg
            ).action
        )

    native_value, native_gradient = jax.value_and_grad(native_objective)(1.0)
    jax_value, jax_gradient = jax.value_and_grad(jax_objective)(1.0)
    np.testing.assert_allclose(native_value, jax_value, rtol=2.0e-8, atol=2.0e-9)
    np.testing.assert_allclose(
        native_gradient, jax_gradient, rtol=3.0e-7, atol=3.0e-8
    )
    (_, native_tangent), (_, jax_tangent) = (
        jax.jvp(objective, (1.0,), (0.07,))
        for objective in (native_objective, jax_objective)
    )
    np.testing.assert_allclose(
        native_tangent, jax_tangent, rtol=3.0e-7, atol=3.0e-8
    )
    compiled = jax.jit(native_objective)(1.0)
    np.testing.assert_allclose(compiled, native_value, rtol=2.0e-10, atol=2.0e-10)


def test_zero_forcing_diagnostics_have_finite_zero_gradient() -> None:
    grid, q_numpy, _, _ = _problem(13)
    q = jnp.asarray(q_numpy)
    h = jnp.zeros_like(q)
    cfg = PeriodicPoissonConfig(
        operator_floor_rel=2.0e-4,
        cg_tol=1.0e-10,
        cg_maxiter=800,
        gauge_strength=1.0,
    )

    for solver in (
        solve_periodic_weighted_poisson3d_batch_tesseract,
        solve_periodic_weighted_poisson3d_batch_jax,
    ):
        def objective(scale):
            result = solver(scale * q, h, grid, cfg)
            return jnp.sum(result.action) + jnp.sum(result.relative_residual)

        value, gradient = jax.value_and_grad(objective)(1.0)
        assert np.isfinite(value)
        assert np.isfinite(gradient)
        np.testing.assert_allclose(value, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(gradient, 0.0, atol=1.0e-14)
