from __future__ import annotations

from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from mfsi.poisson import PoissonConfig, solve_weighted_poisson
from mfsi.poisson_variational_tesseract import (
    VariationalPoissonConfig,
    is_tesseract_variational_poisson_available,
    solve_variational_poisson_batch_native,
    solve_variational_poisson_batch_tesseract,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = ROOT / "native" / "variational_poisson_tesseract" / "build"
if str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))

native = pytest.importorskip(
    "_variational_poisson_native",
    reason="build native/variational_poisson_tesseract before running native tests",
)


def _cosine_manufactured_system():
    ny, nx, dx = 15, 21, 0.2
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    length_x = nx * dx
    length_y = ny * dx
    potential = (
        np.cos(2.0 * np.pi * xx / length_x)
        * np.cos(np.pi * yy / length_y)
    )
    eigenvalue = (2.0 * np.pi / length_x) ** 2 + (np.pi / length_y) ** 2
    forcing = -eigenvalue * potential
    log_q = np.full_like(potential, -np.log(nx * ny))
    return log_q[None], forcing[None], potential, eigenvalue, dx


def test_variational_native_recovers_neumann_cosine_manufactured_solution():
    log_q, forcing, expected, eigenvalue, dx = _cosine_manufactured_system()
    cfg = VariationalPoissonConfig(dx=dx, maximum_mode=3)
    result = solve_variational_poisson_batch_native(log_q, forcing, cfg)

    assert bool(result["converged"][0])
    assert result["retained_rank"][0] == result["basis_size"][0] == 15
    np.testing.assert_allclose(result["potential"][0], expected, rtol=1e-12, atol=1e-12)
    expected_action = eigenvalue * np.mean(expected * expected)
    np.testing.assert_allclose(result["action"][0], expected_action, rtol=1e-12)
    np.testing.assert_allclose(result["objective"][0], -0.5 * expected_action, rtol=1e-12)
    assert result["scaled_weak_relative_residual"][0] < 1e-12
    assert result["retained_scaled_weak_relative_residual"][0] < 1e-12
    assert result["discarded_scaled_load_relative_residual"][0] < 1e-12
    assert result["weak_relative_residual"][0] < 1e-12
    assert abs(result["gauge_residual"][0]) < 1e-14
    assert result["gauge_relative_residual"][0] < 1e-14
    assert abs(result["compatibility_relative_residual"][0]) < 1e-14
    assert result["energy_load_identity_relative_error"][0] < 1e-12

    # Along the exact one-mode trial direction, the reported solution is the
    # strict minimizer of 0.5 E_q|grad psi|^2 + E_q[h psi].
    for amplitude in (0.6, 0.9, 1.1, 1.4):
        perturbed_objective = expected_action * (0.5 * amplitude**2 - amplitude)
        assert perturbed_objective > result["objective"][0]


def test_variational_native_is_log_normalization_shift_invariant():
    ny, nx, dx = 13, 17, 0.2
    yy, xx = np.mgrid[:ny, :nx]
    log_q = -0.08 * ((xx - 7.2) ** 2 + 1.4 * (yy - 5.4) ** 2)
    weights = np.exp(log_q - np.max(log_q))
    weights /= np.sum(weights)
    forcing = (
        np.cos(np.pi * (xx + 0.5) / nx)
        + 0.3 * np.cos(2.0 * np.pi * (yy + 0.5) / ny)
    )
    forcing -= np.sum(weights * forcing)
    result = solve_variational_poisson_batch_native(
        np.stack((log_q, log_q + 100_000.0)),
        np.stack((forcing, forcing)),
        VariationalPoissonConfig(dx=dx, maximum_mode=4),
    )

    assert np.all(result["converged"] == 1.0)
    assert np.all(result["retained_rank"] == result["basis_size"])
    assert np.max(result["scaled_weak_relative_residual"]) < 1e-10
    assert np.max(result["retained_scaled_weak_relative_residual"]) < 1e-10
    assert np.max(np.abs(result["gauge_residual"])) < 1e-13
    np.testing.assert_allclose(result["action"][0], result["action"][1], rtol=1e-11)
    np.testing.assert_allclose(
        result["potential"][0], result["potential"][1], rtol=1e-10, atol=2e-12
    )


def test_variational_native_accepts_log_weights_beyond_float64_density_range():
    ny, nx, dx = 13, 17, 0.2
    yy, xx = np.mgrid[:ny, :nx]
    log_q = -0.04 * ((xx - 8.0) ** 2 + (yy - 6.0) ** 2)
    log_q[0, 0] = -2_000.0
    log_q[-1, -1] = -3_000.0
    shifted = np.exp(log_q - np.max(log_q))
    weights = shifted / np.sum(shifted)
    forcing = np.cos(np.pi * (xx + 0.5) / nx)
    forcing -= np.sum(weights * forcing)

    result = solve_variational_poisson_batch_native(
        log_q[None],
        forcing[None],
        VariationalPoissonConfig(dx=dx, maximum_mode=4),
    )

    assert np.max(log_q) - np.min(log_q) >= 3_000.0
    assert bool(result["converged"][0])
    assert np.isfinite(result["potential"]).all()
    assert np.isfinite(result["action"][0])
    assert result["scaled_weak_relative_residual"][0] < 1e-10
    assert abs(result["gauge_residual"][0]) < 1e-13


def test_variational_tesseract_endpoint_matches_direct_native_call():
    pytest.importorskip("tesseract_jax")
    assert is_tesseract_variational_poisson_available()
    log_q, forcing, _, _, dx = _cosine_manufactured_system()
    log_q = np.concatenate((log_q, log_q - 23.0), axis=0)
    forcing = np.concatenate((forcing, 0.7 * forcing), axis=0)
    cfg = VariationalPoissonConfig(dx=dx, maximum_mode=3)

    direct = solve_variational_poisson_batch_native(log_q, forcing, cfg)
    endpoint = solve_variational_poisson_batch_tesseract(log_q, forcing, cfg)

    assert set(endpoint) == set(direct)
    for name in direct:
        np.testing.assert_allclose(np.asarray(endpoint[name]), direct[name], rtol=0.0, atol=0.0)


def test_variational_and_existing_strong_solver_converge_on_same_smooth_problem():
    ny, nx, dx = 35, 41, 0.1
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    expected = (
        np.cos(2.0 * np.pi * xx / (nx * dx))
        * np.cos(np.pi * yy / (ny * dx))
    )
    eigenvalue = (2.0 * np.pi / (nx * dx)) ** 2 + (np.pi / (ny * dx)) ** 2
    forcing = -eigenvalue * expected
    q_density = np.full_like(expected, 1.0 / (nx * ny * dx * dx))

    strong = solve_weighted_poisson(
        q_density,
        forcing,
        PoissonConfig(
            dx=dx,
            operator_floor_rel=0.0,
            cg_tol=1.0e-11,
            cg_maxiter=1_000,
        ),
    )
    weak = solve_variational_poisson_batch_native(
        np.full((1, ny, nx), -np.log(nx * ny)),
        forcing[None],
        VariationalPoissonConfig(dx=dx, maximum_mode=3),
    )

    assert float(strong.relative_residual) < 1e-10
    assert bool(weak["converged"][0])
    # The strong solver uses a second-order finite-volume eigenvalue while the
    # Ritz solver differentiates its cosine basis analytically. Their O(dx^2)
    # discrepancy must already be small on this grid.
    np.testing.assert_allclose(
        np.asarray(strong.potential), weak["potential"][0], rtol=2e-3, atol=2e-3
    )
    np.testing.assert_allclose(float(strong.action), weak["action"][0], rtol=2e-3)


def test_variational_wrapper_rejects_invalid_inputs_before_native_boundary():
    cfg = VariationalPoissonConfig(dx=0.2)
    with pytest.raises(ValueError, match="same .* shape"):
        solve_variational_poisson_batch_native(
            np.zeros((1, 5, 5)), np.zeros((1, 5, 4)), cfg
        )
    invalid = np.zeros((1, 5, 5))
    invalid[0, 2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_variational_poisson_batch_native(invalid, np.zeros_like(invalid), cfg)
