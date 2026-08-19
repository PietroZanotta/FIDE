from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from experiments.ocean_drifters.local_poisson import (
    LocalPoissonConfig,
    _row_scaled_conservative_system,
    _score_matrix,
    arithmetic_face_log_conductances,
    gaussian_kde_log_density_and_score,
    solve_log_row_scaled_fv,
    solve_score_form,
    stable_action_diagnostics,
)


def _weighted_center(log_q: np.ndarray, value: np.ndarray) -> np.ndarray:
    weights = np.exp(log_q - np.max(log_q))
    weights /= weights.sum()
    return value - np.sum(weights * value)


def _smooth_problem(nx: int, ny: int) -> tuple[np.ndarray, ...]:
    lx, ly = 2.0, 1.0
    dx = lx / nx
    assert math.isclose(dx, ly / ny)
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    alpha = 0.35
    log_q = alpha * np.cos(2.0 * np.pi * xx / lx) * np.cos(np.pi * yy / ly)
    log_q -= np.log(np.exp(log_q - np.max(log_q)).sum()) + np.max(log_q)
    score_x = -alpha * (2.0 * np.pi / lx) * np.sin(2.0 * np.pi * xx / lx) * np.cos(np.pi * yy / ly)
    score_y = -alpha * (np.pi / ly) * np.cos(2.0 * np.pi * xx / lx) * np.sin(np.pi * yy / ly)
    psi = np.cos(np.pi * xx / lx) * np.cos(2.0 * np.pi * yy / ly)
    psi_x = -(np.pi / lx) * np.sin(np.pi * xx / lx) * np.cos(2.0 * np.pi * yy / ly)
    psi_y = -(2.0 * np.pi / ly) * np.cos(np.pi * xx / lx) * np.sin(2.0 * np.pi * yy / ly)
    laplacian = -((np.pi / lx) ** 2 + (2.0 * np.pi / ly) ** 2) * psi
    h = laplacian + score_x * psi_x + score_y * psi_y
    weights = np.exp(log_q)
    h -= np.sum(weights * h) / np.sum(weights)
    return log_q, h, score_x, score_y, _weighted_center(log_q, psi), dx


def test_log_row_scaling_is_solution_equivalent() -> None:
    log_q, h, _, _, _, dx = _smooth_problem(16, 8)
    scaled, scaled_rhs, diagnostics = _row_scaled_conservative_system(log_q, h, dx)
    assert diagnostics["row_scaling_exactly_representable"]
    assert scaled is not None and scaled_rhs is not None

    log_x, log_y, diagonal = arithmetic_face_log_conductances(log_q, dx)
    ny, nx = log_q.shape
    indices = np.arange(nx * ny).reshape(log_q.shape)
    rows = [np.arange(nx * ny), indices[:, :-1].ravel(), indices[:, 1:].ravel(),
            indices[:-1, :].ravel(), indices[1:, :].ravel()]
    columns = [np.arange(nx * ny), indices[:, 1:].ravel(), indices[:, :-1].ravel(),
               indices[1:, :].ravel(), indices[:-1, :].ravel()]
    values = [np.exp(diagonal).ravel(), -np.exp(log_x).ravel(), -np.exp(log_x).ravel(),
              -np.exp(log_y).ravel(), -np.exp(log_y).ravel()]
    raw = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(nx * ny, nx * ny),
    ).tocsr()
    raw_rhs = -(np.exp(log_q) * h).ravel()
    pin = int(np.argmax(log_q))
    raw_pinned = raw.tolil(copy=True); raw_pinned.rows[pin] = [pin]; raw_pinned.data[pin] = [1.0]
    scaled_pinned = scaled.tolil(copy=True); scaled_pinned.rows[pin] = [pin]; scaled_pinned.data[pin] = [1.0]
    raw_rhs[pin] = 0.0; scaled_rhs[pin] = 0.0
    raw_solution = _weighted_center(log_q, spsolve(raw_pinned.tocsr(), raw_rhs).reshape(log_q.shape))
    scaled_solution = _weighted_center(log_q, spsolve(scaled_pinned.tocsr(), scaled_rhs).reshape(log_q.shape))
    np.testing.assert_allclose(scaled_solution, raw_solution, rtol=2e-12, atol=2e-12)


def test_extreme_absolute_coefficients_are_solved_without_flooring() -> None:
    ny, nx, dx = 7, 11, 0.25
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    # The global range is larger than float64's full exponent span while each
    # local row ratio remains representable.  Direct exp(log_q) therefore has
    # both overflow and underflow, but exact relative row scaling is sufficient.
    log_q = 1000.0 - 300.0 * xx - 20.0 * yy
    expected = np.cos(np.pi * (xx + 0.5) / nx)
    scaled, _, diagnostics = _row_scaled_conservative_system(log_q, np.ones_like(log_q), dx)
    assert scaled is not None
    assert diagnostics["maximum_log_face_conductance"] > math.log(np.finfo(float).max)
    assert diagnostics["log_conductance_range"] > 3000.0

    # Manufacture h from the scaled equation itself, which is invariant to the
    # impossible absolute coefficient scale.
    _, _, diagonal = arithmetic_face_log_conductances(log_q, dx)
    row_exponent = diagonal
    matrix, _, _ = _row_scaled_conservative_system(log_q, np.zeros_like(log_q), dx)
    assert matrix is not None
    h = -(matrix @ expected.ravel()).reshape(log_q.shape) / np.exp(log_q - row_exponent)
    result = solve_log_row_scaled_fv(
        log_q, h, LocalPoissonConfig(dx=dx, relative_tolerance=1e-11)
    )
    assert result["converged"]
    assert result["genuine_scaled_conductance_underflow_count"] == 0
    np.testing.assert_allclose(
        result["potential"], _weighted_center(log_q, expected), rtol=2e-9, atol=2e-9
    )


def test_smooth_density_both_formulations_converge_to_known_solution_and_action() -> None:
    errors = {"fv": [], "score": []}
    identity = {"fv": [], "score": []}
    for nx, ny in ((24, 12), (48, 24), (96, 48)):
        log_q, h, sx, sy, expected, dx = _smooth_problem(nx, ny)
        config = LocalPoissonConfig(dx=dx, relative_tolerance=2e-10)
        results = {
            "fv": solve_log_row_scaled_fv(log_q, h, config),
            "score": solve_score_form(log_q, h, sx, sy, config),
        }
        for name, result in results.items():
            assert result["converged"]
            errors[name].append(
                np.linalg.norm(result["potential"] - expected) / np.linalg.norm(expected)
            )
            identity[name].append(result["action_identity_relative_error"])
    for name in errors:
        assert errors[name][-1] < errors[name][0] / 3.0
    assert max(identity["fv"]) < 1e-9
    assert identity["score"][-1] < identity["score"][0] / 20.0


def test_constant_density_score_operator_is_neumann_poisson() -> None:
    zeros = np.zeros((6, 9))
    score_matrix, laplacian, diagnostics = _score_matrix(zeros, zeros, 0.3)
    np.testing.assert_allclose(score_matrix.toarray(), laplacian.toarray(), atol=0.0)
    assert diagnostics["score_magnitude_maximum"] == 0.0


def test_action_is_gauge_invariant_and_weak_identity_is_reported() -> None:
    log_q, h, _, _, expected, dx = _smooth_problem(32, 16)
    first = stable_action_diagnostics(log_q, h, expected, dx)
    shifted = stable_action_diagnostics(log_q, h, expected + 12345.0, dx)
    np.testing.assert_allclose(shifted["action"], first["action"], rtol=5e-13)
    np.testing.assert_allclose(shifted["weak_action"], first["weak_action"], rtol=2e-10)


def test_gaussian_kde_score_uses_stable_responsibilities() -> None:
    atoms = np.asarray([[-2.0, 0.0], [1.0, 0.5], [4.0, -1.0]])
    bandwidth = np.asarray([[0.7, 0.1], [0.1, 1.2]])
    points = np.asarray([[1000.0, -900.0], [0.2, 0.1]])
    log_density, score = gaussian_kde_log_density_and_score(points, atoms, bandwidth)
    assert np.isfinite(log_density).all() and np.isfinite(score).all()
    step = 1e-5
    for axis in range(2):
        plus = points.copy(); minus = points.copy()
        plus[:, axis] += step; minus[:, axis] -= step
        log_plus, _ = gaussian_kde_log_density_and_score(plus, atoms, bandwidth)
        log_minus, _ = gaussian_kde_log_density_and_score(minus, atoms, bandwidth)
        np.testing.assert_allclose(score[:, axis], (log_plus - log_minus) / (2.0 * step), rtol=2e-6)


def test_tangent_projection_energy_is_a_lower_bound() -> None:
    rng = np.random.default_rng(20260818)
    gradient = rng.normal(size=(200, 2))
    weights = rng.random(200); weights /= weights.sum()
    full = float(np.sum(weights[:, None] * gradient * gradient))
    direction = np.asarray([1.0, -0.4]); direction /= np.linalg.norm(direction)
    tangent = float(np.sum(weights * (gradient @ direction) ** 2))
    assert tangent <= full + 20.0 * np.finfo(float).eps * full
