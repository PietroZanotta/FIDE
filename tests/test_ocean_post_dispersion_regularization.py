from __future__ import annotations

import numpy as np

from experiments.ocean_drifters.direct_qr_ritz import prepare_direct_ritz_basis
from experiments.ocean_drifters.full_action_repair import TrialBasis
from experiments.ocean_drifters.post_dispersion_regularization import (
    normalized_trapezoid_weights,
    post_dispersion_source_indices,
    solve_conductivity_regularized_ritz,
)
from experiments.ocean_drifters.temporal_refinement import (
    nested_source_grids,
    summarize_temporal_levels,
)


def _prepared_problem() -> tuple[object, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260819)
    count, size = 80, 5
    basis = TrialBasis(
        values=rng.normal(size=(count, size)),
        gradient_x=rng.normal(size=(count, size)),
        gradient_y=rng.normal(size=(count, size)),
        names=tuple(f"test_{index}" for index in range(size)),
    )
    physical_gram = (
        basis.gradient_x.T @ basis.gradient_x
        + basis.gradient_y.T @ basis.gradient_y
        + np.eye(size)
    ) / count
    prepared = prepare_direct_ritz_basis(basis, physical_gram)
    weights = rng.random(count)
    weights /= weights.sum()
    forcing = rng.normal(size=count)
    forcing -= weights @ forcing
    return prepared, weights, forcing


def test_post_dispersion_window_includes_predeclared_boundary() -> None:
    normalized_times = np.arange(181, dtype=np.float64) / 180.0
    indices = post_dispersion_source_indices(
        normalized_times,
        start_day_inclusive=12.0,
        end_day_inclusive=45.0,
    )
    assert indices[0] == 48
    assert indices[-1] == 180
    assert len(indices) == 133


def test_window_trapezoid_weights_match_vortices_convention() -> None:
    times = np.asarray([0.0, 0.25, 1.0])
    weights = normalized_trapezoid_weights(times)
    np.testing.assert_allclose(weights, [0.125, 0.5, 0.375])
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_temporal_refinement_ladder_is_nested_and_certifies_stable_rows() -> None:
    cfg = {
        "window_start_source_index": 48,
        "window_end_source_index": 180,
        "source_horizon_days": 45.0,
        "source_steps": [12, 6, 3, 1],
        "node_counts": [12, 23, 45, 133],
        "maximum_consecutive_relative_action_change": 0.05,
        "tangent_full_relative_tolerance": 1e-8,
        "production_authorized_if_certified": True,
    }
    grids = nested_source_grids(48, 180, cfg["source_steps"])
    assert [len(grid) for grid in grids] == cfg["node_counts"]
    rows = [
        {
            "source_time_index": int(source),
            "tangent_action": 2.0,
            "direct_action_qr": 3.0,
            "local_valid": True,
        }
        for source in grids[-1]
    ]
    summary = summarize_temporal_levels(rows, cfg)
    assert summary["temporal_quadrature_refinement_certified"] is True
    assert summary["production_authorized"] is True
    assert summary["maximum_consecutive_relative_action_change_observed"] < 1e-14


def test_regularized_ritz_matches_explicit_dense_equation() -> None:
    prepared, weights, forcing = _prepared_problem()
    epsilon = 2.0e-6
    result = solve_conductivity_regularized_ritz(
        prepared, weights, forcing, epsilon
    )
    assert result.success

    mean = weights @ prepared.values
    load = prepared.raw_to_whitened.T @ (
        -(prepared.values - mean).T @ (weights * forcing)
    )
    gx = prepared.gradient_x_whitened
    gy = prepared.gradient_y_whitened
    physical = gx.T @ (weights[:, None] * gx) + gy.T @ (
        weights[:, None] * gy
    )
    regularizer = epsilon * np.max(weights) * (gx.T @ gx + gy.T @ gy)
    expected = np.linalg.solve(physical + regularizer, load)

    np.testing.assert_allclose(result.coefficients, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result.physical_action,
        expected @ physical @ expected,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.operator_action,
        result.physical_action + result.regularization_action,
        rtol=2e-14,
        atol=2e-14,
    )


def test_regularization_is_not_applied_to_rhs_or_reported_action() -> None:
    prepared, weights, forcing = _prepared_problem()
    small = solve_conductivity_regularized_ritz(
        prepared, weights, forcing, 2.0e-7
    )
    large = solve_conductivity_regularized_ritz(
        prepared, weights, forcing, 2.0e-5
    )
    assert small.success and large.success
    assert small.regularization_action > 0.0
    assert large.regularization_action > 0.0
    assert small.operator_action > small.physical_action
    assert large.operator_action > large.physical_action
    # Added conductivity reduces the dual/operator energy for a fixed qh load.
    assert large.operator_action < small.operator_action
