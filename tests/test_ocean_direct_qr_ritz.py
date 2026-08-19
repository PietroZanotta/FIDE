from __future__ import annotations

import numpy as np

from experiments.ocean_drifters.direct_qr_ritz import (
    solve_raw_direct_ritz,
    solve_whitened_direct_ritz,
)


def _matrix_with_singular_values(
    singular_values: np.ndarray,
    *,
    rows: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = len(singular_values)
    left, _ = np.linalg.qr(rng.normal(size=(rows, dimension)))
    right, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    return left @ np.diag(singular_values) @ right.T, right


def test_direct_qr_matches_normal_equation_when_well_conditioned():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(80, 7))
    load = rng.normal(size=7)
    direct = solve_whitened_direct_ritz(matrix, load)
    expected = float(load @ np.linalg.solve(matrix.T @ matrix, load))
    np.testing.assert_allclose(direct.action_qr, expected, rtol=2e-14)
    np.testing.assert_allclose(direct.action_svd, expected, rtol=2e-14)


def test_ill_conditioned_direct_qr_and_svd_agree_when_normal_equation_does_not():
    matrix, right = _matrix_with_singular_values(np.geomspace(1.0, 1.0e-11, 8))
    load = right @ np.array([0.0] * 7 + [1.0e-11])
    direct = solve_whitened_direct_ritz(matrix, load)
    normal = float(load @ np.linalg.solve(matrix.T @ matrix, load))
    assert 1.0e10 <= direct.kappa_c <= 1.0e12
    assert direct.qr_svd_relative_discrepancy < 1.0e-12
    assert abs(normal - direct.action_svd) / direct.action_svd > 0.1


def test_direct_action_is_invariant_under_trial_coordinate_changes():
    rng = np.random.default_rng(22)
    matrix = rng.normal(size=(120, 6))
    load = rng.normal(size=6)
    physical_gram = rng.normal(size=(6, 6))
    physical_gram = physical_gram.T @ physical_gram + np.eye(6)
    baseline = solve_raw_direct_ritz(
        matrix, load, physical_gram, structural_basis=np.eye(6)
    )

    orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    nonorthogonal = np.eye(6) + 0.1 * np.triu(np.ones((6, 6)), 1)
    maps = [
        np.eye(6)[:, [4, 0, 5, 2, 1, 3]],
        np.diag(np.geomspace(1.0e-4, 1.0e4, 6)),
        orthogonal,
        nonorthogonal,
    ]
    for coordinate_map in maps:
        transformed = solve_raw_direct_ritz(
            matrix @ coordinate_map,
            coordinate_map.T @ load,
            coordinate_map.T @ physical_gram @ coordinate_map,
            structural_basis=np.eye(6),
        )
        np.testing.assert_allclose(
            transformed.direct.action_qr,
            baseline.direct.action_qr,
            rtol=3.0e-8,
        )


def test_enriched_space_action_bounds_sensor_subspace_action():
    rng = np.random.default_rng(33)
    matrix = rng.normal(size=(150, 8))
    load = rng.normal(size=8)
    tangent = solve_whitened_direct_ritz(matrix[:, :3], load[:3])
    enriched = solve_whitened_direct_ritz(matrix, load)
    assert tangent.action_qr <= enriched.action_qr * (1.0 + 1.0e-14)


def test_small_positive_singular_direction_is_not_truncated():
    matrix = np.diag([1.0, 0.2, 1.0e-11])
    load = np.array([0.0, 0.0, 1.0e-11])
    direct = solve_whitened_direct_ritz(matrix, load)
    assert direct.lapack_full_column_rank
    np.testing.assert_allclose(direct.action_qr, 1.0, rtol=1.0e-15)
    np.testing.assert_allclose(direct.action_svd, 1.0, rtol=1.0e-15)
    assert np.max(direct.action_contributions) == 1.0


def test_direct_actions_are_monotone_for_exactly_nested_spaces():
    rng = np.random.default_rng(44)
    matrix = rng.normal(size=(120, 7))
    load = rng.normal(size=7)
    actions = [
        solve_whitened_direct_ritz(matrix[:, :size], load[:size]).action_qr
        for size in range(2, 8)
    ]
    assert np.all(np.diff(actions) >= -1.0e-13 * np.max(actions))


def test_direct_solver_survives_nonpositive_float64_normal_eigenvalue():
    matrix, right = _matrix_with_singular_values(np.geomspace(1.0, 1.0e-10, 8))
    load = right @ np.array([0.0] * 7 + [1.0e-10])
    direct = solve_whitened_direct_ritz(matrix, load)
    normal = matrix.T @ matrix
    assert direct.sigma_min > 0.0
    assert np.linalg.eigvalsh(0.5 * (normal + normal.T))[0] <= 0.0
    assert direct.qr_success and direct.svd_success
    assert direct.qr_svd_relative_discrepancy < 1.0e-12
    np.testing.assert_allclose(direct.action_qr, 1.0, rtol=5.0e-7)
