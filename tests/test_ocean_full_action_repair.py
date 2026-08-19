import numpy as np

from experiments.ocean_drifters.full_action_repair import (
    assemble_variational_system,
    gaussian_sensor_basis,
    generalized_cutoff_actions,
    old_equilibrated_cutoff_actions,
    solve_full_rank_ritz,
    transformed_system,
)


def _spd_problem(size: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(47)
    factor = rng.normal(size=(size, size))
    stiffness = factor.T @ factor + np.eye(size)
    forcing = rng.normal(size=size)
    physical_gram = np.diag(np.geomspace(0.7, 2.1, size))
    return stiffness, forcing, physical_gram


def test_repaired_action_is_invariant_to_basis_rescaling() -> None:
    stiffness, forcing, physical_gram = _spd_problem()
    baseline = solve_full_rank_ritz(stiffness, forcing, physical_gram)
    coordinate_map = np.diag(np.geomspace(0.2, 5.0, len(forcing)))
    transformed = solve_full_rank_ritz(
        *transformed_system(stiffness, forcing, physical_gram, coordinate_map)
    )
    assert baseline.certified
    assert transformed.certified
    assert np.isclose(transformed.action, baseline.action, rtol=2e-13)
    assert np.allclose(
        transformed.generalized_eigenvalues,
        baseline.generalized_eigenvalues,
        rtol=2e-13,
        atol=2e-13,
    )


def test_repaired_action_is_invariant_to_orthogonal_mixing() -> None:
    stiffness, forcing, physical_gram = _spd_problem()
    rng = np.random.default_rng(91)
    coordinate_map, _ = np.linalg.qr(rng.normal(size=stiffness.shape))
    baseline = solve_full_rank_ritz(stiffness, forcing, physical_gram)
    transformed = solve_full_rank_ritz(
        *transformed_system(stiffness, forcing, physical_gram, coordinate_map)
    )
    assert transformed.certified
    assert np.isclose(transformed.action, baseline.action, rtol=2e-13)


def test_enriched_sensor_trial_space_contains_tangent_witness() -> None:
    rng = np.random.default_rng(12)
    points = rng.uniform(-2.0, 2.0, size=(600, 2))
    centers = np.array([[-1.0, -0.5], [0.8, -0.7], [-0.3, 1.1], [1.2, 0.9]])
    sensor = gaussian_sensor_basis(points, centers, sigma=0.65)
    weights = rng.uniform(0.2, 1.0, size=len(points))
    weights /= weights.sum()
    forcing = rng.normal(size=len(points))
    sensor_k, sensor_f = assemble_variational_system(sensor, weights, forcing)
    sensor_action = solve_full_rank_ritz(sensor_k, sensor_f, np.eye(4)).action

    extra_values = np.column_stack((points[:, 0], points[:, 1], points[:, 0] * points[:, 1]))
    extra_x = np.column_stack((np.ones(len(points)), np.zeros(len(points)), points[:, 1]))
    extra_y = np.column_stack((np.zeros(len(points)), np.ones(len(points)), points[:, 0]))
    from experiments.ocean_drifters.full_action_repair import TrialBasis

    enriched = TrialBasis(
        values=np.column_stack((sensor.values, extra_values)),
        gradient_x=np.column_stack((sensor.gradient_x, extra_x)),
        gradient_y=np.column_stack((sensor.gradient_y, extra_y)),
        names=sensor.names + ("x", "y", "xy"),
    )
    full_k, full_f = assemble_variational_system(enriched, weights, forcing)
    full_action = solve_full_rank_ritz(full_k, full_f, np.eye(7)).action
    assert sensor_action <= full_action + 5e-12 * max(full_action, 1.0)


def test_nested_full_rank_ritz_actions_are_nondecreasing() -> None:
    stiffness, forcing, physical_gram = _spd_problem(size=8)
    actions = []
    for size in (3, 4, 6, 8):
        result = solve_full_rank_ritz(
            stiffness[:size, :size],
            forcing[:size],
            physical_gram[:size, :size],
        )
        assert result.certified
        actions.append(result.action)
    assert np.all(np.diff(actions) >= -1e-12)


def test_generalized_spectral_sum_matches_direct_spd_action() -> None:
    stiffness, forcing, physical_gram = _spd_problem()
    result = solve_full_rank_ritz(stiffness, forcing, physical_gram)
    expected = float(forcing @ np.linalg.solve(stiffness, forcing))
    assert result.certified
    assert np.isclose(result.action, expected, rtol=2e-13)
    assert np.isclose(result.spectral_action, expected, rtol=2e-13)


def test_primary_solver_keeps_small_positive_forced_mode() -> None:
    stiffness = np.diag([1.0, 1.0e-16])
    forcing = np.array([1.0, 1.0e-8])
    result = solve_full_rank_ritz(stiffness, forcing, np.eye(2))
    truncated = generalized_cutoff_actions(result)
    assert np.isclose(result.action, 2.0, rtol=2e-15)
    assert np.isclose(result.generalized_contributions[0], 1.0, rtol=2e-15)
    assert np.isclose(truncated[1.0e-14], 1.0, rtol=2e-15)


def test_structural_redundancy_uses_fixed_H_not_small_weighted_eigenvalue() -> None:
    # Three raw functions represent two physical directions: phi_3=phi_1+phi_2.
    representation = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    physical_gram = representation.T @ representation
    physical_stiffness = np.diag([1.0, 1.0e-8])
    stiffness = representation.T @ physical_stiffness @ representation
    physical_forcing = np.array([1.0, 1.0e-4])
    forcing = representation.T @ physical_forcing
    result = solve_full_rank_ritz(stiffness, forcing, physical_gram)
    assert result.structural_rank == 2
    assert np.isclose(result.action, 2.0, rtol=1e-7)
    assert result.minimum_generalized_eigenvalue > 0.0


def test_old_coordinate_cutoff_can_change_but_untruncated_action_does_not() -> None:
    stiffness = np.diag([1.0, 1.0e-8, 3.0e-10])
    forcing = np.array([-2.30197061e-1, -3.52950842e-5, -9.23118556e-6])
    physical_gram = np.eye(3)
    coordinate_map = np.array([
        [0.79640914, 0.33188893, 0.40050631],
        [0.15085142, 0.98633901, -0.26337101],
        [0.21532610, 0.48328086, 0.51564881],
    ])
    baseline = solve_full_rank_ritz(stiffness, forcing, physical_gram)
    transformed_values = transformed_system(
        stiffness, forcing, physical_gram, coordinate_map
    )
    transformed = solve_full_rank_ritz(*transformed_values)
    old_baseline = old_equilibrated_cutoff_actions(stiffness, forcing)
    old_transformed = old_equilibrated_cutoff_actions(
        transformed_values[0], transformed_values[1]
    )
    assert np.isclose(transformed.action, baseline.action, rtol=2e-6)
    assert any(
        not np.isclose(old_transformed[tolerance], old_baseline[tolerance], rtol=1e-4)
        for tolerance in old_baseline
    )
