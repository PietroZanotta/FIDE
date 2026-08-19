from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.active_nematic.measurements import PeriodicGaussianSensors
from experiments.active_nematic.periodic_numerics import (
    PeriodicGrid2D,
    PeriodicGrid3D,
    PeriodicPoissonBatchResult,
    PeriodicPoissonConfig,
    periodic_weighted_laplacian3d,
    periodic_weighted_laplacian,
    rasterize_periodic_particles3d,
    solve_periodic_weighted_poisson3d_batch_jax,
    solve_periodic_weighted_poisson,
)
from experiments.active_nematic.periodic_reference import periodic_delta
from experiments.active_nematic.risk import periodic_mmd2
from experiments.active_nematic.experiment import (
    ActiveNematicExperiment,
    ObservationTrialBank,
)


def test_periodic_measurement_gradient_matches_autodiff() -> None:
    family = PeriodicGaussianSensors(
        box_size=10.0,
        width=1.2,
        n_sensors=2,
        channels=("occupancy", "polarity_cos", "polarity_sin"),
    )
    state = jnp.asarray([9.9, 0.1, 0.4], dtype=jnp.float64)
    eta = jnp.asarray([0.2, 9.8, 4.0, 5.0], dtype=jnp.float64)
    analytic = family.feature_gradients(state, eta)
    automatic = jax.jacfwd(lambda value: family.features(value, eta))(state)
    np.testing.assert_allclose(analytic, automatic, rtol=2.0e-10, atol=2.0e-10)


def test_periodic_measurements_are_seam_invariant() -> None:
    family = PeriodicGaussianSensors(10.0, width=1.0, n_sensors=1, channels=("occupancy",))
    eta = jnp.asarray([0.1, 9.9])
    first = family.features(jnp.asarray([9.8, 0.2]), eta)
    shifted = family.features(jnp.asarray([-0.2, 10.2]), eta)
    np.testing.assert_allclose(first, shifted, atol=1.0e-14)


def test_periodic_mmd_identifies_equivalent_wrapped_samples() -> None:
    x = jnp.asarray([[0.1, 9.9], [5.0, 4.0]])
    y = x + jnp.asarray([10.0, -10.0])
    value = periodic_mmd2(
        x, y, periods=jnp.asarray([10.0, 10.0]), bandwidths=jnp.asarray([0.5, 1.0])
    )
    assert float(value) < 1.0e-12


def test_shortest_periodic_endpoint_displacement() -> None:
    delta = periodic_delta(
        jnp.asarray([0.1, 9.9, 0.05]),
        jnp.asarray([9.9, 0.1, 2.0 * np.pi - 0.05]),
        jnp.asarray([10.0, 10.0, 2.0 * np.pi]),
    )
    np.testing.assert_allclose(delta, np.asarray([0.2, -0.2, 0.1]), atol=1.0e-12)


def test_periodic_weighted_poisson_manufactured_solution() -> None:
    grid = PeriodicGrid2D(box_size=2.0 * np.pi, n=16)
    coordinates = (np.arange(grid.n) + 0.5) * grid.dx
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    q = jnp.ones((grid.n, grid.n), dtype=jnp.float64)
    expected = jnp.asarray(np.cos(xx) + 0.4 * np.sin(2.0 * yy))
    h = -periodic_weighted_laplacian(expected, q, grid.dx)
    result = solve_periodic_weighted_poisson(
        q,
        h,
        grid,
        PeriodicPoissonConfig(operator_floor_rel=0.0, cg_tol=1.0e-11, cg_maxiter=200),
    )
    recovered = result.potential - jnp.mean(result.potential)
    target = expected - jnp.mean(expected)
    np.testing.assert_allclose(recovered, target, rtol=2.0e-8, atol=2.0e-8)
    assert float(result.relative_residual) < 1.0e-9


def test_polarity_channels_reject_position_only_state() -> None:
    family = PeriodicGaussianSensors(
        box_size=10.0, channels=("occupancy", "polarity_cos")
    )
    with pytest.raises(ValueError, match="polarity channels"):
        family.features(jnp.asarray([1.0, 2.0]), jnp.zeros(2 * family.n_sensors))


def _experiment_config(channels=("occupancy",)):
    return {
        "seed": 4,
        "physics": {"box_size": 10.0},
        "measurement": {
            "n_sensors": 1,
            "sensor_width": 1.2,
            "channels": list(channels),
            "acquisition_k": 3,
            "obs_noise_std": 0.0,
        },
        "moment_reconstruction": {"internal_knots": 1, "smoothing": 1.0e-5},
        "projection": {"max_steps": 30, "residual_tol": 1.0e-8},
        "particle_mfsi": {"covariance_ridge": 1.0e-7, "tangent_ridge": 1.0e-7},
        "full_action": {
            "grid_n": 8,
            "grid_shape_polarity": [6, 6, 6],
            "polarity_metric_radius": 0.75,
            "backend_3d": "jax",
            "operator_floor_rel": 1.0e-4,
            "cg_maxiter": 250,
            "cg_tol": 1.0e-8,
        },
        "law": {
            "mmd_bandwidths": [0.8, 1.6],
            "grid_shape_position": [8, 8],
            "grid_shape_polarity": [8, 8, 6],
            "epsilon_r": 0.01,
        },
        "validity": {
            "max_calibration_residual": 1.0e-3,
            "min_ess_fraction": 0.01,
            "max_poisson_relative_residual": 1.0e-3,
        },
        "optimization": {"invalid_penalty": 1000.0},
    }


def test_position_only_experiment_objective_is_differentiable() -> None:
    times = jnp.asarray([0.0, 0.5, 1.0])
    base = jnp.asarray(
        [[1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 2.5],
         [5.0, 3.0], [6.0, 3.5], [7.0, 4.0], [8.0, 4.5]]
    )
    nodes = jnp.stack([base + jnp.asarray([0.2 * time, 0.0]) for time in times])
    velocity = jnp.broadcast_to(jnp.asarray([0.2, 0.0]), nodes.shape)
    weights = jnp.full(nodes.shape[:2], 1.0 / nodes.shape[1])
    experiment = ActiveNematicExperiment(
        _experiment_config(),
        times=times,
        truth_particles=nodes,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    indices = jnp.broadcast_to(jnp.arange(nodes.shape[1]), (1, 3, nodes.shape[1]))
    bank = ObservationTrialBank(indices, jnp.zeros((1, 3, 1)))
    eta = jnp.asarray([2.5, 2.0])
    value, gradient = jax.value_and_grad(lambda design: experiment.mean_metric(design, bank, "law_risk"))(eta)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()


def test_time_guard_points_zero_boundary_quadrature_weights() -> None:
    times = jnp.linspace(0.0, 1.0, 5)
    nodes = jnp.broadcast_to(
        jnp.asarray(
            [[1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 2.5]]
        ),
        (5, 4, 2),
    )
    config = _experiment_config()
    config["evaluation"] = {"time_guard_points": 1}
    experiment = ActiveNematicExperiment(
        config,
        times=times,
        truth_particles=nodes,
        reference_nodes=nodes,
        reference_velocity=jnp.zeros_like(nodes),
        reference_weights=jnp.full((5, 4), 0.25),
    )
    np.testing.assert_allclose(
        experiment.time_weights,
        np.asarray([0.0, 0.25, 0.5, 0.25, 0.0]),
    )
    provenance = experiment.full_action_provenance()
    assert provenance["time_guard_points"] == 1
    np.testing.assert_allclose(provenance["time_weights"], experiment.time_weights)


def test_authoritative_audit_uses_robust_projection() -> None:
    times = jnp.asarray([0.0, 0.5, 1.0])
    nodes = jnp.broadcast_to(
        jnp.asarray(
            [[1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 2.5],
             [5.0, 3.0], [6.0, 3.5], [7.0, 4.0], [8.0, 4.5]]
        ),
        (3, 8, 2),
    )
    experiment = ActiveNematicExperiment(
        _experiment_config(),
        times=times,
        truth_particles=nodes,
        reference_nodes=nodes,
        reference_velocity=jnp.zeros_like(nodes),
        reference_weights=jnp.full((3, 8), 1.0 / 8.0),
    )
    indices = jnp.broadcast_to(jnp.arange(8), (1, 3, 8))
    bank = ObservationTrialBank(indices, jnp.zeros((1, 3, 1)))
    audit = experiment.audit_metric(
        jnp.asarray([2.5, 2.0]), bank, "full_action"
    )
    assert audit["projection_solver"] == "robust_empirical_tilt_exact"
    assert audit["valid"]
    np.testing.assert_allclose(audit["value"], 0.0, atol=1.0e-12)
    assert audit["max_calibration_residual"] < 1.0e-10
    assert audit["min_empirical_hull_support_gap"] >= 0.0


def test_polarity_state_uses_differentiable_3d_full_action() -> None:
    times = jnp.asarray([0.0, 0.5, 1.0])
    particles = jnp.asarray(
        [[[1.0, 1.0, 0.1], [4.0, 5.0, 1.2]],
         [[1.2, 1.0, 0.2], [4.2, 5.0, 1.3]],
         [[1.4, 1.0, 0.3], [4.4, 5.0, 1.4]]]
    )
    experiment = ActiveNematicExperiment(
        _experiment_config(("occupancy", "polarity_cos", "polarity_sin")),
        times=times,
        truth_particles=particles,
        reference_nodes=particles,
        reference_velocity=jnp.zeros_like(particles),
        reference_weights=jnp.full((3, 2), 0.5),
    )
    bank = ObservationTrialBank(jnp.zeros((1, 3, 1), dtype=jnp.int32), jnp.zeros((1, 3, 3)))
    eta = jnp.asarray([2.0, 2.0])
    assert experiment.full_action_supported
    value, gradient = jax.value_and_grad(
        lambda design: experiment.mean_metric(design, bank, "full_action")
    )(eta)
    assert np.isfinite(float(value))
    assert float(value) > 1.0
    assert np.isfinite(np.asarray(gradient)).all()


def test_anisotropic_periodic_poisson3d_manufactured_solution() -> None:
    grid = PeriodicGrid3D(
        box_size=2.0 * np.pi,
        shape=(9, 8, 7),
        polarity_metric_radius=0.65,
    )
    x = np.arange(grid.shape[0])[:, None, None] * grid.dx
    y = np.arange(grid.shape[1])[None, :, None] * grid.dy
    theta_metric = (
        np.arange(grid.shape[2])[None, None, :] * float(grid.dtheta_metric)
    )
    expected = jnp.asarray(
        np.cos(x) + 0.3 * np.sin(2.0 * y) + 0.2 * np.cos(theta_metric / 0.65)
    )
    q = jnp.ones((1, *grid.shape), dtype=jnp.float64)
    h = -periodic_weighted_laplacian3d(expected, q[0], grid.spacings)[None]
    result = solve_periodic_weighted_poisson3d_batch_jax(
        q,
        h,
        grid,
        PeriodicPoissonConfig(
            operator_floor_rel=0.0, cg_tol=1.0e-11, cg_maxiter=500
        ),
    )
    recovered = result.potential[0] - jnp.mean(result.potential[0])
    target = expected - jnp.mean(expected)
    np.testing.assert_allclose(recovered, target, rtol=3.0e-8, atol=3.0e-8)
    assert float(result.relative_residual[0]) < 1.0e-9


def test_polarity_raster_preserves_mass_compatibility_and_periodic_seams() -> None:
    grid = PeriodicGrid3D(8.0, (9, 8, 7), polarity_metric_radius=0.6)
    particles = jnp.asarray(
        [[0.1, 7.9, 0.05], [7.8, 0.2, 2.1], [4.0, 3.0, 6.2]]
    )
    weights = jnp.asarray([0.2, 0.3, 0.5])
    forcing = jnp.asarray([1.0, -2.0, 0.8])
    first = rasterize_periodic_particles3d(
        particles, weights, forcing, grid, bandwidth=0.5
    )
    shifted = rasterize_periodic_particles3d(
        particles + jnp.asarray([8.0, -8.0, 2.0 * np.pi]),
        weights,
        forcing,
        grid,
        bandwidth=0.5,
    )
    np.testing.assert_allclose(first.q, shifted.q, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(first.h, shifted.h, rtol=0.0, atol=2.0e-12)
    assert float(jnp.min(first.q)) >= 0.0
    np.testing.assert_allclose(
        grid.cell_volume * jnp.sum(first.q), 1.0, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        grid.cell_volume * jnp.sum(first.q * first.h),
        0.0,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_full_action_batches_trials_and_times_in_one_poisson_call(monkeypatch) -> None:
    times = jnp.asarray([0.0, 0.5, 1.0])
    particles = jnp.asarray(
        [
            [[1.0, 1.0, 0.1], [3.0, 2.0, 0.8], [5.0, 4.0, 1.7]],
            [[1.2, 1.0, 0.2], [3.1, 2.1, 0.9], [5.2, 4.0, 1.8]],
            [[1.4, 1.0, 0.3], [3.2, 2.2, 1.0], [5.4, 4.0, 1.9]],
        ]
    )
    experiment = ActiveNematicExperiment(
        _experiment_config(("occupancy", "polarity_cos", "polarity_sin")),
        times=times,
        truth_particles=particles,
        reference_nodes=particles,
        reference_velocity=jnp.zeros_like(particles),
        reference_weights=jnp.full((3, 3), 1.0 / 3.0),
    )
    indices = jnp.asarray(
        [
            [[0, 1], [1, 2], [0, 2]],
            [[1, 2], [0, 2], [0, 1]],
        ],
        dtype=jnp.int32,
    )
    bank = ObservationTrialBank(indices, jnp.zeros((2, 3, 3)))
    calls = []

    def fake_solve(q, h):
        calls.append((q.shape, h.shape))
        batch = q.shape[0]
        return PeriodicPoissonBatchResult(
            action=jnp.arange(batch, dtype=jnp.float64),
            potential=jnp.zeros_like(q),
            relative_residual=jnp.zeros((batch,), dtype=jnp.float64),
            weighted_mean_potential=jnp.zeros((batch,), dtype=jnp.float64),
            operator_floor=jnp.zeros((batch,), dtype=jnp.float64),
        )

    monkeypatch.setattr(experiment, "_solve_poisson3d_batch", fake_solve)
    rows = experiment._batch_trial_values(
        jnp.asarray([2.0, 2.0]), bank, full=True
    )
    assert calls == [((6, 6, 6, 6), (6, 6, 6, 6))]
    np.testing.assert_allclose([row[2] for row in rows], [1.0, 4.0])
