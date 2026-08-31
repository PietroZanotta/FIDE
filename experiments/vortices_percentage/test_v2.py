from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for path in (REPO_ROOT / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from core import (
    config_fingerprint,
    frozen_common_reference_scott_bandwidth,
    frozen_reference_scott_bandwidth,
    independent_poisson,
    make_grid,
    rasterize_v2,
    sha256_file,
    solve_v2,
    weighted_gradient_relative_error,
)
from continuity_commutator import (
    column_normalized_kernel_1d,
    truncated_gaussian_log_normalizer_gradient,
)
from mfsi.poisson import PoissonConfig, weighted_laplacian
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.raster import (
    rasterize_projected_particles_reflected_rect,
    reflected_flux_divergence_rect,
    reflected_gaussian_cell_mass_matrix_1d,
    reflected_gaussian_face_flux_matrix_1d,
    reflected_particle_flux_rect,
)


def test_legacy_v1_scientific_inputs_are_unchanged():
    assert sha256_file(HERE / "base_experiment_config.json") == (
        "8f57f167675718b19d7ffc1741a8175adbe22069ff4043634b62df8dcf100ed0"
    )
    assert sha256_file(HERE / "experiment.py") == (
        "5bcd5b3c96668cabf6d7a8b2b1944f48f490635763b997172584328551a9a4c4"
    )


def test_config_hash_changes_for_every_scientific_setting_mutation():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    baseline = config_fingerprint(config)
    for path, value in (
        (("raster", "physical_bandwidth"), 0.061),
        (("raster", "boundary_rule"), "periodic"),
        (("raster", "reflected_image_pairs"), 3),
        (("projection", "type"), "soft"),
        (("continuity", "equation"), "K psi = s"),
        (("development", "particle_counts"), [4096, 8192]),
    ):
        changed = copy.deepcopy(config)
        changed[path[0]][path[1]] = value
        assert config_fingerprint(changed) != baseline


def test_reference_bandwidth_is_physical_and_grid_independent():
    rng = np.random.default_rng(7)
    nodes = rng.normal(size=(5, 256, 2))
    weights = np.full((5, 256), 1.0 / 256.0)
    bandwidth, by_time = frozen_reference_scott_bandwidth(nodes, weights)
    assert bandwidth > 0.0
    assert by_time.shape == (5,)
    # The rule has no grid argument; converting it to cell units changes while
    # its physical value remains exactly fixed.
    coarse, fine = make_grid(64, 32), make_grid(256, 128)
    assert bandwidth == bandwidth
    assert bandwidth / fine.dx == pytest.approx(4.0 * bandwidth / coarse.dx)
    common, values = frozen_common_reference_scott_bandwidth(
        [(nodes, weights), (1.1 * nodes, weights), (0.9 * nodes, weights)]
    )
    assert values.shape == (3,)
    assert common == pytest.approx(float(np.median(values)))


def test_v2_reflected_operator_conserves_mass_source_is_positive_and_float64():
    grid = make_grid(64, 32)
    nodes = jnp.asarray(
        [[0.03, 0.04], [0.45, 0.73], [1.25, 0.31], [1.97, 0.94]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([0.12, 0.23, 0.41, 0.24], dtype=jnp.float64)
    forcing = jnp.asarray([0.8, -0.4, 0.17, -0.25], dtype=jnp.float64)
    forcing -= jnp.sum(weights * forcing)
    result = rasterize_v2(nodes, weights, forcing, grid, bandwidth=0.09)
    assert result.q.dtype == jnp.float64
    assert result.source.dtype == jnp.float64
    assert float(jnp.min(result.q)) > 0.0
    np.testing.assert_allclose(np.sum(result.mass), 1.0, atol=3e-15)
    np.testing.assert_allclose(np.sum(result.source) * grid.cell_area, 0.0, atol=3e-15)
    np.testing.assert_allclose(result.q * result.h, result.source, rtol=3e-15, atol=1e-15)

    x_edges = jnp.linspace(grid.x_min, grid.x_max, grid.nx + 1)
    y_edges = jnp.linspace(grid.y_min, grid.y_max, grid.ny + 1)
    kx = reflected_gaussian_cell_mass_matrix_1d(
        x_edges, nodes[:, 0], bandwidth=0.09, image_pairs=4
    )
    ky = reflected_gaussian_cell_mass_matrix_1d(
        y_edges, nodes[:, 1], bandwidth=0.09, image_pairs=4
    )
    np.testing.assert_allclose(np.sum(kx, axis=0), 1.0, atol=3e-15)
    np.testing.assert_allclose(np.sum(ky, axis=0), 1.0, atol=3e-15)
    expected_mass = (ky * weights[None, :]) @ kx.T
    expected_source = (ky * (weights * forcing)[None, :]) @ kx.T
    expected_source -= expected_mass * jnp.sum(expected_source) / jnp.sum(expected_mass)
    np.testing.assert_allclose(result.mass, expected_mass, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        result.source * grid.cell_area, expected_source, rtol=2e-15, atol=2e-15
    )


def test_grid_axis_orientation_is_y_then_x():
    grid = make_grid(40, 20)
    result = rasterize_v2(
        jnp.asarray([[1.73, 0.23]], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        grid,
        bandwidth=0.025,
    )
    iy, ix = np.unravel_index(int(np.argmax(np.asarray(result.q))), grid.shape)
    x = float(np.asarray(grid.x_centers())[ix])
    y = float(np.asarray(grid.y_centers())[iy])
    assert abs(x - 1.73) < 0.06
    assert abs(y - 0.23) < 0.06


def test_v2_sign_energy_moment_identity_and_independent_solver():
    grid = make_grid(32, 16)
    points = np.asarray(grid.points())
    q = 0.4 + np.exp(-((points[..., 0] - 1.0) ** 2 + (points[..., 1] - 0.5) ** 2) / 0.2)
    q /= np.sum(q) * grid.cell_area
    psi_true = np.sin(np.pi * points[..., 0] / 2.0) * np.sin(np.pi * points[..., 1])
    operator = np.asarray(weighted_laplacian(jnp.asarray(psi_true), jnp.asarray(q), grid.dx))
    source = -operator
    solved = solve_v2(q, source, grid)
    solved_flip = solve_v2(q, -source, grid)
    np.testing.assert_allclose(solved.action, solved_flip.action, rtol=2e-11)
    np.testing.assert_allclose(solved.potential, -solved_flip.potential, rtol=2e-10, atol=2e-10)
    assert float(solved.relative_residual[0]) < 1e-10
    energy = grid.cell_area * float(np.sum(np.asarray(solved.potential[0]) * (-source)))
    np.testing.assert_allclose(float(solved.action[0]), energy, rtol=2e-11)

    # For delta=-grad(psi), the discrete weak moment rate is
    # -<grad(phi),grad(psi)> = integral(phi*s).
    feature = points[..., 0] ** 2 + 0.3 * points[..., 1]
    correction_rate = -grid.cell_area * float(
        np.sum(feature * np.asarray(weighted_laplacian(solved.potential[0], q, grid.dx)))
    )
    expected_rate = grid.cell_area * float(np.sum(feature * source))
    np.testing.assert_allclose(correction_rate, expected_rate, rtol=2e-11, atol=2e-11)

    independent = independent_poisson(q, source, grid)
    np.testing.assert_allclose(independent["action"], solved.action[0], rtol=2e-9)
    assert independent["relative_residual"] < 2e-10
    assert weighted_gradient_relative_error(
        q, np.asarray(solved.potential[0]), independent["potential"], grid
    ) < 2e-9


def test_v2_weak_full_continuity_for_moving_reweighted_particles():
    grid = make_grid(96, 48)
    nodes = np.asarray(
        [[0.55, 0.35], [0.72, 0.67], [1.18, 0.41], [1.45, 0.64]],
        dtype=np.float64,
    )
    weights = np.asarray([0.20, 0.25, 0.30, 0.25], dtype=np.float64)
    velocity = np.asarray(
        [[0.04, 0.02], [-0.03, 0.01], [0.02, -0.04], [-0.025, 0.03]],
        dtype=np.float64,
    )
    forcing = np.asarray([0.5, -0.4, 0.25, -0.2], dtype=np.float64)
    forcing -= weights @ forcing
    bandwidth = 0.09
    epsilon = 1e-5
    weights_plus = weights * np.exp(epsilon * forcing)
    weights_minus = weights * np.exp(-epsilon * forcing)
    weights_plus /= weights_plus.sum()
    weights_minus /= weights_minus.sum()
    q_plus = np.asarray(
        rasterize_v2(
            nodes + epsilon * velocity, weights_plus, np.zeros(4), grid,
            bandwidth=bandwidth,
        ).q
    )
    q_minus = np.asarray(
        rasterize_v2(
            nodes - epsilon * velocity, weights_minus, np.zeros(4), grid,
            bandwidth=bandwidth,
        ).q
    )
    center = rasterize_v2(nodes, weights, forcing, grid, bandwidth=bandwidth)
    flux_x, flux_y = reflected_particle_flux_rect(
        nodes, weights, velocity, grid, bandwidth=bandwidth, image_pairs=4
    )
    positive_defect = (
        (q_plus - q_minus) / (2.0 * epsilon)
        + reflected_flux_divergence_rect(flux_x, flux_y, grid)
    )
    source = np.asarray(center.source)
    points = np.asarray(grid.points())
    tests = (
        np.ones(grid.shape),
        points[..., 0],
        points[..., 1],
        np.exp(-((points[..., 0] - 1.0) ** 2 + (points[..., 1] - 0.5) ** 2) / 0.2),
    )
    for feature in tests:
        weak_error = grid.cell_area * np.sum(feature * (positive_defect - source))
        weak_scale = max(grid.cell_area * np.sum(np.abs(feature * source)), 1e-12)
        assert abs(weak_error) / weak_scale < 1e-8

    solved = solve_v2(center.q, center.source, grid)
    for feature in tests:
        correction_rate = -grid.cell_area * np.sum(
            feature * np.asarray(weighted_laplacian(solved.potential[0], center.q, grid.dx))
        )
        source_rate = grid.cell_area * np.sum(feature * source)
        np.testing.assert_allclose(correction_rate, source_rate, rtol=2e-10, atol=2e-10)


def test_column_normalized_kernel_commutator_sign_and_derivative():
    bandwidth = 0.09
    points = np.asarray(
        [[0.02, 0.03], [1.0, 0.5], [1.98, 0.97]], dtype=np.float64
    )
    analytic = truncated_gaussian_log_normalizer_gradient(
        points, bandwidth=bandwidth
    )
    # Moving inward increases retained Gaussian mass; moving outward decreases it.
    assert analytic[0, 0] > 0.0 and analytic[0, 1] > 0.0
    assert abs(analytic[1, 0]) < 1.0e-12 and abs(analytic[1, 1]) < 1.0e-6
    assert analytic[2, 0] < 0.0 and analytic[2, 1] < 0.0

    centers = np.linspace(0.01, 1.99, 100)
    operator = column_normalized_kernel_1d(centers, bandwidth)
    reconstructed_source_derivative = (
        -operator.target_derivative
        - operator.kernel * operator.source_log_normalizer_gradient[None, :]
    )
    epsilon = 1.0e-7

    def normalized_kernel(source_centers):
        displacement = centers[:, None] - source_centers[None, :]
        kernel = np.exp(-0.5 * (displacement / bandwidth) ** 2)
        return kernel / np.sum(kernel, axis=0, keepdims=True)

    finite = (
        normalized_kernel(centers + epsilon)
        - normalized_kernel(centers - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        reconstructed_source_derivative, finite, rtol=2.0e-6, atol=2.0e-8
    )


def test_reflected_kernel_commutes_with_manufactured_particle_continuity():
    grid = make_grid(64, 32)
    bandwidth = 0.08
    epsilon = 1.0e-6
    nodes = np.asarray(
        [[0.015, 0.30], [1.985, 0.72], [0.65, 0.012], [1.30, 0.988]],
        dtype=np.float64,
    )
    weights = np.asarray([0.17, 0.31, 0.23, 0.29], dtype=np.float64)
    velocity = np.asarray(
        [[0.04, 0.01], [-0.03, -0.01], [0.01, 0.035], [-0.01, -0.025]],
        dtype=np.float64,
    )
    forcing = np.asarray([0.35, -0.22, 0.17, -0.11], dtype=np.float64)
    forcing -= weights @ forcing
    weights_plus = weights * np.exp(epsilon * forcing)
    weights_minus = weights * np.exp(-epsilon * forcing)
    weights_plus /= np.sum(weights_plus)
    weights_minus /= np.sum(weights_minus)
    minus = rasterize_projected_particles_reflected_rect(
        nodes - epsilon * velocity,
        weights_minus,
        np.zeros_like(weights),
        grid,
        bandwidth=bandwidth,
    )
    plus = rasterize_projected_particles_reflected_rect(
        nodes + epsilon * velocity,
        weights_plus,
        np.zeros_like(weights),
        grid,
        bandwidth=bandwidth,
    )
    center = rasterize_projected_particles_reflected_rect(
        nodes, weights, forcing, grid, bandwidth=bandwidth
    )
    flux_x, flux_y = reflected_particle_flux_rect(
        nodes, weights, velocity, grid, bandwidth=bandwidth
    )
    np.testing.assert_allclose(flux_x[:, [0, -1]], 0.0, atol=2.0e-14)
    np.testing.assert_allclose(flux_y[[0, -1], :], 0.0, atol=2.0e-14)
    residual = (
        (plus.q - minus.q) / (2.0 * epsilon)
        + reflected_flux_divergence_rect(flux_x, flux_y, grid)
        - center.source
    )
    source_l2 = np.sqrt(np.sum(center.source**2) * grid.cell_area)
    residual_l2 = np.sqrt(np.sum(residual**2) * grid.cell_area)
    assert residual_l2 / source_l2 < 2.0e-8
    assert abs(np.sum(center.mass) - 1.0) < 3.0e-15
    assert np.min(center.q) > 0.0


def test_reflected_scalar_derivative_matches_odd_face_flux_exactly():
    edges = jnp.linspace(0.0, 1.0, 65)
    sources = jnp.asarray([0.012, 0.27, 0.73, 0.988], dtype=jnp.float64)
    bandwidth = 0.08
    epsilon = 1.0e-6
    plus = reflected_gaussian_cell_mass_matrix_1d(
        edges, sources + epsilon, bandwidth=bandwidth, image_pairs=4
    )
    minus = reflected_gaussian_cell_mass_matrix_1d(
        edges, sources - epsilon, bandwidth=bandwidth, image_pairs=4
    )
    finite_source_derivative = (plus - minus) / (2.0 * epsilon)
    face = reflected_gaussian_face_flux_matrix_1d(
        edges, sources, bandwidth=bandwidth, image_pairs=4
    )
    exact_cell_derivative = -jnp.diff(face, axis=0)
    np.testing.assert_allclose(
        finite_source_derivative, exact_cell_derivative, rtol=3e-9, atol=3e-10
    )
    np.testing.assert_array_equal(np.asarray(face)[[0, -1]], 0.0)


def test_four_reflected_image_pairs_are_saturated_at_frozen_bandwidth():
    bandwidth = 0.05883961987664522
    for lower, upper, cells in ((0.0, 2.0, 256), (0.0, 1.0, 128)):
        edges = jnp.linspace(lower, upper, cells + 1)
        sources = jnp.linspace(lower + 1.0e-6, upper - 1.0e-6, 257)
        scalar_four = reflected_gaussian_cell_mass_matrix_1d(
            edges, sources, bandwidth=bandwidth, image_pairs=4
        )
        scalar_five = reflected_gaussian_cell_mass_matrix_1d(
            edges, sources, bandwidth=bandwidth, image_pairs=5
        )
        flux_four = reflected_gaussian_face_flux_matrix_1d(
            edges, sources, bandwidth=bandwidth, image_pairs=4
        )
        flux_five = reflected_gaussian_face_flux_matrix_1d(
            edges, sources, bandwidth=bandwidth, image_pairs=5
        )
        np.testing.assert_array_equal(scalar_four, scalar_five)
        np.testing.assert_array_equal(flux_four, flux_five)
        np.testing.assert_allclose(np.sum(scalar_four, axis=0), 1.0, atol=2e-15)
        assert float(np.min(scalar_four)) > 0.0


def test_v2_production_does_not_call_column_normalized_raster(monkeypatch):
    import mfsi.raster as raster_module

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy source-column raster called by V2 production")

    monkeypatch.setattr(
        raster_module, "rasterize_projected_particles_positive_rect", forbidden
    )
    grid = make_grid(16, 8)
    result = rasterize_v2(
        np.asarray([[0.4, 0.3], [1.6, 0.7]]),
        np.asarray([0.45, 0.55]),
        np.asarray([0.2, -0.2 * 0.45 / 0.55]),
        grid,
        bandwidth=0.08,
        image_pairs=4,
    )
    assert np.all(np.asarray(result.q) > 0.0)


def test_hard_projection_lambda_dot_matches_recalibration_finite_difference():
    rng = np.random.default_rng(123)
    phi = jnp.asarray(rng.normal(size=(500, 3)), dtype=jnp.float64)
    base = jnp.asarray(rng.uniform(0.2, 1.0, size=500), dtype=jnp.float64)
    base /= jnp.sum(base)
    projector = EmpiricalIProjector(
        IProjectionConfig(max_steps=300, residual_tol=1e-12, newton_ridge=1e-12),
        trajectory_backend="jax",
    )
    target = np.asarray(projector.project(phi, base, jnp.asarray([0.08, -0.05, 0.03])).moments)
    target_dot = np.asarray([0.02, -0.01, 0.015])
    center = projector.project(phi, base, jnp.asarray(target))
    analytic = np.linalg.solve(np.asarray(center.covariance), target_dot)
    epsilon = 1e-4
    plus = projector.project(phi, base, jnp.asarray(target + epsilon * target_dot), lam0=center.lam)
    minus = projector.project(phi, base, jnp.asarray(target - epsilon * target_dot), lam0=center.lam)
    finite = (np.asarray(plus.lam) - np.asarray(minus.lam)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite, rtol=3e-6, atol=3e-7)


def test_frozen_bank_and_geometry_identity_if_development_artifacts_exist():
    root = HERE / "inputs" / "development_pareto" / "confirmatory_validation_2048" / "bank_19892"
    bank = root / "fresh_validation_bank.npz"
    summary = root / "summary.json"
    if not bank.exists() or not summary.exists():
        pytest.skip("ignored V1 development bank is not materialized")
    assert sha256_file(bank) == "b25fe9be6a467c451671cad110f44a63e24b9f7787a9af2b34b16aed096bc5bf"
    data = json.loads(summary.read_text(encoding="utf-8"))
    geometries = {row["key"]: row["geometry"] for row in data["manifest"]["geometries"]}
    np.testing.assert_allclose(
        geometries["ea6c90af64ce4356"],
        [1.06370935080401, 0.4550192034057061, 0.49853661140128447,
         0.7420342952622412, 1.7597617944632438, 0.24,
         0.2497059564587267, 0.6463819589743185],
        rtol=0.0,
        atol=0.0,
    )
