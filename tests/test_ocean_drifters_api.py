from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mfsi.config import load_config
from mfsi.measurements import GaussianPointSensors2D

from experiments.ocean_drifters.experiment import (
    FrozenArtifactError,
    OceanDriftersExperiment,
    run_experiment,
)
from experiments.ocean_drifters.action import _positive_kernel_reconstruct
from experiments.ocean_drifters.full_action import _forcing
from experiments.ocean_drifters.full_action_production import (
    OceanFullActionProduction,
)
from experiments.ocean_drifters.final_evaluation import _validate_release_manifest
from experiments.ocean_drifters.poisson_backend import (
    OCEAN_VARIATIONAL_POISSON_BACKEND,
    OceanVariationalPoissonConfig,
    solve_ocean_variational_poisson_batch,
    solve_ocean_variational_poisson_quadrature,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/ocean_drifters/config.json"


def test_ocean_frozen_api_loads_without_final_test() -> None:
    cfg = load_config(CONFIG)
    experiment = OceanDriftersExperiment(cfg)
    assert isinstance(experiment.measurements(), GaussianPointSensors2D)
    assert experiment.cohort.inference.shape == (200, 181, 2)
    assert experiment.cohort.validation.shape == (70, 181, 2)
    assert experiment.sensor_bank.centers_km.shape == (512, 4, 2)
    manifest = experiment.numerical_admissibility()
    assert manifest["admissible_layout_count"] == 512
    assert manifest["excluded_design_ids"] == []
    assert manifest["final_test_accessed"] is False


def test_ocean_post_dispersion_action_matches_vortex_result_shape() -> None:
    cfg = load_config(CONFIG)
    experiment = OceanDriftersExperiment(cfg)
    action = experiment.post_dispersion_action()

    assert action.design_indices == (216,)
    assert action.start_day == 12.0
    assert action.end_day == 45.0
    np.testing.assert_allclose(action.normalized_times[[0, -1]], [0.0, 1.0])
    np.testing.assert_allclose(action.time_weights.sum(), 1.0)
    np.testing.assert_allclose(action.action_density_scale, (33.0 / 45.0) ** 2)

    tangent = action.exact_tangent_result(216)
    full = action.exact_full_result(216)
    assert tangent["valid"] and full["valid"]
    assert set(("valid", "value", "rows")).issubset(tangent)
    assert set(("valid", "value", "rows")).issubset(full)
    assert len(tangent["rows"]) == len(full["rows"]) == 133
    assert tangent["value"] == pytest.approx(153061.76319751356)
    assert full["value"] == pytest.approx(481208.82166242064)
    assert tangent["value"] <= full["value"]

    layouts = action.evaluate_layouts_exact()
    assert len(layouts) == 1
    assert all(row["valid"] for row in layouts)
    assert all(row["tangent_lower_bound_valid"] for row in layouts)
    assert layouts[0]["regularization_diagnostic_time_node_count"] == 7
    assert layouts[0]["integrated_regularization_bias"] < 0.05


def test_ocean_post_dispersion_stage_writes_canonical_result(
    tmp_path: Path,
) -> None:
    cfg = load_config(CONFIG)
    payload = run_experiment(cfg, tmp_path, stage="post_dispersion_action")

    assert payload["stage"] == "post_dispersion_action"
    assert payload["post_dispersion_action"]["window_days"] == [12.0, 45.0]
    assert payload["post_dispersion_action"]["all_layouts_valid"] is True
    assert payload["post_dispersion_action"][
        "temporal_quadrature_refinement_certified"
    ] is True
    assert payload["post_dispersion_action"]["production_run"] is True
    assert payload["post_dispersion_action"]["full_action_production_valid"] is True
    assert payload["statuses"]["post_dispersion_action_valid"] is True
    assert payload["statuses"]["full_action_valid"] is True
    assert payload["statuses"]["full_horizon_action_valid"] is False
    assert payload["final_test_accessed"] is False
    assert (tmp_path / "result.json").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "result.json" in manifest["files"]


def test_ocean_post_dispersion_window_is_ocean_local() -> None:
    ocean = load_config(CONFIG)
    assert ocean["scientific"]["action_window_days"] == [12.0, 45.0]
    for relative in (
        "experiments/toy_example/config.json",
        "experiments/vortices/config.json",
    ):
        other = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert "action_window_days" not in other.get("scientific", {})
        assert "post_dispersion_regularization_audit" not in other.get(
            "action", {}
        )


def test_ocean_risk_api_uses_complete_validation_ids() -> None:
    cfg = load_config(CONFIG, smoke=True)
    assert cfg["law"]["max_relative_risk_violation"] == 0.05
    assert "frozen_additive_epsilon" not in cfg["law"]
    experiment = OceanDriftersExperiment(cfg)
    result = experiment.scientific_risk(
        design_indices=np.asarray([216]),
        time_positions=np.asarray([1, 2, 3]),
        bootstrap_replicates=4,
    )
    assert result["risk"].shape == (1,)
    assert result["risk_by_time"].shape == (1, 3)
    assert result["bootstrap_risk"].shape == (1, 4)
    assert np.isfinite(result["risk"]).all()
    assert result["summary"]["bootstrap_unit"] == "complete validation drifter ID"
    assert result["summary"]["relative_risk_allowance"] == 0.05
    assert result["summary"]["risk_ceiling"] == pytest.approx(
        1.05 * result["summary"]["R_star"]
    )
    assert result["summary"]["final_test_accessed"] is False


def test_ocean_final_evaluation_remains_locked(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    with pytest.raises(PermissionError, match="final-test trajectories are locked"):
        run_experiment(cfg, tmp_path, stage="final_evaluation")
    manifest = json.loads((tmp_path / "numerical_admissibility_manifest.json").read_text())
    assert manifest["final_test_accessed"] is False


def test_ocean_final_evaluation_dry_run_reproduces_validation_risk(
    tmp_path: Path,
) -> None:
    cfg = load_config(CONFIG)
    payload = run_experiment(cfg, tmp_path, stage="final_evaluation_dry_run")
    result = payload["final_evaluation_dry_run"]

    assert result["status"] == "ready_for_explicit_one_shot_authorization"
    assert result["selected_design_id"] == "design_000216"
    assert result["validation_id_count"] == 70
    assert result["evaluation_time_count"] == 19
    assert result["reproduced_validation_risk"] == pytest.approx(
        0.006490757661634413, abs=2e-12
    )
    assert result["acceptance_upper_bound"] == pytest.approx(
        0.023351205336727864, abs=2e-15
    )
    assert result["locked_cohort_path_resolved"] is False
    assert result["locked_cohort_hashed"] is False
    assert result["locked_cohort_opened"] is False
    assert result["split_manifest_opened"] is False
    assert payload["final_test_accessed"] is False
    assert (tmp_path / "result.json").is_file()


def test_ocean_final_evaluation_requires_both_authorization_keys() -> None:
    cfg = load_config(CONFIG)
    cfg["scientific"]["final_test_access_allowed"] = True
    with pytest.raises(FrozenArtifactError, match="two-key authorization"):
        OceanDriftersExperiment(cfg)

    cfg = load_config(CONFIG)
    cfg["final_evaluation"][
        "authorization_status"
    ] = "explicit_user_authorization_recorded"
    with pytest.raises(FrozenArtifactError, match="two-key authorization"):
        OceanDriftersExperiment(cfg)


def test_ocean_final_evaluation_release_manifest_is_hash_anchored() -> None:
    cfg = load_config(CONFIG)
    cfg["final_evaluation"]["release_manifest_expected_sha256"] = "0" * 64
    experiment = OceanDriftersExperiment(cfg)
    with pytest.raises(FrozenArtifactError, match="release manifest hash changed"):
        _validate_release_manifest(experiment)


def test_positive_kernel_reconstruction_is_convex_and_has_analytic_derivative() -> None:
    observation_days = np.asarray([0.0, 1.0, 2.0])
    raw = np.asarray([[[0.1], [0.7], [0.3]]])
    evaluation_days = np.linspace(0.0, 2.0, 17)
    values, derivative, weights = _positive_kernel_reconstruct(
        raw, observation_days, evaluation_days, bandwidth_days=0.6
    )

    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(values >= raw.min())
    assert np.all(values <= raw.max())
    assert np.isfinite(derivative).all()

    step = 1e-5
    plus, _, _ = _positive_kernel_reconstruct(
        raw, observation_days, evaluation_days + step, bandwidth_days=0.6
    )
    minus, _, _ = _positive_kernel_reconstruct(
        raw, observation_days, evaluation_days - step, bandwidth_days=0.6
    )
    np.testing.assert_allclose(derivative, (plus - minus) / (2.0 * step), rtol=2e-8, atol=2e-10)


def test_ocean_tangent_status_reads_canonical_output(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    experiment = OceanDriftersExperiment(cfg)
    readiness = tmp_path / "tangent_action.csv"
    readiness.write_text("valid\nTrue\nFalse\n", encoding="utf-8")
    experiment.cfg["action"]["tangent_readiness_table"] = str(readiness)
    status = experiment.action_status("tangent_action")
    assert status["layout_count"] == 2
    assert status["valid_count"] == 1
    assert status["backend"] == "tesseract_cpp"


def test_ocean_full_forcing_is_compatible_for_exact_projected_moments() -> None:
    rng = np.random.default_rng(20260817)
    weights = rng.random(17)
    weights /= weights.sum()
    phi = rng.normal(size=(17, 4))
    material = rng.normal(size=(17, 4))
    target = weights @ phi
    h, mean, relative = _forcing(
        phi,
        target,
        material,
        weights,
        rng.normal(size=4),
        rng.normal(size=4),
    )

    assert np.isfinite(h).all()
    assert abs(mean) < 1e-14
    assert relative < 1e-14


def test_poisson_pilot_selection_is_frozen_and_tangent_ready() -> None:
    cfg = load_config(CONFIG)
    experiment = OceanDriftersExperiment(cfg)
    selection_path = ROOT / cfg["action"]["poisson_pilot"]["selection_table"]
    selection = selection_path.read_text(encoding="utf-8")

    assert selection.count("\n") == cfg["action"]["poisson_pilot"]["layout_count"] + 1
    assert "design_000216" not in selection
    assert ",True,False" in selection


def test_ocean_variational_poisson_keeps_the_common_result_fields() -> None:
    pytest.importorskip("tesseract_jax")
    ny, nx, dx = 11, 15, 0.2
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    expected = np.cos(np.pi * xx / (nx * dx))
    eigenvalue = (np.pi / (nx * dx)) ** 2
    forcing = -eigenvalue * expected
    result = solve_ocean_variational_poisson_batch(
        np.full((1, ny, nx), -np.log(nx * ny)),
        forcing[None],
        OceanVariationalPoissonConfig(dx=dx, maximum_mode=2),
    )

    assert result.action.shape == (1,)
    assert result.potential.shape == (1, ny, nx)
    assert result.relative_residual.shape == (1,)
    assert result.weighted_mean_potential.shape == (1,)
    np.testing.assert_array_equal(result.operator_floor, np.zeros(1))
    assert bool(result.converged[0])
    assert result.relative_residual[0] < 1e-10
    assert abs(result.weighted_mean_potential[0]) < 1e-13
    np.testing.assert_allclose(result.potential[0], expected, rtol=1e-12, atol=1e-12)
    assert "weak_relative_residual" in result.diagnostics
    assert "scaled_weak_relative_residual" in result.diagnostics
    assert "discarded_scaled_load_relative_residual" in result.diagnostics
    assert "gauge_relative_residual" in result.diagnostics


def test_ocean_nonuniform_quadrature_matches_structured_ritz_contract() -> None:
    pytest.importorskip("tesseract_jax")
    ny, nx, dx = 11, 15, 0.2
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    forcing = -(
        np.pi / (nx * dx)
    ) ** 2 * np.cos(np.pi * xx / (nx * dx))
    log_q = np.full((ny, nx), -np.log(nx * ny))
    config = OceanVariationalPoissonConfig(dx=dx, maximum_mode=2)
    structured = solve_ocean_variational_poisson_batch(
        log_q[None], forcing[None], config
    )
    adaptive = solve_ocean_variational_poisson_quadrature(
        points,
        log_q.ravel(),
        forcing.ravel(),
        np.asarray([0.0, nx * dx, 0.0, ny * dx]),
        config,
    )

    np.testing.assert_allclose(adaptive.action, structured.action, rtol=2e-13)
    np.testing.assert_allclose(
        adaptive.potential.reshape((1, ny, nx)),
        structured.potential,
        rtol=2e-12,
        atol=2e-12,
    )
    assert bool(adaptive.converged[0])
    assert "compatibility_relative_residual" in adaptive.diagnostics


def test_variational_backend_selection_is_confined_to_ocean_experiment() -> None:
    ocean = load_config(CONFIG)
    assert ocean["action"]["poisson_backend"] == OCEAN_VARIATIONAL_POISSON_BACKEND
    for relative in (
        "experiments/toy_example/config.json",
        "experiments/vortices/config.json",
    ):
        other = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert other.get("action", {}).get("poisson_backend") != (
            OCEAN_VARIATIONAL_POISSON_BACKEND
        )


def test_ocean_rejects_unknown_poisson_backend() -> None:
    cfg = load_config(CONFIG)
    cfg["action"]["poisson_backend"] = "unknown_global_override"
    with pytest.raises(FrozenArtifactError, match="ocean Poisson backend"):
        OceanDriftersExperiment(cfg)


def test_ocean_soft_projection_contract_is_ocean_only_and_inference_scaled() -> None:
    cfg = load_config(CONFIG)
    soft = cfg["action"]["soft_moment_projection"]
    assert soft["finite_sample_standard_error_floor_rule"] == (
        "inverse_inference_trajectory_count"
    )
    assert soft["tangent_rank_sensitivity_relative_tolerances"] == [
        1e-10, 1e-12, 1e-14
    ]
    assert soft["maximum_relative_tangent_rank_action_change"] == 0.1
    assert soft["minimum_valid_layout_fraction"] == 0.95

    invalid = load_config(CONFIG)
    invalid["action"]["soft_moment_projection"][
        "finite_sample_standard_error_floor_rule"
    ] = "validation_tuned"
    with pytest.raises(FrozenArtifactError, match="inference-only"):
        OceanDriftersExperiment(invalid)


def test_ocean_full_action_production_contract_is_ocean_local() -> None:
    cfg = load_config(CONFIG)
    production = cfg["action"]["full_action_production"]
    assert production["layout_count"] == 1
    assert production["time_count"] == 181
    assert production["grid_resolution"] == [511, 273]
    assert production["adaptive_through_source_index"] == 9
    assert production["minimum_valid_layout_fraction"] == 0.95
    assert production["production_run_authorized"] is False
    assert cfg["scientific"]["final_test_access_allowed"] is False

    for relative in (
        "experiments/toy_example/config.json",
        "experiments/vortices/config.json",
    ):
        other = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        assert "full_action_production" not in other.get("action", {})


def test_ocean_production_runner_is_locked_after_relative_risk_change(
    tmp_path: Path,
) -> None:
    cfg = load_config(CONFIG)
    experiment = OceanDriftersExperiment(cfg)
    with pytest.raises(RuntimeError, match="production is not authorized"):
        OceanFullActionProduction(experiment, tmp_path, tmp_path)
