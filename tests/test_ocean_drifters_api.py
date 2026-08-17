from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mfsi.config import load_config
from mfsi.measurements import GaussianPointSensors2D

from experiments.ocean_drifters.experiment import (
    OceanDriftersExperiment,
    run_experiment,
)
from experiments.ocean_drifters.action import _positive_kernel_reconstruct


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


def test_ocean_risk_api_uses_complete_validation_ids() -> None:
    cfg = load_config(CONFIG, smoke=True)
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
    assert result["summary"]["final_test_accessed"] is False


def test_ocean_final_evaluation_remains_locked(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    with pytest.raises(PermissionError, match="final-test trajectories are locked"):
        run_experiment(cfg, tmp_path, stage="final_evaluation")
    manifest = json.loads((tmp_path / "numerical_admissibility_manifest.json").read_text())
    assert manifest["final_test_accessed"] is False


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
