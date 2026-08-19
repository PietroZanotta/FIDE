#!/usr/bin/env python3
"""Finalize versioned risk-freeze and dense-moment inputs from soft-law output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mfsi.cache import file_sha256  # noqa: E402
from mfsi.config import load_config  # noqa: E402
from experiments.ocean_drifters.action import _features  # noqa: E402
from experiments.ocean_drifters.experiment import OceanDriftersExperiment  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "experiments/ocean_drifters/config.json")
    experiment = OceanDriftersExperiment(cfg)
    processed = ROOT / "experiments/ocean_drifters/processed"
    table_dir = ROOT / "experiments/ocean_drifters/analysis/tables"
    risk_path = processed / "iprojection_soft_grid_validation_risk.npz"
    parameter_path = processed / "iprojection_soft_parameters.npz"
    with np.load(risk_path, allow_pickle=False) as data:
        risks = np.asarray(data["risks"], dtype=np.float64)
        bootstrap = np.asarray(data["bootstrap_risk"], dtype=np.float64)
    with np.load(parameter_path, allow_pickle=False) as data:
        stationarity = np.asarray(data["stationarity_residual"], dtype=np.float64)
        hard_residual = np.asarray(data["hard_moment_residual"], dtype=np.float64)
        penalty_trace = np.asarray(data["penalty_trace"], dtype=np.float64)

    best = int(np.argmin(risks))
    epsilon = float(np.std(bootstrap[best], ddof=1))
    near = np.flatnonzero(risks <= risks[best] + epsilon)
    near = near[np.argsort(risks[near])]
    freeze = {
        "schema_version": 2,
        "projection_method": "soft finite-sample covariance-of-the-mean I-projection",
        "action_values_inspected_before_freeze": False,
        "best_design_id": str(experiment.sensor_bank.design_ids[best]),
        "best_validation_risk": float(risks[best]),
        "epsilon_choice": "one bootstrap standard error of the point-estimate best layout",
        "final_test_artifact_loaded": False,
        "frozen_additive_epsilon": epsilon,
        "near_optimal_design_ids": experiment.sensor_bank.design_ids[near].tolist(),
        "near_optimal_layout_count": int(len(near)),
        "rationale": (
            "the previously declared one-best-layout-bootstrap-SE rule, rerun after "
            "the inference-only finite-sample projection correction and before "
            "corrected tangent/full-action selection"
        ),
        "risk_ceiling": float(risks[best] + epsilon),
        "risk_projection_artifact_sha256": file_sha256(risk_path),
        "final_test_accessed": False,
    }
    freeze_path = table_dir / "soft_law_risk_freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    times = np.asarray(experiment.cohort.normalized_time, dtype=np.float64)
    inference = np.asarray(experiment.cohort.inference, dtype=np.float64)
    raw = np.empty((len(near), len(times), 4), dtype=np.float64)
    for local, design in enumerate(near):
        raw[local] = _features(
            inference.reshape((-1, 2)),
            experiment.sensor_bank.centers_km[design],
            experiment.sensor_bank.sigma_km,
        ).reshape((len(inference), len(times), 4)).mean(axis=0)
    dense_path = processed / "soft_law_dense_moments.npz"
    np.savez_compressed(
        dense_path,
        design_indices=near,
        normalized_times=times,
        raw_moments=raw,
        projection_method=np.asarray("soft_finite_sample_covariance_of_mean"),
        final_test_accessed=np.asarray(False),
    )

    diagnostic_path = table_dir / "soft_law_projection_diagnostics.csv"
    with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "design_index", "design_id", "maximum_moment_residual",
            "maximum_soft_stationarity_residual", "maximum_penalty_trace",
            "mean_projection_kl", "minimum_log10_intrinsic_ess",
            "worst_covariance_condition_regularized",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for design in range(len(risks)):
            writer.writerow({
                "design_index": design,
                "design_id": experiment.sensor_bank.design_ids[design],
                "maximum_moment_residual": float(np.max(hard_residual[design])),
                "maximum_soft_stationarity_residual": float(np.max(stationarity[design])),
                "maximum_penalty_trace": float(np.max(penalty_trace[design])),
                "mean_projection_kl": "",
                "minimum_log10_intrinsic_ess": "",
                "worst_covariance_condition_regularized": "",
            })
    print(json.dumps({
        "risk_freeze": str(freeze_path),
        "dense_moments": str(dense_path),
        "projection_diagnostics": str(diagnostic_path),
        "near_optimal_layout_count": int(len(near)),
        "best_design_id": str(experiment.sensor_bank.design_ids[best]),
        "final_test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
