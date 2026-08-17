#!/usr/bin/env python3
"""Validate frozen Phase-2 artifacts and leakage/numerical contracts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import gaussian_features_numpy, load_phase2_config, resolve  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.poisson_tesseract import is_tesseract_poisson_available  # noqa: E402
from mfsi.projection_tesseract import is_tesseract_iprojection_available  # noqa: E402


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    cfg = load_phase2_config()
    processed = resolve(cfg["processed_dir"])
    analysis = resolve(cfg["analysis_dir"])
    model_dir = resolve(cfg["model_dir"])
    required = [
        processed / "cohort_45d.npz",
        processed / "development_270.npz",
        processed / "splits/split_manifest.csv",
        processed / "splits/repeated_cv_manifest.csv",
        processed / "sensor_bank.npz",
        processed / "measurement_trajectories.npz",
        processed / "iprojection_primary.npz",
        model_dir / "reference.npz",
        model_dir / "reference_bank.npz",
        analysis / "tables/iprojection_diagnostics.csv",
        analysis / "tables/repeated_cv_summary.json",
        analysis / "tables/reference_support_lp_summary.json",
        analysis / "mfsi_phase2_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing Phase-2 artifacts: {missing}"
    with np.load(processed / "cohort_45d.npz", allow_pickle=False) as data:
        cohort_X = np.asarray(data["X"])
        cohort_ids = np.asarray(data["ids"]).astype(str)
    assert cohort_X.shape == (339, 181, 2)
    metadata = csv_rows(processed / "metadata.csv")
    assert len(metadata) == 339
    assert all(row["all_finite"] == "True" and row["all_drogued"] == "True" and row["exact_6_hour"] == "True" for row in metadata)
    assert np.isfinite(cohort_X).all()
    assert len(np.unique(cohort_ids)) == 339
    split_rows = csv_rows(processed / "splits/split_manifest.csv")
    split_ids = {name: {row["drifter_id"] for row in split_rows if row["split"] == name}
                 for name in ["inference", "validation", "final_test"]}
    assert {name: len(values) for name, values in split_ids.items()} == {
        "inference": 200, "validation": 70, "final_test": 69,
    }
    assert not (split_ids["inference"] & split_ids["validation"])
    assert not (split_ids["inference"] & split_ids["final_test"])
    assert not (split_ids["validation"] & split_ids["final_test"])
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        development_X = np.asarray(data["X"])
        development_ids = set(np.asarray(data["ids"]).astype(str))
        roles = np.asarray(data["split"]).astype(str)
    assert development_X.shape == (270, 181, 2)
    assert development_ids == split_ids["inference"] | split_ids["validation"]
    assert not development_ids & split_ids["final_test"]
    assert dict(zip(*np.unique(roles, return_counts=True), strict=True)) == {
        "inference": 200, "validation": 70,
    }
    cv_rows = csv_rows(processed / "splits/repeated_cv_manifest.csv")
    for repeat in range(3):
        fold = [row for row in cv_rows if int(row["repeat"]) == repeat]
        assert len(fold) == 270 and {row["drifter_id"] for row in fold} == development_ids
        counts = {role: sum(row["role"] == role for row in fold) for role in ["inference", "validation"]}
        assert counts == {"inference": 200, "validation": 70}
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"])
        sigma = float(data["sigma_km"])
        bounds = np.asarray(data["bounds_km"])
        minimum = float(data["min_separation_km"])
    assert centers.shape == (512, 4, 2) and sigma == 200.0
    assert np.all((centers[..., 0] >= bounds[0]) & (centers[..., 0] <= bounds[1]))
    assert np.all((centers[..., 1] >= bounds[2]) & (centers[..., 1] <= bounds[3]))
    distances = np.linalg.norm(centers[:, :, None] - centers[:, None, :], axis=-1)
    distances += np.eye(4)[None] * 1e30
    assert distances.min() >= minimum - 1e-10
    with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"])
        assert not bool(data["final_test_accessed"])
    assert measurements.shape == (512, 181, 4) and np.isfinite(measurements).all()
    inference = development_X[roles == "inference"]
    expected = gaussian_features_numpy(inference[:, 0], centers[0], sigma).mean(axis=0)
    np.testing.assert_allclose(measurements[0, 0], expected, rtol=2e-6, atol=2e-7)
    diagnostics = csv_rows(analysis / "tables/iprojection_diagnostics.csv")
    assert len(diagnostics) == 512 * 19
    valid_residuals = [float(row["verified_moment_residual"]) for row in diagnostics if row["valid"] == "True"]
    assert valid_residuals and max(valid_residuals) <= float(cfg["projection"]["accept_residual"])
    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        feasible = np.asarray(data["feasible"], dtype=bool)
        risks = np.asarray(data["risks"])
    assert feasible.shape == (512,) and int(feasible.sum()) == 2
    assert np.isfinite(risks).all()
    with (analysis / "tables/iprojection_risk_summary.json").open(encoding="utf-8") as handle:
        risk_summary = json.load(handle)
    with (analysis / "tables/repeated_cv_summary.json").open(encoding="utf-8") as handle:
        cv_summary = json.load(handle)
    with (analysis / "tables/reference_support_lp_summary.json").open(encoding="utf-8") as handle:
        support_summary = json.load(handle)
    assert risk_summary["final_test_artifact_loaded"] is False
    assert risk_summary["weighted_poisson_invoked"] is False
    assert cv_summary["final_test_artifact_loaded"] is False
    assert cv_summary["designs_feasible_all_repeats"] == 0
    assert support_summary["unique_support_counts"] == {
        "2000": 200, "10000": 200, "50000": 200, "200000": 200,
    }
    assert support_summary["lp_infeasible_at_200000"] == 16
    assert is_tesseract_iprojection_available() and is_tesseract_poisson_available()
    report = (analysis / "mfsi_phase2_report.md").read_text(encoding="utf-8")
    for phrase in ["339", "270 development / 69", "sigma = 200 km", "2 of 512", "no layout is feasible"]:
        assert phrase in report, phrase
    print("Validated Phase 2: cohort=339, development/final=270/69, bank=512, primary feasible=2, all-repeat feasible=0.")


if __name__ == "__main__":
    main()
