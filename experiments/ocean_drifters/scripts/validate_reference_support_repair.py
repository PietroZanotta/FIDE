#!/usr/bin/env python3
"""Validate the gated continuous-endpoint support-repair attempt."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import load_phase2_config, resolve  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.reference import MLPReferenceFlow  # noqa: E402


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    phase2 = load_phase2_config()
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_continuous_endpoints"
    required = [
        SCRIPT_DIR.parent / "configs/reference_support_repair.json",
        processed / "endpoint_density_estimator/gaussian_kde_endpoints.npz",
        model_dir / "reference.npz",
        model_dir / "reference_bank_eval_200000.npz",
        analysis / "tables/reference_support_cases.json",
        analysis / "tables/reference_support_lp_old.csv",
        analysis / "tables/reference_endpoint_metrics.csv",
        analysis / "tables/reference_endpoint_acceptance.json",
        analysis / "tables/reference_support_repair_summary.json",
        analysis / "reference_support_repair.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing repair artifacts: {missing}"
    cases = read_json(analysis / "tables/reference_support_cases.json")
    assert cases["case_count"] == 20 and len(cases["cases"]) == 20
    assert [case["case"] for case in cases["cases"]] == list(range(20))
    assert all(len(case["target_moments"]) == 4 for case in cases["cases"])
    old_rows = read_csv(analysis / "tables/reference_support_lp_old.csv")
    assert len(old_rows) == 80
    metrics = read_csv(analysis / "tables/reference_endpoint_metrics.csv")
    repaired = {row["endpoint"]: row for row in metrics if row["model"] == "continuous_kde"}
    assert repaired["day0"]["accepted"] == "True"
    assert repaired["day45"]["accepted"] == "False"
    assert float(repaired["day45"]["outside_domain_fraction"]) > 0.01
    endpoint = read_json(analysis / "tables/reference_endpoint_acceptance.json")
    summary = read_json(analysis / "tables/reference_support_repair_summary.json")
    assert endpoint["passed"] is False and endpoint["intermediate_positions_used"] is False
    assert endpoint["final_test_artifact_loaded"] is False
    assert summary["endpoint_acceptance_passed"] is False
    assert summary["support_audit_run"] is False
    assert summary["full_bank_sweep_authorized"] is False
    assert summary["full_bank_sweep_completed"] is False
    assert summary["final_test_artifact_loaded"] is False
    assert not (analysis / "tables/reference_support_lp_continuous.csv").exists()
    with np.load(processed / "endpoint_density_estimator/gaussian_kde_endpoints.npz", allow_pickle=False) as data:
        assert int(data["inference_n"]) == 200
        assert not bool(data["final_test_accessed"])
        for key in ["H0_km2", "H1_km2"]:
            matrix = np.asarray(data[key], dtype=np.float64)
            assert matrix.shape == (2, 2) and np.linalg.eigvalsh(matrix).min() > 0.0
    with np.load(model_dir / "reference_bank_eval_200000.npz", allow_pickle=False) as data:
        nodes = np.asarray(data["nodes_km"])
        initial = np.asarray(data["initial_km"])
        assert not bool(data["final_test_accessed"])
    assert nodes.shape == (19, 200000, 2) and initial.shape == (200000, 2)
    assert np.isfinite(nodes).all() and np.isfinite(initial).all()
    unique = read_csv(analysis / "tables/reference_continuous_unique_paths.csv")
    assert [int(row["particle_count"]) for row in unique] == [2000, 10000, 50000, 200000]
    assert all(
        int(row["unique_initial_exact"]) == int(row["particle_count"])
        and int(row["unique_day45_exact"]) == int(row["particle_count"])
        for row in unique
    )
    flow = MLPReferenceFlow.from_npz(
        model_dir / "reference.npz",
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    metadata = flow.metadata or {}
    assert metadata["endpoint_only"] is True
    assert metadata["intermediate_positions_used_for_training"] is False
    assert metadata["architecture_optimizer_schedule_changed"] is False
    report = (analysis / "reference_support_repair.md").read_text(encoding="utf-8")
    for phrase in ["did not pass", "2.538%", "200,000", "not run", "bounded-domain endpoint-only stochastic bridge"]:
        assert phrase in report, phrase
    print("Validated gated support repair: 200k unique paths, endpoint gate failed, no intermediate/full-bank audit run.")


if __name__ == "__main__":
    main()
