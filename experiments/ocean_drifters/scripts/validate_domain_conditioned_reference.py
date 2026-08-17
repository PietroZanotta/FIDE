#!/usr/bin/env python3
"""Validate Phase-2C endpoint conditioning and gated stop behavior."""

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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def payload(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    phase2 = load_phase2_config()
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_conditioned_endpoints"
    required = [
        SCRIPT_DIR.parent / "configs/domain_conditioned_reference.json",
        processed / "endpoint_density_estimator_conditioned/conditioned_kde_endpoints.npz",
        model_dir / "reference.npz",
        model_dir / "reference_bank_eval_200000.npz",
        analysis / "tables/conditioned_kde_normalization.csv",
        analysis / "tables/conditioned_endpoint_metrics.csv",
        analysis / "tables/conditioned_endpoint_acceptance.json",
        analysis / "tables/domain_conditioned_reference_summary.json",
        analysis / "domain_conditioned_reference_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing domain-conditioned artifacts: {missing}"
    normalization = {row["endpoint"]: row for row in rows(analysis / "tables/conditioned_kde_normalization.csv")}
    assert float(normalization["day0"]["Z_hat"]) == 1.0
    assert 0.9805 < float(normalization["day45"]["Z_hat"]) < 0.9813
    unbounded_path = processed / "endpoint_density_estimator/gaussian_kde_endpoints.npz"
    conditioned_path = processed / "endpoint_density_estimator_conditioned/conditioned_kde_endpoints.npz"
    with np.load(unbounded_path, allow_pickle=False) as old, np.load(conditioned_path, allow_pickle=False) as new:
        np.testing.assert_array_equal(old["H0_km2"], new["H0_km2"])
        np.testing.assert_array_equal(old["H1_km2"], new["H1_km2"])
        domain = np.asarray(new["domain_km"], dtype=np.float64)
        for key in ["conditioned_audit_x0_km", "conditioned_audit_x1_km"]:
            sample = np.asarray(new[key], dtype=np.float64)
            assert sample.shape == (100000, 2)
            assert np.all((sample[:, 0] >= domain[0]) & (sample[:, 0] <= domain[1]))
            assert np.all((sample[:, 1] >= domain[2]) & (sample[:, 1] <= domain[3]))
        assert not bool(new["final_test_accessed"])
    metric_rows = rows(analysis / "tables/conditioned_endpoint_metrics.csv")
    conditioned_estimators = [row for row in metric_rows if row["model"] == "conditioned_kde"]
    generated = {row["endpoint"]: row for row in metric_rows if row["model"] == "conditioned_generated"}
    assert len(conditioned_estimators) == 2 and all(row["accepted"] == "True" for row in conditioned_estimators)
    assert generated["day0"]["accepted"] == "True"
    assert generated["day45"]["accepted"] == "False"
    outside = float(generated["day45"]["outside_domain_fraction"])
    assert outside == 0.010465 and outside > 0.01
    acceptance = payload(analysis / "tables/conditioned_endpoint_acceptance.json")
    summary = payload(analysis / "tables/domain_conditioned_reference_summary.json")
    assert acceptance["conditioned_estimator_passed"] is True
    assert acceptance["generated_reference_passed"] is False
    assert acceptance["passed"] is False
    assert acceptance["final_test_artifact_loaded"] is False
    assert summary["endpoint_gate_passed"] is False
    assert summary["support_audit_run"] is False
    assert summary["full_bank_sweep_run"] is False
    assert summary["stochastic_bridge_justified"] is False
    assert summary["final_test_artifact_loaded"] is False
    assert not (analysis / "tables/reference_support_lp_conditioned.csv").exists()
    assert not (analysis / "tables/conditioned_full_bank_feasibility.csv").exists()
    unique = rows(analysis / "tables/conditioned_reference_unique_paths.csv")
    assert [int(row["particle_count"]) for row in unique] == [2000, 10000, 50000, 200000]
    assert all(
        int(row[key]) == int(row["particle_count"])
        for row in unique
        for key in ["unique_initial_exact", "unique_midpoint_exact", "unique_day45_exact"]
    )
    with np.load(model_dir / "reference_bank_eval_200000.npz", allow_pickle=False) as bank:
        assert bank["nodes_km"].shape == (19, 200000, 2)
        assert not bool(bank["final_test_accessed"])
    flow = MLPReferenceFlow.from_npz(
        model_dir / "reference.npz",
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    metadata = flow.metadata or {}
    assert metadata["endpoint_only"] is True
    assert metadata["intermediate_positions_used_for_training"] is False
    assert metadata["architecture_optimizer_bridge_integration_changed"] is False
    report = (analysis / "domain_conditioned_reference_report.md").read_text(encoding="utf-8")
    for phrase in ["Z0 = 1.000000", "ZT = 0.980895", "1.0465%", "not run", "not yet justified"]:
        assert phrase in report, phrase
    print("Validated Phase 2C: conditioned estimator passes, generated day-45 leakage=1.0465%, downstream audits correctly gated.")


if __name__ == "__main__":
    main()
