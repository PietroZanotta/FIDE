#!/usr/bin/env python3
"""Validate Phase-2D domain-preserving reference and gated support decision."""

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
from mfsi.reference import DomainPreservingReferenceFlow  # noqa: E402


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
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_domain_preserving"
    required = [
        SCRIPT_DIR.parent / "configs/domain_preserving_reference.json",
        model_dir / "reference.npz",
        model_dir / "reference_bank_eval_200000.npz",
        analysis / "tables/domain_preserving_map_audit.csv",
        analysis / "tables/domain_preserving_endpoint_metrics.csv",
        analysis / "tables/domain_preserving_endpoint_acceptance.json",
        analysis / "tables/domain_preserving_unique_paths.csv",
        analysis / "tables/reference_support_lp_domain_preserving.csv",
        analysis / "tables/domain_preserving_reference_summary.json",
        analysis / "domain_preserving_reference_report.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing domain-preserving artifacts: {missing}"
    map_rows = {row["dataset"]: row for row in rows(analysis / "tables/domain_preserving_map_audit.csv")}
    assert all(float(row["map_epsilon"]) == 1e-6 for row in map_rows.values())
    assert int(map_rows["conditioned_training_day0"]["samples_requiring_map_clipping"]) == 0
    assert int(map_rows["conditioned_training_day45"]["samples_requiring_map_clipping"]) == 1
    assert float(map_rows["conditioned_training_day45"]["maximum_physical_coordinate_displacement_km"]) < 0.0012
    assert int(map_rows["empirical_inference_day0"]["exact_boundary_sample_count"]) == 0
    assert int(map_rows["empirical_inference_day45"]["exact_boundary_sample_count"]) == 0
    metrics = rows(analysis / "tables/domain_preserving_endpoint_metrics.csv")
    generated = {row["endpoint"]: row for row in metrics if row["model"] == "domain_preserving"}
    assert generated["day0"]["accepted"] == "True" and generated["day45"]["accepted"] == "True"
    assert float(generated["day45"]["outside_domain_fraction"]) == 0.0
    endpoint = payload(analysis / "tables/domain_preserving_endpoint_acceptance.json")
    assert endpoint["passed"] is True
    assert endpoint["strict_outside_particle_count_all_times"] == 0
    assert endpoint["exact_boundary_coordinate_count_all_times"] == 0
    assert endpoint["final_test_artifact_loaded"] is False
    unique = rows(analysis / "tables/domain_preserving_unique_paths.csv")
    assert [int(row["particle_count"]) for row in unique] == [2000, 10000, 50000, 200000]
    assert all(
        int(row[key]) == int(row["particle_count"])
        for row in unique
        for key in ["unique_initial_exact", "unique_midpoint_exact", "unique_day45_exact"]
    )
    audit = rows(analysis / "tables/reference_support_lp_domain_preserving.csv")
    assert len(audit) == 80
    expected_feasible = {2000: 8, 10000: 8, 50000: 9, 200000: 10}
    for size, count in expected_feasible.items():
        subset = [row for row in audit if int(row["particle_count"]) == size]
        assert sum(float(row["minimum_linf_residual"]) <= 2e-7 for row in subset) == count
    largest = [row for row in audit if int(row["particle_count"]) == 200000]
    assert sum(row["native_healthy"] == "True" for row in largest) == 2
    assert sum(row["classification"] == "A_support_repaired" for row in largest) == 2
    assert sum(row["classification"] == "B_feasible_but_unhealthy" for row in largest) == 8
    assert sum(row["classification"] == "C_convex_hull_failure" for row in largest) == 10
    summary = payload(analysis / "tables/domain_preserving_reference_summary.json")
    assert summary["endpoint_gate_passed"] is True
    assert summary["support_audit_run"] is True
    assert summary["support_audit_passed"] is False
    assert summary["lp_feasible_cases"] == 10 and summary["native_healthy_cases"] == 2
    assert summary["full_bank_sweep_authorized"] is False
    assert summary["full_bank_sweep_run"] is False
    assert summary["stochastic_bridge_justified"] is True
    assert summary["final_test_artifact_loaded"] is False
    assert not (analysis / "tables/domain_preserving_full_bank_feasibility.csv").exists()
    with np.load(model_dir / "reference_bank_eval_200000.npz", allow_pickle=False) as bank:
        nodes = np.asarray(bank["nodes_km"])
        velocities = np.asarray(bank["velocity_km_per_normalized_time"])
        assert not bool(bank["final_test_accessed"])
    assert nodes.shape == velocities.shape == (19, 200000, 2)
    assert np.isfinite(nodes).all() and np.isfinite(velocities).all()
    bounds = np.asarray([-650.0, 3000.0, -950.0, 1000.0])
    assert np.all((nodes[..., 0] > bounds[0]) & (nodes[..., 0] < bounds[1]))
    assert np.all((nodes[..., 1] > bounds[2]) & (nodes[..., 1] < bounds[3]))
    flow = DomainPreservingReferenceFlow.from_npz(
        model_dir / "reference.npz",
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    metadata = flow.metadata or {}
    assert metadata["endpoint_only"] is True
    assert metadata["intermediate_positions_used_for_training"] is False
    assert metadata["architecture_optimizer_bridge_integration_changed"] is False
    assert metadata["physical_velocity"] == "J_T(z) times latent_velocity"
    report = (analysis / "domain_preserving_reference_report.md").read_text(encoding="utf-8")
    for phrase in ["epsilon_map=1e-6", "1.14 m", "10/20", "2/20", "stochastic endpoint-only bridge"]:
        assert phrase in report, phrase
    print("Validated Phase 2D: endpoint/domain pass, LP=10/20, native healthy=2/20, stochastic bridge justified.")


if __name__ == "__main__":
    main()
