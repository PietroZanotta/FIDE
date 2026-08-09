#!/usr/bin/env python3
"""Validate Stage-2 coupling diagnostic artifacts."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "coupling_study" / "diagnostics"


def main() -> None:
    decomposition = json.loads((OUTPUT / "objective_decomposition.json").read_text())
    resampling = json.loads((OUTPUT / "soft_plan_resampling.json").read_text())
    contrasts = json.loads((OUTPUT / "coupling_contrasts.json").read_text())
    decision = json.loads((OUTPUT / "diagnostic_decision.json").read_text())
    required = (
        "objective_decomposition.json", "objective_decomposition.csv",
        "soft_plan_resampling.json", "soft_plan_resampling.csv",
        "soft_plan_mmd_resampling.csv", "coupling_contrasts.json",
        "metric_alignment.json", "diagnostic_decision.json",
        "diagnostic_summary.md", "objective_decomposition.png",
        "soft_plan_resampling.png", "mmd_resampling.png",
        "geometric_ot_effects.png", "metric_change_vs_mmd.png",
    )
    gates = {
        "all_required_artifacts_present": all((OUTPUT / name).exists() for name in required),
        "five_scientific_banks": contrasts["n"] == 5,
        "all_three_splits_decomposed": (
            len(decomposition["paired_differences"]) == 15
            and set(decomposition["aggregate_paired_differences"])
            == {"train", "selection", "evaluation"}
        ),
        "twenty_plan_resamples": (
            resampling["resamples_per_fixed_plan"] == 20
            and len(resampling["rows"]) == 5 * 2 * 20
        ),
        "twenty_mmd_resamples": (
            resampling["mmd2_resampling"]["resamples_per_fixed_pipeline"] == 20
            and len(resampling["mmd2_resampling"]["rows"]) == 5 * 2 * 20
        ),
        "mmd_pipeline_reconstructed": (
            resampling["mmd2_resampling"]["aggregate"][
                "maximum_original_reconstruction_error"
            ] < 1e-6
        ),
        "resamples_not_treated_as_scientific_n": (
            resampling["resamples_are_scientific_replicates"] is False
            and resampling["scientific_replication_n"] == 5
        ),
        "no_joint_optimization": not decision["joint_schedule_coupling_currently_justified"],
        "heldout_metrics_not_used_for_design": not decision["q4_or_final_mmd_used_for_design"],
        "exactly_one_next_experiment": decision["next_experiment"].startswith("one coupling-only extension"),
    }
    print(json.dumps(gates, indent=2))
    if not all(gates.values()):
        raise SystemExit("coupling diagnostic validation failed")


if __name__ == "__main__":
    main()
