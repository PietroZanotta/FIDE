#!/usr/bin/env python3
"""Validate completed Stage-4 differentiable fiber-design artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage4_fiber_design"


def validate(path):
    summary = json.loads((path / "summary.json").read_text())
    validation = summary["validation"]
    configuration = summary["configuration"]
    return {
        "stage_is_four": summary["stage"] == 4,
        "five_scientific_banks": (
            summary["scientific_replication_n"] == 5
            and summary["seeds"] == [401, 402, 403, 404, 405]
        ),
        "exactly_three_observables": configuration["observable_count"] == 3,
        "paper_physical_dimension": (
            configuration["particle_count"] == 32
            and configuration["state_dimension"] == 64
        ),
        "all_physical_components_frozen": all(summary["frozen"].values()),
        "endpoint_laws_unchanged": validation["endpoint_laws_unchanged"],
        "endpoint_equivalence_exact": validation["endpoint_equivalence_enforced"],
        "hidden_q4_gap_preserved": validation["hidden_q4_gap_preserved"],
        "bank_roles_distinct": validation["bank_roles_distinct"],
        "hand_control_nested": validation["hand_candidate_nested"],
        "hand_subspace_matches_original": validation["hand_observable_subspace_matches_original"],
        "construction_gradient_validated": validation["gradient"]["passed"],
        "no_q4_leakage": validation["q4_used_for_adaptation_or_selection"] is False,
        "no_other_intervention": all(
            validation[name] is False for name in (
                "rollout_used", "schedule_optimized", "coupling_changed",
                "neural_model_trained",
            )
        ),
        "full_optimizer_protocol": (
            configuration["optimizer_steps"] == 40
            and configuration["candidate_interval"] == 5
            and configuration["learning_rate"] == 0.03
        ),
        "required_artifacts": all(
            (path / name).exists() for name in (
                "summary.json", "REPORT.md", "stage4_metrics.csv",
                "stage4_summary.png", "stage4_selection_q4.png",
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checks = validate(args.path)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Stage-4 artifact validation failed")


if __name__ == "__main__":
    main()
