#!/usr/bin/env python3
"""Validate completed Stage-3 rollout-adaptation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage3_rollout_adaptation"


def validate(path):
    summary = json.loads((path / "summary.json").read_text())
    validation = summary["validation"]
    return {
        "stage_is_three": summary["stage"] == 3,
        "five_scientific_banks": (
            summary["scientific_replication_n"] == 5
            and summary["seeds"] == [401, 402, 403, 404, 405]
        ),
        "only_three_new_parameters": summary["configuration"]["new_trainable_parameters"] == 3,
        "all_protocol_components_frozen": all(summary["frozen"].values()),
        "neural_weights_frozen": validation["neural_parameters_frozen"],
        "bank_roles_distinct": validation["bank_roles_distinct"],
        "no_q4_leakage": validation["q4_used_for_adaptation_or_selection"] is False,
        "no_evaluation_leakage": validation["final_evaluation_used_for_adaptation_or_selection"] is False,
        "rollout_gradient_validated": validation["gradient"]["passed"],
        "functional_heun_matches_established": validation["functional_heun_parity_passed"],
        "mmd_kernel_matches_established": validation["mmd_parity"]["passed"],
        "fixed_solver_protocol": (
            summary["configuration"]["heun_steps"] == 24
            and summary["configuration"]["nfe"] == 48
            and summary["configuration"]["evaluation_times"] == [0.25, 0.5, 0.75, 1.0]
        ),
        "required_artifacts": all(
            (path / name).exists() for name in (
                "summary.json", "REPORT.md", "stage3_metrics.csv",
                "stage3_summary.png", "stage3_optimization.png", "stage3_paths.png",
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
        raise SystemExit("Stage-3 artifact validation failed")


if __name__ == "__main__":
    main()
