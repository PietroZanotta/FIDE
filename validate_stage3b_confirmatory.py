#!/usr/bin/env python3
"""Validate Stage 3B confirmatory artifacts and frozen protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage3b_confirmatory"


def validate(path):
    summary = json.loads((path / "summary.json").read_text())
    protocol = summary["protocol"]
    validation = summary["validation"]
    return {
        "stage_is_3b": summary["stage"] == "3B",
        "ten_new_seeds": (
            summary["scientific_replication_n"] == 10
            and summary["seeds"] == list(range(406, 416))
            and validation["models_are_new_seeds"]
        ),
        "protocol_predeclared": protocol["status"] == "predeclared-before-new-seed-execution",
        "frozen_hyperparameters": (
            protocol["frozen_stage3_settings"]["learning_rate"] == 0.04
            and protocol["frozen_stage3_settings"]["optimizer_steps"] == 40
            and protocol["frozen_stage3_settings"]["checkpoint_steps"]
            == [0, 5, 10, 15, 20, 25, 30, 35, 40]
            and protocol["frozen_stage3_settings"]["heun_steps"] == 24
        ),
        "all_three_adaptations_present": all(
            {"scalar", "stopped_state", "full"} <= set(report["optimizations"])
            for report in summary["seed_reports"]
        ),
        "models_frozen": validation["models_frozen"],
        "banks_distinct": validation["bank_roles_distinct"],
        "forward_control_exact": validation["full_stopped_forward_trajectories_identical"],
        "gradient_control_nontrivial": validation["full_stopped_gradients_differ"],
        "no_q4_or_evaluation_leakage": (
            validation["q4_used_for_adaptation_or_selection"] is False
            and validation["evaluation_used_for_adaptation_or_selection"] is False
        ),
        "required_artifacts": all(
            (path / name).exists() for name in (
                "summary.json", "REPORT.md", "stage3b_metrics.csv",
                "stage3b_summary.png", "stage3b_contrasts.png",
                "stage3b_selection.png",
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
        raise SystemExit("Stage 3B validation failed")


if __name__ == "__main__":
    main()
