#!/usr/bin/env python3
"""Validate completed Stage-2 coupling-study artifacts without rerunning them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coupling_study import METHODS


def validate(path: Path) -> dict[str, bool]:
    summary = json.loads((path / "summary.json").read_text())
    checks = summary["aggregate"]["checks"]
    gates = {
        "stage_is_coupling_only": (
            summary["stage"] == 2
            and summary["joint_schedule_coupling_optimization"] is False
        ),
        "all_methods_present": all(
            all(method in seed["methods"] for method in METHODS)
            for seed in summary["seed_reports"]
        ),
        "schedules_are_frozen": all(
            seed["fixed_schedule"]["reoptimized"] is False
            for seed in summary["seed_reports"]
        ),
        "endpoint_marginals_preserved": checks["endpoint_marginals_preserved"],
        "finite_endpoint_moments_calibrated": checks["finite_endpoint_moments_calibrated"],
        "implicit_gradient_validated": checks["gradient_validation_passed"],
        "no_final_mmd_leakage": not checks["test_mmd_used_for_training_or_selection"],
        "independent_bank_roles_recorded": all(
            {"coupling_optimization", "coupling_validation", "final_evaluation"}
            <= set(seed["banks"])
            for seed in summary["seed_reports"]
        ),
        "pair_sampling_seeds_recorded": all(
            "random_seed" in sampling
            for seed in summary["seed_reports"]
            for method in METHODS
            for sampling in seed["methods"][method]["neural_downstream"][
                "sampling_diagnostics"
            ].values()
        ),
        "required_outputs_present": all(
            (path / name).exists()
            for name in (
                "summary.json", "coupling_metrics.csv", "REPORT.md",
                "coupling_summary.png", "coupling_time_diagnostics.png",
                "coupling_paired_effects.png", "coupling_plan_diagnostics.png",
            )
        ),
    }
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "results" / "coupling_study" / "standard",
    )
    args = parser.parse_args()
    gates = validate(args.path)
    print(json.dumps(gates, indent=2))
    if not all(gates.values()):
        raise SystemExit("coupling-study artifact validation failed")


if __name__ == "__main__":
    main()
