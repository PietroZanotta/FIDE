#!/usr/bin/env python3
"""Validate completed Stage-2B artifacts without rerunning the experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "independent", "geometric_sinkhorn", "fiber_aware", "fiber_aware_gram",
)


def validate(path: Path) -> dict[str, bool]:
    summary = json.loads((path / "summary.json").read_text())
    decision = json.loads((path / "decision.json").read_text())
    configuration = summary["configuration"]
    validation = summary["validation"]
    return {
        "stage_is_exactly_2b_coupling_only": (
            summary["stage"] == "2B"
            and summary["joint_schedule_coupling_optimization"] is False
        ),
        "five_standard_banks_present": (
            summary["scientific_replication_n"] == 5
            and [report["seed"] for report in summary["seed_reports"]]
            == [401, 402, 403, 404, 405]
        ),
        "all_four_comparators_present": all(
            all(method in report["methods"] for method in METHODS)
            for report in summary["seed_reports"]
        ),
        "exact_parameter_counts": (
            configuration["total_coupling_parameters"] == 45
            and configuration["existing_phi_parameters"] == 9
            and configuration["new_gram_parameters"] == 36
        ),
        "original_phi_features_unchanged": validation[
            "first_nine_features_unchanged"
        ],
        "gradient_validated": validation["gradient_passed"],
        "endpoint_marginals_preserved": (
            validation["marginals_passed"]
            and validation["maximum_plan_marginal_linf"] < 5e-8
        ),
        "frozen_protocol_recorded": all(summary["frozen"].values()),
        "no_q4_or_mmd_optimization_leakage": (
            configuration["q4_used_for_optimization"] is False
            and configuration["final_mmd_used_for_optimization_or_selection"] is False
        ),
        "numerical_sinkhorn_iterations_recorded": (
            configuration["sinkhorn_iterations"] == 100
            and configuration["rich_sinkhorn_iterations"] == 500
            and all(
                method["plan"]["sinkhorn_iterations"] == 500
                for report in summary["seed_reports"]
                for role in report["role_metrics"].values()
                for name, method in role.items()
                if name == "fiber_aware_gram"
            )
        ),
        "stop_decision_is_consistent": (
            decision == summary["decision"]
            and decision["stage2b_success"] is False
            and decision["stop_fiber_aware_coupling_development_for_this_paper"]
            and decision["joint_schedule_coupling_optimization_justified"] is False
            and decision["preferred_coupling_baseline_for_later_work"]
            == "geometric_sinkhorn"
        ),
        "required_outputs_present": all(
            (path / name).exists()
            for name in (
                "summary.json", "decision.json", "stage2b_metrics.csv", "REPORT.md",
                "stage2b_summary.png", "stage2b_paired_effects.png",
                "stage2b_time_diagnostics.png",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=(Path(__file__).resolve().parent / "results" / "coupling_study"
                 / "stage2b_moment_gram"),
    )
    args = parser.parse_args()
    checks = validate(args.path)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Stage-2B artifact validation failed")


if __name__ == "__main__":
    main()
