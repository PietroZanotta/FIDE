#!/usr/bin/env python3
"""Validate completed MFSI Stage-4B confirmatory artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage4b_fiber_design_confirmatory"
REQUIRED = (
    "stage4b_protocol.json", "config.json", "seed_provenance.json",
    "gradient_check.json", "forward_equivalence.json", "per_seed_metrics.csv",
    "paired_contrasts.json", "fiber_geometry.csv", "selected_subspaces.json",
    "REPORT.md",
)
SEEDS = list(range(426, 436))
METHODS = {"hand", "stop_grad", "full_grad"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def validate(path: Path):
    protocol = json.loads((path / "stage4b_protocol.json").read_text())
    config = json.loads((path / "config.json").read_text())
    provenance = json.loads((path / "seed_provenance.json").read_text())
    gradient = json.loads((path / "gradient_check.json").read_text())
    forward = json.loads((path / "forward_equivalence.json").read_text())
    contrasts = json.loads((path / "paired_contrasts.json").read_text())
    selections = json.loads((path / "selected_subspaces.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    metric_rows = list(csv.DictReader((path / "per_seed_metrics.csv").open()))
    geometry_rows = list(csv.DictReader((path / "fiber_geometry.csv").open()))
    by_seed_method = {
        (int(row["seed"]), row["method"]): row for row in metric_rows
    }
    primary = [
        float(by_seed_method[(seed, "full_grad")]["construction_objective"])
        - float(by_seed_method[(seed, "hand")]["construction_objective"])
        for seed in SEEDS
    ]
    mechanism = [
        float(by_seed_method[(seed, "full_grad")]["construction_objective"])
        - float(by_seed_method[(seed, "stop_grad")]["construction_objective"])
        for seed in SEEDS
    ]
    primary_recomputed = paired(primary)
    mechanism_recomputed = paired(mechanism)
    primary_saved = contrasts["full_grad_minus_hand"]
    mechanism_saved = contrasts["full_grad_minus_stop_grad"]
    allowed_steps = set(range(0, 41, 5))
    return {
        "required_artifacts": all((path / name).exists() for name in REQUIRED),
        "predeclared_seed_block": protocol["seeds"] == SEEDS,
        "scientific_replication_n_is_10": config["scientific_replication_n"] == 10,
        "fresh_disjoint_offsets": config["bank_offsets"] == {
            "adaptation": 111000, "selection": 112000,
            "evaluation": 113000, "gradient_validation": 114000,
        },
        "three_methods_and_30_metric_rows": (
            len(metric_rows) == 30
            and {row["method"] for row in metric_rows} == METHODS
        ),
        "three_geometry_contrasts_per_seed": len(geometry_rows) == 30,
        "forward_equivalence_passed": (
            forward["passed"]
            and forward["maximum_absolute_difference"] <= 1e-10
        ),
        "full_gradient_check_passed": gradient["passed"],
        "gradient_ablation_passed": gradient["gradient_ablation"]["passed"],
        "all_selections_frozen_before_q4": selections[
            "selection_frozen_before_evaluation_q4"
        ],
        "checkpoint_zero_and_cadence_preserved": all(
            item[method]["candidate_steps"] == list(range(0, 41, 5))
            and item[method]["selected_step"] in allowed_steps
            for item in selections["seeds"]
            for method in ("full_grad", "stop_grad")
        ),
        "paired_primary_reproduces": np.allclose(
            primary_recomputed,
            [primary_saved["mean_paired_difference"], primary_saved["paired_sd"],
             primary_saved["ci95_low"], primary_saved["ci95_high"]],
            rtol=1e-13, atol=1e-13,
        ),
        "paired_mechanism_reproduces": np.allclose(
            mechanism_recomputed,
            [mechanism_saved["mean_paired_difference"], mechanism_saved["paired_sd"],
             mechanism_saved["ci95_low"], mechanism_saved["ci95_high"]],
            rtol=1e-13, atol=1e-13,
        ),
        "structural_residuals_pass": (
            summary["checks"]["maximum_orthonormality_residual"] < 2e-10
            and summary["checks"]["maximum_endpoint_residual"] < 1e-10
            and summary["checks"]["maximum_nullspace_residual"] < 1e-10
        ),
        "bank_roles_disjoint": summary["checks"]["all_bank_roles_disjoint"],
        "no_q4_leakage": summary["checks"]["q4_used_for_adaptation_or_selection"] is False,
        "no_evaluation_leakage": summary["checks"]["evaluation_used_for_selection"] is False,
        "stage5_decision_matches_strong_rule": (
            summary["stage5_scientifically_justified"]
            == (primary_saved["success"] and mechanism_saved["success"])
        ),
        "recorded_hashes_match": (
            provenance["protocol_sha256"] == sha256(path / "stage4b_protocol.json")
            and provenance["driver_sha256"]
            == sha256(ROOT / "stage4b_fiber_design_confirmatory.py")
            and provenance["frozen_stage4_driver_sha256"]
            == sha256(ROOT / "stage4_fiber_design.py")
            and provenance["level2_source_sha256"]
            == sha256(ROOT / "level2_paper_study.py")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checks = validate(args.path)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Stage-4B artifact validation failed")


if __name__ == "__main__":
    main()
