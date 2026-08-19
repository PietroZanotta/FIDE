"""Audit cosine-order convergence on frozen production failure controls."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.config import load_config
from experiments.ocean_drifters.action import _read_csv, _write_csv
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action_production import (
    OceanFullActionProduction,
)


def _representatives(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rank_failures = sorted(
        (
            row for row in rows
            if row["rank_sensitivity_valid"] == "False"
            and int(row["source_time_index"]) >= 10
        ),
        key=lambda row: float(row["maximum_relative_rank_action_change"]),
    )
    lower_failures = sorted(
        (row for row in rows if row["tangent_full_inequality_valid"] == "False"),
        key=lambda row: (
            float(row["tangent_action_density"])
            / float(row["full_action_density"])
        ),
    )
    selected = [
        ("median_rank_failure", rank_failures[len(rank_failures) // 2]),
        ("maximum_rank_failure", rank_failures[-1]),
        ("maximum_lower_bound_failure", lower_failures[-1]),
    ]
    output: list[dict[str, object]] = []
    for label, row in selected:
        source = int(row["source_time_index"])
        output.append({
            "case_label": label,
            "design_index": int(row["design_index"]),
            "source_time_index": source,
        })
        control = next(
            candidate for candidate in rows
            if candidate["design_id"] == "design_000218"
            and int(candidate["source_time_index"]) == source
        )
        output.append({
            "case_label": f"control_for_{label}",
            "design_index": int(control["design_index"]),
            "source_time_index": source,
        })
    return output


def main() -> None:
    cfg = load_config(EXPERIMENT_DIR / "config.json")
    experiment = OceanDriftersExperiment(cfg)
    production = OceanFullActionProduction(
        experiment,
        EXPERIMENT_DIR / "analysis",
        EXPERIMENT_DIR / "outputs/full_action_production",
    )
    source_rows = _read_csv(
        EXPERIMENT_DIR / "analysis/tables/full_action_production_time.csv"
    )
    cases = _representatives(source_rows)
    output: list[dict[str, object]] = []
    original_variational = dict(production.runner.variational_cfg)
    for source in sorted({int(case["source_time_index"]) for case in cases}):
        local_cases = [
            case for case in cases if int(case["source_time_index"]) == source
        ]
        designs = np.asarray([
            int(case["design_index"]) for case in local_cases
        ], dtype=int)
        production.runner.source_indices = np.asarray([source], dtype=int)
        production.runner.designs = designs
        points, dx, log_base, velocity = production.runner._reference_grid(
            production.resolution,
            source_indices=production.runner.source_indices,
            cache_namespace="full_action_production_reference",
        )
        systems = production.runner._systems_for_grid(
            production.resolution, points, dx, log_base, velocity
        )
        label_by_design = {
            int(case["design_index"]): str(case["case_label"])
            for case in local_cases
        }
        for maximum_mode in (3, 4, 5, 6, 7):
            production.runner.variational_cfg = {
                **original_variational,
                "maximum_mode": maximum_mode,
            }
            solved, _ = production.runner._solve_variational(systems, dx)
            for row in solved:
                output.append({
                    "case_label": label_by_design[int(row["design_index"])],
                    "design_index": int(row["design_index"]),
                    "design_id": row["design_id"],
                    "source_time_index": source,
                    "day": float(row["day"]),
                    "maximum_mode": maximum_mode,
                    "basis_size": int(row["basis_size"]),
                    "retained_rank": int(row["retained_rank"]),
                    "condition_proxy": float(row["condition_proxy"]),
                    "full_action_density": float(row["full_action_density"]),
                    "tangent_action_density": float(row["tangent_action_density"]),
                    "maximum_relative_rank_action_change": float(
                        row["maximum_relative_rank_action_change"]
                    ),
                    "rank_sensitivity_valid": bool(row["rank_sensitivity_valid"]),
                    "tangent_full_inequality_valid": bool(
                        row["tangent_full_inequality_valid"]
                    ),
                    "solver_success": bool(row["solver_success"]),
                    "density_modified": False,
                    "operator_floor": 0.0,
                    "final_test_accessed": False,
                })
    _write_csv(
        EXPERIMENT_DIR
        / "analysis/tables/full_action_rank_repair_mode_sweep.csv",
        output,
    )
    print(f"wrote {len(output)} frozen representative mode audits")


if __name__ == "__main__":
    main()
