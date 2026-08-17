#!/usr/bin/env python3
"""Freeze Phase-2F additive risk tolerance before any action calculation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from phase2_common import resolve, sha256, write_csv, write_json  # noqa: E402


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    analysis = SCRIPT_DIR.parent / "analysis"
    table_dir = analysis / "tables"
    summary_path = table_dir / "phase2f_validation_risk_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        risk_summary = json.load(handle)
    risk_rows = rows(table_dir / "validation_risk.csv")
    best_risk = float(risk_summary["best_validation_risk"])
    epsilon = float(risk_summary["best_bootstrap_risk_se"])
    selected = sorted(
        [
            row for row in risk_rows
            if row["fully_grid_usable"] == "True"
            and float(row["validation_mmd_risk"]) <= best_risk + epsilon
        ],
        key=lambda row: float(row["validation_mmd_risk"]),
    )
    frozen_rows = []
    for rank, row in enumerate(selected, start=1):
        frozen_rows.append({
            "near_optimal_rank": rank,
            "design_index": int(row["design_index"]), "design_id": row["design_id"],
            "style": row["style"], "validation_risk": float(row["validation_mmd_risk"]),
            "risk_difference_from_best": float(row["validation_mmd_risk"]) - best_risk,
            "bootstrap_risk_se": float(row["bootstrap_risk_se"]),
            "bootstrap_risk_ci_lower": float(row["bootstrap_risk_ci_lower"]),
            "bootstrap_risk_ci_upper": float(row["bootstrap_risk_ci_upper"]),
            **{
                key: float(row[key]) for key in row
                if key.startswith("s") and (key.endswith("_x_km") or key.endswith("_y_km"))
            },
        })
    write_csv(table_dir / "near_optimal_set.csv", frozen_rows)
    freeze = {
        "best_design_id": risk_summary["best_design_id"],
        "best_validation_risk": best_risk,
        "frozen_additive_epsilon": epsilon,
        "risk_ceiling": best_risk + epsilon,
        "epsilon_choice": "one bootstrap standard error of the point-estimate best layout",
        "rationale": (
            "chosen from the predeclared uncertainty-scale candidates before action; "
            "it retains a nontrivial geometrically diverse set without adopting the 458-layout "
            "paired-indistinguishability band whose allowed gap is 5.9 times the best risk"
        ),
        "near_optimal_layout_count": len(frozen_rows),
        "near_optimal_design_ids": [row["design_id"] for row in frozen_rows],
        "action_values_inspected_before_freeze": False,
        "final_test_artifact_loaded": False,
        "validation_risk_summary_sha256": sha256(summary_path),
        "validation_risk_table_sha256": sha256(table_dir / "validation_risk.csv"),
    }
    write_json(table_dir / "phase2f_risk_freeze.json", freeze)
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == "__main__":
    main()
