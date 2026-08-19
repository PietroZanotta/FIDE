"""Summarize certified selection and held-out effects across reference seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _selected_full_audit(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row
        for row in result["selection_candidates"]["full_exact"]
        if np.allclose(np.asarray(row["eta"], dtype=np.float64), eta)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one exact full-action audit for design {design}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "run" / "manifest_position_polarity.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["runs"]:
        result_path = Path(entry["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        law = _selected_full_audit(result, "law")
        full = _selected_full_audit(result, "full")
        validation_law = result["validation"]["law"]["summary"]
        validation_full = result["validation"]["full"]["summary"]
        epsilon = float(result["risk_max"] - result["risk_star"])
        validation_law_risk = float(validation_law["law_risk"]["mean"])
        validation_full_risk = float(validation_full["law_risk"]["mean"])
        validation_law_action = float(validation_law["full_action"]["mean"])
        validation_full_action = float(validation_full["full_action"]["mean"])
        rows.append(
            {
                "reference_seed": int(result["reference_seed"]),
                "result": str(result_path),
                "selection_certified": bool(result["selection_certified"]),
                "selection_trials": int(
                    result["observation_banks"]["selection"]["trials"]
                ),
                "validation_trials": int(
                    result["observation_banks"]["validation"]["trials"]
                ),
                "epsilon_r": epsilon,
                "selection_law_design_risk": float(law["law"]["value"]),
                "selection_full_design_risk": float(full["law"]["value"]),
                "selection_law_design_action": float(law["action"]["value"]),
                "selection_full_design_action": float(full["action"]["value"]),
                "selection_full_to_law_action_ratio": float(
                    full["action"]["value"] / law["action"]["value"]
                ),
                "validation_law_design_risk": validation_law_risk,
                "validation_full_design_risk": validation_full_risk,
                "validation_risk_difference": (
                    validation_full_risk - validation_law_risk
                ),
                "validation_information_equivalent": bool(
                    validation_full_risk <= validation_law_risk + epsilon
                ),
                "validation_law_design_action": validation_law_action,
                "validation_full_design_action": validation_full_action,
                "validation_full_to_law_action_ratio": float(
                    validation_full_action / validation_law_action
                ),
                "validation_action_advantage": bool(
                    validation_full_action < validation_law_action
                ),
                "validation_law_valid_trials": int(validation_law["valid_trials"]),
                "validation_full_valid_trials": int(validation_full["valid_trials"]),
            }
        )

    ratios = np.asarray(
        [row["validation_full_to_law_action_ratio"] for row in rows],
        dtype=np.float64,
    )
    finite_ratios = ratios[np.isfinite(ratios)]
    payload = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "reference_seed_count": len(rows),
        "all_selection_certified": all(row["selection_certified"] for row in rows),
        "validation_information_equivalent_seeds": sum(
            row["validation_information_equivalent"] for row in rows
        ),
        "validation_action_advantage_seeds": sum(
            row["validation_action_advantage"] for row in rows
        ),
        "validation_joint_success_seeds": sum(
            row["validation_information_equivalent"]
            and row["validation_action_advantage"]
            for row in rows
        ),
        "validation_action_ratio_mean": (
            float(np.mean(finite_ratios)) if len(finite_ratios) else float("nan")
        ),
        "validation_action_ratio_median": (
            float(np.median(finite_ratios)) if len(finite_ratios) else float("nan")
        ),
        "rows": rows,
    }
    target = args.output or (
        args.manifest.parent / "audits" / "reference_seed_summary.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(target)


if __name__ == "__main__":
    main()
