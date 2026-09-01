#!/usr/bin/env python3
"""Fast read-only verification of the saved prospective-vortices result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "outputs" / "prospective_reflected_single_seed_pareto_repaired"
EXPECTED_ALLOWANCES = [0.5, 1.0, 2.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run_dir: Path) -> dict[str, Any]:
    summary_path = HERE / "results" / "validation_summary.json"
    visual_path = HERE / "plots" / "visualization_manifest.json"
    summary = load_json(summary_path)
    visuals = load_json(visual_path)
    failures: list[str] = []

    if summary.get("status") != "VALIDATION_COMPLETE_1_OF_3_STRICT_POINTS":
        failures.append("unexpected saved-result status")
    if summary.get("validation_reference") != "E1":
        failures.append("saved result is not the fresh repaired E1 validation")
    points = summary.get("points", [])
    if [float(row["allowance_percent"]) for row in points] != EXPECTED_ALLOWANCES:
        failures.append("saved allowance set is not exactly 0.5%, 1%, and 2%")
    if [bool(row.get("strict_success")) for row in points] != [False, False, True]:
        failures.append("strict-success pattern is not expected [false, false, true]")
    if not all(bool(row.get("risk_pass")) for row in points):
        failures.append("at least one held-out point lacks risk certification")
    if not all(bool(row.get("numerical_certification_pass")) for row in points):
        failures.append("at least one held-out point lacks numerical certification")
    held_key = "held_out_E1"
    if not all(
        float(points[index][held_key]["paired_action_difference_95_ci"][0]) == 0.0
        and float(points[index][held_key]["paired_action_difference_95_ci"][1]) == 0.0
        for index in (0, 1)
    ):
        failures.append("0.5% and 1% are not certified as identical to Law")
    if not float(points[2][held_key]["paired_action_difference_95_ci"][1]) < 0.0:
        failures.append("2% paired action interval is not strictly below zero")

    raw = {
        "combined_frozen_manifest_sha256": run_dir / "results" / "combined_frozen_manifest.json",
        "validation_result_sha256": run_dir / "results" / "validation_result.json",
    }
    raw_present = all(path.exists() for path in raw.values())
    if raw_present:
        for key, path in raw.items():
            if sha256_file(path) != summary["source_artifacts"][key]:
                failures.append(f"raw source hash mismatch: {path}")

    if visuals.get("status") != "COMPLETE_HELD_OUT_E1_VISUALIZATION_SET":
        failures.append("visualization set is incomplete")
    if visuals.get("selection_state_changed") is not False:
        failures.append("visualization manifest does not certify read-only rendering")
    for row in visuals.get("artifacts", []):
        path = HERE.parent.parent / row["path"]
        if not path.exists():
            failures.append(f"missing visualization: {path}")
        elif sha256_file(path) != row["sha256"]:
            failures.append(f"visualization hash mismatch: {path}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "allowances_percent": EXPECTED_ALLOWANCES,
        "strict_success_points": sum(bool(row.get("strict_success")) for row in points),
        "raw_authority_present_and_verified": raw_present and not any(
            failure.startswith("raw source hash mismatch") for failure in failures
        ),
        "visual_artifacts_verified": len(visuals.get("artifacts", [])),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify(args.run_dir.expanduser().resolve())
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["status"])
        for failure in result["failures"]:
            print(f"- {failure}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
