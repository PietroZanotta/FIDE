#!/usr/bin/env python3
"""Fail-closed V2.1 preflight that preserves all historical authorities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from mfsi.poisson_tesseract import is_tesseract_poisson_available

from v2_1_contract import CONFIG, load_resolved_config, sha256_file


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "VORTICES_V2_1_FREEZE_MANIFEST.json"
OLD_MANIFEST = HERE / "VORTICES_V2_FREEZE_MANIFEST.json"
REFERENCE_SUMMARY = HERE / "outputs" / "prospective_v2" / "references" / "reference_stage_summary.json"
DEVELOPMENT_AUDIT = HERE / "outputs" / "development_v2_failure_feasibility_first" / "exact_l_r_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_hash_map(mapping: dict[str, str], label: str) -> dict[str, Any]:
    for relative, expected in mapping.items():
        path = REPO / relative
        require(path.is_file(), f"{label} missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected, f"{label} hash mismatch: {relative}: {actual} != {expected}")
    return {"status": "PASS", "files": len(mapping)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    config, overlay = load_resolved_config(require_frozen=not args.allow_draft)
    checks: dict[str, Any] = {}

    old = load_json(OLD_MANIFEST)
    checks["old_v2_frozen_files"] = check_hash_map(old["frozen_files"], "old V2 frozen file")
    checks["shared_dependencies"] = check_hash_map(old["shared_dependencies"], "shared dependency")
    checks["toy_unchanged"] = check_hash_map(old["toy_immutability"], "Toy immutable")
    checks["v1_unchanged"] = check_hash_map(old["v1_immutability"], "V1 immutable")

    failure = HERE / "outputs" / "prospective_v2" / "selection" / "selection_failure.json"
    old_bank = HERE / "outputs" / "prospective_v2" / "selection" / "shared_selection_bank.npz"
    require(sha256_file(failure) == "a80df34da49699cf2bcf4b9091cff0af356b4fdd9ea41480ce3b6497d6932598", "failed V2 authority changed")
    require(sha256_file(old_bank) == "1096a255beffa781ee5a9bec881a2778b11f3bf5b8674389d7120180f5280d3b", "failed V2 bank changed")
    checks["failed_v2_preserved"] = {"status": "PASS"}

    reference_summary = load_json(REFERENCE_SUMMARY)
    require(reference_summary["reference_stage_complete"], "reference stage is incomplete")
    require(reference_summary["common_physical_bandwidth"] == 0.058816544123815116, "common bandwidth changed")
    require(len(reference_summary["references"]) == 3, "reference count changed")
    for row in reference_summary["references"]:
        require(row["qualification"] == "PASS", f"reference {row['training_seed']} is unqualified")
        require(sha256_file(Path(row["checkpoint"])) == row["checkpoint_sha256"], "reference checkpoint hash mismatch")
        require(sha256_file(Path(row["rollout_bank"])) == row["rollout_bank_sha256"], "reference rollout hash mismatch")
        require(sha256_file(Path(row["qualification_receipt"])) == row["qualification_receipt_sha256"], "qualification receipt hash mismatch")
    checks["references_and_bandwidth"] = {"status": "PASS", "references": 3}

    require(is_tesseract_poisson_available(), "native sparse Tesseract backend unavailable")
    checks["native_sparse_backend"] = {"status": "PASS"}
    require(config["optimization"]["full"]["selection_order"] == "exact_L_and_R_then_feasible_proxy_rank", "feasibility-first override missing")
    require(overlay["two_digit_randomness"]["policy"].startswith("all newly chosen"), "two-digit randomness policy missing")
    checks["v2_1_contract"] = {"status": "PASS", "config_status": overlay["status"]}

    require(DEVELOPMENT_AUDIT.is_file(), "development exact-L/R audit is incomplete")
    development = load_json(DEVELOPMENT_AUDIT)
    require(development["status"] == "EXACT_L_R_COMPLETE", "development audit status mismatch")
    require(development["candidate_count"] == 166, "development candidate count mismatch")
    require(development["feasible_counts"]["0.5"] > 0, "feasibility-first mechanism not demonstrated")
    checks["development_mechanism_evidence"] = {
        "status": "PASS",
        "data_role": development["data_role"],
        "feasible_counts": development["feasible_counts"],
        "sha256": sha256_file(DEVELOPMENT_AUDIT),
    }

    validation_paths = [
        HERE / "outputs" / "prospective_v2" / "validation" / "shared_validation_bank.npz",
        HERE / "outputs" / "prospective_v2_1" / "stress_test" / "shared_stress_test_bank.npz",
        HERE / "outputs" / "prospective_v2_1" / "validation" / "shared_validation_bank.npz",
    ]
    require(not any(path.exists() for path in validation_paths), "a forbidden pre-selection stress/validation bank exists")
    checks["no_stress_or_validation_bank"] = {"status": "PASS"}

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    checks["repository_head"] = {"status": "PASS", "head": head}
    if not args.allow_draft:
        manifest = load_json(MANIFEST)
        require(manifest["status"] == "FROZEN_PROSPECTIVE_BEFORE_V2_1_SELECTION_BANK", "manifest status mismatch")
        checks["v2_1_frozen_files"] = check_hash_map(manifest["frozen_files"], "V2.1 frozen file")
        prospective = HERE / "outputs" / "prospective_v2_1"
        existing = [path for path in prospective.rglob("*") if path.is_file()] if prospective.exists() else []
        require(not existing, f"V2.1 prospective outputs already exist: {existing[:5]}")
        checks["no_v2_1_prospective_output"] = {"status": "PASS"}
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        raise SystemExit(1)
