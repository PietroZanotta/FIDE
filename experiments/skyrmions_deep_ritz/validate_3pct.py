"""Independent saved-artifact gate between the 3% experiment and Pareto."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent


def validate(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    invalidation_path = result_path.parent / "anchor_invalidation.json"
    invalidation = (
        json.loads(invalidation_path.read_text(encoding="utf-8"))
        if invalidation_path.exists() else {}
    )
    law = result.get("law_anchor", {})
    full = result.get("full_3_percent", {})
    validation = result.get("validation", {})
    search_cfg = result.get("config", {}).get("search", {})
    allowance = float(search_cfg.get("risk_allowance_percent", 3.0))
    minimum_reduction = float(search_cfg.get("minimum_action_reduction_fraction", 0.01))
    anchor = float(law.get("risk", math.nan))
    risk = float(full.get("selection_risk", math.nan))
    action_reduction = float(full.get("action_reduction_vs_law", math.nan))
    validation_action_reduction = float(
        result.get("validation_contrast", {}).get("full_vs_law_action_reduction", math.nan)
    )
    bank_manifest = result.get("bank_manifest", [])
    identifiers = [row.get("identifier") for row in bank_manifest]
    access_log = result.get("bank_access_log", [])

    checks = {
        "authoritative_profile": (
            not bool(result.get("smoke"))
            and result.get("execution_profile") == "authoritative"
        ),
        "law_anchor_positive_finite": math.isfinite(anchor) and anchor > 0.0,
        "law_anchor_not_invalidated": not (
            invalidation.get("unresolved") is True
            and invalidation.get("invalidated_config_hash") == result.get("config_hash")
        ),
        "selection_risk_within_3pct": (
            math.isfinite(risk) and math.isfinite(anchor)
            and risk <= (1.0 + allowance / 100.0) * anchor + 1.0e-12
        ),
        "full_selection_certificate": bool(full.get("valid")) and bool(full.get("certificate", {}).get("valid")),
        "meaningful_action_reduction": math.isfinite(action_reduction) and action_reduction >= minimum_reduction,
        "law_independent_validation": bool(validation.get("law", {}).get("valid")),
        "full_independent_validation": bool(validation.get("full", {}).get("valid")),
        "independent_action_reduction_reproduced": (
            math.isfinite(validation_action_reduction)
            and validation_action_reduction >= minimum_reduction
        ),
        "frozen_endpoint_only_reference": (
            result.get("reference", {}).get("endpoint_only") is True
            and result.get("reference", {}).get("frozen_for_all_designs") is True
        ),
        "bank_identifiers_unique": len(identifiers) == len(set(identifiers)) and None not in identifiers,
        "no_validation_bank_used_by_selection": not any(
            row.get("consumer") == "selection" and row.get("role") == "validation"
            for row in access_log
        ),
        "law_checkpoint_present": (result_path.parent / str(law.get("checkpoint", "missing"))).is_file(),
        "full_checkpoint_present": (result_path.parent / str(full.get("checkpoint", "missing"))).is_file(),
        "no_tangent_hidden_decomposition": result.get("forbidden_decompositions_computed") is False,
    }
    passed = all(checks.values()) and bool(result.get("milestone_success"))
    return {
        "schema_version": 1,
        "kind": "skyrmion_authoritative_3pct_validation_gate",
        "result": str(result_path),
        "allowance_percent": allowance,
        "checks": checks,
        "passed": passed,
        "pareto_unlocked": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the authoritative skyrmion 3% milestone")
    parser.add_argument("result", nargs="?", type=Path, default=SCRIPT_DIR / "outputs" / "run" / "result.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.result)
    output = args.output or args.result.parent / "three_percent_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"pareto_unlocked={report['pareto_unlocked']}")
    print(f"report={output}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
