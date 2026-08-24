from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a saved skyrmion Deep Ritz result")
    parser.add_argument("result", nargs="?", type=Path, default=SCRIPT_DIR / "outputs" / "run" / "result.json")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    failures: list[str] = []
    full = result.get("full_3_percent", {})
    cert = full.get("certificate", {})
    if result.get("comparisons") != ["Law", "Full Deep Ritz"]:
        failures.append("unexpected design comparison set")
    if result.get("forbidden_decompositions_computed") is not False:
        failures.append("Tangent/Hidden decomposition must not be computed")
    if not full.get("valid") or not cert.get("valid"):
        failures.append("Full winner is not certified")
    if float(full.get("selection_risk", math.inf)) > float(full.get("risk_limit", -math.inf)):
        failures.append("Full winner exceeds declared risk limit")
    for label in ("law", "full"):
        if not result.get("validation", {}).get(label, {}).get("valid"):
            failures.append(f"{label} independent validation failed")
    print(f"experiment: {result.get('experiment')}")
    print(f"smoke: {result.get('smoke')}")
    print(f"milestone_success: {result.get('milestone_success')}")
    print(f"selection risk/action: {full.get('selection_risk')} / {full.get('selection_action')}")
    print(
        "certificates: weak={maximum_weak_residual} energy={maximum_energy_residual} "
        "gauge={maximum_gauge_residual} moment-rate={maximum_moment_rate_residual}".format(**cert)
    )
    if not result.get("smoke") and not result.get("milestone_success"):
        failures.append("authoritative 3% milestone did not pass")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

