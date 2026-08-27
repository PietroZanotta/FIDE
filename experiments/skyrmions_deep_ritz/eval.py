"""Display the saved primary Deep Ritz result without recomputation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, source_label


DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "pareto_authoritative" / "risk_3pct" / "result.json"
DEFAULT_TANGENT = SCRIPT_DIR / "outputs" / "pareto_authoritative" / "tangent_analysis" / "tangent_pareto.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--tangent", type=Path, default=DEFAULT_TANGENT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    tangent = json.loads(args.tangent.read_text(encoding="utf-8"))
    tan_row = next(row for row in tangent.get("rows", []) if float(row["allowance_percent"]) == 3.0)
    law = result.get("validation", {}).get("law", {})
    full = result.get("validation", {}).get("full", {})
    tan_law = tangent.get("law", {}).get("validation_certificate", {})
    full_primary = result.get("full_3_percent", {})

    failures = []
    if result.get("comparisons") != ["Law", "Full Deep Ritz"]:
        failures.append("unexpected primary comparison set")
    if result.get("forbidden_decompositions_computed") is not False:
        failures.append("the primary run unexpectedly computed a forbidden decomposition")
    if not result.get("milestone_success") or not full_primary.get("valid"):
        failures.append("the saved primary Full result is invalid")
    if float(full_primary.get("selection_risk", math.inf)) > float(full_primary.get("risk_limit", -math.inf)):
        failures.append("the primary Full result exceeds its saved risk limit")
    if not all(block.get("valid") for block in (law, full)) or not tan_row.get("valid"):
        failures.append("one or more saved validation records is invalid")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    law_full = float(law["action"])
    law_selection = result["law_anchor"]
    tan_law_selection = tangent["law"]["selection_certificate"]
    records = (
        ("selection", "Law", law_selection["risk"], tan_law_selection["action"], MISSING, tan_law_selection["action_standard_error"], law_selection["action"], MISSING, MISSING, 0.0),
        ("selection", "Tangent", tan_row["selection_risk"], tan_row["selection_tangent_action"], MISSING, tan_row["selection_certificate"]["action_standard_error"], MISSING, MISSING, MISSING, tan_row["selection_tangent_action_reduction_vs_law"]),
        ("selection", "Full", full_primary["selection_risk"], MISSING, MISSING, MISSING, full_primary["selection_action"], MISSING, full_primary["certificate"]["action_standard_error"], full_primary["action_reduction_vs_law"]),
        ("validation", "Law", law["risk"], tan_law["action"], MISSING, tan_law["action_standard_error"], law_full, MISSING, law["certificate"]["action_standard_error"], 0.0),
        ("validation", "Tangent", tan_row["validation_risk"], tan_row["validation_tangent_action"], MISSING, tan_row["validation_tangent_action_standard_error"], MISSING, MISSING, MISSING, tan_row["validation_tangent_action_reduction_vs_law"]),
        ("validation", "Full", full["risk"], MISSING, MISSING, MISSING, full["action"], MISSING, full["certificate"]["action_standard_error"], 1.0 - float(full["action"]) / law_full),
    )
    print_heading(
        "SKYRMIONS — DEEP RITZ",
        "Saved selection and validation result — 3% Law-relative allowance",
        [source_label(args.result, REPOSITORY_ROOT), source_label(args.tangent, REPOSITORY_ROOT)],
    )
    print_table(
        ("stage", "method", "risk", "Tangent A", "Tangent SD", "Tangent SE", "Full A", "Full SD", "Full SE", "objective Δ vs Law"),
        [
            (stage, method, number(risk), number(tan_a), str(tan_sd), number(tan_se), number(full_a), str(full_sd), number(full_se), percent(delta))
            for stage, method, risk, tan_a, tan_sd, tan_se, full_a, full_sd, full_se, delta in records
        ],
    )
    print_uncertainty_note(
        "The saved certificates contain Monte Carlo SEs but no sample counts or raw values; SD is therefore not identifiable (—) at either stage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
