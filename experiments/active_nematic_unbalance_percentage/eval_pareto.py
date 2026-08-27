"""Display the saved authoritative active-nematic Pareto results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, source_label


DEFAULT_RESULT = SCRIPT_DIR / "published_results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = result.get("pareto_rows", [])

    failures = []
    if result.get("status") != "PASS":
        failures.append("published status is not valid")
    if len(rows) != 6:
        failures.append(f"expected 6 allowances, found {len(rows)}")
    if not all(row.get("certified") for row in rows):
        failures.append("one or more Pareto rows is uncertified")
    if not result.get("checks", {}).get("selection_curve_nested"):
        failures.append("selection curve is not nested")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    table_rows = []
    for row in rows:
        p = f"{float(row['allowance_percent']):.1f}%"
        law = float(row["validation_law_action"])
        table_rows.extend(
            (
                (p, "selection", "Law", MISSING, number(row["selection_law_full_action"]), MISSING, MISSING, percent(0.0)),
                (p, "selection", "Tangent", MISSING, number(row["selection_tangent_geometry_full_action"]), MISSING, MISSING, percent(1.0 - float(row["selection_tangent_geometry_full_action"]) / float(row["selection_law_full_action"]))),
                (p, "selection", "Full", number(row["selection_full_risk"]), number(row["selection_full_action"]), MISSING, MISSING, percent(row["selection_full_vs_law_reduction"])),
                (p, "validation", "Law", number(row["validation_law_risk"]), number(law), MISSING, MISSING, percent(0.0)),
                (
                    p,
                    "validation", "Tangent",
                    number(row["validation_tangent_risk"]),
                    number(row["validation_tangent_action"]),
                    MISSING,
                    number(row["validation_tangent_action_view_se"]),
                    percent(row["validation_tangent_vs_law_reduction"]),
                ),
                (
                    p,
                    "validation", "Full",
                    number(row["validation_full_risk"]),
                    number(row["validation_full_action"]),
                    MISSING,
                    number(row["validation_full_action_view_se"]),
                    percent(row["validation_full_vs_law_reduction"]),
                ),
            )
        )

    print_heading(
        "ACTIVE NEMATIC",
        "Saved selection and validation Pareto results",
        [source_label(args.result, REPOSITORY_ROOT)],
    )
    print_table(
        ("p", "stage", "method", "risk", "action mean", "action SD", "action SE", "action Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation SE is the saved "
        "physical-fold jackknife SE; per-allowance view samples were not "
        "preserved, so validation SD is not identifiable (—)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
