"""Display the saved official B1 Galerkin validation Pareto results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, source_label


RUN_DIR = SCRIPT_DIR / "outputs" / "official_b1_galerkin_pareto_v1"
DEFAULT_FINAL = RUN_DIR / "final_summary.json"
DEFAULT_SELECTION = RUN_DIR / "selection" / "cross_evaluation.json"
EXPECTED_HASHES = {
    "final": "e7f38eeda7d1aefea1ed2bc701bc35b5926f3b8f504bc3a75df0392ee5ddd9d3",
    "selection": "4ccf6ff16a03f4f0e3169d2caabf6be7d1c15d230e8f0ea2e6e8e402901db86c",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    final = json.loads(args.final.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = final.get("validation_rows", [])
    selection_rows = selection.get("rows", [])

    failures = []
    if _sha256(args.final) != EXPECTED_HASHES["final"] or _sha256(args.selection) != EXPECTED_HASHES["selection"]:
        failures.append("a published artifact hash differs")
    if len(rows) != 18 or len(selection_rows) != 18 or not selection.get("passed"):
        failures.append("the saved selection or validation grid is incomplete")
    if not all(
        row.get("classification") == "PASS" and row.get("numerically_certified")
        and row.get("strict_p_validation_pass") and row.get("p_plus_5pp_validation_pass")
        for row in rows
    ):
        failures.append("one or more saved validation rows is invalid")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    order = {"Law": 0, "Tangent": 1, "Full": 2}
    rows.sort(key=lambda row: (float(row["allowance_percent"]), order[row["selected_by"]]))
    selected = {(float(row["allowance_percent"]), row["selected_by"]): row for row in selection_rows}
    table_rows = []
    for row in rows:
        key = (float(row["allowance_percent"]), row["selected_by"])
        selection_row = selected[key]
        law_selection = selected[(key[0], "Law")]
        table_rows.append((
            f"{key[0]:.1f}%", "selection", row["selected_by"],
            number(selection_row["risk"]), number(selection_row["tangent_action"]), MISSING, MISSING,
            number(selection_row["full_action"]), MISSING, MISSING,
            percent(1.0 - float(selection_row["full_action"]) / float(law_selection["full_action"])),
        ))
        tangent_se = row.get("diagnostics", {}).get("tangent_fit", {}).get("action_standard_error")
        table_rows.append((
            f"{float(row['allowance_percent']):.1f}%",
            "validation", row["selected_by"],
            number(row["validation_risk"]),
            number(row["tangent_action"]),
            MISSING,
            number(tangent_se),
            number(row["full_audit_action"]),
            MISSING,
            number(row["action_standard_error"]),
            percent(row["full_reduction_vs_law"]),
        ))

    print_heading(
        "SKYRMIONS — B1 GALERKIN K=280",
        "Saved selection and validation Pareto results",
        [source_label(args.final, REPOSITORY_ROOT), source_label(args.selection, REPOSITORY_ROOT)],
    )
    print_table(
        ("p", "stage", "method", "risk", "Tangent A", "Tangent SD", "Tangent SE", "Full A", "Full SD", "Full SE", "Full Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation certificates report empirical audit-sample SEs but retain no sample counts or trial values, so validation SD is not identifiable (—)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
