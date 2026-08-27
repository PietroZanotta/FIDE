"""Display the saved primary official B1 Galerkin result without recomputation."""

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


DEFAULT_RESULT = SCRIPT_DIR / "published_official_b1_final_summary.json"
DEFAULT_SELECTION = SCRIPT_DIR / "published_official_b1_cross_evaluation.json"
EXPECTED_SHA256 = "e7f38eeda7d1aefea1ed2bc701bc35b5926f3b8f504bc3a75df0392ee5ddd9d3"
EXPECTED_SELECTION_SHA256 = "4ccf6ff16a03f4f0e3169d2caabf6be7d1c15d230e8f0ea2e6e8e402901db86c"
PRIMARY_ALLOWANCE = 5.0


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _row_values(row: dict) -> tuple:
    tangent_se = row.get("diagnostics", {}).get("tangent_fit", {}).get("action_standard_error")
    return (
        "validation", row["selected_by"],
        number(row["validation_risk"]),
        number(row["tangent_action"]),
        MISSING,
        number(tangent_se),
        number(row["full_audit_action"]),
        MISSING,
        number(row["action_standard_error"]),
        percent(row["full_reduction_vs_law"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = [row for row in result.get("validation_rows", []) if float(row["allowance_percent"]) == PRIMARY_ALLOWANCE]
    selection_rows = [row for row in selection.get("rows", []) if float(row["allowance_percent"]) == PRIMARY_ALLOWANCE]

    failures = []
    if _sha256(args.result) != EXPECTED_SHA256:
        failures.append("published final-summary hash mismatch")
    if _sha256(args.selection) != EXPECTED_SELECTION_SHA256:
        failures.append("published selection hash mismatch")
    if result.get("status") != "COMPLETE" or len(rows) != 3 or len(selection_rows) != 3:
        failures.append("the saved 5% result is incomplete")
    if not all(
        row.get("classification") == "PASS" and row.get("numerically_certified")
        and row.get("strict_p_validation_pass") and row.get("p_plus_5pp_validation_pass")
        for row in rows
    ):
        failures.append("one or more saved 5% validation rows is invalid")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    order = {"Law": 0, "Tangent": 1, "Full": 2}
    rows.sort(key=lambda row: order[row["selected_by"]])
    selection_rows.sort(key=lambda row: order[row["selected_by"]])
    selected_law_full = next(row["full_action"] for row in selection_rows if row["selected_by"] == "Law")
    table_rows = []
    for selected, validated in zip(selection_rows, rows):
        table_rows.append((
            "selection", selected["selected_by"], number(selected["risk"]),
            number(selected["tangent_action"]), MISSING, MISSING,
            number(selected["full_action"]), MISSING, MISSING,
            percent(1.0 - float(selected["full_action"]) / float(selected_law_full)),
        ))
        table_rows.append(_row_values(validated))
    print_heading(
        "SKYRMIONS — B1 GALERKIN K=280",
        "Saved selection and validation result — 5% Law-relative allowance",
        [source_label(args.result, REPOSITORY_ROOT), source_label(args.selection, REPOSITORY_ROOT)],
    )
    print_table(
        ("stage", "method", "risk", "Tangent A", "Tangent SD", "Tangent SE", "Full A", "Full SD", "Full SE", "Full Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation certificates report empirical audit-sample SEs but retain no sample counts or trial values, so validation SD is not identifiable (—)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
