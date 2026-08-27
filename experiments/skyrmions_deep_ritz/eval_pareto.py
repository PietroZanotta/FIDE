"""Display the tracked Deep Ritz Full and Tangent Pareto results."""

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


DEFAULT_FULL = SCRIPT_DIR / "outputs" / "pareto_authoritative" / "pareto.json"
DEFAULT_TANGENT = SCRIPT_DIR / "outputs" / "pareto_authoritative" / "tangent_analysis" / "tangent_pareto.json"
EXPECTED_HASHES = {
    "full": "b1a864f9b90c10891f2c4742199e4f07e5fcfe90765009cd89d36dc9d8816ea4",
    "tangent": "88e55fbc010a9b10ddfe9d4f42162788758f4c909ab7b1365832370132e56846",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--tangent", type=Path, default=DEFAULT_TANGENT)
    args = parser.parse_args()
    full = json.loads(args.full.read_text(encoding="utf-8"))
    tangent = json.loads(args.tangent.read_text(encoding="utf-8"))
    full_rows = full.get("rows", [])
    tangent_rows = tangent.get("rows", [])
    tangent_by_p = {float(row["allowance_percent"]): row for row in tangent_rows}

    failures = []
    if _sha256(args.full) != EXPECTED_HASHES["full"] or _sha256(args.tangent) != EXPECTED_HASHES["tangent"]:
        failures.append("a published artifact hash differs")
    if len(full_rows) != 6 or len(tangent_rows) != 6 or set(tangent_by_p) != {float(row["allowance_percent"]) for row in full_rows}:
        failures.append("the saved allowance grids are incomplete or different")
    if not full.get("certified") or not tangent.get("certified"):
        failures.append("a saved Pareto artifact is uncertified")
    if not all(row.get("valid") and row.get("milestone_success") for row in full_rows) or not all(row.get("valid") for row in tangent_rows):
        failures.append("one or more saved Pareto rows is invalid")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    law_tan = tangent["law"]["validation_certificate"]
    law_risk = tangent["law"]["validation_risk"]
    table_rows = []
    for full_row in full_rows:
        p_value = float(full_row["allowance_percent"])
        p = f"{p_value:.1f}%"
        tan = tangent_by_p[p_value]
        law_full_selection = float(full_row["selection_action"]) / (1.0 - float(full_row["action_reduction_vs_law"]))
        table_rows.extend((
            (p, "selection", "Law", number(full["frozen_law_risk"]), number(tangent["law"]["selection_certificate"]["action"]), MISSING, number(tangent["law"]["selection_certificate"]["action_standard_error"]), number(law_full_selection), MISSING, MISSING, percent(0.0)),
            (p, "selection", "Tangent", number(tan["selection_risk"]), number(tan["selection_tangent_action"]), MISSING, number(tan["selection_certificate"]["action_standard_error"]), MISSING, MISSING, MISSING, percent(tan["selection_tangent_action_reduction_vs_law"])),
            (p, "selection", "Full", number(full_row["selection_risk"]), MISSING, MISSING, MISSING, number(full_row["selection_action"]), MISSING, MISSING, percent(full_row["action_reduction_vs_law"])),
            (p, "validation", "Law", number(law_risk), number(law_tan["action"]), MISSING, number(law_tan["action_standard_error"]), number(full_row["validation_law_action"]), MISSING, MISSING, percent(0.0)),
            (p, "validation", "Tangent", number(tan["validation_risk"]), number(tan["validation_tangent_action"]), MISSING, number(tan["validation_tangent_action_standard_error"]), MISSING, MISSING, MISSING, percent(tan["validation_tangent_action_reduction_vs_law"])),
            (p, "validation", "Full", number(full_row["validation_risk"]), MISSING, MISSING, MISSING, number(full_row["validation_action"]), MISSING, number(full_row["validation_action_standard_error"]), percent(full_row["validation_action_reduction_vs_law"])),
        ))

    print_heading(
        "SKYRMIONS — DEEP RITZ",
        "Saved selection and validation Pareto results",
        [source_label(args.full, REPOSITORY_ROOT), source_label(args.tangent, REPOSITORY_ROOT)],
    )
    print_table(
        ("p", "stage", "method", "risk", "Tangent A", "Tangent SD", "Tangent SE", "Full A", "Full SD", "Full SE", "objective Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "The saved Pareto records contain Monte Carlo SEs but not the sample counts or trial values needed to recover SD (—)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
