"""Display the saved toy-example validation result without recomputation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, sample_sd_from_se, source_label


DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "run" / "result.json"
DEFAULT_SELECTION = SCRIPT_DIR / "outputs" / "run" / "result.candidate_summary.csv"
EXPECTED_SELECTION_SHA256 = "abb1a04a128892f0d7e19fe67a98be6f82d49fbc7e0d185eda12fce58141a56f"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    with args.selection.open(newline="", encoding="utf-8") as stream:
        selection = {row["design"]: row for row in csv.DictReader(stream)}
    validation = result.get("validation", {})
    certificates = result.get("selection_certificates", {})

    failures = []
    if _sha256(args.selection) != EXPECTED_SELECTION_SHA256:
        failures.append("saved selection summary hash mismatch")
    for method in ("law", "tangent", "full"):
        if method not in validation or float(validation[method].get("valid_fraction", 0.0)) < 0.95:
            failures.append(f"saved {method} validation is incomplete")
        if not certificates.get(method, {}).get("certified"):
            failures.append(f"saved {method} selection certificate is invalid")
        if method not in selection:
            failures.append(f"saved {method} selection values are missing")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    law_full = float(validation["law"]["full_action"]["mean"])
    table_rows = []
    for method in ("law", "tangent", "full"):
        selected = selection[method]
        table_rows.append((
            "selection", method.title(), number(selected["finite_risk_selection"]), MISSING,
            number(selected["tangent_action_selection"]), MISSING,
            number(selected["full_action_selection"]), MISSING, MISSING,
            percent(1.0 - float(selected["full_action_selection"]) / float(selection["law"]["full_action_selection"])),
        ))
        block = validation[method]
        risk = block["law_risk"]
        tangent = block["tangent_action"]
        full = block["full_action"]
        table_rows.append((
            "validation", method.title(),
            number(risk["mean"]),
            number(sample_sd_from_se(risk["se"], risk["n"])),
            number(tangent["mean"]),
            number(sample_sd_from_se(tangent["se"], tangent["n"])),
            number(full["mean"]),
            number(sample_sd_from_se(full["se"], full["n"])),
            number(full["se"]),
            percent(1.0 - float(full["mean"]) / law_full),
        ))

    print_heading(
        "TOY EXAMPLE",
        "Saved selection and independent-validation result",
        [source_label(args.result, REPOSITORY_ROOT), source_label(args.selection, REPOSITORY_ROOT)],
    )
    print_table(
        ("stage", "method", "risk", "risk SD", "Tangent A", "Tangent SD", "Full A", "Full SD", "Full SE", "Full Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation SD is reconstructed from the saved ordinary SE and n=128 independent trials (SD = SE√n)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
