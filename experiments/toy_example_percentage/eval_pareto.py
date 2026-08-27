"""Display the saved corrected toy-example validation Pareto results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, source_label


DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "pareto" / "corrected_nested_full_sweep.json"
DEFAULT_SUMMARY = SCRIPT_DIR / "outputs" / "pareto" / "authoritative_run_summary.json"
DEFAULT_METHODS = SCRIPT_DIR / "outputs" / "pareto" / "pareto_methods_validation.csv"
DEFAULT_SELECTION = SCRIPT_DIR / "outputs" / "pareto" / "pareto_methods_selection.csv"
DEFAULT_TRIALS = SCRIPT_DIR / "outputs" / "pareto" / "validation_trial_summaries.csv"
EXPECTED_HASHES = {
    "result": "114df72191c0b519e6e45cf7c574060a47ac6c64201eba7ed7432f2f11fc2c7e",
    "summary": "2e29a178e3850ccb35c067006fb565bcb4f5fb845740ab4b6d4e4859034b80db",
    "methods": "4f648a6c6478fa76c443b5fcdbfc1f8b487c625075fb9d0f521f045465349c46",
    "selection": "921d9edb24d10d220b018092705abec318677c9d310f583d53827ecea9824401",
    "trials": "7215f0c01230e6c7946be3fc5f2fe7d65e5233a20b8270e1b3d33713973ee6ed",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    receipt = json.loads(args.summary.read_text(encoding="utf-8"))
    methods = _csv(args.methods)
    selection_rows = _csv(args.selection)
    selection = {(float(row["risk_allowance_percent"]), row["method"]): row for row in selection_rows}
    trials = _csv(args.trials)
    grouped: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        grouped[(float(row["allowance_percent"]), row["method"])].append(row)

    failures = []
    for name, path in (("result", args.result), ("summary", args.summary), ("methods", args.methods), ("selection", args.selection), ("trials", args.trials)):
        if _sha256(path) != EXPECTED_HASHES[name]:
            failures.append(f"{name} artifact hash mismatch")
    if len(result.get("rows", [])) != 6 or len(methods) != 18 or len(selection_rows) != 18 or len(trials) != 2304:
        failures.append("the saved Pareto tables are incomplete")
    if result.get("summary", {}).get("status") != "PASS" or receipt.get("status") != "PASS":
        failures.append("a saved Pareto summary is invalid")
    if not result.get("summary", {}).get("selection_curve_nested"):
        failures.append("the saved selection curve is not nested")

    table_rows = []
    order = {"law": 0, "tangent": 1, "full": 2}
    for row in sorted(methods, key=lambda item: (float(item["risk_allowance_percent"]), order[item["method"]])):
        key = (float(row["risk_allowance_percent"]), row["method"])
        selected = selection.get(key)
        selected_law = selection.get((key[0], "law"))
        if selected is None or selected_law is None:
            failures.append(f"selection values are missing for {key}")
            continue
        table_rows.append((
            f"{key[0]:.1f}%", "selection", key[1].title(),
            number(selected["selection_R"]), MISSING,
            number(selected["selection_tangent_action"]), MISSING,
            number(selected["selection_full_action"]), MISSING, MISSING,
            percent(1.0 - float(selected["selection_full_action"]) / float(selected_law["selection_full_action"])),
        ))
        values = grouped.get(key, [])
        if len(values) != 128:
            failures.append(f"expected 128 trials for {key}")
            continue
        risk_values = [float(item["law_risk"]) for item in values if item["law_risk"]]
        full_values = [float(item["full_action"]) for item in values]
        risk_mean = float(row["validation_R_mean"])
        risk_sd = (
            statistics.stdev(risk_values)
            if len(risk_values) == 128
            else float(row["validation_R_se"]) * math.sqrt(128)
        )
        full_mean = statistics.mean(full_values)
        full_se = statistics.stdev(full_values) / math.sqrt(len(full_values))
        if risk_values and not math.isclose(statistics.mean(risk_values), risk_mean, rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"risk mean mismatch for {key}")
        if not math.isclose(full_mean, float(row["validation_full_action_mean"]), rel_tol=0.0, abs_tol=1e-10):
            failures.append(f"Full-action mean mismatch for {key}")
        table_rows.append((
            f"{key[0]:.1f}%", "validation", key[1].title(), number(risk_mean), number(risk_sd),
            MISSING, MISSING, number(full_mean), number(statistics.stdev(full_values)), number(full_se),
            percent(float(row["validation_full_action_reduction_vs_law_percent"]) / 100.0),
        ))
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    print_heading(
        "TOY EXAMPLE",
        "Saved selection and validation Pareto results",
        [source_label(path, REPOSITORY_ROOT) for path in (args.result, args.selection, args.methods, args.trials)],
    )
    print_table(
        ("p", "stage", "method", "risk", "risk SD", "Tangent A", "Tangent SD", "Full A", "Full SD", "Full SE", "Full Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation Full-action SD and SE come from 128 saved independent trials; risk SD is direct for Full and reconstructed as SE√n for Law/Tangent. Validation Tangent-action trials were not saved (—)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
