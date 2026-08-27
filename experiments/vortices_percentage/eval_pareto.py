"""Display the saved corrected vortices validation Pareto results."""

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


DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "pareto" / "corrected_authoritative_pareto.json"
DEFAULT_SUMMARY = SCRIPT_DIR / "outputs" / "pareto" / "authoritative_run_summary.json"
DEFAULT_METHODS = SCRIPT_DIR / "outputs" / "pareto" / "pareto_methods_validation.csv"
DEFAULT_SELECTION = SCRIPT_DIR / "outputs" / "pareto" / "pareto_methods_selection.csv"
DEFAULT_TRIALS = SCRIPT_DIR / "outputs" / "pareto" / "validation_trial_summaries.csv"
EXPECTED_HASHES = {
    "result": "cdf84cd0e8277c8b3f89bf950d82a349f18972fff9986a5b1a217e213fac89aa",
    "summary": "4414ec8308bd1121ce1d3bea28f4a4bd3bfe68defd15e7ed4ef25edfb8017daa",
    "methods": "6c70e8abc7850adc06e0f1856a50fd26e926811b937c4d8e6b2939e8945edb10",
    "selection": "d40f8cc97a24c6ebf77f1834b62d93266e1670919e2beb527bf3c038712ee4ea",
    "trials": "cfa54f8efb0bc4f25dccacb2dd32b106d72016b5d08e59b8eb3e4312c71b0ed2",
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
        grouped[(float(row["allowance_percent"]), row["design"])].append(row)

    failures = []
    for name, path in (("result", args.result), ("summary", args.summary), ("methods", args.methods), ("selection", args.selection), ("trials", args.trials)):
        if _sha256(path) != EXPECTED_HASHES[name]:
            failures.append(f"{name} artifact hash mismatch")
    if len(result.get("rows", [])) != 6 or len(methods) != 18 or len(selection_rows) != 18 or len(trials) != 1536:
        failures.append("the saved Pareto tables are incomplete")
    summary = result.get("summary", {})
    if summary.get("status") != "PASS" or receipt.get("status") != "PASS" or not summary.get("selection_curve_nested"):
        failures.append("a saved Pareto summary is invalid")

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
        if len(values) != 64:
            failures.append(f"expected 64 trials for {key}")
            continue
        risk_values = [float(item["law_risk"]) for item in values]
        tangent_values = [float(item["tangent_action"]) for item in values]
        full_values = [float(item["full_action"]) for item in values]
        risk_mean = statistics.mean(risk_values)
        full_mean = statistics.mean(full_values)
        full_se = statistics.stdev(full_values) / math.sqrt(len(full_values))
        if not math.isclose(risk_mean, float(row["validation_R_mean"]), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"risk mean mismatch for {key}")
        if not math.isclose(full_mean, float(row["validation_full_action_mean"]), rel_tol=0.0, abs_tol=1e-10):
            failures.append(f"Full-action mean mismatch for {key}")
        table_rows.append((
            f"{key[0]:.1f}%", "validation", key[1].title(), number(risk_mean), number(statistics.stdev(risk_values)),
            number(statistics.mean(tangent_values)), number(statistics.stdev(tangent_values)),
            number(full_mean), number(statistics.stdev(full_values)), number(full_se),
            percent(float(row["validation_full_action_reduction_vs_law_percent"]) / 100.0),
        ))
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    print_heading(
        "VORTICES / DOUBLE-GYRE",
        "Saved selection and validation Pareto results",
        [source_label(path, REPOSITORY_ROOT) for path in (args.result, args.selection, args.methods, args.trials)],
    )
    print_table(
        ("p", "stage", "method", "risk", "risk SD", "Tangent A", "Tangent SD", "Full A", "Full SD", "Full SE", "Full Δ vs Law"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation SD and SE are computed directly from 64 saved independent trials per allowance and method."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
