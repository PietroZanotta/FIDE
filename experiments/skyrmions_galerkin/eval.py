"""Display the saved primary official B1 Galerkin result without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import MISSING, number, percent, print_heading, print_table, print_uncertainty_note, source_label
from skyrmions_galerkin import per_seed_pareto


RUN_DIR = SCRIPT_DIR / "outputs" / "official_b1_galerkin_pareto_v1"
DEFAULT_RESULT = RUN_DIR / "final_summary.json"
DEFAULT_SELECTION = RUN_DIR / "selection" / "cross_evaluation.json"
EXPECTED_SHA256 = "e7f38eeda7d1aefea1ed2bc701bc35b5926f3b8f504bc3a75df0392ee5ddd9d3"
EXPECTED_SELECTION_SHA256 = "4ccf6ff16a03f4f0e3169d2caabf6be7d1c15d230e8f0ea2e6e8e402901db86c"
PRIMARY_ALLOWANCE = 5.0


def _main_per_seed(seed_id: str, allowance: float | None) -> int:
    run_dir = per_seed_pareto.seed_output_root(seed_id)
    summary_path = run_dir / "evaluations" / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read {summary_path}: {exc}", file=sys.stderr)
        return 2
    rows = summary.get("rows", [])
    if allowance is not None:
        rows = [row for row in rows if float(row["allowance_percent"]) == allowance]
    expected_count = 1 if allowance is not None else 6
    failures = []
    if summary.get("seed_id") != seed_id or len(rows) != expected_count:
        failures.append("the saved per-risk evaluation grid is incomplete")
    if not all(row.get("classification") == "PASS" for row in rows):
        failures.append("one or more requested risk allowances failed")
    table_rows = []
    for row in rows:
        for method in ("Tangent", "Full"):
            value = row["methods"][method]
            table_rows.append((
                f"{float(row['allowance_percent']):.1f}%",
                method,
                "PASS" if value["certified"] else "FAIL",
                number(value["risk"]),
                number(value["risk_ceiling"]),
                percent(
                    None if value["risk_change_vs_law_percent"] is None
                    else float(value["risk_change_vs_law_percent"]) / 100.0
                ),
                number(value["action"]),
            ))
    scope = (
        "All six saved risk-allowance evaluations"
        if allowance is None else f"Saved {allowance:g}% risk-allowance evaluation"
    )
    print_heading(
        f"SKYRMIONS — B1 GALERKIN K=280 — {seed_id}",
        scope,
        [source_label(summary_path, REPOSITORY_ROOT)],
    )
    print_table(
        ("p", "method", "status", "risk", "ceiling", "ΔR vs Law", "action"),
        table_rows,
    )
    print_uncertainty_note(
        "Selection-only deterministic evaluation of frozen per-seed artifacts; validation was not accessed."
    )
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2
    return 0


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
    parser.add_argument("--seed", choices=per_seed_pareto.SUPPORTED_SEEDS)
    parser.add_argument("--allowance", type=float, choices=per_seed_pareto.engine.ALLOWANCES)
    args = parser.parse_args()
    if args.seed is not None:
        return _main_per_seed(args.seed, args.allowance)
    if args.allowance is not None:
        parser.error("--allowance requires --seed")
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
