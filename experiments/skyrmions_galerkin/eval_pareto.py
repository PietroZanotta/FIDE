"""Display saved B1 Galerkin Pareto results without rerunning the study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
THREE_REFERENCE_RUN_DIR = (
    SCRIPT_DIR / "outputs" / "skyrmion_b1_galerkin_pareto_3references_v1"
)
DEFAULT_THREE_REFERENCE_PARETO = THREE_REFERENCE_RUN_DIR / "pareto.json"
DEFAULT_THREE_REFERENCE_SELECTION = THREE_REFERENCE_RUN_DIR / "full_search" / "selection.json"
EXPECTED_THREE_REFERENCE_ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _close(left: object, right: object, *, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def evaluate_three_reference(
    pareto_path: Path, selection_path: Path
) -> tuple[dict, list[tuple[str, ...]]]:
    """Validate and format the robust three-reference saved Pareto result."""
    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    allowances = pareto.get("allowances", [])
    flows = tuple(pareto.get("flow_ids", []))
    errors: list[str] = []

    observed_allowances = tuple(float(row.get("allowance_percent", math.nan)) for row in allowances)
    if observed_allowances != EXPECTED_THREE_REFERENCE_ALLOWANCES:
        errors.append("the saved allowance grid is incomplete or out of order")
    if len(flows) != 3 or len(set(flows)) != 3:
        errors.append("the result does not contain three distinct reference flows")
    if pareto.get("risk_gate") != "separate for every flow":
        errors.append("the saved result does not declare separate per-flow risk gates")
    if pareto.get("objective") != "equal-weight mean action":
        errors.append("the saved result does not declare the equal-weight action objective")
    if pareto.get("validation_accessed") is not False:
        errors.append("validation-access isolation is not recorded")

    full_gap_count = 0
    table_rows: list[tuple[str, ...]] = []
    previous_tangent_action = math.inf
    for row in allowances:
        allowance = float(row["allowance_percent"])
        law = row.get("Law") or {}
        tangent = row.get("Tangent") or {}
        full = row.get("Full")
        law_risk = law.get("risk_by_flow", {})
        tangent_risk = tangent.get("risk_by_flow", {})
        ceilings = row.get("risk_ceiling_by_flow", {})
        per_flow_action = tangent.get("per_flow_action", {})
        if set(law_risk) != set(flows) or set(tangent_risk) != set(flows):
            errors.append(f"{allowance:g}% has incomplete per-flow risk records")
            continue
        if set(ceilings) != set(flows) or set(per_flow_action) != set(flows):
            errors.append(f"{allowance:g}% has incomplete gate or action records")
            continue
        if tangent.get("status") != "CERTIFIED":
            errors.append(f"{allowance:g}% Tangent is not certified")

        risk_changes = []
        budget_uses = []
        for flow in flows:
            expected_ceiling = float(law_risk[flow]) * (1.0 + allowance / 100.0)
            if not _close(ceilings[flow], expected_ceiling):
                errors.append(f"{allowance:g}% has an invalid {flow} risk ceiling")
            if float(tangent_risk[flow]) > float(ceilings[flow]) + 1e-12:
                errors.append(f"{allowance:g}% Tangent exceeds the {flow} risk ceiling")
            change = float(tangent_risk[flow]) / float(law_risk[flow]) - 1.0
            risk_changes.append(change)
            budget_uses.append(change / (allowance / 100.0))

        mean_risk = sum(float(tangent_risk[flow]) for flow in flows) / len(flows)
        mean_action = sum(float(per_flow_action[flow]) for flow in flows) / len(flows)
        tangent_action = float(tangent.get("tangent_action", math.nan))
        if not _close(tangent.get("mean_risk"), mean_risk):
            errors.append(f"{allowance:g}% Tangent mean risk is inconsistent")
        if not _close(tangent_action, mean_action):
            errors.append(f"{allowance:g}% Tangent mean action is inconsistent")
        if tangent_action > previous_tangent_action + 1e-12:
            errors.append("Tangent action is not nested/non-increasing with allowance")
        previous_tangent_action = tangent_action

        table_rows.append((
            f"{allowance:.1f}%", "Tangent", "CERTIFIED",
            number(mean_risk), percent(max(risk_changes)), percent(max(budget_uses)),
            number(tangent_action),
        ))
        if full is None:
            full_gap_count += 1
            table_rows.append((
                f"{allowance:.1f}%", "Full K280", "NO CERTIFIED POINT",
                MISSING, MISSING, MISSING, MISSING,
            ))
        else:
            table_rows.append((
                f"{allowance:.1f}%", "Full K280", str(full.get("status", "UNKNOWN")),
                number(full.get("mean_risk")), MISSING, MISSING,
                number(full.get("full_action")),
            ))

    selection_rows = selection.get("allowances", [])
    selection_gaps = sum(
        row.get("status") == "NO_CERTIFIED_FULL_POINT" for row in selection_rows
    )
    if len(selection_rows) != len(allowances) or selection_gaps != full_gap_count:
        errors.append("the Full selection receipt disagrees with the Pareto summary")
    if pareto.get("full_gap_count") != full_gap_count:
        errors.append("the declared Full gap count is inconsistent")
    expected_status = "COMPLETE_WITH_GAPS" if full_gap_count else "COMPLETE"
    if pareto.get("status") != expected_status:
        errors.append("the declared completion status is inconsistent")
    if errors:
        raise ValueError("; ".join(errors))
    return pareto, table_rows


def _main_three_reference(pareto_path: Path, selection_path: Path) -> int:
    try:
        pareto, table_rows = evaluate_three_reference(pareto_path, selection_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_heading(
        "SKYRMIONS — B1 GALERKIN K=280 — THREE REFERENCES",
        "Saved robust-selection Pareto results",
        [source_label(pareto_path, REPOSITORY_ROOT), source_label(selection_path, REPOSITORY_ROOT)],
    )
    print_table(
        ("p", "method", "status", "mean risk", "worst ΔR", "max budget used", "mean action"),
        table_rows,
    )
    print()
    print(
        f"result: {pareto['status']} — Tangent is certified at all "
        f"{len(pareto['allowances'])} allowances; Full K280 has "
        f"{pareto['full_gap_count']} certification gaps."
    )
    print_uncertainty_note(
        "This is deterministic post-processing of frozen selection artifacts; "
        "no validation data were accessed and no sampling uncertainty is reported."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--three-reference", action="store_true",
        help="evaluate the robust three-reference Pareto result instead of the official single-reference result",
    )
    parser.add_argument("--pareto", type=Path, default=DEFAULT_THREE_REFERENCE_PARETO)
    parser.add_argument(
        "--three-reference-selection", type=Path,
        default=DEFAULT_THREE_REFERENCE_SELECTION,
    )
    args = parser.parse_args()
    if args.three_reference:
        return _main_three_reference(args.pareto, args.three_reference_selection)
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
