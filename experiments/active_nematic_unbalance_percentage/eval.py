"""Display the saved primary active-nematic result without recomputation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import number, percent, print_heading, print_table, print_uncertainty_note, source_label


DEFAULT_RESULT = SCRIPT_DIR / "published_results.json"
PRIMARY_ALLOWANCE = 3.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_se(values: list[float]) -> dict[str, float | int] | None:
    """Compatibility helper retained for experiment tests and saved audits."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "se": float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(len(array)),
    }


def _selected_full_audit(result: dict[str, Any], design: str) -> dict[str, Any] | None:
    """Return the exact Full audit matching a selected saved geometry, if present."""
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row for row in result["selection_candidates"]["full"]
        if np.allclose(np.asarray(row["eta"], dtype=np.float64), eta, rtol=0.0, atol=1.0e-8)
    ]
    return min(matches, key=lambda row: float(row["audit"]["value"])) if matches else None


def _raw_result_path(source: Path, payload: dict[str, Any]) -> Path:
    """Resolve a raw one-point Pareto summary to its complete saved receipt."""
    if "validation_designs" in payload and "selection_candidates" in payload:
        return source
    rows = payload.get("rows", [])
    if len(rows) == 1:
        row = rows[0]
    else:
        row = next(
            (
                candidate
                for candidate in rows
                if float(candidate.get("allowance_percent", math.nan))
                == PRIMARY_ALLOWANCE
            ),
            None,
        )
        if row is None:
            raise ValueError(
                "raw Pareto summary must contain one row or a 3% primary row"
            )
    declared = Path(row.get("result", "")).expanduser()
    if declared.is_file():
        return declared.resolve()
    allowance = float(row["allowance_percent"])
    tag = f"risk_{f'{allowance:g}'.replace('.', 'p')}pct"
    local = source.parent / tag / "result.json"
    if not local.is_file():
        raise FileNotFoundError(f"missing complete saved receipt {local}")
    return local.resolve()


def _raw_selection_candidate(
    result: dict[str, Any], design: str
) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row
        for row in result["selection_candidates"]["full"]
        if np.allclose(
            np.asarray(row["eta"], dtype=np.float64),
            eta,
            rtol=0.0,
            atol=1.0e-8,
        )
    ]
    if not matches:
        raise ValueError(f"missing exact Full audit for selected {design} geometry")
    return min(matches, key=lambda row: float(row["audit"]["value"]))


def _view_action_means(validation: dict[str, Any]) -> list[float]:
    return [
        float(view["summary"]["metrics"]["full_unbalanced_action_total"]["mean"])
        for view in validation["views"]
    ]


def _display_raw_result(source: Path, result_path: Path) -> int:
    result = _load(result_path)
    if not result.get("selection_certified"):
        print("error: raw saved selection is not certified", file=sys.stderr)
        return 2
    if result.get("validation_designs") is None:
        print("error: raw saved result has no validation evaluation", file=sys.stderr)
        return 2

    allowance = float(result["allowance_percent"])
    methods = (
        ("Law", "law"),
        ("Tangent", "tangent"),
        ("Full", "unbalanced_full"),
    )
    selected = {
        label: _raw_selection_candidate(result, design)
        for label, design in methods
    }
    selection_law = float(selected["Law"]["audit"]["value"])
    validation_law = float(
        result["validation_designs"]["law"]["physical_view_action"]["mean"]
    )

    rows = []
    for label, _ in methods:
        candidate = selected[label]
        action = float(candidate["audit"]["value"])
        view_values = [float(value) for value in candidate["audit"]["view_values"]]
        rows.append(
            (
                "selection",
                label,
                number(candidate["law_screen"]["value"]),
                number(action),
                number(statistics.stdev(view_values) if len(view_values) > 1 else 0.0),
                number(None),
                percent(1.0 - action / selection_law),
            )
        )
    for label, design in methods:
        validation = result["validation_designs"][design]
        physical = validation["physical_view_action"]
        action = float(physical["mean"])
        view_values = _view_action_means(validation)
        risk = validation["summary"]["metrics"]["law_risk_total"]["mean"]
        rows.append(
            (
                "validation",
                label,
                number(risk),
                number(action),
                number(statistics.stdev(view_values) if len(view_values) > 1 else 0.0),
                number(physical["se_across_views"]),
                percent(1.0 - action / validation_law),
            )
        )

    reference_seeds = result["config"]["reference_training"].get("seeds", [])
    sources = [source_label(source, REPOSITORY_ROOT)]
    if result_path.resolve() != source.resolve():
        sources.append(source_label(result_path, REPOSITORY_ROOT))
    print_heading(
        "ACTIVE NEMATIC",
        (
            "Saved selection and validation result — "
            f"{allowance:g}% Law-relative allowance, "
            f"{len(reference_seeds)} reference seed"
            f"{'s' if len(reference_seeds) != 1 else ''}"
        ),
        sources,
    )
    print_table(
        (
            "stage",
            "method",
            "risk",
            "action",
            "action SD",
            "action SE",
            "action Δ vs Law",
        ),
        rows,
    )
    print_uncertainty_note(
        "Selection action is the robust maximum across physical/reference "
        "views; selection SD is the sample SD of those view actions and has "
        "no sampling SE. Validation action is the mean across physical folds, "
        "SD is the sample SD of fold means, and SE is the predeclared "
        "leave-one-physical-fold-out jackknife SE. A one-reference-seed run "
        "does not estimate reference-seed uncertainty."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    result = _load(args.result)
    if "pareto_rows" not in result:
        try:
            result_path = _raw_result_path(args.result, result)
            return _display_raw_result(args.result, result_path)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    rows = result.get("pareto_rows", [])
    primary = next(
        (row for row in rows if float(row["allowance_percent"]) == PRIMARY_ALLOWANCE),
        None,
    )
    views = result.get("primary_validation_view_actions", {})

    failures = [name for name, passed in result.get("checks", {}).items() if not passed]
    if result.get("status") != "PASS":
        failures.append("published result status is not valid")
    if primary is None:
        failures.append("the saved 3% row is missing")
    for method in ("Law", "Tangent", "Full"):
        values = views.get(method, [])
        if len(values) != 12:
            failures.append(f"expected 12 saved {method} view actions")
    if failures:
        print("error: " + "; ".join(failures), file=sys.stderr)
        return 2

    law_action = float(primary["validation_law_action"])
    law_se = float(rows[0]["validation_full_action_view_se"])
    selection_records = (
        ("selection", "Law", None, primary["selection_law_full_action"], None, None, 0.0),
        (
            "selection", "Tangent", None,
            primary["selection_tangent_geometry_full_action"], None, None,
            1.0 - float(primary["selection_tangent_geometry_full_action"]) / float(primary["selection_law_full_action"]),
        ),
        (
            "selection", "Full", primary["selection_full_risk"],
            primary["selection_full_action"], None, None,
            primary["selection_full_vs_law_reduction"],
        ),
    )
    validation_records = (
        (
            "validation", "Law",
            primary["validation_law_risk"],
            primary["validation_law_action"],
            statistics.stdev(views["Law"]),
            law_se,
            0.0,
        ),
        (
            "validation", "Tangent",
            primary["validation_tangent_risk"],
            primary["validation_tangent_action"],
            statistics.stdev(views["Tangent"]),
            primary["validation_tangent_action_view_se"],
            1.0 - float(primary["validation_tangent_action"]) / law_action,
        ),
        (
            "validation", "Full",
            primary["validation_full_risk"],
            primary["validation_full_action"],
            statistics.stdev(views["Full"]),
            primary["validation_full_action_view_se"],
            primary["validation_full_vs_law_reduction"],
        ),
    )

    print_heading(
        "ACTIVE NEMATIC",
        "Saved selection and validation result — 3% Law-relative allowance",
        [source_label(args.result, REPOSITORY_ROOT)],
    )
    print_table(
        ("stage", "method", "risk", "action mean", "action SD", "action SE", "action Δ vs Law"),
        [
            (stage, method, number(risk), number(mean), number(sd), number(se), percent(delta))
            for stage, method, risk, mean, sd, se, delta
            in selection_records + validation_records
        ],
    )
    print_uncertainty_note(
        "Selection has no sampling uncertainty. Validation SD is the sample SD "
        "across 12 physical/reference views; validation SE is the predeclared "
        "leave-one-physical-fold-out jackknife SE."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
