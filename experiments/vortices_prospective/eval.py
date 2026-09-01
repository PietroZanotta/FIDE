"""Display a frozen prospective-vortices V6a validation result.

This command is read-only: it does not generate references or hidden banks,
optimize geometries, rerun validation, or modify the frozen result directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR.parent))

from saved_result_display import (  # noqa: E402
    number,
    percent,
    print_heading,
    print_table,
    print_uncertainty_note,
    source_label,
)


DEFAULT_RESULT = (
    SCRIPT_DIR
    / "outputs"
    / "prospective_v6a_2pct_1reference_fast_v1"
    / "results"
    / "validation_result.json"
)
METHODS = (("Law", "Law"), ("Tangent", "Tangent"), ("Full", "v6a"))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("validation result must be a JSON object")
    return value


def _display(path: Path, result: dict[str, Any]) -> int:
    points = result.get("points", [])
    if len(points) != 1:
        print(
            f"error: single-run evaluator expected one allowance, found {len(points)}",
            file=sys.stderr,
        )
        return 2
    point = points[0]
    methods = point.get("methods", {})
    missing = [key for _, key in METHODS if key not in methods]
    if missing:
        print(f"error: missing method results: {', '.join(missing)}", file=sys.stderr)
        return 2

    law_action = float(methods["Law"]["equal_reference_mean_full_action"])
    rows = []
    for label, key in METHODS:
        method = methods[key]
        distribution = method["pooled_full_distribution"]
        action = float(method["equal_reference_mean_full_action"])
        rows.append(
            (
                label,
                number(method["equal_reference_mean_risk"]),
                number(action),
                number(distribution["sd"]),
                number(distribution["se"]),
                percent(1.0 - action / law_action),
            )
        )

    allowance = 100.0 * float(point["allowance"])
    references = result.get("reference_ids", [])
    print_heading(
        "PROSPECTIVE VORTICES",
        (
            f"Saved V6a hidden validation — {allowance:g}% Law-relative allowance, "
            f"{len(references)} evaluation-reference seed"
            f"{'s' if len(references) != 1 else ''}"
        ),
        [source_label(path, REPOSITORY_ROOT)],
    )
    print_table(
        (
            "method",
            "risk",
            "action mean",
            "action SD",
            "action SE",
            "action Δ vs Law",
        ),
        rows,
    )

    paired = point["v6a_minus_law"]
    difference = paired["difference_full_minus_law"]
    print()
    print(f"paired Full - Law mean: {number(difference['mean'])}")
    print(
        "paired t 95% CI: "
        f"[{number(paired['paired_t_95_ci'][0])}, "
        f"{number(paired['paired_t_95_ci'][1])}]"
    )
    print(
        "paired bootstrap 95% CI: "
        f"[{number(paired['paired_bootstrap_95_ci'][0])}, "
        f"{number(paired['paired_bootstrap_95_ci'][1])}]"
    )
    print(f"Full lower-trial fraction: {percent(paired['fraction_full_lower'])}")
    conclusion = (
        "strict success"
        if bool(point.get("strict_success"))
        else "statistically unresolved"
    )
    print(
        "gates: "
        f"risk={'PASS' if point.get('all_reference_risk_pass') else 'FAIL'}, "
        f"numerical={'PASS' if point.get('numerical_certification_pass') else 'FAIL'}, "
        f"conclusion={conclusion}"
    )
    print_uncertainty_note(
        f"The saved result contains {int(difference['n'])} aligned trial pairs. "
        "A one-reference run cannot estimate variation across reference-flow seeds."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    try:
        result = _load(args.result)
        return _display(args.result, result)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
