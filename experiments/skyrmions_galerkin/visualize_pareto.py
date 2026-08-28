"""Plot the official B1 Galerkin action and risk-allowance Pareto summary.

This is deterministic post-processing of frozen selection and validation JSON
artifacts. It does not simulate, optimize, or revalidate any geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from pareto_cost_risk_visualization import (  # noqa: E402
    make_cost_risk_figure,
    save_cost_risk_figure,
)
from percentage_pareto_visualization import METHOD_LABELS  # noqa: E402


RUN_DIR = SCRIPT_DIR / "outputs" / "official_b1_galerkin_pareto_v1"
DEFAULT_SELECTION = RUN_DIR / "selection" / "pareto_selection.json"
DEFAULT_VALIDATION = SCRIPT_DIR / "published_official_b1_final_summary.json"
DEFAULT_OUTPUT_STEM = (
    SCRIPT_DIR / "figures" / "skyrmion_galerkin_pareto_methods"
)
FIGURE_TITLE = "Skyrmions Galerkin · cost and risk use along the frontier"
VALIDATION_RISK_NOTE = (
    "validation: p + 5 pp · negative = below frozen Law"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _indexed_rows(rows: list[dict[str, Any]], *, source: str) -> dict[tuple[float, str], dict]:
    indexed: dict[tuple[float, str], dict] = {}
    for row in rows:
        method = str(row["selected_by"]).strip().lower()
        if method not in METHOD_LABELS:
            raise ValueError(f"unknown {source} method {row['selected_by']!r}")
        key = (float(row["allowance_percent"]), method)
        if key in indexed:
            raise ValueError(f"duplicate {source} row for {key}")
        indexed[key] = row
    return indexed


def load_method_records(selection_path: Path, validation_path: Path) -> list[dict]:
    """Translate official Galerkin artifacts into the shared plotting schema."""
    selection_payload = _load_json(selection_path)
    cross_evaluation = selection_payload.get("cross_evaluation", selection_payload)
    if cross_evaluation.get("common_metric") != "authoritative K280 Full action":
        raise ValueError("selection artifact does not use authoritative K280 Full action")
    if not bool(cross_evaluation.get("passed")):
        raise ValueError("selection cross-evaluation did not pass")

    validation_payload = _load_json(validation_path)
    validation_rows = validation_payload.get(
        "validation_rows", validation_payload.get("rows", [])
    )
    selection = _indexed_rows(cross_evaluation.get("rows", []), source="selection")
    validation = _indexed_rows(validation_rows, source="validation")
    if selection.keys() != validation.keys():
        missing_validation = sorted(selection.keys() - validation.keys())
        missing_selection = sorted(validation.keys() - selection.keys())
        raise ValueError(
            "selection/validation row mismatch: "
            f"missing validation={missing_validation}, missing selection={missing_selection}"
        )

    records: list[dict] = []
    for allowance, method in sorted(selection):
        selected = selection[(allowance, method)]
        validated = validation[(allowance, method)]
        if not bool(validated.get("numerically_certified")):
            raise ValueError(f"validation row is not certified for {(allowance, method)}")
        records.append(
            {
                "risk_allowance_percent": allowance,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "selection_full_action": float(selected["full_action"]),
                "selection_budget_used_percent": 100.0
                * float(selected["budget_used"]),
                "validation_full_action_mean": float(validated["full_audit_action"]),
                "validation_full_action_se": float(validated["action_standard_error"]),
                "validation_R_change_vs_law_percent": 100.0
                * float(validated["validation_risk_increase"]),
                "validation_budget_used_percent": 100.0
                * float(validated["validation_risk_increase"])
                / (allowance / 100.0 + 0.05),
            }
        )
    if len(records) != 18:
        raise ValueError(f"expected 18 Law/Tangent/Full records, found {len(records)}")
    return records


def make_figure(records: list[dict]) -> plt.Figure:
    return make_cost_risk_figure(
        records,
        title=FIGURE_TITLE,
        validation_risk_note=VALIDATION_RISK_NOTE,
        risk_limit_label="100% = protocol ceiling",
    )


def main() -> int:
    args = _parse_args()
    records = load_method_records(args.selection, args.validation)
    outputs = save_cost_risk_figure(
        records,
        args.output_stem,
        title=FIGURE_TITLE,
        validation_risk_note=VALIDATION_RISK_NOTE,
        risk_limit_label="100% = protocol ceiling",
        dpi=args.dpi,
    )
    for path in outputs:
        print(f"saved {path}")
    if args.show:
        figure = make_figure(records)
        plt.show()
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
