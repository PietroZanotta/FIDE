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
THREE_REFERENCE_RUN_DIR = (
    SCRIPT_DIR / "outputs" / "skyrmion_b1_galerkin_pareto_3references_v1"
)
DEFAULT_THREE_REFERENCE_PARETO = THREE_REFERENCE_RUN_DIR / "pareto.json"
DEFAULT_THREE_REFERENCE_OUTPUT_STEM = (
    SCRIPT_DIR / "figures" / "skyrmion_b1_galerkin_pareto_3references_v1"
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
    parser.add_argument(
        "--three-reference", action="store_true",
        help="plot the robust three-reference Pareto summary",
    )
    parser.add_argument(
        "--pareto", type=Path, default=DEFAULT_THREE_REFERENCE_PARETO,
        help="three-reference pareto.json artifact",
    )
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


def load_three_reference_records(pareto_path: Path) -> tuple[list[str], list[dict]]:
    """Translate the robust Pareto summary into per-allowance plot records."""
    payload = _load_json(pareto_path)
    flows = [str(flow) for flow in payload.get("flow_ids", [])]
    if len(flows) != 3 or len(set(flows)) != 3:
        raise ValueError("three-reference result must contain three distinct flows")
    records: list[dict] = []
    for row in payload.get("allowances", []):
        allowance = float(row["allowance_percent"])
        law = row.get("Law") or {}
        tangent = row.get("Tangent") or {}
        if tangent.get("status") != "CERTIFIED":
            raise ValueError(f"Tangent is not certified at {allowance:g}%")
        law_risk = law.get("risk_by_flow", {})
        tangent_risk = tangent.get("risk_by_flow", {})
        if set(law_risk) != set(flows) or set(tangent_risk) != set(flows):
            raise ValueError(f"incomplete per-flow risks at {allowance:g}%")
        risk_change = {
            flow: 100.0 * (float(tangent_risk[flow]) / float(law_risk[flow]) - 1.0)
            for flow in flows
        }
        for flow, value in risk_change.items():
            if value > allowance + 1e-8:
                raise ValueError(f"{flow} exceeds its risk gate at {allowance:g}%")
        records.append(
            {
                "allowance_percent": allowance,
                "tangent_action": float(tangent["tangent_action"]),
                "risk_change_percent": risk_change,
                "budget_used_percent": {
                    flow: 100.0 * risk_change[flow] / allowance for flow in flows
                },
                "full_certified": row.get("Full") is not None,
            }
        )
    if [record["allowance_percent"] for record in records] != [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        raise ValueError("expected the frozen 0.5%, 1%, 2%, 3%, 4%, 5% allowance grid")
    return flows, records


def make_three_reference_figure(flows: list[str], records: list[dict]) -> plt.Figure:
    """Plot action, realized per-flow risk, and gate utilization."""
    allowances = [record["allowance_percent"] for record in records]
    actions = [record["tangent_action"] for record in records]
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    figure.suptitle(
        "Skyrmions B1 Galerkin · robust Pareto across three reference flows",
        fontsize=14,
        fontweight="bold",
    )

    action_axis, risk_axis, budget_axis = axes
    action_axis.plot(
        allowances, actions, marker="o", linewidth=2.2, markersize=6,
        color="#6A3D9A", label="Tangent (certified)",
    )
    action_axis.set_title("Mean tangent action")
    action_axis.set_xlabel("Risk allowance p (%)")
    action_axis.set_ylabel("Equal-weight mean action")
    action_axis.legend(frameon=False, loc="upper right")
    action_axis.text(
        0.57, 0.58,
        "Full K=280:\nno certified point\nat all 6 allowances",
        transform=action_axis.transAxes,
        fontsize=9,
        color="#9C2F2F",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF3F3", "edgecolor": "#D9A4A4"},
    )

    risk_axis.plot(
        allowances, allowances, linestyle="--", linewidth=1.5,
        color="#555555", label="per-flow ceiling",
    )
    for flow, color in zip(flows, colors):
        risk_axis.plot(
            allowances,
            [record["risk_change_percent"][flow] for record in records],
            marker="o", linewidth=2.0, color=color, label=flow,
        )
    risk_axis.axhline(0.0, color="#999999", linewidth=0.8)
    risk_axis.set_title("Realized scientific-risk change")
    risk_axis.set_xlabel("Risk allowance p (%)")
    risk_axis.set_ylabel("Change from frozen Law (%)")
    risk_axis.legend(frameon=False, fontsize=8)

    budget_axis.axhline(
        100.0, linestyle="--", linewidth=1.5, color="#555555",
        label="protocol ceiling",
    )
    for flow, color in zip(flows, colors):
        budget_axis.plot(
            allowances,
            [record["budget_used_percent"][flow] for record in records],
            marker="o", linewidth=2.0, color=color, label=flow,
        )
    budget_axis.axhline(0.0, color="#999999", linewidth=0.8)
    budget_axis.set_title("Per-flow risk-budget use")
    budget_axis.set_xlabel("Risk allowance p (%)")
    budget_axis.set_ylabel("Signed budget use (%)\n(negative = below Law)")
    budget_axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(allowances)
        axis.grid(True, alpha=0.22, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5, -0.01,
        "Frozen selection artifacts only · ceilings are (1+p) times each flow's own Law risk · validation not accessed",
        ha="center", fontsize=8.5, color="#555555",
    )
    return figure


def save_three_reference_figure(
    flows: list[str], records: list[dict], output_stem: Path, *, dpi: int
) -> list[Path]:
    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure = make_three_reference_figure(flows, records)
    outputs = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    plt.close(figure)
    return outputs


def main() -> int:
    args = _parse_args()
    if args.three_reference:
        flows, records = load_three_reference_records(args.pareto)
        output_stem = (
            args.output_stem
            if args.output_stem != DEFAULT_OUTPUT_STEM
            else DEFAULT_THREE_REFERENCE_OUTPUT_STEM
        )
        outputs = save_three_reference_figure(
            flows, records, output_stem, dpi=args.dpi
        )
        for path in outputs:
            print(f"saved {path}")
        if args.show:
            figure = make_three_reference_figure(flows, records)
            plt.show()
            plt.close(figure)
        return 0
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
