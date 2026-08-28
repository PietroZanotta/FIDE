"""Plot active-nematic validation cost components and Pareto risk usage.

The saved selection receipts contain common Full-action totals, while the
held-out validation receipts additionally retain the physical move/reaction
decomposition. This script only reads those authoritative receipts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from pareto_cost_risk_visualization import (  # noqa: E402
    INK,
    MUTED,
    PANEL_BACKGROUND,
    paper_style,
)
from percentage_pareto_visualization import (  # noqa: E402
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
)


DEFAULT_PARETO_DIR = SCRIPT_DIR / "outputs" / "pareto_robust"
DEFAULT_OUTPUT_STEM = (
    SCRIPT_DIR / "figures" / "active_nematic_pareto_components"
)
DESIGN_KEYS = {"law": "law", "tangent": "tangent", "full": "unbalanced_full"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO_DIR)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _selected_full_audit(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row
        for row in result["selection_candidates"]["full"]
        if np.allclose(
            np.asarray(row["eta"], dtype=np.float64), eta, rtol=0.0, atol=1.0e-10
        )
    ]
    if not matches:
        raise ValueError(f"selected {design} geometry lacks a Full-action audit")
    return min(matches, key=lambda row: float(row["audit"]["value"]))


def _weighted_metric(
    metrics: dict[str, Any], component: str, weight_plus: float, weight_minus: float
) -> float:
    return float(
        weight_plus * float(metrics[f"{component}_action_plus"]["mean"])
        + weight_minus * float(metrics[f"{component}_action_minus"]["mean"])
    )


def _physical_fold_component(
    payload: dict[str, Any],
    component: str,
    weight_plus: float,
    weight_minus: float,
) -> tuple[float, float]:
    """Return the view mean and physical-fold jackknife SE for one component."""
    by_fold: dict[tuple[int, ...], list[float]] = {}
    values = []
    for view in payload["views"]:
        value = _weighted_metric(
            view["summary"]["metrics"], component, weight_plus, weight_minus
        )
        values.append(value)
        by_fold.setdefault(tuple(int(item) for item in view["run_indices"]), []).append(
            value
        )
    fold_values = np.asarray(
        [np.mean(group) for group in by_fold.values()], dtype=np.float64
    )
    fold_mean = float(np.mean(fold_values))
    jackknife_se = (
        float(
            np.sqrt(
                (len(fold_values) - 1)
                / len(fold_values)
                * np.sum((fold_values - fold_mean) ** 2)
            )
        )
        if len(fold_values) > 1
        else 0.0
    )
    return float(np.mean(values)), jackknife_se


def load_component_records(pareto_dir: Path) -> list[dict]:
    """Load all methods and allowances without rerunning scientific evaluation."""
    pareto_dir = pareto_dir.expanduser().resolve()
    paths = sorted(pareto_dir.glob("risk_*pct/result.json"))
    if len(paths) != 6:
        raise ValueError(f"expected six authoritative Pareto results, found {len(paths)}")
    results = sorted((_load_json(path) for path in paths), key=lambda row: row["allowance_percent"])

    records: list[dict] = []
    for result in results:
        allowance = float(result["allowance_percent"])
        if allowance <= 0.0 or not bool(result.get("selection_certified")):
            raise ValueError(f"invalid or uncertified allowance {allowance:g}%")
        unbalanced = result["config"]["unbalanced"]
        weight_plus = float(unbalanced.get("species_weight_plus", 1.0))
        weight_minus = float(unbalanced.get("species_weight_minus", 1.0))
        selection_anchor = np.asarray(result["risk_view_star"], dtype=np.float64)
        law_validation_risk = float(
            result["validation_designs"]["law"]["summary"]["metrics"]
            ["law_risk_total"]["mean"]
        )

        for method in METHODS:
            design = DESIGN_KEYS[method]
            selected = _selected_full_audit(result, design)
            selection_risk = np.asarray(
                selected["law_screen"]["view_values"], dtype=np.float64
            )
            if selection_risk.shape != selection_anchor.shape:
                raise ValueError(f"selection risk-view mismatch for {(allowance, method)}")
            selection_usage = 100.0 * float(
                np.max(
                    (selection_risk - selection_anchor)
                    / ((allowance / 100.0) * np.abs(selection_anchor))
                )
            )
            if method == "law" and abs(selection_usage) < 1.0e-8:
                selection_usage = 0.0
            if selection_usage > 100.0 + 1.0e-7:
                raise ValueError(f"selection exceeds risk allowance for {(allowance, method)}")

            validation = result["validation_designs"][design]
            move_mean, move_se = _physical_fold_component(
                validation, "move", weight_plus, weight_minus
            )
            reaction_mean, reaction_se = _physical_fold_component(
                validation, "reaction", weight_plus, weight_minus
            )
            total = float(validation["physical_view_action"]["mean"])
            if not np.isclose(move_mean + reaction_mean, total, rtol=2.0e-8, atol=1.0e-10):
                raise ValueError(f"action decomposition failed for {(allowance, method)}")
            validation_risk = float(
                validation["summary"]["metrics"]["law_risk_total"]["mean"]
            )
            validation_usage = (
                100.0
                * (validation_risk - law_validation_risk)
                / abs(law_validation_risk)
                / (allowance / 100.0)
            )
            records.append(
                {
                    "risk_allowance_percent": allowance,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "validation_move_action_mean": move_mean,
                    "validation_move_action_se": move_se,
                    "validation_reaction_action_mean": reaction_mean,
                    "validation_reaction_action_se": reaction_se,
                    "selection_budget_used_percent": selection_usage,
                    "validation_budget_used_percent": validation_usage,
                }
            )
    if len(records) != 18:
        raise ValueError(f"expected 18 method records, found {len(records)}")
    return records


def _series(
    records: list[dict], method: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in records if row["method"] == method),
        key=lambda row: float(row["risk_allowance_percent"]),
    )
    return (
        np.asarray([row["risk_allowance_percent"] for row in selected], dtype=np.float64),
        np.asarray([row[field] for row in selected], dtype=np.float64),
    )


def make_figure(records: list[dict]) -> plt.Figure:
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.55), sharex=True)
    fig.subplots_adjust(left=0.067, right=0.985, bottom=0.17, top=0.76, wspace=0.24)
    move_ax, reaction_ax, risk_ax = axes

    for method in METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        style = {
            "color": color,
            "marker": marker,
            "ms": 6.4,
            "lw": 2.0,
            "markeredgecolor": PANEL_BACKGROUND,
            "markeredgewidth": 0.9,
            "zorder": 4,
        }
        x, move = _series(records, method, "validation_move_action_mean")
        _, move_se = _series(records, method, "validation_move_action_se")
        _, reaction = _series(records, method, "validation_reaction_action_mean")
        _, reaction_se = _series(records, method, "validation_reaction_action_se")
        _, selection_budget = _series(records, method, "selection_budget_used_percent")
        _, validation_budget = _series(records, method, "validation_budget_used_percent")

        move_ax.plot(x, move, **style)
        move_ax.fill_between(
            x, move - 1.96 * move_se, move + 1.96 * move_se,
            color=color, alpha=0.12, linewidth=0, zorder=1,
        )
        reaction_ax.plot(x, reaction, **style)
        reaction_ax.fill_between(
            x, reaction - 1.96 * reaction_se, reaction + 1.96 * reaction_se,
            color=color, alpha=0.12, linewidth=0, zorder=1,
        )
        risk_ax.plot(x, selection_budget, **style)
        risk_ax.plot(
            x,
            validation_budget,
            **{
                **style,
                "ls": "--",
                "markerfacecolor": PANEL_BACKGROUND,
                "markeredgecolor": color,
                "zorder": 3,
            },
        )

    allowances = sorted({float(row["risk_allowance_percent"]) for row in records})
    for ax in axes:
        ax.set_xlabel("Allowed extra risk  (%)")
        ax.set_xticks(allowances)
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.04)
    move_ax.margins(y=0.16)
    reaction_ax.margins(y=0.16)
    move_ax.set_ylabel(r"Validation action component  $A_h$")
    move_ax.set_title("A   Moving defects", loc="left", fontsize=11, fontweight="bold")
    reaction_ax.set_ylabel(r"Validation action component  $A_h$")
    reaction_ax.set_title(
        "B   Creating / removing mass", loc="left", fontsize=11, fontweight="bold"
    )
    move_ax.text(
        0.98, 0.05, "shading: 95% physical-fold interval",
        transform=move_ax.transAxes, ha="right", fontsize=7.6, color=MUTED,
    )

    risk_ax.axhline(100.0, color="#4B9A73", ls=":", lw=1.3, zorder=2)
    risk_ax.axhline(0.0, color="#7A8088", ls=":", lw=0.9, alpha=0.7, zorder=2)
    risk_ax.set_ylim(-7.0, 107.0)
    risk_ax.set_ylabel("Allowed-risk budget used  (%)")
    risk_ax.set_title("C   Risk-allowance usage", loc="left", fontsize=11, fontweight="bold")
    risk_ax.text(
        0.98, 0.94, "100% = full allowance", transform=risk_ax.transAxes,
        ha="right", fontsize=7.6, color="#3F805F",
    )
    risk_ax.text(
        0.98, 0.05, "validation change is relative to validation Law",
        transform=risk_ax.transAxes, ha="right", fontsize=7.6, color=MUTED,
    )

    method_handles = [
        Line2D(
            [0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            markeredgecolor=PANEL_BACKGROUND, markeredgewidth=0.9, lw=2.0,
            markersize=7.2, label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    risk_handles = [
        Line2D([0], [0], color=INK, lw=2.0, label="Risk: selection"),
        Line2D([0], [0], color=INK, lw=2.0, ls="--", label="Risk: validation"),
    ]
    fig.legend(
        handles=method_handles + risk_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.855),
        ncol=5,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.6,
    )
    fig.suptitle(
        "Active nematic · transport, mass creation, and risk use",
        x=0.067, y=0.95, ha="left", fontsize=18, fontweight="bold", color="#20242B",
    )
    return fig


def _output_paths(stem: Path) -> tuple[Path, Path]:
    stem = stem.expanduser().resolve()
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> int:
    args = _parse_args()
    records = load_component_records(args.pareto_dir)
    figure = make_figure(records)
    png, pdf = _output_paths(args.output_stem)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    print(f"saved {png}")
    print(f"saved {pdf}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

