"""Render the vortices information--transport Pareto sweep.

This is pure post-processing: it reads ``pareto.json`` and never launches an
optimization. ``run_pareto.py`` also calls :func:`save_pareto_figure` after each
completed epsilon point so the figure stays current during a long sweep.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "pareto" / "pareto.json"
PAPER = "#F4F1EA"
PANEL = "#FBFAF6"
INK = "#252931"
MUTED = "#69717D"
LAW = "#2C7FB8"
FULL = "#D1495B"
DOMINATED = "#D9A441"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nondominated(x: np.ndarray, y: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if not eligible[i]:
            continue
        dominates = eligible & (x <= x[i]) & (y <= y[i]) & (
            (x < x[i]) | (y < y[i])
        )
        keep[i] = not np.any(dominates)
    return keep


def _epsilon_label(value: float) -> str:
    return rf"$\epsilon_R={value:g}$"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#B8B5AE",
            "axes.labelcolor": INK,
            "xtick.color": "#555B64",
            "ytick.color": "#555B64",
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "savefig.facecolor": PAPER,
        }
    )


def _arrays(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    usable = [row for row in rows if all(_finite(row.get(key)) for key in keys)]
    if not usable:
        missing = ", ".join(keys)
        raise ValueError(f"pareto.json contains no complete rows for: {missing}")
    usable.sort(key=lambda row: float(row["epsilon_r"]))
    return {
        "rows": np.asarray(usable, dtype=object),
        **{
            key: np.asarray([float(row[key]) for row in usable], dtype=np.float64)
            for key in keys
        },
    }


def _draw_points(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    certified: np.ndarray,
    efficient: np.ndarray,
    *,
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
) -> None:
    order = np.argsort(x)
    ax.plot(x[order], y[order], color="#A7A9AD", lw=1.15, zorder=1)
    if xerr is not None or yerr is not None:
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="#777D85",
            elinewidth=0.9,
            capsize=2.5,
            alpha=0.72,
            zorder=2,
        )
    dominated = certified & ~efficient
    failed = ~certified
    ax.scatter(
        x[dominated], y[dominated], s=62, marker="o", facecolor=PANEL,
        edgecolor=DOMINATED, linewidth=1.7, zorder=4,
    )
    ax.scatter(
        x[efficient], y[efficient], s=68, marker="o", facecolor=FULL,
        edgecolor="white", linewidth=1.0, zorder=5,
    )
    ax.scatter(
        x[failed], y[failed], s=70, marker="X", facecolor="#8B9098",
        edgecolor="white", linewidth=0.8, zorder=5,
    )


def make_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    required = (
        "epsilon_r",
        "R_star",
        "full_R_selection",
        "law_A_selection",
        "full_A_selection",
        "law_R_validation",
        "law_A_validation",
        "full_R_validation",
        "full_A_validation",
    )
    data = _arrays(rows, required)
    used = data.pop("rows").tolist()
    eps = data["epsilon_r"]
    selection_excess = data["full_R_selection"] - data["R_star"]
    selection_action = data["full_A_selection"]
    validation_risk = data["full_R_validation"]
    validation_action = data["full_A_validation"]
    certified = np.asarray([bool(row.get("full_certified", False)) for row in used])
    action_se = np.asarray(
        [float(row.get("full_A_validation_se", 0.0) or 0.0) for row in used]
    )
    risk_se = np.asarray(
        [float(row.get("full_R_validation_se", 0.0) or 0.0) for row in used]
    )
    efficient_selection = _nondominated(
        selection_excess, selection_action, certified
    )
    efficient_validation = _nondominated(validation_risk, validation_action, certified)

    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 5.4))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.17, top=0.72, wspace=0.24)

    law_action_selection = float(np.median(data["law_A_selection"]))
    law_action = float(np.median(data["law_A_validation"]))
    law_risk = float(np.median(data["law_R_validation"]))

    left.axvspan(
        min(float(np.min(selection_excess)), 0.0), 0.0,
        color=LAW, alpha=0.055, zorder=0,
    )
    left.axvline(0.0, color=LAW, ls="--", lw=1.0, alpha=0.72)
    left.axhline(law_action_selection, color=LAW, ls=":", lw=1.25, alpha=0.82)
    _draw_points(
        left,
        selection_excess,
        selection_action,
        certified,
        efficient_selection,
    )
    offsets = ((6, 9), (6, -18), (6, 9), (-68, -18), (6, 9), (-68, 9))
    for i, (x, y, epsilon) in enumerate(zip(selection_excess, selection_action, eps)):
        left.annotate(
            _epsilon_label(float(epsilon)),
            (x, y),
            xytext=offsets[i % len(offsets)],
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    left.text(
        0.02, 0.965, r"blue region: selection risk below $R^\star$",
        transform=left.transAxes, va="top", color=LAW, fontsize=7.8,
    )
    left.set_xlabel(r"Exact selection-law excess  $R_{\rm sel}-R^\star$")
    left.set_ylabel(r"Exact selection full action  $A_{\rm full}$")
    left.set_title("A   Certified selection frontier", loc="left", fontweight="bold")
    left.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)

    right.axhline(law_action, color=LAW, ls=":", lw=1.25, alpha=0.82)
    right.axvline(law_risk, color=LAW, ls=":", lw=1.25, alpha=0.82)
    right.scatter(
        [law_risk], [law_action], marker="*", s=145, facecolor=LAW,
        edgecolor="white", linewidth=0.8, zorder=6,
    )
    _draw_points(
        right,
        validation_risk,
        validation_action,
        certified,
        efficient_validation,
        xerr=risk_se,
        yerr=action_se,
    )
    for i, (x, y, epsilon, row) in enumerate(
        zip(validation_risk, validation_action, eps, used)
    ):
        reduction = row.get("validation_action_reduction")
        suffix = f" · {100.0 * float(reduction):.0f}% less A" if _finite(reduction) else ""
        right.annotate(
            _epsilon_label(float(epsilon)) + suffix,
            (x, y),
            xytext=offsets[i % len(offsets)],
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    right.annotate(
        "Law baseline",
        (law_risk, law_action),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8,
        color=LAW,
        fontweight="bold",
    )
    right.set_xlabel(r"Validation finite-law risk  $R_{\rm val}$")
    right.set_ylabel(r"Validation full action  $A_{\rm full}$")
    right.set_title("B   Independent-validation frontier", loc="left", fontweight="bold")
    right.ticklabel_format(axis="x", style="plain", useOffset=False)

    for ax in (left, right):
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.14, y=0.20)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FULL,
               markeredgecolor="white", markersize=7, label="nondominated + certified"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PANEL,
               markeredgecolor=DOMINATED, markeredgewidth=1.5, markersize=7,
               label="certified but dominated"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#8B9098",
               markeredgecolor="white", markersize=7, label="failed certificate"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=LAW,
               markeredgecolor="white", markersize=10, label="Law baseline"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.815),
               ncol=4, frameon=False, fontsize=8.3)
    fig.suptitle(
        "Vortices · information–transport Pareto sweep",
        x=0.075, y=0.965, ha="left", fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(
        0.075, 0.915,
        "Each point is a full sensor-design run; labels give the allowed finite-law excess.",
        ha="left", fontsize=9.2, color=MUTED,
    )
    fig.text(
        0.5, 0.045,
        "Error bars show ±1 SE on independent validation. Lines guide the eye; dominance is computed within the completed sweep.",
        ha="center", fontsize=8.1, color=MUTED,
    )
    return fig


def save_pareto_figure(
    rows: list[dict[str, Any]], output: Path, *, dpi: int = 210
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a completed vortices Pareto sweep.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=210)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        rows = json.loads(args.input.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read Pareto data {args.input}: {exc}") from exc
    if not isinstance(rows, list):
        raise SystemExit(f"Expected a JSON list in {args.input}")
    output = args.output or args.input.with_name("pareto.png")
    fig = make_figure(rows)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved={output}", flush=True)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
