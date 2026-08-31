"""Shared paper-style action-cost and risk-budget Pareto visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from percentage_pareto_visualization import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
)


PAPER_BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#FFFFFF"
INK = "#252A33"
MUTED = "#66707C"


def paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": "#C9C3B8",
            "axes.labelcolor": INK,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#555D67",
            "ytick.color": "#555D67",
            "figure.facecolor": PAPER_BACKGROUND,
            "savefig.facecolor": PAPER_BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series(
    records: list[dict], method: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (record for record in records if record.get("method") == method),
        key=lambda record: float(record["risk_allowance_percent"]),
    )
    if not selected:
        raise ValueError(f"no {method} records to plot")
    return (
        np.asarray(
            [float(record["risk_allowance_percent"]) for record in selected],
            dtype=np.float64,
        ),
        np.asarray([float(record[field]) for record in selected], dtype=np.float64),
    )


def make_cost_risk_figure(
    records: list[dict],
    *,
    title: str = "Cost and risk use along the risk–allowance frontier",
    validation_risk_note: str = "negative = lower risk than validation Law",
    risk_limit_label: str = "100% = full allowance",
) -> plt.Figure:
    """Plot Law-relative Full action and nominal risk-budget use for three methods."""
    if not records:
        raise ValueError("no method records to plot")

    paper_style()
    fig, (action_ax, risk_ax) = plt.subplots(1, 2, figsize=(11.8, 4.55), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.17, top=0.76, wspace=0.12)
    all_budget_values: list[float] = []

    law_x, law_selection = _series(records, "law", "selection_full_action")
    law_validation_x, law_validation = _series(
        records, "law", "validation_full_action_mean"
    )
    if not np.array_equal(law_x, law_validation_x):
        raise ValueError("inconsistent selection/validation allowances for Law")
    if not (
        np.all(np.isfinite(law_selection))
        and np.all(np.isfinite(law_validation))
        and np.all(law_selection > 0.0)
        and np.all(law_validation > 0.0)
    ):
        raise ValueError("Law action references must be finite and positive")

    for method in METHODS:
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        x, selection_action = _series(records, method, "selection_full_action")
        validation_x, validation_action = _series(
            records, method, "validation_full_action_mean"
        )
        if not (
            np.array_equal(x, validation_x)
            and np.array_equal(x, law_x)
        ):
            raise ValueError(f"inconsistent action allowances for {method}")
        selection = 100.0 * selection_action / law_selection
        validation = 100.0 * validation_action / law_validation

        selection_style = {
            "color": color,
            "marker": marker,
            "ms": 6.6,
            "lw": 2.0,
            "markeredgecolor": PANEL_BACKGROUND,
            "markeredgewidth": 0.9,
            "zorder": 4,
        }
        validation_style = {
            **selection_style,
            "ls": "--",
            "markerfacecolor": PANEL_BACKGROUND,
            "markeredgecolor": color,
            "zorder": 3,
        }
        action_ax.plot(x, selection, **selection_style)
        action_ax.plot(x, validation, **validation_style)

        budget_x, selection_budget = _series(
            records, method, "selection_budget_used_percent"
        )
        method_records = [record for record in records if record.get("method") == method]
        explicit_validation_budget = all(
            "validation_budget_used_percent" in record for record in method_records
        )
        validation_field = (
            "validation_budget_used_percent"
            if explicit_validation_budget
            else "validation_R_change_vs_law_percent"
        )
        change_x, validation_value = _series(records, method, validation_field)
        if not (np.array_equal(x, budget_x) and np.array_equal(x, change_x)):
            raise ValueError(f"inconsistent risk allowances for {method}")
        validation_budget = (
            validation_value
            if explicit_validation_budget
            else 100.0 * validation_value / x
        )
        risk_ax.plot(x, selection_budget, **selection_style)
        risk_ax.plot(x, validation_budget, **validation_style)
        all_budget_values.extend(selection_budget[np.isfinite(selection_budget)].tolist())
        all_budget_values.extend(validation_budget[np.isfinite(validation_budget)].tolist())

    allowances = sorted({float(record["risk_allowance_percent"]) for record in records})
    for ax in (action_ax, risk_ax):
        ax.set_xlabel("Allowed extra risk  (%)")
        ax.set_xticks(allowances)
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.04)

    action_ax.axhline(100.0, color="#7A8088", ls=":", lw=0.9, alpha=0.7, zorder=2)
    action_ax.margins(y=0.13)
    action_ax.set_ylabel("Full-action cost  (% of Law)")
    action_ax.set_title(
        "A   Action cost relative to Law", loc="left", fontsize=11, fontweight="bold"
    )
    action_ax.text(
        0.98,
        0.05,
        "100% = Law action on the same bank",
        transform=action_ax.transAxes,
        ha="right",
        fontsize=7.8,
        color=MUTED,
    )

    risk_min = min([0.0, *all_budget_values])
    risk_max = max([100.0, *all_budget_values])
    risk_span = max(risk_max - risk_min, 1.0)
    risk_ax.axhline(100.0, color="#4B9A73", ls=":", lw=1.3, zorder=2)
    risk_ax.axhline(0.0, color="#7A8088", ls=":", lw=0.9, alpha=0.7, zorder=2)
    risk_ax.set_ylim(risk_min - 0.06 * risk_span, risk_max + 0.06 * risk_span)
    risk_ax.set_ylabel("Law-relative risk budget used  (%)")
    risk_ax.set_title(
        "B   Risk-allowance usage", loc="left", fontsize=11, fontweight="bold"
    )
    risk_ax.text(
        0.98,
        0.94,
        risk_limit_label,
        transform=risk_ax.transAxes,
        ha="right",
        fontsize=7.8,
        color="#3F805F",
    )
    risk_ax.text(
        0.98,
        0.05,
        validation_risk_note,
        transform=risk_ax.transAxes,
        ha="right",
        fontsize=7.8,
        color=MUTED,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markeredgecolor=PANEL_BACKGROUND,
            markeredgewidth=0.9,
            lw=2.0,
            markersize=7.2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    bank_handles = [
        Line2D([0], [0], color=INK, lw=2.0, label="Selection"),
        Line2D([0], [0], color=INK, lw=2.0, ls="--", label="Validation"),
    ]
    fig.legend(
        handles=method_handles + bank_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.855),
        ncol=5,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.6,
    )
    fig.suptitle(
        title,
        x=0.09,
        y=0.95,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#20242B",
    )
    return fig


def save_cost_risk_figure(
    records: list[dict],
    output: Path,
    *,
    title: str = "Cost and risk use along the risk–allowance frontier",
    validation_risk_note: str = "negative = lower risk than validation Law",
    risk_limit_label: str = "100% = full allowance",
    dpi: int = 300,
) -> list[Path]:
    """Write raster and vector versions of the two-panel cost/risk figure."""
    stem = Path(output).expanduser().resolve()
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig = make_cost_risk_figure(
        records,
        title=title,
        validation_risk_note=validation_risk_note,
        risk_limit_label=risk_limit_label,
    )
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png, pdf]
