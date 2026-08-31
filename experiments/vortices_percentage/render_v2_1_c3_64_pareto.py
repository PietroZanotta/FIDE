#!/usr/bin/env python3
"""Render the confirmed 0.5--2% Vortices comparison in the Toy plot style."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

from pareto_cost_risk_visualization import make_cost_risk_figure, paper_style  # noqa: E402
from percentage_pareto_visualization import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    make_figure,
)

PUBLISHED = HERE / "outputs" / "published"
PUBLISHED_DATA = PUBLISHED / "pareto_data.json"
INFERENCE = PUBLISHED / "simultaneous_inference.json"
OUTPUT = HERE / "plots"
ALLOWANCES = (0.5, 1.0, 2.0)
TAGS = ("0p5", "1p0", "2p0")
METHODS = ("law", "tangent", "full")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    return png, pdf


def _method_series(
    records: list[dict[str, Any]], method: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(
        (row for row in records if row["method"] == method),
        key=lambda row: float(row["risk_allowance_percent"]),
    )
    return (
        np.asarray([row["risk_allowance_percent"] for row in rows], dtype=float),
        np.asarray([row[field] for row in rows], dtype=float),
    )


def make_relative_metrics_figure(records: list[dict[str, Any]]) -> plt.Figure:
    """Toy-style relative Tangent action, Full action, and risk-budget use."""
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.75), sharex=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.75, wspace=0.19)
    allowances = sorted({float(row["risk_allowance_percent"]) for row in records})
    panel_specs = (
        (
            axes[0],
            "selection_tangent_action",
            "validation_tangent_action_mean",
            "A   Tangent action relative to Law",
            r"Tangent-action cost  (% of Law)",
        ),
        (
            axes[1],
            "selection_full_action",
            "validation_full_action_mean",
            "B   Full action relative to Law",
            r"Full-action cost  (% of Law)",
        ),
    )
    for ax, selection_field, holdout_field, title, ylabel in panel_specs:
        law_x, law_selection = _method_series(records, "law", selection_field)
        holdout_x, law_holdout = _method_series(records, "law", holdout_field)
        if not np.array_equal(law_x, holdout_x):
            raise RuntimeError(f"inconsistent Law coordinates for {selection_field}")
        for method in METHODS:
            x, selection = _method_series(records, method, selection_field)
            xh, holdout = _method_series(records, method, holdout_field)
            if not (np.array_equal(x, law_x) and np.array_equal(xh, law_x)):
                raise RuntimeError(f"inconsistent coordinates for {method}/{selection_field}")
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            common = {
                "color": color,
                "marker": marker,
                "ms": 6.6,
                "lw": 2.0,
                "markeredgewidth": 0.9,
            }
            ax.plot(
                x,
                100.0 * selection / law_selection,
                markeredgecolor="white",
                zorder=4,
                **common,
            )
            ax.plot(
                x,
                100.0 * holdout / law_holdout,
                ls="--",
                markerfacecolor="white",
                markeredgecolor=color,
                zorder=3,
                **common,
            )
        ax.axhline(100.0, color="#7A8088", ls=":", lw=0.9, alpha=0.7)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.margins(y=0.13)
        ax.text(
            0.98,
            0.035,
            "100% = Law metric on the same bank",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.6,
            color="#66707C",
        )

    risk = axes[2]
    all_risk: list[float] = []
    for method in METHODS:
        x, selection = _method_series(records, method, "selection_budget_used_percent")
        xh, holdout = _method_series(records, method, "validation_budget_used_percent")
        if not np.array_equal(x, xh):
            raise RuntimeError(f"inconsistent risk coordinates for {method}")
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        common = {
            "color": color,
            "marker": marker,
            "ms": 6.6,
            "lw": 2.0,
            "markeredgewidth": 0.9,
        }
        risk.plot(x, selection, markeredgecolor="white", zorder=4, **common)
        risk.plot(
            x,
            holdout,
            ls="--",
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=3,
            **common,
        )
        all_risk.extend(selection.tolist())
        all_risk.extend(holdout.tolist())
    risk.axhline(100.0, color="#4B9A73", ls=":", lw=1.3)
    risk.axhline(0.0, color="#7A8088", ls=":", lw=0.9, alpha=0.7)
    span = max(100.0, max(all_risk)) - min(0.0, min(all_risk))
    risk.set_ylim(min(0.0, min(all_risk)) - 0.06 * span, max(100.0, max(all_risk)) + 0.06 * span)
    risk.set_title("C   Risk-allowance usage", loc="left", fontsize=11, fontweight="bold")
    risk.set_ylabel("Law-relative risk budget used  (%)")
    risk.text(
        0.98,
        0.94,
        "100% = full allowance",
        transform=risk.transAxes,
        ha="right",
        fontsize=7.6,
        color="#3F805F",
    )

    for ax in axes:
        ax.set_xlabel("Allowed extra risk  (%)")
        ax.set_xticks(allowances)
        ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
        ax.margins(x=0.04)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markeredgecolor="white",
            markeredgewidth=0.9,
            lw=2.0,
            markersize=7.2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    bank_handles = [
        Line2D([0], [0], color="#252A33", lw=2.0, label="Selection"),
        Line2D([0], [0], color="#252A33", lw=2.0, ls="--", label="Holdout"),
    ]
    fig.legend(
        handles=method_handles + bank_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.84),
        ncol=5,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.6,
    )
    fig.suptitle(
        "Relative action and risk use along the risk–allowance frontier",
        x=0.065,
        y=0.95,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.065,
        0.035,
        "Tangent-action metrics and the Tangent-design holdout are supplementary descriptive common-bank evaluations outside the primary Law–Full inference family.",
        ha="left",
        fontsize=7.8,
        color="#66707C",
    )
    return fig


def main() -> int:
    published = load_json(PUBLISHED_DATA)
    inference = load_json(INFERENCE)
    if inference.get("status") != "PASS":
        raise RuntimeError("a passing independent 64-trial holdout receipt is required")
    records = published.get("records", [])
    if published.get("status") != "COMPLETE_TOY_STYLE_RELATIVE_TANGENT_FULL_RISK_COMPARISON" or len(records) != 9:
        raise RuntimeError("the compact published Pareto records are incomplete")
    by_key = {(float(row["risk_allowance_percent"]), str(row["method"])): row for row in records}
    effects = np.asarray(inference["effects"], dtype=np.float64)
    lower = np.asarray(inference["simultaneous_lower"], dtype=np.float64)
    upper = np.asarray(inference["simultaneous_upper"], dtype=np.float64)
    if effects.shape != lower.shape or effects.shape != upper.shape or effects.shape != (3, 3):
        raise RuntimeError("the compact simultaneous-inference arrays are invalid")
    full_only_rows: list[dict[str, Any]] = []
    for index, allowance in enumerate(ALLOWANCES):
        law = by_key[(allowance, "law")]
        full = by_key[(allowance, "full")]
        full_only_rows.append(
            {
                "risk_allowance_percent": allowance,
                "R_star": float(full["R_star"]),
                "full_R_excess_selection": float(full["selection_R"]) - float(full["R_star"]),
                "full_A_selection": float(full["selection_full_action"]),
                "law_A_selection": float(law["selection_full_action"]),
                "full_certified": bool(full["certified"]),
                "law_R_validation": float(law["validation_R_mean"]),
                "full_R_validation": float(full["validation_R_mean"]),
                "law_A_validation": float(law["validation_full_action_mean"]),
                "full_A_validation": float(full["validation_full_action_mean"]),
                "validation_action_reduction": float(np.mean(effects[:, index])),
                "validation_ci_lower": float(np.min(lower[:, index])),
                "validation_ci_upper": float(np.max(upper[:, index])),
                "interval_semantics": "conservative envelope of the three primary simultaneous intervals",
                "candidate_id": str(full["candidate_id"]),
                "selection_receipt_sha256": str(full["selection_receipt_sha256"]),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT / "pareto_0p5_to_2pct_data.json"
    csv_path = OUTPUT / "pareto_0p5_to_2pct.csv"
    atomic_json(data_path, published)
    write_csv(csv_path, records)

    frontier = make_figure(full_only_rows, experiment_label="Vortices experiment")
    frontier.axes[1].set_title("B   Independent holdout check", loc="left")
    frontier.axes[1].set_xlabel("Holdout finite-risk change vs Law  (%)")
    frontier.axes[1].set_ylabel("Holdout Full-action reduction vs Law  (%)")
    frontier.axes[2].set_ylabel("Holdout Full-action reduction vs Law  (%)")
    for item in frontier.texts:
        if item.get_text().startswith("Selection-bank certification is authoritative"):
            item.set_text("Selection receipts define the partial frontier; the independent 64-trial holdout confirms all nine primary effects.")
    frontier_files = save_figure(frontier, OUTPUT / "pareto_frontier_3panel_0p5_to_2pct", dpi=240)
    plt.close(frontier)

    figure = make_cost_risk_figure(
        records,
        title="Cost and risk use along the risk–allowance frontier",
        validation_risk_note="negative = below holdout Law risk",
        risk_limit_label="100% = full allowance",
    )
    if figure.legends:
        figure.legends[0].texts[-1].set_text("Holdout")
    for item in figure.axes[1].texts:
        if item.get_text().startswith("negative ="):
            item.set_position((0.98, 0.018))
    figure.text(0.09, 0.035, "Tangent holdout is a supplementary descriptive common-bank evaluation; it is outside the primary Law–Full inference family.", ha="left", fontsize=7.8, color="#66707C")
    full_action_risk_files = save_figure(figure, OUTPUT / "pareto_methods_full_action_risk_0p5_to_2pct")
    plt.close(figure)

    relative_figure = make_relative_metrics_figure(records)
    relative_files = save_figure(relative_figure, OUTPUT / "pareto_relative_metrics_0p5_to_2pct")
    canonical_files = save_figure(relative_figure, OUTPUT / "pareto_0p5_to_2pct")
    method_alias_files = save_figure(relative_figure, OUTPUT / "pareto_methods_0p5_to_2pct")
    plt.close(relative_figure)

    files = (data_path, csv_path, *relative_files, *canonical_files, *method_alias_files, *full_action_risk_files, *frontier_files)
    manifest = {
        "schema_version": 4,
        "status": "COMPLETE_PUBLISHED_ARTIFACT_PLOTS",
        "files": {path.name: sha256_file(path) for path in files},
        "renderer_sha256": sha256_file(Path(__file__)),
        "published_data_sha256": sha256_file(PUBLISHED_DATA),
        "simultaneous_inference_sha256": sha256_file(INFERENCE),
        "scientific_execution_performed": False,
        "selection_state_modified": False,
        "original_3_to_5pct_selection_remains_paused": True,
    }
    atomic_json(OUTPUT / "pareto_0p5_to_2pct_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
