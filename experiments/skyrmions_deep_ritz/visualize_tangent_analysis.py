"""Publication visuals for the additive skyrmion Tangent analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .visualize_authoritative import (
    BLUE,
    GRAY,
    GREEN,
    NAVY,
    ORANGE,
    PURPLE,
    STYLE,
    _draw_domain,
    _resolve_result_path,
    _save,
)
from .visualize_field_observables import (
    _density_frames,
    _field_figure,
    _observable_trajectory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TANGENT_COLOR = "#7A5195"
FULL_COLOR = "#EF8354"


def _load(
    tangent_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    tangent = json.loads(tangent_path.read_text(encoding="utf-8"))
    if not tangent.get("certified") or tangent.get("exploratory"):
        raise RuntimeError("combined visuals require a certified Tangent extension")
    full_path = Path(tangent["existing_full_pareto"])
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if not full.get("certified") or full.get("exploratory_override"):
        raise RuntimeError("combined visuals require a certified Full Pareto sweep")
    results = [
        json.loads(_resolve_result_path(row["result"], full_path).read_text(encoding="utf-8"))
        for row in full["rows"]
    ]
    return tangent, full, results, full_path


def _comparison_dashboard(
    tangent: dict[str, Any],
    full: dict[str, Any],
    results: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    tangent_rows = tangent["rows"]
    full_rows = full["rows"]
    allowance = np.asarray([row["allowance_percent"] for row in tangent_rows])
    tangent_risk = np.asarray([row["extra_risk_percent"] for row in tangent_rows])
    full_risk = np.asarray([row["extra_risk_percent"] for row in full_rows])
    tangent_action = np.asarray([row["validation_tangent_action"] for row in tangent_rows])
    tangent_se = np.asarray(
        [row["validation_tangent_action_standard_error"] for row in tangent_rows]
    )
    full_action = np.asarray([row["validation_action"] for row in full_rows])
    full_se = np.asarray([row["validation_action_standard_error"] for row in full_rows])
    tangent_gain = 100.0 * np.asarray(
        [row["validation_tangent_action_reduction_vs_law"] for row in tangent_rows]
    )
    full_gain = 100.0 * np.asarray(
        [row["validation_action_reduction_vs_law"] for row in full_rows]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), layout="constrained")
    fig.suptitle(
        "Certified skyrmion Pareto · Full correction and Tangent lower bound",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )

    ax = axes[0, 0]
    ax.errorbar(
        tangent_risk,
        tangent_action,
        yerr=tangent_se,
        marker="^",
        color=TANGENT_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.2,
        capsize=3,
        label="Tangent-selected · Tangent action",
    )
    ax.errorbar(
        full_risk,
        full_action,
        yerr=full_se,
        marker="o",
        color=FULL_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.2,
        capsize=3,
        label="Full-selected · Full action",
    )
    ax.scatter(
        [0.0],
        [tangent["law"]["validation_certificate"]["action"]],
        marker="^",
        s=85,
        color=TANGENT_COLOR,
        zorder=4,
    )
    ax.scatter(
        [0.0],
        [full_rows[0]["validation_law_action"]],
        marker="o",
        s=65,
        color=FULL_COLOR,
        zorder=4,
    )
    ax.set(
        title="A   Independently validated action",
        xlabel="actual extra selection risk (%)",
        ylabel="kinetic action ↓",
    )
    ax.legend(fontsize=8.8)

    ax = axes[0, 1]
    ax.plot(
        allowance,
        tangent_gain,
        "^-",
        color=TANGENT_COLOR,
        linewidth=2.2,
        label="Tangent vs Tangent Law",
    )
    ax.plot(
        allowance,
        full_gain,
        "o-",
        color=FULL_COLOR,
        linewidth=2.2,
        label="Full vs Full Law",
    )
    ax.axvspan(3.0, 5.0, color=GREEN, alpha=0.07)
    ax.annotate(
        "Full plateaus; Tangent continues",
        xy=(4.0, tangent_gain[4]),
        xytext=(-85, -42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": TANGENT_COLOR},
        color=NAVY,
        fontsize=9,
    )
    ax.set(
        title="B   Reduction against each method's Law baseline",
        xlabel="allowed extra scientific risk (%)",
        ylabel="validated action reduction (%) ↑",
    )
    ax.legend(fontsize=8.8)

    ax = axes[1, 0]
    ax.plot(allowance, tangent_risk, "^-", color=TANGENT_COLOR, linewidth=2.2, label="Tangent")
    ax.plot(allowance, full_risk, "o-", color=FULL_COLOR, linewidth=2.2, label="Full")
    ax.plot(allowance, allowance, "--", color=GRAY, linewidth=1.2, label="risk ceiling")
    ax.set(
        title="C   How much of each risk allowance is used",
        xlabel="allowed extra scientific risk (%)",
        ylabel="actual extra selection risk (%)",
    )
    ax.legend(fontsize=8.8)

    ax = axes[1, 1]
    target_index = int(np.argmin(np.abs(allowance - 3.0)))
    tangent_law_time = np.asarray(tangent["law"]["validation_certificate"]["action_by_time"])
    tangent_time = np.asarray(tangent_rows[target_index]["validation_certificate"]["action_by_time"])
    full_law_time = np.asarray(results[target_index]["validation"]["law"]["certificate"]["kinetic_by_time"])
    full_time = np.asarray(results[target_index]["validation"]["full"]["certificate"]["kinetic_by_time"])
    time = np.linspace(0.0, 1.0, len(tangent_time))
    ax.plot(time, tangent_law_time, "--", color=TANGENT_COLOR, alpha=0.48, label="Tangent Law")
    ax.plot(time, tangent_time, "^-", color=TANGENT_COLOR, markevery=2, label="Tangent 3%")
    ax.plot(time, full_law_time, "--", color=FULL_COLOR, alpha=0.48, label="Full Law")
    ax.plot(time, full_time, "o-", color=FULL_COLOR, markevery=2, label="Full 3%")
    ax.set(
        title="D   Where the 3% action occurs in time",
        xlabel="normalized time",
        ylabel="action density",
    )
    ax.legend(ncols=2, fontsize=8.5)

    fig.text(
        0.5,
        -0.015,
        "Tangent is the minimum correction satisfying only the selected moment rates; "
        "Full enforces the many-body continuity equation.",
        ha="center",
        color=GRAY,
        fontsize=9.2,
    )
    return _save(fig, output, "authoritative_tangent_comparison")


def _sensor_layouts(
    tangent: dict[str, Any],
    results: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    reference = results[0]
    physics = reference["config"]["physics"]
    measurement = reference["config"]["measurement"]
    box = np.asarray(physics["box"], dtype=float)
    pins = np.asarray(physics["pinning_centers"], dtype=float)
    width = float(measurement["sensor_width"])
    panels = [("Law", np.asarray(tangent["law"]["eta"]), "risk anchor")]
    for row in tangent["rows"]:
        panels.append((
            f"{row['allowance_percent']:g}%",
            np.asarray(row["eta"]),
            f"+{row['extra_risk_percent']:.2f}% risk\n"
            f"{100.0 * row['validation_tangent_action_reduction_vs_law']:.1f}% validated gain",
        ))
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(3.05 * len(panels), 3.35),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    fig.suptitle(
        "Tangent-selected sensor layouts along the certified risk sweep",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    for ax, (title, eta, subtitle) in zip(np.atleast_1d(axes), panels):
        _draw_domain(
            ax,
            centers=eta.reshape(-1, 2),
            pinning=pins,
            box=box,
            width=width,
        )
        ax.set_title(title, color=NAVY, pad=20)
        ax.text(
            0.5,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=GRAY,
        )
        ax.set_xlabel("x")
    np.atleast_1d(axes)[0].set_ylabel("y")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="X", color="none", markerfacecolor=PURPLE, markeredgecolor=PURPLE, label="pinning center"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", label="fixed local-density sensor"),
        ],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.035),
    )
    return _save(fig, output, "tangent_sensor_layout_evolution")


def _field_figures(
    tangent: dict[str, Any],
    results: list[dict[str, Any]],
    full_path: Path,
    output: Path,
) -> list[Path]:
    result = results[0]
    first_result_path = _resolve_result_path(
        json.loads(full_path.read_text(encoding="utf-8"))["rows"][0]["result"],
        full_path,
    )
    with np.load(first_result_path.parent / "truth_banks.npz") as truth:
        times = np.asarray(truth["times"], dtype=np.float64)
        configurations = np.asarray(truth["design"], dtype=np.float64)
    box = np.asarray(result["config"]["physics"]["box"], dtype=np.float64)
    densities = _density_frames(configurations, box=box)
    paths: list[Path] = []
    for row in tangent["rows"]:
        eta = np.asarray(row["eta"], dtype=np.float64)
        observable = _observable_trajectory(configurations, times, eta, result)
        allowance = float(row["allowance_percent"])
        paths.extend(_field_figure(
            times=times,
            densities=densities,
            observable=observable,
            eta=eta,
            result=result,
            title=f"Certified Tangent design · {allowance:g}% risk allowance",
            subtitle="Frozen truth trajectory with Tangent-selected local-density observables",
            output=output,
            stem=f"field_observables_tangent_{allowance:g}pct".replace(".", "p"),
        ))
    return paths


def _write_markdown(
    tangent: dict[str, Any],
    full: dict[str, Any],
    output: Path,
) -> Path:
    full_by_allowance = {
        float(row["allowance_percent"]): row for row in full["rows"]
    }
    lines = [
        "# Certified skyrmion Tangent extension",
        "",
        "The original Law/Full artifacts were retained verbatim. The Tangent curve was "
        "computed from frozen banks with closed-form Gram solves; no Deep Ritz model was rerun.",
        "",
        "Tangent is a lower bound that enforces the selected moment rates only. Full Deep Ritz "
        "enforces the complete many-body continuity equation, so the two curves should not be "
        "described as equivalent realized transports.",
        "",
        "| allowance | Tangent actual risk | Tangent validation action | Tangent gain | Full validation action | Full gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tangent["rows"]:
        full_row = full_by_allowance[float(row["allowance_percent"])]
        lines.append(
            f"| {row['allowance_percent']:g}% | {row['extra_risk_percent']:.3f}% | "
            f"{row['validation_tangent_action']:.6f} | "
            f"{100.0 * row['validation_tangent_action_reduction_vs_law']:.2f}% | "
            f"{full_row['validation_action']:.6f} | "
            f"{100.0 * full_row['validation_action_reduction_vs_law']:.2f}% |"
        )
    lines.extend([
        "",
        f"Saved feasible geometries rescored: {tangent['selection_protocol']['saved_feasible_geometries_scored']}.",
        f"New Tangent-only refined geometries retained: {tangent['selection_protocol']['tangent_local_refined_geometries']}.",
        f"Full Deep Ritz rerun: {tangent['full_deep_ritz_rerun']}.",
    ])
    path = output / "tangent_analysis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the certified skyrmion Tangent extension")
    parser.add_argument(
        "tangent",
        nargs="?",
        type=Path,
        default=(
            SCRIPT_DIR
            / "outputs"
            / "pareto_authoritative"
            / "tangent_analysis"
            / "tangent_pareto.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tangent, full, results, full_path = _load(args.tangent)
    output = args.output or args.tangent.parent / "figures"
    field_output = output / "field_observables"
    output.mkdir(parents=True, exist_ok=True)
    field_output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        paths = []
        paths.extend(_comparison_dashboard(tangent, full, results, output))
        paths.extend(_sensor_layouts(tangent, results, output))
        paths.extend(_field_figures(tangent, results, full_path, field_output))
    paths.append(_write_markdown(tangent, full, output))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
