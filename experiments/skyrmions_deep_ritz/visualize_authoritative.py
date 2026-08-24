"""Publication-style visuals for a certified skyrmion Pareto sweep."""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

NAVY = "#16324F"
BLUE = "#2B6CB0"
ORANGE = "#E8792E"
GREEN = "#238B8D"
PURPLE = "#7656A5"
GRAY = "#667085"
LIGHT_GRAY = "#D7DEE8"
SENSOR_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": NAVY,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E8ECF2",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.85,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _resolve_result_path(raw: str, pareto_path: Path) -> Path:
    path = Path(raw)
    candidates = (path, Path.cwd() / path, REPO_ROOT / path, pareto_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve Pareto result path: {raw}")


def _load(pareto_path: Path, *, allow_exploratory: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(pareto_path.read_text(encoding="utf-8"))
    if not data.get("certified") and not allow_exploratory:
        raise RuntimeError(
            "publication visuals require certified=true; pass --allow-exploratory "
            "only for explicitly labeled diagnostic output"
        )
    results = [
        json.loads(_resolve_result_path(row["result"], pareto_path).read_text(encoding="utf-8"))
        for row in data["rows"]
    ]
    return data, results


def _periodic_delta(start: np.ndarray, end: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = end - start
    return delta - np.round(delta / box) * box


def _match_to_law(centers: np.ndarray, law: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match unordered sensors to Law with minimum periodic squared displacement."""

    best_centers: np.ndarray | None = None
    best_delta: np.ndarray | None = None
    best_cost = np.inf
    for order in permutations(range(len(centers))):
        ordered = centers[np.asarray(order)]
        delta = _periodic_delta(law, ordered, box)
        cost = float(np.sum(delta * delta))
        if cost < best_cost:
            best_cost = cost
            best_centers = ordered
            best_delta = delta
    assert best_centers is not None and best_delta is not None
    return best_centers, best_delta


def _unique_design_groups(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    grouped: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(round(float(value), 12) for value in row["eta"])
        grouped.setdefault(key, []).append(index)
    output = []
    for indices in grouped.values():
        allowances = [float(rows[index]["allowance_percent"]) for index in indices]
        lo, hi = min(allowances), max(allowances)
        label = f"{lo:g}%" if lo == hi else f"{lo:g}–{hi:g}%"
        output.append((indices[0], label))
    return output


def _draw_domain(
    ax: plt.Axes,
    *,
    centers: np.ndarray,
    pinning: np.ndarray,
    box: np.ndarray,
    width: float,
    colors: tuple[str, ...] = SENSOR_COLORS,
) -> None:
    ax.grid(False)
    ax.set_xlim(0.0, float(box[0]))
    ax.set_ylim(0.0, float(box[1]))
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 1.0, 2.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.scatter(
        pinning[:, 0], pinning[:, 1], marker="X", s=42, color=PURPLE,
        alpha=0.72, linewidths=0.4, edgecolors="white", zorder=2,
    )
    for index, (center, color) in enumerate(zip(centers, colors), start=1):
        ax.add_patch(Circle(center, width, facecolor=color, edgecolor="none", alpha=0.12, zorder=1))
        ax.scatter(*center, s=66, color=color, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(
            center[0], center[1], str(index), ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold", zorder=5,
        )


def _save(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=260, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _summary_figure(
    data: dict[str, Any],
    results: list[dict[str, Any]],
    output: Path,
    *,
    label: str,
) -> list[Path]:
    rows = data["rows"]
    allowance = np.asarray([row["allowance_percent"] for row in rows], dtype=float)
    actual_risk = np.asarray([row["extra_risk_percent"] for row in rows], dtype=float)
    validation_action = np.asarray([row["validation_action"] for row in rows], dtype=float)
    validation_se = np.asarray([row["validation_action_standard_error"] for row in rows], dtype=float)
    selection_reduction = 100.0 * np.asarray([row["action_reduction_vs_law"] for row in rows])
    validation_reduction = 100.0 * np.asarray(
        [row["validation_action_reduction_vs_law"] for row in rows]
    )
    knee = int(np.argmax(validation_reduction))
    unique = _unique_design_groups(rows)

    law_validation = float(rows[0]["validation_law_action"])
    law_validation_se = float(results[0]["validation"]["law"]["certificate"]["action_standard_error"])
    knee_result = results[knee]
    measurement = knee_result["config"]["measurement"]
    physics = knee_result["config"]["physics"]
    box = np.asarray(physics["box"], dtype=float)
    pins = np.asarray(physics["pinning_centers"], dtype=float)
    law_centers = np.asarray(knee_result["law_anchor"]["eta"], dtype=float).reshape(-1, 2)
    full_centers = np.asarray(rows[knee]["eta"], dtype=float).reshape(-1, 2)
    full_centers, displacement = _match_to_law(full_centers, law_centers, box)

    selection_cert = knee_result["full_3_percent"]["certificate"]
    validation_cert = knee_result["validation"]["full"]["certificate"]
    residual_keys = (
        "maximum_weak_residual",
        "maximum_energy_residual",
        "maximum_gauge_residual",
        "maximum_moment_rate_residual",
    )
    residual_labels = ("weak", "energy", "gauge", "moment-rate")
    selection_ratios = np.asarray(
        [selection_cert[key] / selection_cert["thresholds"][key] for key in residual_keys]
    )
    validation_ratios = np.asarray(
        [validation_cert[key] / validation_cert["thresholds"][key] for key in residual_keys]
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), layout="constrained")
    fig.suptitle(
        f"{label} skyrmion design: performance, geometry, and certificates",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )

    ax = axes[0, 0]
    ax.errorbar(
        [0.0], [law_validation], yerr=[law_validation_se], fmt="*", markersize=14,
        color=NAVY, capsize=3, label="Law", zorder=5,
    )
    unique_indices = [index for index, _ in unique]
    ax.fill_between(
        np.r_[0.0, actual_risk[unique_indices]],
        np.r_[law_validation, validation_action[unique_indices]],
        law_validation,
        color=GREEN,
        alpha=0.10,
    )
    ax.errorbar(
        actual_risk[unique_indices], validation_action[unique_indices],
        yerr=validation_se[unique_indices], fmt="o-", color=GREEN,
        markerfacecolor="white", markeredgewidth=2, linewidth=2.2,
        capsize=3, label="Full · independent validation", zorder=4,
    )
    for (index, group_label) in unique:
        ax.annotate(
            group_label,
            (actual_risk[index], validation_action[index]),
            xytext=(5, -15 if index == knee else 7),
            textcoords="offset points",
            fontsize=9,
            color=NAVY,
        )
    ax.set(
        title="A   Out-of-sample action vs actual risk",
        xlabel="actual extra selection risk (%)",
        ylabel="independent-validation action ↓",
    )
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(allowance, selection_reduction, "o-", color=BLUE, linewidth=2.2, label="selection bank")
    ax.plot(
        allowance, validation_reduction, "s--", color=ORANGE,
        linewidth=2.2, label="independent validation bank",
    )
    ax.axvspan(allowance[knee], allowance[-1], color=GREEN, alpha=0.08)
    ax.axvline(allowance[knee], color=GREEN, linestyle=":", linewidth=1.5)
    ax.annotate(
        f"Pareto knee\n{validation_reduction[knee]:.1f}% validated reduction",
        xy=(allowance[knee], validation_reduction[knee]),
        xytext=(-92, -52),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.2},
        color=NAVY,
        fontsize=9.5,
    )
    ax.set(
        title="B   Benefit saturates at the 3% allowance",
        xlabel="allowed extra scientific risk (%)",
        ylabel="action reduction vs Law (%) ↑",
    )
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    ax.grid(False)
    ax.set_xlim(0.0, float(box[0]))
    ax.set_ylim(0.0, float(box[1]))
    ax.set_aspect("equal")
    ax.scatter(pins[:, 0], pins[:, 1], marker="X", s=52, color=PURPLE, alpha=0.75, zorder=2)
    width = float(measurement["sensor_width"])
    for index, (law_center, full_center, delta, color) in enumerate(
        zip(law_centers, full_centers, displacement, SENSOR_COLORS), start=1
    ):
        ax.add_patch(Circle(law_center, width, fill=False, linestyle="--", linewidth=1.2, edgecolor=color, alpha=0.7))
        ax.scatter(*law_center, s=58, facecolor="white", edgecolor=color, linewidth=1.6, zorder=4)
        endpoint = law_center + delta
        ax.annotate(
            "", xy=endpoint, xytext=law_center,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
            zorder=3,
        )
        ax.add_patch(Circle(endpoint, width, facecolor=color, edgecolor="none", alpha=0.12))
        ax.scatter(*endpoint, s=68, color=color, edgecolor="white", linewidth=1.0, zorder=5)
        ax.text(*endpoint, str(index), ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=6)
    ax.set(
        title="C   Sensor geometry: Law → 3% Pareto knee",
        xlabel="periodic x",
        ylabel="periodic y",
        xticks=[0.0, 1.0, 2.0],
        yticks=[0.0, 0.5, 1.0],
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=GRAY, label="Law sensor"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", label="3% sensor"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor=PURPLE, markeredgecolor=PURPLE, label="pinning center"),
        ],
        loc="lower left",
        ncols=3,
        fontsize=8.5,
    )

    ax = axes[1, 1]
    x = np.arange(len(residual_labels))
    width_bar = 0.34
    ax.bar(x - width_bar / 2, selection_ratios, width_bar, color=BLUE, label="selection")
    ax.bar(x + width_bar / 2, validation_ratios, width_bar, color=ORANGE, label="validation")
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.2, label="certificate limit")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-8, 2.0)
    ax.set_xticks(x, residual_labels)
    ax.set(ylabel="residual / allowed threshold ↓")
    ax.set_title("D   3% winner remains inside every certificate")
    ess = 100.0 * float(rows[knee]["minimum_ess_fraction"])
    ess_limit = 100.0 * float(knee_result["config"]["forcing"]["minimum_ess_fraction"])
    ax.text(
        0.02, 0.06,
        f"minimum ESS: {ess:.1f}%  ·  required: {ess_limit:.1f}%",
        transform=ax.transAxes,
        color=NAVY,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)

    return _save(fig, output, "authoritative_summary")


def _sensor_evolution_figure(
    data: dict[str, Any],
    results: list[dict[str, Any]],
    output: Path,
    *,
    label: str,
) -> list[Path]:
    rows = data["rows"]
    unique = _unique_design_groups(rows)
    reference = results[0]
    physics = reference["config"]["physics"]
    measurement = reference["config"]["measurement"]
    box = np.asarray(physics["box"], dtype=float)
    pins = np.asarray(physics["pinning_centers"], dtype=float)
    law = np.asarray(reference["law_anchor"]["eta"], dtype=float).reshape(-1, 2)
    width = float(measurement["sensor_width"])

    panels: list[tuple[str, np.ndarray, str]] = [
        ("Law", law, f"risk anchor {data['frozen_law_risk']:.4f}"),
    ]
    for index, group_label in unique:
        centers = np.asarray(rows[index]["eta"], dtype=float).reshape(-1, 2)
        centers, _ = _match_to_law(centers, law, box)
        subtitle = (
            f"+{rows[index]['extra_risk_percent']:.2f}% actual risk\n"
            f"{100.0 * rows[index]['validation_action_reduction_vs_law']:.1f}% validated gain"
        )
        panels.append((group_label, centers, subtitle))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.15 * len(panels), 3.35), sharex=True, sharey=True, layout="constrained")
    fig.suptitle(
        f"{label} sensor layouts along the Pareto front",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    for ax, (title, centers, subtitle) in zip(np.atleast_1d(axes), panels):
        _draw_domain(ax, centers=centers, pinning=pins, box=box, width=width)
        ax.set_title(title, color=NAVY, pad=20)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=GRAY)
        ax.set_xlabel("x")
    np.atleast_1d(axes)[0].set_ylabel("y")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="X", color="none", markerfacecolor=PURPLE, markeredgecolor=PURPLE, label="pinning center"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", label="sensor center; halo = width"),
        ],
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, -0.035),
    )
    return _save(fig, output, "sensor_layout_evolution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication visuals for a certified skyrmion Pareto sweep")
    parser.add_argument(
        "pareto",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "pareto_authoritative" / "pareto.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-exploratory",
        action="store_true",
        help="allow non-certified input; figures remain explicitly labeled Exploratory",
    )
    args = parser.parse_args()
    data, results = _load(args.pareto, allow_exploratory=args.allow_exploratory)
    output = args.output or args.pareto.parent / "publication_figures"
    output.mkdir(parents=True, exist_ok=True)
    label = "Certified" if data.get("certified") else "Exploratory"

    with plt.rc_context(STYLE):
        paths = []
        paths.extend(_summary_figure(data, results, output, label=label))
        paths.extend(_sensor_evolution_figure(data, results, output, label=label))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
