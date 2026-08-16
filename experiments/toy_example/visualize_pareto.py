"""Visualize the toy example's epsilon_R Pareto sweep.

The authoritative frontier uses exact selection-bank risk and full action.  A
second panel shows the same designs on the independent validation bank, and a
third reports the Full-vs-Law validation reduction with its bootstrap interval.

The input may be ``pareto.csv``, ``pareto.json``, a directory containing either
file, or a single full-run ``result.json`` (useful for previewing the two endpoint
designs before running a full sweep).

Examples
--------
From the repository root::

    .venv/bin/python experiments/toy_example/visualize_pareto.py
    .venv/bin/python experiments/toy_example/visualize_pareto.py \
        experiments/toy_example/outputs/pareto/pareto.csv --show
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "pareto"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the toy epsilon_R Pareto frontier.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="pareto directory/CSV/JSON, or a single result.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output image (default: <input directory>/pareto_frontier.png)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    numeric = (
        "epsilon_r",
        "R_star",
        "R_max",
        "full_theta1_deg",
        "full_theta2_deg",
        "law_theta1_deg",
        "law_theta2_deg",
        "full_R_selection",
        "full_R_excess_selection",
        "full_R_slack_selection",
        "full_L_selection",
        "law_L_selection",
        "law_R_selection",
        "law_A_selection",
        "full_A_selection",
        "law_R_validation",
        "full_R_validation",
        "law_A_validation",
        "full_A_validation",
        "validation_action_reduction",
        "validation_ci_lower",
        "validation_ci_upper",
        "proxy_spearman",
    )
    out = dict(row)
    for key in numeric:
        out[key] = _float_or_none(out.get(key))
    out["full_certified"] = _bool(out.get("full_certified", False))
    return out


def _row_from_result(data: dict[str, Any], *, law_anchor: bool) -> dict[str, Any]:
    certs = data.get("selection_certificates", {})
    full = certs.get("law" if law_anchor else "full", {})
    law = certs.get("law", {})
    validation = data.get("validation", {})
    full_validation = validation.get("law" if law_anchor else "full", {})
    law_validation = validation.get("law", {})
    contrast = data.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    bootstrap = data.get("contrasts", {}).get("full_vs_law_ratio_of_means_bootstrap_95", {})
    screens = data.get("law_screens", {})
    epsilon = 0.0 if law_anchor else float(screens.get("epsilon_r", 0.0))
    eta = full.get("eta_deg", [None, None])
    law_eta = law.get("eta_deg", [None, None])
    return _normalize_row(
        {
            "epsilon_r": epsilon,
            "R_star": screens.get("R_star"),
            "R_max": screens.get("R_star") if law_anchor else screens.get("R_max"),
            "full_theta1_deg": eta[0],
            "full_theta2_deg": eta[1],
            "law_theta1_deg": law_eta[0],
            "law_theta2_deg": law_eta[1],
            "full_R_selection": full.get("R_selection"),
            "full_R_excess_selection": 0.0 if law_anchor else full.get("R_excess_from_star"),
            "full_R_slack_selection": 0.0 if law_anchor else full.get("R_slack_to_max"),
            "full_L_selection": full.get("L_selection"),
            "full_certified": full.get("certified", False),
            "law_L_selection": law.get("L_selection"),
            "law_R_selection": law.get("R_selection"),
            "law_A_selection": law.get("full_action_selection"),
            "full_A_selection": full.get("full_action_selection"),
            "law_R_validation": law_validation.get("law_risk", {}).get("mean"),
            "full_R_validation": full_validation.get("law_risk", {}).get("mean"),
            "law_A_validation": law_validation.get("full_action", {}).get("mean"),
            "full_A_validation": full_validation.get("full_action", {}).get("mean"),
            "validation_action_reduction": 0.0 if law_anchor else contrast.get("ratio_of_means_reduction"),
            "validation_ci_lower": 0.0 if law_anchor else bootstrap.get("lower"),
            "validation_ci_upper": 0.0 if law_anchor else bootstrap.get("upper"),
        }
    )


def _load_json(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if isinstance(payload, list):
        return [_normalize_row(row) for row in payload]
    if isinstance(payload, dict) and payload.get("experiment") == "toy_example":
        if payload.get("smoke"):
            raise SystemExit("A smoke result does not contain a Pareto comparison.")
        return [_row_from_result(payload, law_anchor=True), _row_from_result(payload, law_anchor=False)]
    raise SystemExit(f"Unsupported JSON structure in {path}")


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        csv_path = path / "pareto.csv"
        json_path = path / "pareto.json"
        if csv_path.exists():
            path = csv_path
        elif json_path.exists():
            path = json_path
        else:
            raise SystemExit(
                f"No pareto.csv or pareto.json found in {path}.\n"
                "Run `.venv/bin/python experiments/toy_example/run_pareto.py` first, "
                "or pass a full-run result.json for a two-point preview."
            )
    if not path.exists():
        raise SystemExit(
            f"Input not found: {path}\n"
            "Run `.venv/bin/python experiments/toy_example/run_pareto.py` first."
        )
    if path.suffix.lower() == ".csv":
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = [_normalize_row(row) for row in csv.DictReader(handle)]
        except OSError as exc:
            raise SystemExit(f"Could not read {path}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        rows = _load_json(path)
    else:
        raise SystemExit("Input must be a Pareto directory, .csv, or .json file.")
    rows = [row for row in rows if row.get("epsilon_r") is not None]
    rows.sort(key=lambda row: float(row["epsilon_r"]))
    if not rows:
        raise SystemExit(f"No usable Pareto rows in {path}")
    return rows, path


def _nondominated_indices(rows: list[dict[str, Any]]) -> list[int]:
    """Return points that are not worse in both selection objectives."""
    valid = [
        i
        for i, row in enumerate(rows)
        if row["full_certified"]
        and _finite(row.get("full_R_excess_selection"))
        and _finite(row.get("full_A_selection"))
    ]
    nondominated: list[int] = []
    for i in valid:
        xi = float(rows[i]["full_R_excess_selection"])
        yi = float(rows[i]["full_A_selection"])
        dominated = False
        for j in valid:
            if i == j:
                continue
            xj = float(rows[j]["full_R_excess_selection"])
            yj = float(rows[j]["full_A_selection"])
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            nondominated.append(i)
    return sorted(nondominated, key=lambda i: float(rows[i]["full_R_excess_selection"]))


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#F7F5F1",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#F7F5F1",
        }
    )


def _point_colors(rows: Iterable[dict[str, Any]]) -> tuple[mpl.colors.Normalize, mpl.colors.Colormap, list[Any]]:
    eps = 1.0e3 * np.asarray([float(row["epsilon_r"]) for row in rows], dtype=float)
    vmax = max(float(np.max(eps)), 1.0e-12)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.colormaps["viridis"]
    return norm, cmap, [cmap(norm(value)) for value in eps]


def _plot_selection(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    colors: list[Any],
    nondominated: list[int],
) -> None:
    scale = 1.0e3
    for i, (row, color) in enumerate(zip(rows, colors)):
        x, y = row.get("full_R_excess_selection"), row.get("full_A_selection")
        if not (_finite(x) and _finite(y)):
            continue
        certified = bool(row["full_certified"])
        marker = "o" if i in nondominated else ("X" if certified else "x")
        size = 96 if i in nondominated else 58
        ax.scatter(float(x) * scale, float(y), s=size, marker=marker, color=color, edgecolor="white" if marker == "o" else color, linewidth=1.25, zorder=4)
        label = "Law anchor" if abs(float(row["epsilon_r"])) < 1.0e-15 else rf"$\epsilon_R={float(row['epsilon_r']):g}$"
        ax.annotate(label, (float(x) * scale, float(y)), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#363A40")
        budget = float(row["epsilon_r"]) * scale
        if certified and budget > float(x) * scale + 1.0e-10:
            ax.plot([float(x) * scale, budget], [float(y), float(y)], color=color, lw=0.8, alpha=0.35, zorder=1)
    if len(nondominated) >= 2:
        xs = [float(rows[i]["full_R_excess_selection"]) * scale for i in nondominated]
        ys = [float(rows[i]["full_A_selection"]) for i in nondominated]
        ax.plot(xs, ys, color="#253342", lw=2.0, zorder=2, label="nondominated frontier")
    ax.axvline(0.0, color="#8D939B", lw=0.8)
    ax.grid(color="#AEB2B8", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"Achieved selection risk excess $(R-R^\star)\times 10^3$")
    ax.set_ylabel(r"Exact selection full action $A_{\rm full}$")
    ax.set_title("A   Authoritative selection-bank frontier", loc="left")
    ax.text(
        0.02,
        0.02,
        "circles: nondominated   ×: dominated/uncertified\nthin segments: unused risk budget",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#696E76",
        fontsize=8,
    )
    if len(nondominated) >= 2:
        ax.legend(loc="upper right", frameon=False)


def _plot_validation(ax: plt.Axes, rows: list[dict[str, Any]], colors: list[Any]) -> None:
    valid_indices: list[int] = []
    for i, (row, color) in enumerate(zip(rows, colors)):
        risk, action = row.get("full_R_validation"), row.get("full_A_validation")
        if not (_finite(risk) and _finite(action)):
            continue
        valid_indices.append(i)
        ax.scatter(float(risk), float(action), s=72, color=color, edgecolor="white", linewidth=1.1, zorder=3)
    if len(valid_indices) >= 2:
        ax.plot(
            [float(rows[i]["full_R_validation"]) for i in valid_indices],
            [float(rows[i]["full_A_validation"]) for i in valid_indices],
            color="#7D838C",
            lw=1.0,
            alpha=0.65,
            zorder=1,
        )
    law_risk = next((row.get("law_R_validation") for row in rows if _finite(row.get("law_R_validation"))), None)
    law_action = next((row.get("law_A_validation") for row in rows if _finite(row.get("law_A_validation"))), None)
    if _finite(law_risk):
        ax.axvline(float(law_risk), color="#2878B5", ls="--", lw=1.0, alpha=0.6)
    if _finite(law_action):
        ax.axhline(float(law_action), color="#2878B5", ls="--", lw=1.0, alpha=0.6)
    ax.ticklabel_format(axis="x", style="plain", useOffset=True)
    ax.grid(color="#AEB2B8", lw=0.6, alpha=0.3)
    ax.set_xlabel("Mean validation law risk")
    ax.set_ylabel(r"Mean validation full action $A_{\rm full}$")
    ax.set_title("B   Independent validation check", loc="left")
    ax.text(0.02, 0.02, "dashed lines: Law design", transform=ax.transAxes, color="#696E76", fontsize=8)


def _plot_reduction(ax: plt.Axes, rows: list[dict[str, Any]], colors: list[Any]) -> None:
    plotted = False
    for row, color in zip(rows, colors):
        eps = row.get("epsilon_r")
        value = row.get("validation_action_reduction")
        lower, upper = row.get("validation_ci_lower"), row.get("validation_ci_upper")
        if not (_finite(eps) and _finite(value)):
            continue
        yerr = None
        if _finite(lower) and _finite(upper):
            yerr = np.asarray([[100.0 * (float(value) - float(lower))], [100.0 * (float(upper) - float(value))]])
        ax.errorbar(
            float(eps) * 1.0e3,
            100.0 * float(value),
            yerr=yerr,
            fmt="o",
            ms=7,
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        plotted = True
    ax.axhline(0.0, color="#8D939B", lw=0.9)
    ax.grid(color="#AEB2B8", lw=0.6, alpha=0.3)
    ax.set_xlabel(r"Allowed risk excess $\epsilon_R\times 10^3$")
    ax.set_ylabel("Full-vs-Law action reduction (%)")
    ax.set_title("C   Validation benefit", loc="left")
    ax.text(0.02, 0.02, "bars: bootstrap 95% CI", transform=ax.transAxes, color="#696E76", fontsize=8)
    if not plotted:
        ax.text(0.5, 0.5, "No validation contrasts available", ha="center", va="center", transform=ax.transAxes)


def make_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    _style()
    nondominated = _nondominated_indices(rows)
    norm, cmap, colors = _point_colors(rows)
    fig = plt.figure(figsize=(14.8, 7.7))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.07,
        right=0.91,
        bottom=0.11,
        top=0.85,
        wspace=0.28,
        hspace=0.38,
    )
    _plot_selection(fig.add_subplot(grid[:, 0]), rows, colors, nondominated)
    _plot_validation(fig.add_subplot(grid[0, 1]), rows, colors)
    _plot_reduction(fig.add_subplot(grid[1, 1]), rows, colors)
    cax = fig.add_axes((0.935, 0.18, 0.014, 0.58))
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label(r"risk allowance $\epsilon_R\times 10^3$", labelpad=8)
    colorbar.outline.set_visible(False)
    fig.suptitle("Toy example · law-risk / transport-action Pareto frontier", x=0.07, y=0.95, ha="left", fontsize=19, fontweight="bold")
    fig.text(
        0.07,
        0.895,
        "Each sweep point minimizes exact full MFSI action subject to the lexicographic population and finite-law screens.",
        ha="left",
        color="#555B64",
        fontsize=10.5,
    )
    certified = sum(bool(row["full_certified"]) for row in rows)
    fig.text(
        0.91,
        0.95,
        f"{len(rows)} sweep points  ·  {certified} certified  ·  {len(nondominated)} nondominated",
        ha="right",
        va="top",
        fontsize=8.8,
        color="#6B7078",
    )
    return fig


def main() -> None:
    args = _parse_args()
    rows, resolved_input = _load_rows(args.input)
    base_dir = resolved_input if resolved_input.is_dir() else resolved_input.parent
    output = args.output or base_dir / "pareto_frontier.png"
    output = output.expanduser().resolve()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(rows)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {output}")
    if len(rows) == 2 and abs(float(rows[0]["epsilon_r"])) < 1.0e-15:
        print("note: two-point preview from one run; use run_pareto.py for a full sweep")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
