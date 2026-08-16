"""Create a publication-friendly overview of a saved toy-example run.

The plot is intentionally a post-processing step: it reads ``result.json`` and
does not rerun any optimization or Tesseract solves.  The population panels are
drawn from the analytic mixture in :mod:`domain`, reimplemented here with NumPy
so the visualizer only needs NumPy and Matplotlib.

Examples
--------
From the repository root::

    .venv/bin/python experiments/toy_example/visualize.py
    .venv/bin/python experiments/toy_example/visualize.py \
        experiments/toy_example/outputs/run/result.json --show
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
DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "run" / "result.json"

DESIGN_ORDER = ("population", "law", "tangent", "full")
DISPLAY_NAMES = {
    "population": "Population",
    "law": "Law",
    "tangent": "Tangent",
    "full": "Full MFSI",
}
COLORS = {
    "population": "#5B6573",
    "law": "#2878B5",
    "tangent": "#E59F00",
    "full": "#C43C39",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a completed toy-example result without rerunning it."
    )
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULT,
        help=f"saved result.json (default: {DEFAULT_RESULT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output image (default: <result directory>/toy_visualization.png)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args()


def _load_result(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Result not found: {path}\nRun `python run.py` first or pass a result.json path."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if data.get("experiment") != "toy_example":
        raise SystemExit(f"Expected a toy_example result, got {data.get('experiment')!r}")
    if data.get("smoke"):
        raise SystemExit("The overview requires a full run; the supplied result is a smoke run.")
    return data


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "#F7F5F1",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#F7F5F1",
        }
    )


def _mixture_density(
    xx: np.ndarray,
    yy: np.ndarray,
    t: float,
    alpha: float,
    *,
    radius: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate the three-component, antipodally symmetric toy density."""
    s2 = sigma * sigma
    norm = 1.0 / (2.0 * np.pi * s2)

    def component(angle: float) -> np.ndarray:
        mx, my = radius * np.cos(angle), radius * np.sin(angle)
        plus = np.exp(-0.5 * ((xx - mx) ** 2 + (yy - my) ** 2) / s2)
        minus = np.exp(-0.5 * ((xx + mx) ** 2 + (yy + my) ** 2) / s2)
        return 0.5 * norm * (plus + minus)

    return (
        (1.0 - t) ** 2 * component(0.0)
        + 2.0 * t * (1.0 - t) * component(alpha)
        + t**2 * component(0.5 * np.pi)
    )


def _designs(data: dict[str, Any]) -> dict[str, np.ndarray]:
    selection = data.get("selection", {})
    designs: dict[str, np.ndarray] = {}
    for name in DESIGN_ORDER:
        value = selection.get(f"{name}_optimum_deg")
        if isinstance(value, list) and len(value) == 2 and all(_finite(x) for x in value):
            designs[name] = np.asarray(value, dtype=float)
    return designs


def _draw_population_strip(
    fig: plt.Figure,
    spec: mpl.gridspec.SubplotSpec,
    cfg: dict[str, Any],
) -> None:
    pop = cfg["population"]
    half_width = float(pop["domain_half_width"])
    radius = float(pop["radius"])
    sigma = float(pop["sigma"])
    alpha = np.deg2rad(0.5 * (float(pop["alpha_min_deg"]) + float(pop["alpha_max_deg"])))
    grid = np.linspace(-half_width, half_width, 280)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    times = np.linspace(0.0, 1.0, 5)
    densities = [
        _mixture_density(xx, yy, float(t), alpha, radius=radius, sigma=sigma)
        for t in times
    ]
    vmax = max(float(np.max(z)) for z in densities)
    levels = np.linspace(0.0, vmax, 18)
    inner = spec.subgridspec(1, len(times), wspace=0.06)
    axes: list[plt.Axes] = []
    image = None
    for index, (t, density) in enumerate(zip(times, densities)):
        ax = fig.add_subplot(inner[0, index])
        axes.append(ax)
        image = ax.contourf(xx, yy, density, levels=levels, cmap="magma", extend="max")
        ax.contour(xx, yy, density, levels=levels[3::3], colors="white", linewidths=0.35, alpha=0.38)
        ax.set_aspect("equal")
        ax.set_xlim(-2.35, 2.35)
        ax.set_ylim(-2.35, 2.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t:.2g}", pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].text(
        -0.08,
        1.13,
        rf"A   Hidden population path   ·   representative $\alpha={np.rad2deg(alpha):.0f}^\circ$",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.012, aspect=28)
        cbar.set_label("density", labelpad=5)
        cbar.outline.set_visible(False)


def _draw_sensor_geometry(
    ax: plt.Axes,
    data: dict[str, Any],
    designs: dict[str, np.ndarray],
) -> None:
    cfg = data["config"]
    pop = cfg["population"]
    measurement = cfg["measurement"]
    radius = float(measurement["sensor_radius"])
    width = float(measurement["sensor_width"])
    alpha = np.deg2rad(0.5 * (float(pop["alpha_min_deg"]) + float(pop["alpha_max_deg"])))
    lim = max(float(pop["domain_half_width"]), radius + 2.0 * width)
    grid = np.linspace(-lim, lim, 260)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    density = _mixture_density(
        xx,
        yy,
        0.5,
        alpha,
        radius=float(pop["radius"]),
        sigma=float(pop["sigma"]),
    )
    ax.contourf(xx, yy, density, levels=16, cmap="Greys", alpha=0.23)
    circle = plt.Circle((0.0, 0.0), radius, fill=False, color="#8C9199", lw=1.1, ls="--")
    ax.add_patch(circle)

    # Merge exactly coincident designs (Population and Law often agree here).
    groups: list[tuple[list[str], np.ndarray]] = []
    for name in DESIGN_ORDER:
        eta = designs.get(name)
        if eta is None:
            continue
        match = next((group for group in groups if np.allclose(eta, group[1], atol=1.0e-7)), None)
        if match is not None:
            match[0].append(name)
        else:
            groups.append(([name], eta))

    handles: list[Line2D] = []
    for names, eta in groups:
        # Prefer the more downstream objective's color for a shared optimum.
        name = names[-1]
        points = radius * np.column_stack((np.cos(np.deg2rad(eta)), np.sin(np.deg2rad(eta))))
        color = COLORS[name]
        ax.plot(points[:, 0], points[:, 1], color=color, lw=2.0, alpha=0.86, zorder=4)
        ax.scatter(
            points[:, 0], points[:, 1], s=68, color=color, edgecolor="white", linewidth=1.3, zorder=5
        )
        for sensor_index, (x, y) in enumerate(points, start=1):
            ax.text(x, y, str(sensor_index), color="white", fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=6)
        for x, y in points:
            ax.add_patch(plt.Circle((x, y), width, fill=False, color=color, lw=0.8, alpha=0.45))
        label = " / ".join(DISPLAY_NAMES[item] for item in names)
        handles.append(Line2D([0], [0], color=color, marker="o", lw=2, markersize=5, label=label))

    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=2, handlelength=1.5)
    ax.axhline(0, color="#9FA3AA", lw=0.55, alpha=0.5)
    ax.axvline(0, color="#9FA3AA", lw=0.55, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("B   Selected sensor pairs", loc="left")
    ax.text(
        0.02,
        0.98,
        "Dashed ring: sensor radius\nThin rings: 1σ sensor width",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#60656F",
        fontsize=8,
    )


def _draw_selection_tradeoff(ax: plt.Axes, data: dict[str, Any]) -> None:
    certs = data.get("selection_certificates", {})
    screens = data.get("law_screens", {})
    r_star = float(screens["R_star"])
    epsilon_r = float(screens["epsilon_r"])
    scale = 1.0e3
    plotted = 0
    x_values: list[float] = []
    population_law_overlap = False
    if "population" in certs and "law" in certs:
        pc, lc = certs["population"], certs["law"]
        population_law_overlap = all(
            _finite(pc.get(key))
            and _finite(lc.get(key))
            and np.isclose(float(pc[key]), float(lc[key]), rtol=0.0, atol=1.0e-12)
            for key in ("R_selection", "full_action_selection")
        )
    for name in DESIGN_ORDER:
        cert = certs.get(name, {})
        risk = cert.get("R_selection")
        action = cert.get("full_action_selection")
        if not (_finite(risk) and _finite(action)):
            continue
        x = (float(risk) - r_star) * scale
        x_values.append(x)
        marker = "o" if cert.get("certified", False) else "X"
        ax.scatter(x, float(action), s=80, marker=marker, color=COLORS[name], edgecolor="white", linewidth=1.2, zorder=4)
        if not (population_law_overlap and name == "population"):
            label = "Population / Law" if population_law_overlap and name == "law" else DISPLAY_NAMES[name]
            offsets = {"law": (6, 5), "tangent": (6, 5), "full": (6, -13)}
            ax.annotate(
                label,
                (x, float(action)),
                xytext=offsets.get(name, (6, 5)),
                textcoords="offset points",
                color=COLORS[name],
                fontsize=8.5,
            )
        plotted += 1
    extent = x_values + [epsilon_r * scale, 0.0]
    lo, hi = min(extent), max(extent)
    margin = max(0.04 * (hi - lo), 0.012)
    ax.set_xlim(lo - margin, hi + margin)
    ax.axvspan(lo - margin, epsilon_r * scale, color="#4C9F70", alpha=0.08, zorder=0)
    ax.axvline(epsilon_r * scale, color="#4C9F70", ls="--", lw=1.2, label=r"risk budget $\epsilon_R$")
    ax.axvline(0.0, color="#8D939B", lw=0.8, zorder=0)
    ax.grid(color="#AEB2B8", lw=0.6, alpha=0.28)
    ax.set_xlabel(r"Selection risk excess $(R-R^\star)\times 10^3$")
    ax.set_ylabel(r"Selection full action $A_{\rm full}$")
    ax.set_title("C   Information–transport trade-off", loc="left")
    if plotted:
        ax.legend(loc="upper right", frameon=False)


def _draw_validation(ax: plt.Axes, data: dict[str, Any]) -> None:
    validation = data.get("validation", {})
    available = [name for name in DESIGN_ORDER if _finite(validation.get(name, {}).get("full_action", {}).get("mean"))]
    if not available:
        ax.text(0.5, 0.5, "No validation summary available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = np.arange(len(available), dtype=float)
    means = np.asarray([validation[name]["full_action"]["mean"] for name in available], dtype=float)
    ses = np.asarray([validation[name]["full_action"].get("se", np.nan) for name in available], dtype=float)
    colors = [COLORS[name] for name in available]
    for xi, mean, se, color in zip(x, means, ses, colors):
        ax.errorbar(
            xi,
            mean,
            yerr=1.96 * se,
            fmt="o",
            ms=8.5,
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=5,
            capthick=1.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
    law_mean = validation.get("law", {}).get("full_action", {}).get("mean")
    if _finite(law_mean):
        ax.axhline(float(law_mean), color=COLORS["law"], ls="--", lw=1.0, alpha=0.65)
    ax.set_xticks(x, [DISPLAY_NAMES[name].replace(" ", "\n") for name in available])
    ax.set_ylabel(r"Validation full action $A_{\rm full}$")
    ax.set_title("D   Independent validation", loc="left")
    ax.grid(axis="y", color="#AEB2B8", lw=0.6, alpha=0.3)
    contrast = data.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    bootstrap = data.get("contrasts", {}).get("full_vs_law_ratio_of_means_bootstrap_95", {})
    reduction = contrast.get("ratio_of_means_reduction")
    lower, upper = bootstrap.get("lower"), bootstrap.get("upper")
    if all(_finite(value) for value in (reduction, lower, upper)):
        ax.text(
            0.98,
            0.98,
            "Full vs Law\n"
            f"{100.0 * float(reduction):.1f}% less action\n"
            f"95% bootstrap CI [{100.0 * float(lower):.1f}, {100.0 * float(upper):.1f}]%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#42464D",
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#D3D0C8", "alpha": 0.9},
        )
    ax.text(
        0.02,
        0.02,
        "points: mean   bars: 95% normal CI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#70747B",
    )


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    designs = _designs(data)
    fig = plt.figure(figsize=(15.8, 9.6), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.95, 1.25),
        left=0.055,
        right=0.97,
        bottom=0.075,
        top=0.84,
        hspace=0.27,
    )
    _draw_population_strip(fig, outer[0, 0], data["config"])
    bottom = outer[1, 0].subgridspec(1, 3, width_ratios=(1.04, 1.0, 1.0), wspace=0.28)
    _draw_sensor_geometry(fig.add_subplot(bottom[0, 0]), data, designs)
    _draw_selection_tradeoff(fig.add_subplot(bottom[0, 1]), data)
    _draw_validation(fig.add_subplot(bottom[0, 2]), data)

    cfg = data["config"]
    fig.suptitle("Toy example · what the sensors learn and what transport costs", x=0.055, y=0.965, ha="left", fontsize=19, fontweight="bold")
    fig.text(
        0.055,
        0.925,
        "Two Gaussian lobes rotate from the horizontal to the vertical; sensor placement balances finite-law fidelity against full MFSI action.",
        ha="left",
        color="#555B64",
        fontsize=10.5,
    )
    fig.text(
        0.97,
        0.965,
        f"n={cfg['measurement']['finite_n']}  ·  acquisitions={cfg['measurement']['acquisition_k']}  ·  validation trials={cfg['randomness']['validation_trials']}",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#6B7078",
    )
    return fig


def main() -> None:
    args = _parse_args()
    result_path = args.result.expanduser().resolve()
    output = args.output or result_path.with_name("toy_visualization.png")
    output = output.expanduser().resolve()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _load_result(result_path)
    fig = make_figure(data)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
