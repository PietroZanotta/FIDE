"""Create a publication-friendly overview of a completed vortices run.

The visualizer is deliberately pure post-processing: it reads the saved truth
bank, result JSON, candidate summary, and validation rows.  It never trains a
reference model or reruns an optimization, I-projection, or Poisson solve.

Examples
--------
From the repository root::

    .venv/bin/python experiments/vortices/visualize.py
    .venv/bin/python experiments/vortices/visualize.py \
        experiments/vortices/outputs/run/result.json --show
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT = SCRIPT_DIR / "outputs" / "run" / "result.json"
DESIGN_ORDER = ("population", "law", "tangent", "full")
DISPLAY_NAMES = {
    "population": "Population oracle",
    "law": "Finite law",
    "tangent": "Tangent MFSI",
    "full": "Full MFSI",
}
SHORT_NAMES = {
    "population": "Population",
    "law": "Law",
    "tangent": "Tangent",
    "full": "Full",
}
COLORS = {
    "population": "#69717D",
    "law": "#2C7FB8",
    "tangent": "#E39D24",
    "full": "#D1495B",
}
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "vortex_density",
    ("#F5F1E8", "#D9C7A4", "#D77A61", "#8D3D55", "#272442"),
)
OCCUPANCY_CMAP = LinearSegmentedColormap.from_list(
    "vortex_occupancy", ("#FBFAF6", "#DDE5E6", "#738B96", "#263946")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a completed vortices run without rerunning it."
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
        help="output image (default: <result directory>/vortices_visualization.png)",
    )
    parser.add_argument("--dpi", type=int, default=210, help="output resolution")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Result not found: {path}\nRun the vortices experiment first or pass a result.json."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_inputs(result_path: Path):
    data = _load_json(result_path)
    if data.get("experiment") != "vortices_double_gyre":
        raise SystemExit(
            f"Expected a vortices_double_gyre result, got {data.get('experiment')!r}."
        )
    if data.get("smoke"):
        raise SystemExit("The dashboard requires a full run, not a smoke result.")
    run_dir = result_path.parent
    truth_name = data.get("truth", {}).get("truth_bank", "truth_bank.npz")
    truth_path = run_dir / truth_name
    try:
        with np.load(truth_path) as bank:
            times = np.asarray(bank["times"], dtype=np.float64)
            particles = np.asarray(bank["particles"], dtype=np.float64)
    except (FileNotFoundError, KeyError, OSError) as exc:
        raise SystemExit(f"Could not load truth bank {truth_path}: {exc}") from exc
    candidates = _load_csv(run_dir / "result.candidate_summary.csv")
    validation = _load_csv(run_dir / "result.validation_trials.csv")
    return data, times, particles, candidates, validation


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.1,
            "figure.facecolor": "#F4F1EA",
            "axes.facecolor": "#FBFAF6",
            "savefig.facecolor": "#F4F1EA",
        }
    )


def _smooth2d(values: np.ndarray, passes: int = 2) -> np.ndarray:
    """Small dependency-free separable blur for empirical histograms."""
    out = np.asarray(values, dtype=np.float64)
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for _ in range(int(passes)):
        padded = np.pad(out, ((0, 0), (2, 2)), mode="edge")
        out = sum(kernel[k] * padded[:, k : k + out.shape[1]] for k in range(5))
        padded = np.pad(out, ((2, 2), (0, 0)), mode="edge")
        out = sum(kernel[k] * padded[k : k + out.shape[0], :] for k in range(5))
    return out


def _density(particles: np.ndarray, *, nx: int = 180, ny: int = 90) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        particles[:, 0], particles[:, 1], bins=(nx, ny), range=((0.0, 2.0), (0.0, 1.0))
    )
    return _smooth2d(hist.T, passes=2)


def _double_gyre_velocity(
    xx: np.ndarray, yy: np.ndarray, t: float, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = float(cfg.get("amplitude", 0.1))
    epsilon = float(cfg.get("epsilon", 0.25))
    horizon = float(cfg.get("horizon", 10.0))
    period = float(cfg.get("period", 10.0))
    tau = horizon * float(t)
    omega = 2.0 * np.pi / period
    a = epsilon * np.sin(omega * tau)
    b = 1.0 - 2.0 * a
    f = a * xx * xx + b * xx
    dfdx = 2.0 * a * xx + b
    vx = -np.pi * amplitude * np.sin(np.pi * f) * np.cos(np.pi * yy)
    vy = np.pi * amplitude * np.cos(np.pi * f) * np.sin(np.pi * yy) * dfdx
    return horizon * vx, horizon * vy


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color="#262A30",
    )


def _draw_truth_strip(
    fig: plt.Figure,
    spec: mpl.gridspec.SubplotSpec,
    data: dict[str, Any],
    times: np.ndarray,
    particles: np.ndarray,
) -> None:
    target_times = np.linspace(float(times[0]), float(times[-1]), 5)
    indices = [int(np.argmin(np.abs(times - t))) for t in target_times]
    densities = [_density(particles[index]) for index in indices]
    vmax = float(np.quantile(np.concatenate([z.ravel() for z in densities]), 0.9985))
    vmax = max(vmax, 1.0)
    norm = PowerNorm(gamma=0.52, vmin=0.0, vmax=vmax)
    inner = spec.subgridspec(1, len(indices), wspace=0.055)
    axes: list[plt.Axes] = []
    for panel, (index, density) in enumerate(zip(indices, densities, strict=True)):
        ax = fig.add_subplot(inner[0, panel])
        axes.append(ax)
        ax.imshow(
            density,
            origin="lower",
            extent=(0.0, 2.0, 0.0, 1.0),
            cmap=FLOW_CMAP,
            norm=norm,
            interpolation="bilinear",
            aspect="equal",
        )
        gx = np.linspace(0.02, 1.98, 31)
        gy = np.linspace(0.02, 0.98, 17)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        u, v = _double_gyre_velocity(xx, yy, float(times[index]), data["config"]["truth"])
        ax.streamplot(
            gx,
            gy,
            u,
            v,
            density=0.52,
            linewidth=0.38,
            arrowsize=0.45,
            color=(1.0, 1.0, 1.0, 0.34),
        )
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$t={times[index]:.2f}$", pad=4, color="#343941")
        for spine in ax.spines.values():
            spine.set_visible(False)
    _panel_label(axes[0], "A   Hidden double-gyre population")
    axes[-1].text(
        0.98,
        0.04,
        "color: particle density\nlines: instantaneous flow",
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#202532", "ec": "none", "alpha": 0.58},
    )


def _draw_sensor_strip(
    fig: plt.Figure,
    spec: mpl.gridspec.SubplotSpec,
    data: dict[str, Any],
    particles: np.ndarray,
) -> None:
    mean_density = np.mean(
        np.stack([_density(particles[index]) for index in range(0, len(particles), 2)]), axis=0
    )
    vmax = max(float(np.quantile(mean_density, 0.998)), 1.0)
    inner = spec.subgridspec(1, len(DESIGN_ORDER), wspace=0.12)
    measurement = data["config"]["measurement"]
    width = float(measurement.get("sensor_width", 0.12))
    margin = float(measurement.get("boundary_margin", 2.0 * width))
    validation = data.get("validation", {})
    axes: list[plt.Axes] = []
    for panel, name in enumerate(DESIGN_ORDER):
        ax = fig.add_subplot(inner[0, panel])
        axes.append(ax)
        ax.imshow(
            mean_density,
            origin="lower",
            extent=(0.0, 2.0, 0.0, 1.0),
            cmap=OCCUPANCY_CMAP,
            norm=PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax),
            interpolation="bilinear",
            aspect="equal",
        )
        ax.add_patch(
            Rectangle(
                (margin, margin),
                2.0 - 2.0 * margin,
                1.0 - 2.0 * margin,
                fill=False,
                edgecolor=(1.0, 1.0, 1.0, 0.56),
                linewidth=0.75,
                linestyle=(0, (3, 2)),
            )
        )
        centers = np.asarray(data.get("selection_centers", {}).get(name, []), dtype=float)
        color = COLORS[name]
        for sensor_index, center in enumerate(centers, start=1):
            x, y = map(float, center)
            ax.add_patch(Circle((x, y), 2.0 * width, color=color, alpha=0.09, ec="none"))
            ax.add_patch(
                Circle((x, y), width, fill=False, color=color, alpha=0.95, linewidth=1.25)
            )
            ax.scatter(
                [x], [y], s=40, color=color, edgecolor="white", linewidth=1.0, zorder=5
            )
            ax.text(
                x,
                y,
                str(sensor_index),
                color="white",
                fontsize=6.3,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )
        block = validation.get(name, {})
        valid = 100.0 * float(block.get("valid_fraction", 0.0))
        risk = block.get("law_risk", {}).get("mean")
        note = f"valid {valid:.0f}%"
        if _finite(risk):
            note += f"  ·  R={float(risk):.4f}"
        ax.text(
            0.02,
            0.035,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.24", "fc": "#202631", "ec": "none", "alpha": 0.67},
        )
        ax.set_title(DISPLAY_NAMES[name], color=color, pad=4)
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    _panel_label(axes[0], "B   Four objectives, four sensor geometries")
    axes[-1].text(
        0.98,
        1.09,
        "rings: 1σ sensing width   ·   dashed box: admissible centers",
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#676C74",
    )


def _candidate_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("design", ""): row for row in rows}


def _draw_selection_tradeoff(
    ax: plt.Axes, data: dict[str, Any], candidates: list[dict[str, str]]
) -> None:
    screens = data["law_screens"]
    r_star = float(screens["R_star"])
    epsilon = float(screens["R_max"]) - r_star
    lookup = _candidate_lookup(candidates)
    ax.axvspan(-0.08, 1.0, color="#4B9A73", alpha=0.085, zorder=0)
    ax.axvline(1.0, color="#438A68", lw=1.25, ls="--")
    for name in ("law", "tangent", "full"):
        cert = data.get("selection_certificates", {}).get(name, {})
        action = lookup.get(name, {}).get("full_action_selection")
        risk = cert.get("R_selection")
        if not (_finite(risk) and _finite(action)):
            continue
        x = (float(risk) - r_star) / max(epsilon, 1.0e-15)
        y = float(action)
        ax.scatter(
            x,
            y,
            s=105,
            color=COLORS[name],
            edgecolor="white",
            linewidth=1.3,
            zorder=4,
        )
        offsets = {"law": (7, 7), "tangent": (7, 5), "full": (7, -14)}
        ax.annotate(
            f"{SHORT_NAMES[name]}  A={y:.2f}",
            (x, y),
            xytext=offsets[name],
            textcoords="offset points",
            color=COLORS[name],
            fontsize=8.2,
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.04,
        "green region satisfies the finite-law budget",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#47765F",
        fontsize=7.5,
    )
    ax.set_xlim(-0.08, 1.08)
    ax.set_xlabel(r"Used finite-law budget  $(R-R^\star)/\epsilon_R$")
    ax.set_ylabel(r"Selection full action $A_{\rm full}$")
    ax.set_title("C   Certified selection trade-off", loc="left")
    ax.grid(color="#AEB2B8", lw=0.55, alpha=0.27)


def _validation_values(
    rows: list[dict[str, str]], design: str, metric: str
) -> np.ndarray:
    values = []
    for row in rows:
        if row.get("design") != design or row.get("valid", "").lower() != "true":
            continue
        value = row.get(metric)
        if _finite(value) and float(value) > 0.0:
            values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _draw_validation_actions(
    ax: plt.Axes, data: dict[str, Any], rows: list[dict[str, str]]
) -> None:
    order = ("law", "tangent", "full")
    rng = np.random.default_rng(20260816)
    tick_labels: list[str] = []
    for position, name in enumerate(order, start=1):
        values = _validation_values(rows, name, "full_action")
        total = sum(row.get("design") == name for row in rows)
        tick_labels.append(f"{SHORT_NAMES[name]}\n{values.size}/{total} valid")
        if values.size == 0:
            continue
        box = ax.boxplot(
            [values],
            positions=[position],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": COLORS[name], "linewidth": 1.1},
            capprops={"color": COLORS[name], "linewidth": 1.1},
        )
        box["boxes"][0].set(facecolor=COLORS[name], alpha=0.76, edgecolor=COLORS[name])
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            s=9,
            color=COLORS[name],
            alpha=0.3,
            linewidth=0,
            zorder=2,
        )
        mean = float(np.mean(values))
        ax.scatter(
            [position], [mean], marker="D", s=34, color="#20252D", edgecolor="white", lw=0.8, zorder=5
        )
        ax.annotate(
            f"mean {mean:.2f}\nmedian {np.median(values):.2f}",
            (position, mean),
            xytext=(7, 1),
            textcoords="offset points",
            fontsize=7.2,
            color="#3F444C",
        )
    contrast = data.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    reduction = contrast.get("ratio_of_means_reduction")
    if _finite(reduction):
        ax.text(
            0.97,
            0.96,
            f"Full uses {100.0 * float(reduction):.1f}% less mean action than Law",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["full"],
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#DDD8CE", "alpha": 0.9},
        )
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(order) + 1), tick_labels)
    ax.set_ylabel(r"Validation full action $A_{\rm full}$  (log scale)")
    ax.set_title("D   Independent-trial action", loc="left")
    ax.grid(axis="y", which="both", color="#AEB2B8", lw=0.55, alpha=0.27)
    legend = Line2D(
        [0], [0], marker="D", color="none", markerfacecolor="#20252D", markeredgecolor="white", markersize=6, label="mean"
    )
    ax.legend(handles=[legend], loc="lower left", frameon=False)


def _draw_timing(ax: plt.Axes, data: dict[str, Any]) -> None:
    stage = data.get("selection_timings_seconds", {})
    phase = data.get("timings_seconds", {}).get("phases_seconds", {})
    entries = [
        ("Setup", float(phase.get("setup_and_cached_inputs", 0.0)), "#87919D"),
        ("Population", float(stage.get("stage_1_population", 0.0)), COLORS["population"]),
        ("Finite law", float(stage.get("stage_2_finite_law", 0.0)), COLORS["law"]),
        ("Tangent", float(stage.get("stage_3_tangent", 0.0)), COLORS["tangent"]),
        ("Full MFSI", float(stage.get("stage_4_full", 0.0)), COLORS["full"]),
        ("Validation", float(phase.get("validation_and_certification", 0.0)), "#7768AE"),
    ]
    entries = [entry for entry in entries if entry[1] > 0.0]
    labels = [entry[0] for entry in entries]
    values = np.asarray([entry[1] for entry in entries], dtype=float) / 60.0
    colors = [entry[2] for entry in entries]
    y = np.arange(len(entries))
    ax.barh(y, values, color=colors, alpha=0.84, height=0.62)
    for yi, value, (_, seconds, _) in zip(y, values, entries, strict=True):
        ax.text(value + 0.06, yi, _format_duration(seconds), va="center", fontsize=7.8, color="#41464E")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall time (minutes)")
    ax.set_title("E   Compute profile", loc="left")
    ax.grid(axis="x", color="#AEB2B8", lw=0.55, alpha=0.27)
    total = data.get("timings_seconds", {}).get("total_seconds")
    if _finite(total):
        ax.text(
            0.98,
            0.96,
            f"total  {_format_duration(float(total))}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color="#363B43",
        )


def make_figure(
    data: dict[str, Any],
    times: np.ndarray,
    particles: np.ndarray,
    candidates: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(16.4, 11.0), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.78, 0.83, 1.18),
        left=0.05,
        right=0.975,
        bottom=0.07,
        top=0.875,
        hspace=0.34,
    )
    _draw_truth_strip(fig, outer[0, 0], data, times, particles)
    _draw_sensor_strip(fig, outer[1, 0], data, particles)
    bottom = outer[2, 0].subgridspec(1, 3, width_ratios=(1.05, 1.12, 0.9), wspace=0.3)
    _draw_selection_tradeoff(fig.add_subplot(bottom[0, 0]), data, candidates)
    _draw_validation_actions(fig.add_subplot(bottom[0, 1]), data, validation_rows)
    _draw_timing(fig.add_subplot(bottom[0, 2]), data)

    cfg = data["config"]
    contrast = data.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    reduction = contrast.get("ratio_of_means_reduction")
    result_line = (
        f"The full-action design preserves the declared law screen and cuts validation action by "
        f"{100.0 * float(reduction):.1f}% relative to the law-only design."
        if _finite(reduction)
        else "Sensor placement trades finite-law fidelity against transport regularity."
    )
    fig.suptitle(
        "Vortices · learning where to observe a double gyre",
        x=0.05,
        y=0.968,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#22262C",
    )
    fig.text(0.05, 0.932, result_line, ha="left", fontsize=10.6, color="#50565F")
    fig.text(
        0.975,
        0.968,
        f"{cfg['truth']['truth_particles']:,} truth particles  ·  "
        f"{cfg['measurement']['n_sensors']} sensors  ·  "
        f"{cfg['randomness']['validation_trials']} validation trials",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#686E77",
    )
    fig.text(
        0.05,
        0.025,
        "Population is an exact-moment oracle baseline; its finite/noisy validity is diagnostic. "
        "Law, Tangent, and Full are independently validated on the same disjoint trial bank.",
        ha="left",
        fontsize=7.8,
        color="#6B7078",
    )
    return fig


def main() -> None:
    args = _parse_args()
    result_path = args.result.expanduser().resolve()
    output = args.output or result_path.with_name("vortices_visualization.png")
    output = output.expanduser().resolve()
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    data, times, particles, candidates, validation = _load_inputs(result_path)
    fig = make_figure(data, times, particles, candidates, validation)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {output}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
