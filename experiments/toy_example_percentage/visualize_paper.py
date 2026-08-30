"""Create the paper-style toy population / corrected-law / sensor figure.

This is a deterministic post-processing script.  It reads the authoritative
5% Full geometry and frozen validation/reference banks, but does not rerun
training, design optimization, or validation.  Every invocation writes both a
PNG and a vector PDF.

From the repository root::

    .venv/bin/python experiments/toy_example_percentage/visualize_paper.py
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import root


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto" / "corrected_nested_full_sweep.json"
DEFAULT_REFERENCE_BANK = SCRIPT_DIR / "outputs" / "pareto" / "frozen_inputs" / "reference_bank.npz"
DEFAULT_VALIDATION_BANK = SCRIPT_DIR / "outputs" / "pareto" / "frozen_inputs" / "validation_bank.npz"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "figures" / "toy_population_correction_sensors"

TIME_INDICES = (0, 6, 14, 20)
SENSOR_COLORS = ("#16A6A1", "#F28E5B")
PAPER_BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#FFFFFF"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pareto", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=DEFAULT_VALIDATION_BANK)
    parser.add_argument("--allowance", type=float, default=5.0)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="output path without extension; .png and .pdf are both written",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.facecolor": PANEL_BACKGROUND,
            "axes.edgecolor": "#C9C3B8",
            "axes.linewidth": 0.65,
            "figure.facecolor": PAPER_BACKGROUND,
            "savefig.facecolor": PAPER_BACKGROUND,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_geometry(pareto: dict[str, Any], allowance: float) -> np.ndarray:
    matches = [
        row for row in pareto.get("rows", [])
        if math.isclose(float(row["allowance_percent"]), allowance, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Pareto row at {allowance:g}%, found {len(matches)}")
    return np.deg2rad(np.asarray(matches[0]["geometry_deg"], dtype=np.float64))


def _sensor_features(points: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    displacement = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(displacement**2, axis=-1) / width**2)


def _mixture_density(
    xx: np.ndarray,
    yy: np.ndarray,
    time: float,
    alpha: float,
    *,
    radius: float,
    sigma: float,
) -> np.ndarray:
    variance = sigma**2
    normalizer = 1.0 / (2.0 * np.pi * variance)

    def component(angle: float) -> np.ndarray:
        center = radius * np.asarray([np.cos(angle), np.sin(angle)])
        plus = np.exp(-0.5 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / variance)
        minus = np.exp(-0.5 * ((xx + center[0]) ** 2 + (yy + center[1]) ** 2) / variance)
        return 0.5 * normalizer * (plus + minus)

    return (
        (1.0 - time) ** 2 * component(0.0)
        + 2.0 * time * (1.0 - time) * component(alpha)
        + time**2 * component(0.5 * np.pi)
    )


def _project_weights(
    points: np.ndarray,
    base_weights: np.ndarray,
    centers: np.ndarray,
    width: float,
    target: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    features = _sensor_features(points, centers, width)
    base = np.asarray(base_weights, dtype=np.float64).copy()
    base /= np.sum(base)
    log_base = np.full_like(base, -np.inf)
    positive = base > 0.0
    log_base[positive] = np.log(base[positive])

    def weights(lam: np.ndarray) -> np.ndarray:
        logits = log_base + features @ lam
        logits -= np.max(logits[positive])
        tilted = np.exp(logits)
        return tilted / np.sum(tilted)

    def residual(lam: np.ndarray) -> np.ndarray:
        return weights(lam) @ features - target

    solution = root(residual, np.zeros(features.shape[1], dtype=np.float64))
    projected = weights(solution.x)
    maximum_residual = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or maximum_residual > 1.0e-8:
        raise RuntimeError(
            "sensor-moment projection failed: "
            f"success={solution.success}, residual={maximum_residual:.3e}"
        )
    base_ess = 1.0 / np.sum(base**2)
    projected_ess = 1.0 / np.sum(projected**2)
    return projected, projected_ess / base_ess, solution.x


def _particle_density(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    half_width: float,
    grid_n: int,
    bandwidth: float,
) -> np.ndarray:
    edges = np.linspace(-half_width, half_width, grid_n + 1)
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=(edges, edges), weights=weights
    )
    cell_width = 2.0 * half_width / grid_n
    smoothed = gaussian_filter(
        histogram,
        sigma=bandwidth / cell_width,
        mode="constant",
        truncate=4.0,
    )
    mass = float(np.sum(smoothed))
    if mass <= 0.0:
        raise RuntimeError("rasterized corrected law has zero mass")
    return smoothed / (mass * cell_width**2)


def _clean_density_axis(ax: plt.Axes, half_width: float) -> None:
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _sensor_cmap(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sensor",
        [PANEL_BACKGROUND, mpl.colors.to_rgba(color, 0.28), color, "#151A21"],
    )


def _add_sensor_marker(
    ax: plt.Axes,
    center: np.ndarray,
    width: float,
    color: str,
    *,
    compact: bool,
) -> None:
    ax.add_patch(
        plt.Circle(
            center,
            width,
            fill=False,
            color=color,
            lw=0.8 if compact else 1.05,
            ls=(0, (2.2, 2.2)),
            alpha=0.9,
            zorder=5,
        )
    )
    ax.scatter(
        center[0], center[1],
        s=18 if compact else 28,
        marker="x",
        color=color,
        linewidth=1.4,
        zorder=6,
    )


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_json(args.config)
    pareto = _load_json(args.pareto)
    angles = _full_geometry(pareto, args.allowance)
    measurement = cfg["measurement"]
    population = cfg["population"]
    sensor_radius = float(measurement["sensor_radius"])
    sensor_width = float(measurement["sensor_width"])
    sensor_centers = sensor_radius * np.column_stack((np.cos(angles), np.sin(angles)))

    with np.load(args.validation_bank, allow_pickle=False) as validation:
        alphas = np.asarray(validation["alphas"], dtype=np.float64)
        trial = int(np.argmin(np.abs(alphas - np.deg2rad(45.0))))
        alpha = float(alphas[trial])
        hidden_masses = np.asarray(validation["masses"][trial], dtype=np.float64)
        sample_indices = np.asarray(validation["sample_indices"][trial], dtype=np.int64)
        detector_z = np.asarray(validation["detector_z"][trial], dtype=np.float64)
        acquisition_indices = np.asarray(validation["acquisition_indices"], dtype=np.int64)

    with np.load(args.reference_bank, allow_pickle=False) as reference:
        times = np.asarray(reference["times"], dtype=np.float64)
        reference_particles = np.asarray(reference["reference_particles"], dtype=np.float64)
        base_weights = np.asarray(reference["base_weights"], dtype=np.float64)
        in_domain = np.asarray(reference["in_domain_mask"], dtype=bool)

    if any(index not in set(acquisition_indices.tolist()) for index in TIME_INDICES):
        raise ValueError("paper time indices must all be frozen acquisition nodes")
    if hidden_masses.shape[0] != len(times):
        raise ValueError("validation and reference banks use different time grids")

    hidden_grid_n = int(round(math.sqrt(hidden_masses.shape[-1])))
    if hidden_grid_n**2 != hidden_masses.shape[-1]:
        raise ValueError("hidden validation masses do not form a square grid")
    half_width = float(population["domain_half_width"])
    hidden_dx = 2.0 * half_width / hidden_grid_n
    hidden_centers_1d = -half_width + (np.arange(hidden_grid_n) + 0.5) * hidden_dx
    hidden_xx, hidden_yy = np.meshgrid(hidden_centers_1d, hidden_centers_1d, indexing="xy")
    hidden_points = np.column_stack((hidden_xx.ravel(), hidden_yy.ravel()))
    hidden_features = _sensor_features(hidden_points, sensor_centers, sensor_width)

    observation_targets: list[np.ndarray] = []
    corrected_weights: list[np.ndarray] = []
    ess_fractions: list[float] = []
    multipliers: list[np.ndarray] = []
    for time_index in TIME_INDICES:
        acquisition_position = int(np.flatnonzero(acquisition_indices == time_index)[0])
        target = (
            np.mean(hidden_features[sample_indices[acquisition_position]], axis=0)
            + float(measurement["obs_noise_std"]) * detector_z[acquisition_position]
        )
        if time_index in (0, len(times) - 1):
            target = hidden_masses[time_index] @ hidden_features
        masked_base = np.where(in_domain[time_index], base_weights[time_index], 0.0)
        projected, ess_fraction, lam = _project_weights(
            reference_particles[time_index],
            masked_base,
            sensor_centers,
            sensor_width,
            target,
        )
        observation_targets.append(target)
        corrected_weights.append(projected)
        ess_fractions.append(ess_fraction)
        multipliers.append(lam)

    return {
        "config": cfg,
        "angles": angles,
        "sensor_centers": sensor_centers,
        "sensor_width": sensor_width,
        "half_width": half_width,
        "times": times,
        "alpha": alpha,
        "trial": trial,
        "hidden_features": hidden_features,
        "hidden_masses": hidden_masses,
        "reference_particles": reference_particles,
        "observation_targets": observation_targets,
        "corrected_weights": corrected_weights,
        "ess_fractions": ess_fractions,
        "multipliers": multipliers,
        "allowance": args.allowance,
    }


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    cfg = data["config"]
    population = cfg["population"]
    half_width = float(data["half_width"])
    display_limit = 2.45
    display_n = 241
    display_centers = np.linspace(-half_width, half_width, display_n)
    xx, yy = np.meshgrid(display_centers, display_centers, indexing="xy")
    extent = (-half_width, half_width, -half_width, half_width)

    times = [float(data["times"][index]) for index in TIME_INDICES]
    hidden = [
        _mixture_density(
            xx,
            yy,
            time,
            float(data["alpha"]),
            radius=float(population["radius"]),
            sigma=float(population["sigma"]),
        )
        for time in times
    ]
    bandwidth = float(
        cfg.get("raster", {}).get("authoritative_positive", {}).get(
            "frozen_bandwidth", 0.417530106552
        )
    )
    # The accepted experiment records this frozen Scott bandwidth in its report;
    # older config files need the explicit fallback above.
    corrected = [
        _particle_density(
            data["reference_particles"][time_index],
            weights,
            half_width=half_width,
            grid_n=display_n,
            bandwidth=bandwidth,
        )
        for time_index, weights in zip(TIME_INDICES, data["corrected_weights"])
    ]

    sensor_centers = np.asarray(data["sensor_centers"])
    sensor_width = float(data["sensor_width"])
    display_points = np.column_stack((xx.ravel(), yy.ravel()))
    display_sensor_features = _sensor_features(display_points, sensor_centers, sensor_width)
    sensor_views = [
        [density * display_sensor_features[:, sensor].reshape(xx.shape) for sensor in range(2)]
        for density in hidden
    ]

    truth_vmax = max(max(float(np.max(item)) for item in hidden), max(float(np.max(item)) for item in corrected))
    sensor_vmax = max(float(np.max(item)) for pair in sensor_views for item in pair)
    density_cmap = mpl.colormaps["magma"]
    sensor_cmaps = tuple(_sensor_cmap(color) for color in SENSOR_COLORS)

    fig = plt.figure(figsize=(14.6, 10.4), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 0.64),
        left=0.105,
        right=0.965,
        bottom=0.105,
        top=0.845,
        wspace=0.055,
        hspace=0.13,
    )
    top_axes: list[plt.Axes] = []
    middle_axes: list[plt.Axes] = []

    for column, (time, hidden_density, corrected_density) in enumerate(zip(times, hidden, corrected)):
        truth_ax = fig.add_subplot(outer[0, column])
        corrected_ax = fig.add_subplot(outer[1, column])
        top_axes.append(truth_ax)
        middle_axes.append(corrected_ax)

        truth_image = truth_ax.imshow(
            hidden_density,
            origin="lower",
            extent=extent,
            cmap=density_cmap,
            vmin=0.0,
            vmax=truth_vmax,
            interpolation="bilinear",
        )
        truth_ax.contour(
            xx,
            yy,
            hidden_density,
            levels=np.linspace(0.18 * truth_vmax, 0.82 * truth_vmax, 4),
            colors="white",
            linewidths=0.42,
            alpha=0.45,
        )
        corrected_ax.imshow(
            corrected_density,
            origin="lower",
            extent=extent,
            cmap=density_cmap,
            vmin=0.0,
            vmax=truth_vmax,
            interpolation="bilinear",
        )
        corrected_ax.contour(
            xx,
            yy,
            corrected_density,
            levels=np.linspace(0.18 * truth_vmax, 0.82 * truth_vmax, 4),
            colors="white",
            linewidths=0.42,
            alpha=0.42,
        )
        for sensor, center in enumerate(sensor_centers):
            _add_sensor_marker(
                corrected_ax,
                center,
                sensor_width,
                SENSOR_COLORS[sensor],
                compact=False,
            )

        for ax in (truth_ax, corrected_ax):
            _clean_density_axis(ax, display_limit)
        truth_ax.set_title(rf"$t={time:.1f}$", fontsize=12.5, fontweight="bold", pad=7, color="#272C34")

        sensor_spec = outer[2, column].subgridspec(1, 2, wspace=0.045)
        for sensor in range(2):
            sensor_ax = fig.add_subplot(sensor_spec[0, sensor])
            sensor_ax.imshow(
                sensor_views[column][sensor],
                origin="lower",
                extent=extent,
                cmap=sensor_cmaps[sensor],
                vmin=0.0,
                vmax=sensor_vmax,
                interpolation="bilinear",
            )
            _add_sensor_marker(
                sensor_ax,
                sensor_centers[sensor],
                sensor_width,
                SENSOR_COLORS[sensor],
                compact=True,
            )
            _clean_density_axis(sensor_ax, display_limit)
            sensor_ax.text(
                0.5,
                1.035,
                rf"sensor {sensor + 1}   $y={data['observation_targets'][column][sensor]:.3f}$",
                transform=sensor_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color=SENSOR_COLORS[sensor],
            )

    cbar = fig.colorbar(
        truth_image,
        ax=top_axes + middle_axes,
        fraction=0.017,
        pad=0.010,
        aspect=34,
    )
    cbar.set_label("probability density", fontsize=10.0, color="#555A62")
    cbar.ax.tick_params(labelsize=8.5, length=3, colors="#666A70")
    cbar.outline.set_visible(False)

    # Keep row labels close to the first panel so the figure does not acquire a
    # wide decorative gutter on the left.
    fig.text(0.078, 0.695, "HIDDEN\nPOPULATION", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#4A3A62")
    fig.text(0.078, 0.448, "CORRECTED\nLAW", rotation=90, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#8B3E46")
    fig.text(0.078, 0.205, "WHAT EACH\nSENSOR SEES", rotation=90, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#3E6670")

    angles_deg = np.rad2deg(data["angles"])
    fig.suptitle(
        "Analytic Gaussian-mixture transport",
        x=0.105,
        y=0.955,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.105,
        0.895,
        (
            rf"Authoritative Full geometry at {data['allowance']:.0f}% allowance  ·  "
            rf"sensor angles {angles_deg[0]:.1f}° and {angles_deg[1]:.1f}°  ·  "
            rf"frozen validation trial {data['trial']} ($\alpha={np.rad2deg(data['alpha']):.1f}°$)"
        ),
        ha="left",
        fontsize=8.8,
        color="#6A6F76",
    )
    # Paper-figure house style: no in-panel ESS badge, footer prose, or
    # upper-right context tag.  The title and compact provenance line suffice.
    return fig


def _output_paths(stem: Path) -> tuple[Path, Path]:
    stem = stem.expanduser().resolve()
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    return stem.with_suffix(".png"), stem.with_suffix(".pdf")


def main() -> int:
    args = _parse_args()
    data = _prepare_data(args)
    figure = make_figure(data)
    png, pdf = _output_paths(args.output_stem)
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    print(f"saved {png}")
    print(f"saved {pdf}")
    print(
        "projection diagnostics: "
        f"minimum relative ESS={min(data['ess_fractions']):.6f}, "
        f"maximum |lambda|={max(float(np.max(np.abs(value))) for value in data['multipliers']):.6f}"
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
