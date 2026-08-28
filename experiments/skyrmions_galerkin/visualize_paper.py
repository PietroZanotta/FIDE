"""Paper-style view of the official B1 Galerkin skyrmion experiment.

This is deterministic post-processing of the frozen fresh-validation truth,
reference, noise, and selected-geometry artifacts.  It never simulates,
trains, optimizes, or validates.  Every run writes both PNG and PDF outputs.
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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import root


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR / "outputs" / "official_b1_galerkin_pareto_v1"
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_SELECTION = RUN_DIR / "selection" / "pareto_selection.json"
DEFAULT_TRUTH = RUN_DIR / "fresh_validation" / "truth.npz"
DEFAULT_REFERENCE = RUN_DIR / "fresh_validation" / "paper_reference_audit.npz"
DEFAULT_NOISE = RUN_DIR / "fresh_validation" / "measurement_noise.npz"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "figures" / "skyrmion_population_correction_sensors"

TIME_INDICES = (0, 4, 8, 12)
SENSOR_COLORS = ("#1CA6A3", "#F28E5B", "#9271C2", "#5AAA70")
PAPER_BACKGROUND = "#F4F1EA"
PANEL_BACKGROUND = "#FBFAF6"
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "skyrmion_density",
    ("#F7F3EA", "#D9C8AA", "#C66B67", "#713B62", "#20233C"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--noise", type=Path, default=DEFAULT_NOISE)
    parser.add_argument("--allowance", type=float, default=5.0)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
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


def _selected_centers(selection: dict[str, Any], allowance: float) -> np.ndarray:
    matches = [
        row
        for row in selection.get("winners", [])
        if math.isclose(float(row["allowance_percent"]), allowance, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one selected row at {allowance:g}%, found {len(matches)}")
    return np.asarray(matches[0]["Full"]["eta"], dtype=np.float64).reshape(-1, 2)


def _minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return (delta + 0.5 * box) % box - 0.5 * box


def _features(
    configurations: np.ndarray,
    centers: np.ndarray,
    width: float,
    box: np.ndarray,
) -> np.ndarray:
    delta = _minimum_image(
        configurations[:, :, None, :] - centers[None, None, :, :], box
    )
    distance2 = np.sum(delta**2, axis=-1)
    return np.mean(np.exp(-0.5 * distance2 / width**2), axis=1)


def _acquisition_indices(time_n: int, count: int) -> np.ndarray:
    indices = np.asarray(
        [round(index * (time_n - 1) / (count - 1)) for index in range(count)],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != count:
        raise ValueError("acquisition schedule contains duplicate nodes")
    return indices


def _bspline_basis(times: np.ndarray, knots: np.ndarray, derivative: int = 0) -> np.ndarray:
    degree = 3
    count = len(knots) - degree - 1
    coefficients = np.eye(count, dtype=np.float64)
    columns = []
    for index in range(count):
        spline = BSpline(knots, coefficients[index], degree, extrapolate=False)
        if derivative:
            spline = spline.derivative(derivative)
        columns.append(np.asarray(spline(times), dtype=np.float64))
    return np.stack(columns, axis=-1)[..., 1:-1]


def _roughness_matrix(knots: np.ndarray, order: int) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    spans = np.unique(knots)
    size = _bspline_basis(np.asarray([0.5]), knots).shape[-1]
    matrix = np.zeros((size, size), dtype=np.float64)
    for left, right in zip(spans[:-1], spans[1:], strict=True):
        if right <= left:
            continue
        times = 0.5 * (right - left) * nodes + 0.5 * (left + right)
        scaled = 0.5 * (right - left) * weights
        second = _bspline_basis(times, knots, derivative=2)
        matrix += np.einsum("q,qa,qb->ab", scaled, second, second)
    return 0.5 * (matrix + matrix.T)


def _reconstruct(
    times: np.ndarray,
    acquisition: np.ndarray,
    observations: np.ndarray,
    endpoint0: np.ndarray,
    endpoint1: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    internal = int(cfg["internal_knots"])
    knots = np.concatenate(
        (np.zeros(4), np.linspace(0.0, 1.0, internal + 2)[1:-1], np.ones(4))
    )
    observed_times = times[acquisition]
    basis_observed = _bspline_basis(observed_times, knots)
    basis_all = _bspline_basis(times, knots)
    roughness = _roughness_matrix(knots, int(cfg["roughness_quadrature_order"]))
    linear_observed = (
        (1.0 - observed_times[:, None]) * endpoint0
        + observed_times[:, None] * endpoint1
    )
    gram = basis_observed.T @ basis_observed
    scale = max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    normal = (
        gram
        + float(cfg["smoothing"]) * roughness
        + float(cfg["ridge_rel"]) * scale * np.eye(len(gram))
    )
    coefficients = np.linalg.solve(
        normal, basis_observed.T @ (observations - linear_observed)
    )
    linear_all = (1.0 - times[:, None]) * endpoint0 + times[:, None] * endpoint1
    return linear_all + basis_all @ coefficients


def _project_weights(
    configurations: np.ndarray,
    base_weights: np.ndarray,
    centers: np.ndarray,
    width: float,
    box: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    features = _features(configurations, centers, width, box)
    base = np.asarray(base_weights, dtype=np.float64).copy()
    base /= np.sum(base)
    positive = base > 0.0
    log_base = np.full_like(base, -np.inf)
    log_base[positive] = np.log(base[positive])

    def weights(multiplier: np.ndarray) -> np.ndarray:
        logits = log_base + features @ multiplier
        logits -= np.max(logits[positive])
        tilted = np.exp(logits)
        return tilted / np.sum(tilted)

    def residual(multiplier: np.ndarray) -> np.ndarray:
        return weights(multiplier) @ features - target

    solution = root(
        residual,
        np.zeros(features.shape[1], dtype=np.float64),
        method="hybr",
        options={"xtol": 1.0e-11, "maxfev": 2000},
    )
    projected = weights(solution.x)
    maximum_residual = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or maximum_residual > 1.0e-8:
        raise RuntimeError(
            f"moment projection failed: success={solution.success}, "
            f"residual={maximum_residual:.3e}"
        )
    return projected, solution.x, maximum_residual


def _particle_density(
    configurations: np.ndarray,
    configuration_weights: np.ndarray,
    *,
    nx: int = 240,
    ny: int = 120,
    bandwidth: float = 0.018,
) -> np.ndarray:
    particle_count = configurations.shape[1]
    points = configurations.reshape(-1, 2)
    weights = np.repeat(configuration_weights / particle_count, particle_count)
    histogram, _, _ = np.histogram2d(
        points[:, 1],
        points[:, 0],
        bins=(ny, nx),
        range=((0.0, 1.0), (0.0, 2.0)),
        weights=weights,
    )
    cell_width = 2.0 / nx
    smoothed = gaussian_filter(
        histogram,
        sigma=bandwidth / cell_width,
        mode="wrap",
        truncate=4.0,
    )
    return smoothed / (np.sum(smoothed) * cell_width**2)


def _sensor_window(
    points: np.ndarray, center: np.ndarray, width: float, box: np.ndarray
) -> np.ndarray:
    delta = _minimum_image(points - center, box)
    return np.exp(-0.5 * np.sum(delta**2, axis=-1) / width**2)


def _sensor_cmap(color: str, index: int) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"skyrmion_sensor_{index}",
        [PANEL_BACKGROUND, mpl.colors.to_rgba(color, 0.28), color, "#151A21"],
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_periodic_sensor(
    ax: plt.Axes,
    center: np.ndarray,
    width: float,
    color: str,
    *,
    compact: bool,
) -> None:
    for offset_x in (-2.0, 0.0, 2.0):
        for offset_y in (-1.0, 0.0, 1.0):
            shifted = center + np.asarray([offset_x, offset_y])
            ax.add_patch(
                plt.Circle(
                    shifted,
                    width,
                    fill=False,
                    color=color,
                    lw=0.7 if compact else 1.0,
                    ls=(0, (2.2, 2.2)),
                    alpha=0.95,
                    zorder=5,
                )
            )
    ax.scatter(
        center[0],
        center[1],
        s=14 if compact else 27,
        marker="x",
        color=color,
        linewidth=1.3 if compact else 1.5,
        zorder=6,
    )


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_json(args.config)
    centers = _selected_centers(_load_json(args.selection), args.allowance)
    measurement = cfg["measurement"]
    width = float(measurement["sensor_width"])
    box = np.asarray(cfg["physics"]["box"], dtype=np.float64)

    with np.load(args.truth, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=np.float64)
        truth = np.asarray(arrays["configurations"], dtype=np.float64)
    with np.load(args.reference, allow_pickle=False) as arrays:
        reference = np.asarray(arrays["configurations"], dtype=np.float64)
        base_weights = np.asarray(arrays["base_weights"], dtype=np.float64)
    with np.load(args.noise, allow_pickle=False) as arrays:
        detector_noise = np.asarray(arrays["detector_noise"], dtype=np.float64)

    acquisition = _acquisition_indices(
        len(times), int(measurement["acquisition_count"])
    )
    if any(index not in set(acquisition.tolist()) for index in TIME_INDICES):
        raise ValueError("paper time indices must all be acquisition nodes")
    truth_features = np.stack(
        [_features(truth[index], centers, width, box) for index in range(len(times))]
    )
    acquired = truth_features[acquisition]
    finite_count = min(int(measurement["finite_configurations"]), truth.shape[1])
    observations = acquired[:, :finite_count].mean(axis=1) + detector_noise
    endpoint0 = truth_features[0].mean(axis=0)
    endpoint1 = truth_features[-1].mean(axis=0)
    reconstructed = _reconstruct(
        times,
        acquisition,
        observations,
        endpoint0,
        endpoint1,
        cfg["moment_reconstruction"],
    )

    corrected_weights = []
    multipliers = []
    residuals = []
    displayed_observations = []
    for time_index in TIME_INDICES:
        acquisition_position = int(np.flatnonzero(acquisition == time_index)[0])
        weights, multiplier, residual = _project_weights(
            reference[time_index],
            base_weights[time_index],
            centers,
            width,
            box,
            reconstructed[time_index],
        )
        corrected_weights.append(weights)
        multipliers.append(multiplier)
        residuals.append(residual)
        displayed_observations.append(
            endpoint0
            if time_index == 0
            else endpoint1 if time_index == len(times) - 1 else observations[acquisition_position]
        )

    return {
        "config": cfg,
        "allowance": args.allowance,
        "times": times,
        "centers": centers,
        "width": width,
        "box": box,
        "truth": truth,
        "reference": reference,
        "corrected_weights": corrected_weights,
        "multipliers": multipliers,
        "residuals": residuals,
        "observations": displayed_observations,
    }


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    nx, ny = 240, 120
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    extent = (0.0, 2.0, 0.0, 1.0)

    hidden = [
        _particle_density(
            data["truth"][index],
            np.full(data["truth"].shape[1], 1.0 / data["truth"].shape[1]),
            nx=nx,
            ny=ny,
        )
        for index in TIME_INDICES
    ]
    corrected = [
        _particle_density(data["reference"][index], weights, nx=nx, ny=ny)
        for index, weights in zip(TIME_INDICES, data["corrected_weights"], strict=True)
    ]
    sensor_views = [
        [
            density
            * _sensor_window(
                grid_points, data["centers"][sensor], data["width"], data["box"]
            ).reshape(ny, nx)
            for sensor in range(4)
        ]
        for density in hidden
    ]

    density_values = np.concatenate([row.ravel() for row in hidden + corrected])
    density_norm = PowerNorm(
        gamma=0.52, vmin=0.0, vmax=float(np.quantile(density_values, 0.998))
    )
    sensor_values = np.concatenate(
        [row.ravel() for group in sensor_views for row in group]
    )
    sensor_norm = PowerNorm(
        gamma=0.45, vmin=0.0, vmax=float(np.quantile(sensor_values, 0.998))
    )
    sensor_cmaps = tuple(
        _sensor_cmap(color, index) for index, color in enumerate(SENSOR_COLORS)
    )

    fig = plt.figure(figsize=(15.4, 9.25), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 1.04),
        left=0.105,
        right=0.94,
        bottom=0.105,
        top=0.865,
        wspace=0.055,
        hspace=0.13,
    )
    image = None
    for column, time_index in enumerate(TIME_INDICES):
        truth_ax = fig.add_subplot(outer[0, column])
        corrected_ax = fig.add_subplot(outer[1, column])
        image = truth_ax.imshow(
            hidden[column],
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            norm=density_norm,
            interpolation="bilinear",
        )
        corrected_ax.imshow(
            corrected[column],
            origin="lower",
            extent=extent,
            cmap=DENSITY_CMAP,
            norm=density_norm,
            interpolation="bilinear",
        )
        for sensor, center in enumerate(data["centers"]):
            _add_periodic_sensor(
                corrected_ax,
                center,
                data["width"],
                SENSOR_COLORS[sensor],
                compact=False,
            )
        for ax in (truth_ax, corrected_ax):
            _clean_axis(ax)
        truth_ax.set_title(
            rf"$t={data['times'][time_index]:.2f}$",
            fontsize=12.2,
            fontweight="bold",
            pad=6,
            color="#272C34",
        )

        sensor_spec = outer[2, column].subgridspec(2, 2, wspace=0.035, hspace=0.18)
        for sensor in range(4):
            sensor_ax = fig.add_subplot(sensor_spec[sensor // 2, sensor % 2])
            sensor_ax.imshow(
                sensor_views[column][sensor],
                origin="lower",
                extent=extent,
                cmap=sensor_cmaps[sensor],
                norm=sensor_norm,
                interpolation="bilinear",
            )
            _add_periodic_sensor(
                sensor_ax,
                data["centers"][sensor],
                data["width"],
                SENSOR_COLORS[sensor],
                compact=True,
            )
            _clean_axis(sensor_ax)
            sensor_ax.text(
                0.5,
                1.025,
                rf"S{sensor + 1}  $y={data['observations'][column][sensor]:.3f}$",
                transform=sensor_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color=SENSOR_COLORS[sensor],
            )

    if image is None:
        raise RuntimeError("no density panels were constructed")
    colorbar_axis = fig.add_axes((0.951, 0.435, 0.013, 0.365))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("skyrmion probability density", fontsize=9.6, color="#555A62")
    colorbar.ax.tick_params(labelsize=8.2, length=3, colors="#666A70")
    colorbar.outline.set_visible(False)

    fig.text(0.078, 0.727, "HIDDEN\nPOPULATION", rotation=90, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#4A3A62")
    fig.text(0.078, 0.497, "CORRECTED\nLAW", rotation=90, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#8B3E46")
    fig.text(0.078, 0.244, "WHAT EACH\nSENSOR SEES", rotation=90, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#3E6670")
    fig.suptitle(
        "Four sensors track a driven skyrmion population",
        x=0.105,
        y=0.955,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.105,
        0.905,
        rf"Official B1 Galerkin Full geometry at {data['allowance']:g}% allowance  ·  frozen fresh-validation bank",
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
        f"maximum calibration residual={max(data['residuals']):.3e}, "
        f"maximum |lambda|={max(float(np.max(np.abs(row))) for row in data['multipliers']):.6f}"
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
