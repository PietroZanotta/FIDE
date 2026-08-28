"""Create the paper-style vortices population / corrected-law / sensor figure.

The script is deterministic post-processing.  It reads the authoritative 5%
Full geometry and the frozen truth, reference, and validation banks.  It does
not train a reference model, optimize sensor locations, or rerun validation.
Every invocation writes both a PNG and a vector PDF.

From the repository root::

    .venv/bin/python experiments/vortices_percentage/visualize_paper.py
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
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto" / "corrected_authoritative_pareto.json"
DEFAULT_TRUTH_BANK = SCRIPT_DIR / "outputs" / "pareto" / "frozen_inputs" / "truth_bank.npz"
DEFAULT_REFERENCE_BANK = SCRIPT_DIR / "outputs" / "pareto" / "frozen_inputs" / "reference_bank.npz"
DEFAULT_VALIDATION_BANK = SCRIPT_DIR / "outputs" / "pareto" / "frozen_inputs" / "validation_bank.npz"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "figures" / "vortices_population_correction_sensors"

# All four panels are frozen acquisition nodes on the 21-node scientific grid.
TIME_INDICES = (0, 5, 15, 20)
SENSOR_COLORS = ("#1CA6A3", "#F28E5B", "#9271C2", "#5AAA70")
PAPER_BACKGROUND = "#F4F1EA"
PANEL_BACKGROUND = "#FBFAF6"
FLOW_CMAP = LinearSegmentedColormap.from_list(
    "vortex_density",
    ("#F7F3EA", "#DCC8A6", "#D77A61", "#8D3D55", "#272442"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pareto", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--truth-bank", type=Path, default=DEFAULT_TRUTH_BANK)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=DEFAULT_VALIDATION_BANK)
    parser.add_argument("--allowance", type=float, default=5.0)
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="frozen validation trial to illustrate (default: 0)",
    )
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


def _full_centers(pareto: dict[str, Any], allowance: float) -> np.ndarray:
    matches = [
        row
        for row in pareto.get("rows", [])
        if math.isclose(float(row["allowance_percent"]), allowance, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Pareto row at {allowance:g}%, found {len(matches)}")
    return np.asarray(matches[0]["full_centers"], dtype=np.float64)


def _sensor_features(points: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    displacement = points[:, None, :] - centers[None, :, :]
    return np.exp(-0.5 * np.sum(displacement**2, axis=-1) / width**2)


def _acquisition_indices(time_n: int, count: int) -> np.ndarray:
    indices = np.rint(np.linspace(0, time_n - 1, count)).astype(np.int64)
    indices[0], indices[-1] = 0, time_n - 1
    if len(np.unique(indices)) != count:
        raise ValueError("rounded acquisition schedule contains duplicate time nodes")
    return indices


def _bspline_basis(times: np.ndarray, knots: np.ndarray, derivative: int = 0) -> np.ndarray:
    degree = 3
    n_full = len(knots) - degree - 1
    coefficients = np.eye(n_full, dtype=np.float64)
    columns: list[np.ndarray] = []
    for index in range(n_full):
        spline = BSpline(knots, coefficients[index], degree, extrapolate=False)
        if derivative:
            spline = spline.derivative(derivative)
        columns.append(np.asarray(spline(times), dtype=np.float64))
    # Dropping the first and last clamped basis functions makes every retained
    # basis column vanish exactly at both endpoints.
    return np.stack(columns, axis=-1)[..., 1:-1]


def _roughness_matrix(knots: np.ndarray, quadrature_order: int) -> np.ndarray:
    spans = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    n_basis = _bspline_basis(np.asarray([0.5]), knots).shape[-1]
    roughness = np.zeros((n_basis, n_basis), dtype=np.float64)
    for left, right in zip(spans[:-1], spans[1:], strict=True):
        if right <= left:
            continue
        times = 0.5 * (right - left) * nodes + 0.5 * (left + right)
        scaled_weights = 0.5 * (right - left) * weights
        second = _bspline_basis(times, knots, derivative=2)
        roughness += np.einsum("q,qa,qb->ab", scaled_weights, second, second)
    return 0.5 * (roughness + roughness.T)


def _bounded_moment_values(values: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    feature_bounds = cfg.get("feature_bounds")
    if feature_bounds is None:
        return values
    margin = float(cfg.get("feature_bound_interior_margin", 0.0))
    lower = float(feature_bounds[0]) + margin
    upper = float(feature_bounds[1]) - margin
    width = float(cfg.get("feature_bound_transition_width", margin))

    def blend(s: np.ndarray) -> np.ndarray:
        return s**3 * (6.0 - 8.0 * s + 3.0 * s**2)

    lower_s = np.clip((values - lower) / width, 0.0, 1.0)
    upper_s = np.clip((upper - values) / width, 0.0, 1.0)
    lower_transition = lower + width * blend(lower_s)
    upper_transition = upper - width * blend(upper_s)
    return np.where(
        values <= lower,
        lower,
        np.where(
            values < lower + width,
            lower_transition,
            np.where(
                values >= upper,
                upper,
                np.where(values > upper - width, upper_transition, values),
            ),
        ),
    )


def _reconstruct_moments(
    times: np.ndarray,
    acquisition_indices: np.ndarray,
    observations: np.ndarray,
    endpoint_start: np.ndarray,
    endpoint_end: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    internal_count = int(cfg.get("internal_knots", 3))
    interior = np.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    knots = np.concatenate((np.zeros(4), interior, np.ones(4)))
    observed_times = times[acquisition_indices]
    basis_observed = _bspline_basis(observed_times, knots)
    basis_evaluation = _bspline_basis(times, knots)
    roughness = _roughness_matrix(
        knots, int(cfg.get("roughness_quadrature_order", 8))
    )

    linear_observed = (
        (1.0 - observed_times[:, None]) * endpoint_start[None, :]
        + observed_times[:, None] * endpoint_end[None, :]
    )
    gram = basis_observed.T @ basis_observed
    scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
    normal = (
        gram
        + float(cfg.get("smoothing", 1.0e-4)) * roughness
        + float(cfg.get("ridge_rel", 1.0e-10)) * scale * np.eye(gram.shape[0])
    )
    coefficients = np.linalg.solve(
        normal, basis_observed.T @ (observations - linear_observed)
    )
    linear_evaluation = (
        (1.0 - times[:, None]) * endpoint_start[None, :]
        + times[:, None] * endpoint_end[None, :]
    )
    values = linear_evaluation + basis_evaluation @ coefficients
    return _bounded_moment_values(values, cfg)


def _project_weights(
    points: np.ndarray,
    base_weights: np.ndarray,
    centers: np.ndarray,
    width: float,
    target: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    features = _sensor_features(points, centers, width)
    base = np.asarray(base_weights, dtype=np.float64).copy()
    base /= np.sum(base)
    positive = base > 0.0
    log_base = np.full_like(base, -np.inf)
    log_base[positive] = np.log(base[positive])

    def weights(lam: np.ndarray) -> np.ndarray:
        logits = log_base + features @ lam
        logits -= np.max(logits[positive])
        tilted = np.exp(logits)
        return tilted / np.sum(tilted)

    def residual(lam: np.ndarray) -> np.ndarray:
        return weights(lam) @ features - target

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
            "sensor-moment projection failed: "
            f"success={solution.success}, residual={maximum_residual:.3e}"
        )
    base_ess = 1.0 / np.sum(base**2)
    projected_ess = 1.0 / np.sum(projected**2)
    return projected, projected_ess / base_ess, solution.x, maximum_residual


def _particle_density(
    points: np.ndarray,
    weights: np.ndarray,
    *,
    nx: int = 240,
    ny: int = 120,
    visual_bandwidth: float = 0.015,
) -> np.ndarray:
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
        sigma=visual_bandwidth / cell_width,
        mode="constant",
        truncate=4.0,
    )
    mass = float(np.sum(smoothed))
    if mass <= 0.0:
        raise RuntimeError("rasterized particle law has zero mass")
    return smoothed / (mass * cell_width**2)


def _double_gyre_velocity(
    xx: np.ndarray, yy: np.ndarray, time: float, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = float(cfg.get("amplitude", 0.1))
    epsilon = float(cfg.get("epsilon", 0.25))
    horizon = float(cfg.get("horizon", 10.0))
    period = float(cfg.get("period", 10.0))
    physical_time = horizon * time
    a = epsilon * np.sin(2.0 * np.pi * physical_time / period)
    b = 1.0 - 2.0 * a
    f = a * xx**2 + b * xx
    dfdx = 2.0 * a * xx + b
    velocity_x = -np.pi * amplitude * np.sin(np.pi * f) * np.cos(np.pi * yy)
    velocity_y = np.pi * amplitude * np.cos(np.pi * f) * np.sin(np.pi * yy) * dfdx
    return horizon * velocity_x, horizon * velocity_y


def _sensor_cmap(color: str, index: int) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"sensor_{index}",
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
        linewidth=1.25 if compact else 1.5,
        zorder=6,
    )


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_json(args.config)
    centers = _full_centers(_load_json(args.pareto), args.allowance)
    measurement = cfg["measurement"]
    width = float(measurement["sensor_width"])

    with np.load(args.truth_bank, allow_pickle=False) as truth_bank:
        times = np.asarray(truth_bank["times"], dtype=np.float64)
        truth_particles = np.asarray(truth_bank["particles"], dtype=np.float64)
    with np.load(args.reference_bank, allow_pickle=False) as reference_bank:
        reference_times = np.asarray(reference_bank["times"], dtype=np.float64)
        reference_particles = np.asarray(reference_bank["nodes"], dtype=np.float64)
        base_weights = np.asarray(reference_bank["weights"], dtype=np.float64)
    with np.load(args.validation_bank, allow_pickle=False) as validation_bank:
        sample_indices_bank = np.asarray(validation_bank["sample_indices"], dtype=np.int64)
        detector_z_bank = np.asarray(validation_bank["detector_z"], dtype=np.float64)

    if not np.array_equal(times, reference_times):
        raise ValueError("truth and reference banks use different time grids")
    if args.trial < 0 or args.trial >= sample_indices_bank.shape[0]:
        raise ValueError(
            f"trial must lie in [0, {sample_indices_bank.shape[0] - 1}], got {args.trial}"
        )
    acquisition_indices = _acquisition_indices(
        len(times), int(measurement["acquisition_k"])
    )
    if any(index not in set(acquisition_indices.tolist()) for index in TIME_INDICES):
        raise ValueError("paper time indices must all be frozen acquisition nodes")

    sample_indices = sample_indices_bank[args.trial]
    detector_z = detector_z_bank[args.trial]
    truth_features = [
        _sensor_features(truth_particles[index], centers, width)
        for index in acquisition_indices
    ]
    exact_observations = np.stack([features.mean(axis=0) for features in truth_features])
    observations = np.stack(
        [
            features[sample_indices[position]].mean(axis=0)
            + float(measurement["obs_noise_std"]) * detector_z[position]
            for position, features in enumerate(truth_features)
        ]
    )
    observations[0] = exact_observations[0]
    observations[-1] = exact_observations[-1]
    reconstructed = _reconstruct_moments(
        times,
        acquisition_indices,
        observations,
        exact_observations[0],
        exact_observations[-1],
        cfg["moment_reconstruction"],
    )

    corrected_weights: list[np.ndarray] = []
    ess_fractions: list[float] = []
    multipliers: list[np.ndarray] = []
    projection_residuals: list[float] = []
    displayed_observations: list[np.ndarray] = []
    for time_index in TIME_INDICES:
        acquisition_position = int(np.flatnonzero(acquisition_indices == time_index)[0])
        projected, ess_fraction, multiplier, residual = _project_weights(
            reference_particles[time_index],
            base_weights[time_index],
            centers,
            width,
            reconstructed[time_index],
        )
        corrected_weights.append(projected)
        ess_fractions.append(ess_fraction)
        multipliers.append(multiplier)
        projection_residuals.append(residual)
        displayed_observations.append(observations[acquisition_position])

    return {
        "config": cfg,
        "allowance": args.allowance,
        "trial": args.trial,
        "times": times,
        "centers": centers,
        "sensor_width": width,
        "truth_particles": truth_particles,
        "reference_particles": reference_particles,
        "corrected_weights": corrected_weights,
        "ess_fractions": ess_fractions,
        "multipliers": multipliers,
        "projection_residuals": projection_residuals,
        "observations": displayed_observations,
        "reconstructed": [reconstructed[index] for index in TIME_INDICES],
    }


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    extent = (0.0, 2.0, 0.0, 1.0)
    nx, ny = 240, 120
    x_centers = (np.arange(nx) + 0.5) * 2.0 / nx
    y_centers = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")

    hidden = [
        _particle_density(
            data["truth_particles"][index],
            np.full(data["truth_particles"].shape[1], 1.0 / data["truth_particles"].shape[1]),
            nx=nx,
            ny=ny,
        )
        for index in TIME_INDICES
    ]
    corrected = [
        _particle_density(
            data["reference_particles"][index], weights, nx=nx, ny=ny
        )
        for index, weights in zip(TIME_INDICES, data["corrected_weights"], strict=True)
    ]
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_features = _sensor_features(
        grid_points, data["centers"], float(data["sensor_width"])
    )
    sensor_views = [
        [density * grid_features[:, sensor].reshape((ny, nx)) for sensor in range(4)]
        for density in hidden
    ]

    density_values = np.concatenate([density.ravel() for density in hidden + corrected])
    density_vmax = float(np.quantile(density_values, 0.998))
    density_norm = PowerNorm(gamma=0.52, vmin=0.0, vmax=density_vmax)
    sensor_values = np.concatenate(
        [view.ravel() for time_views in sensor_views for view in time_views]
    )
    sensor_vmax = float(np.quantile(sensor_values, 0.998))
    sensor_norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=sensor_vmax)
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
    top_axes: list[plt.Axes] = []
    middle_axes: list[plt.Axes] = []
    image = None

    for column, time_index in enumerate(TIME_INDICES):
        time = float(data["times"][time_index])
        truth_ax = fig.add_subplot(outer[0, column])
        corrected_ax = fig.add_subplot(outer[1, column])
        top_axes.append(truth_ax)
        middle_axes.append(corrected_ax)

        image = truth_ax.imshow(
            hidden[column],
            origin="lower",
            extent=extent,
            cmap=FLOW_CMAP,
            norm=density_norm,
            interpolation="bilinear",
        )
        gx = np.linspace(0.02, 1.98, 31)
        gy = np.linspace(0.02, 0.98, 17)
        flow_xx, flow_yy = np.meshgrid(gx, gy, indexing="xy")
        velocity_x, velocity_y = _double_gyre_velocity(
            flow_xx, flow_yy, time, data["config"]["truth"]
        )
        truth_ax.streamplot(
            gx,
            gy,
            velocity_x,
            velocity_y,
            density=0.52,
            linewidth=0.38,
            arrowsize=0.45,
            color=(1.0, 1.0, 1.0, 0.38),
        )
        corrected_ax.imshow(
            corrected[column],
            origin="lower",
            extent=extent,
            cmap=FLOW_CMAP,
            norm=density_norm,
            interpolation="bilinear",
        )
        for sensor, center in enumerate(data["centers"]):
            _add_sensor_marker(
                corrected_ax,
                center,
                float(data["sensor_width"]),
                SENSOR_COLORS[sensor],
                compact=False,
            )
        for ax in (truth_ax, corrected_ax):
            _clean_axis(ax)
        truth_ax.set_title(
            rf"$t={time:.2f}$",
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
            _add_sensor_marker(
                sensor_ax,
                data["centers"][sensor],
                float(data["sensor_width"]),
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
    colorbar.set_label("probability density", fontsize=9.6, color="#555A62")
    colorbar.ax.tick_params(labelsize=8.2, length=3, colors="#666A70")
    colorbar.outline.set_visible(False)

    fig.text(
        0.078,
        0.727,
        "HIDDEN\nPOPULATION",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#4A3A62",
    )
    fig.text(
        0.078,
        0.497,
        "CORRECTED\nLAW",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#8B3E46",
    )
    fig.text(
        0.078,
        0.244,
        "WHAT EACH\nSENSOR SEES",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#3E6670",
    )
    fig.suptitle(
        "Four sensors in a moving double gyre",
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
        rf"Authoritative Full geometry at {data['allowance']:g}% allowance  ·  frozen validation trial {data['trial']}",
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
        f"minimum ESS/base ESS={min(data['ess_fractions']):.6f}, "
        f"maximum calibration residual={max(data['projection_residuals']):.3e}, "
        f"maximum |lambda|={max(float(np.max(np.abs(value))) for value in data['multipliers']):.6f}"
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
