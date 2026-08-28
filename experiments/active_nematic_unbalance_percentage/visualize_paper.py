"""Paper-style view of the authoritative active-nematic Pareto experiment.

This is deterministic post-processing of the frozen defect, validation-noise,
reference-bank, view-manifest, and selected-geometry artifacts.  It never runs
the active-nematic simulator, trains a reference, optimizes, or validates.  A
run always writes both PNG and PDF outputs.
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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, SymLogNorm
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import root


SCRIPT_DIR = Path(__file__).resolve().parent
PARETO_DIR = SCRIPT_DIR / "outputs" / "pareto_robust"
FROZEN_DIR = PARETO_DIR / "frozen_inputs"
DEFAULT_PARETO = PARETO_DIR / "authoritative_pareto.json"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "figures" / "active_nematic_defect_correction_sensors"

DISPLAY_INDICES = (0, 4, 8, 10)
SENSOR_COLORS = ("#159B9A", "#EF855B")
PAPER_BACKGROUND = "#F4F1EA"
PANEL_BACKGROUND = "#FBFAF6"
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "signed_defect_density",
    ("#203B73", "#78A5C8", "#F8F5ED", "#D98875", "#8F283A"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--frozen-inputs", type=Path, default=FROZEN_DIR)
    parser.add_argument("--allowance", type=float, default=3.0)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    parser.add_argument("--trial", type=int, default=0)
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


def _selected_centers(pareto: dict[str, Any], allowance: float) -> np.ndarray:
    rows = [
        row
        for row in pareto["rows"]
        if math.isclose(float(row["allowance_percent"]), allowance, abs_tol=1.0e-12)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one authoritative row at {allowance:g}%, found {len(rows)}")
    return np.mod(np.asarray(rows[0]["full_eta"], dtype=np.float64).reshape(-1, 2), 32.0)


def _validation_runs(manifest: dict[str, Any], fold: int) -> np.ndarray:
    views = manifest["validation_run_views"]
    if not 0 <= fold < len(views):
        raise ValueError(f"validation fold must be in [0, {len(views) - 1}]")
    return np.asarray(views[fold], dtype=np.int64)


def _nested_acquisition_indices(time_n: int, count: int) -> np.ndarray:
    if count < 2 or count > time_n:
        raise ValueError("acquisition count must satisfy 2 <= count <= time count")
    indices = np.unique(np.rint(np.linspace(0, time_n - 1, count)).astype(np.int32))
    if len(indices) != count:
        interior = np.arange(1, time_n - 1, dtype=np.int32)
        middle = interior[
            np.rint(np.linspace(0, len(interior) - 1, count - 2)).astype(int)
        ]
        indices = np.concatenate(([0], middle, [time_n - 1])).astype(np.int32)
    return indices


def _resample_trajectory(
    states: np.ndarray,
    offsets: np.ndarray,
    runs: np.ndarray,
    *,
    time_n: int,
    particle_n: int,
    seed: int,
) -> np.ndarray:
    """Reproduce TwoSpeciesDefectBank.resample_normalized_trajectory exactly."""
    rng = np.random.default_rng(seed)
    trajectory = []
    for time_index in range(time_n):
        chunks = [states[offsets[run, time_index] : offsets[run, time_index + 1]] for run in runs]
        rows = np.concatenate([chunk for chunk in chunks if len(chunk)], axis=0)
        # Every run contributes total mass one, so pooled rows have uniform
        # normalized probabilities for the production views (equal counts).
        probabilities = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
        trajectory.append(rows[rng.choice(len(rows), size=particle_n, p=probabilities)])
    return np.stack(trajectory)


def _sensor_features(states: np.ndarray, centers: np.ndarray, box: float, width: float) -> np.ndarray:
    position = states[..., :2]
    delta = position[..., None, :] - centers
    phase = (2.0 * np.pi / box) * delta
    scale = box / (2.0 * np.pi)
    distance2 = 2.0 * scale**2 * np.sum(1.0 - np.cos(phase), axis=-1)
    window = np.exp(-0.5 * distance2 / width**2)
    beta = states[..., 2, None]
    return np.stack((window, window * np.cos(beta), window * np.sin(beta)), axis=-1).reshape(
        states.shape[:-1] + (3 * len(centers),)
    )


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
        scaled_weights = 0.5 * (right - left) * weights
        second = _bspline_basis(times, knots, derivative=2)
        matrix += np.einsum("q,qa,qb->ab", scaled_weights, second, second)
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
        (1.0 - observed_times[:, None]) * endpoint0 + observed_times[:, None] * endpoint1
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


def _mass_trajectory(
    normalized_times: np.ndarray,
    acquisition: np.ndarray,
    plus_mass: np.ndarray,
    minus_mass: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    imbalance = float(np.mean(plus_mass[acquisition] - minus_mass[acquisition]))
    pair = 0.5 * (plus_mass[acquisition] + minus_mass[acquisition])
    log_pair = _reconstruct(
        normalized_times,
        acquisition,
        np.log(pair)[:, None],
        np.log(pair[[0]]),
        np.log(pair[[-1]]),
        {
            **cfg["moment_reconstruction"],
            "smoothing": float(cfg["unbalanced"].get("mass_smoothing", 1.0e-4)),
        },
    )[:, 0]
    pair_evaluation = np.exp(log_pair)
    return pair_evaluation + 0.5 * imbalance, pair_evaluation - 0.5 * imbalance


def _observations_and_targets(
    truth: np.ndarray,
    centers: np.ndarray,
    box: float,
    width: float,
    acquisition: np.ndarray,
    sample_indices: np.ndarray,
    detector_z: np.ndarray,
    noise_std: float,
    times: np.ndarray,
    reconstruction_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    features = _sensor_features(truth, centers, box, width)
    acquired = features[acquisition]
    sampled = np.stack(
        [rows[indices].mean(axis=0) for rows, indices in zip(acquired, sample_indices, strict=True)]
    )
    exact = acquired.mean(axis=1)
    observed = sampled + noise_std * detector_z
    endpoint = (acquisition == 0) | (acquisition == len(times) - 1)
    observed[endpoint] = exact[endpoint]
    target = _reconstruct(times, acquisition, observed, exact[0], exact[-1], reconstruction_cfg)
    return observed, target


def _project_weights(
    states: np.ndarray,
    base_weights: np.ndarray,
    centers: np.ndarray,
    box: float,
    width: float,
    target: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    features = _sensor_features(states, centers, box, width)
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
        initial,
        method="hybr",
        options={"xtol": 1.0e-11, "maxfev": 4000},
    )
    projected = weights(solution.x)
    maximum_residual = float(np.max(np.abs(residual(solution.x))))
    if not solution.success or maximum_residual > 1.0e-8:
        raise RuntimeError(
            f"moment projection failed: success={solution.success}, residual={maximum_residual:.3e}"
        )
    return projected, solution.x, maximum_residual


def _spatial_density(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    box: float,
    grid_n: int = 180,
    bandwidth: float = 0.72,
) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        states[:, 1],
        states[:, 0],
        bins=(grid_n, grid_n),
        range=((0.0, box), (0.0, box)),
        weights=weights,
    )
    cell_width = box / grid_n
    smoothed = gaussian_filter(
        histogram,
        sigma=bandwidth / cell_width,
        mode="wrap",
        truncate=4.0,
    )
    return smoothed / (np.sum(smoothed) * cell_width**2)


def _sensor_window_grid(xx: np.ndarray, yy: np.ndarray, center: np.ndarray, box: float, width: float) -> np.ndarray:
    delta = np.stack((xx - center[0], yy - center[1]), axis=-1)
    phase = (2.0 * np.pi / box) * delta
    scale = box / (2.0 * np.pi)
    distance2 = 2.0 * scale**2 * np.sum(1.0 - np.cos(phase), axis=-1)
    return np.exp(-0.5 * distance2 / width**2)


def _sensor_cmap(color: str, index: int) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"active_nematic_sensor_{index}",
        [PANEL_BACKGROUND, mpl.colors.to_rgba(color, 0.30), color, "#15202A"],
    )


def _clean_axis(ax: plt.Axes, box: float) -> None:
    ax.set_xlim(0.0, box)
    ax.set_ylim(0.0, box)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_periodic_sensor(
    ax: plt.Axes,
    center: np.ndarray,
    width: float,
    box: float,
    color: str,
    *,
    compact: bool,
) -> None:
    for offset_x in (-box, 0.0, box):
        for offset_y in (-box, 0.0, box):
            shifted = center + np.asarray([offset_x, offset_y])
            ax.add_patch(
                plt.Circle(
                    shifted,
                    width,
                    fill=False,
                    color=color,
                    lw=0.65 if compact else 1.05,
                    ls=(0, (2.3, 2.3)),
                    alpha=0.95,
                    zorder=5,
                )
            )
    ax.scatter(
        center[0],
        center[1],
        s=17 if compact else 31,
        marker="x",
        color=color,
        linewidth=1.35 if compact else 1.6,
        zorder=6,
    )


def _prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    frozen = args.frozen_inputs
    cfg = _load_json(frozen / "effective_config.json")
    manifest = _load_json(frozen / "view_manifest.json")
    centers = _selected_centers(_load_json(args.pareto), args.allowance)
    runs = _validation_runs(manifest, args.validation_fold)
    measurement = cfg["measurement"]
    box = float(cfg["physics"]["box_size"])
    width = float(measurement["sensor_width"])

    with np.load(frozen / "two_species_defect_bank.npz", allow_pickle=False) as arrays:
        physical_times = np.asarray(arrays["times"], dtype=np.float64)
        plus_states = np.asarray(arrays["plus_states"], dtype=np.float64)
        minus_states = np.asarray(arrays["minus_states"], dtype=np.float64)
        plus_offsets = np.asarray(arrays["plus_offsets"], dtype=np.int64)
        minus_offsets = np.asarray(arrays["minus_offsets"], dtype=np.int64)
        plus_counts = np.asarray(arrays["plus_counts"], dtype=np.float64)
        minus_counts = np.asarray(arrays["minus_counts"], dtype=np.float64)
    normalized_times = (physical_times - physical_times[0]) / (physical_times[-1] - physical_times[0])
    acquisition = _nested_acquisition_indices(len(physical_times), int(measurement["acquisition_k"]))
    if any(index not in set(acquisition.tolist()) for index in DISPLAY_INDICES):
        raise ValueError("paper time indices must all be acquisition nodes")

    particle_n = int(cfg["randomness"]["truth_particles"])
    truth_seed = int(cfg["seed"]) + 4200 + args.validation_fold
    plus_truth = _resample_trajectory(
        plus_states,
        plus_offsets,
        runs,
        time_n=len(physical_times),
        particle_n=particle_n,
        seed=truth_seed,
    )
    minus_truth = _resample_trajectory(
        minus_states,
        minus_offsets,
        runs,
        time_n=len(physical_times),
        particle_n=particle_n,
        seed=truth_seed + 1,
    )

    with np.load(frozen / "validation_bank.npz", allow_pickle=False) as arrays:
        trial_count = int(arrays["plus_sample_indices"].shape[0])
        if not 0 <= args.trial < trial_count:
            raise ValueError(f"trial must be in [0, {trial_count - 1}]")
        plus_indices = np.asarray(arrays["plus_sample_indices"][args.trial], dtype=np.int64)
        minus_indices = np.asarray(arrays["minus_sample_indices"][args.trial], dtype=np.int64)
        plus_z = np.asarray(arrays["plus_detector_z"][args.trial], dtype=np.float64)
        minus_z = np.asarray(arrays["minus_detector_z"][args.trial], dtype=np.float64)

    noise_std = float(measurement["obs_noise_std"])
    plus_observed, plus_target = _observations_and_targets(
        plus_truth,
        centers,
        box,
        width,
        acquisition,
        plus_indices,
        plus_z,
        noise_std,
        normalized_times,
        cfg["moment_reconstruction"],
    )
    minus_observed, minus_target = _observations_and_targets(
        minus_truth,
        centers,
        box,
        width,
        acquisition,
        minus_indices,
        minus_z,
        noise_std,
        normalized_times,
        cfg["moment_reconstruction"],
    )

    true_plus_mass = plus_counts[runs].mean(axis=0)
    true_minus_mass = minus_counts[runs].mean(axis=0)
    target_plus_mass, target_minus_mass = _mass_trajectory(
        normalized_times,
        acquisition,
        true_plus_mass,
        true_minus_mass,
        cfg,
    )

    reference_dir = frozen / f"reference_seed_{args.reference_seed}"
    reference: dict[str, dict[str, np.ndarray]] = {}
    for species in ("plus", "minus"):
        path = reference_dir / f"{species}_reference_bank.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as arrays:
            reference[species] = {
                "nodes": np.asarray(arrays["nodes"], dtype=np.float64),
                "weights": np.asarray(arrays["weights"], dtype=np.float64),
            }

    corrected: dict[str, list[np.ndarray]] = {"plus": [], "minus": []}
    multipliers: dict[str, list[np.ndarray]] = {"plus": [], "minus": []}
    residuals: list[float] = []
    for species, target in (("plus", plus_target), ("minus", minus_target)):
        initial = np.zeros(target.shape[1], dtype=np.float64)
        for time_index in range(len(physical_times)):
            weights, initial, residual = _project_weights(
                reference[species]["nodes"][time_index],
                reference[species]["weights"][time_index],
                centers,
                box,
                width,
                target[time_index],
                initial,
            )
            corrected[species].append(weights)
            multipliers[species].append(initial.copy())
            residuals.append(residual)

    displayed_observations: dict[str, list[np.ndarray]] = {"plus": [], "minus": []}
    for species, observed in (("plus", plus_observed), ("minus", minus_observed)):
        for time_index in DISPLAY_INDICES:
            position = int(np.flatnonzero(acquisition == time_index)[0])
            displayed_observations[species].append(observed[position])

    return {
        "allowance": args.allowance,
        "validation_fold": args.validation_fold,
        "reference_seed": args.reference_seed,
        "trial": args.trial,
        "physical_times": physical_times,
        "normalized_times": normalized_times,
        "box": box,
        "width": width,
        "centers": centers,
        "truth": {"plus": plus_truth, "minus": minus_truth},
        "true_mass": {"plus": true_plus_mass, "minus": true_minus_mass},
        "reference": reference,
        "corrected_weights": corrected,
        "target_mass": {"plus": target_plus_mass, "minus": target_minus_mass},
        "observations": displayed_observations,
        "multipliers": multipliers,
        "residuals": residuals,
    }


def make_figure(data: dict[str, Any]) -> plt.Figure:
    _style()
    grid_n = 180
    box = data["box"]
    coordinates = (np.arange(grid_n) + 0.5) * box / grid_n
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    extent = (0.0, box, 0.0, box)

    hidden_signed = []
    corrected_signed = []
    hidden_total = []
    for time_index in DISPLAY_INDICES:
        uniform = np.full(data["truth"]["plus"].shape[1], 1.0 / data["truth"]["plus"].shape[1])
        plus_hidden = data["true_mass"]["plus"][time_index] * _spatial_density(
            data["truth"]["plus"][time_index], uniform, box=box, grid_n=grid_n
        )
        minus_hidden = data["true_mass"]["minus"][time_index] * _spatial_density(
            data["truth"]["minus"][time_index], uniform, box=box, grid_n=grid_n
        )
        plus_corrected = data["target_mass"]["plus"][time_index] * _spatial_density(
            data["reference"]["plus"]["nodes"][time_index],
            data["corrected_weights"]["plus"][time_index],
            box=box,
            grid_n=grid_n,
        )
        minus_corrected = data["target_mass"]["minus"][time_index] * _spatial_density(
            data["reference"]["minus"]["nodes"][time_index],
            data["corrected_weights"]["minus"][time_index],
            box=box,
            grid_n=grid_n,
        )
        hidden_signed.append(plus_hidden - minus_hidden)
        corrected_signed.append(plus_corrected - minus_corrected)
        hidden_total.append(plus_hidden + minus_hidden)

    sensor_views = [
        [
            density * _sensor_window_grid(xx, yy, center, box, data["width"])
            for center in data["centers"]
        ]
        for density in hidden_total
    ]
    signed_values = np.concatenate([row.ravel() for row in hidden_signed + corrected_signed])
    signed_max = float(np.quantile(np.abs(signed_values), 0.9975))
    signed_norm = SymLogNorm(
        linthresh=max(0.025 * signed_max, 1.0e-8),
        linscale=0.65,
        vmin=-signed_max,
        vmax=signed_max,
        base=10,
    )
    sensor_values = np.concatenate([row.ravel() for group in sensor_views for row in group])
    sensor_norm = PowerNorm(
        gamma=0.52,
        vmin=0.0,
        vmax=float(np.quantile(sensor_values, 0.998)),
    )
    sensor_cmaps = tuple(_sensor_cmap(color, index) for index, color in enumerate(SENSOR_COLORS))

    fig = plt.figure(figsize=(14.9, 10.2), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 0.72),
        left=0.105,
        right=0.938,
        bottom=0.095,
        top=0.86,
        wspace=0.065,
        hspace=0.13,
    )
    image = None
    for column, time_index in enumerate(DISPLAY_INDICES):
        hidden_ax = fig.add_subplot(outer[0, column])
        corrected_ax = fig.add_subplot(outer[1, column])
        image = hidden_ax.imshow(
            hidden_signed[column],
            origin="lower",
            extent=extent,
            cmap=SIGNED_CMAP,
            norm=signed_norm,
            interpolation="bilinear",
        )
        corrected_ax.imshow(
            corrected_signed[column],
            origin="lower",
            extent=extent,
            cmap=SIGNED_CMAP,
            norm=signed_norm,
            interpolation="bilinear",
        )
        for sensor, center in enumerate(data["centers"]):
            _add_periodic_sensor(
                corrected_ax,
                center,
                data["width"],
                box,
                SENSOR_COLORS[sensor],
                compact=False,
            )
        for ax in (hidden_ax, corrected_ax):
            _clean_axis(ax, box)
        if column == 0:
            time_label = rf"INITIAL CONDITION  ·  $t={data['normalized_times'][time_index]:.2f}$"
        elif column == len(DISPLAY_INDICES) - 1:
            time_label = rf"FINAL CONDITION  ·  $t={data['normalized_times'][time_index]:.2f}$"
        else:
            time_label = rf"$t={data['normalized_times'][time_index]:.2f}$"
        hidden_ax.set_title(
            time_label,
            fontsize=10.8 if column in (0, len(DISPLAY_INDICES) - 1) else 12.2,
            fontweight="bold",
            pad=6,
            color="#272C34",
        )

        sensor_spec = outer[2, column].subgridspec(1, 2, wspace=0.055)
        for sensor in range(2):
            sensor_ax = fig.add_subplot(sensor_spec[0, sensor])
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
                box,
                SENSOR_COLORS[sensor],
                compact=True,
            )
            _clean_axis(sensor_ax, box)
            plus_y = data["observations"]["plus"][column][3 * sensor]
            minus_y = data["observations"]["minus"][column][3 * sensor]
            sensor_ax.text(
                0.5,
                1.035,
                rf"S{sensor + 1}  $y_+={plus_y:.3f}$  $y_-={minus_y:.3f}$",
                transform=sensor_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.25,
                fontweight="bold",
                color=SENSOR_COLORS[sensor],
            )

    if image is None:
        raise RuntimeError("no density panels were constructed")
    colorbar_axis = fig.add_axes((0.951, 0.425, 0.014, 0.38))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(
        "signed defect density   (− charge  /  + charge)",
        fontsize=9.4,
        color="#555A62",
    )
    colorbar.ax.tick_params(labelsize=8.0, length=3, colors="#666A70")
    colorbar.outline.set_visible(False)

    fig.text(0.078, 0.727, "HIDDEN\nDEFECT POPULATION", rotation=90, ha="center", va="center", fontsize=9.8, fontweight="bold", color="#4A3A62")
    fig.text(0.078, 0.495, "CORRECTED\nLAW", rotation=90, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#8B3E46")
    fig.text(0.078, 0.225, "WHAT EACH\nSENSOR SEES", rotation=90, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#3E6670")
    fig.suptitle(
        "Two sensors track active-nematic defects",
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
        rf"Inference window: physical time {data['physical_times'][0]:.0f}–{data['physical_times'][-1]:.0f}, normalized to $t=0$–$1$  ·  authoritative Full geometry at {data['allowance']:g}%  ·  fold {data['validation_fold']}  ·  reference {data['reference_seed']}  ·  trial {data['trial']}",
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
        f"max calibration residual={max(data['residuals']):.3e}, "
        f"max |lambda|={max(float(np.max(np.abs(row))) for rows in data['multipliers'].values() for row in rows):.6f}"
    )
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
