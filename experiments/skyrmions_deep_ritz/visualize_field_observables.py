"""Field-only trajectory visuals for a certified skyrmion Pareto sweep.

The backgrounds are empirical one-particle densities from the frozen truth
trajectory.  Sensor curves are the same sparse-observation reconstructions
used by selection, not values inferred from the Deep Ritz solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from scipy.ndimage import gaussian_filter

from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor

from .measurements import LocalDensitySensors
from .visualize_authoritative import (
    GRAY,
    LIGHT_GRAY,
    NAVY,
    SENSOR_COLORS,
    STYLE,
    _load,
    _resolve_result_path,
    _save,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _truth_path(result_path: Path) -> Path:
    candidate = result_path.parent / "truth_banks.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"missing frozen truth trajectory: {candidate}")
    return candidate


def _density_frames(
    configurations: np.ndarray,
    *,
    box: np.ndarray,
    bins: tuple[int, int] = (120, 60),
) -> np.ndarray:
    """Return periodic, smoothed empirical one-particle density frames."""

    frames = []
    for frame in configurations:
        density, _, _ = np.histogram2d(
            frame[..., 0].reshape(-1),
            frame[..., 1].reshape(-1),
            bins=bins,
            range=((0.0, float(box[0])), (0.0, float(box[1]))),
            density=True,
        )
        frames.append(gaussian_filter(density.T, sigma=1.5, mode="wrap"))
    return np.asarray(frames)


def _observable_trajectory(
    configurations: np.ndarray,
    times: np.ndarray,
    eta: np.ndarray,
    result: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Reproduce the experiment's sparse observations and spline target."""

    measurement = result["config"]["measurement"]
    physics = result["config"]["physics"]
    family = LocalDensitySensors(
        int(measurement["n_sensors"]),
        float(measurement["sensor_width"]),
        tuple(float(value) for value in physics["box"]),
        float(measurement["min_separation"]),
    )
    features = np.asarray(
        family.features(jnp.asarray(configurations), jnp.asarray(eta)),
        dtype=np.float64,
    )
    count = int(measurement["acquisition_count"])
    indices = np.unique(np.rint(np.linspace(0, len(times) - 1, count)).astype(int))
    finite_n = min(int(measurement["finite_configurations"]), features.shape[1])
    observations = np.mean(features[indices, :finite_n], axis=1)

    offsets = result["config"]["banks"]["seed_offsets"]
    noise_seed = int(result["config"]["seed"]) + int(offsets["observation"])
    noise = float(measurement["observation_noise_std"]) * np.asarray(
        jax.random.normal(
            jax.random.PRNGKey(noise_seed), observations.shape, dtype=jnp.float64
        )
    )
    observations = observations + noise

    reconstructor = AnchoredCubicSplineReconstructor(
        times[indices],
        times,
        AnchoredCubicSplineConfig(**result["config"]["moment_reconstruction"]),
    )
    reconstructed = reconstructor.reconstruct(
        observations,
        np.mean(features[0], axis=0),
        np.mean(features[-1], axis=0),
    )
    return {
        "truth": np.mean(features, axis=1),
        "observations": observations,
        "acquisition_indices": indices,
        "target": np.asarray(reconstructed.c),
    }


def _duplicate_allowances(rows: list[dict[str, Any]]) -> dict[int, str]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(rows):
        key = tuple(round(float(value), 12) for value in row["eta"])
        groups.setdefault(key, []).append(index)
    labels: dict[int, str] = {}
    for indices in groups.values():
        if len(indices) < 2:
            continue
        allowances = [float(rows[index]["allowance_percent"]) for index in indices]
        label = f"same selected design across {min(allowances):g}–{max(allowances):g}% allowances"
        labels.update({index: label for index in indices})
    return labels


def _field_figure(
    *,
    times: np.ndarray,
    densities: np.ndarray,
    observable: dict[str, np.ndarray],
    eta: np.ndarray,
    result: dict[str, Any],
    title: str,
    subtitle: str,
    output: Path,
    stem: str,
) -> list[Path]:
    physics = result["config"]["physics"]
    measurement = result["config"]["measurement"]
    box = np.asarray(physics["box"], dtype=float)
    centers = np.asarray(eta, dtype=float).reshape(-1, 2)
    sensor_width = float(measurement["sensor_width"])
    snapshot_indices = np.unique(
        np.rint(np.linspace(0, len(times) - 1, 5)).astype(int)
    )

    density_low, density_high = np.percentile(densities, [1.0, 99.5])
    sensor_low, sensor_high = np.min(observable["target"]), np.max(observable["target"])
    sensor_norm = Normalize(vmin=float(sensor_low), vmax=float(sensor_high))
    sensor_cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(16.2, 7.15), layout="constrained")
    grid = fig.add_gridspec(2, len(snapshot_indices), height_ratios=(1.2, 0.85))
    image_axes = [fig.add_subplot(grid[0, column]) for column in range(len(snapshot_indices))]
    trace_ax = fig.add_subplot(grid[1, :])
    fig.suptitle(title, fontsize=17, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", color=GRAY, fontsize=10)

    image = None
    for ax, time_index in zip(image_axes, snapshot_indices):
        image = ax.imshow(
            densities[time_index],
            origin="lower",
            extent=(0.0, float(box[0]), 0.0, float(box[1])),
            cmap="magma",
            vmin=float(density_low),
            vmax=float(density_high),
            interpolation="bilinear",
            aspect="equal",
        )
        for sensor_index, (center, identity_color) in enumerate(
            zip(centers, SENSOR_COLORS)
        ):
            value = float(observable["target"][time_index, sensor_index])
            ax.add_patch(
                Circle(
                    center,
                    sensor_width,
                    facecolor="none",
                    edgecolor=identity_color,
                    linewidth=2.0,
                    alpha=0.95,
                    zorder=4,
                )
            )
            ax.scatter(
                *center,
                s=105,
                color=sensor_cmap(sensor_norm(value)),
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
            )
            ax.text(
                *center,
                str(sensor_index + 1),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
        ax.set(
            title=f"t = {times[time_index]:.2f}",
            xlim=(0.0, float(box[0])),
            ylim=(0.0, float(box[1])),
            xticks=[0.0, 1.0, 2.0],
            yticks=[0.0, 0.5, 1.0],
            xlabel="x",
        )
        ax.grid(False)
    image_axes[0].set_ylabel("y")
    for ax in image_axes[1:]:
        ax.set_yticklabels([])

    assert image is not None
    density_bar = fig.colorbar(image, ax=image_axes, location="bottom", shrink=0.72, pad=0.03)
    density_bar.set_label("empirical one-particle density")
    sensor_bar = fig.colorbar(
        ScalarMappable(norm=sensor_norm, cmap=sensor_cmap),
        ax=image_axes,
        location="right",
        shrink=0.70,
        pad=0.012,
    )
    sensor_bar.set_label(r"sensor fill: reconstructed $c_j(t)$")

    acquisition_indices = observable["acquisition_indices"]
    for sensor_index, color in enumerate(SENSOR_COLORS):
        trace_ax.plot(
            times,
            observable["truth"][:, sensor_index],
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.42,
        )
        trace_ax.plot(
            times,
            observable["target"][:, sensor_index],
            color=color,
            linewidth=2.25,
            label=f"sensor {sensor_index + 1}",
        )
        trace_ax.scatter(
            times[acquisition_indices],
            observable["observations"][:, sensor_index],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=40,
            zorder=4,
        )
    for time_index in snapshot_indices:
        trace_ax.axvline(times[time_index], color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    trace_ax.set(
        xlabel="normalized time",
        ylabel=r"local-density observable $c_j(t)$",
        xlim=(float(times[0]), float(times[-1])),
        title="The fixed sensors observe time-varying local density",
    )
    trace_ax.legend(loc="upper left", ncols=4)
    trace_ax.legend(
        handles=[
            *trace_ax.get_legend_handles_labels()[0],
            Line2D([0], [0], color=GRAY, linestyle="--", alpha=0.5, label="full truth-bank mean"),
            Line2D([0], [0], marker="o", color="white", markerfacecolor=GRAY, label="sparse acquired value"),
        ],
        labels=[
            *trace_ax.get_legend_handles_labels()[1],
            "full truth-bank mean",
            "sparse acquired value",
        ],
        loc="upper left",
        ncols=3,
        fontsize=8.5,
    )
    trace_ax.text(
        0.995,
        0.04,
        f"rings are fixed at width ℓ = {sensor_width:g}; sensor fills encode reconstructed $c_j(t)$",
        transform=trace_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=GRAY,
    )
    return _save(fig, output, stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot certified skyrmion density and observable evolution"
    )
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
        help="allow non-certified input; output remains explicitly labeled Exploratory",
    )
    args = parser.parse_args()
    data, results = _load(args.pareto, allow_exploratory=args.allow_exploratory)
    output = args.output or args.pareto.parent / "publication_figures" / "field_observables"
    output.mkdir(parents=True, exist_ok=True)
    status = "Certified" if data.get("certified") else "Exploratory"

    result_paths = [
        _resolve_result_path(row["result"], args.pareto) for row in data["rows"]
    ]
    truth_path = _truth_path(result_paths[0])
    with np.load(truth_path) as truth:
        times = np.asarray(truth["times"], dtype=np.float64)
        configurations = np.asarray(truth["design"], dtype=np.float64)
    box = np.asarray(results[0]["config"]["physics"]["box"], dtype=np.float64)
    densities = _density_frames(configurations, box=box)
    duplicate_labels = _duplicate_allowances(data["rows"])

    figures: list[Path] = []
    law_eta = np.asarray(results[0]["law_anchor"]["eta"], dtype=np.float64)
    law_observable = _observable_trajectory(
        configurations, times, law_eta, results[0]
    )
    with plt.rc_context(STYLE):
        figures.extend(
            _field_figure(
                times=times,
                densities=densities,
                observable=law_observable,
                eta=law_eta,
                result=results[0],
                title=f"{status} Law design · observed skyrmion field",
                subtitle="Frozen truth trajectory with the four selected local-density observables",
                output=output,
                stem="field_observables_law",
            )
        )
        for index, (row, result) in enumerate(zip(data["rows"], results)):
            eta = np.asarray(row["eta"], dtype=np.float64)
            observable = _observable_trajectory(configurations, times, eta, result)
            allowance = float(row["allowance_percent"])
            plateau = duplicate_labels.get(index, "independently selected geometry")
            figures.extend(
                _field_figure(
                    times=times,
                    densities=densities,
                    observable=observable,
                    eta=eta,
                    result=result,
                    title=f"{status} Full design · {allowance:g}% risk allowance",
                    subtitle=f"Frozen truth trajectory · {plateau}",
                    output=output,
                    stem=f"field_observables_{allowance:g}pct".replace(".", "p"),
                )
            )
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
