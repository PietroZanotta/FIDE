"""Animate the paper-style toy visualization over the scientific time grid.

The GIF uses the same authoritative 5% Full sensor geometry, frozen validation
trial, reference particles, density rasterization, colors, and sensor markings
as ``visualize_paper.py``.  Time is represented by animation rather than by
four side-by-side columns.  The output is a compact, universally supported GIF
that can be embedded directly in a Markdown README.

From the repository root::

    .venv/bin/python experiments/toy_example_percentage/visualize_paper_gif.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mfsi.moments import QuadraticBridgeConfig, fit_quadratic_bridge_gls  # noqa: E402
import visualize_paper as paper  # noqa: E402


DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "toy_population_correction_sensors.gif"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=paper.DEFAULT_CONFIG)
    parser.add_argument("--pareto", type=Path, default=paper.DEFAULT_PARETO)
    parser.add_argument("--reference-bank", type=Path, default=paper.DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=paper.DEFAULT_VALIDATION_BANK)
    parser.add_argument("--allowance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--grid-n", type=int, default=201)
    parser.add_argument(
        "--endpoint-pause-ms",
        type=int,
        default=500,
        help="display duration of the first and last frames",
    )
    return parser.parse_args()


def _output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved if resolved.suffix.lower() == ".gif" else resolved.with_suffix(".gif")


def _prepare_animation_data(args: argparse.Namespace) -> dict[str, Any]:
    cfg = paper._load_json(args.config)
    pareto = paper._load_json(args.pareto)
    angles = paper._full_geometry(pareto, args.allowance)
    measurement = cfg["measurement"]
    population = cfg["population"]
    sensor_width = float(measurement["sensor_width"])
    sensor_radius = float(measurement["sensor_radius"])
    sensor_centers = sensor_radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )

    with np.load(args.validation_bank, allow_pickle=False) as validation:
        alphas = np.asarray(validation["alphas"], dtype=np.float64)
        trial = int(np.argmin(np.abs(alphas - np.deg2rad(45.0))))
        alpha = float(alphas[trial])
        hidden_masses = np.asarray(validation["masses"][trial], dtype=np.float64)
        sample_indices = np.asarray(
            validation["sample_indices"][trial], dtype=np.int64
        )
        detector_z = np.asarray(validation["detector_z"][trial], dtype=np.float64)
        acquisition_indices = np.asarray(
            validation["acquisition_indices"], dtype=np.int64
        )

    with np.load(args.reference_bank, allow_pickle=False) as reference:
        times = np.asarray(reference["times"], dtype=np.float64)
        reference_particles = np.asarray(
            reference["reference_particles"], dtype=np.float64
        )
        base_weights = np.asarray(reference["base_weights"], dtype=np.float64)
        in_domain = np.asarray(reference["in_domain_mask"], dtype=bool)

    if hidden_masses.shape[0] != len(times):
        raise ValueError("validation and reference banks use different time grids")
    if np.any(np.diff(acquisition_indices) <= 0):
        raise ValueError("acquisition indices must be strictly increasing")

    hidden_grid_n = int(round(math.sqrt(hidden_masses.shape[-1])))
    if hidden_grid_n**2 != hidden_masses.shape[-1]:
        raise ValueError("hidden validation masses do not form a square grid")
    half_width = float(population["domain_half_width"])
    hidden_dx = 2.0 * half_width / hidden_grid_n
    hidden_centers = -half_width + (np.arange(hidden_grid_n) + 0.5) * hidden_dx
    hidden_xx, hidden_yy = np.meshgrid(hidden_centers, hidden_centers, indexing="xy")
    hidden_points = np.column_stack((hidden_xx.ravel(), hidden_yy.ravel()))
    hidden_features = paper._sensor_features(
        hidden_points, sensor_centers, sensor_width
    )

    acquisition_masses = hidden_masses[acquisition_indices]
    exact_targets = acquisition_masses @ hidden_features
    second_moments = np.einsum(
        "kg,gi,gj->kij", acquisition_masses, hidden_features, hidden_features
    )
    covariance = second_moments - np.einsum(
        "ki,kj->kij", exact_targets, exact_targets
    )
    reconstruction_cfg = cfg.get("moment_reconstruction", {})
    covariance /= float(measurement["finite_n"])
    covariance += (
        float(measurement["obs_noise_std"]) ** 2
        + float(reconstruction_cfg.get("variance_floor", 1.0e-10))
    ) * np.eye(len(sensor_centers), dtype=np.float64)[None, :, :]

    observations = (
        np.mean(hidden_features[sample_indices], axis=1)
        + float(measurement["obs_noise_std"]) * detector_z
    )
    observations[[0, -1]] = exact_targets[[0, -1]]
    reconstruction = fit_quadratic_bridge_gls(
        jnp.asarray(times[acquisition_indices]),
        jnp.asarray(observations),
        jnp.asarray(covariance),
        jnp.asarray(exact_targets[0]),
        jnp.asarray(exact_targets[-1]),
        jnp.asarray(times),
        QuadraticBridgeConfig(
            ridge_rel=float(reconstruction_cfg.get("ridge_rel", 1.0e-12)),
            variance_floor=float(
                reconstruction_cfg.get("variance_floor", 1.0e-10)
            ),
        ),
    )
    targets = np.asarray(reconstruction.c, dtype=np.float64)

    corrected_weights = []
    ess_fractions = []
    multipliers = []
    for time_index in range(len(times)):
        masked_base = np.where(
            in_domain[time_index], base_weights[time_index], 0.0
        )
        weights, ess_fraction, multiplier = paper._project_weights(
            reference_particles[time_index],
            masked_base,
            sensor_centers,
            sensor_width,
            targets[time_index],
        )
        corrected_weights.append(weights)
        ess_fractions.append(ess_fraction)
        multipliers.append(multiplier)

    return {
        "config": cfg,
        "angles": angles,
        "allowance": float(args.allowance),
        "trial": trial,
        "alpha": alpha,
        "half_width": half_width,
        "times": times,
        "sensor_centers": sensor_centers,
        "sensor_width": sensor_width,
        "reference_particles": reference_particles,
        "targets": targets,
        "corrected_weights": corrected_weights,
        "ess_fractions": np.asarray(ess_fractions),
        "multipliers": np.asarray(multipliers),
    }


def _precompute_fields(data: dict[str, Any], grid_n: int) -> dict[str, Any]:
    if grid_n < 64:
        raise ValueError("grid-n must be at least 64")
    cfg = data["config"]
    population = cfg["population"]
    half_width = float(data["half_width"])
    centers = np.linspace(-half_width, half_width, grid_n)
    xx, yy = np.meshgrid(centers, centers, indexing="xy")
    extent = (-half_width, half_width, -half_width, half_width)

    hidden = [
        paper._mixture_density(
            xx,
            yy,
            float(time),
            float(data["alpha"]),
            radius=float(population["radius"]),
            sigma=float(population["sigma"]),
        )
        for time in data["times"]
    ]
    bandwidth = float(
        cfg.get("raster", {}).get("authoritative_positive", {}).get(
            "frozen_bandwidth", 0.417530106552
        )
    )
    corrected = [
        paper._particle_density(
            particles,
            weights,
            half_width=half_width,
            grid_n=grid_n,
            bandwidth=bandwidth,
        )
        for particles, weights in zip(
            data["reference_particles"], data["corrected_weights"], strict=True
        )
    ]
    display_points = np.column_stack((xx.ravel(), yy.ravel()))
    sensor_features = paper._sensor_features(
        display_points,
        np.asarray(data["sensor_centers"]),
        float(data["sensor_width"]),
    )
    sensor_views = [
        [density * sensor_features[:, sensor].reshape(xx.shape) for sensor in range(2)]
        for density in hidden
    ]
    return {
        "xx": xx,
        "yy": yy,
        "extent": extent,
        "hidden": hidden,
        "corrected": corrected,
        "sensor_views": sensor_views,
        "density_vmax": max(
            max(float(np.max(value)) for value in hidden),
            max(float(np.max(value)) for value in corrected),
        ),
        "sensor_vmax": max(
            float(np.max(value)) for pair in sensor_views for value in pair
        ),
    }


def _render_frames(
    data: dict[str, Any], fields: dict[str, Any], *, dpi: int
) -> list[Image.Image]:
    paper._style()
    fig = plt.figure(figsize=(11.7, 5.9), dpi=dpi, constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.47),
        left=0.045,
        right=0.925,
        bottom=0.14,
        top=0.82,
        wspace=0.075,
        hspace=0.16,
    )
    hidden_ax = fig.add_subplot(grid[:, 0])
    corrected_ax = fig.add_subplot(grid[:, 1])
    sensor_axes = (fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2]))
    all_axes = (hidden_ax, corrected_ax, *sensor_axes)
    progress_ax = fig.add_axes((0.065, 0.052, 0.84, 0.024))
    colorbar_ax = fig.add_axes((0.945, 0.29, 0.012, 0.40))
    density_map = mpl.colormaps["magma"]
    sensor_maps = tuple(paper._sensor_cmap(color) for color in paper.SENSOR_COLORS)
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=fields["density_vmax"]),
        cmap=density_map,
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("probability density", fontsize=9.0, color="#555A62")
    colorbar.ax.tick_params(labelsize=7.8, length=2.5, colors="#666A70")
    colorbar.outline.set_visible(False)

    angles = np.rad2deg(data["angles"])
    fig.suptitle(
        "Two sensors, one evolving population",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.045,
        0.905,
        (
            rf"Authoritative Full geometry at {data['allowance']:.0f}% allowance  ·  "
            rf"angles {angles[0]:.1f}° / {angles[1]:.1f}°  ·  "
            rf"frozen validation trial {data['trial']} ($\alpha={np.rad2deg(data['alpha']):.1f}°$)"
        ),
        ha="left",
        fontsize=8.2,
        color="#6A6F76",
    )
    time_text = fig.text(
        0.925,
        0.855,
        "",
        ha="right",
        fontsize=13,
        fontweight="bold",
        color="#313741",
    )
    frames: list[Image.Image] = []
    display_limit = 2.45
    contour_levels = np.linspace(
        0.18 * fields["density_vmax"], 0.82 * fields["density_vmax"], 4
    )

    for frame_index, time in enumerate(data["times"]):
        hidden = fields["hidden"][frame_index]
        corrected = fields["corrected"][frame_index]
        targets = data["targets"][frame_index]
        for ax in all_axes:
            ax.clear()

        hidden_ax.imshow(
            hidden,
            origin="lower",
            extent=fields["extent"],
            cmap=density_map,
            vmin=0.0,
            vmax=fields["density_vmax"],
            interpolation="bilinear",
        )
        hidden_ax.contour(
            fields["xx"], fields["yy"], hidden,
            levels=contour_levels, colors="white", linewidths=0.46, alpha=0.45,
        )
        corrected_ax.imshow(
            corrected,
            origin="lower",
            extent=fields["extent"],
            cmap=density_map,
            vmin=0.0,
            vmax=fields["density_vmax"],
            interpolation="bilinear",
        )
        corrected_ax.contour(
            fields["xx"], fields["yy"], corrected,
            levels=contour_levels, colors="white", linewidths=0.46, alpha=0.42,
        )
        for sensor, center in enumerate(data["sensor_centers"]):
            paper._add_sensor_marker(
                corrected_ax, center, float(data["sensor_width"]),
                paper.SENSOR_COLORS[sensor], compact=False,
            )

        hidden_ax.set_title(
            "HIDDEN POPULATION", fontsize=10.5, fontweight="bold", color="#4A3A62"
        )
        corrected_ax.set_title(
            "CORRECTED LAW", fontsize=10.5, fontweight="bold", color="#8B3E46"
        )
        for sensor in range(2):
            sensor_axes[sensor].imshow(
                fields["sensor_views"][frame_index][sensor],
                origin="lower",
                extent=fields["extent"],
                cmap=sensor_maps[sensor],
                vmin=0.0,
                vmax=fields["sensor_vmax"],
                interpolation="bilinear",
            )
            paper._add_sensor_marker(
                sensor_axes[sensor],
                data["sensor_centers"][sensor],
                float(data["sensor_width"]),
                paper.SENSOR_COLORS[sensor],
                compact=False,
            )
            sensor_axes[sensor].set_title(
                rf"SENSOR {sensor + 1} SEES   $y={targets[sensor]:.3f}$",
                fontsize=8.7,
                fontweight="bold",
                color=paper.SENSOR_COLORS[sensor],
            )
        for ax in all_axes:
            paper._clean_density_axis(ax, display_limit)

        time_text.set_text(rf"$t={float(time):.2f}$")
        progress_ax.clear()
        progress_ax.set_xlim(0.0, 1.0)
        progress_ax.set_ylim(-1.0, 1.0)
        progress_ax.axis("off")
        progress_ax.plot([0.0, 1.0], [0.0, 0.0], color="#D1CBC0", lw=4.0, solid_capstyle="round")
        progress_ax.plot([0.0, float(time)], [0.0, 0.0], color="#596675", lw=4.0, solid_capstyle="round")
        progress_ax.scatter([float(time)], [0.0], s=38, color="#D84C5B", edgecolor=paper.PAPER_BACKGROUND, linewidth=1.0, zorder=3)
        progress_ax.text(0.0, -0.75, "0", ha="center", va="top", fontsize=7.5, color="#777B82")
        progress_ax.text(1.0, -0.75, "1", ha="center", va="top", fontsize=7.5, color="#777B82")
        progress_ax.text(0.5, -0.75, "time", ha="center", va="top", fontsize=7.5, color="#777B82")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy(), mode="RGB"))

    plt.close(fig)
    return frames


def _save_readme_gif(
    frames: list[Image.Image],
    output: Path,
    *,
    fps: float,
    endpoint_pause_ms: int,
) -> None:
    if not frames:
        raise ValueError("no animation frames to save")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)

    # A shared palette prevents frame-to-frame color flicker and gives Markdown
    # renderers a substantially smaller asset than independent RGB frames.
    palette_width = 160
    thumbnails = [
        frame.resize(
            (palette_width, max(1, round(frame.height * palette_width / frame.width))),
            Image.Resampling.LANCZOS,
        )
        for frame in frames
    ]
    palette_source = Image.new(
        "RGB",
        (palette_width * len(thumbnails), max(image.height for image in thumbnails)),
        paper.PAPER_BACKGROUND,
    )
    for index, thumbnail in enumerate(thumbnails):
        palette_source.paste(thumbnail, (index * palette_width, 0))
    palette = palette_source.quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    frame_ms = max(20, int(round(1000.0 / fps)))
    durations = [frame_ms] * len(quantized)
    durations[0] = max(frame_ms, endpoint_pause_ms)
    durations[-1] = max(frame_ms, endpoint_pause_ms)
    quantized[0].save(
        output,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )


def main() -> int:
    args = _parse_args()
    data = _prepare_animation_data(args)
    fields = _precompute_fields(data, args.grid_n)
    frames = _render_frames(data, fields, dpi=args.dpi)
    output = _output_path(args.output)
    _save_readme_gif(
        frames,
        output,
        fps=args.fps,
        endpoint_pause_ms=args.endpoint_pause_ms,
    )
    print(f"saved {output}")
    print(
        f"frames={len(frames)}, dimensions={frames[0].width}x{frames[0].height}, "
        f"bytes={output.stat().st_size}"
    )
    print(
        "projection diagnostics: "
        f"minimum relative ESS={float(np.min(data['ess_fractions'])):.6f}, "
        f"maximum |lambda|={float(np.max(np.abs(data['multipliers']))):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
