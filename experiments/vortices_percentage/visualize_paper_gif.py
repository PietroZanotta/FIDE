"""Animate the paper-style vortices visualization over its scientific time grid.

The output keeps the hidden population and corrected law as two large panels,
with all four sensor views stacked at the right.  It uses the same frozen banks,
Full geometry, moment reconstruction, and exponential-family projection as
``visualize_paper.py`` and writes a README-friendly looping GIF.

From the repository root::

    .venv/bin/python experiments/vortices_percentage/visualize_paper_gif.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import visualize_paper as paper  # noqa: E402


DEFAULT_OUTPUT = SCRIPT_DIR / "figures" / "vortices_population_correction_sensors.gif"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=paper.DEFAULT_CONFIG)
    parser.add_argument("--pareto", type=Path, default=paper.DEFAULT_PARETO)
    parser.add_argument("--truth-bank", type=Path, default=paper.DEFAULT_TRUTH_BANK)
    parser.add_argument("--reference-bank", type=Path, default=paper.DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=paper.DEFAULT_VALIDATION_BANK)
    parser.add_argument("--allowance", type=float, default=5.0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--grid-nx", type=int, default=220)
    parser.add_argument("--endpoint-pause-ms", type=int, default=500)
    return parser.parse_args()


def _output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved if resolved.suffix.lower() == ".gif" else resolved.with_suffix(".gif")


def _prepare_animation_data(args: argparse.Namespace) -> dict[str, Any]:
    cfg = paper._load_json(args.config)
    centers = paper._full_centers(paper._load_json(args.pareto), args.allowance)
    measurement = cfg["measurement"]
    width = float(measurement["sensor_width"])

    with np.load(args.truth_bank, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=np.float64)
        truth_particles = np.asarray(arrays["particles"], dtype=np.float64)
    with np.load(args.reference_bank, allow_pickle=False) as arrays:
        reference_times = np.asarray(arrays["times"], dtype=np.float64)
        reference_particles = np.asarray(arrays["nodes"], dtype=np.float64)
        base_weights = np.asarray(arrays["weights"], dtype=np.float64)
    with np.load(args.validation_bank, allow_pickle=False) as arrays:
        sample_indices_bank = np.asarray(arrays["sample_indices"], dtype=np.int64)
        detector_z_bank = np.asarray(arrays["detector_z"], dtype=np.float64)

    if not np.array_equal(times, reference_times):
        raise ValueError("truth and reference banks use different time grids")
    if args.trial < 0 or args.trial >= sample_indices_bank.shape[0]:
        raise ValueError(
            f"trial must lie in [0, {sample_indices_bank.shape[0] - 1}], got {args.trial}"
        )
    acquisition = paper._acquisition_indices(
        len(times), int(measurement["acquisition_k"])
    )
    sample_indices = sample_indices_bank[args.trial]
    detector_z = detector_z_bank[args.trial]
    acquired_features = [
        paper._sensor_features(truth_particles[index], centers, width)
        for index in acquisition
    ]
    exact = np.stack([features.mean(axis=0) for features in acquired_features])
    observations = np.stack(
        [
            features[sample_indices[position]].mean(axis=0)
            + float(measurement["obs_noise_std"]) * detector_z[position]
            for position, features in enumerate(acquired_features)
        ]
    )
    observations[[0, -1]] = exact[[0, -1]]
    targets = paper._reconstruct_moments(
        times,
        acquisition,
        observations,
        exact[0],
        exact[-1],
        cfg["moment_reconstruction"],
    )

    corrected_weights: list[np.ndarray] = []
    ess_fractions: list[float] = []
    multipliers: list[np.ndarray] = []
    residuals: list[float] = []
    for index in range(len(times)):
        weights, ess, multiplier, residual = paper._project_weights(
            reference_particles[index],
            base_weights[index],
            centers,
            width,
            targets[index],
        )
        corrected_weights.append(weights)
        ess_fractions.append(ess)
        multipliers.append(multiplier)
        residuals.append(residual)

    return {
        "config": cfg,
        "allowance": float(args.allowance),
        "trial": int(args.trial),
        "times": times,
        "centers": centers,
        "width": width,
        "truth_particles": truth_particles,
        "reference_particles": reference_particles,
        "targets": targets,
        "corrected_weights": corrected_weights,
        "ess_fractions": np.asarray(ess_fractions),
        "multipliers": np.asarray(multipliers),
        "residuals": np.asarray(residuals),
    }


def _precompute_fields(data: dict[str, Any], nx: int) -> dict[str, Any]:
    if nx < 128:
        raise ValueError("grid-nx must be at least 128")
    ny = nx // 2
    extent = (0.0, 2.0, 0.0, 1.0)
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    uniform = np.full(
        data["truth_particles"].shape[1],
        1.0 / data["truth_particles"].shape[1],
    )
    hidden = [
        paper._particle_density(points, uniform, nx=nx, ny=ny)
        for points in data["truth_particles"]
    ]
    corrected = [
        paper._particle_density(points, weights, nx=nx, ny=ny)
        for points, weights in zip(
            data["reference_particles"], data["corrected_weights"], strict=True
        )
    ]
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_features = paper._sensor_features(
        grid_points, data["centers"], float(data["width"])
    )
    sensor_views = [
        [density * grid_features[:, sensor].reshape(ny, nx) for sensor in range(4)]
        for density in hidden
    ]
    density_values = np.concatenate([field.ravel() for field in hidden + corrected])
    sensor_values = np.concatenate(
        [field.ravel() for group in sensor_views for field in group]
    )
    return {
        "extent": extent,
        "xx": xx,
        "yy": yy,
        "hidden": hidden,
        "corrected": corrected,
        "sensor_views": sensor_views,
        "density_norm": mpl.colors.PowerNorm(
            gamma=0.52,
            vmin=0.0,
            vmax=float(np.quantile(density_values, 0.998)),
        ),
        "sensor_norm": mpl.colors.PowerNorm(
            gamma=0.45,
            vmin=0.0,
            vmax=float(np.quantile(sensor_values, 0.998)),
        ),
    }


def _render_frames(
    data: dict[str, Any], fields: dict[str, Any], *, dpi: int
) -> list[Image.Image]:
    paper._style()
    fig = plt.figure(figsize=(13.0, 4.6), dpi=dpi, constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=(1.0, 1.0, 0.29),
        left=0.035,
        right=0.925,
        bottom=0.15,
        top=0.80,
        wspace=0.06,
        hspace=0.28,
    )
    hidden_ax = fig.add_subplot(grid[:, 0])
    corrected_ax = fig.add_subplot(grid[:, 1])
    sensor_axes = tuple(fig.add_subplot(grid[index, 2]) for index in range(4))
    all_axes = (hidden_ax, corrected_ax, *sensor_axes)
    progress_ax = fig.add_axes((0.055, 0.05, 0.85, 0.025))
    colorbar_ax = fig.add_axes((0.947, 0.28, 0.012, 0.40))
    sensor_maps = tuple(
        paper._sensor_cmap(color, index)
        for index, color in enumerate(paper.SENSOR_COLORS)
    )
    scalar = mpl.cm.ScalarMappable(norm=fields["density_norm"], cmap=paper.FLOW_CMAP)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("probability density", fontsize=8.6, color="#555A62")
    colorbar.ax.tick_params(labelsize=7.2, length=2.5, colors="#666A70")
    colorbar.outline.set_visible(False)

    fig.suptitle(
        "Four sensors in a moving double gyre",
        x=0.035,
        y=0.965,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.035,
        0.895,
        rf"Authoritative Full geometry at {data['allowance']:g}% allowance  ·  frozen validation trial {data['trial']}",
        ha="left",
        fontsize=8.2,
        color="#6A6F76",
    )
    time_text = fig.text(
        0.925,
        0.845,
        "",
        ha="right",
        fontsize=12.5,
        fontweight="bold",
        color="#313741",
    )
    gx = np.linspace(0.02, 1.98, 31)
    gy = np.linspace(0.02, 0.98, 17)
    flow_xx, flow_yy = np.meshgrid(gx, gy, indexing="xy")
    frames: list[Image.Image] = []
    start, end = float(data["times"][0]), float(data["times"][-1])

    for frame_index, time in enumerate(data["times"]):
        for ax in all_axes:
            ax.clear()
        hidden_ax.imshow(
            fields["hidden"][frame_index],
            origin="lower",
            extent=fields["extent"],
            cmap=paper.FLOW_CMAP,
            norm=fields["density_norm"],
            interpolation="bilinear",
        )
        velocity_x, velocity_y = paper._double_gyre_velocity(
            flow_xx, flow_yy, float(time), data["config"]["truth"]
        )
        hidden_ax.streamplot(
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
            fields["corrected"][frame_index],
            origin="lower",
            extent=fields["extent"],
            cmap=paper.FLOW_CMAP,
            norm=fields["density_norm"],
            interpolation="bilinear",
        )
        for sensor, center in enumerate(data["centers"]):
            paper._add_sensor_marker(
                corrected_ax,
                center,
                float(data["width"]),
                paper.SENSOR_COLORS[sensor],
                compact=False,
            )
        hidden_ax.set_title(
            "HIDDEN POPULATION", fontsize=10.2, fontweight="bold", color="#4A3A62"
        )
        corrected_ax.set_title(
            "CORRECTED LAW", fontsize=10.2, fontweight="bold", color="#8B3E46"
        )
        for sensor, sensor_ax in enumerate(sensor_axes):
            sensor_ax.imshow(
                fields["sensor_views"][frame_index][sensor],
                origin="lower",
                extent=fields["extent"],
                cmap=sensor_maps[sensor],
                norm=fields["sensor_norm"],
                interpolation="bilinear",
            )
            paper._add_sensor_marker(
                sensor_ax,
                data["centers"][sensor],
                float(data["width"]),
                paper.SENSOR_COLORS[sensor],
                compact=True,
            )
            sensor_ax.set_title(
                rf"S{sensor + 1} SEES  $y={data['targets'][frame_index, sensor]:.3f}$",
                fontsize=6.7,
                fontweight="bold",
                color=paper.SENSOR_COLORS[sensor],
                pad=1.5,
            )
        for ax in all_axes:
            paper._clean_axis(ax)

        time_text.set_text(rf"$t={float(time):.2f}$")
        progress = 0.0 if end == start else (float(time) - start) / (end - start)
        progress_ax.clear()
        progress_ax.set_xlim(0.0, 1.0)
        progress_ax.set_ylim(-1.0, 1.0)
        progress_ax.axis("off")
        progress_ax.plot([0.0, 1.0], [0.0, 0.0], color="#D1CBC0", lw=4.0, solid_capstyle="round")
        progress_ax.plot([0.0, progress], [0.0, 0.0], color="#596675", lw=4.0, solid_capstyle="round")
        progress_ax.scatter([progress], [0.0], s=38, color="#D84C5B", edgecolor=paper.PAPER_BACKGROUND, linewidth=1.0, zorder=3)
        progress_ax.text(0.0, -0.75, f"{start:g}", ha="center", va="top", fontsize=7.3, color="#777B82")
        progress_ax.text(1.0, -0.75, f"{end:g}", ha="center", va="top", fontsize=7.3, color="#777B82")
        progress_ax.text(0.5, -0.75, "time", ha="center", va="top", fontsize=7.3, color="#777B82")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy(), mode="RGB"))

    plt.close(fig)
    return frames


def _save_readme_gif(
    frames: list[Image.Image], output: Path, *, fps: float, endpoint_pause_ms: int
) -> None:
    if not frames:
        raise ValueError("no animation frames to save")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    fields = _precompute_fields(data, args.grid_nx)
    frames = _render_frames(data, fields, dpi=args.dpi)
    output = _output_path(args.output)
    _save_readme_gif(
        frames, output, fps=args.fps, endpoint_pause_ms=args.endpoint_pause_ms
    )
    print(f"saved {output}")
    print(
        f"frames={len(frames)}, dimensions={frames[0].width}x{frames[0].height}, "
        f"bytes={output.stat().st_size}"
    )
    print(
        "projection diagnostics: "
        f"minimum ESS/base ESS={float(np.min(data['ess_fractions'])):.6f}, "
        f"maximum calibration residual={float(np.max(data['residuals'])):.3e}, "
        f"maximum |lambda|={float(np.max(np.abs(data['multipliers']))):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
