"""Animate the paper-style active-nematic defect visualization.

The animation uses the authoritative Full geometry and frozen defect,
validation-noise, reference-bank, and view-manifest artifacts.  Signed defect
density is shown in the two large panels; the two periodic sensor views are
stacked at the right and retain separate positive/negative charge readouts.
The output is a compact looping GIF intended for direct README embedding.

From the repository root::

    .venv/bin/python experiments/active_nematic_unbalance_percentage/visualize_paper_gif.py
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


DEFAULT_OUTPUT = (
    SCRIPT_DIR / "figures" / "active_nematic_defect_correction_sensors.gif"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, default=paper.DEFAULT_PARETO)
    parser.add_argument("--frozen-inputs", type=Path, default=paper.FROZEN_DIR)
    parser.add_argument("--allowance", type=float, default=3.0)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--reference-seed", type=int, default=20260818)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--grid-n", type=int, default=180)
    parser.add_argument("--endpoint-pause-ms", type=int, default=500)
    return parser.parse_args()


def _output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved if resolved.suffix.lower() == ".gif" else resolved.with_suffix(".gif")


def _precompute_fields(data: dict[str, Any], grid_n: int) -> dict[str, Any]:
    if grid_n < 128:
        raise ValueError("grid-n must be at least 128")
    box = float(data["box"])
    coordinates = (np.arange(grid_n) + 0.5) * box / grid_n
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    uniform = np.full(
        data["truth"]["plus"].shape[1],
        1.0 / data["truth"]["plus"].shape[1],
    )
    hidden_signed: list[np.ndarray] = []
    corrected_signed: list[np.ndarray] = []
    hidden_total: list[np.ndarray] = []

    for time_index in range(len(data["physical_times"])):
        plus_hidden = data["true_mass"]["plus"][time_index] * paper._spatial_density(
            data["truth"]["plus"][time_index],
            uniform,
            box=box,
            grid_n=grid_n,
        )
        minus_hidden = data["true_mass"]["minus"][time_index] * paper._spatial_density(
            data["truth"]["minus"][time_index],
            uniform,
            box=box,
            grid_n=grid_n,
        )
        plus_corrected = data["target_mass"]["plus"][time_index] * paper._spatial_density(
            data["reference"]["plus"]["nodes"][time_index],
            data["corrected_weights"]["plus"][time_index],
            box=box,
            grid_n=grid_n,
        )
        minus_corrected = data["target_mass"]["minus"][time_index] * paper._spatial_density(
            data["reference"]["minus"]["nodes"][time_index],
            data["corrected_weights"]["minus"][time_index],
            box=box,
            grid_n=grid_n,
        )
        hidden_signed.append(plus_hidden - minus_hidden)
        corrected_signed.append(plus_corrected - minus_corrected)
        hidden_total.append(plus_hidden + minus_hidden)

    windows = [
        paper._sensor_window_grid(
            xx, yy, center, box, float(data["width"])
        )
        for center in data["centers"]
    ]
    sensor_views = [
        [density * windows[sensor] for sensor in range(len(windows))]
        for density in hidden_total
    ]
    signed_values = np.concatenate(
        [field.ravel() for field in hidden_signed + corrected_signed]
    )
    signed_max = float(np.quantile(np.abs(signed_values), 0.9975))
    sensor_values = np.concatenate(
        [field.ravel() for group in sensor_views for field in group]
    )
    return {
        "extent": (0.0, box, 0.0, box),
        "hidden_signed": hidden_signed,
        "corrected_signed": corrected_signed,
        "sensor_views": sensor_views,
        "signed_norm": mpl.colors.SymLogNorm(
            linthresh=max(0.025 * signed_max, 1.0e-8),
            linscale=0.65,
            vmin=-signed_max,
            vmax=signed_max,
            base=10,
        ),
        "sensor_norm": mpl.colors.PowerNorm(
            gamma=0.52,
            vmin=0.0,
            vmax=float(np.quantile(sensor_values, 0.998)),
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
    sensor_maps = tuple(
        paper._sensor_cmap(color, index)
        for index, color in enumerate(paper.SENSOR_COLORS)
    )
    scalar = mpl.cm.ScalarMappable(
        norm=fields["signed_norm"], cmap=paper.SIGNED_CMAP
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.ax.tick_params(labelsize=7.2, length=2.5, colors="#666A70")
    colorbar.outline.set_visible(False)
    fig.text(
        0.993,
        0.49,
        "signed defect density  (− / + charge)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555A62",
    )

    fig.suptitle(
        "Two sensors track active-nematic defects",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#20242B",
    )
    fig.text(
        0.045,
        0.905,
        (
            rf"Authoritative Full geometry at {data['allowance']:g}% allowance  ·  "
            rf"fold {data['validation_fold']}  ·  reference {data['reference_seed']}  ·  "
            rf"trial {data['trial']}"
        ),
        ha="left",
        fontsize=8.2,
        color="#6A6F76",
    )
    time_text = fig.text(
        0.925,
        0.955,
        "",
        ha="right",
        fontsize=12.5,
        fontweight="bold",
        color="#313741",
    )
    mass_text = fig.text(
        0.925,
        0.905,
        "",
        ha="right",
        fontsize=7.5,
        color="#6A6F76",
    )
    frames: list[Image.Image] = []
    box = float(data["box"])

    for frame_index, normalized_time in enumerate(data["normalized_times"]):
        for ax in all_axes:
            ax.clear()
        hidden_ax.imshow(
            fields["hidden_signed"][frame_index],
            origin="lower",
            extent=fields["extent"],
            cmap=paper.SIGNED_CMAP,
            norm=fields["signed_norm"],
            interpolation="bilinear",
        )
        corrected_ax.imshow(
            fields["corrected_signed"][frame_index],
            origin="lower",
            extent=fields["extent"],
            cmap=paper.SIGNED_CMAP,
            norm=fields["signed_norm"],
            interpolation="bilinear",
        )
        for sensor, center in enumerate(data["centers"]):
            paper._add_periodic_sensor(
                corrected_ax,
                center,
                float(data["width"]),
                box,
                paper.SENSOR_COLORS[sensor],
                compact=False,
            )
        hidden_ax.set_title(
            "HIDDEN DEFECT POPULATION",
            fontsize=10.2,
            fontweight="bold",
            color="#4A3A62",
        )
        corrected_ax.set_title(
            "CORRECTED LAW",
            fontsize=10.2,
            fontweight="bold",
            color="#8B3E46",
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
            paper._add_periodic_sensor(
                sensor_ax,
                data["centers"][sensor],
                float(data["width"]),
                box,
                paper.SENSOR_COLORS[sensor],
                compact=True,
            )
            plus_y = data["targets"]["plus"][frame_index, 3 * sensor]
            minus_y = data["targets"]["minus"][frame_index, 3 * sensor]
            sensor_ax.set_title(
                rf"S{sensor + 1} SEES  $y_+={plus_y:.3f}$  $y_-={minus_y:.3f}$",
                fontsize=7.0,
                fontweight="bold",
                color=paper.SENSOR_COLORS[sensor],
                pad=2.0,
            )
        for ax in all_axes:
            paper._clean_axis(ax, box)

        physical_time = float(data["physical_times"][frame_index])
        time_text.set_text(rf"$t={float(normalized_time):.2f}$")
        mass_text.set_text(
            rf"physical time {physical_time:g}  ·  target mass "
            rf"$m_+={data['target_mass']['plus'][frame_index]:.1f}$ / "
            rf"$m_-={data['target_mass']['minus'][frame_index]:.1f}$"
        )
        progress = float(normalized_time)
        progress_ax.clear()
        progress_ax.set_xlim(0.0, 1.0)
        progress_ax.set_ylim(-1.0, 1.0)
        progress_ax.axis("off")
        progress_ax.plot(
            [0.0, 1.0], [0.0, 0.0], color="#D1CBC0", lw=4.0, solid_capstyle="round"
        )
        progress_ax.plot(
            [0.0, progress], [0.0, 0.0], color="#596675", lw=4.0, solid_capstyle="round"
        )
        progress_ax.scatter(
            [progress],
            [0.0],
            s=38,
            color="#D84C5B",
            edgecolor=paper.PAPER_BACKGROUND,
            linewidth=1.0,
            zorder=3,
        )
        progress_ax.text(0.0, -0.75, "0", ha="center", va="top", fontsize=7.5, color="#777B82")
        progress_ax.text(1.0, -0.75, "1", ha="center", va="top", fontsize=7.5, color="#777B82")
        progress_ax.text(0.5, -0.75, "normalized time", ha="center", va="top", fontsize=7.5, color="#777B82")

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
    data = paper._prepare_data(args)
    fields = _precompute_fields(data, args.grid_n)
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
        f"maximum calibration residual={float(np.max(data['residuals'])):.3e}, "
        f"maximum |lambda|="
        f"{max(float(np.max(np.abs(row))) for rows in data['multipliers'].values() for row in rows):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
