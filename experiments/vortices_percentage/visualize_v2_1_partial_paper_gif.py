#!/usr/bin/env python3
"""Animate the completed 2% V2.1 Full geometry on the independent holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEGACY_DIR = HERE
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(1, str(LEGACY_DIR))

import visualize_v2_1_partial_paper as static  # noqa: E402
import visualize_paper as paper  # noqa: E402
import visualize_paper_gif as legacy_gif  # noqa: E402


DEFAULT_OUTPUT = static.DEFAULT_OUTPUT_DIR / "vortices_v2_1_full_2p0.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=static.DEFAULT_CONFIG)
    parser.add_argument("--truth-bank", type=Path, default=static.DEFAULT_TRUTH_BANK)
    parser.add_argument("--reference-bank", type=Path, default=static.DEFAULT_REFERENCE_BANK)
    parser.add_argument("--validation-bank", type=Path, default=static.DEFAULT_VALIDATION_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--grid-nx", type=int, default=220)
    parser.add_argument("--endpoint-pause-ms", type=int, default=500)
    return parser.parse_args()


def render_frames(
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
        "V2.1 Full geometry at 2% allowance  ·  independent holdout trial 0",
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
        for axis in all_axes:
            axis.clear()
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
        for axis in all_axes:
            paper._clean_axis(axis)

        time_text.set_text(rf"$t={float(time):.2f}$")
        progress = 0.0 if end == start else (float(time) - start) / (end - start)
        progress_ax.clear()
        progress_ax.set_xlim(0.0, 1.0)
        progress_ax.set_ylim(-1.0, 1.0)
        progress_ax.axis("off")
        progress_ax.plot(
            [0.0, 1.0], [0.0, 0.0], color="#D1CBC0", lw=4.0, solid_capstyle="round"
        )
        progress_ax.plot(
            [0.0, progress],
            [0.0, 0.0],
            color="#596675",
            lw=4.0,
            solid_capstyle="round",
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
        progress_ax.text(
            0.0, -0.75, f"{start:g}", ha="center", va="top", fontsize=7.3, color="#777B82"
        )
        progress_ax.text(
            1.0, -0.75, f"{end:g}", ha="center", va="top", fontsize=7.3, color="#777B82"
        )
        progress_ax.text(
            0.5, -0.75, "time", ha="center", va="top", fontsize=7.3, color="#777B82"
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy()))

    plt.close(fig)
    return frames


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = static.verify_inputs(args)
    adapter = static.write_geometry_adapter(output.parent, rows)
    render_args = SimpleNamespace(
        config=args.config,
        pareto=adapter,
        truth_bank=args.truth_bank,
        reference_bank=args.reference_bank,
        validation_bank=args.validation_bank,
        allowance=2.0,
        trial=args.trial,
    )
    data = legacy_gif._prepare_animation_data(render_args)
    fields = legacy_gif._precompute_fields(data, args.grid_nx)
    frames = render_frames(data, fields, dpi=args.dpi)
    legacy_gif._save_readme_gif(
        frames, output, fps=args.fps, endpoint_pause_ms=args.endpoint_pause_ms
    )
    preview = output.with_name(f"{output.stem}_preview.png")
    frames[0].save(preview)
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE_V2_1_2PCT_CONFIRMED_HOLDOUT_GIF",
        "data_role": "INDEPENDENT_HOLDOUT_VISUALIZATION_ONLY",
        "allowance_percent": 2.0,
        "trial": args.trial,
        "namespace": 23,
        "reference_seed": 310000101,
        "frames": len(frames),
        "fps": args.fps,
        "dimensions": [frames[0].width, frames[0].height],
        "minimum_ess_fraction": float(np.min(data["ess_fractions"])),
        "maximum_calibration_residual": float(np.max(data["residuals"])),
        "maximum_absolute_multiplier": float(np.max(np.abs(data["multipliers"]))),
        "gif": str(output.relative_to(REPO)),
        "gif_sha256": static.sha256_file(output),
        "preview": str(preview.relative_to(REPO)),
        "preview_sha256": static.sha256_file(preview),
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "renderer_sha256": static.sha256_file(Path(__file__)),
        "source_renderer": str((LEGACY_DIR / "visualize_paper_gif.py").relative_to(REPO)),
        "selection_state_changed": False,
    }
    static.atomic_json(manifest_path, manifest)
    print(f"saved {output}")
    print(f"saved {preview}")
    print(f"saved {manifest_path}")
    print(
        f"frames={len(frames)}, dimensions={frames[0].width}x{frames[0].height}, "
        f"bytes={output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
