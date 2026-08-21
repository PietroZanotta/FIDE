"""Visualize the vortices percentage-risk Pareto sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from percentage_pareto_visualization import (
    METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    load_point_results,
    load_rows,
    make_figure,
    method_records,
    save_figure,
    save_method_figure,
    save_method_tables,
)
import visualize as experiment_visualization

DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "pareto"


def save_pareto_figure(rows: list[dict], output: Path, *, dpi: int = 220) -> Path:
    return save_figure(rows, output, experiment_label="Vortices / double gyre", dpi=dpi)


def _mean_truth_density(result_path: Path, data: dict) -> np.ndarray:
    truth_name = data.get("truth", {}).get("truth_bank", "truth_bank.npz")
    with np.load(result_path.with_name(truth_name)) as bank:
        particles = np.asarray(bank["particles"], dtype=np.float64)
    indices = np.unique(np.linspace(0, len(particles) - 1, min(len(particles), 13), dtype=int))
    return np.mean(
        np.stack([experiment_visualization._density(particles[index]) for index in indices]),
        axis=0,
    )


def save_sensor_atlas(
    rows: list[dict], pareto_source: Path, output: Path, *, dpi: int = 220
) -> Path:
    """Show the actual four-sensor geometries at every Pareto allowance."""
    loaded = load_point_results(rows, pareto_source)
    mean_density = _mean_truth_density(loaded[0][1], loaded[0][2])
    vmax = max(float(np.quantile(mean_density, 0.998)), 1.0)
    measurement = loaded[0][2]["config"]["measurement"]
    width = float(measurement.get("sensor_width", 0.12))
    margin = float(measurement.get("boundary_margin", 2.0 * width))
    experiment_visualization._style()
    fig, axes = plt.subplots(
        len(METHODS), len(loaded), figsize=(2.65 * len(loaded), 7.0), squeeze=False,
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.83, wspace=0.05, hspace=0.13)
    for column, (row, _, data) in enumerate(loaded):
        for method_index, method in enumerate(METHODS):
            ax = axes[method_index, column]
            ax.imshow(
                mean_density, origin="lower", extent=(0.0, 2.0, 0.0, 1.0),
                cmap=experiment_visualization.OCCUPANCY_CMAP,
                norm=PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax),
                interpolation="bilinear", aspect="equal",
            )
            ax.add_patch(
                Rectangle(
                    (margin, margin), 2.0 - 2.0 * margin, 1.0 - 2.0 * margin,
                    fill=False, edgecolor=(1.0, 1.0, 1.0, 0.56), linewidth=0.7,
                    linestyle=(0, (3, 2)),
                )
            )
            centers = np.asarray(data.get("selection_centers", {}).get(method, []), dtype=float)
            color = METHOD_COLORS[method]
            for index, (x, y) in enumerate(centers, start=1):
                ax.add_patch(Circle((x, y), 2.0 * width, color=color, alpha=0.08, ec="none"))
                ax.add_patch(Circle((x, y), width, fill=False, color=color, lw=1.0, alpha=0.95))
                ax.scatter([x], [y], s=37, color=color, edgecolor="white", linewidth=0.8, zorder=5)
                ax.text(x, y, str(index), color="white", fontsize=5.7, fontweight="bold", ha="center", va="center", zorder=6)
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if method_index == 0:
                ax.set_title(f"{float(row['risk_allowance_percent']):g}%", pad=5)
            if column == 0:
                ax.set_ylabel(METHOD_LABELS[method], color=color, fontweight="bold", labelpad=8)
    fig.suptitle(
        "Vortices · Pareto sensor-position atlas", x=0.075, y=0.96,
        ha="left", fontsize=18, fontweight="bold",
    )
    fig.text(
        0.075, 0.905,
        "Columns increase the allowed finite-law risk; rows compare Law, Tangent, and Full sensor centers over time-averaged particle occupancy.",
        ha="left", fontsize=9.4, color="#5E646D",
    )
    fig.text(
        0.985, 0.905, "rings: 1σ sensing width  ·  dashed box: admissible centers",
        ha="right", fontsize=7.7, color="#686E77",
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _representative_result(rows: list[dict], pareto_source: Path) -> tuple[Path, dict]:
    loaded = load_point_results(rows, pareto_source)
    return min(loaded, key=lambda item: abs(float(item[0]["risk_allowance_percent"]) - 3.0))[1:]


def save_experiment_figure(
    rows: list[dict], pareto_source: Path, output: Path, *, dpi: int = 220
) -> Path:
    result_path, _ = _representative_result(rows, pareto_source)
    data, times, particles, candidates, validation = experiment_visualization._load_inputs(result_path)
    fig = experiment_visualization.make_figure(data, times, particles, candidates, validation)
    screens = data["law_screens"]
    allowance = 100.0 * float(screens["epsilon_r"]) / abs(float(screens["R_star"]))
    fig.text(
        0.975, 0.025, f"representative Pareto point: {allowance:g}% allowance",
        ha="right", fontsize=7.8, color="#6B7078",
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def save_pareto_suite(
    rows: list[dict], pareto_source: Path, output_dir: Path, *, dpi: int = 220
) -> list[Path]:
    """Write the frontier, method tables, comparisons, and geometry figures."""
    output_dir = Path(output_dir).expanduser().resolve()
    records = method_records(rows, pareto_source)
    outputs = [save_pareto_figure(rows, output_dir / "pareto.png", dpi=dpi)]
    outputs.extend(save_method_tables(records, output_dir))
    outputs.append(save_method_figure(records, output_dir / "pareto_methods.png", experiment_label="Vortices / double gyre", dpi=dpi))
    outputs.append(save_sensor_atlas(rows, pareto_source, output_dir / "pareto_sensor_layouts.png", dpi=dpi))
    outputs.append(save_experiment_figure(rows, pareto_source, output_dir / "experiment_sensors.png", dpi=dpi))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows, resolved = load_rows(args.input)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Could not read Pareto data {args.input}: {exc}") from exc
    output = (args.output or resolved.with_name("pareto.png")).expanduser().resolve()
    if args.show:
        fig = make_figure(rows, experiment_label="Vortices / double gyre")
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
        print(f"saved={output}", flush=True)
        plt.show()
        plt.close(fig)
    else:
        outputs = save_pareto_suite(rows, resolved, output.parent, dpi=args.dpi)
        if output.name != "pareto.png":
            save_pareto_figure(rows, output, dpi=args.dpi)
            outputs[0] = output
        for path in outputs:
            print(f"saved={path}", flush=True)


if __name__ == "__main__":
    main()
