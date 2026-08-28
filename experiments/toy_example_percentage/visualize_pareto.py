"""Visualize the toy percentage-risk Pareto sweep."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

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
    save_method_tables,
)
from pareto_cost_risk_visualization import (
    make_cost_risk_figure,
    save_cost_risk_figure,
)
import visualize as experiment_visualization

DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "pareto"


def save_pareto_figure(rows: list[dict], output: Path, *, dpi: int = 220) -> Path:
    return save_figure(rows, output, experiment_label="Toy example", dpi=dpi)


def _sensor_angles(data: dict, method: str) -> np.ndarray:
    value = data.get("selection", {}).get(f"{method}_optimum_deg", [])
    return np.asarray(value, dtype=float)


def save_sensor_atlas(
    rows: list[dict], pareto_source: Path, output: Path, *, dpi: int = 220
) -> Path:
    """Show how each objective moves the two toy sensors across allowances."""
    loaded = load_point_results(rows, pareto_source)
    cfg = loaded[0][2]["config"]
    pop = cfg["population"]
    measurement = cfg["measurement"]
    sensor_radius = float(measurement["sensor_radius"])
    sensor_width = float(measurement["sensor_width"])
    alpha = np.deg2rad(0.5 * (float(pop["alpha_min_deg"]) + float(pop["alpha_max_deg"])))
    limit = max(float(pop["domain_half_width"]), sensor_radius + 2.0 * sensor_width)
    grid = np.linspace(-limit, limit, 230)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    density = experiment_visualization._mixture_density(
        xx, yy, 0.5, alpha, radius=float(pop["radius"]), sigma=float(pop["sigma"])
    )
    experiment_visualization._style()
    fig, axes = plt.subplots(
        len(METHODS), len(loaded), figsize=(2.55 * len(loaded), 7.9), squeeze=False,
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.84, wspace=0.05, hspace=0.12)
    for column, (row, _, data) in enumerate(loaded):
        for method_index, method in enumerate(METHODS):
            ax = axes[method_index, column]
            ax.contourf(xx, yy, density, levels=16, cmap="Greys", alpha=0.22)
            ax.add_patch(plt.Circle((0.0, 0.0), sensor_radius, fill=False, color="#9298A0", lw=0.8, ls="--"))
            angles = _sensor_angles(data, method)
            points = sensor_radius * np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
            color = METHOD_COLORS[method]
            if len(points):
                ax.plot(points[:, 0], points[:, 1], color=color, lw=1.5, alpha=0.9)
            for index, (x, y) in enumerate(points, start=1):
                ax.add_patch(plt.Circle((x, y), sensor_width, fill=False, color=color, lw=0.75, alpha=0.48))
                ax.scatter([x], [y], s=46, color=color, edgecolor="white", linewidth=0.9, zorder=4)
                ax.text(x, y, str(index), color="white", fontsize=6.1, fontweight="bold", ha="center", va="center", zorder=5)
            ax.set_aspect("equal")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if method_index == 0:
                ax.set_title(f"{float(row['risk_allowance_percent']):g}%", pad=5)
            if column == 0:
                ax.set_ylabel(METHOD_LABELS[method], color=color, fontweight="bold", labelpad=8)
            ax.text(
                0.04, 0.04, " / ".join(f"{angle:.1f}°" for angle in angles),
                transform=ax.transAxes, fontsize=6.4, color=color,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.76},
            )
    fig.suptitle(
        "Toy example · Pareto sensor-position atlas", x=0.075, y=0.96,
        ha="left", fontsize=18, fontweight="bold",
    )
    fig.text(
        0.075, 0.91,
        "Columns increase the allowed finite-law risk; rows compare the Law, Tangent, and Full objectives over the midpoint population density.",
        ha="left", fontsize=9.4, color="#5E646D",
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
    """Save the established experiment/sensor dashboard for a representative point."""
    _, data = _representative_result(rows, pareto_source)
    fig = experiment_visualization.make_figure(data)
    screens = data["law_screens"]
    allowance = 100.0 * float(screens["epsilon_r"]) / abs(float(screens["R_star"]))
    fig.text(
        0.97, 0.025, f"representative Pareto point: {allowance:g}% allowance",
        ha="right", fontsize=7.8, color="#6B7078",
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def make_action_cost_figure(records: list[dict]) -> plt.Figure:
    return make_cost_risk_figure(records)


def save_action_cost_figure(
    records: list[dict], output: Path, *, dpi: int = 300
) -> list[Path]:
    return save_cost_risk_figure(records, output, dpi=dpi)


def save_pareto_suite(
    rows: list[dict], pareto_source: Path, output_dir: Path, *, dpi: int = 220
) -> list[Path]:
    """Write the frontier, method tables, comparisons, and geometry figures."""
    output_dir = Path(output_dir).expanduser().resolve()
    records = method_records(rows, pareto_source)
    outputs = [save_pareto_figure(rows, output_dir / "pareto.png", dpi=dpi)]
    outputs.extend(save_method_tables(records, output_dir))
    outputs.extend(
        save_action_cost_figure(records, output_dir / "pareto_methods.png", dpi=dpi)
    )
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
        fig = make_figure(rows, experiment_label="Toy example")
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
