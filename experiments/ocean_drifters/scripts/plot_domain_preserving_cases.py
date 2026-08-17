#!/usr/bin/env python3
"""Representative frozen-case support and sensor-footprint diagnostics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import load_phase2_config, resolve, write_csv  # noqa: E402


def main() -> None:
    cfg = load_phase2_config()
    processed = resolve(cfg["processed_dir"])
    analysis = resolve(cfg["analysis_dir"])
    table_path = analysis / "tables/reference_support_lp_domain_preserving.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    largest = max(int(row["particle_count"]) for row in all_rows)
    rows = {int(row["case"]): row for row in all_rows if int(row["particle_count"]) == largest}
    chosen = [0, 8, 15, 16]
    with np.load(
        SCRIPT_DIR.parent / "models/reference_flow_domain_preserving/reference_bank_eval_200000.npz",
        allow_pickle=False,
    ) as data:
        nodes = np.asarray(data["nodes_km"], dtype=np.float64)
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
    inference = X[split == "inference"]
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
    with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"], dtype=np.float64)
    figure_dir = analysis / "figures/reference_support/domain_preserving_paths"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260821 + 901)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    range_rows = []
    colors = plt.cm.tab10(np.arange(4))
    for ax, case in zip(axes.ravel(), chosen, strict=True):
        row = rows[case]
        design_index = int(row["design_index"])
        source_index = int(row["source_time_index"])
        t = int(np.flatnonzero(evaluation_indices == source_index)[0])
        points = nodes[t]
        sample = points[rng.choice(len(points), 6000, replace=False)]
        ax.hexbin(sample[:, 0], sample[:, 1], gridsize=55, bins="log", mincnt=1, cmap="Greys", alpha=.75)
        ax.scatter(
            inference[:, source_index, 0], inference[:, source_index, 1],
            s=13, alpha=.55, color="#e45756", label="inference positions",
        )
        delta = points[:, None] - centers[design_index]
        phi = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2)
        targets = measurements[design_index, source_index]
        for sensor, center in enumerate(centers[design_index]):
            ax.add_patch(plt.Circle(center, sigma, fill=False, lw=1.4, color=colors[sensor]))
            ax.scatter(*center, marker="X", s=65, color=colors[sensor], edgecolor="black")
            range_rows.append({
                "case": case, "design_id": row["design_id"], "day": row["day"],
                "classification": row["classification"], "sensor": sensor + 1,
                "target_moment": targets[sensor],
                "reference_observable_min": phi[:, sensor].min(),
                "reference_observable_max": phi[:, sensor].max(),
                "coordinatewise_feasible": bool(
                    phi[:, sensor].min() <= targets[sensor] <= phi[:, sensor].max()
                ),
            })
        ax.set_title(
            f"case {case}: {row['classification'].split('_', 1)[0]}\n"
            f"day {float(row['day']):g}, LP={float(row['minimum_linf_residual']):.2e}, "
            f"ESS={float(row['native_ess_fraction']):.2e}"
        )
        ax.set_aspect("equal"); ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)")
    fig.suptitle("Frozen diagnostic cases: domain-preserving support and Gaussian footprints")
    fig.tight_layout(); fig.savefig(figure_dir / "representative_sensor_support.png", dpi=190); plt.close(fig)
    write_csv(analysis / "tables/domain_preserving_observable_ranges.csv", range_rows)
    print("Wrote representative domain-preserving sensor-support diagnostics.")


if __name__ == "__main__":
    main()
