from __future__ import annotations

"""Read-only endpoint-distribution audit for the frozen V6 design references."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.stats import wasserstein_distance


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "outputs" / "prospective_v6_beta_ablation_positive_raster_v1"
DEFAULT_OUTPUT = DEFAULT_RUN / "diagnostics" / "reference_flow_quality_v1"
BOUNDS = ((0.0, 2.0), (0.0, 1.0))
HISTOGRAM_BINS = (256, 128)
MMD_BANDWIDTHS = (0.02, 0.04, 0.08, 0.16)
SLICED_DIRECTIONS = 256
METRIC_SEED = 2026082801


def _histogram(points: np.ndarray) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        points[:, 1],
        points[:, 0],
        bins=(HISTOGRAM_BINS[1], HISTOGRAM_BINS[0]),
        range=(BOUNDS[1], BOUNDS[0]),
    )
    total = float(np.sum(histogram))
    if total <= 0.0:
        raise ValueError("endpoint sample has no mass inside the physical box")
    return histogram / total


def _mmd_kernel() -> np.ndarray:
    nx, ny = HISTOGRAM_BINS
    dx = (BOUNDS[0][1] - BOUNDS[0][0]) / nx
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / ny
    ox = np.arange(-(nx - 1), nx, dtype=np.float64) * dx
    oy = np.arange(-(ny - 1), ny, dtype=np.float64) * dy
    xx, yy = np.meshgrid(ox, oy, indexing="xy")
    kernels = [
        np.exp(-(xx * xx + yy * yy) / (2.0 * bandwidth * bandwidth))
        for bandwidth in MMD_BANDWIDTHS
    ]
    return np.mean(kernels, axis=0)


def _sliced_wasserstein(
    target: np.ndarray,
    learned: np.ndarray,
    directions: np.ndarray,
) -> float:
    values = [
        wasserstein_distance(target @ direction, learned @ direction)
        for direction in directions
    ]
    return float(np.mean(values))


def _distribution_metrics(
    target: np.ndarray,
    learned: np.ndarray,
    directions: np.ndarray,
    kernel: np.ndarray,
) -> dict[str, float]:
    target_hist = _histogram(target)
    learned_hist = _histogram(learned)
    target_kernel = fftconvolve(target_hist, kernel, mode="same")
    learned_kernel = fftconvolve(learned_hist, kernel, mode="same")
    mmd2 = (
        np.sum(target_hist * target_kernel)
        + np.sum(learned_hist * learned_kernel)
        - 2.0 * np.sum(target_hist * learned_kernel)
    )
    target_mean = np.mean(target, axis=0)
    learned_mean = np.mean(learned, axis=0)
    target_cov = np.cov(target, rowvar=False)
    learned_cov = np.cov(learned, rowvar=False)
    outside = (
        (learned[:, 0] < BOUNDS[0][0])
        | (learned[:, 0] > BOUNDS[0][1])
        | (learned[:, 1] < BOUNDS[1][0])
        | (learned[:, 1] > BOUNDS[1][1])
    )
    return {
        "sliced_wasserstein_1": _sliced_wasserstein(target, learned, directions),
        "multiscale_gaussian_mmd": float(np.sqrt(max(float(mmd2), 0.0))),
        "histogram_total_variation": float(0.5 * np.sum(np.abs(target_hist - learned_hist))),
        "histogram_hellinger": float(
            np.sqrt(0.5 * np.sum((np.sqrt(target_hist) - np.sqrt(learned_hist)) ** 2))
        ),
        "mean_error_l2": float(np.linalg.norm(learned_mean - target_mean)),
        "covariance_error_frobenius": float(np.linalg.norm(learned_cov - target_cov)),
        "covariance_error_relative": float(
            np.linalg.norm(learned_cov - target_cov)
            / max(np.linalg.norm(target_cov), np.finfo(np.float64).tiny)
        ),
        "outside_box_fraction": float(np.mean(outside)),
    }


def _training_summary(checkpoint_path: Path) -> dict[str, float | int]:
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(np.asarray(checkpoint["metadata_json"]).item()))
    history = list(metadata.get("training_history", []))
    losses = np.asarray([row["conditional_fm_loss"] for row in history], dtype=np.float64)
    return {
        "logged_points": int(len(history)),
        "initial_logged_loss": float(losses[0]),
        "final_logged_loss": float(losses[-1]),
        "best_logged_loss": float(np.min(losses)),
        "best_logged_step": int(history[int(np.argmin(losses))]["step"]),
    }


def _density(points: np.ndarray) -> np.ndarray:
    return gaussian_filter(_histogram(points), sigma=1.35, mode="nearest")


def _highest_density_levels(density: np.ndarray) -> list[float]:
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    levels = []
    for mass in (0.90, 0.70, 0.50):
        levels.append(float(flat[min(int(np.searchsorted(cumulative, mass)), len(flat) - 1)]))
    return sorted(set(levels))


def _plot(
    target_x0: np.ndarray,
    target_x1: np.ndarray,
    learned: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: list[dict[str, Any]],
    output: Path,
) -> None:
    target_density = [_density(target_x0), _density(target_x1)]
    learned_density = {
        key: (_density(value[0]), _density(value[1])) for key, value in learned.items()
    }
    endpoint_vmax = []
    for endpoint in range(2):
        values = [target_density[endpoint]] + [value[endpoint] for value in learned_density.values()]
        endpoint_vmax.append(max(float(np.max(np.log1p(1.0e6 * item))) for item in values))

    rows = len(learned)
    fig, axes = plt.subplots(rows, 4, figsize=(14.0, 3.35 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]
    extent = (BOUNDS[0][0], BOUNDS[0][1], BOUNDS[1][0], BOUNDS[1][1])
    metric_by_key = {(row["reference_id"], row["endpoint"]): row for row in metrics}
    images = [None, None]
    for row_index, (reference_id, densities) in enumerate(learned_density.items()):
        for endpoint in range(2):
            target_ax = axes[row_index, 2 * endpoint]
            learned_ax = axes[row_index, 2 * endpoint + 1]
            for ax, density, title in (
                (target_ax, target_density[endpoint], f"Target $x_{endpoint}$"),
                (learned_ax, densities[endpoint], f"{reference_id} rollout at $t={endpoint}$"),
            ):
                image = ax.imshow(
                    np.log1p(1.0e6 * density),
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=0.0,
                    vmax=endpoint_vmax[endpoint],
                    interpolation="nearest",
                    aspect="equal",
                )
                images[endpoint] = image
                ax.set_xlim(*BOUNDS[0])
                ax.set_ylim(*BOUNDS[1])
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            levels = _highest_density_levels(target_density[endpoint])
            if levels:
                learned_ax.contour(
                    target_density[endpoint],
                    levels=levels,
                    colors="cyan",
                    linewidths=0.8,
                    origin="lower",
                    extent=extent,
                )
            row_metric = metric_by_key[(reference_id, f"x{endpoint}")]
            learned_ax.text(
                0.02,
                0.02,
                f"SW1={row_metric['sliced_wasserstein_1']:.4g}\n"
                f"MMD={row_metric['multiscale_gaussian_mmd']:.4g}\n"
                f"TV={row_metric['histogram_total_variation']:.4g}",
                transform=learned_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 2.5},
            )
        axes[row_index, 0].text(
            -0.22,
            0.5,
            reference_id,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    for endpoint in range(2):
        fig.colorbar(
            images[endpoint],
            ax=axes[:, 2 * endpoint : 2 * endpoint + 2],
            shrink=0.82,
            label="log(1 + scaled probability mass)",
        )
    fig.suptitle(
        "V6 design-reference endpoint distributions\n"
        "Cyan contours on rollout panels enclose target high-density regions",
        fontsize=13,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def run(run_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_path = run_root / "shared" / "endpoint_reference" / "endpoint_data.npz"
    manifest_path = run_root / "shared" / "results" / "design_reference_manifest.json"
    with np.load(endpoint_path, allow_pickle=False) as endpoint:
        target_x0 = np.asarray(endpoint["x0"], dtype=np.float64)
        target_x1 = np.asarray(endpoint["x1"], dtype=np.float64)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(METRIC_SEED)
    angles = rng.uniform(0.0, np.pi, size=SLICED_DIRECTIONS)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    kernel = _mmd_kernel()
    with np.load(Path(manifest["references"][0]["rollout"]), allow_pickle=False) as first_rollout:
        rollout_samples = int(first_rollout["nodes"].shape[1])
    baseline_indices = rng.choice(len(target_x1), size=rollout_samples, replace=False)
    baseline_x1 = target_x1[baseline_indices]
    rows: list[dict[str, Any]] = []
    learned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    training: dict[str, dict[str, float | int]] = {}
    for reference in manifest["references"]:
        reference_id = str(reference["reference_id"])
        rollout_path = Path(reference["rollout"])
        checkpoint_path = Path(reference["checkpoint"])
        with np.load(rollout_path, allow_pickle=False) as rollout:
            nodes = np.asarray(rollout["nodes"], dtype=np.float64)
        learned[reference_id] = (nodes[0], nodes[-1])
        training[reference_id] = _training_summary(checkpoint_path)
        for endpoint_name, target, sample in (
            ("x0", target_x0, nodes[0]),
            ("x1", target_x1, nodes[-1]),
        ):
            rows.append(
                {
                    "reference_id": reference_id,
                    "endpoint": endpoint_name,
                    "target_samples": int(len(target)),
                    "rollout_samples": int(len(sample)),
                    **_distribution_metrics(target, sample, directions, kernel),
                }
            )
    baseline = {
        "endpoint": "x1",
        "description": "same-size target subsample versus the full x1 target",
        **_distribution_metrics(target_x1, baseline_x1, directions, kernel),
    }
    result = {
        "schema_version": 1,
        "role": "read_only_v6_design_reference_endpoint_quality_audit",
        "run_root": str(run_root.resolve()),
        "reference_ids": list(learned),
        "endpoint_target_samples": {"x0": int(len(target_x0)), "x1": int(len(target_x1))},
        "rollout_samples": int(next(iter(learned.values()))[0].shape[0]),
        "metric_specification": {
            "seed": METRIC_SEED,
            "sliced_wasserstein_directions": SLICED_DIRECTIONS,
            "histogram_bins": list(HISTOGRAM_BINS),
            "multiscale_mmd_bandwidths": list(MMD_BANDWIDTHS),
            "physical_bounds": [list(item) for item in BOUNDS],
        },
        "sampling_floor": baseline,
        "training_history": training,
        "endpoint_metrics": rows,
        "interpretation_scope": (
            "Endpoint data used for these metrics are the endpoint-only training distributions, "
            "not a held-out endpoint set. Intermediate truth states remain unused."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(target_x0, target_x1, learned, rows, output_dir / "endpoint_distributions.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.run_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
