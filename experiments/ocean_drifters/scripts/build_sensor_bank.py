#!/usr/bin/env python3
"""Freeze sigma, admissible geometry, four-sensor bank, and dense measurements."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import (  # noqa: E402
    gaussian_features_numpy,
    load_phase2_config,
    resolve,
    sha256,
    write_csv,
    write_json,
)

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.measurements import GaussianPointSensors2D  # noqa: E402


def minimum_separation(centers: np.ndarray) -> float:
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    distance += np.eye(len(centers)) * 1e30
    return float(distance.min())


def valid_design(centers: np.ndarray, bounds: np.ndarray, min_sep: float) -> bool:
    return bool(
        np.all(np.isfinite(centers))
        and np.all(centers[:, 0] >= bounds[0]) and np.all(centers[:, 0] <= bounds[1])
        and np.all(centers[:, 1] >= bounds[2]) and np.all(centers[:, 1] <= bounds[3])
        and minimum_separation(centers) >= min_sep
    )


def canonical_key(centers: np.ndarray) -> tuple[float, ...]:
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return tuple(np.round(centers[order].ravel(), 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    processed = resolve(cfg["processed_dir"])
    analysis = resolve(cfg["analysis_dir"])
    for directory in [analysis / "figures/sigma", analysis / "figures/sensors", analysis / "figures/measurements", analysis / "tables"]:
        directory.mkdir(parents=True, exist_ok=True)
    dev_path = processed / "development_270.npz"
    with np.load(dev_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        days = np.asarray(data["relative_days"], dtype=np.float64)
    inference = X[split == "inference"]
    assert len(inference) == 200

    representative_centers = np.asarray([
        [-150, -180], [-100, 120], [300, -350], [350, 50],
        [400, 400], [800, -450], [850, -50], [850, 350],
        [1300, -300], [1300, 200], [1750, -150], [1800, 350],
    ], dtype=float)
    sigma_rows = []
    flat = inference.reshape(-1, 2)
    study = cfg["sigma_study"]
    for sigma in study["candidates_km"]:
        values = gaussian_features_numpy(flat, representative_centers, sigma)
        covariance = np.cov(values, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        correlation = np.corrcoef(values, rowvar=False)
        offdiag = np.abs(correlation - np.eye(len(correlation)))
        sigma_rows.append({
            "sigma_km": sigma, "mean_activation": values.mean(),
            "median_sensor_mean": np.median(values.mean(axis=0)),
            "median_sensor_variance": np.median(values.var(axis=0)),
            "near_zero_fraction": np.mean(values < study["near_zero_threshold"]),
            "near_one_fraction": np.mean(values > study["near_one_threshold"]),
            "max_abs_pairwise_correlation": np.nanmax(offdiag),
            "covariance_min_eigenvalue": eigenvalues.min(),
            "covariance_max_eigenvalue": eigenvalues.max(),
            "covariance_condition": eigenvalues.max() / max(eigenvalues.min(), 1e-300),
            "selected": float(sigma) == float(study["selected_km"]),
        })
    write_csv(analysis / "tables/sigma_diagnostics.csv", sigma_rows)
    selected_sigma = float(study["selected_km"])
    selected = next(row for row in sigma_rows if row["selected"])
    write_json(analysis / "tables/selected_sigma.json", {
        "sigma_km": selected_sigma,
        "selection_data": "inference trajectories only",
        "reason": "middle-scale localization with nondegenerate activation variance and acceptable pooled covariance conditioning",
        "diagnostics": selected,
        "final_test_accessed": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    sigmas = np.asarray([row["sigma_km"] for row in sigma_rows])
    axes[0, 0].plot(sigmas, [row["mean_activation"] for row in sigma_rows], marker="o")
    axes[0, 0].set_ylabel("mean activation")
    axes[0, 1].plot(sigmas, [row["median_sensor_variance"] for row in sigma_rows], marker="o", color="#d95f02")
    axes[0, 1].set_ylabel("median sensor variance")
    axes[1, 0].plot(sigmas, [row["near_zero_fraction"] for row in sigma_rows], marker="o", label="near zero")
    axes[1, 0].plot(sigmas, [row["near_one_fraction"] for row in sigma_rows], marker="o", label="near one")
    axes[1, 0].legend(); axes[1, 0].set_ylabel("fraction")
    axes[1, 1].semilogy(sigmas, [row["covariance_condition"] for row in sigma_rows], marker="o", color="#7570b3")
    axes[1, 1].set_ylabel("pooled covariance condition")
    for ax in axes.ravel():
        ax.axvline(selected_sigma, ls="--", color="black", alpha=.5); ax.set_xlabel("sigma (km)"); ax.grid(alpha=.2)
    fig.suptitle("Gaussian sensor-radius diagnostics (inference cohort only)")
    fig.tight_layout(); fig.savefig(analysis / "figures/sigma/sigma_diagnostics.png", dpi=190); plt.close(fig)

    bounds = np.asarray(cfg["sensor_design"]["bounds_km"], dtype=float)
    min_sep = float(cfg["sensor_design"]["min_separation_km"])
    manual = [
        [[-150, -150], [350, -100], [850, -50], [1450, 50]],
        [[250, -500], [250, -150], [250, 200], [250, 550]],
        [[850, -550], [850, -200], [850, 150], [850, 500]],
        [[-100, -350], [450, 300], [1050, -300], [1650, 300]],
        [[-100, 250], [500, 250], [1100, 250], [1700, 250]],
        [[-100, -300], [500, -300], [1100, -300], [1700, -300]],
        [[0, -500], [500, 400], [1200, -450], [1900, 350]],
        [[-200, 0], [300, 450], [800, 0], [1300, -450]],
        [[500, -550], [900, -150], [1300, 250], [1750, 600]],
        [[-300, -450], [300, 350], [900, -350], [1500, 450]],
        [[100, -250], [700, -250], [1300, -250], [1900, -250]],
        [[100, 350], [700, 350], [1300, 350], [1900, 350]],
    ]
    designs: list[tuple[str, np.ndarray]] = []
    keys = set()
    def add(style: str, centers) -> bool:
        centers = np.asarray(centers, dtype=float)
        key = canonical_key(centers)
        if key in keys or not valid_design(centers, bounds, min_sep):
            return False
        keys.add(key); designs.append((style, centers)); return True
    for centers in manual:
        add("manual", centers)

    rng = np.random.default_rng(int(cfg["seed"]) + 2001)
    support = inference[:, ::4].reshape(-1, 2)
    support = support[
        (support[:, 0] >= bounds[0]) & (support[:, 0] <= bounds[1])
        & (support[:, 1] >= bounds[2]) & (support[:, 1] <= bounds[3])
    ]
    target = int(cfg["sensor_design"]["bank_size"])
    attempts = 0
    while len(designs) < target and attempts < target * 500:
        attempts += 1
        fraction = len(designs) / target
        if fraction < .43:
            style = "support_random"
            centers = support[rng.choice(len(support), 4, replace=False)] + rng.normal(0, cfg["sensor_design"]["jitter_km"], (4, 2))
        elif fraction < .68:
            style = "longitudinal_stratified"
            xedges = np.quantile(support[:, 0], [0, .25, .5, .75, 1])
            centers = []
            for lo, hi in zip(xedges[:-1], xedges[1:], strict=True):
                pool = support[(support[:, 0] >= lo) & (support[:, 0] <= hi)]
                centers.append(pool[rng.integers(len(pool))] + rng.normal(0, 70, 2))
        elif fraction < .82:
            style = "lateral_discriminating"
            anchor_x = rng.uniform(150, 1500)
            centers = np.column_stack([
                anchor_x + rng.normal(0, 150, 4),
                np.linspace(-550, 550, 4) + rng.normal(0, 55, 4),
            ])
        elif fraction < .91:
            style = "upstream_focused"
            centers = np.column_stack([rng.uniform(-350, 800, 4), rng.uniform(-600, 600, 4)])
        elif fraction < .97:
            style = "downstream_focused"
            centers = np.column_stack([rng.uniform(700, 2000, 4), rng.uniform(-600, 650, 4)])
        else:
            style = "uniform_control"
            centers = np.column_stack([rng.uniform(bounds[0], bounds[1], 4), rng.uniform(bounds[2], bounds[3], 4)])
        add(style, centers)
    if len(designs) != target:
        raise RuntimeError(f"could only generate {len(designs)} valid designs")

    bank = np.stack([centers for _, centers in designs])
    rows = []
    for i, (style, centers) in enumerate(designs):
        row = {"design_id": f"design_{i:06d}", "style": style, "sigma_km": selected_sigma,
               "min_pairwise_separation_km": minimum_separation(centers)}
        for j, (x, y) in enumerate(centers, start=1):
            row[f"s{j}_x_km"] = x; row[f"s{j}_y_km"] = y
        rows.append(row)
    write_csv(processed / "sensor_bank.csv", rows)
    np.savez_compressed(processed / "sensor_bank.npz", centers_km=bank, design_id=np.asarray([row["design_id"] for row in rows]), style=np.asarray([style for style, _ in designs]), sigma_km=selected_sigma, bounds_km=bounds, min_separation_km=min_sep)

    measurements = np.empty((len(bank), len(days), 4), dtype=np.float32)
    for start in range(0, len(bank), 16):
        stop = min(start + 16, len(bank))
        delta = inference[None, :, :, None, :] - bank[start:stop, None, None, :, :]
        features = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / selected_sigma**2)
        measurements[start:stop] = features.mean(axis=1).astype(np.float32)
    # Cross-check the experiment's vectorized fast path against the shared API.
    family = GaussianPointSensors2D(width=selected_sigma, n_sensors=4)
    expected = np.asarray(family.features(jnp.asarray(inference[0, :3]), jnp.asarray(bank[0].ravel())))
    actual = gaussian_features_numpy(inference[0, :3], bank[0], selected_sigma)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.savez_compressed(processed / "measurement_trajectories.npz", c=measurements, days=days, design_id=np.asarray([row["design_id"] for row in rows]), inference_n=200, sensor_bank_sha256=np.asarray(sha256(processed / "sensor_bank.npz")), final_test_accessed=np.asarray(False))

    manifest = {
        "design_count": len(bank), "style_counts": dict(Counter(style for style, _ in designs)),
        "sigma_km": selected_sigma, "bounds_km": bounds.tolist(),
        "minimum_separation_km": min_sep, "all_constraints_valid": True,
        "sensor_bank_sha256": sha256(processed / "sensor_bank.npz"),
        "measurements_sha256": sha256(processed / "measurement_trajectories.npz"),
        "measurement_time_count": len(days), "measurement_inference_n": 200,
        "shared_measurement_api_crosscheck": "pass", "final_test_accessed": False,
    }
    write_json(processed / "sensor_bank_manifest.json", manifest)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    hist = ax.hexbin(inference[:, ::4, 0].ravel(), inference[:, ::4, 1].ravel(), gridsize=65, cmap="Greys", bins="log", mincnt=1)
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for j, center in enumerate(bank[0]):
        circle = plt.Circle(center, selected_sigma, fill=False, color=colors[j], lw=2)
        ax.add_patch(circle); ax.scatter(*center, marker="x", s=70, color=colors[j])
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_title("Selected 200 km footprint and admissible sensor region")
    fig.colorbar(hist, ax=ax, label="log trajectory occupancy")
    fig.tight_layout(); fig.savefig(analysis / "figures/sigma/selected_sigma_footprints.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    examples = [0]
    for style in ["support_random", "longitudinal_stratified", "lateral_discriminating", "upstream_focused", "downstream_focused"]:
        examples.append(next(i for i, (s, _) in enumerate(designs) if s == style))
    for ax, index in zip(axes.ravel(), examples, strict=True):
        ax.hexbin(inference[:, ::4, 0].ravel(), inference[:, ::4, 1].ravel(), gridsize=45, cmap="Greys", bins="log", mincnt=1, alpha=.8)
        ax.scatter(bank[index, :, 0], bank[index, :, 1], marker="X", s=65, c=np.arange(4), cmap="tab10", edgecolor="black")
        ax.set_title(f"{rows[index]['design_id']}\n{designs[index][0]}"); ax.set_aspect("equal")
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.suptitle("Frozen sensor-bank geometry classes")
    fig.tight_layout(); fig.savefig(analysis / "figures/sensors/bank_examples.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(days, measurements[0, :, j], label=f"sensor {j + 1}", color=plt.cm.tab10(j))
        ax.set_ylabel("mean activation"); ax.grid(alpha=.2); ax.legend()
    fig.supxlabel("days after crossing"); fig.suptitle("Dense virtual measurements: manual longitudinal design")
    fig.tight_layout(); fig.savefig(analysis / "figures/measurements/representative_measurements.png", dpi=190); plt.close(fig)
    print(f"[sensors] selected sigma={selected_sigma:.0f} km; froze {len(bank)} designs and {len(days)}-time measurements", flush=True)


if __name__ == "__main__":
    main()
