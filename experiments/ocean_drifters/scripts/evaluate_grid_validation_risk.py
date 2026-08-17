#!/usr/bin/env python3
"""Held-out validation-law-risk ranking for continuous-grid I-projections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from phase2_common import load_phase2_config, resolve, rff_map, sha256, write_csv, write_json  # noqa: E402
from run_continuous_grid_iprojection import grid_points, load_config as load_grid_config  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None) -> dict[str, Any]:
    source = Path(path) if path else SCRIPT_DIR.parent / "configs/grid_validation_law_risk.json"
    with source.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_distance(a: np.ndarray, b: np.ndarray) -> float:
    cost = cdist(a, b)
    row, column = linear_sum_assignment(cost)
    return float(cost[row, column].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["phase2_config"]))
    grid_cfg = load_grid_config(resolve(cfg["continuous_grid_config"]))
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    table_dir = analysis / "tables"
    figure_dir = analysis / "figures/grid_law_risk"
    figure_dir.mkdir(parents=True, exist_ok=True)

    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        positions = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
    inference = positions[split == "inference"]
    validation = positions[split == "validation"]
    if len(inference) != 200 or len(validation) != 70:
        raise RuntimeError("the frozen 200/70 development split changed")
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        design_ids = np.asarray(data["design_id"]).astype(str)
        styles = np.asarray(data["style"]).astype(str)
        sigma = float(data["sigma_km"])
    with np.load(resolve(cfg["frozen_rff_source"]), allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        evaluation_days = np.asarray(data["evaluation_days"], dtype=np.float64)
        omega = np.asarray(data["rff_omega"], dtype=np.float64)
        phase = np.asarray(data["rff_phase"], dtype=np.float64)
        bandwidth = float(data["bandwidth_km"])
        old_risks = np.asarray(data["risks"], dtype=np.float64)
        old_feasible = np.asarray(data["feasible"], dtype=bool)
        old_best = int(data["best_design_index"])
    if omega.shape[1] != int(cfg["rff_features"]):
        raise RuntimeError("frozen RFF feature count differs from the predeclared grid-risk config")

    projection_rows = read_csv(resolve(cfg["full_projection_table"]))
    design_rows = read_csv(resolve(cfg["full_design_table"]))
    expected = len(centers) * len(evaluation_indices)
    if len(projection_rows) != expected:
        raise RuntimeError(f"expected {expected} full-grid projections, found {len(projection_rows)}")
    source_to_position = {int(value): i for i, value in enumerate(evaluation_indices)}
    lambdas = np.full((len(centers), len(evaluation_indices), 4), np.nan, dtype=np.float64)
    usable = np.zeros((len(centers), len(evaluation_indices)), dtype=bool)
    kl = np.full(usable.shape, np.nan, dtype=np.float64)
    log10_ess = np.full(usable.shape, np.nan, dtype=np.float64)
    condition = np.full(usable.shape, np.nan, dtype=np.float64)
    residual = np.full(usable.shape, np.nan, dtype=np.float64)
    iterations = np.zeros(usable.shape, dtype=np.int32)
    for row in projection_rows:
        design = int(row["design_index"])
        time_index = source_to_position[int(row["source_time_index"])]
        lambdas[design, time_index] = [float(row[f"lambda_{j}"]) for j in range(4)]
        usable[design, time_index] = row["usable"] == "True"
        kl[design, time_index] = float(row["kl_divergence"])
        log10_ess[design, time_index] = float(row["log10_intrinsic_ess"])
        condition[design, time_index] = float(row["covariance_condition_regularized"])
        residual[design, time_index] = float(row["verified_l2_residual"])
        iterations[design, time_index] = int(row["iterations"])
    eligible = usable.all(axis=1)
    declared_eligible = {
        int(row["design_index"]): row["fully_usable"] == "True" for row in design_rows
    }
    if len(declared_eligible) != len(centers) or any(
        declared_eligible[i] != bool(eligible[i]) for i in range(len(centers))
    ):
        raise RuntimeError("full-bank per-design eligibility is inconsistent with projection rows")
    eligible_indices = np.flatnonzero(eligible)
    if len(eligible_indices) != 485:
        raise RuntimeError(f"expected 485 fully usable layouts, found {len(eligible_indices)}")

    nx, ny = (int(value) for value in cfg["grid_resolution"])
    expected_grid = tuple(int(value) for value in grid_cfg["full_bank"]["resolution"])
    if (nx, ny) != expected_grid:
        raise RuntimeError("risk grid differs from the accepted full-sweep grid")
    bounds = np.asarray(phase2["domain"]["final_box_km"], dtype=np.float64)
    points, _, _ = grid_points(bounds, nx, ny)
    grid_rff = rff_map(points, omega, phase, dtype=np.float32)
    grid_rff_device = jax.device_put(jnp.asarray(grid_rff))
    validation_rff_by_id = rff_map(
        validation[:, evaluation_indices], omega, phase, dtype=np.float32
    )
    validation_embedding = validation_rff_by_id.mean(axis=0, dtype=np.float64)

    checkpoint_hash = sha256(resolve(grid_cfg["reference_checkpoint"]))
    cache_dir = SCRIPT_DIR.parent / "cache/reference_density_grids"
    projected_embedding = np.full(
        (len(centers), len(evaluation_indices), omega.shape[1]), np.nan, dtype=np.float32
    )
    base_embedding = np.empty((len(evaluation_indices), omega.shape[1]), dtype=np.float64)
    batch_size = int(cfg["projection_embedding_batch_size"])
    started = time.perf_counter()
    for time_index, source_index in enumerate(evaluation_indices):
        cache = cache_dir / (
            f"density_{checkpoint_hash[:12]}_t{int(source_index):03d}_{nx}x{ny}.npz"
        )
        with np.load(cache, allow_pickle=False) as data:
            log_base = np.asarray(data["log_base_mass"], dtype=np.float64)
        base_weights = np.exp(log_base)
        base_embedding[time_index] = base_weights @ grid_rff.astype(np.float64)
        for start in range(0, len(eligible_indices), batch_size):
            selected = eligible_indices[start:start + batch_size]
            delta = points[None, :, None, :] - centers[selected, None, :, :]
            phi = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2)
            logits = log_base[None] + np.einsum(
                "bnm,bm->bn", phi, lambdas[selected, time_index]
            )
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= weights.sum(axis=1, keepdims=True)
            embedding = jnp.asarray(weights, dtype=jnp.float32) @ grid_rff_device
            projected_embedding[selected, time_index] = np.asarray(embedding)
        print(
            f"[grid law risk] day {evaluation_days[time_index]:g}: "
            f"485 embeddings complete; elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    difference = (
        projected_embedding[eligible_indices].astype(np.float64)
        - validation_embedding[None]
    )
    risk_by_time_eligible = np.sum(difference * difference, axis=-1)
    risks = np.full(len(centers), np.nan, dtype=np.float64)
    risks[eligible_indices] = risk_by_time_eligible.mean(axis=1)
    risk_by_time = np.full((len(centers), len(evaluation_indices)), np.nan, dtype=np.float64)
    risk_by_time[eligible_indices] = risk_by_time_eligible
    base_difference = base_embedding - validation_embedding
    base_risk_by_time = np.sum(base_difference * base_difference, axis=-1)

    bootstrap_rng = np.random.default_rng(
        int(cfg["seed"]) + int(phase2["law_risk"]["bootstrap_seed_offset"])
    )
    bootstrap_indices = bootstrap_rng.integers(
        0, len(validation), size=(int(cfg["bootstrap_replicates"]), len(validation))
    )
    bootstrap_embedding = validation_rff_by_id[bootstrap_indices].mean(axis=1, dtype=np.float64)
    projected_sq = np.sum(
        projected_embedding[eligible_indices].astype(np.float64) ** 2, axis=-1
    )
    bootstrap_sq = np.sum(bootstrap_embedding**2, axis=-1)
    cross = np.einsum(
        "dtf,btf->dbt",
        projected_embedding[eligible_indices].astype(np.float64), bootstrap_embedding,
        optimize=True,
    )
    bootstrap_risk_eligible = np.mean(
        projected_sq[:, None, :] + bootstrap_sq[None] - 2.0 * cross, axis=-1
    )
    bootstrap_risk = np.full(
        (len(centers), int(cfg["bootstrap_replicates"])), np.nan, dtype=np.float64
    )
    bootstrap_risk[eligible_indices] = bootstrap_risk_eligible

    best = int(eligible_indices[np.argmin(risks[eligible_indices])])
    order = eligible_indices[np.argsort(risks[eligible_indices])]
    best_se = float(np.std(bootstrap_risk[best], ddof=1))
    epsilon_values = sorted({
        max(float(cfg["minimum_reported_epsilon"]), multiplier * best_se)
        for multiplier in cfg["epsilon_multipliers_of_best_bootstrap_se"]
    })
    epsilon_rows: list[dict] = []
    for epsilon in epsilon_values:
        near = eligible_indices[risks[eligible_indices] <= risks[best] + epsilon]
        epsilon_rows.append({
            "epsilon": epsilon,
            "epsilon_over_best_risk": epsilon / risks[best],
            "near_optimal_count": len(near),
            "near_optimal_fraction_of_eligible": len(near) / len(eligible_indices),
            "maximum_geometry_distance_from_best_km": max(
                (set_distance(centers[best], centers[index]) for index in near), default=0.0
            ),
        })
    widest = epsilon_values[-1]
    near = eligible_indices[risks[eligible_indices] <= risks[best] + widest]
    alternative = int(max(near, key=lambda index: set_distance(centers[best], centers[index])))
    poor = int(eligible_indices[np.argmax(risks[eligible_indices])])

    summary_rows: list[dict] = []
    for design in range(len(centers)):
        row = {
            "rank_among_eligible": (
                int(np.flatnonzero(order == design)[0]) + 1 if eligible[design] else ""
            ),
            "design_index": design, "design_id": design_ids[design],
            "style": styles[design], "sigma_km": sigma,
            "fully_grid_usable": bool(eligible[design]),
            "validation_mmd_risk": risks[design],
            "bootstrap_risk_se": (
                float(np.std(bootstrap_risk[design], ddof=1)) if eligible[design] else math.nan
            ),
            "minimum_log10_intrinsic_ess": float(np.nanmin(log10_ess[design])),
            "median_log10_intrinsic_ess": float(np.nanmedian(log10_ess[design])),
            "worst_covariance_condition_regularized": float(np.nanmax(condition[design])),
            "mean_projection_kl": float(np.nanmean(kl[design])),
            "maximum_projection_kl": float(np.nanmax(kl[design])),
            "maximum_moment_residual": float(np.nanmax(residual[design])),
            "maximum_native_iterations": int(np.max(iterations[design])),
            "projection_failure_count": int(np.sum(~usable[design])),
        }
        for sensor, (x, y) in enumerate(centers[design], start=1):
            row[f"s{sensor}_x_km"] = x
            row[f"s{sensor}_y_km"] = y
        summary_rows.append(row)
    write_csv(
        table_dir / "grid_sensor_bank_risk.csv",
        sorted(summary_rows, key=lambda row: (not row["fully_grid_usable"], row["validation_mmd_risk"])),
    )
    write_csv(table_dir / "grid_epsilon_candidates.csv", epsilon_rows)
    time_rows: list[dict] = []
    for design in range(len(centers)):
        for time_index, day in enumerate(evaluation_days):
            time_rows.append({
                "design_index": design, "design_id": design_ids[design],
                "day": float(day), "source_time_index": int(evaluation_indices[time_index]),
                "fully_grid_usable": bool(eligible[design]),
                "validation_mmd2": risk_by_time[design, time_index],
            })
    write_csv(table_dir / "grid_risk_by_time.csv", time_rows)

    primary_ranks = np.argsort(np.argsort(risks[eligible_indices]))
    primary_top = set(order[:20].tolist())
    stability_rows: list[dict] = []
    for replicate in range(bootstrap_risk.shape[1]):
        values = bootstrap_risk[eligible_indices, replicate]
        replicate_order = eligible_indices[np.argsort(values)]
        rho = float(spearmanr(primary_ranks, np.argsort(np.argsort(values))).statistic)
        stability_rows.append({
            "replicate": replicate,
            "spearman_rank_correlation": rho,
            "top20_overlap": len(primary_top & set(replicate_order[:20].tolist())),
            "bootstrap_best_design_id": design_ids[int(replicate_order[0])],
            "primary_best_rank_in_bootstrap": int(np.flatnonzero(replicate_order == best)[0]) + 1,
        })
    write_csv(table_dir / "grid_risk_bootstrap_stability.csv", stability_rows)

    np.savez_compressed(
        processed / "iprojection_grid_validation_risk.npz",
        evaluation_indices=evaluation_indices, evaluation_days=evaluation_days,
        design_id=design_ids, eligible=eligible, risks=risks,
        risk_by_time=risk_by_time, projected_rff_embedding=projected_embedding,
        base_risk_by_time=base_risk_by_time, bootstrap_risk=bootstrap_risk,
        bandwidth_km=np.asarray(bandwidth), rff_omega=omega, rff_phase=phase,
        best_design_index=np.asarray(best), near_alternative_index=np.asarray(alternative),
        poor_design_index=np.asarray(poor), final_test_accessed=np.asarray(False),
    )

    old_joint = eligible & old_feasible
    old_comparison_rows = [{
        "design_index": int(index), "design_id": design_ids[index],
        "old_particle_risk": old_risks[index], "continuous_grid_risk": risks[index],
        "absolute_risk_difference": abs(old_risks[index] - risks[index]),
    } for index in np.flatnonzero(old_joint)]
    write_csv(table_dir / "particle_vs_grid_validation_risk.csv", old_comparison_rows)

    summary = {
        "method": "continuous-grid I-projection with frozen 128-feature Gaussian RFF MMD",
        "eligible_design_count": len(eligible_indices),
        "excluded_unresolved_design_count": int(np.sum(~eligible)),
        "evaluation_time_count": len(evaluation_indices),
        "validation_id_count": len(validation),
        "kernel_bandwidth_km": bandwidth,
        "rff_features": omega.shape[1],
        "time_weighting": cfg["time_weighting"],
        "baseline_continuous_reference_risk": float(base_risk_by_time.mean()),
        "best_design_index": best, "best_design_id": design_ids[best],
        "best_validation_risk": float(risks[best]),
        "best_bootstrap_risk_se": best_se,
        "near_alternative_index": alternative,
        "near_alternative_id": design_ids[alternative],
        "near_alternative_risk": float(risks[alternative]),
        "near_alternative_geometry_distance_km": set_distance(
            centers[best], centers[alternative]
        ),
        "poor_design_index": poor, "poor_design_id": design_ids[poor],
        "poor_validation_risk": float(risks[poor]),
        "epsilon_candidates": epsilon_rows,
        "bootstrap_median_spearman": float(np.median([
            row["spearman_rank_correlation"] for row in stability_rows
        ])),
        "bootstrap_median_top20_overlap": float(np.median([
            row["top20_overlap"] for row in stability_rows
        ])),
        "old_particle_best_design_id": design_ids[old_best],
        "old_particle_best_grid_eligible": bool(eligible[old_best]),
        "old_particle_best_grid_risk": float(risks[old_best]) if eligible[old_best] else None,
        "grid_embedding_seconds": time.perf_counter() - started,
        "final_test_artifact_loaded": False,
        "config_sha256": sha256(cfg["_config_path"]),
        "full_projection_table_sha256": sha256(resolve(cfg["full_projection_table"])),
        "frozen_rff_source_sha256": sha256(resolve(cfg["frozen_rff_source"])),
    }
    write_json(table_dir / "grid_validation_risk_summary.json", summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].hist(risks[eligible], bins=35, color="#4c78a8", alpha=0.85)
    axes[0, 0].axvline(risks[best], color="black", linestyle="--")
    axes[0, 0].set_xlabel("validation RFF-MMD² risk")
    axes[0, 0].set_ylabel("fully usable layouts")
    axes[0, 1].plot(np.arange(1, len(order) + 1), risks[order], color="#d95f02")
    axes[0, 1].set_xlabel("eligible design rank")
    axes[0, 1].set_ylabel("validation risk")
    axes[1, 0].scatter(np.nanmean(kl[eligible], axis=1), risks[eligible], s=12, alpha=0.5)
    axes[1, 0].set_xlabel("mean projection KL")
    axes[1, 0].set_ylabel("validation risk")
    axes[1, 1].scatter(-np.nanmin(log10_ess[eligible], axis=1), risks[eligible], s=12, alpha=0.5)
    axes[1, 1].set_xscale("symlog", linthresh=1.0)
    axes[1, 1].set_xlabel("worst −log10 intrinsic ESS")
    axes[1, 1].set_ylabel("validation risk")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    fig.suptitle("Continuous-grid validation-law-risk ranking")
    fig.savefig(figure_dir / "grid_bank_risk_diagnostics.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for index, label, color in [
        (best, "best", "#4c78a8"),
        (alternative, "near geometric alternative", "#59a14f"),
        (poor, "poor-risk control", "#e45756"),
    ]:
        axis.plot(evaluation_days, risk_by_time[index], marker="o", markersize=3, label=f"{label}: {design_ids[index]}", color=color)
    axis.plot(evaluation_days, base_risk_by_time, linestyle="--", color="black", label="unprojected continuous reference")
    axis.set_xlabel("day")
    axis.set_ylabel("validation RFF-MMD²")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(figure_dir / "selected_risk_by_time.png", dpi=190)
    plt.close(fig)

    representative = [(best, "best"), (alternative, "near-optimal alternative"), (poor, "poor-risk control")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, (index, label) in zip(axes, representative, strict=True):
        axis.hexbin(
            inference[:, ::4, 0].ravel(), inference[:, ::4, 1].ravel(),
            gridsize=45, bins="log", mincnt=1, cmap="Greys",
        )
        axis.scatter(
            centers[index, :, 0], centers[index, :, 1], marker="X", s=90,
            c=np.arange(4), cmap="tab10", edgecolor="black",
        )
        axis.set_title(f"{label}\n{design_ids[index]}  R={risks[index]:.5f}")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    fig.supxlabel("x (km)")
    fig.supylabel("y (km)")
    fig.savefig(figure_dir / "grid_risk_selected_layouts.png", dpi=190)
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
