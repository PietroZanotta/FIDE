#!/usr/bin/env python3
"""Native I-projection sweep and held-out RBF-MMD risk evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# One native solve has batch size one. Independent Python workers provide the
# bank-level parallelism; nested OpenMP teams would only oversubscribe the CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_PROC_BIND", "close")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import (  # noqa: E402
    gaussian_features_numpy,
    load_phase2_config,
    resolve,
    rff_map,
    rff_parameters,
    sha256,
    write_csv,
    write_json,
)

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.projection import EmpiricalIProjector, IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import (  # noqa: E402
    is_tesseract_iprojection_available,
    solve_i_projection_trajectory_tesseract_forward,
)
from mfsi.poisson_tesseract import is_tesseract_poisson_available  # noqa: E402

jax.config.update("jax_enable_x64", True)


def stable_weights(phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    logits = np.einsum("tnm,tm->tn", phi, lam)
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights


def set_distance(a: np.ndarray, b: np.ndarray) -> float:
    cost = cdist(a, b)
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def evaluate_one(
    index: int,
    centers: np.ndarray,
    nodes: np.ndarray,
    targets: np.ndarray,
    sigma: float,
    projection_cfg: IProjectionConfig,
    accept: dict,
):
    phi = gaussian_features_numpy(nodes, centers, sigma)
    n = phi.shape[1]
    log_base = np.full(phi.shape[:2], -math.log(n), dtype=np.float64)
    native = solve_i_projection_trajectory_tesseract_forward(
        phi, log_base, targets[None, ...], projection_cfg
    )
    lam = np.asarray(native["lambda_values"][0])
    weights = stable_weights(phi, lam)
    moments = np.einsum("tn,tnm->tm", weights, phi)
    residual = moments - targets
    centered = phi - moments[:, None, :]
    covariance = np.einsum("tn,tni,tnj->tij", weights, centered, centered)
    eig = np.linalg.eigvalsh(covariance)
    ridge = float(projection_cfg.newton_ridge)
    regularized_condition = (eig[:, -1] + ridge) / np.maximum(eig[:, 0] + ridge, 1e-300)
    raw_condition = eig[:, -1] / np.maximum(eig[:, 0], 1e-300)
    ess_fraction = (1.0 / np.maximum(np.sum(weights * weights, axis=-1), 1e-300)) / n
    kl = np.sum(weights * np.log(np.maximum(weights * n, 1e-300)), axis=-1)
    residual_norm = np.linalg.norm(residual, axis=-1)
    support_margin = np.min(
        np.minimum(targets - phi.min(axis=1), phi.max(axis=1) - targets), axis=-1
    )
    converged = np.asarray(native["converged"][0], dtype=bool)
    failure = (
        ~converged | ~np.isfinite(residual_norm)
        | (residual_norm > accept["accept_residual"])
        | (ess_fraction < accept["min_ess_fraction"])
        | (regularized_condition > accept["max_covariance_condition"])
        | (support_margin < -1.0e-12)
    )
    warning = eig[:, 0] < accept["min_covariance_eigenvalue"]
    rows = []
    for t in range(len(targets)):
        reasons = []
        if not converged[t]: reasons.append("native_nonconvergence")
        if residual_norm[t] > accept["accept_residual"]: reasons.append("moment_residual")
        if ess_fraction[t] < accept["min_ess_fraction"]: reasons.append("low_ess")
        if regularized_condition[t] > accept["max_covariance_condition"]: reasons.append("regularized_covariance_condition")
        if support_margin[t] < 0: reasons.append("coordinate_support_infeasible")
        rows.append({
            "design_index": index, "time_index": t,
            "native_converged": bool(converged[t]),
            "native_iterations": int(native["iterations"][0, t]),
            "native_reported_residual": float(native["residual_norm"][0, t]),
            "verified_moment_residual": float(residual_norm[t]),
            "lambda_1": lam[t, 0], "lambda_2": lam[t, 1],
            "lambda_3": lam[t, 2], "lambda_4": lam[t, 3],
            "covariance_min_eigenvalue": eig[t, 0],
            "covariance_max_eigenvalue": eig[t, -1],
            "covariance_condition_raw": raw_condition[t],
            "covariance_condition_regularized": regularized_condition[t],
            "ess_fraction": ess_fraction[t], "effective_sample_size": ess_fraction[t] * n,
            "projection_kl": kl[t], "coordinate_support_margin": support_margin[t],
            "conditioning_warning": bool(warning[t]),
            "valid": not bool(failure[t]), "failure_reason": ";".join(reasons),
        })
    return index, lam, weights.astype(np.float32), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    if not is_tesseract_iprojection_available():
        raise RuntimeError("native I-projection Tesseract was explicitly requested but is unavailable")
    poisson_available = is_tesseract_poisson_available()
    if not poisson_available:
        raise RuntimeError("Phase-3 weighted-Poisson Tesseract health contract failed")
    processed = resolve(cfg["processed_dir"])
    model_dir = resolve(cfg["model_dir"])
    analysis = resolve(cfg["analysis_dir"])
    for directory in [analysis / "figures/iprojection", analysis / "figures/risk", analysis / "figures/sensors", analysis / "tables"]:
        directory.mkdir(parents=True, exist_ok=True)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        days = np.asarray(data["relative_days"], dtype=np.float64)
    inference = X[split == "inference"]
    validation = X[split == "validation"]
    assert len(inference) == 200 and len(validation) == 70
    with np.load(model_dir / "reference_bank.npz", allow_pickle=False) as data:
        reference = np.asarray(data["nodes_km"], dtype=np.float64)
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        design_ids = np.asarray(data["design_id"]).astype(str)
        styles = np.asarray(data["style"]).astype(str)
        sigma = float(data["sigma_km"])
    with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"], dtype=np.float64)
    if args.limit is not None:
        centers = centers[:args.limit]; design_ids = design_ids[:args.limit]
        styles = styles[:args.limit]; measurements = measurements[:args.limit]
    stride = int(cfg["projection"]["evaluation_stride"])
    evaluation_indices = np.unique(np.r_[np.arange(0, len(days), stride), len(days) - 1]).astype(int)
    evaluation_days = days[evaluation_indices]
    nodes = reference[evaluation_indices]
    targets = measurements[:, evaluation_indices]
    p = cfg["projection"]
    projection_cfg = IProjectionConfig(
        max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
        newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
        implicit_ridge=float(p["implicit_ridge"]),
    )
    print(
        f"[iprojection] native sweep: designs={len(centers)}, times={len(evaluation_indices)}, "
        f"particles={nodes.shape[1]}, workers={p['workers']}", flush=True,
    )
    started = time.perf_counter()
    lambdas = np.empty((len(centers), len(evaluation_indices), 4), dtype=np.float64)
    weights = np.empty((len(centers), len(evaluation_indices), nodes.shape[1]), dtype=np.float32)
    diagnostics_by_index = {}
    with ThreadPoolExecutor(max_workers=int(p["workers"])) as executor:
        futures = [
            executor.submit(
                evaluate_one, i, centers[i], nodes, targets[i], sigma, projection_cfg, p
            ) for i in range(len(centers))
        ]
        completed = 0
        for future in as_completed(futures):
            i, lam, w, rows = future.result()
            lambdas[i] = lam; weights[i] = w; diagnostics_by_index[i] = rows
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(f"[iprojection] {completed}/{len(futures)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    native_seconds = time.perf_counter() - started
    diagnostic_rows = []
    for i in range(len(centers)):
        for row, day, source_index in zip(diagnostics_by_index[i], evaluation_days, evaluation_indices, strict=True):
            row["design_id"] = design_ids[i]; row["day"] = float(day); row["source_time_index"] = int(source_index)
            diagnostic_rows.append(row)
    write_csv(analysis / "tables/iprojection_diagnostics.csv", diagnostic_rows)

    risk_cfg = cfg["law_risk"]
    rng = np.random.default_rng(int(cfg["seed"]) + 3001)
    bandwidth_points = np.concatenate([
        inference[:, evaluation_indices].reshape(-1, 2),
        validation[:, evaluation_indices].reshape(-1, 2),
    ])
    sample_n = min(int(risk_cfg["bandwidth_sample_points"]), len(bandwidth_points))
    bandwidth_points = bandwidth_points[rng.choice(len(bandwidth_points), sample_n, replace=False)]
    bandwidth = float(np.median(pdist(bandwidth_points)))
    omega, phase = rff_parameters(int(cfg["seed"]) + 3101, int(risk_cfg["rff_features"]), bandwidth)
    reference_rff = np.stack([rff_map(points, omega, phase) for points in nodes])
    validation_rff_by_id = np.stack([
        np.stack([rff_map(validation[i, index:index + 1], omega, phase)[0] for index in evaluation_indices])
        for i in range(len(validation))
    ])
    validation_embedding = validation_rff_by_id.mean(axis=0)
    # GPU batched contraction makes the weighted RFF mean inexpensive.
    projected_embedding = np.asarray(jnp.einsum(
        "dtn,tnf->dtf", jnp.asarray(weights), jnp.asarray(reference_rff)
    ))
    risk_by_time = np.sum((projected_embedding - validation_embedding[None]) ** 2, axis=-1)
    risks = risk_by_time.mean(axis=-1)
    base_embedding = reference_rff.mean(axis=1)
    base_risk_by_time = np.sum((base_embedding - validation_embedding) ** 2, axis=-1)

    bootstrap_rng = np.random.default_rng(int(cfg["seed"]) + int(risk_cfg["bootstrap_seed_offset"]))
    bootstrap_indices = bootstrap_rng.integers(0, len(validation), size=(int(risk_cfg["bootstrap_replicates"]), len(validation)))
    bootstrap_embedding = validation_rff_by_id[bootstrap_indices].mean(axis=1)
    bootstrap_risk = np.asarray(jnp.mean(jnp.sum(
        (jnp.asarray(projected_embedding)[:, None] - jnp.asarray(bootstrap_embedding)[None]) ** 2,
        axis=-1,
    ), axis=-1))

    summary_rows = []
    feasible = np.empty(len(centers), dtype=bool)
    for i in range(len(centers)):
        rows = diagnostics_by_index[i]
        failures = sum(not row["valid"] for row in rows)
        warnings = sum(row["conditioning_warning"] for row in rows)
        feasible[i] = failures == 0
        row = {
            "design_id": design_ids[i], "style": styles[i], "sigma_km": sigma,
            "validation_mmd_risk": risks[i], "bootstrap_risk_se": np.std(bootstrap_risk[i], ddof=1),
            "minimum_ess_fraction": min(x["ess_fraction"] for x in rows),
            "median_ess_fraction": np.median([x["ess_fraction"] for x in rows]),
            "worst_covariance_condition_raw": max(x["covariance_condition_raw"] for x in rows),
            "worst_covariance_condition_regularized": max(x["covariance_condition_regularized"] for x in rows),
            "minimum_covariance_eigenvalue": min(x["covariance_min_eigenvalue"] for x in rows),
            "projection_failure_count": failures, "conditioning_warning_count": warnings,
            "mean_projection_kl": np.mean([x["projection_kl"] for x in rows]),
            "max_moment_residual": max(x["verified_moment_residual"] for x in rows),
            "max_native_iterations": max(x["native_iterations"] for x in rows),
            "feasible": bool(feasible[i]),
        }
        for j, (x, y) in enumerate(centers[i], start=1):
            row[f"s{j}_x_km"] = x; row[f"s{j}_y_km"] = y
        summary_rows.append(row)
    feasible_indices = np.flatnonzero(feasible)
    if not feasible_indices.size:
        raise RuntimeError("No sensor design passed the explicit projection validity contract")
    best = int(feasible_indices[np.argmin(risks[feasible_indices])])
    order = feasible_indices[np.argsort(risks[feasible_indices])]
    best_se = float(np.std(bootstrap_risk[best], ddof=1))
    epsilon_values = sorted(set([best_se, 2 * best_se, max(best_se / 2, 1e-5)]))
    epsilon_rows = []
    for epsilon in epsilon_values:
        near = feasible_indices[risks[feasible_indices] <= risks[best] + epsilon]
        diversity = max((set_distance(centers[best], centers[i]) for i in near), default=0.0)
        epsilon_rows.append({
            "epsilon": epsilon, "near_optimal_count": len(near),
            "near_optimal_fraction_of_feasible": len(near) / len(feasible_indices),
            "max_geometry_distance_from_best_km": diversity,
        })
    write_csv(analysis / "tables/epsilon_candidates.csv", epsilon_rows)
    write_csv(analysis / "tables/sensor_bank_results.csv", sorted(summary_rows, key=lambda row: (not row["feasible"], row["validation_mmd_risk"])))
    risk_time_rows = []
    for i in range(len(centers)):
        for day, value in zip(evaluation_days, risk_by_time[i], strict=True):
            risk_time_rows.append({"design_id": design_ids[i], "day": day, "mmd2": value})
    write_csv(analysis / "tables/risk_by_time.csv", risk_time_rows)

    rank_correlations = []
    primary_ranks = np.argsort(np.argsort(risks[feasible_indices]))
    primary_top = set(feasible_indices[np.argsort(risks[feasible_indices])[:20]].tolist())
    for b in range(bootstrap_risk.shape[1]):
        sample = bootstrap_risk[feasible_indices, b]
        rho = float(spearmanr(primary_ranks, np.argsort(np.argsort(sample))).statistic)
        sample_top = set(feasible_indices[np.argsort(sample)[:20]].tolist())
        rank_correlations.append({"replicate": b, "spearman_rank_correlation": rho, "top20_overlap": len(primary_top & sample_top)})
    write_csv(analysis / "tables/risk_bootstrap_stability.csv", rank_correlations)

    # Use the widest reported finite-sample tolerance for the geometric
    # alternative. Narrower candidates can legitimately contain only the best.
    near_epsilon = epsilon_values[-1]
    near = feasible_indices[risks[feasible_indices] <= risks[best] + near_epsilon]
    alternative = int(max(near, key=lambda i: set_distance(centers[best], centers[i])))
    poor = int(feasible_indices[np.argmax(risks[feasible_indices])])

    tight_cfg = IProjectionConfig(
        max_steps=int(p["max_steps"] * 2), residual_tol=float(p["tight_residual_tol"]),
        newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
    )
    robustness_rows = []
    for i in sorted(set([0, best, alternative, poor])):
        phi = gaussian_features_numpy(nodes, centers[i], sigma)
        log_base = np.full(phi.shape[:2], -math.log(phi.shape[1]))
        tight = solve_i_projection_trajectory_tesseract_forward(phi, log_base, targets[i:i + 1], tight_cfg)
        tight_lam = tight["lambda_values"][0]
        tight_weights = stable_weights(phi, tight_lam)
        tight_embedding = np.einsum("tn,tnf->tf", tight_weights.astype(np.float32), reference_rff)
        tight_risk = np.mean(np.sum((tight_embedding - validation_embedding) ** 2, axis=-1))
        half_phi = phi[:, :2000]
        half_log = np.full(half_phi.shape[:2], -math.log(half_phi.shape[1]))
        half = solve_i_projection_trajectory_tesseract_forward(half_phi, half_log, targets[i:i + 1], projection_cfg)
        half_weights = stable_weights(half_phi, half["lambda_values"][0])
        half_embedding = np.einsum("tn,tnf->tf", half_weights.astype(np.float32), reference_rff[:, :2000])
        half_risk = np.mean(np.sum((half_embedding - validation_embedding) ** 2, axis=-1))
        robustness_rows.append({
            "design_id": design_ids[i], "role": "best" if i == best else ("near_alternative" if i == alternative else ("poor" if i == poor else "manual_control")),
            "primary_risk": risks[i], "tight_tolerance_risk": tight_risk,
            "absolute_tight_risk_change": abs(tight_risk - risks[i]),
            "max_lambda_change_tight": np.max(np.abs(tight_lam - lambdas[i])),
            "half_reference_bank_risk": half_risk,
            "absolute_half_bank_risk_change": abs(half_risk - risks[i]),
            "tight_max_residual": float(np.max(tight["residual_norm"])),
            "half_bank_max_residual": float(np.max(half["residual_norm"])),
        })
    write_csv(analysis / "tables/numerical_robustness.csv", robustness_rows)

    np.savez_compressed(
        processed / "iprojection_primary.npz", lambdas=lambdas,
        evaluation_indices=evaluation_indices, evaluation_days=evaluation_days,
        design_id=design_ids, feasible=feasible, risks=risks,
        risk_by_time=risk_by_time, bandwidth_km=bandwidth, rff_omega=omega,
        rff_phase=phase, best_design_index=best, near_alternative_index=alternative,
        poor_design_index=poor,
    )
    manifest = {
        "backend": "tesseract_cpp", "native_iprojection_available": True,
        "native_weighted_poisson_available": poisson_available,
        "weighted_poisson_invoked": False,
        "weighted_poisson_reason": "Phase 2 explicitly stops before action/Poisson computation",
        "design_count": len(centers), "evaluation_time_count": len(evaluation_indices),
        "reference_particles": nodes.shape[1], "native_sweep_seconds": native_seconds,
        "feasible_designs": int(feasible.sum()), "conditioning_warning_designs": int(sum(row["conditioning_warning_count"] > 0 for row in summary_rows)),
        "kernel": risk_cfg["kernel"], "kernel_bandwidth_km": bandwidth,
        "rff_features": int(risk_cfg["rff_features"]),
        "baseline_reference_risk": float(base_risk_by_time.mean()),
        "best_design_id": design_ids[best], "best_validation_risk": float(risks[best]),
        "near_alternative_id": design_ids[alternative],
        "near_alternative_geometry_distance_km": set_distance(centers[best], centers[alternative]),
        "poor_design_id": design_ids[poor], "epsilon_candidates": epsilon_rows,
        "bootstrap_median_spearman": float(np.median([row["spearman_rank_correlation"] for row in rank_correlations])),
        "bootstrap_median_top20_overlap": float(np.median([row["top20_overlap"] for row in rank_correlations])),
        "final_test_artifact_loaded": False,
        "sensor_bank_sha256": sha256(processed / "sensor_bank.npz"),
        "reference_bank_sha256": sha256(model_dir / "reference_bank.npz"),
    }
    write_json(analysis / "tables/iprojection_risk_summary.json", manifest)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(risks[feasible], bins=35, color="#4c78a8", alpha=.85); axes[0, 0].axvline(risks[best], color="black", ls="--")
    axes[0, 0].set_xlabel("validation RFF-MMD² risk"); axes[0, 0].set_ylabel("designs")
    axes[0, 1].scatter([row["minimum_ess_fraction"] for row in summary_rows], risks, s=10, alpha=.5)
    axes[0, 1].set_xlabel("minimum ESS fraction"); axes[0, 1].set_ylabel("risk")
    axes[1, 0].scatter([row["worst_covariance_condition_regularized"] for row in summary_rows], risks, s=10, alpha=.5)
    axes[1, 0].set_xscale("log"); axes[1, 0].set_xlabel("worst regularized covariance condition"); axes[1, 0].set_ylabel("risk")
    ranked = risks[order]
    axes[1, 1].plot(np.arange(len(ranked)), ranked, color="#d95f02"); axes[1, 1].set_xlabel("feasible design rank"); axes[1, 1].set_ylabel("risk")
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.suptitle("Frozen sensor-bank I-projection and validation-risk diagnostics")
    fig.tight_layout(); fig.savefig(analysis / "figures/risk/bank_risk_diagnostics.png", dpi=190); plt.close(fig)

    representative = [(best, "best"), (alternative, "near-optimal geometric alternative"), (poor, "poor-risk control"), (0, "manual control")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (i, label) in zip(axes.ravel(), representative, strict=True):
        ax.hexbin(inference[:, ::4, 0].ravel(), inference[:, ::4, 1].ravel(), gridsize=48, bins="log", mincnt=1, cmap="Greys")
        ax.scatter(centers[i, :, 0], centers[i, :, 1], marker="X", s=90, c=np.arange(4), cmap="tab10", edgecolor="black")
        ax.set_title(f"{label}\n{design_ids[i]}  R={risks[i]:.5f}"); ax.set_aspect("equal")
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.suptitle("Geometrically distinct sensor layouts")
    fig.tight_layout(); fig.savefig(analysis / "figures/sensors/risk_selected_layouts.png", dpi=190); plt.close(fig)

    chosen_days = [0, 10, 20, 45]
    chosen_t = [int(np.argmin(np.abs(evaluation_days - day))) for day in chosen_days]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
    plot_rng = np.random.default_rng(int(cfg["seed"]) + 3301)
    for column, (day, t) in enumerate(zip(chosen_days, chosen_t, strict=True)):
        ref_points = nodes[t]
        base_idx = plot_rng.choice(len(ref_points), 450, replace=False)
        projected_idx = plot_rng.choice(len(ref_points), 450, replace=True, p=np.asarray(weights[best, t], dtype=float) / np.sum(weights[best, t]))
        axes[0, column].scatter(ref_points[base_idx, 0], ref_points[base_idx, 1], s=7, alpha=.35, color="#4c78a8")
        axes[1, column].scatter(validation[:, evaluation_indices[t], 0], validation[:, evaluation_indices[t], 1], s=16, alpha=.7, color="#e45756")
        axes[2, column].scatter(ref_points[projected_idx, 0], ref_points[projected_idx, 1], s=7, alpha=.35, color="#54a24b")
        for row in range(3): axes[row, column].set_title(("reference", "validation", "I-projected")[row] + f" day {day}"); axes[row, column].set_aspect("equal"); axes[row, column].grid(alpha=.15)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.suptitle(f"Best design information projection: {design_ids[best]}")
    fig.tight_layout(); fig.savefig(analysis / "figures/iprojection/best_projection_snapshots.png", dpi=190); plt.close(fig)
    print(
        f"[risk] feasible={feasible.sum()}/{len(feasible)} best={design_ids[best]} "
        f"R*={risks[best]:.6g} baseline={base_risk_by_time.mean():.6g} native={native_seconds:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
