#!/usr/bin/env python3
"""Phase 2B: continuous-endpoint KDE reference and frozen support audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

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
from mfsi.flow_matching import FlowMatchingConfig, train_reference_flow  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import (  # noqa: E402
    is_tesseract_iprojection_available,
    solve_i_projection_trajectory_tesseract_forward,
)
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_repair_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else SCRIPT_DIR.parent / "configs/reference_support_repair.json"
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["_config_path"] = str(path.resolve())
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class GaussianKDEEndpointSource:
    x0_atoms: jax.Array
    x1_atoms: jax.Array
    x0_cholesky: jax.Array
    x1_cholesky: jax.Array

    def sample(self, key: jax.Array, n: int, endpoint: int) -> jax.Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be zero or one")
        atoms = self.x0_atoms if endpoint == 0 else self.x1_atoms
        cholesky = self.x0_cholesky if endpoint == 0 else self.x1_cholesky
        index_key, noise_key = jax.random.split(key)
        indices = jax.random.randint(index_key, (int(n),), 0, atoms.shape[0])
        noise = jax.random.normal(noise_key, (int(n), 2), dtype=jnp.float64)
        return atoms[indices] + noise @ cholesky.T


def scott_bandwidth(atoms: np.ndarray, ridge_relative: float) -> tuple[np.ndarray, float]:
    atoms = np.asarray(atoms, dtype=np.float64)
    factor = float(len(atoms) ** (-1.0 / (atoms.shape[1] + 4.0)))
    bandwidth = factor**2 * np.cov(atoms, rowvar=False, ddof=1)
    ridge = float(ridge_relative) * float(np.trace(bandwidth)) / atoms.shape[1]
    bandwidth = bandwidth + ridge * np.eye(atoms.shape[1])
    np.linalg.cholesky(bandwidth)
    return bandwidth, factor


def sample_kde(
    atoms: np.ndarray, bandwidth: np.ndarray, rng: np.random.Generator, count: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(atoms), size=int(count))
    noise = rng.normal(size=(int(count), 2)) @ np.linalg.cholesky(bandwidth).T
    return atoms[indices] + noise, indices


def log_kde_density(points: np.ndarray, atoms: np.ndarray, bandwidth: np.ndarray, chunk: int = 5000) -> np.ndarray:
    inverse = np.linalg.inv(bandwidth)
    log_norm = -math.log(2.0 * math.pi) - 0.5 * np.linalg.slogdet(bandwidth)[1] - math.log(len(atoms))
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        delta = points[start:start + chunk, None] - atoms[None]
        quadratic = np.einsum("nki,ij,nkj->nk", delta, inverse, delta)
        output[start:start + chunk] = logsumexp(-0.5 * quadratic, axis=1) + log_norm
    return output


def leave_one_out_log_density(atoms: np.ndarray, bandwidth: np.ndarray, floor: float) -> np.ndarray:
    inverse = np.linalg.inv(bandwidth)
    delta = atoms[:, None] - atoms[None]
    quadratic = np.einsum("nki,ij,nkj->nk", delta, inverse, delta)
    np.fill_diagonal(quadratic, np.inf)
    log_norm = -math.log(2.0 * math.pi) - 0.5 * np.linalg.slogdet(bandwidth)[1] - math.log(len(atoms) - 1)
    return np.maximum(logsumexp(-0.5 * quadratic, axis=1) + log_norm, math.log(floor))


def biased_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    xx = np.exp(-cdist(x, x, "sqeuclidean") / (2.0 * bandwidth**2)).mean()
    yy = np.exp(-cdist(y, y, "sqeuclidean") / (2.0 * bandwidth**2)).mean()
    xy = np.exp(-cdist(x, y, "sqeuclidean") / (2.0 * bandwidth**2)).mean()
    return float(xx + yy - 2.0 * xy)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(
        2.0 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()
    )


def flow_config(block: dict, seed: int) -> FlowMatchingConfig:
    return FlowMatchingConfig(
        seed=int(seed), hidden_width=int(block["hidden_width"]),
        hidden_layers=int(block["hidden_layers"]), train_steps=int(block["train_steps"]),
        batch_size=int(block["batch_size"]), learning_rate=float(block["learning_rate"]),
        min_learning_rate_ratio=float(block["min_learning_rate_ratio"]),
        adam_beta1=float(block["adam_beta1"]), adam_beta2=float(block["adam_beta2"]),
        adam_eps=float(block["adam_eps"]), grad_clip_norm=float(block["grad_clip_norm"]),
        bridge_schedule=str(block["bridge_schedule"]),
        bridge_noise_std=float(block["bridge_noise_std_normalized"]),
        log_every=int(block["log_every"]),
    )


def freeze_old_diagnostic(
    analysis: Path, processed: Path, tolerance: float
) -> list[dict]:
    table_dir = analysis / "tables"
    old_source = table_dir / "reference_support_lp.csv"
    old_copy = table_dir / "reference_support_lp_old.csv"
    if not old_copy.exists():
        shutil.copy2(old_source, old_copy)
    figure_source = analysis / "figures/iprojection/reference_support_lp_scaling.png"
    figure_copy = analysis / "figures/reference_support/lp_scaling/reference_support_lp_old.png"
    figure_copy.parent.mkdir(parents=True, exist_ok=True)
    if not figure_copy.exists():
        shutil.copy2(figure_source, figure_copy)
    rows = csv_rows(old_copy)
    maximum_size = max(int(row["nominal_particle_count"]) for row in rows)
    selected = sorted(
        [row for row in rows if int(row["nominal_particle_count"]) == maximum_size],
        key=lambda row: int(row["case"]),
    )
    if len(selected) != 20 or [int(row["case"]) for row in selected] != list(range(20)):
        raise RuntimeError("the preserved diagnostic does not contain the exact 20 cases")
    with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"], dtype=np.float64)
    cases = []
    for row in selected:
        design_index = int(row["design_index"])
        source_index = int(row["source_time_index"])
        cases.append({
            "case": int(row["case"]), "design_id": row["design_id"],
            "design_index": design_index, "style": row["style"],
            "day": float(row["day"]), "source_time_index": source_index,
            "target_moments": measurements[design_index, source_index].tolist(),
            "frozen_lp_tolerance": tolerance,
            "old_minimum_linf_residual": float(row["minimum_linf_residual"]),
            "old_exact_lp_success": row["exact_lp_success"] == "True",
            "old_native_failure_reason": row["original_failure_reason"],
            "old_native_residual": float(row["original_newton_residual"]),
        })
    manifest_path = table_dir / "reference_support_cases.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous["cases"] != cases:
            raise RuntimeError("frozen 20-case manifest changed")
    else:
        write_json(manifest_path, {
            "case_count": 20, "selection": "exact preserved Phase-2 failures",
            "final_test_artifact_loaded": False, "cases": cases,
        })
    return cases


def sparse_simplex_lp(phi: np.ndarray, target: np.ndarray, tolerance: float) -> dict:
    """Minimize L-infinity moment residual and test exact simplex feasibility."""
    phi = np.asarray(phi, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n, m = phi.shape
    objective = np.r_[np.zeros(n), 1.0]
    upper = csc_matrix(np.vstack([
        np.c_[phi.T, -np.ones(m)],
        np.c_[-phi.T, -np.ones(m)],
    ]))
    equality = csc_matrix(np.c_[np.ones((1, n)), np.zeros((1, 1))])
    minimum = linprog(
        objective, A_ub=upper, b_ub=np.r_[target, -target],
        A_eq=equality, b_eq=np.asarray([1.0]), bounds=(0.0, None), method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not minimum.success:
        raise RuntimeError(f"minimum-residual LP failed: {minimum.message}")
    weights = minimum.x[:-1]
    residual = weights @ phi - target
    linf = float(np.max(np.abs(residual)))
    exact_success = False
    exact_residual = math.nan
    if linf <= max(20.0 * tolerance, 1e-7):
        exact = linprog(
            np.zeros(n), A_eq=csc_matrix(np.vstack([np.ones(n), phi.T])),
            b_eq=np.r_[1.0, target], bounds=(0.0, None), method="highs",
            options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
        )
        exact_success = bool(exact.success)
        if exact.success:
            exact_residual = float(np.max(np.abs(exact.x @ phi - target)))
    return {
        "exact_lp_success": exact_success,
        "exact_lp_verified_linf_residual": exact_residual,
        "minimum_linf_residual": linf,
        "minimum_l2_residual_at_linf_solution": float(np.linalg.norm(residual)),
        "lp_active_weight_count": int(np.sum(weights > 1e-10)),
        "lp_maximum_weight": float(weights.max()),
    }


def stable_weights(phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    logits = phi @ lam
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def native_single(
    phi: np.ndarray, target: np.ndarray, config: IProjectionConfig, acceptance: dict
) -> dict:
    log_base = np.full((1, len(phi)), -math.log(len(phi)), dtype=np.float64)
    native = solve_i_projection_trajectory_tesseract_forward(
        phi[None], log_base, target[None, None], config
    )
    lam = np.asarray(native["lambda_values"][0, 0], dtype=np.float64)
    weights = stable_weights(phi, lam)
    achieved = weights @ phi
    residual = float(np.linalg.norm(achieved - target))
    centered = phi - achieved
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    eig = np.linalg.eigvalsh(covariance)
    condition = float((eig[-1] + config.newton_ridge) / max(eig[0] + config.newton_ridge, 1e-300))
    ess_fraction = float((1.0 / np.sum(weights * weights)) / len(weights))
    converged = bool(native["converged"][0, 0])
    healthy = bool(
        converged and residual <= acceptance["accept_residual"]
        and ess_fraction >= acceptance["min_ess_fraction"]
        and condition <= acceptance["max_covariance_condition"]
    )
    return {
        "native_converged": converged,
        "native_iterations": int(native["iterations"][0, 0]),
        "native_reported_residual": float(native["residual_norm"][0, 0]),
        "native_verified_l2_residual": residual,
        "native_ess_fraction": ess_fraction,
        "native_effective_sample_size": ess_fraction * len(weights),
        "native_covariance_min_eigenvalue": float(eig[0]),
        "native_covariance_max_eigenvalue": float(eig[-1]),
        "native_covariance_condition_regularized": condition,
        "native_multiplier_norm": float(np.linalg.norm(lam)),
        "native_maximum_weight": float(weights.max()),
        "native_healthy": healthy,
    }


def endpoint_metrics(
    empirical: np.ndarray,
    generated: np.ndarray,
    model: str,
    endpoint: str,
    bandwidth: float,
    domain: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    sample = generated[rng.choice(len(generated), min(2500, len(generated)), replace=False)]
    empirical_cov = np.cov(empirical, rowvar=False, ddof=1)
    generated_cov = np.cov(generated, rowvar=False, ddof=1)
    rms_spread = float(np.sqrt(np.trace(empirical_cov)))
    outside = (
        (generated[:, 0] < domain[0]) | (generated[:, 0] > domain[1])
        | (generated[:, 1] < domain[2]) | (generated[:, 1] > domain[3])
    )
    return {
        "model": model, "endpoint": endpoint,
        "sample_n": len(sample), "distribution_n": len(generated),
        "biased_rbf_mmd2": biased_mmd2(sample, empirical, bandwidth),
        "energy_distance_km": energy_distance(sample, empirical),
        "mean_error_km": float(np.linalg.norm(generated.mean(axis=0) - empirical.mean(axis=0))),
        "mean_error_over_empirical_rms_spread": float(np.linalg.norm(generated.mean(axis=0) - empirical.mean(axis=0)) / rms_spread),
        "relative_covariance_frobenius_error": float(np.linalg.norm(generated_cov - empirical_cov) / np.linalg.norm(empirical_cov)),
        "outside_domain_fraction": float(outside.mean()),
        "generated_mean_x_km": float(generated[:, 0].mean()),
        "generated_mean_y_km": float(generated[:, 1].mean()),
        "generated_cov_xx": float(generated_cov[0, 0]),
        "generated_cov_xy": float(generated_cov[0, 1]),
        "generated_cov_yy": float(generated_cov[1, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--force-bank", action="store_true")
    args = parser.parse_args()
    repair = load_repair_config(args.config)
    phase2 = load_phase2_config(resolve(repair["base_phase2_config"]))
    if not is_tesseract_iprojection_available():
        raise RuntimeError("native I-projection Tesseract is required")
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    old_model_dir = resolve(phase2["model_dir"])
    model_dir = SCRIPT_DIR.parent / "models/reference_flow_continuous_endpoints"
    estimator_dir = processed / "endpoint_density_estimator"
    table_dir = analysis / "tables"
    endpoint_fig_dir = analysis / "figures/reference_support/endpoints"
    compare_fig_dir = analysis / "figures/reference_support/old_vs_new"
    lp_fig_dir = analysis / "figures/reference_support/lp_scaling"
    for directory in [model_dir, estimator_dir, table_dir, endpoint_fig_dir, compare_fig_dir, lp_fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    p = phase2["projection"]
    cases = freeze_old_diagnostic(analysis, processed, float(p["accept_residual"]))
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
        days = np.asarray(data["relative_days"], dtype=np.float64)
        domain = np.asarray(data["domain_km"], dtype=np.float64)
    inference = X[split == "inference"]
    validation = X[split == "validation"]
    if inference.shape != (200, 181, 2) or validation.shape != (70, 181, 2):
        raise RuntimeError("frozen primary 200/70 split changed")
    x0 = inference[:, 0]
    x1 = inference[:, -1]
    kde_cfg = repair["endpoint_estimator"]
    H0, factor0 = scott_bandwidth(x0, float(kde_cfg["covariance_ridge_relative_trace"]))
    H1, factor1 = scott_bandwidth(x1, float(kde_cfg["covariance_ridge_relative_trace"]))
    estimator_path = estimator_dir / "gaussian_kde_endpoints.npz"
    np.savez_compressed(
        estimator_path, x0_atoms_km=x0, x1_atoms_km=x1,
        H0_km2=H0, H1_km2=H1, scott_factor_0=factor0, scott_factor_1=factor1,
        rule=np.asarray("n^(-1/(d+4)) full empirical covariance; endpoint-only"),
        inference_n=np.asarray(200), final_test_accessed=np.asarray(False),
    )
    audit_rng = np.random.default_rng(int(repair["seed"]) + 100)
    audit_n = int(kde_cfg["audit_samples"])
    samples0, _ = sample_kde(x0, H0, audit_rng, audit_n)
    samples1, _ = sample_kde(x1, H1, audit_rng, audit_n)
    estimator_rows = []
    mass_levels = [0.50, 0.90, 0.95, 0.99]
    for label, atoms, bandwidth, samples, factor in [
        ("day0", x0, H0, samples0, factor0), ("day45", x1, H1, samples1, factor1)
    ]:
        loo = leave_one_out_log_density(atoms, bandwidth, float(kde_cfg["loo_density_floor"]))
        density_sample = log_kde_density(samples, atoms, bandwidth)
        row = {
            "endpoint": label, "n_atoms": len(atoms), "scott_factor": factor,
            "bandwidth_xx_km2": bandwidth[0, 0], "bandwidth_xy_km2": bandwidth[0, 1],
            "bandwidth_yy_km2": bandwidth[1, 1],
            "bandwidth_min_eigenvalue_km2": np.linalg.eigvalsh(bandwidth)[0],
            "bandwidth_max_eigenvalue_km2": np.linalg.eigvalsh(bandwidth)[-1],
            "bandwidth_condition": np.linalg.cond(bandwidth),
            "mean_loo_log_density": loo.mean(), "minimum_loo_log_density": loo.min(),
            "empirical_mean_x_km": atoms[:, 0].mean(), "empirical_mean_y_km": atoms[:, 1].mean(),
            "kde_mean_x_km": samples[:, 0].mean(), "kde_mean_y_km": samples[:, 1].mean(),
            "kde_robust_spread_km": np.median(np.linalg.norm(samples - np.median(samples, axis=0), axis=1)),
            "kde_outside_domain_fraction": np.mean(
                (samples[:, 0] < domain[0]) | (samples[:, 0] > domain[1])
                | (samples[:, 1] < domain[2]) | (samples[:, 1] > domain[3])
            ),
        }
        for level in mass_levels:
            row[f"hdr_{int(level * 100)}pct_log_density_threshold"] = np.quantile(density_sample, 1.0 - level)
        estimator_rows.append(row)
    write_csv(table_dir / "reference_endpoint_kde_audit.csv", estimator_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, label, atoms, bandwidth, samples in [
        (axes[0], "day 0", x0, H0, samples0), (axes[1], "day 45", x1, H1, samples1)
    ]:
        low = np.quantile(samples, .002, axis=0); high = np.quantile(samples, .998, axis=0)
        gx, gy = np.meshgrid(np.linspace(low[0], high[0], 170), np.linspace(low[1], high[1], 170))
        grid = np.c_[gx.ravel(), gy.ravel()]
        density = np.exp(log_kde_density(grid, atoms, bandwidth)).reshape(gx.shape)
        ax.contourf(gx, gy, density, levels=14, cmap="Blues", alpha=.8)
        ax.scatter(atoms[:, 0], atoms[:, 1], s=12, color="#e45756", alpha=.65, label="inference endpoints")
        sample_index = audit_rng.choice(len(samples), 1200, replace=False)
        ax.scatter(samples[sample_index, 0], samples[sample_index, 1], s=3, color="black", alpha=.12, label="KDE samples")
        ax.set_title(f"Full-covariance Scott KDE: {label}"); ax.set_aspect("equal"); ax.grid(alpha=.15)
        ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    axes[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(endpoint_fig_dir / "endpoint_kde_contours.png", dpi=190); plt.close(fig)

    block = phase2["reference_training"]
    center = np.asarray(block["normalization_center_km"], dtype=np.float64)
    scale = float(block["normalization_scale_km"])
    normalized_x0 = (x0 - center) / scale
    normalized_x1 = (x1 - center) / scale
    normalized_H0 = H0 / scale**2
    normalized_H1 = H1 / scale**2
    train_cfg = flow_config(block, int(repair["flow"]["training_seed"]))
    signature = json.dumps({
        "estimator_sha256": sha256(estimator_path), "training": asdict(train_cfg),
        "center": center.tolist(), "scale": scale,
        "architecture_optimizer_schedule_inherited": True,
    }, sort_keys=True)
    checkpoint = model_dir / "reference.npz"
    flow = None
    if checkpoint.exists() and not args.force_training:
        candidate = MLPReferenceFlow.from_npz(
            checkpoint, substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"])
        )
        if (candidate.metadata or {}).get("repair_signature") == signature:
            flow = candidate
            print("[repair] reusing continuous-endpoint checkpoint", flush=True)
    if flow is None:
        source = GaussianKDEEndpointSource(
            jnp.asarray(normalized_x0), jnp.asarray(normalized_x1),
            jnp.asarray(np.linalg.cholesky(normalized_H0)),
            jnp.asarray(np.linalg.cholesky(normalized_H1)),
        )
        started = time.perf_counter()
        flow, history = train_reference_flow(
            source, train_cfg,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        training_seconds = time.perf_counter() - started
        metadata = dict(flow.metadata or {})
        metadata.update({
            "repair_signature": signature, "endpoint_source": "full-covariance Scott Gaussian KDE",
            "endpoint_only": True, "intermediate_positions_used_for_training": False,
            "architecture_optimizer_schedule_changed": False,
            "history": history, "training_seconds": training_seconds,
        })
        flow = MLPReferenceFlow(
            flow.params,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
            metadata=metadata,
        )
        save_npz_checkpoint(checkpoint, flow.params, metadata)
        write_csv(table_dir / "reference_continuous_training_history.csv", history)
        print(f"[repair] trained continuous-endpoint flow in {training_seconds:.1f}s", flush=True)

    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        frozen_mmd_bandwidth = float(data["bandwidth_km"])
    evaluation_times = times[evaluation_indices]
    evaluation_days = days[evaluation_indices]
    sizes = [int(value) for value in repair["reference_bank"]["sizes"]]
    largest = max(sizes)
    bank_path = model_dir / f"reference_bank_eval_{largest}.npz"
    bank_signature = json.dumps({
        "checkpoint_sha256": sha256(checkpoint), "estimator_sha256": sha256(estimator_path),
        "seed": int(repair["reference_bank"]["seed"]), "particles": largest,
        "evaluation_times": evaluation_times.tolist(),
    }, sort_keys=True)
    nodes = initial = None
    if bank_path.exists() and not args.force_bank:
        with np.load(bank_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == bank_signature:
                nodes = np.asarray(cached["nodes_km"], dtype=np.float64)
                initial = np.asarray(cached["initial_km"], dtype=np.float64)
                print(f"[repair] reusing {largest:,}-path continuous bank", flush=True)
    if nodes is None:
        bank_rng = np.random.default_rng(int(repair["reference_bank"]["seed"]))
        initial, mixture_indices = sample_kde(x0, H0, bank_rng, largest)
        nodes = np.empty((len(evaluation_times), largest, 2), dtype=np.float64)
        chunk = int(repair["reference_bank"]["rollout_chunk_size"])
        started = time.perf_counter()
        for start in range(0, largest, chunk):
            stop = min(start + chunk, largest)
            normalized = (initial[start:stop] - center) / scale
            nodes[:, start:stop] = np.asarray(flow.rollout(
                jnp.asarray(normalized), jnp.asarray(evaluation_times)
            )) * scale + center
            if stop == chunk or stop % 50000 == 0 or stop == largest:
                print(f"[repair] rollout {stop:,}/{largest:,}", flush=True)
        rollout_seconds = time.perf_counter() - started
        np.savez_compressed(
            bank_path, nodes_km=nodes, initial_km=initial,
            initial_mixture_indices=mixture_indices, evaluation_indices=evaluation_indices,
            evaluation_times=evaluation_times, evaluation_days=evaluation_days,
            signature=np.asarray(bank_signature), checkpoint_sha256=np.asarray(sha256(checkpoint)),
            estimator_sha256=np.asarray(sha256(estimator_path)), final_test_accessed=np.asarray(False),
        )
        print(f"[repair] rolled {largest:,} unique paths in {rollout_seconds:.1f}s", flush=True)
    unique_rows = []
    for size in sizes:
        unique_rows.append({
            "particle_count": size,
            "unique_initial_exact": int(len(np.unique(initial[:size], axis=0))),
            "unique_day45_exact": int(len(np.unique(nodes[-1, :size], axis=0))),
        })
    write_csv(table_dir / "reference_continuous_unique_paths.csv", unique_rows)
    if any(row["unique_initial_exact"] != row["particle_count"] or row["unique_day45_exact"] != row["particle_count"] for row in unique_rows):
        raise RuntimeError("continuous endpoint sampling did not produce unique paths")

    with np.load(old_model_dir / "reference_bank.npz", allow_pickle=False) as data:
        old_nodes = np.asarray(data["nodes_km"], dtype=np.float64)[evaluation_indices]
    endpoint_rng = np.random.default_rng(int(repair["seed"]) + 300)
    metric_rows = [
        endpoint_metrics(x0, old_nodes[0], "old_discrete", "day0", frozen_mmd_bandwidth, domain, endpoint_rng),
        endpoint_metrics(x1, old_nodes[-1], "old_discrete", "day45", frozen_mmd_bandwidth, domain, endpoint_rng),
        endpoint_metrics(x0, nodes[0], "continuous_kde", "day0", frozen_mmd_bandwidth, domain, endpoint_rng),
        endpoint_metrics(x1, nodes[-1], "continuous_kde", "day45", frozen_mmd_bandwidth, domain, endpoint_rng),
    ]
    acceptance = repair["endpoint_acceptance"]
    repaired_metrics = [row for row in metric_rows if row["model"] == "continuous_kde"]
    for row in metric_rows:
        row["frozen_rbf_bandwidth_km"] = frozen_mmd_bandwidth
        row["accepted"] = bool(
            row["biased_rbf_mmd2"] <= float(acceptance["maximum_biased_rbf_mmd2_each_endpoint"])
            and row["mean_error_over_empirical_rms_spread"] <= float(acceptance["maximum_mean_error_over_empirical_rms_spread"])
            and row["relative_covariance_frobenius_error"] <= float(acceptance["maximum_relative_covariance_frobenius_error"])
            and row["outside_domain_fraction"] <= float(acceptance["maximum_generated_endpoint_outside_domain_fraction"])
        ) if row["model"] == "continuous_kde" else "baseline"
    endpoint_passed = all(row["accepted"] is True for row in repaired_metrics)
    write_csv(table_dir / "reference_endpoint_metrics.csv", metric_rows)
    write_json(table_dir / "reference_endpoint_acceptance.json", {
        "criteria_predeclared_in": repair["_config_path"], "criteria": acceptance,
        "passed": endpoint_passed, "metrics": repaired_metrics,
        "intermediate_positions_used": False, "final_test_artifact_loaded": False,
    })
    if not endpoint_passed:
        write_json(table_dir / "reference_support_repair_summary.json", {
            "endpoint_acceptance_passed": False,
            "endpoint_failure_reason": "generated day-45 outside-domain mass exceeds the predeclared maximum",
            "support_acceptance_passed": None,
            "support_audit_run": False,
            "support_audit_stop_reason": "endpoint acceptance must pass before intermediate cases are examined",
            "case_count_frozen": len(cases),
            "full_bank_sweep_authorized": False,
            "full_bank_sweep_completed": False,
            "unique_paths": unique_rows,
            "final_test_artifact_loaded": False
        })
        raise RuntimeError("continuous reference failed predeclared endpoint-fidelity acceptance; LP audit stopped")
    print("[repair] endpoint-fidelity acceptance passed", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    display_days = [5.0, 10.0, 20.0]
    plot_rng = np.random.default_rng(int(repair["seed"]) + 400)
    for column, day in enumerate(display_days):
        t = int(np.argmin(np.abs(evaluation_days - day)))
        source_index = evaluation_indices[t]
        old_sample = old_nodes[t, plot_rng.choice(old_nodes.shape[1], 1500, replace=False)]
        new_sample = nodes[t, plot_rng.choice(nodes.shape[1], 2500, replace=False)]
        axes[0, column].scatter(old_sample[:, 0], old_sample[:, 1], s=4, alpha=.2, label="old discrete")
        axes[0, column].scatter(inference[:, source_index, 0], inference[:, source_index, 1], s=13, alpha=.65, color="#e45756", label="inference")
        axes[1, column].scatter(new_sample[:, 0], new_sample[:, 1], s=3, alpha=.15, color="#54a24b", label="continuous reference")
        axes[1, column].scatter(validation[:, source_index, 0], validation[:, source_index, 1], s=13, alpha=.65, color="#f58518", label="validation")
        axes[0, column].set_title(f"old reference, day {day:g}")
        axes[1, column].set_title(f"continuous reference, day {day:g}")
        for row in range(2): axes[row, column].set_aspect("equal"); axes[row, column].grid(alpha=.15)
    axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
    fig.supxlabel("x (km)"); fig.supylabel("y (km)"); fig.tight_layout()
    fig.savefig(compare_fig_dir / "old_vs_new_intermediate_support.png", dpi=190); plt.close(fig)

    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
    projection_cfg = IProjectionConfig(
        max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
        newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
        implicit_ridge=float(p["implicit_ridge"]),
    )
    native_acceptance = {
        "accept_residual": float(p["accept_residual"]),
        "min_ess_fraction": float(repair["support_acceptance"]["minimum_native_ess_fraction"]),
        "max_covariance_condition": float(repair["support_acceptance"]["maximum_native_covariance_condition"]),
    }
    audit_rows = []
    started = time.perf_counter()
    for size in sizes:
        print(f"[repair LP] M={size:,}", flush=True)
        for case in cases:
            design_index = int(case["design_index"])
            eval_position = int(np.flatnonzero(evaluation_indices == int(case["source_time_index"]))[0])
            points = nodes[eval_position, :size]
            phi = gaussian_features_numpy(points, centers[design_index], sigma)
            target = np.asarray(case["target_moments"], dtype=np.float64)
            lp = sparse_simplex_lp(phi, target, float(case["frozen_lp_tolerance"]))
            native = native_single(phi, target, projection_cfg, native_acceptance)
            coordinate_margin = float(np.min(np.minimum(
                target - phi.min(axis=0), phi.max(axis=0) - target
            )))
            audit_rows.append({
                "case": case["case"], "design_id": case["design_id"],
                "design_index": design_index, "style": case["style"], "day": case["day"],
                "source_time_index": case["source_time_index"],
                "target_1": target[0], "target_2": target[1],
                "target_3": target[2], "target_4": target[3],
                "particle_count": size, "unique_reference_particles": size,
                "old_minimum_linf_residual": case["old_minimum_linf_residual"],
                "coordinate_support_margin": coordinate_margin,
                **lp, **native,
            })
        print(f"[repair LP] completed M={size:,}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    continuous_table = table_dir / "reference_support_lp_continuous.csv"
    write_csv(continuous_table, audit_rows)
    largest_rows = [row for row in audit_rows if row["particle_count"] == largest]
    lp_feasible = sum(
        row["minimum_linf_residual"] <= float(p["accept_residual"]) for row in largest_rows
    )
    native_healthy = sum(row["native_healthy"] for row in largest_rows)
    support_cfg = repair["support_acceptance"]
    support_passed = bool(
        lp_feasible >= int(support_cfg["minimum_lp_feasible_cases_at_largest_bank"])
        and native_healthy >= int(support_cfg["minimum_native_healthy_cases_at_largest_bank"])
    )
    write_json(table_dir / "reference_support_repair_summary.json", {
        "endpoint_acceptance_passed": endpoint_passed,
        "support_acceptance_passed": support_passed,
        "largest_particle_count": largest, "lp_feasible_cases": lp_feasible,
        "native_healthy_cases": native_healthy, "case_count": len(cases),
        "thresholds": support_cfg, "old_lp_feasible_cases": 4,
        "unique_paths": unique_rows, "final_test_artifact_loaded": False,
        "full_bank_sweep_authorized": support_passed,
        "full_bank_sweep_completed": False,
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for case in range(len(cases)):
        rows = [row for row in audit_rows if row["case"] == case]
        ax.plot(
            [row["particle_count"] for row in rows],
            [max(float(row["minimum_linf_residual"]), 1e-12) for row in rows],
            marker="o", ms=3, alpha=.65,
        )
    ax.axhline(float(p["accept_residual"]), color="black", ls="--", label="frozen feasibility tolerance")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=.2); ax.legend()
    ax.set_xlabel("genuinely unique reference particles M")
    ax.set_ylabel("minimum simplex LP residual (L-infinity)")
    ax.set_title("Same 20 cases after continuous-endpoint reference repair")
    fig.tight_layout(); fig.savefig(lp_fig_dir / "reference_support_lp_continuous.png", dpi=190); plt.close(fig)
    print(
        f"[repair] support audit: LP feasible={lp_feasible}/20, "
        f"native healthy={native_healthy}/20, passed={support_passed}", flush=True,
    )


if __name__ == "__main__":
    main()
