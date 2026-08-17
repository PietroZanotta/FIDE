#!/usr/bin/env python3
"""Dense moment splines and tangent-action readiness for Phase-2F finalists."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from phase2_common import gaussian_features_numpy, load_phase2_config, resolve, sha256, write_csv, write_json  # noqa: E402
from repair_reference_support import log_kde_density  # noqa: E402
from run_continuous_grid_iprojection import grid_points, projection_config, read_csv  # noqa: E402
from run_phase2f_adaptive_iprojection import summarize_at_lambda, warm_start_trust_region  # noqa: E402
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor  # noqa: E402
from mfsi.projection_tesseract import solve_i_projection_trajectory_tesseract_forward  # noqa: E402
from mfsi.reference import DomainPreservingReferenceFlow  # noqa: E402
from mfsi.reference_density import backward_latent_with_log_density_correction, logistic_log_abs_det_jacobian  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None) -> dict[str, Any]:
    source = Path(path) if path else SCRIPT_DIR.parent / "configs/mfsi_phase2f.json"
    with source.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def compute_moments(
    inference: np.ndarray, centers: np.ndarray, sigma: float
) -> np.ndarray:
    output = np.empty((len(centers), inference.shape[1], centers.shape[1]), dtype=np.float64)
    for start in range(0, len(centers), 8):
        stop = min(start + 8, len(centers))
        delta = inference[None, :, :, None, :] - centers[start:stop, None, None, :, :]
        output[start:stop] = np.exp(
            -0.5 * np.sum(delta * delta, axis=-1) / sigma**2
        ).mean(axis=1)
    return output


def fit_splines(
    cfg: dict, phase2: dict, analysis: Path, processed: Path,
    inference: np.ndarray, times: np.ndarray, centers: np.ndarray, sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    table_dir = analysis / "tables"
    near_rows = read_csv(table_dir / "near_optimal_set.csv")
    design_indices = np.asarray([int(row["design_index"]) for row in near_rows], dtype=int)
    raw = compute_moments(inference, centers[design_indices], sigma)
    spline_cfg = cfg["moment_spline"]
    candidates = [float(value) for value in spline_cfg["smoothing_candidates"]]
    gcv_rows: list[dict] = []
    chosen_score = math.inf
    chosen_smoothing = candidates[0]
    n = len(times)
    for smoothing in candidates:
        reconstructor = AnchoredCubicSplineReconstructor(
            times, times,
            AnchoredCubicSplineConfig(
                internal_knots=int(spline_cfg["internal_knots"]), smoothing=smoothing,
                ridge_rel=float(spline_cfg["ridge_relative"]),
                roughness_quadrature_order=int(spline_cfg["roughness_quadrature_order"]),
            ),
        )
        gram = np.asarray(reconstructor.B_obs.T @ reconstructor.B_obs)
        normal = gram + smoothing * np.asarray(reconstructor.roughness_matrix)
        ridge_scale = max(float(np.trace(gram)) / max(reconstructor.n_basis, 1), 1.0)
        normal += float(spline_cfg["ridge_relative"]) * ridge_scale * np.eye(reconstructor.n_basis)
        effective_df = float(np.trace(np.linalg.solve(normal, gram))) + 2.0
        normalized_gcv = []
        rss_values = []
        roughness_values = []
        for values in raw:
            fit = reconstructor.reconstruct(values, values[0], values[-1])
            fitted = np.asarray(fit.c)
            rss_component = np.sum((values - fitted) ** 2, axis=0)
            scale = np.maximum(np.sum((values - values.mean(axis=0)) ** 2, axis=0), 1e-14)
            normalized_gcv.extend(
                (rss_component / scale / max((1.0 - effective_df / n) ** 2, 1e-14)).tolist()
            )
            rss_values.append(float(fit.residual_sum_squares))
            roughness_values.append(float(fit.roughness))
        score = float(np.mean(normalized_gcv))
        gcv_rows.append({
            "smoothing": smoothing, "internal_knots": int(spline_cfg["internal_knots"]),
            "effective_degrees_of_freedom_per_component": effective_df,
            "aggregate_normalized_gcv": score,
            "mean_residual_sum_squares": float(np.mean(rss_values)),
            "mean_roughness": float(np.mean(roughness_values)),
        })
        if score < chosen_score:
            chosen_score, chosen_smoothing = score, smoothing
    write_csv(table_dir / "dense_moment_spline_gcv.csv", gcv_rows)

    reconstructor = AnchoredCubicSplineReconstructor(
        times, times,
        AnchoredCubicSplineConfig(
            internal_knots=int(spline_cfg["internal_knots"]), smoothing=chosen_smoothing,
            ridge_rel=float(spline_cfg["ridge_relative"]),
            roughness_quadrature_order=int(spline_cfg["roughness_quadrature_order"]),
        ),
    )
    smoothed = np.empty_like(raw)
    derivative = np.empty_like(raw)
    diagnostic_rows: list[dict] = []
    for local, design in enumerate(design_indices):
        fit = reconstructor.reconstruct(raw[local], raw[local, 0], raw[local, -1])
        smoothed[local] = np.asarray(fit.c)
        derivative[local] = np.asarray(fit.c_dot)
        diagnostic_rows.append({
            "design_index": int(design), "design_id": near_rows[local]["design_id"],
            "smoothing": chosen_smoothing,
            "residual_sum_squares": float(fit.residual_sum_squares),
            "roughness": float(fit.roughness),
            "maximum_absolute_fit_residual": float(np.max(np.abs(smoothed[local] - raw[local]))),
            "minimum_smoothed_moment": float(smoothed[local].min()),
            "maximum_smoothed_moment": float(smoothed[local].max()),
            "maximum_absolute_derivative_per_normalized_time": float(np.max(np.abs(derivative[local]))),
        })
    write_csv(table_dir / "dense_moment_spline_diagnostics.csv", diagnostic_rows)
    np.savez_compressed(
        processed / "phase2f_dense_moments.npz",
        design_indices=design_indices, normalized_times=times,
        raw_moments=raw, smoothed_moments=smoothed,
        moment_derivative=derivative, smoothing=np.asarray(chosen_smoothing),
        final_test_accessed=np.asarray(False),
    )
    write_json(table_dir / "phase2f_spline_freeze.json", {
        "smoothing": chosen_smoothing,
        "selection": spline_cfg["selection"],
        "internal_knots": int(spline_cfg["internal_knots"]),
        "near_optimal_layout_count": len(design_indices),
        "action_values_inspected_before_freeze": False,
        "final_test_artifact_loaded": False,
        "risk_freeze_sha256": sha256(table_dir / "phase2f_risk_freeze.json"),
    })

    figure_dir = analysis / "figures/moment_splines"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    for observable, axis in enumerate(axes.ravel()):
        axis.plot(times * 45.0, raw[0, :, observable], color="#9ecae1", linewidth=1, label="raw inference moment")
        axis.plot(times * 45.0, smoothed[0, :, observable], color="#08519c", linewidth=1.5, label="frozen spline")
        axis.set_title(f"best layout sensor {observable + 1}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.supxlabel("day")
    fig.supylabel("Gaussian sensor moment")
    fig.savefig(figure_dir / "best_layout_moment_splines.png", dpi=180)
    plt.close(fig)
    return design_indices, smoothed, derivative, chosen_smoothing


def dense_reference(
    cfg: dict, phase2: dict, processed: Path, times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    grid_cfg_path = resolve(cfg["continuous_grid_config"])
    with grid_cfg_path.open(encoding="utf-8") as handle:
        grid_cfg = json.load(handle)
    bounds = np.asarray(phase2["domain"]["final_box_km"], dtype=np.float64)
    nx, ny = 256, 136
    points, dx, dy = grid_points(bounds, nx, ny)
    checkpoint = resolve(cfg["reference_checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    flow = DomainPreservingReferenceFlow.from_npz(
        checkpoint,
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    with np.load(resolve(cfg["conditioned_endpoint_estimator"]), allow_pickle=False) as data:
        atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
        bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
    analysis = resolve(phase2["analysis_dir"])
    z0 = float(next(
        row for row in read_csv(analysis / "tables/conditioned_kde_normalization.csv")
        if row["endpoint"] == "day0"
    )["Z_hat"])
    bounds_array = np.asarray(flow.bounds, dtype=np.float64)
    chunk = int(grid_cfg["grid"]["density_chunk_size"])
    cache_dir = SCRIPT_DIR.parent / "cache/projected_laws/dense_reference_256x136"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_base_all = np.empty((len(times), len(points)), dtype=np.float64)
    velocity_all = np.empty((len(times), len(points), 2), dtype=np.float64)
    zt = flow.to_latent(jnp.asarray(points))
    velocity_fn = jax.jit(lambda time: flow.velocity(jnp.asarray(points), time))

    @lru_cache(maxsize=None)
    def backward_fn(steps: int):
        return jax.jit(lambda z, time: backward_latent_with_log_density_correction(
            flow.params, z, time, steps=steps
        ))

    for source_index, time_value in enumerate(times):
        cache = cache_dir / f"reference_{checkpoint_hash[:12]}_t{source_index:03d}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                log_base_all[source_index] = data["log_base_mass"]
                velocity_all[source_index] = data["velocity"]
            continue
        steps = max(1, int(math.ceil(source_index / 5.0)))
        log_density = np.empty(len(points), dtype=np.float64)
        backward = backward_fn(steps)
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            local_zt = zt[start:stop]
            z_initial, correction = backward(local_zt, jnp.asarray(time_value))
            x_initial = np.asarray(flow.to_physical(z_initial))
            log_initial = log_kde_density(x_initial, atoms, bandwidth, chunk=chunk) - math.log(z0)
            log_density[start:stop] = (
                log_initial
                + np.asarray(logistic_log_abs_det_jacobian(z_initial, bounds_array))
                + np.asarray(correction)
                - np.asarray(logistic_log_abs_det_jacobian(local_zt, bounds_array))
            )
        log_raw = log_density + math.log(dx * dy)
        log_base = log_raw - logsumexp(log_raw)
        velocity = np.asarray(velocity_fn(jnp.asarray(time_value)))
        log_base_all[source_index] = log_base
        velocity_all[source_index] = velocity
        np.savez_compressed(
            cache, log_base_mass=log_base, velocity=velocity,
            source_time_index=np.asarray(source_index), normalized_time=np.asarray(time_value),
            rk4_steps=np.asarray(steps), final_test_accessed=np.asarray(False),
        )
        if source_index == 0 or source_index % 10 == 0 or source_index == len(times) - 1:
            print(f"[phase2f tangent] dense reference {source_index + 1}/{len(times)}", flush=True)
    return points, log_base_all, velocity_all, dx, dy


def tangent_readiness(
    cfg: dict, phase2: dict, analysis: Path, processed: Path,
    times: np.ndarray, centers_all: np.ndarray, sigma: float,
    design_indices: np.ndarray, targets: np.ndarray, target_dot: np.ndarray,
) -> None:
    table_dir = analysis / "tables"
    points, log_base_all, velocity_all, dx, dy = dense_reference(cfg, phase2, processed, times)
    solver = projection_config(cfg["adaptive_iprojection"]["solver"])
    fallback = cfg["adaptive_iprojection"]["warm_start_trust_region_fallback"]
    readiness = cfg["tangent_readiness"]
    design_rows: list[dict] = []
    time_rows: list[dict] = []
    cache_dir = SCRIPT_DIR.parent / "cache/projected_laws/tangent_256x136"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for local, design in enumerate(design_indices):
        cache = cache_dir / f"design_{int(design):06d}.npz"
        phi = gaussian_features_numpy(points, centers_all[design], sigma)
        delta = points[:, None, :] - centers_all[design][None]
        gradient = -(delta / sigma**2) * phi[:, :, None]
        phi_all = np.ascontiguousarray(np.broadcast_to(phi, (len(times), *phi.shape)))
        native = solve_i_projection_trajectory_tesseract_forward(
            phi_all, log_base_all, targets[local:local + 1], solver
        )
        lambdas = np.asarray(native["lambda_values"][0], dtype=np.float64)
        projection_ok = np.zeros(len(times), dtype=bool)
        tangent_density = np.full(len(times), np.nan, dtype=np.float64)
        lambda_dot = np.full_like(lambdas, np.nan)
        warm = np.zeros(4, dtype=np.float64)
        ranks = []
        design_time_rows: list[dict] = []
        for time_index in range(len(times)):
            result = summarize_at_lambda(
                phi, log_base_all[time_index], targets[local, time_index],
                lambdas[time_index], iterations=int(native["iterations"][0, time_index]),
                solver_name="native_dense_trajectory",
                convergence_tolerance=float(readiness["maximum_moment_residual"]),
                ridge=solver.newton_ridge,
            )
            if not result["converged"]:
                result = warm_start_trust_region(
                    phi, log_base_all[time_index], targets[local, time_index], warm,
                    fallback,
                    acceptance_tolerance=float(readiness["maximum_moment_residual"]),
                    ridge=solver.newton_ridge,
                )
                lambdas[time_index] = [result[f"lambda_{j}"] for j in range(4)]
            warm = lambdas[time_index]
            weights = np.asarray(result["projected_weights"])
            projection_ok[time_index] = result["converged"]
            moment_velocity = np.einsum(
                "n,nmd,nd->nm", weights, gradient, velocity_all[time_index]
            )
            expected_m = moment_velocity.sum(axis=0)
            gram = np.einsum("n,nmd,nkd->mk", weights, gradient, gradient)
            gram_eig, gram_vec = np.linalg.eigh(gram)
            gram_threshold = float(readiness["gram_relative_rank_tolerance"]) * max(gram_eig[-1], 1e-300)
            retained = gram_eig > gram_threshold
            rank = int(retained.sum())
            ranks.append(rank)
            r = expected_m - target_dot[local, time_index]
            projected_r = gram_vec[:, retained] @ (gram_vec[:, retained].T @ r) if rank else np.zeros_like(r)
            compatibility = float(np.linalg.norm(r - projected_r) / max(np.linalg.norm(r), 1e-14))
            if rank and compatibility <= float(readiness["gram_compatibility_relative_tolerance"]):
                tangent_density[time_index] = float(
                    np.sum((gram_vec[:, retained].T @ r) ** 2 / gram_eig[retained])
                )
            achieved = weights @ phi
            centered = phi - achieved
            covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
            cov_eig = np.linalg.eigvalsh(covariance)
            cov_condition = float(cov_eig[-1] / max(cov_eig[0], 1e-300))
            lambda_m = moment_velocity @ lambdas[time_index]
            cov_phi_lambda_m = np.einsum(
                "n,ni,n->i", weights, centered, lambda_m - weights @ lambda_m
            )
            lambda_rhs = target_dot[local, time_index] - expected_m - cov_phi_lambda_m
            covariance_ready = bool(
                cov_eig[0] >= float(readiness["covariance_minimum_eigenvalue"])
                and cov_condition <= float(readiness["covariance_maximum_condition"])
            )
            if covariance_ready:
                lambda_dot[time_index] = np.linalg.solve(covariance, lambda_rhs)
            design_time_rows.append({
                "design_index": int(design), "design_id": f"design_{int(design):06d}",
                "source_time_index": time_index, "day": float(times[time_index] * 45.0),
                "projection_converged": bool(result["converged"]),
                "moment_residual": result["verified_l2_residual"],
                "lambda_norm": result["lambda_norm"],
                "covariance_min_eigenvalue": float(cov_eig[0]),
                "covariance_condition": cov_condition,
                "covariance_ready_for_lambda_dot": covariance_ready,
                "gram_rank": rank, "gram_minimum_retained_eigenvalue": float(gram_eig[retained].min()) if rank else math.nan,
                "gram_compatibility_relative_residual": compatibility,
                "tangent_action_density": tangent_density[time_index],
                "lambda_dot_norm": float(np.linalg.norm(lambda_dot[time_index])) if covariance_ready else math.nan,
            })
        time_rows.extend(design_time_rows)
        all_projection = bool(projection_ok.all())
        all_compatibility = all(
            float(row["gram_compatibility_relative_residual"])
            <= float(readiness["gram_compatibility_relative_tolerance"])
            for row in design_time_rows
        )
        all_covariance = all(row["covariance_ready_for_lambda_dot"] for row in design_time_rows)
        rank_stable = len(set(ranks)) == 1
        tangent_finite = bool(np.isfinite(tangent_density).all())
        tangent_total = float(np.trapezoid(tangent_density, times)) if tangent_finite else math.nan
        design_rows.append({
            "design_index": int(design), "design_id": f"design_{int(design):06d}",
            "dense_projection_ready": all_projection,
            "gram_compatibility_ready": all_compatibility,
            "gram_rank_constant": rank_stable,
            "gram_rank_values": ";".join(str(value) for value in sorted(set(ranks))),
            "lambda_dot_covariance_ready": all_covariance,
            "tangent_action_finite": tangent_finite,
            "tangent_action": tangent_total,
            "tangent_action_ready": bool(
                all_projection and all_compatibility and rank_stable and all_covariance and tangent_finite
            ),
            "maximum_moment_residual": max(float(row["moment_residual"]) for row in design_time_rows),
            "maximum_covariance_condition": max(float(row["covariance_condition"]) for row in design_time_rows),
            "minimum_covariance_eigenvalue": min(float(row["covariance_min_eigenvalue"]) for row in design_time_rows),
            "maximum_gram_compatibility_residual": max(float(row["gram_compatibility_relative_residual"]) for row in design_time_rows),
        })
        np.savez_compressed(
            cache, lambda_value=lambdas, lambda_dot=lambda_dot,
            tangent_action_density=tangent_density, projection_converged=projection_ok,
            target_moments=targets[local], target_derivative=target_dot[local],
            final_test_accessed=np.asarray(False),
        )
        print(
            f"[phase2f tangent] {local + 1}/{len(design_indices)} design={int(design)} "
            f"ready={design_rows[-1]['tangent_action_ready']}", flush=True,
        )
    write_csv(table_dir / "tangent_action_readiness.csv", design_rows)
    write_csv(table_dir / "tangent_action_time.csv", time_rows)
    write_csv(table_dir / "tangent_action.csv", [{
        "design_index": row["design_index"], "design_id": row["design_id"],
        "tangent_action": row["tangent_action"], "valid": row["tangent_action_ready"],
    } for row in design_rows])
    write_json(table_dir / "phase2f_tangent_summary.json", {
        "near_optimal_layout_count": len(design_rows),
        "dense_projection_ready_count": sum(row["dense_projection_ready"] for row in design_rows),
        "tangent_action_ready_count": sum(row["tangent_action_ready"] for row in design_rows),
        "lambda_dot_covariance_ready_count": sum(row["lambda_dot_covariance_ready"] for row in design_rows),
        "final_test_artifact_loaded": False,
        "spline_freeze_sha256": sha256(table_dir / "phase2f_spline_freeze.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("spline", "density", "tangent", "all"), default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["phase2_config"]))
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        positions = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
    inference = positions[split == "inference"]
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
    dense_path = processed / "phase2f_dense_moments.npz"
    if args.stage in ("spline", "all") or not dense_path.exists():
        design_indices, targets, target_dot, _ = fit_splines(
            cfg, phase2, analysis, processed, inference, times, centers, sigma
        )
    else:
        with np.load(dense_path, allow_pickle=False) as data:
            design_indices = np.asarray(data["design_indices"], dtype=int)
            targets = np.asarray(data["smoothed_moments"], dtype=np.float64)
            target_dot = np.asarray(data["moment_derivative"], dtype=np.float64)
    if args.stage == "spline":
        return
    if args.stage == "density":
        dense_reference(cfg, phase2, processed, times)
        return
    tangent_readiness(
        cfg, phase2, analysis, processed, times, centers, sigma,
        design_indices, targets, target_dot,
    )


if __name__ == "__main__":
    main()
