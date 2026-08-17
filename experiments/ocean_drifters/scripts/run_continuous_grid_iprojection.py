#!/usr/bin/env python3
"""Phase 2E: deterministic-grid audit of the frozen continuous reference law."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from phase2_common import (  # noqa: E402
    gaussian_features_numpy,
    load_phase2_config,
    resolve,
    sha256,
    write_csv,
    write_json,
)
from repair_reference_support import log_kde_density, sparse_simplex_lp  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import (  # noqa: E402
    is_tesseract_iprojection_available,
    solve_i_projection_trajectory_tesseract_forward,
)
from mfsi.reference import DomainPreservingReferenceFlow  # noqa: E402
from mfsi.reference_density import (  # noqa: E402
    backward_latent_with_log_density_correction,
    logistic_log_abs_det_jacobian,
)

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None) -> dict[str, Any]:
    source = Path(path) if path else SCRIPT_DIR.parent / "configs/continuous_grid_iprojection.json"
    with source.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grid_points(bounds: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, float, float]:
    dx = float(bounds[1] - bounds[0]) / nx
    dy = float(bounds[3] - bounds[2]) / ny
    x = bounds[0] + (np.arange(nx) + 0.5) * dx
    y = bounds[2] + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel())), dx, dy


def projection_config(cfg: dict[str, Any]) -> IProjectionConfig:
    return IProjectionConfig(**{
        key: cfg[key] for key in asdict(IProjectionConfig()).keys() if key in cfg
    })


def containment_rows(development: Path, bounds: np.ndarray, evaluation_indices: np.ndarray) -> list[dict]:
    with np.load(development, allow_pickle=False) as data:
        x = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"])
    rows: list[dict] = []
    for role in ("inference", "validation"):
        points = x[split == role][:, evaluation_indices].reshape(-1, 2)
        strict = (
            (points[:, 0] > bounds[0]) & (points[:, 0] < bounds[1])
            & (points[:, 1] > bounds[2]) & (points[:, 1] < bounds[3])
        )
        outside = (
            (points[:, 0] < bounds[0]) | (points[:, 0] > bounds[1])
            | (points[:, 1] < bounds[2]) | (points[:, 1] > bounds[3])
        )
        boundary = ~(strict | outside)
        rows.append({
            "role": role,
            "trajectory_count": int(np.sum(split == role)),
            "evaluation_time_count": len(evaluation_indices),
            "observation_count": len(points),
            "strictly_inside_count": int(strict.sum()),
            "strictly_inside_fraction": float(strict.mean()),
            "outside_count": int(outside.sum()),
            "boundary_count": int(boundary.sum()),
            "minimum_distance_to_xmin_km": float(np.min(points[:, 0] - bounds[0])),
            "minimum_distance_to_xmax_km": float(np.min(bounds[1] - points[:, 0])),
            "minimum_distance_to_ymin_km": float(np.min(points[:, 1] - bounds[2])),
            "minimum_distance_to_ymax_km": float(np.min(bounds[3] - points[:, 1])),
        })
    return rows


def density_cache_path(cache_dir: Path, checkpoint_hash: str, source_index: int, nx: int, ny: int) -> Path:
    return cache_dir / f"density_{checkpoint_hash[:12]}_t{source_index:03d}_{nx}x{ny}.npz"


def evaluate_density_grid(
    *,
    flow: DomainPreservingReferenceFlow,
    points: np.ndarray,
    time_value: float,
    source_index: int,
    nx: int,
    ny: int,
    steps: int,
    chunk_size: int,
    atoms: np.ndarray,
    bandwidth: np.ndarray,
    conditioning_normalizer: float,
    cell_area: float,
    cache_dir: Path,
    checkpoint_hash: str,
    force: bool,
) -> dict[str, Any]:
    cache = density_cache_path(cache_dir, checkpoint_hash, source_index, nx, ny)
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as data:
            if (
                int(data["steps"]) == steps
                and float(data["time_value"]) == time_value
                and str(data["checkpoint_sha256"].item()) == checkpoint_hash
            ):
                return {key: np.asarray(data[key]) for key in data.files}

    bounds = np.asarray(flow.bounds, dtype=np.float64)
    log_density = np.empty(len(points), dtype=np.float64)
    started = time.perf_counter()

    # A separate compiled function is intentional: each frozen time has a static
    # RK4 step count and can then stream arbitrary chunks without retracing.
    backward = jax.jit(lambda z: backward_latent_with_log_density_correction(
        flow.params, z, jnp.asarray(time_value, dtype=jnp.float64), steps=steps
    ))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        zt = flow.to_latent(jnp.asarray(points[start:stop]))
        z0, correction = backward(zt)
        x0 = np.asarray(flow.to_physical(z0), dtype=np.float64)
        log_p0 = log_kde_density(x0, atoms, bandwidth, chunk=chunk_size) - math.log(
            conditioning_normalizer
        )
        log_qz0 = log_p0 + np.asarray(logistic_log_abs_det_jacobian(z0, bounds))
        log_qzt = log_qz0 + np.asarray(correction)
        log_density[start:stop] = log_qzt - np.asarray(
            logistic_log_abs_det_jacobian(zt, bounds)
        )
    if not np.isfinite(log_density).all():
        raise RuntimeError(f"nonfinite continuous density at time index {source_index}")
    log_raw_mass = log_density + math.log(cell_area)
    log_normalization = float(logsumexp(log_raw_mass))
    log_base_mass = log_raw_mass - log_normalization
    result = {
        "log_density": log_density,
        "log_base_mass": log_base_mass,
        "log_normalization": np.asarray(log_normalization),
        "normalization": np.asarray(math.exp(log_normalization)),
        "steps": np.asarray(steps),
        "time_value": np.asarray(time_value),
        "source_index": np.asarray(source_index),
        "nx": np.asarray(nx),
        "ny": np.asarray(ny),
        "checkpoint_sha256": np.asarray(checkpoint_hash),
        "elapsed_seconds": np.asarray(time.perf_counter() - started),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def weighted_summary(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ points
    centered = points - mean
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, covariance


def projection_summary(
    phi: np.ndarray,
    log_base: np.ndarray,
    target: np.ndarray,
    solver_cfg: IProjectionConfig,
) -> dict[str, Any]:
    native = solve_i_projection_trajectory_tesseract_forward(
        phi[None], log_base[None], target[None, None], solver_cfg
    )
    lam = np.asarray(native["lambda_values"][0, 0], dtype=np.float64)
    log_ratio_unnormalized = phi @ lam
    log_partition = float(logsumexp(log_base + log_ratio_unnormalized))
    log_ratio = log_ratio_unnormalized - log_partition
    log_weight = log_base + log_ratio
    weights = np.exp(log_weight)
    achieved = weights @ phi
    centered = phi - achieved
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    log_second_moment = float(logsumexp(log_base + 2.0 * log_ratio))
    intrinsic_ess = float(math.exp(-log_second_moment)) if log_second_moment < 710 else 0.0
    return {
        "converged": bool(native["converged"][0, 0]),
        "iterations": int(native["iterations"][0, 0]),
        "reported_residual": float(native["residual_norm"][0, 0]),
        "verified_l2_residual": float(np.linalg.norm(achieved - target)),
        "verified_linf_residual": float(np.max(np.abs(achieved - target))),
        "lambda_0": float(lam[0]), "lambda_1": float(lam[1]),
        "lambda_2": float(lam[2]), "lambda_3": float(lam[3]),
        "lambda_norm": float(np.linalg.norm(lam)),
        "kl_divergence": float(weights @ log_ratio),
        "intrinsic_ess": intrinsic_ess,
        "log10_intrinsic_ess": float(-log_second_moment / math.log(10.0)),
        "covariance_min_eigenvalue": float(eigenvalues[0]),
        "covariance_max_eigenvalue": float(eigenvalues[-1]),
        "covariance_condition_regularized": float(
            (eigenvalues[-1] + solver_cfg.newton_ridge)
            / max(eigenvalues[0] + solver_cfg.newton_ridge, 1e-300)
        ),
        "maximum_quadrature_weight": float(weights.max()),
    }


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(1.0, abs(new))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("geometry", "density", "projection", "audit"), default="audit")
    parser.add_argument("--force-density", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["base_phase2_config"]))
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    table_dir = analysis / "tables"
    cache_dir = SCRIPT_DIR.parent / "cache/reference_density_grids"
    bounds = np.asarray(phase2["domain"]["final_box_km"], dtype=np.float64)
    checkpoint = resolve(cfg["reference_checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    flow = DomainPreservingReferenceFlow.from_npz(
        checkpoint,
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    with (table_dir / "reference_support_cases.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    current_particle_rows = read_csv(table_dir / "reference_support_lp_domain_preserving.csv")
    current_particle_count = max(int(row["particle_count"]) for row in current_particle_rows)
    current_particle_feasible = {
        int(row["case"]): row["exact_lp_success"] == "True"
        for row in current_particle_rows if int(row["particle_count"]) == current_particle_count
    }
    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        evaluation_days = np.asarray(data["evaluation_days"], dtype=np.float64)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        normalized_times = np.asarray(data["normalized_time"], dtype=np.float64)
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        sigma = float(data["sigma_km"])
    with np.load(resolve(cfg["conditioned_endpoint_estimator"]), allow_pickle=False) as data:
        atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
        bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
        if bool(data["final_test_accessed"]):
            raise RuntimeError("conditioned endpoint estimator reports final-test access")
    normalization_rows = read_csv(table_dir / "conditioned_kde_normalization.csv")
    z0 = float(next(row for row in normalization_rows if row["endpoint"] == "day0")["Z_hat"])
    if z0 <= 0.0:
        raise RuntimeError("invalid initial KDE domain-conditioning normalizer")

    containment = containment_rows(processed / "development_270.npz", bounds, evaluation_indices)
    write_csv(table_dir / "continuous_reference_target_domain_containment.csv", containment)
    if containment[0]["outside_count"] or containment[0]["boundary_count"]:
        raise RuntimeError("inference target positions are not strictly inside the frozen domain")

    geometry_rows: list[dict] = []
    grids: dict[tuple[int, int], tuple[np.ndarray, float, float]] = {}
    for nx, ny in cfg["grid"]["resolutions"]:
        points, dx, dy = grid_points(bounds, int(nx), int(ny))
        grids[(int(nx), int(ny))] = (points, dx, dy)
        for case in cases:
            phi = gaussian_features_numpy(points, centers[int(case["design_index"])], sigma)
            result = sparse_simplex_lp(phi, np.asarray(case["target_moments"]), float(case["frozen_lp_tolerance"]))
            geometry_rows.append({
                "case": int(case["case"]), "design_id": case["design_id"],
                "design_index": int(case["design_index"]), "style": case["style"],
                "day": float(case["day"]), "source_time_index": int(case["source_time_index"]),
                "grid_nx": int(nx), "grid_ny": int(ny), "grid_point_count": len(points),
                "dx_km": dx, "dy_km": dy, **result,
            })
        print(f"[continuous grid] domain hull {nx}x{ny} complete", flush=True)
    write_csv(table_dir / "domain_moment_hull_20cases.csv", geometry_rows)
    if args.stage == "geometry":
        return

    with np.load(resolve(cfg["reference_particle_bank"]), allow_pickle=False) as data:
        particle_nodes = np.asarray(data["nodes_km"], dtype=np.float64)
        bank_indices = np.asarray(data["evaluation_indices"], dtype=int)
        bank_times = np.asarray(data["evaluation_times"], dtype=np.float64)
        if bool(data["final_test_accessed"]):
            raise RuntimeError("reference bank reports final-test access")

    distinct_source_indices = sorted({int(case["source_time_index"]) for case in cases})
    density_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    density_rows: list[dict] = []
    comparison_rows: list[dict] = []
    source_stride = int(cfg["grid"]["evaluation_stride_source_steps"])
    substeps = int(cfg["grid"]["flow_rk4_substeps_per_evaluation_interval"])
    chunk_size = int(cfg["grid"]["density_chunk_size"])

    for source_index in distinct_source_indices:
        eval_position = int(np.flatnonzero(bank_indices == source_index)[0])
        time_value = float(bank_times[eval_position])
        day = float(evaluation_days[np.flatnonzero(evaluation_indices == source_index)[0]])
        particle = particle_nodes[eval_position]
        particle_mean = particle.mean(axis=0)
        particle_cov = np.cov(particle, rowvar=False, ddof=0)
        designs_at_time = sorted({int(c["design_index"]) for c in cases if int(c["source_time_index"]) == source_index})
        particle_sensor = {
            design: gaussian_features_numpy(particle, centers[design], sigma).mean(axis=0)
            for design in designs_at_time
        }
        steps = max(1, int(round(source_index / source_stride)) * substeps)
        for nx, ny in cfg["grid"]["resolutions"]:
            nx, ny = int(nx), int(ny)
            points, dx, dy = grids[(nx, ny)]
            density = evaluate_density_grid(
                flow=flow, points=points, time_value=time_value, source_index=source_index,
                nx=nx, ny=ny, steps=steps, chunk_size=chunk_size, atoms=atoms,
                bandwidth=bandwidth, conditioning_normalizer=z0, cell_area=dx * dy,
                cache_dir=cache_dir, checkpoint_hash=checkpoint_hash, force=args.force_density,
            )
            density_by_key[(source_index, nx, ny)] = density
            log_density = np.asarray(density["log_density"])
            weights = np.exp(np.asarray(density["log_base_mass"]))
            grid_mean, grid_cov = weighted_summary(points, weights)
            mean_error = float(np.linalg.norm(grid_mean - particle_mean))
            cov_error = float(np.linalg.norm(grid_cov - particle_cov) / np.linalg.norm(particle_cov))
            sensor_errors = []
            for design in designs_at_time:
                grid_sensor = weights @ gaussian_features_numpy(points, centers[design], sigma)
                sensor_errors.extend(np.abs(grid_sensor - particle_sensor[design]).tolist())
            density_rows.append({
                "day": day, "source_time_index": source_index, "normalized_time": time_value,
                "grid_nx": nx, "grid_ny": ny, "grid_point_count": len(points),
                "dx_km": dx, "dy_km": dy,
                "raw_normalization": float(density["normalization"]),
                "absolute_normalization_error": abs(float(density["normalization"]) - 1.0),
                "minimum_log_density": float(log_density.min()),
                "maximum_log_density": float(log_density.max()),
                "log10_density_dynamic_range": float((log_density.max() - log_density.min()) / math.log(10.0)),
                "all_density_values_finite_and_positive_in_log_space": bool(np.isfinite(log_density).all()),
                "backward_rk4_steps": steps,
                "density_evaluation_seconds": float(density["elapsed_seconds"]),
            })
            comparison_rows.append({
                "day": day, "source_time_index": source_index,
                "grid_nx": nx, "grid_ny": ny,
                "grid_mean_x_km": float(grid_mean[0]), "grid_mean_y_km": float(grid_mean[1]),
                "particle_mean_x_km": float(particle_mean[0]), "particle_mean_y_km": float(particle_mean[1]),
                "mean_error_km": mean_error,
                "relative_covariance_frobenius_error": cov_error,
                "maximum_sensor_expectation_error": float(max(sensor_errors, default=0.0)),
            })
            print(f"[continuous grid] density day {day:g} {nx}x{ny}: Z={float(density['normalization']):.6g}", flush=True)
    write_csv(table_dir / "reference_grid_normalization.csv", density_rows)
    write_csv(table_dir / "reference_grid_particle_comparison.csv", comparison_rows)
    if args.stage == "density":
        return
    if not is_tesseract_iprojection_available():
        raise RuntimeError("native I-projection Tesseract is required")

    solver_cfg = projection_config(phase2["projection"])
    projection_rows: list[dict] = []
    for case in cases:
        source_index = int(case["source_time_index"])
        target = np.asarray(case["target_moments"], dtype=np.float64)
        design_index = int(case["design_index"])
        for nx, ny in cfg["grid"]["resolutions"]:
            nx, ny = int(nx), int(ny)
            points = grids[(nx, ny)][0]
            phi = gaussian_features_numpy(points, centers[design_index], sigma)
            result = projection_summary(
                phi, np.asarray(density_by_key[(source_index, nx, ny)]["log_base_mass"]),
                target, solver_cfg,
            )
            projection_rows.append({
                "case": int(case["case"]), "design_id": case["design_id"],
                "design_index": design_index, "style": case["style"],
                "day": float(case["day"]), "source_time_index": source_index,
                "grid_nx": nx, "grid_ny": ny, "grid_point_count": len(points), **result,
            })
            print(
                f"[continuous grid] projection case {case['case']} {nx}x{ny}: "
                f"converged={result['converged']} residual={result['verified_l2_residual']:.3g}",
                flush=True,
            )
    write_csv(table_dir / "grid_iprojection_20cases.csv", projection_rows)

    # The inherited Phase-2 solver deliberately clipped multipliers at 500.  A
    # hit at that ceiling is not evidence against existence of the continuous
    # projection, so retry only those cases with the separately recorded
    # high-dynamic-range numerical configuration.  This changes neither p_i nor
    # the moment target.
    retry_cfg = projection_config(cfg["high_dynamic_range_retry"])
    retry_rows: list[dict] = []
    resolved_rows: list[dict] = []
    for primary in projection_rows:
        selected = dict(primary)
        selected["solver_tier"] = "inherited_phase2"
        if not primary["converged"]:
            nx, ny = int(primary["grid_nx"]), int(primary["grid_ny"])
            source_index = int(primary["source_time_index"])
            case = cases[int(primary["case"])]
            points = grids[(nx, ny)][0]
            phi = gaussian_features_numpy(points, centers[int(case["design_index"])], sigma)
            retried = projection_summary(
                phi, np.asarray(density_by_key[(source_index, nx, ny)]["log_base_mass"]),
                np.asarray(case["target_moments"]), retry_cfg,
            )
            retry_row = {
                "case": int(case["case"]), "design_id": case["design_id"],
                "design_index": int(case["design_index"]), "style": case["style"],
                "day": float(case["day"]), "source_time_index": source_index,
                "grid_nx": nx, "grid_ny": ny, "grid_point_count": len(points),
                "trigger_primary_converged": bool(primary["converged"]),
                "trigger_primary_verified_l2_residual": primary["verified_l2_residual"],
                **retried,
            }
            retry_rows.append(retry_row)
            if retried["converged"] or retried["verified_l2_residual"] < primary["verified_l2_residual"]:
                selected = {
                    key: value for key, value in primary.items()
                    if key not in retried
                }
                selected.update(retried)
                selected["solver_tier"] = "high_dynamic_range_retry"
        resolved_rows.append(selected)
    write_csv(table_dir / "grid_iprojection_high_dynamic_range_retry.csv", retry_rows)
    write_csv(table_dir / "grid_iprojection_20cases_resolved.csv", resolved_rows)

    tolerance = float(cfg["projection_stability"]["moment_residual_tolerance"])
    resolutions = [(int(nx), int(ny)) for nx, ny in cfg["grid"]["resolutions"]]
    medium, fine = resolutions[-2:]
    stability_rows: list[dict] = []
    for case in cases:
        case_rows = [row for row in resolved_rows if row["case"] == int(case["case"])]
        by_resolution = {(row["grid_nx"], row["grid_ny"]): row for row in case_rows}
        med, fin = by_resolution[medium], by_resolution[fine]
        lam_med = np.asarray([med[f"lambda_{j}"] for j in range(4)])
        lam_fin = np.asarray([fin[f"lambda_{j}"] for j in range(4)])
        lambda_change = float(np.linalg.norm(lam_fin - lam_med) / max(1.0, np.linalg.norm(lam_fin)))
        kl_change = relative_change(float(fin["kl_divergence"]), float(med["kl_divergence"]))
        ess_change = abs(float(fin["log10_intrinsic_ess"]) - float(med["log10_intrinsic_ess"]))
        fine_geometry = next(
            row for row in geometry_rows
            if row["case"] == int(case["case"]) and (row["grid_nx"], row["grid_ny"]) == fine
        )
        stable = bool(
            med["converged"] and fin["converged"]
            and med["verified_l2_residual"] <= tolerance and fin["verified_l2_residual"] <= tolerance
            and fine_geometry["minimum_linf_residual"] <= tolerance
            and lambda_change <= float(cfg["projection_stability"]["maximum_medium_fine_relative_lambda_change"])
            and kl_change <= float(cfg["projection_stability"]["maximum_medium_fine_relative_kl_change"])
            and ess_change <= float(cfg["projection_stability"]["maximum_medium_fine_log10_intrinsic_ess_change"])
        )
        stability_rows.append({
            "case": int(case["case"]), "design_id": case["design_id"],
            "day": float(case["day"]), "source_time_index": int(case["source_time_index"]),
            "particle_200000_exact_lp_success": current_particle_feasible[int(case["case"])],
            "domain_grid_fine_minimum_linf_residual": fine_geometry["minimum_linf_residual"],
            "medium_fine_relative_lambda_change": lambda_change,
            "medium_fine_relative_kl_change": kl_change,
            "medium_fine_log10_intrinsic_ess_change": ess_change,
            "fine_verified_l2_residual": fin["verified_l2_residual"],
            "fine_kl_divergence": fin["kl_divergence"],
            "fine_intrinsic_ess": fin["intrinsic_ess"],
            "stable_grid_iprojection": stable,
        })
    write_csv(table_dir / "grid_iprojection_20case_stability.csv", stability_rows)

    summary = {
        "phase": "2E continuous-reference deterministic-grid I-projection audit",
        "config": cfg["_config_path"],
        "config_sha256": sha256(cfg["_config_path"]),
        "reference_checkpoint_sha256": checkpoint_hash,
        "continuous_reference_changed": False,
        "scientific_model_changed": False,
        "final_test_artifact_loaded": False,
        "target_inference_positions_all_strictly_inside_domain": not bool(containment[0]["outside_count"] or containment[0]["boundary_count"]),
        "case_count": len(cases),
        "stable_case_count": int(sum(row["stable_grid_iprojection"] for row in stability_rows)),
        "minimum_stable_cases_for_full_sweep": int(cfg["projection_stability"]["minimum_stable_cases"]),
    }
    summary["full_512_grid_sweep_authorized"] = bool(
        summary["stable_case_count"] >= summary["minimum_stable_cases_for_full_sweep"]
    )
    write_json(table_dir / "continuous_grid_iprojection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
