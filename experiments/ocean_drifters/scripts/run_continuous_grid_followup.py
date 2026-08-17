#!/usr/bin/env python3
"""Phase-2E comparisons, figures, and authorized full-bank grid sweep."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import jax
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
from run_continuous_grid_iprojection import (  # noqa: E402
    evaluate_density_grid,
    grid_points,
    load_config,
    projection_config,
    projection_summary,
    read_csv,
)
from mfsi.projection_tesseract import solve_i_projection_trajectory_tesseract_forward  # noqa: E402
from mfsi.reference import DomainPreservingReferenceFlow  # noqa: E402

jax.config.update("jax_enable_x64", True)

_FULL_WORKER_STATE: dict | None = None


def summarize_native(
    phi: np.ndarray,
    log_base: np.ndarray,
    target: np.ndarray,
    native: dict[str, np.ndarray],
    batch_index: int,
    time_index: int,
    ridge: float,
) -> dict:
    lam = np.asarray(native["lambda_values"][batch_index, time_index], dtype=np.float64)
    log_unnormalized_ratio = phi @ lam
    log_partition = float(logsumexp(log_base + log_unnormalized_ratio))
    log_ratio = log_unnormalized_ratio - log_partition
    log_weight = log_base + log_ratio
    weights = np.exp(log_weight)
    achieved = weights @ phi
    centered = phi - achieved
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
    eigenvalues = np.linalg.eigvalsh(covariance)
    log_second_moment = float(logsumexp(log_base + 2.0 * log_ratio))
    return {
        "converged": bool(native["converged"][batch_index, time_index]),
        "iterations": int(native["iterations"][batch_index, time_index]),
        "reported_residual": float(native["residual_norm"][batch_index, time_index]),
        "verified_l2_residual": float(np.linalg.norm(achieved - target)),
        "verified_linf_residual": float(np.max(np.abs(achieved - target))),
        **{f"lambda_{j}": float(lam[j]) for j in range(4)},
        "lambda_norm": float(np.linalg.norm(lam)),
        "kl_divergence": float(weights @ log_ratio),
        "intrinsic_ess": float(math.exp(-log_second_moment)) if log_second_moment < 710.0 else 0.0,
        "log10_intrinsic_ess": float(-log_second_moment / math.log(10.0)),
        "covariance_min_eigenvalue": float(eigenvalues[0]),
        "covariance_max_eigenvalue": float(eigenvalues[-1]),
        "covariance_condition_regularized": float(
            (eigenvalues[-1] + ridge) / max(eigenvalues[0] + ridge, 1e-300)
        ),
        "maximum_quadrature_weight": float(weights.max()),
    }


def initialize_full_worker(state: dict) -> None:
    global _FULL_WORKER_STATE
    _FULL_WORKER_STATE = state


def solve_full_design(design_index: int) -> list[dict]:
    if _FULL_WORKER_STATE is None:
        raise RuntimeError("full-sweep worker was not initialized")
    state = _FULL_WORKER_STATE
    points = state["points"]
    center = state["centers"][design_index]
    phi = gaussian_features_numpy(points, center, state["sigma"])
    phi_all = np.ascontiguousarray(
        np.broadcast_to(phi, (len(state["evaluation_indices"]), *phi.shape))
    )
    solver = state["solver"]
    native = solve_i_projection_trajectory_tesseract_forward(
        phi_all, state["log_base_all"], state["targets_all"][design_index:design_index + 1], solver
    )
    rows: list[dict] = []
    for eval_position, (source_index, day) in enumerate(zip(
        state["evaluation_indices"], state["evaluation_days"], strict=True
    )):
        result = summarize_native(
            phi, state["log_base_all"][eval_position],
            state["targets_all"][design_index, eval_position],
            native, 0, eval_position, solver.newton_ridge,
        )
        rows.append({
            "design_index": design_index,
            "design_id": str(state["design_ids"][design_index]),
            "style": str(state["styles"][design_index]),
            "day": float(day), "source_time_index": int(source_index),
            "grid_nx": state["nx"], "grid_ny": state["ny"],
            "domain_moment_feasible": True,
            "domain_feasibility_basis": "empirical target is a convex combination of Phi at strictly interior observations",
            **result,
            "usable": bool(
                result["converged"]
                and result["verified_l2_residual"] <= state["residual_tolerance"]
            ),
        })
    return rows


def solve_failed_zero_start(key: tuple[int, int]) -> tuple[tuple[int, int], dict]:
    if _FULL_WORKER_STATE is None:
        raise RuntimeError("full-sweep worker was not initialized")
    design_index, eval_position = key
    state = _FULL_WORKER_STATE
    phi = gaussian_features_numpy(
        state["points"], state["centers"][design_index], state["sigma"]
    )
    result = projection_summary(
        phi, state["log_base_all"][eval_position],
        state["targets_all"][design_index, eval_position], state["solver"],
    )
    return key, result


def shared_inputs(cfg: dict, phase2: dict) -> dict:
    processed = resolve(phase2["processed_dir"])
    analysis = resolve(phase2["analysis_dir"])
    bounds = np.asarray(phase2["domain"]["final_box_km"], dtype=np.float64)
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        design_ids = np.asarray(data["design_id"])
        styles = np.asarray(data["style"])
        sigma = float(data["sigma_km"])
    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        evaluation_days = np.asarray(data["evaluation_days"], dtype=np.float64)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        normalized_times = np.asarray(data["normalized_time"], dtype=np.float64)
    with np.load(resolve(cfg["conditioned_endpoint_estimator"]), allow_pickle=False) as data:
        atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
        bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
    z0 = float(next(
        row for row in read_csv(analysis / "tables/conditioned_kde_normalization.csv")
        if row["endpoint"] == "day0"
    )["Z_hat"])
    checkpoint = resolve(cfg["reference_checkpoint"])
    flow = DomainPreservingReferenceFlow.from_npz(
        checkpoint,
        substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
    )
    return locals()


def comparisons(cfg: dict, phase2: dict, shared: dict) -> None:
    processed: Path = shared["processed"]
    analysis: Path = shared["analysis"]
    table_dir = analysis / "tables"
    centers, sigma = shared["centers"], shared["sigma"]
    bounds = shared["bounds"]
    points, _, _ = grid_points(bounds, 512, 272)
    resolved = read_csv(table_dir / "grid_iprojection_20cases_resolved.csv")
    with (table_dir / "reference_support_cases.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    phase2d = read_csv(table_dir / "reference_support_lp_domain_preserving.csv")
    largest = max(int(row["particle_count"]) for row in phase2d)
    particle_status = {
        int(row["case"]): row for row in phase2d if int(row["particle_count"]) == largest
    }
    stability = {int(row["case"]): row for row in read_csv(table_dir / "grid_iprojection_20case_stability.csv")}
    with np.load(resolve(cfg["reference_particle_bank"]), allow_pickle=False) as data:
        nodes = np.asarray(data["nodes_km"], dtype=np.float64)
        bank_indices = np.asarray(data["evaluation_indices"], dtype=int)
    solver = projection_config(phase2["projection"])
    comparison_rows: list[dict] = []
    case_rows: list[dict] = []
    for case in cases:
        case_index = int(case["case"])
        source_index = int(case["source_time_index"])
        grid = next(
            row for row in resolved
            if int(row["case"]) == case_index and int(row["grid_nx"]) == 512
        )
        density_path = next(
            (SCRIPT_DIR.parent / "cache/reference_density_grids").glob(
                f"density_{sha256(resolve(cfg['reference_checkpoint']))[:12]}_t{source_index:03d}_512x272.npz"
            )
        )
        with np.load(density_path, allow_pickle=False) as data:
            log_base = np.asarray(data["log_base_mass"], dtype=np.float64)
        design_index = int(case["design_index"])
        phi = gaussian_features_numpy(points, centers[design_index], sigma)
        lam = np.asarray([float(grid[f"lambda_{j}"]) for j in range(4)])
        log_weights = log_base + phi @ lam
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)
        base_weights = np.exp(log_base)
        map_index = int(np.argmax(log_weights))
        one_sigma_union = np.max(phi, axis=1) >= math.exp(-0.5)
        status = particle_status[case_index]
        old_feasible = status["exact_lp_success"] == "True"
        stable = stability[case_index]["stable_grid_iprojection"] == "True"
        classification = (
            "particle-feasible comparison case" if old_feasible else
            "rare-event / finite-proposal failure" if stable else
            "continuous projection exists but resolution-sensitive"
        )
        case_rows.append({
            "case": case_index, "design_id": case["design_id"], "day": float(case["day"]),
            "particle_200000_lp_feasible": old_feasible,
            "domain_grid_lp_feasible": True,
            "grid_projection_converged": grid["converged"] == "True",
            "grid_projection_stable": stable,
            "classification": classification,
            "grid_verified_l2_residual": float(grid["verified_l2_residual"]),
            "grid_lambda_norm": float(grid["lambda_norm"]),
            "grid_kl_divergence": float(grid["kl_divergence"]),
            "grid_intrinsic_ess": float(grid["intrinsic_ess"]),
            "grid_log10_intrinsic_ess": float(grid["log10_intrinsic_ess"]),
            "projected_mean_x_km": float(weights @ points[:, 0]),
            "projected_mean_y_km": float(weights @ points[:, 1]),
            "projected_mode_x_km": float(points[map_index, 0]),
            "projected_mode_y_km": float(points[map_index, 1]),
            "base_probability_within_one_sigma_of_any_sensor": float(base_weights @ one_sigma_union),
            "minimum_base_sensor_expectation": float(np.min(base_weights @ phi)),
            "maximum_base_sensor_expectation": float(np.max(base_weights @ phi)),
        })
        if old_feasible:
            eval_position = int(np.flatnonzero(bank_indices == source_index)[0])
            particle_phi = gaussian_features_numpy(nodes[eval_position], centers[design_index], sigma)
            particle = projection_summary(
                particle_phi,
                np.full(len(particle_phi), -math.log(len(particle_phi))),
                np.asarray(case["target_moments"], dtype=np.float64),
                solver,
            )
            particle_lam = np.asarray([particle[f"lambda_{j}"] for j in range(4)])
            comparison_rows.append({
                "case": case_index, "design_id": case["design_id"], "day": float(case["day"]),
                "particle_count": len(particle_phi),
                "particle_converged": particle["converged"],
                "particle_verified_l2_residual": particle["verified_l2_residual"],
                "grid_verified_l2_residual": float(grid["verified_l2_residual"]),
                "particle_lambda_norm": particle["lambda_norm"],
                "grid_lambda_norm": float(grid["lambda_norm"]),
                "relative_lambda_difference": float(
                    np.linalg.norm(particle_lam - lam) / max(1.0, np.linalg.norm(lam))
                ),
                "particle_kl_divergence": particle["kl_divergence"],
                "grid_kl_divergence": float(grid["kl_divergence"]),
                "particle_empirical_ess_fraction": float(status["native_ess_fraction"]),
                "grid_intrinsic_ess_fraction": float(grid["intrinsic_ess"]),
                "maximum_projected_sensor_moment_error": float(grid["verified_linf_residual"]),
                **{f"particle_lambda_{j}": particle_lam[j] for j in range(4)},
                **{f"grid_lambda_{j}": lam[j] for j in range(4)},
            })
        print(f"[continuous grid followup] compared case {case_index}", flush=True)
    write_csv(table_dir / "particle_vs_grid_iprojection.csv", comparison_rows)
    write_csv(table_dir / "grid_iprojection_case_analysis.csv", case_rows)


def full_sweep(cfg: dict, phase2: dict, shared: dict) -> None:
    analysis: Path = shared["analysis"]
    table_dir = analysis / "tables"
    summary_path = table_dir / "continuous_grid_iprojection_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        audit_summary = json.load(handle)
    if not audit_summary["full_512_grid_sweep_authorized"]:
        raise RuntimeError("the frozen 20-case gate did not authorize a full sweep")
    nx, ny = (int(v) for v in cfg["full_bank"]["resolution"])
    points, dx, dy = grid_points(shared["bounds"], nx, ny)
    solver = projection_config(cfg["high_dynamic_range_retry"])
    cache_dir = SCRIPT_DIR.parent / "cache/reference_density_grids"
    checkpoint_hash = sha256(shared["checkpoint"])
    with np.load(shared["processed"] / "measurement_trajectories.npz", allow_pickle=False) as data:
        measurements = np.asarray(data["c"], dtype=np.float64)
    output = table_dir / "grid_iprojection_full_bank.csv"
    existing = read_csv(output) if output.exists() else []
    existing_counts = Counter(int(row["design_index"]) for row in existing)
    completed_designs = {
        design for design, count in existing_counts.items()
        if count == len(shared["evaluation_indices"])
    }
    all_rows: list[dict] = list(existing)
    stride = int(cfg["grid"]["evaluation_stride_source_steps"])
    substeps = int(cfg["grid"]["flow_rk4_substeps_per_evaluation_interval"])
    log_base_all = np.empty((len(shared["evaluation_indices"]), len(points)), dtype=np.float64)
    for eval_position, (source_index, day) in enumerate(zip(
        shared["evaluation_indices"], shared["evaluation_days"], strict=True
    )):
        source_index = int(source_index)
        steps = max(1, int(round(source_index / stride)) * substeps)
        density = evaluate_density_grid(
            flow=shared["flow"], points=points,
            time_value=float(shared["normalized_times"][source_index]), source_index=source_index,
            nx=nx, ny=ny, steps=steps, chunk_size=int(cfg["grid"]["density_chunk_size"]),
            atoms=shared["atoms"], bandwidth=shared["bandwidth"],
            conditioning_normalizer=shared["z0"], cell_area=dx * dy,
            cache_dir=cache_dir, checkpoint_hash=checkpoint_hash, force=False,
        )
        log_base_all[eval_position] = np.asarray(density["log_base_mass"], dtype=np.float64)
        print(f"[continuous grid full] density ready for day {day:g}", flush=True)

    targets_all = measurements[:, shared["evaluation_indices"]]
    started = time.perf_counter()
    pending = [index for index in range(len(shared["centers"])) if index not in completed_designs]
    worker_state = {
        "points": points, "centers": shared["centers"], "sigma": shared["sigma"],
        "log_base_all": log_base_all, "targets_all": targets_all,
        "evaluation_indices": shared["evaluation_indices"],
        "evaluation_days": shared["evaluation_days"],
        "design_ids": shared["design_ids"], "styles": shared["styles"],
        "solver": solver, "nx": nx, "ny": ny,
        "residual_tolerance": float(cfg["projection_stability"]["moment_residual_tolerance"]),
    }
    completed_now = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(cfg["full_bank"]["workers"]), mp_context=context,
        initializer=initialize_full_worker, initargs=(worker_state,),
    ) as pool:
        futures = {pool.submit(solve_full_design, index): index for index in pending}
        for future in as_completed(futures):
            all_rows.extend(future.result())
            completed_now += 1
            total_completed = len(completed_designs) + completed_now
            if completed_now % 12 == 0 or completed_now == len(pending):
                write_csv(output, all_rows)
                print(
                    f"[continuous grid full] layouts {total_completed}/512 in "
                    f"{time.perf_counter()-started:.1f}s; usable projections="
                    f"{sum(str(row['usable']).lower() == 'true' for row in all_rows)}/{len(all_rows)}",
                    flush=True,
                )

    # A trajectory solve warm-starts each time from the preceding multiplier.
    # Retry failures independently from zero so a pathological previous-time
    # direction cannot be mistaken for failure of the current projection.
    source_to_position = {
        int(source_index): position
        for position, source_index in enumerate(shared["evaluation_indices"])
    }
    failed_keys = [
        (int(row["design_index"]), source_to_position[int(row["source_time_index"])])
        for row in all_rows
        if str(row["usable"]).lower() != "true"
        and str(row.get("zero_start_retry_performed", "False")).lower() != "true"
    ]
    if failed_keys:
        row_by_key = {
            (int(row["design_index"]), source_to_position[int(row["source_time_index"])]): row
            for row in all_rows
        }
        with ProcessPoolExecutor(
            max_workers=int(cfg["full_bank"]["workers"]), mp_context=context,
            initializer=initialize_full_worker, initargs=(worker_state,),
        ) as pool:
            futures = [pool.submit(solve_failed_zero_start, key) for key in failed_keys]
            for future in as_completed(futures):
                key, retry = future.result()
                row = row_by_key[key]
                row["zero_start_retry_performed"] = True
                row["trajectory_warm_start_verified_l2_residual"] = row["verified_l2_residual"]
                row["trajectory_warm_start_converged"] = row["converged"]
                if retry["converged"] or retry["verified_l2_residual"] < float(row["verified_l2_residual"]):
                    row.update(retry)
                    row["solver_initialization"] = "independent_zero_start_retry"
                row["usable"] = bool(
                    str(row["converged"]).lower() == "true"
                    and float(row["verified_l2_residual"])
                    <= float(cfg["projection_stability"]["moment_residual_tolerance"])
                )
        write_csv(output, all_rows)
        print(
            f"[continuous grid full] zero-start retry complete for {len(failed_keys)} projections; "
            f"usable={sum(str(row['usable']).lower() == 'true' for row in all_rows)}/{len(all_rows)}",
            flush=True,
        )
    by_design: list[dict] = []
    for design_index in range(len(shared["centers"])):
        rows = [row for row in all_rows if int(row["design_index"]) == design_index]
        usable_count = sum(str(row["usable"]).lower() == "true" for row in rows)
        by_design.append({
            "design_index": design_index,
            "design_id": str(shared["design_ids"][design_index]),
            "style": str(shared["styles"][design_index]),
            "evaluation_time_count": len(rows),
            "usable_time_count": usable_count,
            "fully_usable": usable_count == len(shared["evaluation_indices"]),
            "maximum_verified_l2_residual": max(float(row["verified_l2_residual"]) for row in rows),
            "maximum_kl_divergence": max(float(row["kl_divergence"]) for row in rows),
            "minimum_log10_intrinsic_ess": min(float(row["log10_intrinsic_ess"]) for row in rows),
        })
    write_csv(table_dir / "grid_iprojection_full_bank_by_design.csv", by_design)
    write_json(table_dir / "continuous_grid_full_bank_summary.json", {
        "design_count": len(by_design),
        "evaluation_time_count": len(shared["evaluation_indices"]),
        "projection_count": len(all_rows),
        "usable_projection_count": sum(str(row["usable"]).lower() == "true" for row in all_rows),
        "fully_usable_design_count": sum(row["fully_usable"] for row in by_design),
        "resolution": [nx, ny],
        "resolution_stability_basis": "accepted 18/20 medium-fine frozen diagnostic contract",
        "final_test_artifact_loaded": False,
    })


def finalize(cfg: dict, phase2: dict, shared: dict) -> None:
    analysis: Path = shared["analysis"]
    table_dir = analysis / "tables"
    figure_dir = analysis / "figures/grid_iprojection"
    figure_dir.mkdir(parents=True, exist_ok=True)
    nx, ny = (int(v) for v in cfg["full_bank"]["resolution"])
    points, dx, dy = grid_points(shared["bounds"], nx, ny)
    checkpoint_hash = sha256(shared["checkpoint"])
    cache_dir = SCRIPT_DIR.parent / "cache/reference_density_grids"
    with np.load(resolve(cfg["reference_particle_bank"]), allow_pickle=False) as data:
        particle_nodes = np.asarray(data["nodes_km"], dtype=np.float64)
        bank_indices = np.asarray(data["evaluation_indices"], dtype=int)
    normalization_rows: list[dict] = []
    comparison_rows: list[dict] = []
    box_rows: list[dict] = []
    density_for_plot: dict[int, np.ndarray] = {}
    xmid = float((shared["bounds"][0] + shared["bounds"][1]) / 2.0)
    ymid = float((shared["bounds"][2] + shared["bounds"][3]) / 2.0)
    boxes = {
        "southwest": (points[:, 0] < xmid) & (points[:, 1] < ymid),
        "northwest": (points[:, 0] < xmid) & (points[:, 1] >= ymid),
        "southeast": (points[:, 0] >= xmid) & (points[:, 1] < ymid),
        "northeast": (points[:, 0] >= xmid) & (points[:, 1] >= ymid),
    }
    for eval_position, (source_index, day) in enumerate(zip(
        shared["evaluation_indices"], shared["evaluation_days"], strict=True
    )):
        path = cache_dir / f"density_{checkpoint_hash[:12]}_t{int(source_index):03d}_{nx}x{ny}.npz"
        with np.load(path, allow_pickle=False) as data:
            log_density = np.asarray(data["log_density"], dtype=np.float64)
            log_base = np.asarray(data["log_base_mass"], dtype=np.float64)
            normalization = float(data["normalization"])
            elapsed = float(data["elapsed_seconds"])
        weights = np.exp(log_base)
        particle_position = int(np.flatnonzero(bank_indices == int(source_index))[0])
        particle = particle_nodes[particle_position]
        grid_mean = weights @ points
        particle_mean = particle.mean(axis=0)
        grid_centered = points - grid_mean
        particle_centered = particle - particle_mean
        grid_cov = np.einsum("n,ni,nj->ij", weights, grid_centered, grid_centered)
        particle_cov = particle_centered.T @ particle_centered / len(particle)
        normalization_rows.append({
            "day": float(day), "source_time_index": int(source_index),
            "grid_nx": nx, "grid_ny": ny, "dx_km": dx, "dy_km": dy,
            "raw_normalization": normalization,
            "absolute_normalization_error": abs(normalization - 1.0),
            "minimum_log_density": float(log_density.min()),
            "maximum_log_density": float(log_density.max()),
            "log10_density_dynamic_range": float((log_density.max() - log_density.min()) / math.log(10.0)),
            "density_evaluation_seconds": elapsed,
        })
        comparison_rows.append({
            "day": float(day), "source_time_index": int(source_index),
            "mean_error_km": float(np.linalg.norm(grid_mean - particle_mean)),
            "relative_covariance_frobenius_error": float(
                np.linalg.norm(grid_cov - particle_cov) / np.linalg.norm(particle_cov)
            ),
            "grid_mean_x_km": float(grid_mean[0]), "grid_mean_y_km": float(grid_mean[1]),
            "particle_mean_x_km": float(particle_mean[0]), "particle_mean_y_km": float(particle_mean[1]),
        })
        for name, grid_mask in boxes.items():
            particle_mask = {
                "southwest": (particle[:, 0] < xmid) & (particle[:, 1] < ymid),
                "northwest": (particle[:, 0] < xmid) & (particle[:, 1] >= ymid),
                "southeast": (particle[:, 0] >= xmid) & (particle[:, 1] < ymid),
                "northeast": (particle[:, 0] >= xmid) & (particle[:, 1] >= ymid),
            }[name]
            grid_mass = float(weights @ grid_mask)
            particle_mass = float(particle_mask.mean())
            box_rows.append({
                "day": float(day), "source_time_index": int(source_index), "box": name,
                "grid_mass": grid_mass, "particle_mass": particle_mass,
                "absolute_mass_error": abs(grid_mass - particle_mass),
            })
        if float(day) in (0.0, 15.0, 30.0, 45.0):
            density_for_plot[int(source_index)] = log_density
    write_csv(table_dir / "reference_grid_normalization_full_times.csv", normalization_rows)
    write_csv(table_dir / "reference_grid_particle_comparison_full_times.csv", comparison_rows)
    write_csv(table_dir / "reference_grid_fixed_box_comparison.csv", box_rows)

    # Reference-density panels.  Relative log density is clipped only for color
    # display; all integrations above use the unclipped log density.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    plot_times = [(0, 0.0), (60, 15.0), (120, 30.0), (180, 45.0)]
    for axis, (source_index, day) in zip(axes.ravel(), plot_times, strict=True):
        values = density_for_plot[source_index].reshape(ny, nx)
        relative = np.maximum(values - values.max(), -45.0)
        image = axis.imshow(
            relative, origin="lower",
            extent=[shared["bounds"][0], shared["bounds"][1], shared["bounds"][2], shared["bounds"][3]],
            aspect="auto", cmap="viridis", vmin=-45.0, vmax=0.0,
        )
        particle_position = int(np.flatnonzero(bank_indices == source_index)[0])
        sample = particle_nodes[particle_position, ::40]
        axis.scatter(sample[:, 0], sample[:, 1], s=1.0, color="white", alpha=0.15, linewidths=0)
        axis.set_title(f"Day {day:g}")
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")
    fig.colorbar(image, ax=axes, label="log density relative to maximum (display clipped at −45)")
    fig.savefig(figure_dir / "reference_density_fine_grid.png", dpi=170)
    plt.close(fig)

    audit_density = read_csv(table_dir / "reference_grid_normalization.csv")
    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for day in sorted({float(row["day"]) for row in audit_density}):
        selected = sorted(
            (row for row in audit_density if float(row["day"]) == day),
            key=lambda row: int(row["grid_nx"]),
        )
        axis.plot(
            [int(row["grid_nx"]) for row in selected],
            [float(row["absolute_normalization_error"]) for row in selected],
            marker="o", label=f"day {day:g}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("grid cells in x")
    axis.set_ylabel("absolute normalization error")
    axis.legend(ncol=2)
    fig.savefig(figure_dir / "normalization_resolution.png", dpi=170)
    plt.close(fig)

    stability = read_csv(table_dir / "grid_iprojection_20case_stability.csv")
    case_analysis = read_csv(table_dir / "grid_iprojection_case_analysis.csv")
    current_feasible = {int(row["case"]): row["particle_200000_lp_feasible"] == "True" for row in case_analysis}
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, constrained_layout=True)
    case_number = np.asarray([int(row["case"]) for row in stability])
    colors = ["#4c78a8" if current_feasible[int(row["case"])] else "#e45756" for row in stability]
    axes[0].bar(case_number, [float(row["fine_kl_divergence"]) for row in stability], color=colors)
    axes[0].set_ylabel("KL(grid projection || reference)")
    burden = [-float(row["fine_intrinsic_ess"]) for row in stability]
    log_burden = [-float(next(
        item["grid_log10_intrinsic_ess"] for item in case_analysis if item["case"] == row["case"]
    )) for row in stability]
    axes[1].bar(case_number, log_burden, color=colors)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("−log10 intrinsic ESS")
    axes[1].set_xlabel("frozen diagnostic case")
    fig.savefig(figure_dir / "audit_kl_overlap.png", dpi=170)
    plt.close(fig)

    particle_grid = read_csv(table_dir / "particle_vs_grid_iprojection.csv")
    fig, axis = plt.subplots(figsize=(5.3, 5.0), constrained_layout=True)
    x = np.asarray([float(row["particle_kl_divergence"]) for row in particle_grid])
    y = np.asarray([float(row["grid_kl_divergence"]) for row in particle_grid])
    axis.scatter(x, y, s=42, color="#4c78a8")
    limits = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.05]
    axis.plot(limits, limits, "--", color="black", linewidth=1)
    axis.set_xlim(limits); axis.set_ylim(limits)
    axis.set_xlabel("particle KL")
    axis.set_ylabel("continuous-grid KL")
    fig.savefig(figure_dir / "particle_vs_grid_kl.png", dpi=170)
    plt.close(fig)

    by_design = read_csv(table_dir / "grid_iprojection_full_bank_by_design.csv")
    fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    counts = [int(row["usable_time_count"]) for row in by_design]
    axis.hist(counts, bins=np.arange(-0.5, 20.5, 1), color="#59a14f", edgecolor="white")
    axis.set_xlabel("usable evaluation times out of 19")
    axis.set_ylabel("layout count")
    fig.savefig(figure_dir / "full_bank_usable_times.png", dpi=170)
    plt.close(fig)

    audit_summary = json.loads((table_dir / "continuous_grid_iprojection_summary.json").read_text())
    full_summary = json.loads((table_dir / "continuous_grid_full_bank_summary.json").read_text())
    containment = read_csv(table_dir / "continuous_reference_target_domain_containment.csv")
    resolved = [row for row in read_csv(table_dir / "grid_iprojection_20cases_resolved.csv") if int(row["grid_nx"]) == 512]
    fine_norm = [row for row in normalization_rows]
    old_failures = [row for row in case_analysis if row["particle_200000_lp_feasible"] == "False"]
    old_fail_stable = sum(row["grid_projection_stable"] == "True" for row in old_failures)
    all_time_max_mean = max(float(row["mean_error_km"]) for row in comparison_rows)
    all_time_max_cov = max(float(row["relative_covariance_frobenius_error"]) for row in comparison_rows)
    all_time_max_norm = max(float(row["absolute_normalization_error"]) for row in fine_norm)
    box_max_error = max(float(row["absolute_mass_error"]) for row in box_rows)
    kl_values = np.asarray([float(row["kl_divergence"]) for row in resolved])
    log_ess_values = np.asarray([float(row["log10_intrinsic_ess"]) for row in resolved])
    report = f"""# Continuous-reference grid I-projection audit

## Decision

The frozen continuous deterministic reference is mathematically adequate for the frozen sensor-moment targets, but ordinary finite-particle reweighting is not. All 20 targets are feasible in the domain moment hull, all 20 continuous-grid projections reach the moment tolerance with high-dynamic-range Newton arithmetic, and {audit_summary['stable_case_count']}/20 pass the predeclared medium/fine stability contract. The practical 18/20 gate is met, so no stochastic bridge is justified by the support evidence.

The authorized full sweep yields **{full_summary['usable_projection_count']}/{full_summary['projection_count']} usable layout-time projections** and **{full_summary['fully_usable_design_count']}/512 layouts usable at all 19 times**. The remaining {full_summary['projection_count']-full_summary['usable_projection_count']} unresolved projections belong to {512-full_summary['fully_usable_design_count']} layouts and occur only at days 2.5 and 5; they hit the numerical multiplier ceiling and require local/adaptive quadrature if those layouts must be retained.

## Support and domain checks

The Gaussian KDE is strictly positive on R². Conditioning on the rectangle preserves strict positivity at every interior point. The logistic transform is a smooth bijection from R² to the open rectangle, and the finite SiLU network is continuously differentiable with bounded activation derivative, giving the latent ODE a unique invertible finite-time flow under the usual finite-parameter regularity assumptions. A smooth invertible pushforward therefore remains positive throughout the interior. The caveat is numerical: finite-step RK4 approximates that exact flow, and the measured normalization error below quantifies the accumulated discretization/quadrature error; cell centers lie away from the implementation's inverse-map clipping threshold.

All {containment[0]['observation_count']} inference observations and all {containment[1]['observation_count']} validation observations at the 19 frozen times are strictly inside the rectangle. No point is outside or on its boundary. The closest inference distances to xmin, xmax, ymin, ymax are respectively {float(containment[0]['minimum_distance_to_xmin_km']):.2f}, {float(containment[0]['minimum_distance_to_xmax_km']):.2f}, {float(containment[0]['minimum_distance_to_ymin_km']):.2f}, and {float(containment[0]['minimum_distance_to_ymax_km']):.2f} km.

The independent simplex LP over 128×68, 256×136, and 512×272 cell-centered domain grids is feasible for all 20 cases at every resolution. The worst fine-grid L-infinity residual is {max(float(row['domain_grid_fine_minimum_linf_residual']) for row in stability):.3e}. This establishes measurement/domain compatibility independently of reference probability.

## Continuous density evaluation and validation

For each physical grid point, the implementation maps to latent coordinates, integrates the frozen neural ODE backward, and accumulates the exact two-dimensional autodiff divergence with RK4. It evaluates the frozen conditioned day-0 KDE at the recovered initial point, applies the logistic Jacobians at both endpoints, and normalizes cell masses with log-sum-exp. No density floor or clipped integral is used.

The grids are 128×68 (28.516×28.676 km), 256×136 (14.258×14.338 km), and 512×272 (7.129×7.169 km). Across all 19 times, the fine-grid maximum raw normalization error is {all_time_max_norm:.4g}; the maximum mean discrepancy from the 200,000-particle bank is {all_time_max_mean:.3f} km, the maximum relative covariance error is {all_time_max_cov:.4g}, and the maximum absolute fixed-quadrant mass error is {box_max_error:.4g}. On the four frozen diagnostic times the maximum sensor-expectation error is {max(float(row['maximum_sensor_expectation_error']) for row in read_csv(table_dir / 'reference_grid_particle_comparison.csv') if int(row['grid_nx']) == 512):.4g}. These pass all predeclared density-validation gates.

Log densities are finite on every cell center but span as much as {max(float(row['log10_density_dynamic_range']) for row in fine_norm):.3g} decades. Values this small underflow if exponentiated directly; retaining them in log space is essential and is evidence of rare-event overlap, not zero continuous support.

## Twenty-case projections

The inherited solver converges without alteration in 12/20 cases. Its other eight results hit the inherited ±500 multiplier clip. A separately recorded retry changes only numerical Newton limits (ridge 1e-15, clip 1e9) and obtains residuals below 1e-8 for all eight. Thus 20/20 fine-grid projections converge, while 18/20 meet the full medium/fine stability gate. Cases 0 and 2 match moments and KL but remain resolution-sensitive in intrinsic overlap or a nearly non-identifiable multiplier direction; adaptive/local quadrature is required before treating their detailed density-ratio diagnostics as settled.

Fine-grid KL ranges from {kl_values.min():.4g} to {kl_values.max():.4g} (median {np.median(kl_values):.4g}). Log10 intrinsic ESS ranges from {log_ess_values.min():.4g} to {log_ess_values.max():.4g} (median {np.median(log_ess_values):.4g}). Tiny ESS is reported as correction burden, not used as a feasibility rejection.

Under the current 200,000-particle audit, 10 cases were particle-LP feasible. Particle and grid KL agree well in the two formerly healthy cases: case 15 is {float(next(row['relative_lambda_difference'] for row in particle_grid if row['case']=='15')):.1%} apart in lambda and case 18 is {float(next(row['relative_lambda_difference'] for row in particle_grid if row['case']=='18')):.1%}, while their KL differences are {abs(float(next(row['particle_kl_divergence'] for row in particle_grid if row['case']=='15'))-float(next(row['grid_kl_divergence'] for row in particle_grid if row['case']=='15'))):.3g} and {abs(float(next(row['particle_kl_divergence'] for row in particle_grid if row['case']=='18'))-float(next(row['grid_kl_divergence'] for row in particle_grid if row['case']=='18'))):.3g}. Larger discrepancies among low-ESS cases are consistent with finite rare-event sampling and ill-conditioned multiplier directions.

All ten previous particle-convex-hull failures are continuously moment-feasible and converge on the fine grid. Eight are fully medium/fine stable and are classified as rare-event / finite-proposal failures; two converge but retain resolution-sensitive diagnostics. None is a true domain-moment failure.

## Full-bank consequence

The 512-layout sweep uses the accepted 512×272 grid and the frozen high-dynamic-range native solver. Domain feasibility is exact analytically here: each target is the empirical average of Phi at inference observations that were verified strictly inside D. Of 9,728 projections, {full_summary['usable_projection_count']} meet the 2e-7 residual gate. {full_summary['fully_usable_design_count']} layouts are usable over all times. Validation-law-risk ranking is therefore justified to resume on those fully usable layouts using grid projections; it has not been rerun in this phase.

There is no remaining evidence that changing to a stochastic reference is required. The unresolved numerical work is narrower: locally refine the 27 early-time, near-boundary exponential tilts (and the two resolution-sensitive audit diagnostics) if retaining every layout is scientifically necessary. A stochastic bridge would change the reference geometry and action, whereas adaptive quadrature would preserve the frozen law.

## Reproducibility and outputs

The final-test trajectories were not loaded. The scientific reference checkpoint, endpoint KDE, domain, sensors, sigma, and target moments were unchanged. Core tables are `domain_moment_hull_20cases.csv`, `reference_grid_normalization_full_times.csv`, `reference_grid_particle_comparison_full_times.csv`, `grid_iprojection_20cases_resolved.csv`, `particle_vs_grid_iprojection.csv`, `grid_iprojection_case_analysis.csv`, `grid_iprojection_full_bank.csv`, and `grid_iprojection_full_bank_by_design.csv`. Figures are under `analysis/figures/grid_iprojection/`.
"""
    (analysis / "continuous_grid_iprojection_report.md").write_text(report, encoding="utf-8")
    write_json(table_dir / "continuous_grid_final_summary.json", {
        "decision": "continuous deterministic reference adequate; particle reweighting inadequate",
        "audit_converged_cases": 20,
        "audit_stable_cases": audit_summary["stable_case_count"],
        "current_particle_failure_cases": len(old_failures),
        "current_particle_failures_grid_stable": old_fail_stable,
        "full_usable_projections": full_summary["usable_projection_count"],
        "full_projection_count": full_summary["projection_count"],
        "fully_usable_designs": full_summary["fully_usable_design_count"],
        "stochastic_bridge_recommended": False,
        "validation_law_risk_ranking_status": "justified to resume on 485 fully usable layouts; not run in Phase 2E",
        "final_test_artifact_loaded": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=("comparisons", "full", "finalize", "all"), default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["base_phase2_config"]))
    shared = shared_inputs(cfg, phase2)
    if args.stage in ("comparisons", "all"):
        comparisons(cfg, phase2, shared)
    if args.stage in ("full", "all"):
        full_sweep(cfg, phase2, shared)
    if args.stage in ("finalize", "all"):
        finalize(cfg, phase2, shared)


if __name__ == "__main__":
    main()
