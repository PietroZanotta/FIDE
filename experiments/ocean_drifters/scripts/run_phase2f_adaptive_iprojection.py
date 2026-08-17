#!/usr/bin/env python3
"""Phase 2F adaptive closure of the 27 concentrated grid I-projections."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from phase2_common import gaussian_features_numpy, load_phase2_config, resolve, sha256, write_csv, write_json  # noqa: E402
from repair_reference_support import log_kde_density  # noqa: E402
from run_continuous_grid_iprojection import grid_points, projection_config, projection_summary, read_csv  # noqa: E402
from mfsi.reference import DomainPreservingReferenceFlow  # noqa: E402
from mfsi.reference_density import backward_latent_with_log_density_correction, logistic_log_abs_det_jacobian  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_config(path: str | Path | None) -> dict[str, Any]:
    source = Path(path) if path else SCRIPT_DIR.parent / "configs/mfsi_phase2f.json"
    with source.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def evaluate_log_density_points(
    flow: DomainPreservingReferenceFlow,
    points: np.ndarray,
    *,
    time_value: float,
    steps: int,
    atoms: np.ndarray,
    bandwidth: np.ndarray,
    conditioning_normalizer: float,
    chunk_size: int,
) -> np.ndarray:
    bounds = np.asarray(flow.bounds, dtype=np.float64)
    output = np.empty(len(points), dtype=np.float64)
    backward = jax.jit(lambda z: backward_latent_with_log_density_correction(
        flow.params, z, jnp.asarray(time_value, dtype=jnp.float64), steps=steps
    ))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        zt = flow.to_latent(jnp.asarray(points[start:stop]))
        z0, correction = backward(zt)
        x0 = np.asarray(flow.to_physical(z0), dtype=np.float64)
        log_initial = log_kde_density(x0, atoms, bandwidth, chunk=chunk_size)
        log_initial -= math.log(conditioning_normalizer)
        output[start:stop] = (
            log_initial
            + np.asarray(logistic_log_abs_det_jacobian(z0, bounds))
            + np.asarray(correction)
            - np.asarray(logistic_log_abs_det_jacobian(zt, bounds))
        )
    if not np.isfinite(output).all():
        raise RuntimeError("adaptive density evaluation produced nonfinite log density")
    return output


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(1.0, abs(new))


def solve_case(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        cfg = load_config(payload["config"])
        phase2 = load_phase2_config(resolve(cfg["phase2_config"]))
        grid_cfg_path = resolve(cfg["continuous_grid_config"])
        with grid_cfg_path.open(encoding="utf-8") as handle:
            grid_cfg = json.load(handle)
        adaptive = cfg["adaptive_iprojection"]
        solver = projection_config(adaptive["solver"])
        bounds = np.asarray(phase2["domain"]["final_box_km"], dtype=np.float64)
        nx, ny = (int(value) for value in adaptive["base_resolution"])
        points, dx0, dy0 = grid_points(bounds, nx, ny)
        widths_x = np.full(len(points), dx0, dtype=np.float64)
        widths_y = np.full(len(points), dy0, dtype=np.float64)
        cell_levels = np.zeros(len(points), dtype=np.int16)
        design_index = int(payload["design_index"])
        source_index = int(payload["source_time_index"])
        processed = resolve(phase2["processed_dir"])
        analysis = resolve(phase2["analysis_dir"])
        with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
            centers = np.asarray(data["centers_km"], dtype=np.float64)[design_index]
            sigma = float(data["sigma_km"])
            design_id = str(data["design_id"][design_index])
        with np.load(processed / "measurement_trajectories.npz", allow_pickle=False) as data:
            target = np.asarray(data["c"][design_index, source_index], dtype=np.float64)
        with np.load(processed / "development_270.npz", allow_pickle=False) as data:
            time_value = float(data["normalized_time"][source_index])
            day = float(data["relative_days"][source_index])
        checkpoint = resolve(cfg["reference_checkpoint"])
        checkpoint_hash = sha256(checkpoint)
        density_cache = SCRIPT_DIR.parent / "cache/reference_density_grids" / (
            f"density_{checkpoint_hash[:12]}_t{source_index:03d}_{nx}x{ny}.npz"
        )
        with np.load(density_cache, allow_pickle=False) as data:
            log_density = np.asarray(data["log_density"], dtype=np.float64)
        estimator = resolve(cfg["conditioned_endpoint_estimator"])
        with np.load(estimator, allow_pickle=False) as data:
            atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
            bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
        normalization = next(
            row for row in read_csv(analysis / "tables/conditioned_kde_normalization.csv")
            if row["endpoint"] == "day0"
        )
        z0 = float(normalization["Z_hat"])
        flow = DomainPreservingReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(phase2["reference"]["rk4_substeps_per_time_interval"]),
        )
        stride = int(grid_cfg["grid"]["evaluation_stride_source_steps"])
        substeps = int(grid_cfg["grid"]["flow_rk4_substeps_per_evaluation_interval"])
        rk4_steps = max(1, int(round(source_index / stride)) * substeps)
        rows: list[dict] = []
        previous: dict[str, Any] | None = None
        resolved = False
        final_result: dict[str, Any] | None = None
        final_weights: np.ndarray | None = None

        for level in range(int(adaptive["maximum_refinement_levels"]) + 1):
            log_raw_mass = log_density + np.log(widths_x * widths_y)
            log_base = log_raw_mass - logsumexp(log_raw_mass)
            phi = gaussian_features_numpy(points, centers, sigma)
            result = projection_summary(phi, log_base, target, solver)
            lam = np.asarray([result[f"lambda_{j}"] for j in range(4)])
            log_weight = log_base + phi @ lam
            log_weight -= logsumexp(log_weight)
            weights = np.exp(log_weight)
            achieved = weights @ phi
            accepted = bool(
                result["converged"]
                and result["verified_l2_residual"] <= float(adaptive["moment_residual_tolerance"])
            )
            stable = False
            lambda_change = kl_change = ess_change = moment_change = math.nan
            if previous is not None:
                previous_lambda = np.asarray([previous[f"lambda_{j}"] for j in range(4)])
                lambda_change = float(
                    np.linalg.norm(lam - previous_lambda) / max(1.0, np.linalg.norm(lam))
                )
                kl_change = relative_change(result["kl_divergence"], previous["kl_divergence"])
                ess_change = abs(
                    result["log10_intrinsic_ess"] - previous["log10_intrinsic_ess"]
                )
                moment_change = float(np.max(np.abs(achieved - previous["achieved_moments"])))
                stable = bool(
                    accepted and previous["accepted"]
                    and lambda_change <= float(adaptive["stability_relative_lambda_change"])
                    and kl_change <= float(adaptive["stability_relative_kl_change"])
                    and ess_change <= float(adaptive["stability_log10_intrinsic_ess_change"])
                    and moment_change <= float(adaptive["stability_projected_moment_change"])
                )
            rows.append({
                "design_index": design_index, "design_id": design_id,
                "day": day, "source_time_index": source_index,
                "refinement_level": level, "quadrature_cell_count": len(points),
                "minimum_cell_width_x_km": float(widths_x.min()),
                "minimum_cell_width_y_km": float(widths_y.min()),
                **result,
                "accepted_moment_residual": accepted,
                "relative_lambda_change_from_previous": lambda_change,
                "relative_kl_change_from_previous": kl_change,
                "log10_intrinsic_ess_change_from_previous": ess_change,
                "projected_moment_linf_change_from_previous": moment_change,
                "stable_under_additional_refinement": stable,
            })
            final_result, final_weights = result, weights
            if stable:
                resolved = True
                break
            if level == int(adaptive["maximum_refinement_levels"]):
                break

            eligible_cell = (
                (0.5 * widths_x >= float(adaptive["minimum_cell_width_km"]))
                & (0.5 * widths_y >= float(adaptive["minimum_cell_width_km"]))
            )
            order = np.argsort(weights)[::-1]
            order = order[eligible_cell[order]]
            if not len(order):
                rows[-1]["refinement_stop_reason"] = "minimum_cell_width_reached"
                break
            cumulative = np.cumsum(weights[order])
            count = int(np.searchsorted(
                cumulative, float(adaptive["projected_mass_refinement_fraction"]), side="left"
            ) + 1)
            count = max(count, int(adaptive["minimum_refined_cells_per_level"]))
            count = min(count, int(adaptive["maximum_refined_cells_per_level"]), len(order))
            capacity = (int(adaptive["maximum_total_cells"]) - len(points)) // 3
            count = min(count, capacity)
            if count <= 0:
                rows[-1]["refinement_stop_reason"] = "maximum_total_cells_reached"
                break
            selected = order[:count]
            keep = np.ones(len(points), dtype=bool)
            keep[selected] = False
            parent = points[selected]
            parent_dx = widths_x[selected]
            parent_dy = widths_y[selected]
            offsets = np.asarray([
                [-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]
            ])
            children = parent[:, None, :] + offsets[None] * np.stack(
                [parent_dx, parent_dy], axis=-1
            )[:, None, :]
            children = children.reshape((-1, 2))
            child_log_density = evaluate_log_density_points(
                flow, children, time_value=time_value, steps=rk4_steps,
                atoms=atoms, bandwidth=bandwidth, conditioning_normalizer=z0,
                chunk_size=int(grid_cfg["grid"]["density_chunk_size"]),
            )
            points = np.concatenate([points[keep], children], axis=0)
            log_density = np.concatenate([log_density[keep], child_log_density])
            widths_x = np.concatenate([widths_x[keep], np.repeat(parent_dx * 0.5, 4)])
            widths_y = np.concatenate([widths_y[keep], np.repeat(parent_dy * 0.5, 4)])
            cell_levels = np.concatenate([
                cell_levels[keep], np.repeat(cell_levels[selected] + 1, 4)
            ])
            rows[-1]["refined_parent_cell_count"] = count
            rows[-1]["refined_projected_mass"] = float(weights[selected].sum())
            previous = {
                **result, "accepted": accepted, "achieved_moments": achieved,
            }

        if final_result is None or final_weights is None:
            raise RuntimeError("adaptive solve produced no level")
        classification = (
            "A_resolved" if resolved
            else "B_mathematically_feasible_but_numerically_unresolved"
        )
        cache_dir = SCRIPT_DIR.parent / "cache/adaptive_iprojection"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"design_{design_index:06d}_t{source_index:03d}.npz"
        final_log_raw = log_density + np.log(widths_x * widths_y)
        final_log_base = final_log_raw - logsumexp(final_log_raw)
        np.savez_compressed(
            cache_path,
            points_km=points, log_density=log_density, log_base_mass=final_log_base,
            cell_width_x_km=widths_x, cell_width_y_km=widths_y,
            cell_level=cell_levels, lambda_value=np.asarray([
                final_result[f"lambda_{j}"] for j in range(4)
            ]), projected_weights=final_weights,
            target_moments=target, design_index=np.asarray(design_index),
            source_time_index=np.asarray(source_index), day=np.asarray(day),
            classification=np.asarray(classification), resolved=np.asarray(resolved),
            config_sha256=np.asarray(sha256(cfg["_config_path"])),
            final_test_accessed=np.asarray(False),
        )
        return {
            "ok": True, "design_index": design_index, "design_id": design_id,
            "source_time_index": source_index, "day": day,
            "classification": classification, "resolved": resolved,
            "final_refinement_level": int(rows[-1]["refinement_level"]),
            "final_quadrature_cell_count": len(points),
            "final_verified_l2_residual": final_result["verified_l2_residual"],
            "final_lambda_norm": final_result["lambda_norm"],
            "final_kl_divergence": final_result["kl_divergence"],
            "final_log10_intrinsic_ess": final_result["log10_intrinsic_ess"],
            "cache_path": str(cache_path.relative_to(ROOT)), "rows": rows,
        }
    except Exception as exc:  # preserve an exact implementation-failure record
        return {
            "ok": False, "design_index": int(payload["design_index"]),
            "source_time_index": int(payload["source_time_index"]),
            "classification": "C_implementation_failure", "resolved": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "rows": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    phase2 = load_phase2_config(resolve(cfg["phase2_config"]))
    analysis = resolve(phase2["analysis_dir"])
    table_dir = analysis / "tables"
    full_rows = read_csv(resolve(cfg["full_projection_table"]))
    failures = [row for row in full_rows if row["usable"] != "True"]
    if len(failures) != 27:
        raise RuntimeError(f"expected the frozen 27 unresolved projections, found {len(failures)}")
    payloads = [{
        "config": cfg["_config_path"],
        "design_index": int(row["design_index"]),
        "source_time_index": int(row["source_time_index"]),
    } for row in failures]
    context = mp.get_context("spawn")
    results: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=int(cfg["adaptive_iprojection"]["workers"]), mp_context=context
    ) as pool:
        futures = [pool.submit(solve_case, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[phase2f adaptive] {completed}/27 design={result['design_index']} "
                f"time={result['source_time_index']} class={result['classification']}",
                flush=True,
            )
    results.sort(key=lambda row: (row["design_index"], row["source_time_index"]))
    level_rows = [row for result in results for row in result["rows"]]
    summary_rows = [{key: value for key, value in result.items() if key != "rows"} for result in results]
    write_csv(table_dir / "adaptive_iprojection_27cases.csv", level_rows)
    write_csv(table_dir / "adaptive_iprojection_27case_summary.csv", summary_rows)

    result_by_key = {
        (result["design_index"], result["source_time_index"]): result for result in results
    }
    all_designs = sorted({int(row["design_index"]) for row in full_rows})
    admissibility_rows: list[dict] = []
    for design in all_designs:
        rows = [row for row in full_rows if int(row["design_index"]) == design]
        unresolved = [row for row in rows if row["usable"] != "True"]
        repaired = [
            result_by_key[(design, int(row["source_time_index"]))] for row in unresolved
        ]
        accepted = all(result["resolved"] for result in repaired)
        exclusions = [result for result in repaired if not result["resolved"]]
        admissibility_rows.append({
            "design_index": design, "design_id": rows[0]["design_id"],
            "style": rows[0]["style"], "frozen_evaluation_time_count": len(rows),
            "original_unresolved_time_count": len(unresolved),
            "adaptive_resolved_time_count": sum(result["resolved"] for result in repaired),
            "numerically_admissible": accepted,
            "excluded_source_time_indices": ";".join(
                str(result["source_time_index"]) for result in exclusions
            ),
            "excluded_days": ";".join(str(result.get("day", "")) for result in exclusions),
            "exclusion_reasons": ";".join(result["classification"] for result in exclusions),
        })
    write_csv(table_dir / "numerical_admissible_layouts.csv", admissibility_rows)
    resolved_count = sum(result["resolved"] for result in results)
    admissible_count = sum(row["numerically_admissible"] for row in admissibility_rows)
    write_json(table_dir / "phase2f_numerical_admissibility_summary.json", {
        "attempted_projection_count": len(results),
        "resolved_projection_count": resolved_count,
        "numerically_unresolved_projection_count": len(results) - resolved_count,
        "implementation_failure_count": sum(not result["ok"] for result in results),
        "numerically_admissible_layout_count": admissible_count,
        "excluded_layout_count": len(admissibility_rows) - admissible_count,
        "excluded_design_ids": [
            row["design_id"] for row in admissibility_rows if not row["numerically_admissible"]
        ],
        "validation_risk_inspected_before_freeze": False,
        "final_test_artifact_loaded": False,
        "config_sha256": sha256(cfg["_config_path"]),
    })
    print(
        f"[phase2f adaptive] resolved={resolved_count}/27; "
        f"numerically admissible layouts={admissible_count}/512",
        flush=True,
    )


if __name__ == "__main__":
    main()
