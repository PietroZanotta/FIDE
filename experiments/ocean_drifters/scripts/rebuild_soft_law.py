#!/usr/bin/env python3
"""Rebuild ocean validation-law embeddings with the finite-sample projection.

This is an ocean-only migration utility.  It reads inference trajectories and the
already frozen RFF estimator, never loads the locked final-test cohort, and writes
versioned artifacts so the exact-moment result remains available for provenance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mfsi.config import load_config  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import (  # noqa: E402
    solve_soft_i_projection_trajectory_tesseract_forward,
)
from experiments.ocean_drifters.action import (  # noqa: E402
    _features,
    _positive_kernel_reconstruct,
)
from experiments.ocean_drifters.experiment import (  # noqa: E402
    OceanDriftersExperiment,
    _cell_centres,
    _rff_map,
)


def _statistics(
    sample_features: np.ndarray,
    days: np.ndarray,
    bandwidth_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = sample_features.shape[0]
    raw_moment = sample_features.mean(axis=0)
    raw_second = np.einsum(
        "ntm,ntk->tmk", sample_features, sample_features
    ) / count
    target, _, _ = _positive_kernel_reconstruct(
        raw_moment[None], days, days, bandwidth_days
    )
    second, _, _ = _positive_kernel_reconstruct(
        raw_second.reshape((1, len(days), 16)), days, days, bandwidth_days
    )
    target = target[0]
    second = second[0].reshape((-1, 4, 4))
    covariance_of_mean = (
        second - np.einsum("ti,tj->tij", target, target)
    ) / (count - 1)
    penalty = covariance_of_mean + (1.0 / count) ** 2 * np.eye(4)
    penalty = 0.5 * (penalty + penalty.swapaxes(-1, -2))
    return target, penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "experiments/ocean_drifters/config.json")
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    experiment = OceanDriftersExperiment(cfg)
    started = time.perf_counter()

    old_path = experiment.paths["risk_projection_embeddings"]
    with np.load(old_path, allow_pickle=False) as old:
        evaluation_indices = np.asarray(old["evaluation_indices"], dtype=int)
        evaluation_days = np.asarray(old["evaluation_days"], dtype=np.float64)
        omega = np.asarray(old["rff_omega"], dtype=np.float64)
        phase = np.asarray(old["rff_phase"], dtype=np.float64)
        bandwidth_km = float(old["bandwidth_km"])

    with np.load(
        experiment._resolve(
            "experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"
        ),
        allow_pickle=False,
    ) as moment_cache:
        metadata = json.loads(
            str(np.asarray(moment_cache["__metadata_json__"]).item())
        )
    bandwidth_days = float(metadata["bandwidths_days"]["nominal"])
    times = np.asarray(experiment.cohort.normalized_time, dtype=np.float64)
    days = times * float(cfg["scientific"]["horizon_days"])
    inference = np.asarray(experiment.cohort.inference, dtype=np.float64)

    nx, ny = (int(value) for value in cfg["projection"]["grid_resolution"])
    points = _cell_centres(
        np.asarray(cfg["scientific"]["domain_km"], dtype=np.float64), nx, ny
    )
    checkpoint = cfg["artifacts"]["reference_checkpoint"]["sha256"][:12]
    log_base = np.empty((len(evaluation_indices), len(points)), dtype=np.float64)
    for local, source in enumerate(evaluation_indices):
        path = experiment.reference_density_cache / (
            f"density_{checkpoint}_t{int(source):03d}_{nx}x{ny}.npz"
        )
        with np.load(path, allow_pickle=False) as data:
            log_base[local] = np.asarray(data["log_base_mass"], dtype=np.float64)

    p = cfg["projection"]
    solver = IProjectionConfig(
        max_steps=int(p["max_steps"]),
        residual_tol=float(cfg["action"]["soft_moment_projection"][
            "stationarity_residual_tolerance"
        ]),
        newton_ridge=float(p["newton_ridge"]),
        step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]),
        line_search_steps=int(p["line_search_steps"]),
        implicit_ridge=0.0,
    )
    design_count = len(experiment.sensor_bank.centers_km)
    time_count = len(evaluation_indices)
    lambdas = np.empty((design_count, time_count, 4), dtype=np.float64)
    stationarity = np.empty((design_count, time_count), dtype=np.float64)
    hard_residual = np.empty_like(stationarity)
    penalty_trace = np.empty_like(stationarity)
    targets = np.empty((design_count, time_count, 4), dtype=np.float64)
    penalties = np.empty((design_count, time_count, 4, 4), dtype=np.float64)

    for design in range(design_count):
        centers = experiment.sensor_bank.centers_km[design]
        sample_features = _features(
            inference.reshape((-1, 2)), centers, experiment.sensor_bank.sigma_km
        ).reshape((len(inference), len(times), 4))
        target_all, penalty_all = _statistics(
            sample_features, days, bandwidth_days
        )
        target = target_all[evaluation_indices]
        penalty = penalty_all[evaluation_indices]
        phi = _features(points, centers, experiment.sensor_bank.sigma_km)
        native = solve_soft_i_projection_trajectory_tesseract_forward(
            np.ascontiguousarray(
                np.broadcast_to(phi, (time_count, *phi.shape))
            ),
            log_base,
            target[None],
            penalty[None],
            solver,
        )
        lambdas[design] = native["lambda_values"][0]
        stationarity[design] = native["residual_norm"][0]
        hard_residual[design] = native["hard_moment_residual_norm"][0]
        penalty_trace[design] = np.trace(penalty, axis1=1, axis2=2)
        targets[design] = target
        penalties[design] = penalty
        if design == 0 or (design + 1) % 16 == 0 or design + 1 == design_count:
            print(
                f"[soft law] projections {design + 1}/{design_count}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    tolerance = float(cfg["action"]["soft_moment_projection"][
        "stationarity_residual_tolerance"
    ])
    if not np.all(stationarity <= tolerance):
        worst = np.unravel_index(np.argmax(stationarity), stationarity.shape)
        raise RuntimeError(
            f"soft projection failed at design/time {worst}: "
            f"{stationarity[worst]:.3e} > {tolerance:.3e}"
        )

    grid_rff = _rff_map(points, omega, phase)
    projected = np.empty(
        (design_count, time_count, omega.shape[1]), dtype=np.float32
    )
    batch_size = int(args.embedding_batch_size)
    for local_time in range(time_count):
        for start in range(0, design_count, batch_size):
            stop = min(start + batch_size, design_count)
            weights = np.empty((stop - start, len(points)), dtype=np.float32)
            for row, design in enumerate(range(start, stop)):
                phi = _features(
                    points,
                    experiment.sensor_bank.centers_km[design],
                    experiment.sensor_bank.sigma_km,
                )
                logits = log_base[local_time] + phi @ lambdas[design, local_time]
                logits -= logsumexp(logits)
                weights[row] = np.exp(logits).astype(np.float32)
            projected[start:stop, local_time] = weights @ grid_rff
        print(
            f"[soft law] embeddings {local_time + 1}/{time_count}; "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    validation = np.asarray(
        experiment.cohort.validation[:, evaluation_indices], dtype=np.float64
    )
    validation_features = _rff_map(validation, omega, phase)
    validation_embedding = validation_features.mean(axis=0, dtype=np.float64)
    difference = projected.astype(np.float64) - validation_embedding[None]
    risk_by_time = np.sum(difference * difference, axis=-1)
    risks = risk_by_time.mean(axis=1)
    rng = np.random.default_rng(int(cfg["law"]["bootstrap_seed"]))
    replicates = int(cfg["law"]["bootstrap_replicates"])
    bootstrap_ids = rng.integers(
        0, len(validation), size=(replicates, len(validation))
    )
    bootstrap_embedding = validation_features[bootstrap_ids].mean(
        axis=1, dtype=np.float64
    )
    projected_sq = np.sum(projected.astype(np.float64) ** 2, axis=-1)
    bootstrap_sq = np.sum(bootstrap_embedding * bootstrap_embedding, axis=-1)
    cross = np.einsum(
        "dtf,btf->dbt", projected.astype(np.float64), bootstrap_embedding,
        optimize=True,
    )
    bootstrap_risk = np.mean(
        projected_sq[:, None] + bootstrap_sq[None] - 2.0 * cross, axis=-1
    )
    best = int(np.argmin(risks))
    epsilon = float(np.std(bootstrap_risk[best], ddof=1))
    near = np.flatnonzero(risks <= risks[best] + epsilon)
    poor = int(np.argmax(risks))

    processed = ROOT / "experiments/ocean_drifters/processed"
    output = processed / "iprojection_soft_grid_validation_risk.npz"
    np.savez_compressed(
        output,
        evaluation_indices=evaluation_indices,
        evaluation_days=evaluation_days,
        design_id=experiment.sensor_bank.design_ids,
        eligible=np.ones(design_count, dtype=bool),
        risks=risks,
        risk_by_time=risk_by_time,
        projected_rff_embedding=projected,
        bootstrap_risk=bootstrap_risk,
        bandwidth_km=np.asarray(bandwidth_km),
        rff_omega=omega,
        rff_phase=phase,
        best_design_index=np.asarray(best),
        near_alternative_index=np.asarray(int(near[-1])),
        poor_design_index=np.asarray(poor),
        projection_method=np.asarray("soft_finite_sample_covariance_of_mean"),
        moment_bandwidth_days=np.asarray(bandwidth_days),
        final_test_accessed=np.asarray(False),
    )
    parameter_output = processed / "iprojection_soft_parameters.npz"
    np.savez_compressed(
        parameter_output,
        evaluation_indices=evaluation_indices,
        lambda_value=lambdas,
        target_moment=targets,
        penalty=penalties,
        penalty_trace=penalty_trace,
        stationarity_residual=stationarity,
        hard_moment_residual=hard_residual,
        final_test_accessed=np.asarray(False),
    )
    summary = {
        "schema_version": 1,
        "method": "soft finite-sample covariance-of-the-mean I-projection",
        "best_design_index": best,
        "best_design_id": str(experiment.sensor_bank.design_ids[best]),
        "best_validation_risk": float(risks[best]),
        "best_bootstrap_standard_error": epsilon,
        "additive_epsilon": epsilon,
        "near_optimal_layout_count": int(len(near)),
        "near_optimal_design_ids": experiment.sensor_bank.design_ids[near].tolist(),
        "maximum_stationarity_residual": float(np.max(stationarity)),
        "maximum_hard_moment_residual": float(np.max(hard_residual)),
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_accessed": False,
    }
    summary_path = (
        ROOT / "experiments/ocean_drifters/analysis/tables/soft_law_rebuild_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
