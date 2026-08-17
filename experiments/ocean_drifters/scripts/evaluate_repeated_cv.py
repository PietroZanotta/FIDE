#!/usr/bin/env python3
"""Repeated ID-level 200/70 development-fold robustness evaluation.

The primary repeat is the frozen design split. Repeats 1 and 2 retrain the
endpoint-only reference and recompute measurements using their own 200 IDs.
The sensor bank, sigma, MMD kernel, and final 69-ID test lock remain frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_PROC_BIND", "close")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_iprojection import evaluate_one  # noqa: E402
from phase2_common import (  # noqa: E402
    EmpiricalEndpointSource,
    load_phase2_config,
    resolve,
    rff_map,
    sha256,
    write_csv,
    write_json,
)

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.flow_matching import FlowMatchingConfig, train_reference_flow  # noqa: E402
from mfsi.projection import IProjectionConfig  # noqa: E402
from mfsi.projection_tesseract import is_tesseract_iprojection_available  # noqa: E402
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint  # noqa: E402

jax.config.update("jax_enable_x64", True)


def read_fold_roles(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(int(row["repeat"]), {})[str(row["drifter_id"])] = row["role"]
    return result


def flow_config(block: dict, repeat: int) -> FlowMatchingConfig:
    return FlowMatchingConfig(
        seed=int(block["seed"]) + repeat,
        hidden_width=int(block["hidden_width"]),
        hidden_layers=int(block["hidden_layers"]),
        train_steps=int(block["train_steps"]),
        batch_size=int(block["batch_size"]),
        learning_rate=float(block["learning_rate"]),
        min_learning_rate_ratio=float(block["min_learning_rate_ratio"]),
        adam_beta1=float(block["adam_beta1"]),
        adam_beta2=float(block["adam_beta2"]),
        adam_eps=float(block["adam_eps"]),
        grad_clip_norm=float(block["grad_clip_norm"]),
        bridge_schedule=str(block["bridge_schedule"]),
        bridge_noise_std=float(block["bridge_noise_std_normalized"]),
        log_every=int(block["log_every"]),
    )


def train_and_rollout(
    cfg: dict,
    repeat: int,
    inference: np.ndarray,
    evaluation_times: np.ndarray,
    fold_dir: Path,
    force: bool,
) -> tuple[np.ndarray, str, float]:
    block = cfg["reference_training"]
    center = np.asarray(block["normalization_center_km"], dtype=np.float64)
    scale = float(block["normalization_scale_km"])
    train_cfg = flow_config(block, repeat)
    x0 = (inference[:, 0] - center) / scale
    x1 = (inference[:, -1] - center) / scale
    signature = json.dumps({
        "repeat": repeat,
        "inference_endpoint_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(inference[:, [0, -1]]).tobytes()
        ).hexdigest(),
        "training": asdict(train_cfg),
        "center": center.tolist(),
        "scale": scale,
        "evaluation_times": evaluation_times.tolist(),
        "particles": int(cfg["reference"]["particles"]),
    }, sort_keys=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = fold_dir / "reference.npz"
    bank_path = fold_dir / "reference_bank_evaluation_times.npz"
    flow = None
    if checkpoint.exists() and not force:
        candidate = MLPReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        if (candidate.metadata or {}).get("cv_signature") == signature:
            flow = candidate
    training_seconds = 0.0
    if flow is None:
        started = time.perf_counter()
        flow, history = train_reference_flow(
            EmpiricalEndpointSource(jnp.asarray(x0), jnp.asarray(x1)),
            train_cfg,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        training_seconds = time.perf_counter() - started
        metadata = dict(flow.metadata or {})
        metadata.update({
            "cv_signature": signature,
            "repeat": repeat,
            "endpoint_data": "this repeat's inference IDs only",
            "intermediate_positions_used_for_training": False,
            "history": history,
            "training_seconds": training_seconds,
        })
        flow = MLPReferenceFlow(
            flow.params,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
            metadata=metadata,
        )
        save_npz_checkpoint(checkpoint, flow.params, metadata)
    nodes = None
    if bank_path.exists() and not force:
        with np.load(bank_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == signature:
                nodes = np.asarray(cached["nodes_km"], dtype=np.float64)
    if nodes is None:
        particle_n = int(cfg["reference"]["particles"])
        repetitions, remainder = divmod(particle_n, len(x0))
        indices = np.tile(np.arange(len(x0), dtype=np.int32), repetitions)
        rng = np.random.default_rng(int(cfg["seed"]) + repeat * 10000 + int(cfg["reference"]["bank_seed_offset"]))
        if remainder:
            indices = np.r_[indices, rng.permutation(len(x0))[:remainder]]
        rng.shuffle(indices)
        normalized = flow.rollout(jnp.asarray(x0[indices]), jnp.asarray(evaluation_times))
        nodes = np.asarray(normalized) * scale + center
        np.savez_compressed(
            bank_path,
            nodes_km=nodes,
            initial_inference_indices=indices,
            evaluation_times=evaluation_times,
            signature=np.asarray(signature),
            checkpoint_sha256=np.asarray(sha256(checkpoint)),
        )
    return nodes, sha256(checkpoint), training_seconds


def fold_measurements(
    inference: np.ndarray,
    centers: np.ndarray,
    sigma: float,
    evaluation_indices: np.ndarray,
) -> np.ndarray:
    points = inference[:, evaluation_indices]
    targets = np.empty((len(centers), len(evaluation_indices), 4), dtype=np.float64)
    for start in range(0, len(centers), 16):
        stop = min(start + 16, len(centers))
        delta = points[None, :, :, None, :] - centers[start:stop, None, None, :, :]
        targets[start:stop] = np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2).mean(axis=1)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    if not is_tesseract_iprojection_available():
        raise RuntimeError("native I-projection Tesseract is required for repeated CV")
    processed = resolve(cfg["processed_dir"])
    model_dir = resolve(cfg["model_dir"])
    analysis = resolve(cfg["analysis_dir"])
    table_dir = analysis / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with np.load(processed / "development_270.npz", allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        ids = np.asarray(data["ids"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
        days = np.asarray(data["relative_days"], dtype=np.float64)
    roles = read_fold_roles(processed / "splits/repeated_cv_manifest.csv")
    repeats = sorted(roles)
    expected_repeats = int(cfg["splits"]["repeated_cv_folds"])
    if repeats != list(range(expected_repeats)):
        raise RuntimeError(f"expected repeats 0..{expected_repeats - 1}, got {repeats}")
    if any(set(mapping) != set(ids) for mapping in roles.values()):
        raise RuntimeError("repeated-CV manifest does not exactly cover the 270 development IDs")
    with np.load(processed / "sensor_bank.npz", allow_pickle=False) as data:
        centers = np.asarray(data["centers_km"], dtype=np.float64)
        design_ids = np.asarray(data["design_id"]).astype(str)
        styles = np.asarray(data["style"]).astype(str)
        sigma = float(data["sigma_km"])
    with np.load(processed / "iprojection_primary.npz", allow_pickle=False) as data:
        evaluation_indices = np.asarray(data["evaluation_indices"], dtype=int)
        primary_risks = np.asarray(data["risks"], dtype=np.float64)
        primary_feasible = np.asarray(data["feasible"], dtype=bool)
        omega = np.asarray(data["rff_omega"], dtype=np.float64)
        phase = np.asarray(data["rff_phase"], dtype=np.float64)
        bandwidth = float(data["bandwidth_km"])
    p = cfg["projection"]
    projection_cfg = IProjectionConfig(
        max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
        newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
        lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
        implicit_ridge=float(p["implicit_ridge"]),
    )
    all_risks = [primary_risks]
    all_feasible = [primary_feasible]
    fold_rows = [{
        "repeat": 0, "inference_n": 200, "validation_n": 70,
        "feasible_designs": int(primary_feasible.sum()),
        "best_design_id": str(design_ids[np.flatnonzero(primary_feasible)[np.argmin(primary_risks[primary_feasible])]]),
        "best_validation_risk": float(primary_risks[primary_feasible].min()),
        "flow_retrained": False, "training_seconds": 0.0,
    }]
    for repeat in repeats[1:]:
        mapping = roles[repeat]
        inference_mask = np.asarray([mapping[value] == "inference" for value in ids])
        validation_mask = ~inference_mask
        if inference_mask.sum() != 200 or validation_mask.sum() != 70:
            raise RuntimeError(f"repeat {repeat} is not an ID-level 200/70 split")
        inference = X[inference_mask]
        validation = X[validation_mask]
        nodes, checkpoint_hash, training_seconds = train_and_rollout(
            cfg, repeat, inference, times[evaluation_indices],
            model_dir / f"cv_repeat_{repeat}", args.force,
        )
        targets = fold_measurements(inference, centers, sigma, evaluation_indices)
        weights = np.empty((len(centers), len(evaluation_indices), nodes.shape[1]), dtype=np.float32)
        diagnostics: dict[int, list[dict]] = {}
        started = time.perf_counter()
        print(f"[cv repeat {repeat}] native sweep {len(centers)} designs", flush=True)
        with ThreadPoolExecutor(max_workers=int(p["workers"])) as executor:
            futures = [
                executor.submit(
                    evaluate_one, i, centers[i], nodes, targets[i], sigma, projection_cfg, p
                ) for i in range(len(centers))
            ]
            completed = 0
            for future in as_completed(futures):
                i, _, w, rows = future.result()
                weights[i] = w
                diagnostics[i] = rows
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(f"[cv repeat {repeat}] {completed}/{len(futures)}", flush=True)
        native_seconds = time.perf_counter() - started
        feasible = np.asarray([
            all(row["valid"] for row in diagnostics[i]) for i in range(len(centers))
        ], dtype=bool)
        reference_rff = np.stack([rff_map(points, omega, phase) for points in nodes])
        validation_embedding = np.stack([
            rff_map(validation[:, index], omega, phase).mean(axis=0)
            for index in evaluation_indices
        ])
        projected_embedding = np.asarray(jnp.einsum(
            "dtn,tnf->dtf", jnp.asarray(weights), jnp.asarray(reference_rff)
        ))
        risks = np.mean(np.sum(
            (projected_embedding - validation_embedding[None]) ** 2, axis=-1
        ), axis=-1)
        all_risks.append(risks)
        all_feasible.append(feasible)
        per_design = []
        for i in range(len(centers)):
            rows = diagnostics[i]
            reasons: dict[str, int] = {}
            for row in rows:
                for reason in row["failure_reason"].split(";") if row["failure_reason"] else []:
                    reasons[reason] = reasons.get(reason, 0) + 1
            per_design.append({
                "repeat": repeat, "design_id": design_ids[i], "style": styles[i],
                "validation_mmd_risk": risks[i], "feasible": bool(feasible[i]),
                "projection_failure_count": sum(not row["valid"] for row in rows),
                "minimum_ess_fraction": min(row["ess_fraction"] for row in rows),
                "max_moment_residual": max(row["verified_moment_residual"] for row in rows),
                "failure_reason_counts": json.dumps(reasons, sort_keys=True),
            })
        write_csv(table_dir / f"repeated_cv_repeat_{repeat}_designs.csv", per_design)
        np.savez_compressed(
            processed / f"iprojection_cv_repeat_{repeat}.npz",
            risks=risks, feasible=feasible, design_id=design_ids,
            evaluation_indices=evaluation_indices, reference_checkpoint_sha256=checkpoint_hash,
            sensor_bank_sha256=sha256(processed / "sensor_bank.npz"),
            final_test_accessed=np.asarray(False),
        )
        if feasible.any():
            best = int(np.flatnonzero(feasible)[np.argmin(risks[feasible])])
            best_id = str(design_ids[best])
            best_risk = float(risks[best])
        else:
            best_id = "none"
            best_risk = math.nan
        fold_rows.append({
            "repeat": repeat, "inference_n": int(inference_mask.sum()),
            "validation_n": int(validation_mask.sum()),
            "feasible_designs": int(feasible.sum()), "best_design_id": best_id,
            "best_validation_risk": best_risk, "flow_retrained": True,
            "training_seconds": training_seconds, "native_sweep_seconds": native_seconds,
            "reference_checkpoint_sha256": checkpoint_hash,
        })
        print(
            f"[cv repeat {repeat}] feasible={feasible.sum()}/{len(feasible)} "
            f"best={best_id} risk={best_risk}", flush=True,
        )

    risk_matrix = np.stack(all_risks)
    feasible_matrix = np.stack(all_feasible)
    aggregate_rows = []
    for i in range(len(centers)):
        valid_values = risk_matrix[feasible_matrix[:, i], i]
        aggregate_rows.append({
            "design_id": design_ids[i], "style": styles[i],
            "feasible_repeat_count": int(feasible_matrix[:, i].sum()),
            "feasible_all_repeats": bool(feasible_matrix[:, i].all()),
            "mean_valid_fold_risk": float(valid_values.mean()) if len(valid_values) else math.nan,
            "std_valid_fold_risk": float(valid_values.std(ddof=1)) if len(valid_values) > 1 else math.nan,
            **{f"repeat_{r}_risk": risk_matrix[r, i] for r in repeats},
            **{f"repeat_{r}_feasible": bool(feasible_matrix[r, i]) for r in repeats},
        })
    write_csv(table_dir / "repeated_cv_design_summary.csv", sorted(
        aggregate_rows,
        key=lambda row: (-row["feasible_repeat_count"], row["mean_valid_fold_risk"]),
    ))
    stability_rows = []
    for left in repeats:
        for right in repeats[left + 1:]:
            joint = feasible_matrix[left] & feasible_matrix[right]
            rho = float(spearmanr(risk_matrix[left, joint], risk_matrix[right, joint]).statistic) if joint.sum() >= 3 else math.nan
            left_top = set(np.flatnonzero(feasible_matrix[left])[np.argsort(risk_matrix[left, feasible_matrix[left]])[:20]])
            right_top = set(np.flatnonzero(feasible_matrix[right])[np.argsort(risk_matrix[right, feasible_matrix[right]])[:20]])
            stability_rows.append({
                "repeat_left": left, "repeat_right": right,
                "jointly_feasible_designs": int(joint.sum()),
                "spearman_on_jointly_feasible": rho,
                "top20_feasible_overlap": len(left_top & right_top),
            })
    write_csv(table_dir / "repeated_cv_fold_summary.csv", fold_rows)
    write_csv(table_dir / "repeated_cv_rank_stability.csv", stability_rows)
    write_json(table_dir / "repeated_cv_summary.json", {
        "scheme": "270 development IDs; three repeated ID-level 200/70 splits; 69 final IDs untouched",
        "fold_count": len(repeats), "folds": fold_rows,
        "designs_feasible_all_repeats": int(feasible_matrix.all(axis=0).sum()),
        "designs_feasible_at_least_two_repeats": int((feasible_matrix.sum(axis=0) >= 2).sum()),
        "fixed_across_repeats": ["sensor bank", "sigma", "RBF bandwidth", "RFF features", "projection acceptance contract"],
        "fold_specific": ["200 inference IDs", "70 validation IDs", "endpoint reference flow", "measurement moments"],
        "bandwidth_km": bandwidth, "final_test_artifact_loaded": False,
        "manifest_sha256": sha256(processed / "splits/repeated_cv_manifest.csv"),
        "stability": stability_rows,
    })
    print(
        f"[cv] all-fold feasible={feasible_matrix.all(axis=0).sum()}, "
        f"at-least-two={((feasible_matrix.sum(axis=0) >= 2).sum())}", flush=True,
    )


if __name__ == "__main__":
    main()
