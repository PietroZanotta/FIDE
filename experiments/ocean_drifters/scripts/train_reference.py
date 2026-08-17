#!/usr/bin/env python3
"""Train and freeze the endpoint-only reference flow on inference IDs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from phase2_common import (  # noqa: E402
    EmpiricalEndpointSource,
    load_phase2_config,
    resolve,
    sha256,
    write_csv,
    write_json,
)

sys.path.insert(0, str(SCRIPT_DIR.parents[2] / "src"))
from mfsi.flow_matching import FlowMatchingConfig, train_reference_flow  # noqa: E402
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint  # noqa: E402

jax.config.update("jax_enable_x64", True)


def biased_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    kxx = np.exp(-cdist(x, x, "sqeuclidean") / (2 * bandwidth**2)).mean()
    kyy = np.exp(-cdist(y, y, "sqeuclidean") / (2 * bandwidth**2)).mean()
    kxy = np.exp(-cdist(x, y, "sqeuclidean") / (2 * bandwidth**2)).mean()
    return float(kxx + kyy - 2 * kxy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_phase2_config(args.config)
    processed = resolve(cfg["processed_dir"])
    model_dir = resolve(cfg["model_dir"])
    analysis = resolve(cfg["analysis_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    (analysis / "figures/reference").mkdir(parents=True, exist_ok=True)
    (analysis / "tables").mkdir(parents=True, exist_ok=True)
    dev_path = processed / "development_270.npz"
    if not dev_path.is_file():
        raise FileNotFoundError("Run freeze_cohort.py first")
    with np.load(dev_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        split = np.asarray(data["split"]).astype(str)
        times = np.asarray(data["normalized_time"], dtype=np.float64)
        days = np.asarray(data["relative_days"], dtype=np.float64)
        domain = np.asarray(data["domain_km"], dtype=np.float64)
    inf = X[split == "inference"]
    val = X[split == "validation"]
    assert inf.shape[0] == 200 and val.shape[0] == 70

    block = cfg["reference_training"]
    center = np.asarray(block["normalization_center_km"], dtype=np.float64)
    scale = float(block["normalization_scale_km"])
    x0 = (inf[:, 0] - center) / scale
    x1 = (inf[:, -1] - center) / scale
    train_cfg = FlowMatchingConfig(
        seed=int(block["seed"]), hidden_width=int(block["hidden_width"]),
        hidden_layers=int(block["hidden_layers"]), train_steps=int(block["train_steps"]),
        batch_size=int(block["batch_size"]), learning_rate=float(block["learning_rate"]),
        min_learning_rate_ratio=float(block["min_learning_rate_ratio"]),
        adam_beta1=float(block["adam_beta1"]), adam_beta2=float(block["adam_beta2"]),
        adam_eps=float(block["adam_eps"]), grad_clip_norm=float(block["grad_clip_norm"]),
        bridge_schedule=str(block["bridge_schedule"]),
        bridge_noise_std=float(block["bridge_noise_std_normalized"]),
        log_every=int(block["log_every"]),
    )
    checkpoint = model_dir / "reference.npz"
    development_sha = sha256(dev_path)
    signature_payload = {
        "experiment": cfg["name"], "development_sha256": development_sha,
        "training": asdict(train_cfg), "normalization_center_km": center.tolist(),
        "normalization_scale_km": scale,
    }
    signature = json.dumps(signature_payload, sort_keys=True)
    flow = None; history = None
    if checkpoint.exists() and not args.force:
        candidate = MLPReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        if (candidate.metadata or {}).get("training_signature") == signature:
            flow = candidate
            history = list((candidate.metadata or {}).get("history", []))
            print("[reference] reusing compatible checkpoint", flush=True)
    if flow is None:
        source = EmpiricalEndpointSource(jnp.asarray(x0), jnp.asarray(x1))
        started = time.perf_counter()
        flow, history = train_reference_flow(
            source, train_cfg,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        elapsed = time.perf_counter() - started
        metadata = dict(flow.metadata or {})
        metadata.update({
            "experiment": cfg["name"], "endpoint_data": "inference IDs only",
            "training_signature": signature, "normalization_center_km": center.tolist(),
            "normalization_scale_km": scale, "history": history,
            "training_seconds": elapsed,
        })
        flow = MLPReferenceFlow(
            flow.params,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
            metadata=metadata,
        )
        save_npz_checkpoint(checkpoint, flow.params, metadata)
        print(f"[reference] trained in {elapsed:.1f}s", flush=True)
    write_csv(analysis / "tables/reference_training_history.csv", history)

    bank_path = model_dir / "reference_bank.npz"
    bank_signature = json.dumps({
        "checkpoint_sha256": sha256(checkpoint), "particles": cfg["reference"]["particles"],
        "seed": int(cfg["seed"]) + int(cfg["reference"]["bank_seed_offset"]),
        "times": times.tolist(),
    }, sort_keys=True)
    nodes_km = velocity_km = initial_indices = None
    if bank_path.exists() and not args.force:
        with np.load(bank_path, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == bank_signature:
                nodes_km = np.asarray(cached["nodes_km"], dtype=np.float64)
                velocity_km = np.asarray(cached["velocity_km_per_normalized_time"], dtype=np.float64)
                initial_indices = np.asarray(cached["initial_inference_indices"], dtype=np.int32)
                print("[reference] reusing compatible reference bank", flush=True)
    if nodes_km is None:
        rng = np.random.default_rng(int(cfg["seed"]) + int(cfg["reference"]["bank_seed_offset"]))
        particle_n = int(cfg["reference"]["particles"])
        repetitions, remainder = divmod(particle_n, len(x0))
        initial_indices = np.tile(np.arange(len(x0), dtype=np.int32), repetitions)
        if remainder:
            initial_indices = np.concatenate([
                initial_indices, rng.permutation(len(x0))[:remainder].astype(np.int32)
            ])
        rng.shuffle(initial_indices)
        initial = jnp.asarray(x0[initial_indices], dtype=jnp.float64)
        print(f"[reference] rolling {len(initial_indices)} particles over {len(times)} times", flush=True)
        nodes_normalized = flow.rollout(initial, jnp.asarray(times, dtype=jnp.float64))
        velocity_normalized = jax.vmap(lambda t, x: flow.velocity(x, t))(
            jnp.asarray(times, dtype=jnp.float64), nodes_normalized
        )
        nodes_km = np.asarray(nodes_normalized) * scale + center
        velocity_km = np.asarray(velocity_normalized) * scale
        np.savez_compressed(
            bank_path, nodes_km=nodes_km, velocity_km_per_normalized_time=velocity_km,
            weights=np.full((len(times), len(initial_indices)), 1 / len(initial_indices)),
            times=times, relative_days=days, initial_inference_indices=initial_indices,
            signature=np.asarray(bank_signature), checkpoint_sha256=np.asarray(sha256(checkpoint)),
        )
    assert nodes_km.shape == (181, int(cfg["reference"]["particles"]), 2)
    assert np.all(np.isfinite(nodes_km)) and np.all(np.isfinite(velocity_km))

    rng = np.random.default_rng(int(cfg["seed"]) + 9001)
    bandwidth_points = np.concatenate([inf[:, ::10].reshape(-1, 2), val[:, ::10].reshape(-1, 2)])
    if len(bandwidth_points) > 3000:
        bandwidth_points = bandwidth_points[rng.choice(len(bandwidth_points), 3000, replace=False)]
    bandwidth = float(np.median(pdist(bandwidth_points)))
    snapshot_days = [0, 5, 10, 20, 30, 45]
    snapshot_indices = [day * 4 for day in snapshot_days]
    reference_rows = []
    sample_indices = rng.choice(nodes_km.shape[1], min(1024, nodes_km.shape[1]), replace=False)
    for day, index in zip(snapshot_days, snapshot_indices, strict=True):
        generated = nodes_km[index, sample_indices]
        reference_rows.append({
            "day": day, "mmd2_to_inference_empirical": biased_mmd2(generated, inf[:, index], bandwidth),
            "mmd2_to_validation_empirical": biased_mmd2(generated, val[:, index], bandwidth),
            "reference_mean_x_km": generated[:, 0].mean(), "reference_mean_y_km": generated[:, 1].mean(),
            "reference_spread_km": np.median(np.linalg.norm(generated - np.median(generated, axis=0), axis=1)),
        })
    write_csv(analysis / "tables/reference_validation.csv", reference_rows)
    outside = (
        (nodes_km[..., 0] < domain[0]) | (nodes_km[..., 0] > domain[1])
        | (nodes_km[..., 1] < domain[2]) | (nodes_km[..., 1] > domain[3])
    )
    diagnostics = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "reference_bank": str(bank_path), "reference_particles": nodes_km.shape[1],
        "kernel_bandwidth_km_for_diagnostics": bandwidth,
        "outside_domain_observation_fraction": float(outside.mean()),
        "particles_ever_outside_domain_fraction": float(outside.any(axis=0).mean()),
        "reference_xmin_km": float(nodes_km[..., 0].min()),
        "reference_xmax_km": float(nodes_km[..., 0].max()),
        "reference_ymin_km": float(nodes_km[..., 1].min()),
        "reference_ymax_km": float(nodes_km[..., 1].max()),
        "max_speed_km_per_normalized_time": float(np.linalg.norm(velocity_km, axis=-1).max()),
        "all_finite": True, "endpoint_only_training": True,
        "intermediate_noaa_positions_used_for_training": False,
        "final_test_artifact_loaded": False,
        "snapshot_metrics": reference_rows,
    }
    write_json(analysis / "tables/reference_diagnostics.json", diagnostics)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, day, index in zip(axes.ravel(), snapshot_days, snapshot_indices, strict=True):
        ax.scatter(nodes_km[index, sample_indices, 0], nodes_km[index, sample_indices, 1], s=6, alpha=.22, label="reference", color="#4c78a8")
        ax.scatter(val[:, index, 0], val[:, index, 1], s=14, alpha=.65, label="validation", color="#e45756")
        ax.set_title(f"day {day}\nMMD²={reference_rows[snapshot_days.index(day)]['mmd2_to_validation_empirical']:.4f}")
        ax.set_aspect("equal"); ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8); fig.supxlabel("x (km)"); fig.supylabel("y (km)")
    fig.suptitle("Endpoint-trained reference vs held-out validation law")
    fig.tight_layout(); fig.savefig(analysis / "figures/reference/reference_vs_empirical.png", dpi=190); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, index, label in zip(axes, [0, -1, -1], ["inference day 0", "inference day 45", "validation day 45"], strict=True):
        empirical = inf[:, index] if "inference" in label else val[:, index]
        generated = nodes_km[index, sample_indices]
        ax.scatter(empirical[:, 0], empirical[:, 1], s=15, alpha=.6, label="empirical")
        ax.scatter(generated[:, 0], generated[:, 1], s=5, alpha=.18, label="generated")
        ax.set_title(label); ax.set_aspect("equal"); ax.grid(alpha=.2)
    axes[0].legend(); fig.supxlabel("x (km)"); fig.supylabel("y (km)")
    fig.suptitle("Reference endpoint fit")
    fig.tight_layout(); fig.savefig(analysis / "figures/reference/endpoint_fit.png", dpi=190); plt.close(fig)

    path_count = min(180, nodes_km.shape[1])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i in range(path_count):
        ax.plot(nodes_km[:, i, 0], nodes_km[:, i, 1], lw=.55, alpha=.16, color="#2a6fbb")
    ax.set_aspect("equal"); ax.grid(alpha=.2); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("Frozen endpoint-trained reference trajectories")
    fig.tight_layout(); fig.savefig(analysis / "figures/reference/reference_paths.png", dpi=190); plt.close(fig)
    print(f"[reference] checkpoint={checkpoint}; diagnostic bandwidth={bandwidth:.1f} km", flush=True)


if __name__ == "__main__":
    main()
