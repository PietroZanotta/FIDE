#!/usr/bin/env python3
"""Reference-only Phase 1 execution harness for the frozen Vortices V2 run."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


V2_DIR = Path(__file__).resolve().parent
REPO_ROOT = V2_DIR.parents[1]
V1_DIR = V2_DIR
for value in (REPO_ROOT / "src", V1_DIR, V2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

jax.config.update("jax_enable_x64", True)

from bounded_reference import (  # noqa: E402
    BOX_HIGH,
    BOX_LOW,
    TRANSFORM_VERSION,
    BoxTransformedReferenceFlow,
    train_box_reference_flow,
)
from core import frozen_reference_scott_bandwidth  # noqa: E402
from domain import EmpiricalEndpointSource  # noqa: E402
from experiment import _flow_matching_cfg, _truth_from_cfg  # noqa: E402
from mfsi.cache import fingerprint, save_npz_cache, write_json_atomic  # noqa: E402
from mfsi.reference import save_npz_checkpoint  # noqa: E402
from selection_contract import (  # noqa: E402
    canonical_json_sha256,
    load_selection_config,
    sha256_file,
    validate_selection_config,
)


SELECTION_CONFIG = V2_DIR / "VORTICES_V2_SELECTION_CONFIG.json"
NUMERICAL_CONFIG = V2_DIR / "config.json"
MANIFEST_PATH = V2_DIR / "VORTICES_V2_FREEZE_MANIFEST.json"
V1_CONFIG_PATH = V1_DIR / "base_experiment_config.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_endpoint_source(path: Path) -> tuple[EmpiricalEndpointSource, str]:
    with np.load(path, allow_pickle=False) as bank:
        if set(("x0", "x1", "__signature__")) - set(bank.files):
            raise RuntimeError("frozen endpoint dataset is incomplete")
        x0 = np.asarray(bank["x0"], dtype=np.float64)
        x1 = np.asarray(bank["x1"], dtype=np.float64)
        signature = str(np.asarray(bank["__signature__"]).item())
    if x0.shape != (50000, 2) or x1.shape != (50000, 2):
        raise RuntimeError(f"unexpected endpoint shapes: {x0.shape}, {x1.shape}")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(x1)):
        raise RuntimeError("nonfinite endpoint data")
    return EmpiricalEndpointSource(jnp.asarray(x0), jnp.asarray(x1)), signature


def build_v1_training_config(selection: dict[str, Any], seed: int) -> dict[str, Any]:
    frozen = selection["reference_replicates"]["training"]
    config = json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))
    block = config["reference_training"]
    block.update(
        {
            "seed": int(seed),
            "hidden_width": int(frozen["hidden_width"]),
            "hidden_layers": int(frozen["hidden_layers"]),
            "train_steps": int(frozen["train_steps"]),
            "batch_size": int(frozen["batch_size"]),
            "learning_rate": float(frozen["learning_rate"]),
            "min_learning_rate_ratio": float(frozen["minimum_learning_rate_ratio"]),
            "adam_beta1": float(frozen["adam_beta1"]),
            "adam_beta2": float(frozen["adam_beta2"]),
            "adam_eps": float(frozen["adam_epsilon"]),
            "grad_clip_norm": float(frozen["gradient_clip_norm"]),
            "bridge_schedule": str(frozen["bridge_schedule"]),
            "bridge_noise_std": float(frozen["bridge_noise_standard_deviation"]),
            "log_every": 500,
        }
    )
    config["reference"]["particles"] = int(
        selection["reference_replicates"]["rollout"]["particles_per_reference"]
    )
    config["reference"]["rk4_substeps_per_time_interval"] = int(
        selection["reference_replicates"]["rollout"][
            "rk4_substeps_per_scientific_interval"
        ]
    )
    return config


def source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    wanted = [
        "experiments/vortices_percentage/bounded_reference.py",
        "experiments/vortices_percentage/experiment.py",
        "src/mfsi/flow_matching.py",
        "src/mfsi/reference.py",
        "experiments/vortices_percentage/core.py",
        "experiments/vortices_percentage/VORTICES_V2_SELECTION_CONFIG.json",
        "experiments/vortices_percentage/VORTICES_V2_FREEZE_MANIFEST.json",
    ]
    result = {}
    for relative in wanted:
        path = REPO_ROOT / relative
        result[relative] = sha256_file(path)
    return result


def checkpoint_is_compatible(
    path: Path,
    *,
    seed: int,
    endpoint_signature: str,
    training_signature: str,
) -> tuple[BoxTransformedReferenceFlow, dict[str, Any]] | None:
    if not path.exists():
        return None
    flow = BoxTransformedReferenceFlow.from_npz(path, substeps_per_interval=16)
    metadata = dict(flow.metadata or {})
    required = (
        metadata.get("experiment") == "vortices_double_gyre"
        and metadata.get("endpoint_signature") == endpoint_signature
        and metadata.get("training_signature") == training_signature
        and metadata.get("training", {}).get("seed") == seed
        and metadata.get("training", {}).get("steps") == 12000
    )
    return (flow, metadata) if required else None


def execute(seed: int) -> dict[str, Any]:
    started_at = utc_now()
    selection = load_selection_config(SELECTION_CONFIG)
    validate_selection_config(selection)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    references = selection["reference_replicates"]
    expected_seeds = list(map(int, references["training_seeds"]))
    if seed not in expected_seeds:
        raise RuntimeError(f"seed {seed} is outside the frozen three-seed set")
    expected_rollout_seed = int(references["rollout"]["seeds"][expected_seeds.index(seed)])
    if expected_rollout_seed != seed + 3001:
        raise RuntimeError("frozen rollout seed rule is inconsistent")

    output_dir = (
        V2_DIR
        / "outputs"
        / "prospective_v2"
        / "references"
        / f"reference_seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "qualification_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(f"qualification receipt already exists: {receipt_path}")

    endpoint_path = V2_DIR / "inputs" / "reference_endpoints.npz"
    endpoint_hash = sha256_file(endpoint_path)
    if endpoint_hash != references["endpoint_dataset"]["sha256"]:
        raise RuntimeError("frozen endpoint dataset hash mismatch")
    source, endpoint_signature = load_endpoint_source(endpoint_path)

    config = build_v1_training_config(selection, seed)
    train_cfg = _flow_matching_cfg(config)
    complete_training = {
        "flow_matching_config": asdict(train_cfg),
        "coordinate_transform": {
            "identity": TRANSFORM_VERSION,
            "box_low": list(BOX_LOW),
            "box_high": list(BOX_HIGH),
        },
        "endpoint_dataset_sha256": endpoint_hash,
        "rollout": references["rollout"],
    }
    training_config_hash = canonical_json_sha256(complete_training)
    training_architecture_hash = canonical_json_sha256(references["training"])
    training_signature = fingerprint(asdict(train_cfg))
    write_json_atomic(output_dir / "training_config.json", complete_training)

    checkpoint_path = output_dir / "reference.npz"
    compatible = checkpoint_is_compatible(
        checkpoint_path,
        seed=seed,
        endpoint_signature=endpoint_signature,
        training_signature=training_signature,
    )
    if compatible is None:
        if checkpoint_path.exists():
            raise RuntimeError("existing checkpoint is incompatible; refusing to overwrite")
        print(f"[reference {seed}] training frozen endpoint-only model", flush=True)
        flow, history = train_box_reference_flow(
            source,
            train_cfg,
            substeps_per_interval=int(
                references["rollout"]["rk4_substeps_per_scientific_interval"]
            ),
        )
        metadata = dict(flow.metadata or {})
        metadata.update(
            {
                "experiment": "vortices_double_gyre",
                "endpoint_signature": endpoint_signature,
                "endpoint_data_sha256": endpoint_hash,
                "training_signature": training_signature,
                "training_config_sha256": training_config_hash,
                "training_architecture_sha256": training_architecture_hash,
                "history": history,
                "completed_at_utc": utc_now(),
            }
        )
        flow = BoxTransformedReferenceFlow(
            params=flow.params,
            substeps_per_interval=int(
                references["rollout"]["rk4_substeps_per_scientific_interval"]
            ),
            metadata=metadata,
            transform_eps=flow.transform_eps,
        )
        save_npz_checkpoint(checkpoint_path, flow.params, metadata)
    else:
        flow, metadata = compatible
        history = list(metadata.get("history", []))
        print(f"[reference {seed}] resuming compatible frozen checkpoint", flush=True)

    checkpoint_hash = sha256_file(checkpoint_path)
    atomic_json(output_dir / "training_history.json", {"history": history})

    rollout_path = output_dir / "reference_bank.npz"
    if rollout_path.exists():
        raise RuntimeError(f"rollout already exists without receipt: {rollout_path}")
    truth = _truth_from_cfg(config)
    n = int(references["rollout"]["particles_per_reference"])
    times = jnp.linspace(0.0, 1.0, 21, dtype=jnp.float64)
    x0 = jnp.asarray(truth.sample_initial_numpy(expected_rollout_seed, n), dtype=jnp.float64)
    print(f"[reference {seed}] rolling {n} particles with seed {expected_rollout_seed}", flush=True)
    nodes_jax = flow.rollout(x0, times)
    velocity_jax = jax.vmap(lambda t, x: flow.velocity(x, t))(times, nodes_jax)
    weights_jax = jnp.full((21, n), 1.0 / float(n), dtype=jnp.float64)
    nodes = np.asarray(nodes_jax, dtype=np.float64)
    velocity = np.asarray(velocity_jax, dtype=np.float64)
    weights = np.asarray(weights_jax, dtype=np.float64)
    rollout_signature = fingerprint(
        {
            "schema": 1,
            "checkpoint_sha256": checkpoint_hash,
            "rollout_seed": expected_rollout_seed,
            "particles": n,
            "times": np.asarray(times).tolist(),
            "rk4_substeps_per_interval": 16,
        }
    )
    save_npz_cache(
        rollout_path,
        {"times": times, "nodes": nodes, "velocity": velocity, "weights": weights},
        signature=rollout_signature,
        metadata={
            "role": "prospective_v2_frozen_reference_rollout",
            "training_seed": seed,
            "rollout_seed": expected_rollout_seed,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    rollout_hash = sha256_file(rollout_path)

    bandwidth, bandwidth_by_time = frozen_reference_scott_bandwidth(nodes, weights)
    weight_sum_errors = np.abs(np.sum(weights, axis=1) - 1.0)
    in_domain = (
        (nodes[..., 0] >= 0.0)
        & (nodes[..., 0] <= 2.0)
        & (nodes[..., 1] >= 0.0)
        & (nodes[..., 1] <= 1.0)
    )
    in_domain_mass = np.sum(np.where(in_domain, weights, 0.0), axis=1)
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(flow.params)]
    final = history[-1] if history else {}
    qualification = references["qualification"]
    checks = {
        "training_seed_exact": seed in expected_seeds,
        "transform_exact": metadata.get("vortex_reference_transform") == TRANSFORM_VERSION,
        "hidden_width_exact": metadata.get("network", {}).get("hidden_width") == 128,
        "hidden_layers_exact": metadata.get("network", {}).get("hidden_layers") == 4,
        "training_steps_metadata_exact": metadata.get("training", {}).get("steps") == 12000,
        "final_logged_step_exact": int(final.get("step", -1)) == 12000,
        "final_loss_finite_and_bounded": bool(
            np.isfinite(final.get("conditional_fm_loss", np.nan))
            and float(final["conditional_fm_loss"])
            <= float(qualification["maximum_final_conditional_fm_loss"])
        ),
        "final_gradient_finite_and_bounded": bool(
            np.isfinite(final.get("grad_norm_preclip", np.nan))
            and float(final["grad_norm_preclip"])
            <= float(qualification["maximum_final_preclip_gradient_norm"])
        ),
        "checkpoint_parameters_finite": bool(all(np.all(np.isfinite(x)) for x in leaves)),
        "nodes_shape_exact": list(nodes.shape) == [21, 32768, 2],
        "velocity_shape_exact": list(velocity.shape) == [21, 32768, 2],
        "weights_shape_exact": list(weights.shape) == [21, 32768],
        "nodes_and_velocity_finite": bool(np.all(np.isfinite(nodes)) and np.all(np.isfinite(velocity))),
        "weights_finite_nonnegative": bool(np.all(np.isfinite(weights)) and np.all(weights >= 0.0)),
        "weight_sums_within_tolerance": bool(
            np.max(weight_sum_errors) <= float(qualification["weight_sum_absolute_tolerance"])
        ),
        "minimum_in_domain_mass_pass": bool(
            np.min(in_domain_mass) >= float(qualification["minimum_in_domain_base_mass"])
        ),
        "all_scott_bandwidths_positive_finite": bool(
            np.all(np.isfinite(bandwidth_by_time)) and np.all(bandwidth_by_time > 0.0)
        ),
        "median_scott_bandwidth_positive_finite": bool(
            np.isfinite(bandwidth) and bandwidth > 0.0
        ),
    }
    qualified = bool(all(checks.values()))
    receipt = {
        "schema_version": 1,
        "status": "PASS" if qualified else "FAIL",
        "qualified": qualified,
        "training_seed": seed,
        "rollout_seed": expected_rollout_seed,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "freeze_manifest_path": str(MANIFEST_PATH.resolve()),
        "freeze_manifest_sha256": sha256_file(MANIFEST_PATH),
        "selection_config_path": str(SELECTION_CONFIG.resolve()),
        "selection_config_sha256": sha256_file(SELECTION_CONFIG),
        "training_config_path": str((output_dir / "training_config.json").resolve()),
        "training_config_sha256": training_config_hash,
        "training_architecture_sha256": training_architecture_hash,
        "numerical_method_config_sha256": sha256_file(NUMERICAL_CONFIG),
        "endpoint_data_path": str(endpoint_path.resolve()),
        "endpoint_data_sha256": endpoint_hash,
        "endpoint_signature": endpoint_signature,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "rollout_bank_path": str(rollout_path.resolve()),
        "rollout_bank_sha256": rollout_hash,
        "qualification_receipt_path": str(receipt_path.resolve()),
        "final_training_step": int(final.get("step", -1)),
        "final_conditional_fm_loss": float(final.get("conditional_fm_loss", np.nan)),
        "final_preclip_gradient_norm": float(final.get("grad_norm_preclip", np.nan)),
        "checkpoint_parameter_count": int(sum(x.size for x in leaves)),
        "checkpoint_parameters_all_finite": bool(checks["checkpoint_parameters_finite"]),
        "rollout_shape": list(nodes.shape),
        "velocity_shape": list(velocity.shape),
        "weight_shape": list(weights.shape),
        "maximum_weight_sum_absolute_error": float(np.max(weight_sum_errors)),
        "minimum_in_domain_base_mass": float(np.min(in_domain_mass)),
        "in_domain_base_mass_by_time": in_domain_mass.tolist(),
        "scott_bandwidth": float(bandwidth),
        "scott_bandwidth_by_time": bandwidth_by_time.tolist(),
        "qualification_checks": checks,
        "source_sha256": source_hashes(manifest),
        "forbidden_sensor_risk_action_inputs_used": False,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    receipt = execute(args.seed)
    raise SystemExit(0 if receipt["qualified"] else 2)


if __name__ == "__main__":
    main()
