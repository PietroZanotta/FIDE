from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from bounded_reference import BoxTransformedReferenceFlow, train_box_reference_flow
from common import artifact_dirs, config_hash, experiment_source_hash, fingerprint, load_config, write_json_atomic
from mfsi.cache import file_sha256
from mfsi.flow_matching import FlowMatchingConfig
from mfsi.reference import save_npz_checkpoint

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class EndpointOnlySource:
    x0: jax.Array
    x1: jax.Array

    def sample(self, key, n: int, endpoint: int):
        if endpoint not in (0, 1):
            raise ValueError("only endpoints 0 and 1 are available")
        values = self.x0 if endpoint == 0 else self.x1
        indices = jax.random.randint(key, (int(n),), 0, len(values))
        return values[indices]


def _training_config(cfg: dict) -> FlowMatchingConfig:
    c = cfg["reference_training"]
    return FlowMatchingConfig(
        seed=int(cfg["seed"]),
        hidden_width=int(c["hidden_width"]),
        hidden_layers=int(c["hidden_layers"]),
        train_steps=int(c["train_steps"]),
        batch_size=int(c["batch_size"]),
        learning_rate=float(c["learning_rate"]),
        min_learning_rate_ratio=float(c["min_learning_rate_ratio"]),
        adam_beta1=float(c["adam_beta1"]),
        adam_beta2=float(c["adam_beta2"]),
        adam_eps=float(c["adam_eps"]),
        grad_clip_norm=float(c["grad_clip_norm"]),
        bridge_schedule=str(c["bridge_schedule"]),
        bridge_noise_std=float(c["bridge_noise_std"]),
        log_every=int(c["log_every"]),
    )


def train_and_rollout(cfg: dict, output_dir: str | Path) -> dict:
    dirs = artifact_dirs(output_dir)
    dirs["endpoint"].mkdir(parents=True, exist_ok=True)
    endpoint_path = dirs["endpoint"] / "endpoint_data.npz"
    if not endpoint_path.exists():
        raise FileNotFoundError("build prospective endpoint data before training the reference")
    checkpoint = dirs["endpoint"] / "reference_checkpoint.npz"
    rollout_path = dirs["endpoint"] / "reference_rollout.npz"
    receipt_path = dirs["endpoint"] / "reference_receipt.json"
    endpoint_sha = file_sha256(endpoint_path)
    signature = fingerprint({
        "schema": 1,
        "config_hash": config_hash(cfg),
        "experiment_source_hash": experiment_source_hash(),
        "endpoint_sha256": endpoint_sha,
        "training": asdict(_training_config(cfg)),
        "reference": cfg["reference"],
        "scientific_nodes": cfg["time"]["scientific_nodes"],
    })
    if checkpoint.exists() and rollout_path.exists() and receipt_path.exists():
        import json
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("signature") == signature:
            print("[reference] reusing endpoint-trained checkpoint and rollout", flush=True)
            return receipt

    started = time.perf_counter()
    with np.load(endpoint_path, allow_pickle=False) as data:
        if str(np.asarray(data["role"]).item()) != "endpoint_only_reference_training":
            raise ValueError("reference training input is not endpoint-only")
        x0 = np.asarray(data["x0"], dtype=np.float64)
        x1 = np.asarray(data["x1"], dtype=np.float64)
        forbidden = set(data.files) - {"x0", "x1", "role", "config_hash"}
        if forbidden:
            raise ValueError(f"unexpected endpoint training arrays: {sorted(forbidden)}")
    source = EndpointOnlySource(jnp.asarray(x0), jnp.asarray(x1))
    ref_cfg = cfg["reference"]
    flow, history = train_box_reference_flow(
        source,
        _training_config(cfg),
        substeps_per_interval=int(ref_cfg["rk4_substeps_per_interval"]),
    )
    metadata = dict(flow.metadata or {})
    metadata.update({
        "experiment": "vortices_prospective",
        "scientific_role": "frozen_endpoint_only_reference",
        "endpoint_sha256": endpoint_sha,
        "signature": signature,
        "training_history": history,
    })
    save_npz_checkpoint(checkpoint, flow.params, metadata)
    flow = BoxTransformedReferenceFlow.from_npz(
        checkpoint, substeps_per_interval=int(ref_cfg["rk4_substeps_per_interval"])
    )

    rng = np.random.default_rng(int(cfg["seed"]) + int(ref_cfg["seed_offset"]))
    indices = rng.integers(0, len(x0), size=int(ref_cfg["particles"]))
    reference_x0 = jnp.asarray(x0[indices], dtype=jnp.float64)
    times = jnp.linspace(0.0, 1.0, int(cfg["time"]["scientific_nodes"]), dtype=jnp.float64)
    nodes = flow.rollout(reference_x0, times)
    velocity = jax.vmap(lambda t, x: flow.velocity(x, t))(times, nodes)
    weights = jnp.full(nodes.shape[:2], 1.0 / nodes.shape[1], dtype=jnp.float64)
    np.savez_compressed(
        rollout_path,
        role=np.asarray("frozen_endpoint_only_reference_rollout"),
        signature=np.asarray(signature), times=np.asarray(times),
        nodes=np.asarray(nodes), velocity=np.asarray(velocity), weights=np.asarray(weights),
    )
    receipt = {
        "schema_version": 1,
        "signature": signature,
        "config_hash": config_hash(cfg),
        "endpoint_sha256": endpoint_sha,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "rollout": str(rollout_path.resolve()),
        "rollout_sha256": file_sha256(rollout_path),
        "training_inputs": ["x0", "x1"],
        "intermediate_target_states_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(train_and_rollout(load_config(args.config), args.output_dir))


if __name__ == "__main__":
    main()
