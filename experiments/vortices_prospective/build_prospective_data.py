from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from aggregate_qois import qoi_features
from common import artifact_dirs, config_hash, experiment_source_hash, load_config, write_json_atomic
from physical import truth_from_config

jax.config.update("jax_enable_x64", True)


def _response_fields(states: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, width: float, chunk: int):
    # The isotropic Gaussian factorizes.  Two skinny kernel matrices plus one
    # matrix multiply avoid materializing [particle, y, x], which is decisive for
    # the production 50k-particle/129x65 table.
    mean_rows, second_rows = [], []
    for time_states in states:
        total = np.zeros((len(y_grid), len(x_grid)), dtype=np.float64)
        total2 = np.zeros_like(total)
        for start in range(0, len(time_states), int(chunk)):
            particles = time_states[start : start + int(chunk)]
            kx = np.exp(-0.5 * ((particles[:, 0, None] - x_grid[None, :]) / float(width)) ** 2)
            ky = np.exp(-0.5 * ((particles[:, 1, None] - y_grid[None, :]) / float(width)) ** 2)
            total += ky.T @ kx
            total2 += (ky * ky).T @ (kx * kx)
        mean_rows.append(total / len(time_states))
        second_rows.append(total2 / len(time_states))
    return np.stack(mean_rows), np.stack(second_rows)


def build(cfg: dict, output_dir: str | Path) -> dict:
    dirs = artifact_dirs(output_dir)
    dirs["endpoint"].mkdir(parents=True, exist_ok=True)
    dirs["prospective"].mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash(cfg)
    source_hash = experiment_source_hash()
    endpoint_path = dirs["endpoint"] / "endpoint_data.npz"
    aggregate_path = dirs["prospective"] / "aggregate_predictions.npz"
    receipt_path = dirs["prospective"] / "build_receipt.json"
    if endpoint_path.exists() and aggregate_path.exists() and receipt_path.exists():
        import json
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("config_hash") == cfg_hash and receipt.get("experiment_source_hash") == source_hash:
            print("[prospective-data] reusing endpoint and aggregate artifacts", flush=True)
            return receipt

    started = time.perf_counter()
    truth = truth_from_config(cfg)
    tcfg = cfg["truth"]
    seed = int(cfg["seed"])
    n_endpoint = int(tcfg["endpoint_particles"])
    endpoint_seed = seed + int(tcfg["endpoint_seed_offset"])
    x0 = truth.sample_initial_numpy(endpoint_seed, n_endpoint)
    endpoints = truth.rollout(
        jnp.asarray(x0), jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        substeps_per_interval=int(tcfg["endpoint_rk4_substeps"]),
    )
    np.savez_compressed(
        endpoint_path,
        x0=np.asarray(endpoints[0]), x1=np.asarray(endpoints[-1]),
        role=np.asarray("endpoint_only_reference_training"), config_hash=np.asarray(cfg_hash),
    )

    times = np.linspace(0.0, 1.0, int(cfg["time"]["scientific_nodes"]), dtype=np.float64)
    bank = truth.make_bank(
        seed=seed + int(tcfg["prospective_seed_offset"]),
        n=int(tcfg["prospective_particles"]),
        times=jnp.asarray(times),
        substeps_per_interval=int(tcfg["rk4_substeps_per_interval"]),
    )
    # Microscopic states exist only in this trusted reducer's memory.  The saved
    # prospective artifact contains aggregate reductions exclusively.
    states = np.asarray(bank.particles, dtype=np.float64)
    pcfg = cfg["aggregate_predictor"]
    x_grid = np.linspace(0.0, 2.0, int(pcfg["grid_nx"]), dtype=np.float64)
    y_grid = np.linspace(0.0, 1.0, int(pcfg["grid_ny"]), dtype=np.float64)
    mean_field, second_field = _response_fields(
        states, x_grid, y_grid, float(cfg["measurement"]["sensor_width"]), int(pcfg["particle_chunk"])
    )
    qoi = np.asarray(jnp.mean(qoi_features(jnp.asarray(states)), axis=1), dtype=np.float64)
    floor = float(cfg["qoi"]["scale_floor"])
    scales = np.maximum(np.std(qoi, axis=0), floor)
    np.savez_compressed(
        aggregate_path,
        role=np.asarray("prospective_aggregate_only"), config_hash=np.asarray(cfg_hash),
        times=times, x_grid=x_grid, y_grid=y_grid,
        response_mean_field=mean_field, response_second_field=second_field,
        scientific_qoi_predictions=qoi, qoi_scales=scales,
    )
    receipt = {
        "schema_version": 1,
        "config_hash": cfg_hash,
        "experiment_source_hash": source_hash,
        "endpoint_path": str(endpoint_path.resolve()),
        "aggregate_path": str(aggregate_path.resolve()),
        "prospective_particles_reduced": int(states.shape[1]),
        "raw_intermediate_states_persisted": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(load_config(args.config), args.output_dir))


if __name__ == "__main__":
    main()
