#!/usr/bin/env python3
"""Reconstruct only the frozen base objects consumed by Stage 3B.

This follows the paper-facing standard pipeline exactly through selected
schedule construction, random-continuous-time MLP training, and Ritz gating.
Unconsumed fixed-grid, representation, and rollout diagnostics are omitted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import level2_paper_study as paper


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "stage3b_base_models"
SEEDS = list(range(406, 416))


def build_base(seed):
    started = time.perf_counter()
    populations = paper.build_physical_populations(seed + 10000, False)
    rng = np.random.default_rng(seed)
    times = jnp.asarray(np.linspace(0.12, 0.88, 6))
    schedule_train = paper.make_bridge_bank(populations, rng, np.asarray(times), 72)
    schedule_select = paper.make_bridge_bank(populations, rng, np.asarray(times), 192)
    # Constructed in the original order even though Stage 3B does not consume it.
    paper.make_bridge_bank(populations, rng, np.asarray(times), 192)
    target = jnp.asarray(populations["target"])
    hand = np.asarray([paper._inverse_softplus(0.55)])
    scalar, _, _ = paper.optimize_schedule(hand, schedule_train, times, target, 45)
    multi_initial = np.asarray([scalar[0], 0.0, 0.0])
    multi_candidate, _, _ = paper.optimize_schedule(
        multi_initial, schedule_train, times, target, 45
    )
    nested_scalar = np.asarray([scalar[0], 0.0, 0.0])
    selection_values = {
        "nested_scalar": float(paper.schedule_objective(
            jnp.asarray(nested_scalar), schedule_select, times, target
        )),
        "multi_candidate": float(paper.schedule_objective(
            jnp.asarray(multi_candidate), schedule_select, times, target
        )),
    }
    choice = min(selection_values, key=selection_values.get)
    multi = nested_scalar if choice == "nested_scalar" else multi_candidate

    # Preserve the original RNG advancement before constructing the gate bank.
    paper.make_bridge_bank(populations, rng, np.asarray(times), 192)
    gate_bank = paper.make_bridge_bank(populations, rng, np.asarray(times), 384)

    continuous_rng = np.random.default_rng(seed + 60000)
    continuous_time_count = 18
    continuous_particles_per_time = 64
    strata = (
        np.arange(continuous_time_count)
        + continuous_rng.uniform(size=continuous_time_count)
    )
    continuous_times = jnp.asarray(
        0.12 + 0.76 * strata / continuous_time_count
    )
    continuous_train = paper.make_bridge_bank(
        populations, continuous_rng, np.asarray(continuous_times),
        continuous_particles_per_time,
    )
    model, trace, training_seconds = paper.train_neural_correction(
        jax.random.PRNGKey(seed), jnp.asarray(multi), continuous_train,
        continuous_times, target, 420,
    )
    gate, gate_gain, gate_standard_error = paper.select_gate(
        model, jnp.asarray(multi), gate_bank, times, target
    )
    return {
        "seed": seed,
        "endpoint": {
            "minus_calibration_residual": populations["minus_residual"],
            "plus_calibration_residual": populations["plus_residual"],
        },
        "schedules": {
            "optimized_multi": {
                "raw": np.asarray(multi).tolist(),
                "selection": choice,
                "candidate_raw": np.asarray(multi_candidate).tolist(),
                "selection_objectives": selection_values,
            }
        },
        "rollout_diagnostics": {
            "continuous_time_training": {
                "design": "unchanged 18 stratified random times x 64 particles",
                "times": np.asarray(continuous_times).tolist(),
                "n_times": continuous_time_count,
                "particles_per_time": continuous_particles_per_time,
                "total_training_configurations": 1152,
                "optimizer_steps": 420,
                "same_initialization_as_paper_protocol": True,
                "gate": gate,
                "gate_bank_gain": gate_gain,
                "gate_bank_standard_error": gate_standard_error,
                "training_seconds": training_seconds,
                "training_initial_loss": trace[0],
                "training_final_loss": trace[-1],
                "model_parameters": paper.serialize_mlp(model),
            }
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="406 407 408 409 410 411 412 413 414 415")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.perf_counter()
    for seed in seeds:
        print(f"[stage3b-base] seed {seed}", flush=True)
        report = build_base(seed)
        (args.output_dir / f"seed_{seed}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reports.append(report)
    summary = {
        "experiment": "stage3b-frozen-base-model-reconstruction",
        "seeds": seeds,
        "procedure": "paper-facing standard pipeline through random-time model and gate",
        "skipped_as_unconsumed": [
            "fixed-grid model", "angular representation diagnostic",
            "rollout diagnostics", "component deployment parity",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "seed_reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
