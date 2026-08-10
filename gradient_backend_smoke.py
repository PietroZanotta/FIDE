#!/usr/bin/env python3
"""Smoke-test the complete rollout and fiber gradient engines."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from backend_runtime import normalize_backend
from gradient_runtime import run_gradient_engine
import level2_paper_study as paper
import stage3_rollout_adaptation as stage3
import stage4_fiber_design as stage4


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "backend_smoke"


def rollout_payload(source, populations):
    continuous = source["rollout_diagnostics"]["continuous_time_training"]
    model = continuous["model_parameters"]
    raw = jnp.asarray(source["schedules"]["optimized_multi"]["raw"])
    target = jnp.asarray(populations["target"])

    def role(offset):
        rng = np.random.default_rng(int(source["seed"]) + offset)
        generation = paper.make_bridge_bank(populations, rng, np.asarray([0.5]), 8)
        oracle_bank = paper.make_bridge_bank(
            populations, rng, np.asarray(stage3.EVALUATION_TIMES), 24
        )
        oracle = stage3.oracle_projection(raw, oracle_bank, target)
        return generation, oracle

    adaptation, adaptation_oracle = role(181000)
    selection, selection_oracle = role(182000)
    payload = {
        **{f"model_{name}": value for name, value in model.items()},
        "gate": continuous["gate"],
        "schedule_raw": raw,
        "control": 0,
        "optimizer_steps": 5,
    }
    for prefix, generation, oracle in (
        ("adaptation", adaptation, adaptation_oracle),
        ("selection", selection, selection_oracle),
    ):
        payload[f"{prefix}_minus"] = generation[0]
        payload[f"{prefix}_plus"] = generation[1]
        payload[f"{prefix}_noise"] = generation[2]
        payload[f"{prefix}_oracle_features"] = oracle[0]
        payload[f"{prefix}_oracle_weights"] = oracle[1]
    return payload


def fiber_payload(source, populations):
    geometry = stage4.endpoint_geometry(populations)
    raw = jnp.asarray(source["schedules"]["optimized_multi"]["raw"])

    def bank(offset, count):
        rng = np.random.default_rng(int(source["seed"]) + offset)
        return paper.make_bridge_bank(
            populations, rng, np.asarray(stage4.TIMES), count
        )

    adaptation = bank(191000, 12)
    selection = bank(192000, 16)
    return {
        "schedule_raw": raw,
        "common_mean": geometry["common_mean"],
        "theta0": geometry["theta0"],
        "basis": geometry["basis"],
        "adaptation_minus": adaptation[0],
        "adaptation_plus": adaptation[1],
        "adaptation_noise": adaptation[2],
        "selection_minus": selection[0],
        "selection_plus": selection[1],
        "selection_noise": selection[2],
        "stopped": 0,
        "optimizer_steps": 5,
    }


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__array__"):
        return np.asarray(value).tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "tesseract"), default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    backend = normalize_backend(args.backend)
    source = json.loads(
        (ROOT / "results" / "level2_paper_study" / "jax" / "seed_401.json").read_text()
    )
    populations = paper.build_physical_populations(10401, False)
    results = {}
    started = time.perf_counter()
    for engine, payload in (
        ("rollout", rollout_payload(source, populations)),
        ("fiber", fiber_payload(source, populations)),
    ):
        engine_started = time.perf_counter()
        output = run_gradient_engine(engine, payload, backend)
        results[engine] = {
            "elapsed_seconds": time.perf_counter() - engine_started,
            "output": jsonable(output),
        }
    report = {
        "backend": backend,
        "seed": 401,
        "optimizer_steps": 5,
        "gradient_scope": "complete objective, reverse pass, Adam, and selection",
        "elapsed_seconds": time.perf_counter() - started,
        "engines": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{backend}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "backend": backend,
        "rollout_selected_step": results["rollout"]["output"]["selected_step"],
        "fiber_selected_step": results["fiber"]["output"]["selected_step"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
