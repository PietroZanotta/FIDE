"""Run calibration-scale parameter gradient checks in an isolated process."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

import jax

from manybody_completion.ablation import AblationMode
from manybody_completion.calibration_experiment import build_calibration_experiment_problem
from manybody_completion.config import load_yaml
from manybody_completion.generator_training import (
    ablation_training_objective,
    parameter_directional_derivative_sweep,
)
from manybody_completion.scalar_training import arrays_to_python

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem = build_calibration_experiment_problem(configuration, REPO_ROOT)
    finite_config = configuration.get("finite_difference", {})
    report: dict[str, object] = {}
    for offset, value in enumerate(finite_config.get("modes", [])):
        mode = AblationMode.parse(value)
        objective = partial(
            ablation_training_objective,
            batch=problem.minibatches[0],
            generator_config=problem.model_config,
            completion_options=problem.completion_options,
            weights=problem.objective_weights,
            mode=mode,
        )
        print(f"[calibration:gradient] checking {mode.value}", flush=True)
        report[mode.value] = parameter_directional_derivative_sweep(
            objective,
            problem.initial_parameters,
            jax.random.PRNGKey(int(finite_config["direction_seed"]) + offset),
            finite_config["epsilons"],
            jit_objective=bool(finite_config.get("jit_objective", True)),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(arrays_to_python(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("[calibration:gradient] complete", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
