#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from manybody_completion.config import load_config
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.training import (
    composed_objective_and_gradient,
    finite_difference_gradient,
    make_conditional_tasks,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffpop_micro.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-report")
    args = parser.parse_args()
    config = load_config(args.config)
    support = build_population_support(int(config["system"]["n_spins"]))
    true_params = PriorParameters.from_mapping(config["true_prior"])
    probe_params = PriorParameters.from_mapping(config["learned_initial"])
    rng = np.random.default_rng(int(config["seed"]) + 991)
    task = make_conditional_tasks(
        true_params,
        support,
        [float(config["target"]["true_tilt"])],
        512,
        rng,
    )[0]
    full_value, full_grad = composed_objective_and_gradient(
        probe_params, support, task, differentiate_dual=True
    )
    stop_value, stop_grad = composed_objective_and_gradient(
        probe_params, support, task, differentiate_dual=False
    )
    finite = finite_difference_gradient(probe_params, support, task)
    error = float(np.linalg.norm(full_grad - finite) / max(np.linalg.norm(finite), 1e-12))
    payload = {
        "full_objective": full_value,
        "stopgrad_objective": stop_value,
        "full_gradient": full_grad.tolist(),
        "stopgrad_gradient": stop_grad.tolist(),
        "finite_difference_gradient": finite.tolist(),
        "relative_full_gradient_error": error,
        "full_minus_stopgrad_gradient_norm": float(np.linalg.norm(full_grad - stop_grad)),
        "passed": bool(error < 2e-3 and np.all(np.isfinite(full_grad))),
    }
    if args.experiment_report:
        payload["experiment_report"] = args.experiment_report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
