#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from manybody_completion.config import load_config
from manybody_completion.homometric import build_population_support
from manybody_completion.network import PriorParameters
from manybody_completion.solvers import calibrate_dual, tilted_ensemble
from manybody_completion.tesseract_backend import serialize_calibration, serialize_tilted


def _load_apply(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diffpop_micro.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    support = build_population_support(int(config["system"]["n_spins"]))
    params = PriorParameters.from_mapping(config["true_prior"])
    sampler = dict(config["sampler"])
    calibration = dict(config["calibration"])
    dual = float(config["target"]["true_tilt"])
    target = float(np.sum(
        np.asarray(__import__("manybody_completion.energy", fromlist=["conditioned_probabilities"])
        .conditioned_probabilities(params, support, dual)) * support.pair
    ))

    root = Path(__file__).resolve().parents[1]
    tilted_apply = _load_apply(
        root / "tesseracts/scientific_tilted_ensemble/tesseract_api.py", "tilted_api"
    )
    dual_apply = _load_apply(
        root / "tesseracts/scientific_dual_calibration/tesseract_api.py", "dual_api"
    )
    payload_tilt = {
        "n_spins": support.n_spins,
        "prior_parameters": params.to_mapping(),
        "dual": dual,
        "sampler_options": sampler,
        "seed": 123,
    }
    local_tilt = serialize_tilted(tilted_ensemble(params, support, dual, seed=123, **sampler))
    api_tilt = tilted_apply(payload_tilt)

    payload_dual = {
        "n_spins": support.n_spins,
        "prior_parameters": params.to_mapping(),
        "target_moment": target,
        "sampler_options": sampler,
        "calibration_options": calibration,
        "seed": 456,
    }
    local_dual = serialize_calibration(
        calibrate_dual(
            params,
            support,
            target,
            sampler_options=sampler,
            calibration_options=calibration,
            seed=456,
        )
    )
    api_dual = dual_apply(payload_dual)
    tilt_error = float(
        np.max(
            np.abs(
                np.asarray(local_tilt["atom_probabilities"])
                - np.asarray(api_tilt["atom_probabilities"])
            )
        )
    )
    dual_error = abs(float(local_dual["dual"]) - float(api_dual["dual"]))
    result = {
        "tilted_atom_probability_max_error": tilt_error,
        "dual_error": dual_error,
        "tilted_status_equal": local_tilt["moment_mean"] == api_tilt["moment_mean"],
        "calibration_status_equal": local_dual["status"] == api_dual["status"],
        "passed": bool(tilt_error == 0.0 and dual_error == 0.0),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
