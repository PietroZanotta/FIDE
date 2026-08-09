"""Run S1 through the built Tesseract and verify a host-side JAX gradient."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    with np.load(REPO_ROOT / "data" / "smoke_problems.npz", allow_pickle=False) as data:
        coordinates = jnp.asarray(data["s1_coordinates"])
        inputs = {
            "coordinates": coordinates,
            "box": jnp.asarray(data["box"]),
            "r0": jnp.asarray(data["s1_r0"]),
            "kappa": jnp.asarray(data["s1_kappa"]),
            "prox_strength": jnp.asarray(data["s1_prox_strength"]),
            "num_steps": 128,
            "step_size": 0.0025,
            "tolerance": 1e-7,
            "max_update_norm": 0.04,
            "line_search_steps": 12,
            "line_search_shrink": 0.5,
            "armijo_coefficient": 1e-4,
        }

    probe = jnp.asarray([[[1.0, -0.3], [-0.7, 0.2]]], dtype=coordinates.dtype)
    tesseract = Tesseract.from_image("manybody-physical-relaxation")
    # Tesseract Core 1.10.0 hardcodes debug=True in from_image(), which adds an
    # unnecessary debugpy port and can fail under Docker Desktop port forwarding.
    tesseract._spawn_config["debug"] = False
    with tesseract:
        result = apply_tesseract(tesseract, inputs)

        def scalar_probe(value):
            current = dict(inputs)
            current["coordinates"] = value
            output = apply_tesseract(tesseract, current)
            return jnp.vdot(output["relaxed_coordinates"], probe)

        gradient = jax.grad(scalar_probe)(coordinates)

    summary = {
        "physical_energy_before": float(result["physical_energy_before"]),
        "physical_energy_after": float(result["physical_energy_after"]),
        "minimum_pair_distance_before": float(result["minimum_pair_distance_before"]),
        "minimum_pair_distance_after": float(result["minimum_pair_distance_after"]),
        "prox_displacement": float(result["prox_displacement"]),
        "converged": bool(result["converged"]),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
        "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gradient_finite"] or summary["gradient_norm"] == 0.0:
        raise SystemExit("Tesseract gradient was non-finite or zero")
    if summary["physical_energy_after"] >= summary["physical_energy_before"]:
        raise SystemExit("physical energy did not decrease")


if __name__ == "__main__":
    main()
