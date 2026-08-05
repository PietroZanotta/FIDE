"""Run S2 through the built projection Tesseract and verify host JAX gradients."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    payload = json.loads(
        (REPO_ROOT / "tesseracts/moment_projection/examples/s2_payload.json").read_text()
    )["inputs"]
    inputs = {
        key: jnp.asarray(value) if isinstance(value, list) else value
        for key, value in payload.items()
    }
    coordinates = inputs["coordinates"]
    target = inputs["target_moments"]
    probe = jax.random.normal(jax.random.PRNGKey(92), coordinates.shape, dtype=coordinates.dtype)

    tesseract = Tesseract.from_image("manybody-moment-projection")
    # Tesseract Core 1.10.0 hardcodes debug=True in from_image(), which adds an
    # unnecessary debugpy port and can fail under Docker Desktop port forwarding.
    tesseract._spawn_config["debug"] = False
    with tesseract:
        result = apply_tesseract(tesseract, inputs)

        def scalar_probe(value_coordinates, value_target):
            current = dict(inputs)
            current["coordinates"] = value_coordinates
            current["target_moments"] = value_target
            output = apply_tesseract(tesseract, current)
            return jnp.vdot(output["projected_coordinates"], probe)

        coordinate_gradient, target_gradient = jax.grad(scalar_probe, argnums=(0, 1))(
            coordinates, target
        )

    summary = {
        "constraint_residual_before": float(result["constraint_residual_before"]),
        "constraint_residual": float(result["constraint_residual"]),
        "correction_norm": float(result["correction_norm"]),
        "effective_rank": int(result["effective_rank"]),
        "rank_deficient": bool(result["rank_deficient"]),
        "converged": bool(result["converged"]),
        "coordinate_gradient_norm": float(jnp.linalg.norm(coordinate_gradient)),
        "target_gradient_norm": float(jnp.linalg.norm(target_gradient)),
        "gradients_finite": bool(
            jnp.all(jnp.isfinite(coordinate_gradient))
            & jnp.all(jnp.isfinite(target_gradient))
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gradients_finite"]:
        raise SystemExit("Tesseract projection gradients were non-finite")
    if summary["constraint_residual"] > summary["constraint_residual_before"] * 1e-2:
        raise SystemExit("Tesseract projection did not reduce the residual enough")
    if summary["rank_deficient"]:
        raise SystemExit("S2 Tesseract projection unexpectedly reported rank deficiency")


if __name__ == "__main__":
    main()
