"""Compare local JAX and in-process Tesseract solver values and gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from manybody_completion.energy import PhysicalParameters
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.solver_factory import build_solver_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination for the backend-parity report.",
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    root = Path(__file__).resolve().parents[1]
    box = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    basis = PairBasis.uniform(3, 0.18, 0.42, 0.06, dtype=jnp.float64)
    coordinates = jax.random.uniform(
        jax.random.PRNGKey(18), (2, 4, 2), dtype=jnp.float64
    )
    target = ensemble_pair_moments(coordinates, box, basis)
    common_config = {
        "physical": {"r0": 0.22, "kappa": 20.0},
        "relaxation": {
            "num_steps": 2,
            "step_size": 0.05,
            "prox_strength": 0.15,
            "max_particle_step": 0.05,
            "tolerance": 1.0,
        },
        "projection": {
            "num_steps": 2,
            "ridge": 1e-5,
            "max_particle_step": 0.08,
            "tolerance": 1e-8,
            "rank_tolerance": 1e-7,
        },
    }
    common = {
        "repository_root": root,
        "box": box,
        "basis": basis,
        "moment_scales": jnp.ones((3,), dtype=jnp.float64),
        "physical": PhysicalParameters(**common_config["physical"]),
    }
    local = build_solver_backend(
        {**common_config, "solver_backend": {"kind": "local_jax"}}, **common
    )
    remote = build_solver_backend(
        {
            **common_config,
            "solver_backend": {"kind": "tesseract", "transport": "local_api"},
        },
        **common,
    )

    local_relaxed, _ = local.relax(coordinates)
    remote_relaxed, _ = remote.relax(coordinates)
    local_projected, _ = local.project(local_relaxed, target)
    remote_projected, _ = remote.project(remote_relaxed, target)
    relax_error = float(jnp.max(jnp.abs(local_relaxed - remote_relaxed)))
    projection_error = float(jnp.max(jnp.abs(local_projected - remote_projected)))

    probe = jnp.arange(coordinates.size, dtype=coordinates.dtype).reshape(
        coordinates.shape
    )

    def local_objective(values):
        return jnp.vdot(local.relax(values)[0], probe)

    def remote_objective(values):
        return jnp.vdot(remote.relax(values)[0], probe)

    gradient_error = float(
        jnp.max(
            jnp.abs(
                jax.grad(local_objective)(coordinates)
                - jax.grad(remote_objective)(coordinates)
            )
        )
    )

    def local_projection_objective(values, moments):
        return jnp.vdot(local.project(values, moments)[0], probe)

    def remote_projection_objective(values, moments):
        return jnp.vdot(remote.project(values, moments)[0], probe)

    local_projection_gradients = jax.grad(
        local_projection_objective, argnums=(0, 1)
    )(local_relaxed, target)
    remote_projection_gradients = jax.grad(
        remote_projection_objective, argnums=(0, 1)
    )(remote_relaxed, target)
    projection_coordinate_gradient_error = float(
        jnp.max(
            jnp.abs(
                local_projection_gradients[0]
                - remote_projection_gradients[0]
            )
        )
    )
    projection_target_gradient_error = float(
        jnp.max(
            jnp.abs(
                local_projection_gradients[1]
                - remote_projection_gradients[1]
            )
        )
    )

    def local_composed_objective(values, moments):
        relaxed = local.relax(values)[0]
        return jnp.vdot(local.project(relaxed, moments)[0], probe)

    def remote_composed_objective(values, moments):
        relaxed = remote.relax(values)[0]
        return jnp.vdot(remote.project(relaxed, moments)[0], probe)

    local_composed_gradients = jax.grad(
        local_composed_objective, argnums=(0, 1)
    )(coordinates, target)
    remote_composed_gradients = jax.grad(
        remote_composed_objective, argnums=(0, 1)
    )(coordinates, target)
    composed_coordinate_gradient_error = float(
        jnp.max(
            jnp.abs(
                local_composed_gradients[0]
                - remote_composed_gradients[0]
            )
        )
    )
    composed_target_gradient_error = float(
        jnp.max(
            jnp.abs(
                local_composed_gradients[1]
                - remote_composed_gradients[1]
            )
        )
    )
    batch = jnp.stack((coordinates, jnp.mod(coordinates + 0.017, box)))

    def local_batch_objective(values):
        return jnp.sum(jax.vmap(local.relax)(values)[0])

    def remote_batch_objective(values):
        return jnp.sum(jax.vmap(remote.relax)(values)[0])

    batch_gradient_error = float(
        jnp.max(
            jnp.abs(
                jax.grad(local_batch_objective)(batch)
                - jax.grad(remote_batch_objective)(batch)
            )
        )
    )
    tolerance = 2e-10
    errors = {
        "relaxation_forward_max_abs_error": relax_error,
        "projection_forward_max_abs_error": projection_error,
        "relaxation_coordinate_gradient_max_abs_error": gradient_error,
        "projection_coordinate_gradient_max_abs_error": (
            projection_coordinate_gradient_error
        ),
        "projection_target_gradient_max_abs_error": projection_target_gradient_error,
        "composed_coordinate_gradient_max_abs_error": (
            composed_coordinate_gradient_error
        ),
        "composed_target_gradient_max_abs_error": composed_target_gradient_error,
        "vmapped_relaxation_gradient_max_abs_error": batch_gradient_error,
    }
    maximum_error = max(errors.values())
    report = {
        "status": "passed" if maximum_error <= tolerance else "failed",
        "scope": (
            "local JAX versus in-process Tesseract relaxation/projection "
            "forward values, individual gradients, and composed gradients"
        ),
        "dtype": "float64",
        "tolerance": tolerance,
        "maximum_error": maximum_error,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit("Tesseract backend differs from local JAX")


if __name__ == "__main__":
    main()
