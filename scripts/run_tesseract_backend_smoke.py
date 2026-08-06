"""Compare local JAX and in-process Tesseract implementations of both solvers."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from manybody_completion.energy import PhysicalParameters
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.solver_factory import build_solver_backend


def main() -> None:
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
    print(f"relaxation max error: {relax_error:.3e}")
    print(f"projection max error: {projection_error:.3e}")
    print(f"relaxation gradient max error: {gradient_error:.3e}")
    print(f"vmapped relaxation gradient max error: {batch_gradient_error:.3e}")
    if max(
        relax_error,
        projection_error,
        gradient_error,
        batch_gradient_error,
    ) > tolerance:
        raise SystemExit("Tesseract backend differs from local JAX")


if __name__ == "__main__":
    main()
