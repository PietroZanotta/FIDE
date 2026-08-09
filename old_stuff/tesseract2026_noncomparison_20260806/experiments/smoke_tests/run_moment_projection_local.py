"""Run the deterministic S2 fixture through the local JAX projection core."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.observables import PairBasis
from manybody_completion.projection import ProjectionOptions, project_ensemble_moments


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    with np.load(REPO_ROOT / "data" / "smoke_problems.npz", allow_pickle=False) as data:
        coordinates = jnp.asarray(data["s2_relaxed_coordinates"])
        target = jnp.asarray(data["s2_target_moments"])
        box = jnp.asarray(data["box"])
        basis = PairBasis(
            centers=jnp.asarray(data["s2_basis_centers"]),
            widths=jnp.asarray(data["s2_basis_widths"]),
        )

    options = ProjectionOptions(
        num_steps=12,
        tolerance=1e-10,
        kkt_tolerance=1e-6,
        ridge=1e-8,
        svd_rcond=1e-7,
        damping=1.0,
        max_step_norm=0.05,
        max_correction_norm=0.25,
        line_search_steps=10,
        line_search_shrink=0.5,
        sufficient_decrease=0.0,
        merit_penalty=1.0,
    )
    projected, diagnostics = project_ensemble_moments(
        coordinates=coordinates,
        target_moments=target,
        box=box,
        basis=basis,
        moment_scales=jnp.ones_like(target),
        basis_mask=jnp.ones_like(target),
        options=options,
    )

    probe = jax.random.normal(jax.random.PRNGKey(91), coordinates.shape, dtype=coordinates.dtype)

    def scalar_probe(value_coordinates, value_target):
        result, _ = project_ensemble_moments(
            coordinates=value_coordinates,
            target_moments=value_target,
            box=box,
            basis=basis,
            moment_scales=jnp.ones_like(target),
            basis_mask=jnp.ones_like(target),
            options=options,
        )
        return jnp.vdot(result, probe)

    coordinate_gradient, target_gradient = jax.grad(scalar_probe, argnums=(0, 1))(
        coordinates, target
    )
    summary = {
        "constraint_residual_before": float(diagnostics["constraint_residual_before"]),
        "constraint_residual": float(diagnostics["constraint_residual"]),
        "residual_reduction_factor": float(
            diagnostics["constraint_residual_before"]
            / jnp.maximum(diagnostics["constraint_residual"], 1e-30)
        ),
        "correction_norm": float(diagnostics["correction_norm"]),
        "kkt_stationarity_norm": float(diagnostics["kkt_stationarity_norm"]),
        "effective_rank": int(diagnostics["effective_rank"]),
        "active_constraints": int(diagnostics["active_constraints"]),
        "rank_deficient": bool(diagnostics["rank_deficient"]),
        "iterations": int(diagnostics["iterations"]),
        "line_search_failures": int(diagnostics["line_search_failures"]),
        "converged": bool(diagnostics["converged"]),
        "coordinate_gradient_norm": float(jnp.linalg.norm(coordinate_gradient)),
        "target_gradient_norm": float(jnp.linalg.norm(target_gradient)),
        "gradients_finite": bool(
            jnp.all(jnp.isfinite(coordinate_gradient))
            & jnp.all(jnp.isfinite(target_gradient))
        ),
        "projected_shape": list(projected.shape),
    }
    output = REPO_ROOT / "artifacts" / "moment_projection_local_s2.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["constraint_residual"] > summary["constraint_residual_before"] * 1e-2:
        raise SystemExit("S2 residual was not reduced by two orders of magnitude")
    if summary["rank_deficient"]:
        raise SystemExit("S2 unexpectedly had a rank-deficient moment Jacobian")
    if not summary["gradients_finite"]:
        raise SystemExit("projection gradients were non-finite")


if __name__ == "__main__":
    main()
