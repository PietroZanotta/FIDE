from __future__ import annotations

"""Profile cumulative differentiable components of the realistic toy stage 4.

The benchmark deliberately uses saved full-run inputs and the configured
4-trial x 7-time gradient workload.  Every reported VJP differentiates with
respect to the sensor angles, so timings include the gradient path used by Adam.
"""

import argparse
import copy
import os
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path[:0] = [str(SRC_DIR), str(SCRIPT_DIR)]

from benchmark_poisson_backends import _load_saved_inputs
from experiment import ToyExperiment
from mfsi.io import write_json
from mfsi.raster import rasterize_projected_particles

jax.config.update("jax_enable_x64", True)


def _block(value):
    return jax.block_until_ready(value)


def _benchmark(fn, eta, repetitions: int) -> dict[str, float]:
    compiled = jax.jit(fn)
    compiled_vg = jax.jit(jax.value_and_grad(fn))

    started = time.perf_counter()
    _block(compiled(eta))
    forward_compile = time.perf_counter() - started
    started = time.perf_counter()
    _block(compiled_vg(eta))
    vjp_compile = time.perf_counter() - started

    forward_samples = []
    vjp_samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        _block(compiled(eta))
        forward_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        _block(compiled_vg(eta))
        vjp_samples.append(time.perf_counter() - started)
    return {
        "forward_seconds": float(statistics.median(forward_samples)),
        "value_gradient_seconds": float(statistics.median(vjp_samples)),
        "first_forward_compile_and_run_seconds": float(forward_compile),
        "first_value_gradient_compile_and_run_seconds": float(vjp_compile),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    repetitions = max(3, int(args.repetitions))

    cfg, reference, nodes, velocity, weights, bank, eta, eta_deg = _load_saved_inputs()
    cfg = copy.deepcopy(cfg)
    cfg["optimization"]["full_gradient_poisson_backend"] = "tesseract_cpp"
    cfg["optimization"]["full_exact_poisson_backend"] = "tesseract_cpp"
    exp = ToyExperiment(
        cfg,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    time_indices = list(map(int, exp.full_gradient_time_idx))
    trial_count = int(bank.masses.shape[0])

    def geometry_loss(eta_arg):
        phi_grid, phi_nodes, grad_nodes = exp._geometry(exp.family.canonicalize(eta_arg))
        return (
            jnp.mean(phi_grid**2)
            + jnp.mean(phi_nodes**2)
            + 0.01 * jnp.mean(grad_nodes**2)
        )

    def reconstruction_loss(eta_arg):
        canonical = exp.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, _ = exp._geometry(canonical)
        polytope, endpoint_violation = exp._prepare_reconstruction_polytope(
            phi_grid, phi_nodes, bank
        )
        values = []
        for trial in range(trial_count):
            rec = exp._reconstruct_from_geometry(
                phi_grid,
                phi_nodes,
                bank,
                trial,
                reconstruction_polytope=polytope,
                endpoint_violation=endpoint_violation,
            )
            values.append(
                jnp.mean(rec.c**2)
                + 0.01 * jnp.mean(rec.c_dot**2)
                + 0.01 * rec.projection_distance
            )
        return jnp.mean(jnp.stack(values))

    def projection_loss(eta_arg):
        canonical = exp.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, _ = exp._geometry(canonical)
        polytope, endpoint_violation = exp._prepare_reconstruction_polytope(
            phi_grid, phi_nodes, bank
        )
        values = []
        for trial in range(trial_count):
            rec = exp._reconstruct_from_geometry(
                phi_grid,
                phi_nodes,
                bank,
                trial,
                reconstruction_polytope=polytope,
                endpoint_violation=endpoint_violation,
            )
            lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
            for t_idx in time_indices:
                projection = exp.projector.project(
                    phi_nodes[t_idx],
                    exp.reference_weights[t_idx],
                    rec.c[t_idx],
                    lam0=lam,
                )
                lam = projection.lam
                # Nonconstant scalar cotangents exercise both lambda's custom VJP
                # and the downstream derivative of the normalized weights.
                values.append(
                    jnp.sum(projection.weights**2)
                    + 0.01 * jnp.sum(projection.lam**2)
                    + jnp.sum(projection.residual**2)
                )
        return jnp.mean(jnp.stack(values))

    def forcing_loss(eta_arg):
        canonical = exp.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, grad_nodes = exp._geometry(canonical)
        polytope, endpoint_violation = exp._prepare_reconstruction_polytope(
            phi_grid, phi_nodes, bank
        )
        values = []
        for trial in range(trial_count):
            rec = exp._reconstruct_from_geometry(
                phi_grid,
                phi_nodes,
                bank,
                trial,
                reconstruction_polytope=polytope,
                endpoint_violation=endpoint_violation,
            )
            lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
            for t_idx in time_indices:
                projection, forcing = exp._particle_forcing_only(
                    phi=phi_nodes[t_idx],
                    grad_phi=grad_nodes[t_idx],
                    velocity=exp.reference_velocity[t_idx],
                    base_weights=exp.reference_weights[t_idx],
                    target=rec.c[t_idx],
                    target_dot=rec.c_dot[t_idx],
                    lam0=lam,
                )
                lam = projection.lam
                values.append(
                    jnp.sum(projection.weights**2)
                    + 0.01 * jnp.mean(forcing**2)
                )
        return jnp.mean(jnp.stack(values))

    def raster_loss(eta_arg):
        canonical = exp.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, grad_nodes = exp._geometry(canonical)
        polytope, endpoint_violation = exp._prepare_reconstruction_polytope(
            phi_grid, phi_nodes, bank
        )
        values = []
        for trial in range(trial_count):
            rec = exp._reconstruct_from_geometry(
                phi_grid,
                phi_nodes,
                bank,
                trial,
                reconstruction_polytope=polytope,
                endpoint_violation=endpoint_violation,
            )
            lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
            for t_idx in time_indices:
                projection, forcing = exp._particle_forcing_only(
                    phi=phi_nodes[t_idx],
                    grad_phi=grad_nodes[t_idx],
                    velocity=exp.reference_velocity[t_idx],
                    base_weights=exp.reference_weights[t_idx],
                    target=rec.c[t_idx],
                    target_dot=rec.c_dot[t_idx],
                    lam0=lam,
                )
                lam = projection.lam
                ras = rasterize_projected_particles(
                    exp.reference_nodes[t_idx],
                    projection.weights,
                    forcing,
                    exp.full_gradient_grid,
                    exp.raster_cfg,
                )
                values.append(jnp.mean(ras.q**2) + 0.01 * jnp.mean(ras.h**2))
        return jnp.mean(jnp.stack(values))

    def legacy_repeated_polytope_raster_loss(eta_arg):
        """Pre-optimization baseline: rebuild the shared polytope per trial."""
        canonical = exp.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, grad_nodes = exp._geometry(canonical)
        values = []
        for trial in range(trial_count):
            rec = exp._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)
            lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
            for t_idx in time_indices:
                projection, forcing = exp._particle_forcing_only(
                    phi=phi_nodes[t_idx],
                    grad_phi=grad_nodes[t_idx],
                    velocity=exp.reference_velocity[t_idx],
                    base_weights=exp.reference_weights[t_idx],
                    target=rec.c[t_idx],
                    target_dot=rec.c_dot[t_idx],
                    lam0=lam,
                )
                lam = projection.lam
                ras = rasterize_projected_particles(
                    exp.reference_nodes[t_idx],
                    projection.weights,
                    forcing,
                    exp.full_gradient_grid,
                    exp.raster_cfg,
                )
                values.append(jnp.mean(ras.q**2) + 0.01 * jnp.mean(ras.h**2))
        return jnp.mean(jnp.stack(values))

    components = {
        "A_geometry": geometry_loss,
        "B_geometry_reconstruction": reconstruction_loss,
        "C_geometry_reconstruction_iprojection": projection_loss,
        "D_geometry_reconstruction_iprojection_forcing": forcing_loss,
        "E_geometry_reconstruction_iprojection_forcing_raster": raster_loss,
        "E0_legacy_repeated_polytope_raster": legacy_repeated_polytope_raster_loss,
        "F_complete_stage4_tesseract_cpp": lambda x: exp.full_action_gradient(x, bank),
    }
    timings = {}
    for name, fn in components.items():
        print(f"[benchmark] {name}", flush=True)
        timings[name] = _benchmark(fn, eta, repetitions)
        print(f"  {timings[name]}", flush=True)

    vjp = {name: row["value_gradient_seconds"] for name, row in timings.items()}
    optimized_value, optimized_grad = jax.jit(jax.value_and_grad(raster_loss))(eta)
    legacy_value, legacy_grad = jax.jit(
        jax.value_and_grad(legacy_repeated_polytope_raster_loss)
    )(eta)
    _block((optimized_value, optimized_grad, legacy_value, legacy_grad))
    result = {
        "schema_version": 1,
        "workload": {
            "design_deg": list(map(float, eta_deg)),
            "trials": trial_count,
            "time_indices": time_indices,
            "reference_particles": int(nodes.shape[1]),
            "gradient_grid_n": int(exp.full_gradient_grid.n),
            "jax_devices": [str(device) for device in jax.devices()],
            "affinity_cpu_count": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "repetitions": repetitions,
        },
        "timings": timings,
        "cumulative_vjp_differences_seconds": {
            "reconstruction_beyond_geometry": vjp["B_geometry_reconstruction"] - vjp["A_geometry"],
            "iprojection_beyond_reconstruction": vjp["C_geometry_reconstruction_iprojection"] - vjp["B_geometry_reconstruction"],
            "forcing_beyond_iprojection": vjp["D_geometry_reconstruction_iprojection_forcing"] - vjp["C_geometry_reconstruction_iprojection"],
            "raster_beyond_forcing": vjp["E_geometry_reconstruction_iprojection_forcing_raster"] - vjp["D_geometry_reconstruction_iprojection_forcing"],
            "poisson_beyond_raster": vjp["F_complete_stage4_tesseract_cpp"] - vjp["E_geometry_reconstruction_iprojection_forcing_raster"],
        },
        "shared_polytope_optimization": {
            "legacy_value_gradient_seconds": vjp["E0_legacy_repeated_polytope_raster"],
            "optimized_value_gradient_seconds": vjp["E_geometry_reconstruction_iprojection_forcing_raster"],
            "speedup": vjp["E0_legacy_repeated_polytope_raster"] / vjp["E_geometry_reconstruction_iprojection_forcing_raster"],
            "value_absolute_error": float(jnp.abs(legacy_value - optimized_value)),
            "gradient_max_absolute_error": float(jnp.max(jnp.abs(legacy_grad - optimized_grad))),
            "gradient_relative_l2_error": float(
                jnp.linalg.norm(legacy_grad - optimized_grad)
                / jnp.maximum(jnp.linalg.norm(legacy_grad), 1.0e-300)
            ),
        },
        "note": (
            "Differences of separately compiled cumulative kernels are diagnostic, "
            "not perfectly additive, because XLA fusion and scalar cotangents differ."
        ),
    }
    output_path = SCRIPT_DIR / "outputs" / "stage4_component_benchmark.json"
    write_json(output_path, result)
    print(f"[benchmark] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
