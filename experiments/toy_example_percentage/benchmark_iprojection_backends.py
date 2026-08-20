from __future__ import annotations

"""Focused realistic benchmark for batched JAX/native I-projection trajectories."""

import argparse
import copy
import os
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path[:0] = [
    str(REPO_ROOT / "src"),
    str(SCRIPT_DIR),
    str(REPO_ROOT / "native" / "iprojection_tesseract" / "build"),
]

from benchmark_poisson_backends import _load_saved_inputs
from experiment import ToyExperiment
from mfsi.io import write_json

import _iprojection_native as native

jax.config.update("jax_enable_x64", True)


def _block(value):
    return jax.block_until_ready(value)


def _median(fn, args, repetitions: int) -> float:
    _block(fn(*args))
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        _block(fn(*args))
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    repetitions = max(3, int(args.repetitions))

    cfg, reference, nodes, velocity, base_weights, bank, eta, eta_deg = _load_saved_inputs()
    cfg_jax = copy.deepcopy(cfg)
    cfg_native = copy.deepcopy(cfg)
    cfg_jax["projection"]["trajectory_backend"] = "jax"
    cfg_native["projection"]["trajectory_backend"] = "tesseract_cpp"
    for item in (cfg_jax, cfg_native):
        item["optimization"]["full_gradient_poisson_backend"] = "tesseract_cpp"
        item["optimization"]["full_exact_poisson_backend"] = "tesseract_cpp"

    def make_exp(config):
        return ToyExperiment(
            config,
            reference,
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=base_weights,
        )

    exp_jax = make_exp(cfg_jax)
    exp_native = make_exp(cfg_native)
    time_idx = exp_jax.full_gradient_time_idx
    trial_idx = jnp.arange(int(bank.masses.shape[0]), dtype=jnp.int32)

    phi_grid, phi_nodes, _ = exp_jax._geometry(eta)
    polytope, endpoint_violation = exp_jax._prepare_reconstruction_polytope(
        phi_grid, phi_nodes, bank
    )
    rec = jax.vmap(
        lambda trial: exp_jax._reconstruct_from_geometry(
            phi_grid,
            phi_nodes,
            bank,
            trial,
            reconstruction_polytope=polytope,
            endpoint_violation=endpoint_violation,
        )
    )(trial_idx)
    fixed_phi = phi_nodes[time_idx]
    fixed_base = exp_jax.reference_weights[time_idx]
    fixed_targets = rec.c[:, time_idx]

    particle_cotangent = jnp.sin(
        0.017 * jnp.arange(fixed_phi.shape[1], dtype=jnp.float64)
    )
    lambda_cotangent = jnp.asarray([0.37, -0.21], dtype=jnp.float64)

    def trajectory_loss(projector, phi_arg, target_arg):
        state = projector.project_trajectory(fixed_phi + phi_arg, fixed_base, target_arg)
        return (
            jnp.mean(state.weights * particle_cotangent[None, None, :])
            + jnp.mean(state.lam * lambda_cotangent[None, None, :])
            + 0.01 * jnp.mean(state.ess_fraction)
        )

    zero_phi = jnp.zeros_like(fixed_phi)
    jax_forward = jax.jit(lambda p, t: trajectory_loss(exp_jax.projector, p, t))
    native_forward = jax.jit(lambda p, t: trajectory_loss(exp_native.projector, p, t))
    jax_vg = jax.jit(jax.value_and_grad(
        lambda p, t: trajectory_loss(exp_jax.projector, p, t), argnums=(0, 1)
    ))
    native_vg = jax.jit(jax.value_and_grad(
        lambda p, t: trajectory_loss(exp_native.projector, p, t), argnums=(0, 1)
    ))
    inputs = (zero_phi, fixed_targets)
    jax_value, jax_grad = jax_vg(*inputs)
    native_value, native_grad = native_vg(*inputs)
    _block((jax_value, jax_grad, native_value, native_grad))

    state_jax = jax.jit(exp_jax.projector.project_trajectory)(
        fixed_phi, fixed_base, fixed_targets
    )
    state_native = jax.jit(exp_native.projector.project_trajectory)(
        fixed_phi, fixed_base, fixed_targets
    )
    _block((state_jax, state_native))

    full_jax = jax.jit(jax.value_and_grad(lambda x: exp_jax.full_action_gradient(x, bank)))
    full_native = jax.jit(jax.value_and_grad(lambda x: exp_native.full_action_gradient(x, bank)))
    full_jax_value, full_jax_grad = full_jax(eta)
    full_native_value, full_native_grad = full_native(eta)
    _block((full_jax_value, full_jax_grad, full_native_value, full_native_grad))

    log_base = jnp.where(fixed_base > 0.0, jnp.log(fixed_base), -jnp.inf)
    direct = native.solve_batch(
        np.ascontiguousarray(fixed_phi),
        np.ascontiguousarray(log_base),
        np.ascontiguousarray(fixed_targets),
        exp_native.projector.cfg.max_steps,
        exp_native.projector.cfg.residual_tol,
        exp_native.projector.cfg.newton_ridge,
        exp_native.projector.cfg.step_cap,
        exp_native.projector.cfg.lambda_clip,
        exp_native.projector.cfg.line_search_steps,
        exp_native.projector.cfg.implicit_ridge,
    )

    gradient_difference = np.concatenate([
        np.ravel(np.asarray(a - b)) for a, b in zip(jax_grad, native_grad, strict=True)
    ])
    gradient_reference = np.concatenate([np.ravel(np.asarray(x)) for x in jax_grad])
    full_gradient_difference = np.asarray(full_jax_grad - full_native_grad)
    result = {
        "schema_version": 1,
        "workload": {
            "design_deg": list(map(float, eta_deg)),
            "shape_phi": list(map(int, fixed_phi.shape)),
            "shape_targets": list(map(int, fixed_targets.shape)),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "affinity_cpu_count": len(os.sched_getaffinity(0)),
            "jax_devices": [str(device) for device in jax.devices()],
            "repetitions": repetitions,
        },
        "timings": {
            "jax_trajectory_forward_seconds": _median(jax_forward, inputs, repetitions),
            "tesseract_trajectory_forward_seconds": _median(native_forward, inputs, repetitions),
            "jax_trajectory_value_gradient_seconds": _median(jax_vg, inputs, repetitions),
            "tesseract_trajectory_value_gradient_seconds": _median(native_vg, inputs, repetitions),
            "jax_complete_stage4_value_gradient_seconds": _median(full_jax, (eta,), repetitions),
            "tesseract_complete_stage4_value_gradient_seconds": _median(full_native, (eta,), repetitions),
        },
        "errors": {
            "lambda_max_absolute": float(jnp.max(jnp.abs(state_jax.lam - state_native.lam))),
            "weights_max_absolute": float(jnp.max(jnp.abs(state_jax.weights - state_native.weights))),
            "residual_max_absolute": float(jnp.max(jnp.abs(state_jax.residual - state_native.residual))),
            "trajectory_value_absolute": float(jnp.abs(jax_value - native_value)),
            "trajectory_gradient_max_absolute": float(np.max(np.abs(gradient_difference))),
            "trajectory_gradient_relative_l2": float(
                np.linalg.norm(gradient_difference) / np.linalg.norm(gradient_reference)
            ),
            "complete_stage4_value_absolute": float(jnp.abs(full_jax_value - full_native_value)),
            "complete_stage4_gradient_max_absolute": float(np.max(np.abs(full_gradient_difference))),
            "complete_stage4_gradient_relative_l2": float(
                np.linalg.norm(full_gradient_difference) / np.linalg.norm(np.asarray(full_jax_grad))
            ),
        },
        "native_diagnostics": {
            "max_iterations": int(np.max(direct["iterations"])),
            "mean_iterations": float(np.mean(direct["iterations"])),
            "max_residual_norm": float(np.max(direct["residual_norm"])),
            "all_converged": bool(np.all(direct["converged"])),
        },
    }
    timings = result["timings"]
    result["speedups"] = {
        "trajectory_forward": timings["jax_trajectory_forward_seconds"] / timings["tesseract_trajectory_forward_seconds"],
        "trajectory_value_gradient": timings["jax_trajectory_value_gradient_seconds"] / timings["tesseract_trajectory_value_gradient_seconds"],
        "complete_stage4_value_gradient": timings["jax_complete_stage4_value_gradient_seconds"] / timings["tesseract_complete_stage4_value_gradient_seconds"],
    }
    output_path = SCRIPT_DIR / "outputs" / "iprojection_backend_benchmark.json"
    write_json(output_path, result)
    print(result, flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
