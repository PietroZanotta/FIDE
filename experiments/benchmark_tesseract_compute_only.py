from __future__ import annotations

"""Compute-only JAX-versus-native benchmarks for the three Tesseracts.

Inputs are placed on their execution device before timing: JAX arrays reside on
the active accelerator and native arrays reside in host memory.  The reported
native timings therefore exclude device transfer and Tesseract wrapper overhead.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_DIR = REPO_ROOT / "experiments" / "toy_example_percentage"
sys.path[:0] = [
    str(REPO_ROOT / "src"),
    str(TOY_DIR),
    str(REPO_ROOT / "native" / "iprojection_tesseract" / "build"),
    str(REPO_ROOT / "native" / "poisson_tesseract" / "build"),
    str(REPO_ROOT / "native" / "galerkin_tesseract" / "build"),
]

from benchmark_poisson_backends import _jax_solver, _load_saved_inputs
from experiment import ToyExperiment

import _galerkin_native as galerkin_native
import _iprojection_native as iprojection_native
import _poisson_native as poisson_native


jax.config.update("jax_enable_x64", True)
WARMUP_REPETITIONS = 3


def _block(value: Any) -> Any:
    return jax.block_until_ready(value)


def _median_jax(
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    repetitions: int,
) -> tuple[float, list[float]]:
    for _ in range(WARMUP_REPETITIONS):
        _block(function(*arguments))
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        _block(function(*arguments))
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples)), samples


def _median_native(
    function: Callable[[], Any], repetitions: int
) -> tuple[float, list[float]]:
    for _ in range(WARMUP_REPETITIONS):
        function()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples)), samples


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.asarray(actual) - np.asarray(expected)
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(difference) / denominator)


def _max_absolute(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _solver_args(experiment: ToyExperiment) -> tuple[Any, ...]:
    cfg = experiment.projector.cfg
    return (
        cfg.max_steps,
        cfg.residual_tol,
        cfg.newton_ridge,
        cfg.step_cap,
        cfg.lambda_clip,
        cfg.line_search_steps,
        cfg.implicit_ridge,
    )


def benchmark_iprojection(
    repetitions: int,
    saved_inputs: tuple[Any, ...],
) -> dict[str, Any]:
    cfg, reference, nodes, velocity, base_weights, bank, eta, eta_deg = saved_inputs
    cfg_jax = copy.deepcopy(cfg)
    cfg_jax["projection"]["trajectory_backend"] = "jax"
    experiment = ToyExperiment(
        cfg_jax,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=base_weights,
    )

    phi_grid, phi_nodes, _ = experiment._geometry(eta)
    polytope, endpoint_violation = experiment._prepare_reconstruction_polytope(
        phi_grid, phi_nodes, bank
    )
    trial_indices = jnp.arange(int(bank.masses.shape[0]), dtype=jnp.int32)
    reconstruction = jax.vmap(
        lambda trial: experiment._reconstruct_from_geometry(
            phi_grid,
            phi_nodes,
            bank,
            trial,
            reconstruction_polytope=polytope,
            endpoint_violation=endpoint_violation,
        )
    )(trial_indices)
    time_indices = experiment.full_gradient_time_idx
    phi_device = jnp.asarray(phi_nodes[time_indices], dtype=jnp.float64)
    base_device = jnp.asarray(
        experiment.reference_weights[time_indices], dtype=jnp.float64
    )
    targets_device = jnp.asarray(
        reconstruction.c[:, time_indices], dtype=jnp.float64
    )
    lambda_bar_device = jnp.asarray(
        np.random.default_rng(20260831).normal(size=targets_device.shape),
        dtype=jnp.float64,
    )

    def lambda_trajectory(phi_arg: jax.Array, targets_arg: jax.Array) -> jax.Array:
        return experiment.projector.project_trajectory(
            phi_arg, base_device, targets_arg
        ).lam

    jax_forward = jax.jit(lambda_trajectory)
    jax_value_gradient = jax.jit(
        jax.value_and_grad(
            lambda phi_arg, targets_arg: jnp.sum(
                lambda_trajectory(phi_arg, targets_arg) * lambda_bar_device
            ),
            argnums=(0, 1),
        )
    )

    # All conversions occur before timing.
    phi_host = np.ascontiguousarray(np.asarray(phi_device), dtype=np.float64)
    base_host = np.ascontiguousarray(np.asarray(base_device), dtype=np.float64)
    log_base_host = np.full_like(base_host, -np.inf)
    positive = base_host > 0.0
    log_base_host[positive] = np.log(base_host[positive])
    targets_host = np.ascontiguousarray(np.asarray(targets_device), dtype=np.float64)
    lambda_bar_host = np.ascontiguousarray(
        np.asarray(lambda_bar_device), dtype=np.float64
    )
    native_args = _solver_args(experiment)

    def native_forward() -> dict[str, np.ndarray]:
        return iprojection_native.solve_batch(
            phi_host, log_base_host, targets_host, *native_args
        )

    def native_value_gradient() -> tuple[float, dict[str, np.ndarray]]:
        forward = native_forward()
        lambda_values = np.ascontiguousarray(
            forward["lambda_values"], dtype=np.float64
        )
        pullback = iprojection_native.vjp_batch(
            phi_host,
            log_base_host,
            targets_host,
            lambda_values,
            lambda_bar_host,
            *native_args,
        )
        return float(np.sum(lambda_values * lambda_bar_host)), pullback

    jax_lambda = np.asarray(_block(jax_forward(phi_device, targets_device)))
    jax_value, jax_gradient = _block(
        jax_value_gradient(phi_device, targets_device)
    )
    native_result = native_forward()
    native_value, native_gradient = native_value_gradient()
    converged = np.asarray(native_result["converged"], dtype=bool)
    if not np.all(converged):
        raise RuntimeError("native I-projection compute-only benchmark did not converge")

    jax_forward_seconds, jax_forward_samples = _median_jax(
        jax_forward, (phi_device, targets_device), repetitions
    )
    native_forward_seconds, native_forward_samples = _median_native(
        native_forward, repetitions
    )
    jax_vg_seconds, jax_vg_samples = _median_jax(
        jax_value_gradient, (phi_device, targets_device), repetitions
    )
    native_vg_seconds, native_vg_samples = _median_native(
        native_value_gradient, repetitions
    )

    gradient_actual = np.concatenate(
        [
            np.ravel(native_gradient["phi"]),
            np.ravel(native_gradient["targets"]),
        ]
    )
    gradient_expected = np.concatenate(
        [np.ravel(np.asarray(jax_gradient[0])), np.ravel(np.asarray(jax_gradient[1]))]
    )
    return {
        "workload": {
            "design_deg": list(map(float, eta_deg)),
            "phi_shape": list(map(int, phi_host.shape)),
            "targets_shape": list(map(int, targets_host.shape)),
            "dtype": "float64",
        },
        "timings_seconds": {
            "jax_forward": jax_forward_seconds,
            "native_forward_compute_only": native_forward_seconds,
            "jax_value_gradient": jax_vg_seconds,
            "native_value_gradient_compute_only": native_vg_seconds,
        },
        "timing_samples_seconds": {
            "jax_forward": jax_forward_samples,
            "native_forward_compute_only": native_forward_samples,
            "jax_value_gradient": jax_vg_samples,
            "native_value_gradient_compute_only": native_vg_samples,
        },
        "speedups": {
            "forward": jax_forward_seconds / native_forward_seconds,
            "value_gradient": jax_vg_seconds / native_vg_seconds,
        },
        "agreement": {
            "lambda_max_absolute": _max_absolute(
                native_result["lambda_values"], jax_lambda
            ),
            "value_absolute": abs(float(jax_value) - native_value),
            "gradient_relative_l2": _relative_l2(
                gradient_actual, gradient_expected
            ),
            "gradient_max_absolute": _max_absolute(
                gradient_actual, gradient_expected
            ),
            "all_native_systems_converged": bool(np.all(converged)),
            "maximum_native_residual": float(
                np.max(native_result["residual_norm"])
            ),
        },
    }


def benchmark_poisson(
    repetitions: int,
    saved_inputs: tuple[Any, ...],
) -> dict[str, Any]:
    cfg, reference, nodes, velocity, base_weights, bank, eta, eta_deg = saved_inputs
    cfg_jax = copy.deepcopy(cfg)
    cfg_jax["optimization"]["full_gradient_poisson_backend"] = "jax"
    experiment = ToyExperiment(
        cfg_jax,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=base_weights,
    )
    canonical = experiment.family.canonicalize(eta)
    phi_grid, phi_nodes, grad_nodes = experiment._geometry(canonical)
    q_batch, h_batch, _ = jax.jit(experiment._full_action_gradient_system_batch)(
        phi_grid, phi_nodes, grad_nodes, bank
    )
    _block((q_batch, h_batch))
    poisson_cfg = experiment.poisson_gradient_cfg
    q_floor = poisson_cfg.operator_floor_rel * jnp.max(
        q_batch, axis=(-2, -1), keepdims=True
    )
    q_device = jnp.asarray(q_batch + q_floor, dtype=jnp.float64)
    rhs_device = jnp.asarray(-(q_batch * h_batch), dtype=jnp.float64)
    flat_q = q_batch.reshape((q_batch.shape[0], -1))
    gauge_device = jnp.asarray(
        (
            flat_q
            / jnp.maximum(
                jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300
            )
        ).reshape(q_batch.shape),
        dtype=jnp.float64,
    )
    cotangent_device = jnp.asarray(
        np.random.default_rng(20260832).normal(size=q_device.shape),
        dtype=jnp.float64,
    )

    jax_forward = jax.jit(
        lambda q, rhs, gauge: _jax_solver(
            q,
            rhs,
            gauge,
            dx=poisson_cfg.dx,
            gauge_strength=poisson_cfg.gauge_strength,
            tol=poisson_cfg.cg_tol,
            maxiter=poisson_cfg.cg_maxiter,
        )
    )
    jax_value_gradient = jax.jit(
        jax.value_and_grad(
            lambda q, rhs, gauge: jnp.sum(
                jax_forward(q, rhs, gauge) * cotangent_device
            ),
            argnums=(0, 1, 2),
        )
    )

    # All conversions occur before timing.
    q_host = np.ascontiguousarray(np.asarray(q_device), dtype=np.float64)
    rhs_host = np.ascontiguousarray(np.asarray(rhs_device), dtype=np.float64)
    gauge_host = np.ascontiguousarray(np.asarray(gauge_device), dtype=np.float64)
    cotangent_host = np.ascontiguousarray(
        np.asarray(cotangent_device), dtype=np.float64
    )

    def solve_native(right_hand_side: np.ndarray) -> dict[str, np.ndarray]:
        result = poisson_native.solve_batch(
            q_host,
            right_hand_side,
            gauge_host,
            poisson_cfg.dx,
            poisson_cfg.gauge_strength,
            poisson_cfg.cg_tol,
            poisson_cfg.cg_maxiter,
            None,
        )
        if not np.all(result["converged"]):
            raise RuntimeError("native Poisson compute-only benchmark did not converge")
        return result

    def native_forward() -> dict[str, np.ndarray]:
        return solve_native(rhs_host)

    def native_value_gradient() -> tuple[float, tuple[np.ndarray, ...]]:
        forward = native_forward()
        psi = np.ascontiguousarray(forward["psi"], dtype=np.float64)
        adjoint = np.ascontiguousarray(
            solve_native(cotangent_host)["psi"], dtype=np.float64
        )
        q_bar = poisson_native.weighted_operator_vjp(
            psi, adjoint, poisson_cfg.dx
        )
        gauge_bar = poisson_native.gauge_vjp(
            psi, adjoint, gauge_host, poisson_cfg.gauge_strength
        )
        return float(np.sum(psi * cotangent_host)), (q_bar, adjoint, gauge_bar)

    jax_psi = np.asarray(
        _block(jax_forward(q_device, rhs_device, gauge_device))
    )
    jax_value, jax_gradient = _block(
        jax_value_gradient(q_device, rhs_device, gauge_device)
    )
    native_result = native_forward()
    native_value, native_gradient = native_value_gradient()

    jax_forward_seconds, jax_forward_samples = _median_jax(
        jax_forward, (q_device, rhs_device, gauge_device), repetitions
    )
    native_forward_seconds, native_forward_samples = _median_native(
        native_forward, repetitions
    )
    jax_vg_seconds, jax_vg_samples = _median_jax(
        jax_value_gradient, (q_device, rhs_device, gauge_device), repetitions
    )
    native_vg_seconds, native_vg_samples = _median_native(
        native_value_gradient, repetitions
    )

    gradient_actual = np.concatenate(
        [np.ravel(np.asarray(value)) for value in native_gradient]
    )
    gradient_expected = np.concatenate(
        [np.ravel(np.asarray(value)) for value in jax_gradient]
    )
    return {
        "workload": {
            "design_deg": list(map(float, eta_deg)),
            "system_shape": list(map(int, q_host.shape)),
            "cg_tolerance": float(poisson_cfg.cg_tol),
            "cg_maximum_iterations": int(poisson_cfg.cg_maxiter),
            "dtype": "float64",
        },
        "timings_seconds": {
            "jax_forward": jax_forward_seconds,
            "native_forward_compute_only": native_forward_seconds,
            "jax_value_gradient": jax_vg_seconds,
            "native_value_gradient_compute_only": native_vg_seconds,
        },
        "timing_samples_seconds": {
            "jax_forward": jax_forward_samples,
            "native_forward_compute_only": native_forward_samples,
            "jax_value_gradient": jax_vg_samples,
            "native_value_gradient_compute_only": native_vg_samples,
        },
        "speedups": {
            "forward": jax_forward_seconds / native_forward_seconds,
            "value_gradient": jax_vg_seconds / native_vg_seconds,
        },
        "agreement": {
            "forward_relative_l2": _relative_l2(
                native_result["psi"], jax_psi
            ),
            "forward_max_absolute": _max_absolute(
                native_result["psi"], jax_psi
            ),
            "value_absolute": abs(float(jax_value) - native_value),
            "gradient_relative_l2": _relative_l2(
                gradient_actual, gradient_expected
            ),
            "gradient_max_absolute": _max_absolute(
                gradient_actual, gradient_expected
            ),
            "all_native_systems_converged": bool(
                np.all(native_result["converged"])
            ),
            "maximum_native_relative_residual": float(
                np.max(native_result["relative_residual"])
            ),
        },
    }


def benchmark_galerkin(repetitions: int) -> dict[str, Any]:
    samples, basis, particles, dimensions = 256, 280, 16, 2
    key = jax.random.PRNGKey(17)
    values_device = jax.random.normal(
        key, (samples, basis), dtype=jnp.float64
    )
    gradients_device = jax.random.normal(
        jax.random.fold_in(key, 1),
        (samples, basis, particles, dimensions),
        dtype=jnp.float64,
    )
    weights_device = jax.nn.softmax(
        jax.random.normal(jax.random.fold_in(key, 2), (samples,), dtype=jnp.float64)
    )
    forcing_device = jax.random.normal(
        jax.random.fold_in(key, 3), (samples,), dtype=jnp.float64
    )

    jax_forward = jax.jit(
        lambda values, gradients, weights, forcing: (
            jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients),
            jnp.einsum("n,n,nk->k", weights, forcing, values),
            jnp.einsum("n,nk->k", weights, values),
            jnp.einsum("n,n->", weights, forcing),
        )
    )
    device_arguments = (
        values_device,
        gradients_device,
        weights_device,
        forcing_device,
    )
    host_arguments = tuple(
        np.ascontiguousarray(np.asarray(value), dtype=np.float64)
        for value in device_arguments
    )

    def native_forward() -> dict[str, np.ndarray]:
        return galerkin_native.assemble_chunk(*host_arguments)

    jax_result = _block(jax_forward(*device_arguments))
    native_result = native_forward()
    jax_forward_seconds, jax_forward_samples = _median_jax(
        jax_forward, device_arguments, repetitions
    )
    native_forward_seconds, native_forward_samples = _median_native(
        native_forward, repetitions
    )
    names = ("gram", "raw_load", "basis_mean", "forcing_sum")
    maximum_discrepancy = max(
        _max_absolute(native_result[name], np.asarray(expected))
        for name, expected in zip(names, jax_result, strict=True)
    )
    return {
        "workload": {
            "input_shape": [samples, basis, particles, dimensions],
            "dtype": "float64",
            "derivative": "forward-only by design",
        },
        "timings_seconds": {
            "jax_forward": jax_forward_seconds,
            "native_forward_compute_only": native_forward_seconds,
        },
        "timing_samples_seconds": {
            "jax_forward": jax_forward_samples,
            "native_forward_compute_only": native_forward_samples,
        },
        "speedups": {"forward": jax_forward_seconds / native_forward_seconds},
        "agreement": {"maximum_absolute": maximum_discrepancy},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments" / "tesseract_compute_only_benchmark.json",
    )
    arguments = parser.parse_args()
    repetitions = max(5, int(arguments.repetitions))

    initial_affinity_count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    saved_inputs = _load_saved_inputs()
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    result = {
        "schema_version": 1,
        "timing_contract": {
            "summary": (
                "Compute only: JAX inputs are resident on the accelerator and "
                "native inputs are resident in host memory before timing."
            ),
            "included": [
                "solver/assembly computation",
                "native output allocation",
                "implicit native reverse computation where applicable",
            ],
            "excluded": [
                "host-device transfer",
                "Tesseract and tesseract-jax wrapper overhead",
                "JAX tracing and compilation",
                "input construction",
            ],
            "repetitions": repetitions,
            "warmup_repetitions": WARMUP_REPETITIONS,
            "statistic": "median after three untimed warm-ups",
        },
        "environment": {
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "gpu": gpu,
            "cpu_count": os.cpu_count(),
            "calling_thread_affinity_count_after_openmp_binding": (
                initial_affinity_count
            ),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "unset"),
            "OMP_PLACES": os.environ.get("OMP_PLACES", "unset"),
            "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY", "unset"),
            "OPENBLAS_NUM_THREADS": os.environ.get(
                "OPENBLAS_NUM_THREADS", "unset"
            ),
        },
        "iprojection": benchmark_iprojection(repetitions, saved_inputs),
        "weighted_poisson": benchmark_poisson(repetitions, saved_inputs),
        "galerkin": benchmark_galerkin(repetitions),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"results={arguments.output}", flush=True)


if __name__ == "__main__":
    main()
