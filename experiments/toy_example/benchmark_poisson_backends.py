from __future__ import annotations

"""Benchmark the realistic toy stage-4 JAX and Tesseract/C++ solver paths."""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
NATIVE_BUILD = REPO_ROOT / "native" / "poisson_tesseract" / "build"
sys.path[:0] = [str(SRC_DIR), str(SCRIPT_DIR), str(NATIVE_BUILD)]

from experiment import ToyExperiment
from mfsi.config import load_config
from mfsi.law_objectives import TrialBank
from mfsi.linear import implicit_cg
from mfsi.poisson import weighted_laplacian, weighted_laplacian_diag
from mfsi.poisson_tesseract import solve_linear_system_batch_tesseract
from mfsi.reference import load_npz_checkpoint

import _poisson_native as native

jax.config.update("jax_enable_x64", True)


def _block(value):
    return jax.block_until_ready(value)


def _median_jax(fn, args, repetitions: int) -> float:
    _block(fn(*args))
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        _block(fn(*args))
        values.append(time.perf_counter() - start)
    return float(statistics.median(values))


def _median_python(fn, repetitions: int) -> float:
    fn()
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return float(statistics.median(values))


def _load_saved_inputs():
    output_dir = SCRIPT_DIR / "outputs" / "run"
    required = [
        output_dir / "reference.npz",
        output_dir / "reference_bank.npz",
        output_dir / "selection_bank.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The realistic benchmark needs saved full-run artifacts. Missing: "
            + ", ".join(missing)
        )

    cfg = load_config(SCRIPT_DIR / "config.json", smoke=False)
    trials = int(cfg["optimization"].get("full_gradient_trials", 4))
    reference = load_npz_checkpoint(output_dir / "reference.npz")
    with np.load(output_dir / "reference_bank.npz") as data:
        nodes = jnp.asarray(data["reference_particles"], dtype=jnp.float64)
        velocity = jnp.asarray(data["reference_velocity"], dtype=jnp.float64)
        weights = jnp.asarray(data["base_weights"], dtype=jnp.float64)
    with np.load(output_dir / "selection_bank.npz") as data:
        bank = TrialBank(
            masses=jnp.asarray(data["masses"][:trials], dtype=jnp.float64),
            sample_indices=jnp.asarray(data["sample_indices"][:trials], dtype=jnp.int32),
            detector_z=jnp.asarray(data["detector_z"][:trials], dtype=jnp.float64),
            alphas=jnp.asarray(data["alphas"][:trials], dtype=jnp.float64),
        )

    result_path = output_dir / "result.json"
    eta_deg = [23.385, 67.952]
    if result_path.is_file():
        saved = json.loads(result_path.read_text())
        eta_deg = saved.get("selection", {}).get("full_optimum_deg", eta_deg)
    eta = jnp.radians(jnp.asarray(eta_deg, dtype=jnp.float64))
    return cfg, reference, nodes, velocity, weights, bank, eta, eta_deg


def _jax_solver(q_operator, rhs, gauge, *, dx, gauge_strength, tol, maxiter):
    height, width = q_operator.shape[-2:]

    def one(q_one, rhs_one, gauge_one):
        gauge_flat = gauge_one.reshape(-1)

        def matvec(z_flat):
            z = z_flat.reshape((height, width))
            return (
                weighted_laplacian(z, q_one, dx).reshape(-1)
                + gauge_strength * gauge_flat * jnp.dot(gauge_flat, z_flat)
            )

        diag = weighted_laplacian_diag(q_one, dx).reshape(-1)
        diag = diag + gauge_strength * gauge_flat**2
        return implicit_cg(
            matvec,
            rhs_one.reshape(-1),
            tol=tol,
            maxiter=maxiter,
            preconditioner=lambda r: r / jnp.maximum(diag, 1.0e-10),
        ).reshape((height, width))

    return jax.vmap(one)(q_operator, rhs, gauge)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--include-exact-audit", action="store_true")
    args = parser.parse_args()
    repetitions = max(3, int(args.repetitions))

    cfg, reference, nodes, velocity, weights, bank, eta, eta_deg = _load_saved_inputs()
    cfg_jax = copy.deepcopy(cfg)
    cfg_native = copy.deepcopy(cfg)
    cfg_jax["optimization"]["full_gradient_poisson_backend"] = "jax"
    cfg_jax["optimization"]["full_exact_poisson_backend"] = "jax"
    cfg_native["optimization"]["full_gradient_poisson_backend"] = "tesseract_cpp"
    cfg_native["optimization"]["full_exact_poisson_backend"] = "tesseract_cpp"
    exp_jax = ToyExperiment(
        cfg_jax,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    exp_native = ToyExperiment(
        cfg_native,
        reference,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    pcfg = exp_jax.poisson_gradient_cfg

    def construct_systems(eta_arg):
        canonical = exp_jax.family.canonicalize(eta_arg)
        phi_grid, phi_nodes, grad_nodes = exp_jax._geometry(canonical)
        return exp_jax._full_action_gradient_system_batch(
            phi_grid, phi_nodes, grad_nodes, bank
        )

    construct_jit = jax.jit(construct_systems)
    q_batch, h_batch, valid = construct_jit(eta)
    _block((q_batch, h_batch, valid))
    expected_batch = int(bank.masses.shape[0]) * len(exp_jax.full_gradient_time_idx)
    if q_batch.shape != (
        expected_batch,
        exp_jax.full_gradient_grid.n,
        exp_jax.full_gradient_grid.n,
    ):
        raise RuntimeError(f"Unexpected realistic batch shape: {q_batch.shape}")

    q_floor = pcfg.operator_floor_rel * jnp.max(
        q_batch, axis=(-2, -1), keepdims=True
    )
    q_operator = q_batch + q_floor
    rhs = -(q_batch * h_batch)
    flat_q = q_batch.reshape((q_batch.shape[0], -1))
    gauge = (
        flat_q
        / jnp.maximum(jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300)
    ).reshape(q_batch.shape)
    rng = np.random.default_rng(20260815)
    cotangent = jnp.asarray(rng.normal(size=q_batch.shape), dtype=jnp.float64)

    jax_forward = jax.jit(
        lambda q, r, g: _jax_solver(
            q,
            r,
            g,
            dx=pcfg.dx,
            gauge_strength=pcfg.gauge_strength,
            tol=pcfg.cg_tol,
            maxiter=pcfg.cg_maxiter,
        )
    )
    native_forward = jax.jit(
        lambda q, r, g: solve_linear_system_batch_tesseract(
            q,
            r,
            g,
            dx=pcfg.dx,
            gauge_strength=pcfg.gauge_strength,
            cg_tol=pcfg.cg_tol,
            cg_maxiter=pcfg.cg_maxiter,
        )
    )
    jax_value_grad = jax.jit(
        jax.value_and_grad(
            lambda q, r, g: jnp.sum(jax_forward(q, r, g) * cotangent),
            argnums=(0, 1, 2),
        )
    )
    native_value_grad = jax.jit(
        jax.value_and_grad(
            lambda q, r, g: jnp.sum(native_forward(q, r, g) * cotangent),
            argnums=(0, 1, 2),
        )
    )

    system_args = (q_operator, rhs, gauge)
    jax_psi = jax_forward(*system_args)
    native_psi = native_forward(*system_args)
    jax_grad = jax_value_grad(*system_args)[1]
    native_grad = native_value_grad(*system_args)[1]
    _block((jax_psi, native_psi, jax_grad, native_grad))

    timings = {
        "A_jax_batched_forward_seconds": _median_jax(
            jax_forward, system_args, repetitions
        ),
        "B_tesseract_cpp_forward_seconds": _median_jax(
            native_forward, system_args, repetitions
        ),
        "C_jax_forward_gradient_seconds": _median_jax(
            jax_value_grad, system_args, repetitions
        ),
        "D_tesseract_cpp_forward_gradient_seconds": _median_jax(
            native_value_grad, system_args, repetitions
        ),
    }

    q_np, rhs_np, gauge_np = map(
        lambda x: np.ascontiguousarray(np.asarray(x, dtype=np.float64)), system_args
    )
    cotangent_np = np.ascontiguousarray(np.asarray(cotangent, dtype=np.float64))
    historical_120 = native.solve_batch(
        q_np,
        rhs_np,
        gauge_np,
        pcfg.dx,
        pcfg.gauge_strength,
        pcfg.cg_tol,
        120,
    )

    def direct_forward():
        return native.solve_batch(
            q_np,
            rhs_np,
            gauge_np,
            pcfg.dx,
            pcfg.gauge_strength,
            pcfg.cg_tol,
            pcfg.cg_maxiter,
        )

    forward_stats = direct_forward()
    if not np.all(forward_stats["converged"]):
        raise RuntimeError(f"Native benchmark forward failure: {forward_stats}")

    def direct_adjoint():
        return native.solve_batch(
            q_np,
            cotangent_np,
            gauge_np,
            pcfg.dx,
            pcfg.gauge_strength,
            pcfg.cg_tol,
            pcfg.cg_maxiter,
        )

    adjoint_stats = direct_adjoint()
    if not np.all(adjoint_stats["converged"]):
        raise RuntimeError(f"Native benchmark adjoint failure: {adjoint_stats}")
    direct_forward_seconds = _median_python(direct_forward, repetitions)
    direct_adjoint_seconds = _median_python(direct_adjoint, repetitions)

    system_construction_seconds = _median_jax(construct_jit, (eta,), repetitions)
    full_jax = jax.jit(
        jax.value_and_grad(lambda x: exp_jax.full_action_gradient(x, bank))
    )
    full_native = jax.jit(
        jax.value_and_grad(lambda x: exp_native.full_action_gradient(x, bank))
    )
    full_jax_value, full_jax_grad = full_jax(eta)
    full_native_value, full_native_grad = full_native(eta)
    _block((full_jax_value, full_jax_grad, full_native_value, full_native_grad))
    full_jax_seconds = _median_jax(full_jax, (eta,), repetitions)
    full_native_seconds = _median_jax(full_native, (eta,), repetitions)

    exact_audit_seconds = None
    exact_native_audit_seconds = None
    exact_steady_jax_seconds = None
    exact_steady_native_seconds = None
    exact_value_absolute = None
    if args.include_exact_audit:
        start = time.perf_counter()
        audit = exp_jax._exact_trial_result(
            eta, bank, 0, compute_law=False, compute_tangent=False, compute_full=True
        )
        exact_audit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        native_audit = exp_native._exact_trial_result(
            eta, bank, 0, compute_law=False, compute_tangent=False, compute_full=True
        )
        exact_native_audit_seconds = time.perf_counter() - start
        if not audit["valid"] or not native_audit["valid"]:
            raise RuntimeError(f"Exact benchmark audit was invalid: {audit}")
        exact_value_absolute = abs(
            float(audit["full_action"]) - float(native_audit["full_action"])
        )

        # Trial zero includes one-time tracing/compilation. Distinct later trials
        # are representative of the hundreds of uncached rows in stage 4.
        steady_trials = range(1, min(int(bank.masses.shape[0]), repetitions + 1))
        jax_exact_times = []
        native_exact_times = []
        for trial in steady_trials:
            start = time.perf_counter()
            exp_jax._exact_trial_result(
                eta,
                bank,
                trial,
                compute_law=False,
                compute_tangent=False,
                compute_full=True,
            )
            jax_exact_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            exp_native._exact_trial_result(
                eta,
                bank,
                trial,
                compute_law=False,
                compute_tangent=False,
                compute_full=True,
            )
            native_exact_times.append(time.perf_counter() - start)
        exact_steady_jax_seconds = float(statistics.median(jax_exact_times))
        exact_steady_native_seconds = float(statistics.median(native_exact_times))

    forward_diff = np.asarray(jax_psi - native_psi)
    forward_ref = np.asarray(jax_psi)
    gradient_diff = np.concatenate(
        [np.ravel(np.asarray(a - b)) for a, b in zip(jax_grad, native_grad, strict=True)]
    )
    gradient_ref = np.concatenate([np.ravel(np.asarray(x)) for x in jax_grad])
    full_grad_diff = np.asarray(full_jax_grad - full_native_grad)
    full_grad_ref = np.asarray(full_jax_grad)
    speedup = full_jax_seconds / full_native_seconds

    try:
        cpu_model = subprocess.check_output(
            ["lscpu"], text=True
        ).split("Model name:", 1)[1].splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        cpu_model = "unknown"
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    result = {
        "shape": list(map(int, q_batch.shape)),
        "dtype": "float64",
        "cg_tol": pcfg.cg_tol,
        "cg_maxiter": pcfg.cg_maxiter,
        "repetitions": repetitions,
        "design_deg": list(map(float, eta_deg)),
        "jax_devices": [str(device) for device in jax.devices()],
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_count": affinity,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "unset"),
        "OMP_PLACES": os.environ.get("OMP_PLACES", "unset"),
        "tesseract_mode": "in-process Tesseract.from_tesseract_api",
        "timings": {
            **timings,
            "native_direct_forward_pcg_seconds": direct_forward_seconds,
            "native_direct_adjoint_pcg_seconds": direct_adjoint_seconds,
            "estimated_tesseract_forward_overhead_seconds": max(
                0.0,
                timings["B_tesseract_cpp_forward_seconds"] - direct_forward_seconds,
            ),
            "system_construction_iprojection_forcing_raster_seconds": system_construction_seconds,
            "stage4_like_jax_value_gradient_seconds": full_jax_seconds,
            "stage4_like_tesseract_cpp_value_gradient_seconds": full_native_seconds,
            "exact_one_trial_audit_seconds": exact_audit_seconds,
            "exact_one_trial_native_cold_seconds": exact_native_audit_seconds,
            "exact_one_trial_jax_steady_seconds": exact_steady_jax_seconds,
            "exact_one_trial_native_steady_seconds": exact_steady_native_seconds,
        },
        "errors": {
            "forward_absolute_l2": float(np.linalg.norm(forward_diff)),
            "forward_relative_l2": float(
                np.linalg.norm(forward_diff) / np.linalg.norm(forward_ref)
            ),
            "forward_max_absolute": float(np.max(np.abs(forward_diff))),
            "gradient_absolute_l2": float(np.linalg.norm(gradient_diff)),
            "gradient_relative_l2": float(
                np.linalg.norm(gradient_diff) / np.linalg.norm(gradient_ref)
            ),
            "gradient_max_absolute": float(np.max(np.abs(gradient_diff))),
            "end_to_end_value_absolute": float(abs(full_jax_value - full_native_value)),
            "end_to_end_value_relative": float(
                abs(full_jax_value - full_native_value)
                / jnp.maximum(jnp.abs(full_jax_value), 1.0e-300)
            ),
            "end_to_end_gradient_relative_l2": float(
                np.linalg.norm(full_grad_diff) / np.linalg.norm(full_grad_ref)
            ),
            "end_to_end_gradient_max_absolute": float(np.max(np.abs(full_grad_diff))),
            "exact_one_trial_action_absolute": exact_value_absolute,
        },
        "native_forward_iterations": {
            "mean": float(np.mean(forward_stats["iterations"])),
            "max": int(np.max(forward_stats["iterations"])),
            "max_relative_residual": float(np.max(forward_stats["relative_residual"])),
        },
        "historical_120_cap_check": {
            "converged_count": int(np.sum(historical_120["converged"])),
            "system_count": int(q_np.shape[0]),
            "mean_relative_residual": float(
                np.mean(historical_120["relative_residual"])
            ),
            "max_relative_residual": float(
                np.max(historical_120["relative_residual"])
            ),
            "note": (
                "The former stage-4 cap returned unconverged iterates in the JAX "
                f"reference path; the shared cap is now {pcfg.cg_maxiter} without "
                "changing tolerance."
            ),
        },
        "native_adjoint_iterations": {
            "mean": float(np.mean(adjoint_stats["iterations"])),
            "max": int(np.max(adjoint_stats["iterations"])),
            "max_relative_residual": float(np.max(adjoint_stats["relative_residual"])),
        },
        "stage4_like_speedup": float(speedup),
        "exact_trial_steady_speedup": (
            float(exact_steady_jax_seconds / exact_steady_native_seconds)
            if exact_steady_jax_seconds is not None
            and exact_steady_native_seconds is not None
            else None
        ),
        "decision": (
            "integrate exact batching; CUDA unnecessary"
            if exact_steady_jax_seconds is not None
            and exact_steady_native_seconds is not None
            and exact_steady_jax_seconds / exact_steady_native_seconds >= 10.0
            else (
                "integrate proxy acceleration; CUDA unnecessary"
                if speedup >= 3.0
                else "speedup below 1.5x; stop and profile/recommend future proxy or CUDA"
            )
        ),
    }
    output_path = SCRIPT_DIR / "outputs" / "poisson_backend_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"results={output_path}", flush=True)


if __name__ == "__main__":
    main()
