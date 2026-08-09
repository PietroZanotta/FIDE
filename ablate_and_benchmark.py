#!/usr/bin/env python3
"""Ablations + microbenchmarks for the two-component JAX MFSI prototype.

Run:
    python ablate_and_benchmark.py

Ablations
---------
1. Envelope theorem: distortion gradients should not need d lambda*/d xi.
2. Correction energy: implicit/unrolled gradients should match finite differences;
   stop-gradient(lambda*) may differ.
3. Bridge design: optimize beta_xi(t) with implicit, unrolled, and stop modes.

Benchmarks
----------
1. Steady-state JIT forward and value+gradient latency versus quadrature size.
2. T1-only, T2-only, staged T1->T2, and fused T1->T2 latency at one time.

These are local JAX microbenchmarks, NOT measurements of future container/RPC
Tesseract overhead.  They answer whether the mathematical component boundary
and solver-aware VJP are sensible before adding distributed infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mfsi_components import (
    TARGET,
    beta_schedule,
    inverse_softplus,
    make_grid,
    mfsi_pipeline,
    moment_fiber_realizer,
    pipeline_objective,
    reference_transport,
)

jax.config.update("jax_enable_x64", True)

A = 0.8
GRID = make_grid(xmax=6.0, n=501)
TIMES = jnp.linspace(0.10, 0.90, 7)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def finite_difference(fun, p, eps=2e-4):
    vals = []
    for k in range(p.size):
        e = jnp.zeros_like(p).at[k].set(eps)
        vals.append((fun(p + e) - fun(p - e)) / (2.0 * eps))
    return jnp.asarray(vals)


def adam_optimize(fun, p0, steps=18, lr=0.10):
    p = p0
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    hist = []
    vg = jax.jit(jax.value_and_grad(fun))
    best_p, best_val = p, np.inf
    for i in range(1, steps + 1):
        val, g = vg(p)
        val.block_until_ready()
        val_f = float(val)
        hist.append(val_f)
        if val_f < best_val:
            best_p, best_val = p, val_f
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1.0 - 0.9**i)
        vh = v / (1.0 - 0.999**i)
        p = p - lr * mh / (jnp.sqrt(vh) + 1e-8)
    val = fun(p)
    if float(val) < best_val:
        best_p = p
    return best_p, hist


def _block_tree(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def benchmark_jitted(fun, arg, repeats=10):
    """Return first-call (compile+execute) and median warm latency in ms."""
    start = perf_counter()
    out = fun(arg)
    _block_tree(out)
    first_ms = (perf_counter() - start) * 1e3

    samples = []
    for _ in range(repeats):
        start = perf_counter()
        out = fun(arg)
        _block_tree(out)
        samples.append((perf_counter() - start) * 1e3)
    return {"first_call_ms": first_ms, "warm_median_ms": float(np.median(samples))}


def summarize_schedule(p, grid=GRID, times=TIMES):
    e = lambda pp: pipeline_objective(pp, grid, times, A, kind="correction_energy", differentiation="implicit")
    d = lambda pp: pipeline_objective(pp, grid, times, A, kind="distortion", differentiation="implicit")
    betas = np.asarray(jax.vmap(lambda t: beta_schedule(t, p))(times))
    ess = np.asarray(jax.vmap(lambda t: mfsi_pipeline(p, grid, t, A, differentiation="implicit")[1].ess_fraction)(times))
    return {
        "params": np.asarray(p).tolist(),
        "beta_mean": float(np.mean(betas)),
        "beta_max_abs_error_from_1": float(np.max(np.abs(betas - 1.0))),
        "correction_energy": float(e(p)),
        "projection_distortion": float(d(p)),
        "min_population_ess_fraction": float(np.min(ess)),
    }


def run_ablations():
    p0 = jnp.array([inverse_softplus(0.35), 0.35, -0.20], dtype=jnp.float64)

    objectives = {}
    for kind in ("distortion", "correction_energy"):
        for mode in ("unrolled", "implicit", "stop"):
            objectives[(kind, mode)] = lambda p, k=kind, m=mode: pipeline_objective(
                p, GRID, TIMES, A, kind=k, differentiation=m
            )

    grad = {}
    for key, fun in objectives.items():
        grad[f"{key[0]}_{key[1]}"] = np.asarray(jax.jit(jax.grad(fun))(p0))

    # Finite differences use the unrolled primal as an independent numerical check.
    d_fd = np.asarray(finite_difference(jax.jit(objectives[("distortion", "unrolled")]), p0))
    e_fd = np.asarray(finite_difference(jax.jit(objectives[("correction_energy", "unrolled")]), p0))

    p_imp, hist_imp = adam_optimize(objectives[("correction_energy", "implicit")], p0)
    p_unr, hist_unr = adam_optimize(objectives[("correction_energy", "unrolled")], p0)
    p_stop, hist_stop = adam_optimize(objectives[("correction_energy", "stop")], p0)
    p_ideal = jnp.array([inverse_softplus(1.0), 0.0, 0.0], dtype=jnp.float64)

    rel = lambda x, y: float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))
    result = {
        "gradient_checks": {k: v.tolist() for k, v in grad.items()},
        "finite_difference": {"distortion": d_fd.tolist(), "correction_energy": e_fd.tolist()},
        "relative_errors": {
            "distortion_unrolled_vs_fd": rel(grad["distortion_unrolled"], d_fd),
            "distortion_implicit_vs_fd": rel(grad["distortion_implicit"], d_fd),
            "distortion_stop_vs_unrolled": rel(grad["distortion_stop"], grad["distortion_unrolled"]),
            "correction_unrolled_vs_fd": rel(grad["correction_energy_unrolled"], e_fd),
            "correction_implicit_vs_fd": rel(grad["correction_energy_implicit"], e_fd),
            "correction_implicit_vs_unrolled": rel(grad["correction_energy_implicit"], grad["correction_energy_unrolled"]),
            "correction_stop_vs_unrolled": rel(grad["correction_energy_stop"], grad["correction_energy_unrolled"]),
        },
        "initial": summarize_schedule(p0),
        "optimized_implicit": summarize_schedule(p_imp),
        "optimized_unrolled": summarize_schedule(p_unr),
        "optimized_stop": summarize_schedule(p_stop),
        "known_ideal": summarize_schedule(p_ideal),
        "optimization_history": {
            "implicit": hist_imp,
            "unrolled": hist_unr,
            "stop": hist_stop,
        },
        "optimized_params": {
            "implicit": np.asarray(p_imp).tolist(),
            "unrolled": np.asarray(p_unr).tolist(),
            "stop": np.asarray(p_stop).tolist(),
        },
    }
    return result, p0, p_imp, p_unr, p_stop


def run_benchmarks(p0):
    sizes = [251, 501]
    scaling = []
    for n in sizes:
        grid = make_grid(xmax=6.0, n=n)
        times = jnp.linspace(0.10, 0.90, 7)
        row = {"grid_n": n}
        for mode in ("unrolled", "implicit", "stop"):
            obj = lambda p, m=mode: pipeline_objective(
                p, grid, times, A, kind="correction_energy", differentiation=m
            )
            fwd = jax.jit(obj)
            vg = jax.jit(jax.value_and_grad(obj))
            row[f"forward_{mode}"] = benchmark_jitted(fwd, p0, repeats=5)
            row[f"value_grad_{mode}"] = benchmark_jitted(vg, p0, repeats=5)
        scaling.append(row)

    # Component-level workload / fusion benchmark at representative t.
    grid = make_grid(xmax=6.0, n=1001)
    t = jnp.asarray(0.5)
    t1 = jax.jit(lambda p: reference_transport(p, grid, t, A))
    ref = t1(p0)
    _block_tree(ref)
    t2 = jax.jit(lambda r: moment_fiber_realizer(r, grid, TARGET, differentiation="implicit"))
    fused = jax.jit(lambda p: mfsi_pipeline(p, grid, t, A, differentiation="implicit")[1])

    t1_b = benchmark_jitted(t1, p0, repeats=10)
    t2_b = benchmark_jitted(t2, ref, repeats=10)
    fused_b = benchmark_jitted(fused, p0, repeats=10)

    # Explicit staged execution to capture two Python/JAX dispatches (still no RPC).
    def staged(p):
        r = t1(p)
        f = t2(r)
        return f.correction_energy

    # Warm both compiled pieces before measuring staged dispatch.
    _block_tree(t1(p0)); _block_tree(t2(ref))
    staged_samples = []
    for _ in range(10):
        start = perf_counter()
        out = staged(p0)
        out.block_until_ready()
        staged_samples.append((perf_counter() - start) * 1e3)

    return {
        "note": "CPU local-JAX timings only; excludes future Tesseract container/RPC overhead.",
        "scaling": scaling,
        "component_breakdown_n1001_t0_5": {
            "reference_transport_T1": t1_b,
            "moment_fiber_realizer_T2": t2_b,
            "fused_T1_to_T2": fused_b,
            "staged_two_dispatch_warm_median_ms": float(np.median(staged_samples)),
        },
    }


def make_plot(ablation, benchmark, p0, p_imp, p_unr, p_stop):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for mode in ("implicit", "unrolled", "stop"):
        raw = np.asarray(ablation["optimization_history"][mode])
        axes[0, 0].semilogy(
            np.maximum(np.minimum.accumulate(raw), 1e-18), label=mode
        )
    axes[0, 0].set_title("Bridge optimization: best correction energy")
    axes[0, 0].set_xlabel("Adam step")
    axes[0, 0].legend()

    tnp = np.asarray(TIMES)
    for p, label in [(p0, "initial"), (p_imp, "implicit"), (p_unr, "unrolled"), (p_stop, "stop")]:
        axes[0, 1].plot(tnp, np.asarray(jax.vmap(lambda t: beta_schedule(t, p))(TIMES)), label=label)
    axes[0, 1].axhline(1.0, linestyle="--", linewidth=1, label="ideal beta=1")
    axes[0, 1].set_title("Learned reference schedule")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].legend()

    ns = np.asarray([r["grid_n"] for r in benchmark["scaling"]])
    for mode in ("unrolled", "implicit", "stop"):
        ys = [r[f"value_grad_{mode}"]["warm_median_ms"] for r in benchmark["scaling"]]
        axes[1, 0].plot(ns, ys, marker="o", label=mode)
    axes[1, 0].set_title("Warm value+gradient latency")
    axes[1, 0].set_xlabel("quadrature grid size")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].legend()

    labels = ["T1", "T2", "fused", "staged"]
    cb = benchmark["component_breakdown_n1001_t0_5"]
    vals = [
        cb["reference_transport_T1"]["warm_median_ms"],
        cb["moment_fiber_realizer_T2"]["warm_median_ms"],
        cb["fused_T1_to_T2"]["warm_median_ms"],
        cb["staged_two_dispatch_warm_median_ms"],
    ]
    axes[1, 1].bar(labels, vals)
    axes[1, 1].set_title("Local component-boundary cost (n=1001)")
    axes[1, 1].set_ylabel("warm median ms")

    fig.tight_layout()
    fig.savefig(OUT / "ablation_benchmark.png", dpi=160)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Implicit-gradient ablations and optional local latency microbenchmarks.")
    parser.add_argument(
        "--timings", action="store_true",
        help="Also run the compilation-heavy local JAX latency microbenchmarks.",
    )
    args = parser.parse_args()

    ablation, p0, p_imp, p_unr, p_stop = run_ablations()
    (OUT / "ablation_metrics.json").write_text(json.dumps(ablation, indent=2))
    report = {
        "relative_errors": ablation["relative_errors"],
        "initial": ablation["initial"],
        "optimized_implicit": ablation["optimized_implicit"],
        "optimized_unrolled": ablation["optimized_unrolled"],
        "optimized_stop": ablation["optimized_stop"],
    }

    if args.timings:
        benchmark = run_benchmarks(p0)
        combined = {"ablation": ablation, "benchmark": benchmark}
        (OUT / "ablation_benchmark_metrics.json").write_text(json.dumps(combined, indent=2))
        make_plot(ablation, benchmark, p0, p_imp, p_unr, p_stop)
        report["component_breakdown"] = benchmark["component_breakdown_n1001_t0_5"]
    else:
        report["timing_note"] = "Pass --timings for the compilation-heavy local JAX latency benchmark."

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
