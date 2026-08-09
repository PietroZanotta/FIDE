#!/usr/bin/env python3
"""Matched Example-A benchmark with a *proper interacting-particle MGD* baseline.

Methods
-------
raw_si
    Deterministic probability flow of the uncorrected MFSI reference bridge.

moment_tangent
    Self-consistent interacting Gram correction that fixes the target moment
    rates but does not attempt to realize the complete I-projected law.

mgd
    Actual Moment Guided Diffusion predictor/corrector (Lempereur et al. 2026,
    Sec. 3.2, Eqs. 18-21), simulated with interacting particles and Brownian
    noise.  For this Example A, the MGD variance-preserving moment path is
    m_t=(0,1).  Appendix E gives an independent oracle: the exact MGD law is
    N(0,1) for all t and all sigma.  The code does NOT substitute that oracle;
    it uses it only as an implementation correctness check.

mfsi_learned / mfsi_learned_safe
    Learned flow-matched reference plus time-conditioned Deep-Ritz correction,
    with and without the optional population-moment safety layer.

mfsi_oracle
    Exact 1D I-projected path with exact weighted-Poisson realization; used only
    as an Experiment-A reporting oracle, never as a learned-model pass criterion.

Fairness / interpretation
-------------------------
* Every method starts from the same deterministic Gaussian quantile ensemble.
* Raw SI, tangent and MFSI use the same Heun grid (240 steps).
* MGD uses its own paper-style stochastic discretization with 1000 steps at
  sigma=1; forcing MGD onto the 240-step deterministic budget would not be a
  faithful implementation. Runtime and step counts are reported separately.
* All methods are evaluated at the same held-out times t=0,.1,...,1 against
  the exact MFSI projected marginal using W1/W2/KS and hidden fourth moment.
* MGD is additionally evaluated against its own analytic Gaussian oracle. This
  distinguishes "MGD is implemented correctly" from "MGD solves MFSI's task".

Run:
    python benchmark_methods.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mfsi_components import (
    TARGET, ensemble_safety_velocity, flatten_mlp, integrate_learned_flow, learned_correction,
    load_learned_model, make_grid, mfsi_pipeline, reference_transport, reference_velocity,
)
from backend_runtime import TesseractRESTBackend, normalize_backend
from mgd import example_a_moment_path, simulate_mgd_predictor_corrector

jax.config.update("jax_enable_x64", True)

A = 0.8
GRID = make_grid(xmax=7.0, n=1601)
N_STEPS = 120
TIME_NODES = jnp.linspace(0.0, 1.0, N_STEPS + 1)
HELDOUT_TIMES = np.linspace(0.0, 1.0, 11)
HELDOUT_INDICES = np.rint(HELDOUT_TIMES * N_STEPS).astype(int)
N_PARTICLES = 2048

MGD_SIGMA = 1.0
MGD_STEPS = 1000
MGD_SEED = 2026
MGD_HELDOUT_INDICES = np.rint(HELDOUT_TIMES * MGD_STEPS).astype(int)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
MODEL_PATH = OUT / "learned_mfsi_example_a.npz"
METHODS = ("raw_si", "moment_tangent", "mgd", "mfsi_learned", "mfsi_learned_safe", "mfsi_oracle")


def _block(x):
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def _timed_jit(fun, arg, repeats=3):
    jf = jax.jit(fun)
    start = perf_counter()
    out = jf(arg)
    _block(out)
    first = (perf_counter() - start) * 1e3
    vals = []
    for _ in range(repeats):
        start = perf_counter()
        out = jf(arg)
        _block(out)
        vals.append((perf_counter() - start) * 1e3)
    return out, {"first_call_ms": float(first), "warm_median_ms": float(np.median(vals))}


def raw_velocity_fields(times):
    return jax.vmap(lambda t: reference_transport(None, GRID, t, A).velocity)(times)


def mfsi_oracle_velocity_fields(times):
    return jax.vmap(lambda t: mfsi_pipeline(None, GRID, t, A, differentiation="implicit")[1].velocity)(times)


def target_densities(times):
    return jax.vmap(lambda t: mfsi_pipeline(None, GRID, t, A, differentiation="implicit")[1].q)(times)


def reference_densities(times):
    return jax.vmap(lambda t: reference_transport(None, GRID, t, A).q_ref)(times)


def initial_quantile_particles(n=N_PARTICLES):
    """Shared deterministic cubature of N(0,1), normalized in first two moments."""
    u = (jnp.arange(n, dtype=jnp.float64) + 0.5) / n
    x = jax.scipy.special.ndtri(u)
    x = x - jnp.mean(x)
    return x / jnp.sqrt(jnp.mean(x * x))


def integrate_velocity_field(field, x0):
    """Heun integration through v(t_k,x_grid), shared by raw SI and MFSI."""
    dt = TIME_NODES[1] - TIME_NODES[0]
    xgrid = GRID.x

    def step(x, pair):
        vk, vk1 = pair
        k1 = jnp.interp(x, xgrid, vk)
        pred = x + dt * k1
        k2 = jnp.interp(pred, xgrid, vk1)
        nxt = x + 0.5 * dt * (k1 + k2)
        return nxt, nxt

    _, tail = jax.lax.scan(step, x0, (field[:-1], field[1:]))
    return jnp.concatenate([x0[None, :], tail], axis=0)


def tangent_velocity_particles(x, t):
    """Self-consistent Eq.45-style moment-rate correction on its own ensemble."""
    u = reference_velocity(x, t, A, None)
    jphi = jnp.stack([jnp.ones_like(x), 2.0 * x], axis=-1)
    G = (jphi.T @ jphi) / x.size
    r = jnp.mean(jphi * u[:, None], axis=0)  # c_dot=0
    coeff = jnp.linalg.solve(G + 1e-10 * jnp.eye(2, dtype=x.dtype), r)
    return u - jphi @ coeff


def integrate_tangent_interacting(x0):
    dt = TIME_NODES[1] - TIME_NODES[0]

    def step(x, k):
        t0, t1 = TIME_NODES[k], TIME_NODES[k + 1]
        k1 = tangent_velocity_particles(x, t0)
        pred = x + dt * k1
        k2 = tangent_velocity_particles(pred, t1)
        nxt = x + 0.5 * dt * (k1 + k2)
        return nxt, nxt

    _, tail = jax.lax.scan(step, x0, jnp.arange(N_STEPS))
    return jnp.concatenate([x0[None, :], tail], axis=0)


def run_mgd(x0, key):
    moments, moment_dot = example_a_moment_path(MGD_STEPS, dtype=x0.dtype)
    return simulate_mgd_predictor_corrector(
        x0,
        key,
        MGD_SIGMA,
        moments,
        moment_dot,
        ridge=1e-7,
    )


def integrate_learned_tesseract(model, x0, n_steps=N_STEPS, *, safety=False):
    """Heun generation through the two served Tesseract Core components."""
    client = TesseractRESTBackend.from_env()
    ref_flat = np.asarray(flatten_mlp(model.reference_params))
    pot_flat = np.asarray(flatten_mlp(model.potential_params))
    times = np.linspace(0.0, 1.0, n_steps + 1)
    dt = 1.0 / n_steps
    x = np.asarray(x0, dtype=np.float64)
    traj = [x.copy()]

    def field(t, z):
        z = np.asarray(z, dtype=np.float64)
        u = client.reference_velocity(ref_flat, t, z)[:, 0]
        ph = np.stack([z, z * z], axis=-1)
        jpu = np.stack([u, 2.0 * z * u], axis=-1)
        out = client.fiber_apply(
            x=z, t=t, velocity=u, phi_values=ph, jphi_u=jpu,
            target=np.asarray(TARGET), log_base_weights=np.zeros(z.size),
            potential_params=pot_flat,
        )
        v = np.asarray(out["velocity"])[:, 0]
        if safety:
            v = np.asarray(ensemble_safety_velocity(jnp.asarray(z), jnp.asarray(v)))
        return v

    for i in range(n_steps):
        v0 = field(float(times[i]), x)
        xp = x + dt * v0
        v1 = field(float(times[i + 1]), xp)
        x = x + 0.5 * dt * (v0 + v1)
        traj.append(x.copy())
    return times, np.stack(traj, axis=0)


def continuous_cdf_and_quantiles(q, particle_u):
    """Trapezoidal CDF and target quantiles on the fixed spatial grid."""
    x = np.asarray(GRID.x, dtype=np.float64)
    qn = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    inc = 0.5 * (qn[:-1] + qn[1:]) * np.diff(x)
    cdf = np.concatenate([[0.0], np.cumsum(inc)])
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return cdf, np.interp(particle_u, cdf, x)


def empirical_metrics(particles, q_target):
    particles = np.sort(np.asarray(particles, dtype=np.float64))
    n = particles.size
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    cdf_t, target_quant = continuous_cdf_and_quantiles(q_target, u)
    diff = particles - target_quant

    target_cdf_at_particles = np.interp(
        particles, np.asarray(GRID.x), cdf_t, left=0.0, right=1.0
    )
    mean = float(np.mean(particles))
    second = float(np.mean(particles**2))
    fourth = float(np.mean(particles**4))
    target_fourth = float(jnp.sum(GRID.w * q_target * GRID.x**4))

    return {
        "w1_to_projected_target": float(np.mean(np.abs(diff))),
        "w2_to_projected_target": float(np.sqrt(np.mean(diff * diff))),
        "ks_to_projected_target": float(np.max(np.abs(target_cdf_at_particles - u))),
        "mean_error": abs(mean - float(TARGET[0])),
        "second_moment_error": abs(second - float(TARGET[1])),
        "fourth_moment": fourth,
        "target_fourth_moment": target_fourth,
        "fourth_moment_error": abs(fourth - target_fourth),
    }


def standard_normal_w1(particles):
    """W1 to the analytic MGD oracle N(0,1) using midpoint normal quantiles."""
    p = np.sort(np.asarray(particles, dtype=np.float64))
    u = (np.arange(p.size, dtype=np.float64) + 0.5) / p.size
    q = np.asarray(jax.scipy.special.ndtri(jnp.asarray(u)))
    return float(np.mean(np.abs(p - q)))


def reference_integration_error(particles, q_reference):
    p = np.sort(np.asarray(particles, dtype=np.float64))
    u = (np.arange(p.size, dtype=np.float64) + 0.5) / p.size
    _, ref_quant = continuous_cdf_and_quantiles(q_reference, u)
    return float(np.mean(np.abs(p - ref_quant)))


def summarize(per_time):
    interior = [r for r in per_time if 0.0 < r["t"] < 1.0]
    return {
        "mean_interior_w1": float(np.mean([r["w1_to_projected_target"] for r in interior])),
        "max_interior_w1": float(np.max([r["w1_to_projected_target"] for r in interior])),
        "mean_interior_w2": float(np.mean([r["w2_to_projected_target"] for r in interior])),
        "mean_interior_ks": float(np.mean([r["ks_to_projected_target"] for r in interior])),
        "max_mean_error": float(np.max([r["mean_error"] for r in per_time])),
        "max_second_moment_error": float(np.max([r["second_moment_error"] for r in per_time])),
        "rmse_fourth_moment_vs_target": float(np.sqrt(np.mean([r["fourth_moment_error"] ** 2 for r in interior]))),
        "endpoint_w1_t1": float(per_time[-1]["w1_to_projected_target"]),
    }


def run(backend="jax"):
    backend = normalize_backend(backend)
    x0 = initial_quantile_particles()

    # Exact target/reference laws at common held-out times.
    held_t = jnp.asarray(HELDOUT_TIMES, dtype=jnp.float64)
    q_target = jax.jit(target_densities)(held_t)
    q_ref = jax.jit(reference_densities)(held_t)
    _block(q_target); _block(q_ref)

    # Deterministic fields and flows.
    field_timings = {}
    raw_field, field_timings["raw_si_field"] = _timed_jit(raw_velocity_fields, TIME_NODES)
    mfsi_field, field_timings["mfsi_oracle_field"] = _timed_jit(mfsi_oracle_velocity_fields, TIME_NODES)

    raw_traj, raw_timing = _timed_jit(lambda field: integrate_velocity_field(field, x0), raw_field)
    mfsi_traj, mfsi_timing = _timed_jit(lambda field: integrate_velocity_field(field, x0), mfsi_field)
    tangent_traj, tangent_timing = _timed_jit(lambda x: integrate_tangent_interacting(x), x0)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Run validate_pipeline.py --retrain first to create {MODEL_PATH}")
    learned_model = load_learned_model(MODEL_PATH)
    if backend == "jax":
        learned_traj, learned_timing = _timed_jit(
            lambda x: integrate_learned_flow(learned_model, x, n_steps=N_STEPS, safety=False)[1], x0, repeats=5
        )
        learned_safe_traj, learned_safe_timing = _timed_jit(
            lambda x: integrate_learned_flow(learned_model, x, n_steps=N_STEPS, safety=True)[1], x0, repeats=5
        )
    else:
        start = perf_counter(); _, learned_traj = integrate_learned_tesseract(learned_model, x0, N_STEPS, safety=False)
        learned_timing = {"first_call_ms": (perf_counter()-start)*1e3, "warm_median_ms": float("nan")}
        start = perf_counter(); _, learned_safe_traj = integrate_learned_tesseract(learned_model, x0, N_STEPS, safety=True)
        learned_safe_timing = {"first_call_ms": (perf_counter()-start)*1e3, "warm_median_ms": float("nan")}

    # Proper MGD: stochastic interacting-particle predictor/corrector.
    # _timed_jit accepts a single positional argument. Wrap (x0,key) explicitly.
    mgd_fun = lambda pair: run_mgd(pair[0], pair[1])
    mgd_result, mgd_timing = _timed_jit(
        mgd_fun,
        (x0, jax.random.PRNGKey(MGD_SEED)),
        repeats=5,
    )

    # Common held-out particle snapshots.
    held_particles = {
        "raw_si": np.asarray(raw_traj)[HELDOUT_INDICES],
        "moment_tangent": np.asarray(tangent_traj)[HELDOUT_INDICES],
        "mgd": np.asarray(mgd_result.trajectory)[MGD_HELDOUT_INDICES],
        "mfsi_learned": np.asarray(learned_traj)[HELDOUT_INDICES],
        "mfsi_learned_safe": np.asarray(learned_safe_traj)[HELDOUT_INDICES],
        "mfsi_oracle": np.asarray(mfsi_traj)[HELDOUT_INDICES],
    }

    per_method = {}
    for name in METHODS:
        rows = []
        for j, t in enumerate(HELDOUT_TIMES):
            row = empirical_metrics(held_particles[name][j], q_target[j])
            row["t"] = float(t)
            if name == "raw_si":
                row["w1_to_own_reference_marginal"] = reference_integration_error(held_particles[name][j], q_ref[j])
            if name == "mgd":
                row["w1_to_own_gaussian_oracle"] = standard_normal_w1(held_particles[name][j])
            rows.append(row)
        per_method[name] = {"summary": summarize(rows), "per_time": rows}

    # Explicit function-evaluation accounting. NFE means velocity-field calls;
    # MGD is a non-neural predictor/corrector and is reported with NFE=0 plus
    # its native step count instead of forcing an artificial equivalence.
    for name in METHODS:
        deterministic = name != "mgd"
        per_method[name]["summary"]["nfe"] = int(2 * N_STEPS if deterministic else 0)
        per_method[name]["summary"]["integration_steps"] = int(N_STEPS if deterministic else MGD_STEPS)

    # Paper metrics: learned correction energy and target projection distortion.
    corr_vals, dist_vals = [], []
    for j, t in enumerate(HELDOUT_TIMES):
        tj = jnp.asarray(t)
        _, fibj = mfsi_pipeline(None, GRID, tj, A, differentiation="implicit")
        dc = learned_correction(learned_model.potential_params, tj, GRID.x)
        corr_vals.append(float(0.5 * jnp.sum(GRID.w * q_target[j] * dc * dc)))
        dist_vals.append(float(fibj.projection_distortion))
    path_functionals = {
        "integrated_learned_correction_energy": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(np.asarray(corr_vals), HELDOUT_TIMES)),
        "integrated_projection_distortion": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(np.asarray(dist_vals), HELDOUT_TIMES)),
        "per_time": [
            {"t": float(t), "learned_correction_energy": ce, "projection_distortion": dp}
            for t, ce, dp in zip(HELDOUT_TIMES, corr_vals, dist_vals)
        ],
    }

    # Oracle and implementation sanity checks.
    xg = GRID.x
    target_checks = []
    for j, t in enumerate(HELDOUT_TIMES):
        q = q_target[j]
        target_checks.append({
            "t": float(t),
            "mass_error": abs(float(jnp.sum(GRID.w * q)) - 1.0),
            "mean_error": abs(float(jnp.sum(GRID.w * q * xg))),
            "second_moment_error": abs(float(jnp.sum(GRID.w * q * xg * xg)) - 1.0),
        })

    raw_self = [r["w1_to_own_reference_marginal"] for r in per_method["raw_si"]["per_time"]]
    mgd_self = [r["w1_to_own_gaussian_oracle"] for r in per_method["mgd"]["per_time"]]

    sanity = {
        "max_target_mass_error": float(max(r["mass_error"] for r in target_checks)),
        "max_target_mean_error": float(max(r["mean_error"] for r in target_checks)),
        "max_target_second_moment_error": float(max(r["second_moment_error"] for r in target_checks)),
        "max_raw_si_w1_to_own_exact_marginal": float(max(raw_self)),
        "max_mgd_w1_to_own_gaussian_oracle": float(max(mgd_self)),
        "mgd_max_internal_corrected_moment_norm": float(jnp.max(mgd_result.corrected_moment_error)),
        "mgd_max_normalized_gram_condition": float(jnp.max(mgd_result.gram_condition)),
        "notes": [
            "MGD particles are actually simulated; N(0,1) is used only as Appendix-E oracle.",
            "MGD uses Eq. (19) corrector sign/scale together with Eq. (21).",
            "Raw-SI self-W1 isolates deterministic flow discretization/cubature error.",
            "mfsi_oracle uses the exact 1D Poisson correction only as an Example-A reporting floor.",
            "mfsi_learned and mfsi_learned_safe use the learned flow-matching + Deep-Ritz model.",
        ],
    }

    # Fail loudly if either our benchmark oracle or MGD implementation is suspect.
    assert sanity["max_target_mass_error"] < 1e-10, sanity
    assert sanity["max_target_mean_error"] < 1e-10, sanity
    assert sanity["max_target_second_moment_error"] < 1e-9, sanity
    assert sanity["max_raw_si_w1_to_own_exact_marginal"] < 8e-4, sanity
    assert sanity["max_mgd_w1_to_own_gaussian_oracle"] < 3e-2, sanity
    assert sanity["mgd_max_internal_corrected_moment_norm"] < 5e-5, sanity
    assert per_method["moment_tangent"]["summary"]["max_second_moment_error"] < 1e-4
    assert per_method["mgd"]["summary"]["max_second_moment_error"] < 5e-5
    assert per_method["mfsi_oracle"]["summary"]["mean_interior_w1"] < 8e-4
    assert per_method["mfsi_learned_safe"]["summary"]["max_second_moment_error"] < 1e-4

    result = {
        "experiment": "Example A full-law benchmark with proper MGD",
        "backend": backend,
        "a": A,
        "target_moments": [0.0, 1.0],
        "grid": {"xmax": 7.0, "n": int(GRID.x.size)},
        "common_evaluation": {
            "particles": N_PARTICLES,
            "heldout_times": HELDOUT_TIMES.tolist(),
            "distribution_metrics": ["W1", "W2", "KS", "fourth moment"],
        },
        "deterministic_integration": {
            "scheme": "Heun / explicit trapezoidal",
            "steps": N_STEPS,
            "velocity_interpolations": 2 * N_STEPS,
        },
        "mgd_integration": {
            "implementation": "interacting-particle predictor/corrector, Sec. 3.2 Eqs. (18)-(21)",
            "sigma": MGD_SIGMA,
            "steps": MGD_STEPS,
            "seed": MGD_SEED,
            "gram_ridge_after_diagonal_normalization": 1e-7,
            "moment_path": "variance-preserving MGD interpolant; m_t=(0,1) exactly",
            "analytic_oracle": "Appendix E: N(0,1) at every t for all sigma",
        },
        "baseline_definitions": {
            "raw_si": "uncorrected stochastic-interpolant probability flow",
            "moment_tangent": "self-consistent interacting Gram correction of reference velocity; fixes moment rates only",
            "mgd": "actual MGD predictor/corrector with empirical Gram solves and Brownian replicas; not an analytic substitution",
            "mfsi_learned": "flow-matched reference + learned Deep-Ritz correction",
            "mfsi_learned_safe": "learned MFSI plus optional population-rate safety correction",
            "mfsi_oracle": "exact 1D weighted-Poisson realization, reporting oracle only",
        },
        "sanity_checks": sanity,
        "methods": per_method,
        "path_functionals": path_functionals,
        "timings_ms": {
            "field_construction": field_timings,
            "integration": {
                "raw_si": raw_timing,
                "moment_tangent": tangent_timing,
                "mgd": mgd_timing,
                "mfsi_learned": learned_timing,
                "mfsi_learned_safe": learned_safe_timing,
                "mfsi_oracle": mfsi_timing,
            },
            "warning": "Local CPU/JAX microbenchmark; methods use their faithful native discretizations, not equal wall-clock budgets.",
        },
    }
    return result, held_particles, q_target


def make_plot(result, held_particles, q_target):
    t = HELDOUT_TIMES
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for name in METHODS:
        rows = result["methods"][name]["per_time"]
        axes[0, 0].plot(t, [r["w1_to_projected_target"] for r in rows], marker="o", label=name)
    axes[0, 0].set_title("Full-law error: W1 to MFSI projected oracle")
    axes[0, 0].set_xlabel("t"); axes[0, 0].set_ylabel("W1"); axes[0, 0].legend(fontsize=8)

    for name in METHODS:
        rows = result["methods"][name]["per_time"]
        axes[0, 1].semilogy(t, np.maximum([r["second_moment_error"] for r in rows], 1e-12), marker="o", label=name)
    axes[0, 1].set_title("Measured second-moment error")
    axes[0, 1].set_xlabel("t"); axes[0, 1].legend(fontsize=8)

    for name in METHODS:
        rows = result["methods"][name]["per_time"]
        axes[1, 0].plot(t, [r["fourth_moment"] for r in rows], marker="o", label=name)
    target_fourth = [r["target_fourth_moment"] for r in result["methods"]["mfsi_oracle"]["per_time"]]
    axes[1, 0].plot(t, target_fourth, linestyle="--", linewidth=2, label="projected target")
    axes[1, 0].set_title("Held-out fourth moment")
    axes[1, 0].set_xlabel("t"); axes[1, 0].legend(fontsize=8)

    # CDF at t=.5.
    j = 5
    q = np.asarray(q_target[j]); xnp = np.asarray(GRID.x)
    inc = 0.5 * (q[:-1] + q[1:]) * np.diff(xnp)
    cdf = np.concatenate([[0.0], np.cumsum(inc)]); cdf /= cdf[-1]
    axes[1, 1].plot(xnp, cdf, linewidth=2, label="projected target")
    for name in METHODS:
        p = np.sort(held_particles[name][j]); ecdf = (np.arange(p.size) + 0.5) / p.size
        axes[1, 1].plot(p, ecdf, linewidth=1, label=name)
    axes[1, 1].set_xlim(-3.0, 3.0); axes[1, 1].set_title("CDF at t=0.5")
    axes[1, 1].set_xlabel("x"); axes[1, 1].legend(fontsize=8)

    fig.tight_layout(); fig.savefig(OUT / "method_benchmark.png", dpi=170); plt.close(fig)

    # Paper-facing density overlay at the central held-out time.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xnp, q, linewidth=2.5, label="projected target")
    bins = np.linspace(-3.5, 3.5, 90)
    for name in ("raw_si", "moment_tangent", "mfsi_learned_safe", "mgd"):
        vals = np.asarray(held_particles[name][j])
        hist, edges = np.histogram(vals, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, linewidth=1.2, label=name)
    ax.set_xlim(-3.5, 3.5); ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("Example A density overlay at t=0.5")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "example_a_density_overlay.png", dpi=180); plt.close(fig)


def write_summary_csv(result):
    cols = [
        "method", "mean_interior_w1", "max_interior_w1", "mean_interior_w2",
        "mean_interior_ks", "max_mean_error", "max_second_moment_error",
        "rmse_fourth_moment_vs_target", "endpoint_w1_t1", "nfe", "integration_steps",
    ]
    with (OUT / "method_benchmark_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, **result["methods"][method]["summary"]})


def main():
    import argparse
    p = argparse.ArgumentParser(description="Experiment-A matched benchmark")
    p.add_argument("--backend", choices=("tesseract", "jax"), default=normalize_backend(None),
                   help="learned-component execution backend; default: tesseract")
    args = p.parse_args()
    result, held_particles, q_target = run(args.backend)
    (OUT / "method_benchmark_metrics.json").write_text(json.dumps(result, indent=2))
    write_summary_csv(result)
    make_plot(result, held_particles, q_target)
    print(json.dumps({
        "sanity_checks": result["sanity_checks"],
        "summaries": {k: v["summary"] for k, v in result["methods"].items()},
        "path_functionals": result["path_functionals"],
        "timings_ms": result["timings_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
