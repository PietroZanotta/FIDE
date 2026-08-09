#!/usr/bin/env python3
"""Independent validation of the interacting-particle MGD implementation.

The quadratic Example A is unusually valuable because MGD Appendix E provides
an analytic law-level oracle: with Gaussian base and phi=(x,x^2), the MGD law
is Gaussian with the prescribed mean/covariance for every volatility sigma.
Here that oracle is N(0,1) for all t.

This script checks:
  1. predictor/corrector MGD against N(0,1) for several sigma values;
  2. empirical moment correction effectiveness at every step;
  3. a direct Theorem-3.1 Euler-Maruyama implementation as an independent
     cross-check (expected to have larger finite-replica moment fluctuations).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mgd import example_a_moment_path, simulate_mgd_predictor_corrector, simulate_mgd_theorem_euler

jax.config.update("jax_enable_x64", True)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
N_PARTICLES = 8192
HELDOUT_TIMES = np.linspace(0.0, 1.0, 11)
SIGMA_CONFIGS = [
    {"sigma2": 0.25, "steps": 1000, "seed": 1201},
    {"sigma2": 1.0, "steps": 1000, "seed": 1202},
    {"sigma2": 2.5, "steps": 1500, "seed": 1203},
]


def x0_quantiles(n=N_PARTICLES):
    u = (jnp.arange(n, dtype=jnp.float64) + 0.5) / n
    x = jax.scipy.special.ndtri(u)
    x = x - jnp.mean(x)
    return x / jnp.sqrt(jnp.mean(x * x))


def normal_w1(x):
    p = np.sort(np.asarray(x, dtype=np.float64))
    u = (np.arange(p.size) + 0.5) / p.size
    q = np.asarray(jax.scipy.special.ndtri(jnp.asarray(u)))
    return float(np.mean(np.abs(p - q)))


def heldout_summary(traj, steps):
    idx = np.rint(HELDOUT_TIMES * steps).astype(int)
    x = np.asarray(traj)[idx]
    rows = []
    for t, z in zip(HELDOUT_TIMES, x):
        rows.append({
            "t": float(t),
            "w1_to_gaussian_oracle": normal_w1(z),
            "mean_error": abs(float(np.mean(z))),
            "second_moment_error": abs(float(np.mean(z * z)) - 1.0),
            "fourth_moment": float(np.mean(z**4)),
            "fourth_moment_error_to_gaussian": abs(float(np.mean(z**4)) - 3.0),
        })
    return rows


def run():
    x0 = x0_quantiles()
    results = {"predictor_corrector_sigma_sweep": []}

    for cfg in SIGMA_CONFIGS:
        sigma = float(np.sqrt(cfg["sigma2"]))
        steps = int(cfg["steps"])
        moments, mdot = example_a_moment_path(steps)
        fun = jax.jit(lambda key: simulate_mgd_predictor_corrector(x0, key, sigma, moments, mdot))
        res = fun(jax.random.PRNGKey(cfg["seed"]))
        res.trajectory.block_until_ready()
        rows = heldout_summary(res.trajectory, steps)
        pred = np.asarray(res.predictor_moment_error)
        corr = np.asarray(res.corrected_moment_error)
        entry = {
            **cfg,
            "sigma": sigma,
            "max_w1_to_gaussian_oracle": float(max(r["w1_to_gaussian_oracle"] for r in rows)),
            "mean_w1_to_gaussian_oracle": float(np.mean([r["w1_to_gaussian_oracle"] for r in rows])),
            "max_mean_error": float(max(r["mean_error"] for r in rows)),
            "max_second_moment_error": float(max(r["second_moment_error"] for r in rows)),
            "rmse_fourth_moment_to_gaussian": float(np.sqrt(np.mean([(r["fourth_moment"] - 3.0)**2 for r in rows]))),
            "max_internal_corrected_moment_norm": float(np.max(corr)),
            "median_corrector_reduction_factor": float(np.median(pred / np.maximum(corr, 1e-30))),
            "max_normalized_gram_condition": float(jnp.max(res.gram_condition)),
            "per_time": rows,
        }
        results["predictor_corrector_sigma_sweep"].append(entry)

    # Independent direct-Theorem implementation at sigma=1.
    steps = 1000
    moments, mdot = example_a_moment_path(steps)
    theorem_fun = jax.jit(lambda key: simulate_mgd_theorem_euler(x0, key, 1.0, moments, mdot))
    theorem = theorem_fun(jax.random.PRNGKey(2202))
    theorem.trajectory.block_until_ready()
    theorem_rows = heldout_summary(theorem.trajectory, steps)
    results["theorem_3_1_euler_crosscheck_sigma1"] = {
        "steps": steps,
        "max_w1_to_gaussian_oracle": float(max(r["w1_to_gaussian_oracle"] for r in theorem_rows)),
        "max_mean_error": float(max(r["mean_error"] for r in theorem_rows)),
        "max_second_moment_error": float(max(r["second_moment_error"] for r in theorem_rows)),
        "per_time": theorem_rows,
        "note": "Finite-replica Euler-Maruyama does not project empirical moments each step; larger empirical fluctuations than predictor/corrector are expected.",
    }

    # Conservative correctness assertions.
    for entry in results["predictor_corrector_sigma_sweep"]:
        assert entry["max_w1_to_gaussian_oracle"] < 2.5e-2, entry
        assert entry["max_second_moment_error"] < 1e-4, entry
        assert entry["max_internal_corrected_moment_norm"] < 1e-4, entry
        assert entry["median_corrector_reduction_factor"] > 100.0, entry
    assert results["theorem_3_1_euler_crosscheck_sigma1"]["max_w1_to_gaussian_oracle"] < 3e-2
    return results


def make_plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for entry in results["predictor_corrector_sigma_sweep"]:
        rows = entry["per_time"]
        label = f"sigma^2={entry['sigma2']}"
        axes[0].plot(HELDOUT_TIMES, [r["w1_to_gaussian_oracle"] for r in rows], marker="o", label=label)
        axes[1].semilogy(HELDOUT_TIMES, np.maximum([r["second_moment_error"] for r in rows], 1e-12), marker="o", label=label)
    axes[0].set_title("Proper MGD vs Appendix-E Gaussian oracle")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("W1"); axes[0].legend(fontsize=8)
    axes[1].set_title("Empirical second-moment error")
    axes[1].set_xlabel("t"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mgd_validation.png", dpi=170); plt.close(fig)


def main():
    results = run()
    (OUT / "mgd_validation_metrics.json").write_text(json.dumps(results, indent=2))
    make_plot(results)
    compact = {
        "predictor_corrector": [
            {k: e[k] for k in ["sigma2", "steps", "max_w1_to_gaussian_oracle", "max_second_moment_error", "median_corrector_reduction_factor"]}
            for e in results["predictor_corrector_sigma_sweep"]
        ],
        "theorem_crosscheck": {k: results["theorem_3_1_euler_crosscheck_sigma1"][k] for k in ["max_w1_to_gaussian_oracle", "max_mean_error", "max_second_moment_error"]},
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
