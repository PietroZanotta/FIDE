#!/usr/bin/env python3
"""Train and validate the learned MFSI Example-A pipeline.

Pass/fail criteria intentionally avoid analytic-oracle path error so the same
logic transfers to later experiments.  The exact 1D oracle is reported only in
an optional debug section.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from mfsi_components import (
    LearnedMFSIModel,
    TARGET,
    calibrate_empirical_implicit,
    empirical_fiber_state,
    flatten_mlp,
    heldout_learned_diagnostics,
    integrate_learned_flow,
    learned_correction,
    load_learned_model,
    make_grid,
    mfsi_pipeline,
    phi,
    reference_velocity,
    reference_velocity_net,
    sample_reference_bridge,
    sample_reference_bridge_times,
    save_learned_model,
    train_deep_ritz,
    train_reference_flow_matching,
    weighted_l2,
)

jax.config.update("jax_enable_x64", True)

A = 0.8
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
MODEL_PATH = OUT / "learned_mfsi_example_a.npz"
HELDOUT_TIMES = jnp.linspace(0.1, 0.9, 9)


def initial_quantile_particles(n=1024):
    u = (jnp.arange(n, dtype=jnp.float64) + 0.5) / n
    x = jax.scipy.special.ndtri(u)
    x = x - jnp.mean(x)
    return x / jnp.sqrt(jnp.mean(x * x))


def reference_holdout(key, params, n=8192):
    kt, kb = jax.random.split(key)
    jitter = jax.random.uniform(kt, (n,), dtype=jnp.float64)
    t = 0.01 + 0.98 * (jnp.arange(n, dtype=jnp.float64) + jitter) / n
    x, target_v = sample_reference_bridge_times(kb, t, A, None)
    pred = reference_velocity_net(params, t, x)
    mse = jnp.mean((pred - target_v) ** 2)
    zero_mse = jnp.mean(target_v * target_v)
    return {
        "fm_mse": float(mse),
        "zero_predictor_mse": float(zero_mse),
        "fm_mse_ratio_to_zero": float(mse / zero_mse),
    }


def empirical_vjp_check():
    """Check particle-aware implicit calibration derivative through Phi(x)."""
    x0 = jnp.linspace(-2.3, 2.1, 257)
    cot = jnp.array([0.31, -0.47])
    logw = jnp.zeros_like(x0)

    def objective(xi):
        x = x0 + xi * jnp.sin(1.3 * x0)
        lam = calibrate_empirical_implicit(logw, phi(x), TARGET)
        return cot @ lam

    g_impl = jax.grad(objective)(jnp.asarray(0.07))
    eps = 2e-5
    g_fd = (objective(0.07 + eps) - objective(0.07 - eps)) / (2 * eps)
    rel = jnp.abs(g_impl - g_fd) / jnp.maximum(jnp.abs(g_fd), 1e-12)
    return {"implicit": float(g_impl), "finite_difference": float(g_fd), "relative_error": float(rel)}


def oracle_debug(model):
    """Development-only diagnostics; never used as a pass criterion."""
    grid = make_grid(6.0, 1001)
    rows = []
    for t in np.linspace(0.1, 0.9, 5):
        ref, fib = mfsi_pipeline(None, grid, jnp.asarray(t), A)
        uhat = reference_velocity_net(model.reference_params, jnp.asarray(t), grid.x)
        dhat = learned_correction(model.potential_params, jnp.asarray(t), grid.x)
        rows.append({
            "t": float(t),
            "reference_velocity_l2": float(weighted_l2(uhat - ref.velocity, ref.q_ref, grid)),
            "correction_l2": float(weighted_l2(dhat - fib.correction, fib.q, grid)),
        })
    return rows



def empirical_oracle_component_check(n_particles=8192):
    """Example-A-only component check against the quadrature oracle.

    This intentionally uses the exact reference SI velocity so it isolates the
    finite-particle I-projection / fiber-calculus approximation from neural
    reference-velocity error.  It is reported only as an Experiment-A oracle
    diagnostic and is never used for training or checkpoint selection.
    """
    grid = make_grid(7.0, 2001)
    rows = []
    keys = jax.random.split(jax.random.PRNGKey(73191), len(HELDOUT_TIMES))
    for key, t in zip(keys, np.asarray(HELDOUT_TIMES)):
        tj = jnp.asarray(t)
        x, _ = sample_reference_bridge(key, tj, n_particles, A, None)
        u = reference_velocity(x, tj, A, None)
        emp = empirical_fiber_state(x, u, TARGET)
        _, exact = mfsi_pipeline(None, grid, tj, A, differentiation="implicit")
        h_exact = jnp.interp(x, grid.x, exact.forcing)
        h_l2 = jnp.sqrt(jnp.sum(emp.projected_weights * (emp.forcing - h_exact) ** 2))
        h_norm = jnp.sqrt(jnp.sum(emp.projected_weights * h_exact ** 2))
        h_rel = h_l2 / jnp.maximum(h_norm, 1e-12)
        lam_abs = jnp.linalg.norm(emp.lambda_ - exact.lambda_)
        ldot_abs = jnp.linalg.norm(emp.lambda_dot - exact.lambda_dot)
        rows.append({
            "t": float(t),
            "lambda_empirical": np.asarray(emp.lambda_).tolist(),
            "lambda_quadrature": np.asarray(exact.lambda_).tolist(),
            "lambda_abs_error": float(lam_abs),
            "lambda_dot_empirical": np.asarray(emp.lambda_dot).tolist(),
            "lambda_dot_quadrature": np.asarray(exact.lambda_dot).tolist(),
            "lambda_dot_abs_error": float(ldot_abs),
            "forcing_weighted_l2_error": float(h_l2),
            "forcing_relative_l2_error": float(h_rel),
            "calibration_residual": float(emp.calibration_residual),
            "ess_fraction": float(emp.ess_fraction),
        })
    return {
        "n_particles": int(n_particles),
        "rows": rows,
        "mean_lambda_abs_error": float(np.mean([r["lambda_abs_error"] for r in rows])),
        "max_lambda_abs_error": float(max(r["lambda_abs_error"] for r in rows)),
        "mean_lambda_dot_abs_error": float(np.mean([r["lambda_dot_abs_error"] for r in rows])),
        "max_lambda_dot_abs_error": float(max(r["lambda_dot_abs_error"] for r in rows)),
        "mean_forcing_weighted_l2_error": float(np.mean([r["forcing_weighted_l2_error"] for r in rows])),
        "max_forcing_weighted_l2_error": float(max(r["forcing_weighted_l2_error"] for r in rows)),
        "mean_forcing_relative_l2_error": float(np.mean([r["forcing_relative_l2_error"] for r in rows])),
        "max_forcing_relative_l2_error": float(max(r["forcing_relative_l2_error"] for r in rows)),
    }


def make_component_validation_plot(component_check, oracle_rows):
    """Paper-facing Example-A component figure; oracle diagnostics only."""
    rows = component_check["rows"]
    t = np.asarray([r["t"] for r in rows])
    le = np.asarray([r["lambda_empirical"] for r in rows])
    lq = np.asarray([r["lambda_quadrature"] for r in rows])
    de = np.asarray([r["lambda_dot_empirical"] for r in rows])
    dq = np.asarray([r["lambda_dot_quadrature"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for k in range(le.shape[1]):
        axes[0,0].plot(t, lq[:,k], linewidth=2, label=f"quadrature λ{k+1}")
        axes[0,0].plot(t, le[:,k], 'o', ms=4, label=f"empirical λ{k+1}")
    axes[0,0].set_title("Multiplier: empirical vs quadrature"); axes[0,0].set_xlabel("t")
    axes[0,0].legend(fontsize=7, ncol=2)
    for k in range(de.shape[1]):
        axes[0,1].plot(t, dq[:,k], linewidth=2, label=f"quadrature λdot{k+1}")
        axes[0,1].plot(t, de[:,k], 'o', ms=4, label=f"empirical λdot{k+1}")
    axes[0,1].set_title("Multiplier derivative"); axes[0,1].set_xlabel("t")
    axes[0,1].legend(fontsize=7, ncol=2)
    axes[1,0].semilogy(t, np.maximum([r["forcing_weighted_l2_error"] for r in rows], 1e-14), marker='o')
    axes[1,0].set_title("Forcing error, weighted L2"); axes[1,0].set_xlabel("t")
    od = {float(r["t"]): r for r in oracle_rows}
    ot = np.asarray(sorted(od))
    axes[1,1].plot(ot, [od[z]["reference_velocity_l2"] for z in ot], marker='o', label="reference velocity")
    axes[1,1].plot(ot, [od[z]["correction_l2"] for z in ot], marker='o', label="Deep-Ritz correction")
    axes[1,1].set_title("Learned-field oracle errors (debug only)"); axes[1,1].set_xlabel("t")
    axes[1,1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "example_a_component_validation.png", dpi=180); plt.close(fig)

def run(train=True, refine=False):
    key = jax.random.PRNGKey(2026)
    k_ref, k_ritz, k_refval, k_raw, k_safe = jax.random.split(key, 5)

    if refine and MODEL_PATH.exists():
        base = load_learned_model(MODEL_PATH)
        ref_params, ref_history = train_reference_flow_matching(
            k_ref, A, steps=500, batch_size=4096, validation_size=8192,
            lr_start=5e-4, lr_end=2e-5, initial_params=base.reference_params,
        )
        potential_params, ritz_history, polish = train_deep_ritz(
            k_ritz, ref_params, A, steps=300, n_times=8, particles_per_time=160,
            bank_pool_size=6, validation_times=10, validation_particles=256,
            lbfgs_maxiter=1, polish_times=8, polish_particles=160,
            lr_start=2e-4, lr_end=1e-5, initial_params=base.potential_params,
        )
        model = LearnedMFSIModel(ref_params, potential_params)
        save_learned_model(MODEL_PATH, model)
    elif train or not MODEL_PATH.exists():
        ref_params, ref_history = train_reference_flow_matching(
            k_ref, A, steps=1200, batch_size=4096, validation_size=8192
        )
        potential_params, ritz_history, polish = train_deep_ritz(
            k_ritz, ref_params, A, steps=1000, n_times=10, particles_per_time=224,
            bank_pool_size=12, validation_times=14, validation_particles=320,
            lbfgs_maxiter=3, polish_times=10, polish_particles=192,
        )
        model = LearnedMFSIModel(ref_params, potential_params)
        save_learned_model(MODEL_PATH, model)
    else:
        model = load_learned_model(MODEL_PATH)
        ref_history, ritz_history, polish = [], [], {"used": False, "loaded_model": True}

    reference_validation = reference_holdout(k_refval, model.reference_params)
    vjp = empirical_vjp_check()
    component_oracle = empirical_oracle_component_check()

    x0 = initial_quantile_particles()
    integrate_raw = jax.jit(lambda x: integrate_learned_flow(model, x, n_steps=120, safety=False))
    integrate_safe = jax.jit(lambda x: integrate_learned_flow(model, x, n_steps=120, safety=True))
    raw_times, raw_traj = integrate_raw(x0)
    safe_times, safe_traj = integrate_safe(x0)
    raw_traj.block_until_ready(); safe_traj.block_until_ready()

    raw_rows = heldout_learned_diagnostics(
        k_raw, model, A, HELDOUT_TIMES,
        raw_traj, raw_times,
        bank_particles=768, mmd_particles=256,
    )
    safe_rows = heldout_learned_diagnostics(
        k_safe, model, A, HELDOUT_TIMES,
        safe_traj, safe_times,
        bank_particles=768, mmd_particles=256,
    )

    def aggregate(rows):
        return {
            "max_calibration_residual": float(max(r["calibration_residual"] for r in rows)),
            "min_ess_fraction": float(min(r["ess_fraction"] for r in rows)),
            "min_covariance_rank": int(min(r["covariance_rank"] for r in rows)),
            "max_covariance_condition": float(max(r["covariance_condition"] for r in rows)),
            "median_weak_form_residual": float(np.median([r["weak_form_residual"] for r in rows])),
            "max_weak_form_residual": float(max(r["weak_form_residual"] for r in rows)),
            "median_projected_mmd": float(np.median([r["projected_mmd"] for r in rows])),
            "max_projected_mmd": float(max(r["projected_mmd"] for r in rows)),
            "max_generated_moment_error": float(max(r["generated_moment_error"] for r in rows)),
        }

    raw_summary = aggregate(raw_rows)
    safe_summary = aggregate(safe_rows)

    # Future-compatible criteria only: no analytic correction/path oracle below.
    criteria = {
        "reference_fm_beats_zero": reference_validation["fm_mse_ratio_to_zero"] < 0.90,
        "empirical_vjp_matches_fd": vjp["relative_error"] < 2e-5,
        "fresh_projection_calibrated": safe_summary["max_calibration_residual"] < 1e-8,
        "fresh_projection_has_overlap": safe_summary["min_ess_fraction"] > 0.10,
        "fresh_projection_full_rank": safe_summary["min_covariance_rank"] == 2,
        "fresh_projection_conditioned": safe_summary["max_covariance_condition"] < 1e4,
        "ritz_weak_form_median": safe_summary["median_weak_form_residual"] < 0.08,
        "ritz_weak_form_max": safe_summary["max_weak_form_residual"] < 0.18,
        "generated_matches_projected_samples": safe_summary["max_projected_mmd"] < 0.13,
        "safety_controls_population_moments": safe_summary["max_generated_moment_error"] < 1e-4,
        "safety_does_not_destroy_path": safe_summary["median_projected_mmd"] <= raw_summary["median_projected_mmd"] + 0.03,
    }

    oracle_rows = oracle_debug(model)
    make_component_validation_plot(component_oracle, oracle_rows)

    results = {
        "reference_validation": reference_validation,
        "particle_implicit_vjp": vjp,
        "example_a_empirical_vs_quadrature": component_oracle,
        "raw_learned_flow": {"summary": raw_summary, "heldout": raw_rows},
        "safe_learned_flow": {"summary": safe_summary, "heldout": safe_rows},
        "training": {
            "reference_history": ref_history,
            "ritz_history": ritz_history,
            "lbfgs_polish": polish,
            "reference_parameter_count": int(flatten_mlp(model.reference_params).size),
            "potential_parameter_count": int(flatten_mlp(model.potential_params).size),
        },
        "criteria": criteria,
        "all_passed": bool(all(criteria.values())),
        "oracle_debug_not_used_for_pass_fail": oracle_rows,
    }
    (OUT / "learned_validation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not results["all_passed"]:
        failed = [k for k, v in criteria.items() if not v]
        raise AssertionError(f"learned MFSI validation failed: {failed}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the learned MFSI toy pipeline.")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain the reference and Deep-Ritz networks from scratch.",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Continue training the packaged checkpoint using oracle-free holdout selection.",
    )
    args = parser.parse_args()
    if args.retrain and args.refine:
        parser.error("choose at most one of --retrain and --refine")
    run(train=args.retrain or not MODEL_PATH.exists(), refine=args.refine)
