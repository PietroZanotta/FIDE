#!/usr/bin/env python3
"""Post-hoc mechanism analysis for the frozen Experiment-D confirmatory run.

This program NEVER trains or selects a model.  It loads the registered
observable/Deep-Ritz checkpoints and recomputes additional diagnostics on the
same deterministic evaluation seeds used by the completed confirmatory run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import example_b as exb
import observable_design_toy as od

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "results" / "observable_design_toy" / "confirmatory" / "R3"
REPORT_PATH = ROOT / "OBSERVABLE_DESIGN_MECHANISM_ANALYSIS.md"
OBJECTIVES = ("info", "cv", "fiber", "random", "full_phi5")
LEARNED = ("info", "cv", "fiber")
R3_OBJECTIVES = ("info", "cv", "fiber", "random")
BASIS = ("x1", "x2", "x1^2", "x1*x2", "x2^2")


def _key(seed: int, stream: int) -> jax.Array:
    return jax.random.fold_in(jax.random.PRNGKey(seed), stream)


def _json_ready(x: Any) -> Any:
    return od.json_ready(x)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, allow_nan=False))


def _checkpoint_hashes(run_dir: Path) -> dict[str, str]:
    """Fingerprint all frozen analysis inputs so mutation is detectable."""
    paths = [run_dir / "design_standardization.npz", *sorted((run_dir / "checkpoints").glob("*.npz"))]
    return {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _load_standardization(run_dir: Path) -> od.Standardization:
    data = np.load(run_dir / "design_standardization.npz")
    return od.Standardization(jnp.asarray(data["center"]), jnp.asarray(data["whitening"]),
                              jnp.asarray(data["covariance_eigenvalues"]))


def _load_observable(run_dir: Path, objective: str, seed: int,
                     standardization: od.Standardization) -> od.ObservableModel:
    data = np.load(run_dir / "checkpoints" / f"observable_{objective}_modelseed_{seed}.npz")
    return od.ObservableModel(jnp.asarray(data["A"]), standardization)


def _load_potential(run_dir: Path, objective: str, seed: int):
    data = np.load(run_dir / "checkpoints" / f"ritz_{objective}_modelseed_{seed}.npz")
    return exb.unflatten(jnp.asarray(data["potential_params"]), exb.RITZ_HIDDEN, 1)


def _weighted_cov(values: jax.Array, weights: jax.Array) -> tuple[jax.Array, jax.Array]:
    mean = weights @ values
    centered = values - mean
    return mean, (centered.T * weights) @ centered


def _components(mu: jax.Array, projector: jax.Array) -> tuple[float, float, float]:
    constrained = projector @ mu
    null = mu - constrained
    ec = jnp.sum(constrained * constrained)
    en = jnp.sum(null * null)
    return float(jnp.sqrt(ec)), float(jnp.sqrt(en)), float(en / (ec + en + 1e-14))


def _trajectory_at(run: dict[str, jax.Array], t: float) -> jax.Array:
    idx = int(np.argmin(np.abs(np.asarray(run["times"]) - t)))
    return run["trajectory"][idx]


def compute_cell(
    *,
    objective: str,
    model_seed: int,
    evaluation_seed: int,
    model: od.ObservableModel,
    reference_params,
    potential_params,
    times: list[float],
    rollout_particles: int,
    target_particles: int,
    flow_steps: int,
) -> dict[str, Any]:
    """Frozen-model diagnostic extraction for one registered crossed cell."""
    eval_key = _key(evaluation_seed, 700)
    kr, kb = jax.random.split(eval_key)
    runs = od.rollout_methods(kr, model, reference_params, potential_params,
                              n_particles=rollout_particles, flow_steps=flow_steps)
    bank_keys = jax.random.split(kb, 2 * len(times))
    A = model.A
    P = A.T @ A
    rows = []
    for i, t_float in enumerate(times):
        t = jnp.asarray(t_float, dtype=jnp.float64)
        x, _ = exb.sample_bridge(bank_keys[2 * i], t, target_particles)
        u = exb.reference_velocity(reference_params, t, x)
        z = od.standardized_dictionary(x, model.standardization)
        ph = z @ A.T
        f = od.project_bank(A, model.standardization, x, u)
        uniform = jnp.ones(x.shape[0], dtype=x.dtype) / x.shape[0]
        mu_ref, cov_z_ref = _weighted_cov(z, uniform)
        mu_q, cov_z_q = _weighted_cov(z, f.projected_weights)
        _, cov_phi_ref = _weighted_cov(ph, uniform)
        _, cov_phi_q = _weighted_cov(ph, f.projected_weights)
        violation = jnp.linalg.norm(A @ mu_ref)
        capture = violation**2 / (jnp.sum(mu_ref * mu_ref) + 1e-14)
        w = f.projected_weights
        ref_c, ref_n, ref_nf = _components(mu_ref, P)
        q_c, q_n, q_nf = _components(mu_q, P)
        y_tan = _trajectory_at(runs["moment_tangent"], t_float)
        y_safe = _trajectory_at(runs["mfsi_learned_safe"], t_float)
        mu_tan = jnp.mean(od.standardized_dictionary(y_tan, model.standardization), axis=0)
        mu_safe = jnp.mean(od.standardized_dictionary(y_safe, model.standardization), axis=0)
        tan_c, tan_n, tan_nf = _components(mu_tan, P)
        safe_c, safe_n, safe_nf = _components(mu_safe, P)
        rows.append({
            "t": t_float,
            "reference_violation": float(violation),
            "capture_fraction": float(capture),
            "reference_mz": np.asarray(mu_ref).tolist(),
            "reference_delta_b": np.asarray(jnp.mean(od.raw_dictionary(x), axis=0)
                                               - model.standardization.center).tolist(),
            "projection_distortion": float(jnp.sum(w * jnp.log(jnp.maximum(w * x.shape[0], 1e-300)))),
            "lambda_norm": float(jnp.linalg.norm(f.lambda_)),
            "ess_fraction": float(f.ess_fraction),
            "max_weight": float(jnp.max(w)),
            "weight_entropy": float(-jnp.sum(w * jnp.log(jnp.maximum(w, 1e-300)))),
            "calibration_residual": float(f.calibration_residual),
            "covariance_condition": float(f.covariance_condition),
            "trace_cov_phi_reference": float(jnp.trace(cov_phi_ref)),
            "trace_cov_phi_projected": float(jnp.trace(cov_phi_q)),
            "phi_variance_reference": np.asarray(jnp.diag(cov_phi_ref)).tolist(),
            "phi_variance_projected": np.asarray(jnp.diag(cov_phi_q)).tolist(),
            "trace_cov_z_reference": float(jnp.trace(cov_z_ref)),
            "trace_cov_z_projected": float(jnp.trace(cov_z_q)),
            "mu_z_reference": np.asarray(mu_ref).tolist(),
            "mu_z_projected": np.asarray(mu_q).tolist(),
            "mu_z_tangent": np.asarray(mu_tan).tolist(),
            "mu_z_mfsi_safe": np.asarray(mu_safe).tolist(),
            "reference_constrained_norm": ref_c,
            "reference_null_norm": ref_n,
            "reference_null_fraction": ref_nf,
            "projected_constrained_norm": q_c,
            "projected_null_norm": q_n,
            "projected_null_fraction": q_nf,
            "tangent_constrained_norm": tan_c,
            "tangent_null_norm": tan_n,
            "tangent_null_fraction": tan_nf,
            "mfsi_safe_constrained_norm": safe_c,
            "mfsi_safe_null_norm": safe_n,
            "mfsi_safe_null_fraction": safe_nf,
        })
    return {"objective": objective, "model_seed": model_seed,
            "evaluation_seed": evaluation_seed, "per_time": rows}


def _null_basis(matrix: np.ndarray, null_dim: int = 2) -> np.ndarray:
    _, _, vh = np.linalg.svd(matrix, full_matrices=True)
    return vh[-null_dim:].T


def _canonical_columns(basis: np.ndarray) -> np.ndarray:
    out = basis.copy()
    for j in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, j])))
        if out[pivot, j] < 0:
            out[:, j] *= -1
    return out


def analyze_nullspaces(run_dir: Path, model_seeds: list[int],
                       standardization: od.Standardization) -> dict[str, Any]:
    output: dict[str, Any] = {"basis_names": BASIS, "objectives": {}}
    for objective in R3_OBJECTIVES:
        entries, pz_all, praw_all, nz_all, nraw_all = [], [], [], [], []
        for seed in model_seeds:
            model = _load_observable(run_dir, objective, seed, standardization)
            A = np.asarray(model.A)
            nz = _null_basis(A)
            pz = nz @ nz.T
            C = np.asarray(model.raw_coefficients)
            nraw = _null_basis(C)
            praw = nraw @ nraw.T
            entries.append({"model_seed": seed, "P_null_z": pz.tolist(), "N_z": nz.tolist(),
                            "P_null_raw": praw.tolist(), "N_raw": nraw.tolist(),
                            "C_raw": C.tolist()})
            pz_all.append(pz); praw_all.append(praw); nz_all.append(nz); nraw_all.append(nraw)
        pairwise = []
        for i in range(len(model_seeds)):
            for j in range(i + 1, len(model_seeds)):
                distance = np.linalg.norm(pz_all[i] - pz_all[j]) / np.sqrt(4.0)
                s = np.linalg.svd(nz_all[i].T @ nz_all[j], compute_uv=False)
                angles = np.arccos(np.clip(s, -1, 1))
                pairwise.append({"seed_i": model_seeds[i], "seed_j": model_seeds[j],
                                 "projection_distance": float(distance),
                                 "principal_angles_rad": angles.tolist(),
                                 "largest_angle_deg": float(np.max(angles) * 180 / np.pi)})
        mean_pz = np.mean(pz_all, axis=0)
        vals_z, vecs_z = np.linalg.eigh(mean_pz)
        order_z = np.argsort(vals_z)[::-1]
        consensus_z = _canonical_columns(vecs_z[:, order_z[:2]])
        mean_praw = np.mean(praw_all, axis=0)
        vals_raw, vecs_raw = np.linalg.eigh(mean_praw)
        order_raw = np.argsort(vals_raw)[::-1]
        consensus_raw = _canonical_columns(vecs_raw[:, order_raw[:2]])
        output["objectives"][objective] = {
            "seeds": entries,
            "pairwise": pairwise,
            "pairwise_distance_mean": float(np.mean([p["projection_distance"] for p in pairwise])),
            "pairwise_distance_min": float(np.min([p["projection_distance"] for p in pairwise])),
            "pairwise_distance_max": float(np.max([p["projection_distance"] for p in pairwise])),
            "largest_angle_deg_mean": float(np.mean([p["largest_angle_deg"] for p in pairwise])),
            "mean_projector_z_eigenvalues": vals_z[order_z].tolist(),
            "consensus_N_z": consensus_z.tolist(),
            "mean_projector_raw_eigenvalues": vals_raw[order_raw].tolist(),
            "consensus_N_raw": consensus_raw.tolist(),
        }
    # Paired cross-objective null-space comparisons parallel the main study.
    cross = {}
    for left, right in (("fiber", "info"), ("fiber", "cv"), ("info", "cv")):
        rows = []
        for seed in model_seeds:
            A = np.asarray(_load_observable(run_dir, left, seed, standardization).A)
            B = np.asarray(_load_observable(run_dir, right, seed, standardization).A)
            PA, PB = np.eye(5) - A.T @ A, np.eye(5) - B.T @ B
            NA, NB = _null_basis(A), _null_basis(B)
            angles = np.arccos(np.clip(np.linalg.svd(NA.T @ NB, compute_uv=False), -1, 1))
            rows.append({"seed": seed, "projection_distance": float(np.linalg.norm(PA - PB) / 2.0),
                         "principal_angles_rad": angles.tolist(),
                         "largest_angle_deg": float(np.max(angles) * 180 / np.pi)})
        cross[f"{left}_vs_{right}"] = rows
    output["paired_cross_objective"] = cross
    return output


def _trapz(values: list[float], times: list[float]) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(values, times))


def aggregate_seed_rows(cells: list[dict[str, Any]], confirmatory: dict[str, Any]) -> list[dict[str, Any]]:
    perf = {(r["objective"], int(r["model_seed"]), int(r["evaluation_seed"])): r
            for r in confirmatory["seed_level_records"]}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["objective"], int(cell["model_seed"])), []).append(cell)
    rows = []
    for (objective, seed), group in sorted(groups.items()):
        flattened = [r for c in group for r in c["per_time"]]
        eval_integrals = []
        for c in group:
            tt = [r["t"] for r in c["per_time"]]
            eval_integrals.append({
                "violation": _trapz([r["reference_violation"] for r in c["per_time"]], tt),
                "capture": _trapz([r["capture_fraction"] for r in c["per_time"]], tt),
                "null_fraction": _trapz([r["reference_null_fraction"] for r in c["per_time"]], tt),
            })
        prows = [perf[(objective, seed, int(c["evaluation_seed"]))] for c in group]
        row = {
            "objective": objective, "model_seed": seed,
            "mean_reference_violation": float(np.mean([r["reference_violation"] for r in flattened])),
            "max_reference_violation": float(np.max([r["reference_violation"] for r in flattened])),
            "integrated_reference_violation": float(np.mean([r["violation"] for r in eval_integrals])),
            "mean_capture_fraction": float(np.mean([r["capture_fraction"] for r in flattened])),
            "integrated_capture_fraction": float(np.mean([r["capture"] for r in eval_integrals])),
            "mean_reference_null_fraction": float(np.mean([r["reference_null_fraction"] for r in flattened])),
            "integrated_reference_null_fraction": float(np.mean([r["null_fraction"] for r in eval_integrals])),
            "mean_ess": float(np.mean([r["ess_fraction"] for r in flattened])),
            "min_ess": float(np.min([r["ess_fraction"] for r in flattened])),
            "mean_projection_distortion": float(np.mean([r["projection_distortion"] for r in flattened])),
            "mean_lambda_norm": float(np.mean([r["lambda_norm"] for r in flattened])),
            "mean_max_weight": float(np.mean([r["max_weight"] for r in flattened])),
            "mean_weight_entropy": float(np.mean([r["weight_entropy"] for r in flattened])),
            "mean_covariance_condition": float(np.mean([r["covariance_condition"] for r in flattened])),
            "mean_trace_cov_phi_reference": float(np.mean([r["trace_cov_phi_reference"] for r in flattened])),
            "mean_trace_cov_phi_projected": float(np.mean([r["trace_cov_phi_projected"] for r in flattened])),
        }
        for prefix in ("reference", "projected", "tangent", "mfsi_safe"):
            for component in ("constrained_norm", "null_norm", "null_fraction"):
                key = f"{prefix}_{component}"
                row[f"mean_{key}"] = float(np.mean([r[key] for r in flattened]))
        for metric in ("tangent_local_mmd", "velocity_gap", "tangent_rollout_mmd",
                       "mfsi_rollout_mmd", "angular_error", "max_moment_error"):
            row[metric] = float(np.mean([r[metric] for r in prows]))
        rows.append(row)
    return rows


def correlations(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictors = ("mean_reference_violation", "mean_capture_fraction", "mean_ess", "min_ess",
                  "mean_projection_distortion", "mean_lambda_norm", "mean_reference_null_fraction")
    outcomes = ("tangent_local_mmd", "velocity_gap", "tangent_rollout_mmd",
                "mfsi_rollout_mmd", "angular_error")
    output: dict[str, Any] = {"guardrail": "exploratory n=10 per objective; no inferential p-value claims",
                              "within_objective": {}, "learned_pooled": {}, "descriptive_ols": {}}
    for objective in LEARNED:
        subset = [r for r in seed_rows if r["objective"] == objective]
        output["within_objective"][objective] = {}
        for pred in predictors:
            output["within_objective"][objective][pred] = {}
            for outcome in outcomes:
                x = np.array([r[pred] for r in subset]); y = np.array([r[outcome] for r in subset])
                output["within_objective"][objective][pred][outcome] = {
                    "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                    "spearman_r": float(spearmanr(x, y).statistic),
                }
    learned = [r for r in seed_rows if r["objective"] in LEARNED]
    for pred in predictors:
        output["learned_pooled"][pred] = {}
        for outcome in outcomes:
            x = np.array([r[pred] for r in learned]); y = np.array([r[outcome] for r in learned])
            output["learned_pooled"][pred][outcome] = {
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "spearman_r": float(spearmanr(x, y).statistic),
            }
    # Descriptive adjustment: outcome ~ centered violation + CV + FIBER (INFO reference).
    violation = np.array([r["mean_reference_violation"] for r in learned])
    X = np.column_stack([np.ones(len(learned)), violation - violation.mean(),
                         [r["objective"] == "cv" for r in learned],
                         [r["objective"] == "fiber" for r in learned]]).astype(float)
    for outcome in outcomes + ("mean_ess",):
        y = np.array([r[outcome] for r in learned])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        output["descriptive_ols"][outcome] = {
            "columns": ["intercept_INFO", "centered_reference_violation", "CV_vs_INFO", "FIBER_vs_INFO"],
            "coefficients": beta.tolist(), "r_squared": float(1 - np.sum((y - X @ beta)**2) / np.sum((y-y.mean())**2)),
            "warning": "descriptive post-hoc adjustment; no causal or significance interpretation",
        }
    return output


def _mean_time(cells: list[dict[str, Any]], objective: str, metric: str, times: list[float]) -> np.ndarray:
    return np.array([np.mean([r[metric] for c in cells if c["objective"] == objective
                              for r in c["per_time"] if r["t"] == t]) for t in times])


def make_figures(run_dir: Path, cells: list[dict[str, Any]], seed_rows: list[dict[str, Any]],
                 nulls: dict[str, Any], times: list[float]) -> None:
    colors = {"info": "#4477aa", "cv": "#ee8844", "fiber": "#228833",
              "random": "#999999", "full_phi5": "#aa3377"}
    # 1. Constraint strength and capture.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for o in OBJECTIVES:
        axes[0].plot(times, _mean_time(cells, o, "reference_violation", times), marker="o", label=o.upper(), color=colors[o])
        axes[1].plot(times, _mean_time(cells, o, "capture_fraction", times), marker="o", label=o.upper(), color=colors[o])
    axes[0].set_ylabel(r"$\|A E_{\tilde Q_t}[z]\|_2$"); axes[0].set_xlabel("t")
    axes[1].set_ylabel("capture fraction"); axes[1].set_xlabel("t"); axes[1].set_ylim(-.03, 1.03)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(run_dir / "constraint_strength_vs_time.png", dpi=180); plt.close(fig)

    # 2. Projection geometry.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for o in OBJECTIVES:
        for ax, metric in zip(axes, ("ess_fraction", "projection_distortion", "lambda_norm")):
            ax.plot(times, _mean_time(cells, o, metric, times), marker="o", label=o.upper(), color=colors[o])
    axes[0].set_ylabel("ESS fraction"); axes[1].set_ylabel("KL projection distortion"); axes[2].set_ylabel(r"$\|\lambda\|_2$")
    for ax in axes: ax.set_xlabel("t")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(run_dir / "projection_geometry_vs_time.png", dpi=180); plt.close(fig)

    # 3. Null-space stability.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, o in zip(axes, LEARNED):
        pairs = nulls["objectives"][o]["pairwise"]
        ax.scatter([p["projection_distance"] for p in pairs], [p["largest_angle_deg"] for p in pairs], alpha=.7, color=colors[o])
        eig = nulls["objectives"][o]["mean_projector_z_eigenvalues"]
        ax.set_title(f"{o.upper()}\nmean-P eig: " + ", ".join(f"{x:.2f}" for x in eig))
        ax.set_xlabel("projection distance"); ax.set_ylabel("largest angle (deg)")
    fig.tight_layout(); fig.savefig(run_dir / "nullspace_stability.png", dpi=180); plt.close(fig)

    # 4. Consensus raw coefficients.
    mats = [np.asarray(nulls["objectives"][o]["consensus_N_raw"]).T for o in LEARNED]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    vmax = max(np.max(np.abs(m)) for m in mats)
    for ax, o, mat in zip(axes, LEARNED, mats):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(5), BASIS, rotation=35, ha="right"); ax.set_yticks([0,1],["consensus 1","consensus 2"]); ax.set_title(o.upper())
    fig.colorbar(im, ax=axes, shrink=.75); fig.subplots_adjust(bottom=.25, wspace=.35)
    fig.savefig(run_dir / "consensus_nullspace_coefficients.png", dpi=180); plt.close(fig)

    # 5. Reference null fraction.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for o in OBJECTIVES:
        ax.plot(times, _mean_time(cells, o, "reference_null_fraction", times), marker="o", label=o.upper(), color=colors[o])
    ax.set_xlabel("t"); ax.set_ylabel("reference low-order null fraction"); ax.set_ylim(-.03,1.03); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "reference_null_fraction_vs_time.png", dpi=180); plt.close(fig)

    # 6. Null norm and trajectories in each objective's consensus standardized coordinates.
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    path_keys = (("mu_z_reference", "reference", "-"), ("mu_z_projected", "projected", "--"),
                 ("mu_z_tangent", "tangent", ":"), ("mu_z_mfsi_safe", "MFSI safe", "-."))
    norm_keys = (("reference_null_norm", "reference", "-"), ("projected_null_norm", "projected", "--"),
                 ("tangent_null_norm", "tangent", ":"), ("mfsi_safe_null_norm", "MFSI safe", "-."))
    for row_i, o in enumerate(LEARNED):
        N = np.asarray(nulls["objectives"][o]["consensus_N_z"])
        for key, label, style in norm_keys:
            axes[row_i, 0].plot(times, _mean_time(cells, o, key, times), style,
                                marker="o", ms=3, label=label)
        for key, label, style in path_keys:
            coords = []
            for t in times:
                mus = np.array([r[key] for c in cells if c["objective"] == o for r in c["per_time"] if r["t"] == t])
                coords.append(mus.mean(axis=0) @ N)
            coords = np.asarray(coords)
            for k in range(2): axes[row_i,k + 1].plot(times, coords[:,k], style, marker="o", ms=3, label=label)
        axes[row_i,0].set_ylabel(o.upper())
        axes[row_i,0].set_title(r"$\|P_{null}\mu_z\|_2$")
        axes[row_i,1].set_title("consensus null coordinate 1"); axes[row_i,2].set_title("consensus null coordinate 2")
    for ax in axes[-1]: ax.set_xlabel("t")
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(run_dir / "null_moment_trajectories.png", dpi=180); plt.close(fig)

    # 7. Mechanism scatterplots.
    learned_rows = [r for r in seed_rows if r["objective"] in LEARNED]
    xvars = (("mean_reference_violation", "reference violation"), ("mean_ess", "mean ESS"),
             ("mean_reference_null_fraction", "reference null fraction"))
    yvars = (("tangent_local_mmd", "local tangent MMD"), ("velocity_gap", "velocity gap"),
             ("tangent_rollout_mmd", "tangent rollout MMD"), ("mfsi_rollout_mmd", "safe-MFSI MMD"),
             ("angular_error", "hidden angular error"))
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for i,(xkey,xlab) in enumerate(xvars):
        for j,(ykey,ylab) in enumerate(yvars):
            ax=axes[i,j]
            for o in LEARNED:
                rows=[r for r in learned_rows if r["objective"]==o]
                ax.scatter([r[xkey] for r in rows],[r[ykey] for r in rows],label=o.upper(),color=colors[o],alpha=.8)
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(run_dir / "mechanism_scatterplots.png", dpi=180); plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_report(run_dir: Path, summary: dict[str, Any], nulls: dict[str, Any],
                 endpoint: dict[str, Any], correlations_data: dict[str, Any]) -> None:
    learned = summary["objective_means"]
    f, i, c = learned["fiber"], learned["info"], learned["cv"]
    ols = correlations_data["descriptive_ols"]
    fiber_weaker = f["mean_reference_violation"] < min(i["mean_reference_violation"], c["mean_reference_violation"])
    fiber_captures_less = f["mean_capture_fraction"] < min(i["mean_capture_fraction"], c["mean_capture_fraction"])
    fiber_more_null = f["mean_reference_null_fraction"] > max(i["mean_reference_null_fraction"], c["mean_reference_null_fraction"])
    fiber_higher_ess = f["mean_ess"] > max(i["mean_ess"], c["mean_ess"])
    adjusted_ess_positive = ols["mean_ess"]["coefficients"][3] > 0
    adjusted_mmd_favorable = ols["mfsi_rollout_mmd"]["coefficients"][3] < 0
    if fiber_weaker and fiber_captures_less:
        strength_sentence = ("FIBER selects lower-strength constraints on these reference banks and "
                             "captures less of the bridge's standardized low-order mean departure.")
    else:
        strength_sentence = ("FIBER is not uniformly the lowest-strength constraint by both violation "
                             "and capture, so an 'easy constraints only' account is not supported by those summaries alone.")
    if fiber_higher_ess and adjusted_ess_positive and fiber_weaker:
        projection_sentence = ("The descriptive evidence favors both smaller raw violation and residual "
                               "objective-associated projection ease: FIBER has higher raw ESS and a positive "
                               "FIBER ESS coefficient after linear adjustment for violation.")
    elif fiber_higher_ess and fiber_weaker:
        projection_sentence = ("The descriptive evidence primarily supports smaller raw violation; the "
                               "violation-adjusted model does not show an additional positive FIBER ESS coefficient.")
    elif fiber_higher_ess:
        projection_sentence = ("FIBER's higher ESS is not explained by smaller mean violation, pointing "
                               "descriptively toward projection geometry or conditioning.")
    else:
        projection_sentence = "FIBER does not have the highest mean ESS in these recomputed diagnostics."
    null_sentence = ("FIBER allocates the largest mean share of reference low-order motion to its unresolved plane."
                     if fiber_more_null else
                     "FIBER does not allocate the largest mean share of reference low-order motion to its unresolved plane.")
    ess_adjusted_sentence = ("Its ESS advantage remains descriptively visible after the linear adjustment."
                             if adjusted_ess_positive else
                             "Its ESS advantage does not remain descriptively visible after the linear adjustment.")
    adjusted_sentence = ("The favorable FIBER safe-MFSI association remains in the descriptive linear adjustment."
                         if adjusted_mmd_favorable else
                         "The favorable FIBER safe-MFSI association does not remain in the descriptive linear adjustment.")
    interior_times = ("0.25", "0.5", "0.75")
    interior_null = {
        o: float(np.mean([summary["objective_time_means"][o][t]["reference_null_fraction"]
                          for t in interior_times])) for o in LEARNED
    }
    fiber_nulls = nulls["objectives"]["fiber"]
    lines = [
        "# Observable-design mechanism analysis",
        "",
        "This is post-hoc mechanism analysis of the frozen Experiment-D confirmatory checkpoints. No observable or downstream model was retrained, no registered metric was changed, and these diagnostics were not used for selection.",
        "",
        "## Bottom line",
        "",
        f"{strength_sentence} Mean standardized reference-fiber violation is `{f['mean_reference_violation']:.4f}`, versus `{i['mean_reference_violation']:.4f}` for INFO and `{c['mean_reference_violation']:.4f}` for CV. Mean capture fraction is `{f['mean_capture_fraction']:.3f}`, versus `{i['mean_capture_fraction']:.3f}` and `{c['mean_capture_fraction']:.3f}`.",
        "",
        f"The learned FIBER observables are not vacuous. Their mean reference Phi variance trace is `{f['mean_trace_cov_phi_reference']:.3f}` (INFO `{i['mean_trace_cov_phi_reference']:.3f}`, CV `{c['mean_trace_cov_phi_reference']:.3f}`), endpoint Phi-space MMD is `{endpoint['fiber']['phi_space_mmd']:.3f}`, and representative endpoint AUROC is `{endpoint['fiber']['classifier_auroc']:.3f}`. FIBER therefore retains nontrivial sample-level information.",
        "",
        f"FIBER's mean reference-motion null fraction is `{f['mean_reference_null_fraction']:.3f}`, compared with INFO `{i['mean_reference_null_fraction']:.3f}` and CV `{c['mean_reference_null_fraction']:.3f}`. {null_sentence} This is a descriptive mechanism diagnostic, not causal proof.",
        "",
        "## 1. Constraint strength and projection geometry",
        "",
        "| Objective | Mean / mean seedwise max / integrated violation | Mean / integrated capture | Mean / min ESS | Mean KL | Mean lambda | Mean max weight | Mean entropy | Ref / projected Phi variance trace |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for o in OBJECTIVES:
        x=learned[o]
        lines.append(f"| {o.upper()} | {x['mean_reference_violation']:.4f} / {x['max_reference_violation']:.4f} / {x['integrated_reference_violation']:.4f} | {x['mean_capture_fraction']:.3f} / {x['integrated_capture_fraction']:.3f} | {x['mean_ess']:.3f} / {x['min_ess']:.3f} | {x['mean_projection_distortion']:.4f} | {x['mean_lambda_norm']:.4f} | {x['mean_max_weight']:.2e} | {x['mean_weight_entropy']:.3f} | {x['mean_trace_cov_phi_reference']:.3f} / {x['mean_trace_cov_phi_projected']:.3f} |")
    lines += [
        "",
        projection_sentence + " There is limited cross-objective overlap in violation, so the adjustment is partly extrapolative and does not establish conditioning at exactly matched violation.",
        "",
        "## 2. Null-space stability and physical directions",
        "",
    ]
    for o in LEARNED:
        n=nulls["objectives"][o]
        lines += [f"### {o.upper()}", "",
                  f"Within-objective null-projector distance: mean `{n['pairwise_distance_mean']:.3f}`, range `{n['pairwise_distance_min']:.3f}`-`{n['pairwise_distance_max']:.3f}`. Mean largest principal angle: `{n['largest_angle_deg_mean']:.1f}` degrees. Mean-projector eigenvalues: `{', '.join(f'{v:.3f}' for v in n['mean_projector_z_eigenvalues'])}`.", "",
                  "Consensus raw null directions (columns are a deterministic eigensystem representation of the consensus subspace; individual signs/basis rotations have no invariant meaning):", "",
                  "```text"]
        raw=np.asarray(n["consensus_N_raw"])
        for k in range(2): lines.append("  " + ", ".join(f"{name}={raw[j,k]:+.3f}" for j,name in enumerate(BASIS)))
        lines += ["```", ""]
    lines += [
        "The primary objects are the projectors and principal angles, not individual printed vectors. FIBER's mean-projector spectrum and pairwise distances quantify whether the unresolved plane is reproducible; see `nullspaces.json` and `nullspace_stability.png`.",
        "",
        f"FIBER's unresolved plane is only partially reproducible: its leading mean-projector eigenvalue is `{fiber_nulls['mean_projector_z_eigenvalues'][0]:.3f}`, but the second is `{fiber_nulls['mean_projector_z_eigenvalues'][1]:.3f}`; mean pairwise distance is `{fiber_nulls['pairwise_distance_mean']:.3f}` and mean largest angle is `{fiber_nulls['largest_angle_deg_mean']:.1f}` degrees. Thus one unresolved direction is moderately stable, while the full two-plane is not tightly seed-stable and is not clearly more reproducible than CV.",
        "",
        "The leading FIBER consensus raw direction is mostly a positive quadratic-scale combination (`x1^2` and `x2^2`). The second mixes `x2^2`, `-x1^2`, and `-x2`; because the second consensus eigenvalue is modest, that coordinate should be treated as exploratory rather than a seed-invariant physical law.",
        "",
        "## 3. Motion along unresolved directions",
        "",
        f"The reference bridge places `{f['mean_reference_null_fraction']:.1%}` of its standardized low-order mean energy in FIBER's null space on average. The corresponding fractions are `{i['mean_reference_null_fraction']:.1%}` for INFO and `{c['mean_reference_null_fraction']:.1%}` for CV. The projected, tangent, and safe-MFSI trajectories in consensus-null coordinates are shown in `null_moment_trajectories.png`.",
        "",
        f"At the three interior times, where the reference departure is appreciable, the mean null fractions are FIBER `{interior_null['fiber']:.1%}`, INFO `{interior_null['info']:.1%}`, and CV `{interior_null['cv']:.1%}`. Endpoint fractions are less interpretable because the total standardized mean departure is close to zero there.",
        "",
        "Mean standardized constrained / null norm along each path:",
        "",
        "| Objective | Reference | I-projected | Tangent | Safe MFSI |",
        "|---|---:|---:|---:|---:|",
    ]
    for o in OBJECTIVES:
        x = learned[o]
        entries = [f"{x[f'mean_{prefix}_constrained_norm']:.3f} / {x[f'mean_{prefix}_null_norm']:.3f}"
                   for prefix in ("reference", "projected", "tangent", "mfsi_safe")]
        lines.append(f"| {o.upper()} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "FIBER's R=3 advantage over full-Phi5 is consistent with allowing these low-order combinations to move while full-Phi5 has no low-order null space. This is supportive but insufficient to label full-Phi5 causally 'overconstrained'; architecture/optimization at different output dimension remains a confound.",
        "",
        "## 4. Relationship to law-level performance",
        "",
        f"Across seed means, FIBER retains the confirmatory law advantage (safe-MFSI MMD `{f['mfsi_rollout_mmd']:.4f}` versus INFO `{i['mfsi_rollout_mmd']:.4f}` and CV `{c['mfsi_rollout_mmd']:.4f}`). In the descriptive learned-seed regression `safe MMD ~ violation + objective`, the FIBER-vs-INFO coefficient is `{ols['mfsi_rollout_mmd']['coefficients'][3]:+.4f}`. This is exploratory adjustment, not causal identification or a significance test.",
        "",
        f"For mean ESS under the same descriptive adjustment, the FIBER-vs-INFO coefficient is `{ols['mean_ess']['coefficients'][3]:+.4f}`. {ess_adjusted_sentence} Ten seeds per objective are too few for a strong adjusted claim.",
        "",
        "All Pearson/Spearman correlations, including within-objective n=10 values, are in `mechanism_correlations.json`. They are exploratory and no p-values are interpreted.",
        "",
        "## 5. Endpoint informativeness",
        "",
        "| Objective | AUROC | Phi-space MMD | Expectation gap | Calibrated max gap | Angular gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for o in LEARNED:
        e=endpoint[o]
        lines.append(f"| {o.upper()} | {e['classifier_auroc']:.3f} | {e['phi_space_mmd']:.3f} | {e['expectation_gap']:.4f} | {e['calibrated_max_gap']:.2e} | {e['angular_gap']:.3f} |")
    lines += [
        "",
        "INFO is not relabeled as more identifying: the frozen representative AUROCs are essentially equal. FIBER remains sample-level discriminative despite matched measured expectations.",
        "",
        "## Guarded conclusion",
        "",
        f"{strength_sentence} Those constraints are nevertheless nonzero-variance and sample-level informative. {null_sentence} {projection_sentence} {adjusted_sentence} These observations support only a mechanism hypothesis; they do not establish causality or rule out objective-specific architecture/optimization effects.",
        "",
        "This post-hoc analysis does not alter the registered confirmatory conclusion and does not prove causality.",
    ]
    REPORT_PATH.write_text("\n".join(lines)+"\n")
    (run_dir / REPORT_PATH.name).write_text("\n".join(lines)+"\n")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--force", action="store_true", help="recompute frozen diagnostic cells")
    args=p.parse_args()
    run_dir=args.run_dir.resolve(); confirmatory=json.loads((run_dir/"results.json").read_text())
    hashes_before = _checkpoint_hashes(run_dir)
    config=json.loads((run_dir/"resolved_config.json").read_text()); phase=config["phases"]["confirmatory"]
    model_seeds=list(map(int,confirmatory["seed_manifest"]["model_seeds"])); eval_seeds=list(map(int,confirmatory["seed_manifest"]["evaluation_seeds"]))
    times=list(map(float,config["evaluation_times"])); standardization=_load_standardization(run_dir)
    reference=exb.load_model()[0]
    cache_dir=run_dir/"mechanism_cells"; cache_dir.mkdir(exist_ok=True)
    cells=[]
    print(f"[mechanism] frozen analysis: {len(OBJECTIVES)} objectives x {len(model_seeds)} models x {len(eval_seeds)} banks",flush=True)
    for mi,seed in enumerate(model_seeds):
        print(f"[mechanism] model {seed} ({mi+1}/{len(model_seeds)})",flush=True)
        for objective in OBJECTIVES:
            model=_load_observable(run_dir,objective,seed,standardization); potential=_load_potential(run_dir,objective,seed)
            for evaluation_seed in eval_seeds:
                path=cache_dir/f"{objective}_model_{seed}_eval_{evaluation_seed}.json"
                if path.exists() and not args.force:
                    cell=json.loads(path.read_text())
                else:
                    print(f"[mechanism]   {objective} eval {evaluation_seed}",flush=True)
                    cell=compute_cell(objective=objective,model_seed=seed,evaluation_seed=evaluation_seed,
                                      model=model,reference_params=reference,potential_params=potential,times=times,
                                      rollout_particles=int(phase["rollout_particles"]),target_particles=int(phase["target_particles"]),
                                      flow_steps=int(phase["flow_steps"]))
                    _write_json(path,cell)
                cells.append(cell)
        jax.clear_caches()
    nulls=analyze_nullspaces(run_dir,model_seeds,standardization)
    seed_rows=aggregate_seed_rows(cells,confirmatory)
    corr=correlations(seed_rows)
    objective_means={o:{k:float(np.mean([r[k] for r in seed_rows if r["objective"]==o]))
                        for k in seed_rows[0] if k not in {"objective","model_seed"}} for o in OBJECTIVES}
    time_metrics = ("reference_violation", "capture_fraction", "reference_null_fraction",
                    "ess_fraction", "projection_distortion", "lambda_norm", "max_weight",
                    "weight_entropy", "phi_variance_reference", "phi_variance_projected")
    objective_time_means = {}
    for objective in OBJECTIVES:
        objective_time_means[objective] = {}
        for t in times:
            matching = [row for cell in cells if cell["objective"] == objective
                        for row in cell["per_time"] if row["t"] == t]
            objective_time_means[objective][f"{t:g}"] = {
                metric: (np.mean(np.asarray([row[metric] for row in matching]), axis=0).tolist()
                         if isinstance(matching[0][metric], list)
                         else float(np.mean([row[metric] for row in matching])))
                for metric in time_metrics
            }
    endpoint={o:{"classifier_auroc":confirmatory["objectives"][o]["endpoint_classifier"]["auroc"],
                 "phi_space_mmd":confirmatory["objectives"][o]["endpoint"]["phi_space_mmd"],
                 "expectation_gap":confirmatory["objectives"][o]["endpoint"]["expectation_gap_norm"],
                 "calibrated_max_gap":confirmatory["objectives"][o]["endpoint"]["calibrated_max_abs_gap"],
                 "angular_gap":confirmatory["objectives"][o]["endpoint"]["hidden_angular_gap_norm"]} for o in LEARNED}
    summary={"status":"post_hoc_frozen_models_only","model_seeds":model_seeds,"evaluation_seeds":eval_seeds,
             "times":times,"objective_means":objective_means,
             "objective_time_means":objective_time_means,"endpoint_frozen_diagnostics":endpoint}
    flat=[]
    for cell in cells:
        for row in cell["per_time"]:
            flat.append({"objective":cell["objective"],"model_seed":cell["model_seed"],"evaluation_seed":cell["evaluation_seed"],
                         **{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in row.items()}})
    _write_csv(run_dir/"mechanism_per_time.csv",flat); _write_csv(run_dir/"mechanism_per_seed.csv",seed_rows)
    _write_json(run_dir/"mechanism_summary.json",summary); _write_json(run_dir/"nullspaces.json",nulls)
    _write_json(run_dir/"mechanism_correlations.json",corr)
    make_figures(run_dir,cells,seed_rows,nulls,times)
    write_report(run_dir,summary,nulls,endpoint,corr)
    hashes_after = _checkpoint_hashes(run_dir)
    if hashes_after != hashes_before:
        raise RuntimeError("Frozen checkpoint or standardization input changed during post-hoc analysis")
    _write_json(run_dir/"mechanism_frozen_input_hashes.json", hashes_after)
    print(json.dumps({"report":str(REPORT_PATH),"cells":len(cells),"per_time_rows":len(flat)},indent=2),flush=True)


if __name__=="__main__":
    main()
