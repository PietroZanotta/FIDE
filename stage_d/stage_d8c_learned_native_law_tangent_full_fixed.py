#!/usr/bin/env python3
"""
Stage D.8c: learned-native Law vs Tangent-TC vs Full-TC comparison under D5.

Purpose
-------
D8 established a finite-resource law-feasible set under the endpoint-trained D5
flow-matching reference, then selected the minimum weighted-Poisson-action design.
This script makes the comparison scientifically native to that learned reference:

  D5-Law      = minimum finite-resource law risk among D8 law-feasible survivors
  D5-Tangent  = minimum finite-resource tangent action among the same survivors
  D5-Full     = minimum finite-resource full weighted-Poisson action among survivors

All three use the same:
  * endpoint-trained D5 particle reference and learned velocity,
  * finite/noisy measurement model,
  * endpoint-anchored GLS moment reconstruction,
  * common physical/D5-particle feasibility polytope,
  * hard empirical I-projection,
  * common-random-number selection bank,
  * disjoint validation bank.

For the reference D8 run, the default survivor set is the seven finite-law-feasible
angle pairs already found by D8.  This avoids re-running the 521-candidate oracle.
A D8 selection/full JSON can alternatively be supplied with --selection-json.

The tangent action is evaluated on the SAME finite reconstructed target curve and
SAME D5 projected particle law as the full-action calculation:

    T = r^T G^{-1} r,
    r = E_q[J Phi u] - c_dot,
    G = E_q[grad Phi grad Phi^T].

The authoritative full realization action remains the D7 weighted-Poisson action.
No old analytic interpolant is used.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Loading / helpers
# -----------------------------------------------------------------------------


def load_module(path: Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jsonify(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonify(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, Mapping):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, jax.Array):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    return x


def mean_se(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    if x.size == 1:
        return {"mean": float(x[0]), "se": float("nan"), "n": 1}
    return {
        "mean": float(np.mean(x)),
        "se": float(np.std(x, ddof=1) / math.sqrt(x.size)),
        "n": int(x.size),
    }


def paired_difference(rows_a, rows_b, key: str) -> Dict[str, float]:
    a = np.asarray([float(r.get(key, np.nan)) for r in rows_a], dtype=np.float64)
    b = np.asarray([float(r.get(key, np.nan)) for r in rows_b], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    s = mean_se(d)
    return {
        "mean_difference_a_minus_b": s["mean"],
        "se_difference": s["se"],
        "n": s["n"],
        "a_better_fraction": float(np.mean(d < 0.0)) if d.size else float("nan"),
    }


def paired_reduction(rows_num, rows_den, key: str) -> Dict[str, float]:
    num = np.asarray([float(r.get(key, np.nan)) for r in rows_num], dtype=np.float64)
    den = np.asarray([float(r.get(key, np.nan)) for r in rows_den], dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-14)
    num = num[mask]
    den = den[mask]
    if num.size == 0:
        return {
            "ratio_of_means_reduction": float("nan"),
            "mean_paired_reduction": float("nan"),
            "se_paired_reduction": float("nan"),
            "n": 0,
        }
    p = 1.0 - num / den
    s = mean_se(p)
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(num) / np.mean(den)),
        "mean_paired_reduction": s["mean"],
        "se_paired_reduction": s["se"],
        "n": s["n"],
    }


def canonical_key(eta: np.ndarray) -> Tuple[float, float]:
    eta = np.mod(np.asarray(eta, dtype=np.float64), np.pi)
    eta = np.sort(eta)
    return tuple(np.round(eta, 12))


def write_json(path: Path, payload: Mapping[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True, allow_nan=True) + "\n")


def write_csv(path: Path, rows: List[Mapping[str, Any]]):
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen and np.isscalar(r[k]):
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# -----------------------------------------------------------------------------
# D8 survivor set
# -----------------------------------------------------------------------------


def default_reference_survivors() -> List[Dict[str, Any]]:
    # Exact grid indices from angle_n=37 reference D8 run.
    # These are the seven candidates that survived both D8 law screens.
    ij = [(4, 13), (4, 14), (5, 13), (5, 14), (6, 13), (6, 14), (6, 15)]
    out = []
    for i, j in ij:
        eta = np.array([i * np.pi / 37.0, j * np.pi / 37.0], dtype=np.float64)
        out.append({
            "eta": eta,
            "theta_deg": np.degrees(eta).tolist(),
            "source": "reference_d8_seven_survivors",
            "grid_indices": [i, j],
        })
    return out


def candidates_from_selection_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())

    def get_rows(obj):
        if isinstance(obj, dict):
            for k in ("finite_risk_feasible_rows", "finite_law_feasible_rows"):
                if isinstance(obj.get(k), list) and obj[k]:
                    return obj[k]
            for v in obj.values():
                r = get_rows(v)
                if r:
                    return r
        return None

    rows = get_rows(payload)
    if not rows:
        raise ValueError(
            f"Could not find finite_risk_feasible_rows in {path}. "
            "Use the D8 selection/full JSON, not validation-only JSON."
        )
    out = []
    for r in rows:
        deg = r.get("theta_deg")
        if deg is None and "eta" in r:
            eta = np.asarray(r["eta"], dtype=np.float64)
        elif deg is not None:
            eta = np.radians(np.asarray(deg, dtype=np.float64))
        else:
            continue
        out.append({
            "eta": eta,
            "theta_deg": np.degrees(eta).tolist(),
            "source": f"selection_json:{Path(path).name}",
            "original_row": r,
        })
    if not out:
        raise ValueError(f"No usable candidate angles in {path}")
    return dedupe_candidates(out)


def dedupe_candidates(cands: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in cands:
        eta = np.asarray(c["eta"], dtype=np.float64)
        key = canonical_key(eta)
        if key in seen:
            continue
        seen.add(key)
        cc = dict(c)
        cc["eta"] = np.asarray(key, dtype=np.float64)
        cc["theta_deg"] = np.degrees(cc["eta"]).tolist()
        out.append(cc)
    return out


# -----------------------------------------------------------------------------
# Exact finite-curve reconstruction used for tangent evaluation
# -----------------------------------------------------------------------------


def reconstruct_finite_curve(
    model, evaluator, d3, c2, measurement_cov, eta, shared, acq_idx,
    cfg, joint_constraints,
):
    t_acq, y_acq, V_acq, exact_acq = c2.design_measurements(
        model=model,
        infer=measurement_cov,
        eta=eta,
        shared=shared,
        acq_idx=acq_idx,
        n=int(cfg.finite_n),
        obs_noise_std=float(cfg.obs_noise_std),
        variance_floor=float(cfg.variance_floor),
    )
    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_j, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_j, eta_j), dtype=np.float64)
    curve = d3.fit_quadratic_bridge_joint_feasible(
        t_obs=t_acq,
        y_obs=y_acq,
        V_obs=V_acq,
        c0=c0,
        c1=c1,
        t_eval=times,
        A=joint_constraints["A"],
        b=joint_constraints["b"],
        feasible_beta=joint_constraints["feasible_beta"],
        ridge_rel=float(cfg.quadratic_ridge_rel),
        variance_floor=float(cfg.variance_floor),
    )
    return curve


# -----------------------------------------------------------------------------
# Robust 2-D empirical tilt fallback for tangent calculation
# -----------------------------------------------------------------------------


def normalized_exp(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def tilt_quantities(phi, base_w, target, lam):
    base_w = np.asarray(base_w, dtype=np.float64)
    base_w = base_w / np.sum(base_w)
    log_base = np.log(np.maximum(base_w, 1e-300))
    w = normalized_exp(log_base + np.asarray(phi) @ np.asarray(lam))
    moment = np.sum(w[:, None] * phi, axis=0)
    centered = phi - moment[None, :]
    C = centered.T @ (w[:, None] * centered)
    resid = float(np.linalg.norm(moment - target))
    return w, moment, C, resid


def lbfgs_empirical_tilt(phi, base_w, target, lam0, cfg):
    phi = np.asarray(phi, dtype=np.float64)
    base_w = np.asarray(base_w, dtype=np.float64)
    base_w = base_w / np.sum(base_w)
    target = np.asarray(target, dtype=np.float64)
    log_base = np.log(np.maximum(base_w, 1e-300))

    clip0 = float(getattr(cfg, "lambda_clip", 1000.0))
    mult = float(getattr(cfg, "calibration_retry_clip_multiplier", 2.0))
    retries = int(getattr(cfg, "calibration_max_retries", 2))
    maxiter = int(getattr(cfg, "calibration_lbfgs_maxiter", 800))
    accept = float(getattr(cfg, "solver_accept_tol", 2e-6))
    lam_start = np.asarray(lam0, dtype=np.float64).copy()
    best = None

    for retry in range(retries + 1):
        clip = clip0 * (mult ** retry)

        def fun(lam):
            return float(logsumexp(log_base + phi @ lam) - np.dot(lam, target))

        def jac(lam):
            w = normalized_exp(log_base + phi @ lam)
            return np.sum(w[:, None] * phi, axis=0) - target

        x0 = np.clip(lam_start, -clip, clip)
        sol = minimize(
            fun, x0, jac=jac, method="L-BFGS-B",
            bounds=[(-clip, clip)] * phi.shape[1],
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-11, "maxls": 50},
        )
        w, moment, C, resid = tilt_quantities(phi, base_w, target, sol.x)
        cand = (resid, np.asarray(sol.x, dtype=np.float64), w, moment, C, sol, retry, clip)
        if best is None or resid < best[0]:
            best = cand
        if np.isfinite(resid) and resid <= accept:
            break
        lam_start = np.asarray(sol.x, dtype=np.float64)

    resid, lam, w, moment, C, sol, retry, clip = best
    return {
        "lambda": lam,
        "weights": w,
        "moment": moment,
        "C": C,
        "residual": float(resid),
        "solver_success": bool(sol.success),
        "retry_count": int(retry),
        "clip": float(clip),
    }


def tangent_curve_action(model, evaluator, d2, d7, eta, target_curve, target_cdot, cfg):
    """Finite-resource tangent action on the D5 learned particle reference.

    This deliberately uses the *D7-v2 robust projected state*, not the older
    D2 convenience solver.  Therefore the tangent objective is evaluated from
    exactly the same endpoint-trained D5 particles, learned velocity, robust
    empirical I-projection, ESS convention, and calibration tolerances as D8.
    No Poisson solve is performed here.
    """
    # D7Evaluator names the D5 rollout arrays ``nodes`` and ``u_nodes``.
    # (The older D2 evaluator used ``learned_nodes`` / ``learned_u_nodes``.)
    nodes = np.asarray(evaluator.nodes, dtype=np.float64)
    unodes = np.asarray(evaluator.u_nodes, dtype=np.float64)
    base_all = np.asarray(evaluator.base_w, dtype=np.float64)
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    ridge = float(model.cfg.newton_ridge)

    tangent = np.full(len(times), np.nan, dtype=np.float64)
    ess = np.zeros(len(times), dtype=np.float64)
    cal = np.full(len(times), np.inf, dtype=np.float64)
    lam_abs = np.full(len(times), np.nan, dtype=np.float64)
    lamdot_abs = np.full(len(times), np.nan, dtype=np.float64)
    gram_min = np.full(len(times), np.nan, dtype=np.float64)
    in_base_mass = np.zeros(len(times), dtype=np.float64)
    hull_inside = np.ones(len(times), dtype=bool)
    retry_count = 0
    lam_warm = np.zeros(2, dtype=np.float64)

    for kt, _t in enumerate(times):
        x_all = nodes[kt]
        u_all = unodes[kt]
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        u = u_all[mask]
        base_w = base_all[mask].copy()
        bm = float(np.sum(base_w))
        in_base_mass[kt] = bm
        base_w /= max(bm, 1e-300)

        phi, grad_phi = d2.sensor_particle_fields(model, eta, x)
        target = np.asarray(target_curve[kt], dtype=np.float64)
        cdot = np.asarray(target_cdot[kt], dtype=np.float64)

        # Keep the same empirical-hull semantics used by D7.evaluate_particle_curve.
        hd = d7.target_hull_diagnostic(phi, target)
        hull_inside[kt] = bool(hd["inside_closed_hull"])
        if not hull_inside[kt]:
            continue

        st = d7.robust_particle_mfsi_state(
            phi=phi,
            grad_phi=grad_phi,
            u=u,
            base_w=base_w,
            target=target,
            c_dot=cdot,
            ridge=ridge,
            cfg=cfg,
            lam0=lam_warm,
        )
        lam_warm = np.asarray(st["lambda"], dtype=np.float64)
        dd = st["diagnostics"]

        tangent[kt] = float(dd["tangent_action"])
        ess[kt] = float(dd["ess_fraction"])
        cal[kt] = float(dd["residual"])
        lam_abs[kt] = float(dd["max_abs_lambda_coordinate"])
        lamdot_abs[kt] = float(np.max(np.abs(np.asarray(st["lambda_dot"], dtype=np.float64))))
        gram_min[kt] = float(dd["min_tangent_gram_eig"])
        retry_count += int(dd.get("retry_count", 0))

    valid = bool(
        np.all(hull_inside)
        and np.all(np.isfinite(tangent))
        and np.max(cal) <= float(getattr(cfg, "max_finite_calibration_resid", 1e-3))
        and np.min(ess) >= float(getattr(cfg, "min_ess_fraction", 0.03))
        and np.min(in_base_mass) >= float(getattr(cfg, "min_in_domain_base_mass", 0.995))
    )

    return {
        "finite_tangent_action": float(np.sum(tw * tangent)) if valid else float("nan"),
        "finite_tangent_min_ess": float(np.min(ess)),
        "finite_tangent_max_calibration_resid": float(np.max(cal)),
        "finite_tangent_max_abs_lambda_coordinate": float(np.nanmax(lam_abs)),
        "finite_tangent_max_abs_lambda_dot_coordinate": float(np.nanmax(lamdot_abs)),
        "finite_tangent_min_gram_eig": float(np.nanmin(gram_min)),
        "finite_tangent_fallback_count": int(retry_count),
        "finite_tangent_valid": valid,
    }


def tangent_from_d7_detail(detail):
    """Extract the finite tangent objective from a D7 compute_action=True detail."""
    fin = detail["finite"]
    return {
        "finite_tangent_action": float(fin["tangent_action"]),
        "finite_tangent_min_ess": float(fin["min_ess_fraction"]),
        "finite_tangent_max_calibration_resid": float(fin["max_calibration_residual"]),
        "finite_tangent_max_abs_lambda_coordinate": float(fin["max_abs_lambda_coordinate"]),
        "finite_tangent_max_abs_lambda_dot_coordinate": float("nan"),
        "finite_tangent_min_gram_eig": float("nan"),
        "finite_tangent_fallback_count": int(fin.get("retry_case_count", 0)),
        "finite_tangent_valid": bool(fin["scientifically_valid"] and np.isfinite(fin["tangent_action"])),
    }


# -----------------------------------------------------------------------------
# Summaries / reuse
# -----------------------------------------------------------------------------


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = [
        "finite_heldout_mmd2", "exact_heldout_mmd2", "measurement_delta_mmd2",
        "finite_action", "exact_action", "measurement_action_inflation",
        "finite_tangent_action", "finite_min_ess", "finite_max_calibration_resid",
        "finite_max_abs_lambda_coordinate", "feasibility_projection_active",
        "feasibility_projection_norm", "quadratic_moment_rmse",
        "finite_tangent_min_ess", "finite_tangent_max_calibration_resid",
        "finite_tangent_max_abs_lambda_coordinate",
        "finite_tangent_max_abs_lambda_dot_coordinate", "finite_tangent_min_gram_eig",
        "finite_tangent_fallback_count",
    ]
    out = {k: mean_se([float(r.get(k, np.nan)) for r in rows]) for k in keys}
    if rows:
        out["finite_valid_fraction"] = float(np.mean([bool(r.get("finite_valid", False)) for r in rows]))
        out["exact_valid_fraction"] = float(np.mean([bool(r.get("exact_valid", False)) for r in rows]))
        out["tangent_valid_fraction"] = float(np.mean([bool(r.get("finite_tangent_valid", False)) for r in rows]))
    return out


def load_reusable_validation(path: Path | None) -> Dict[Tuple[float, float], List[Dict[str, Any]]]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text())
    rows_by_name = payload.get("validation_trial_rows", {})
    val = payload.get("validation", {})
    out = {}
    for name, rows in rows_by_name.items():
        eta = None
        if isinstance(val.get(name), dict):
            eta = val[name].get("eta")
            if eta is None and "eta_deg" in val[name]:
                eta = np.radians(np.asarray(val[name]["eta_deg"], dtype=np.float64)).tolist()
        if eta is None:
            # Known fixed-D8 aliases.
            if name == "robust_d8":
                eta = payload.get("selected_eta_rad") or payload.get("robust_d8_eta_rad")
            elif name == "lift":
                eta = np.radians([1.63, 161.63]).tolist()
            elif name == "tangent":
                eta = np.radians([0.0, 154.70]).tolist()
            elif name == "full":
                eta = np.radians([0.0, 160.0]).tolist()
        if eta is not None and isinstance(rows, list):
            out[canonical_key(np.asarray(eta, dtype=np.float64))] = rows
    return out


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d8-script", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--c2-script", required=True)
    p.add_argument("--d2-script", required=True)
    p.add_argument("--d3-script", required=True)
    p.add_argument("--d5-script", required=True)
    p.add_argument("--d7-script", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="reference")
    p.add_argument("--selection-json", default=None,
                   help="Optional D8 full/selection JSON containing finite_risk_feasible_rows.")
    p.add_argument("--reuse-validation-json", default=None,
                   help="Optional prior D8 validation-only/full JSON; matching selected-design rows are reused for law/full action.")
    p.add_argument("--survivor-eta-deg", nargs=2, type=float, action="append", default=None,
                   metavar=("THETA1", "THETA2"),
                   help="Explicit survivor pair; repeat flag for multiple pairs. Overrides defaults/selection JSON.")
    p.add_argument("--law-trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--validation-trials", type=int, default=None)
    p.add_argument("--output", default="stage_d8c_learned_native_law_tangent_full.json")
    return p


def main():
    wall0 = time.time()
    args = build_parser().parse_args()

    d8 = load_module(Path(args.d8_script), "stage_d8_fixed_for_d8c")
    backend = load_module(Path(args.backend), "stage_b_for_d8c")
    c2 = load_module(Path(args.c2_script), "stage_c2_for_d8c")
    d2 = load_module(Path(args.d2_script), "stage_d2_for_d8c")
    d3 = load_module(Path(args.d3_script), "stage_d3_for_d8c")
    d5 = load_module(Path(args.d5_script), "stage_d5_for_d8c")
    d7 = load_module(Path(args.d7_script), "stage_d7_for_d8c")

    # Preserve the corrected common-polytope projection from the fixed D8 script.
    if hasattr(d8, "install_robust_d3_projection"):
        d8.install_robust_d3_projection(d3)

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))
    cfg = d8.preset_d8_config(args.preset)
    repl = {}
    if args.law_trials is not None:
        repl["law_trials"] = int(args.law_trials)
    if args.action_trials is not None:
        repl["action_trials"] = int(args.action_trials)
    if args.validation_trials is not None:
        repl["validation_trials"] = int(args.validation_trials)
    cfg = dataclasses.replace(cfg, **repl)

    if args.preset == "quick":
        base_b = backend.preset_config("quick")
    else:
        base_b = backend.preset_config("reference")
    stage_b_cfg = dataclasses.replace(base_b, grid_n=int(cfg.grid_n), time_n=int(cfg.time_n))
    model = backend.StageB(stage_b_cfg)
    d7_cfg = d8.d7_compatible_config(cfg, d7)
    evaluator = d7.D7Evaluator(model, d2, d5, params, checkpoint_meta, d7_cfg)
    measurement_cov = d3.MeasurementCovariance(model)

    if args.survivor_eta_deg:
        candidates = dedupe_candidates([
            {"eta": np.radians(np.asarray(v, dtype=np.float64)), "source": "cli"}
            for v in args.survivor_eta_deg
        ])
    elif args.selection_json:
        candidates = candidates_from_selection_json(Path(args.selection_json))
    else:
        if args.preset != "reference":
            raise ValueError(
                "The built-in seven-survivor set is specific to the reference D8 run. "
                "For quick/confirm, pass --selection-json or repeated --survivor-eta-deg."
            )
        candidates = default_reference_survivors()

    # Acquisition/heldout split exactly as D8.
    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in acq_set and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    total_trials = int(cfg.law_trials + cfg.validation_trials)
    trial_bank = []
    for trial in range(total_trials):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 8800, trial]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        trial_bank.append(c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng))
    law_bank = trial_bank[: int(cfg.law_trials)]
    action_bank = law_bank[: int(cfg.action_trials)]
    validation_bank = trial_bank[int(cfg.law_trials):]

    output = Path(args.output)
    selection_progress = output.with_suffix(".selection_progress.json")
    validation_progress = output.with_suffix(".validation_progress.json")

    print("=" * 118)
    print("Stage D.8c — learned-native D5 Law vs Tangent-TC vs Full-TC")
    print("=" * 118)
    print(f"D5 checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"D5 bank       : {cfg.bank_mode}, GH order={cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Finite cond.  : N={cfg.finite_n}, K={cfg.acquisition_k}, noise={cfg.obs_noise_std:.4f}")
    print(f"Candidates    : {len(candidates)} D8 law-feasible survivors")
    print(f"Trials        : law={cfg.law_trials}, action={cfg.action_trials}, validation={cfg.validation_trials}")
    print(f"Held-out idx  : {heldout_idx.tolist()}")

    # Precompute common feasible geometry once per survivor.
    print("\nBuilding D5-native common feasibility geometry...", flush=True)
    for c in candidates:
        eta = np.asarray(c["eta"], dtype=np.float64)
        c["constraints"] = d7.build_joint_beta_constraints(
            model=model, evaluator=evaluator, d2=d2, c2=c2,
            eta=eta, margin=float(cfg.feasibility_margin),
        )
        c["population"] = d8.population_endpoint_fm_law(model, evaluator, d2, d7, eta, cfg)
        print(
            f"  eta={np.degrees(eta).round(3).tolist()} | "
            f"Lpop={c['population']['lift_mmd2']:.8e} | "
            f"constraints={c['constraints']['A'].shape[0]}", flush=True,
        )

    # ------------------------------------------------------------------
    # A. Recompute finite-law objective on the original D8 law bank.
    #    If a prior run completed this stage and then crashed later, reuse it.
    # ------------------------------------------------------------------
    resumed_law = False
    if selection_progress.exists():
        try:
            prev = json.loads(selection_progress.read_text(encoding="utf-8"))
            prev_rows = prev.get("candidates", [])
            prev_map = {}
            for pc in prev_rows:
                if "eta" in pc:
                    prev_map[canonical_key(np.asarray(pc["eta"], dtype=np.float64))] = pc
            if (
                str(prev.get("stage", "")).startswith("D8c selection progress: law")
                and int(prev.get("completed_candidates", 0)) >= len(candidates)
                and all(canonical_key(np.asarray(c["eta"])) in prev_map for c in candidates)
            ):
                for c in candidates:
                    pc = prev_map[canonical_key(np.asarray(c["eta"]))]
                    for k in ("finite_risk", "exact_risk", "measurement_degradation", "law_valid"):
                        if k in pc:
                            c[k] = pc[k]
                resumed_law = all("finite_risk" in c and "law_valid" in c for c in candidates)
        except Exception as exc:
            print(f"  NOTE: could not reuse prior law progress ({exc}); recomputing Stage A.", flush=True)

    if resumed_law:
        print("\n[A] Reusing completed finite-law Stage A from selection_progress JSON...", flush=True)
        for j, c in enumerate(candidates):
            print(
                f"  law {j+1}/{len(candidates)} eta={np.degrees(c['eta']).round(3).tolist()} | "
                f"R={c['finite_risk']['mean']:.8e} +/- {c['finite_risk']['se']:.2e} | valid={c['law_valid']}",
                flush=True,
            )
    else:
        print("\n[A] Re-evaluating finite law on the original D8 CRN law bank...", flush=True)
        for j, c in enumerate(candidates):
            rows = []
            valid = True
            for shared in law_bank:
                row, _ = d7.evaluate_design_trial(
                    model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                    measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=shared,
                    acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                    joint_constraints=c["constraints"], compute_action=False,
                )
                rows.append(row)
                valid = valid and bool(row.get("finite_valid", False)) and bool(row.get("exact_valid", False))
            c["law_rows"] = rows
            c["finite_risk"] = mean_se([r["finite_heldout_mmd2"] for r in rows])
            c["exact_risk"] = mean_se([r["exact_heldout_mmd2"] for r in rows])
            c["measurement_degradation"] = mean_se([r["measurement_delta_mmd2"] for r in rows])
            c["law_valid"] = bool(valid and np.isfinite(c["finite_risk"]["mean"]))
            print(
                f"  law {j+1}/{len(candidates)} eta={np.degrees(c['eta']).round(3).tolist()} | "
                f"R={c['finite_risk']['mean']:.8e} +/- {c['finite_risk']['se']:.2e} | valid={c['law_valid']}",
                flush=True,
            )
            write_json(selection_progress, {
                "stage": "D8c selection progress: law",
                "completed_candidates": j + 1,
                "candidates": [{k: jsonify(v) for k, v in cc.items() if k not in ("constraints", "law_rows")} for cc in candidates[:j+1]],
            })

    valid_law = [c for c in candidates if c["law_valid"]]
    if not valid_law:
        raise RuntimeError("No valid D8c law candidates")
    law_best = min(valid_law, key=lambda c: c["finite_risk"]["mean"])
    rstar = float(law_best["finite_risk"]["mean"])
    rmax = (1.0 + float(cfg.tau_r)) * rstar
    law_feasible = [c for c in valid_law if c["finite_risk"]["mean"] <= rmax + 1e-15]
    print(
        f"  -> D5-Law eta={np.degrees(law_best['eta']).round(6).tolist()} | "
        f"R*={rstar:.8e}; 1% set={len(law_feasible)}/{len(valid_law)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # B. On the SAME law-feasible set, compare tangent and full actions.
    # ------------------------------------------------------------------
    print("\n[B] D5-native tangent vs full action on the same law-feasible set...", flush=True)
    for j, c in enumerate(law_feasible):
        rows = []
        all_valid = True
        for shared in action_bank:
            row, detail = d7.evaluate_design_trial(
                model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                measurement_cov=measurement_cov, eta=np.asarray(c["eta"]), shared=shared,
                acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                joint_constraints=c["constraints"], compute_action=True,
            )
            # D7 already computed tangent_action from the same robust calibrated
            # finite branch used for the Poisson full action.  Reuse it exactly.
            tan = tangent_from_d7_detail(detail)
            rr = dict(row)
            rr.update(tan)
            rows.append(rr)
            all_valid = all_valid and bool(row.get("finite_valid", False)) and bool(tan["finite_tangent_valid"])
        c["action_rows"] = rows
        c["finite_full_action"] = mean_se([r["finite_action"] for r in rows])
        c["finite_tangent_action"] = mean_se([r["finite_tangent_action"] for r in rows])
        c["action_valid"] = bool(all_valid)
        print(
            f"  action {j+1}/{len(law_feasible)} eta={np.degrees(c['eta']).round(3).tolist()} | "
            f"T={c['finite_tangent_action']['mean']:.6e} | "
            f"A={c['finite_full_action']['mean']:.6e} | valid={c['action_valid']}",
            flush=True,
        )
        write_json(selection_progress, {
            "stage": "D8c selection progress: action",
            "completed_candidates": j + 1,
            "law_best_eta_deg": np.degrees(law_best["eta"]).tolist(),
            "rstar": rstar, "rmax": rmax,
            "candidates": [
                {k: jsonify(v) for k, v in cc.items() if k not in ("constraints", "law_rows", "action_rows")}
                for cc in law_feasible[:j+1]
            ],
        })

    valid_action = [c for c in law_feasible if c.get("action_valid", False)]
    if not valid_action:
        raise RuntimeError("No valid D8c action candidates")
    tangent_best = min(valid_action, key=lambda c: c["finite_tangent_action"]["mean"])
    full_best = min(valid_action, key=lambda c: c["finite_full_action"]["mean"])

    print("\nSelected learned-native designs:")
    print(
        f"  D5-Law     : {np.degrees(law_best['eta']).round(6).tolist()} | "
        f"R={law_best['finite_risk']['mean']:.8e}"
    )
    print(
        f"  D5-Tangent : {np.degrees(tangent_best['eta']).round(6).tolist()} | "
        f"T={tangent_best['finite_tangent_action']['mean']:.6e} | "
        f"A={tangent_best['finite_full_action']['mean']:.6e}"
    )
    print(
        f"  D5-Full    : {np.degrees(full_best['eta']).round(6).tolist()} | "
        f"T={full_best['finite_tangent_action']['mean']:.6e} | "
        f"A={full_best['finite_full_action']['mean']:.6e}"
    )
    if args.preset == "reference" and args.selection_json is None and args.survivor_eta_deg is None:
        expected_d8 = np.array([4.0 * np.pi / 37.0, 14.0 * np.pi / 37.0])
        if canonical_key(full_best["eta"]) != canonical_key(expected_d8):
            print(
                "  WARNING: D5-Full did not reproduce the original D8 selected grid point "
                f"{np.degrees(expected_d8).round(6).tolist()}. Inspect action rows before interpretation.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # C. Independent validation on trials law_trials ... law_trials+validation-1.
    # ------------------------------------------------------------------
    print("\n[C] Independent validation of D5-Law / D5-Tangent / D5-Full...", flush=True)
    selected = {
        "d5_law": law_best,
        "d5_tangent": tangent_best,
        "d5_full": full_best,
    }
    reusable = load_reusable_validation(Path(args.reuse_validation_json) if args.reuse_validation_json else None)
    validation_rows: Dict[str, List[Dict[str, Any]]] = {}
    validation_summary: Dict[str, Any] = {}

    # Deduplicate calculations if objectives select the same eta.
    cache: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for name, c in selected.items():
        eta = np.asarray(c["eta"], dtype=np.float64)
        key = canonical_key(eta)
        if key in cache:
            validation_rows[name] = cache[key]
            validation_summary[name] = summarize_rows(cache[key])
            continue

        base_rows = reusable.get(key)
        rows = []
        for trial_i, shared in enumerate(validation_bank):
            # Reuse existing D8 law/full row when exact eta and trial count match;
            # tangent is still recomputed from the same deterministic finite curve.
            if base_rows is not None and trial_i < len(base_rows):
                row = dict(base_rows[trial_i])
                # Reused D8 validation rows predate the tangent field. Reconstruct
                # only the finite target curve and run the robust D7 tangent state;
                # no Poisson solve is repeated.
                curve = reconstruct_finite_curve(
                    model, evaluator, d3, c2, measurement_cov, eta, shared,
                    acq_idx, d7_cfg, c["constraints"],
                )
                tan = tangent_curve_action(
                    model, evaluator, d2, d7, eta,
                    np.asarray(curve["c"], dtype=np.float64),
                    np.asarray(curve["cdot"], dtype=np.float64), d7_cfg,
                )
            else:
                row, detail = d7.evaluate_design_trial(
                    model=model, evaluator=evaluator, d2=d2, d3=d3, c2=c2,
                    measurement_cov=measurement_cov, eta=eta, shared=shared,
                    acq_idx=acq_idx, heldout_mask=heldout_mask, cfg=d7_cfg,
                    joint_constraints=c["constraints"], compute_action=True,
                )
                tan = tangent_from_d7_detail(detail)
            rr = dict(row)
            rr.update(tan)
            rr["validation_trial_index"] = int(cfg.law_trials + trial_i)
            rows.append(rr)

        cache[key] = rows
        validation_rows[name] = rows
        validation_summary[name] = summarize_rows(rows)
        s = validation_summary[name]
        print(
            f"  {name:10s} eta={np.degrees(eta).round(3).tolist()} | "
            f"R={s['finite_heldout_mmd2']['mean']:.8e} | "
            f"T={s['finite_tangent_action']['mean']:.6e} | "
            f"A={s['finite_action']['mean']:.6e}", flush=True,
        )
        write_json(validation_progress, {
            "stage": "D8c validation progress",
            "completed_designs": list(validation_rows.keys()),
            "selected_eta_deg": {k: np.degrees(v["eta"]).tolist() for k, v in selected.items()},
            "validation_summary": validation_summary,
            "validation_trial_rows": validation_rows,
        })

    comparisons = {
        "tangent_vs_law_finite_law": paired_difference(validation_rows["d5_tangent"], validation_rows["d5_law"], "finite_heldout_mmd2"),
        "tangent_vs_law_tangent_reduction": paired_reduction(validation_rows["d5_tangent"], validation_rows["d5_law"], "finite_tangent_action"),
        "tangent_vs_law_full_reduction": paired_reduction(validation_rows["d5_tangent"], validation_rows["d5_law"], "finite_action"),
        "full_vs_law_finite_law": paired_difference(validation_rows["d5_full"], validation_rows["d5_law"], "finite_heldout_mmd2"),
        "full_vs_law_tangent_reduction": paired_reduction(validation_rows["d5_full"], validation_rows["d5_law"], "finite_tangent_action"),
        "full_vs_law_full_reduction": paired_reduction(validation_rows["d5_full"], validation_rows["d5_law"], "finite_action"),
        "full_vs_tangent_finite_law": paired_difference(validation_rows["d5_full"], validation_rows["d5_tangent"], "finite_heldout_mmd2"),
        "full_vs_tangent_tangent_reduction": paired_reduction(validation_rows["d5_full"], validation_rows["d5_tangent"], "finite_tangent_action"),
        "full_vs_tangent_full_reduction": paired_reduction(validation_rows["d5_full"], validation_rows["d5_tangent"], "finite_action"),
    }

    def public_candidate(c):
        return {
            "theta_deg": np.degrees(c["eta"]).tolist(),
            "source": c.get("source"),
            "population_law": c["population"]["lift_mmd2"],
            "finite_risk": c.get("finite_risk"),
            "exact_risk": c.get("exact_risk"),
            "measurement_degradation": c.get("measurement_degradation"),
            "finite_tangent_action": c.get("finite_tangent_action"),
            "finite_full_action": c.get("finite_full_action"),
            "law_valid": c.get("law_valid"),
            "action_valid": c.get("action_valid"),
        }

    payload = {
        "stage": "D.8c learned-native D5 Law vs Tangent-TC vs Full-TC",
        "created_unix_time": time.time(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "paths": {
            "d8_script": str(Path(args.d8_script).resolve()),
            "backend": str(Path(args.backend).resolve()),
            "c2_script": str(Path(args.c2_script).resolve()),
            "d2_script": str(Path(args.d2_script).resolve()),
            "d3_script": str(Path(args.d3_script).resolve()),
            "d5_script": str(Path(args.d5_script).resolve()),
            "d7_script": str(Path(args.d7_script).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "selection_json": str(Path(args.selection_json).resolve()) if args.selection_json else None,
            "reuse_validation_json": str(Path(args.reuse_validation_json).resolve()) if args.reuse_validation_json else None,
        },
        "config": jsonify(cfg),
        "stage_b_config": jsonify(stage_b_cfg),
        "condition": {
            "law_trials": int(cfg.law_trials),
            "action_trials": int(cfg.action_trials),
            "validation_trials": int(cfg.validation_trials),
            "selection_trial_indices": list(range(int(cfg.law_trials))),
            "action_trial_indices": list(range(int(cfg.action_trials))),
            "validation_trial_indices": list(range(int(cfg.law_trials), int(cfg.law_trials + cfg.validation_trials))),
            "acquisition_indices": acq_idx.tolist(),
            "heldout_indices": heldout_idx.tolist(),
            "tau_r": float(cfg.tau_r),
            "finite_risk_star": rstar,
            "finite_risk_max": rmax,
        },
        "candidate_count": len(candidates),
        "law_feasible_count_after_recheck": len(law_feasible),
        "candidate_summary": [public_candidate(c) for c in candidates],
        "selected": {
            "d5_law": public_candidate(law_best),
            "d5_tangent": public_candidate(tangent_best),
            "d5_full": public_candidate(full_best),
        },
        "validation_summary": validation_summary,
        "validation_comparisons": comparisons,
        "validation_trial_rows": validation_rows,
        "interpretation_notes": [
            "All three headline designs are defined under the same endpoint-trained D5 interpolant; the historical Stage-B angles are not design objectives here.",
            "D5-Law minimizes finite-resource held-out law MMD among the D8 survivor set.",
            "D5-Tangent minimizes finite-resource particle tangent action among candidates that remain within the same D8 finite-law tolerance.",
            "D5-Full minimizes finite-resource weighted-Poisson action among that same law-feasible set and should reproduce the D8 selection under the reference configuration.",
            "The tangent action is computed on the same finite reconstructed moment curve and same D5 I-projected particle law as the full-action evaluation.",
            "Selection uses the original CRN law/action banks; validation uses the original disjoint D8 validation bank.",
            "No old analytic interpolant, old A_t/B_t teacher, CNF density, score model, or learned likelihood is used.",
        ],
        "wall_seconds": float(time.time() - wall0),
    }
    write_json(output, payload)

    # Flat CSVs are convenient for plots / manuscript tables.
    cand_csv = output.with_suffix(".candidate_summary.csv")
    val_csv = output.with_suffix(".validation_trials.csv")
    write_csv(cand_csv, [
        {
            "theta1_deg": np.degrees(c["eta"])[0],
            "theta2_deg": np.degrees(c["eta"])[1],
            "population_law": c["population"]["lift_mmd2"],
            "finite_risk": c["finite_risk"]["mean"],
            "finite_risk_se": c["finite_risk"]["se"],
            "finite_tangent_action": c.get("finite_tangent_action", {}).get("mean", np.nan),
            "finite_tangent_action_se": c.get("finite_tangent_action", {}).get("se", np.nan),
            "finite_full_action": c.get("finite_full_action", {}).get("mean", np.nan),
            "finite_full_action_se": c.get("finite_full_action", {}).get("se", np.nan),
            "is_d5_law": canonical_key(c["eta"]) == canonical_key(law_best["eta"]),
            "is_d5_tangent": canonical_key(c["eta"]) == canonical_key(tangent_best["eta"]),
            "is_d5_full": canonical_key(c["eta"]) == canonical_key(full_best["eta"]),
        }
        for c in candidates
    ])
    flat_val = []
    for name, rows in validation_rows.items():
        for r in rows:
            flat_val.append({"design": name, **r})
    write_csv(val_csv, flat_val)

    print("\n" + "=" * 118)
    print("D8c independent-validation comparison")
    print("=" * 118)
    for name in ("d5_law", "d5_tangent", "d5_full"):
        s = validation_summary[name]
        eta = selected[name]["eta"]
        print(
            f"{name:10s} eta={np.degrees(eta).round(3).tolist()} | "
            f"R={s['finite_heldout_mmd2']['mean']:.8e} +/- {s['finite_heldout_mmd2']['se']:.2e} | "
            f"T={s['finite_tangent_action']['mean']:.6e} +/- {s['finite_tangent_action']['se']:.2e} | "
            f"A={s['finite_action']['mean']:.6e} +/- {s['finite_action']['se']:.2e}"
        )
    fl = comparisons["full_vs_law_full_reduction"]
    tl = comparisons["tangent_vs_law_tangent_reduction"]
    ft = comparisons["full_vs_tangent_full_reduction"]
    print(
        f"D5-Full vs D5-Law full-action reduction: "
        f"ratio-of-means={100*fl['ratio_of_means_reduction']:+.2f}% | "
        f"paired={100*fl['mean_paired_reduction']:+.2f}% +/- {100*fl['se_paired_reduction']:.2f}%"
    )
    print(
        f"D5-Tangent vs D5-Law tangent-action reduction: "
        f"ratio-of-means={100*tl['ratio_of_means_reduction']:+.2f}% | "
        f"paired={100*tl['mean_paired_reduction']:+.2f}% +/- {100*tl['se_paired_reduction']:.2f}%"
    )
    print(
        f"D5-Full vs D5-Tangent full-action reduction: "
        f"ratio-of-means={100*ft['ratio_of_means_reduction']:+.2f}% | "
        f"paired={100*ft['mean_paired_reduction']:+.2f}% +/- {100*ft['se_paired_reduction']:.2f}%"
    )
    print(f"Saved JSON : {output.resolve()}")
    print(f"Saved CSV  : {cand_csv.resolve()}")
    print(f"Saved CSV  : {val_csv.resolve()}")


if __name__ == "__main__":
    main()
