#!/usr/bin/env python3
"""
Stage D.8: finite-resource sensor-design oracle under an endpoint-trained
flow-matching particle reference (NO CNF, NO old analytic SI).

Scientific purpose
------------------
D.5 trained a reference velocity from endpoint samples rather than from the old
Stage-B analytic path teacher. D.6 validated frozen exact-population MFSI under
that reference. D.7 added finite/noisy measurements while keeping Lift,
Tangent-TC and Full-TC frozen.

D.8 now lets the physically admissible two-sensor design eta move again.

The oracle is deliberately DISCRETE and LEXICOGRAPHIC, matching Stage C.3/D.4:

  1. Endpoint-trained-FM population-law sufficiency

       L_D5(eta) <= (1 + tau_L) min_eta L_D5(eta).

  2. Endpoint-trained-FM finite-resource law sufficiency, evaluated with common
     random numbers (CRN) over the candidates that pass stage 1

       R_D5,N(eta) <= (1 + tau_R)
                      min_{population-feasible eta} R_D5,N(eta).

  3. Among candidates passing BOTH law screens, minimize endpoint-trained-FM
     finite-resource weighted-Poisson action

       A_D5,N(eta).

The selected D.8 design is then evaluated on a COMPLETELY DISJOINT validation
bank together with the historical Lift, Tangent-TC and Full-TC controls.

Reference / inference restrictions
----------------------------------
* D.5 learned ODE rollout particles are the ONLY reference marginal used.
* No old Stage-B A_t or B_t is used.
* No old analytic-reference particle comparison branch is used.
* No CNF density, log-Jacobian, score model or learned likelihood is used.
* Finite uncertainty enters once through the D.7 endpoint-anchored GLS
  reconstructed moment curve.
* Reconstructed finite curves are constrained to ONE common feasible polytope:
    physical sensor moment hull INTERSECT D.5 particle moment hulls over time.
* Hard empirical MFSI calibration uses the robust D.7 convex-dual solver and the
  D.2/D.6 nonuniform-quadrature ESS convention.

Frozen controls
---------------
The historical controls are ALWAYS inserted into the candidate set even when
not on the dense angular grid:

    Lift        = (1.63 deg, 161.63 deg)
    Tangent-TC  = (0.00 deg, 154.70 deg)
    Full-TC     = (0.00 deg, 160.00 deg)

Recommended reference run
-------------------------
python stage_d8_endpoint_flow_matching_finite_resource_design_oracle.py \
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \
    --c2-script ../stage_c/stage_c2_mfsi_matched_action.py \
    --d2-script stage_d2_flow_matching_particle_mfsi.py \
    --d3-script stage_d3_flow_matching_finite_measurements.py \
    --d5-script stage_d5_endpoint_flow_matching_reference_v2.py \
    --d7-script stage_d7_endpoint_flow_matching_finite_measurements_v2.py \
    --checkpoint stage_d5_endpoint_flow_matching_reference_v2.npz \
    --preset reference \
    --output stage_d8_endpoint_flow_matching_finite_resource_design_oracle.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


PI = math.pi


# -----------------------------------------------------------------------------
# Dynamic imports
# -----------------------------------------------------------------------------


def load_module(path: Path, module_name: str):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def autodetect(names: Sequence[str]) -> Path | None:
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    return None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class D8Config:
    preset: str = "quick"
    seed: int = 20260813

    # Finite-resource condition.
    finite_n: int = 100
    acquisition_k: int = 7
    obs_noise_std: float = 0.01

    # Discrete design oracle.
    angle_n: int = 13
    law_trials: int = 4
    action_trials: int = 2
    validation_trials: int = 4
    tau_l: float = 0.05
    tau_r: float = 0.01

    # Stage-B / D.5 particle discretization.
    grid_n: int = 19
    time_n: int = 13
    bank_mode: str = "gauss-hermite"
    gh_order: int = 20
    particles: int = 8192
    rk4_substeps_per_time_interval: int = 8
    kde_bandwidth: float = 0.0
    kde_truncate: float = 4.0

    # Finite-measurement reconstruction.
    variance_floor: float = 1.0e-10
    quadratic_ridge_rel: float = 1.0e-12
    feasibility_margin: float = 0.0

    # Robust hard empirical I-projection.
    calibration_steps: int = 80
    calibration_tol: float = 2.0e-8
    solver_accept_tol: float = 2.0e-6
    newton_step_cap: float = 10.0
    lambda_clip: float = 300.0
    calibration_lbfgs_maxiter: int = 400
    calibration_max_retries: int = 2
    calibration_retry_clip_multiplier: float = 2.0
    clip_saturation_fraction: float = 0.995

    # Numerical/reference-overlap gates. NOT design objectives.
    max_population_calibration_resid: float = 5.0e-6
    max_finite_calibration_resid: float = 1.0e-3
    min_ess_fraction: float = 0.03
    min_in_domain_base_mass: float = 0.995

    # Frozen historical designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)


def preset_d8_config(name: str) -> D8Config:
    if name == "quick":
        return D8Config()
    if name == "reference":
        return D8Config(
            preset="reference",
            finite_n=100,
            acquisition_k=11,
            obs_noise_std=0.01,
            angle_n=37,
            law_trials=24,
            action_trials=8,
            validation_trials=24,
            tau_l=0.05,
            tau_r=0.01,
            grid_n=51,
            time_n=21,
            gh_order=36,
            particles=32768,
            rk4_substeps_per_time_interval=16,
            calibration_steps=300,
            calibration_tol=1.0e-9,
            solver_accept_tol=2.0e-6,
            newton_step_cap=20.0,
            lambda_clip=1000.0,
            calibration_lbfgs_maxiter=800,
            calibration_max_retries=2,
            max_population_calibration_resid=1.0e-5,
            max_finite_calibration_resid=1.0e-3,
            min_ess_fraction=0.03,
        )
    if name == "confirm":
        return D8Config(
            preset="confirm",
            finite_n=100,
            acquisition_k=11,
            obs_noise_std=0.01,
            angle_n=49,
            law_trials=40,
            action_trials=12,
            validation_trials=40,
            tau_l=0.05,
            tau_r=0.01,
            grid_n=65,
            time_n=27,
            gh_order=48,
            particles=65536,
            rk4_substeps_per_time_interval=24,
            calibration_steps=500,
            calibration_tol=5.0e-10,
            solver_accept_tol=1.0e-6,
            newton_step_cap=25.0,
            lambda_clip=2000.0,
            calibration_lbfgs_maxiter=1200,
            calibration_max_retries=2,
            max_population_calibration_resid=5.0e-6,
            max_finite_calibration_resid=5.0e-4,
            min_ess_fraction=0.03,
        )
    raise ValueError(name)


def d7_compatible_config(cfg: D8Config, d7):
    """Build the exact config object consumed by D.7 helpers/evaluator."""
    base = d7.preset_d7_config(cfg.preset)
    return dataclasses.replace(
        base,
        seed=int(cfg.seed),
        finite_n=int(cfg.finite_n),
        acquisition_k=int(cfg.acquisition_k),
        obs_noise_std=float(cfg.obs_noise_std),
        grid_n=int(cfg.grid_n),
        time_n=int(cfg.time_n),
        bank_mode=str(cfg.bank_mode),
        gh_order=int(cfg.gh_order),
        particles=int(cfg.particles),
        rk4_substeps_per_time_interval=int(cfg.rk4_substeps_per_time_interval),
        kde_bandwidth=float(cfg.kde_bandwidth),
        kde_truncate=float(cfg.kde_truncate),
        variance_floor=float(cfg.variance_floor),
        quadratic_ridge_rel=float(cfg.quadratic_ridge_rel),
        feasibility_margin=float(cfg.feasibility_margin),
        calibration_steps=int(cfg.calibration_steps),
        calibration_tol=float(cfg.calibration_tol),
        solver_accept_tol=float(cfg.solver_accept_tol),
        newton_step_cap=float(cfg.newton_step_cap),
        lambda_clip=float(cfg.lambda_clip),
        calibration_lbfgs_maxiter=int(cfg.calibration_lbfgs_maxiter),
        calibration_max_retries=int(cfg.calibration_max_retries),
        calibration_retry_clip_multiplier=float(cfg.calibration_retry_clip_multiplier),
        clip_saturation_fraction=float(cfg.clip_saturation_fraction),
        max_population_calibration_resid=float(cfg.max_population_calibration_resid),
        max_finite_calibration_resid=float(cfg.max_finite_calibration_resid),
        min_ess_fraction=float(cfg.min_ess_fraction),
        min_in_domain_base_mass=float(cfg.min_in_domain_base_mass),
        lift_design_deg=tuple(cfg.lift_design_deg),
        tangent_design_deg=tuple(cfg.tangent_design_deg),
        full_design_deg=tuple(cfg.full_design_deg),
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


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
    return {
        "mean": float(np.mean(x)),
        "se": float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0,
        "n": int(x.size),
    }


def paired_difference(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]], key: str):
    a = np.asarray([float(r.get(key, np.nan)) for r in rows_a], dtype=np.float64)
    b = np.asarray([float(r.get(key, np.nan)) for r in rows_b], dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    s = mean_se(d)
    return {
        "mean_difference_a_minus_b": s["mean"],
        "se_difference": s["se"],
        "n": s["n"],
        "a_better_fraction": float(np.mean(d < 0.0)) if d.size else float("nan"),
    }


def paired_reduction(rows_num: Sequence[Mapping[str, Any]], rows_den: Sequence[Mapping[str, Any]], key: str):
    num = np.asarray([float(r.get(key, np.nan)) for r in rows_num], dtype=np.float64)
    den = np.asarray([float(r.get(key, np.nan)) for r in rows_den], dtype=np.float64)
    keep = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-14)
    num = num[keep]
    den = den[keep]
    if num.size == 0:
        return {
            "ratio_of_means_reduction": float("nan"),
            "mean_paired_reduction": float("nan"),
            "se_paired_reduction": float("nan"),
            "n": 0,
        }
    paired = 1.0 - num / den
    s = mean_se(paired)
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(num) / np.mean(den)),
        "mean_paired_reduction": s["mean"],
        "se_paired_reduction": s["se"],
        "n": s["n"],
    }


def projective_angle_distance(a: float, b: float) -> float:
    d = abs((float(a) - float(b)) % PI)
    return min(d, PI - d)


def canonical_eta(eta: np.ndarray) -> np.ndarray:
    x = np.mod(np.asarray(eta, dtype=np.float64), PI)
    return np.sort(x)


def unordered_design_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = canonical_eta(a)
    b = canonical_eta(b)
    direct = math.hypot(
        projective_angle_distance(a[0], b[0]),
        projective_angle_distance(a[1], b[1]),
    )
    swap = math.hypot(
        projective_angle_distance(a[0], b[1]),
        projective_angle_distance(a[1], b[0]),
    )
    return min(direct, swap)


def oracle_candidate_designs(
    angle_n: int,
    min_sep_deg: float,
    extra_designs: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    if angle_n < 5:
        raise ValueError("angle_n must be >= 5")
    grid = np.linspace(0.0, PI, int(angle_n), endpoint=False, dtype=np.float64)
    out: List[Dict[str, Any]] = []
    seen: Dict[Tuple[float, float], int] = {}

    def add(eta: np.ndarray, source: str):
        eta = canonical_eta(eta)
        sep = math.degrees(projective_angle_distance(eta[0], eta[1]))
        if sep < float(min_sep_deg) - 1.0e-10:
            return
        key = tuple(np.round(eta, 12))
        if key in seen:
            out[seen[key]]["sources"].append(source)
            return
        seen[key] = len(out)
        out.append({
            "eta": eta,
            "theta_deg": np.degrees(eta).tolist(),
            "sensor_separation_deg": float(sep),
            "sources": [source],
        })

    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            add(np.array([grid[i], grid[j]], dtype=np.float64), "dense_grid")
    for name, eta in extra_designs.items():
        add(np.asarray(eta, dtype=np.float64), f"frozen_{name}")
    return out


def exact_target_curve(model, evaluator, eta: np.ndarray, alpha: float):
    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    c = np.stack([
        np.asarray(
            model.measurement_grid(jnp.asarray(t), alpha_j, eta_j),
            dtype=np.float64,
        )
        for t in times
    ], axis=0)
    cdot = np.stack([
        np.asarray(
            evaluator.cdot_jit(jnp.asarray(t), alpha_j, eta_j),
            dtype=np.float64,
        )
        for t in times
    ], axis=0)
    return c, cdot


def population_endpoint_fm_law(model, evaluator, d2, d7, eta: np.ndarray, cfg: D8Config):
    """Exact-population law screen using only the D.5 endpoint-trained particle reference."""
    alphas = np.asarray(model.alphas, dtype=np.float64)
    aw = np.asarray(model.alpha_w, dtype=np.float64)
    all_mask = np.ones(model.cfg.time_n, dtype=bool)

    vals = []
    raw_vals = []
    min_ess = float("inf")
    max_cal = 0.0
    max_grid_moment = 0.0
    max_lambda_coord = 0.0
    max_clip_fraction = 0.0
    max_retry_count = 0
    all_valid = True
    all_hull = True
    min_domain = 1.0

    for alpha in alphas:
        c, cdot = exact_target_curve(model, evaluator, eta, float(alpha))
        r = d7.evaluate_particle_curve(
            evaluator=evaluator,
            d2=d2,
            eta=eta,
            alpha=float(alpha),
            target_curve=c,
            target_cdot=cdot,
            heldout_mask=all_mask,
            compute_action=False,
            calibration_validity_tol=float(cfg.max_population_calibration_resid),
        )
        raw_vals.append(float(r["raw_heldout_mmd2"]))
        vals.append(float(r["heldout_mmd2"]))
        all_valid = all_valid and bool(r["scientifically_valid"])
        all_hull = all_hull and bool(r["all_targets_inside_empirical_moment_hull"])
        min_ess = min(min_ess, float(r["min_ess_fraction"]))
        max_cal = max(max_cal, float(r["max_calibration_residual"]))
        max_grid_moment = max(max_grid_moment, float(r["max_grid_moment_error_after_kde"]))
        max_lambda_coord = max(max_lambda_coord, float(r["max_abs_lambda_coordinate"]))
        max_clip_fraction = max(max_clip_fraction, float(r["max_clip_fraction"]))
        max_retry_count = max(max_retry_count, int(r["max_retry_count"]))
        min_domain = min(min_domain, float(r["min_in_domain_base_mass"]))

    vals_np = np.asarray(vals, dtype=np.float64)
    raw_np = np.asarray(raw_vals, dtype=np.float64)
    law = float(np.sum(aw * vals_np)) if np.all(np.isfinite(vals_np)) else float("nan")
    raw_law = float(np.sum(aw * raw_np))

    return {
        "lift_mmd2": law,
        "raw_lift_mmd2": raw_law,
        "scientifically_valid": bool(all_valid),
        "all_targets_inside_empirical_moment_hulls": bool(all_hull),
        "min_ess_fraction": float(min_ess),
        "max_calibration_residual": float(max_cal),
        "max_grid_moment_error_after_kde": float(max_grid_moment),
        "max_abs_lambda_coordinate": float(max_lambda_coord),
        "max_clip_fraction": float(max_clip_fraction),
        "max_retry_count": int(max_retry_count),
        "min_in_domain_base_mass": float(min_domain),
    }


def summarize_selection_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = [
        "finite_heldout_mmd2",
        "exact_heldout_mmd2",
        "measurement_delta_mmd2",
        "finite_action",
        "exact_action",
        "measurement_action_inflation",
        "finite_min_ess",
        "finite_min_chi2_ess",
        "finite_max_calibration_resid",
        "finite_max_abs_lambda_coordinate",
        "finite_max_clip_fraction",
        "finite_retry_case_count",
        "finite_outside_hull_count",
        "exact_min_ess",
        "exact_max_calibration_resid",
        "feasibility_projection_active",
        "feasibility_projection_norm",
        "quadratic_moment_rmse",
        "quadratic_moment_max_error",
        "acquisition_mean_rmse",
    ]
    out = {k: mean_se([float(r.get(k, np.nan)) for r in rows]) for k in keys}
    if rows:
        out.update({
            "finite_valid_fraction": float(np.mean([bool(r.get("finite_valid", 0.0)) for r in rows])),
            "exact_valid_fraction": float(np.mean([bool(r.get("exact_valid", 0.0)) for r in rows])),
            "max_finite_calibration_resid": float(max(float(r.get("finite_max_calibration_resid", np.inf)) for r in rows)),
            "min_finite_ess": float(min(float(r.get("finite_min_ess", 0.0)) for r in rows)),
            "projection_active_fraction": float(np.mean([float(r.get("feasibility_projection_active", 0.0)) for r in rows])),
            "max_projection_norm": float(max(float(r.get("feasibility_projection_norm", 0.0)) for r in rows)),
            "max_abs_lambda_coordinate": float(max(float(r.get("finite_max_abs_lambda_coordinate", 0.0)) for r in rows)),
            "max_clip_fraction": float(max(float(r.get("finite_max_clip_fraction", 0.0)) for r in rows)),
            "retry_case_count": int(sum(int(r.get("finite_retry_case_count", 0)) for r in rows)),
        })
    return out


def public_candidate(c: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "theta_deg", "sensor_separation_deg", "sources",
        "population_law", "population_raw_law", "population_min_ess",
        "population_max_cal", "population_max_abs_lambda_coordinate",
        "population_max_clip_fraction", "population_max_retry_count",
        "population_numerically_valid",
        "finite_risk", "finite_degradation", "finite_min_ess", "finite_max_cal",
        "finite_max_abs_lambda_coordinate", "finite_max_clip_fraction",
        "finite_retry_case_count", "finite_numerically_valid",
        "projection_rate", "max_projection_norm",
        "finite_action", "matched_exact_action", "action_inflation",
        "action_numerically_valid",
    ]
    return {k: jsonify(c[k]) for k in keys if k in c}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--c2-script", type=str, default=None)
    p.add_argument("--d2-script", type=str, default=None)
    p.add_argument("--d3-script", type=str, default=None)
    p.add_argument("--d5-script", type=str, default=None)
    p.add_argument("--d7-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d8_endpoint_flow_matching_finite_resource_design_oracle.json")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--angle-n", type=int, default=None)
    p.add_argument("--law-trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--validation-trials", type=int, default=None)
    p.add_argument("--tau-l", type=float, default=None)
    p.add_argument("--tau-r", type=float, default=None)
    p.add_argument("--N", dest="finite_n", type=int, default=None)
    p.add_argument("--K", dest="acquisition_k", type=int, default=None)
    p.add_argument("--noise", "--noise-std", dest="obs_noise_std", type=float, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--time-n", type=int, default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)
    p.add_argument("--feasibility-margin", type=float, default=None)
    p.add_argument("--calibration-steps", type=int, default=None)
    p.add_argument("--lambda-clip", type=float, default=None)
    p.add_argument("--newton-step-cap", type=float, default=None)
    p.add_argument("--lbfgs-maxiter", type=int, default=None)
    p.add_argument("--calibration-max-retries", type=int, default=None)
    p.add_argument("--max-population-calibration-resid", type=float, default=None)
    p.add_argument("--max-finite-calibration-resid", type=float, default=None)
    p.add_argument("--min-ess-fraction", type=float, default=None)
    return p


def main():
    wall0 = time.time()
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect([
        "stage_b2_transport_conditioned_design.py",
        "stage_b2_transport_conditioned_design(5).py",
    ])
    c2_path = Path(args.c2_script) if args.c2_script else autodetect([
        "../stage_c/stage_c2_mfsi_matched_action.py",
        "stage_c2_mfsi_matched_action.py",
    ])
    d2_path = Path(args.d2_script) if args.d2_script else autodetect([
        "stage_d2_flow_matching_particle_mfsi.py",
        "stage_d2_flow_matching_particle_mfsi(2).py",
    ])
    d3_path = Path(args.d3_script) if args.d3_script else autodetect([
        "stage_d3_flow_matching_finite_measurements.py",
        "stage_d3_flow_matching_finite_measurements(3).py",
    ])
    d5_path = Path(args.d5_script) if args.d5_script else autodetect([
        "stage_d5_endpoint_flow_matching_reference_v2.py",
        "stage_d5_endpoint_flow_matching_reference.py",
    ])
    d7_path = Path(args.d7_script) if args.d7_script else autodetect([
        "stage_d7_endpoint_flow_matching_finite_measurements_v2.py",
        "stage_d7_endpoint_flow_matching_finite_measurements.py",
    ])

    missing = [
        name for name, path in (
            ("backend", backend_path),
            ("c2-script", c2_path),
            ("d2-script", d2_path),
            ("d3-script", d3_path),
            ("d5-script", d5_path),
            ("d7-script", d7_path),
        ) if path is None
    ]
    if missing:
        raise FileNotFoundError(f"Could not autodetect: {missing}. Pass explicit paths.")

    backend = load_module(backend_path, "stage_b_backend_d8")
    c2 = load_module(c2_path, "stage_c2_backend_d8")
    d2 = load_module(d2_path, "stage_d2_helpers_d8")
    d3 = load_module(d3_path, "stage_d3_helpers_d8")
    d5 = load_module(d5_path, "stage_d5_backend_d8")
    d7 = load_module(d7_path, "stage_d7_helpers_d8")

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d8_config(args.preset)
    overrides: Dict[str, Any] = {}
    for arg_name, field_name, cast in (
        ("seed", "seed", int),
        ("angle_n", "angle_n", int),
        ("law_trials", "law_trials", int),
        ("action_trials", "action_trials", int),
        ("validation_trials", "validation_trials", int),
        ("tau_l", "tau_l", float),
        ("tau_r", "tau_r", float),
        ("finite_n", "finite_n", int),
        ("acquisition_k", "acquisition_k", int),
        ("obs_noise_std", "obs_noise_std", float),
        ("grid_n", "grid_n", int),
        ("time_n", "time_n", int),
        ("gh_order", "gh_order", int),
        ("particles", "particles", int),
        ("rk4_substeps", "rk4_substeps_per_time_interval", int),
        ("kde_bandwidth", "kde_bandwidth", float),
        ("feasibility_margin", "feasibility_margin", float),
        ("calibration_steps", "calibration_steps", int),
        ("lambda_clip", "lambda_clip", float),
        ("newton_step_cap", "newton_step_cap", float),
        ("lbfgs_maxiter", "calibration_lbfgs_maxiter", int),
        ("calibration_max_retries", "calibration_max_retries", int),
        ("max_population_calibration_resid", "max_population_calibration_resid", float),
        ("max_finite_calibration_resid", "max_finite_calibration_resid", float),
        ("min_ess_fraction", "min_ess_fraction", float),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = cast(value)
    cfg = dataclasses.replace(cfg, **overrides)

    if cfg.action_trials < 1 or cfg.action_trials > cfg.law_trials:
        raise ValueError("Require 1 <= action_trials <= law_trials")
    if cfg.validation_trials < 1:
        raise ValueError("validation_trials must be >= 1")
    if cfg.tau_l < 0.0 or cfg.tau_r < 0.0:
        raise ValueError("tau_l and tau_r must be nonnegative")

    # Scientific model grid/time. D.8 intentionally inherits the D.7/D.6
    # reference regime rather than the Stage-B preset's original 39x39 grid.
    if args.preset == "quick":
        base_b = backend.preset_config("quick")
    else:
        base_b = backend.preset_config("reference")
    stage_b_cfg = dataclasses.replace(
        base_b,
        grid_n=int(cfg.grid_n),
        time_n=int(cfg.time_n),
    )
    model = backend.StageB(stage_b_cfg)

    if cfg.acquisition_k < 3 or cfg.acquisition_k >= model.cfg.time_n:
        raise ValueError("K must be >=3 and < time_n")

    d7_cfg = d7_compatible_config(cfg, d7)
    evaluator = d7.D7Evaluator(model, d2, d5, params, checkpoint_meta, d7_cfg)
    measurement_cov = d3.MeasurementCovariance(model)

    frozen = {
        "lift": np.radians(np.asarray(cfg.lift_design_deg, dtype=np.float64)),
        "tangent": np.radians(np.asarray(cfg.tangent_design_deg, dtype=np.float64)),
        "full": np.radians(np.asarray(cfg.full_design_deg, dtype=np.float64)),
    }
    candidates = oracle_candidate_designs(
        int(cfg.angle_n),
        float(model.cfg.min_sep_deg),
        frozen,
    )

    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in acq_set and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    if heldout_idx.size == 0:
        raise ValueError("Acquisition set leaves no held-out interior times")
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    # Disjoint selection and validation CRN banks.
    total_trials = int(cfg.law_trials + cfg.validation_trials)
    trial_bank = []
    for trial in range(total_trials):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 8800, trial]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        trial_bank.append(
            c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng)
        )
    law_bank = trial_bank[: int(cfg.law_trials)]
    action_bank = law_bank[: int(cfg.action_trials)]
    validation_bank = trial_bank[int(cfg.law_trials):]

    print("=" * 116)
    print("Stage D.8 — endpoint-trained FM finite-resource sensor-design oracle")
    print("=" * 116)
    print(f"Backend       : {Path(backend_path).resolve()}")
    print(f"D5 checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"D7 evaluator  : {Path(d7_path).resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"D5 bank       : {cfg.bank_mode}, GH order={cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Finite cond.  : N={cfg.finite_n}, K={cfg.acquisition_k}, noise={cfg.obs_noise_std:.4f}")
    print(
        f"Oracle        : angle_n={cfg.angle_n}, candidates={len(candidates)}, "
        f"min_sep={model.cfg.min_sep_deg:.1f} deg"
    )
    print(
        f"Trials        : law={cfg.law_trials}, action={cfg.action_trials}, "
        f"independent validation={cfg.validation_trials}"
    )
    print(f"Tolerances    : tau_L={100*cfg.tau_l:.2f}%, tau_R={100*cfg.tau_r:.2f}%")
    print(
        f"Numerics      : population cal<={cfg.max_population_calibration_resid:.1e}, "
        f"finite cal<={cfg.max_finite_calibration_resid:.1e}, "
        f"D2-relative ESS>={cfg.min_ess_fraction:.3f}"
    )
    print(f"Held-out idx  : {heldout_idx.tolist()}")

    # ------------------------------------------------------------------
    # Stage 1: population-law screen. No finite measurements, no Poisson.
    # ------------------------------------------------------------------
    print("\n[1/3] Endpoint-trained-FM population-law screen...", flush=True)
    valid_population: List[Dict[str, Any]] = []
    for j, cand in enumerate(candidates):
        try:
            pop = population_endpoint_fm_law(
                model=model,
                evaluator=evaluator,
                d2=d2,
                d7=d7,
                eta=np.asarray(cand["eta"], dtype=np.float64),
                cfg=cfg,
            )
            cand["population_law"] = float(pop["lift_mmd2"])
            cand["population_raw_law"] = float(pop["raw_lift_mmd2"])
            cand["population_min_ess"] = float(pop["min_ess_fraction"])
            cand["population_max_cal"] = float(pop["max_calibration_residual"])
            cand["population_max_abs_lambda_coordinate"] = float(pop["max_abs_lambda_coordinate"])
            cand["population_max_clip_fraction"] = float(pop["max_clip_fraction"])
            cand["population_max_retry_count"] = int(pop["max_retry_count"])
            cand["population_numerically_valid"] = bool(pop["scientifically_valid"])
            cand["population_diagnostics"] = pop
        except Exception as exc:
            cand["population_law"] = float("inf")
            cand["population_raw_law"] = float("inf")
            cand["population_min_ess"] = 0.0
            cand["population_max_cal"] = float("inf")
            cand["population_numerically_valid"] = False
            cand["population_failure"] = repr(exc)

        if cand["population_numerically_valid"] and np.isfinite(cand["population_law"]):
            valid_population.append(cand)

        if (j + 1) % max(1, len(candidates) // 10) == 0 or j + 1 == len(candidates):
            print(
                f"  population {j+1}/{len(candidates)} | valid={len(valid_population)}",
                flush=True,
            )

    if not valid_population:
        raise RuntimeError("No numerically valid candidate in D.8 population-law screen")

    lstar = min(float(c["population_law"]) for c in valid_population)
    lmax = (1.0 + float(cfg.tau_l)) * lstar
    pop_feasible = [
        c for c in valid_population if float(c["population_law"]) <= lmax
    ]
    population_best = min(valid_population, key=lambda c: float(c["population_law"]))
    print(
        f"  population valid={len(valid_population)}/{len(candidates)} | "
        f"law-feasible={len(pop_feasible)} | L_D5*={lstar:.8e} | "
        f"threshold={lmax:.8e}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: finite-resource law screen with CRN. No Poisson.
    # ------------------------------------------------------------------
    print("\n[2/3] Endpoint-trained-FM finite-resource law screen...", flush=True)
    finite_valid: List[Dict[str, Any]] = []

    for j, cand in enumerate(pop_feasible):
        eta = np.asarray(cand["eta"], dtype=np.float64)
        try:
            constraints = d7.build_joint_beta_constraints(
                model=model,
                evaluator=evaluator,
                d2=d2,
                c2=c2,
                eta=eta,
                margin=float(cfg.feasibility_margin),
            )
            cand["joint_feasibility"] = {
                "constraint_count": int(constraints["A"].shape[0]),
                "physical_hull_metadata": constraints["physical_hull_metadata"],
                "time_metadata": constraints["time_metadata"],
            }
            cand["_constraints"] = constraints

            law_rows = []
            all_valid = True
            for shared in law_bank:
                row, _ = d7.evaluate_design_trial(
                    model=model,
                    evaluator=evaluator,
                    d2=d2,
                    d3=d3,
                    c2=c2,
                    measurement_cov=measurement_cov,
                    eta=eta,
                    shared=shared,
                    acq_idx=acq_idx,
                    heldout_mask=heldout_mask,
                    cfg=d7_cfg,
                    joint_constraints=constraints,
                    compute_action=False,
                )
                law_rows.append(row)
                all_valid = all_valid and bool(row["finite_valid"]) and bool(row["exact_valid"])

            cand["law_rows"] = law_rows
            cand["finite_risk"] = mean_se([r["finite_heldout_mmd2"] for r in law_rows])
            cand["finite_degradation"] = mean_se([r["measurement_delta_mmd2"] for r in law_rows])
            cand["finite_min_ess"] = float(min(r["finite_min_ess"] for r in law_rows))
            cand["finite_max_cal"] = float(max(r["finite_max_calibration_resid"] for r in law_rows))
            cand["finite_max_abs_lambda_coordinate"] = float(
                max(r["finite_max_abs_lambda_coordinate"] for r in law_rows)
            )
            cand["finite_max_clip_fraction"] = float(
                max(r["finite_max_clip_fraction"] for r in law_rows)
            )
            cand["finite_retry_case_count"] = int(
                sum(int(r["finite_retry_case_count"]) for r in law_rows)
            )
            cand["projection_rate"] = float(
                np.mean([r["feasibility_projection_active"] for r in law_rows])
            )
            cand["max_projection_norm"] = float(
                max(r["feasibility_projection_norm"] for r in law_rows)
            )
            cand["finite_numerically_valid"] = bool(
                all_valid
                and np.isfinite(cand["finite_risk"]["mean"])
                and cand["finite_min_ess"] >= cfg.min_ess_fraction
                and cand["finite_max_cal"] <= cfg.max_finite_calibration_resid
            )
        except Exception as exc:
            cand["finite_numerically_valid"] = False
            cand["finite_failure"] = repr(exc)
            cand["finite_risk"] = {"mean": float("inf"), "se": float("nan"), "n": 0}
            cand["finite_degradation"] = {"mean": float("nan"), "se": float("nan"), "n": 0}
            cand["finite_min_ess"] = 0.0
            cand["finite_max_cal"] = float("inf")

        if cand["finite_numerically_valid"]:
            finite_valid.append(cand)

        if (j + 1) % max(1, len(pop_feasible) // 10) == 0 or j + 1 == len(pop_feasible):
            print(
                f"  finite law {j+1}/{len(pop_feasible)} | valid={len(finite_valid)}",
                flush=True,
            )

    if not finite_valid:
        raise RuntimeError(
            "No numerically valid population-feasible candidate under finite measurements"
        )

    rstar = min(float(c["finite_risk"]["mean"]) for c in finite_valid)
    rmax = (1.0 + float(cfg.tau_r)) * rstar
    finite_feasible = [
        c for c in finite_valid if float(c["finite_risk"]["mean"]) <= rmax
    ]
    finite_best = min(finite_valid, key=lambda c: float(c["finite_risk"]["mean"]))
    print(
        f"  finite valid={len(finite_valid)}/{len(pop_feasible)} | "
        f"risk-feasible={len(finite_feasible)} | R_D5,N*={rstar:.8e} | "
        f"threshold={rmax:.8e}",
        flush=True,
    )

    if not finite_feasible:
        raise RuntimeError("No finite-risk-feasible D.8 candidate")

    # ------------------------------------------------------------------
    # Stage 3: finite action ONLY on survivors of both law screens.
    # ------------------------------------------------------------------
    print("\n[3/3] Endpoint-trained-FM finite-resource action screen...", flush=True)
    action_valid: List[Dict[str, Any]] = []

    for j, cand in enumerate(finite_feasible):
        eta = np.asarray(cand["eta"], dtype=np.float64)
        constraints = cand.get("_constraints")
        if constraints is None:
            constraints = d7.build_joint_beta_constraints(
                model=model,
                evaluator=evaluator,
                d2=d2,
                c2=c2,
                eta=eta,
                margin=float(cfg.feasibility_margin),
            )

        rows = []
        all_valid = True
        for shared in action_bank:
            row, _ = d7.evaluate_design_trial(
                model=model,
                evaluator=evaluator,
                d2=d2,
                d3=d3,
                c2=c2,
                measurement_cov=measurement_cov,
                eta=eta,
                shared=shared,
                acq_idx=acq_idx,
                heldout_mask=heldout_mask,
                cfg=d7_cfg,
                joint_constraints=constraints,
                compute_action=True,
            )
            rows.append(row)
            all_valid = (
                all_valid
                and bool(row["finite_valid"])
                and bool(row["exact_valid"])
                and np.isfinite(float(row["finite_action"]))
            )

        cand["action_rows"] = rows
        cand["finite_action"] = mean_se([r["finite_action"] for r in rows])
        cand["matched_exact_action"] = mean_se([r["exact_action"] for r in rows])
        cand["action_inflation"] = mean_se([r["measurement_action_inflation"] for r in rows])
        cand["action_numerically_valid"] = bool(
            all_valid and np.isfinite(cand["finite_action"]["mean"])
        )

        if cand["action_numerically_valid"]:
            action_valid.append(cand)

        print(
            f"  action {j+1}/{len(finite_feasible)} | eta={np.round(cand['theta_deg'], 3).tolist()} | "
            f"A={cand['finite_action']['mean']:.6e} | valid={cand['action_numerically_valid']}",
            flush=True,
        )

    if not action_valid:
        raise RuntimeError("No numerically valid finite-action candidate after D.8 law screens")

    robust = min(action_valid, key=lambda c: float(c["finite_action"]["mean"]))

    # Exact frozen candidate rows, if explicitly present.
    frozen_candidate_rows: Dict[str, Any] = {}
    for name, eta in frozen.items():
        hit = min(candidates, key=lambda c: unordered_design_distance(c["eta"], eta))
        if unordered_design_distance(hit["eta"], eta) < 1.0e-10:
            frozen_candidate_rows[name] = hit

    # ------------------------------------------------------------------
    # Independent post-selection validation.
    # ------------------------------------------------------------------
    validation_designs: Dict[str, np.ndarray] = {
        "robust_d8": np.asarray(robust["eta"], dtype=np.float64),
        "lift": frozen["lift"],
        "tangent": frozen["tangent"],
        "full": frozen["full"],
    }

    # Deduplicate computation when the selected design equals a frozen control.
    unique_validation: Dict[Tuple[float, float], Dict[str, Any]] = {}
    alias_to_key: Dict[str, Tuple[float, float]] = {}
    for name, eta in validation_designs.items():
        key = tuple(np.round(canonical_eta(eta), 12))
        alias_to_key[name] = key
        unique_validation.setdefault(
            key,
            {"eta": canonical_eta(eta), "aliases": []},
        )["aliases"].append(name)

    validation_unique: Dict[Tuple[float, float], Dict[str, Any]] = {}
    validation_trial_rows_unique: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}

    print("\nIndependent post-selection validation...", flush=True)
    for key, spec in unique_validation.items():
        eta = np.asarray(spec["eta"], dtype=np.float64)
        constraints = d7.build_joint_beta_constraints(
            model=model,
            evaluator=evaluator,
            d2=d2,
            c2=c2,
            eta=eta,
            margin=float(cfg.feasibility_margin),
        )

        rows = []
        for shared in validation_bank:
            row, _ = d7.evaluate_design_trial(
                model=model,
                evaluator=evaluator,
                d2=d2,
                d3=d3,
                c2=c2,
                measurement_cov=measurement_cov,
                eta=eta,
                shared=shared,
                acq_idx=acq_idx,
                heldout_mask=heldout_mask,
                cfg=d7_cfg,
                joint_constraints=constraints,
                compute_action=True,
            )
            rows.append(row)

        summary = summarize_selection_rows(rows)
        validation_trial_rows_unique[key] = rows
        validation_unique[key] = {
            "theta_deg": np.degrees(eta).tolist(),
            "aliases": list(spec["aliases"]),
            "summary": summary,
        }

        print(
            f"  {','.join(spec['aliases']):18s} eta={np.degrees(eta).round(3).tolist()} | "
            f"valid={100*summary['finite_valid_fraction']:.1f}% | "
            f"R={summary['finite_heldout_mmd2']['mean']:.8e} | "
            f"A={summary['finite_action']['mean']:.6e}",
            flush=True,
        )

    validation = {
        name: validation_unique[alias_to_key[name]]
        for name in validation_designs
    }
    validation_rows = {
        name: validation_trial_rows_unique[alias_to_key[name]]
        for name in validation_designs
    }

    validation_comparisons = {
        "robust_vs_lift_finite_law": paired_difference(
            validation_rows["robust_d8"], validation_rows["lift"], "finite_heldout_mmd2"
        ),
        "robust_vs_full_finite_law": paired_difference(
            validation_rows["robust_d8"], validation_rows["full"], "finite_heldout_mmd2"
        ),
        "robust_vs_tangent_finite_law": paired_difference(
            validation_rows["robust_d8"], validation_rows["tangent"], "finite_heldout_mmd2"
        ),
        "robust_vs_lift_measurement_degradation": paired_difference(
            validation_rows["robust_d8"], validation_rows["lift"], "measurement_delta_mmd2"
        ),
        "robust_vs_full_measurement_degradation": paired_difference(
            validation_rows["robust_d8"], validation_rows["full"], "measurement_delta_mmd2"
        ),
        "robust_vs_lift_action_reduction": paired_reduction(
            validation_rows["robust_d8"], validation_rows["lift"], "finite_action"
        ),
        "robust_vs_full_action_reduction": paired_reduction(
            validation_rows["robust_d8"], validation_rows["full"], "finite_action"
        ),
        "robust_vs_tangent_action_reduction": paired_reduction(
            validation_rows["robust_d8"], validation_rows["tangent"], "finite_action"
        ),
        "full_vs_lift_finite_law": paired_difference(
            validation_rows["full"], validation_rows["lift"], "finite_heldout_mmd2"
        ),
        "full_vs_lift_measurement_degradation": paired_difference(
            validation_rows["full"], validation_rows["lift"], "measurement_delta_mmd2"
        ),
        "full_vs_lift_action_reduction": paired_reduction(
            validation_rows["full"], validation_rows["lift"], "finite_action"
        ),
    }

    all_validation_valid = all(
        validation[name]["summary"]["finite_valid_fraction"] == 1.0
        and validation[name]["summary"]["exact_valid_fraction"] == 1.0
        for name in validation_designs
    )

    payload = {
        "stage": "D.8",
        "purpose": (
            "Re-optimize the physically admissible two-sensor design under the "
            "teacher-free D.5 endpoint-trained FM particle reference and finite/noisy measurements."
        ),
        "method": (
            "D.5 population-law screen -> D.7 finite-law CRN screen -> minimum D.7 finite "
            "weighted-Poisson action -> disjoint post-selection validation."
        ),
        "backend_path": str(Path(backend_path).resolve()),
        "c2_script_path": str(Path(c2_path).resolve()),
        "d2_script_path": str(Path(d2_path).resolve()),
        "d3_script_path": str(Path(d3_path).resolve()),
        "d5_script_path": str(Path(d5_path).resolve()),
        "d7_script_path": str(Path(d7_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "config": jsonify(cfg),
        "d7_config": jsonify(d7_cfg),
        "stage_b_config": jsonify(stage_b_cfg),
        "condition": {
            "N": int(cfg.finite_n),
            "K": int(cfg.acquisition_k),
            "obs_noise_std": float(cfg.obs_noise_std),
            "angle_n": int(cfg.angle_n),
            "candidate_count": int(len(candidates)),
            "law_trials": int(cfg.law_trials),
            "action_trials": int(cfg.action_trials),
            "validation_trials": int(cfg.validation_trials),
            "tau_L": float(cfg.tau_l),
            "tau_R": float(cfg.tau_r),
            "min_sep_deg": float(model.cfg.min_sep_deg),
            "acquisition_indices": acq_idx.tolist(),
            "heldout_indices": heldout_idx.tolist(),
        },
        "population_law_star": float(lstar),
        "population_law_max": float(lmax),
        "finite_risk_star": float(rstar),
        "finite_risk_max": float(rmax),
        "population_numerically_valid_count": int(len(valid_population)),
        "population_feasible_count": int(len(pop_feasible)),
        "finite_numerically_valid_count": int(len(finite_valid)),
        "finite_risk_feasible_count": int(len(finite_feasible)),
        "finite_action_valid_count": int(len(action_valid)),
        "population_best": public_candidate(population_best),
        "finite_law_best": public_candidate(finite_best),
        "robust_d8": public_candidate(robust),
        "robust_d8_eta_rad": jsonify(robust["eta"]),
        "frozen_candidate_rows": {
            k: public_candidate(v) for k, v in frozen_candidate_rows.items()
        },
        "finite_risk_feasible_rows": [public_candidate(c) for c in finite_feasible],
        "population_feasible_rows": [public_candidate(c) for c in pop_feasible],
        "all_candidates_population": [
            {
                "theta_deg": c["theta_deg"],
                "sensor_separation_deg": c["sensor_separation_deg"],
                "sources": c["sources"],
                "population_law": c.get("population_law", float("inf")),
                "population_raw_law": c.get("population_raw_law", float("inf")),
                "population_min_ess": c.get("population_min_ess", 0.0),
                "population_max_cal": c.get("population_max_cal", float("inf")),
                "population_numerically_valid": c.get("population_numerically_valid", False),
                "population_failure": c.get("population_failure"),
            }
            for c in candidates
        ],
        "independent_validation": validation,
        "independent_validation_comparisons": validation_comparisons,
        "validation_trial_rows": validation_rows,
        "checks": {
            "checkpoint_teacher_free": bool(
                checkpoint_meta["bridge"].get("uses_analytic_A_t") is False
                and checkpoint_meta["bridge"].get("uses_analytic_B_t") is False
                and checkpoint_meta["bridge"].get("uses_analytic_velocity_teacher") is False
            ),
            "no_cnf_used": True,
            "no_old_analytic_reference_branch": True,
            "ess_convention_matches_d2": True,
            "selection_and_validation_banks_disjoint": True,
            "frozen_controls_explicitly_inserted": True,
            "population_screen_nonempty": bool(len(pop_feasible) > 0),
            "finite_law_screen_nonempty": bool(len(finite_feasible) > 0),
            "finite_action_screen_nonempty": bool(len(action_valid) > 0),
            "all_validation_rows_scientifically_valid": bool(all_validation_valid),
        },
        "interpretation": [
            "D.8 is the endpoint-trained-reference analogue of D.4: the two sensor angles move again only after D.5-D.7 froze and validated the new reference/finite-measurement layers.",
            "The design objective is lexicographic: population law sufficiency, then finite-resource law sufficiency, then minimum finite weighted-Poisson action.",
            "No scalar weighted sum of law and action is used.",
            "All selection candidates share identical scientific alpha values, microscopic observations and detector-normal draws through common random numbers.",
            "The validation bank is disjoint from every selection trial and is touched only after robust_d8 is selected.",
            "The finite reconstructed target curve is constrained once to the physical/D.5-particle feasible intersection; there is no per-time or branch-specific clipping after reconstruction.",
            "Hard calibration and D.2-relative ESS are numerical/reference-overlap gates, not design objectives.",
            "The oracle is discrete at angle_n resolution and is not a proof of the exact continuous-angle optimum.",
            "Lift, Tangent-TC and Full-TC are explicitly inserted into the candidate set and independent validation controls.",
            "Absolute weighted-Poisson action remains a particle/grid quantity; the main D.8 claim should be based on matched within-run design contrasts and independent validation.",
        ],
        "wall_seconds": float(time.time() - wall0),
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    print("\n" + "=" * 116)
    print("Stage D.8 oracle result")
    print("=" * 116)
    print(
        f"Population best : {population_best['theta_deg']} | "
        f"L_D5={population_best['population_law']:.8e}"
    )
    print(
        f"Finite-law best : {finite_best['theta_deg']} | "
        f"R_D5,N={finite_best['finite_risk']['mean']:.8e} +/- "
        f"{finite_best['finite_risk']['se']:.2e}"
    )
    print(
        f"Robust D8       : {robust['theta_deg']} | "
        f"L_D5={robust['population_law']:.8e} | "
        f"R_D5,N={robust['finite_risk']['mean']:.8e} +/- "
        f"{robust['finite_risk']['se']:.2e} | "
        f"A_D5,N={robust['finite_action']['mean']:.6e} +/- "
        f"{robust['finite_action']['se']:.2e}"
    )

    rv = validation["robust_d8"]["summary"]
    lv = validation["lift"]["summary"]
    tv = validation["tangent"]["summary"]
    fv = validation["full"]["summary"]
    print("Independent validation:")
    for label, vv in (
        ("Robust D8", rv),
        ("Lift", lv),
        ("Tangent-TC", tv),
        ("Full-TC", fv),
    ):
        print(
            f"  {label:10s}: valid={100*vv['finite_valid_fraction']:.1f}% | "
            f"R={vv['finite_heldout_mmd2']['mean']:.8e} +/- "
            f"{vv['finite_heldout_mmd2']['se']:.2e} | "
            f"A={vv['finite_action']['mean']:.6e} +/- {vv['finite_action']['se']:.2e}"
        )

    comp = validation_comparisons["robust_vs_lift_action_reduction"]
    print(
        f"  Robust-vs-Lift action reduction: ratio-of-means="
        f"{100*comp['ratio_of_means_reduction']:.2f}% | "
        f"paired={100*comp['mean_paired_reduction']:.2f}% +/- "
        f"{100*comp['se_paired_reduction']:.2f}%"
    )
    compf = validation_comparisons["robust_vs_full_action_reduction"]
    print(
        f"  Robust-vs-Full action reduction: ratio-of-means="
        f"{100*compf['ratio_of_means_reduction']:.2f}% | "
        f"paired={100*compf['mean_paired_reduction']:.2f}% +/- "
        f"{100*compf['se_paired_reduction']:.2f}%"
    )
    print("=" * 116)

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(jsonify(payload), indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved D.8 results: {out_path.resolve()}")


if __name__ == "__main__":
    main()
