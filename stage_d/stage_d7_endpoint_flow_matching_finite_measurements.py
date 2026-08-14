#!/usr/bin/env python3
"""
Stage D.7: finite/noisy measurements under an endpoint-trained flow-matching
particle reference (NO CNF, NO old analytic SI).

Scientific purpose
------------------
D.5 trained the reference velocity from endpoint samples rather than from the old
Stage-B analytic path teacher. D.6 then showed that, under exact population
moments, the frozen Lift / Tangent-TC / Full-TC designs admit valid hard empirical
MFSI projections under that endpoint-trained reference.

D.7 adds one approximation layer only:

    finite/noisy population measurements.

The sensor designs remain frozen. There is no sensor re-optimization in D.7.

For each common-random-number scientific trial and each frozen design:

  1. draw N microscopic observations from the Stage-B scientific population,
  2. evaluate the K acquisition-time sensor means and optional detector noise,
  3. fit the same endpoint-anchored GLS quadratic moment trajectory used in
     Stage C / D.3,
  4. if necessary, project the quadratic coefficient onto ONE precomputed common
     feasible polytope lying inside
          (a) the physical sensor-moment hull and
          (b) the D.5 learned-particle moment hull at every evaluation time,
  5. run hard empirical MFSI on the D.5 rollout bank using that reconstructed
     moment curve,
  6. evaluate held-out law MMD^2 and, on a predeclared subset of trials, full
     weighted-Poisson action,
  7. compare finite-measurement performance against the exact-population-moment
     result for the SAME scientific trial and SAME D.5 reference bank.

Important controls
------------------
* The noisy reconstructed target curve enters exactly once.
* There is NO branch-specific clipping or post-hoc target rescue.
* The hard empirical I-projection uses a robust convex-dual solver:
    - damped Newton,
    - exact-gradient L-BFGS-B fallback,
    - adaptive coordinate-ceiling retries only when required.
* A finite row is scientifically valid only when:
    - every target is inside the D.5 empirical moment hull,
    - hard calibration residual is below the declared tolerance,
    - relative ESS is above the declared overlap gate,
    - reference base mass remains in-domain.
* Invalid rows are recorded as diagnostics and excluded from scientific summaries.
* No CNF density, score model, log-Jacobian integration, learned likelihood, old
  Stage-B A_t/B_t reference path, or analytic-reference particle branch is used.

Recommended reference run
-------------------------
python stage_d7_endpoint_flow_matching_finite_measurements.py \
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \
    --c2-script ../stage_c/stage_c2_mfsi_matched_action.py \
    --d2-script stage_d2_flow_matching_particle_mfsi.py \
    --d3-script stage_d3_flow_matching_finite_measurements.py \
    --d5-script stage_d5_endpoint_flow_matching_reference_v2.py \
    --checkpoint stage_d5_endpoint_flow_matching_reference_v2.npz \
    --preset reference \
    --output stage_d7_endpoint_flow_matching_finite_measurements.json
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
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import logsumexp

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
class D7Config:
    preset: str = "quick"
    seed: int = 20260813

    # Finite-resource condition.
    trials: int = 4
    action_trials: int = 2
    finite_n: int = 100
    acquisition_k: int = 7
    obs_noise_std: float = 0.01

    # Stage-B / particle discretization.
    grid_n: int = 19
    time_n: int = 13
    bank_mode: str = "gauss-hermite"
    gh_order: int = 20
    particles: int = 8192
    rk4_substeps_per_time_interval: int = 8
    kde_bandwidth: float = 0.0
    kde_truncate: float = 4.0

    # Measurement-curve fit.
    variance_floor: float = 1.0e-10
    quadratic_ridge_rel: float = 1.0e-12
    feasibility_margin: float = 0.0

    # Robust hard empirical I-projection.
    calibration_steps: int = 80
    calibration_tol: float = 2.0e-8
    calibration_accept_tol: float = 2.0e-6
    newton_step_cap: float = 10.0
    lambda_clip: float = 300.0
    calibration_lbfgs_maxiter: int = 400
    calibration_max_retries: int = 2
    calibration_retry_clip_multiplier: float = 2.0
    clip_saturation_fraction: float = 0.995

    # Numerical validity gates.
    min_ess_fraction: float = 0.03
    min_in_domain_base_mass: float = 0.995

    # Frozen historical designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)


def preset_d7_config(name: str) -> D7Config:
    if name == "quick":
        return D7Config()
    if name == "reference":
        return D7Config(
            preset="reference",
            trials=24,
            action_trials=8,
            finite_n=100,
            acquisition_k=11,
            obs_noise_std=0.01,
            grid_n=51,
            time_n=21,
            gh_order=36,
            particles=32768,
            rk4_substeps_per_time_interval=16,
            calibration_steps=300,
            calibration_tol=1.0e-9,
            calibration_accept_tol=2.0e-6,
            newton_step_cap=20.0,
            lambda_clip=1000.0,
            calibration_lbfgs_maxiter=800,
            calibration_max_retries=2,
        )
    if name == "confirm":
        return D7Config(
            preset="confirm",
            trials=50,
            action_trials=12,
            finite_n=100,
            acquisition_k=11,
            obs_noise_std=0.01,
            grid_n=65,
            time_n=27,
            gh_order=48,
            particles=65536,
            rk4_substeps_per_time_interval=24,
            calibration_steps=500,
            calibration_tol=5.0e-10,
            calibration_accept_tol=1.0e-6,
            newton_step_cap=25.0,
            lambda_clip=2000.0,
            calibration_lbfgs_maxiter=1200,
            calibration_max_retries=2,
        )
    raise ValueError(name)


def d2_compatible_config(cfg: D7Config, d2):
    return d2.D2Config(
        preset=cfg.preset,
        seed=cfg.seed,
        lift_design_deg=cfg.lift_design_deg,
        tangent_design_deg=cfg.tangent_design_deg,
        full_design_deg=cfg.full_design_deg,
        bank_mode=cfg.bank_mode,
        gh_order=cfg.gh_order,
        particles=cfg.particles,
        rk4_substeps_per_time_interval=cfg.rk4_substeps_per_time_interval,
        calibration_steps=cfg.calibration_steps,
        calibration_tol=cfg.calibration_tol,
        newton_step_cap=cfg.newton_step_cap,
        lambda_clip=cfg.lambda_clip,
        kde_bandwidth=cfg.kde_bandwidth,
        kde_truncate=cfg.kde_truncate,
    )


# -----------------------------------------------------------------------------
# Generic helpers
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


def mean_se(values: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"mean": float(np.mean(x)), "se": se, "n": int(len(x))}


def paired_difference(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], key: str):
    a = np.asarray([r.get(key, np.nan) for r in rows_a], dtype=np.float64)
    b = np.asarray([r.get(key, np.nan) for r in rows_b], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    s = mean_se(d)
    return {
        "mean_difference_a_minus_b": s["mean"],
        "se_difference": s["se"],
        "n_pairs": s["n"],
    }


def ratio_of_means_reduction(rows_num: List[Dict[str, Any]], rows_den: List[Dict[str, Any]], key: str):
    num = np.asarray([r.get(key, np.nan) for r in rows_num], dtype=np.float64)
    den = np.asarray([r.get(key, np.nan) for r in rows_den], dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den)
    if not np.any(mask):
        return {"ratio_of_means_reduction": float("nan"), "n_pairs": 0}
    mn = float(np.mean(num[mask]))
    md = float(np.mean(den[mask]))
    return {
        "ratio_of_means_reduction": float(1.0 - mn / md) if abs(md) > 1e-14 else float("nan"),
        "mean_numerator": mn,
        "mean_denominator": md,
        "n_pairs": int(mask.sum()),
    }


def trap_average(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    ww = weights[mask]
    vv = values[mask]
    if len(vv) == 0 or float(np.sum(ww)) <= 0.0:
        return float("nan")
    return float(np.sum(ww * vv) / np.sum(ww))


# -----------------------------------------------------------------------------
# Empirical moment hull
# -----------------------------------------------------------------------------


def hull_equations_from_points(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=14), axis=0)
    if pts.shape[0] < 3 or np.linalg.matrix_rank(pts - np.mean(pts, axis=0)) < 2:
        raise RuntimeError("Particle moment set is rank-deficient.")
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise RuntimeError(f"Could not construct particle moment hull: {exc}") from exc
    return np.asarray(hull.equations, dtype=np.float64)


def target_hull_diagnostic(phi: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    eq = hull_equations_from_points(phi)
    vals = eq[:, :2] @ np.asarray(target, dtype=np.float64) + eq[:, 2]
    mv = float(np.max(vals))
    return {
        "inside_closed_hull": bool(mv <= 1.0e-12),
        "max_facet_violation": mv,
        "interior_margin": float(-mv),
        "facets": int(eq.shape[0]),
    }


# -----------------------------------------------------------------------------
# Robust empirical I-projection
# -----------------------------------------------------------------------------


def _tilt_state(phi: np.ndarray, base_w: np.ndarray, target: np.ndarray, lam: np.ndarray):
    phi = np.asarray(phi, dtype=np.float64)
    base_w = np.asarray(base_w, dtype=np.float64)
    base_w = base_w / np.sum(base_w)
    target = np.asarray(target, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)

    log_base = np.log(np.maximum(base_w, 1.0e-300))
    logits = log_base + phi @ lam
    z = logsumexp(logits)
    w = np.exp(logits - z)
    moment = np.sum(w[:, None] * phi, axis=0)
    centered = phi - moment[None, :]
    C = centered.T @ (w[:, None] * centered)
    F = moment - target
    dual = float(z - np.dot(lam, target))
    return w, moment, C, F, dual


def robust_empirical_tilt(
    phi: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    ridge: float,
    cfg: D7Config,
    lam0: np.ndarray | None = None,
):
    m = phi.shape[1]
    start = np.zeros(m, dtype=np.float64) if lam0 is None else np.asarray(lam0, dtype=np.float64).copy()
    attempts = []
    best = None

    for retry in range(int(cfg.calibration_max_retries) + 1):
        clip = float(cfg.lambda_clip) * (float(cfg.calibration_retry_clip_multiplier) ** retry)
        lam = np.clip(start, -clip, clip)
        newton_iters = 0

        for it in range(int(cfg.calibration_steps)):
            newton_iters = it + 1
            w, moment, C, F, cur = _tilt_state(phi, base_w, target, lam)
            resid = float(np.linalg.norm(F))
            if resid <= cfg.calibration_tol:
                break

            H = C + float(ridge) * np.eye(m)
            try:
                step = np.linalg.solve(H, F)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H, rcond=1.0e-12) @ F

            sn = float(np.linalg.norm(step))
            if sn > cfg.newton_step_cap:
                step *= float(cfg.newton_step_cap) / max(sn, 1.0e-30)

            accepted = False
            for scale in 0.5 ** np.arange(14, dtype=np.float64):
                cand = np.clip(lam - scale * step, -clip, clip)
                *_, val = _tilt_state(phi, base_w, target, cand)
                if np.isfinite(val) and val <= cur + 1.0e-14:
                    lam = cand
                    accepted = True
                    break
            if not accepted:
                break

        # Convex-dual L-BFGS-B fallback with exact gradient.
        def obj_grad(ll):
            w, moment, C, F, dual = _tilt_state(phi, base_w, target, ll)
            return dual, F

        w, moment, C, F, dual = _tilt_state(phi, base_w, target, lam)
        resid_before_lbfgs = float(np.linalg.norm(F))
        lbfgs_used = resid_before_lbfgs > cfg.calibration_tol

        if lbfgs_used:
            sol = minimize(
                fun=lambda ll: obj_grad(ll)[0],
                x0=lam,
                jac=lambda ll: obj_grad(ll)[1],
                method="L-BFGS-B",
                bounds=[(-clip, clip)] * m,
                options={
                    "maxiter": int(cfg.calibration_lbfgs_maxiter),
                    "ftol": 1.0e-15,
                    "gtol": min(float(cfg.calibration_tol), 1.0e-10),
                    "maxls": 50,
                },
            )
            if np.all(np.isfinite(sol.x)):
                lam = np.asarray(sol.x, dtype=np.float64)

        w, moment, C, F, dual = _tilt_state(phi, base_w, target, lam)
        resid = float(np.linalg.norm(F))
        max_abs_coord = float(np.max(np.abs(lam)))
        clip_fraction = max_abs_coord / max(clip, 1.0e-300)
        eig = np.linalg.eigvalsh(0.5 * (C + C.T))
        min_eig = float(np.min(eig))
        max_eig = float(np.max(eig))
        cond = float(max_eig / max(min_eig, 1.0e-300))
        ess = float((np.sum(w * w / np.maximum(base_w / np.sum(base_w), 1.0e-300))) ** -1)
        # Above is not the same relative ESS convention as D2. Use exact D2-like:
        bw = base_w / np.sum(base_w)
        ratio = w / np.maximum(bw, 1.0e-300)
        ess_rel = float(1.0 / np.sum(bw * ratio * ratio))

        rec = {
            "retry": int(retry),
            "effective_coordinate_clip": clip,
            "newton_iterations": int(newton_iters),
            "lbfgs_used": bool(lbfgs_used),
            "residual": resid,
            "lambda_norm": float(np.linalg.norm(lam)),
            "max_abs_lambda_coordinate": max_abs_coord,
            "clip_fraction": clip_fraction,
            "clip_hit": bool(clip_fraction >= cfg.clip_saturation_fraction),
            "min_cov_eig": min_eig,
            "cov_condition_number": cond,
            "ess_fraction": ess_rel,
        }
        attempts.append(rec)

        if best is None or resid < best["diagnostics"]["residual"]:
            best = {
                "lambda": lam.copy(),
                "weights": w.copy(),
                "moment": moment.copy(),
                "C": C.copy(),
                "diagnostics": rec.copy(),
            }

        if resid <= cfg.calibration_accept_tol:
            break
        start = lam

    assert best is not None
    best["diagnostics"]["attempts"] = attempts
    best["diagnostics"]["retry_count"] = int(len(attempts) - 1)
    return best


def robust_particle_mfsi_state(
    phi: np.ndarray,
    grad_phi: np.ndarray,
    u: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    c_dot: np.ndarray,
    ridge: float,
    cfg: D7Config,
    lam0: np.ndarray | None = None,
):
    tilt = robust_empirical_tilt(phi, base_w, target, ridge, cfg, lam0)
    lam = tilt["lambda"]
    w = tilt["weights"]
    moment = tilt["moment"]
    C = tilt["C"]

    m = np.einsum("nmc,nc->nm", grad_phi, u)
    Em = np.sum(w[:, None] * m, axis=0)
    g = m @ lam
    Eg = float(np.dot(w, g))
    centered_phi = phi - moment[None, :]
    cov_phi_g = np.sum(w[:, None] * centered_phi * (g - Eg)[:, None], axis=0)

    H = C + float(ridge) * np.eye(C.shape[0])
    rhs = np.asarray(c_dot, dtype=np.float64) - Em - cov_phi_g
    try:
        lam_dot = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        lam_dot = np.linalg.pinv(H, rcond=1.0e-12) @ rhs

    h = centered_phi @ lam_dot + g - Eg
    h = h - float(np.dot(w, h))

    G = np.einsum("nmc,nkc,n->mk", grad_phi, grad_phi, w)
    rr = Em - np.asarray(c_dot, dtype=np.float64)
    Gs = G + float(ridge) * np.eye(G.shape[0])
    try:
        tangent_action = float(rr @ np.linalg.solve(Gs, rr))
    except np.linalg.LinAlgError:
        tangent_action = float(rr @ (np.linalg.pinv(Gs, rcond=1.0e-12) @ rr))

    diag = dict(tilt["diagnostics"])
    diag.update({
        "lambda_dot_norm": float(np.linalg.norm(lam_dot)),
        "min_tangent_gram_eig": float(np.min(np.linalg.eigvalsh(0.5 * (G + G.T)))),
        "tangent_action": tangent_action,
    })
    return {
        "lambda": lam,
        "lambda_dot": lam_dot,
        "weights": w,
        "moment": moment,
        "C": C,
        "h": h,
        "diagnostics": diag,
    }


# -----------------------------------------------------------------------------
# D5 particle reference
# -----------------------------------------------------------------------------


class D7Evaluator:
    def __init__(self, model, d2, d5, params, checkpoint_meta: Dict[str, Any], cfg: D7Config):
        self.model = model
        self.d2 = d2
        self.d5 = d5
        self.params = params
        self.checkpoint_meta = checkpoint_meta
        self.cfg = cfg
        self.d2cfg = d2_compatible_config(cfg, d2)

        if checkpoint_meta.get("stage") != "D.5":
            raise ValueError(f"Expected D.5 checkpoint, got {checkpoint_meta.get('stage')!r}")
        bridge = checkpoint_meta.get("bridge", {})
        if not (
            bridge.get("uses_analytic_A_t") is False
            and bridge.get("uses_analytic_B_t") is False
            and bridge.get("uses_analytic_velocity_teacher") is False
        ):
            raise ValueError("D.5 checkpoint is not marked teacher-free.")

        cp_phys = checkpoint_meta.get("physical_system", {})
        self.r = float(cp_phys["r"])
        self.sigma = float(cp_phys["sigma"])

        sampler = d5.EndpointSampler(
            r=self.r,
            sigma=self.sigma,
            x0_external=None,
            x1_external=None,
            validation_fraction=0.1,
            seed=cfg.seed,
        )
        x0_np, base_w = d2.initial_reference_bank(sampler, self.d2cfg)
        self.x0 = jnp.asarray(x0_np, dtype=jnp.float64)
        self.base_w = np.asarray(base_w, dtype=np.float64)

        print(
            f"Generating D.5 learned reference bank: mode={cfg.bank_mode}, N={len(base_w)}, "
            f"time_n={model.cfg.time_n}, RK4 substeps/interval={cfg.rk4_substeps_per_time_interval}",
            flush=True,
        )
        rollout_fun = jax.jit(
            lambda z: d2.rollout_learned_to_nodes(
                params,
                d5,
                z,
                int(model.cfg.time_n),
                int(cfg.rk4_substeps_per_time_interval),
            )
        )
        self.nodes = np.asarray(rollout_fun(self.x0), dtype=np.float64)

        ts = np.asarray(model.times, dtype=np.float64)
        unodes = []
        velocity_jit = jax.jit(lambda t, x: d5.velocity_mlp(params, t, x))
        for k, t in enumerate(ts):
            unodes.append(np.asarray(velocity_jit(jnp.asarray(t), jnp.asarray(self.nodes[k])), dtype=np.float64))
        self.u_nodes = np.stack(unodes, axis=0)

        self.cdot_jit = jax.jit(jax.jacfwd(model.measurement_grid, argnums=0))
        self.poisson_jit = jax.jit(model.poisson_solve)


# -----------------------------------------------------------------------------
# Common physical + D5-particle feasible quadratic bridge
# -----------------------------------------------------------------------------


def build_joint_beta_constraints(model, evaluator: D7Evaluator, d2, c2, eta: np.ndarray, margin: float):
    physical_eq, physical_meta = c2.sensor_moment_hull_equations(model, eta)
    if physical_eq is None:
        raise RuntimeError(f"Physical sensor moment hull unavailable: {physical_meta}")

    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_probe = jnp.asarray(0.5 * (model.cfg.alpha_min + model.cfg.alpha_max), dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_probe, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_probe, eta_j), dtype=np.float64)

    A_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    time_meta = []

    for kt, t in enumerate(times):
        x_all = np.asarray(evaluator.nodes[kt], dtype=np.float64)
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        phi, _ = d2.sensor_particle_fields(model, eta, x)
        particle_eq = hull_equations_from_points(phi)

        eqs = [
            ("physical", np.asarray(physical_eq, dtype=np.float64)),
            ("d5_particle", particle_eq),
        ]

        z = float(t * (1.0 - t))
        bridge = (1.0 - t) * c0 + t * c1
        max_endpoint_violation = 0.0

        for _, eq in eqs:
            normals = eq[:, :2]
            offsets = eq[:, 2]
            if abs(z) < 1.0e-14:
                viol = normals @ bridge + offsets + float(margin)
                max_endpoint_violation = max(max_endpoint_violation, float(np.max(viol)))
            else:
                A_rows.append(z * normals)
                b_rows.append(-offsets - normals @ bridge - float(margin))

        if abs(z) < 1.0e-14 and max_endpoint_violation > 2.0e-8:
            raise RuntimeError(
                f"Endpoint moment outside physical/D5 particle hull at t={t:.3f}; "
                f"violation={max_endpoint_violation:.3e}"
            )

        time_meta.append({
            "t": float(t),
            "d5_in_domain_particles": int(x.shape[0]),
            "d5_hull_facets": int(particle_eq.shape[0]),
            "max_endpoint_violation": float(max(0.0, max_endpoint_violation)),
        })

    A = np.concatenate(A_rows, axis=0) if A_rows else np.zeros((0, 2), dtype=np.float64)
    b = np.concatenate(b_rows, axis=0) if b_rows else np.zeros((0,), dtype=np.float64)

    if A.shape[0]:
        feas = linprog(
            c=np.zeros(2, dtype=np.float64),
            A_ub=A,
            b_ub=b,
            bounds=[(None, None), (None, None)],
            method="highs",
        )
        if not feas.success:
            raise RuntimeError(
                "Physical/D5-particle quadratic-beta feasibility intersection is empty."
            )
        feasible_beta = np.asarray(feas.x, dtype=np.float64)
    else:
        feasible_beta = np.zeros(2, dtype=np.float64)

    return {
        "A": A,
        "b": b,
        "c0": c0,
        "c1": c1,
        "feasible_beta": feasible_beta,
        "physical_hull_metadata": physical_meta,
        "time_metadata": time_meta,
    }


# -----------------------------------------------------------------------------
# Particle-curve evaluation
# -----------------------------------------------------------------------------


def evaluate_particle_curve(
    evaluator: D7Evaluator,
    d2,
    eta: np.ndarray,
    alpha: float,
    target_curve: np.ndarray,
    target_cdot: np.ndarray,
    heldout_mask: np.ndarray,
    compute_action: bool,
):
    model = evaluator.model
    cfg = evaluator.cfg

    bw = float(cfg.kde_bandwidth)
    if bw <= 0.0:
        bw = 0.35 * float(model.dx)

    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)

    law = np.zeros(len(times), dtype=np.float64)
    action = np.full(len(times), np.nan, dtype=np.float64)
    tangent = np.full(len(times), np.nan, dtype=np.float64)
    ess = np.zeros(len(times), dtype=np.float64)
    cal = np.zeros(len(times), dtype=np.float64)
    lam_norm = np.zeros(len(times), dtype=np.float64)
    max_lam_coord = np.zeros(len(times), dtype=np.float64)
    clip_frac = np.zeros(len(times), dtype=np.float64)
    retries = np.zeros(len(times), dtype=np.int32)
    hull_inside = np.ones(len(times), dtype=bool)
    hull_violation = np.zeros(len(times), dtype=np.float64)
    in_base_mass = np.zeros(len(times), dtype=np.float64)
    min_cov = np.zeros(len(times), dtype=np.float64)
    cov_cond = np.zeros(len(times), dtype=np.float64)
    poisson = np.full(len(times), np.nan, dtype=np.float64)
    grid_moment_error = np.zeros(len(times), dtype=np.float64)

    phi_grid, _ = model.sensor_fields(eta_j)
    phi_grid_np = np.asarray(phi_grid, dtype=np.float64)
    lam_warm = np.zeros(2, dtype=np.float64)
    worst = None

    for kt, t in enumerate(times):
        x_all = np.asarray(evaluator.nodes[kt], dtype=np.float64)
        u_all = np.asarray(evaluator.u_nodes[kt], dtype=np.float64)
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        u = u_all[mask]
        base_w = np.asarray(evaluator.base_w[mask], dtype=np.float64)
        bm = float(np.sum(base_w))
        in_base_mass[kt] = bm
        base_w /= max(bm, 1.0e-300)

        phi, grad_phi = d2.sensor_particle_fields(model, eta, x)
        target = np.asarray(target_curve[kt], dtype=np.float64)
        cdot = np.asarray(target_cdot[kt], dtype=np.float64)

        hd = target_hull_diagnostic(phi, target)
        hull_inside[kt] = bool(hd["inside_closed_hull"])
        hull_violation[kt] = float(hd["max_facet_violation"])

        if not hull_inside[kt]:
            cal[kt] = np.inf
            ess[kt] = 0.0
            continue

        st = robust_particle_mfsi_state(
            phi=phi,
            grad_phi=grad_phi,
            u=u,
            base_w=base_w,
            target=target,
            c_dot=cdot,
            ridge=float(model.cfg.newton_ridge),
            cfg=cfg,
            lam0=lam_warm,
        )
        lam_warm = np.asarray(st["lambda"], dtype=np.float64)

        q, qmass, h_grid, _ = d2.rasterize_projected_state(
            model=model,
            x=x,
            w=st["weights"],
            h=st["h"],
            bandwidth=bw,
            truncate=float(cfg.kde_truncate),
        )

        _, p_mass = model.external_q_mass(jnp.asarray(t), alpha_j)
        law[kt] = float(model.gaussian_mmd2_mass(jnp.asarray(qmass), p_mass))
        grid_moment = np.sum(phi_grid_np * qmass[None, ...], axis=(1, 2))
        grid_moment_error[kt] = float(np.linalg.norm(grid_moment - target))

        dd = st["diagnostics"]
        cal[kt] = float(dd["residual"])
        ess[kt] = float(dd["ess_fraction"])
        lam_norm[kt] = float(dd["lambda_norm"])
        max_lam_coord[kt] = float(dd["max_abs_lambda_coordinate"])
        clip_frac[kt] = float(dd["clip_fraction"])
        retries[kt] = int(dd["retry_count"])
        min_cov[kt] = float(dd["min_cov_eig"])
        cov_cond[kt] = float(dd["cov_condition_number"])
        tangent[kt] = float(dd["tangent_action"])

        if compute_action:
            full, _, pres, _, _ = evaluator.poisson_jit(
                jnp.asarray(q, dtype=jnp.float64),
                jnp.asarray(h_grid, dtype=jnp.float64),
            )
            action[kt] = float(full)
            poisson[kt] = float(pres)

        if worst is None or cal[kt] > worst["residual"]:
            worst = {
                "t": float(t),
                "target": target.tolist(),
                "achieved_moment": np.asarray(st["moment"]).tolist(),
                "residual": float(cal[kt]),
                "lambda": np.asarray(st["lambda"]).tolist(),
                "max_abs_lambda_coordinate": float(max_lam_coord[kt]),
                "clip_fraction": float(clip_frac[kt]),
                "retry_count": int(retries[kt]),
                "ess_fraction": float(ess[kt]),
                "min_cov_eig": float(min_cov[kt]),
                "cov_condition_number": float(cov_cond[kt]),
                "hull_margin": float(-hull_violation[kt]),
                "attempts": dd.get("attempts", []),
            }

    valid = bool(
        np.all(hull_inside)
        and np.max(cal) <= cfg.calibration_accept_tol
        and np.min(ess) >= cfg.min_ess_fraction
        and np.min(in_base_mass) >= cfg.min_in_domain_base_mass
    )

    raw_heldout = trap_average(law, tw, heldout_mask)
    raw_interior = trap_average(law, tw, interior)
    raw_action = float(np.sum(tw * action)) if compute_action else float("nan")
    raw_tangent = float(np.sum(tw * tangent)) if compute_action else float("nan")

    return {
        "scientifically_valid": valid,
        "heldout_mmd2": raw_heldout if valid else float("nan"),
        "all_interior_mmd2": raw_interior if valid else float("nan"),
        "full_action": raw_action if (valid and compute_action) else float("nan"),
        "tangent_action": raw_tangent if (valid and compute_action) else float("nan"),
        "raw_heldout_mmd2": raw_heldout,
        "raw_all_interior_mmd2": raw_interior,
        "raw_full_action": raw_action,
        "raw_tangent_action": raw_tangent,
        "min_ess_fraction": float(np.min(ess)),
        "mean_ess_fraction": float(np.sum(tw * ess)),
        "max_calibration_residual": float(np.max(cal)),
        "all_targets_inside_empirical_moment_hull": bool(np.all(hull_inside)),
        "outside_hull_count": int(np.size(hull_inside) - np.count_nonzero(hull_inside)),
        "max_hull_facet_violation": float(np.max(hull_violation)),
        "min_in_domain_base_mass": float(np.min(in_base_mass)),
        "max_lambda_norm": float(np.max(lam_norm)),
        "max_abs_lambda_coordinate": float(np.max(max_lam_coord)),
        "max_clip_fraction": float(np.max(clip_frac)),
        "retry_case_count": int(np.count_nonzero(retries)),
        "max_retry_count": int(np.max(retries)),
        "min_projected_cov_eig": float(np.min(min_cov)),
        "max_cov_condition_number": float(np.max(cov_cond)),
        "max_grid_moment_error_after_kde": float(np.max(grid_moment_error)),
        "max_poisson_relative_residual": float(np.nanmax(poisson)) if compute_action else float("nan"),
        "worst_calibration_case": worst,
    }


# -----------------------------------------------------------------------------
# Trial evaluation
# -----------------------------------------------------------------------------


def evaluate_design_trial(
    model,
    evaluator: D7Evaluator,
    d2,
    d3,
    c2,
    measurement_cov,
    eta: np.ndarray,
    shared,
    acq_idx: np.ndarray,
    heldout_mask: np.ndarray,
    cfg: D7Config,
    joint_constraints: Dict[str, Any],
    compute_action: bool,
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
    if np.linalg.norm(c0 - joint_constraints["c0"]) > 1.0e-10 or np.linalg.norm(c1 - joint_constraints["c1"]) > 1.0e-10:
        raise RuntimeError("Unexpected alpha-dependent endpoints.")

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

    exact_curve = np.stack([
        np.asarray(model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        for t in times
    ], axis=0)
    exact_cdot = np.stack([
        np.asarray(evaluator.cdot_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        for t in times
    ], axis=0)

    finite = evaluate_particle_curve(
        evaluator=evaluator,
        d2=d2,
        eta=eta,
        alpha=float(shared.alpha),
        target_curve=np.asarray(curve["c"], dtype=np.float64),
        target_cdot=np.asarray(curve["cdot"], dtype=np.float64),
        heldout_mask=heldout_mask,
        compute_action=compute_action,
    )
    exact = evaluate_particle_curve(
        evaluator=evaluator,
        d2=d2,
        eta=eta,
        alpha=float(shared.alpha),
        target_curve=exact_curve,
        target_cdot=exact_cdot,
        heldout_mask=heldout_mask,
        compute_action=compute_action,
    )

    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    row = {
        "alpha": float(shared.alpha),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((y_acq - exact_acq) ** 2))),
        "quadratic_moment_rmse": float(
            np.sqrt(np.mean(np.sum((np.asarray(curve["c"])[interior] - exact_curve[interior]) ** 2, axis=1)))
        ),
        "quadratic_moment_max_error": float(
            np.max(np.linalg.norm(np.asarray(curve["c"])[interior] - exact_curve[interior], axis=1))
        ),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        "beta_cov_trace": float(np.trace(np.asarray(curve["beta_cov"], dtype=np.float64))),

        "finite_valid": float(bool(finite["scientifically_valid"])),
        "exact_valid": float(bool(exact["scientifically_valid"])),

        "finite_heldout_mmd2": finite["heldout_mmd2"],
        "exact_heldout_mmd2": exact["heldout_mmd2"],
        "measurement_delta_mmd2": (
            float(finite["heldout_mmd2"] - exact["heldout_mmd2"])
            if np.isfinite(finite["heldout_mmd2"]) and np.isfinite(exact["heldout_mmd2"])
            else float("nan")
        ),

        "finite_action": finite["full_action"],
        "exact_action": exact["full_action"],
        "measurement_action_inflation": (
            float(finite["full_action"] / exact["full_action"] - 1.0)
            if compute_action and np.isfinite(finite["full_action"]) and np.isfinite(exact["full_action"])
            else float("nan")
        ),
        "measurement_action_excess": (
            float(finite["full_action"] - exact["full_action"])
            if compute_action and np.isfinite(finite["full_action"]) and np.isfinite(exact["full_action"])
            else float("nan")
        ),

        "finite_min_ess": finite["min_ess_fraction"],
        "finite_max_calibration_resid": finite["max_calibration_residual"],
        "finite_max_abs_lambda_coordinate": finite["max_abs_lambda_coordinate"],
        "finite_max_clip_fraction": finite["max_clip_fraction"],
        "finite_retry_case_count": finite["retry_case_count"],
        "finite_outside_hull_count": finite["outside_hull_count"],

        "exact_min_ess": exact["min_ess_fraction"],
        "exact_max_calibration_resid": exact["max_calibration_residual"],
    }

    detail = {
        "curve": curve,
        "finite": finite,
        "exact_population": exact,
    }
    return row, detail


# -----------------------------------------------------------------------------
# Summaries / contrasts
# -----------------------------------------------------------------------------


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "acquisition_mean_rmse",
        "quadratic_moment_rmse",
        "quadratic_moment_max_error",
        "feasibility_projection_active",
        "feasibility_projection_norm",
        "max_unconstrained_hull_violation",
        "beta_cov_trace",
        "finite_valid",
        "exact_valid",
        "finite_heldout_mmd2",
        "exact_heldout_mmd2",
        "measurement_delta_mmd2",
        "finite_action",
        "exact_action",
        "measurement_action_inflation",
        "measurement_action_excess",
        "finite_min_ess",
        "finite_max_calibration_resid",
        "finite_max_abs_lambda_coordinate",
        "finite_max_clip_fraction",
        "finite_retry_case_count",
        "finite_outside_hull_count",
        "exact_min_ess",
        "exact_max_calibration_resid",
    ]
    return {k: mean_se([r.get(k, np.nan) for r in rows]) for k in keys}


def build_contrasts(rows: Dict[str, List[Dict[str, Any]]]):
    out = {}
    out["full_vs_lift_finite_law"] = paired_difference(rows["full"], rows["lift"], "finite_heldout_mmd2")
    out["full_vs_lift_measurement_degradation"] = paired_difference(rows["full"], rows["lift"], "measurement_delta_mmd2")
    out["full_vs_lift_finite_action_reduction"] = ratio_of_means_reduction(rows["full"], rows["lift"], "finite_action")

    out["full_vs_tangent_finite_law"] = paired_difference(rows["full"], rows["tangent"], "finite_heldout_mmd2")
    out["full_vs_tangent_finite_action_reduction"] = ratio_of_means_reduction(rows["full"], rows["tangent"], "finite_action")
    return out


def print_summary(payload: Dict[str, Any]):
    print("\n" + "=" * 108)
    print("Stage D.7 — endpoint-trained FM reference + finite/noisy measurements")
    print("=" * 108)
    cond = payload["finite_condition"]
    print(
        f"N={cond['N']}, K={cond['K']}, detector noise std={cond['obs_noise_std']:.4f}, "
        f"trials={cond['trials']} (action={cond['action_trials']})"
    )
    print("-" * 108)

    for key, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        s = payload["summaries"][key]
        print(
            f"{label:10s} finite law={s['finite_heldout_mmd2']['mean']:.8f} "
            f"± {s['finite_heldout_mmd2']['se']:.2e} | "
            f"measurement Δ={s['measurement_delta_mmd2']['mean']:+.3e} | "
            f"valid={100*s['finite_valid']['mean']:.1f}%"
        )
        print(
            f"{'':10s} finite A={s['finite_action']['mean']:.3f} "
            f"± {s['finite_action']['se']:.2f} | "
            f"A meas infl.={100*s['measurement_action_inflation']['mean']:+.2f}% | "
            f"ESSmin(mean)={s['finite_min_ess']['mean']:.3f} | "
            f"calmax(mean)={s['finite_max_calibration_resid']['mean']:.2e}"
        )

    print("-" * 108)
    c = payload["contrasts"]
    law = c["full_vs_lift_finite_law"]
    deg = c["full_vs_lift_measurement_degradation"]
    act = c["full_vs_lift_finite_action_reduction"]
    print(
        f"Full-Lift: finite law Δ={law['mean_difference_a_minus_b']:+.3e} "
        f"± {law['se_difference']:.2e} | "
        f"differential measurement degradation={deg['mean_difference_a_minus_b']:+.3e} "
        f"± {deg['se_difference']:.2e} | "
        f"finite-action reduction={100*act['ratio_of_means_reduction']:+.2f}%"
    )

    ft = c["full_vs_tangent_finite_law"]
    at = c["full_vs_tangent_finite_action_reduction"]
    print(
        f"Full-Tangent: finite law Δ={ft['mean_difference_a_minus_b']:+.3e} "
        f"± {ft['se_difference']:.2e} | "
        f"finite-action reduction={100*at['ratio_of_means_reduction']:+.2f}%"
    )

    print("-" * 108)
    for k, v in payload["checks"].items():
        print(f"{k}: {v}")
    print("=" * 108)


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
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d7_endpoint_flow_matching_finite_measurements.json")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--N", dest="finite_n", type=int, default=None)
    p.add_argument("--K", dest="acquisition_k", type=int, default=None)
    p.add_argument("--noise", "--noise-std", dest="obs_noise_std", type=float, default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)
    p.add_argument("--calibration-steps", type=int, default=None)
    p.add_argument("--lambda-clip", type=float, default=None)
    p.add_argument("--newton-step-cap", type=float, default=None)
    p.add_argument("--lbfgs-maxiter", type=int, default=None)
    p.add_argument("--calibration-max-retries", type=int, default=None)
    return p


def main():
    t_start = time.time()
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect([
        "stage_b2_transport_conditioned_design.py",
        "stage_b2_transport_conditioned_design(5).py",
    ])
    c2_path = Path(args.c2_script) if args.c2_script else autodetect([
        "stage_c2_mfsi_matched_action.py",
        "../stage_c/stage_c2_mfsi_matched_action.py",
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

    missing = [
        name for name, path in (
            ("backend", backend_path), ("c2-script", c2_path), ("d2-script", d2_path),
            ("d3-script", d3_path), ("d5-script", d5_path)
        ) if path is None
    ]
    if missing:
        raise FileNotFoundError(f"Could not autodetect: {missing}. Pass explicit paths.")

    backend = load_module(backend_path, "stage_b_backend_d7")
    c2 = load_module(c2_path, "stage_c2_backend_d7")
    d2 = load_module(d2_path, "stage_d2_helpers_d7")
    d3 = load_module(d3_path, "stage_d3_helpers_d7")
    d5 = load_module(d5_path, "stage_d5_backend_d7")

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d7_config(args.preset)
    overrides = {}
    for arg_name, field_name, cast in (
        ("seed", "seed", int),
        ("trials", "trials", int),
        ("action_trials", "action_trials", int),
        ("finite_n", "finite_n", int),
        ("acquisition_k", "acquisition_k", int),
        ("obs_noise_std", "obs_noise_std", float),
        ("gh_order", "gh_order", int),
        ("particles", "particles", int),
        ("rk4_substeps", "rk4_substeps_per_time_interval", int),
        ("kde_bandwidth", "kde_bandwidth", float),
        ("calibration_steps", "calibration_steps", int),
        ("lambda_clip", "lambda_clip", float),
        ("newton_step_cap", "newton_step_cap", float),
        ("lbfgs_maxiter", "calibration_lbfgs_maxiter", int),
        ("calibration_max_retries", "calibration_max_retries", int),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = cast(value)
    cfg = dataclasses.replace(cfg, **overrides)

    if cfg.action_trials > cfg.trials:
        raise ValueError("action_trials cannot exceed trials")
    if cfg.finite_n < 2:
        raise ValueError("N must be >=2")

    # Preserve the established reference grid/time regime.
    if args.preset == "quick":
        stage_b_cfg = backend.preset_config("quick")
        stage_b_cfg = dataclasses.replace(stage_b_cfg, grid_n=cfg.grid_n, time_n=cfg.time_n)
    else:
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=cfg.grid_n, time_n=cfg.time_n)

    model = backend.StageB(stage_b_cfg)
    if cfg.acquisition_k < 3 or cfg.acquisition_k >= model.cfg.time_n:
        raise ValueError("K must be >=3 and < time_n")

    cp_phys = checkpoint_meta.get("physical_system", {})
    for key in ("r", "sigma"):
        saved = float(cp_phys[key])
        current = float(getattr(stage_b_cfg, key))
        if abs(saved - current) > 1.0e-12:
            raise ValueError(f"D5 checkpoint {key}={saved} != Stage-B {current}")

    evaluator = D7Evaluator(model, d2, d5, params, checkpoint_meta, cfg)
    measurement_cov = d3.MeasurementCovariance(model)

    designs_deg = {
        "lift": cfg.lift_design_deg,
        "tangent": cfg.tangent_design_deg,
        "full": cfg.full_design_deg,
    }
    designs = {
        k: np.radians(np.asarray(v, dtype=np.float64))
        for k, v in designs_deg.items()
    }

    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in acq_set and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    if heldout_idx.size == 0:
        raise ValueError("Acquisition set leaves no held-out interior time nodes")
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    print("=" * 108)
    print("Stage D.7 — endpoint-trained FM reference + finite/noisy measurements")
    print("=" * 108)
    print(f"Backend       : {Path(backend_path).resolve()}")
    print(f"D5 checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"D5 bank       : {cfg.bank_mode}, GH order={cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Condition     : N={cfg.finite_n}, K={cfg.acquisition_k}, noise std={cfg.obs_noise_std}")
    print(f"Trials        : {cfg.trials} (full action: {cfg.action_trials})")
    print(f"Held-out idx  : {heldout_idx.tolist()}")

    print("\nBuilding physical/D5-particle moment-feasibility intersections...", flush=True)
    joint = {}
    for name, eta in designs.items():
        joint[name] = build_joint_beta_constraints(
            model=model,
            evaluator=evaluator,
            d2=d2,
            c2=c2,
            eta=eta,
            margin=float(cfg.feasibility_margin),
        )
        print(
            f"  {name:8s}: beta inequalities={joint[name]['A'].shape[0]}, "
            f"physical facets={joint[name]['physical_hull_metadata'].get('facets', 'n/a')}",
            flush=True,
        )

    print("Building common-random-number finite-measurement trial bank...", flush=True)
    trial_bank = []
    for trial in range(int(cfg.trials)):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), trial]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        trial_bank.append(c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng))

    rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in designs}
    details: Dict[str, List[Dict[str, Any]]] = {k: [] for k in designs}

    for trial, shared in enumerate(trial_bank):
        do_action = trial < int(cfg.action_trials)
        for name, eta in designs.items():
            row, detail = evaluate_design_trial(
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
                cfg=cfg,
                joint_constraints=joint[name],
                compute_action=do_action,
            )
            rows[name].append(row)
            details[name].append(detail)

        print(
            f"  trial {trial + 1:2d}/{cfg.trials} alpha={shared.alpha:.5f} action={do_action}",
            flush=True,
        )

    summaries = {k: summarize_rows(v) for k, v in rows.items()}
    contrasts = build_contrasts(rows)

    all_rows = [r for rr in rows.values() for r in rr]
    projection_active = np.asarray([r["feasibility_projection_active"] for r in all_rows], dtype=np.float64)
    projection_norm = np.asarray([r["feasibility_projection_norm"] for r in all_rows], dtype=np.float64)

    finite_valid_fraction = {
        name: float(np.mean([bool(r["finite_valid"]) for r in rr]))
        for name, rr in rows.items()
    }
    exact_valid_fraction = {
        name: float(np.mean([bool(r["exact_valid"]) for r in rr]))
        for name, rr in rows.items()
    }

    checks = {
        "checkpoint_teacher_free": bool(
            checkpoint_meta["bridge"].get("uses_analytic_A_t") is False
            and checkpoint_meta["bridge"].get("uses_analytic_B_t") is False
            and checkpoint_meta["bridge"].get("uses_analytic_velocity_teacher") is False
        ),
        "no_cnf_used": True,
        "no_old_analytic_reference_branch": True,
        "finite_rows_all_valid": bool(all(v == 1.0 for v in finite_valid_fraction.values())),
        "exact_population_rows_all_valid": bool(all(v == 1.0 for v in exact_valid_fraction.values())),
        "finite_valid_fraction_by_design": finite_valid_fraction,
        "exact_valid_fraction_by_design": exact_valid_fraction,
        "common_feasibility_projection_active_fraction": float(np.mean(projection_active)),
        "max_common_feasibility_projection_norm": float(np.max(projection_norm)),
    }

    payload = {
        "stage": "D.7",
        "purpose": (
            "Finite/noisy measurement robustness of frozen Lift/Tangent-TC/Full-TC "
            "under the endpoint-trained D.5 flow-matching particle reference."
        ),
        "backend_path": str(Path(backend_path).resolve()),
        "c2_script_path": str(Path(c2_path).resolve()),
        "d2_script_path": str(Path(d2_path).resolve()),
        "d3_script_path": str(Path(d3_path).resolve()),
        "d5_script_path": str(Path(d5_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "config": jsonify(cfg),
        "finite_condition": {
            "N": int(cfg.finite_n),
            "K": int(cfg.acquisition_k),
            "obs_noise_std": float(cfg.obs_noise_std),
            "trials": int(cfg.trials),
            "action_trials": int(cfg.action_trials),
            "acquisition_indices": acq_idx.tolist(),
            "heldout_indices": heldout_idx.tolist(),
        },
        "designs_deg": {k: list(v) for k, v in designs_deg.items()},
        "joint_feasibility": {
            k: {
                "num_beta_inequalities": int(v["A"].shape[0]),
                "physical_hull_metadata": v["physical_hull_metadata"],
                "time_metadata": v["time_metadata"],
            }
            for k, v in joint.items()
        },
        "summaries": summaries,
        "contrasts": contrasts,
        "trial_rows": rows,
        "trial_details": details,
        "checks": checks,
        "interpretation": [
            "D.7 adds finite/noisy measurements to the teacher-free D.5/D.6 reference while keeping sensor designs frozen.",
            "Finite uncertainty enters once through a shared endpoint-anchored GLS quadratic moment trajectory.",
            "The fitted quadratic coefficient is projected, only when needed, onto the intersection of the physical sensor-moment hull and the D.5 learned-particle moment hull over all evaluation times.",
            "There is no branch-specific target clipping after reconstruction.",
            "Hard empirical calibration uses a robust convex-dual solve and invalid rows are excluded from scientific summaries rather than silently regularized.",
            "The primary finite-resource question is whether Full-TC retains near-Lift finite law while reducing finite weighted-Poisson action under the endpoint-trained reference.",
            "Sensor re-optimization remains deferred to D.8.",
        ],
        "elapsed_seconds": float(time.time() - t_start),
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    print_summary(payload)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved D.7 results: {out_path}")


if __name__ == "__main__":
    main()
