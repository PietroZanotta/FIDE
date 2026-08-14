#!/usr/bin/env python3
"""
Stage D.6: frozen-design MFSI under a genuinely endpoint-trained flow-matching reference.

Scientific purpose
------------------
D.5 removed the old Stage-B analytic path/velocity teacher and trained a reference
velocity from endpoint information with conditional flow matching (CFM). D.6 is the
first downstream scientific test of that new reference geometry.

The three previously established designs are frozen:

    Lift        = (1.63 deg, 161.63 deg)
    Tangent-TC  = (0.00 deg, 154.70 deg)
    Full-TC     = (0.00 deg, 160.00 deg)

For each design, D.6:

  1. constructs a deterministic weighted Q0 bank,
  2. pushes it through the D.5 learned ODE,
  3. treats those rollout particles as the reference marginal,
  4. performs the hard empirical moment I-projection,
  5. computes lambda_dot and the MFSI forcing from particle statistics and u_theta,
  6. rasterizes q and q h onto the validated Stage-B grid,
  7. uses the existing weighted-Poisson solver as the deterministic action oracle,
  8. reports Lift / full-action / tangent-action / calibration / ESS diagnostics.

No old analytic reference path is used in the D.6 inference pipeline:
  * no A_t,
  * no B_t,
  * no Stage-B reference_density/reference_q_mass,
  * no exact analytic reference velocity,
  * no CNF density reconstruction.

The Stage-B backend is used only for:
  * physical endpoint/scientific-law parameters,
  * sensor fields,
  * the scientific target law P_t^alpha and its moments,
  * the established grid and weighted-Poisson solver.

Teacher-free reference validation
---------------------------------
For the current synthetic two-lobe benchmark, the D.5 CFM bridge itself has an
exact marginal law because it is defined from independent endpoint draws:

    X_t = a(t) X0 + b(t) X1 + gamma(t) Z.

With X0 and X1 the known two-lobe Gaussian endpoints, this is a four-component
Gaussian mixture. D.6 uses that *declared D.5 bridge marginal* only as a validation
oracle for the learned ODE rollout. This is not the old Stage-B analytic SI and
does not use A_t or B_t.

Numerical validity in v3
------------------------
The hard empirical I-projection is the same convex dual as in D.2, but v3 makes
its numerical status explicit.  The reference preset uses 300 damped-Newton
steps, coordinate clip 1000 and step cap 20, followed by an exact-gradient
L-BFGS-B fallback.  If the residual remains above the scientific acceptance
gate, the coordinate ceiling is retried geometrically and every retry is logged.
Official law/action metrics are null unless hull feasibility, hard calibration,
ESS and in-domain mass gates all pass.

The reference scientific grid is 51x51 with 21 time nodes, matching the
established D2/D3/D4 reference regime.

Recommended reference run
-------------------------
python stage_d6_endpoint_flow_matching_frozen_designs_v3.py \
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \
    --d2-script stage_d2_flow_matching_particle_mfsi.py \
    --d5-script stage_d5_endpoint_flow_matching_reference_v2.py \
    --checkpoint stage_d5_endpoint_flow_matching_reference_v2.npz \
    --preset reference \
    --run-ceiling-sensitivity \
    --ceiling-multiplier 2 \
    --output stage_d6_endpoint_flow_matching_frozen_designs_v3.json
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
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial import ConvexHull, QhullError

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


@dataclasses.dataclass(frozen=True)
class D6Config:
    preset: str = "quick"
    seed: int = 20260813

    # Frozen historical designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)

    # Empirical reference bank / learned ODE.
    bank_mode: str = "gauss-hermite"
    gh_order: int = 20
    particles: int = 8192
    rk4_substeps_per_time_interval: int = 8

    # Hard empirical I-projection.
    calibration_steps: int = 80
    calibration_tol: float = 2.0e-8
    newton_step_cap: float = 10.0
    lambda_clip: float = 300.0

    # Robust convex-dual fallback / adaptive numerical budget.
    calibration_lbfgs_maxiter: int = 400
    calibration_max_retries: int = 2
    calibration_retry_clip_multiplier: float = 2.0
    clip_saturation_fraction: float = 0.995

    # A row is scientifically usable only if hard calibration reaches this gate.
    calibration_accept_tol: float = 2.0e-6

    # Optional matched ceiling-sensitivity check.
    ceiling_sensitivity_multiplier: float = 2.0
    ceiling_action_relative_tol: float = 0.02
    ceiling_law_relative_tol: float = 0.005

    # Particle -> grid anti-aliasing.
    kde_bandwidth: float = 0.0
    kde_truncate: float = 4.0

    # Numerical checks.
    ess_warn_fraction: float = 0.03
    max_allowed_calibration_resid: float = 2.0e-6
    min_in_domain_base_mass: float = 0.995


def preset_d6_config(name: str) -> D6Config:
    if name == "quick":
        return D6Config()
    if name == "reference":
        return D6Config(
            preset="reference",
            gh_order=36,
            particles=32768,
            rk4_substeps_per_time_interval=16,
            # Expanded hard-tilt budget. The primary coordinate ceiling is 1000;
            # the robust solver can retry at 2x/4x only if the residual remains high.
            calibration_steps=300,
            calibration_tol=1.0e-9,
            newton_step_cap=20.0,
            lambda_clip=1000.0,
            calibration_lbfgs_maxiter=800,
            calibration_max_retries=2,
            calibration_retry_clip_multiplier=2.0,
            calibration_accept_tol=2.0e-6,
        )
    if name == "confirm":
        return D6Config(
            preset="confirm",
            gh_order=48,
            particles=65536,
            rk4_substeps_per_time_interval=24,
            calibration_steps=500,
            calibration_tol=5.0e-10,
            newton_step_cap=25.0,
            lambda_clip=2000.0,
            calibration_lbfgs_maxiter=1200,
            calibration_max_retries=2,
            calibration_retry_clip_multiplier=2.0,
            calibration_accept_tol=1.0e-6,
        )
    raise ValueError(name)


def d2_compatible_config(cfg: D6Config, d2):
    """Create the exact config object expected by the reused D.2 particle helpers."""
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
        ess_warn_fraction=cfg.ess_warn_fraction,
        max_allowed_calibration_resid=cfg.max_allowed_calibration_resid,
        min_in_domain_fraction=cfg.min_in_domain_base_mass,
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


def weighted_mean_cov(x: np.ndarray, w: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)
    m = np.sum(w[:, None] * x, axis=0)
    xc = x - m[None, :]
    cov = xc.T @ (w[:, None] * xc)
    return m, cov


def bridge_coefficients_np(t: float, schedule: str):
    t = float(t)
    if schedule == "linear":
        return 1.0 - t, t
    if schedule == "trig":
        return math.cos(0.5 * PI * t), math.sin(0.5 * PI * t)
    raise ValueError(f"Unsupported D.5 bridge schedule {schedule!r}")


def exact_d5_bridge_mean_cov(t: float, r: float, sigma: float, schedule: str, noise_std: float):
    a, b = bridge_coefficients_np(t, schedule)
    g = float(noise_std) * math.sin(PI * float(t))
    cov0 = np.diag([r * r + sigma * sigma, sigma * sigma])
    cov1 = np.diag([sigma * sigma, r * r + sigma * sigma])
    cov = a * a * cov0 + b * b * cov1 + g * g * np.eye(2)
    return np.zeros(2, dtype=np.float64), cov


def exact_d5_bridge_mass(model, t: float, r: float, sigma: float, schedule: str, noise_std: float):
    """
    Exact marginal of the declared D.5 product-coupled stochastic bridge.

    X0 = s0 * (r,0) + eps0
    X1 = s1 * (0,r) + eps1
    eps0,eps1 ~ N(0,sigma^2 I), s0,s1 in {-1,+1}
    X_t = a X0 + b X1 + gamma Z

    Hence the marginal is a four-component isotropic Gaussian mixture.
    """
    a, b = bridge_coefficients_np(t, schedule)
    g = float(noise_std) * math.sin(PI * float(t))
    var = sigma * sigma * (a * a + b * b) + g * g
    var = max(var, 1.0e-14)
    xy = np.asarray(model.xy, dtype=np.float64)
    density = np.zeros(xy.shape[:-1], dtype=np.float64)
    norm = 1.0 / (2.0 * PI * var)
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            mu = np.array([a * s0 * r, b * s1 * r], dtype=np.float64)
            d2 = np.sum((xy - mu[None, None, :]) ** 2, axis=-1)
            density += 0.25 * norm * np.exp(-0.5 * d2 / var)
    z = float(np.sum(density) * model.cell_area)
    density /= max(z, 1.0e-300)
    return density, density * float(model.cell_area)


def rasterize_base_mass(model, d2, x: np.ndarray, w: np.ndarray, bandwidth: float, truncate: float):
    mask = d2.in_domain_mask(model, x)
    x_in = np.asarray(x[mask], dtype=np.float64)
    w_in = np.asarray(w[mask], dtype=np.float64)
    base_mass = float(np.sum(w_in))
    w_in = w_in / max(base_mass, 1.0e-300)

    n = int(model.cfg.grid_n)
    L = float(model.cfg.L)
    edges = np.linspace(-L, L, n + 1, dtype=np.float64)
    mass_hist, _, _ = np.histogram2d(
        x_in[:, 1], x_in[:, 0], bins=(edges, edges), weights=w_in
    )
    sigma_cells = float(bandwidth) / float(model.dx)
    qmass = gaussian_filter(
        mass_hist,
        sigma=sigma_cells,
        mode="constant",
        cval=0.0,
        truncate=float(truncate),
    )
    qmass /= max(float(np.sum(qmass)), 1.0e-300)
    return qmass, {
        "in_domain_count_fraction": float(np.mean(mask)),
        "in_domain_base_mass": base_mass,
    }



def moment_hull_diagnostic(phi: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """
    Exact 2-D convex-hull feasibility diagnostic for the empirical sensor moment.

    For exponential tilting of a finite positive base measure, the attainable
    moment set is the relative interior of conv{Phi(x_i)}; boundary points are
    approached only as |lambda| -> infinity.  A positive facet violation means
    the requested target is outside the empirical moment hull.
    """
    pts = np.asarray(phi, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # Remove numerical duplicates before Qhull.
    pts = np.unique(np.round(pts, decimals=14), axis=0)
    if len(pts) < 3:
        return {
            "hull_available": False,
            "inside_closed_hull": False,
            "max_facet_violation": float("inf"),
            "min_signed_interior_margin": float("-inf"),
            "hull_vertices": int(len(pts)),
        }
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return {
            "hull_available": False,
            "inside_closed_hull": False,
            "max_facet_violation": float("inf"),
            "min_signed_interior_margin": float("-inf"),
            "hull_vertices": 0,
        }

    # scipy.spatial.ConvexHull: equations are normal.dot(x) + offset <= 0 inside.
    vals = hull.equations[:, :-1] @ target + hull.equations[:, -1]
    max_violation = float(np.max(vals))
    return {
        "hull_available": True,
        "inside_closed_hull": bool(max_violation <= 1.0e-12),
        "max_facet_violation": max_violation,
        "min_signed_interior_margin": float(-max_violation),
        "hull_vertices": int(len(hull.vertices)),
    }



def _normalized_exp_weights_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / max(float(np.sum(ez)), 1.0e-300)


def _weighted_mean_np(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(w)[:, None] * np.asarray(x), axis=0)


def _relative_ess_np(projected_w: np.ndarray, base_w: np.ndarray) -> float:
    ep = 1.0 / max(float(np.sum(projected_w * projected_w)), 1.0e-300)
    eb = 1.0 / max(float(np.sum(base_w * base_w)), 1.0e-300)
    return float(ep / max(eb, 1.0e-300))


def solve_empirical_tilt_robust(
    phi: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    ridge: float,
    cfg: D6Config,
    lam0: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Robust convex-dual solver for the hard empirical I-projection.

    The dual objective is
        log sum_i a_i exp(lambda^T phi_i) - lambda^T target,
    whose gradient is exactly the moment residual and whose Hessian is the
    weighted covariance of phi.  We use damped Newton first, then bounded
    L-BFGS-B as a convex fallback.  If calibration still fails, the coordinate
    ceiling is enlarged geometrically; every enlargement is reported.
    """
    phi = np.asarray(phi, dtype=np.float64)
    base_w = np.asarray(base_w, dtype=np.float64)
    base_w = base_w / max(float(np.sum(base_w)), 1.0e-300)
    target = np.asarray(target, dtype=np.float64)
    log_base = np.log(np.maximum(base_w, 1.0e-300))
    m = int(phi.shape[1])
    initial = np.zeros(m, dtype=np.float64) if lam0 is None else np.asarray(lam0, dtype=np.float64).copy()

    def stats(lam: np.ndarray):
        logits = log_base + phi @ lam
        w = _normalized_exp_weights_np(logits)
        moment = _weighted_mean_np(w, phi)
        F = moment - target
        centered = phi - moment[None, :]
        C = centered.T @ (w[:, None] * centered)
        dual = float(logsumexp(logits) - np.dot(lam, target))
        return dual, w, moment, F, C

    best = None
    attempts = []
    clip = float(cfg.lambda_clip)
    lam_seed = initial.copy()

    for retry in range(int(cfg.calibration_max_retries) + 1):
        lam = np.clip(lam_seed, -clip, clip)
        newton_iters = 0
        line_search_failures = 0

        for it in range(int(cfg.calibration_steps)):
            newton_iters = it + 1
            cur_dual, w, moment, F, C = stats(lam)
            resid = float(np.linalg.norm(F))
            if resid <= float(cfg.calibration_tol):
                break

            H = 0.5 * (C + C.T) + float(ridge) * np.eye(m)
            try:
                step = np.linalg.solve(H, F)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H, rcond=1.0e-12) @ F

            sn = float(np.linalg.norm(step))
            if sn > float(cfg.newton_step_cap):
                step *= float(cfg.newton_step_cap) / max(sn, 1.0e-300)

            accepted = False
            for scale in 0.5 ** np.arange(14, dtype=np.float64):
                cand = np.clip(lam - scale * step, -clip, clip)
                cand_dual, *_ = stats(cand)
                if np.isfinite(cand_dual) and cand_dual <= cur_dual + 1.0e-13:
                    lam = cand
                    accepted = True
                    break
            if not accepted:
                line_search_failures += 1
                break

        # Convex fallback: directly minimize the same dual with its exact gradient.
        def fun_grad(ll):
            dual, _, _, F, _ = stats(np.asarray(ll, dtype=np.float64))
            return dual, F

        before = stats(lam)
        before_resid = float(np.linalg.norm(before[3]))
        lbfgs_used = before_resid > float(cfg.calibration_tol)
        lbfgs_success = None
        lbfgs_message = None
        lbfgs_nit = 0
        if lbfgs_used:
            opt = minimize(
                fun_grad,
                x0=np.asarray(lam, dtype=np.float64),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-clip, clip)] * m,
                options={
                    "maxiter": int(cfg.calibration_lbfgs_maxiter),
                    "ftol": 1.0e-15,
                    "gtol": min(float(cfg.calibration_tol), 1.0e-10),
                    "maxls": 50,
                },
            )
            lbfgs_success = bool(opt.success)
            lbfgs_message = str(opt.message)
            lbfgs_nit = int(getattr(opt, "nit", 0))
            cand = np.asarray(opt.x, dtype=np.float64)
            after = stats(cand)
            if np.isfinite(after[0]) and float(np.linalg.norm(after[3])) <= before_resid:
                lam = cand

        dual, w, moment, F, C = stats(lam)
        resid = float(np.linalg.norm(F))
        max_abs = float(np.max(np.abs(lam)))
        clip_fraction = max_abs / max(clip, 1.0e-300)
        at_clip = bool(clip_fraction >= float(cfg.clip_saturation_fraction))
        eig = np.linalg.eigvalsh(0.5 * (C + C.T))
        min_eig = float(np.min(eig))
        max_eig = float(np.max(eig))
        cond = float(max_eig / max(min_eig, 1.0e-300))
        attempt = {
            "retry_index": int(retry),
            "coordinate_clip": float(clip),
            "residual": resid,
            "dual": float(dual),
            "newton_iterations": int(newton_iters),
            "line_search_failures": int(line_search_failures),
            "lbfgs_used": bool(lbfgs_used),
            "lbfgs_success": lbfgs_success,
            "lbfgs_message": lbfgs_message,
            "lbfgs_iterations": int(lbfgs_nit),
            "max_abs_lambda": max_abs,
            "lambda_norm": float(np.linalg.norm(lam)),
            "clip_fraction": clip_fraction,
            "at_coordinate_clip": at_clip,
            "ess_fraction": _relative_ess_np(w, base_w),
            "min_cov_eig": min_eig,
            "cov_condition_number": cond,
        }
        attempts.append(attempt)

        candidate = (resid, float(dual), lam.copy(), w.copy(), moment.copy(), C.copy(), attempt)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate

        if resid <= float(cfg.calibration_accept_tol):
            break

        # Only a numerical budget is being changed; the I-projection objective is unchanged.
        lam_seed = lam.copy()
        clip *= float(cfg.calibration_retry_clip_multiplier)

    assert best is not None
    resid, dual, lam, w, moment, C, best_attempt = best
    diag = dict(best_attempt)
    diag.update({
        "residual": float(resid),
        "dual": float(dual),
        "iterations": int(best_attempt["newton_iterations"] + best_attempt["lbfgs_iterations"]),
        "accepted": bool(resid <= float(cfg.calibration_accept_tol)),
        "strictly_converged": bool(resid <= float(cfg.calibration_tol)),
        "retries_used": int(best_attempt["retry_index"]),
        "effective_coordinate_clip": float(best_attempt["coordinate_clip"]),
        "attempts": attempts,
    })
    return lam, w, moment, C, diag


def particle_mfsi_state_robust(
    phi: np.ndarray,
    grad_phi: np.ndarray,
    u: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    c_dot: np.ndarray,
    ridge: float,
    cfg: D6Config,
    lam0: np.ndarray | None = None,
) -> Dict[str, Any]:
    lam, w, moment, C, cal = solve_empirical_tilt_robust(
        phi, base_w, target, ridge, cfg, lam0
    )

    m = np.einsum("nmc,nc->nm", grad_phi, u)
    Em = _weighted_mean_np(w, m)
    g = m @ lam
    Eg = float(np.dot(w, g))
    centered_phi = phi - moment[None, :]
    cov_phi_g = np.sum(w[:, None] * centered_phi * (g - Eg)[:, None], axis=0)

    H = 0.5 * (C + C.T) + float(ridge) * np.eye(C.shape[0])
    rhs = np.asarray(c_dot, dtype=np.float64) - Em - cov_phi_g
    try:
        lam_dot = np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        lam_dot = np.linalg.pinv(H, rcond=1.0e-12) @ rhs

    h = centered_phi @ lam_dot + g - Eg
    mean_h_before = float(np.dot(w, h))
    h = h - np.dot(w, h)
    mean_h_after = float(np.dot(w, h))

    G = np.einsum("nmc,nkc,n->mk", grad_phi, grad_phi, w)
    rvec = Em - np.asarray(c_dot, dtype=np.float64)
    Gs = 0.5 * (G + G.T) + float(ridge) * np.eye(G.shape[0])
    try:
        tangent_action = float(rvec @ np.linalg.solve(Gs, rvec))
    except np.linalg.LinAlgError:
        tangent_action = float(rvec @ (np.linalg.pinv(Gs, rcond=1.0e-12) @ rvec))

    cal.update({
        "lambda_dot_norm": float(np.linalg.norm(lam_dot)),
        "weighted_mean_h_before_center": mean_h_before,
        "weighted_mean_h_after_center": mean_h_after,
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
        "diagnostics": cal,
    }


# -----------------------------------------------------------------------------
# D.6 evaluator
# -----------------------------------------------------------------------------


class D6Evaluator:
    def __init__(
        self,
        model,
        d2,
        d5,
        params,
        checkpoint_meta: Dict[str, Any],
        cfg: D6Config,
    ):
        self.model = model
        self.d2 = d2
        self.d5 = d5
        self.params = params
        self.checkpoint_meta = checkpoint_meta
        self.cfg = cfg
        self.d2cfg = d2_compatible_config(cfg, d2)

        cp_stage = checkpoint_meta.get("stage")
        if cp_stage != "D.5":
            raise ValueError(f"Expected a D.5 checkpoint, got stage={cp_stage!r}")

        cp_phys = checkpoint_meta.get("physical_system", {})
        self.r = float(cp_phys["r"])
        self.sigma = float(cp_phys["sigma"])

        bridge = checkpoint_meta.get("bridge", {})
        if bool(bridge.get("uses_analytic_A_t", True)):
            raise ValueError("Checkpoint metadata says analytic A_t was used.")
        if bool(bridge.get("uses_analytic_B_t", True)):
            raise ValueError("Checkpoint metadata says analytic B_t was used.")
        if bool(bridge.get("uses_analytic_velocity_teacher", True)):
            raise ValueError("Checkpoint metadata says an analytic velocity teacher was used.")
        if bridge.get("endpoint_coupling") != "independent product coupling":
            raise ValueError(
                "This D.6 implementation currently validates the synthetic independent-product D.5 bridge only."
            )
        self.bridge_schedule = str(bridge["schedule"])
        self.noise_std = float(bridge["noise_std"])

        # Build the same synthetic endpoint sampler used by D.5. For Gauss-Hermite
        # mode, only r/sigma are consumed by D.2's deterministic initial bank.
        self.endpoint_sampler = d5.EndpointSampler(
            r=self.r,
            sigma=self.sigma,
            x0_external=None,
            x1_external=None,
            validation_fraction=0.1,
            seed=cfg.seed,
        )

        x0_np, base_w = d2.initial_reference_bank(self.endpoint_sampler, self.d2cfg)
        self.x0 = jnp.asarray(x0_np, dtype=jnp.float64)
        self.base_w = np.asarray(base_w, dtype=np.float64)

        self.velocity_jit = jax.jit(lambda t, x: d5.velocity_mlp(params, t, x))
        self.cdot_jit = jax.jit(jax.jacfwd(model.measurement_grid, argnums=0))
        self.poisson_jit = jax.jit(model.poisson_solve)

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
        for k, t in enumerate(ts):
            unodes.append(
                np.asarray(
                    self.velocity_jit(jnp.asarray(t), jnp.asarray(self.nodes[k])),
                    dtype=np.float64,
                )
            )
        self.u_nodes = np.stack(unodes, axis=0)

    def _bandwidth(self, bandwidth: float | None = None):
        if bandwidth is not None:
            return float(bandwidth)
        bw = float(self.cfg.kde_bandwidth)
        return 0.35 * float(self.model.dx) if bw <= 0.0 else bw

    def reference_bank_diagnostics(self) -> Dict[str, Any]:
        bw = self._bandwidth()
        rows = []
        for k, t in enumerate(np.asarray(self.model.times, dtype=np.float64)):
            x = self.nodes[k]
            mean, cov = weighted_mean_cov(x, self.base_w)
            exact_mean, exact_cov = exact_d5_bridge_mean_cov(
                float(t), self.r, self.sigma, self.bridge_schedule, self.noise_std
            )
            qmass, dom = rasterize_base_mass(
                self.model,
                self.d2,
                x,
                self.base_w,
                bw,
                self.cfg.kde_truncate,
            )
            _, bridge_mass = exact_d5_bridge_mass(
                self.model,
                float(t),
                self.r,
                self.sigma,
                self.bridge_schedule,
                self.noise_std,
            )
            mmd = float(
                self.model.gaussian_mmd2_mass(
                    jnp.asarray(qmass, dtype=jnp.float64),
                    jnp.asarray(bridge_mass, dtype=jnp.float64),
                )
            )
            rows.append({
                "t": float(t),
                "learned_mean": mean.tolist(),
                "exact_bridge_mean": exact_mean.tolist(),
                "mean_l2_error": float(np.linalg.norm(mean - exact_mean)),
                "learned_covariance": cov.tolist(),
                "exact_bridge_covariance": exact_cov.tolist(),
                "covariance_fro_error": float(np.linalg.norm(cov - exact_cov, ord="fro")),
                "learned_vs_exact_declared_bridge_mmd2": mmd,
                **dom,
            })

        interior = rows[1:-1]
        endpoint = rows[-1]
        return {
            "kde_bandwidth": bw,
            "kde_sigma_cells": float(bw / self.model.dx),
            "times": rows,
            "mean_interior_bridge_mmd2": float(
                np.mean([r["learned_vs_exact_declared_bridge_mmd2"] for r in interior])
                if interior else 0.0
            ),
            "max_interior_bridge_mmd2": float(
                max([r["learned_vs_exact_declared_bridge_mmd2"] for r in interior], default=0.0)
            ),
            "endpoint_bridge_mmd2": float(endpoint["learned_vs_exact_declared_bridge_mmd2"]),
            "endpoint_mean_l2_error": float(endpoint["mean_l2_error"]),
            "endpoint_covariance_fro_error": float(endpoint["covariance_fro_error"]),
            "max_mean_l2_error": float(max(r["mean_l2_error"] for r in rows)),
            "max_covariance_fro_error": float(max(r["covariance_fro_error"] for r in rows)),
            "min_in_domain_base_mass": float(min(r["in_domain_base_mass"] for r in rows)),
        }

    def evaluate_design(
        self,
        eta: np.ndarray,
        bandwidth: float | None = None,
        calibration_cfg: D6Config | None = None,
    ):
        """Evaluate one frozen design, with validity separated from raw diagnostics."""
        bw = self._bandwidth(bandwidth)
        cal_cfg = self.cfg if calibration_cfg is None else calibration_cfg

        times = np.asarray(self.model.times, dtype=np.float64)
        alphas = np.asarray(self.model.alphas, dtype=np.float64)
        tw = np.asarray(self.model.time_w, dtype=np.float64)
        aw = np.asarray(self.model.alpha_w, dtype=np.float64)

        shape = (len(alphas), len(times))
        law = np.zeros(shape, dtype=np.float64)
        action = np.zeros(shape, dtype=np.float64)
        tangent = np.zeros(shape, dtype=np.float64)
        cal_resid = np.zeros(shape, dtype=np.float64)
        ess = np.zeros(shape, dtype=np.float64)
        lam_norm = np.zeros(shape, dtype=np.float64)
        max_abs_lam = np.zeros(shape, dtype=np.float64)
        clip_fraction = np.zeros(shape, dtype=np.float64)
        effective_clip = np.zeros(shape, dtype=np.float64)
        retries_used = np.zeros(shape, dtype=np.float64)
        min_cov = np.zeros(shape, dtype=np.float64)
        cov_cond = np.zeros(shape, dtype=np.float64)
        lambda_dot_norm = np.zeros(shape, dtype=np.float64)
        min_gram = np.zeros(shape, dtype=np.float64)
        poisson_resid = np.zeros(shape, dtype=np.float64)
        grid_moment_err = np.zeros(shape, dtype=np.float64)
        source_compat = np.zeros(shape, dtype=np.float64)
        hull_violation = np.zeros(shape, dtype=np.float64)
        hull_inside = np.zeros(shape, dtype=bool)
        hull_margin = np.zeros(shape, dtype=np.float64)
        iterations = np.zeros(shape, dtype=np.float64)
        at_clip = np.zeros(shape, dtype=bool)
        used_lbfgs = np.zeros(shape, dtype=bool)
        in_fraction = np.zeros(len(times), dtype=np.float64)
        in_base_mass = np.zeros(len(times), dtype=np.float64)

        lam_warm = [np.zeros(2, dtype=np.float64) for _ in alphas]
        phi_grid, _ = self.model.sensor_fields(jnp.asarray(eta, dtype=jnp.float64))
        phi_grid_np = np.asarray(phi_grid, dtype=np.float64)

        worst_calibration_case: Dict[str, Any] | None = None
        worst_lambda_case: Dict[str, Any] | None = None

        for kt, t in enumerate(times):
            x_all = self.nodes[kt]
            u_all = self.u_nodes[kt]
            mask = self.d2.in_domain_mask(self.model, x_all)
            in_fraction[kt] = float(np.mean(mask))
            x = x_all[mask]
            u = u_all[mask]
            base_w = self.base_w[mask].copy()
            base_mass = float(np.sum(base_w))
            in_base_mass[kt] = base_mass
            base_w /= max(base_mass, 1.0e-300)

            if x.shape[0] < 100:
                raise RuntimeError(f"Too few in-domain particles at t={t}: {x.shape[0]}")

            phi, grad_phi = self.d2.sensor_particle_fields(self.model, eta, x)

            for ka, alpha in enumerate(alphas):
                target = np.asarray(
                    self.model.measurement_grid(
                        jnp.asarray(t), jnp.asarray(alpha), jnp.asarray(eta)
                    ),
                    dtype=np.float64,
                )
                c_dot = np.asarray(
                    self.cdot_jit(
                        jnp.asarray(t), jnp.asarray(alpha), jnp.asarray(eta)
                    ),
                    dtype=np.float64,
                )

                hull_diag = moment_hull_diagnostic(phi, target)
                hull_violation[ka, kt] = float(hull_diag["max_facet_violation"])
                hull_inside[ka, kt] = bool(hull_diag["inside_closed_hull"])
                hull_margin[ka, kt] = float(hull_diag["min_signed_interior_margin"])

                st = particle_mfsi_state_robust(
                    phi,
                    grad_phi,
                    u,
                    base_w,
                    target,
                    c_dot,
                    float(self.model.cfg.newton_ridge),
                    cal_cfg,
                    lam_warm[ka],
                )
                lam_warm[ka] = st["lambda"]
                diag = st["diagnostics"]

                q, qmass, h_grid, ras = self.d2.rasterize_projected_state(
                    self.model,
                    x,
                    st["weights"],
                    st["h"],
                    bw,
                    self.cfg.kde_truncate,
                )

                full, _, pres, _, _ = self.poisson_jit(
                    jnp.asarray(q, dtype=jnp.float64),
                    jnp.asarray(h_grid, dtype=jnp.float64),
                )
                _, p_mass = self.model.external_q_mass(jnp.asarray(t), jnp.asarray(alpha))
                lift = self.model.gaussian_mmd2_mass(
                    jnp.asarray(qmass, dtype=jnp.float64), p_mass
                )
                grid_moment = np.sum(phi_grid_np * qmass[None, ...], axis=(1, 2))

                law[ka, kt] = float(lift)
                action[ka, kt] = float(full)
                tangent[ka, kt] = float(diag["tangent_action"])
                cal_resid[ka, kt] = float(diag["residual"])
                iterations[ka, kt] = float(diag.get("iterations", 0))
                ess[ka, kt] = float(diag["ess_fraction"])
                lam_norm[ka, kt] = float(diag["lambda_norm"])
                max_abs_lam[ka, kt] = float(diag["max_abs_lambda"])
                clip_fraction[ka, kt] = float(diag["clip_fraction"])
                effective_clip[ka, kt] = float(diag["effective_coordinate_clip"])
                retries_used[ka, kt] = float(diag["retries_used"])
                at_clip[ka, kt] = bool(diag["at_coordinate_clip"])
                used_lbfgs[ka, kt] = bool(diag.get("lbfgs_used", False))
                min_cov[ka, kt] = float(diag["min_cov_eig"])
                cov_cond[ka, kt] = float(diag["cov_condition_number"])
                lambda_dot_norm[ka, kt] = float(diag["lambda_dot_norm"])
                min_gram[ka, kt] = float(diag["min_tangent_gram_eig"])
                poisson_resid[ka, kt] = float(pres)
                grid_moment_err[ka, kt] = float(np.linalg.norm(grid_moment - target))
                source_compat[ka, kt] = abs(float(ras["source_mass_after_center"]))

                case = {
                    "t": float(t),
                    "alpha_rad": float(alpha),
                    "alpha_deg": float(np.degrees(alpha)),
                    "target": target.tolist(),
                    "achieved_moment": np.asarray(st["moment"], dtype=np.float64).tolist(),
                    "residual": float(diag["residual"]),
                    "lambda": np.asarray(st["lambda"], dtype=np.float64).tolist(),
                    "lambda_norm": float(diag["lambda_norm"]),
                    "max_abs_lambda": float(diag["max_abs_lambda"]),
                    "effective_coordinate_clip": float(diag["effective_coordinate_clip"]),
                    "clip_fraction": float(diag["clip_fraction"]),
                    "at_coordinate_clip": bool(diag["at_coordinate_clip"]),
                    "retries_used": int(diag["retries_used"]),
                    "ess_fraction": float(diag["ess_fraction"]),
                    "min_cov_eig": float(diag["min_cov_eig"]),
                    "cov_condition_number": float(diag["cov_condition_number"]),
                    "lambda_dot_norm": float(diag["lambda_dot_norm"]),
                    "hull_inside": bool(hull_diag["inside_closed_hull"]),
                    "hull_facet_violation": float(hull_diag["max_facet_violation"]),
                    "hull_interior_margin": float(hull_diag["min_signed_interior_margin"]),
                    "solver_attempts": diag.get("attempts", []),
                }
                if (
                    worst_calibration_case is None
                    or case["residual"] > worst_calibration_case["residual"]
                ):
                    worst_calibration_case = case
                if (
                    worst_lambda_case is None
                    or case["max_abs_lambda"] > worst_lambda_case["max_abs_lambda"]
                ):
                    worst_lambda_case = case

        W = aw[:, None] * tw[None, :]
        raw_L = float(np.sum(W * law))
        raw_A = float(np.sum(W * action))
        raw_T = float(np.sum(W * tangent))

        all_hull = bool(np.all(hull_inside))
        max_cal = float(np.max(cal_resid))
        min_ess = float(np.min(ess))
        min_domain = float(np.min(in_base_mass))
        finite_raw = bool(
            np.all(np.isfinite(law))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(tangent))
        )
        calibration_valid = bool(max_cal <= float(cal_cfg.calibration_accept_tol))
        scientific_valid = bool(
            finite_raw
            and all_hull
            and calibration_valid
            and min_ess >= float(self.cfg.ess_warn_fraction)
            and min_domain >= float(self.cfg.min_in_domain_base_mass)
        )

        return {
            # Official metrics are null unless the hard population fiber was reached.
            "lift_mmd2": raw_L if scientific_valid else None,
            "full_action": raw_A if scientific_valid else None,
            "tangent_action": raw_T if scientific_valid else None,
            "hidden_action": float(raw_A - raw_T) if scientific_valid else None,
            # Always retain raw values for numerical diagnosis.
            "raw_lift_mmd2": raw_L,
            "raw_full_action": raw_A,
            "raw_tangent_action": raw_T,
            "raw_hidden_action": float(raw_A - raw_T),
            "scientific_metrics_valid": scientific_valid,
            "calibration_valid": calibration_valid,
            "kde_bandwidth": bw,
            "kde_sigma_cells": float(bw / self.model.dx),
            "particle_count": int(self.x0.shape[0]),
            "bank_mode": self.cfg.bank_mode,
            "gh_order": int(self.cfg.gh_order) if self.cfg.bank_mode == "gauss-hermite" else None,
            "calibration_primary_coordinate_clip": float(cal_cfg.lambda_clip),
            "calibration_accept_tol": float(cal_cfg.calibration_accept_tol),
            "min_in_domain_fraction": float(np.min(in_fraction)),
            "mean_in_domain_fraction": float(np.sum(tw * in_fraction)),
            "min_in_domain_base_mass": min_domain,
            "mean_in_domain_base_mass": float(np.sum(tw * in_base_mass)),
            "max_calibration_residual": max_cal,
            "mean_calibration_residual": float(np.sum(W * cal_resid)),
            "max_calibration_iterations": int(np.max(iterations)),
            "all_targets_inside_empirical_moment_hull": all_hull,
            "outside_hull_count": int(np.size(hull_inside) - np.count_nonzero(hull_inside)),
            "max_empirical_moment_hull_facet_violation": float(np.max(hull_violation)),
            "min_empirical_moment_hull_interior_margin": float(np.min(hull_margin)),
            "min_ess_fraction": min_ess,
            "mean_ess_fraction": float(np.sum(W * ess)),
            "max_lambda_norm": float(np.max(lam_norm)),
            "max_abs_lambda_coordinate": float(np.max(max_abs_lam)),
            "max_lambda_clip_fraction": float(np.max(clip_fraction)),
            "max_effective_coordinate_clip": float(np.max(effective_clip)),
            "coordinate_clip_saturation_count": int(np.count_nonzero(at_clip)),
            "calibration_retry_case_count": int(np.count_nonzero(retries_used > 0)),
            "max_calibration_retries_used": int(np.max(retries_used)),
            "lbfgs_fallback_case_count": int(np.count_nonzero(used_lbfgs)),
            "min_calibration_cov_eig": float(np.min(min_cov)),
            "max_calibration_cov_condition_number": float(np.max(cov_cond)),
            "max_lambda_dot_norm": float(np.max(lambda_dot_norm)),
            "min_tangent_gram_eig": float(np.min(min_gram)),
            "max_poisson_relative_residual": float(np.max(poisson_resid)),
            "max_grid_moment_error_after_kde": float(np.max(grid_moment_err)),
            "mean_grid_moment_error_after_kde": float(np.sum(W * grid_moment_err)),
            "max_abs_smoothed_source_compatibility": float(np.max(source_compat)),
            "worst_calibration_case": worst_calibration_case,
            "worst_lambda_case": worst_lambda_case,
        }

    def bandwidth_check(self, eta: np.ndarray, bandwidths: Sequence[float]):
        rows = []
        for bw in bandwidths:
            print(f"  D.6 Full-TC bandwidth check h={bw:.4f}", flush=True)
            r = self.evaluate_design(eta, bandwidth=float(bw))
            rows.append({
                "kde_bandwidth": float(bw),
                "scientific_metrics_valid": bool(r["scientific_metrics_valid"]),
                "lift_mmd2": r["lift_mmd2"],
                "full_action": r["full_action"],
                "tangent_action": r["tangent_action"],
                "raw_lift_mmd2": r["raw_lift_mmd2"],
                "raw_full_action": r["raw_full_action"],
                "min_ess_fraction": r["min_ess_fraction"],
                "max_calibration_residual": r["max_calibration_residual"],
                "max_abs_lambda_coordinate": r["max_abs_lambda_coordinate"],
            })
        return rows

    def ceiling_sensitivity(
        self,
        design_map: Dict[str, np.ndarray],
        multiplier: float,
    ) -> Dict[str, Any]:
        """Matched full recomputation with a larger primary lambda ceiling."""
        cfg2 = dataclasses.replace(
            self.cfg,
            lambda_clip=float(self.cfg.lambda_clip) * float(multiplier),
        )
        rows: Dict[str, Any] = {}
        for name, eta in design_map.items():
            print(
                f"Ceiling sensitivity {name}: primary coordinate clip={cfg2.lambda_clip:.1f}",
                flush=True,
            )
            rows[name] = self.evaluate_design(eta, calibration_cfg=cfg2)
        return {
            "multiplier": float(multiplier),
            "primary_coordinate_clip": float(cfg2.lambda_clip),
            "designs": rows,
        }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _valid_pair(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return bool(
        a.get("scientific_metrics_valid", False)
        and b.get("scientific_metrics_valid", False)
        and a.get("lift_mmd2") is not None
        and b.get("lift_mmd2") is not None
        and a.get("full_action") is not None
        and b.get("full_action") is not None
    )


def full_vs_lift(designs: Dict[str, Any]):
    full = designs["full"]
    lift = designs["lift"]
    if not _valid_pair(full, lift):
        return {
            "valid": False,
            "reason": "Full-TC and/or Lift did not pass hard-calibration/reference-overlap validity gates.",
        }
    return {
        "valid": True,
        "law_relative_penalty": float(full["lift_mmd2"] / lift["lift_mmd2"] - 1.0),
        "action_reduction_fraction": float(1.0 - full["full_action"] / lift["full_action"]),
        "tangent_action_reduction_fraction": float(
            1.0 - full["tangent_action"] / lift["tangent_action"]
        ),
        "full_minus_lift_action": float(full["full_action"] - lift["full_action"]),
    }


def tangent_vs_lift(designs: Dict[str, Any]):
    tangent = designs["tangent"]
    lift = designs["lift"]
    if not _valid_pair(tangent, lift):
        return {
            "valid": False,
            "reason": "Tangent-TC and/or Lift did not pass hard-calibration/reference-overlap validity gates.",
        }
    return {
        "valid": True,
        "law_relative_penalty": float(tangent["lift_mmd2"] / lift["lift_mmd2"] - 1.0),
        "action_reduction_fraction": float(
            1.0 - tangent["full_action"] / lift["full_action"]
        ),
    }


def full_vs_tangent(designs: Dict[str, Any]):
    full = designs["full"]
    tangent = designs["tangent"]
    if not _valid_pair(full, tangent):
        return {
            "valid": False,
            "reason": "Full-TC and/or Tangent-TC did not pass hard-calibration/reference-overlap validity gates.",
        }
    return {
        "valid": True,
        "law_relative_change": float(full["lift_mmd2"] / tangent["lift_mmd2"] - 1.0),
        "full_action_reduction_fraction": float(
            1.0 - full["full_action"] / tangent["full_action"]
        ),
        "tangent_action_reduction_fraction": float(
            1.0 - full["tangent_action"] / tangent["tangent_action"]
        ),
    }


def compare_ceiling_runs(primary: Dict[str, Any], sensitivity: Dict[str, Any], cfg: D6Config):
    out = {"available": True, "designs": {}}
    stable = True
    for name in ("lift", "tangent", "full"):
        a = primary[name]
        b = sensitivity["designs"][name]
        row = {
            "primary_valid": bool(a.get("scientific_metrics_valid", False)),
            "sensitivity_valid": bool(b.get("scientific_metrics_valid", False)),
        }
        if row["primary_valid"] and row["sensitivity_valid"]:
            law_rel = float(b["lift_mmd2"] / a["lift_mmd2"] - 1.0)
            act_rel = float(b["full_action"] / a["full_action"] - 1.0)
            row.update({
                "law_relative_change": law_rel,
                "full_action_relative_change": act_rel,
                "stable": bool(
                    abs(law_rel) <= float(cfg.ceiling_law_relative_tol)
                    and abs(act_rel) <= float(cfg.ceiling_action_relative_tol)
                ),
            })
            stable = stable and row["stable"]
        else:
            row["stable"] = False
            stable = False
        out["designs"][name] = row
    out["all_designs_stable"] = bool(stable)
    return out



def print_summary(payload: Dict[str, Any]):
    print("\n" + "=" * 110)
    print("Stage D.6 frozen-design MFSI under endpoint-trained FM reference")
    print("=" * 110)

    ref = payload["reference_bank_diagnostics"]
    print(
        "Reference bank vs declared D.5 bridge: "
        f"mean interior MMD^2={ref['mean_interior_bridge_mmd2']:.6e} | "
        f"max interior={ref['max_interior_bridge_mmd2']:.6e} | "
        f"endpoint={ref['endpoint_bridge_mmd2']:.6e}"
    )
    print(
        "Reference endpoint moments: "
        f"mean err={ref['endpoint_mean_l2_error']:.3e} | "
        f"cov err={ref['endpoint_covariance_fro_error']:.3e} | "
        f"in-domain base mass min={ref['min_in_domain_base_mass']:.6f}"
    )
    print("-" * 110)

    for key, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        r = payload["designs"][key]
        status = "VALID" if r["scientific_metrics_valid"] else "INVALID"
        if r["scientific_metrics_valid"]:
            metric_text = (
                f"L={r['lift_mmd2']:.8f} | A_full={r['full_action']:.3f} | "
                f"A_tan={r['tangent_action']:.3f}"
            )
        else:
            metric_text = (
                f"raw L={r['raw_lift_mmd2']:.8f} | raw A_full={r['raw_full_action']:.3f} | "
                f"raw A_tan={r['raw_tangent_action']:.3f}"
            )
        print(
            f"{label:10s} [{status:7s}] {metric_text} | "
            f"ESSmin={r['min_ess_fraction']:.3f} | calmax={r['max_calibration_residual']:.2e} | "
            f"hull_out={r['outside_hull_count']}"
        )
        print(
            f"             max|lambda_j|={r['max_abs_lambda_coordinate']:.2f} | "
            f"max clip fraction={r['max_lambda_clip_fraction']:.3f} | "
            f"max effective clip={r['max_effective_coordinate_clip']:.1f} | "
            f"retry cases={r['calibration_retry_case_count']} | clip hits={r['coordinate_clip_saturation_count']}"
        )
        wc = r.get("worst_calibration_case")
        if wc is not None:
            print(
                f"             worst calibration: t={wc['t']:.3f}, alpha={wc['alpha_deg']:.2f} deg, "
                f"resid={wc['residual']:.3e}, ESS={wc['ess_fraction']:.3f}, "
                f"min eig(C)={wc['min_cov_eig']:.3e}, cond(C)={wc['cov_condition_number']:.3e}"
            )

    print("-" * 110)
    c = payload["contrasts"]["full_vs_lift"]
    if c.get("valid", False):
        print(
            "Full-TC vs Lift: "
            f"law penalty={100.0*c['law_relative_penalty']:+.3f}% | "
            f"full-action reduction={100.0*c['action_reduction_fraction']:+.2f}%"
        )
    else:
        print(f"Full-TC vs Lift: NOT SCIENTIFICALLY REPORTABLE ({c['reason']})")

    ct = payload["contrasts"]["full_vs_tangent"]
    if ct.get("valid", False):
        print(
            "Full-TC vs Tangent-TC: "
            f"law change={100.0*ct['law_relative_change']:+.3f}% | "
            f"full-action reduction={100.0*ct['full_action_reduction_fraction']:+.2f}%"
        )
    else:
        print(f"Full-TC vs Tangent-TC: NOT SCIENTIFICALLY REPORTABLE ({ct['reason']})")

    sens = payload.get("calibration_ceiling_sensitivity")
    if sens is not None:
        comp = payload.get("calibration_ceiling_comparison")
        print("-" * 110)
        print(
            f"Ceiling sensitivity: primary clip x {sens['multiplier']:.2f} -> "
            f"{sens['primary_coordinate_clip']:.1f}"
        )
        if comp is not None:
            for name, row in comp["designs"].items():
                if row.get("primary_valid") and row.get("sensitivity_valid"):
                    print(
                        f"  {name:8s}: law change={100.0*row['law_relative_change']:+.3f}% | "
                        f"action change={100.0*row['full_action_relative_change']:+.3f}% | "
                        f"stable={row['stable']}"
                    )
                else:
                    print(f"  {name:8s}: sensitivity comparison invalid")
            print(f"  all designs stable: {comp['all_designs_stable']}")

    print("-" * 110)
    for k, v in payload["checks"].items():
        print(f"{k}: {v}")
    print("=" * 110)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--d2-script", type=str, default=None)
    p.add_argument("--d5-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d6_endpoint_flow_matching_frozen_designs_v3.json")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--bank-mode", choices=("gauss-hermite", "iid"), default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)

    # Hard-calibration overrides. These change only the numerical solve budget, not the objective.
    p.add_argument("--calibration-steps", type=int, default=None)
    p.add_argument("--lambda-clip", type=float, default=None)
    p.add_argument("--newton-step-cap", type=float, default=None)
    p.add_argument("--calibration-accept-tol", type=float, default=None)
    p.add_argument("--lbfgs-maxiter", type=int, default=None)
    p.add_argument("--calibration-max-retries", type=int, default=None)
    p.add_argument("--retry-clip-multiplier", type=float, default=None)

    p.add_argument("--run-ceiling-sensitivity", action="store_true")
    p.add_argument("--ceiling-multiplier", type=float, default=None)

    p.add_argument("--run-bandwidth-check", action="store_true")
    p.add_argument(
        "--bandwidths",
        type=str,
        default="0.06,0.08,0.10,0.12",
        help="Comma-separated physical KDE bandwidths for optional Full-TC check.",
    )
    return p


def main():
    t0 = time.time()
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect(
        [
            "stage_b2_transport_conditioned_design.py",
            "stage_b2_transport_conditioned_design(5).py",
        ]
    )
    if backend_path is None:
        raise FileNotFoundError("Pass --backend /path/to/stage_b2_transport_conditioned_design.py")

    d2_path = Path(args.d2_script) if args.d2_script else autodetect(
        [
            "stage_d2_flow_matching_particle_mfsi.py",
            "stage_d2_flow_matching_particle_mfsi(2).py",
        ]
    )
    if d2_path is None:
        raise FileNotFoundError("Pass --d2-script /path/to/stage_d2_flow_matching_particle_mfsi.py")

    d5_path = Path(args.d5_script) if args.d5_script else autodetect(
        [
            "stage_d5_endpoint_flow_matching_reference_v2.py",
            "stage_d5_endpoint_flow_matching_reference.py",
        ]
    )
    if d5_path is None:
        raise FileNotFoundError("Pass --d5-script /path/to/stage_d5_endpoint_flow_matching_reference_v2.py")

    backend = load_module(backend_path, "stage_b2_backend_d6")
    d2 = load_module(d2_path, "stage_d2_helpers_d6")
    d5 = load_module(d5_path, "stage_d5_backend_d6")

    params, checkpoint_meta = d5.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d6_config(args.preset)
    overrides = {}
    for arg_name, field_name, cast in (
        ("seed", "seed", int),
        ("bank_mode", "bank_mode", str),
        ("gh_order", "gh_order", int),
        ("particles", "particles", int),
        ("rk4_substeps", "rk4_substeps_per_time_interval", int),
        ("kde_bandwidth", "kde_bandwidth", float),
        ("calibration_steps", "calibration_steps", int),
        ("lambda_clip", "lambda_clip", float),
        ("newton_step_cap", "newton_step_cap", float),
        ("calibration_accept_tol", "calibration_accept_tol", float),
        ("lbfgs_maxiter", "calibration_lbfgs_maxiter", int),
        ("calibration_max_retries", "calibration_max_retries", int),
        ("retry_clip_multiplier", "calibration_retry_clip_multiplier", float),
        ("ceiling_multiplier", "ceiling_sensitivity_multiplier", float),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = cast(value)
    cfg = dataclasses.replace(cfg, **overrides)

    # D.6 uses the Stage-B backend for the scientific population/sensors/grid.
    # It deliberately never calls its analytic reference path methods.
    if args.preset == "quick":
        base = backend.preset_config("quick")
        stage_b_cfg = dataclasses.replace(base, grid_n=19, time_n=13)
    elif args.preset == "reference":
        # Match the established D2/D3/D4 reference numerical regime.
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=51, time_n=21)
    else:
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=65, time_n=27)

    cp_phys = checkpoint_meta.get("physical_system", {})
    parameter_checks = {}
    for key in ("r", "sigma"):
        current = float(getattr(stage_b_cfg, key))
        saved = float(cp_phys[key])
        diff = abs(saved - current)
        parameter_checks[key] = {
            "checkpoint": saved,
            "stage_b": current,
            "abs_diff": diff,
        }
        if diff > 1.0e-12:
            raise ValueError(f"D.5 checkpoint {key}={saved} != Stage-B {current}")

    # kappa is intentionally NOT a D.6 compatibility requirement because the new
    # bridge does not use the old analytic Stage-B interpolation geometry.
    if "kappa" in cp_phys:
        parameter_checks["kappa_provenance_only"] = {
            "checkpoint": float(cp_phys["kappa"]),
            "used_by_d6_reference": False,
        }

    model = backend.StageB(stage_b_cfg)
    evaluator = D6Evaluator(model, d2, d5, params, checkpoint_meta, cfg)

    reference_bank_diagnostics = evaluator.reference_bank_diagnostics()

    designs = {}
    design_deg = {
        "lift": cfg.lift_design_deg,
        "tangent": cfg.tangent_design_deg,
        "full": cfg.full_design_deg,
    }
    design_eta = {
        name: np.radians(np.asarray(deg, dtype=np.float64))
        for name, deg in design_deg.items()
    }
    for name, deg in design_deg.items():
        print(f"Evaluating D.6 {name}: {deg[0]:.2f} deg, {deg[1]:.2f} deg", flush=True)
        row = evaluator.evaluate_design(design_eta[name])
        row["theta_deg"] = [float(deg[0]), float(deg[1])]
        designs[name] = row

    contrasts = {
        "full_vs_lift": full_vs_lift(designs),
        "tangent_vs_lift": tangent_vs_lift(designs),
        "full_vs_tangent": full_vs_tangent(designs),
    }

    ceiling_sensitivity = None
    ceiling_comparison = None
    if args.run_ceiling_sensitivity:
        ceiling_sensitivity = evaluator.ceiling_sensitivity(
            design_eta, float(cfg.ceiling_sensitivity_multiplier)
        )
        ceiling_comparison = compare_ceiling_runs(designs, ceiling_sensitivity, cfg)

    bandwidth_check = None
    if args.run_bandwidth_check:
        bws = [float(x.strip()) for x in args.bandwidths.split(",") if x.strip()]
        bandwidth_check = evaluator.bandwidth_check(
            np.radians(np.asarray(cfg.full_design_deg, dtype=np.float64)),
            bws,
        )

    all_design_rows = list(designs.values())
    finite_outputs = all(
        np.isfinite(v)
        for row in all_design_rows
        for key, v in row.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    max_cal = max(r["max_calibration_residual"] for r in all_design_rows)
    min_ess = min(r["min_ess_fraction"] for r in all_design_rows)
    min_domain = min(r["min_in_domain_base_mass"] for r in all_design_rows)
    all_hulls_feasible = all(r["all_targets_inside_empirical_moment_hull"] for r in all_design_rows)
    max_hull_violation = max(r["max_empirical_moment_hull_facet_violation"] for r in all_design_rows)
    all_valid = all(r["scientific_metrics_valid"] for r in all_design_rows)
    total_clip_hits = sum(r["coordinate_clip_saturation_count"] for r in all_design_rows)
    max_retry_cases = sum(r["calibration_retry_case_count"] for r in all_design_rows)
    fvl = contrasts["full_vs_lift"]

    checks = {
        "finite_raw_outputs": bool(finite_outputs),
        "reference_bank_finite": bool(
            all(
                np.isfinite(v)
                for v in (
                    reference_bank_diagnostics["mean_interior_bridge_mmd2"],
                    reference_bank_diagnostics["max_interior_bridge_mmd2"],
                    reference_bank_diagnostics["endpoint_bridge_mmd2"],
                    reference_bank_diagnostics["endpoint_mean_l2_error"],
                    reference_bank_diagnostics["endpoint_covariance_fro_error"],
                )
            )
        ),
        "all_population_targets_inside_empirical_moment_hulls": bool(all_hulls_feasible),
        "max_population_target_hull_facet_violation": float(max_hull_violation),
        "empirical_calibration_residual_small": bool(max_cal <= cfg.calibration_accept_tol),
        "all_three_designs_scientifically_valid": bool(all_valid),
        "ess_above_warning_fraction": bool(min_ess >= cfg.ess_warn_fraction),
        "in_domain_base_mass_high": bool(min_domain >= cfg.min_in_domain_base_mass),
        "coordinate_clip_saturation_count": int(total_clip_hits),
        "adaptive_clip_retry_case_count": int(max_retry_cases),
        "full_tc_action_below_lift": (
            bool(fvl["action_reduction_fraction"] > 0.0) if fvl.get("valid", False) else None
        ),
        "calibration_ceiling_sensitivity_stable": (
            None if ceiling_comparison is None else bool(ceiling_comparison["all_designs_stable"])
        ),
        "checkpoint_teacher_free": bool(
            checkpoint_meta["bridge"].get("uses_analytic_A_t") is False
            and checkpoint_meta["bridge"].get("uses_analytic_B_t") is False
            and checkpoint_meta["bridge"].get("uses_analytic_velocity_teacher") is False
        ),
        "no_cnf_used": True,
        "no_old_analytic_reference_called_by_d6": True,
    }


    payload = {
        "stage": "D.6",
        "purpose": (
            "Frozen Lift/Tangent-TC/Full-TC MFSI evaluation under the D.5 "
            "endpoint-trained flow-matching particle reference."
        ),
        "method": (
            "D.5 learned-ODE particle reference + empirical hard I-projection + "
            "smoothed particle weighted-Poisson realization; no CNF and no old "
            "Stage-B analytic reference path."
        ),
        "backend_path": str(Path(backend_path).resolve()),
        "d2_helpers_path": str(Path(d2_path).resolve()),
        "d5_script_path": str(Path(d5_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "physical_parameter_checks": parameter_checks,
        "config": jsonify(cfg),
        "stage_b_scientific_grid": {
            "grid_n": int(stage_b_cfg.grid_n),
            "time_n": int(stage_b_cfg.time_n),
            "alpha_n": int(stage_b_cfg.alpha_n),
            "dx": float(model.dx),
        },
        "reference_bank_diagnostics": reference_bank_diagnostics,
        "designs": designs,
        "contrasts": contrasts,
        "calibration_ceiling_sensitivity": ceiling_sensitivity,
        "calibration_ceiling_comparison": ceiling_comparison,
        "full_tc_bandwidth_check": bandwidth_check,
        "checks": checks,
        "interpretation": [
            "D.6 is the first frozen-design test after removing the old analytic path teacher in D.5.",
            "The learned ODE rollout supplies the empirical reference marginal used by the hard I-projection.",
            "The exact four-component D.5 bridge marginal is used only to validate that the learned ODE reproduces the declared endpoint-defined bridge; it is not the old Stage-B analytic SI.",
            "The MFSI forcing is computed from weighted particle statistics using C lambda_dot = c_dot - E[J Phi u] - Cov(Phi, lambda^T J Phi u).",
            "The weighted-grid Poisson solve remains the deterministic action oracle; neural Poisson learning remains deferred.",
            "Finite/noisy measurements and sensor re-optimization remain off in D.6 so the effect of changing the reference bridge is isolated.",
            "D.6-v3 explicitly checks each population target against the empirical D.5 particle moment hull; official law/action metrics are set to null unless the hard projection and reference-overlap gates pass.",
            "The hard empirical I-projection is solved as the same convex dual as D2, with damped Newton plus L-BFGS-B fallback and reported adaptive coordinate-ceiling retries; only numerical budget changes, never the objective.",
            "The reference preset uses the established 51x51, 21-time-node D2/D3/D4 numerical regime rather than the earlier accidental 39x39 D6 grid.",
            "Worst calibration and worst-lambda states record t, alpha, achieved/target moments, actual max coordinate multiplier, covariance conditioning, ESS, hull margin, and every solver attempt.",
            "An optional --run-ceiling-sensitivity recomputes all three designs at a larger fixed primary coordinate ceiling so action/law stability can be checked before declaring D6 passed.",
        ],
        "elapsed_seconds": float(time.time() - t0),
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    print_summary(payload)
    out = Path(args.output)
    out.write_text(
        json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved D.6 results: {out}")


if __name__ == "__main__":
    main()
