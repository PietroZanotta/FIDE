#!/usr/bin/env python3
"""
Stage D.3: finite/noisy measurements with a learned flow-matching particle reference (NO CNF).

Scientific purpose
------------------
D.2 removed the analytic reference marginal from MFSI by using a reference bank
obtained from flow-matching rollout particles. D.3 composes that learned/sample-
based reference with the validated Stage-C finite-measurement layer while keeping
sensor designs frozen.

The experiment therefore asks a matched question:

    Does the Full-TC law/action advantage survive BOTH
      (i) learned-reference error from flow matching and
      (ii) finite/noisy population measurements?

No CNF is used. There is no learned density, log-Jacobian, score model, or
change-of-variables likelihood.

Pipeline
--------
For each scientific trial alpha and frozen design eta:

  1. draw finite microscopic measurements from the same Stage-B scientific law,
  2. fit the endpoint-anchored GLS quadratic moment curve exactly as in Stage C,
  3. project the fitted beta, only if necessary, onto a COMMON feasible set that
     lies inside the physical moment hull and inside BOTH the analytic-particle
     and learned-FM particle moment hulls at every evaluation time,
  4. run hard empirical MFSI on the analytic-particle and learned-FM reference
     banks using the SAME reconstructed moment curve,
  5. compute law MMD on held-out time nodes and weighted-Poisson action on the
     complete Stage-B trapezoidal time quadrature.

The common feasibility projection is important: noisy measurements should enter
once, through one reconstructed c_hat(t). We do not allow the two reference
branches to silently clip c_hat(t) differently.

Particle MFSI
-------------
D.3 reuses the D.2 particle calculus. For reference particles x_i with base
weights a_i, hard I-projection weights are

    w_i(lambda) propto a_i exp(lambda^T Phi(x_i)),

and the multiplier derivative satisfies

    C lambda_dot
      = c_dot - E_w[J Phi u] - Cov_w(Phi, lambda^T J Phi u).

The sample forcing is

    h_i = lambda_dot^T(Phi_i - E_w Phi)
          + lambda^T J Phi_i u_i
          - E_w[lambda^T J Phi u].

The weighted particles and signed source are rasterized exactly as in D.2 and
passed to the validated Stage-B weighted-Poisson solver. Absolute particle action
remains a particle/grid discretization quantity; the primary learned-reference
comparison is learned-FM particles versus matched analytic particles.

Default reference condition
---------------------------
The reference preset uses the same representative finite-resource condition as
Stage C.3:

    N=100, K=11, detector noise std=0.01,

with 24 common-random-number scientific trials and 8 full-action trials.

Recommended run
---------------
python stage_d3_flow_matching_finite_measurements.py \\
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
    --c2-script ../stage_c/stage_c2_mfsi_matched_action.py \\
    --d0-script stage_d0_flow_matching_reference.py \\
    --d2-script stage_d2_flow_matching_particle_mfsi.py \\
    --checkpoint stage_d0_flow_matching_reference.npz \\
    --preset reference \\
    --output stage_d3_flow_matching_finite_measurements.json
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
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial import ConvexHull, QhullError

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Module loading
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


def autodetect(names: Sequence[str]) -> Path | None:
    here = Path(__file__).resolve().parent
    for name in names:
        for p in (Path(name), here / name):
            if p.exists():
                return p
    return None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class D3Config:
    preset: str = "quick"
    seed: int = 20260812

    trials: int = 4
    action_trials: int = 2
    finite_n: int = 100
    acquisition_k: int = 7
    obs_noise_std: float = 0.01

    # Stage-B/D.2 discretization.
    grid_n: int = 19
    time_n: int = 13
    bank_mode: str = "gauss-hermite"
    gh_order: int = 20
    particles: int = 8192
    rk4_substeps_per_time_interval: int = 8
    kde_bandwidth: float = 0.0
    kde_truncate: float = 4.0

    # Empirical I-projection / finite-measurement safeguards.
    calibration_steps: int = 24
    calibration_tol: float = 2.0e-8
    newton_step_cap: float = 5.0
    lambda_clip: float = 80.0
    variance_floor: float = 1.0e-10
    quadratic_ridge_rel: float = 1.0e-12
    feasibility_margin: float = 0.0

    # Frozen designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)


def preset_d3_config(name: str) -> D3Config:
    if name == "quick":
        return D3Config()
    if name == "reference":
        return D3Config(
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
            calibration_steps=28,
            calibration_tol=1.0e-9,
        )
    if name == "confirm":
        return D3Config(
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
            calibration_steps=32,
            calibration_tol=5.0e-10,
        )
    raise ValueError(name)


# -----------------------------------------------------------------------------
# Utilities
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


def mean_se(values: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), 0
    if x.size == 1:
        return float(x[0]), float("nan"), 1
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(x.size)), int(x.size)


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    keys = sorted({k for r in rows for k, v in r.items() if np.isscalar(v)})
    out: Dict[str, Any] = {}
    for k in keys:
        try:
            vals = [float(r.get(k, float("nan"))) for r in rows]
        except (TypeError, ValueError):
            continue
        m, se, n = mean_se(vals)
        out[k] = {"mean": m, "se": se, "n": n}
    return out


def paired_difference(rows_a: List[Dict[str, float]], rows_b: List[Dict[str, float]], key: str):
    a = np.asarray([r.get(key, np.nan) for r in rows_a], dtype=np.float64)
    b = np.asarray([r.get(key, np.nan) for r in rows_b], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    m, se, n = mean_se(d)
    return {
        "mean_difference_a_minus_b": m,
        "se_difference": se,
        "n": n,
        "a_better_fraction": float(np.mean(d < 0.0)) if n else float("nan"),
    }


def reduction_fraction(rows_num: List[Dict[str, float]], rows_den: List[Dict[str, float]], key: str):
    num = np.asarray([r.get(key, np.nan) for r in rows_num], dtype=np.float64)
    den = np.asarray([r.get(key, np.nan) for r in rows_den], dtype=np.float64)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-14)
    num = num[mask]
    den = den[mask]
    if num.size == 0:
        return {
            "ratio_of_means_reduction": float("nan"),
            "mean_paired_reduction": float("nan"),
            "se_paired_reduction": float("nan"),
            "n": 0,
        }
    paired = 1.0 - num / den
    pm, pse, n = mean_se(paired)
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(num) / np.mean(den)),
        "mean_paired_reduction": pm,
        "se_paired_reduction": pse,
        "n": n,
    }


def trap_average(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    if mask is not None:
        w *= np.asarray(mask, dtype=np.float64)
    z = float(np.sum(w))
    if z <= 0.0:
        return float("nan")
    return float(np.sum(w * v) / z)


# -----------------------------------------------------------------------------
# Measurement covariance only: no analytic-reference MFSI is used here
# -----------------------------------------------------------------------------


class MeasurementCovariance:
    def __init__(self, model):
        self.model = model
        self.population_cov_jit = jax.jit(self._population_cov)

    def _population_cov(self, t, alpha, eta):
        phi, _ = self.model.sensor_fields(eta)
        _, pmass = self.model.external_q_mass(t, alpha)
        moment = jnp.sum(phi * pmass[None, ...], axis=(1, 2))
        centered = phi - moment[:, None, None]
        return jnp.einsum("myx,nyx,yx->mn", centered, centered, pmass)


# -----------------------------------------------------------------------------
# Common physical + particle feasibility geometry for the quadratic beta
# -----------------------------------------------------------------------------


def hull_equations_from_points(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.shape[0] < 3 or np.linalg.matrix_rank(pts - np.mean(pts, axis=0)) < 2:
        raise RuntimeError("Particle moment set is rank-deficient; hard 2D calibration is not identifiable.")
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise RuntimeError(f"Could not construct particle moment hull: {exc}") from exc
    return np.asarray(hull.equations, dtype=np.float64)


def build_joint_beta_constraints(model, evaluator, d2, c2, eta: np.ndarray, margin: float):
    """Build A beta <= b feasible for physical, analytic-particle and FM-particle hulls.

    Hulls are time dependent for the particle references. The physical sensor hull
    is common across time. The same beta is constrained against every branch so
    one finite-measurement curve is used downstream in both matched controls.
    """
    physical_eq, physical_meta = c2.sensor_moment_hull_equations(model, eta)
    if physical_eq is None:
        raise RuntimeError(f"Physical sensor moment hull unavailable: {physical_meta}")

    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    # Endpoints do not depend on scientific alpha in this benchmark.
    alpha_probe = jnp.asarray(0.5 * (model.cfg.alpha_min + model.cfg.alpha_max), dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_probe, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_probe, eta_j), dtype=np.float64)

    A_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    time_meta = []

    for kt, t in enumerate(times):
        eqs = [("physical", np.asarray(physical_eq, dtype=np.float64))]
        branch_info = {}
        for label, nodes in (
            ("analytic_particle", evaluator.analytic_nodes),
            ("learned_fm_particle", evaluator.learned_nodes),
        ):
            x_all = np.asarray(nodes[kt], dtype=np.float64)
            mask = d2.in_domain_mask(model, x_all)
            x = x_all[mask]
            phi, _ = d2.sensor_particle_fields(model, eta, x)
            eq = hull_equations_from_points(phi)
            eqs.append((label, eq))
            branch_info[label] = {
                "in_domain_particles": int(x.shape[0]),
                "facets": int(eq.shape[0]),
            }

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
                f"Exact endpoint moment is outside a particle reference hull at t={t:.3f}; "
                f"max violation={max_endpoint_violation:.3e}. Increase the D.2 bank resolution."
            )
        time_meta.append({
            "t": float(t),
            "max_endpoint_violation": float(max(0.0, max_endpoint_violation)),
            **branch_info,
        })

    A = np.concatenate(A_rows, axis=0) if A_rows else np.zeros((0, 2), dtype=np.float64)
    b = np.concatenate(b_rows, axis=0) if b_rows else np.zeros((0,), dtype=np.float64)

    # Verify that the common intersection is nonempty independently of any noisy trial.
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
                "Physical/analytic-particle/learned-particle moment hull intersection is empty. "
                "Increase the reference-bank resolution or improve the FM model before D.3."
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


def fit_quadratic_bridge_joint_feasible(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    V_obs: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    t_eval: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    feasible_beta: np.ndarray,
    ridge_rel: float,
    variance_floor: float,
):
    """GLS quadratic fit projected onto a precomputed common beta polytope."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    V_obs = np.asarray(V_obs, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m = int(c0.size)
    H_data = np.zeros((m, m), dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    used = 0
    for t, y, V in zip(t_obs, y_obs, V_obs):
        z = float(t * (1.0 - t))
        if abs(z) < 1.0e-14:
            continue
        bridge = (1.0 - t) * c0 + t * c1
        resid = y - bridge
        Vreg = 0.5 * (V + V.T) + float(variance_floor) * np.eye(m)
        Vinv = np.linalg.inv(Vreg)
        H_data += (z * z) * Vinv
        g += z * (Vinv @ resid)
        used += 1
    if used == 0:
        raise ValueError("Need at least one interior acquisition time for quadratic GLS.")

    scale = max(float(np.trace(H_data)) / max(m, 1), 1.0)
    Hreg = H_data + (float(ridge_rel) * scale) * np.eye(m)
    beta_u = np.linalg.solve(Hreg, g)
    Hinv = np.linalg.inv(Hreg)
    beta_cov = 0.5 * (Hinv @ H_data @ Hinv.T + (Hinv @ H_data @ Hinv.T).T)

    beta = beta_u.copy()
    max_viol = float(max(0.0, np.max(A @ beta_u - b))) if A.shape[0] else 0.0
    projection_active = max_viol > 1.0e-10
    projection_norm = 0.0
    constraint_success = True

    if projection_active:
        def obj(bb):
            return 0.5 * float(bb @ Hreg @ bb) - float(g @ bb)

        def jac(bb):
            return Hreg @ bb - g

        x0 = beta_u if np.all(A @ beta_u <= b + 1.0e-10) else np.asarray(feasible_beta, dtype=np.float64)
        lc = LinearConstraint(A, -np.inf * np.ones_like(b), b)
        sol = minimize(
            obj,
            x0,
            jac=jac,
            constraints=[lc],
            method="SLSQP",
            options={"ftol": 1.0e-12, "maxiter": 500, "disp": False},
        )
        constraint_success = bool(sol.success) and np.all(A @ sol.x <= b + 1.0e-7)
        if not constraint_success:
            raise RuntimeError(
                "Common finite-measurement feasibility projection failed; refusing to use branch-specific clipping."
            )
        beta = np.asarray(sol.x, dtype=np.float64)
        projection_norm = float(np.linalg.norm(beta - beta_u))

    z = t_eval * (1.0 - t_eval)
    c = (1.0 - t_eval[:, None]) * c0[None, :] + t_eval[:, None] * c1[None, :] + z[:, None] * beta[None, :]
    cdot = (c1 - c0)[None, :] + (1.0 - 2.0 * t_eval[:, None]) * beta[None, :]

    return {
        "beta": beta,
        "beta_unconstrained": beta_u,
        "beta_cov": beta_cov,
        "c": c,
        "cdot": cdot,
        "feasibility_projection_active": bool(projection_active),
        "feasibility_projection_norm": float(projection_norm),
        "constraint_solver_success": bool(constraint_success),
        "max_unconstrained_hull_violation": float(max_viol),
    }


# -----------------------------------------------------------------------------
# One particle branch and one target moment curve
# -----------------------------------------------------------------------------


def evaluate_particle_curve(
    evaluator,
    d2,
    eta: np.ndarray,
    alpha: float,
    target_curve: np.ndarray,
    target_cdot: np.ndarray,
    branch: str,
    heldout_mask: np.ndarray,
    compute_action: bool,
):
    model = evaluator.model
    cfg = evaluator.cfg
    if branch == "analytic_particle":
        xnodes = evaluator.analytic_nodes
        unodes = evaluator.analytic_u_nodes
    elif branch == "learned_fm_particle":
        xnodes = evaluator.learned_nodes
        unodes = evaluator.learned_u_nodes
    else:
        raise ValueError(branch)

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
    ess = np.zeros(len(times), dtype=np.float64)
    cal = np.zeros(len(times), dtype=np.float64)
    lam = np.zeros(len(times), dtype=np.float64)
    min_cov = np.zeros(len(times), dtype=np.float64)
    poisson = np.full(len(times), np.nan, dtype=np.float64)
    grid_moment_error = np.zeros(len(times), dtype=np.float64)

    phi_grid, _ = model.sensor_fields(eta_j)
    phi_grid_np = np.asarray(phi_grid, dtype=np.float64)
    lam_warm = np.zeros(2, dtype=np.float64)

    for kt, t in enumerate(times):
        x_all = np.asarray(xnodes[kt], dtype=np.float64)
        u_all = np.asarray(unodes[kt], dtype=np.float64)
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        u = u_all[mask]
        base_w = np.asarray(evaluator.base_w[mask], dtype=np.float64)
        base_w /= max(float(np.sum(base_w)), 1.0e-300)
        phi, grad_phi = d2.sensor_particle_fields(model, eta, x)

        st = d2.particle_mfsi_state(
            phi=phi,
            grad_phi=grad_phi,
            u=u,
            base_w=base_w,
            target=np.asarray(target_curve[kt], dtype=np.float64),
            c_dot=np.asarray(target_cdot[kt], dtype=np.float64),
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
        law[kt] = float(
            model.gaussian_mmd2_mass(jnp.asarray(qmass, dtype=jnp.float64), p_mass)
        )

        grid_moment = np.sum(phi_grid_np * qmass[None, ...], axis=(1, 2))
        grid_moment_error[kt] = float(np.linalg.norm(grid_moment - target_curve[kt]))
        ess[kt] = float(st["diagnostics"]["ess_fraction"])
        cal[kt] = float(st["diagnostics"]["residual"])
        lam[kt] = float(st["diagnostics"]["lambda_norm"])
        min_cov[kt] = float(st["diagnostics"]["min_cov_eig"])

        if compute_action:
            full, _, pres, _, _ = evaluator.poisson_jit(
                jnp.asarray(q, dtype=jnp.float64),
                jnp.asarray(h_grid, dtype=jnp.float64),
            )
            action[kt] = float(full)
            poisson[kt] = float(pres)

    return {
        "heldout_mmd2": trap_average(law, tw, heldout_mask),
        "all_interior_mmd2": trap_average(law, tw, interior),
        "full_action": float(np.sum(tw * action)) if compute_action else float("nan"),
        "min_ess_fraction": float(np.min(ess)),
        "mean_ess_fraction": float(np.sum(tw * ess)),
        "max_calibration_residual": float(np.max(cal)),
        "max_lambda_norm": float(np.max(lam)),
        "min_projected_cov_eig": float(np.min(min_cov)),
        "max_grid_moment_error_after_kde": float(np.max(grid_moment_error)),
        "max_poisson_relative_residual": float(np.nanmax(poisson)) if compute_action else float("nan"),
    }


# -----------------------------------------------------------------------------
# Trial evaluation
# -----------------------------------------------------------------------------


def evaluate_design_trial(
    model,
    evaluator,
    d2,
    c2,
    measurement_cov: MeasurementCovariance,
    eta: np.ndarray,
    shared,
    acq_idx: np.ndarray,
    heldout_mask: np.ndarray,
    cfg: D3Config,
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

    # Endpoint identities are alpha independent in this benchmark; still verify
    # against the constraints precomputed before the trial loop.
    if np.linalg.norm(c0 - joint_constraints["c0"]) > 1.0e-10 or np.linalg.norm(c1 - joint_constraints["c1"]) > 1.0e-10:
        raise RuntimeError("Unexpected alpha-dependent endpoints; D.3 common feasibility geometry is invalid.")

    curve = fit_quadratic_bridge_joint_feasible(
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

    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    results: Dict[str, Any] = {}
    for branch in ("analytic_particle", "learned_fm_particle"):
        finite = evaluate_particle_curve(
            evaluator=evaluator,
            d2=d2,
            eta=eta,
            alpha=float(shared.alpha),
            target_curve=np.asarray(curve["c"], dtype=np.float64),
            target_cdot=np.asarray(curve["cdot"], dtype=np.float64),
            branch=branch,
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
            branch=branch,
            heldout_mask=heldout_mask,
            compute_action=compute_action,
        )
        results[branch] = {
            "finite": finite,
            "exact_population": exact,
            "measurement_delta_heldout_mmd2": float(finite["heldout_mmd2"] - exact["heldout_mmd2"]),
            "measurement_action_inflation": float(finite["full_action"] / exact["full_action"] - 1.0)
                if compute_action else float("nan"),
            "measurement_action_excess": float(finite["full_action"] - exact["full_action"])
                if compute_action else float("nan"),
        }

    # Flatten primary metrics for simple across-trial summaries.
    a = results["analytic_particle"]
    l = results["learned_fm_particle"]
    row = {
        "alpha": float(shared.alpha),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((y_acq - exact_acq) ** 2))),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum((np.asarray(curve["c"])[interior] - exact_curve[interior]) ** 2, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(np.asarray(curve["c"])[interior] - exact_curve[interior], axis=1))),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        "beta_cov_trace": float(np.trace(np.asarray(curve["beta_cov"], dtype=np.float64))),

        "analytic_finite_heldout_mmd2": a["finite"]["heldout_mmd2"],
        "analytic_exact_heldout_mmd2": a["exact_population"]["heldout_mmd2"],
        "analytic_measurement_delta_mmd2": a["measurement_delta_heldout_mmd2"],
        "analytic_finite_action": a["finite"]["full_action"],
        "analytic_exact_action": a["exact_population"]["full_action"],
        "analytic_measurement_action_inflation": a["measurement_action_inflation"],
        "analytic_finite_min_ess": a["finite"]["min_ess_fraction"],
        "analytic_finite_max_calibration_resid": a["finite"]["max_calibration_residual"],

        "learned_finite_heldout_mmd2": l["finite"]["heldout_mmd2"],
        "learned_exact_heldout_mmd2": l["exact_population"]["heldout_mmd2"],
        "learned_measurement_delta_mmd2": l["measurement_delta_heldout_mmd2"],
        "learned_finite_action": l["finite"]["full_action"],
        "learned_exact_action": l["exact_population"]["full_action"],
        "learned_measurement_action_inflation": l["measurement_action_inflation"],
        "learned_finite_min_ess": l["finite"]["min_ess_fraction"],
        "learned_finite_max_calibration_resid": l["finite"]["max_calibration_residual"],

        "learned_minus_analytic_finite_mmd2": float(l["finite"]["heldout_mmd2"] - a["finite"]["heldout_mmd2"]),
        "learned_minus_analytic_exact_mmd2": float(l["exact_population"]["heldout_mmd2"] - a["exact_population"]["heldout_mmd2"]),
        "learned_minus_analytic_measurement_delta_mmd2": float(l["measurement_delta_heldout_mmd2"] - a["measurement_delta_heldout_mmd2"]),
        "learned_minus_analytic_finite_action": float(l["finite"]["full_action"] - a["finite"]["full_action"])
            if compute_action else float("nan"),
        "learned_minus_analytic_finite_action_relative": float(l["finite"]["full_action"] / a["finite"]["full_action"] - 1.0)
            if compute_action else float("nan"),
    }
    return row, results


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def build_contrasts(rows: Dict[str, List[Dict[str, float]]]):
    out: Dict[str, Any] = {}
    for branch in ("analytic", "learned"):
        law_key = f"{branch}_finite_heldout_mmd2"
        degr_key = f"{branch}_measurement_delta_mmd2"
        action_key = f"{branch}_finite_action"
        exact_action_key = f"{branch}_exact_action"
        out[f"{branch}_full_vs_lift_finite_law"] = paired_difference(rows["full"], rows["lift"], law_key)
        out[f"{branch}_full_vs_lift_measurement_degradation"] = paired_difference(rows["full"], rows["lift"], degr_key)
        out[f"{branch}_full_vs_lift_finite_action_reduction"] = reduction_fraction(rows["full"], rows["lift"], action_key)
        out[f"{branch}_full_vs_lift_exact_action_reduction"] = reduction_fraction(rows["full"], rows["lift"], exact_action_key)

    out["full_vs_lift_differential_learning_effect_on_finite_law"] = paired_difference(
        rows["full"], rows["lift"], "learned_minus_analytic_finite_mmd2"
    )
    out["full_vs_lift_differential_learning_effect_on_measurement_degradation"] = paired_difference(
        rows["full"], rows["lift"], "learned_minus_analytic_measurement_delta_mmd2"
    )
    out["full_vs_lift_differential_learning_effect_on_finite_action"] = paired_difference(
        rows["full"], rows["lift"], "learned_minus_analytic_finite_action_relative"
    )
    return out


def build_attribution_diagnostics(rows: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
    """Compact per-trial diagnostics for D3 attribution / outlier analysis."""
    projection_cases = []
    top_action_cases: Dict[str, List[Dict[str, float]]] = {}
    correlations: Dict[str, Dict[str, float]] = {}

    for design, rr in rows.items():
        for trial_idx, r in enumerate(rr):
            if float(r.get("feasibility_projection_active", 0.0)) > 0.5:
                projection_cases.append({
                    "design": design,
                    "trial_index_1based": int(trial_idx + 1),
                    "alpha": float(r.get("alpha", np.nan)),
                    "projection_norm": float(r.get("feasibility_projection_norm", np.nan)),
                    "max_unconstrained_hull_violation": float(r.get("max_unconstrained_hull_violation", np.nan)),
                    "quadratic_moment_rmse": float(r.get("quadratic_moment_rmse", np.nan)),
                    "learned_finite_heldout_mmd2": float(r.get("learned_finite_heldout_mmd2", np.nan)),
                    "learned_measurement_delta_mmd2": float(r.get("learned_measurement_delta_mmd2", np.nan)),
                    "learned_finite_action": float(r.get("learned_finite_action", np.nan)),
                    "learned_exact_action": float(r.get("learned_exact_action", np.nan)),
                    "learned_measurement_action_inflation": float(r.get("learned_measurement_action_inflation", np.nan)),
                    "learned_finite_min_ess": float(r.get("learned_finite_min_ess", np.nan)),
                    "learned_finite_max_calibration_resid": float(r.get("learned_finite_max_calibration_resid", np.nan)),
                })

        action_rows = []
        for trial_idx, r in enumerate(rr):
            val = float(r.get("learned_finite_action", np.nan))
            if np.isfinite(val):
                action_rows.append({
                    "trial_index_1based": int(trial_idx + 1),
                    "alpha": float(r.get("alpha", np.nan)),
                    "learned_finite_action": val,
                    "learned_exact_action": float(r.get("learned_exact_action", np.nan)),
                    "measurement_action_inflation": float(r.get("learned_measurement_action_inflation", np.nan)),
                    "projection_active": bool(float(r.get("feasibility_projection_active", 0.0)) > 0.5),
                    "projection_norm": float(r.get("feasibility_projection_norm", np.nan)),
                    "calibration_residual": float(r.get("learned_finite_max_calibration_resid", np.nan)),
                    "min_ess": float(r.get("learned_finite_min_ess", np.nan)),
                })
        action_rows.sort(key=lambda z: z["learned_finite_action"], reverse=True)
        top_action_cases[design] = action_rows[:5]

        # Simple Pearson diagnostics. These are descriptive only; NaNs/constant columns are skipped.
        candidates = {
            "measurement_delta_vs_acquisition_rmse": ("learned_measurement_delta_mmd2", "acquisition_mean_rmse"),
            "measurement_delta_vs_quadratic_rmse": ("learned_measurement_delta_mmd2", "quadratic_moment_rmse"),
            "measurement_delta_vs_beta_cov_trace": ("learned_measurement_delta_mmd2", "beta_cov_trace"),
            "action_inflation_vs_quadratic_rmse": ("learned_measurement_action_inflation", "quadratic_moment_rmse"),
            "action_inflation_vs_projection_norm": ("learned_measurement_action_inflation", "feasibility_projection_norm"),
            "action_inflation_vs_calibration_residual": ("learned_measurement_action_inflation", "learned_finite_max_calibration_resid"),
            "action_inflation_vs_min_ess": ("learned_measurement_action_inflation", "learned_finite_min_ess"),
        }
        correlations[design] = {}
        for label, (yk, xk) in candidates.items():
            x = np.asarray([float(r.get(xk, np.nan)) for r in rr], dtype=np.float64)
            y = np.asarray([float(r.get(yk, np.nan)) for r in rr], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) >= 3 and np.std(x[mask]) > 0.0 and np.std(y[mask]) > 0.0:
                correlations[design][label] = float(np.corrcoef(x[mask], y[mask])[0, 1])
            else:
                correlations[design][label] = float("nan")

    return {
        "projection_cases": projection_cases,
        "projection_case_count": int(len(projection_cases)),
        "top_learned_action_cases": top_action_cases,
        "descriptive_correlations": correlations,
    }


def print_summary(payload: Dict[str, Any]):
    cond = payload["condition"]
    print("\n" + "=" * 104)
    print("Stage D.3 — learned FM particle reference + finite/noisy measurements (NO CNF)")
    print("=" * 104)
    print(
        f"N={cond['N']}, K={cond['K']}, detector noise std={cond['obs_noise_std']:.4f}, "
        f"trials={cond['trials']} (action={cond['action_trials']})"
    )
    print("-" * 104)
    for name, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        s = payload["design_summary"][name]
        print(
            f"{label:10s} learned finite law={s['learned_finite_heldout_mmd2']['mean']:.8f} "
            f"± {s['learned_finite_heldout_mmd2']['se']:.2e} | "
            f"meas Δ={s['learned_measurement_delta_mmd2']['mean']:+.3e} | "
            f"FM-vs-exact-particle law={s['learned_minus_analytic_finite_mmd2']['mean']:+.3e}"
        )
        print(
            f"{'':10s} learned finite A={s['learned_finite_action']['mean']:.3f} "
            f"± {s['learned_finite_action']['se']:.2f} | "
            f"A meas infl.={100*s['learned_measurement_action_inflation']['mean']:+.2f}% | "
            f"ESSmin(mean)={s['learned_finite_min_ess']['mean']:.3f} | "
            f"calmax(mean)={s['learned_finite_max_calibration_resid']['mean']:.2e}"
        )
    print("-" * 104)
    c = payload["contrasts"]
    for branch, label in (("analytic", "analytic particles"), ("learned", "learned FM particles")):
        law = c[f"{branch}_full_vs_lift_finite_law"]
        deg = c[f"{branch}_full_vs_lift_measurement_degradation"]
        act = c[f"{branch}_full_vs_lift_finite_action_reduction"]
        exa = c[f"{branch}_full_vs_lift_exact_action_reduction"]
        print(
            f"Full-Lift [{label}]: finite law Δ={law['mean_difference_a_minus_b']:+.3e} "
            f"± {law['se_difference']:.2e} | differential measurement degradation={deg['mean_difference_a_minus_b']:+.3e} "
            f"± {deg['se_difference']:.2e}"
        )
        print(
            f"{'':28s} exact-action reduction={100*exa['ratio_of_means_reduction']:+.2f}% | "
            f"finite-action reduction={100*act['ratio_of_means_reduction']:+.2f}%"
        )
    proj = payload["projection_summary"]
    print(
        f"Common feasibility projection active in {100*proj['active_fraction']:.1f}% of design-trials; "
        f"max projection norm={proj['max_projection_norm']:.3e}."
    )
    attr = payload.get("attribution_diagnostics", {})
    cases = attr.get("projection_cases", [])
    if cases:
        print("Projected cases:")
        for c in cases:
            print(
                f"  {c['design']:8s} trial={c['trial_index_1based']:2d} alpha={c['alpha']:.5f} "
                f"proj={c['projection_norm']:.3e} cal={c['learned_finite_max_calibration_resid']:.3e} "
                f"ESS={c['learned_finite_min_ess']:.3f}"
            )
    for design in ("lift", "tangent", "full"):
        top = attr.get("top_learned_action_cases", {}).get(design, [])
        if top:
            r = top[0]
            print(
                f"Largest learned finite action [{design}]: trial={r['trial_index_1based']} "
                f"alpha={r['alpha']:.5f} A={r['learned_finite_action']:.3f} "
                f"infl={100*r['measurement_action_inflation']:+.2f}% "
                f"projected={r['projection_active']}"
            )
    print("=" * 104)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--c2-script", type=str, default=None)
    p.add_argument("--d0-script", type=str, default=None)
    p.add_argument("--d2-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d3_flow_matching_finite_measurements.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--N", dest="finite_n", type=int, default=None)
    p.add_argument("--K", dest="acquisition_k", type=int, default=None)
    p.add_argument(
        "--noise", "--noise-std", dest="obs_noise_std", type=float, default=None,
        help="Additive detector-noise standard deviation (aliases: --noise, --noise-std).",
    )
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--time-n", type=int, default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)
    p.add_argument("--feasibility-margin", type=float, default=None)
    return p


def main():
    wall0 = time.time()
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect(
        ["stage_b2_transport_conditioned_design.py", "stage_b2_transport_conditioned_design(4).py"]
    )
    c2_path = Path(args.c2_script) if args.c2_script else autodetect(
        ["stage_c2_mfsi_matched_action.py", "stage_c2_mfsi_matched_action(1).py"]
    )
    d0_path = Path(args.d0_script) if args.d0_script else autodetect(["stage_d0_flow_matching_reference.py"])
    d2_path = Path(args.d2_script) if args.d2_script else autodetect(["stage_d2_flow_matching_particle_mfsi.py"])
    if backend_path is None:
        raise FileNotFoundError("Pass --backend /path/to/stage_b2_transport_conditioned_design.py")
    if c2_path is None:
        raise FileNotFoundError("Pass --c2-script /path/to/stage_c2_mfsi_matched_action.py")
    if d0_path is None:
        raise FileNotFoundError("Pass --d0-script /path/to/stage_d0_flow_matching_reference.py")
    if d2_path is None:
        raise FileNotFoundError("Pass --d2-script /path/to/stage_d2_flow_matching_particle_mfsi.py")

    backend = load_module(backend_path, "stage_b2_backend_d3")
    c2 = load_module(c2_path, "stage_c2_backend_d3")
    d0 = load_module(d0_path, "stage_d0_backend_d3")
    d2 = load_module(d2_path, "stage_d2_backend_d3")
    params, checkpoint_meta = d0.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d3_config(args.preset)
    overrides: Dict[str, Any] = {}
    for key in (
        "seed", "trials", "action_trials", "finite_n", "acquisition_k", "obs_noise_std",
        "grid_n", "time_n", "gh_order", "kde_bandwidth", "feasibility_margin",
    ):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    if args.rk4_substeps is not None:
        overrides["rk4_substeps_per_time_interval"] = int(args.rk4_substeps)
    cfg = dataclasses.replace(cfg, **overrides)

    if cfg.action_trials > cfg.trials:
        raise ValueError("action_trials cannot exceed trials")
    if cfg.finite_n < 2:
        raise ValueError("N must be >= 2")
    if cfg.acquisition_k < 3 or cfg.acquisition_k >= cfg.time_n:
        raise ValueError("K must satisfy 3 <= K < time_n")
    if cfg.feasibility_margin < 0.0:
        raise ValueError("feasibility_margin must be nonnegative")

    base = backend.preset_config("quick" if args.preset == "quick" else "reference")
    stage_b_cfg = dataclasses.replace(base, grid_n=int(cfg.grid_n), time_n=int(cfg.time_n))
    model = backend.StageB(stage_b_cfg)
    teacher = d0.AnalyticReferenceTeacher(stage_b_cfg)

    # Match D.2 implementation parameters exactly.
    d2_cfg = d2.D2Config(
        preset=cfg.preset,
        seed=int(cfg.seed),
        lift_design_deg=cfg.lift_design_deg,
        tangent_design_deg=cfg.tangent_design_deg,
        full_design_deg=cfg.full_design_deg,
        bank_mode=cfg.bank_mode,
        gh_order=int(cfg.gh_order),
        particles=int(cfg.particles),
        rk4_substeps_per_time_interval=int(cfg.rk4_substeps_per_time_interval),
        calibration_steps=int(cfg.calibration_steps),
        calibration_tol=float(cfg.calibration_tol),
        newton_step_cap=float(cfg.newton_step_cap),
        lambda_clip=float(cfg.lambda_clip),
        kde_bandwidth=float(cfg.kde_bandwidth),
        kde_truncate=float(cfg.kde_truncate),
    )
    evaluator = d2.ParticleReferenceMFSI(model, d0, params, teacher, d2_cfg)
    measurement_cov = MeasurementCovariance(model)

    # Check checkpoint/physical compatibility.
    cp_phys = checkpoint_meta.get("physical_system", {})
    for key, current in (("r", stage_b_cfg.r), ("sigma", stage_b_cfg.sigma), ("kappa", stage_b_cfg.kappa)):
        saved = float(cp_phys.get(key, current))
        if abs(saved - float(current)) > 1.0e-12:
            raise ValueError(f"D0 checkpoint {key}={saved} != Stage-B {current}")

    designs = {
        "lift": np.radians(np.asarray(cfg.lift_design_deg, dtype=np.float64)),
        "tangent": np.radians(np.asarray(cfg.tangent_design_deg, dtype=np.float64)),
        "full": np.radians(np.asarray(cfg.full_design_deg, dtype=np.float64)),
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

    print("=" * 104)
    print("Stage D.3 — learned FM particle reference + finite/noisy measurements (NO CNF)")
    print("=" * 104)
    print(f"Backend       : {Path(backend_path).resolve()}")
    print(f"D0 checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"FM bank       : {d2_cfg.bank_mode}, GH order={d2_cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Condition     : N={cfg.finite_n}, K={cfg.acquisition_k}, noise std={cfg.obs_noise_std}")
    print(f"Trials        : {cfg.trials} (full action: {cfg.action_trials})")
    print(f"Held-out idx  : {heldout_idx.tolist()}")

    print("\nBuilding common physical/particle moment-feasibility intersections...", flush=True)
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

    rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in designs}
    nested_details: Dict[str, List[Dict[str, Any]]] = {k: [] for k in designs}

    for trial, shared in enumerate(trial_bank):
        do_action = trial < int(cfg.action_trials)
        for name, eta in designs.items():
            row, detail = evaluate_design_trial(
                model=model,
                evaluator=evaluator,
                d2=d2,
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
            nested_details[name].append(detail)
        print(f"  trial {trial + 1:2d}/{cfg.trials} alpha={shared.alpha:.5f} action={do_action}", flush=True)

    design_summary = {name: summarize_rows(rr) for name, rr in rows.items()}
    contrasts = build_contrasts(rows)

    all_rows = [r for rr in rows.values() for r in rr]
    projection_active = np.asarray([r["feasibility_projection_active"] for r in all_rows], dtype=np.float64)
    projection_norm = np.asarray([r["feasibility_projection_norm"] for r in all_rows], dtype=np.float64)
    projection_summary = {
        "active_fraction": float(np.mean(projection_active)),
        "max_projection_norm": float(np.max(projection_norm)),
        "mean_projection_norm": float(np.mean(projection_norm)),
    }
    attribution_diagnostics = build_attribution_diagnostics(rows)

    payload = {
        "stage": "D.3 learned flow-matching particle-reference MFSI with finite/noisy measurements; no CNF",
        "created_unix_time": time.time(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend_path": str(Path(backend_path).resolve()),
        "c2_script_path": str(Path(c2_path).resolve()),
        "d0_script_path": str(Path(d0_path).resolve()),
        "d2_script_path": str(Path(d2_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "stage_d3_config": jsonify(cfg),
        "stage_b_config": jsonify(stage_b_cfg),
        "stage_d2_config": jsonify(d2_cfg),
        "condition": {
            "N": int(cfg.finite_n),
            "K": int(cfg.acquisition_k),
            "obs_noise_std": float(cfg.obs_noise_std),
            "trials": int(cfg.trials),
            "action_trials": int(cfg.action_trials),
            "acquisition_indices": acq_idx.tolist(),
            "acquisition_times": np.asarray(model.times, dtype=np.float64)[acq_idx].tolist(),
            "heldout_indices": heldout_idx.tolist(),
            "heldout_times": np.asarray(model.times, dtype=np.float64)[heldout_idx].tolist(),
        },
        "reference_bank_diagnostics": evaluator.reference_bank_diagnostics(),
        "joint_feasibility": {
            name: {
                "constraint_count": int(joint[name]["A"].shape[0]),
                "physical_hull_metadata": joint[name]["physical_hull_metadata"],
                "time_metadata": joint[name]["time_metadata"],
            }
            for name in designs
        },
        "design_summary": design_summary,
        "contrasts": contrasts,
        "projection_summary": projection_summary,
        "attribution_diagnostics": attribution_diagnostics,
        "trial_rows": rows,
        "trial_branch_details": nested_details,
        "interpretation_notes": [
            "The reference preset uses grid_n=51 and time_n=21 so attribution runs remain on the same fine D2/D3 numerical regime used for the reported reference result.",
            "No CNF density is reconstructed. The learned reference marginal is represented only by FM rollout particles.",
            "Finite measurement uncertainty enters once through the endpoint-anchored GLS moment trajectory, following the validated Stage-C acquisition model.",
            "The same finite-measurement curve is used for analytic-particle and learned-FM particle branches.",
            "If the noisy unconstrained curve is not hard-calibratable by both particle branches, beta is projected onto their common time-dependent moment-hull intersection together with the physical moment hull.",
            "Exact-population and finite-measurement quantities are evaluated on the same alpha trial and the same particle branch, so measurement degradation is a paired consequence of finite data.",
            "learned-FM versus analytic-particle differences use the same quadrature bank, empirical I-projection, rasterization and Poisson solver, isolating learned-reference error from particle/grid discretization as far as this controlled benchmark permits.",
            "Full action uses the complete Stage-B trapezoidal time quadrature, including half-weighted endpoints.",
            "Primary design comparisons remain frozen Lift, Tangent-TC and Full-TC; D.3 does not yet re-optimize sensor angles.",
        ],
        "wall_seconds": float(time.time() - wall0),
    }

    print_summary(payload)
    out = Path(args.output)
    out.write_text(json.dumps(jsonify(payload), indent=2, allow_nan=True) + "\n")
    print(f"Saved diagnostics: {out.resolve()}")


if __name__ == "__main__":
    main()
