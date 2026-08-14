#!/usr/bin/env python3
"""
Stage D.4: finite-resource sensor-design oracle with a learned flow-matching
particle reference (NO CNF).

Purpose
-------
D.0 learned the Stage-B reference velocity by flow matching.
D.2 replaced the analytic reference marginal by FM rollout particles.
D.3 composed that learned particle reference with finite/noisy moment measurements.
D.4 now lets the physically admissible two-sensor design eta move again.

The oracle is deliberately discrete and lexicographic, matching Stage C.3:

  1. Learned-FM population-law sufficiency

       L_FM(eta) <= (1 + tau_L) min_eta L_FM(eta).

  2. Learned-FM finite-resource law sufficiency, on common random numbers

       R_FM,N(eta) <= (1 + tau_R) min_{population-feasible eta} R_FM,N(eta).

  3. Among designs passing both law screens, minimize the learned-FM
     finite-resource weighted-Poisson action A_FM,N(eta).

A completely untouched validation bank is used only after the design has been
selected.  Lift, Tangent-TC and Full-TC are inserted explicitly into the
candidate set and are always included in post-selection validation.

No CNF density, score model, log-Jacobian or learned likelihood is used.
The learned reference marginal is represented only by a deterministic weighted
Gauss-Hermite initial bank pushed through the D.0 flow-matching ODE.  Hard
moment projection, lambda_dot and forcing are computed from weighted particle
statistics exactly as in D.2/D.3.  Particle q and q h are deposited on the
existing Stage-B grid only for MMD and the deterministic weighted-Poisson
realization.

Finite measurement model
------------------------
At the Stage-C acquisition times, N microscopic states are observed and the
sensor mean receives optional additive detector noise.  The resulting noisy
measurements are fit by the same endpoint-anchored GLS quadratic bridge used in
Stage C/D.3.  Measurement uncertainty enters once, through this fitted curve.
For each candidate design, the fitted beta is constrained to the intersection
of

  * the physical sensor moment hull, and
  * the time-dependent learned-FM particle moment hull.

Thus the hard empirical MFSI calibration is never rescued by branch-specific
clipping after the fact.

Numerical validity
------------------
A candidate is considered evaluable only if empirical hard calibration remains
below the declared population/finite calibration thresholds and relative ESS remains
above min_ess_fraction.  The finite-measurement threshold is deliberately looser
because noisy targets can approach the empirical moment-hull boundary; the attained
residual is always reported and is not part of the objective.
These are numerical/reference-overlap gates, not design objectives, and are
reported explicitly.  If a design is rejected by one of these gates, that is
recorded rather than silently regularized.

Reference run
-------------
python stage_d4_flow_matching_finite_resource_design_oracle.py \\
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
    --c2-script ../stage_c/stage_c2_mfsi_matched_action.py \\
    --d0-script stage_d0_flow_matching_reference.py \\
    --d2-script stage_d2_flow_matching_particle_mfsi.py \\
    --d3-script stage_d3_flow_matching_finite_measurements.py \\
    --checkpoint stage_d0_flow_matching_reference.npz \\
    --preset reference \\
    --output stage_d4_flow_matching_finite_resource_design_oracle.json
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

from scipy.optimize import linprog
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


Array = jax.Array
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
class D4Config:
    preset: str = "quick"
    seed: int = 20260812

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

    # Stage-B / D.2 discretization.
    grid_n: int = 19
    time_n: int = 13
    bank_mode: str = "gauss-hermite"
    gh_order: int = 20
    particles: int = 8192
    rk4_substeps_per_time_interval: int = 8
    kde_bandwidth: float = 0.0
    kde_truncate: float = 4.0

    # Hard empirical I-projection.
    calibration_steps: int = 24
    calibration_tol: float = 2.0e-8
    newton_step_cap: float = 5.0
    lambda_clip: float = 80.0

    # Finite-measurement fitting.
    variance_floor: float = 1.0e-10
    quadratic_ridge_rel: float = 1.0e-12
    feasibility_margin: float = 0.0

    # Numerical/reference overlap gates.  These do not enter the objective.
    max_population_calibration_resid: float = 5.0e-6
    max_finite_calibration_resid: float = 1.0e-3
    min_ess_fraction: float = 0.03

    # Frozen Stage-B/C designs, always included as candidates and validation controls.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)


def preset_d4_config(name: str) -> D4Config:
    if name == "quick":
        return D4Config()
    if name == "reference":
        return D4Config(
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
            calibration_steps=28,
            calibration_tol=1.0e-9,
            max_population_calibration_resid=1.0e-5,
            max_finite_calibration_resid=1.0e-3,
            min_ess_fraction=0.03,
        )
    if name == "confirm":
        return D4Config(
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
            calibration_steps=32,
            calibration_tol=5.0e-10,
            max_population_calibration_resid=5.0e-6,
            max_finite_calibration_resid=5.0e-4,
            min_ess_fraction=0.03,
        )
    raise ValueError(name)


# -----------------------------------------------------------------------------
# Small helpers
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
    if isinstance(x, (jax.Array,)):
        return np.asarray(x).tolist()
    return x


def mean_se(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
    return {"mean": mean, "se": se, "n": int(x.size)}


def paired_difference(rows_a: Sequence[Mapping[str, float]], rows_b: Sequence[Mapping[str, float]], key: str):
    a = np.asarray([float(r[key]) for r in rows_a], dtype=np.float64)
    b = np.asarray([float(r[key]) for r in rows_b], dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    s = mean_se(d)
    return {
        "mean_difference_a_minus_b": s["mean"],
        "se_difference": s["se"],
        "n": s["n"],
        "a_better_fraction": float(np.mean(d < 0.0)) if d.size else float("nan"),
    }


def paired_reduction(rows_num: Sequence[Mapping[str, float]], rows_den: Sequence[Mapping[str, float]], key: str):
    num = np.asarray([float(r[key]) for r in rows_num], dtype=np.float64)
    den = np.asarray([float(r[key]) for r in rows_den], dtype=np.float64)
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
    direct = math.hypot(projective_angle_distance(a[0], b[0]), projective_angle_distance(a[1], b[1]))
    swap = math.hypot(projective_angle_distance(a[0], b[1]), projective_angle_distance(a[1], b[0]))
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


# -----------------------------------------------------------------------------
# Learned-FM-only feasibility geometry
# -----------------------------------------------------------------------------


def build_learned_beta_constraints(model, evaluator, d2, d3, c2, eta: np.ndarray, margin: float):
    """Build A beta <= b for physical hull intersect learned-FM particle hulls."""
    physical_eq, physical_meta = c2.sensor_moment_hull_equations(model, eta)
    if physical_eq is None:
        raise RuntimeError(f"Physical sensor moment hull unavailable: {physical_meta}")
    physical_eq = np.asarray(physical_eq, dtype=np.float64)

    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_probe = jnp.asarray(0.5 * (model.cfg.alpha_min + model.cfg.alpha_max), dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_probe, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_probe, eta_j), dtype=np.float64)

    A_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    time_meta: List[Dict[str, Any]] = []

    for kt, t in enumerate(times):
        x_all = np.asarray(evaluator.learned_nodes[kt], dtype=np.float64)
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        phi, _ = d2.sensor_particle_fields(model, eta, x)
        learned_eq = d3.hull_equations_from_points(phi)

        z = float(t * (1.0 - t))
        bridge = (1.0 - t) * c0 + t * c1
        max_endpoint_violation = 0.0
        for eq in (physical_eq, learned_eq):
            normals = np.asarray(eq[:, :2], dtype=np.float64)
            offsets = np.asarray(eq[:, 2], dtype=np.float64)
            if abs(z) < 1.0e-14:
                viol = normals @ bridge + offsets + float(margin)
                max_endpoint_violation = max(max_endpoint_violation, float(np.max(viol)))
            else:
                A_rows.append(z * normals)
                b_rows.append(-offsets - normals @ bridge - float(margin))

        if abs(z) < 1.0e-14 and max_endpoint_violation > 2.0e-8:
            raise RuntimeError(
                f"Exact endpoint outside learned/physical moment hull at t={t:.3f}; "
                f"max violation={max_endpoint_violation:.3e}"
            )
        time_meta.append({
            "t": float(t),
            "learned_in_domain_particles": int(x.shape[0]),
            "learned_facets": int(learned_eq.shape[0]),
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
            raise RuntimeError("Physical/learned-FM beta-feasibility intersection is empty")
        feasible_beta = np.asarray(feas.x, dtype=np.float64)
    else:
        feasible_beta = np.zeros(2, dtype=np.float64)

    polygon = _constraint_polygon(A, b) if A.shape[0] else None
    return {
        "A": A,
        "b": b,
        "c0": c0,
        "c1": c1,
        "feasible_beta": feasible_beta,
        "polygon": polygon,
        "physical_hull_metadata": physical_meta,
        "time_metadata": time_meta,
    }



# -----------------------------------------------------------------------------
# Fast/robust 2D finite-resource helpers
# -----------------------------------------------------------------------------


def _constraint_polygon(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Return ordered vertices of the 2D polytope A beta <= b when it has interior.

    D4 has exactly two measured moments, so the GLS feasibility projection is a
    two-dimensional convex QP.  Building the feasible polygon once per design is
    both more robust and much faster than asking SLSQP to process the full stack
    of redundant half-space inequalities on every scientific trial.

    Returns None if there are no constraints. Raises if the intersection has no
    strict 2D interior; such a set is unsuitable for finite exponential tilting
    without an explicit interior margin anyway.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.shape[0] == 0:
        return None
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("D4 fast feasibility solver requires exactly two beta coordinates")

    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1.0e-14
    A = A[keep]
    b = b[keep]
    norms = norms[keep]
    if A.shape[0] == 0:
        return None

    # Chebyshev center: maximize radius r subject to a_i^T x + ||a_i|| r <= b_i.
    # A strictly positive radius gives HalfspaceIntersection a valid interior point.
    A_lp = np.concatenate([A, norms[:, None]], axis=1)
    c_lp = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    center = linprog(
        c=c_lp,
        A_ub=A_lp,
        b_ub=b,
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not center.success:
        raise RuntimeError(f"Could not find interior point of D4 beta polytope: {center.message}")
    radius = float(center.x[2])
    if not np.isfinite(radius) or radius <= 1.0e-11:
        raise RuntimeError(
            "D4 beta-feasibility intersection has essentially zero 2D interior "
            f"(Chebyshev radius={radius:.3e}). Use a positive --feasibility-margin, "
            "increase particle-bank resolution, or inspect the candidate geometry."
        )

    halfspaces = np.concatenate([A, -b[:, None]], axis=1)
    try:
        hs = HalfspaceIntersection(halfspaces, np.asarray(center.x[:2], dtype=np.float64))
        pts = np.asarray(hs.intersections, dtype=np.float64)
        if pts.shape[0] < 3:
            raise RuntimeError("D4 beta polytope has fewer than three vertices")
        hull = ConvexHull(pts)
        return pts[np.asarray(hull.vertices, dtype=int)]
    except (QhullError, RuntimeError) as exc:
        raise RuntimeError(f"Could not construct D4 beta feasibility polygon: {exc}") from exc


def _project_beta_to_polygon(beta_u: np.ndarray, H: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Exact H-metric projection of beta_u onto an ordered convex 2D polygon."""
    beta_u = np.asarray(beta_u, dtype=np.float64)
    H = 0.5 * (np.asarray(H, dtype=np.float64) + np.asarray(H, dtype=np.float64).T)
    poly = np.asarray(polygon, dtype=np.float64)
    best = None
    best_val = float("inf")
    n = poly.shape[0]
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        d = p1 - p0
        Hd = H @ d
        den = float(d @ Hd)
        if den <= 1.0e-30:
            ss = (0.0, 1.0)
        else:
            sstar = -float(d @ (H @ (p0 - beta_u))) / den
            ss = (float(np.clip(sstar, 0.0, 1.0)),)
        for ss0 in ss:
            cand = p0 + ss0 * d
            e = cand - beta_u
            val = 0.5 * float(e @ H @ e)
            if val < best_val:
                best_val = val
                best = cand.copy()
    if best is None:
        raise RuntimeError("Failed to project beta onto D4 feasibility polygon")
    return best


def fit_quadratic_bridge_feasible_2d(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    V_obs: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    t_eval: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    polygon: np.ndarray | None,
    ridge_rel: float,
    variance_floor: float,
):
    """Same endpoint-anchored GLS bridge as D3, with an exact 2D convex projection."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    V_obs = np.asarray(V_obs, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    t_eval = np.asarray(t_eval, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m = int(c0.size)
    if m != 2:
        raise ValueError("D4 fast GLS projection is specialized to the two-sensor benchmark")
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
        # solve is more stable than forming inv(Vreg)
        Vinv_resid = np.linalg.solve(Vreg, resid)
        Vinv = np.linalg.solve(Vreg, np.eye(m))
        H_data += (z * z) * Vinv
        g += z * Vinv_resid
        used += 1
    if used == 0:
        raise ValueError("Need at least one interior acquisition time for quadratic GLS")

    scale = max(float(np.trace(H_data)) / max(m, 1), 1.0)
    Hreg = H_data + (float(ridge_rel) * scale) * np.eye(m)
    beta_u = np.linalg.solve(Hreg, g)
    Hinv = np.linalg.solve(Hreg, np.eye(m))
    beta_cov = 0.5 * (Hinv @ H_data @ Hinv.T + (Hinv @ H_data @ Hinv.T).T)

    max_viol = float(max(0.0, np.max(A @ beta_u - b))) if A.shape[0] else 0.0
    projection_active = max_viol > 1.0e-10
    beta = beta_u.copy()
    if projection_active:
        if polygon is None:
            raise RuntimeError("Projected D4 GLS fit requested but no feasibility polygon is available")
        beta = _project_beta_to_polygon(beta_u, Hreg, polygon)
        final_viol = float(np.max(A @ beta - b)) if A.shape[0] else 0.0
        if final_viol > 2.0e-7:
            raise RuntimeError(f"2D beta projection violated feasibility by {final_viol:.3e}")

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
        "feasibility_projection_norm": projection_norm,
        "constraint_solver_success": True,
        "max_unconstrained_hull_violation": max_viol,
    }


def _prepare_candidate_law_cache(model, evaluator, d2, eta: np.ndarray) -> Dict[str, Any]:
    """Cache eta-dependent particle observables once, then reuse them over trials/alphas."""
    eta = np.asarray(eta, dtype=np.float64)
    ngrid = int(model.cfg.grid_n)
    L = float(model.cfg.L)
    bw = float(evaluator.cfg.kde_bandwidth)
    if bw <= 0.0:
        bw = 0.35 * float(model.dx)
    sigma_cells = float(bw / model.dx)
    per_time = []
    for kt in range(model.cfg.time_n):
        x_all = np.asarray(evaluator.learned_nodes[kt], dtype=np.float64)
        mask = d2.in_domain_mask(model, x_all)
        x = x_all[mask]
        base_w = np.asarray(evaluator.base_w[mask], dtype=np.float64)
        base_w /= max(float(np.sum(base_w)), 1.0e-300)
        phi, _ = d2.sensor_particle_fields(model, eta, x)
        # Same cell convention as D2.rasterize_projected_state: [y, x].
        ix = np.floor((x[:, 0] + L) * ngrid / (2.0 * L)).astype(np.int64)
        iy = np.floor((x[:, 1] + L) * ngrid / (2.0 * L)).astype(np.int64)
        ix = np.clip(ix, 0, ngrid - 1)
        iy = np.clip(iy, 0, ngrid - 1)
        cell_id = iy * ngrid + ix
        per_time.append({"phi": phi, "base_w": base_w, "cell_id": cell_id})
    phi_grid, _ = model.sensor_fields(jnp.asarray(eta, dtype=jnp.float64))
    return {
        "eta": eta,
        "times": per_time,
        "phi_grid": np.asarray(phi_grid, dtype=np.float64),
        "ngrid": ngrid,
        "sigma_cells": sigma_cells,
        "truncate": float(evaluator.cfg.kde_truncate),
    }


def _pmass_curve(model, alpha: float, cache: Dict[bytes, List[Array]]) -> List[Array]:
    key = np.asarray(float(alpha), dtype=np.float64).tobytes()
    if key not in cache:
        rows = []
        for t in np.asarray(model.times, dtype=np.float64):
            _, pmass = model.external_q_mass(jnp.asarray(t), jnp.asarray(float(alpha)))
            rows.append(jnp.asarray(pmass, dtype=jnp.float64))
        cache[key] = rows
    return cache[key]


def _qmass_from_weights(prep_t: Mapping[str, Any], w: np.ndarray, prep: Mapping[str, Any]) -> np.ndarray:
    ngrid = int(prep["ngrid"])
    hist = np.bincount(
        np.asarray(prep_t["cell_id"], dtype=np.int64),
        weights=np.asarray(w, dtype=np.float64),
        minlength=ngrid * ngrid,
    ).reshape(ngrid, ngrid)
    sigma = float(prep["sigma_cells"])
    if sigma > 0.0:
        qmass = gaussian_filter(
            hist, sigma=sigma, mode="constant", cval=0.0, truncate=float(prep["truncate"])
        )
    else:
        qmass = hist
    z = float(np.sum(qmass))
    if not np.isfinite(z) or z <= 0.0:
        raise RuntimeError("Fast rasterized projected mass vanished")
    return qmass / z


def evaluate_particle_law_curve_fast(
    model,
    evaluator,
    d2,
    prep: Mapping[str, Any],
    alpha: float,
    target_curve: np.ndarray,
    heldout_mask: np.ndarray,
    pmass_cache: Dict[bytes, List[Array]],
) -> Dict[str, float]:
    """Law/calibration-only particle evaluation; deliberately skips forcing and Poisson work."""
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    target_curve = np.asarray(target_curve, dtype=np.float64)
    pmasses = _pmass_curve(model, float(alpha), pmass_cache)
    law = np.zeros(len(times), dtype=np.float64)
    ess = np.zeros(len(times), dtype=np.float64)
    cal = np.zeros(len(times), dtype=np.float64)
    lam_norm = np.zeros(len(times), dtype=np.float64)
    grid_err = np.zeros(len(times), dtype=np.float64)
    lam_warm = np.zeros(2, dtype=np.float64)
    for kt, _t in enumerate(times):
        pt = prep["times"][kt]
        lam, w, _moment, _C, diag = d2.solve_empirical_tilt(
            phi=np.asarray(pt["phi"], dtype=np.float64),
            base_w=np.asarray(pt["base_w"], dtype=np.float64),
            target=target_curve[kt],
            ridge=float(model.cfg.newton_ridge),
            cfg=evaluator.cfg,
            lam0=lam_warm,
        )
        lam_warm = np.asarray(lam, dtype=np.float64)
        qmass = _qmass_from_weights(pt, w, prep)
        law[kt] = float(
            model.gaussian_mmd2_mass(jnp.asarray(qmass, dtype=jnp.float64), pmasses[kt])
        )
        grid_moment = np.sum(np.asarray(prep["phi_grid"]) * qmass[None, ...], axis=(1, 2))
        grid_err[kt] = float(np.linalg.norm(grid_moment - target_curve[kt]))
        ess[kt] = float(diag["ess_fraction"])
        cal[kt] = float(diag["residual"])
        lam_norm[kt] = float(diag["lambda_norm"])
    return {
        "heldout_mmd2": d3_trap_average(law, tw, heldout_mask),
        "min_ess_fraction": float(np.min(ess)),
        "max_calibration_residual": float(np.max(cal)),
        "max_lambda_norm": float(np.max(lam_norm)),
        "max_grid_moment_error_after_kde": float(np.max(grid_err)),
    }


def d3_trap_average(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    if mask is not None:
        w *= np.asarray(mask, dtype=np.float64)
    z = float(np.sum(w))
    if z <= 0.0:
        return float("nan")
    return float(np.sum(w * v) / z)


def exact_target_c_only(model, eta: np.ndarray, alpha: float) -> np.ndarray:
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    return np.stack([
        np.asarray(model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        for t in np.asarray(model.times, dtype=np.float64)
    ], axis=0)


def population_learned_law_fast(model, evaluator, d2, eta: np.ndarray, pmass_cache) -> Dict[str, float]:
    """Exact same Stage-1 law screen, without computing quantities the screen never uses."""
    alphas = np.asarray(model.alphas, dtype=np.float64)
    aw = np.asarray(model.alpha_w, dtype=np.float64)
    all_mask = np.ones(model.cfg.time_n, dtype=bool)
    prep = _prepare_candidate_law_cache(model, evaluator, d2, eta)
    vals = []
    min_ess = float("inf")
    max_cal = 0.0
    max_grid_moment = 0.0
    max_lambda = 0.0
    for alpha in alphas:
        c = exact_target_c_only(model, eta, float(alpha))
        r = evaluate_particle_law_curve_fast(
            model, evaluator, d2, prep, float(alpha), c, all_mask, pmass_cache
        )
        vals.append(float(r["heldout_mmd2"]))
        min_ess = min(min_ess, float(r["min_ess_fraction"]))
        max_cal = max(max_cal, float(r["max_calibration_residual"]))
        max_grid_moment = max(max_grid_moment, float(r["max_grid_moment_error_after_kde"]))
        max_lambda = max(max_lambda, float(r["max_lambda_norm"]))
    return {
        "lift_mmd2": float(np.sum(aw * np.asarray(vals, dtype=np.float64))),
        "min_ess_fraction": min_ess,
        "max_calibration_residual": max_cal,
        "max_grid_moment_error_after_kde": max_grid_moment,
        "max_lambda_norm": max_lambda,
    }


def evaluate_learned_trial_law_fast(
    model,
    evaluator,
    d2,
    c2,
    measurement_cov,
    eta: np.ndarray,
    shared,
    acq_idx: np.ndarray,
    heldout_mask: np.ndarray,
    cfg: D4Config,
    constraints: Dict[str, Any],
    prep: Mapping[str, Any],
    pmass_cache,
    compute_exact_law: bool,
) -> Dict[str, float]:
    """Finite-measurement trial for Stage 2; no forcing, source raster, or Poisson solve."""
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
    if np.linalg.norm(c0 - constraints["c0"]) > 1.0e-10 or np.linalg.norm(c1 - constraints["c1"]) > 1.0e-10:
        raise RuntimeError("Unexpected alpha-dependent endpoints")
    curve = fit_quadratic_bridge_feasible_2d(
        t_obs=t_acq,
        y_obs=y_acq,
        V_obs=V_acq,
        c0=c0,
        c1=c1,
        t_eval=times,
        A=constraints["A"],
        b=constraints["b"],
        polygon=constraints.get("polygon"),
        ridge_rel=float(cfg.quadratic_ridge_rel),
        variance_floor=float(cfg.variance_floor),
    )
    finite = evaluate_particle_law_curve_fast(
        model, evaluator, d2, prep, float(shared.alpha), np.asarray(curve["c"]), heldout_mask, pmass_cache
    )
    exact_c = exact_target_c_only(model, eta, float(shared.alpha))
    exact_mmd = float("nan")
    if compute_exact_law:
        exact = evaluate_particle_law_curve_fast(
            model, evaluator, d2, prep, float(shared.alpha), exact_c, heldout_mask, pmass_cache
        )
        exact_mmd = float(exact["heldout_mmd2"])
    interior = np.ones(model.cfg.time_n, dtype=bool)
    interior[[0, -1]] = False
    curve_err = np.asarray(curve["c"])[interior] - exact_c[interior]
    return {
        "alpha": float(shared.alpha),
        "finite_heldout_mmd2": float(finite["heldout_mmd2"]),
        "exact_heldout_mmd2": exact_mmd,
        "measurement_delta_mmd2": float(finite["heldout_mmd2"] - exact_mmd) if np.isfinite(exact_mmd) else float("nan"),
        "finite_action": float("nan"),
        "exact_action": float("nan"),
        "measurement_action_inflation": float("nan"),
        "finite_min_ess": float(finite["min_ess_fraction"]),
        "finite_max_calibration_resid": float(finite["max_calibration_residual"]),
        "exact_min_ess": float("nan"),
        "exact_max_calibration_resid": float("nan"),
        "finite_max_poisson_resid": float("nan"),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum(curve_err * curve_err, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(curve_err, axis=1))),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((np.asarray(y_acq) - np.asarray(exact_acq)) ** 2))),
    }


def evaluate_learned_trial_action_fast(
    model,
    evaluator,
    d2,
    d3,
    c2,
    measurement_cov,
    eta: np.ndarray,
    shared,
    acq_idx: np.ndarray,
    heldout_mask: np.ndarray,
    cfg: D4Config,
    constraints: Dict[str, Any],
    prep: Mapping[str, Any],
    pmass_cache,
    compute_exact_action: bool,
) -> Dict[str, float]:
    """Full finite action + cheap exact-law comparator; exact Poisson action is optional."""
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
    if np.linalg.norm(c0 - constraints["c0"]) > 1.0e-10 or np.linalg.norm(c1 - constraints["c1"]) > 1.0e-10:
        raise RuntimeError("Unexpected alpha-dependent endpoints")
    curve = fit_quadratic_bridge_feasible_2d(
        t_obs=t_acq, y_obs=y_acq, V_obs=V_acq, c0=c0, c1=c1, t_eval=times,
        A=constraints["A"], b=constraints["b"], polygon=constraints.get("polygon"),
        ridge_rel=float(cfg.quadratic_ridge_rel), variance_floor=float(cfg.variance_floor),
    )
    finite = d3.evaluate_particle_curve(
        evaluator=evaluator,
        d2=d2,
        eta=eta,
        alpha=float(shared.alpha),
        target_curve=np.asarray(curve["c"], dtype=np.float64),
        target_cdot=np.asarray(curve["cdot"], dtype=np.float64),
        branch="learned_fm_particle",
        heldout_mask=heldout_mask,
        compute_action=True,
    )
    exact_c = exact_target_c_only(model, eta, float(shared.alpha))
    exact_law = evaluate_particle_law_curve_fast(
        model, evaluator, d2, prep, float(shared.alpha), exact_c, heldout_mask, pmass_cache
    )
    exact_action = float("nan")
    if compute_exact_action:
        _, exact_cdot = exact_target_curve(model, evaluator, eta, float(shared.alpha))
        exact_full = d3.evaluate_particle_curve(
            evaluator=evaluator, d2=d2, eta=eta, alpha=float(shared.alpha),
            target_curve=exact_c, target_cdot=exact_cdot,
            branch="learned_fm_particle", heldout_mask=heldout_mask, compute_action=True,
        )
        exact_action = float(exact_full["full_action"])
    interior = np.ones(model.cfg.time_n, dtype=bool)
    interior[[0, -1]] = False
    curve_err = np.asarray(curve["c"])[interior] - exact_c[interior]
    return {
        "alpha": float(shared.alpha),
        "finite_heldout_mmd2": float(finite["heldout_mmd2"]),
        "exact_heldout_mmd2": float(exact_law["heldout_mmd2"]),
        "measurement_delta_mmd2": float(finite["heldout_mmd2"] - exact_law["heldout_mmd2"]),
        "finite_action": float(finite["full_action"]),
        "exact_action": exact_action,
        "measurement_action_inflation": float(finite["full_action"] / exact_action - 1.0)
            if np.isfinite(exact_action) and abs(exact_action) > 1.0e-14 else float("nan"),
        "finite_min_ess": float(finite["min_ess_fraction"]),
        "finite_max_calibration_resid": float(finite["max_calibration_residual"]),
        "exact_min_ess": float(exact_law["min_ess_fraction"]),
        "exact_max_calibration_resid": float(exact_law["max_calibration_residual"]),
        "finite_max_poisson_resid": float(finite["max_poisson_relative_residual"]),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum(curve_err * curve_err, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(curve_err, axis=1))),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((np.asarray(y_acq) - np.asarray(exact_acq)) ** 2))),
    }


# -----------------------------------------------------------------------------
# Learned-FM population and finite-resource evaluation
# -----------------------------------------------------------------------------


def exact_target_curve(model, evaluator, eta: np.ndarray, alpha: float):
    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    c = np.stack([
        np.asarray(model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        for t in times
    ], axis=0)
    cdot = np.stack([
        np.asarray(evaluator.cdot_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        for t in times
    ], axis=0)
    return c, cdot


def population_learned_law(model, evaluator, d2, d3, eta: np.ndarray) -> Dict[str, float]:
    """Learned-FM exact-population law loss, no Poisson solves."""
    alphas = np.asarray(model.alphas, dtype=np.float64)
    aw = np.asarray(model.alpha_w, dtype=np.float64)
    all_mask = np.ones(model.cfg.time_n, dtype=bool)

    vals = []
    min_ess = float("inf")
    max_cal = 0.0
    max_grid_moment = 0.0
    max_lambda = 0.0
    for alpha in alphas:
        c, cdot = exact_target_curve(model, evaluator, eta, float(alpha))
        r = d3.evaluate_particle_curve(
            evaluator=evaluator,
            d2=d2,
            eta=eta,
            alpha=float(alpha),
            target_curve=c,
            target_cdot=cdot,
            branch="learned_fm_particle",
            heldout_mask=all_mask,
            compute_action=False,
        )
        vals.append(float(r["heldout_mmd2"]))
        min_ess = min(min_ess, float(r["min_ess_fraction"]))
        max_cal = max(max_cal, float(r["max_calibration_residual"]))
        max_grid_moment = max(max_grid_moment, float(r["max_grid_moment_error_after_kde"]))
        max_lambda = max(max_lambda, float(r["max_lambda_norm"]))

    return {
        "lift_mmd2": float(np.sum(aw * np.asarray(vals, dtype=np.float64))),
        "min_ess_fraction": float(min_ess),
        "max_calibration_residual": float(max_cal),
        "max_grid_moment_error_after_kde": float(max_grid_moment),
        "max_lambda_norm": float(max_lambda),
    }


def evaluate_learned_trial(
    model,
    evaluator,
    d2,
    d3,
    c2,
    measurement_cov,
    eta: np.ndarray,
    shared,
    acq_idx: np.ndarray,
    heldout_mask: np.ndarray,
    cfg: D4Config,
    constraints: Dict[str, Any],
    compute_action: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """One scientific trial, learned-FM particle branch only."""
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
    if np.linalg.norm(c0 - constraints["c0"]) > 1.0e-10 or np.linalg.norm(c1 - constraints["c1"]) > 1.0e-10:
        raise RuntimeError("Unexpected alpha-dependent endpoints")

    curve = d3.fit_quadratic_bridge_joint_feasible(
        t_obs=t_acq,
        y_obs=y_acq,
        V_obs=V_acq,
        c0=c0,
        c1=c1,
        t_eval=times,
        A=constraints["A"],
        b=constraints["b"],
        feasible_beta=constraints["feasible_beta"],
        ridge_rel=float(cfg.quadratic_ridge_rel),
        variance_floor=float(cfg.variance_floor),
    )
    exact_c, exact_cdot = exact_target_curve(model, evaluator, eta, float(shared.alpha))

    finite = d3.evaluate_particle_curve(
        evaluator=evaluator,
        d2=d2,
        eta=eta,
        alpha=float(shared.alpha),
        target_curve=np.asarray(curve["c"], dtype=np.float64),
        target_cdot=np.asarray(curve["cdot"], dtype=np.float64),
        branch="learned_fm_particle",
        heldout_mask=heldout_mask,
        compute_action=compute_action,
    )
    exact = d3.evaluate_particle_curve(
        evaluator=evaluator,
        d2=d2,
        eta=eta,
        alpha=float(shared.alpha),
        target_curve=exact_c,
        target_cdot=exact_cdot,
        branch="learned_fm_particle",
        heldout_mask=heldout_mask,
        compute_action=compute_action,
    )

    interior = np.ones(model.cfg.time_n, dtype=bool)
    interior[[0, -1]] = False
    curve_err = np.asarray(curve["c"], dtype=np.float64)[interior] - exact_c[interior]

    row = {
        "alpha": float(shared.alpha),
        "finite_heldout_mmd2": float(finite["heldout_mmd2"]),
        "exact_heldout_mmd2": float(exact["heldout_mmd2"]),
        "measurement_delta_mmd2": float(finite["heldout_mmd2"] - exact["heldout_mmd2"]),
        "finite_action": float(finite["full_action"]),
        "exact_action": float(exact["full_action"]),
        "measurement_action_inflation": float(finite["full_action"] / exact["full_action"] - 1.0)
            if compute_action else float("nan"),
        "finite_min_ess": float(finite["min_ess_fraction"]),
        "finite_max_calibration_resid": float(finite["max_calibration_residual"]),
        "exact_min_ess": float(exact["min_ess_fraction"]),
        "exact_max_calibration_resid": float(exact["max_calibration_residual"]),
        "finite_max_poisson_resid": float(finite["max_poisson_relative_residual"]),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum(curve_err * curve_err, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(curve_err, axis=1))),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((np.asarray(y_acq) - np.asarray(exact_acq)) ** 2))),
    }
    return row, {"finite": finite, "exact_population": exact, "curve": curve}


def summarize_trial_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    keys = [
        "finite_heldout_mmd2", "exact_heldout_mmd2", "measurement_delta_mmd2",
        "finite_action", "exact_action", "measurement_action_inflation",
        "finite_min_ess", "finite_max_calibration_resid",
        "feasibility_projection_active", "feasibility_projection_norm",
        "quadratic_moment_rmse", "quadratic_moment_max_error",
    ]
    out = {k: mean_se([float(r[k]) for r in rows]) for k in keys}
    if rows:
        out["max_finite_calibration_resid"] = float(max(r["finite_max_calibration_resid"] for r in rows))
        out["min_finite_ess"] = float(min(r["finite_min_ess"] for r in rows))
        out["projection_active_fraction"] = float(np.mean([r["feasibility_projection_active"] for r in rows]))
        out["max_projection_norm"] = float(max(r["feasibility_projection_norm"] for r in rows))
    return out


def public_candidate(c: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "theta_deg", "sensor_separation_deg", "sources",
        "population_law", "population_min_ess", "population_max_cal",
        "population_numerically_valid", "finite_risk", "finite_degradation",
        "finite_min_ess", "finite_max_cal", "finite_numerically_valid",
        "projection_rate", "finite_action", "matched_exact_action",
        "action_inflation", "max_poisson_resid",
    ]
    return {k: jsonify(c[k]) for k in keys if k in c}


# -----------------------------------------------------------------------------
# CLI / main oracle
# -----------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--c2-script", type=str, default=None)
    p.add_argument("--d0-script", type=str, default=None)
    p.add_argument("--d2-script", type=str, default=None)
    p.add_argument("--d3-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d4_flow_matching_finite_resource_design_oracle.json")

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--angle-n", type=int, default=None)
    p.add_argument("--law-trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--validation-trials", type=int, default=None)
    p.add_argument("--tau-l", type=float, default=None)
    p.add_argument("--tau-r", type=float, default=None)
    p.add_argument("--N", dest="finite_n", type=int, default=None)
    p.add_argument("--K", dest="acquisition_k", type=int, default=None)
    p.add_argument("--noise", dest="obs_noise_std", type=float, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--time-n", type=int, default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)
    p.add_argument("--feasibility-margin", type=float, default=None)
    p.add_argument("--max-population-calibration-resid", type=float, default=None)
    p.add_argument("--max-finite-calibration-resid", type=float, default=None)
    p.add_argument("--min-ess-fraction", type=float, default=None)
    p.add_argument(
        "--resume-stage1", type=str, default=None,
        help="Reuse a saved Stage-1 population-screen cache from a prior compatible run."
    )
    p.add_argument(
        "--compute-matched-exact-action", action="store_true",
        help="Also solve the expensive exact-target Poisson action in Stage 3/validation. "
             "Not needed for D4 selection or the primary finite-action comparison."
    )
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
    d3_path = Path(args.d3_script) if args.d3_script else autodetect(["stage_d3_flow_matching_finite_measurements.py"])
    for label, p in (("backend", backend_path), ("c2", c2_path), ("d0", d0_path), ("d2", d2_path), ("d3", d3_path)):
        if p is None:
            raise FileNotFoundError(f"Could not autodetect {label}; pass the corresponding CLI path")

    backend = load_module(backend_path, "stage_b2_backend_d4")
    c2 = load_module(c2_path, "stage_c2_backend_d4")
    d0 = load_module(d0_path, "stage_d0_backend_d4")
    d2 = load_module(d2_path, "stage_d2_backend_d4")
    d3 = load_module(d3_path, "stage_d3_backend_d4")
    params, checkpoint_meta = d0.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d4_config(args.preset)
    overrides: Dict[str, Any] = {}
    for key in (
        "seed", "angle_n", "law_trials", "action_trials", "validation_trials",
        "tau_l", "tau_r", "finite_n", "acquisition_k", "obs_noise_std",
        "grid_n", "time_n", "gh_order", "kde_bandwidth", "feasibility_margin",
        "max_population_calibration_resid", "max_finite_calibration_resid", "min_ess_fraction",
    ):
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    if args.rk4_substeps is not None:
        overrides["rk4_substeps_per_time_interval"] = int(args.rk4_substeps)
    cfg = dataclasses.replace(cfg, **overrides)

    if cfg.action_trials < 1 or cfg.action_trials > cfg.law_trials:
        raise ValueError("Require 1 <= action_trials <= law_trials")
    if cfg.validation_trials < 1:
        raise ValueError("validation_trials must be >= 1")
    if cfg.acquisition_k < 3 or cfg.acquisition_k >= cfg.time_n:
        raise ValueError("K must satisfy 3 <= K < time_n")
    if cfg.tau_l < 0.0 or cfg.tau_r < 0.0:
        raise ValueError("tau_l and tau_r must be nonnegative")

    base = backend.preset_config("quick" if args.preset == "quick" else "reference")
    stage_b_cfg = dataclasses.replace(base, grid_n=int(cfg.grid_n), time_n=int(cfg.time_n))
    model = backend.StageB(stage_b_cfg)
    teacher = d0.AnalyticReferenceTeacher(stage_b_cfg)

    cp_phys = checkpoint_meta.get("physical_system", {})
    for key, current in (("r", stage_b_cfg.r), ("sigma", stage_b_cfg.sigma), ("kappa", stage_b_cfg.kappa)):
        saved = float(cp_phys.get(key, current))
        if abs(saved - float(current)) > 1.0e-12:
            raise ValueError(f"D0 checkpoint {key}={saved} != Stage-B {current}")

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
    measurement_cov = d3.MeasurementCovariance(model)
    # External target masses depend only on (t, alpha), not on sensor design.
    # Cache them once and reuse them across the entire oracle.
    pmass_cache: Dict[bytes, List[Array]] = {}

    frozen = {
        "lift": np.radians(np.asarray(cfg.lift_design_deg, dtype=np.float64)),
        "tangent": np.radians(np.asarray(cfg.tangent_design_deg, dtype=np.float64)),
        "full": np.radians(np.asarray(cfg.full_design_deg, dtype=np.float64)),
    }
    candidates = oracle_candidate_designs(int(cfg.angle_n), float(model.cfg.min_sep_deg), frozen)

    acq_sets = c2.nested_acquisition_sets(model.cfg.time_n, [int(cfg.acquisition_k)])
    acq_idx = np.asarray(acq_sets[int(cfg.acquisition_k)], dtype=int)
    acq_set = set(acq_idx.tolist())
    heldout_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in acq_set and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    if heldout_idx.size == 0:
        raise ValueError("No held-out interior times")
    heldout_mask = np.zeros(model.cfg.time_n, dtype=bool)
    heldout_mask[heldout_idx] = True

    total_trials = int(cfg.law_trials + cfg.validation_trials)
    trial_bank = []
    for trial in range(total_trials):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 4400, trial]))
        alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        trial_bank.append(c2.draw_shared_trial(model, alpha, acq_idx, int(cfg.finite_n), rng))
    law_bank = trial_bank[: int(cfg.law_trials)]
    action_bank = trial_bank[: int(cfg.action_trials)]
    validation_bank = trial_bank[int(cfg.law_trials):]

    print("=" * 108)
    print("Stage D.4 — learned-FM finite-resource sensor-design oracle (NO CNF)")
    print("=" * 108)
    print(f"Backend       : {Path(backend_path).resolve()}")
    print(f"D0 checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Grid/time     : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"FM bank       : {d2_cfg.bank_mode}, GH order={d2_cfg.gh_order}, Nbank={evaluator.x0.shape[0]}")
    print(f"Finite cond.  : N={cfg.finite_n}, K={cfg.acquisition_k}, noise={cfg.obs_noise_std:.4f}")
    print(f"Oracle        : angle_n={cfg.angle_n}, candidates={len(candidates)}, min_sep={model.cfg.min_sep_deg:.1f} deg")
    print(f"Trials        : law={cfg.law_trials}, action={cfg.action_trials}, independent validation={cfg.validation_trials}")
    print(f"Tolerances    : tau_L={100*cfg.tau_l:.2f}%, tau_R={100*cfg.tau_r:.2f}%")
    print(f"Numerics      : max population cal={cfg.max_population_calibration_resid:.1e}, "
          f"max finite cal={cfg.max_finite_calibration_resid:.1e}, min ESS={cfg.min_ess_fraction:.3f}")

    # ------------------------------------------------------------------
    # Stage 1: learned-FM population-law screen. No Poisson solves.
    # ------------------------------------------------------------------
    stage1_default = Path(args.output).with_suffix(Path(args.output).suffix + ".stage1.json")
    stage1_cache_path = Path(args.resume_stage1).expanduser().resolve() if args.resume_stage1 else stage1_default
    valid_population: List[Dict[str, Any]] = []
    loaded_stage1 = False
    if args.resume_stage1:
        cached = json.loads(stage1_cache_path.read_text(encoding="utf-8"))
        meta = cached.get("meta", {})
        expected = {
            "angle_n": int(cfg.angle_n),
            "grid_n": int(cfg.grid_n),
            "time_n": int(cfg.time_n),
            "gh_order": int(cfg.gh_order),
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        }
        for k, v in expected.items():
            if meta.get(k) != v:
                raise RuntimeError(f"Stage-1 cache mismatch for {k}: cached={meta.get(k)!r}, current={v!r}")
        rows = cached.get("candidates", [])
        if len(rows) != len(candidates):
            raise RuntimeError("Stage-1 cache candidate count does not match current oracle")
        for cand, row in zip(candidates, rows):
            if np.max(np.abs(np.asarray(cand["theta_deg"]) - np.asarray(row["theta_deg"]))) > 1.0e-8:
                raise RuntimeError("Stage-1 cache candidate ordering/angles do not match current oracle")
            for k in (
                "population_law", "population_min_ess", "population_max_cal",
                "population_numerically_valid", "population_failure",
            ):
                if k in row:
                    cand[k] = row[k]
            if cand.get("population_numerically_valid", False):
                valid_population.append(cand)
        loaded_stage1 = True
        print(f"\n[1/3] Loaded Stage-1 population screen from {stage1_cache_path}", flush=True)
    else:
        print("\n[1/3] Learned-FM population-law screen (fast law-only path)...", flush=True)
        for j, cand in enumerate(candidates):
            try:
                pop = population_learned_law_fast(model, evaluator, d2, cand["eta"], pmass_cache)
                cand["population_law"] = float(pop["lift_mmd2"])
                cand["population_min_ess"] = float(pop["min_ess_fraction"])
                cand["population_max_cal"] = float(pop["max_calibration_residual"])
                cand["population_numerically_valid"] = bool(
                    pop["min_ess_fraction"] >= cfg.min_ess_fraction
                    and pop["max_calibration_residual"] <= cfg.max_population_calibration_resid
                )
                cand["population_diagnostics"] = pop
            except Exception as exc:
                cand["population_law"] = float("inf")
                cand["population_min_ess"] = 0.0
                cand["population_max_cal"] = float("inf")
                cand["population_numerically_valid"] = False
                cand["population_failure"] = f"{type(exc).__name__}: {exc}"
            if cand["population_numerically_valid"]:
                valid_population.append(cand)
            if (j + 1) % max(1, len(candidates) // 10) == 0 or j + 1 == len(candidates):
                print(f"  population {j+1}/{len(candidates)} | numerically valid={len(valid_population)}", flush=True)

        stage1_payload = {
            "meta": {
                "angle_n": int(cfg.angle_n),
                "grid_n": int(cfg.grid_n),
                "time_n": int(cfg.time_n),
                "gh_order": int(cfg.gh_order),
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            },
            "candidates": [
                {
                    "theta_deg": c["theta_deg"],
                    "population_law": c.get("population_law", float("inf")),
                    "population_min_ess": c.get("population_min_ess", 0.0),
                    "population_max_cal": c.get("population_max_cal", float("inf")),
                    "population_numerically_valid": c.get("population_numerically_valid", False),
                    "population_failure": c.get("population_failure"),
                }
                for c in candidates
            ],
        }
        stage1_cache_path.write_text(json.dumps(jsonify(stage1_payload), indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"  saved Stage-1 cache: {stage1_cache_path.resolve()}", flush=True)

    if not valid_population:
        raise RuntimeError("No numerically valid candidate in learned-FM population screen")
    lstar = min(float(c["population_law"]) for c in valid_population)
    lmax = (1.0 + float(cfg.tau_l)) * lstar
    pop_feasible = [c for c in valid_population if float(c["population_law"]) <= lmax]
    population_best = min(valid_population, key=lambda c: float(c["population_law"]))
    print(
        f"  learned population valid={len(valid_population)}/{len(candidates)}; "
        f"law feasible={len(pop_feasible)}; L_FM*= {lstar:.8e}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: learned-FM finite-resource law screen, common random numbers.
    # ------------------------------------------------------------------
    print("\n[2/3] Learned-FM finite-resource law screen (law-only fast path)...", flush=True)
    finite_valid: List[Dict[str, Any]] = []
    for j, cand in enumerate(pop_feasible):
        eta = np.asarray(cand["eta"], dtype=np.float64)
        cand_label = np.round(np.degrees(eta), 5).tolist()
        try:
            constraints = build_learned_beta_constraints(
                model, evaluator, d2, d3, c2, eta, float(cfg.feasibility_margin)
            )
            prep = _prepare_candidate_law_cache(model, evaluator, d2, eta)
            cand["_constraints"] = constraints
            cand["_law_prep"] = prep
            cand["learned_feasibility"] = {
                "constraint_count": int(constraints["A"].shape[0]),
                "polygon_vertex_count": int(0 if constraints.get("polygon") is None else len(constraints["polygon"])),
                "physical_hull_metadata": constraints["physical_hull_metadata"],
                "time_metadata": constraints["time_metadata"],
            }
            law_rows = []
            gate_failure = None
            for trial_i, shared in enumerate(law_bank):
                row = evaluate_learned_trial_law_fast(
                    model, evaluator, d2, c2, measurement_cov,
                    eta, shared, acq_idx, heldout_mask, cfg, constraints, prep,
                    pmass_cache, compute_exact_law=False,
                )
                law_rows.append(row)
                # Numerical validity is monotone in the min-ESS/max-residual gates.
                # Once a candidate has failed a gate, later trials cannot rescue it.
                if row["finite_min_ess"] < cfg.min_ess_fraction:
                    gate_failure = (
                        f"trial {trial_i}: ESS={row['finite_min_ess']:.3e} "
                        f"< {cfg.min_ess_fraction:.3e}"
                    )
                    break
                if row["finite_max_calibration_resid"] > cfg.max_finite_calibration_resid:
                    gate_failure = (
                        f"trial {trial_i}: calibration residual="
                        f"{row['finite_max_calibration_resid']:.3e} > "
                        f"{cfg.max_finite_calibration_resid:.3e}"
                    )
                    break

            cand["law_rows"] = law_rows
            cand["finite_risk"] = mean_se([r["finite_heldout_mmd2"] for r in law_rows])
            # Exact-target comparison is intentionally deferred: it is diagnostic,
            # not part of the Stage-2 selection objective.
            cand["finite_degradation"] = {"mean": float("nan"), "se": float("nan"), "n": 0}
            cand["finite_min_ess"] = float(min(r["finite_min_ess"] for r in law_rows))
            cand["finite_max_cal"] = float(max(r["finite_max_calibration_resid"] for r in law_rows))
            cand["projection_rate"] = float(np.mean([r["feasibility_projection_active"] for r in law_rows]))
            cand["max_projection_norm"] = float(max(r["feasibility_projection_norm"] for r in law_rows))
            cand["finite_numerically_valid"] = gate_failure is None
            if gate_failure is not None:
                cand["finite_failure"] = gate_failure
                print(f"  REJECT eta={cand_label}: {gate_failure}", flush=True)
        except Exception as exc:
            cand["finite_numerically_valid"] = False
            cand["finite_failure"] = f"{type(exc).__name__}: {exc}"
            cand["finite_risk"] = {"mean": float("inf"), "se": float("nan"), "n": 0}
            cand["finite_degradation"] = {"mean": float("nan"), "se": float("nan"), "n": 0}
            cand["finite_min_ess"] = 0.0
            cand["finite_max_cal"] = float("inf")
            print(f"  FAIL   eta={cand_label}: {type(exc).__name__}: {exc}", flush=True)
        if cand["finite_numerically_valid"]:
            finite_valid.append(cand)
            print(
                f"  PASS   eta={cand_label}: R={cand['finite_risk']['mean']:.8e}, "
                f"ESSmin={cand['finite_min_ess']:.3f}, calmax={cand['finite_max_cal']:.3e}",
                flush=True,
            )
        if (j + 1) % max(1, len(pop_feasible) // 10) == 0 or j + 1 == len(pop_feasible):
            print(f"  finite law {j+1}/{len(pop_feasible)} | numerically valid={len(finite_valid)}", flush=True)

    if not finite_valid:
        print("\nNo Stage-2 candidate survived. Detailed reasons:", flush=True)
        for cand in pop_feasible:
            print(
                f"  eta={np.round(cand['theta_deg'], 5).tolist()} | "
                f"reason={cand.get('finite_failure', 'failed numerical gate')}",
                flush=True,
            )
        raise RuntimeError(
            "No numerically valid population-feasible candidate under finite measurements. "
            "The per-candidate failure messages above distinguish solver/geometry failures "
            "from ESS/calibration-gate failures."
        )
    rstar = min(float(c["finite_risk"]["mean"]) for c in finite_valid)
    rmax = (1.0 + float(cfg.tau_r)) * rstar
    finite_feasible = [c for c in finite_valid if float(c["finite_risk"]["mean"]) <= rmax]
    finite_best = min(finite_valid, key=lambda c: float(c["finite_risk"]["mean"]))
    print(
        f"  finite valid={len(finite_valid)}/{len(pop_feasible)}; "
        f"risk feasible={len(finite_feasible)}; R_FM,N*= {rstar:.8e}",
        flush=True,
    )

    if not finite_feasible:
        raise RuntimeError("No finite-risk-feasible design; increase tau_R or angle resolution")

    # ------------------------------------------------------------------
    # Stage 3: expensive finite action only on law survivors.
    # ------------------------------------------------------------------
    print("\n[3/3] Learned-FM finite-resource action screen...", flush=True)
    for j, cand in enumerate(finite_feasible):
        eta = np.asarray(cand["eta"], dtype=np.float64)
        constraints = cand.get("_constraints")
        if constraints is None:
            constraints = build_learned_beta_constraints(
                model, evaluator, d2, d3, c2, eta, float(cfg.feasibility_margin)
            )
        prep = cand.get("_law_prep")
        if prep is None:
            prep = _prepare_candidate_law_cache(model, evaluator, d2, eta)
        action_rows = []
        for shared in action_bank:
            row = evaluate_learned_trial_action_fast(
                model, evaluator, d2, d3, c2, measurement_cov,
                eta, shared, acq_idx, heldout_mask, cfg, constraints, prep,
                pmass_cache, compute_exact_action=bool(args.compute_matched_exact_action),
            )
            action_rows.append(row)
        cand["action_rows"] = action_rows
        cand["finite_action"] = mean_se([r["finite_action"] for r in action_rows])
        cand["matched_exact_action"] = mean_se([r["exact_action"] for r in action_rows])
        cand["action_inflation"] = mean_se([r["measurement_action_inflation"] for r in action_rows])
        cand["max_poisson_resid"] = float(max(r["finite_max_poisson_resid"] for r in action_rows))
        print(
            f"  action {j+1}/{len(finite_feasible)} | eta={cand['theta_deg']} | "
            f"A={cand['finite_action']['mean']:.6e}",
            flush=True,
        )

    robust = min(finite_feasible, key=lambda c: float(c["finite_action"]["mean"]))

    # Identify exact frozen candidate rows if present.
    frozen_candidate_rows: Dict[str, Any] = {}
    for name, eta in frozen.items():
        hit = min(candidates, key=lambda c: unordered_design_distance(c["eta"], eta))
        if unordered_design_distance(hit["eta"], eta) < 1.0e-10:
            frozen_candidate_rows[name] = hit

    # ------------------------------------------------------------------
    # Independent post-selection validation.
    # ------------------------------------------------------------------
    validation_designs: Dict[str, np.ndarray] = {
        "robust_d4": np.asarray(robust["eta"], dtype=np.float64),
        "lift": frozen["lift"],
        "tangent": frozen["tangent"],
        "full": frozen["full"],
    }
    # Deduplicate only computation; aliases are restored in the public output.
    unique_validation: Dict[Tuple[float, float], Dict[str, Any]] = {}
    alias_to_key: Dict[str, Tuple[float, float]] = {}
    for name, eta in validation_designs.items():
        key = tuple(np.round(canonical_eta(eta), 12))
        alias_to_key[name] = key
        unique_validation.setdefault(key, {"eta": canonical_eta(eta), "aliases": []})["aliases"].append(name)

    validation_unique: Dict[Tuple[float, float], Dict[str, Any]] = {}
    validation_trial_rows_unique: Dict[Tuple[float, float], List[Dict[str, float]]] = {}
    print("\nIndependent validation...", flush=True)
    for key, spec in unique_validation.items():
        eta = np.asarray(spec["eta"], dtype=np.float64)
        constraints = build_learned_beta_constraints(
            model, evaluator, d2, d3, c2, eta, float(cfg.feasibility_margin)
        )
        prep = _prepare_candidate_law_cache(model, evaluator, d2, eta)
        rows = []
        for shared in validation_bank:
            row = evaluate_learned_trial_action_fast(
                model, evaluator, d2, d3, c2, measurement_cov,
                eta, shared, acq_idx, heldout_mask, cfg, constraints, prep,
                pmass_cache, compute_exact_action=bool(args.compute_matched_exact_action),
            )
            rows.append(row)
        validation_trial_rows_unique[key] = rows
        validation_unique[key] = {
            "theta_deg": np.degrees(eta).tolist(),
            "aliases": spec["aliases"],
            "summary": summarize_trial_rows(rows),
        }
        s = validation_unique[key]["summary"]
        print(
            f"  {','.join(spec['aliases']):18s} eta={np.degrees(eta).round(3).tolist()} | "
            f"R={s['finite_heldout_mmd2']['mean']:.8e} | A={s['finite_action']['mean']:.6e}",
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
        "robust_vs_lift_finite_law": paired_difference(validation_rows["robust_d4"], validation_rows["lift"], "finite_heldout_mmd2"),
        "robust_vs_full_finite_law": paired_difference(validation_rows["robust_d4"], validation_rows["full"], "finite_heldout_mmd2"),
        "robust_vs_lift_measurement_degradation": paired_difference(validation_rows["robust_d4"], validation_rows["lift"], "measurement_delta_mmd2"),
        "robust_vs_full_measurement_degradation": paired_difference(validation_rows["robust_d4"], validation_rows["full"], "measurement_delta_mmd2"),
        "robust_vs_lift_action_reduction": paired_reduction(validation_rows["robust_d4"], validation_rows["lift"], "finite_action"),
        "robust_vs_full_action_reduction": paired_reduction(validation_rows["robust_d4"], validation_rows["full"], "finite_action"),
        "full_vs_lift_finite_law": paired_difference(validation_rows["full"], validation_rows["lift"], "finite_heldout_mmd2"),
        "full_vs_lift_measurement_degradation": paired_difference(validation_rows["full"], validation_rows["lift"], "measurement_delta_mmd2"),
        "full_vs_lift_action_reduction": paired_reduction(validation_rows["full"], validation_rows["lift"], "finite_action"),
    }

    payload = {
        "stage": "D.4 learned-FM finite-resource sensor-design oracle; no CNF",
        "method": "learned-FM particle population-law screen -> learned-FM finite-law screen -> minimum learned-FM finite action; independent validation",
        "backend_path": str(Path(backend_path).resolve()),
        "c2_script_path": str(Path(c2_path).resolve()),
        "d0_script_path": str(Path(d0_path).resolve()),
        "d2_script_path": str(Path(d2_path).resolve()),
        "d3_script_path": str(Path(d3_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "config": jsonify(cfg),
        "stage_b_config": jsonify(stage_b_cfg),
        "stage_d2_config": jsonify(d2_cfg),
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
            "fast_law_only_screens": True,
            "compute_matched_exact_action": bool(args.compute_matched_exact_action),
            "stage1_cache_path": str(stage1_cache_path.resolve()),
        },
        "reference_bank_diagnostics": evaluator.reference_bank_diagnostics(),
        "population_law_star": float(lstar),
        "population_law_max": float(lmax),
        "finite_risk_star": float(rstar),
        "finite_risk_max": float(rmax),
        "population_numerically_valid_count": int(len(valid_population)),
        "population_feasible_count": int(len(pop_feasible)),
        "finite_numerically_valid_count": int(len(finite_valid)),
        "finite_risk_feasible_count": int(len(finite_feasible)),
        "population_best": public_candidate(population_best),
        "finite_law_best": public_candidate(finite_best),
        "robust_d4": public_candidate(robust),
        "robust_d4_eta_rad": jsonify(robust["eta"]),
        "frozen_candidate_rows": {k: public_candidate(v) for k, v in frozen_candidate_rows.items()},
        "finite_risk_feasible_rows": [public_candidate(c) for c in finite_feasible],
        "population_feasible_rows": [public_candidate(c) for c in pop_feasible],
        "all_candidates_population": [
            {
                "theta_deg": c["theta_deg"],
                "sensor_separation_deg": c["sensor_separation_deg"],
                "sources": c["sources"],
                "population_law": c.get("population_law", float("inf")),
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
        "interpretation_notes": [
            "D.4 optimizes the same physically parameterized two-sensor family as Stage C.3, but all three oracle stages are evaluated with the learned FM particle reference.",
            "No CNF density is reconstructed. Reference marginals are FM rollout particles only.",
            "All candidate designs share the same scientific scenarios, microscopic sample bank and detector-normal draws within the selection bank (common random numbers).",
            "The validation bank is disjoint from all selection trials and is used only after robust_d4 has been selected.",
            "The finite-measurement GLS curve is constrained to the physical sensor hull intersected with the learned-FM particle hull at every evaluation time.",
            "Empirical calibration and ESS gates are numerical/reference-overlap validity checks and are not part of the design objective.",
            "The oracle is discrete at angle_n resolution; it is not a proof of the exact continuous-angle optimum.",
            "Frozen Lift, Tangent-TC and Full-TC are inserted explicitly even if they are not dense-grid nodes.",
            "Finite action is evaluated only after the population-law and finite-risk screens, preserving the Stage C.3 lexicographic design principle.",
            "Stage-1 and Stage-2 law screens use a law-only evaluator: the I-projection weights and MMD are unchanged, while lambda_dot, forcing, source rasterization, and Poisson solves are skipped because they cannot affect law-screen membership.",
            "The 2D GLS feasibility projection is solved by exact H-metric projection onto the precomputed beta-feasibility polygon rather than repeated generic SLSQP calls.",
            "Matched exact-target Poisson action is optional (--compute-matched-exact-action) because it is diagnostic and does not enter D4 selection or the primary finite-action comparison.",
        ],
        "wall_seconds": float(time.time() - wall0),
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    print("\n" + "=" * 108)
    print("Stage D.4 oracle result")
    print("=" * 108)
    print(
        f"Population best : {population_best['theta_deg']} | "
        f"L_FM={population_best['population_law']:.8e}"
    )
    print(
        f"Finite-law best : {finite_best['theta_deg']} | "
        f"R_FM,N={finite_best['finite_risk']['mean']:.8e} +/- {finite_best['finite_risk']['se']:.2e}"
    )
    print(
        f"Robust D4       : {robust['theta_deg']} | "
        f"L_FM={robust['population_law']:.8e} | "
        f"R_FM,N={robust['finite_risk']['mean']:.8e} +/- {robust['finite_risk']['se']:.2e} | "
        f"A_FM,N={robust['finite_action']['mean']:.6e} +/- {robust['finite_action']['se']:.2e}"
    )
    rv = validation["robust_d4"]["summary"]
    lv = validation["lift"]["summary"]
    fv = validation["full"]["summary"]
    print("Independent validation:")
    print(
        f"  Robust D4: R={rv['finite_heldout_mmd2']['mean']:.8e} +/- {rv['finite_heldout_mmd2']['se']:.2e} | "
        f"A={rv['finite_action']['mean']:.6e} +/- {rv['finite_action']['se']:.2e}"
    )
    print(
        f"  Lift     : R={lv['finite_heldout_mmd2']['mean']:.8e} +/- {lv['finite_heldout_mmd2']['se']:.2e} | "
        f"A={lv['finite_action']['mean']:.6e} +/- {lv['finite_action']['se']:.2e}"
    )
    print(
        f"  Full-TC  : R={fv['finite_heldout_mmd2']['mean']:.8e} +/- {fv['finite_heldout_mmd2']['se']:.2e} | "
        f"A={fv['finite_action']['mean']:.6e} +/- {fv['finite_action']['se']:.2e}"
    )
    comp = validation_comparisons["robust_vs_lift_action_reduction"]
    print(
        f"  Robust-vs-Lift action reduction: ratio-of-means={100*comp['ratio_of_means_reduction']:.2f}% | "
        f"paired={100*comp['mean_paired_reduction']:.2f}% +/- {100*comp['se_paired_reduction']:.2f}%"
    )
    print("=" * 108)

    out = Path(args.output)
    out.write_text(json.dumps(jsonify(payload), indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"Saved diagnostics: {out.resolve()}")


if __name__ == "__main__":
    main()
