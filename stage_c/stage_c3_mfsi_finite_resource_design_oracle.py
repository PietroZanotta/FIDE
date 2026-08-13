#!/usr/bin/env python3
"""
Stage C.3: finite-resource-aware MFSI experimental-design oracle.

This script keeps the validated Stage B numerical backend and changes only the
finite-measurement layer.  The central principle is that measurement uncertainty
enters ONCE, through the estimator of the moment trajectory.  Each realized
trajectory is then reconstructed by the ordinary hard-fiber MFSI I-projection.

Observation model
-----------------
At acquisition times t_k, draw N microscopic states and report empirical sensor
means, with optional additive detector noise.  The full two-sensor covariance is

    V_eta(t_k) = Cov_{P_tk}[Phi_eta] / N + sigma_obs^2 I.

Endpoints are exact anchors.  Interior observations are fitted with the same
endpoint-anchored quadratic bridge for every K,

    c_hat(t) = (1-t)c0 + t c1 + beta_hat t(1-t),

using full-covariance GLS.  The GLS covariance Cov(beta_hat) is retained as a
coherent path-level uncertainty model; it is NOT reintroduced as a second
pointwise likelihood inside MFSI.

Hard-fiber MFSI reconstruction
------------------------------
For every realized c_hat(t), solve the ordinary MFSI calibration

    E_{q_lambda}[Phi_eta] = c_hat(t)

and use the corresponding hard-fiber multiplier sensitivity

    C_q lambda_dot = c_hat_dot - (d/dt E_q[Phi_eta])_{lambda fixed},

where C_q = Cov_q(Phi_eta).  The resulting forcing is passed to the same
weighted-Poisson solve used by Stage B.

Finite-noise hard constraints can occasionally leave the feasible moment set.
To prevent a meaningless multiplier blow-up, the quadratic GLS coefficient is
projected, only when necessary, onto the convex feasible set induced by
conv{Phi_eta(x)} across all evaluation times.  The projection frequency and size
are reported as diagnostics.

Primary finite-resource quantities
-----------------------------------
For each frozen design and resource condition, report

  * R_N: Monte-Carlo finite-resource held-out law MMD^2,
  * Delta_N = R_N - L_infinity for the same design,
  * A_N: the actual MFSI weighted-Poisson action of the noisy reconstructed path,
  * A_infinity,matched: the exact-population action evaluated on the SAME scientific
    alpha trials and the SAME full time quadrature as A_N,
  * action inflation A_N/A_infinity,matched,
  * paired finite-resource action-reduction fractions between frozen designs.

Statistical-conditioning diagnostic
-----------------------------------
The local information-geometric sensitivity of hard MFSI is evaluated using

    E[ KL(q_{c+dc} || q_c) ] ~= 1/2 tr(C_q^dagger V_c),

with V_c(t) = t^2(1-t)^2 Cov(beta_hat).  This is a cheap design diagnostic for
noise amplification by the MFSI calibration map.  We also report

    tr(C_q^dagger V_c C_q^dagger),

which is the local variance trace of the multiplier perturbation.  These are
conditioning diagnostics, not replacements for the downstream finite-resource
law risk R_N.

By default the script can still reproduce the frozen-design Stage C.2 consequence study.
With --run-design-oracle it additionally performs a finite-resource sensor-design oracle:
first enforce population-law sufficiency, then finite-resource law sufficiency, then
minimize the actual finite-resource MFSI weighted-Poisson action among survivors.
All candidate designs share the same scientific scenarios, particle draws, and detector
perturbations (common random numbers).  Its action reporting is deliberately
matched to Stage B: both use the complete Stage-B trapezoidal time quadrature.
This makes the finite-resource action ratio and the exact-population action ratio
directly comparable on the same alpha trials.
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
import jax.scipy as jsp

from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull, QhullError

Array = jax.Array
PI = math.pi


# -----------------------------------------------------------------------------
# Dynamic loading of the validated Stage B.2 backend
# -----------------------------------------------------------------------------


def load_backend(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Stage B.2 backend not found: {path}")
    spec = importlib.util.spec_from_file_location("stage_b2_backend", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def autodetect_backend() -> Path | None:
    candidates = [
        Path("stage_b2_transport_conditioned_design.py"),
        Path("stage_b2_transport_conditioned_design(1).py"),
        Path(__file__).with_name("stage_b2_transport_conditioned_design.py"),
        Path(__file__).with_name("stage_b2_transport_conditioned_design(1).py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Stage C configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CConfig:
    preset: str = "quick"
    trials: int = 8
    action_trials: int = 2
    n_list: Tuple[int, ...] = (25, 100)
    k_list: Tuple[int, ...] = (5, 9)
    noise_list: Tuple[float, ...] = (0.0, 0.01)
    seed: int = 20260812

    # Numerical safeguards for hard-fiber calibration and GLS.
    variance_floor: float = 1.0e-10
    lambda_clip: float = 80.0
    newton_step_cap: float = 5.0
    quadratic_ridge_rel: float = 1.0e-12

    # Hard-fiber finite-measurement safeguards / diagnostics.
    feasibility_margin: float = 0.0
    conditioning_rcond: float = 1.0e-10

    # Stage C needs held-out time nodes between acquisitions.
    grid_n: int = 19
    time_n: int = 13
    alpha_eval_mode: str = "random"  # random or quadrature

    # Startup diagnostics.
    control_alpha_n: int = 3

    # Frozen Stage B.3 designs, in degrees.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)

    # Optional Stage C.3 finite-resource design oracle.
    oracle_angle_n: int = 13
    oracle_trials: int = 8
    oracle_action_trials: int = 8
    oracle_validation_trials: int = 8
    oracle_n: int = 100
    oracle_k: int = 7
    oracle_noise_std: float = 0.01
    oracle_tau_l: float = 0.05
    oracle_tau_r: float = 0.01


def preset_cconfig(name: str) -> CConfig:
    if name == "quick":
        return CConfig()
    if name == "reference":
        return CConfig(
            preset="reference",
            trials=24,
            action_trials=5,
            n_list=(25, 100, 400),
            k_list=(7, 11),
            noise_list=(0.0, 0.01),
            grid_n=39,
            time_n=21,
            oracle_angle_n=19,
            oracle_trials=12,
            oracle_action_trials=12,
            oracle_validation_trials=12,
            oracle_n=100,
            oracle_k=11,
            oracle_noise_std=0.01,
        )
    if name == "confirm":
        return CConfig(
            preset="confirm",
            trials=50,
            action_trials=8,
            n_list=(25, 50, 100, 250, 1000),
            k_list=(7, 11, 17),
            noise_list=(0.0, 0.01),
            grid_n=65,
            time_n=27,
            oracle_angle_n=37,
            oracle_trials=24,
            oracle_action_trials=24,
            oracle_validation_trials=24,
            oracle_n=100,
            oracle_k=17,
            oracle_noise_std=0.01,
        )
    raise ValueError(f"Unknown Stage C preset {name!r}")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def jsonify(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonify(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, jax.Array):
        return np.asarray(x).tolist()
    if isinstance(x, Mapping):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    return x


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


def mean_se(x: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(a))
    if a.size == 1:
        return mean, float("nan")
    return mean, float(np.std(a, ddof=1) / np.sqrt(a.size))


def trap_average(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).copy()
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float64)
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w * v) / denom)


def degrees_to_eta(pair_deg: Tuple[float, float]) -> np.ndarray:
    return np.deg2rad(np.asarray(pair_deg, dtype=np.float64))


def nested_acquisition_sets(time_n: int, k_list: Sequence[int]) -> Dict[int, np.ndarray]:
    """Build deterministic nested acquisition sets using farthest-point refinement.

    Endpoints are always included. For K1 < K2, the K1 set is a subset of the K2 set.
    """
    ks = sorted(set(int(k) for k in k_list))
    if not ks:
        raise ValueError("k_list must be nonempty")
    if ks[0] < 3:
        raise ValueError("Every K must be at least 3: two endpoint anchors plus one interior observation")
    if ks[-1] >= time_n:
        raise ValueError(f"max K={ks[-1]} must be smaller than time_n={time_n} to retain held-out times")

    selected_order = [0, time_n - 1]
    selected = {0, time_n - 1}

    # Refine the currently largest gap.  Unlike a naive farthest-point tie break,
    # this cannot spend several consecutive additions inside the same interval.
    while len(selected_order) < ks[-1]:
        srt = sorted(selected)
        gaps = [(hi - lo, lo, hi) for lo, hi in zip(srt[:-1], srt[1:]) if hi - lo > 1]
        if not gaps:
            raise ValueError("Not enough distinct grid nodes for requested acquisition budget")
        _, lo, hi = max(gaps, key=lambda x: (x[0], -x[1]))
        best = lo + (hi - lo) // 2
        if best in selected:
            best += 1
        selected.add(best)
        selected_order.append(best)

    return {k: np.asarray(sorted(selected_order[:k]), dtype=int) for k in ks}


# -----------------------------------------------------------------------------
# Fixed-capacity endpoint-anchored quadratic temporal model
# -----------------------------------------------------------------------------


def fit_quadratic_bridge_gls(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    V_obs: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    t_eval: np.ndarray,
    ridge_rel: float,
    variance_floor: float,
    hull_equations: np.ndarray | None = None,
    feasibility_margin: float = 0.0,
) -> Dict[str, np.ndarray | float | bool]:
    """Fit c(t)=(1-t)c0+t*c1+beta*t(1-t) with full-covariance GLS.

    The unconstrained GLS covariance is
        H_reg^{-1} H_data H_reg^{-T}.

    If ``hull_equations`` is supplied, each row is [a_1,...,a_m,b] with
    a^T c + b <= 0 describing the convex moment polytope.  The fitted beta is
    then projected by the SAME quadratic GLS objective onto the intersection of
    those constraints across every evaluation time.  Since beta=0 gives the
    straight segment between feasible endpoints, the constrained problem is
    feasible whenever the endpoint anchors are feasible.

    The covariance returned is the local unconstrained GLS covariance.  When the
    feasibility projection is active it should be interpreted as a pre-projection
    local uncertainty diagnostic, not the exact covariance of the constrained
    estimator.
    """
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    V_obs = np.asarray(V_obs, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    t_eval = np.asarray(t_eval, dtype=np.float64)

    m = int(c0.size)
    H_data = np.zeros((m, m), dtype=np.float64)
    g = np.zeros(m, dtype=np.float64)
    used = 0

    for t, y, V in zip(t_obs, y_obs, V_obs):
        z = float(t * (1.0 - t))
        if abs(z) < 1e-14:
            continue
        bridge = (1.0 - t) * c0 + t * c1
        resid = y - bridge
        Vreg = 0.5 * (V + V.T) + variance_floor * np.eye(m)
        Vinv = np.linalg.inv(Vreg)
        H_data += (z * z) * Vinv
        g += z * (Vinv @ resid)
        used += 1

    if used == 0:
        raise ValueError("Quadratic bridge fit needs at least one interior acquisition time")

    scale = max(float(np.trace(H_data)) / max(m, 1), 1.0)
    Hreg = H_data + (ridge_rel * scale) * np.eye(m)
    beta_unconstrained = np.linalg.solve(Hreg, g)
    Hreg_inv = np.linalg.inv(Hreg)
    beta_cov = Hreg_inv @ H_data @ Hreg_inv.T
    beta_cov = 0.5 * (beta_cov + beta_cov.T)

    beta = beta_unconstrained.copy()
    projection_active = False
    projection_norm = 0.0
    constraint_success = True
    max_unconstrained_violation = 0.0

    if hull_equations is not None:
        eq = np.asarray(hull_equations, dtype=np.float64)
        if eq.ndim != 2 or eq.shape[1] != m + 1:
            raise ValueError("hull_equations must have shape [n_facets, m+1]")
        normals = eq[:, :m]
        offsets = eq[:, m]
        A_rows = []
        b_rows = []
        for t in t_eval:
            z = float(t * (1.0 - t))
            bridge = (1.0 - t) * c0 + t * c1
            if abs(z) < 1e-14:
                continue
            # normals @ (bridge + z beta) + offsets <= -margin
            A_rows.append(z * normals)
            b_rows.append(-offsets - normals @ bridge - feasibility_margin)
        if A_rows:
            A = np.concatenate(A_rows, axis=0)
            b = np.concatenate(b_rows, axis=0)
            violation = A @ beta_unconstrained - b
            max_unconstrained_violation = float(max(0.0, np.max(violation)))
            if max_unconstrained_violation > 1e-10:
                projection_active = True

                def obj(bb):
                    return 0.5 * float(bb @ Hreg @ bb) - float(g @ bb)

                def jac(bb):
                    return Hreg @ bb - g

                # beta=0 is a reliable feasible initializer because the convex
                # hull contains the straight segment between feasible endpoints.
                x0 = np.zeros(m, dtype=np.float64)
                if np.all(A @ beta_unconstrained <= b + 1e-10):
                    x0 = beta_unconstrained.copy()
                lc = LinearConstraint(A, -np.inf * np.ones_like(b), b)
                sol = minimize(
                    obj, x0, jac=jac, constraints=[lc], method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 500, "disp": False},
                )
                constraint_success = bool(sol.success) and np.all(A @ sol.x <= b + 1e-7)
                if constraint_success:
                    beta = np.asarray(sol.x, dtype=np.float64)
                else:
                    # Convex fallback that is feasible whenever the endpoint
                    # segment is feasible.  Expose the failure in diagnostics.
                    beta = np.zeros(m, dtype=np.float64)
                projection_norm = float(np.linalg.norm(beta - beta_unconstrained))

    z_eval = t_eval * (1.0 - t_eval)
    c = (1.0 - t_eval[:, None]) * c0[None, :] + t_eval[:, None] * c1[None, :] + z_eval[:, None] * beta[None, :]
    cdot = (c1 - c0)[None, :] + (1.0 - 2.0 * t_eval[:, None]) * beta[None, :]

    return {
        "beta": beta,
        "beta_unconstrained": beta_unconstrained,
        "beta_cov": beta_cov,
        "c": c,
        "cdot": cdot,
        "feasibility_projection_active": bool(projection_active),
        "feasibility_projection_norm": float(projection_norm),
        "constraint_solver_success": bool(constraint_success),
        "max_unconstrained_hull_violation": float(max_unconstrained_violation),
    }


# -----------------------------------------------------------------------------
# Shared paired finite-population trial bank
# -----------------------------------------------------------------------------


@dataclass
class SharedTrialData:
    alpha: float
    # Each stored vector has length N_max; smaller N uses a prefix.
    sample_indices: Dict[int, np.ndarray]
    # Common standard-normal detector perturbations, later scaled by sigma_obs.
    detector_z: Dict[int, np.ndarray]


def draw_shared_trial(
    model,
    alpha: float,
    master_acq_idx: np.ndarray,
    n_max: int,
    rng: np.random.Generator,
) -> SharedTrialData:
    samples: Dict[int, np.ndarray] = {}
    noise: Dict[int, np.ndarray] = {}
    times = np.asarray(model.times, dtype=np.float64)

    for idx in master_acq_idx:
        t = float(times[idx])
        _, pmass = model.external_q_mass(jnp.asarray(t), jnp.asarray(alpha))
        p = np.asarray(pmass, dtype=np.float64).reshape(-1)
        p = np.maximum(p, 0.0)
        p /= np.sum(p)
        samples[int(idx)] = rng.choice(p.size, size=n_max, replace=True, p=p)
        noise[int(idx)] = rng.standard_normal(2)

    return SharedTrialData(alpha=float(alpha), sample_indices=samples, detector_z=noise)


# -----------------------------------------------------------------------------
# Hard-fiber MFSI calibration + exact full-law action
# -----------------------------------------------------------------------------


class StageCInference:
    def __init__(self, model, cfg: CConfig):
        self.model = model
        self.cfg = cfg
        self.eye2 = jnp.eye(2, dtype=jnp.float64)
        self._build_jitted()

    def _population_sensor_cov(self, t: Array, alpha: Array, eta: Array) -> Array:
        phi, _ = self.model.sensor_fields(eta)
        _, pmass = self.model.external_q_mass(t, alpha)
        moment = jnp.sum(phi * pmass[None, ...], axis=(1, 2))
        centered = phi - moment[:, None, None]
        return jnp.einsum("myx,nyx,yx->mn", centered, centered, pmass)

    def _exact_measurement_dot(self, t: Array, alpha: Array, eta: Array) -> Array:
        return jax.jacfwd(self.model.measurement_grid, argnums=0)(t, alpha, eta)

    def _hard_lambda_raw(self, t: Array, eta: Array, c_target: Array) -> Array:
        """Hard MFSI multiplier for an arbitrary feasible target moment."""
        model = self.model
        phi, _ = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        flat_phi = phi.reshape((2, -1))
        log_base = jnp.log(jnp.maximum(qref_mass.reshape(-1), 1e-300))

        def moment_cov(lam):
            mass = model.tilted_mass_from_fields(lam, qref_mass, phi)
            moment = jnp.sum(phi * mass[None, ...], axis=(1, 2))
            centered = phi - moment[:, None, None]
            C = jnp.einsum("myx,nyx,yx->mn", centered, centered, mass)
            return mass, moment, C

        def dual(lam):
            return jsp.special.logsumexp(log_base + lam @ flat_phi) - lam @ c_target

        def body(_, lam):
            _, moment, C = moment_cov(lam)
            F = moment - c_target
            H = C + model.cfg.newton_ridge * self.eye2
            step = jnp.linalg.solve(H, F)
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(1.0, self.cfg.newton_step_cap / jnp.maximum(step_norm, 1e-12))
            scales = model.cfg.newton_damping * (0.5 ** jnp.arange(9, dtype=jnp.float64))
            cands = lam[None, :] - scales[:, None] * step[None, :]
            vals = jax.vmap(dual)(cands)
            best = cands[jnp.argmin(vals)]
            return jnp.clip(best, -self.cfg.lambda_clip, self.cfg.lambda_clip)

        return jax.lax.fori_loop(0, model.cfg.newton_steps, body, jnp.zeros(2, dtype=jnp.float64))

    def _hard_state(
        self,
        t: Array,
        alpha: Array,
        eta: Array,
        c_target: Array,
        c_dot: Array,
    ):
        model = self.model
        phi, grad_phi = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        lam = self._hard_lambda_raw(t, eta, c_target)
        qmass = model.tilted_mass_from_fields(lam, qref_mass, phi)
        q = qmass / model.cell_area
        moment = jnp.sum(phi * qmass[None, ...], axis=(1, 2))
        centered = phi - moment[:, None, None]
        C = jnp.einsum("myx,nyx,yx->mn", centered, centered, qmass)

        def fixed_lambda_moment(tt):
            _, qr_mass = model.reference_q_mass(tt)
            mass = model.tilted_mass_from_fields(lam, qr_mass, phi)
            return jnp.sum(phi * mass[None, ...], axis=(1, 2))

        dm_ref_dt = jax.jacfwd(fixed_lambda_moment)(t)
        H = C + model.cfg.newton_ridge * self.eye2
        lam_dot = jnp.linalg.solve(H, c_dot - dm_ref_dt)

        B = model.B_matrix(t)
        u = model.xy_flat @ B.T
        u = u.reshape((model.cfg.grid_n, model.cfg.grid_n, 2))
        jphi_u = jnp.einsum("myxc,yxc->myx", grad_phi, u)

        term_time = jnp.einsum("m,myx->yx", lam_dot, phi - moment[:, None, None])
        adv_scalar = jnp.einsum("m,myx->yx", lam, jphi_u)
        adv_centered = adv_scalar - jnp.sum(qmass * adv_scalar)
        h_raw = term_time + adv_centered
        mean_h_raw = jnp.sum(qmass * h_raw)
        h = h_raw - mean_h_raw

        calibration_resid = jnp.linalg.norm(moment - c_target)
        min_cov_eig = jnp.min(jnp.linalg.eigvalsh(C))
        true_c = model.measurement_grid(t, alpha, eta)
        moment_error = jnp.linalg.norm(moment - true_c)

        return {
            "q": q,
            "qmass": qmass,
            "h": h,
            "lam": lam,
            "moment": moment,
            "C": C,
            "calibration_resid": calibration_resid,
            "min_cov_eig": min_cov_eig,
            "moment_error": moment_error,
            "mean_h_raw": mean_h_raw,
        }

    def _law_metrics(self, t, alpha, eta, c_target, c_dot):
        st = self._hard_state(t, alpha, eta, c_target, c_dot)
        _, p_mass = self.model.external_q_mass(t, alpha)
        lift = self.model.gaussian_mmd2_mass(st["qmass"], p_mass)
        return jnp.array([
            lift,
            st["moment_error"],
            st["calibration_resid"],
            st["min_cov_eig"],
            jnp.linalg.norm(st["lam"]),
            jnp.abs(st["mean_h_raw"]),
        ])

    def _full_metrics(self, t, alpha, eta, c_target, c_dot):
        st = self._hard_state(t, alpha, eta, c_target, c_dot)
        full, _, poisson_rel, _, _ = self.model.poisson_solve(st["q"], st["h"])
        _, p_mass = self.model.external_q_mass(t, alpha)
        lift = self.model.gaussian_mmd2_mass(st["qmass"], p_mass)
        return jnp.array([lift, full, poisson_rel])

    def _conditioning_metrics(self, t, eta, c_target, beta_cov):
        """Local hard-MFSI information sensitivity induced by Cov(beta_hat)."""
        model = self.model
        phi, _ = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        lam = self._hard_lambda_raw(t, eta, c_target)
        qmass = model.tilted_mass_from_fields(lam, qref_mass, phi)
        moment = jnp.sum(phi * qmass[None, ...], axis=(1, 2))
        centered = phi - moment[:, None, None]
        C = jnp.einsum("myx,nyx,yx->mn", centered, centered, qmass)

        w, U = jnp.linalg.eigh(0.5 * (C + C.T))
        wmax = jnp.maximum(jnp.max(w), 1e-30)
        keep = w > (self.cfg.conditioning_rcond * wmax)
        winv = jnp.where(keep, 1.0 / jnp.maximum(w, 1e-30), 0.0)
        Cpinv = (U * winv[None, :]) @ U.T

        z = t * (1.0 - t)
        Vc = (z * z) * beta_cov
        expected_kl = 0.5 * jnp.trace(Cpinv @ Vc)
        lambda_var_trace = jnp.trace(Cpinv @ Vc @ Cpinv)
        return jnp.array([
            expected_kl,
            lambda_var_trace,
            jnp.min(w),
            jnp.trace(Vc),
        ])

    def _hard_lift(self, t, alpha, eta):
        model = self.model
        phi, _ = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        lam = model.solve_lambda(t, alpha, eta)
        qmass = model.tilted_mass_from_fields(lam, qref_mass, phi)
        _, p_mass = model.external_q_mass(t, alpha)
        return model.gaussian_mmd2_mass(qmass, p_mass)

    def _hard_full(self, t, alpha, eta):
        return self.model.one_time_lift_full(t, alpha, eta)

    def _build_jitted(self):
        self.population_cov_jit = jax.jit(self._population_sensor_cov)
        self.exact_measurement_dot_jit = jax.jit(self._exact_measurement_dot)
        self.hard_state_jit = jax.jit(self._hard_state)
        self.law_metrics_jit = jax.jit(self._law_metrics)
        self.full_metrics_jit = jax.jit(self._full_metrics)
        self.conditioning_metrics_jit = jax.jit(self._conditioning_metrics)
        self.hard_lift_jit = jax.jit(self._hard_lift)
        self.hard_full_jit = jax.jit(self._hard_full)


# -----------------------------------------------------------------------------
# Finite-population acquisition and measurement covariance at acquisition times
# -----------------------------------------------------------------------------


def design_measurements(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    n: int,
    obs_noise_std: float,
    variance_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return t_acq, empirical means, exact 2x2 mean covariances, exact means."""
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)
    phi, _ = model.sensor_fields(eta_j)
    phi_flat = np.asarray(phi, dtype=np.float64).reshape(2, -1)
    times = np.asarray(model.times, dtype=np.float64)

    ys: List[np.ndarray] = []
    Vs: List[np.ndarray] = []
    exacts: List[np.ndarray] = []

    for idx in acq_idx:
        t = float(times[idx])
        t_j = jnp.asarray(t, dtype=jnp.float64)
        exact = np.asarray(model.measurement_grid(t_j, alpha_j, eta_j), dtype=np.float64)
        Sigma = np.asarray(infer.population_cov_jit(t_j, alpha_j, eta_j), dtype=np.float64)
        V = Sigma / float(n) + (obs_noise_std ** 2) * np.eye(2)
        V = 0.5 * (V + V.T) + variance_floor * np.eye(2)

        draw = shared.sample_indices[int(idx)][:n]
        vals = phi_flat[:, draw].T  # [N,2]
        if idx == 0 or idx == len(times) - 1:
            y = exact.copy()
        else:
            y = np.mean(vals, axis=0) + obs_noise_std * shared.detector_z[int(idx)]

        ys.append(y)
        Vs.append(V)
        exacts.append(exact)

    return (
        times[acq_idx],
        np.asarray(ys, dtype=np.float64),
        np.asarray(Vs, dtype=np.float64),
        np.asarray(exacts, dtype=np.float64),
    )


def sensor_moment_hull_equations(model, eta: np.ndarray) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    """Half-space representation of conv{Phi_eta(x)} for the finite grid."""
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    phi, _ = model.sensor_fields(eta_j)
    points = np.asarray(phi, dtype=np.float64).reshape(2, -1).T
    points = np.unique(points, axis=0)
    if points.shape[0] < 3 or np.linalg.matrix_rank(points - np.mean(points, axis=0)) < 2:
        return None, {"available": False, "reason": "rank_deficient_moment_set", "vertices": 0}
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        return None, {"available": False, "reason": f"qhull_failure: {exc}", "vertices": 0}
    # scipy.spatial.ConvexHull.equations: normal @ x + offset <= 0 inside.
    return np.asarray(hull.equations, dtype=np.float64), {
        "available": True,
        "reason": "ok",
        "vertices": int(len(hull.vertices)),
        "facets": int(hull.equations.shape[0]),
    }


def statistical_conditioning_metrics(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    alpha: float,
    acq_idx: np.ndarray,
    n: int,
    obs_noise_std: float,
    cfg: CConfig,
    hull_equations: np.ndarray | None,
) -> Dict[str, Any]:
    """Local hard-MFSI statistical conditioning for one scenario/resource condition.

    The central path is the GLS fit to the exact acquisition means.  Measurement
    uncertainty enters through the coherent coefficient covariance Cov(beta_hat),
    giving V_c(t)=z(t)^2 Cov(beta_hat).  The primary local diagnostic is

        1/2 tr(C_q^dagger V_c),

    the second-order expected KL perturbation of the hard I-projected law.
    """
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_j, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_j, eta_j), dtype=np.float64)

    exact_acq = []
    V_acq = []
    for idx in acq_idx:
        t = float(times[idx])
        t_j = jnp.asarray(t, dtype=jnp.float64)
        exact_acq.append(np.asarray(model.measurement_grid(t_j, alpha_j, eta_j), dtype=np.float64))
        Sigma = np.asarray(infer.population_cov_jit(t_j, alpha_j, eta_j), dtype=np.float64)
        m = Sigma.shape[0]
        V = Sigma / float(n) + (obs_noise_std ** 2) * np.eye(m)
        V_acq.append(0.5 * (V + V.T) + cfg.variance_floor * np.eye(m))
    exact_acq = np.asarray(exact_acq, dtype=np.float64)
    V_acq = np.asarray(V_acq, dtype=np.float64)

    mean_curve = fit_quadratic_bridge_gls(
        times[acq_idx], exact_acq, V_acq, c0, c1, times,
        cfg.quadratic_ridge_rel, cfg.variance_floor,
        hull_equations=hull_equations,
        feasibility_margin=cfg.feasibility_margin,
    )

    rows = []
    for i, t in enumerate(times):
        rows.append(np.asarray(infer.conditioning_metrics_jit(
            jnp.asarray(t, dtype=jnp.float64),
            eta_j,
            jnp.asarray(mean_curve["c"][i]),
            jnp.asarray(mean_curve["beta_cov"]),
        ), dtype=np.float64))
    rows = np.asarray(rows, dtype=np.float64)
    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    eigs = np.linalg.eigvalsh(np.asarray(mean_curve["beta_cov"], dtype=np.float64))
    return {
        "stat_expected_local_kl": trap_average(rows[:, 0], tw, interior),
        "stat_lambda_variance_trace": trap_average(rows[:, 1], tw, interior),
        "stat_min_projected_cov_eig": float(np.min(rows[interior, 2])),
        "stat_mean_moment_variance_trace": trap_average(rows[:, 3], tw, interior),
        "beta_cov_trace": float(np.trace(mean_curve["beta_cov"])),
        "beta_cov_max_eig": float(np.max(eigs)),
        "conditioning_center_feasibility_projection_active": bool(mean_curve["feasibility_projection_active"]),
    }


# -----------------------------------------------------------------------------
# One frozen-design evaluation
# -----------------------------------------------------------------------------


def evaluate_design_trial(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    obs_noise_std: float,
    cfg: CConfig,
    compute_action: bool,
    hull_equations: np.ndarray | None,
) -> Dict[str, float]:
    t_acq, y_acq, V_acq, exact_acq = design_measurements(
        model=model,
        infer=infer,
        eta=eta,
        shared=shared,
        acq_idx=acq_idx,
        n=n,
        obs_noise_std=obs_noise_std,
        variance_floor=cfg.variance_floor,
    )

    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_j, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_j, eta_j), dtype=np.float64)

    curve = fit_quadratic_bridge_gls(
        t_obs=t_acq,
        y_obs=y_acq,
        V_obs=V_acq,
        c0=c0,
        c1=c1,
        t_eval=times,
        ridge_rel=cfg.quadratic_ridge_rel,
        variance_floor=cfg.variance_floor,
        hull_equations=hull_equations,
        feasibility_margin=cfg.feasibility_margin,
    )

    tw = np.asarray(model.time_w, dtype=np.float64)
    heldout = np.zeros(len(times), dtype=bool)
    heldout[np.asarray(test_idx, dtype=int)] = True
    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    noisy_rows = []
    oracle_lifts = []
    full_rows = []
    oracle_full_rows = []
    exact_curve = []

    for i, t in enumerate(times):
        exact_curve.append(np.asarray(model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64))
        args = (
            jnp.asarray(t, dtype=jnp.float64),
            alpha_j,
            eta_j,
            jnp.asarray(curve["c"][i]),
            jnp.asarray(curve["cdot"][i]),
        )
        noisy_rows.append(np.asarray(infer.law_metrics_jit(*args), dtype=np.float64))
        oracle_lifts.append(float(infer.hard_lift_jit(jnp.asarray(t), alpha_j, eta_j)))
        if compute_action:
            full_rows.append(np.asarray(infer.full_metrics_jit(*args), dtype=np.float64))
            oracle_full_rows.append(np.asarray(infer.hard_full_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64))

    noisy = np.asarray(noisy_rows, dtype=np.float64)
    oracle_lifts = np.asarray(oracle_lifts, dtype=np.float64)
    exact_curve = np.asarray(exact_curve, dtype=np.float64)

    conditioning = statistical_conditioning_metrics(
        model=model,
        infer=infer,
        eta=eta,
        alpha=shared.alpha,
        acq_idx=acq_idx,
        n=n,
        obs_noise_std=obs_noise_std,
        cfg=cfg,
        hull_equations=hull_equations,
    )

    result = {
        "heldout_mmd2": trap_average(noisy[:, 0], tw, heldout),
        "all_interior_mmd2": trap_average(noisy[:, 0], tw, interior),
        "oracle_heldout_mmd2": trap_average(oracle_lifts, tw, heldout),
        "delta_vs_population_heldout_mmd2": trap_average(noisy[:, 0] - oracle_lifts, tw, heldout),
        "heldout_moment_error": trap_average(noisy[:, 1], tw, heldout),
        "max_hard_calibration_resid": float(np.max(noisy[:, 2])),
        "min_projected_cov_eig": float(np.min(noisy[:, 3])),
        "max_lambda_norm": float(np.max(noisy[:, 4])),
        "max_mean_h_raw": float(np.max(noisy[:, 5])),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((y_acq - exact_acq) ** 2))),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum((curve["c"][interior] - exact_curve[interior]) ** 2, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(curve["c"][interior] - exact_curve[interior], axis=1))),
        "beta_norm": float(np.linalg.norm(curve["beta"])),
        "beta_unconstrained_norm": float(np.linalg.norm(curve["beta_unconstrained"])),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
        "constraint_solver_success": float(bool(curve["constraint_solver_success"])),
        "max_unconstrained_hull_violation": float(curve["max_unconstrained_hull_violation"]),
        **conditioning,
    }

    if compute_action:
        full_rows = np.asarray(full_rows, dtype=np.float64)
        oracle_full_rows = np.asarray(oracle_full_rows, dtype=np.float64)
        # IMPORTANT: match Stage B exactly on the time integration convention.
        # Stage B uses sum_t time_w[t] * A_t over ALL time nodes, including the
        # half-weighted endpoints.  Do not renormalize after dropping endpoints.
        finite_action = float(np.sum(tw * full_rows[:, 1]))
        oracle_action = float(np.sum(tw * oracle_full_rows[:, 1]))
        result.update({
            "finite_full_action": finite_action,
            "oracle_full_action_trial": oracle_action,
            "action_inflation_ratio": finite_action / max(oracle_action, 1e-14),
            "action_excess": finite_action - oracle_action,
            "max_poisson_rel_resid": float(np.max(full_rows[:, 2])),
        })
    else:
        result.update({
            "finite_full_action": float("nan"),
            "oracle_full_action_trial": float("nan"),
            "action_inflation_ratio": float("nan"),
            "action_excess": float("nan"),
            "max_poisson_rel_resid": float("nan"),
        })

    return result


# -----------------------------------------------------------------------------
# Startup controls
# -----------------------------------------------------------------------------


def population_design_check(model, designs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = {}
    for name, eta in designs.items():
        row = np.asarray(model.design_metrics_jit(jnp.asarray(eta, dtype=jnp.float64)), dtype=np.float64)
        out[name] = {
            "theta_deg": np.rad2deg(eta).tolist(),
            "lift_mmd2": float(row[0]),
            "full_action": float(row[1]),
            "tangent_action": float(row[2]),
            "hidden_action": float(row[3]),
            "max_poisson_rel_resid": float(row[6]),
            "max_calibration_resid": float(row[8]),
        }
    return out


def quadratic_temporal_bias_control(
    model,
    infer: StageCInference,
    designs: Dict[str, np.ndarray],
    acq_sets: Dict[int, np.ndarray],
    cfg: CConfig,
) -> Dict[str, Any]:
    """Fit exact Stage B moments with the quadratic model and report finite-grid bias."""
    times = np.asarray(model.times, dtype=np.float64)
    alphas = np.linspace(model.cfg.alpha_min, model.cfg.alpha_max, cfg.control_alpha_n)
    rows = []

    for alpha in alphas:
        alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
        for dname, eta in designs.items():
            eta_j = jnp.asarray(eta, dtype=jnp.float64)
            exact_all = np.asarray([
                np.asarray(model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
                for t in times
            ])
            c0, c1 = exact_all[0], exact_all[-1]
            for k, idx in acq_sets.items():
                # Isotropic weights: this control measures model-class bias, not statistical weighting.
                V_obs = np.repeat(np.eye(2)[None, :, :], len(idx), axis=0)
                curve = fit_quadratic_bridge_gls(
                    t_obs=times[idx],
                    y_obs=exact_all[idx],
                    V_obs=V_obs,
                    c0=c0,
                    c1=c1,
                    t_eval=times,
                    ridge_rel=cfg.quadratic_ridge_rel,
                    variance_floor=cfg.variance_floor,
                )
                err = np.linalg.norm(curve["c"] - exact_all, axis=1)
                rows.append({
                    "alpha": float(alpha),
                    "design": dname,
                    "K": int(k),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "max_error": float(np.max(err)),
                })

    return {
        "rows": rows,
        "worst_rmse": float(max(r["rmse"] for r in rows)),
        "worst_max_error": float(max(r["max_error"] for r in rows)),
        "note": "Nonzero values are finite-grid normalization bias relative to the continuous quadratic mixture identity.",
    }


def arbitrary_target_hard_solver_control(
    model,
    infer: StageCInference,
    designs: Dict[str, np.ndarray],
    cfg: CConfig,
) -> Dict[str, Any]:
    """Check custom arbitrary-target hard calibration against the Stage B hard solve."""
    times_all = np.asarray(model.times, dtype=np.float64)
    control_idx = np.unique(np.rint(np.linspace(0, len(times_all) - 1, 5)).astype(int))
    times = times_all[control_idx]
    alpha = 0.5 * (model.cfg.alpha_min + model.cfg.alpha_max)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    rows = []

    for dname, eta in designs.items():
        eta_j = jnp.asarray(eta, dtype=jnp.float64)
        max_mass_l1 = 0.0
        max_lift_diff = 0.0
        max_resid = 0.0
        for t in times:
            t_j = jnp.asarray(t, dtype=jnp.float64)
            c = model.measurement_grid(t_j, alpha_j, eta_j)
            cdot = infer.exact_measurement_dot_jit(t_j, alpha_j, eta_j)
            st = infer.hard_state_jit(t_j, alpha_j, eta_j, c, cdot)

            phi, _ = model.sensor_fields(eta_j)
            _, qref_mass = model.reference_q_mass(t_j)
            lam_backend = model.solve_lambda(t_j, alpha_j, eta_j)
            qmass_backend = model.tilted_mass_from_fields(lam_backend, qref_mass, phi)
            qmass_custom = np.asarray(st["qmass"], dtype=np.float64)
            qmass_backend_np = np.asarray(qmass_backend, dtype=np.float64)
            mass_l1 = float(np.sum(np.abs(qmass_custom - qmass_backend_np)))

            _, p_mass = model.external_q_mass(t_j, alpha_j)
            lift_custom = float(model.gaussian_mmd2_mass(st["qmass"], p_mass))
            lift_backend = float(model.gaussian_mmd2_mass(qmass_backend, p_mass))
            max_mass_l1 = max(max_mass_l1, mass_l1)
            max_lift_diff = max(max_lift_diff, abs(lift_custom - lift_backend))
            max_resid = max(max_resid, float(st["calibration_resid"]))

        rows.append({
            "design": dname,
            "max_mass_l1": max_mass_l1,
            "max_lift_abs_diff": max_lift_diff,
            "max_hard_calibration_resid": max_resid,
        })

    return {
        "alpha": float(alpha),
        "time_indices": control_idx.tolist(),
        "rows": rows,
        "worst_mass_l1": float(max(r["max_mass_l1"] for r in rows)),
        "worst_lift_abs_diff": float(max(r["max_lift_abs_diff"] for r in rows)),
    }


# -----------------------------------------------------------------------------
# Monte Carlo aggregation
# -----------------------------------------------------------------------------


METRICS = (
    "heldout_mmd2",
    "all_interior_mmd2",
    "oracle_heldout_mmd2",
    "delta_vs_population_heldout_mmd2",
    "heldout_moment_error",
    "acquisition_mean_rmse",
    "quadratic_moment_rmse",
    "quadratic_moment_max_error",
    "beta_norm",
    "beta_unconstrained_norm",
    "max_hard_calibration_resid",
    "min_projected_cov_eig",
    "max_lambda_norm",
    "max_mean_h_raw",
    "feasibility_projection_active",
    "feasibility_projection_norm",
    "constraint_solver_success",
    "max_unconstrained_hull_violation",
    "stat_expected_local_kl",
    "stat_lambda_variance_trace",
    "stat_min_projected_cov_eig",
    "stat_mean_moment_variance_trace",
    "beta_cov_trace",
    "beta_cov_max_eig",
    "finite_full_action",
    "oracle_full_action_trial",
    "action_inflation_ratio",
    "action_excess",
    "max_poisson_rel_resid",
)


def summarize_trials(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in METRICS:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        m, se = mean_se(vals)
        out[key] = {"mean": m, "se": se, "n": int(vals.size)}
    return out


def paired_comparison(
    rows_a: List[Dict[str, float]],
    rows_b: List[Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    a = np.asarray([r[metric] for r in rows_a], dtype=np.float64)
    b = np.asarray([r[metric] for r in rows_b], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    m, se = mean_se(d)
    return {
        "mean_difference_a_minus_b": m,
        "se_difference": se,
        "n": int(d.size),
        "a_better_fraction": float(np.mean(d < 0.0)) if d.size else float("nan"),
    }


def paired_reduction_fraction(
    rows_num: List[Dict[str, float]],
    rows_den: List[Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    """Action reduction summaries for paired trials.

    The PRIMARY estimand matches the Stage B population statement:

        1 - E[A_num] / E[A_den].

    Its standard error is a paired delta-method estimate using the empirical
    covariance of (A_num, A_den).  We additionally report the mean of the
    trial-wise reductions 1-A_num_i/A_den_i as a conditional diagnostic.
    """
    a = np.asarray([r[metric] for r in rows_num], dtype=np.float64)
    b = np.asarray([r[metric] for r in rows_den], dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (b > 0.0)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return {
            "expected_action_reduction_fraction": float("nan"),
            "se_expected_action_reduction_fraction": float("nan"),
            "paired_mean_reduction_fraction": float("nan"),
            "se_paired_mean_reduction_fraction": float("nan"),
            "n": 0,
            "positive_reduction_fraction": float("nan"),
            "min_paired_reduction_fraction": float("nan"),
            "max_paired_reduction_fraction": float("nan"),
        }

    ma = float(np.mean(a))
    mb = float(np.mean(b))
    expected_red = 1.0 - ma / mb
    if a.size > 1:
        cov_obs = np.cov(np.vstack([a, b]), ddof=1)
        cov_mean = cov_obs / float(a.size)
        grad = np.asarray([-1.0 / mb, ma / (mb * mb)], dtype=np.float64)
        var = float(grad @ cov_mean @ grad)
        se_expected = float(np.sqrt(max(var, 0.0)))
    else:
        se_expected = float("nan")

    paired = 1.0 - a / b
    pm, pse = mean_se(paired)
    return {
        "expected_action_reduction_fraction": float(expected_red),
        "se_expected_action_reduction_fraction": float(se_expected),
        "paired_mean_reduction_fraction": float(pm),
        "se_paired_mean_reduction_fraction": float(pse),
        "n": int(a.size),
        "positive_reduction_fraction": float(np.mean(paired > 0.0)),
        "min_paired_reduction_fraction": float(np.min(paired)),
        "max_paired_reduction_fraction": float(np.max(paired)),
    }


def build_trial_bank(
    model,
    cfg: CConfig,
    master_acq_idx: np.ndarray,
) -> List[SharedTrialData]:
    n_max = max(cfg.n_list)
    bank: List[SharedTrialData] = []
    for trial in range(cfg.trials):
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, trial]))
        if cfg.alpha_eval_mode == "quadrature":
            alpha_vals = np.asarray(model.alphas, dtype=np.float64)
            alpha = float(alpha_vals[trial % len(alpha_vals)])
        else:
            alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))
        bank.append(draw_shared_trial(model, alpha, master_acq_idx, n_max, rng))
    return bank


def run_condition(
    model,
    infer: StageCInference,
    designs: Dict[str, np.ndarray],
    cfg: CConfig,
    trial_bank: List[SharedTrialData],
    acq_idx: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    k: int,
    obs_noise_std: float,
) -> Dict[str, Any]:
    rows = {name: [] for name in designs}
    hulls = {name: sensor_moment_hull_equations(model, eta) for name, eta in designs.items()}

    for trial, shared in enumerate(trial_bank):
        do_action = trial < cfg.action_trials
        for name, eta in designs.items():
            rows[name].append(evaluate_design_trial(
                model=model,
                infer=infer,
                eta=eta,
                shared=shared,
                acq_idx=acq_idx,
                test_idx=test_idx,
                n=n,
                obs_noise_std=obs_noise_std,
                cfg=cfg,
                compute_action=do_action,
                hull_equations=hulls[name][0],
            ))

        print(
            f"    condition N={n:4d}, K={k:2d}, noise={obs_noise_std:.4f}: "
            f"trial {trial + 1}/{cfg.trials}",
            flush=True,
        )

    summary = {name: summarize_trials(rr) for name, rr in rows.items()}
    comparisons = {
        "full_vs_lift_heldout_mmd2": paired_comparison(rows["full"], rows["lift"], "heldout_mmd2"),
        "full_vs_tangent_heldout_mmd2": paired_comparison(rows["full"], rows["tangent"], "heldout_mmd2"),
        "full_vs_lift_delta_vs_population_mmd2": paired_comparison(rows["full"], rows["lift"], "delta_vs_population_heldout_mmd2"),
        "full_vs_tangent_delta_vs_population_mmd2": paired_comparison(rows["full"], rows["tangent"], "delta_vs_population_heldout_mmd2"),
        "full_vs_lift_action_inflation": paired_comparison(rows["full"], rows["lift"], "action_inflation_ratio"),
        "full_vs_tangent_action_inflation": paired_comparison(rows["full"], rows["tangent"], "action_inflation_ratio"),
        "full_vs_lift_stat_expected_local_kl": paired_comparison(rows["full"], rows["lift"], "stat_expected_local_kl"),
        "full_vs_tangent_stat_expected_local_kl": paired_comparison(rows["full"], rows["tangent"], "stat_expected_local_kl"),
        "full_vs_lift_finite_action": paired_comparison(rows["full"], rows["lift"], "finite_full_action"),
        "full_vs_tangent_finite_action": paired_comparison(rows["full"], rows["tangent"], "finite_full_action"),
        "full_vs_lift_finite_action_reduction_fraction": paired_reduction_fraction(
            rows["full"], rows["lift"], "finite_full_action"
        ),
        "full_vs_tangent_finite_action_reduction_fraction": paired_reduction_fraction(
            rows["full"], rows["tangent"], "finite_full_action"
        ),
        "full_vs_lift_matched_oracle_action_reduction_fraction": paired_reduction_fraction(
            rows["full"], rows["lift"], "oracle_full_action_trial"
        ),
        "full_vs_tangent_matched_oracle_action_reduction_fraction": paired_reduction_fraction(
            rows["full"], rows["tangent"], "oracle_full_action_trial"
        ),
    }

    return {
        "N": int(n),
        "K": int(k),
        "obs_noise_std": float(obs_noise_std),
        "acquisition_indices": acq_idx.tolist(),
        "acquisition_times": np.asarray(model.times, dtype=np.float64)[acq_idx].tolist(),
        "common_test_indices": np.asarray(test_idx, dtype=int).tolist(),
        "common_test_times": np.asarray(model.times, dtype=np.float64)[np.asarray(test_idx, dtype=int)].tolist(),
        "design_summary": summary,
        "paired_comparisons": comparisons,
        "moment_hull_metadata": {name: meta for name, (_, meta) in hulls.items()},
    }


def print_condition_summary(cond: Dict[str, Any]) -> None:
    print("\n" + "-" * 96)
    print(
        f"Stage C condition: N={cond['N']}, K={cond['K']}, "
        f"detector noise std={cond['obs_noise_std']:.4f}"
    )
    print("-" * 96)
    for name, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        s = cond["design_summary"][name]
        h = s["heldout_mmd2"]
        ex = s["delta_vs_population_heldout_mmd2"]
        ai = s["action_inflation_ratio"]
        fa = s["finite_full_action"]
        oa = s["oracle_full_action_trial"]
        qrmse = s["quadratic_moment_rmse"]
        skl = s["stat_expected_local_kl"]
        fproj = s["feasibility_projection_active"]
        print(
            f"{label:10s} held-out MMD^2={h['mean']:.6e} ± {h['se']:.2e} | "
            f"Delta_N={ex['mean']:+.3e} ± {ex['se']:.2e} | "
            f"quad RMSE={qrmse['mean']:.2e} | local KL={skl['mean']:.3e} | "
            f"feas.proj={100*fproj['mean']:.1f}%"
        )
        print(
            f"{'':10s} finite A_N={fa['mean']:.6e} ± {fa['se']:.2e} | "
            f"matched A_inf={oa['mean']:.6e} ± {oa['se']:.2e} | "
            f"A_N/A_inf={ai['mean']:.4f} ± {ai['se']:.2e} (n={ai['n']})"
        )

    for key, label in (
        ("full_vs_lift_heldout_mmd2", "Full - Lift law"),
        ("full_vs_tangent_heldout_mmd2", "Full - Tangent law"),
        ("full_vs_lift_delta_vs_population_mmd2", "Full - Lift degr."),
        ("full_vs_tangent_delta_vs_population_mmd2", "Full - Tangent degr."),
        ("full_vs_lift_stat_expected_local_kl", "Full - Lift local KL"),
        ("full_vs_tangent_stat_expected_local_kl", "Full - Tangent local KL"),
        ("full_vs_lift_finite_action", "Full - Lift finite A"),
        ("full_vs_tangent_finite_action", "Full - Tangent finite A"),
    ):
        c = cond["paired_comparisons"][key]
        print(
            f"{label:24s}: paired Δ={c['mean_difference_a_minus_b']:+.3e} "
            f"± {c['se_difference']:.2e}; Full better in {100*c['a_better_fraction']:.1f}%"
        )

    fl_fin = cond["paired_comparisons"]["full_vs_lift_finite_action_reduction_fraction"]
    fl_orc = cond["paired_comparisons"]["full_vs_lift_matched_oracle_action_reduction_fraction"]
    ft_fin = cond["paired_comparisons"]["full_vs_tangent_finite_action_reduction_fraction"]
    ft_orc = cond["paired_comparisons"]["full_vs_tangent_matched_oracle_action_reduction_fraction"]
    print(
        f"Full vs Lift action reduction : finite={100*fl_fin['expected_action_reduction_fraction']:+.2f}% "
        f"± {100*fl_fin['se_expected_action_reduction_fraction']:.2f}% | "
        f"matched exact={100*fl_orc['expected_action_reduction_fraction']:+.2f}% "
        f"± {100*fl_orc['se_expected_action_reduction_fraction']:.2f}% (n={fl_fin['n']})"
    )
    print(
        f"Full vs Tangent reduction     : finite={100*ft_fin['expected_action_reduction_fraction']:+.2f}% "
        f"± {100*ft_fin['se_expected_action_reduction_fraction']:.2f}% | "
        f"matched exact={100*ft_orc['expected_action_reduction_fraction']:+.2f}% "
        f"± {100*ft_orc['se_expected_action_reduction_fraction']:.2f}% (n={ft_fin['n']})"
    )



# -----------------------------------------------------------------------------
# Stage C.3 finite-resource sensor-design oracle
# -----------------------------------------------------------------------------


def projective_angle_distance(a: float, b: float) -> float:
    """Distance between two pi-periodic sensor orientations in [0, pi/2]."""
    d = abs((float(a) - float(b)) % PI)
    return min(d, PI - d)


def canonical_eta(eta: np.ndarray) -> np.ndarray:
    x = np.mod(np.asarray(eta, dtype=np.float64), PI)
    return np.sort(x)


def oracle_candidate_designs(
    angle_n: int,
    min_sep_deg: float,
    extra_designs: Mapping[str, np.ndarray] | None = None,
) -> List[Dict[str, Any]]:
    """Permutation-reduced dense angle grid plus explicitly supplied designs."""
    if angle_n < 5:
        raise ValueError("oracle_angle_n must be at least 5")
    angles = np.linspace(0.0, PI, int(angle_n), endpoint=False, dtype=np.float64)
    min_sep = math.radians(float(min_sep_deg))
    raw: List[Tuple[np.ndarray, str]] = []
    for i, a in enumerate(angles):
        for j in range(i + 1, len(angles)):
            b = angles[j]
            if projective_angle_distance(a, b) >= min_sep - 1e-12:
                raw.append((np.asarray([a, b], dtype=np.float64), "grid"))
    if extra_designs:
        for name, eta in extra_designs.items():
            ce = canonical_eta(eta)
            if projective_angle_distance(ce[0], ce[1]) >= min_sep - 1e-12:
                raw.append((ce, f"frozen:{name}"))

    out: List[Dict[str, Any]] = []
    seen = set()
    for eta, source in raw:
        eta = canonical_eta(eta)
        key = tuple(np.round(eta, 12))
        if key in seen:
            # Preserve a frozen label if the same grid point was already present.
            if source.startswith("frozen:"):
                for row in out:
                    if tuple(np.round(row["eta"], 12)) == key:
                        row["sources"].append(source)
                        break
            continue
        seen.add(key)
        out.append({
            "eta": eta,
            "sources": [source],
            "theta_deg": np.rad2deg(eta).tolist(),
            "sensor_separation_deg": float(np.degrees(projective_angle_distance(eta[0], eta[1]))),
        })
    return out


def population_lift_only(model, infer: StageCInference, eta: np.ndarray) -> float:
    """Stage-B population law loss without any Poisson solves."""
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    alphas = np.asarray(model.alphas, dtype=np.float64)
    aw = np.asarray(model.alpha_w, dtype=np.float64)
    by_alpha = []
    for alpha in alphas:
        vals = [
            float(infer.hard_lift_jit(jnp.asarray(t), jnp.asarray(alpha), eta_j))
            for t in times
        ]
        by_alpha.append(float(np.sum(tw * np.asarray(vals, dtype=np.float64))))
    return float(np.sum(aw * np.asarray(by_alpha, dtype=np.float64)))


def evaluate_design_law_trial(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    test_idx: np.ndarray,
    n: int,
    obs_noise_std: float,
    cfg: CConfig,
    hull_equations: np.ndarray | None,
) -> Dict[str, float]:
    """Lightweight finite-resource law evaluation used in the dense oracle screen."""
    t_acq, y_acq, V_acq, _ = design_measurements(
        model, infer, eta, shared, acq_idx, n, obs_noise_std, cfg.variance_floor
    )
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_j, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_j, eta_j), dtype=np.float64)
    curve = fit_quadratic_bridge_gls(
        t_acq, y_acq, V_acq, c0, c1, times,
        cfg.quadratic_ridge_rel, cfg.variance_floor,
        hull_equations=hull_equations,
        feasibility_margin=cfg.feasibility_margin,
    )
    heldout = np.zeros(len(times), dtype=bool)
    heldout[np.asarray(test_idx, dtype=int)] = True
    finite = []
    oracle = []
    for i in np.flatnonzero(heldout):
        t = float(times[i])
        lm = np.asarray(infer.law_metrics_jit(
            jnp.asarray(t), alpha_j, eta_j,
            jnp.asarray(curve["c"][i]), jnp.asarray(curve["cdot"][i]),
        ), dtype=np.float64)
        finite.append((i, float(lm[0])))
        oracle.append((i, float(infer.hard_lift_jit(jnp.asarray(t), alpha_j, eta_j))))
    fvec = np.full(len(times), np.nan, dtype=np.float64)
    ovec = np.full(len(times), np.nan, dtype=np.float64)
    for i, v in finite:
        fvec[i] = v
    for i, v in oracle:
        ovec[i] = v
    return {
        "heldout_mmd2": trap_average(np.nan_to_num(fvec), tw, heldout),
        "oracle_heldout_mmd2": trap_average(np.nan_to_num(ovec), tw, heldout),
        "delta_vs_population_heldout_mmd2": trap_average(np.nan_to_num(fvec - ovec), tw, heldout),
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
        "feasibility_projection_norm": float(curve["feasibility_projection_norm"]),
    }


def evaluate_design_action_trial(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    n: int,
    obs_noise_std: float,
    cfg: CConfig,
    hull_equations: np.ndarray | None,
) -> Dict[str, float]:
    """Matched finite and exact MFSI action for one design/scenario."""
    t_acq, y_acq, V_acq, _ = design_measurements(
        model, infer, eta, shared, acq_idx, n, obs_noise_std, cfg.variance_floor
    )
    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), alpha_j, eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), alpha_j, eta_j), dtype=np.float64)
    curve = fit_quadratic_bridge_gls(
        t_acq, y_acq, V_acq, c0, c1, times,
        cfg.quadratic_ridge_rel, cfg.variance_floor,
        hull_equations=hull_equations,
        feasibility_margin=cfg.feasibility_margin,
    )
    finite_vals = []
    exact_vals = []
    max_resid = 0.0
    for i, t in enumerate(times):
        f = np.asarray(infer.full_metrics_jit(
            jnp.asarray(t), alpha_j, eta_j,
            jnp.asarray(curve["c"][i]), jnp.asarray(curve["cdot"][i]),
        ), dtype=np.float64)
        e = np.asarray(infer.hard_full_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64)
        finite_vals.append(float(f[1]))
        exact_vals.append(float(e[1]))
        max_resid = max(max_resid, float(f[2]))
    finite_action = float(np.sum(tw * np.asarray(finite_vals, dtype=np.float64)))
    exact_action = float(np.sum(tw * np.asarray(exact_vals, dtype=np.float64)))
    return {
        "finite_full_action": finite_action,
        "oracle_full_action_trial": exact_action,
        "action_inflation_ratio": finite_action / max(exact_action, 1e-14),
        "max_poisson_rel_resid": max_resid,
        "feasibility_projection_active": float(bool(curve["feasibility_projection_active"])),
    }


def _summarize_scalar_rows(rows: Sequence[Mapping[str, float]], key: str) -> Dict[str, float]:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    m, se = mean_se(vals)
    return {"mean": m, "se": se, "n": int(vals.size)}


def finite_resource_design_oracle(
    model,
    infer: StageCInference,
    cfg: CConfig,
    frozen_designs: Dict[str, np.ndarray],
    trial_bank: List[SharedTrialData],
    acq_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict[str, Any]:
    """Lexicographic finite-resource design oracle.

    1) population law: L_inf <= (1+tau_L) L_inf_star
    2) finite law:     R_N   <= (1+tau_R) R_N_star among population-feasible designs
    3) minimize finite MFSI action A_N among designs satisfying both.
    """
    candidates = oracle_candidate_designs(
        cfg.oracle_angle_n, model.cfg.min_sep_deg, extra_designs=frozen_designs
    )
    if cfg.oracle_trials < 1 or cfg.oracle_trials > len(trial_bank):
        raise ValueError("oracle_trials must be between 1 and the available trial-bank size")
    if cfg.oracle_action_trials < 1 or cfg.oracle_action_trials > cfg.oracle_trials:
        raise ValueError("oracle_action_trials must be between 1 and oracle_trials")
    if cfg.oracle_validation_trials < 1:
        raise ValueError("oracle_validation_trials must be at least 1")
    needed = cfg.oracle_trials + cfg.oracle_validation_trials
    if needed > len(trial_bank):
        raise ValueError("trial bank is too small for oracle selection plus independent validation")
    law_bank = trial_bank[: cfg.oracle_trials]
    action_bank = trial_bank[: cfg.oracle_action_trials]
    validation_bank = trial_bank[cfg.oracle_trials:needed]

    print("\n" + "=" * 96)
    print("Stage C.3 — finite-resource sensor-design oracle")
    print("=" * 96)
    print(
        f"Oracle condition: N={cfg.oracle_n}, K={cfg.oracle_k}, noise={cfg.oracle_noise_std:.4f}; "
        f"angle nodes={cfg.oracle_angle_n}, candidates={len(candidates)}, "
        f"law trials={cfg.oracle_trials}, action trials={cfg.oracle_action_trials}, "
        f"independent validation trials={cfg.oracle_validation_trials}"
    )
    print(
        f"Constraints: population tau_L={100*cfg.oracle_tau_l:.2f}% | "
        f"finite-risk tau_R={100*cfg.oracle_tau_r:.2f}% | "
        f"min separation={model.cfg.min_sep_deg:.2f}°"
    )

    # Stage 1: population law loss only (no Poisson solves).
    for j, cand in enumerate(candidates):
        cand["population_lift_mmd2"] = population_lift_only(model, infer, cand["eta"])
        if (j + 1) % max(1, len(candidates) // 10) == 0 or j + 1 == len(candidates):
            print(f"  population-law screen {j+1}/{len(candidates)}", flush=True)
    lstar = min(float(c["population_lift_mmd2"]) for c in candidates)
    lmax = (1.0 + cfg.oracle_tau_l) * lstar
    pop_feasible = [c for c in candidates if float(c["population_lift_mmd2"]) <= lmax]
    print(f"  population feasible: {len(pop_feasible)}/{len(candidates)}; L_inf*= {lstar:.6e}")

    # Stage 2: finite-resource law risk on common random numbers.
    for j, cand in enumerate(pop_feasible):
        hull, meta = sensor_moment_hull_equations(model, cand["eta"])
        law_rows = [
            evaluate_design_law_trial(
                model, infer, cand["eta"], shared, acq_idx, test_idx,
                cfg.oracle_n, cfg.oracle_noise_std, cfg, hull,
            )
            for shared in law_bank
        ]
        cand["moment_hull_metadata"] = meta
        cand["finite_risk"] = _summarize_scalar_rows(law_rows, "heldout_mmd2")
        cand["finite_degradation"] = _summarize_scalar_rows(law_rows, "delta_vs_population_heldout_mmd2")
        cand["feasibility_projection_rate"] = float(np.mean([r["feasibility_projection_active"] for r in law_rows]))
        if (j + 1) % max(1, len(pop_feasible) // 10) == 0 or j + 1 == len(pop_feasible):
            print(f"  finite-law screen {j+1}/{len(pop_feasible)}", flush=True)
    rstar = min(float(c["finite_risk"]["mean"]) for c in pop_feasible)
    rmax = (1.0 + cfg.oracle_tau_r) * rstar
    finite_feasible = [c for c in pop_feasible if float(c["finite_risk"]["mean"]) <= rmax]
    print(f"  finite-risk feasible: {len(finite_feasible)}/{len(pop_feasible)}; R_N*= {rstar:.6e}")

    # Stage 3: expensive finite-resource MFSI action only on the survivors.
    for j, cand in enumerate(finite_feasible):
        hull, _ = sensor_moment_hull_equations(model, cand["eta"])
        action_rows = [
            evaluate_design_action_trial(
                model, infer, cand["eta"], shared, acq_idx,
                cfg.oracle_n, cfg.oracle_noise_std, cfg, hull,
            )
            for shared in action_bank
        ]
        cand["finite_action"] = _summarize_scalar_rows(action_rows, "finite_full_action")
        cand["matched_exact_action"] = _summarize_scalar_rows(action_rows, "oracle_full_action_trial")
        cand["action_inflation_ratio"] = _summarize_scalar_rows(action_rows, "action_inflation_ratio")
        cand["max_poisson_rel_resid"] = float(max(r["max_poisson_rel_resid"] for r in action_rows))
        if (j + 1) % max(1, len(finite_feasible) // 10) == 0 or j + 1 == len(finite_feasible):
            print(f"  finite-action screen {j+1}/{len(finite_feasible)}", flush=True)

    if not finite_feasible:
        raise RuntimeError("No finite-risk-feasible design; increase oracle_tau_r or angle resolution")
    robust = min(finite_feasible, key=lambda c: float(c["finite_action"]["mean"]))
    finite_best = min(pop_feasible, key=lambda c: float(c["finite_risk"]["mean"]))
    population_best = min(candidates, key=lambda c: float(c["population_lift_mmd2"]))

    # Identify the frozen designs inside the evaluated candidate set when present.
    frozen_rows: Dict[str, Any] = {}
    for name, eta in frozen_designs.items():
        target = canonical_eta(eta)
        hit = min(candidates, key=lambda c: float(np.linalg.norm(c["eta"] - target)))
        if np.linalg.norm(hit["eta"] - target) < 1e-10:
            frozen_rows[name] = hit

    # Independent post-selection validation: no candidate in this bank influenced
    # the oracle choice.  Validate robust finite-TC against the frozen baselines.
    validation_designs: Dict[str, np.ndarray] = {"robust_finite_tc": np.asarray(robust["eta"], dtype=np.float64)}
    for name, eta in frozen_designs.items():
        validation_designs[f"frozen_{name}"] = np.asarray(eta, dtype=np.float64)
    validation: Dict[str, Any] = {}
    validation_trial_rows: Dict[str, Any] = {}

    for name, eta in validation_designs.items():
        hull, _ = sensor_moment_hull_equations(model, eta)
        law_rows = [
            evaluate_design_law_trial(
                model, infer, eta, shared, acq_idx, test_idx,
                cfg.oracle_n, cfg.oracle_noise_std, cfg, hull,
            )
            for shared in validation_bank
        ]
        action_rows = [
            evaluate_design_action_trial(
                model, infer, eta, shared, acq_idx,
                cfg.oracle_n, cfg.oracle_noise_std, cfg, hull,
            )
            for shared in validation_bank
        ]
        validation_trial_rows[name] = {
            "law": law_rows,
            "action": action_rows,
        }
        validation[name] = {
            "theta_deg": np.rad2deg(canonical_eta(eta)).tolist(),
            "population_lift_mmd2": population_lift_only(model, infer, eta),
            "finite_risk": _summarize_scalar_rows(law_rows, "heldout_mmd2"),
            "finite_degradation": _summarize_scalar_rows(law_rows, "delta_vs_population_heldout_mmd2"),
            "finite_action": _summarize_scalar_rows(action_rows, "finite_full_action"),
            "matched_exact_action": _summarize_scalar_rows(action_rows, "oracle_full_action_trial"),
            "action_inflation_ratio": _summarize_scalar_rows(action_rows, "action_inflation_ratio"),
            "feasibility_projection_rate": float(np.mean([r["feasibility_projection_active"] for r in law_rows])),
        }

        # -------------------------------------------------------------------------
    # Independent-validation bookkeeping
    # -------------------------------------------------------------------------

    vr = validation_trial_rows["robust_finite_tc"]
    vl = validation_trial_rows["frozen_lift"]
    vt = validation_trial_rows["frozen_tangent"]

    validation_comparisons = {
        # Absolute paired finite-law differences.
        "robust_vs_lift_law": paired_comparison(
            vr["law"], vl["law"], "heldout_mmd2"
        ),
        "robust_vs_tangent_law": paired_comparison(
            vr["law"], vt["law"], "heldout_mmd2"
        ),

        # Measurement-induced degradation differences.
        "robust_vs_lift_degradation": paired_comparison(
            vr["law"], vl["law"], "delta_vs_population_heldout_mmd2"
        ),
        "robust_vs_tangent_degradation": paired_comparison(
            vr["law"], vt["law"], "delta_vs_population_heldout_mmd2"
        ),

        # Absolute paired action differences.
        "robust_vs_lift_action": paired_comparison(
            vr["action"], vl["action"], "finite_full_action"
        ),
        "robust_vs_tangent_action": paired_comparison(
            vr["action"], vt["action"], "finite_full_action"
        ),

        # Primary action reduction estimand:
        # 1 - E[A_robust] / E[A_baseline].
        "robust_vs_lift_action_reduction": paired_reduction_fraction(
            vr["action"], vl["action"], "finite_full_action"
        ),
        "robust_vs_tangent_action_reduction": paired_reduction_fraction(
            vr["action"], vt["action"], "finite_full_action"
        ),
    }

    # Relative law penalty using ratio of mean finite risks.
    robust_risk_mean = validation["robust_finite_tc"]["finite_risk"]["mean"]
    lift_risk_mean = validation["frozen_lift"]["finite_risk"]["mean"]
    tangent_risk_mean = validation["frozen_tangent"]["finite_risk"]["mean"]

    validation_comparisons["robust_vs_lift_relative_law_penalty"] = (
        robust_risk_mean / lift_risk_mean - 1.0
    )
    validation_comparisons["robust_vs_tangent_relative_law_penalty"] = (
        robust_risk_mean / tangent_risk_mean - 1.0
    )

    def public_row(c: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: jsonify(v) for k, v in c.items() if k != "eta"} | {"eta_rad": jsonify(c["eta"])}

    print("\nOracle selections:")
    print(
        f"  population-law best : {population_best['theta_deg']} | "
        f"L_inf={population_best['population_lift_mmd2']:.6e}"
    )
    print(
        f"  finite-law best     : {finite_best['theta_deg']} | "
        f"R_N={finite_best['finite_risk']['mean']:.6e} ± {finite_best['finite_risk']['se']:.2e}"
    )
    print(
        f"  robust finite-TC    : {robust['theta_deg']} | "
        f"L_inf={robust['population_lift_mmd2']:.6e} | "
        f"R_N={robust['finite_risk']['mean']:.6e} ± {robust['finite_risk']['se']:.2e} | "
        f"A_N={robust['finite_action']['mean']:.6e} ± {robust['finite_action']['se']:.2e}"
    )
    rv = validation["robust_finite_tc"]
    print(
        f"  independent validation of robust design: R_N={rv['finite_risk']['mean']:.6e} ± {rv['finite_risk']['se']:.2e} | "
        f"A_N={rv['finite_action']['mean']:.6e} ± {rv['finite_action']['se']:.2e}"
    )

    lv = validation["frozen_lift"]
    tv = validation["frozen_tangent"]

    print("\nIndependent validation — same held-out bank:")
    print(
        f"  Lift       : "
        f"R_N={lv['finite_risk']['mean']:.6e} ± {lv['finite_risk']['se']:.2e} | "
        f"A_N={lv['finite_action']['mean']:.6e} ± {lv['finite_action']['se']:.2e}"
    )
    print(
        f"  Robust TC  : "
        f"R_N={rv['finite_risk']['mean']:.6e} ± {rv['finite_risk']['se']:.2e} | "
        f"A_N={rv['finite_action']['mean']:.6e} ± {rv['finite_action']['se']:.2e}"
    )
    print(
        f"  Tangent-TC : "
        f"R_N={tv['finite_risk']['mean']:.6e} ± {tv['finite_risk']['se']:.2e} | "
        f"A_N={tv['finite_action']['mean']:.6e} ± {tv['finite_action']['se']:.2e}"
    )

    c_law = validation_comparisons["robust_vs_lift_law"]
    c_deg = validation_comparisons["robust_vs_lift_degradation"]
    c_act = validation_comparisons["robust_vs_lift_action_reduction"]
    rel_law = validation_comparisons[
        "robust_vs_lift_relative_law_penalty"
    ]

    print(
        f"  Robust - Lift law       : "
        f"paired Δ={c_law['mean_difference_a_minus_b']:+.3e} "
        f"± {c_law['se_difference']:.2e}"
    )
    print(
        f"  Robust/Lift law penalty : {100.0 * rel_law:+.2f}%"
    )
    print(
        f"  Robust - Lift degr.     : "
        f"paired Δ={c_deg['mean_difference_a_minus_b']:+.3e} "
        f"± {c_deg['se_difference']:.2e}"
    )
    print(
        f"  Robust vs Lift A reduction: "
        f"{100.0 * c_act['expected_action_reduction_fraction']:+.2f}% "
        f"± {100.0 * c_act['se_expected_action_reduction_fraction']:.2f}% "
        f"(n={c_act['n']})"
    )

    return {
        "condition": {
            "N": int(cfg.oracle_n),
            "K": int(cfg.oracle_k),
            "obs_noise_std": float(cfg.oracle_noise_std),
            "angle_n": int(cfg.oracle_angle_n),
            "law_trials": int(cfg.oracle_trials),
            "action_trials": int(cfg.oracle_action_trials),
            "validation_trials": int(cfg.oracle_validation_trials),
            "tau_L": float(cfg.oracle_tau_l),
            "tau_R": float(cfg.oracle_tau_r),
            "min_sep_deg": float(model.cfg.min_sep_deg),
            "acquisition_indices": np.asarray(acq_idx, dtype=int).tolist(),
            "test_indices": np.asarray(test_idx, dtype=int).tolist(),
        },
        "population_lift_star": float(lstar),
        "population_lift_max": float(lmax),
        "finite_risk_star": float(rstar),
        "finite_risk_max": float(rmax),
        "candidate_count": int(len(candidates)),
        "population_feasible_count": int(len(pop_feasible)),
        "finite_risk_feasible_count": int(len(finite_feasible)),
        "population_law_best": public_row(population_best),
        "finite_law_best": public_row(finite_best),
        "robust_finite_tc": public_row(robust),
        "frozen_design_rows": {k: public_row(v) for k, v in frozen_rows.items()},
        "independent_validation": validation,
        "independent_validation_comparisons": validation_comparisons,
        "finite_risk_feasible_rows": [public_row(c) for c in finite_feasible],
        "population_feasible_rows": [public_row(c) for c in pop_feasible],
        "all_candidates_population": [
            {
                "theta_deg": c["theta_deg"],
                "eta_rad": jsonify(c["eta"]),
                "sensor_separation_deg": c["sensor_separation_deg"],
                "sources": c["sources"],
                "population_lift_mmd2": c["population_lift_mmd2"],
            }
            for c in candidates
        ],
        "interpretation": [
            "The oracle is lexicographic: population law sufficiency, then finite-resource law sufficiency, then minimum finite-resource MFSI action.",
            "All candidate designs use the same scientific scenarios, microscopic sample indices, acquisition sets, and detector standard-normal draws (common random numbers).",
            "Population-law screening uses the Stage-B hard-fiber MFSI law loss but avoids Poisson solves; finite action is evaluated only for designs satisfying both law constraints.",
            "The finite-risk oracle is a discrete dense-grid oracle at oracle_angle_n resolution, not a proof of the exact continuous-angle optimum.",
            "Frozen Lift/Tangent/Full designs are inserted explicitly even when they do not lie on the dense angular grid.",
            "The selected robust design and all frozen baselines are re-evaluated on an independent validation trial bank that was not used for candidate selection.",
        ],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None, help="Path to Stage B.2 Python backend")
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--action-trials", type=int, default=None)
    p.add_argument("--n-list", type=str, default=None, help="Comma-separated finite population sizes")
    p.add_argument("--k-list", type=str, default=None, help="Comma-separated acquisition-time counts")
    p.add_argument("--noise-list", type=str, default=None, help="Comma-separated additive mean-noise std values")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grid-n", type=int, default=None)
    p.add_argument("--time-n", type=int, default=None)
    p.add_argument("--alpha-eval-mode", choices=("random", "quadrature"), default=None)
    p.add_argument("--feasibility-margin", type=float, default=None, help="Interior margin for the hard-MFSI moment polytope")
    p.add_argument("--conditioning-rcond", type=float, default=None, help="Relative eigenvalue cutoff for C_q pseudoinverse diagnostics")
    p.add_argument("--run-design-oracle", action="store_true", help="Run the Stage C.3 finite-resource sensor-design oracle after startup checks")
    p.add_argument("--oracle-only", action="store_true", help="Skip the frozen-design resource sweep and run only the finite-resource design oracle")
    p.add_argument("--oracle-angle-n", type=int, default=None, help="Angular nodes on [0,pi) for the permutation-reduced dense oracle grid")
    p.add_argument("--oracle-trials", type=int, default=None, help="Common-random-number trials for finite-law oracle screening")
    p.add_argument("--oracle-action-trials", type=int, default=None, help="Trials used for finite MFSI action among law-feasible oracle designs")
    p.add_argument("--oracle-validation-trials", type=int, default=None, help="Independent trials used only after oracle selection to validate the selected and frozen designs")
    p.add_argument("--oracle-n", type=int, default=None, help="Finite population size used by the design oracle")
    p.add_argument("--oracle-k", type=int, default=None, help="Acquisition count used by the design oracle")
    p.add_argument("--oracle-noise", dest="oracle_noise_std", type=float, default=None, help="Detector-noise std used by the design oracle")
    p.add_argument("--oracle-tau-l", type=float, default=None, help="Population-law relative tolerance for the oracle")
    p.add_argument("--oracle-tau-r", type=float, default=None, help="Finite-law relative tolerance for the oracle")
    p.add_argument("--output", type=str, default="stage_c3_mfsi_finite_resource_design_oracle_results.json")
    return p


def apply_overrides(cfg: CConfig, args: argparse.Namespace) -> CConfig:
    kw: Dict[str, Any] = {}
    for name in (
        "trials", "action_trials", "seed", "grid_n", "time_n", "alpha_eval_mode",
        "feasibility_margin", "conditioning_rcond",
        "oracle_angle_n", "oracle_trials", "oracle_action_trials", "oracle_validation_trials", "oracle_n", "oracle_k",
        "oracle_noise_std", "oracle_tau_l", "oracle_tau_r",
    ):
        val = getattr(args, name)
        if val is not None:
            kw[name] = val
    if args.n_list is not None:
        kw["n_list"] = parse_int_tuple(args.n_list)
    if args.k_list is not None:
        kw["k_list"] = parse_int_tuple(args.k_list)
    if args.noise_list is not None:
        kw["noise_list"] = parse_float_tuple(args.noise_list)
    return dataclasses.replace(cfg, **kw)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = apply_overrides(preset_cconfig(args.preset), args)

    backend_path = Path(args.backend) if args.backend else autodetect_backend()
    if backend_path is None:
        raise FileNotFoundError(
            "Could not auto-detect Stage B.2 backend. Pass --backend /path/to/stage_b2_transport_conditioned_design.py"
        )
    backend = load_backend(backend_path)

    base = backend.preset_config("reference" if cfg.preset != "quick" else "quick")
    bcfg = dataclasses.replace(base, grid_n=cfg.grid_n, time_n=cfg.time_n)
    model = backend.StageB(bcfg)
    infer = StageCInference(model, cfg)

    designs = {
        "lift": degrees_to_eta(cfg.lift_design_deg),
        "tangent": degrees_to_eta(cfg.tangent_design_deg),
        "full": degrees_to_eta(cfg.full_design_deg),
    }

    if cfg.action_trials > cfg.trials:
        raise ValueError("action_trials cannot exceed trials")
    if cfg.feasibility_margin < 0.0:
        raise ValueError("feasibility_margin must be nonnegative")
    if cfg.conditioning_rcond <= 0.0:
        raise ValueError("conditioning_rcond must be positive")
    if cfg.oracle_tau_l < 0.0 or cfg.oracle_tau_r < 0.0:
        raise ValueError("oracle tolerances must be nonnegative")
    if cfg.oracle_n < 2:
        raise ValueError("oracle_n must be at least 2")
    if min(cfg.n_list) < 2:
        raise ValueError("Every finite-population N must be at least 2")
    all_k = tuple(sorted(set(cfg.k_list + ((cfg.oracle_k,) if (args.run_design_oracle or args.oracle_only) else tuple()))))
    acq_sets = nested_acquisition_sets(model.cfg.time_n, all_k)
    master_acq_idx = acq_sets[max(acq_sets)]
    common_test_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in set(master_acq_idx.tolist()) and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    if common_test_idx.size == 0:
        raise ValueError("Largest K leaves no common held-out test times; reduce max K or increase time_n")

    print("=" * 78)
    print("Stage C.3 — finite-measurement MFSI + matched action + finite-resource design oracle")
    print("=" * 78)
    print(f"Backend           : {Path(backend_path).resolve()}")
    print(f"Preset            : {cfg.preset}")
    print(f"Grid / time       : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"Trials            : {cfg.trials} (full-action subset: {cfg.action_trials})")
    print(f"N budgets         : {cfg.n_list} (nested sample prefixes)")
    print(f"K budgets         : {cfg.k_list} (nested acquisition sets)")
    print(f"Detector noise    : {cfg.noise_list} (shared z scaled by sigma)")
    print("Inverse            : hard-fiber MFSI on the GLS-reconstructed moment path")
    print(f"Feasibility margin : {cfg.feasibility_margin:g}")
    print(f"Conditioning rcond : {cfg.conditioning_rcond:g}")
    print("Frozen designs:")
    for name, eta in designs.items():
        print(f"  {name:8s}: ({np.rad2deg(eta[0]):.2f}°, {np.rad2deg(eta[1]):.2f}°)")
    print("Acquisition sets:")
    for k in sorted(acq_sets):
        print(f"  K={k:2d}: indices={acq_sets[k].tolist()}")
    print(f"Common held-out test indices: {common_test_idx.tolist()}")

    print("\nCompiling / checking population Stage B metrics at the frozen designs...", flush=True)
    t0 = time.time()
    pop = population_design_check(model, designs)
    for name in ("lift", "tangent", "full"):
        p = pop[name]
        print(
            f"  {name:8s}: Lift={p['lift_mmd2']:.6e} | Full={p['full_action']:.6e} | "
            f"Tangent={p['tangent_action']:.6e}"
        )
    print(f"Population check completed in {time.time()-t0:.1f}s")
    pop_full_vs_lift_reduction = 1.0 - pop["full"]["full_action"] / pop["lift"]["full_action"]
    pop_full_vs_tangent_reduction = 1.0 - pop["full"]["full_action"] / pop["tangent"]["full_action"]
    print(
        f"  Population Full-vs-Lift action reduction   = {100*pop_full_vs_lift_reduction:.2f}%"
    )
    print(
        f"  Population Full-vs-Tangent action reduction= {100*pop_full_vs_tangent_reduction:.2f}%"
    )

    print("\nRunning exact-measurement quadratic temporal-model bias control...", flush=True)
    temporal_control = quadratic_temporal_bias_control(model, infer, designs, acq_sets, cfg)
    print(
        f"  worst quadratic exact-measurement RMSE={temporal_control['worst_rmse']:.3e} | "
        f"max error={temporal_control['worst_max_error']:.3e}"
    )

    print("Running arbitrary-target hard-MFSI consistency control...", flush=True)
    hard_control = arbitrary_target_hard_solver_control(model, infer, designs, cfg)
    print(
        f"  worst custom-vs-backend mass L1={hard_control['worst_mass_l1']:.3e} | "
        f"MMD^2 difference={hard_control['worst_lift_abs_diff']:.3e}"
    )

    print("\nBuilding common-random-number trial bank...", flush=True)
    need_oracle = bool(args.run_design_oracle or args.oracle_only)
    bank_trials = max(cfg.trials, (cfg.oracle_trials + cfg.oracle_validation_trials) if need_oracle else 0)
    bank_nmax = max(max(cfg.n_list), cfg.oracle_n if need_oracle else 0)
    bank_cfg = dataclasses.replace(
        cfg, trials=bank_trials, n_list=tuple(sorted(set(cfg.n_list + ((cfg.oracle_n,) if need_oracle else tuple()))))
    )
    trial_bank = build_trial_bank(model, bank_cfg, master_acq_idx)
    print(f"  built {len(trial_bank)} paired trials with N_max={bank_nmax}")

    conditions = []
    if not args.oracle_only:
        for noise in cfg.noise_list:
            for k in cfg.k_list:
                idx = acq_sets[int(k)]
                for n in cfg.n_list:
                    cond = run_condition(
                        model=model,
                        infer=infer,
                        designs=designs,
                        cfg=cfg,
                        trial_bank=trial_bank[:cfg.trials],
                        acq_idx=idx,
                        test_idx=common_test_idx,
                        n=int(n),
                        k=int(k),
                        obs_noise_std=float(noise),
                    )
                    conditions.append(cond)
                    print_condition_summary(cond)

    oracle_result = None
    if need_oracle:
        if cfg.oracle_k not in acq_sets:
            raise RuntimeError("internal error: oracle K acquisition set was not built")
        oracle_result = finite_resource_design_oracle(
            model=model, infer=infer, cfg=cfg, frozen_designs=designs,
            trial_bank=trial_bank, acq_idx=acq_sets[cfg.oracle_k], test_idx=common_test_idx,
        )

    result = {
        "stage": "C.3 finite-resource-aware hard-fiber MFSI design oracle with matched Stage-B action accounting",
        "created_unix_time": time.time(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend_path": str(Path(backend_path).resolve()),
        "stage_c_config": jsonify(cfg),
        "stage_b_config": jsonify(bcfg),
        "designs_rad": {k: v.tolist() for k, v in designs.items()},
        "designs_deg": {k: np.rad2deg(v).tolist() for k, v in designs.items()},
        "acquisition_sets": {str(k): v.tolist() for k, v in acq_sets.items()},
        "common_test_indices": common_test_idx.tolist(),
        "common_test_times": np.asarray(model.times, dtype=np.float64)[common_test_idx].tolist(),
        "population_design_check": pop,
        "population_action_reductions": {
            "full_vs_lift_fraction": float(pop_full_vs_lift_reduction),
            "full_vs_tangent_fraction": float(pop_full_vs_tangent_reduction),
        },
        "quadratic_temporal_bias_control": temporal_control,
        "arbitrary_target_hard_solver_control": hard_control,
        "conditions": conditions,
        "finite_resource_design_oracle": oracle_result,
        "interpretation_notes": [
            "The frozen-design sweep remains a consequence study. When --run-design-oracle/--oracle-only is used, Stage C.3 additionally re-optimizes the two sensor angles on a dense finite-resource oracle grid.",
            "Finite samples are drawn from the same discrete external law used by the Stage B inverse.",
            "All N/K/noise conditions use common random numbers within each trial; N uses nested sample prefixes and K uses nested acquisition sets.",
            "Primary held-out MMD is evaluated on one fixed common test-time set: the complement of the largest acquisition set, so every K is scored on identical unobserved times.",
            "The temporal estimator has fixed capacity for every K: an endpoint-anchored quadratic bridge with one beta vector in R^2.",
            "Measurement covariance is used once in the GLS estimator. The reconstructed path is then passed to ordinary hard-fiber MFSI; there is no second pointwise soft likelihood.",
            "Because finite-noise moments can leave the MFSI moment set, beta is projected when necessary onto the convex constraints induced by conv{Phi_eta(x)} over all evaluation times. Projection frequency and norm are reported.",
            "The local statistical-conditioning diagnostic is 0.5 tr(C_q^dagger V_c(t)) with V_c(t)=t^2(1-t)^2 Cov(beta_hat), integrated over time.",
            "Delta_N = finite-data held-out MMD^2 minus the same design's population/hard-fiber held-out MMD^2 is the direct robustness degradation diagnostic.",
            "finite_full_action is the actual weighted-Poisson MFSI action after finite measurement noise has changed the reconstructed moment path; it is not population action plus an ad-hoc penalty.",
            "Stage C action integration now uses the same complete Stage-B trapezoidal time quadrature (including half-weighted endpoints) as the population backend; no interior-time renormalization is used for action.",
            "oracle_full_action_trial is the exact-population MFSI action evaluated on the same alpha trial and same time quadrature as finite_full_action, so A_N/A_inf is a matched finite-measurement inflation ratio.",
            "The primary finite action-reduction fraction is 1-E[A_full]/E[A_baseline], matching the Stage B population estimand; a paired mean of trial-wise reductions is also stored as a secondary diagnostic. Matched exact fractions separate alpha-sampling effects from finite-measurement effects.",
            "The reported GLS beta covariance is the local unconstrained covariance. If the feasibility projection is active, treat it as a pre-projection local uncertainty diagnostic rather than the exact covariance of the constrained estimator.",
            "Full-action statistics use only action_trials Monte Carlo scenarios because each requires weighted-Poisson solves at every time node.",
        ],
    }

    output = Path(args.output)
    output.write_text(json.dumps(jsonify(result), indent=2, allow_nan=True) + "\n")
    print("\n" + "=" * 78)
    print(f"Wrote Stage C results to: {output.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
