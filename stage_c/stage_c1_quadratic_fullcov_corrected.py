"""
Stage C.1: finite/noisy population measurements with an analytic stochastic interpolant.

This script deliberately adds ONE new source of difficulty to the validated Stage B setup:
finite/noisy population moments. It reuses the Stage B.2 numerical backend for the
scientific law, physical sensors, analytic stochastic-interpolant reference, grid MMD,
and weighted-Poisson solve.

Scientific question
-------------------
Freeze the Stage B.3 designs (Lift, Tangent-TC, Full-TC) and ask two distinct questions:

  1. Practical performance: which frozen experiment has the smallest finite-data
     held-out law error?
  2. Finite-data robustness: how much does each design degrade relative to its own
     population/hard-fiber baseline?

The second question is reported explicitly through

    Delta_N(eta) = L_N(eta) - L_infinity(eta).

Finite-data observation model
-----------------------------
At acquisition times t_k, draw N microscopic states from the SAME discretized external
population used by Stage B and form empirical sensor means. Optional additive detector
noise is applied to the reported population mean. Endpoints remain exact anchors.

The two sensor means are estimated from the same microscopic particles, so the mean
uncertainty is a full 2x2 covariance matrix

    V_eta(t) = Cov_{P_t}[Phi_eta] / N + sigma_obs^2 I.

For this controlled Stage C benchmark V_eta(t) and dV_eta/dt are evaluated from the
known simulation law, not estimated from the finite sample. This isolates uncertainty
in the measured mean from uncertainty in the uncertainty estimate.

Temporal reconstruction
-----------------------
The continuous scientific mixture has quadratic time weights. To avoid confounding the
acquisition budget K with temporal-model capacity, every K uses the SAME fixed-capacity
endpoint-anchored quadratic bridge

    c_hat(t) = (1-t)c0 + t c1 + beta t(1-t),

where beta in R^2 is estimated by generalized least squares using the full 2x2 V_k.
The derivative is analytic:

    c_hat_dot(t) = c1 - c0 + beta (1-2t).

Because Stage B normalizes the external law on a finite grid, its numerical measurement
trajectory is only approximately quadratic. The script therefore reports an automatic
exact-measurement temporal-model bias control instead of silently assuming zero bias.

Uncertainty-aware inverse
-------------------------
Rather than forcing a noisy empirical mean as an exact constraint, use the soft
information projection

    min_q KL(q || q_ref)
          + 1/2 (E_q[Phi]-c_hat)^T V^{-1} (E_q[Phi]-c_hat).

Its optimizer is still an exponential tilt q_lambda and lambda solves

    E_{q_lambda}[Phi] + V lambda = c_hat.

The calibration Jacobian is Cov_q(Phi) + V. Differentiating the calibration equation
through time gives

    (Cov_q(Phi)+V) lambda_dot
      = c_hat_dot - (d/dt E_q[Phi])_{lambda fixed} - V_dot lambda.

As N -> infinity and detector noise -> 0, V -> 0 and the Stage B hard information
projection is recovered. A direct V->0 hard-fiber recovery control is run at startup.

Paired resource sweeps
----------------------
All N, K, and detector-noise conditions share common random numbers within a trial:
  * the same latent scientific scenario alpha,
  * one master N_max sample at every acquisition time,
  * nested prefixes for smaller N,
  * nested acquisition-time sets for smaller K,
  * one standard-normal detector perturbation scaled by each requested noise level.

This makes learning curves and pairwise comparisons substantially cleaner.

Outputs
-------
For each (N, K, detector-noise level), the script reports paired Monte Carlo summaries of
  * held-out-time law MMD^2,
  * Delta_N relative to the corresponding population/hard-fiber reconstruction,
  * moment-curve and calibration diagnostics,
  * optional noisy full action and action inflation on a smaller subset of trials.

The script produces one JSON result file. It remains a Stage C.1 consequence study:
it does not re-optimize sensor angles under measurement noise.

Examples
--------
Quick smoke test:
    python stage_c_finite_measurement_robustness_corrected.py \
        --backend stage_b2_transport_conditioned_design.py --preset quick

Main reference run:
    python stage_c_finite_measurement_robustness_corrected.py \
        --backend stage_b2_transport_conditioned_design.py --preset reference \
        --trials 40 --action-trials 8 --output stage_c1_reference_corrected.json
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

    # Numerical safeguards for the uncertainty-aware calibration.
    variance_floor: float = 1.0e-10
    lambda_clip: float = 80.0
    newton_step_cap: float = 5.0
    quadratic_ridge_rel: float = 1.0e-12

    # Stage C needs held-out time nodes between acquisitions.
    grid_n: int = 19
    time_n: int = 13
    alpha_eval_mode: str = "random"  # random or quadrature

    # Startup diagnostics.
    control_alpha_n: int = 3
    population_limit_variance: float = 1.0e-10

    # Frozen Stage B.3 designs, in degrees.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)


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
) -> Dict[str, np.ndarray]:
    """Fit c(t)=(1-t)c0+t*c1+beta*t(1-t) with full-covariance GLS."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    V_obs = np.asarray(V_obs, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    t_eval = np.asarray(t_eval, dtype=np.float64)

    H = np.zeros((2, 2), dtype=np.float64)
    g = np.zeros(2, dtype=np.float64)
    used = 0

    for t, y, V in zip(t_obs, y_obs, V_obs):
        z = float(t * (1.0 - t))
        if abs(z) < 1e-14:
            continue
        bridge = (1.0 - t) * c0 + t * c1
        resid = y - bridge
        Vreg = 0.5 * (V + V.T) + variance_floor * np.eye(2)
        Vinv = np.linalg.inv(Vreg)
        H += (z * z) * Vinv
        g += z * (Vinv @ resid)
        used += 1

    if used == 0:
        raise ValueError("Quadratic bridge fit needs at least one interior acquisition time")

    scale = max(float(np.trace(H)) / 2.0, 1.0)
    Hreg = H + (ridge_rel * scale) * np.eye(2)
    beta = np.linalg.solve(Hreg, g)

    z_eval = t_eval * (1.0 - t_eval)
    c = (1.0 - t_eval[:, None]) * c0[None, :] + t_eval[:, None] * c1[None, :] + z_eval[:, None] * beta[None, :]
    cdot = (c1 - c0)[None, :] + (1.0 - 2.0 * t_eval[:, None]) * beta[None, :]

    return {"beta": beta, "c": c, "cdot": cdot}


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
# Uncertainty-aware exponential tilt + exact full-law action
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

    def _population_sensor_cov_dot(self, t: Array, alpha: Array, eta: Array) -> Array:
        return jax.jacfwd(self._population_sensor_cov, argnums=0)(t, alpha, eta)

    def _exact_measurement_dot(self, t: Array, alpha: Array, eta: Array) -> Array:
        return jax.jacfwd(self.model.measurement_grid, argnums=0)(t, alpha, eta)

    def _soft_lambda_raw(self, t: Array, eta: Array, c_hat: Array, V: Array) -> Array:
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
            return (
                jsp.special.logsumexp(log_base + lam @ flat_phi)
                - lam @ c_hat
                + 0.5 * (lam @ V @ lam)
            )

        def body(_, lam):
            _, moment, C = moment_cov(lam)
            F = moment + V @ lam - c_hat
            H = C + V + model.cfg.newton_ridge * self.eye2
            step = jnp.linalg.solve(H, F)
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(1.0, self.cfg.newton_step_cap / jnp.maximum(step_norm, 1e-12))
            scales = model.cfg.newton_damping * (0.5 ** jnp.arange(9, dtype=jnp.float64))
            cands = lam[None, :] - scales[:, None] * step[None, :]
            vals = jax.vmap(dual)(cands)
            best = cands[jnp.argmin(vals)]
            return jnp.clip(best, -self.cfg.lambda_clip, self.cfg.lambda_clip)

        return jax.lax.fori_loop(0, model.cfg.newton_steps, body, jnp.zeros(2, dtype=jnp.float64))

    def _soft_state(
        self,
        t: Array,
        alpha: Array,
        eta: Array,
        c_hat: Array,
        c_dot: Array,
        V: Array,
        V_dot: Array,
    ):
        model = self.model
        phi, grad_phi = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        lam = self._soft_lambda_raw(t, eta, c_hat, V)
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
        H = C + V + model.cfg.newton_ridge * self.eye2
        rhs = c_dot - dm_ref_dt - V_dot @ lam
        lam_dot = jnp.linalg.solve(H, rhs)

        B = model.B_matrix(t)
        u = model.xy_flat @ B.T
        u = u.reshape((model.cfg.grid_n, model.cfg.grid_n, 2))
        jphi_u = jnp.einsum("myxc,yxc->myx", grad_phi, u)

        # General exponential-tilt forcing: center by the ACTUAL q moment.
        term_time = jnp.einsum("m,myx->yx", lam_dot, phi - moment[:, None, None])
        adv_scalar = jnp.einsum("m,myx->yx", lam, jphi_u)
        adv_centered = adv_scalar - jnp.sum(qmass * adv_scalar)
        h_raw = term_time + adv_centered
        mean_h_raw = jnp.sum(qmass * h_raw)
        h = h_raw - mean_h_raw

        soft_resid = jnp.linalg.norm(moment + V @ lam - c_hat)
        shrinkage = jnp.linalg.norm(moment - c_hat)
        min_soft_hessian_eig = jnp.min(jnp.linalg.eigvalsh(C + V))
        true_c = model.measurement_grid(t, alpha, eta)
        moment_error = jnp.linalg.norm(moment - true_c)

        return {
            "q": q,
            "qmass": qmass,
            "h": h,
            "lam": lam,
            "moment": moment,
            "soft_resid": soft_resid,
            "shrinkage": shrinkage,
            "min_soft_hessian_eig": min_soft_hessian_eig,
            "moment_error": moment_error,
            "mean_h_raw": mean_h_raw,
        }

    def _law_metrics(self, t, alpha, eta, c_hat, c_dot, V, V_dot):
        st = self._soft_state(t, alpha, eta, c_hat, c_dot, V, V_dot)
        _, p_mass = self.model.external_q_mass(t, alpha)
        lift = self.model.gaussian_mmd2_mass(st["qmass"], p_mass)
        return jnp.array([
            lift,
            st["moment_error"],
            st["soft_resid"],
            st["shrinkage"],
            st["min_soft_hessian_eig"],
            jnp.linalg.norm(st["lam"]),
            jnp.abs(st["mean_h_raw"]),
        ])

    def _full_metrics(self, t, alpha, eta, c_hat, c_dot, V, V_dot):
        st = self._soft_state(t, alpha, eta, c_hat, c_dot, V, V_dot)
        full, _, poisson_rel, _, _ = self.model.poisson_solve(st["q"], st["h"])
        _, p_mass = self.model.external_q_mass(t, alpha)
        lift = self.model.gaussian_mmd2_mass(st["qmass"], p_mass)
        return jnp.array([lift, full, poisson_rel])

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
        self.population_cov_dot_jit = jax.jit(self._population_sensor_cov_dot)
        self.exact_measurement_dot_jit = jax.jit(self._exact_measurement_dot)
        self.soft_state_jit = jax.jit(self._soft_state)
        self.law_metrics_jit = jax.jit(self._law_metrics)
        self.full_metrics_jit = jax.jit(self._full_metrics)
        self.hard_lift_jit = jax.jit(self._hard_lift)
        self.hard_full_jit = jax.jit(self._hard_full)


# -----------------------------------------------------------------------------
# Finite-population acquisition and exact covariance curve
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


def measurement_covariance_curve(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    alpha: float,
    n: int,
    obs_noise_std: float,
    variance_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    times = np.asarray(model.times, dtype=np.float64)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    Vs = []
    Vdots = []
    for t in times:
        t_j = jnp.asarray(t, dtype=jnp.float64)
        Sigma = np.asarray(infer.population_cov_jit(t_j, alpha_j, eta_j), dtype=np.float64)
        Sigma_dot = np.asarray(infer.population_cov_dot_jit(t_j, alpha_j, eta_j), dtype=np.float64)
        V = Sigma / float(n) + (obs_noise_std ** 2) * np.eye(2)
        V = 0.5 * (V + V.T) + variance_floor * np.eye(2)
        Vdot = Sigma_dot / float(n)
        Vdot = 0.5 * (Vdot + Vdot.T)
        Vs.append(V)
        Vdots.append(Vdot)
    return np.asarray(Vs), np.asarray(Vdots)


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
    )
    V_curve, Vdot_curve = measurement_covariance_curve(
        model=model,
        infer=infer,
        eta=eta,
        alpha=shared.alpha,
        n=n,
        obs_noise_std=obs_noise_std,
        variance_floor=cfg.variance_floor,
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
            jnp.asarray(V_curve[i]),
            jnp.asarray(Vdot_curve[i]),
        )
        noisy_rows.append(np.asarray(infer.law_metrics_jit(*args), dtype=np.float64))
        oracle_lifts.append(float(infer.hard_lift_jit(jnp.asarray(t), alpha_j, eta_j)))
        if compute_action:
            full_rows.append(np.asarray(infer.full_metrics_jit(*args), dtype=np.float64))
            oracle_full_rows.append(np.asarray(infer.hard_full_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64))

    noisy = np.asarray(noisy_rows, dtype=np.float64)
    oracle_lifts = np.asarray(oracle_lifts, dtype=np.float64)
    exact_curve = np.asarray(exact_curve, dtype=np.float64)

    result = {
        "heldout_mmd2": trap_average(noisy[:, 0], tw, heldout),
        "all_interior_mmd2": trap_average(noisy[:, 0], tw, interior),
        "oracle_heldout_mmd2": trap_average(oracle_lifts, tw, heldout),
        "delta_vs_population_heldout_mmd2": trap_average(noisy[:, 0] - oracle_lifts, tw, heldout),
        "heldout_moment_error": trap_average(noisy[:, 1], tw, heldout),
        "max_soft_calibration_resid": float(np.max(noisy[:, 2])),
        "mean_shrinkage_to_measurement": trap_average(noisy[:, 3], tw, interior),
        "min_soft_hessian_eig": float(np.min(noisy[:, 4])),
        "max_lambda_norm": float(np.max(noisy[:, 5])),
        "max_mean_h_raw": float(np.max(noisy[:, 6])),
        "acquisition_mean_rmse": float(np.sqrt(np.mean((y_acq - exact_acq) ** 2))),
        "quadratic_moment_rmse": float(np.sqrt(np.mean(np.sum((curve["c"][interior] - exact_curve[interior]) ** 2, axis=1)))),
        "quadratic_moment_max_error": float(np.max(np.linalg.norm(curve["c"][interior] - exact_curve[interior], axis=1))),
        "beta_norm": float(np.linalg.norm(curve["beta"])),
    }

    if compute_action:
        full_rows = np.asarray(full_rows, dtype=np.float64)
        oracle_full_rows = np.asarray(oracle_full_rows, dtype=np.float64)
        noisy_action = trap_average(full_rows[:, 1], tw, interior)
        oracle_action = trap_average(oracle_full_rows[:, 1], tw, interior)
        result.update({
            "noisy_full_action": noisy_action,
            "oracle_full_action_trial": oracle_action,
            "action_inflation_ratio": noisy_action / max(oracle_action, 1e-14),
            "max_poisson_rel_resid": float(np.max(full_rows[:, 2])),
        })
    else:
        result.update({
            "noisy_full_action": float("nan"),
            "oracle_full_action_trial": float("nan"),
            "action_inflation_ratio": float("nan"),
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


def hard_fiber_recovery_control(
    model,
    infer: StageCInference,
    designs: Dict[str, np.ndarray],
    cfg: CConfig,
) -> Dict[str, Any]:
    """Feed exact Stage B c(t), exact c_dot(t), tiny V and compare soft to hard q."""
    times_all = np.asarray(model.times, dtype=np.float64)
    control_idx = np.unique(np.rint(np.linspace(0, len(times_all) - 1, 5)).astype(int))
    times = times_all[control_idx]
    alpha = 0.5 * (model.cfg.alpha_min + model.cfg.alpha_max)
    alpha_j = jnp.asarray(alpha, dtype=jnp.float64)
    eps = float(cfg.population_limit_variance)
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
            V = eps * jnp.eye(2, dtype=jnp.float64)
            Vdot = jnp.zeros((2, 2), dtype=jnp.float64)
            st = infer.soft_state_jit(t_j, alpha_j, eta_j, c, cdot, V, Vdot)

            phi, _ = model.sensor_fields(eta_j)
            _, qref_mass = model.reference_q_mass(t_j)
            lam_hard = model.solve_lambda(t_j, alpha_j, eta_j)
            qmass_hard = model.tilted_mass_from_fields(lam_hard, qref_mass, phi)
            qmass_soft = np.asarray(st["qmass"], dtype=np.float64)
            qmass_hard_np = np.asarray(qmass_hard, dtype=np.float64)
            mass_l1 = float(np.sum(np.abs(qmass_soft - qmass_hard_np)))

            _, p_mass = model.external_q_mass(t_j, alpha_j)
            lift_soft = float(model.gaussian_mmd2_mass(st["qmass"], p_mass))
            lift_hard = float(model.gaussian_mmd2_mass(qmass_hard, p_mass))
            max_mass_l1 = max(max_mass_l1, mass_l1)
            max_lift_diff = max(max_lift_diff, abs(lift_soft - lift_hard))
            max_resid = max(max_resid, float(st["soft_resid"]))

        rows.append({
            "design": dname,
            "max_mass_l1": max_mass_l1,
            "max_lift_abs_diff": max_lift_diff,
            "max_soft_calibration_resid": max_resid,
        })

    return {
        "alpha": float(alpha),
        "time_indices": control_idx.tolist(),
        "variance_epsilon": eps,
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
    "max_soft_calibration_resid",
    "mean_shrinkage_to_measurement",
    "min_soft_hessian_eig",
    "max_lambda_norm",
    "max_mean_h_raw",
    "noisy_full_action",
    "oracle_full_action_trial",
    "action_inflation_ratio",
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
    }


def print_condition_summary(cond: Dict[str, Any]) -> None:
    print("\n" + "-" * 78)
    print(
        f"Stage C condition: N={cond['N']}, K={cond['K']}, "
        f"detector noise std={cond['obs_noise_std']:.4f}"
    )
    print("-" * 78)
    for name, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        s = cond["design_summary"][name]
        h = s["heldout_mmd2"]
        ex = s["delta_vs_population_heldout_mmd2"]
        ai = s["action_inflation_ratio"]
        qrmse = s["quadratic_moment_rmse"]
        print(
            f"{label:10s} held-out MMD^2={h['mean']:.6e} ± {h['se']:.2e} | "
            f"Delta_N={ex['mean']:+.3e} ± {ex['se']:.2e} | "
            f"quad RMSE={qrmse['mean']:.2e} | "
            f"action inflation={ai['mean']:.4f} ± {ai['se']:.2e} (n={ai['n']})"
        )

    for key, label in (
        ("full_vs_lift_heldout_mmd2", "Full - Lift law"),
        ("full_vs_tangent_heldout_mmd2", "Full - Tangent law"),
        ("full_vs_lift_delta_vs_population_mmd2", "Full - Lift degr."),
        ("full_vs_tangent_delta_vs_population_mmd2", "Full - Tangent degr."),
    ):
        c = cond["paired_comparisons"][key]
        print(
            f"{label:20s}: paired Δ={c['mean_difference_a_minus_b']:+.3e} "
            f"± {c['se_difference']:.2e}; Full better in {100*c['a_better_fraction']:.1f}%"
        )


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
    p.add_argument("--output", type=str, default="stage_c1_finite_measurement_results_corrected.json")
    return p


def apply_overrides(cfg: CConfig, args: argparse.Namespace) -> CConfig:
    kw: Dict[str, Any] = {}
    for name in ("trials", "action_trials", "seed", "grid_n", "time_n", "alpha_eval_mode"):
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
    if min(cfg.n_list) < 2:
        raise ValueError("Every finite-population N must be at least 2")
    acq_sets = nested_acquisition_sets(model.cfg.time_n, cfg.k_list)
    master_acq_idx = acq_sets[max(acq_sets)]
    common_test_idx = np.asarray([
        i for i in range(model.cfg.time_n)
        if i not in set(master_acq_idx.tolist()) and i not in (0, model.cfg.time_n - 1)
    ], dtype=int)
    if common_test_idx.size == 0:
        raise ValueError("Largest K leaves no common held-out test times; reduce max K or increase time_n")

    print("=" * 78)
    print("Stage C.1 — corrected finite/noisy measurements + analytic SI")
    print("=" * 78)
    print(f"Backend           : {Path(backend_path).resolve()}")
    print(f"Preset            : {cfg.preset}")
    print(f"Grid / time       : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"Trials            : {cfg.trials} (full-action subset: {cfg.action_trials})")
    print(f"N budgets         : {cfg.n_list} (nested sample prefixes)")
    print(f"K budgets         : {cfg.k_list} (nested acquisition sets)")
    print(f"Detector noise    : {cfg.noise_list} (shared z scaled by sigma)")
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

    print("\nRunning exact-measurement quadratic temporal-model bias control...", flush=True)
    temporal_control = quadratic_temporal_bias_control(model, infer, designs, acq_sets, cfg)
    print(
        f"  worst quadratic exact-measurement RMSE={temporal_control['worst_rmse']:.3e} | "
        f"max error={temporal_control['worst_max_error']:.3e}"
    )

    print("Running V->0 hard-fiber recovery control...", flush=True)
    hard_control = hard_fiber_recovery_control(model, infer, designs, cfg)
    print(
        f"  worst soft-vs-hard mass L1={hard_control['worst_mass_l1']:.3e} | "
        f"MMD^2 difference={hard_control['worst_lift_abs_diff']:.3e}"
    )

    print("\nBuilding common-random-number trial bank...", flush=True)
    trial_bank = build_trial_bank(model, cfg, master_acq_idx)
    print(f"  built {len(trial_bank)} paired trials with N_max={max(cfg.n_list)}")

    conditions = []
    for noise in cfg.noise_list:
        for k in cfg.k_list:
            idx = acq_sets[int(k)]
            for n in cfg.n_list:
                cond = run_condition(
                    model=model,
                    infer=infer,
                    designs=designs,
                    cfg=cfg,
                    trial_bank=trial_bank,
                    acq_idx=idx,
                    test_idx=common_test_idx,
                    n=int(n),
                    k=int(k),
                    obs_noise_std=float(noise),
                )
                conditions.append(cond)
                print_condition_summary(cond)

    result = {
        "stage": "C.1 corrected finite/noisy measurements with analytic stochastic interpolant",
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
        "quadratic_temporal_bias_control": temporal_control,
        "hard_fiber_recovery_control": hard_control,
        "conditions": conditions,
        "interpretation_notes": [
            "Frozen-design C.1 test: no sensor angles are re-optimized under noise.",
            "Finite samples are drawn from the same discrete external law used by the Stage B inverse.",
            "All N/K/noise conditions use common random numbers within each trial; N uses nested sample prefixes and K uses nested acquisition sets.",
            "Primary held-out MMD is evaluated on one fixed common test-time set: the complement of the largest acquisition set, so every K is scored on identical unobserved times.",
            "The temporal estimator has fixed capacity for every K: an endpoint-anchored quadratic bridge with one beta vector in R^2.",
            "Because the Stage B finite-grid normalization makes the numerical moment path only approximately quadratic, an exact-measurement temporal-model bias control is reported explicitly.",
            "The uncertainty-aware soft fiber uses the full 2x2 mean covariance V=Cov_P(Phi)/N+sigma_obs^2 I and solves E_q[Phi]+V lambda=c_hat.",
            "V and V_dot are evaluated from the known simulation law in C.1 so uncertainty in the mean is isolated from covariance-estimation noise.",
            "Delta_N = finite-data held-out MMD^2 minus the same design's population/hard-fiber held-out MMD^2 is a primary robustness quantity.",
            "A direct V->0 control checks that the soft inverse recovers the Stage B hard-fiber law when exact moments are supplied.",
            "Full-action statistics use only action_trials Monte Carlo replicates because each replicate requires weighted-Poisson solves at every time node.",
        ],
    }

    output = Path(args.output)
    output.write_text(json.dumps(jsonify(result), indent=2, allow_nan=True) + "\n")
    print("\n" + "=" * 78)
    print(f"Wrote Stage C results to: {output.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()