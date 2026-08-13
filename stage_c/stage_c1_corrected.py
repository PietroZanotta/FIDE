"""
Stage C.1: finite/noisy population measurements with an analytic stochastic interpolant.

This script deliberately adds ONE new source of difficulty to the validated Stage B setup:
finite/noisy population moments.  It reuses the Stage B.2 numerical backend for the
scientific law, physical sensors, analytic stochastic-interpolant reference, grid MMD,
and weighted-Poisson solve.

Primary scientific question
---------------------------
Freeze the Stage B.3 designs (Lift, Tangent-TC, Full-TC) and ask whether the design
selected for lower exact full-law transport burden is more robust when moments are
estimated from finite populations.

Finite-data inverse
-------------------
At acquisition times t_k, draw N microscopic states from the *same discretized external
population used by Stage B* and form empirical sensor means.  Optional additive detector
noise is applied to the reported population mean.  A smooth moment curve c_hat(t) is fit
with exact endpoint anchors.

Instead of forcing a noisy empirical mean as a hard moment equality, use the uncertainty-
aware soft information projection

    min_q KL(q || q_ref) + 1/2 (E_q[Phi]-c_hat)^T V^{-1}(E_q[Phi]-c_hat).

Its optimizer is still an exponential tilt q_lambda and lambda solves

    E_{q_lambda}[Phi] + V lambda = c_hat.

The calibration Jacobian is Cov_q(Phi) + V.  As N -> infinity and detector noise -> 0,
V -> 0 and the Stage B hard information projection is recovered.

Temporal smoothing
------------------
Raw independent noise must NOT be finite-differenced directly because the full action
uses a time derivative.  We fit a small RBF smoother to the measured moment trajectory,
with exact endpoint anchors, and differentiate the smoother analytically.  The same
smoother is used for every frozen design.

Outputs
-------
For each (N, acquisition-time budget K, detector-noise level), the script reports paired
Monte Carlo summaries of
  * held-out-time law MMD^2,
  * difference in MMD^2 relative to the corresponding population/hard-fiber reconstruction,
  * moment-curve error and calibration diagnostics,
  * optional noisy full-action and action inflation on a smaller subset of trials.

The script produces one JSON result file.  It is intentionally a Stage C.1 consequence
study: it does not yet re-optimize the sensor angles under measurement noise.

Examples
--------
Quick smoke test:
    python stage_c_finite_measurement_robustness.py \
        --backend stage_b2_transport_conditioned_design.py --preset quick

Stronger run:
    python stage_c_finite_measurement_robustness.py \
        --backend stage_b2_transport_conditioned_design.py --preset reference \
        --trials 40 --action-trials 8 --output stage_c1_reference.json
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    n_list: Tuple[int, ...] = (50, 200)
    k_list: Tuple[int, ...] = (5, 9)
    noise_list: Tuple[float, ...] = (0.0, 0.01)
    seed: int = 20260812

    # Smooth c_hat(t) with an affine + Gaussian-RBF basis.
    smooth_lengthscale: float = 0.22
    smooth_ridge: float = 2.0e-2
    variance_smooth_ridge: float = 8.0e-2

    # Numerical safeguards for the uncertainty-aware calibration.
    variance_floor: float = 1.0e-8
    lambda_clip: float = 80.0
    newton_step_cap: float = 5.0

    # Stage C uses a denser time grid than the original quick Stage B preset so
    # that there are genuinely held-out times between acquisitions.
    grid_n: int = 19
    time_n: int = 13
    alpha_eval_mode: str = "random"  # random or quadrature

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
            smooth_lengthscale=0.18,
            smooth_ridge=1.5e-2,
            variance_smooth_ridge=6.0e-2,
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
            smooth_lengthscale=0.16,
            smooth_ridge=1.0e-2,
            variance_smooth_ridge=5.0e-2,
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


def acquisition_indices(time_n: int, k: int) -> np.ndarray:
    if k < 4:
        raise ValueError("K must be at least 4 for the smooth temporal reconstruction")
    if k >= time_n:
        raise ValueError(f"K={k} must be smaller than time_n={time_n} to retain held-out times")
    idx = np.rint(np.linspace(0, time_n - 1, k)).astype(int)
    idx = np.unique(idx)
    if len(idx) != k:
        raise ValueError(f"Could not place {k} distinct acquisition times on a grid with {time_n} nodes")
    idx[0] = 0
    idx[-1] = time_n - 1
    return idx


def degrees_to_eta(pair_deg: Tuple[float, float]) -> np.ndarray:
    return np.deg2rad(np.asarray(pair_deg, dtype=np.float64))


# -----------------------------------------------------------------------------
# Tiny smooth temporal model: affine + Gaussian RBFs with exact endpoint anchors
# -----------------------------------------------------------------------------


class RBFSmoother:
    """Small deterministic smoother with analytic derivative.

    f(t) = a0 + a1 t + sum_j beta_j exp(-(t-c_j)^2/(2 ell^2)).

    The moment fit is weighted ridge regression subject to exact values at t=0,1.
    This is intentionally simple and auditable; it avoids finite-differencing raw noise.
    """

    def __init__(self, centers: np.ndarray, lengthscale: float):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.lengthscale = float(lengthscale)
        if self.lengthscale <= 0:
            raise ValueError("smooth_lengthscale must be positive")

    def basis(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        z = (t[:, None] - self.centers[None, :]) / self.lengthscale
        rbf = np.exp(-0.5 * z * z)
        return np.concatenate([np.ones((len(t), 1)), t[:, None], rbf], axis=1)

    def basis_derivative(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        z = (t[:, None] - self.centers[None, :]) / self.lengthscale
        rbf = np.exp(-0.5 * z * z)
        drbf = -(t[:, None] - self.centers[None, :]) / (self.lengthscale ** 2) * rbf
        return np.concatenate([np.zeros((len(t), 1)), np.ones((len(t), 1)), drbf], axis=1)

    def fit_anchored(
        self,
        t_obs: np.ndarray,
        y_obs: np.ndarray,
        var_obs: np.ndarray,
        y0: float,
        y1: float,
        ridge_rel: float,
        var_floor: float,
    ) -> np.ndarray:
        B = self.basis(t_obs)
        var = np.maximum(np.asarray(var_obs, dtype=np.float64), var_floor)
        w = 1.0 / var
        H_data = B.T @ (w[:, None] * B)
        g = B.T @ (w * np.asarray(y_obs, dtype=np.float64))

        p = B.shape[1]
        penalty = np.zeros(p, dtype=np.float64)
        penalty[2:] = 1.0
        scale = max(float(np.trace(H_data)) / max(p, 1), 1.0)
        H = H_data + (ridge_rel * scale) * np.diag(penalty) + 1e-12 * np.eye(p)

        Aeq = self.basis(np.asarray([0.0, 1.0]))
        KKT = np.block([
            [H, Aeq.T],
            [Aeq, np.zeros((2, 2), dtype=np.float64)],
        ])
        rhs = np.concatenate([g, np.asarray([y0, y1], dtype=np.float64)])
        sol = np.linalg.solve(KKT, rhs)
        return sol[:p]

    def fit_unconstrained(
        self,
        t_obs: np.ndarray,
        y_obs: np.ndarray,
        ridge_rel: float,
    ) -> np.ndarray:
        B = self.basis(t_obs)
        H_data = B.T @ B
        g = B.T @ np.asarray(y_obs, dtype=np.float64)
        p = B.shape[1]
        penalty = np.zeros(p, dtype=np.float64)
        penalty[2:] = 1.0
        scale = max(float(np.trace(H_data)) / max(p, 1), 1.0)
        H = H_data + (ridge_rel * scale) * np.diag(penalty) + 1e-12 * np.eye(p)
        return np.linalg.solve(H, g)

    def eval(self, coef: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.basis(t) @ coef, self.basis_derivative(t) @ coef


# -----------------------------------------------------------------------------
# Finite-population acquisition on the SAME discrete external law as Stage B
# -----------------------------------------------------------------------------


@dataclass
class SharedTrialData:
    alpha: float
    # Flattened grid-cell indices drawn at each acquisition grid index.
    sample_indices: Dict[int, np.ndarray]
    # Common standard-normal detector perturbations, paired across designs.
    detector_z: Dict[int, np.ndarray]


def draw_shared_trial(
    model,
    alpha: float,
    acq_idx: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> SharedTrialData:
    samples: Dict[int, np.ndarray] = {}
    noise: Dict[int, np.ndarray] = {}
    times = np.asarray(model.times, dtype=np.float64)
    for idx in acq_idx:
        t = float(times[idx])
        _, pmass = model.external_q_mass(jnp.asarray(t), jnp.asarray(alpha))
        p = np.asarray(pmass, dtype=np.float64).reshape(-1)
        p = np.maximum(p, 0.0)
        p /= np.sum(p)
        samples[int(idx)] = rng.choice(p.size, size=n, replace=True, p=p)
        noise[int(idx)] = rng.standard_normal(2)
    return SharedTrialData(alpha=float(alpha), sample_indices=samples, detector_z=noise)


def design_measurements(
    model,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    obs_noise_std: float,
    variance_floor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return t_acq, empirical means, diagonal mean variances, exact means.

    Endpoints are anchored to their exact Stage B population moments.  Interior means
    use actual finite-population draws.  The variance model still reflects the finite N
    budget at endpoints, but q_ref already satisfies the exact endpoint moments, so
    lambda=0 remains the endpoint solution.
    """
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    phi, _ = model.sensor_fields(eta_j)
    phi_flat = np.asarray(phi, dtype=np.float64).reshape(2, -1)
    times = np.asarray(model.times, dtype=np.float64)

    ys: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    exacts: List[np.ndarray] = []
    n = len(next(iter(shared.sample_indices.values())))

    for idx in acq_idx:
        t = float(times[idx])
        exact = np.asarray(
            model.measurement_grid(jnp.asarray(t), jnp.asarray(shared.alpha), eta_j),
            dtype=np.float64,
        )

        draw = shared.sample_indices[int(idx)]
        vals = phi_flat[:, draw].T  # [N,2]
        if n > 1:
            sample_var = np.var(vals, axis=0, ddof=1) / float(n)
        else:
            sample_var = np.full(2, variance_floor)
        mean_var = np.maximum(sample_var + obs_noise_std ** 2, variance_floor)

        if idx == 0 or idx == len(times) - 1:
            y = exact.copy()
        else:
            y = np.mean(vals, axis=0) + obs_noise_std * shared.detector_z[int(idx)]

        ys.append(y)
        vs.append(mean_var)
        exacts.append(exact)

    return (
        times[acq_idx],
        np.asarray(ys, dtype=np.float64),
        np.asarray(vs, dtype=np.float64),
        np.asarray(exacts, dtype=np.float64),
    )


def smooth_measurements(
    model,
    eta: np.ndarray,
    alpha: float,
    t_acq: np.ndarray,
    y_acq: np.ndarray,
    v_acq: np.ndarray,
    cfg: CConfig,
) -> Dict[str, np.ndarray]:
    times = np.asarray(model.times, dtype=np.float64)
    smoother = RBFSmoother(t_acq, cfg.smooth_lengthscale)
    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    c0 = np.asarray(model.measurement_grid(jnp.asarray(0.0), jnp.asarray(alpha), eta_j), dtype=np.float64)
    c1 = np.asarray(model.measurement_grid(jnp.asarray(1.0), jnp.asarray(alpha), eta_j), dtype=np.float64)

    c = np.zeros((len(times), 2), dtype=np.float64)
    cdot = np.zeros_like(c)
    vdiag = np.zeros_like(c)
    vdot = np.zeros_like(c)

    for m in range(2):
        coef = smoother.fit_anchored(
            t_obs=t_acq,
            y_obs=y_acq[:, m],
            var_obs=v_acq[:, m],
            y0=float(c0[m]),
            y1=float(c1[m]),
            ridge_rel=cfg.smooth_ridge,
            var_floor=cfg.variance_floor,
        )
        c[:, m], cdot[:, m] = smoother.eval(coef, times)

        # Smooth log variance to guarantee positivity.  This affects the soft-fiber
        # regularization only; it is not used as an evaluation target.
        logv = np.log(np.maximum(v_acq[:, m], cfg.variance_floor))
        vcoef = smoother.fit_unconstrained(t_acq, logv, cfg.variance_smooth_ridge)
        logv_eval, logv_dot = smoother.eval(vcoef, times)
        lo = math.log(cfg.variance_floor)
        hi = math.log(0.25)  # sensor values lie in [0,1], so this is already generous
        clipped = np.clip(logv_eval, lo, hi)
        active = (logv_eval >= lo) & (logv_eval <= hi)
        vdiag[:, m] = np.exp(clipped)
        vdot[:, m] = vdiag[:, m] * logv_dot * active.astype(np.float64)

    return {"c": c, "cdot": cdot, "vdiag": vdiag, "vdot": vdot}


# -----------------------------------------------------------------------------
# Uncertainty-aware exponential tilt + exact full-law action
# -----------------------------------------------------------------------------


class StageCInference:
    def __init__(self, model, cfg: CConfig):
        self.model = model
        self.cfg = cfg
        self.eye2 = jnp.eye(2, dtype=jnp.float64)
        self._build_jitted()

    def _soft_lambda_raw(self, t: Array, eta: Array, c_hat: Array, vdiag: Array) -> Array:
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
                + 0.5 * jnp.sum(vdiag * lam * lam)
            )

        def body(_, lam):
            _, moment, C = moment_cov(lam)
            F = moment + vdiag * lam - c_hat
            H = C + jnp.diag(vdiag) + model.cfg.newton_ridge * self.eye2
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
        vdiag: Array,
        vdiag_dot: Array,
    ):
        model = self.model
        phi, grad_phi = model.sensor_fields(eta)
        _, qref_mass = model.reference_q_mass(t)
        lam = self._soft_lambda_raw(t, eta, c_hat, vdiag)
        qmass = model.tilted_mass_from_fields(lam, qref_mass, phi)
        q = qmass / model.cell_area
        moment = jnp.sum(phi * qmass[None, ...], axis=(1, 2))
        centered = phi - moment[:, None, None]
        C = jnp.einsum("myx,nyx,yx->mn", centered, centered, qmass)

        # Fixed-lambda reference contribution to d/dt E_q[Phi].
        def fixed_lambda_moment(tt):
            _, qr_mass = model.reference_q_mass(tt)
            mass = model.tilted_mass_from_fields(lam, qr_mass, phi)
            return jnp.sum(phi * mass[None, ...], axis=(1, 2))

        dm_ref_dt = jax.jacfwd(fixed_lambda_moment)(t)
        H = C + jnp.diag(vdiag) + model.cfg.newton_ridge * self.eye2
        rhs = c_dot - dm_ref_dt - vdiag_dot * lam
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

        soft_resid = jnp.linalg.norm(moment + vdiag * lam - c_hat)
        shrinkage = jnp.linalg.norm(moment - c_hat)
        min_soft_hessian_eig = jnp.min(jnp.linalg.eigvalsh(C + jnp.diag(vdiag)))
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

    def _law_metrics(self, t, alpha, eta, c_hat, c_dot, vdiag, vdiag_dot):
        st = self._soft_state(t, alpha, eta, c_hat, c_dot, vdiag, vdiag_dot)
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

    def _full_metrics(self, t, alpha, eta, c_hat, c_dot, vdiag, vdiag_dot):
        st = self._soft_state(t, alpha, eta, c_hat, c_dot, vdiag, vdiag_dot)
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
        self.law_metrics_jit = jax.jit(self._law_metrics)
        self.full_metrics_jit = jax.jit(self._full_metrics)
        self.hard_lift_jit = jax.jit(self._hard_lift)
        self.hard_full_jit = jax.jit(self._hard_full)


# -----------------------------------------------------------------------------
# One frozen-design evaluation
# -----------------------------------------------------------------------------


def evaluate_design_trial(
    model,
    infer: StageCInference,
    eta: np.ndarray,
    shared: SharedTrialData,
    acq_idx: np.ndarray,
    obs_noise_std: float,
    cfg: CConfig,
    compute_action: bool,
) -> Dict[str, float]:
    t_acq, y_acq, v_acq, exact_acq = design_measurements(
        model=model,
        eta=eta,
        shared=shared,
        acq_idx=acq_idx,
        obs_noise_std=obs_noise_std,
        variance_floor=cfg.variance_floor,
    )
    curve = smooth_measurements(
        model=model,
        eta=eta,
        alpha=shared.alpha,
        t_acq=t_acq,
        y_acq=y_acq,
        v_acq=v_acq,
        cfg=cfg,
    )

    times = np.asarray(model.times, dtype=np.float64)
    tw = np.asarray(model.time_w, dtype=np.float64)
    heldout = np.ones(len(times), dtype=bool)
    heldout[acq_idx] = False
    # Endpoints are known by construction and should not dominate the Stage C test.
    heldout[0] = False
    heldout[-1] = False
    interior = np.ones(len(times), dtype=bool)
    interior[[0, -1]] = False

    eta_j = jnp.asarray(eta, dtype=jnp.float64)
    alpha_j = jnp.asarray(shared.alpha, dtype=jnp.float64)

    noisy_rows = []
    oracle_lifts = []
    full_rows = []
    oracle_full_rows = []

    for i, t in enumerate(times):
        args = (
            jnp.asarray(t, dtype=jnp.float64),
            alpha_j,
            eta_j,
            jnp.asarray(curve["c"][i]),
            jnp.asarray(curve["cdot"][i]),
            jnp.asarray(curve["vdiag"][i]),
            jnp.asarray(curve["vdot"][i]),
        )
        noisy_rows.append(np.asarray(infer.law_metrics_jit(*args), dtype=np.float64))
        oracle_lifts.append(float(infer.hard_lift_jit(jnp.asarray(t), alpha_j, eta_j)))
        if compute_action:
            full_rows.append(np.asarray(infer.full_metrics_jit(*args), dtype=np.float64))
            oracle_full_rows.append(np.asarray(infer.hard_full_jit(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64))

    noisy = np.asarray(noisy_rows, dtype=np.float64)
    oracle_lifts = np.asarray(oracle_lifts, dtype=np.float64)

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
        "smoothed_moment_rmse": float(np.sqrt(np.mean([
            np.sum((curve["c"][i] - np.asarray(
                model.measurement_grid(jnp.asarray(t), alpha_j, eta_j), dtype=np.float64
            )) ** 2)
            for i, t in enumerate(times) if interior[i]
        ]))),
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
# Population sanity check and Monte Carlo aggregation
# -----------------------------------------------------------------------------


METRICS = (
    "heldout_mmd2",
    "all_interior_mmd2",
    "oracle_heldout_mmd2",
    "delta_vs_population_heldout_mmd2",
    "heldout_moment_error",
    "acquisition_mean_rmse",
    "smoothed_moment_rmse",
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


def run_condition(
    model,
    infer: StageCInference,
    designs: Dict[str, np.ndarray],
    cfg: CConfig,
    n: int,
    k: int,
    obs_noise_std: float,
) -> Dict[str, Any]:
    acq_idx = acquisition_indices(model.cfg.time_n, k)
    rows = {name: [] for name in designs}

    for trial in range(cfg.trials):
        # Common random numbers across designs make the comparisons paired.
        ss = np.random.SeedSequence([cfg.seed, n, k, int(round(obs_noise_std * 1e6)), trial])
        rng = np.random.default_rng(ss)
        if cfg.alpha_eval_mode == "quadrature":
            alpha_vals = np.asarray(model.alphas, dtype=np.float64)
            alpha = float(alpha_vals[trial % len(alpha_vals)])
        else:
            alpha = float(rng.uniform(model.cfg.alpha_min, model.cfg.alpha_max))

        shared = draw_shared_trial(model, alpha, acq_idx, n, rng)
        do_action = trial < cfg.action_trials
        for name, eta in designs.items():
            rows[name].append(evaluate_design_trial(
                model=model,
                infer=infer,
                eta=eta,
                shared=shared,
                acq_idx=acq_idx,
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
        print(
            f"{label:10s} held-out MMD^2={h['mean']:.6e} ± {h['se']:.2e} | "
            f"Δvs-pop={ex['mean']:+.3e} ± {ex['se']:.2e} | "
            f"action inflation={ai['mean']:.4f} ± {ai['se']:.2e} (n={ai['n']})"
        )
    for key, label in (
        ("full_vs_lift_heldout_mmd2", "Full - Lift"),
        ("full_vs_tangent_heldout_mmd2", "Full - Tangent"),
    ):
        c = cond["paired_comparisons"][key]
        print(
            f"{label:18s}: paired held-out Δ={c['mean_difference_a_minus_b']:+.3e} "
            f"± {c['se_difference']:.2e}; Full better in {100*c['a_better_fraction']:.1f}% of trials"
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
    p.add_argument("--smooth-lengthscale", type=float, default=None)
    p.add_argument("--smooth-ridge", type=float, default=None)
    p.add_argument("--variance-smooth-ridge", type=float, default=None)
    p.add_argument("--alpha-eval-mode", choices=("random", "quadrature"), default=None)
    p.add_argument("--output", type=str, default="stage_c1_finite_measurement_results.json")
    return p


def apply_overrides(cfg: CConfig, args: argparse.Namespace) -> CConfig:
    kw: Dict[str, Any] = {}
    for name in (
        "trials", "action_trials", "seed", "grid_n", "time_n",
        "smooth_lengthscale", "smooth_ridge", "variance_smooth_ridge", "alpha_eval_mode",
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

    # Reuse the validated Stage B numerical model, changing only resolution for
    # this Stage C run.  Scientific parameters remain the same.
    base = backend.preset_config("reference" if cfg.preset != "quick" else "quick")
    bcfg = dataclasses.replace(base, grid_n=cfg.grid_n, time_n=cfg.time_n)
    model = backend.StageB(bcfg)
    infer = StageCInference(model, cfg)

    designs = {
        "lift": degrees_to_eta(cfg.lift_design_deg),
        "tangent": degrees_to_eta(cfg.tangent_design_deg),
        "full": degrees_to_eta(cfg.full_design_deg),
    }

    for k in cfg.k_list:
        acquisition_indices(model.cfg.time_n, k)  # validation
    if cfg.action_trials > cfg.trials:
        raise ValueError("action_trials cannot exceed trials")

    print("=" * 78)
    print("Stage C.1 — finite/noisy measurements + analytic SI")
    print("=" * 78)
    print(f"Backend           : {Path(backend_path).resolve()}")
    print(f"Preset            : {cfg.preset}")
    print(f"Grid / time       : {model.cfg.grid_n}x{model.cfg.grid_n} / {model.cfg.time_n}")
    print(f"Trials            : {cfg.trials} (full-action subset: {cfg.action_trials})")
    print(f"N budgets         : {cfg.n_list}")
    print(f"K budgets         : {cfg.k_list}")
    print(f"Detector noise    : {cfg.noise_list}")
    print("Frozen designs:")
    for name, eta in designs.items():
        print(f"  {name:8s}: ({np.rad2deg(eta[0]):.2f}°, {np.rad2deg(eta[1]):.2f}°)")

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

    conditions = []
    for noise in cfg.noise_list:
        for k in cfg.k_list:
            for n in cfg.n_list:
                cond = run_condition(model, infer, designs, cfg, n=n, k=k, obs_noise_std=noise)
                conditions.append(cond)
                print_condition_summary(cond)

    result = {
        "stage": "C.1 finite/noisy measurements with analytic stochastic interpolant",
        "created_unix_time": time.time(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend_path": str(Path(backend_path).resolve()),
        "stage_c_config": jsonify(cfg),
        "stage_b_config": jsonify(bcfg),
        "designs_rad": {k: v.tolist() for k, v in designs.items()},
        "designs_deg": {k: np.rad2deg(v).tolist() for k, v in designs.items()},
        "population_design_check": pop,
        "conditions": conditions,
        "interpretation_notes": [
            "Frozen-design C.1 test: no sensor angles are re-optimized under noise.",
            "Finite samples are drawn from the same discrete external law used by the Stage B inverse, so N->infinity recovers the Stage B population moments without a truncation confound.",
            "Moment trajectories are smoothed before differentiation; raw independent measurement noise is never finite-differenced.",
            "The uncertainty-aware soft fiber solves E_q[Phi] + V lambda = c_hat and approaches the Stage B hard fiber as V->0.",
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