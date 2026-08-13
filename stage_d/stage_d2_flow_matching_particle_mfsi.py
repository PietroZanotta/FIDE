#!/usr/bin/env python3
"""
Stage D.2: particle-reference MFSI from a learned flow-matching model (NO CNF).

Purpose
-------
Stage D.0 learned a velocity u_theta(t,x) by flow matching on the validated
Stage-B reference interpolation. Stage D.1 kept the analytic reference marginal
and changed only the velocity. Stage D.2 removes that last analytic-marginal
oracle from MFSI:

  1. sample X_0 from the Stage-B reference endpoint,
  2. roll X_0 forward with the learned FM ODE dX/dt = u_theta(t,X),
  3. treat the resulting particles as the empirical reference marginal Q_ref,t^theta,
  4. solve the hard moment I-projection by self-normalized exponential weights,
  5. compute lambda_dot and the MFSI forcing entirely from weighted particle
     statistics and the learned velocity,
  6. smooth the projected particle mass and the signed source q h onto the
     existing Stage-B grid, and use the validated weighted-Poisson solver only
     as a deterministic action oracle.

There is NO CNF density reconstruction, no log-Jacobian integration, no score
model, and no learned density model.

Matched control
---------------
The same initial particle bank is also pushed through the exact analytic Stage-B
reference map. The analytic-particle control uses the SAME particle count, hard
empirical I-projection, KDE/rasterization bandwidth, grid and Poisson solver.
Thus

    analytic-grid oracle -> analytic-particle control -> learned-FM particles

separates finite-bank/rasterization error from learned-reference error.

Particle MFSI calculus
----------------------
At a fixed time, let x_i be reference particles and let

    w_i(lambda) propto exp(lambda^T Phi(x_i))

be the empirical I-projection weights satisfying sum_i w_i Phi_i = c(t).
Define

    m_i = J Phi(x_i) u(t,x_i),
    g_i = lambda^T m_i,
    C   = Cov_w(Phi,Phi).

Differentiating the moment constraint along the reference flow gives

    C lambda_dot
      = c_dot - E_w[m] - Cov_w(Phi, g).

The exact sample forcing is then

    h_i = lambda_dot^T (Phi_i - E_w[Phi]) + g_i - E_w[g].

This is the standard MFSI forcing formula evaluated on the empirical FM
reference law; it requires only samples, u_theta and J Phi u_theta.

Grid realization
----------------
The projected probability mass and signed source are deposited to the Stage-B
cell grid and Gaussian-smoothed with one physical bandwidth:

    q_mass(grid)  ~ KDE of {x_i, w_i},
    qh_mass(grid) ~ KDE of {x_i, w_i h_i}.

The signed source is centered after smoothing so its grid integral is exactly
zero. The resulting q and h are passed to the same Stage-B weighted-Poisson
solver. Because the Poisson action can be sensitive to the smoothing scale,
D.2 reports a matched analytic-particle control and supports an optional
Full-TC bandwidth check.

Recommended run
---------------
python stage_d2_flow_matching_particle_mfsi.py \\
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
    --d0-script stage_d0_flow_matching_reference.py \\
    --checkpoint stage_d0_flow_matching_reference.npz \\
    --preset reference \\
    --output stage_d2_flow_matching_particle_mfsi.json
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
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

Array = jax.Array


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
class D2Config:
    preset: str = "quick"
    seed: int = 20260812

    # Frozen Stage-B/C designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)

    # Empirical reference bank / learned ODE.  Gauss-Hermite is the default in
    # this 2D Gaussian-mixture benchmark because hard exponential tilts can put
    # substantial weight in tails that iid banks resolve very inefficiently.
    bank_mode: str = "gauss-hermite"  # gauss-hermite or iid
    gh_order: int = 20                 # bank size = 2 * gh_order^2
    particles: int = 8192              # used only by iid mode
    rk4_substeps_per_time_interval: int = 8

    # Hard empirical I-projection.
    calibration_steps: int = 24
    calibration_tol: float = 2.0e-8
    newton_step_cap: float = 5.0
    lambda_clip: float = 80.0

    # Particle-to-grid density/source smoothing. Physical units, not cells.
    kde_bandwidth: float = 0.0  # <=0 means 0.35 * Stage-B dx (sub-cell anti-aliasing)
    kde_truncate: float = 4.0

    # Diagnostics.
    ess_warn_fraction: float = 0.03
    max_allowed_calibration_resid: float = 2.0e-6
    min_in_domain_fraction: float = 0.995


def preset_d2_config(name: str) -> D2Config:
    if name == "quick":
        return D2Config()
    if name == "reference":
        return D2Config(
            preset="reference",
            gh_order=36,
            particles=32768,
            rk4_substeps_per_time_interval=16,
            calibration_steps=28,
            calibration_tol=1.0e-9,
            kde_bandwidth=0.0,
        )
    if name == "confirm":
        return D2Config(
            preset="confirm",
            gh_order=48,
            particles=65536,
            rk4_substeps_per_time_interval=24,
            calibration_steps=32,
            calibration_tol=5.0e-10,
            kde_bandwidth=0.0,
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
    if isinstance(x, jax.Array):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    return x


def weighted_mean(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.sum(w[:, None] * x, axis=0)


def normalized_exp_weights(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    ez = np.exp(z)
    return ez / np.sum(ez)


def relative_ess(projected_w: np.ndarray, base_w: np.ndarray) -> float:
    """ESS(projected)/ESS(base), appropriate for nonuniform quadrature weights."""
    ep = 1.0 / np.sum(projected_w * projected_w)
    eb = 1.0 / np.sum(base_w * base_w)
    return float(ep / max(eb, 1.0e-30))


def initial_reference_bank(teacher, cfg: D2Config):
    if cfg.bank_mode == "iid":
        key = jax.random.PRNGKey(int(cfg.seed))
        x0 = np.asarray(teacher.sample_x0(key, int(cfg.particles)), dtype=np.float64)
        base_w = np.full(x0.shape[0], 1.0 / x0.shape[0], dtype=np.float64)
        return x0, base_w
    if cfg.bank_mode != "gauss-hermite":
        raise ValueError(f"Unknown bank_mode {cfg.bank_mode!r}")

    # numpy.polynomial.hermite.hermgauss integrates exp(-z^2).  For a
    # standard normal use x=sqrt(2) z and weights/sqrt(pi).
    z, wz = np.polynomial.hermite.hermgauss(int(cfg.gh_order))
    one_w = wz / math.sqrt(math.pi)
    zz1, zz2 = np.meshgrid(z, z, indexing="ij")
    ww1, ww2 = np.meshgrid(one_w, one_w, indexing="ij")
    noise = math.sqrt(2.0) * float(teacher.sigma) * np.stack(
        [zz1.reshape(-1), zz2.reshape(-1)], axis=-1
    )
    w2 = (ww1 * ww2).reshape(-1)
    plus = noise + np.array([float(teacher.r), 0.0])
    minus = noise + np.array([-float(teacher.r), 0.0])
    x0 = np.concatenate([plus, minus], axis=0).astype(np.float64)
    base_w = np.concatenate([0.5 * w2, 0.5 * w2], axis=0).astype(np.float64)
    base_w /= np.sum(base_w)
    return x0, base_w


def sensor_particle_fields(model, eta: np.ndarray, x: np.ndarray):
    """Return phi[n,m] and grad_phi[n,m,coord] using Stage-B sensor formulas."""
    eta = np.asarray(eta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    centers = float(model.cfg.sensor_radius) * np.stack(
        [np.cos(eta), np.sin(eta)], axis=-1
    )
    diff = x[:, None, :] - centers[None, :, :]
    ell2 = float(model.cfg.sensor_width) ** 2
    phi = np.exp(-0.5 * np.sum(diff * diff, axis=-1) / ell2)
    grad = -(diff / ell2) * phi[..., None]
    return phi, grad


# -----------------------------------------------------------------------------
# Learned FM rollout: keep only Stage-B time nodes, not all RK4 substeps
# -----------------------------------------------------------------------------


def rollout_learned_to_nodes(params, d0, x0: Array, time_n: int, substeps: int) -> Array:
    """Learned ODE states at uniform Stage-B nodes, shape [time_n,N,2]."""
    intervals = int(time_n) - 1
    total_steps = intervals * int(substeps)
    dt = 1.0 / float(total_steps)

    def rk4_one(x, global_i):
        t = global_i.astype(jnp.float64) * dt
        k1 = d0.velocity_mlp(params, t, x)
        k2 = d0.velocity_mlp(params, t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = d0.velocity_mlp(params, t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = d0.velocity_mlp(params, t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def interval_step(carry, interval_i):
        x = carry
        start = interval_i * substeps
        x = jax.lax.fori_loop(
            0,
            substeps,
            lambda j, xx: rk4_one(xx, start + j),
            x,
        )
        return x, x

    idx = jnp.arange(intervals, dtype=jnp.int32)
    _, nodes = jax.lax.scan(interval_step, x0, idx)
    return jnp.concatenate([x0[None, ...], nodes], axis=0)


# -----------------------------------------------------------------------------
# Low-dimensional empirical I-projection
# -----------------------------------------------------------------------------


def solve_empirical_tilt(
    phi: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    ridge: float,
    cfg: D2Config,
    lam0: np.ndarray | None = None,
):
    """Damped Newton solve for a weighted empirical/quadrature reference law."""
    phi = np.asarray(phi, dtype=np.float64)
    base_w = np.asarray(base_w, dtype=np.float64)
    base_w = base_w / np.sum(base_w)
    log_base = np.log(np.maximum(base_w, 1.0e-300))
    target = np.asarray(target, dtype=np.float64)
    n, m = phi.shape
    lam = np.zeros(m, dtype=np.float64) if lam0 is None else np.asarray(lam0, dtype=np.float64).copy()

    def dual(ll):
        return float(logsumexp(log_base + phi @ ll) - np.dot(ll, target))

    iterations = 0
    for it in range(cfg.calibration_steps):
        iterations = it + 1
        logits = log_base + phi @ lam
        w = normalized_exp_weights(logits)
        moment = weighted_mean(w, phi)
        F = moment - target
        resid = float(np.linalg.norm(F))
        if resid <= cfg.calibration_tol:
            break
        centered = phi - moment[None, :]
        C = centered.T @ (w[:, None] * centered)
        H = C + float(ridge) * np.eye(m)
        try:
            step = np.linalg.solve(H, F)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H, rcond=1.0e-12) @ F
        sn = float(np.linalg.norm(step))
        if sn > cfg.newton_step_cap:
            step *= cfg.newton_step_cap / max(sn, 1.0e-30)

        cur = dual(lam)
        accepted = False
        for scale in 0.5 ** np.arange(10, dtype=np.float64):
            cand = np.clip(lam - scale * step, -cfg.lambda_clip, cfg.lambda_clip)
            val = dual(cand)
            if np.isfinite(val) and val <= cur + 1.0e-14:
                lam = cand
                accepted = True
                break
        if not accepted:
            lam = np.clip(lam - 1.0e-3 * step, -cfg.lambda_clip, cfg.lambda_clip)

    logits = log_base + phi @ lam
    w = normalized_exp_weights(logits)
    moment = weighted_mean(w, phi)
    centered = phi - moment[None, :]
    C = centered.T @ (w[:, None] * centered)
    resid = float(np.linalg.norm(moment - target))
    return lam, w, moment, C, {
        "residual": resid,
        "iterations": int(iterations),
        "ess_fraction": relative_ess(w, base_w),
        "lambda_norm": float(np.linalg.norm(lam)),
        "min_cov_eig": float(np.min(np.linalg.eigvalsh(0.5 * (C + C.T)))),
    }


# -----------------------------------------------------------------------------
# Particle MFSI forcing
# -----------------------------------------------------------------------------


def particle_mfsi_state(
    phi: np.ndarray,
    grad_phi: np.ndarray,
    u: np.ndarray,
    base_w: np.ndarray,
    target: np.ndarray,
    c_dot: np.ndarray,
    ridge: float,
    cfg: D2Config,
    lam0: np.ndarray | None = None,
):
    lam, w, moment, C, cal = solve_empirical_tilt(phi, base_w, target, ridge, cfg, lam0)

    # m_i[m] = grad phi_m(x_i) . u(x_i)
    m = np.einsum("nmc,nc->nm", grad_phi, u)
    Em = weighted_mean(w, m)
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
    mean_h_before = float(np.dot(w, h))
    h = h - np.dot(w, h)
    mean_h_after = float(np.dot(w, h))

    # Moment-tangent comparator on the same empirical projected law.
    G = np.einsum("nmc,nkc,n->mk", grad_phi, grad_phi, w)
    r = Em - np.asarray(c_dot, dtype=np.float64)
    Gs = G + float(ridge) * np.eye(G.shape[0])
    try:
        tangent_action = float(r @ np.linalg.solve(Gs, r))
    except np.linalg.LinAlgError:
        tangent_action = float(r @ (np.linalg.pinv(Gs, rcond=1.0e-12) @ r))

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
# Particle -> Stage-B grid smoothing
# -----------------------------------------------------------------------------


def in_domain_mask(model, x: np.ndarray) -> np.ndarray:
    L = float(model.cfg.L)
    # Stage-B cells tile [-L,L]. Points exactly on the upper edge are measure zero.
    return (
        (x[:, 0] >= -L) & (x[:, 0] < L)
        & (x[:, 1] >= -L) & (x[:, 1] < L)
    )


def rasterize_projected_state(
    model,
    x: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    bandwidth: float,
    truncate: float,
):
    n = int(model.cfg.grid_n)
    L = float(model.cfg.L)
    edges = np.linspace(-L, L, n + 1, dtype=np.float64)

    # histogram2d(first coordinate, second coordinate) -> [row,col].
    # Feed y first and x second to match Stage-B arrays q[y,x].
    mass_hist, _, _ = np.histogram2d(
        x[:, 1], x[:, 0], bins=(edges, edges), weights=w
    )
    source_hist, _, _ = np.histogram2d(
        x[:, 1], x[:, 0], bins=(edges, edges), weights=w * h
    )

    sigma_cells = float(bandwidth) / float(model.dx)
    qmass = gaussian_filter(
        mass_hist, sigma=sigma_cells, mode="constant", cval=0.0, truncate=float(truncate)
    )
    qh_mass = gaussian_filter(
        source_hist, sigma=sigma_cells, mode="constant", cval=0.0, truncate=float(truncate)
    )

    mass_before_norm = float(np.sum(qmass))
    if not np.isfinite(mass_before_norm) or mass_before_norm <= 0.0:
        raise RuntimeError("Rasterized projected mass vanished.")
    qmass = qmass / mass_before_norm
    qh_mass = qh_mass / mass_before_norm

    # Enforce the discrete Poisson compatibility condition exactly.
    source_mean_before = float(np.sum(qh_mass))
    qh_mass = qh_mass - qmass * source_mean_before
    source_mean_after = float(np.sum(qh_mass))

    q = qmass / float(model.cell_area)
    h_grid = np.divide(
        qh_mass,
        qmass,
        out=np.zeros_like(qh_mass),
        where=qmass > 1.0e-300,
    )
    return q, qmass, h_grid, {
        "kde_bandwidth": float(bandwidth),
        "kde_sigma_cells": sigma_cells,
        "smoothed_mass_before_normalization": mass_before_norm,
        "source_mass_before_center": source_mean_before,
        "source_mass_after_center": source_mean_after,
    }


# -----------------------------------------------------------------------------
# D.2 evaluator
# -----------------------------------------------------------------------------


class ParticleReferenceMFSI:
    def __init__(self, model, d0, params, teacher, cfg: D2Config):
        self.model = model
        self.d0 = d0
        self.params = params
        self.teacher = teacher
        self.cfg = cfg
        self.velocity_jit = jax.jit(lambda t, x: d0.velocity_mlp(params, t, x))
        self.cdot_jit = jax.jit(jax.jacfwd(model.measurement_grid, argnums=0))
        self.poisson_jit = jax.jit(model.poisson_solve)

        x0_np, base_w = initial_reference_bank(teacher, cfg)
        self.x0 = jnp.asarray(x0_np, dtype=jnp.float64)
        self.base_w = np.asarray(base_w, dtype=np.float64)

        print(
            f"Generating learned FM reference bank: mode={cfg.bank_mode}, N={len(base_w)}, "
            f"time_n={model.cfg.time_n}, RK4 substeps/interval={cfg.rk4_substeps_per_time_interval}",
            flush=True,
        )
        rollout_fun = jax.jit(
            lambda z: rollout_learned_to_nodes(
                params, d0, z, int(model.cfg.time_n), int(cfg.rk4_substeps_per_time_interval)
            )
        )
        self.learned_nodes = np.asarray(rollout_fun(self.x0), dtype=np.float64)

        # Matched exact-flow particle control: same x0, exact Stage-B map, no RK4 error.
        ts = np.asarray(model.times, dtype=np.float64)
        exact = [np.asarray(teacher.pushforward(jnp.asarray(t), self.x0), dtype=np.float64) for t in ts]
        self.analytic_nodes = np.stack(exact, axis=0)

        # Learned velocity at learned nodes and exact velocity at analytic nodes.
        learned_u = []
        exact_u = []
        for k, t in enumerate(ts):
            learned_u.append(
                np.asarray(self.velocity_jit(jnp.asarray(t), jnp.asarray(self.learned_nodes[k])), dtype=np.float64)
            )
            exact_u.append(
                np.asarray(teacher.velocity(jnp.asarray(t), jnp.asarray(self.analytic_nodes[k])), dtype=np.float64)
            )
        self.learned_u_nodes = np.stack(learned_u, axis=0)
        self.analytic_u_nodes = np.stack(exact_u, axis=0)

    def reference_bank_diagnostics(self) -> Dict[str, Any]:
        rows = []
        for k, t in enumerate(np.asarray(self.model.times, dtype=np.float64)):
            xl = self.learned_nodes[k]
            xa = self.analytic_nodes[k]
            err2 = np.sum((xl - xa) ** 2, axis=-1)
            pow2 = np.sum(xa * xa, axis=-1)
            paired_rmse = math.sqrt(float(np.sum(self.base_w * err2)))
            power = max(float(np.sum(self.base_w * pow2)), 1.0e-30)
            ml = in_domain_mask(self.model, xl)
            ma = in_domain_mask(self.model, xa)
            rows.append({
                "t": float(t),
                "paired_rmse": paired_rmse,
                "relative_state_l2": float(paired_rmse / math.sqrt(power)),
                "learned_in_domain_fraction": float(np.mean(ml)),
                "analytic_in_domain_fraction": float(np.mean(ma)),
                "learned_in_domain_base_mass": float(np.sum(self.base_w[ml])),
                "analytic_in_domain_base_mass": float(np.sum(self.base_w[ma])),
            })
        return {
            "times": rows,
            "max_relative_state_l2": float(max(r["relative_state_l2"] for r in rows)),
            "min_learned_in_domain_fraction": float(min(r["learned_in_domain_fraction"] for r in rows)),
            "min_analytic_in_domain_fraction": float(min(r["analytic_in_domain_fraction"] for r in rows)),
            "min_learned_in_domain_base_mass": float(min(r["learned_in_domain_base_mass"] for r in rows)),
            "min_analytic_in_domain_base_mass": float(min(r["analytic_in_domain_base_mass"] for r in rows)),
        }

    def _evaluate_bank_design(self, eta: np.ndarray, bank: str, bandwidth: float | None = None):
        if bank not in ("analytic_particle", "learned_fm_particle"):
            raise ValueError(bank)
        if bandwidth is None:
            bw = float(self.cfg.kde_bandwidth)
            if bw <= 0.0:
                bw = 0.35 * float(self.model.dx)
        else:
            bw = float(bandwidth)
        xnodes = self.analytic_nodes if bank == "analytic_particle" else self.learned_nodes
        unodes = self.analytic_u_nodes if bank == "analytic_particle" else self.learned_u_nodes

        times = np.asarray(self.model.times, dtype=np.float64)
        alphas = np.asarray(self.model.alphas, dtype=np.float64)
        tw = np.asarray(self.model.time_w, dtype=np.float64)
        aw = np.asarray(self.model.alpha_w, dtype=np.float64)

        # Metrics [alpha,time].
        law = np.zeros((len(alphas), len(times)), dtype=np.float64)
        action = np.zeros_like(law)
        tangent = np.zeros_like(law)
        cal_resid = np.zeros_like(law)
        ess = np.zeros_like(law)
        lam_norm = np.zeros_like(law)
        min_cov = np.zeros_like(law)
        min_gram = np.zeros_like(law)
        poisson_resid = np.zeros_like(law)
        grid_moment_err = np.zeros_like(law)
        source_compat = np.zeros_like(law)
        in_fraction = np.zeros(len(times), dtype=np.float64)
        in_base_mass = np.zeros(len(times), dtype=np.float64)

        # Warm starts: one multiplier per scientific alpha as time advances.
        lam_warm = [np.zeros(2, dtype=np.float64) for _ in alphas]

        phi_grid, _ = self.model.sensor_fields(jnp.asarray(eta, dtype=jnp.float64))
        phi_grid_np = np.asarray(phi_grid, dtype=np.float64)

        for kt, t in enumerate(times):
            x_all = xnodes[kt]
            u_all = unodes[kt]
            mask = in_domain_mask(self.model, x_all)
            frac = float(np.mean(mask))
            in_fraction[kt] = frac
            x = x_all[mask]
            u = u_all[mask]
            base_w = self.base_w[mask].copy()
            base_mass = float(np.sum(base_w))
            in_base_mass[kt] = base_mass
            base_w /= max(base_mass, 1.0e-300)
            if x.shape[0] < 100:
                raise RuntimeError(f"Too few in-domain particles at t={t}: {x.shape[0]}")
            phi, grad_phi = sensor_particle_fields(self.model, eta, x)

            for ka, alpha in enumerate(alphas):
                target = np.asarray(
                    self.model.measurement_grid(
                        jnp.asarray(t), jnp.asarray(alpha), jnp.asarray(eta)
                    ),
                    dtype=np.float64,
                )
                c_dot = np.asarray(
                    self.cdot_jit(jnp.asarray(t), jnp.asarray(alpha), jnp.asarray(eta)),
                    dtype=np.float64,
                )

                st = particle_mfsi_state(
                    phi,
                    grad_phi,
                    u,
                    base_w,
                    target,
                    c_dot,
                    float(self.model.cfg.newton_ridge),
                    self.cfg,
                    lam_warm[ka],
                )
                lam_warm[ka] = st["lambda"]

                q, qmass, h_grid, ras = rasterize_projected_state(
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
                tangent[ka, kt] = float(st["diagnostics"]["tangent_action"])
                cal_resid[ka, kt] = float(st["diagnostics"]["residual"])
                ess[ka, kt] = float(st["diagnostics"]["ess_fraction"])
                lam_norm[ka, kt] = float(st["diagnostics"]["lambda_norm"])
                min_cov[ka, kt] = float(st["diagnostics"]["min_cov_eig"])
                min_gram[ka, kt] = float(st["diagnostics"]["min_tangent_gram_eig"])
                poisson_resid[ka, kt] = float(pres)
                grid_moment_err[ka, kt] = float(np.linalg.norm(grid_moment - target))
                source_compat[ka, kt] = abs(float(ras["source_mass_after_center"]))

        W = aw[:, None] * tw[None, :]
        L = float(np.sum(W * law))
        A = float(np.sum(W * action))
        T = float(np.sum(W * tangent))
        return {
            "lift_mmd2": L,
            "full_action": A,
            "tangent_action": T,
            "hidden_action": float(A - T),
            "kde_bandwidth": bw,
            "kde_sigma_cells": float(bw / self.model.dx),
            "particle_count": int(self.x0.shape[0]),
            "bank_mode": self.cfg.bank_mode,
            "gh_order": int(self.cfg.gh_order) if self.cfg.bank_mode == "gauss-hermite" else None,
            "min_in_domain_fraction": float(np.min(in_fraction)),
            "mean_in_domain_fraction": float(np.sum(tw * in_fraction)),
            "min_in_domain_base_mass": float(np.min(in_base_mass)),
            "mean_in_domain_base_mass": float(np.sum(tw * in_base_mass)),
            "max_calibration_residual": float(np.max(cal_resid)),
            "mean_calibration_residual": float(np.sum(W * cal_resid)),
            "min_ess_fraction": float(np.min(ess)),
            "mean_ess_fraction": float(np.sum(W * ess)),
            "max_lambda_norm": float(np.max(lam_norm)),
            "min_calibration_cov_eig": float(np.min(min_cov)),
            "min_tangent_gram_eig": float(np.min(min_gram)),
            "max_poisson_relative_residual": float(np.max(poisson_resid)),
            "max_grid_moment_error_after_kde": float(np.max(grid_moment_err)),
            "mean_grid_moment_error_after_kde": float(np.sum(W * grid_moment_err)),
            "max_abs_smoothed_source_compatibility": float(np.max(source_compat)),
        }

    def evaluate_design(self, eta: np.ndarray):
        eta_j = jnp.asarray(eta, dtype=jnp.float64)
        oracle = self.model.design_metrics_jit(eta_j)
        oracle = self._backend_metrics_dict(np.asarray(oracle, dtype=np.float64))
        analytic_particle = self._evaluate_bank_design(eta, "analytic_particle")
        learned_particle = self._evaluate_bank_design(eta, "learned_fm_particle")
        return {
            "grid_oracle": oracle,
            "analytic_particle": analytic_particle,
            "learned_fm_particle": learned_particle,
            "learned_minus_analytic_particle": {
                "lift_mmd2": float(learned_particle["lift_mmd2"] - analytic_particle["lift_mmd2"]),
                "full_action": float(learned_particle["full_action"] - analytic_particle["full_action"]),
                "full_action_relative_change": float(
                    learned_particle["full_action"] / analytic_particle["full_action"] - 1.0
                ),
            },
            "analytic_particle_minus_grid_oracle": {
                "lift_mmd2": float(analytic_particle["lift_mmd2"] - oracle["lift_mmd2"]),
                "full_action": float(analytic_particle["full_action"] - oracle["full_action"]),
                "full_action_relative_change": float(
                    analytic_particle["full_action"] / oracle["full_action"] - 1.0
                ),
            },
        }

    def _backend_metrics_dict(self, vals: np.ndarray):
        names = getattr(self.model, "METRIC_NAMES", None)
        # StageB instance does not expose module constants; map the packed values directly.
        keys = [
            "lift_mmd2", "full_action", "tangent_action", "hidden_action", "info_score",
            "max_abs_mean_h", "max_poisson_rel_resid", "max_abs_qmean_psi",
            "max_calibration_resid", "min_calibration_cov_eig", "min_tangent_gram_eig",
            "max_operator_floor", "min_pointwise_hidden_action",
        ]
        return {k: float(vals[i]) for i, k in enumerate(keys)}

    def bandwidth_check(self, eta: np.ndarray, bandwidths: Sequence[float]):
        rows = []
        for bw in bandwidths:
            print(f"  bandwidth check h={bw:.3f}", flush=True)
            a = self._evaluate_bank_design(eta, "analytic_particle", bandwidth=float(bw))
            l = self._evaluate_bank_design(eta, "learned_fm_particle", bandwidth=float(bw))
            rows.append({
                "kde_bandwidth": float(bw),
                "analytic_particle_full_action": a["full_action"],
                "learned_fm_particle_full_action": l["full_action"],
                "learned_vs_analytic_relative_action_change": float(
                    l["full_action"] / a["full_action"] - 1.0
                ),
                "analytic_particle_lift_mmd2": a["lift_mmd2"],
                "learned_fm_particle_lift_mmd2": l["lift_mmd2"],
                "analytic_min_ess_fraction": a["min_ess_fraction"],
                "learned_min_ess_fraction": l["min_ess_fraction"],
            })
        return rows


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def full_vs_lift(designs: Dict[str, Any], branch: str):
    full = designs["full"][branch]
    lift = designs["lift"][branch]
    return {
        "law_relative_penalty": float(full["lift_mmd2"] / lift["lift_mmd2"] - 1.0),
        "action_reduction_fraction": float(1.0 - full["full_action"] / lift["full_action"]),
        "full_minus_lift_action": float(full["full_action"] - lift["full_action"]),
    }


def print_summary(payload: Dict[str, Any]):
    print("\n" + "=" * 98)
    print("Stage D.2 particle-reference MFSI summary (flow matching; NO CNF)")
    print("=" * 98)
    for name, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        row = payload["designs"][name]
        g = row["grid_oracle"]
        a = row["analytic_particle"]
        l = row["learned_fm_particle"]
        print(
            f"{label:10s} grid:    L={g['lift_mmd2']:.8f} | A={g['full_action']:.3f}"
        )
        print(
            f"{'':10s} particles exact: L={a['lift_mmd2']:.8f} | A={a['full_action']:.3f} | "
            f"ESSmin={a['min_ess_fraction']:.3f} | calmax={a['max_calibration_residual']:.2e}"
        )
        print(
            f"{'':10s} particles FM:    L={l['lift_mmd2']:.8f} | A={l['full_action']:.3f} | "
            f"ESSmin={l['min_ess_fraction']:.3f} | calmax={l['max_calibration_residual']:.2e}"
        )
        print(
            f"{'':10s} FM-vs-exact particles: dA={100.0*row['learned_minus_analytic_particle']['full_action_relative_change']:+.3f}% | "
            f"dL={row['learned_minus_analytic_particle']['lift_mmd2']:+.3e}"
        )
    print("-" * 98)
    for branch, label in (
        ("grid_oracle", "grid oracle"),
        ("analytic_particle", "analytic particles"),
        ("learned_fm_particle", "learned FM particles"),
    ):
        c = payload["contrasts"][branch]
        print(
            f"Full vs Lift [{label}]: law penalty={100*c['law_relative_penalty']:+.3f}% | "
            f"action reduction={100*c['action_reduction_fraction']:+.2f}%"
        )
    print("=" * 98)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--d0-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d2_flow_matching_particle_mfsi.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--bank-mode", choices=("gauss-hermite", "iid"), default=None)
    p.add_argument("--gh-order", type=int, default=None)
    p.add_argument("--particles", type=int, default=None)
    p.add_argument("--rk4-substeps", type=int, default=None)
    p.add_argument("--kde-bandwidth", type=float, default=None)
    p.add_argument("--run-bandwidth-check", action="store_true")
    p.add_argument(
        "--bandwidths",
        type=str,
        default="0.08,0.12,0.16",
        help="Comma-separated physical KDE bandwidths for optional Full-TC check.",
    )
    return p


def main():
    t0 = time.time()
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect(
        ["stage_b2_transport_conditioned_design.py", "stage_b2_transport_conditioned_design(4).py"]
    )
    if backend_path is None:
        raise FileNotFoundError("Pass --backend /path/to/stage_b2_transport_conditioned_design.py")
    d0_path = Path(args.d0_script) if args.d0_script else autodetect(
        ["stage_d0_flow_matching_reference.py"]
    )
    if d0_path is None:
        raise FileNotFoundError("Pass --d0-script /path/to/stage_d0_flow_matching_reference.py")

    backend = load_module(backend_path, "stage_b2_backend_d2")
    d0 = load_module(d0_path, "stage_d0_backend_d2")
    params, checkpoint_meta = d0.load_checkpoint(Path(args.checkpoint))

    cfg = preset_d2_config(args.preset)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.bank_mode is not None:
        overrides["bank_mode"] = str(args.bank_mode)
    if args.gh_order is not None:
        overrides["gh_order"] = int(args.gh_order)
    if args.particles is not None:
        overrides["particles"] = int(args.particles)
    if args.rk4_substeps is not None:
        overrides["rk4_substeps_per_time_interval"] = int(args.rk4_substeps)
    if args.kde_bandwidth is not None:
        overrides["kde_bandwidth"] = float(args.kde_bandwidth)
    cfg = dataclasses.replace(cfg, **overrides)

    if args.preset == "quick":
        stage_b_cfg = backend.preset_config("quick")
    elif args.preset == "reference":
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=39, time_n=21)
    else:
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=65, time_n=27)

    # Checkpoint/physical compatibility.
    cp_phys = checkpoint_meta.get("physical_system", {})
    physical_checks = {}
    for key, current in (
        ("r", float(stage_b_cfg.r)),
        ("sigma", float(stage_b_cfg.sigma)),
        ("kappa", float(stage_b_cfg.kappa)),
    ):
        saved = float(cp_phys.get(key, current))
        diff = abs(saved - current)
        physical_checks[key] = {"checkpoint": saved, "stage_b": current, "abs_diff": diff}
        if diff > 1.0e-12:
            raise ValueError(f"D0 checkpoint {key}={saved} != Stage-B {current}")

    model = backend.StageB(stage_b_cfg)
    teacher = d0.AnalyticReferenceTeacher(stage_b_cfg)
    evaluator = ParticleReferenceMFSI(model, d0, params, teacher, cfg)
    reference_bank_diagnostics = evaluator.reference_bank_diagnostics()

    design_deg = {
        "lift": cfg.lift_design_deg,
        "tangent": cfg.tangent_design_deg,
        "full": cfg.full_design_deg,
    }
    designs = {}
    for name, deg in design_deg.items():
        print(f"Evaluating {name}: {deg[0]:.2f} deg, {deg[1]:.2f} deg", flush=True)
        eta = np.radians(np.asarray(deg, dtype=np.float64))
        row = evaluator.evaluate_design(eta)
        row["theta_deg"] = list(map(float, deg))
        designs[name] = row

    contrasts = {
        branch: full_vs_lift(designs, branch)
        for branch in ("grid_oracle", "analytic_particle", "learned_fm_particle")
    }
    contrasts["learned_minus_analytic_particle_action_reduction"] = float(
        contrasts["learned_fm_particle"]["action_reduction_fraction"]
        - contrasts["analytic_particle"]["action_reduction_fraction"]
    )
    contrasts["learned_minus_analytic_particle_law_penalty"] = float(
        contrasts["learned_fm_particle"]["law_relative_penalty"]
        - contrasts["analytic_particle"]["law_relative_penalty"]
    )

    bandwidth_check = None
    if args.run_bandwidth_check:
        bws = [float(x.strip()) for x in args.bandwidths.split(",") if x.strip()]
        print("Running Full-TC KDE bandwidth check...", flush=True)
        bandwidth_check = evaluator.bandwidth_check(
            np.radians(np.asarray(cfg.full_design_deg, dtype=np.float64)), bws
        )

    all_branches = [
        designs[d][b]
        for d in designs
        for b in ("analytic_particle", "learned_fm_particle")
    ]
    min_ess = min(r["min_ess_fraction"] for r in all_branches)
    max_cal = max(r["max_calibration_residual"] for r in all_branches)
    min_domain = min(r["min_in_domain_base_mass"] for r in all_branches)
    checks = {
        "finite_outputs": bool(all(
            np.isfinite(v)
            for r in all_branches
            for v in r.values()
            if isinstance(v, (int, float))
        )),
        "empirical_calibration_residual_small": bool(max_cal < cfg.max_allowed_calibration_resid),
        "ess_above_warning_fraction": bool(min_ess >= cfg.ess_warn_fraction),
        "in_domain_fraction_high": bool(min_domain >= cfg.min_in_domain_fraction),
        "full_tc_learned_action_below_lift": bool(
            designs["full"]["learned_fm_particle"]["full_action"]
            < designs["lift"]["learned_fm_particle"]["full_action"]
        ),
        "no_cnf_used": True,
    }

    payload = {
        "stage": "D.2",
        "method": "flow-matching ODE particle reference + empirical hard I-projection + smoothed particle Poisson realization; no CNF",
        "backend_path": str(Path(backend_path).resolve()),
        "d0_script_path": str(Path(d0_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_metadata": checkpoint_meta,
        "physical_parameter_checks": physical_checks,
        "config": jsonify(cfg),
        "stage_b_resolution": {
            "grid_n": int(stage_b_cfg.grid_n),
            "time_n": int(stage_b_cfg.time_n),
            "alpha_n": int(stage_b_cfg.alpha_n),
            "dx": float(model.dx),
        },
        "reference_bank_diagnostics": reference_bank_diagnostics,
        "designs": designs,
        "contrasts": contrasts,
        "full_tc_bandwidth_check": bandwidth_check,
        "checks": checks,
        "interpretation": [
            "D.2 no longer evaluates the analytic reference density inside the I-projection. The learned FM ODE rollout itself supplies the empirical reference marginal.",
            "The matched analytic-particle control uses the same initial weighted particles, empirical I-projection, anti-alias smoothing, grid and Poisson solver, so its difference from the grid oracle measures finite-bank/quadrature/rasterization error.",
            "The learned-FM-particle minus analytic-particle contrast is the primary learned-reference effect in D.2.",
            "The empirical MFSI forcing is computed from weighted particle statistics using C lambda_dot = c_dot - E[J Phi u] - Cov(Phi, lambda^T J Phi u).",
            "Sub-cell Gaussian smoothing is used only as anti-aliasing when depositing q and q h on the deterministic Poisson grid; it is not a CNF density model. Absolute particle-grid action is therefore a separate discretization, and the primary D.2 action contrast is learned-FM particles versus the matched analytic-particle control.",
            "Finite/noisy scientific measurements remain off in D.2 so learned-reference approximation is isolated before composition with Stage C.",
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
    out.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved diagnostics: {out}")


if __name__ == "__main__":
    main()
