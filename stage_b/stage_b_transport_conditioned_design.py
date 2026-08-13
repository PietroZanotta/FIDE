#!/usr/bin/env python3
"""
Stage B: differentiable transport-conditioned experimental design on nonlinear moment fibers.

Single-file JAX experiment. It implements:
  * a symmetric two-lobe scientific population family P_t^alpha,
  * two localized Gaussian sensors eta=(theta1, theta2),
  * an analytic anisotropic stochastic-interpolant reference q_ref,t and velocity u_t,
  * nonlinear exponential-family I-projection onto the two measured moments,
  * implicit differentiation through the 2x2 multiplier solve,
  * the MFSI weighted-Poisson minimum-energy correction on a 2D grid,
  * a local kinetic-tangent comparator,
  * Gaussian-kernel MMD^2 as pointwise law-reconstruction (Lift) loss,
  * a dense two-angle design oracle,
  * end-to-end finite-difference gradient checks,
  * an optional differentiable full-transport optimizer,
  * clear terminal summaries and a single JSON result file.

Recommended first run:
    python stage_b_transport_conditioned_design.py --preset quick

Stronger follow-up after the quick run passes:
    python stage_b_transport_conditioned_design.py --preset reference

The optional --optimizer flag is intentionally not part of the first diagnostic run;
the dense oracle plus end-to-end gradient check should be inspected first.

The script intentionally avoids Tesseract wrappers and neural PDE solvers in Stage B so
that the nonlinear inverse/design geometry is isolated from packaging and approximation error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.sparse.linalg import cg

Array = jax.Array
PI = math.pi


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # Scientific population / sensor geometry
    r: float = 1.5
    sigma: float = 0.30
    sensor_radius: float = 1.5
    sensor_width: float = 0.45
    alpha_min: float = PI / 6.0
    alpha_max: float = PI / 3.0
    kappa: float = 0.30

    # Spatial / temporal numerical resolution
    L: float = 3.2
    grid_n: int = 19
    time_n: int = 7
    alpha_n: int = 3
    dense_angle_n: int = 7

    # MMD law metric
    mmd_bandwidth: float = 0.55

    # Calibration and Poisson solves
    newton_steps: int = 18
    newton_ridge: float = 1e-7
    newton_damping: float = 1.0
    operator_floor_rel: float = 2e-5
    cg_tol: float = 2e-8
    cg_maxiter: int = 220
    gauge_strength: float = 1.0

    # Design constraints / optimization
    lift_tau: float = 0.05
    min_sep_deg: float = 10.0
    tc_penalty: float = 2.0e3
    sep_penalty: float = 2.0e2
    optimizer_steps: int = 30
    optimizer_lr: float = 0.035
    optimizer_starts: int = 2

    # Diagnostics
    fd_eps: float = 2e-4
    run_optimizer: bool = False


def preset_config(name: str) -> Config:
    if name == "quick":
        return Config()
    if name == "reference":
        return Config(
            grid_n=39,
            time_n=15,
            alpha_n=5,
            dense_angle_n=17,
            cg_tol=1e-8,
            cg_maxiter=520,
            optimizer_steps=90,
            optimizer_starts=4,
            fd_eps=1.5e-4,
        )
    raise ValueError(f"unknown preset {name!r}")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def trap_weights(n: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5
    w /= (n - 1)
    return w


def gauss_legendre_uniform(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (b - a) * z + 0.5 * (a + b)
    # Uniform-prior expectation: integral/(b-a), hence weights w/2.
    return x.astype(np.float64), (0.5 * w).astype(np.float64)


def angle_periodic_distance(a: float, b: float) -> float:
    """Distance on a pi-periodic angle circle."""
    d = abs((a - b) % PI)
    return min(d, PI - d)


def unordered_design_distance(x: np.ndarray, y: np.ndarray) -> float:
    d_direct = math.hypot(
        angle_periodic_distance(float(x[0]), float(y[0])),
        angle_periodic_distance(float(x[1]), float(y[1])),
    )
    d_swap = math.hypot(
        angle_periodic_distance(float(x[0]), float(y[1])),
        angle_periodic_distance(float(x[1]), float(y[0])),
    )
    return min(d_direct, d_swap)


def canonical_eta_np(eta: np.ndarray) -> np.ndarray:
    x = np.mod(np.asarray(eta, dtype=np.float64), PI)
    return np.sort(x)


def jsonify(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonify(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (jax.Array,)):
        return np.asarray(x).tolist()
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    return x


# -----------------------------------------------------------------------------
# Stage B model factory
# -----------------------------------------------------------------------------

class StageB:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        n = cfg.grid_n
        dx = 2.0 * cfg.L / n
        centers = -cfg.L + (np.arange(n, dtype=np.float64) + 0.5) * dx
        xx, yy = np.meshgrid(centers, centers, indexing="xy")
        xy = np.stack([xx, yy], axis=-1)

        self.dx = float(dx)
        self.cell_area = float(dx * dx)
        self.xy = jnp.asarray(xy)
        self.xy_flat = self.xy.reshape((-1, 2))

        self.times = jnp.asarray(np.linspace(0.0, 1.0, cfg.time_n, dtype=np.float64))
        self.time_w = jnp.asarray(trap_weights(cfg.time_n))
        alphas, alpha_w = gauss_legendre_uniform(cfg.alpha_n, cfg.alpha_min, cfg.alpha_max)
        self.alphas = jnp.asarray(alphas)
        self.alpha_w = jnp.asarray(alpha_w)

        # Exact discrete Gaussian-kernel MMD convolution stencil.
        offs = (np.arange(-(n - 1), n, dtype=np.float64) * dx)
        ox, oy = np.meshgrid(offs, offs, indexing="xy")
        kernel = np.exp(-(ox * ox + oy * oy) / (2.0 * cfg.mmd_bandwidth ** 2))
        self.mmd_kernel = jnp.asarray(kernel)

        self.eye2 = jnp.eye(2, dtype=jnp.float64)
        self._build_jitted_functions()

    # ------------------------------------------------------------------
    # Geometry / densities
    # ------------------------------------------------------------------

    @staticmethod
    def e(theta: Array) -> Array:
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)])

    @staticmethod
    def rotation(theta: Array) -> Array:
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)

    def A_matrix(self, t: Array) -> Array:
        omega = 0.5 * jnp.pi * t
        s = self.cfg.kappa * jnp.sin(jnp.pi * t)
        D = jnp.diag(jnp.array([jnp.exp(s), jnp.exp(-s)]))
        return self.rotation(omega) @ D

    def B_matrix(self, t: Array) -> Array:
        # B = dot(A) A^{-1} = omega_dot J + R diag(sdot,-sdot) R^T.
        omega = 0.5 * jnp.pi * t
        R = self.rotation(omega)
        J = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)
        sdot = self.cfg.kappa * jnp.pi * jnp.cos(jnp.pi * t)
        S = jnp.diag(jnp.array([sdot, -sdot]))
        return 0.5 * jnp.pi * J + R @ S @ R.T

    def g_density(self, alpha: Array) -> Array:
        mu = self.cfg.r * self.e(alpha)
        x = self.xy
        s2 = self.cfg.sigma ** 2
        norm = 1.0 / (2.0 * jnp.pi * s2)
        d1 = jnp.sum((x - mu) ** 2, axis=-1)
        d2 = jnp.sum((x + mu) ** 2, axis=-1)
        return 0.5 * norm * (jnp.exp(-0.5 * d1 / s2) + jnp.exp(-0.5 * d2 / s2))

    def reference_density(self, t: Array) -> Array:
        # A_t is volume preserving, so q_ref,t(x)=G_0(A_t^{-1}x).
        A = self.A_matrix(t)
        invA = jnp.linalg.inv(A)
        y = self.xy_flat @ invA.T
        mu = jnp.array([self.cfg.r, 0.0], dtype=jnp.float64)
        s2 = self.cfg.sigma ** 2
        norm = 1.0 / (2.0 * jnp.pi * s2)
        d1 = jnp.sum((y - mu) ** 2, axis=-1)
        d2 = jnp.sum((y + mu) ** 2, axis=-1)
        q = 0.5 * norm * (jnp.exp(-0.5 * d1 / s2) + jnp.exp(-0.5 * d2 / s2))
        return q.reshape((self.cfg.grid_n, self.cfg.grid_n))

    def normalize_density(self, q: Array) -> Tuple[Array, Array]:
        z = jnp.sum(q) * self.cell_area
        qn = q / z
        mass = qn * self.cell_area
        return qn, mass

    def reference_q_mass(self, t: Array) -> Tuple[Array, Array]:
        return self.normalize_density(self.reference_density(t))

    def external_density(self, t: Array, alpha: Array) -> Array:
        w0 = (1.0 - t) ** 2
        wm = 2.0 * t * (1.0 - t)
        w1 = t ** 2
        return w0 * self.g_density(0.0) + wm * self.g_density(alpha) + w1 * self.g_density(0.5 * jnp.pi)

    def external_q_mass(self, t: Array, alpha: Array) -> Tuple[Array, Array]:
        return self.normalize_density(self.external_density(t, alpha))

    # ------------------------------------------------------------------
    # Sensors and exact population measurements
    # ------------------------------------------------------------------

    def sensor_fields(self, eta: Array) -> Tuple[Array, Array]:
        """Return phi[m,y,x] and grad_phi[m,y,x,coord]."""
        centers = self.cfg.sensor_radius * jax.vmap(self.e)(eta)  # [2,2]
        diff = self.xy[None, ...] - centers[:, None, None, :]
        ell2 = self.cfg.sensor_width ** 2
        phi = jnp.exp(-0.5 * jnp.sum(diff * diff, axis=-1) / ell2)
        grad = -(diff / ell2) * phi[..., None]
        return phi, grad

    def gaussian_sensor_expectation(self, alpha: Array, eta: Array) -> Array:
        mu = self.cfg.r * self.e(alpha)
        centers = self.cfg.sensor_radius * jax.vmap(self.e)(eta)
        v = self.cfg.sensor_width ** 2 + self.cfg.sigma ** 2
        pref = self.cfg.sensor_width ** 2 / v  # d=2
        dplus = jnp.sum((centers - mu[None, :]) ** 2, axis=-1)
        dminus = jnp.sum((centers + mu[None, :]) ** 2, axis=-1)
        return 0.5 * pref * (
            jnp.exp(-0.5 * dplus / v) + jnp.exp(-0.5 * dminus / v)
        )

    def measurement_exact(self, t: Array, alpha: Array, eta: Array) -> Array:
        w0 = (1.0 - t) ** 2
        wm = 2.0 * t * (1.0 - t)
        w1 = t ** 2
        return (
            w0 * self.gaussian_sensor_expectation(0.0, eta)
            + wm * self.gaussian_sensor_expectation(alpha, eta)
            + w1 * self.gaussian_sensor_expectation(0.5 * jnp.pi, eta)
        )

    def measurement_grid(self, t: Array, alpha: Array, eta: Array) -> Array:
        # Finite-domain population measurement used by the numerical inverse.
        # This guarantees that the target moment is feasible on the same support
        # as the exponential-family calibration. The analytic infinite-domain
        # formula is retained as an independent truncation diagnostic.
        phi, _ = self.sensor_fields(eta)
        _, pmass = self.external_q_mass(t, alpha)
        return jnp.sum(phi * pmass[None, ...], axis=(1, 2))

    # ------------------------------------------------------------------
    # Nonlinear information projection
    # ------------------------------------------------------------------

    def tilted_mass_from_fields(self, lam: Array, qref_mass: Array, phi: Array) -> Array:
        flat_phi = phi.reshape((2, -1))
        logits = jnp.log(jnp.maximum(qref_mass.reshape(-1), 1e-300)) + lam @ flat_phi
        mass = jax.nn.softmax(logits)
        return mass.reshape(qref_mass.shape)

    def calibration_residual_from_fields(
        self, lam: Array, t: Array, alpha: Array, eta: Array, phi: Array
    ) -> Array:
        _, qref_mass = self.reference_q_mass(t)
        mass = self.tilted_mass_from_fields(lam, qref_mass, phi)
        moment = jnp.sum(phi * mass[None, ...], axis=(1, 2))
        return moment - self.measurement_grid(t, alpha, eta)

    def calibration_cov_from_fields(
        self, lam: Array, t: Array, eta: Array, phi: Array
    ) -> Array:
        _, qref_mass = self.reference_q_mass(t)
        mass = self.tilted_mass_from_fields(lam, qref_mass, phi)
        m = jnp.sum(phi * mass[None, ...], axis=(1, 2))
        centered = phi - m[:, None, None]
        C = jnp.einsum("myx,nyx,yx->mn", centered, centered, mass)
        return C

    def _solve_lambda_raw(self, t: Array, alpha: Array, eta: Array) -> Array:
        phi, _ = self.sensor_fields(eta)

        _, qref_mass = self.reference_q_mass(t)
        target = self.measurement_grid(t, alpha, eta)
        flat_phi = phi.reshape((2, -1))
        log_base = jnp.log(jnp.maximum(qref_mass.reshape(-1), 1e-300))

        def dual(lam):
            return jsp.special.logsumexp(log_base + lam @ flat_phi) - lam @ target

        def body(_, lam):
            F = self.calibration_residual_from_fields(lam, t, alpha, eta, phi)
            C = self.calibration_cov_from_fields(lam, t, eta, phi)
            step = jnp.linalg.solve(C + self.cfg.newton_ridge * self.eye2, F)
            # Trust-region cap is essential when a localized sensor has almost no
            # reference mass: the raw Newton step can otherwise be enormous even
            # though the convex optimum has a moderate multiplier.
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(1.0, 5.0 / jnp.maximum(step_norm, 1e-12))
            # Fixed candidate backtracking: choose the dual-decreasing step.
            scales = self.cfg.newton_damping * (0.5 ** jnp.arange(9, dtype=jnp.float64))
            cands = lam[None, :] - scales[:, None] * step[None, :]
            vals = jax.vmap(dual)(cands)
            best = cands[jnp.argmin(vals)]
            return jnp.clip(best, -80.0, 80.0)

        return jax.lax.fori_loop(0, self.cfg.newton_steps, body, jnp.zeros(2, dtype=jnp.float64))

    def _build_solve_lambda(self):
        model = self

        @jax.custom_vjp
        def solve_lambda(t: Array, alpha: Array, eta: Array) -> Array:
            return model._solve_lambda_raw(t, alpha, eta)

        def fwd(t, alpha, eta):
            lam = model._solve_lambda_raw(t, alpha, eta)
            return lam, (lam, t, alpha, eta)

        def bwd(saved, bar_lam):
            lam, t, alpha, eta = saved
            phi, _ = model.sensor_fields(eta)
            C = model.calibration_cov_from_fields(lam, t, eta, phi)
            adj = jnp.linalg.solve(C.T + model.cfg.newton_ridge * model.eye2, bar_lam)

            def fixed_lam_residual(tt, aa, ee):
                ph, _ = model.sensor_fields(ee)
                return model.calibration_residual_from_fields(lam, tt, aa, ee, ph)

            _, pullback = jax.vjp(fixed_lam_residual, t, alpha, eta)
            bt, ba, be = pullback(-adj)
            return bt, ba, be

        solve_lambda.defvjp(fwd, bwd)
        self.solve_lambda = solve_lambda

    def projected_state(self, t: Array, alpha: Array, eta: Array):
        phi, grad_phi = self.sensor_fields(eta)
        qref, qref_mass = self.reference_q_mass(t)
        lam = self.solve_lambda(t, alpha, eta)
        qmass = self.tilted_mass_from_fields(lam, qref_mass, phi)
        q = qmass / self.cell_area
        c = self.measurement_grid(t, alpha, eta)

        # Implicit time derivative lambda_dot from dF/dt + C lambda_dot = 0.
        C = self.calibration_cov_from_fields(lam, t, eta, phi)

        def residual_at_time(tt):
            return self.calibration_residual_from_fields(lam, tt, alpha, eta, phi)

        dFdt = jax.jacfwd(residual_at_time)(t)
        lam_dot = jnp.linalg.solve(C + self.cfg.newton_ridge * self.eye2, -dFdt)
        return q, qmass, phi, grad_phi, lam, lam_dot, c, C

    # ------------------------------------------------------------------
    # MFSI forcing and weighted-Poisson solver
    # ------------------------------------------------------------------

    def msfi_forcing(self, t: Array, alpha: Array, eta: Array):
        q, qmass, phi, grad_phi, lam, lam_dot, c, C = self.projected_state(t, alpha, eta)
        B = self.B_matrix(t)
        u = self.xy_flat @ B.T
        u = u.reshape((self.cfg.grid_n, self.cfg.grid_n, 2))
        jphi_u = jnp.einsum("myxc,yxc->myx", grad_phi, u)

        # Stable MFSI forcing formula:
        # h = lambda_dot^T(Phi-c) + lambda^T JPhi u - E_q[lambda^T JPhi u].
        term_time = jnp.einsum("m,myx->yx", lam_dot, phi - c[:, None, None])
        adv_scalar = jnp.einsum("m,myx->yx", lam, jphi_u)
        adv_centered = adv_scalar - jnp.sum(qmass * adv_scalar)
        h_raw = term_time + adv_centered
        mean_h_raw = jnp.sum(qmass * h_raw)
        h = h_raw - mean_h_raw  # exact discrete compatibility safeguard

        c_dot = jax.jacfwd(lambda tt: self.measurement_grid(tt, alpha, eta))(t)
        ref_moment_rate = jnp.sum(jphi_u * qmass[None, ...], axis=(1, 2))
        r = ref_moment_rate - c_dot

        # Continuous kinetic-tangent Gram, evaluated under the same q-grid quadrature.
        G = jnp.einsum("myxc,nyxc,yx->mn", grad_phi, grad_phi, qmass)
        tangent_action = r @ jnp.linalg.solve(G + self.cfg.newton_ridge * self.eye2, r)

        return {
            "q": q,
            "qmass": qmass,
            "phi": phi,
            "grad_phi": grad_phi,
            "lam": lam,
            "lam_dot": lam_dot,
            "c": c,
            "C": C,
            "h": h,
            "mean_h_raw": mean_h_raw,
            "G": G,
            "r": r,
            "tangent_action": tangent_action,
        }

    def weighted_laplacian(self, psi: Array, qop: Array) -> Array:
        dx2 = self.dx * self.dx
        qx = 0.5 * (qop[:, :-1] + qop[:, 1:])
        qy = 0.5 * (qop[:-1, :] + qop[1:, :])
        out = jnp.zeros_like(psi)

        diffx = psi[:, :-1] - psi[:, 1:]
        out = out.at[:, :-1].add(qx * diffx / dx2)
        out = out.at[:, 1:].add(-qx * diffx / dx2)

        diffy = psi[:-1, :] - psi[1:, :]
        out = out.at[:-1, :].add(qy * diffy / dx2)
        out = out.at[1:, :].add(-qy * diffy / dx2)
        return out

    def weighted_laplacian_diag(self, qop: Array) -> Array:
        dx2 = self.dx * self.dx
        qx = 0.5 * (qop[:, :-1] + qop[:, 1:]) / dx2
        qy = 0.5 * (qop[:-1, :] + qop[1:, :]) / dx2
        d = jnp.zeros_like(qop)
        d = d.at[:, :-1].add(qx)
        d = d.at[:, 1:].add(qx)
        d = d.at[:-1, :].add(qy)
        d = d.at[1:, :].add(qy)
        return d

    def poisson_solve(self, q: Array, h: Array):
        # Mild positive operator floor stabilizes the weighted elliptic solve in
        # near-vacuum cells; the exact q remains in the forcing and diagnostics.
        q_floor = self.cfg.operator_floor_rel * jnp.max(q)
        qop = q + q_floor

        rhs = -(q * h).reshape(-1)
        v = q.reshape(-1)
        v = v / jnp.maximum(jnp.linalg.norm(v), 1e-300)

        def matvec(z_flat):
            z = z_flat.reshape(q.shape)
            kz = self.weighted_laplacian(z, qop).reshape(-1)
            return kz + self.cfg.gauge_strength * v * jnp.dot(v, z_flat)

        diag = self.weighted_laplacian_diag(qop).reshape(-1) + self.cfg.gauge_strength * v * v

        def precond(r):
            return r / jnp.maximum(diag, 1e-10)

        psi_flat, _ = cg(
            matvec,
            rhs,
            tol=self.cfg.cg_tol,
            atol=0.0,
            maxiter=self.cfg.cg_maxiter,
            M=precond,
        )
        psi = psi_flat.reshape(q.shape)
        Kpsi = self.weighted_laplacian(psi, qop)

        # Discrete Dirichlet energy for the regularized weighted operator.
        full_action = self.cell_area * jnp.sum(psi * Kpsi)
        resid = Kpsi.reshape(-1) - rhs
        rel_resid = jnp.linalg.norm(resid) / jnp.maximum(jnp.linalg.norm(rhs), 1e-14)
        qmean_psi = jnp.sum((q * self.cell_area) * psi)
        return full_action, psi, rel_resid, qmean_psi, q_floor

    # ------------------------------------------------------------------
    # Law metric / pointwise scenario metrics
    # ------------------------------------------------------------------

    def gaussian_mmd2_mass(self, p: Array, q: Array) -> Array:
        # p,q are discrete probability masses on the common Cartesian grid.
        kp = jsp.signal.fftconvolve(p, self.mmd_kernel, mode="same")
        kq = jsp.signal.fftconvolve(q, self.mmd_kernel, mode="same")
        pp = jnp.sum(p * kp)
        qq = jnp.sum(q * kq)
        pq = jnp.sum(p * kq)
        return jnp.maximum(pp + qq - 2.0 * pq, 0.0)

    def one_time_metrics(self, t: Array, alpha: Array, eta: Array) -> Array:
        state = self.msfi_forcing(t, alpha, eta)
        full_action, _, poisson_rel, qmean_psi, q_floor = self.poisson_solve(state["q"], state["h"])
        _, p_mass = self.external_q_mass(t, alpha)
        lift = self.gaussian_mmd2_mass(state["qmass"], p_mass)

        cal_res = jnp.linalg.norm(
            jnp.sum(state["phi"] * state["qmass"][None, ...], axis=(1, 2)) - state["c"]
        )
        min_cov_eig = jnp.min(jnp.linalg.eigvalsh(state["C"]))
        min_gram_eig = jnp.min(jnp.linalg.eigvalsh(state["G"]))
        hidden = full_action - state["tangent_action"]

        # Packed output keeps nested vmap/JIT simple.
        return jnp.array([
            lift,
            full_action,
            state["tangent_action"],
            hidden,
            jnp.abs(state["mean_h_raw"]),
            poisson_rel,
            jnp.abs(qmean_psi),
            cal_res,
            min_cov_eig,
            min_gram_eig,
            q_floor,
        ])

    def one_time_lift_full(self, t: Array, alpha: Array, eta: Array) -> Array:
        state = self.msfi_forcing(t, alpha, eta)
        full_action, _, _, _, _ = self.poisson_solve(state["q"], state["h"])
        _, p_mass = self.external_q_mass(t, alpha)
        lift = self.gaussian_mmd2_mass(state["qmass"], p_mass)
        return jnp.array([lift, full_action])

    def design_lift_full(self, eta: Array) -> Array:
        rows = jax.vmap(
            lambda alpha: jax.vmap(lambda t: self.one_time_lift_full(t, alpha, eta))(self.times)
        )(self.alphas)
        by_alpha = jnp.sum(self.time_w[None, :, None] * rows, axis=1)
        return jnp.sum(self.alpha_w[:, None] * by_alpha, axis=0)

    def design_lift_tangent(self, eta: Array) -> Array:
        # Cheaper objective used for Lift/tangent gradient checks: no Poisson CG.
        def one_time(t, alpha):
            state = self.msfi_forcing(t, alpha, eta)
            _, p_mass = self.external_q_mass(t, alpha)
            lift = self.gaussian_mmd2_mass(state["qmass"], p_mass)
            return jnp.array([lift, state["tangent_action"]])

        all_rows = jax.vmap(
            lambda alpha: jax.vmap(lambda t: one_time(t, alpha))(self.times)
        )(self.alphas)
        by_alpha = jnp.sum(self.time_w[None, :, None] * all_rows, axis=1)
        return jnp.sum(self.alpha_w[:, None] * by_alpha, axis=0)

    def info_score(self, eta: Array) -> Array:
        def one_alpha(alpha):
            def one_t(t):
                dc_da = jax.jacfwd(lambda aa: self.measurement_exact(t, aa, eta))(alpha)
                return jnp.sum(dc_da * dc_da)
            return jnp.sum(self.time_w * jax.vmap(one_t)(self.times))
        vals = jax.vmap(one_alpha)(self.alphas)
        return jnp.sum(self.alpha_w * vals)

    def design_metrics(self, eta: Array) -> Array:
        # Compute the expensive Poisson solves exactly once, then reuse them for
        # both averaged objectives and max/min diagnostics.
        all_rows = jax.vmap(
            lambda alpha: jax.vmap(lambda t: self.one_time_metrics(t, alpha, eta))(self.times)
        )(self.alphas)
        by_alpha = jnp.sum(self.time_w[None, :, None] * all_rows, axis=1)
        avg = jnp.sum(self.alpha_w[:, None] * by_alpha, axis=0)
        info = self.info_score(eta)

        # Diagnostics that should be maxima/minima over scenario/time, not averages.
        max_mean_h = jnp.max(all_rows[..., 4])
        max_poisson_rel = jnp.max(all_rows[..., 5])
        max_qmean_psi = jnp.max(all_rows[..., 6])
        max_cal_res = jnp.max(all_rows[..., 7])
        min_cov_eig = jnp.min(all_rows[..., 8])
        min_gram_eig = jnp.min(all_rows[..., 9])
        max_q_floor = jnp.max(all_rows[..., 10])
        min_hidden = jnp.min(all_rows[..., 3])

        # [lift, full, tangent, hidden(avg), info, diagnostics...]
        return jnp.array([
            avg[0], avg[1], avg[2], avg[1] - avg[2], info,
            max_mean_h, max_poisson_rel, max_qmean_psi, max_cal_res,
            min_cov_eig, min_gram_eig, max_q_floor, min_hidden,
        ])

    # ------------------------------------------------------------------
    # JIT wrappers and design objectives
    # ------------------------------------------------------------------

    def _build_jitted_functions(self):
        self._build_solve_lambda()
        self.design_metrics_jit = jax.jit(self.design_metrics)
        self.design_lift_full_jit = jax.jit(self.design_lift_full)
        self.design_lift_tangent_jit = jax.jit(self.design_lift_tangent)
        self.info_score_jit = jax.jit(self.info_score)

    def sep_distance_jax(self, eta: Array) -> Array:
        # pi-periodic separation; arccos(cos(2d))/2 is in [0,pi/2].
        d = eta[0] - eta[1]
        return 0.5 * jnp.arccos(jnp.clip(jnp.cos(2.0 * d), -1.0, 1.0))

    def full_tc_penalized(self, eta_raw: Array, lift_max: Array) -> Array:
        eta = jnp.mod(eta_raw, jnp.pi)
        lf = self.design_lift_full(eta)
        lift, full = lf[0], lf[1]
        viol = jax.nn.relu((lift - lift_max) / jnp.maximum(lift_max, 1e-12))
        min_sep = math.radians(self.cfg.min_sep_deg)
        sep_viol = jax.nn.relu((min_sep - self.sep_distance_jax(eta)) / min_sep)
        return full + self.cfg.tc_penalty * viol * viol + self.cfg.sep_penalty * sep_viol * sep_viol

    # ------------------------------------------------------------------
    # Independent diagnostics
    # ------------------------------------------------------------------

    def measurement_grid_error(self, eta: Array, t: float, alpha: float) -> float:
        phi, _ = self.sensor_fields(eta)
        _, pmass = self.external_q_mass(jnp.array(t), jnp.array(alpha))
        c_grid = jnp.sum(phi * pmass[None, ...], axis=(1, 2))
        c_exact = self.measurement_exact(jnp.array(t), jnp.array(alpha), eta)
        return float(jnp.max(jnp.abs(c_grid - c_exact)))


# -----------------------------------------------------------------------------
# Dense design scan and optimizer
# -----------------------------------------------------------------------------

METRIC_NAMES = [
    "lift_mmd2",
    "full_action",
    "tangent_action",
    "hidden_action",
    "info_score",
    "max_abs_mean_h",
    "max_poisson_rel_resid",
    "max_abs_qmean_psi",
    "max_calibration_resid",
    "min_calibration_cov_eig",
    "min_tangent_gram_eig",
    "max_operator_floor",
    "min_pointwise_hidden_action",
]


def metrics_dict(row: np.ndarray) -> Dict[str, float]:
    return {name: float(row[i]) for i, name in enumerate(METRIC_NAMES)}


def dense_scan(model: StageB) -> List[Dict[str, Any]]:
    cfg = model.cfg
    angles = np.linspace(0.0, PI, cfg.dense_angle_n, endpoint=False, dtype=np.float64)
    min_sep = math.radians(cfg.min_sep_deg)
    etas = []
    for i, th1 in enumerate(angles):
        for j in range(i + 1, len(angles)):
            th2 = angles[j]
            if angle_periodic_distance(float(th1), float(th2)) >= min_sep:
                etas.append([th1, th2])
    etas_np = np.asarray(etas, dtype=np.float64)
    rows: List[Dict[str, Any]] = []

    print(f"\nDense scan: {cfg.dense_angle_n} angular nodes on [0, pi), {len(etas_np)} permutation-reduced designs")
    batch_size = 3 if cfg.grid_n <= 30 else 2
    batched = jax.jit(jax.vmap(model.design_metrics))
    for k in range(0, len(etas_np), batch_size):
        batch_np = etas_np[k:k + batch_size]
        m_batch = np.asarray(batched(jnp.asarray(batch_np)))
        for eta_vals, m in zip(batch_np, m_batch):
            th1, th2 = eta_vals
            rows.append({
                "theta1_rad": float(th1),
                "theta2_rad": float(th2),
                "theta1_deg": float(np.degrees(th1)),
                "theta2_deg": float(np.degrees(th2)),
                **metrics_dict(m),
            })
        print(f"  evaluated {min(k + len(batch_np), len(etas_np))}/{len(etas_np)} designs", flush=True)
    return rows


def choose_dense_designs(rows: List[Dict[str, Any]], tau: float) -> Dict[str, Dict[str, Any]]:
    lift_row = min(rows, key=lambda r: r["lift_mmd2"])
    info_row = max(rows, key=lambda r: r["info_score"])
    lift_star = lift_row["lift_mmd2"]
    lift_max = (1.0 + tau) * lift_star
    feasible = [r for r in rows if r["lift_mmd2"] <= lift_max]
    if not feasible:
        raise RuntimeError("dense grid contains no Lift-feasible design; increase dense resolution")
    tangent_row = min(feasible, key=lambda r: r["tangent_action"])
    full_row = min(feasible, key=lambda r: r["full_action"])
    action_only = min(rows, key=lambda r: r["full_action"])
    return {
        "lift": dict(lift_row),
        "info": dict(info_row),
        "tangent_tc": dict(tangent_row),
        "full_tc": dict(full_row),
        "action_only": dict(action_only),
        "lift_star": {"value": lift_star},
        "lift_max": {"value": lift_max},
        "feasible_count": {"value": len(feasible)},
    }


def tau_sweep(rows: List[Dict[str, Any]], taus=(0.0, 0.025, 0.05, 0.10)) -> List[Dict[str, Any]]:
    lift_star = min(r["lift_mmd2"] for r in rows)
    out = []
    for tau in taus:
        bound = (1.0 + tau) * lift_star
        feasible = [r for r in rows if r["lift_mmd2"] <= bound + 1e-15]
        if not feasible:
            out.append({"tau": tau, "feasible_count": 0})
            continue
        f = min(feasible, key=lambda r: r["full_action"])
        t = min(feasible, key=lambda r: r["tangent_action"])
        out.append({
            "tau": float(tau),
            "lift_bound": float(bound),
            "feasible_count": len(feasible),
            "full_tc": {
                "theta_deg": [f["theta1_deg"], f["theta2_deg"]],
                "lift_mmd2": f["lift_mmd2"],
                "full_action": f["full_action"],
                "tangent_action": f["tangent_action"],
                "hidden_action": f["hidden_action"],
            },
            "tangent_tc": {
                "theta_deg": [t["theta1_deg"], t["theta2_deg"]],
                "lift_mmd2": t["lift_mmd2"],
                "full_action": t["full_action"],
                "tangent_action": t["tangent_action"],
                "hidden_action": t["hidden_action"],
            },
        })
    return out


def adam_optimize_full_tc(model: StageB, lift_max: float) -> Dict[str, Any]:
    cfg = model.cfg
    starts = np.array([
        [0.12 * PI, 0.58 * PI],
        [0.22 * PI, 0.78 * PI],
        [0.05 * PI, 0.45 * PI],
        [0.36 * PI, 0.88 * PI],
        [0.15 * PI, 0.70 * PI],
        [0.30 * PI, 0.62 * PI],
    ], dtype=np.float64)[: cfg.optimizer_starts]

    loss_and_grad = jax.jit(jax.value_and_grad(lambda e: model.full_tc_penalized(e, lift_max)))

    def one_start(x0: np.ndarray) -> Dict[str, Any]:
        x = jnp.asarray(x0)
        m = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        b1, b2 = 0.9, 0.999
        for step in range(1, cfg.optimizer_steps + 1):
            val, g = loss_and_grad(x)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            mh = m / (1.0 - b1 ** step)
            vh = v / (1.0 - b2 ** step)
            x = x - cfg.optimizer_lr * mh / (jnp.sqrt(vh) + 1e-8)
            x = jnp.mod(x, jnp.pi)
        eta = canonical_eta_np(np.asarray(x))
        metrics = np.asarray(model.design_metrics_jit(jnp.asarray(eta)))
        return {
            "theta_rad": eta.tolist(),
            "theta_deg": np.degrees(eta).tolist(),
            "penalized_objective": float(model.full_tc_penalized(jnp.asarray(eta), lift_max)),
            **metrics_dict(metrics),
            "feasible": bool(metrics[0] <= lift_max * (1.0 + 2e-4)),
        }

    candidates = [one_start(x0) for x0 in starts]
    feasible = [r for r in candidates if r["feasible"]]
    best = min(feasible if feasible else candidates, key=lambda r: r["penalized_objective"])
    return {"best": best, "all_starts": candidates}


# -----------------------------------------------------------------------------
# Validation and reporting
# -----------------------------------------------------------------------------

def run_gradient_checks(model: StageB, lift_max: float) -> Dict[str, Any]:
    cfg = model.cfg
    eta0 = jnp.asarray([0.21 * PI, 0.69 * PI], dtype=jnp.float64)
    eps = cfg.fd_eps

    # The expensive full-law derivative is checked at one representative
    # scenario/time point; this still differentiates through sensor physics,
    # nonlinear calibration, lambda_dot, forcing, and the Poisson linear solve.
    probe_t = jnp.array(0.47, dtype=jnp.float64)
    probe_alpha = jnp.array(0.5 * (model.cfg.alpha_min + model.cfg.alpha_max), dtype=jnp.float64)

    def full_probe(e):
        st = model.msfi_forcing(probe_t, probe_alpha, e)
        return model.poisson_solve(st["q"], st["h"])[0]

    full_fun = jax.jit(full_probe)
    lift_fun = jax.jit(lambda e: model.design_lift_tangent(e)[0])
    tangent_fun = jax.jit(lambda e: model.design_lift_tangent(e)[1])

    out: Dict[str, Any] = {"probe_theta_deg": np.degrees(np.asarray(eta0)).tolist()}

    for name, fun in [
        ("lift", lift_fun),
        ("full_action", full_fun),
        ("tangent_action", tangent_fun),
    ]:
        ad = np.asarray(jax.grad(fun)(eta0))
        fd = np.zeros(2, dtype=np.float64)
        for k in range(2):
            d = np.zeros(2, dtype=np.float64)
            d[k] = eps
            fp = float(fun(eta0 + jnp.asarray(d)))
            fm = float(fun(eta0 - jnp.asarray(d)))
            fd[k] = (fp - fm) / (2.0 * eps)
        denom = max(np.linalg.norm(ad), np.linalg.norm(fd), 1e-10)
        rel = float(np.linalg.norm(ad - fd) / denom)
        out[name] = {
            "ad": ad.tolist(),
            "fd": fd.tolist(),
            "relative_error": rel,
        }

    # Independent analytic-vs-grid measurement check.
    out["measurement_grid_abs_error"] = model.measurement_grid_error(
        eta0, t=0.43, alpha=0.41 * PI
    )

    probe_metrics = np.asarray(model.design_metrics_jit(eta0))
    out["probe_metrics"] = metrics_dict(probe_metrics)
    return out


def selected_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "theta1_deg", "theta2_deg", "lift_mmd2", "full_action", "tangent_action",
        "hidden_action", "info_score", "max_poisson_rel_resid", "max_calibration_resid",
        "min_calibration_cov_eig", "min_tangent_gram_eig", "min_pointwise_hidden_action",
    ]
    return {k: row[k] for k in keys if k in row}


def print_key_results(
    model: StageB,
    selected: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    grad_checks: Dict[str, Any],
    opt: Dict[str, Any] | None,
):
    lift = selected["lift"]
    full = selected["full_tc"]
    tang = selected["tangent_tc"]
    info = selected["info"]
    lift_max = selected["lift_max"]["value"]

    headroom = 1.0 - full["full_action"] / lift["full_action"]
    tangent_to_full_gain = 1.0 - full["full_action"] / tang["full_action"]
    lift_sacrifice = full["lift_mmd2"] / lift["lift_mmd2"] - 1.0
    eta_full = np.array([full["theta1_rad"], full["theta2_rad"]])
    eta_tang = np.array([tang["theta1_rad"], tang["theta2_rad"]])
    basin_sep = unordered_design_distance(eta_full, eta_tang)
    hidden_neg_count = sum(r["hidden_action"] < -1e-6 for r in rows)
    point_hidden_neg_count = sum(r["min_pointwise_hidden_action"] < -1e-5 for r in rows)

    print("\n" + "=" * 78)
    print("STAGE B KEY RESULTS")
    print("=" * 78)
    print(f"Lift optimum              : ({lift['theta1_deg']:.2f}°, {lift['theta2_deg']:.2f}°)")
    print(f"Info-Population optimum   : ({info['theta1_deg']:.2f}°, {info['theta2_deg']:.2f}°)")
    print(f"Tangent-TC optimum        : ({tang['theta1_deg']:.2f}°, {tang['theta2_deg']:.2f}°)")
    print(f"Full-transport TC optimum : ({full['theta1_deg']:.2f}°, {full['theta2_deg']:.2f}°)")
    print()
    print(f"Lift* MMD^2               : {lift['lift_mmd2']:.8e}")
    print(f"5% Lift bound             : {lift_max:.8e}")
    print(f"Full-TC Lift MMD^2        : {full['lift_mmd2']:.8e}  ({100*lift_sacrifice:+.2f}% vs Lift*)")
    print(f"Lift-design full action   : {lift['full_action']:.8e}")
    print(f"Full-TC full action       : {full['full_action']:.8e}")
    print(f"Transport headroom used   : {100*headroom:.2f}% action reduction vs Lift design")
    print()
    print(f"Tangent-TC tangent action : {tang['tangent_action']:.8e}")
    print(f"Tangent-TC full action    : {tang['full_action']:.8e}")
    print(f"Full-TC tangent action    : {full['tangent_action']:.8e}")
    print(f"Full-TC full action       : {full['full_action']:.8e}")
    print(f"Full vs tangent-TC gain   : {100*tangent_to_full_gain:.2f}% lower full action")
    print(f"TC design-basin distance  : {math.degrees(basin_sep):.2f}° (permutation/pi invariant)")
    print(f"Full-TC hidden action     : {full['hidden_action']:.8e}")
    print()
    print("Numerical diagnostics")
    print(f"  max calibration residual at Full-TC : {full['max_calibration_resid']:.3e}")
    print(f"  max Poisson relative residual       : {full['max_poisson_rel_resid']:.3e}")
    print(f"  min calibration covariance eig      : {full['min_calibration_cov_eig']:.3e}")
    print(f"  min tangent Gram eig                 : {full['min_tangent_gram_eig']:.3e}")
    print(f"  dense designs with avg hidden<0      : {hidden_neg_count}/{len(rows)}")
    print(f"  dense designs with point hidden<0    : {point_hidden_neg_count}/{len(rows)}")
    print(f"  measurement analytic-vs-grid error   : {grad_checks['measurement_grid_abs_error']:.3e}")
    print("  gradient relative errors (AD vs FD):")
    for k in ["lift", "full_action", "tangent_action"]:
        print(f"    {k:16s}: {grad_checks[k]['relative_error']:.3e}")

    if opt is not None:
        best = opt["best"]
        oracle_eta = np.array([full["theta1_rad"], full["theta2_rad"]])
        opt_eta = np.radians(np.asarray(best["theta_deg"], dtype=np.float64))
        d = unordered_design_distance(opt_eta, oracle_eta)
        print("\nDifferentiable optimizer")
        print(f"  optimizer Full-TC        : ({best['theta_deg'][0]:.2f}°, {best['theta_deg'][1]:.2f}°)")
        print(f"  optimizer feasible       : {best['feasible']}")
        print(f"  distance to dense oracle : {math.degrees(d):.2f}°")
        print(f"  optimizer full action    : {best['full_action']:.8e}")
        print(f"  optimizer Lift MMD^2     : {best['lift_mmd2']:.8e}")

    print("=" * 78)


def checks_from_results(
    model: StageB,
    selected: Dict[str, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    grad_checks: Dict[str, Any],
    opt: Dict[str, Any] | None,
) -> Dict[str, bool]:
    full = selected["full_tc"]
    lift_max = selected["lift_max"]["value"]
    checks = {
        "full_tc_is_lift_feasible": bool(full["lift_mmd2"] <= lift_max * (1.0 + 1e-10)),
        "calibration_residual_small": bool(full["max_calibration_resid"] < 2e-6),
        "poisson_residual_small": bool(full["max_poisson_rel_resid"] < 3e-4),
        "forcing_mean_small": bool(full["max_abs_mean_h"] < 2e-7),
        "measurement_grid_error_small": bool(grad_checks["measurement_grid_abs_error"] < 2e-4),
        "lift_gradient_fd": bool(grad_checks["lift"]["relative_error"] < 2e-2),
        "full_action_gradient_fd": bool(grad_checks["full_action"]["relative_error"] < 5e-2),
        "tangent_gradient_fd": bool(grad_checks["tangent_action"]["relative_error"] < 2e-2),
        "average_full_action_not_below_tangent": bool(
            min(r["hidden_action"] for r in rows) > -5e-4
        ),
    }
    if opt is not None:
        checks["optimizer_feasible"] = bool(opt["best"]["feasible"])
    return checks


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "reference"], default="quick")
    parser.add_argument("--output", default="stage_b_results.json")
    parser.add_argument("--optimizer", action="store_true", help="also run differentiable Full-TC Adam optimization")
    args = parser.parse_args()

    cfg = preset_config(args.preset)
    if args.optimizer:
        cfg = dataclasses.replace(cfg, run_optimizer=True)

    t0 = time.time()
    print("Stage B: differentiable transport-conditioned design on nonlinear moment fibers")
    print(f"preset={args.preset}, grid={cfg.grid_n}x{cfg.grid_n}, time_n={cfg.time_n}, "
          f"alpha_n={cfg.alpha_n}, dense_angle_n={cfg.dense_angle_n}")
    print(f"JAX backend={jax.default_backend()}, devices={jax.devices()}, x64={jax.config.jax_enable_x64}")

    model = StageB(cfg)

    # Force one compilation early and print a probe so failures are immediate.
    probe_eta = jnp.asarray([0.20 * PI, 0.68 * PI])
    print("\nCompiling/evaluating probe design ...", flush=True)
    probe = np.asarray(model.design_metrics_jit(probe_eta))
    print("Probe metrics:")
    for k, v in metrics_dict(probe).items():
        print(f"  {k:30s} {v:.8e}")

    rows = dense_scan(model)
    selected = choose_dense_designs(rows, cfg.lift_tau)
    sweep = tau_sweep(rows)

    print("\nRunning end-to-end gradient checks ...", flush=True)
    grad_checks = run_gradient_checks(model, selected["lift_max"]["value"])

    opt = None
    if cfg.run_optimizer:
        print("\nRunning differentiable Full-TC optimizer ...", flush=True)
        opt = adam_optimize_full_tc(model, selected["lift_max"]["value"])

    checks = checks_from_results(model, selected, rows, grad_checks, opt)
    all_checks_pass = all(checks.values())

    lift = selected["lift"]
    full = selected["full_tc"]
    tang = selected["tangent_tc"]
    derived = {
        "transport_headroom_fraction": 1.0 - full["full_action"] / lift["full_action"],
        "full_tc_lift_sacrifice_fraction": full["lift_mmd2"] / lift["lift_mmd2"] - 1.0,
        "full_tc_vs_tangent_tc_full_action_reduction_fraction": 1.0 - full["full_action"] / tang["full_action"],
        "full_vs_tangent_design_distance_deg": math.degrees(unordered_design_distance(
            np.array([full["theta1_rad"], full["theta2_rad"]]),
            np.array([tang["theta1_rad"], tang["theta2_rad"]]),
        )),
        "dense_hidden_action_negative_count": int(sum(r["hidden_action"] < -1e-6 for r in rows)),
        "dense_pointwise_hidden_negative_count": int(sum(r["min_pointwise_hidden_action"] < -1e-5 for r in rows)),
    }

    result = {
        "stage": "B",
        "preset": args.preset,
        "purpose": "nonlinear moment-fiber design: compare local tangent conditioning with full weighted-Poisson law-path conditioning under a matched Lift constraint",
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
            "x64": bool(jax.config.jax_enable_x64),
        },
        "config": jsonify(cfg),
        "metric_definitions": {
            "lift_mmd2": "time/scenario averaged exact discrete Gaussian-kernel MMD^2 between the I-projected law and external scientific law on the common grid",
            "full_action": "time/scenario averaged weighted-Poisson minimum correction action using a mildly floored elliptic operator",
            "tangent_action": "time/scenario averaged local kinetic moment-rate minimum action r^T G^{-1} r",
            "hidden_action": "full_action - tangent_action; expected nonnegative in the continuum, with small discretization error allowed",
            "info_score": "expected squared derivative of exact population measurements with respect to latent interior orientation alpha",
        },
        "selected_dense_designs": {k: selected_summary(v) if isinstance(v, dict) and "theta1_deg" in v else v for k, v in selected.items()},
        "derived": derived,
        "tau_sweep": sweep,
        "gradient_checks": grad_checks,
        "optimizer": opt,
        "checks": checks,
        "all_checks_pass": all_checks_pass,
        "dense_landscape": rows,
        "runtime_seconds": time.time() - t0,
    }

    print_key_results(model, selected, rows, grad_checks, opt)
    print("Declared checks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
    print(f"all_checks_pass = {all_checks_pass}")

    outpath = Path(args.output)
    outpath.write_text(json.dumps(jsonify(result), indent=2), encoding="utf-8")
    print(f"\nSaved full results to: {outpath.resolve()}")
    print(f"Runtime: {result['runtime_seconds']:.1f} s")


if __name__ == "__main__":
    main()
