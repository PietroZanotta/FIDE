#!/usr/bin/env python3
"""
Stage B.2: constrained and fine-resolution nonlinear transport-conditioned experimental design on moment fibers.

Single-file JAX experiment. It implements:
  * a symmetric two-lobe scientific population family P_t^alpha,
  * two localized Gaussian sensors eta=(theta1, theta2),
  * an analytic anisotropic stochastic-interpolant reference q_ref,t and velocity u_t,
  * nonlinear exponential-family I-projection onto the two measured moments,
  * implicit differentiation through the 2x2 multiplier solve,
  * the MFSI weighted-Poisson minimum-energy correction on a 2D grid,
  * a local kinetic-tangent comparator,
  * Gaussian-kernel MMD^2 as pointwise law-reconstruction (Lift) loss,
  * a dense two-angle design oracle with a hard projective sensor-separation constraint,
  * end-to-end finite-difference gradient checks,
  * a Lift-tolerance Pareto sweep,
  * true constrained SLSQP optimization with JAX objective/constraint gradients,
  * fine-resolution continuous re-optimization and a local basin oracle,
  * an optional fixed-design spatial/time discretization convergence study,
  * clear terminal summaries and a single JSON result file.

Recommended quick smoke test:
    python stage_b2_transport_conditioned_design.py --preset quick --skip-local-scan

Main Stage B.2 run:
    python stage_b2_transport_conditioned_design.py --preset reference --convergence \
        --output stage_b2_results_reference.json

Stage B.2 always uses a true constrained SLSQP outer solve; there is no soft-penalty
optimizer in the final comparison.

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

try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError("Stage B.2 requires scipy for the constrained SLSQP optimizer. Install it with `pip install scipy`.") from exc

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
    min_sep_deg: float = 20.0
    tc_penalty: float = 2.0e3
    sep_penalty: float = 2.0e4
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
            dense_angle_n=37,
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

    angular_spacing_deg = 180.0 / cfg.dense_angle_n
    print(
        f"\nDense scan: {cfg.dense_angle_n} angular nodes on [0, pi), "
        f"spacing={angular_spacing_deg:.2f}°, hard min separation={cfg.min_sep_deg:.2f}°, "
        f"{len(etas_np)} permutation-reduced admissible designs"
    )
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
                "sensor_separation_deg": float(np.degrees(angle_periodic_distance(float(th1), float(th2)))),
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
    """Matched-Lift Pareto sweep using the same admissible dense design set."""
    lift_row = min(rows, key=lambda r: r["lift_mmd2"])
    lift_star = lift_row["lift_mmd2"]
    lift_full_action = lift_row["full_action"]
    out = []
    for tau in taus:
        bound = (1.0 + tau) * lift_star
        feasible = [r for r in rows if r["lift_mmd2"] <= bound + 1e-15]
        if not feasible:
            out.append({"tau": float(tau), "feasible_count": 0})
            continue
        f = min(feasible, key=lambda r: r["full_action"])
        t = min(feasible, key=lambda r: r["tangent_action"])
        f_eta = np.array([f["theta1_rad"], f["theta2_rad"]], dtype=np.float64)
        t_eta = np.array([t["theta1_rad"], t["theta2_rad"]], dtype=np.float64)
        out.append({
            "tau": float(tau),
            "lift_bound": float(bound),
            "feasible_count": len(feasible),
            "lift_design_full_action": float(lift_full_action),
            "transport_headroom_fraction_vs_lift": float(1.0 - f["full_action"] / lift_full_action),
            "full_tc_lift_sacrifice_fraction": float(f["lift_mmd2"] / lift_star - 1.0),
            "full_tc_vs_tangent_tc_full_action_reduction_fraction": float(
                1.0 - f["full_action"] / t["full_action"]
            ),
            "full_vs_tangent_design_distance_deg": float(
                math.degrees(unordered_design_distance(f_eta, t_eta))
            ),
            "full_tc": {
                "theta_deg": [f["theta1_deg"], f["theta2_deg"]],
                "sensor_separation_deg": f["sensor_separation_deg"],
                "lift_mmd2": f["lift_mmd2"],
                "full_action": f["full_action"],
                "tangent_action": f["tangent_action"],
                "hidden_action": f["hidden_action"],
            },
            "tangent_tc": {
                "theta_deg": [t["theta1_deg"], t["theta2_deg"]],
                "sensor_separation_deg": t["sensor_separation_deg"],
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
        sep_deg = math.degrees(angle_periodic_distance(float(eta[0]), float(eta[1])))
        return {
            "theta_rad": eta.tolist(),
            "theta_deg": np.degrees(eta).tolist(),
            "sensor_separation_deg": float(sep_deg),
            "penalized_objective": float(model.full_tc_penalized(jnp.asarray(eta), lift_max)),
            **metrics_dict(metrics),
            "feasible": bool(
                metrics[0] <= lift_max * (1.0 + 2e-4)
                and sep_deg >= cfg.min_sep_deg - 1e-6
            ),
        }

    candidates = [one_start(x0) for x0 in starts]
    feasible = [r for r in candidates if r["feasible"]]
    best = min(feasible if feasible else candidates, key=lambda r: r["penalized_objective"])
    return {"best": best, "all_starts": candidates}



def fixed_design_convergence_study(
    base_model: StageB,
    selected: Dict[str, Dict[str, Any]],
    resolutions=((31, 11), (39, 15), (51, 21)),
) -> Dict[str, Any]:
    """
    Re-evaluate a small set of frozen designs as spatial/time resolution changes.

    This is intentionally *not* a new design optimization at each resolution.
    It tests whether the reported Lift/tangent/full actions for scientifically
    relevant fixed designs are discretization-stable.
    """
    base_cfg = base_model.cfg
    design_map = {
        "lift": np.array(
            [selected["lift"]["theta1_rad"], selected["lift"]["theta2_rad"]],
            dtype=np.float64,
        ),
        "tangent_tc": np.array(
            [selected["tangent_tc"]["theta1_rad"], selected["tangent_tc"]["theta2_rad"]],
            dtype=np.float64,
        ),
        "full_tc": np.array(
            [selected["full_tc"]["theta1_rad"], selected["full_tc"]["theta2_rad"]],
            dtype=np.float64,
        ),
        "generic_probe": np.array([0.21 * PI, 0.69 * PI], dtype=np.float64),
    }

    print("\nFixed-design discretization convergence study")
    print("  designs are frozen; only spatial/time resolution changes")
    by_resolution: List[Dict[str, Any]] = []

    for grid_n, time_n in resolutions:
        if grid_n == base_cfg.grid_n and time_n == base_cfg.time_n:
            model = base_model
        else:
            cfg = dataclasses.replace(
                base_cfg,
                grid_n=int(grid_n),
                time_n=int(time_n),
                dense_angle_n=base_cfg.dense_angle_n,
                run_optimizer=False,
            )
            model = StageB(cfg)

        cache: Dict[Tuple[float, float], Dict[str, float]] = {}
        design_results: Dict[str, Any] = {}
        for label, eta in design_map.items():
            eta = canonical_eta_np(eta)
            key = tuple(np.round(eta, 13))
            if key not in cache:
                vals = np.asarray(model.design_metrics_jit(jnp.asarray(eta, dtype=jnp.float64)))
                cache[key] = metrics_dict(vals)
            design_results[label] = {
                "theta_deg": np.degrees(eta).tolist(),
                "sensor_separation_deg": math.degrees(
                    angle_periodic_distance(float(eta[0]), float(eta[1]))
                ),
                **cache[key],
            }

        by_resolution.append({
            "grid_n": int(grid_n),
            "time_n": int(time_n),
            "alpha_n": int(base_cfg.alpha_n),
            "designs": design_results,
        })
        print(f"  completed grid={grid_n}x{grid_n}, time_n={time_n}", flush=True)

    # Compare every resolution with the highest-resolution result.
    reference = by_resolution[-1]
    relative_to_finest: Dict[str, Any] = {}
    for label in design_map:
        ref = reference["designs"][label]
        entries = []
        for row in by_resolution:
            cur = row["designs"][label]
            item = {
                "grid_n": row["grid_n"],
                "time_n": row["time_n"],
            }
            for metric in ("lift_mmd2", "full_action", "tangent_action", "hidden_action"):
                denom = max(abs(ref[metric]), 1e-12)
                item[f"{metric}_relative_error_vs_finest"] = abs(cur[metric] - ref[metric]) / denom
            entries.append(item)
        relative_to_finest[label] = entries

    return {
        "resolutions": by_resolution,
        "relative_errors_vs_finest": relative_to_finest,
        "finest_resolution": {
            "grid_n": reference["grid_n"],
            "time_n": reference["time_n"],
        },
    }


def print_convergence_summary(conv: Dict[str, Any]) -> None:
    finest = conv["finest_resolution"]
    print("\nDiscretization convergence summary")
    print(f"  finest reference: grid={finest['grid_n']}x{finest['grid_n']}, time_n={finest['time_n']}")
    for label, rows in conv["relative_errors_vs_finest"].items():
        medium = rows[-2] if len(rows) >= 2 else rows[-1]
        print(
            f"  {label:12s} medium->finest relative change: "
            f"Lift={medium['lift_mmd2_relative_error_vs_finest']:.3e}, "
            f"Full={medium['full_action_relative_error_vs_finest']:.3e}, "
            f"Tangent={medium['tangent_action_relative_error_vs_finest']:.3e}"
        )



# -----------------------------------------------------------------------------
# Stage B.2: true constrained optimization + fine-resolution confirmation
# -----------------------------------------------------------------------------

def design_row(model: StageB, eta: np.ndarray) -> Dict[str, Any]:
    eta = canonical_eta_np(np.asarray(eta, dtype=np.float64))
    vals = np.asarray(model.design_metrics_jit(jnp.asarray(eta, dtype=jnp.float64)))
    return {
        "theta1_rad": float(eta[0]),
        "theta2_rad": float(eta[1]),
        "theta1_deg": float(np.degrees(eta[0])),
        "theta2_deg": float(np.degrees(eta[1])),
        "sensor_separation_deg": float(np.degrees(angle_periodic_distance(float(eta[0]), float(eta[1])))),
        **metrics_dict(vals),
    }


def _unique_starts(starts: List[np.ndarray], min_sep_deg: float, max_count: int = 12) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    seen = set()
    min_sep = math.radians(min_sep_deg)
    for x in starts:
        eta = canonical_eta_np(np.asarray(x, dtype=np.float64))
        if angle_periodic_distance(float(eta[0]), float(eta[1])) < min_sep - 1e-12:
            continue
        key = tuple(np.round(eta, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append(eta)
        if len(out) >= max_count:
            break
    return out


def starts_from_dense(
    rows: List[Dict[str, Any]],
    selected: Dict[str, Dict[str, Any]],
    lift_max: float | None,
    max_count: int,
    min_sep_deg: float,
) -> List[np.ndarray]:
    seeds: List[np.ndarray] = []
    for key in ("lift", "full_tc", "tangent_tc", "action_only", "info"):
        r = selected[key]
        seeds.append(np.array([r["theta1_rad"], r["theta2_rad"]], dtype=np.float64))

    if lift_max is None:
        ranked = sorted(rows, key=lambda r: r["lift_mmd2"])
    else:
        feasible = [r for r in rows if r["lift_mmd2"] <= lift_max * (1.0 + 1e-12)]
        ranked = sorted(feasible, key=lambda r: r["full_action"])
        ranked += sorted(feasible, key=lambda r: r["tangent_action"])

    for r in ranked:
        seeds.append(np.array([r["theta1_rad"], r["theta2_rad"]], dtype=np.float64))
    return _unique_starts(seeds, min_sep_deg, max_count=max_count)


def constrained_slsqp(
    model: StageB,
    objective: str,
    starts: List[np.ndarray],
    lift_max: float | None,
    maxiter: int = 80,
    ftol: float = 1e-8,
) -> Dict[str, Any]:
    """True constrained continuous optimization with analytic JAX gradients.

    The canonical domain is 0 <= theta1 < theta2 < pi.  Projective sensor
    separation >= delta is equivalent to

        theta2 - theta1 >= delta,
        theta2 - theta1 <= pi - delta.

    Hence the separation constraints are linear and smooth.  For tangent/full
    optimization the Lift constraint is imposed directly by SLSQP; no penalty
    approximation is used.
    """
    min_sep = math.radians(model.cfg.min_sep_deg)
    upper = np.nextafter(PI, 0.0)

    lift_fun_j = jax.jit(lambda e: model.design_lift_tangent(e)[0])
    lift_grad_j = jax.jit(jax.grad(lambda e: model.design_lift_tangent(e)[0]))

    if objective == "lift":
        obj_j = lift_fun_j
        grad_j = lift_grad_j
        metric_key = "lift_mmd2"
    elif objective == "tangent":
        obj_j = jax.jit(lambda e: model.design_lift_tangent(e)[1])
        grad_j = jax.jit(jax.grad(lambda e: model.design_lift_tangent(e)[1]))
        metric_key = "tangent_action"
    elif objective == "full":
        obj_j = jax.jit(lambda e: model.design_lift_full(e)[1])
        grad_j = jax.jit(jax.grad(lambda e: model.design_lift_full(e)[1]))
        metric_key = "full_action"
    else:
        raise ValueError(f"unknown objective {objective!r}")

    def obj_np(x):
        return float(obj_j(jnp.asarray(x, dtype=jnp.float64)))

    def grad_np(x):
        return np.asarray(grad_j(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: float(x[1] - x[0] - min_sep),
            "jac": lambda x: np.array([-1.0, 1.0], dtype=np.float64),
        },
        {
            "type": "ineq",
            "fun": lambda x: float(PI - min_sep - (x[1] - x[0])),
            "jac": lambda x: np.array([1.0, -1.0], dtype=np.float64),
        },
    ]
    if lift_max is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda x: float(lift_max - lift_fun_j(jnp.asarray(x, dtype=jnp.float64))),
            "jac": lambda x: -np.asarray(lift_grad_j(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64),
        })

    candidates: List[Dict[str, Any]] = []
    starts = _unique_starts(starts, model.cfg.min_sep_deg, max_count=max(1, len(starts)))
    for i, x0 in enumerate(starts):
        res = minimize(
            obj_np,
            x0=np.asarray(x0, dtype=np.float64),
            jac=grad_np,
            method="SLSQP",
            bounds=[(0.0, upper), (0.0, upper)],
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False},
        )
        eta = canonical_eta_np(np.asarray(res.x, dtype=np.float64))
        row = design_row(model, eta)
        sep_ok = row["sensor_separation_deg"] >= model.cfg.min_sep_deg - 2e-5
        lift_ok = True if lift_max is None else row["lift_mmd2"] <= lift_max * (1.0 + 2e-6)
        candidates.append({
            "start_index": int(i),
            "start_theta_deg": np.degrees(np.asarray(x0)).tolist(),
            "scipy_success": bool(res.success),
            "scipy_status": int(res.status),
            "scipy_message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
            "nfev": int(getattr(res, "nfev", -1)),
            "njev": int(getattr(res, "njev", -1)),
            "objective_name": objective,
            "objective_value": float(row[metric_key]),
            "feasible": bool(sep_ok and lift_ok),
            **row,
        })

    feasible = [c for c in candidates if c["feasible"]]
    pool = feasible if feasible else candidates
    best = min(pool, key=lambda c: c[metric_key])
    return {
        "objective": objective,
        "lift_max": None if lift_max is None else float(lift_max),
        "best": best,
        "all_starts": candidates,
        "has_feasible_candidate": bool(feasible),
        "has_successful_feasible_candidate": bool(any(c["feasible"] and c["scipy_success"] for c in candidates)),
    }


def make_fine_model(base_cfg: Config, grid_n: int, time_n: int) -> StageB:
    cfg = dataclasses.replace(
        base_cfg,
        grid_n=int(grid_n),
        time_n=int(time_n),
        run_optimizer=False,
    )
    return StageB(cfg)


def fine_local_scan(
    model: StageB,
    centers: List[np.ndarray],
    lift_max: float,
    radius_deg: float = 7.5,
    points: int = 5,
) -> Dict[str, Any]:
    """Fine-resolution local oracle around the Lift/Tangent/Full basins."""
    if points < 3 or points % 2 == 0:
        raise ValueError("local scan --local-points must be an odd integer >= 3")
    offsets = np.linspace(-math.radians(radius_deg), math.radians(radius_deg), points)
    min_sep = math.radians(model.cfg.min_sep_deg)
    candidates: List[np.ndarray] = []
    for c in centers:
        for da in offsets:
            for db in offsets:
                eta = canonical_eta_np(np.asarray(c, dtype=np.float64) + np.array([da, db]))
                if angle_periodic_distance(float(eta[0]), float(eta[1])) >= min_sep - 1e-12:
                    candidates.append(eta)
    candidates = _unique_starts(candidates, model.cfg.min_sep_deg, max_count=100000)

    print(
        f"\nFine local scan: {len(candidates)} unique designs, "
        f"radius=±{radius_deg:.2f}°, points/axis={points}"
    )
    rows: List[Dict[str, Any]] = []
    for i, eta in enumerate(candidates):
        rows.append(design_row(model, eta))
        if (i + 1) % 10 == 0 or i + 1 == len(candidates):
            print(f"  evaluated {i+1}/{len(candidates)} fine local designs", flush=True)

    lift = min(rows, key=lambda r: r["lift_mmd2"])
    feasible = [r for r in rows if r["lift_mmd2"] <= lift_max * (1.0 + 1e-12)]
    full = min(feasible, key=lambda r: r["full_action"]) if feasible else None
    tangent = min(feasible, key=lambda r: r["tangent_action"]) if feasible else None
    return {
        "radius_deg": float(radius_deg),
        "points_per_axis": int(points),
        "candidate_count": len(rows),
        "feasible_count": len(feasible),
        "best_lift": lift,
        "best_full_feasible": full,
        "best_tangent_feasible": tangent,
        "rows": rows,
    }


def custom_convergence_study(
    base_cfg: Config,
    design_map: Dict[str, np.ndarray],
    resolutions=((31, 11), (39, 15), (51, 21)),
    model_cache: Dict[Tuple[int, int], StageB] | None = None,
) -> Dict[str, Any]:
    print("\nB.2 fixed-design discretization convergence study")
    print("  final designs are frozen; only spatial/time resolution changes")
    out = []
    model_cache = {} if model_cache is None else dict(model_cache)
    for grid_n, time_n in resolutions:
        key = (int(grid_n), int(time_n))
        model = model_cache.get(key)
        if model is None:
            model = make_fine_model(base_cfg, grid_n, time_n)
        designs = {}
        for label, eta in design_map.items():
            designs[label] = design_row(model, eta)
        out.append({"grid_n": grid_n, "time_n": time_n, "designs": designs})
        print(f"  completed grid={grid_n}x{grid_n}, time_n={time_n}", flush=True)

    finest = out[-1]
    rel = {}
    for label in design_map:
        ref = finest["designs"][label]
        entries = []
        for row in out:
            cur = row["designs"][label]
            d = {"grid_n": row["grid_n"], "time_n": row["time_n"]}
            for metric in ("lift_mmd2", "full_action", "tangent_action", "hidden_action"):
                d[f"{metric}_relative_error_vs_finest"] = abs(cur[metric] - ref[metric]) / max(abs(ref[metric]), 1e-12)
            entries.append(d)
        rel[label] = entries
    return {
        "resolutions": out,
        "relative_errors_vs_finest": rel,
        "finest_resolution": {"grid_n": finest["grid_n"], "time_n": finest["time_n"]},
    }


def print_b2_summary(
    cfg: Config,
    dense_selected: Dict[str, Dict[str, Any]],
    medium: Dict[str, Any],
    fine: Dict[str, Any],
    local: Dict[str, Any] | None,
    worst_hidden: Dict[str, Any],
    convergence: Dict[str, Any] | None,
) -> None:
    print("\n" + "=" * 86)
    print("STAGE B.2 — CONSTRAINED + FINE-RESOLUTION KEY RESULTS")
    print("=" * 86)
    print(f"Hard projective sensor separation : >= {cfg.min_sep_deg:.2f}°")
    print(f"Primary matched-Lift tolerance    : {100*cfg.lift_tau:.2f}%")

    print("\nMedium-resolution continuous SLSQP")
    for key, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        b = medium[key]["best"]
        print(
            f"  {label:11s}: ({b['theta1_deg']:.2f}°, {b['theta2_deg']:.2f}°) | "
            f"Lift={b['lift_mmd2']:.8e} | Full={b['full_action']:.8e} | Tangent={b['tangent_action']:.8e} | "
            f"feasible={b['feasible']} | scipy_success={b['scipy_success']}"
        )

    lift = fine["lift"]["best"]
    tang = fine["tangent"]["best"]
    full = fine["full"]["best"]
    lift_star = fine["lift_star"]
    lift_max = fine["lift_max"]
    headroom = 1.0 - full["full_action"] / lift["full_action"]
    vs_tangent = 1.0 - full["full_action"] / tang["full_action"]
    sacrifice = full["lift_mmd2"] / lift_star - 1.0
    basin = math.degrees(unordered_design_distance(
        np.array([full["theta1_rad"], full["theta2_rad"]]),
        np.array([tang["theta1_rad"], tang["theta2_rad"]]),
    ))

    print("\nFINE-RESOLUTION FINAL COMPARISON")
    print(f"  fine resolution               : {fine['grid_n']}x{fine['grid_n']}, time_n={fine['time_n']}")
    print(f"  fine Lift*                    : {lift_star:.8e}")
    print(f"  fine Lift bound               : {lift_max:.8e}")
    print(f"  Lift optimum                  : ({lift['theta1_deg']:.2f}°, {lift['theta2_deg']:.2f}°)")
    print(f"  Tangent-TC optimum            : ({tang['theta1_deg']:.2f}°, {tang['theta2_deg']:.2f}°)")
    print(f"  Full-TC optimum               : ({full['theta1_deg']:.2f}°, {full['theta2_deg']:.2f}°)")
    print(f"  Full-TC Lift sacrifice        : {100*sacrifice:+.3f}%")
    print(f"  Lift-design full action       : {lift['full_action']:.8e}")
    print(f"  Tangent-TC full action        : {tang['full_action']:.8e}")
    print(f"  Full-TC full action           : {full['full_action']:.8e}")
    print(f"  Transport headroom vs Lift    : {100*headroom:+.2f}%")
    print(f"  Full-TC gain vs Tangent-TC    : {100*vs_tangent:+.2f}%")
    print(f"  Full-vs-Tangent basin distance: {basin:.2f}°")
    print(f"  Full-TC hidden action         : {full['hidden_action']:.8e}")
    print(f"  Full-TC feasible              : {full['feasible']}")

    print("\nSLSQP status at fine resolution")
    for key, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        b = fine[key]["best"]
        print(
            f"  {label:11s}: success={b['scipy_success']} feasible={b['feasible']} "
            f"nit={b['nit']} message={b['scipy_message']}"
        )

    if local is not None:
        print("\nFine local-oracle check")
        print(f"  candidates={local['candidate_count']}, Lift-feasible={local['feasible_count']}")
        if local["best_full_feasible"] is not None:
            lfull = local["best_full_feasible"]
            print(
                f"  best local feasible Full action={lfull['full_action']:.8e} at "
                f"({lfull['theta1_deg']:.2f}°, {lfull['theta2_deg']:.2f}°)"
            )
            print(f"  SLSQP/local full-action ratio={full['full_action']/lfull['full_action']:.6f}")

    print("\nWorst pointwise hidden-action diagnostic")
    print(
        f"  medium dense worst: {worst_hidden['medium']['min_pointwise_hidden_action']:.8e} at "
        f"({worst_hidden['medium']['theta1_deg']:.2f}°, {worst_hidden['medium']['theta2_deg']:.2f}°)"
    )
    print(f"  same frozen design at fine resolution: {worst_hidden['fine']['min_pointwise_hidden_action']:.8e}")

    if convergence is not None:
        print_convergence_summary(convergence)
    print("=" * 86)

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
        "theta1_deg", "theta2_deg", "sensor_separation_deg", "lift_mmd2", "full_action", "tangent_action",
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
    print("STAGE B.1 KEY RESULTS")
    print("=" * 78)
    print(f"Lift optimum              : ({lift['theta1_deg']:.2f}°, {lift['theta2_deg']:.2f}°)")
    print(f"Info-Population optimum   : ({info['theta1_deg']:.2f}°, {info['theta2_deg']:.2f}°)")
    print(f"Tangent-TC optimum        : ({tang['theta1_deg']:.2f}°, {tang['theta2_deg']:.2f}°)")
    print(f"Full-transport TC optimum : ({full['theta1_deg']:.2f}°, {full['theta2_deg']:.2f}°)")
    print(f"Hard sensor separation    : >= {model.cfg.min_sep_deg:.2f}° projective distance")
    print(f"Lift sensor separation    : {lift['sensor_separation_deg']:.2f}°")
    print(f"Tangent-TC separation     : {tang['sensor_separation_deg']:.2f}°")
    print(f"Full-TC separation        : {full['sensor_separation_deg']:.2f}°")
    print()
    print(f"Lift* MMD^2               : {lift['lift_mmd2']:.8e}")
    print(f"{100*model.cfg.lift_tau:.1f}% Lift bound          : {lift_max:.8e}")
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
    print("Matched-Lift tolerance sweep")
    for s in sweep:
        if s.get("feasible_count", 0) == 0:
            print(f"  tau={100*s['tau']:5.1f}% : no feasible dense design")
            continue
        print(
            f"  tau={100*s['tau']:5.1f}% | feasible={s['feasible_count']:4d} | "
            f"headroom={100*s['transport_headroom_fraction_vs_lift']:+7.2f}% | "
            f"Full-vs-Tangent={100*s['full_tc_vs_tangent_tc_full_action_reduction_fraction']:+7.2f}% | "
            f"basin distance={s['full_vs_tangent_design_distance_deg']:6.2f}°"
        )
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
        print(f"  optimizer separation     : {best['sensor_separation_deg']:.2f}°")

    if convergence is not None:
        print_convergence_summary(convergence)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["quick", "reference"], default="reference")
    parser.add_argument("--output", default="stage_b2_results.json")
    parser.add_argument("--dense-angle-n", type=int, default=None)
    parser.add_argument("--min-sep-deg", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--fine-grid-n", type=int, default=None, help="final spatial grid; reference default 51")
    parser.add_argument("--fine-time-n", type=int, default=None, help="final time points; reference default 21")
    parser.add_argument("--slsqp-maxiter", type=int, default=70)
    parser.add_argument("--slsqp-starts", type=int, default=6)
    parser.add_argument("--local-radius-deg", type=float, default=7.5)
    parser.add_argument("--local-points", type=int, default=5)
    parser.add_argument("--skip-local-scan", action="store_true")
    parser.add_argument("--convergence", action="store_true", help="re-evaluate final frozen designs at 31^2/11, 39^2/15, 51^2/21")
    args = parser.parse_args()

    cfg = preset_config(args.preset)
    overrides = {"run_optimizer": False}
    if args.dense_angle_n is not None:
        overrides["dense_angle_n"] = int(args.dense_angle_n)
    if args.min_sep_deg is not None:
        overrides["min_sep_deg"] = float(args.min_sep_deg)
    if args.tau is not None:
        overrides["lift_tau"] = float(args.tau)
    cfg = dataclasses.replace(cfg, **overrides)

    if cfg.dense_angle_n < 7:
        raise ValueError("--dense-angle-n must be >= 7")
    if not (0.0 < cfg.min_sep_deg < 90.0):
        raise ValueError("--min-sep-deg must lie in (0,90)")
    if cfg.lift_tau < 0.0:
        raise ValueError("--tau must be nonnegative")

    fine_grid_n = int(args.fine_grid_n or (51 if args.preset == "reference" else 31))
    fine_time_n = int(args.fine_time_n or (21 if args.preset == "reference" else 11))
    if fine_grid_n < cfg.grid_n or fine_time_n < cfg.time_n:
        raise ValueError("fine resolution must be at least as large as the base preset")

    t0 = time.time()
    print("Stage B.2: true constrained optimization + fine-resolution confirmation")
    print(
        f"base={cfg.grid_n}x{cfg.grid_n}/time{cfg.time_n}, fine={fine_grid_n}x{fine_grid_n}/time{fine_time_n}, "
        f"alpha_n={cfg.alpha_n}, dense_angle_n={cfg.dense_angle_n}, "
        f"min_sep={cfg.min_sep_deg:.1f}°, tau={100*cfg.lift_tau:.1f}%"
    )
    print(f"JAX backend={jax.default_backend()}, devices={jax.devices()}, x64={jax.config.jax_enable_x64}")

    # ------------------------------------------------------------------
    # 1. Medium-resolution dense oracle and numerical controls
    # ------------------------------------------------------------------
    model = StageB(cfg)
    probe_eta = jnp.asarray([0.20 * PI, 0.68 * PI])
    print("\nCompiling/evaluating probe design ...", flush=True)
    probe = np.asarray(model.design_metrics_jit(probe_eta))
    for k, v in metrics_dict(probe).items():
        print(f"  {k:30s} {v:.8e}")

    rows = dense_scan(model)
    dense_selected = choose_dense_designs(rows, cfg.lift_tau)
    sweep = tau_sweep(rows)

    print("\nRunning end-to-end gradient checks ...", flush=True)
    grad_checks = run_gradient_checks(model, dense_selected["lift_max"]["value"])

    # ------------------------------------------------------------------
    # 2. True constrained continuous optimization at medium resolution
    # ------------------------------------------------------------------
    print("\nRunning medium-resolution constrained SLSQP optimizers ...", flush=True)
    medium_starts = starts_from_dense(rows, dense_selected, None, max_count=args.slsqp_starts, min_sep_deg=cfg.min_sep_deg)
    med_lift = constrained_slsqp(model, "lift", medium_starts, None, maxiter=args.slsqp_maxiter)
    med_lift_star = med_lift["best"]["lift_mmd2"]
    med_lift_max = (1.0 + cfg.lift_tau) * med_lift_star

    feasible_dense_starts = starts_from_dense(rows, dense_selected, med_lift_max, max_count=args.slsqp_starts, min_sep_deg=cfg.min_sep_deg)
    medium_seed_pool = _unique_starts(
        [np.array([med_lift["best"]["theta1_rad"], med_lift["best"]["theta2_rad"]])] + feasible_dense_starts,
        cfg.min_sep_deg,
        max_count=args.slsqp_starts,
    )
    med_tangent = constrained_slsqp(model, "tangent", medium_seed_pool, med_lift_max, maxiter=args.slsqp_maxiter)
    med_full = constrained_slsqp(model, "full", medium_seed_pool, med_lift_max, maxiter=args.slsqp_maxiter)
    medium = {"lift": med_lift, "tangent": med_tangent, "full": med_full, "lift_star": med_lift_star, "lift_max": med_lift_max}

    # ------------------------------------------------------------------
    # 3. Fine-resolution Lift optimum and local basin oracle
    # ------------------------------------------------------------------
    print("\nBuilding fine-resolution model ...", flush=True)
    fine_model = make_fine_model(cfg, fine_grid_n, fine_time_n)

    fine_lift_starts = _unique_starts([
        np.array([med_lift["best"]["theta1_rad"], med_lift["best"]["theta2_rad"]]),
        np.array([dense_selected["lift"]["theta1_rad"], dense_selected["lift"]["theta2_rad"]]),
        np.array([dense_selected["full_tc"]["theta1_rad"], dense_selected["full_tc"]["theta2_rad"]]),
        np.array([dense_selected["tangent_tc"]["theta1_rad"], dense_selected["tangent_tc"]["theta2_rad"]]),
    ] + medium_starts, cfg.min_sep_deg, max_count=args.slsqp_starts)
    print("Running fine-resolution Lift SLSQP ...", flush=True)
    fine_lift = constrained_slsqp(fine_model, "lift", fine_lift_starts, None, maxiter=args.slsqp_maxiter)
    fine_lift_star = fine_lift["best"]["lift_mmd2"]
    fine_lift_max = (1.0 + cfg.lift_tau) * fine_lift_star

    centers = [
        np.array([med_lift["best"]["theta1_rad"], med_lift["best"]["theta2_rad"]]),
        np.array([med_tangent["best"]["theta1_rad"], med_tangent["best"]["theta2_rad"]]),
        np.array([med_full["best"]["theta1_rad"], med_full["best"]["theta2_rad"]]),
        np.array([dense_selected["lift"]["theta1_rad"], dense_selected["lift"]["theta2_rad"]]),
        np.array([dense_selected["tangent_tc"]["theta1_rad"], dense_selected["tangent_tc"]["theta2_rad"]]),
        np.array([dense_selected["full_tc"]["theta1_rad"], dense_selected["full_tc"]["theta2_rad"]]),
    ]

    local = None
    local_extra_starts: List[np.ndarray] = []
    if not args.skip_local_scan:
        local = fine_local_scan(
            fine_model,
            centers,
            fine_lift_max,
            radius_deg=args.local_radius_deg,
            points=args.local_points,
        )
        for key in ("best_full_feasible", "best_tangent_feasible", "best_lift"):
            r = local.get(key)
            if r is not None:
                local_extra_starts.append(np.array([r["theta1_rad"], r["theta2_rad"]], dtype=np.float64))

    # ------------------------------------------------------------------
    # 4. Fine-resolution constrained Tangent-TC and Full-TC optimizers
    # ------------------------------------------------------------------
    local_full_start = []
    local_tangent_start = []
    if local is not None and local.get("best_full_feasible") is not None:
        r = local["best_full_feasible"]
        local_full_start = [np.array([r["theta1_rad"], r["theta2_rad"]], dtype=np.float64)]
    if local is not None and local.get("best_tangent_feasible") is not None:
        r = local["best_tangent_feasible"]
        local_tangent_start = [np.array([r["theta1_rad"], r["theta2_rad"]], dtype=np.float64)]

    fine_tangent_starts = _unique_starts([
        np.array([fine_lift["best"]["theta1_rad"], fine_lift["best"]["theta2_rad"]]),
        np.array([med_tangent["best"]["theta1_rad"], med_tangent["best"]["theta2_rad"]]),
        np.array([dense_selected["tangent_tc"]["theta1_rad"], dense_selected["tangent_tc"]["theta2_rad"]]),
    ] + local_tangent_start + [
        np.array([med_full["best"]["theta1_rad"], med_full["best"]["theta2_rad"]]),
        np.array([dense_selected["full_tc"]["theta1_rad"], dense_selected["full_tc"]["theta2_rad"]]),
    ], cfg.min_sep_deg, max_count=args.slsqp_starts)

    fine_full_starts = _unique_starts([
        np.array([fine_lift["best"]["theta1_rad"], fine_lift["best"]["theta2_rad"]]),
        np.array([med_full["best"]["theta1_rad"], med_full["best"]["theta2_rad"]]),
        np.array([dense_selected["full_tc"]["theta1_rad"], dense_selected["full_tc"]["theta2_rad"]]),
    ] + local_full_start + [
        np.array([med_tangent["best"]["theta1_rad"], med_tangent["best"]["theta2_rad"]]),
        np.array([dense_selected["tangent_tc"]["theta1_rad"], dense_selected["tangent_tc"]["theta2_rad"]]),
    ], cfg.min_sep_deg, max_count=args.slsqp_starts)

    print("\nRunning fine-resolution constrained Tangent-TC SLSQP ...", flush=True)
    fine_tangent = constrained_slsqp(fine_model, "tangent", fine_tangent_starts, fine_lift_max, maxiter=args.slsqp_maxiter)
    print("Running fine-resolution constrained Full-TC SLSQP ...", flush=True)
    fine_full = constrained_slsqp(fine_model, "full", fine_full_starts, fine_lift_max, maxiter=args.slsqp_maxiter)

    fine = {
        "grid_n": fine_grid_n,
        "time_n": fine_time_n,
        "lift_star": float(fine_lift_star),
        "lift_max": float(fine_lift_max),
        "lift": fine_lift,
        "tangent": fine_tangent,
        "full": fine_full,
    }

    # ------------------------------------------------------------------
    # 5. Targeted hidden-action and convergence diagnostics
    # ------------------------------------------------------------------
    worst_medium = min(rows, key=lambda r: r["min_pointwise_hidden_action"])
    worst_eta = np.array([worst_medium["theta1_rad"], worst_medium["theta2_rad"]], dtype=np.float64)
    worst_fine = design_row(fine_model, worst_eta)
    worst_hidden = {"medium": worst_medium, "fine": worst_fine}

    convergence = None
    if args.convergence:
        conv_resolutions = []
        for pair in ((31, 11), (cfg.grid_n, cfg.time_n), (fine_grid_n, fine_time_n)):
            if pair not in conv_resolutions:
                conv_resolutions.append(pair)
        convergence = custom_convergence_study(
            cfg,
            {
                "lift": np.array([fine_lift["best"]["theta1_rad"], fine_lift["best"]["theta2_rad"]]),
                "tangent_tc": np.array([fine_tangent["best"]["theta1_rad"], fine_tangent["best"]["theta2_rad"]]),
                "full_tc": np.array([fine_full["best"]["theta1_rad"], fine_full["best"]["theta2_rad"]]),
                "worst_hidden_dense": worst_eta,
            },
            resolutions=tuple(conv_resolutions),
            model_cache={(cfg.grid_n, cfg.time_n): model, (fine_grid_n, fine_time_n): fine_model},
        )

    # ------------------------------------------------------------------
    # 6. Final checks and reporting
    # ------------------------------------------------------------------
    base_checks = checks_from_results(model, dense_selected, rows, grad_checks, None)
    fine_lift_b = fine_lift["best"]
    fine_tang_b = fine_tangent["best"]
    fine_full_b = fine_full["best"]
    final_checks = {
        **base_checks,
        "medium_lift_slsqp_feasible": bool(med_lift["best"]["feasible"]),
        "medium_tangent_slsqp_feasible": bool(med_tangent["best"]["feasible"]),
        "medium_full_slsqp_feasible": bool(med_full["best"]["feasible"]),
        "medium_lift_has_successful_feasible_start": bool(med_lift["has_successful_feasible_candidate"]),
        "medium_tangent_has_successful_feasible_start": bool(med_tangent["has_successful_feasible_candidate"]),
        "medium_full_has_successful_feasible_start": bool(med_full["has_successful_feasible_candidate"]),
        "fine_lift_slsqp_feasible": bool(fine_lift_b["feasible"]),
        "fine_tangent_slsqp_feasible": bool(fine_tang_b["feasible"]),
        "fine_full_slsqp_feasible": bool(fine_full_b["feasible"]),
        "fine_lift_has_successful_feasible_start": bool(fine_lift["has_successful_feasible_candidate"]),
        "fine_tangent_has_successful_feasible_start": bool(fine_tangent["has_successful_feasible_candidate"]),
        "fine_full_has_successful_feasible_start": bool(fine_full["has_successful_feasible_candidate"]),
        "fine_full_within_lift_bound": bool(fine_full_b["lift_mmd2"] <= fine_lift_max * (1.0 + 2e-6)),
        "fine_full_sensor_separation": bool(fine_full_b["sensor_separation_deg"] >= cfg.min_sep_deg - 2e-5),
        "worst_hidden_recheck_not_materially_negative": bool(worst_fine["min_pointwise_hidden_action"] > -1e-2),
    }
    all_checks_pass = all(final_checks.values())

    fine_headroom = 1.0 - fine_full_b["full_action"] / fine_lift_b["full_action"]
    fine_vs_tangent = 1.0 - fine_full_b["full_action"] / fine_tang_b["full_action"]
    fine_sacrifice = fine_full_b["lift_mmd2"] / fine_lift_star - 1.0
    fine_basin = math.degrees(unordered_design_distance(
        np.array([fine_full_b["theta1_rad"], fine_full_b["theta2_rad"]]),
        np.array([fine_tang_b["theta1_rad"], fine_tang_b["theta2_rad"]]),
    ))

    result = {
        "stage": "B.2",
        "purpose": "true constrained continuous optimization and fine-resolution confirmation of transport-conditioned experimental design under matched Lift quality",
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
            "x64": bool(jax.config.jax_enable_x64),
        },
        "base_config": jsonify(cfg),
        "b2_config": {
            "fine_grid_n": fine_grid_n,
            "fine_time_n": fine_time_n,
            "slsqp_maxiter": args.slsqp_maxiter,
            "slsqp_starts": args.slsqp_starts,
            "local_scan_enabled": not args.skip_local_scan,
            "local_radius_deg": args.local_radius_deg,
            "local_points": args.local_points,
            "convergence_enabled": bool(args.convergence),
        },
        "why_feasibility_matters": "The design claim compares transport action only among experiments with Lift loss <= (1+tau) times the best Lift loss. Without feasibility, lower action can be obtained by sacrificing scientific reconstruction quality, so the comparison is not matched-information.",
        "medium_dense_selected": {k: selected_summary(v) if isinstance(v, dict) and "theta1_deg" in v else v for k, v in dense_selected.items()},
        "medium_tau_sweep": sweep,
        "gradient_checks": grad_checks,
        "medium_continuous": medium,
        "fine_continuous": fine,
        "fine_local_scan": local,
        "worst_pointwise_hidden_recheck": worst_hidden,
        "convergence_study": convergence,
        "fine_derived": {
            "transport_headroom_fraction_vs_lift": float(fine_headroom),
            "full_tc_vs_tangent_tc_full_action_reduction_fraction": float(fine_vs_tangent),
            "full_tc_lift_sacrifice_fraction": float(fine_sacrifice),
            "full_vs_tangent_design_distance_deg": float(fine_basin),
        },
        "checks": final_checks,
        "all_checks_pass": bool(all_checks_pass),
        "dense_landscape": rows,
        "runtime_seconds": time.time() - t0,
    }

    print_b2_summary(cfg, dense_selected, medium, fine, local, worst_hidden, convergence)
    print("\nDeclared checks:")
    for k, v in final_checks.items():
        print(f"  {'PASS' if v else 'FAIL':4s}  {k}")
    print(f"all_checks_pass = {all_checks_pass}")

    outpath = Path(args.output)
    outpath.write_text(json.dumps(jsonify(result), indent=2), encoding="utf-8")
    print(f"\nSaved full results to: {outpath.resolve()}")
    print(f"Runtime: {result['runtime_seconds']:.1f} s")


if __name__ == "__main__":
    main()
