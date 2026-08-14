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

Recommended reference run
-------------------------
python stage_d6_endpoint_flow_matching_frozen_designs.py \
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \
    --d2-script stage_d2_flow_matching_particle_mfsi.py \
    --d5-script stage_d5_endpoint_flow_matching_reference_v2.py \
    --checkpoint stage_d5_endpoint_flow_matching_reference_v2.npz \
    --preset reference \
    --output stage_d6_endpoint_flow_matching_frozen_designs.json
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
    calibration_steps: int = 24
    calibration_tol: float = 2.0e-8
    newton_step_cap: float = 5.0
    lambda_clip: float = 80.0

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
            # Match the expanded hard-tilt budget validated in the final D4 run.
            calibration_steps=100,
            calibration_tol=1.0e-9,
            newton_step_cap=10.0,
            lambda_clip=300.0,
        )
    if name == "confirm":
        return D6Config(
            preset="confirm",
            gh_order=48,
            particles=65536,
            rk4_substeps_per_time_interval=24,
            calibration_steps=160,
            calibration_tol=5.0e-10,
            newton_step_cap=12.0,
            lambda_clip=500.0,
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

    def evaluate_design(self, eta: np.ndarray, bandwidth: float | None = None):
        bw = self._bandwidth(bandwidth)

        times = np.asarray(self.model.times, dtype=np.float64)
        alphas = np.asarray(self.model.alphas, dtype=np.float64)
        tw = np.asarray(self.model.time_w, dtype=np.float64)
        aw = np.asarray(self.model.alpha_w, dtype=np.float64)

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
        hull_violation = np.zeros_like(law)
        hull_inside = np.zeros_like(law, dtype=bool)
        hull_margin = np.zeros_like(law)
        iterations = np.zeros_like(law)
        in_fraction = np.zeros(len(times), dtype=np.float64)
        in_base_mass = np.zeros(len(times), dtype=np.float64)

        lam_warm = [np.zeros(2, dtype=np.float64) for _ in alphas]

        phi_grid, _ = self.model.sensor_fields(jnp.asarray(eta, dtype=jnp.float64))
        phi_grid_np = np.asarray(phi_grid, dtype=np.float64)

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
                        jnp.asarray(t),
                        jnp.asarray(alpha),
                        jnp.asarray(eta),
                    ),
                    dtype=np.float64,
                )
                c_dot = np.asarray(
                    self.cdot_jit(
                        jnp.asarray(t),
                        jnp.asarray(alpha),
                        jnp.asarray(eta),
                    ),
                    dtype=np.float64,
                )

                hull_diag = moment_hull_diagnostic(phi, target)
                hull_violation[ka, kt] = float(hull_diag["max_facet_violation"])
                hull_inside[ka, kt] = bool(hull_diag["inside_closed_hull"])
                hull_margin[ka, kt] = float(hull_diag["min_signed_interior_margin"])

                st = self.d2.particle_mfsi_state(
                    phi,
                    grad_phi,
                    u,
                    base_w,
                    target,
                    c_dot,
                    float(self.model.cfg.newton_ridge),
                    self.d2cfg,
                    lam_warm[ka],
                )
                lam_warm[ka] = st["lambda"]

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

                _, p_mass = self.model.external_q_mass(
                    jnp.asarray(t), jnp.asarray(alpha)
                )
                lift = self.model.gaussian_mmd2_mass(
                    jnp.asarray(qmass, dtype=jnp.float64),
                    p_mass,
                )
                grid_moment = np.sum(phi_grid_np * qmass[None, ...], axis=(1, 2))

                law[ka, kt] = float(lift)
                action[ka, kt] = float(full)
                tangent[ka, kt] = float(st["diagnostics"]["tangent_action"])
                cal_resid[ka, kt] = float(st["diagnostics"]["residual"])
                iterations[ka, kt] = float(st["diagnostics"].get("iterations", 0))
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
            "max_calibration_iterations": int(np.max(iterations)),
            "all_targets_inside_empirical_moment_hull": bool(np.all(hull_inside)),
            "outside_hull_count": int(np.size(hull_inside) - np.count_nonzero(hull_inside)),
            "max_empirical_moment_hull_facet_violation": float(np.max(hull_violation)),
            "min_empirical_moment_hull_interior_margin": float(np.min(hull_margin)),
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

    def bandwidth_check(self, eta: np.ndarray, bandwidths: Sequence[float]):
        rows = []
        for bw in bandwidths:
            print(f"  D.6 Full-TC bandwidth check h={bw:.4f}", flush=True)
            r = self.evaluate_design(eta, bandwidth=float(bw))
            rows.append({
                "kde_bandwidth": float(bw),
                "lift_mmd2": r["lift_mmd2"],
                "full_action": r["full_action"],
                "tangent_action": r["tangent_action"],
                "min_ess_fraction": r["min_ess_fraction"],
                "max_calibration_residual": r["max_calibration_residual"],
            })
        return rows


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def full_vs_lift(designs: Dict[str, Any]):
    full = designs["full"]
    lift = designs["lift"]
    return {
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
    return {
        "law_relative_penalty": float(tangent["lift_mmd2"] / lift["lift_mmd2"] - 1.0),
        "action_reduction_fraction": float(
            1.0 - tangent["full_action"] / lift["full_action"]
        ),
    }


def full_vs_tangent(designs: Dict[str, Any]):
    full = designs["full"]
    tangent = designs["tangent"]
    return {
        "law_relative_change": float(full["lift_mmd2"] / tangent["lift_mmd2"] - 1.0),
        "full_action_reduction_fraction": float(
            1.0 - full["full_action"] / tangent["full_action"]
        ),
        "tangent_action_reduction_fraction": float(
            1.0 - full["tangent_action"] / tangent["tangent_action"]
        ),
    }


def print_summary(payload: Dict[str, Any]):
    print("\n" + "=" * 98)
    print("Stage D.6 frozen-design MFSI under endpoint-trained FM reference (NO old analytic SI)")
    print("=" * 98)

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
    print("-" * 98)

    for key, label in (
        ("lift", "Lift"),
        ("tangent", "Tangent-TC"),
        ("full", "Full-TC"),
    ):
        r = payload["designs"][key]
        print(
            f"{label:10s} L={r['lift_mmd2']:.8f} | "
            f"A_full={r['full_action']:.3f} | "
            f"A_tan={r['tangent_action']:.3f} | "
            f"ESSmin={r['min_ess_fraction']:.3f} | "
            f"calmax={r['max_calibration_residual']:.2e} | "
            f"hull_out={r['outside_hull_count']} | "
            f"hull_viol={r['max_empirical_moment_hull_facet_violation']:.2e} | "
            f"|lam|max={r['max_lambda_norm']:.1f}"
        )

    c = payload["contrasts"]["full_vs_lift"]
    print("-" * 98)
    print(
        "Full-TC vs Lift: "
        f"law penalty={100.0*c['law_relative_penalty']:+.3f}% | "
        f"full-action reduction={100.0*c['action_reduction_fraction']:+.2f}%"
    )
    ct = payload["contrasts"]["full_vs_tangent"]
    print(
        "Full-TC vs Tangent-TC: "
        f"law change={100.0*ct['law_relative_change']:+.3f}% | "
        f"full-action reduction={100.0*ct['full_action_reduction_fraction']:+.2f}%"
    )
    print("-" * 98)
    for k, v in payload["checks"].items():
        print(f"{k}: {v}")
    print("=" * 98)


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
    p.add_argument("--output", type=str, default="stage_d6_endpoint_flow_matching_frozen_designs_v2.json")

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
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = cast(value)
    cfg = dataclasses.replace(cfg, **overrides)

    # D.6 uses the Stage-B backend for the scientific population/sensors/grid.
    # It deliberately never calls its analytic reference path methods.
    if args.preset == "quick":
        stage_b_cfg = backend.preset_config("quick")
    elif args.preset == "reference":
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=39, time_n=21)
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
    for name, deg in design_deg.items():
        print(f"Evaluating D.6 {name}: {deg[0]:.2f} deg, {deg[1]:.2f} deg", flush=True)
        eta = np.radians(np.asarray(deg, dtype=np.float64))
        row = evaluator.evaluate_design(eta)
        row["theta_deg"] = [float(deg[0]), float(deg[1])]
        designs[name] = row

    contrasts = {
        "full_vs_lift": full_vs_lift(designs),
        "tangent_vs_lift": tangent_vs_lift(designs),
        "full_vs_tangent": full_vs_tangent(designs),
    }

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
        for v in row.values()
        if isinstance(v, (int, float))
    )
    max_cal = max(r["max_calibration_residual"] for r in all_design_rows)
    min_ess = min(r["min_ess_fraction"] for r in all_design_rows)
    min_domain = min(r["min_in_domain_base_mass"] for r in all_design_rows)
    all_hulls_feasible = all(r["all_targets_inside_empirical_moment_hull"] for r in all_design_rows)
    max_hull_violation = max(r["max_empirical_moment_hull_facet_violation"] for r in all_design_rows)

    checks = {
        "finite_outputs": bool(finite_outputs),
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
        "empirical_calibration_residual_small": bool(
            max_cal < cfg.max_allowed_calibration_resid
        ),
        "all_population_targets_inside_empirical_moment_hulls": bool(all_hulls_feasible),
        "max_population_target_hull_facet_violation": float(max_hull_violation),
        "ess_above_warning_fraction": bool(min_ess >= cfg.ess_warn_fraction),
        "in_domain_base_mass_high": bool(min_domain >= cfg.min_in_domain_base_mass),
        "full_tc_action_below_lift": bool(
            designs["full"]["full_action"] < designs["lift"]["full_action"]
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
        "full_tc_bandwidth_check": bandwidth_check,
        "checks": checks,
        "interpretation": [
            "D.6 is the first frozen-design test after removing the old analytic path teacher in D.5.",
            "The learned ODE rollout supplies the empirical reference marginal used by the hard I-projection.",
            "The exact four-component D.5 bridge marginal is used only to validate that the learned ODE reproduces the declared endpoint-defined bridge; it is not the old Stage-B analytic SI.",
            "The MFSI forcing is computed from weighted particle statistics using C lambda_dot = c_dot - E[J Phi u] - Cov(Phi, lambda^T J Phi u).",
            "The weighted-grid Poisson solve remains the deterministic action oracle; neural Poisson learning remains deferred.",
            "Finite/noisy measurements and sensor re-optimization remain off in D.6 so the effect of changing the reference bridge is isolated.",
            "D.6-v2 explicitly checks each population target against the empirical D.5 particle moment hull; action comparisons should be interpreted only when the target is feasible and hard calibration converges.",
            "The reference/confirm presets use the expanded hard-tilt solver budget already validated during the final D4 study (more Newton steps and wider lambda range).",
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
