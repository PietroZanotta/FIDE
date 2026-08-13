#!/usr/bin/env python3
"""
Stage D.1: MFSI with a learned flow-matching reference velocity (no CNF).

Goal
----
Stage D.0 trained a neural velocity u_theta(t,x) by flow matching on the SAME
prescribed Stage-B reference probability path Q_ref,t.  Stage D.1 now asks the
controlled question:

    Does the Stage-B Full-TC vs Lift transport advantage survive when the
    reference velocity is learned rather than analytic?

Crucially, this script does NOT build a continuous-normalizing-flow density.
That is unnecessary for this experiment.  Flow matching learns a velocity field
for a chosen/sampleable probability path; the path itself is the training target.
Here the prescribed path Q_ref,t is deliberately kept identical to Stage B so
that the ONLY new approximation is u_exact -> u_theta.

MFSI target law
---------------
For each scientific scenario alpha and frozen sensor design eta, we retain the
same exact reference marginal q_ref,t and the same hard I-projected target law
q_eta,t as Stage B.  Therefore the law-reconstruction loss is intentionally
unchanged in D.1.  What changes is the reference velocity relative to which the
minimum-energy MFSI correction is computed.

Why the ordinary closed forcing formula needs one extra term
-------------------------------------------------------------
The standard Stage-B forcing assumes (q_ref, u_ref) satisfies the continuity
equation exactly.  A learned flow-matching velocity is only approximate.  If

    delta_u = u_theta - u_exact,

then the projected target density q is unchanged, but its continuity defect
relative to u_theta changes by

    div(q delta_u).

Since Stage B already gives the exact forcing h_exact satisfying

    partial_t q + div(q u_exact) = q h_exact,

we construct the learned-velocity forcing exactly at the continuum level as

    q h_theta = q h_exact + div(q delta_u).

This is the key D.1 identity.  It avoids pretending that the learned velocity
transports q_ref perfectly and avoids any learned-density/CNF approximation.
The divergence term is discretized on the existing Stage-B cell-centered grid
with conservative face fluxes and zero boundary flux, so its discrete integral
is exactly zero.

Controls
--------
For each frozen design (Lift, Tangent-TC, Full-TC), report:
  * the unchanged Stage-B projected-law MMD^2,
  * exact-reference full MFSI action,
  * learned-FM-reference full MFSI action,
  * exact and learned tangent actions,
  * learned-reference action inflation/deflation,
  * Full-vs-Lift action reduction under both velocities,
  * reference-path continuity-defect diagnostics,
  * projected-path continuity-defect diagnostics,
  * Poisson residuals and compatibility checks.

Because q_eta,t is unchanged, any D.1 action change is attributable to the
learned flow-matching velocity (plus the explicitly reported spatial
finite-difference error in the continuity-defect term), not to a new density
estimator, finite-particle I-projection, or finite measurements.

Example
-------
python stage_d1_flow_matching_mfsi.py \
    --backend ../stage_b/stage_b2_transport_conditioned_design.py \
    --d0-script stage_d0_flow_matching_reference.py \
    --checkpoint stage_d0_flow_matching_reference.npz \
    --preset reference \
    --output stage_d1_flow_matching_mfsi.json
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
from typing import Any, Dict, Mapping, Tuple

import numpy as np

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


def autodetect(names) -> Path | None:
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
class D1Config:
    preset: str = "quick"

    # Frozen Stage-B/C designs.
    lift_design_deg: Tuple[float, float] = (1.63, 161.63)
    tangent_design_deg: Tuple[float, float] = (0.0, 154.70)
    full_design_deg: Tuple[float, float] = (0.0, 160.0)

    # Numerical compatibility tolerance after constructing q h_theta.
    compatibility_tol: float = 5.0e-8


def preset_d1_config(name: str) -> D1Config:
    if name in ("quick", "reference", "confirm"):
        return D1Config(preset=name)
    raise ValueError(name)


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


def conservative_divergence(flux: Array, dx: float) -> Array:
    """
    Conservative divergence of a cell-centered vector flux on the Stage-B grid.

    Interior face fluxes are arithmetic averages of neighboring cell values;
    boundary normal flux is zero.  Hence sum(div F) * cell_area = 0 to roundoff.

    flux[y,x,coord], coord 0=x and 1=y.
    """
    fx = flux[..., 0]
    fy = flux[..., 1]

    face_x = 0.5 * (fx[:, :-1] + fx[:, 1:])
    div = jnp.zeros_like(fx)
    div = div.at[:, :-1].add(face_x / dx)
    div = div.at[:, 1:].add(-face_x / dx)

    face_y = 0.5 * (fy[:-1, :] + fy[1:, :])
    div = div.at[:-1, :].add(face_y / dx)
    div = div.at[1:, :].add(-face_y / dx)

    # The sign above is left-minus-right because Stage B's weighted_laplacian
    # is the positive operator -div(q grad).  Flip to the conventional div F.
    return -div


def weighted_relative_velocity_error(qmass: Array, du: Array, u_exact: Array) -> Array:
    num = jnp.sum(qmass * jnp.sum(du * du, axis=-1))
    den = jnp.sum(qmass * jnp.sum(u_exact * u_exact, axis=-1))
    return jnp.sqrt(num / jnp.maximum(den, 1.0e-30))


# -----------------------------------------------------------------------------
# D.1 evaluator
# -----------------------------------------------------------------------------


class LearnedVelocityMFSI:
    def __init__(self, model, d0, params, cfg: D1Config):
        self.model = model
        self.d0 = d0
        self.params = params
        self.cfg = cfg
        self.velocity_jit = jax.jit(lambda t, x: d0.velocity_mlp(params, t, x))
        self.poisson_jit = jax.jit(model.poisson_solve)
        self.cdot_jit = jax.jit(jax.jacfwd(model.measurement_grid, argnums=0))
        self.one_time_jit = jax.jit(self._one_time)

    def learned_velocity_grid(self, t: Array) -> Array:
        u = self.velocity_jit(t, self.model.xy_flat)
        return u.reshape((self.model.cfg.grid_n, self.model.cfg.grid_n, 2))

    def exact_velocity_grid(self, t: Array) -> Array:
        B = self.model.B_matrix(t)
        u = self.model.xy_flat @ B.T
        return u.reshape((self.model.cfg.grid_n, self.model.cfg.grid_n, 2))

    def _one_time(self, t: Array, alpha: Array, eta: Array) -> Array:
        # Exact Stage-B projected target path and exact-reference forcing.
        state = self.model.msfi_forcing(t, alpha, eta)
        q = state["q"]
        qmass = state["qmass"]
        h_exact = state["h"]

        u_exact = self.exact_velocity_grid(t)
        u_learned = self.learned_velocity_grid(t)
        du = u_learned - u_exact

        # Learned-velocity continuity defect for the SAME projected target law:
        # q h_theta = q h_exact + div(q delta_u).
        div_q_du = conservative_divergence(q[..., None] * du, self.model.dx)
        qh_learned = q * h_exact + div_q_du

        # Conservative div integrates to zero.  The exact Stage-B h is already
        # centered, but remove any residual constant mode from discretization.
        compat_before = jnp.sum(qh_learned) * self.model.cell_area
        qh_learned = qh_learned - q * compat_before
        compat_after = jnp.sum(qh_learned) * self.model.cell_area
        h_learned = qh_learned / jnp.maximum(q, 1.0e-300)

        learned_action, _, learned_poisson_rel, _, _ = self.poisson_jit(q, h_learned)
        exact_action, _, exact_poisson_rel, _, _ = self.poisson_jit(q, h_exact)

        # Tangent comparator under learned vs exact reference velocities.
        grad_phi = state["grad_phi"]
        jphi_exact = jnp.einsum("myxc,yxc->myx", grad_phi, u_exact)
        jphi_learned = jnp.einsum("myxc,yxc->myx", grad_phi, u_learned)
        c_dot = self.cdot_jit(t, alpha, eta)
        rate_exact = jnp.sum(jphi_exact * qmass[None, ...], axis=(1, 2))
        rate_learned = jnp.sum(jphi_learned * qmass[None, ...], axis=(1, 2))
        r_exact = rate_exact - c_dot
        r_learned = rate_learned - c_dot
        G = state["G"]
        Gs = G + self.model.cfg.newton_ridge * self.model.eye2
        tangent_exact = r_exact @ jnp.linalg.solve(Gs, r_exact)
        tangent_learned = r_learned @ jnp.linalg.solve(Gs, r_learned)

        # Law target is intentionally unchanged from Stage B.
        _, p_mass = self.model.external_q_mass(t, alpha)
        lift = self.model.gaussian_mmd2_mass(qmass, p_mass)

        # Reference-path continuity defect: exact q_ref but learned u_theta.
        qref, qref_mass = self.model.reference_q_mass(t)
        div_qref_du = conservative_divergence(qref[..., None] * du, self.model.dx)
        ref_defect_l2 = jnp.sqrt(self.model.cell_area * jnp.sum(div_qref_du * div_qref_du))

        # Projected-path defect caused solely by learned velocity error.
        projected_defect_l2 = jnp.sqrt(self.model.cell_area * jnp.sum(div_q_du * div_q_du))
        qh_exact_l2 = jnp.sqrt(self.model.cell_area * jnp.sum((q * h_exact) ** 2))
        projected_defect_relative_to_exact_forcing = (
            projected_defect_l2 / jnp.maximum(qh_exact_l2, 1.0e-30)
        )

        vel_rel_qref = weighted_relative_velocity_error(qref_mass, du, u_exact)
        vel_rel_projected = weighted_relative_velocity_error(qmass, du, u_exact)

        # Direct moment-rate residual on the reference law for THIS sensor map.
        # For exact u: d/dt E_qref[Phi] = E_qref[J Phi u_exact].
        phi, grad_phi_ref = self.model.sensor_fields(eta)
        jphi_ref_exact = jnp.einsum("myxc,yxc->myx", grad_phi_ref, u_exact)
        jphi_ref_learned = jnp.einsum("myxc,yxc->myx", grad_phi_ref, u_learned)
        ref_rate_exact = jnp.sum(jphi_ref_exact * qref_mass[None, ...], axis=(1, 2))
        ref_rate_learned = jnp.sum(jphi_ref_learned * qref_mass[None, ...], axis=(1, 2))
        ref_moment_rate_error = jnp.linalg.norm(ref_rate_learned - ref_rate_exact)
        ref_moment_rate_scale = jnp.maximum(jnp.linalg.norm(ref_rate_exact), 1.0e-12)
        ref_moment_rate_rel = ref_moment_rate_error / ref_moment_rate_scale

        return jnp.array([
            lift,
            exact_action,
            learned_action,
            tangent_exact,
            tangent_learned,
            jnp.abs(compat_before),
            jnp.abs(compat_after),
            exact_poisson_rel,
            learned_poisson_rel,
            ref_defect_l2,
            projected_defect_l2,
            projected_defect_relative_to_exact_forcing,
            vel_rel_qref,
            vel_rel_projected,
            ref_moment_rate_error,
            ref_moment_rate_rel,
            jnp.abs(state["mean_h_raw"]),
            jnp.min(jnp.linalg.eigvalsh(state["C"])),
            jnp.min(jnp.linalg.eigvalsh(state["G"])),
        ])

    def evaluate_design(self, eta: np.ndarray) -> Dict[str, Any]:
        eta_j = jnp.asarray(eta, dtype=jnp.float64)
        rows = jax.vmap(
            lambda alpha: jax.vmap(lambda t: self.one_time_jit(t, alpha, eta_j))(self.model.times)
        )(self.model.alphas)

        # Integrated metrics use exactly the Stage-B alpha/time quadrature.
        integrated = jnp.sum(
            self.model.alpha_w[:, None, None]
            * self.model.time_w[None, :, None]
            * rows,
            axis=(0, 1),
        )
        a = np.asarray(rows, dtype=np.float64)
        z = np.asarray(integrated, dtype=np.float64)

        exact_action = float(z[1])
        learned_action = float(z[2])
        tangent_exact = float(z[3])
        tangent_learned = float(z[4])

        return {
            "lift_mmd2": float(z[0]),
            "exact_full_action": exact_action,
            "learned_full_action": learned_action,
            "learned_full_action_relative_change": float(learned_action / exact_action - 1.0),
            "exact_tangent_action": tangent_exact,
            "learned_tangent_action": tangent_learned,
            "learned_tangent_action_relative_change": float(tangent_learned / tangent_exact - 1.0),
            "exact_hidden_action": float(exact_action - tangent_exact),
            "learned_hidden_action": float(learned_action - tangent_learned),
            "max_abs_qh_compatibility_before_center": float(np.max(a[..., 5])),
            "max_abs_qh_compatibility_after_center": float(np.max(a[..., 6])),
            "max_exact_poisson_relative_residual": float(np.max(a[..., 7])),
            "max_learned_poisson_relative_residual": float(np.max(a[..., 8])),
            "mean_reference_continuity_defect_l2": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 9]
            )),
            "max_reference_continuity_defect_l2": float(np.max(a[..., 9])),
            "mean_projected_continuity_defect_l2": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 10]
            )),
            "max_projected_continuity_defect_l2": float(np.max(a[..., 10])),
            "mean_projected_defect_relative_to_exact_forcing": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 11]
            )),
            "max_projected_defect_relative_to_exact_forcing": float(np.max(a[..., 11])),
            "mean_velocity_relative_l2_under_qref": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 12]
            )),
            "max_velocity_relative_l2_under_qref": float(np.max(a[..., 12])),
            "mean_velocity_relative_l2_under_projected_q": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 13]
            )),
            "max_velocity_relative_l2_under_projected_q": float(np.max(a[..., 13])),
            "mean_reference_sensor_moment_rate_abs_error": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 14]
            )),
            "max_reference_sensor_moment_rate_abs_error": float(np.max(a[..., 14])),
            "mean_reference_sensor_moment_rate_relative_error": float(np.sum(
                np.asarray(self.model.alpha_w)[:, None] * np.asarray(self.model.time_w)[None, :] * a[..., 15]
            )),
            "max_reference_sensor_moment_rate_relative_error": float(np.max(a[..., 15])),
            "max_abs_exact_forcing_mean_raw": float(np.max(a[..., 16])),
            "min_calibration_cov_eig": float(np.min(a[..., 17])),
            "min_tangent_gram_eig": float(np.min(a[..., 18])),
        }


# -----------------------------------------------------------------------------
# Independent spatial-discretization check for the learned continuity defect
# -----------------------------------------------------------------------------


def fixed_design_resolution_check(backend, d0, params, cfg: D1Config, design_deg):
    """
    Evaluate only the learned/exact action contrast at two Stage-B grid sizes.

    This specifically checks the finite-difference div(q delta_u) term.  It does
    not re-train the FM network and does not re-optimize sensors.
    """
    base = backend.preset_config("reference")
    if cfg.preset == "quick":
        resolutions = [(19, 7), (31, 11)]
    elif cfg.preset == "reference":
        resolutions = [(39, 15), (51, 21)]
    else:
        resolutions = [(51, 21), (65, 27)]
    out = []
    for grid_n, time_n in resolutions:
        bcfg = dataclasses.replace(base, grid_n=grid_n, time_n=time_n)
        model = backend.StageB(bcfg)
        evaluator = LearnedVelocityMFSI(model, d0, params, cfg)
        eta = np.radians(np.asarray(design_deg, dtype=np.float64))
        row = evaluator.evaluate_design(eta)
        out.append({
            "grid_n": int(grid_n),
            "time_n": int(time_n),
            "exact_full_action": row["exact_full_action"],
            "learned_full_action": row["learned_full_action"],
            "learned_full_action_relative_change": row["learned_full_action_relative_change"],
            "max_abs_qh_compatibility_after_center": row["max_abs_qh_compatibility_after_center"],
        })
    if len(out) >= 2:
        coarse, fine = out[-2], out[-1]
        out[-1]["relative_change_in_learned_action_shift_vs_previous"] = float(
            abs(
                (fine["learned_full_action"] - fine["exact_full_action"])
                - (coarse["learned_full_action"] - coarse["exact_full_action"])
            ) / max(abs(fine["learned_full_action"] - fine["exact_full_action"]), 1.0e-12)
        )
    return out


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def full_vs_lift(designs: Dict[str, Any], action_key: str):
    full = designs["full"]
    lift = designs["lift"]
    return {
        "law_relative_penalty": float(full["lift_mmd2"] / lift["lift_mmd2"] - 1.0),
        "action_reduction_fraction": float(1.0 - full[action_key] / lift[action_key]),
        "full_minus_lift_action": float(full[action_key] - lift[action_key]),
    }


def print_summary(payload: Dict[str, Any]):
    print("\n" + "=" * 92)
    print("Stage D.1 flow-matching MFSI summary (learned velocity; no CNF)")
    print("=" * 92)
    for name, label in (("lift", "Lift"), ("tangent", "Tangent-TC"), ("full", "Full-TC")):
        r = payload["designs"][name]
        print(
            f"{label:10s} L={r['lift_mmd2']:.8f} | "
            f"A_exact={r['exact_full_action']:.3f} | A_FM={r['learned_full_action']:.3f} | "
            f"dA={100*r['learned_full_action_relative_change']:+.3f}%"
        )
        print(
            f"{'':10s} vel err(q) mean/max={r['mean_velocity_relative_l2_under_projected_q']:.3e}/"
            f"{r['max_velocity_relative_l2_under_projected_q']:.3e} | "
            f"max projected continuity defect/forcing={r['max_projected_defect_relative_to_exact_forcing']:.3e}"
        )
    print("-" * 92)
    ex = payload["contrasts"]["exact_velocity"]
    fm = payload["contrasts"]["learned_fm_velocity"]
    print(
        f"Full vs Lift [exact velocity]: law penalty={100*ex['law_relative_penalty']:+.3f}% | "
        f"action reduction={100*ex['action_reduction_fraction']:+.2f}%"
    )
    print(
        f"Full vs Lift [learned FM velocity]: law penalty={100*fm['law_relative_penalty']:+.3f}% | "
        f"action reduction={100*fm['action_reduction_fraction']:+.2f}%"
    )
    print("=" * 92)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--d0-script", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--output", type=str, default="stage_d1_flow_matching_mfsi.json")
    p.add_argument(
        "--skip-resolution-check", action="store_true",
        help="Skip the two-resolution Full-TC check for div(q delta_u).",
    )
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()

    backend_path = Path(args.backend) if args.backend else autodetect(
        ["stage_b2_transport_conditioned_design.py", "stage_b2_transport_conditioned_design(4).py"]
    )
    if backend_path is None:
        raise FileNotFoundError("Stage-B backend not found; pass --backend")
    d0_path = Path(args.d0_script) if args.d0_script else autodetect(["stage_d0_flow_matching_reference.py"])
    if d0_path is None:
        raise FileNotFoundError("Stage-D.0 source not found; pass --d0-script")

    backend = load_module(backend_path, "stage_b2_backend_d1")
    d0 = load_module(d0_path, "stage_d0_module_d1")
    params, checkpoint_meta = d0.load_checkpoint(Path(args.checkpoint))
    cfg = preset_d1_config(args.preset)

    if args.preset == "quick":
        stage_b_cfg = backend.preset_config("quick")
    elif args.preset == "reference":
        # Match the validated Stage-B fine-resolution reporting grid, not the
        # intermediate 39x39/time15 optimization grid.
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=39, time_n=15)
    else:
        base = backend.preset_config("reference")
        stage_b_cfg = dataclasses.replace(base, grid_n=65, time_n=27)

    # Verify physical system compatibility with D0 checkpoint.
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
    evaluator = LearnedVelocityMFSI(model, d0, params, cfg)

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
        "exact_velocity": full_vs_lift(designs, "exact_full_action"),
        "learned_fm_velocity": full_vs_lift(designs, "learned_full_action"),
    }
    contrasts["change_in_full_vs_lift_action_reduction"] = float(
        contrasts["learned_fm_velocity"]["action_reduction_fraction"]
        - contrasts["exact_velocity"]["action_reduction_fraction"]
    )

    resolution_check = None
    if not args.skip_resolution_check:
        print("Running fixed-design spatial check for Full-TC learned continuity defect...", flush=True)
        resolution_check = fixed_design_resolution_check(
            backend, d0, params, cfg, cfg.full_design_deg
        )

    all_compat = max(r["max_abs_qh_compatibility_after_center"] for r in designs.values())
    checks = {
        "finite_outputs": bool(all(
            np.isfinite(v)
            for r in designs.values()
            for k, v in r.items()
            if isinstance(v, (int, float))
        )),
        "learned_qh_compatibility_small": bool(all_compat < cfg.compatibility_tol),
        "projected_law_unchanged_by_construction": True,
        "full_tc_learned_action_below_lift": bool(
            designs["full"]["learned_full_action"] < designs["lift"]["learned_full_action"]
        ),
    }

    payload = {
        "stage": "D.1",
        "method": "flow-matching learned reference velocity on the prescribed Stage-B/FM marginal path; no CNF density",
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
        },
        "designs": designs,
        "contrasts": contrasts,
        "full_tc_resolution_check": resolution_check,
        "checks": checks,
        "interpretation": [
            "The flow-matching target marginal path is held fixed; D.1 changes only the learned reference velocity.",
            "No CNF density, log-Jacobian integration, KDE, or finite-particle I-projection is introduced.",
            "The learned forcing includes the continuity defect div(q (u_theta-u_exact)); omitting it would incorrectly assume the learned velocity transports the prescribed path exactly.",
            "Law MMD is unchanged by construction in D.1.  The scientific quantity under test is the stability of transport/correction action to learned-velocity approximation.",
            "Finite/noisy measurements remain excluded so learning error is isolated before composition with Stage C.",
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
