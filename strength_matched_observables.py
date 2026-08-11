"""Core utilities for the prespecified Experiment-E follow-up controls.

This module adds no new registered Experiment-D metric.  It provides frozen
design-bank matching, strength-constrained FIBER training, and validated time
reparameterizations of the existing Experiment-B reference bridge.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

import example_b as exb
import observable_design_toy as od

jax.config.update("jax_enable_x64", True)

Array = jax.Array
GEOMETRIES = ("default", "smoothstep", "cosine")


def warp_time(t: Array, geometry: str) -> tuple[Array, Array]:
    """Return s(t), ds/dt for endpoint-preserving reference schedules."""
    t = jnp.asarray(t, dtype=jnp.float64)
    if geometry == "default":
        return t, jnp.ones_like(t)
    if geometry == "smoothstep":
        return 3.0 * t * t - 2.0 * t * t * t, 6.0 * t * (1.0 - t)
    if geometry == "cosine":
        return 0.5 * (1.0 - jnp.cos(jnp.pi * t)), 0.5 * jnp.pi * jnp.sin(jnp.pi * t)
    raise ValueError(f"unknown reference geometry: {geometry}")


def sample_bridge_geometry(key: Array, t: Array, n: int, geometry: str) -> tuple[Array, Array]:
    s, ds = warp_time(t, geometry)
    x, dx_ds = exb.sample_bridge(key, s, n)
    return x, ds * dx_ds


def reference_velocity_geometry(reference_params, t: Array, x: Array, geometry: str) -> Array:
    s, ds = warp_time(t, geometry)
    return ds * exb.reference_velocity(reference_params, s, x)


def strength_matrix(
    key: Array,
    times: Array,
    n_particles: int,
    standardization: od.Standardization,
    geometry: str = "default",
) -> tuple[Array, Array, Array]:
    """Fixed design-bank M=mean_t m_t m_t' plus means and average covariance."""
    keys = jax.random.split(key, len(times))
    means, covariances = [], []
    for k, t in zip(keys, times):
        x, _ = sample_bridge_geometry(k, t, n_particles, geometry)
        z = od.standardized_dictionary(x, standardization)
        mean = jnp.mean(z, axis=0)
        centered = z - mean
        means.append(mean)
        covariances.append(centered.T @ centered / z.shape[0])
    means_array = jnp.stack(means)
    return (
        jnp.mean(jax.vmap(lambda value: jnp.outer(value, value))(means_array), axis=0),
        means_array,
        jnp.mean(jnp.stack(covariances), axis=0),
    )


def constraint_strength(A: Array, matrix: Array) -> Array:
    return jnp.einsum("ri,ij,rj->", A, matrix, A)


def normalized_strength(A: Array, means: Array, eps: float = 1e-14) -> Array:
    projected = means @ A.T
    return jnp.mean(jnp.sum(projected * projected, axis=-1) /
                    (jnp.sum(means * means, axis=-1) + eps))


def random_stiefel_pool(key: Array, count: int, R: int = 3, dimension: int = 5) -> Array:
    """Haar row-subspaces using the same Gaussian/QR construction as Experiment D."""
    B = jax.random.normal(key, (count, R, dimension), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(jnp.swapaxes(B, -1, -2))
    return jnp.swapaxes(q, -1, -2)


def make_fiber_banks_geometry(
    key: Array,
    times: Array,
    delta_t: float,
    n_particles: int,
    reference_params,
    geometry: str,
) -> dict[str, Array]:
    keys = jax.random.split(key, 2 * len(times))
    x0, x1, u0, u1 = [], [], [], []
    for i, t in enumerate(times):
        xa, _ = sample_bridge_geometry(keys[2 * i], t, n_particles, geometry)
        tb = jnp.minimum(t + delta_t, 1.0)
        xb, _ = sample_bridge_geometry(keys[2 * i + 1], tb, n_particles, geometry)
        x0.append(xa); x1.append(xb)
        u0.append(reference_velocity_geometry(reference_params, t, xa, geometry))
        u1.append(reference_velocity_geometry(reference_params, tb, xb, geometry))
    return {"times": jnp.asarray(times), "x0": jnp.stack(x0), "x1": jnp.stack(x1),
            "u0": jnp.stack(u0), "u1": jnp.stack(u1)}


def _fiber_metadata(value: Array, aux: dict[str, Array], history: list[dict[str, Any]],
                    found: bool) -> dict[str, Any]:
    return {
        "fiber_validation_objective": float(value),
        "local_tangent_mmd2": float(jnp.mean(aux["local_mmd2"])),
        "min_ess": float(jnp.min(jnp.concatenate([aux["ess_t"], aux["ess_next"]]))),
        "max_calibration_residual": float(jnp.max(jnp.concatenate([aux["residual_t"], aux["residual_next"]]))),
        "max_condition": float(jnp.max(jnp.concatenate([aux["condition_t"], aux["condition_next"]]))),
        "feasible_checkpoint_found": found,
        "history": history,
    }


def train_fiber_geometry(
    key: Array,
    standardization: od.Standardization,
    B0: Array,
    reference_params,
    *,
    geometry: str,
    steps: int,
    n_times: int,
    n_particles: int,
    delta_t: float,
    lr: float = 1e-3,
) -> tuple[Array, dict[str, Any]]:
    """The unchanged FIBER objective on an endpoint-preserving time schedule."""
    kt, kv = jax.random.split(key)
    times = jnp.linspace(0.08, 0.92 - delta_t, n_times)
    train_banks = make_fiber_banks_geometry(kt, times, delta_t, n_particles, reference_params, geometry)
    val_banks = make_fiber_banks_geometry(kv, times, delta_t, n_particles, reference_params, geometry)

    def loss_fn(B):
        return od.fiber_objective_from_A(od.stiefel_rows(B), standardization, train_banks, delta_t)[0]

    vg = jax.jit(jax.value_and_grad(loss_fn))
    vfun = jax.jit(lambda B: od.fiber_objective_from_A(
        od.stiefel_rows(B), standardization, val_banks, delta_t))
    B, state = B0, od._adam_init(B0)
    best, best_val, found = B0, float("inf"), False
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        loss, grad = vg(B)
        rate = exb.core.cosine_lr(step - 1, steps, lr, lr * 0.05)
        B, state = od._update(B, grad, state, step, rate, weight_decay=0.0)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            val, aux = vfun(B)
            feasible = od.fiber_checkpoint_feasible(aux, B.shape[0])
            if feasible and (not found or float(val) < best_val):
                found, best_val, best = True, float(val), B
            history.append({"step": step, "train_objective": float(loss),
                            "validation_objective": float(val), "feasible": feasible})
    if not found:
        best = B0
    value, aux = vfun(best)
    meta = _fiber_metadata(value, aux, history, found)
    meta["reference_geometry"] = geometry
    return od.stiefel_rows(best), meta


def train_strength_constrained_fiber(
    key: Array,
    standardization: od.Standardization,
    B0: Array,
    reference_params,
    strength_design_matrix: Array,
    *,
    target_strength: float,
    relative_tolerance: float,
    gamma_start: float,
    gamma_end: float,
    steps: int,
    n_times: int,
    n_particles: int,
    delta_t: float,
    lr: float = 1e-3,
) -> tuple[Array, dict[str, Any]]:
    """Unchanged FIBER loss plus the prespecified squared strength penalty."""
    kt, kv = jax.random.split(key)
    times = jnp.linspace(0.08, 0.92 - delta_t, n_times)
    train_banks = od.make_fiber_banks(kt, times, delta_t, n_particles, reference_params)
    val_banks = od.make_fiber_banks(kv, times, delta_t, n_particles, reference_params)
    target = jnp.asarray(target_strength, dtype=jnp.float64)

    def total_loss(B, gamma):
        A = od.stiefel_rows(B)
        fiber_loss = od.fiber_objective_from_A(A, standardization, train_banks, delta_t)[0]
        strength = constraint_strength(A, strength_design_matrix)
        return fiber_loss + gamma * (strength - target) ** 2

    vg = jax.jit(jax.value_and_grad(total_loss))
    vfun = jax.jit(lambda B: od.fiber_objective_from_A(
        od.stiefel_rows(B), standardization, val_banks, delta_t))
    B, state = B0, od._adam_init(B0)
    best, best_val, found = B0, float("inf"), False
    closest, closest_error, closest_gate = B0, float("inf"), False
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        fraction = (step - 1) / max(steps - 1, 1)
        gamma = gamma_start * (gamma_end / gamma_start) ** fraction
        loss, grad = vg(B, jnp.asarray(gamma))
        rate = exb.core.cosine_lr(step - 1, steps, lr, lr * 0.05)
        B, state = od._update(B, grad, state, step, rate, weight_decay=0.0)
        if step == 1 or step % max(steps // 20, 1) == 0 or step == steps:
            val, aux = vfun(B)
            A = od.stiefel_rows(B)
            achieved = float(constraint_strength(A, strength_design_matrix))
            rel_error = abs(achieved - target_strength) / (target_strength + 1e-14)
            gate = od.fiber_checkpoint_feasible(aux, B.shape[0])
            feasible = gate and rel_error <= relative_tolerance
            if gate and rel_error < closest_error:
                closest, closest_error, closest_gate = B, rel_error, True
            if feasible and (not found or float(val) < best_val):
                found, best_val, best = True, float(val), B
            history.append({"step": step, "gamma": gamma, "train_total": float(loss),
                            "validation_fiber": float(val), "achieved_strength": achieved,
                            "relative_strength_error": rel_error, "calibration_gate": gate,
                            "accepted": feasible})
    if not found:
        best = closest if closest_gate else B0
    A = od.stiefel_rows(best)
    value, aux = vfun(best)
    achieved = float(constraint_strength(A, strength_design_matrix))
    relative_error = abs(achieved - target_strength) / (target_strength + 1e-14)
    meta = _fiber_metadata(value, aux, history, found)
    meta.update({"target_strength": target_strength, "achieved_strength": achieved,
                 "relative_strength_error": relative_error,
                 "strength_tolerance": relative_tolerance,
                 "strength_target_accepted": bool(found and relative_error <= relative_tolerance),
                 "gamma_start": gamma_start, "gamma_end": gamma_end})
    return A, meta


def _ritz_bank_geometry(
    key: Array,
    model: od.ObservableModel,
    reference_params,
    n_times: int,
    n_particles: int,
    geometry: str,
) -> dict[str, Array]:
    kt, kb = jax.random.split(key)
    times = exb.stratified_times(kt, n_times, lo=0.04, hi=0.96)
    keys = jax.random.split(kb, n_times)
    xs, ws, hs = [], [], []
    for k, t in zip(keys, times):
        x, _ = sample_bridge_geometry(k, t, n_particles, geometry)
        u = reference_velocity_geometry(reference_params, t, x, geometry)
        f = od.project_bank(model.A, model.standardization, x, u)
        xs.append(x); ws.append(f.projected_weights); hs.append(f.forcing)
    return {"times": times, "x": jnp.stack(xs), "weights": jnp.stack(ws), "h": jnp.stack(hs)}


def _ritz_loss(params, bank: dict[str, Array]) -> Array:
    values = jax.vmap(exb.ritz_state_loss, in_axes=(None, 0, 0, 0, 0))(
        params, bank["times"], bank["x"], bank["weights"], bank["h"])
    return jnp.mean(values)


def train_downstream_ritz_geometry(
    key: Array,
    model: od.ObservableModel,
    reference_params,
    *,
    geometry: str,
    steps: int,
    n_times: int,
    n_particles: int,
) -> tuple[Any, dict[str, Any]]:
    ki, kt, kv = jax.random.split(key, 3)
    input_dim = exb.STATE_DIM + 1 + 2 * exb.TIME_FREQ
    params = exb.core.init_mlp(ki, input_dim, exb.RITZ_HIDDEN, 1)
    train_bank = _ritz_bank_geometry(kt, model, reference_params, n_times, n_particles, geometry)
    val_bank = _ritz_bank_geometry(kv, model, reference_params, max(n_times, 3), n_particles, geometry)
    vg = jax.jit(jax.value_and_grad(lambda p: _ritz_loss(p, train_bank)))
    vfun = jax.jit(lambda p: _ritz_loss(p, val_bank))
    state = od._adam_init(params)
    best, best_val, history = params, float(vfun(params)), []
    for step in range(1, steps + 1):
        loss, grad = vg(params)
        rate = exb.core.cosine_lr(step - 1, steps, 1.5e-3, 4e-5)
        params, state = od._update(params, grad, state, step, rate)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            value = float(vfun(params))
            if value < best_val:
                best, best_val = params, value
            history.append({"step": step, "train_ritz_loss": float(loss), "validation_ritz_loss": value})
    return best, {"heldout_ritz_loss": best_val, "history": history,
                  "reference_geometry": geometry}


@partial(jax.jit, static_argnames=("n_steps", "geometry"))
def _compiled_rollouts_geometry(
    x0: Array,
    reference_params,
    potential_params,
    A: Array,
    whitening: Array,
    *,
    n_steps: int,
    geometry: str,
):
    reference = lambda t, x: reference_velocity_geometry(reference_params, t, x, geometry)
    learned = lambda t, x: reference(t, x) - exb.potential_grad(potential_params, t, x)
    tangent = lambda t, x: od._safety_velocity_arrays(A, whitening, x, reference(t, x))
    learned_safe = lambda t, x: od._safety_velocity_arrays(A, whitening, x, learned(t, x))
    times, raw = exb.integrate_field(x0, reference, n_steps)
    _, tan = exb.integrate_field(x0, tangent, n_steps)
    _, mfsi = exb.integrate_field(x0, learned, n_steps)
    _, safe = exb.integrate_field(x0, learned_safe, n_steps)
    return times, raw, tan, mfsi, safe


def rollout_methods_geometry(
    key: Array,
    model: od.ObservableModel,
    reference_params,
    potential_params,
    *,
    n_particles: int,
    flow_steps: int,
    geometry: str,
) -> dict[str, dict[str, Array]]:
    x0 = exb.whiten_empirical(exb.sample_ring(key, n_particles))
    times, raw, tangent, mfsi, safe = _compiled_rollouts_geometry(
        x0, reference_params, potential_params, model.A, model.standardization.whitening,
        n_steps=flow_steps, geometry=geometry)
    safe.block_until_ready()
    return {
        "raw_si": {"times": times, "trajectory": raw},
        "moment_tangent": {"times": times, "trajectory": tangent},
        "mfsi_learned": {"times": times, "trajectory": mfsi},
        "mfsi_learned_safe": {"times": times, "trajectory": safe},
    }


def _trajectory_at(run: dict[str, Array], t: float) -> Array:
    index = int(np.argmin(np.abs(np.asarray(run["times"]) - t)))
    return run["trajectory"][index]


def evaluate_downstream_geometry(
    key: Array,
    model: od.ObservableModel,
    reference_params,
    potential_params,
    *,
    geometry: str,
    times: Iterable[float],
    n_particles: int,
    target_particles: int,
    flow_steps: int,
    local_dt: float,
) -> dict[str, Any]:
    """Experiment-D metrics under a fixed endpoint-preserving reference schedule."""
    kr, kb = jax.random.split(key)
    runs = rollout_methods_geometry(kr, model, reference_params, potential_params,
                                    n_particles=n_particles, flow_steps=flow_steps,
                                    geometry=geometry)
    times = list(times)
    keys = jax.random.split(kb, 2 * len(times))
    per_method = {name: [] for name in runs}
    target_rows, local_rows = [], []
    for i, t_float in enumerate(times):
        t = jnp.asarray(t_float, dtype=jnp.float64)
        x, _ = sample_bridge_geometry(keys[2 * i], t, target_particles, geometry)
        u = reference_velocity_geometry(reference_params, t, x, geometry)
        f = od.project_bank(model.A, model.standardization, x, u)
        w = f.projected_weights
        m = min(384, target_particles, n_particles)
        xt, wt = x[:m], w[:m]
        wt = wt / jnp.sum(wt)
        target_ang = w @ exb.angular_features(x)
        target_rows.append({
            "t": t_float, "ess_fraction": float(f.ess_fraction),
            "condition": float(f.covariance_condition),
            "calibration_residual": float(f.calibration_residual),
            "projection_distortion": float(jnp.sum(w * jnp.log(jnp.maximum(w * x.shape[0], 1e-300)))),
            "lambda_norm": float(jnp.linalg.norm(f.lambda_)),
        })
        for name, run in runs.items():
            y = _trajectory_at(run, t_float)
            ph_mean = jnp.mean(od.observable_values(model.A, model.standardization, y), axis=0)
            mmd = jnp.sqrt(od.weighted_mmd2(xt, wt, y[:m]))
            angular = jnp.linalg.norm(jnp.mean(exb.angular_features(y), axis=0) - target_ang)
            per_method[name].append({
                "t": t_float, "mmd": float(mmd),
                "max_moment_error": float(jnp.max(jnp.abs(ph_mean))),
                "mean_moment_error": float(jnp.mean(jnp.abs(ph_mean))),
                "angular_error": float(angular),
            })
        if t_float + local_dt <= 1.0:
            tn = jnp.asarray(t_float + local_dt, dtype=jnp.float64)
            xn, _ = sample_bridge_geometry(keys[2 * i + 1], tn, target_particles, geometry)
            un = reference_velocity_geometry(reference_params, tn, xn, geometry)
            fn = od.project_bank(model.A, model.standardization, xn, un)
            vtan = od.tangent_velocity(model.A, model.standardization, x, u, w)
            vmfsi = u - exb.potential_grad(potential_params, t, x)
            gap2 = jnp.sum(w * jnp.sum((vmfsi - vtan) ** 2, axis=-1))
            local_rows.append({
                "t": t_float,
                "tangent_next_mmd": float(jnp.sqrt(od.weighted_mmd2(
                    (x + local_dt * vtan)[:m], w[:m], xn[:m], fn.projected_weights[:m]))),
                "mfsi_next_mmd": float(jnp.sqrt(od.weighted_mmd2(
                    (x + local_dt * vmfsi)[:m], w[:m], xn[:m], fn.projected_weights[:m]))),
                "velocity_gap_mse": float(gap2),
            })
    summary = {}
    for name, rows in per_method.items():
        interior = [r for r in rows if 0.0 < r["t"] < 1.0]
        summary[name] = {
            "mean_interior_mmd": float(np.mean([r["mmd"] for r in interior])),
            "max_moment_error": float(np.max([r["max_moment_error"] for r in rows])),
            "mean_moment_error": float(np.mean([r["mean_moment_error"] for r in rows])),
            "mean_interior_angular_error": float(np.mean([r["angular_error"] for r in interior])),
            "endpoint_mmd": float(rows[-1]["mmd"]),
        }
    local_summary = {
        "mean_tangent_next_mmd": float(np.mean([r["tangent_next_mmd"] for r in local_rows])),
        "mean_mfsi_next_mmd": float(np.mean([r["mfsi_next_mmd"] for r in local_rows])),
        "mean_velocity_gap_mse": float(np.mean([r["velocity_gap_mse"] for r in local_rows])),
    }
    return {"summary": summary, "per_method": per_method, "target": target_rows,
            "local": local_rows, "local_summary": local_summary}


def downstream_record(result: dict[str, Any]) -> dict[str, float]:
    target = result["target"]
    return {
        "tangent_local_mmd": result["local_summary"]["mean_tangent_next_mmd"],
        "tangent_rollout_mmd": result["summary"]["moment_tangent"]["mean_interior_mmd"],
        "mfsi_rollout_mmd": result["summary"]["mfsi_learned_safe"]["mean_interior_mmd"],
        "velocity_gap": result["local_summary"]["mean_velocity_gap_mse"],
        "angular_error": result["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"],
        "min_ess": min(row["ess_fraction"] for row in target),
        "mean_ess": float(np.mean([row["ess_fraction"] for row in target])),
        "mean_projection_distortion": float(np.mean([row["projection_distortion"] for row in target])),
        "mean_lambda_norm": float(np.mean([row["lambda_norm"] for row in target])),
        "max_moment_error": result["summary"]["mfsi_learned_safe"]["max_moment_error"],
    }
