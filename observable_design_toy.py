"""Learned-observable design for the Experiment-B two-dimensional toy.

This module is deliberately layered on :mod:`example_b`.  It changes only the
measured observable map; endpoint sampling, the stochastic interpolant,
reference velocity, empirical I-projection, Deep-Ritz integrand, MMD bandwidth
convention, angular diagnostics, and Heun rollout all come from the validated
Experiment-B implementation.

The public functions in this file are kept small and functional so that the
three observable objectives and their derivatives can be tested independently
of the command-line experiment driver.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import example_b as exb

jax.config.update("jax_enable_x64", True)

Array = jax.Array
RAW_DIM = 5
BASIS_NAMES = ("x1", "x2", "x1^2", "x1*x2", "x2^2")
OBJECTIVES = ("info", "cv", "fiber")


@dataclass(frozen=True)
class Standardization:
    """Frozen design-split centering and whitening of the raw dictionary."""

    center: Array
    whitening: Array
    covariance_eigenvalues: Array


@dataclass(frozen=True)
class ObservableModel:
    """A row-orthonormal subspace in standardized dictionary coordinates."""

    A: Array
    standardization: Standardization

    @property
    def R(self) -> int:
        return int(self.A.shape[0])

    @property
    def raw_coefficients(self) -> Array:
        return self.A @ self.standardization.whitening

    @property
    def raw_intercept(self) -> Array:
        return -(self.raw_coefficients @ self.standardization.center)


def raw_dictionary(x: Array) -> Array:
    """The exact five-observable dictionary from Experiment B."""
    return exb.phi(x)


def raw_dictionary_jacobian(x: Array) -> Array:
    return exb.jphi(x)


def fit_standardization(x_minus: Array, x_plus: Array, eig_floor: float = 1e-10) -> Standardization:
    """Fit one fixed full-rank symmetric whitening transform on design data."""
    bm = raw_dictionary(x_minus)
    bp = raw_dictionary(x_plus)
    center = 0.5 * (jnp.mean(bm, axis=0) + jnp.mean(bp, axis=0))
    pooled = jnp.concatenate([bm - center, bp - center], axis=0)
    cov = pooled.T @ pooled / pooled.shape[0]
    cov = 0.5 * (cov + cov.T)
    vals, vecs = jnp.linalg.eigh(cov)
    floor = eig_floor * jnp.maximum(jnp.max(vals), 1.0)
    safe = jnp.maximum(vals, floor)
    whitening = vecs @ jnp.diag(1.0 / jnp.sqrt(safe)) @ vecs.T
    return Standardization(center=center, whitening=whitening, covariance_eigenvalues=vals)


def standardized_dictionary(x: Array, standardization: Standardization) -> Array:
    # Row convention is equivalent to z = W @ (b-c) for each column vector.
    return (raw_dictionary(x) - standardization.center) @ standardization.whitening.T


def standardized_jacobian(x: Array, standardization: Standardization) -> Array:
    return jnp.einsum("ab,...bd->...ad", standardization.whitening, raw_dictionary_jacobian(x))


def stiefel_rows(B: Array) -> Array:
    """Differentiable QR retraction from R x 5 to row-orthonormal A."""
    q, r = jnp.linalg.qr(B.T, mode="reduced")
    # Fix the otherwise arbitrary QR signs.  The smooth approximation avoids a
    # zero derivative at exactly-zero diagonal entries while preserving scale.
    diag = jnp.diag(r)
    signs = jnp.where(diag < 0.0, -1.0, 1.0)
    return (q * signs[None, :]).T


def initialize_stiefel(key: Array, R: int) -> tuple[Array, Array]:
    if not 1 <= R <= RAW_DIM:
        raise ValueError(f"R must be in [1,{RAW_DIM}], got {R}")
    B = jax.random.normal(key, (R, RAW_DIM), dtype=jnp.float64)
    return B, stiefel_rows(B)


def observable_values(A: Array, standardization: Standardization, x: Array) -> Array:
    return standardized_dictionary(x, standardization) @ A.T


def observable_jacobian(A: Array, standardization: Standardization, x: Array) -> Array:
    return jnp.einsum("ra,...ad->...rd", A, standardized_jacobian(x, standardization))


def observable_rate(A: Array, standardization: Standardization, x: Array, velocity: Array) -> Array:
    return jnp.einsum("...rd,...d->...r", observable_jacobian(A, standardization, x), velocity)


def project_bank(A: Array, standardization: Standardization, x: Array, velocity: Array):
    ph = observable_values(A, standardization, x)
    rate = observable_rate(A, standardization, x, velocity)
    return exb.core.empirical_fiber_state(x, velocity, jnp.zeros(A.shape[0], dtype=x.dtype), ph=ph, jphi_u=rate)


def tangent_velocity(
    A: Array,
    standardization: Standardization,
    x: Array,
    base_velocity: Array,
    weights: Array | None = None,
) -> Array:
    """Minimum-energy velocity correction tangent to E[Phi_A]=0."""
    n = x.shape[0]
    weights = jnp.ones(n, dtype=x.dtype) / n if weights is None else weights / jnp.sum(weights)
    jp = observable_jacobian(A, standardization, x)
    rate = jnp.einsum("nrd,nd->nr", jp, base_velocity)
    r = weights @ rate
    G = jnp.einsum("n,nrd,nsd->rs", weights, jp, jp)
    coeff, _, _ = exb.core._stable_cov_solve(G, r, damping=1e-10)
    return base_velocity - jnp.einsum("nrd,r->nd", jp, coeff)


def safety_velocity(model: ObservableModel, x: Array, velocity: Array) -> Array:
    return tangent_velocity(model.A, model.standardization, x, velocity)


def weighted_mmd2(x: Array, wx: Array, y: Array, wy: Array | None = None) -> Array:
    """Differentiable weighted squared RBF-MMD with Experiment-B bandwidths."""
    wx = wx / jnp.sum(wx)
    if wy is None:
        wy = jnp.ones(y.shape[0], dtype=y.dtype) / y.shape[0]
    else:
        wy = wy / jnp.sum(wy)
    mean = wx @ x
    var = jnp.sum(wx * jnp.sum((x - mean) ** 2, axis=-1)) / x.shape[-1] + 1e-6
    bws = jnp.sqrt(var) * jnp.array([0.5, 1.0, 2.0], dtype=x.dtype)
    dxx = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    dyy = jnp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    dxy = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)

    def one(s):
        kxx = jnp.exp(-0.5 * dxx / (s * s))
        kyy = jnp.exp(-0.5 * dyy / (s * s))
        kxy = jnp.exp(-0.5 * dxy / (s * s))
        return wx @ (kxx @ wx) + wy @ (kyy @ wy) - 2.0 * wx @ (kxy @ wy)

    return jnp.maximum(jnp.mean(jax.vmap(one)(bws)), 0.0)


def subspace_distance(A: Array, B: Array) -> Array:
    if A.shape != B.shape:
        raise ValueError("subspaces must have the same shape")
    return jnp.linalg.norm(A.T @ A - B.T @ B) / jnp.sqrt(2.0 * A.shape[0])


def principal_angles(A: Array, B: Array) -> Array:
    s = jnp.linalg.svd(A @ B.T, compute_uv=False)
    return jnp.arccos(jnp.clip(s, -1.0, 1.0))


def _mlp_apply(params, x):
    return exb.core.mlp_apply(params, x)


def _binary_cross_entropy(logits: Array, labels: Array) -> Array:
    return jnp.mean(jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _adam_init(params):
    return exb.core._tree_zeros_like(params), exb.core._tree_zeros_like(params)


def _update(params, grads, state, step: int, lr: Array, weight_decay: float = 1e-7):
    grads = exb._global_clip(grads, 5.0)
    p, m, v = exb.core._adamw_update(params, grads, state[0], state[1], step, lr, weight_decay)
    return p, (m, v)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = ~pos
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # Average tied ranks.
    unique, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    npos, nneg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def classification_metrics(logits: Array, labels: Array) -> dict[str, float]:
    logits_np = np.asarray(logits)
    labels_np = np.asarray(labels)
    return {
        "cross_entropy": float(_binary_cross_entropy(logits, labels)),
        "accuracy": float(np.mean((logits_np >= 0.0) == labels_np.astype(bool))),
        "auroc": _auc(labels_np, logits_np),
    }


def _endpoint_dataset(key: Array, n_per_class: int) -> tuple[Array, Array]:
    k0, k1 = jax.random.split(key)
    x0 = exb.sample_ring(k0, n_per_class)
    x1 = exb.sample_four_lobes(k1, n_per_class)
    x = jnp.concatenate([x0, x1], axis=0)
    y = jnp.concatenate([jnp.zeros(n_per_class), jnp.ones(n_per_class)]).astype(jnp.float64)
    return x, y


def train_info(
    key: Array,
    standardization: Standardization,
    B0: Array,
    *,
    steps: int,
    n_train: int,
    n_validation: int,
    hidden: tuple[int, ...] = (32, 32),
    lr: float = 2e-3,
) -> tuple[Array, dict[str, Any]]:
    """Jointly train A and a supervised endpoint-state information proxy."""
    kt, kv, ki = jax.random.split(key, 3)
    xt, yt = _endpoint_dataset(kt, n_train // 2)
    xv, yv = _endpoint_dataset(kv, n_validation // 2)
    classifier = exb.core.init_mlp(ki, B0.shape[0], hidden, 1)
    params = (B0, classifier)
    state = _adam_init(params)

    def loss_fn(p):
        B, clf = p
        logits = _mlp_apply(clf, observable_values(stiefel_rows(B), standardization, xt))[..., 0]
        return _binary_cross_entropy(logits, yt)

    def val_fn(p):
        B, clf = p
        logits = _mlp_apply(clf, observable_values(stiefel_rows(B), standardization, xv))[..., 0]
        return _binary_cross_entropy(logits, yv), logits

    vg = jax.jit(jax.value_and_grad(loss_fn))
    vfun = jax.jit(val_fn)
    best = params
    best_val = float("inf")
    history = []
    for step in range(1, steps + 1):
        loss, grads = vg(params)
        rate = exb.core.cosine_lr(step - 1, steps, lr, lr * 0.05)
        params, state = _update(params, grads, state, step, rate)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            val, logits = vfun(params)
            vf = float(val)
            if vf < best_val:
                best_val, best = vf, params
            history.append({"step": step, "train_cross_entropy": float(loss), "validation_cross_entropy": vf})
    val, logits = vfun(best)
    metrics = classification_metrics(logits, yv)
    metrics.update({"validation_cross_entropy": float(val), "history": history})
    return stiefel_rows(best[0]), metrics


def _cv_features(t: Array, z: Array) -> Array:
    tt = jnp.broadcast_to(t, z.shape[:-1])
    return jnp.concatenate([z, exb.core.time_fourier_features(tt, exb.TIME_FREQ)], axis=-1)


def _bridge_dataset(key: Array, reference_params, n: int) -> tuple[Array, Array, Array]:
    kt, kx = jax.random.split(key)
    times = exb.stratified_times(kt, n)
    x, _ = exb.sample_bridge_times(kx, times)
    u = exb.reference_velocity(reference_params, times, x)
    return times, x, u


def train_cv(
    key: Array,
    standardization: Standardization,
    B0: Array,
    reference_params,
    *,
    steps: int,
    n_train: int,
    n_validation: int,
    hidden: tuple[int, ...] = (32, 32),
    lr: float = 2e-3,
) -> tuple[Array, dict[str, Any]]:
    """Train A for variance-normalized reduced-flow closure."""
    kt, kv, ki = jax.random.split(key, 3)
    tt, xt, ut = _bridge_dataset(kt, reference_params, n_train)
    tv, xv, uv = _bridge_dataset(kv, reference_params, n_validation)
    input_dim = B0.shape[0] + 1 + 2 * exb.TIME_FREQ
    field = exb.core.init_mlp(ki, input_dim, hidden, B0.shape[0])
    params = (B0, field)
    state = _adam_init(params)

    def closure(p, times, x, u):
        B, net = p
        A = stiefel_rows(B)
        z = observable_values(A, standardization, x)
        dz = observable_rate(A, standardization, x, u)
        pred = _mlp_apply(net, _cv_features(times, z))
        mse = jnp.mean((pred - dz) ** 2)
        variance = jnp.mean((dz - jnp.mean(dz, axis=0)) ** 2) + 1e-8
        return mse / variance, (mse, variance)

    loss_fn = lambda p: closure(p, tt, xt, ut)[0]
    vg = jax.jit(jax.value_and_grad(loss_fn))
    vfun = jax.jit(lambda p: closure(p, tv, xv, uv))
    best = params
    best_val = float("inf")
    history = []
    for step in range(1, steps + 1):
        loss, grads = vg(params)
        rate = exb.core.cosine_lr(step - 1, steps, lr, lr * 0.05)
        params, state = _update(params, grads, state, step, rate)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            val, aux = vfun(params)
            vf = float(val)
            if vf < best_val:
                best_val, best = vf, params
            history.append({"step": step, "train_normalized_mse": float(loss), "validation_normalized_mse": vf})
    normalized, (mse, variance) = vfun(best)
    return stiefel_rows(best[0]), {
        "closure_mse": float(mse),
        "closure_normalized_mse": float(normalized),
        "closure_R2": float(1.0 - mse / variance),
        "history": history,
    }


def reduced_flow_closure_diagnostic(
    key: Array,
    model: ObservableModel,
    reference_params,
    *,
    steps: int,
    n_train: int,
    n_validation: int,
    hidden: tuple[int, ...] = (32, 32),
) -> dict[str, float]:
    """Fit a fresh reduced field with frozen A for a comparable closure score."""
    kt, kv, ki = jax.random.split(key, 3)
    tt, xt, ut = _bridge_dataset(kt, reference_params, n_train)
    tv, xv, uv = _bridge_dataset(kv, reference_params, n_validation)
    input_dim = model.R + 1 + 2 * exb.TIME_FREQ
    params = exb.core.init_mlp(ki, input_dim, hidden, model.R)
    state = _adam_init(params)

    def score(p, times, x, u):
        z = observable_values(model.A, model.standardization, x)
        dz = observable_rate(model.A, model.standardization, x, u)
        pred = _mlp_apply(p, _cv_features(times, z))
        mse = jnp.mean((pred - dz) ** 2)
        variance = jnp.mean((dz - jnp.mean(dz, axis=0)) ** 2) + 1e-8
        return mse / variance, (mse, variance)

    vg = jax.jit(jax.value_and_grad(lambda p: score(p, tt, xt, ut)[0]))
    for step in range(1, steps + 1):
        _, grad = vg(params)
        rate = exb.core.cosine_lr(step - 1, steps, 2e-3, 1e-4)
        params, state = _update(params, grad, state, step, rate)
    normalized, (mse, variance) = score(params, tv, xv, uv)
    return {"closure_mse": float(mse), "closure_normalized_mse": float(normalized),
            "closure_R2": float(1.0 - mse / variance)}


def make_fiber_banks(
    key: Array,
    times: Array,
    delta_t: float,
    n_particles: int,
    reference_params,
) -> dict[str, Array]:
    """Create paired fresh banks at t and t+dt; no evaluation data enter here."""
    keys = jax.random.split(key, 2 * len(times))
    x0, x1, u0, u1 = [], [], [], []
    for i, t in enumerate(times):
        xa, _ = exb.sample_bridge(keys[2 * i], t, n_particles)
        tb = jnp.minimum(t + delta_t, 1.0)
        xb, _ = exb.sample_bridge(keys[2 * i + 1], tb, n_particles)
        x0.append(xa); x1.append(xb)
        u0.append(exb.reference_velocity(reference_params, t, xa))
        u1.append(exb.reference_velocity(reference_params, tb, xb))
    return {"times": jnp.asarray(times), "x0": jnp.stack(x0), "x1": jnp.stack(x1),
            "u0": jnp.stack(u0), "u1": jnp.stack(u1)}


def fiber_objective_from_A(
    A: Array,
    standardization: Standardization,
    banks: dict[str, Array],
    delta_t: float,
    *,
    ess_floor: float = 0.20,
    penalty_scale: float = 5.0,
) -> tuple[Array, dict[str, Array]]:
    """Method-blind tangent-pushforward to next-I-projected-law objective."""
    target = jnp.zeros(A.shape[0], dtype=A.dtype)

    def one(t, x0, x1, u0, u1):
        del u1
        ph0 = observable_values(A, standardization, x0)
        ph1 = observable_values(A, standardization, x1)
        rate0 = observable_rate(A, standardization, x0, u0)
        f0 = exb.core.empirical_fiber_state(x0, u0, target, ph=ph0, jphi_u=rate0)
        # Velocity is evaluated on the calibrated q_t bank, exactly as specified.
        vtan = tangent_velocity(A, standardization, x0, u0, f0.projected_weights)
        pushed = x0 + delta_t * vtan
        # u at t+dt is immaterial to calibration, but empirical_fiber_state also
        # computes forcing, so use zeros and the corresponding zero rate here.
        zeros = jnp.zeros_like(x1)
        f1 = exb.core.empirical_fiber_state(
            x1, zeros, target, ph=ph1, jphi_u=jnp.zeros_like(ph1)
        )
        mmd2 = weighted_mmd2(pushed, f0.projected_weights, x1, f1.projected_weights)
        ess_penalty = jax.nn.softplus(20.0 * (ess_floor - f0.ess_fraction)) / 20.0
        ess_penalty += jax.nn.softplus(20.0 * (ess_floor - f1.ess_fraction)) / 20.0
        residual = f0.calibration_residual + f1.calibration_residual
        loss = mmd2 / (delta_t * delta_t) + penalty_scale * ess_penalty + 100.0 * residual
        return loss, (mmd2, f0.ess_fraction, f1.ess_fraction,
                      f0.calibration_residual, f1.calibration_residual,
                      f0.covariance_rank, f1.covariance_rank,
                      f0.covariance_condition, f1.covariance_condition)

    losses, aux = jax.vmap(one)(banks["times"], banks["x0"], banks["x1"], banks["u0"], banks["u1"])
    return jnp.mean(losses), {
        "local_mmd2": aux[0], "ess_t": aux[1], "ess_next": aux[2],
        "residual_t": aux[3], "residual_next": aux[4],
        "rank_t": aux[5], "rank_next": aux[6],
        "condition_t": aux[7], "condition_next": aux[8],
    }


def fiber_checkpoint_feasible(aux: dict[str, Array], R: int, residual_tol: float = 1e-6) -> bool:
    return bool(
        float(jnp.min(jnp.concatenate([aux["ess_t"], aux["ess_next"]]))) >= 0.20
        and float(jnp.max(jnp.concatenate([aux["residual_t"], aux["residual_next"]]))) <= residual_tol
        and int(jnp.min(jnp.concatenate([aux["rank_t"], aux["rank_next"]]))) >= R
        and bool(jnp.all(jnp.isfinite(jnp.concatenate([aux["condition_t"], aux["condition_next"]]))))
    )


def train_fiber(
    key: Array,
    standardization: Standardization,
    B0: Array,
    reference_params,
    *,
    steps: int,
    n_times: int,
    n_particles: int,
    delta_t: float,
    lr: float = 1e-3,
) -> tuple[Array, dict[str, Any]]:
    """Train A only against held-out method-blind projected-law closure."""
    kt, kv = jax.random.split(key)
    times = jnp.linspace(0.08, 0.92 - delta_t, n_times)
    train_banks = make_fiber_banks(kt, times, delta_t, n_particles, reference_params)
    val_banks = make_fiber_banks(kv, times, delta_t, n_particles, reference_params)

    def loss_fn(B):
        return fiber_objective_from_A(stiefel_rows(B), standardization, train_banks, delta_t)[0]

    vfun = jax.jit(lambda B: fiber_objective_from_A(stiefel_rows(B), standardization, val_banks, delta_t))
    vg = jax.jit(jax.value_and_grad(loss_fn))
    B = B0
    state = _adam_init(B)
    best = B
    best_val = float("inf")
    found_feasible = False
    history = []
    for step in range(1, steps + 1):
        loss, grads = vg(B)
        rate = exb.core.cosine_lr(step - 1, steps, lr, lr * 0.05)
        B, state = _update(B, grads, state, step, rate, weight_decay=0.0)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            val, aux = vfun(B)
            feasible = fiber_checkpoint_feasible(aux, B.shape[0])
            vf = float(val)
            if feasible and (not found_feasible or vf < best_val):
                found_feasible, best_val, best = True, vf, B
            history.append({
                "step": step, "train_objective": float(loss), "validation_objective": vf,
                "feasible": feasible,
                "min_ess": float(jnp.min(jnp.concatenate([aux["ess_t"], aux["ess_next"]]))),
                "max_residual": float(jnp.max(jnp.concatenate([aux["residual_t"], aux["residual_next"]]))),
            })
    if not found_feasible:
        # Retain the initialization so a pathological optimizer cannot win via
        # an infeasible projection.  The run is explicitly flagged as failed.
        best = B0
    val, aux = vfun(best)
    return stiefel_rows(best), {
        "fiber_validation_objective": float(val),
        "local_tangent_mmd2": float(jnp.mean(aux["local_mmd2"])),
        "min_ess": float(jnp.min(jnp.concatenate([aux["ess_t"], aux["ess_next"]]))),
        "max_calibration_residual": float(jnp.max(jnp.concatenate([aux["residual_t"], aux["residual_next"]]))),
        "max_condition": float(jnp.max(jnp.concatenate([aux["condition_t"], aux["condition_next"]]))),
        "feasible_checkpoint_found": found_feasible,
        "history": history,
    }


def endpoint_equivalence(
    key: Array,
    model: ObservableModel,
    n: int,
) -> dict[str, Any]:
    k0, k1 = jax.random.split(key)
    x0, x1 = exb.sample_ring(k0, n), exb.sample_four_lobes(k1, n)
    p0 = observable_values(model.A, model.standardization, x0)
    p1 = observable_values(model.A, model.standardization, x1)
    gap = jnp.mean(p0, axis=0) - jnp.mean(p1, axis=0)
    # Endpoint calibration is computed independently, purely as a diagnostic.
    zeros0, zeros1 = jnp.zeros_like(x0), jnp.zeros_like(x1)
    f0 = project_bank(model.A, model.standardization, x0, zeros0)
    f1 = project_bank(model.A, model.standardization, x1, zeros1)
    calibrated_gap = f0.projected_weights @ p0 - f1.projected_weights @ p1
    m = min(512, n)
    w = jnp.ones(m, dtype=x0.dtype) / m
    phi_mmd = jnp.sqrt(weighted_mmd2(p0[:m], w, p1[:m], w))
    angular_gap = jnp.mean(exb.angular_features(x0), axis=0) - jnp.mean(exb.angular_features(x1), axis=0)
    return {
        "mean_minus": np.asarray(jnp.mean(p0, axis=0)).tolist(),
        "mean_plus": np.asarray(jnp.mean(p1, axis=0)).tolist(),
        "expectation_gap": np.asarray(gap).tolist(),
        "expectation_gap_norm": float(jnp.linalg.norm(gap)),
        "max_abs_expectation_gap": float(jnp.max(jnp.abs(gap))),
        "calibrated_expectation_gap": np.asarray(calibrated_gap).tolist(),
        "calibrated_max_abs_gap": float(jnp.max(jnp.abs(calibrated_gap))),
        "phi_space_mmd": float(phi_mmd),
        "hidden_angular_gap": np.asarray(angular_gap).tolist(),
        "hidden_angular_gap_norm": float(jnp.linalg.norm(angular_gap)),
    }


def fit_frozen_representation_classifier(
    key: Array,
    model: ObservableModel,
    *,
    n_train: int,
    n_evaluation: int,
    steps: int = 100,
    hidden: tuple[int, ...] = (32, 32),
) -> dict[str, float]:
    kt, ke, ki = jax.random.split(key, 3)
    xt, yt = _endpoint_dataset(kt, n_train // 2)
    xe, ye = _endpoint_dataset(ke, n_evaluation // 2)
    zt = jax.lax.stop_gradient(observable_values(model.A, model.standardization, xt))
    ze = jax.lax.stop_gradient(observable_values(model.A, model.standardization, xe))
    params = exb.core.init_mlp(ki, model.R, hidden, 1)
    state = _adam_init(params)

    def loss_fn(p):
        return _binary_cross_entropy(_mlp_apply(p, zt)[..., 0], yt)

    vg = jax.jit(jax.value_and_grad(loss_fn))
    for step in range(1, steps + 1):
        _, grad = vg(params)
        params, state = _update(params, grad, state, step, jnp.asarray(1e-3))
    logits = _mlp_apply(params, ze)[..., 0]
    return classification_metrics(logits, ye)


def projection_diagnostics(
    key: Array,
    model: ObservableModel,
    reference_params,
    times: Iterable[float],
    n_particles: int,
) -> list[dict[str, Any]]:
    times = list(times)
    rows = []
    keys = jax.random.split(key, len(times))
    for k, t in zip(keys, times):
        x, _ = exb.sample_bridge(k, jnp.asarray(t), n_particles)
        u = exb.reference_velocity(reference_params, jnp.asarray(t), x)
        f = project_bank(model.A, model.standardization, x, u)
        eig = jnp.linalg.eigvalsh(0.5 * (f.covariance + f.covariance.T))
        w = f.projected_weights
        rows.append({
            "t": float(t), "calibration_residual": float(f.calibration_residual),
            "ess_fraction": float(f.ess_fraction), "weight_entropy": float(-jnp.sum(w * jnp.log(jnp.maximum(w, 1e-300)))),
            "max_weight": float(jnp.max(w)), "lambda_norm": float(jnp.linalg.norm(f.lambda_)),
            "covariance_rank": int(f.covariance_rank), "covariance_min_eigenvalue": float(jnp.min(eig)),
            "covariance_condition": float(f.covariance_condition),
            "projection_distortion": float(jnp.sum(w * jnp.log(jnp.maximum(w * n_particles, 1e-300)))),
        })
    return rows


def _generic_ritz_bank(key, model: ObservableModel, reference_params, n_times: int, n_particles: int):
    kt, kb = jax.random.split(key)
    times = exb.stratified_times(kt, n_times, lo=0.04, hi=0.96)
    keys = jax.random.split(kb, n_times)
    xs, ws, hs = [], [], []
    for k, t in zip(keys, times):
        x, _ = exb.sample_bridge(k, t, n_particles)
        u = exb.reference_velocity(reference_params, t, x)
        f = project_bank(model.A, model.standardization, x, u)
        xs.append(x); ws.append(f.projected_weights); hs.append(f.forcing)
    return {"times": times, "x": jnp.stack(xs), "weights": jnp.stack(ws), "h": jnp.stack(hs)}


def _generic_bank_ritz_loss(params, bank):
    values = jax.vmap(exb.ritz_state_loss, in_axes=(None, 0, 0, 0, 0))(
        params, bank["times"], bank["x"], bank["weights"], bank["h"]
    )
    return jnp.mean(values)


def train_downstream_ritz(
    key: Array,
    model: ObservableModel,
    reference_params,
    *,
    steps: int,
    n_times: int,
    n_particles: int,
) -> tuple[Any, dict[str, Any]]:
    """Experiment-B Deep-Ritz architecture/optimizer with generic Phi_A banks."""
    ki, kt, kv = jax.random.split(key, 3)
    input_dim = exb.STATE_DIM + 1 + 2 * exb.TIME_FREQ
    params = exb.core.init_mlp(ki, input_dim, exb.RITZ_HIDDEN, 1)
    train_bank = _generic_ritz_bank(kt, model, reference_params, n_times, n_particles)
    val_bank = _generic_ritz_bank(kv, model, reference_params, max(n_times, 3), n_particles)
    state = _adam_init(params)
    vg = jax.jit(jax.value_and_grad(lambda p: _generic_bank_ritz_loss(p, train_bank)))
    vfun = jax.jit(lambda p: _generic_bank_ritz_loss(p, val_bank))
    best, best_val = params, float(vfun(params))
    history = []
    for step in range(1, steps + 1):
        loss, grad = vg(params)
        rate = exb.core.cosine_lr(step - 1, steps, 1.5e-3, 4e-5)
        params, state = _update(params, grad, state, step, rate)
        if step == 1 or step % max(steps // 10, 1) == 0 or step == steps:
            val = float(vfun(params))
            if val < best_val:
                best, best_val = params, val
            history.append({"step": step, "train_ritz_loss": float(loss), "validation_ritz_loss": val})
    return best, {"heldout_ritz_loss": best_val, "history": history}


def rollout_methods(
    key: Array,
    model: ObservableModel,
    reference_params,
    potential_params,
    *,
    n_particles: int,
    flow_steps: int,
) -> dict[str, dict[str, Array]]:
    x0 = exb.whiten_empirical(exb.sample_ring(key, n_particles))
    learned = lambda t, x: exb.reference_velocity(reference_params, t, x) - exb.potential_grad(potential_params, t, x)
    fields = {
        "raw_si": lambda t, x: exb.reference_velocity(reference_params, t, x),
        "moment_tangent": lambda t, x: safety_velocity(model, x, exb.reference_velocity(reference_params, t, x)),
        "mfsi_learned": learned,
        "mfsi_learned_safe": lambda t, x: safety_velocity(model, x, learned(t, x)),
    }
    runs = {}
    for name, field in fields.items():
        times, trajectory = jax.jit(lambda z: exb.integrate_field(z, field, flow_steps))(x0)
        trajectory.block_until_ready()
        runs[name] = {"times": times, "trajectory": trajectory}
    return runs


def _trajectory_at(run: dict[str, Array], t: float) -> Array:
    idx = int(np.argmin(np.abs(np.asarray(run["times"]) - t)))
    return run["trajectory"][idx]


def evaluate_downstream(
    key: Array,
    model: ObservableModel,
    reference_params,
    potential_params,
    *,
    times: Iterable[float],
    n_particles: int,
    target_particles: int,
    flow_steps: int,
    local_dt: float,
) -> dict[str, Any]:
    kr, kb = jax.random.split(key)
    runs = rollout_methods(kr, model, reference_params, potential_params,
                           n_particles=n_particles, flow_steps=flow_steps)
    times = list(times)
    keys = jax.random.split(kb, 2 * len(times))
    per_method = {name: [] for name in runs}
    target_rows, local_rows = [], []
    for i, t in enumerate(times):
        x, _ = exb.sample_bridge(keys[2 * i], jnp.asarray(t), target_particles)
        u = exb.reference_velocity(reference_params, jnp.asarray(t), x)
        f = project_bank(model.A, model.standardization, x, u)
        m = min(384, target_particles, n_particles)
        xt, wt = x[:m], f.projected_weights[:m]
        wt = wt / jnp.sum(wt)
        target_ang = f.projected_weights @ exb.angular_features(x)
        target_rows.append({"t": t, "ess_fraction": float(f.ess_fraction),
                            "condition": float(f.covariance_condition),
                            "calibration_residual": float(f.calibration_residual)})
        for name, run in runs.items():
            y = _trajectory_at(run, t)
            ph_mean = jnp.mean(observable_values(model.A, model.standardization, y), axis=0)
            mmd = jnp.sqrt(weighted_mmd2(xt, wt, y[:m]))
            angular_error = jnp.linalg.norm(jnp.mean(exb.angular_features(y), axis=0) - target_ang)
            per_method[name].append({"t": t, "mmd": float(mmd),
                                     "max_moment_error": float(jnp.max(jnp.abs(ph_mean))),
                                     "mean_moment_error": float(jnp.mean(jnp.abs(ph_mean))),
                                     "angular_error": float(angular_error)})
        if t + local_dt <= 1.0:
            tn = t + local_dt
            xn, _ = exb.sample_bridge(keys[2 * i + 1], jnp.asarray(tn), target_particles)
            un = exb.reference_velocity(reference_params, jnp.asarray(tn), xn)
            fn = project_bank(model.A, model.standardization, xn, un)
            vtan = tangent_velocity(model.A, model.standardization, x, u, f.projected_weights)
            vmfsi = u - exb.potential_grad(potential_params, jnp.asarray(t), x)
            tan_push = x + local_dt * vtan
            mfsi_push = x + local_dt * vmfsi
            gap2 = jnp.sum(f.projected_weights * jnp.sum((vmfsi - vtan) ** 2, axis=-1))
            uenergy = jnp.sum(f.projected_weights * jnp.sum(u * u, axis=-1))
            ctan, cmfsi = vtan - u, vmfsi - u
            dot = jnp.sum(f.projected_weights * jnp.sum(ctan * cmfsi, axis=-1))
            et = jnp.sum(f.projected_weights * jnp.sum(ctan * ctan, axis=-1))
            em = jnp.sum(f.projected_weights * jnp.sum(cmfsi * cmfsi, axis=-1))
            local_rows.append({
                "t": t,
                "tangent_next_mmd": float(jnp.sqrt(weighted_mmd2(tan_push[:m], f.projected_weights[:m], xn[:m], fn.projected_weights[:m]))),
                "mfsi_next_mmd": float(jnp.sqrt(weighted_mmd2(mfsi_push[:m], f.projected_weights[:m], xn[:m], fn.projected_weights[:m]))),
                "velocity_gap_mse": float(gap2), "velocity_gap_rms": float(jnp.sqrt(gap2)),
                "normalized_velocity_gap": float(gap2 / jnp.maximum(uenergy, 1e-12)),
                "correction_cosine": float(dot / jnp.sqrt(jnp.maximum(et * em, 1e-24))),
                "tangent_correction_energy": float(et), "mfsi_correction_energy": float(em),
            })
    summary = {}
    for name, rows in per_method.items():
        interior = [r for r in rows if r["t"] > 0.0 and r["t"] < 1.0]
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
        "rms_velocity_gap": float(np.sqrt(np.mean([r["velocity_gap_mse"] for r in local_rows]))),
        "mean_normalized_velocity_gap": float(np.mean([r["normalized_velocity_gap"] for r in local_rows])),
    }
    return {"summary": summary, "per_method": per_method, "target": target_rows,
            "local": local_rows, "local_summary": local_summary}


def rotated_endpoint_diagnostics(
    key: Array,
    model: ObservableModel,
    angles: Iterable[float],
    n: int,
) -> list[dict[str, Any]]:
    """Held-out geometric feasibility with A frozen and target recomputed."""
    angles = list(angles)
    rows = []
    keys = jax.random.split(key, 2 * len(angles))
    for i, angle in enumerate(angles):
        ca, sa = np.cos(angle), np.sin(angle)
        rot = jnp.asarray([[ca, -sa], [sa, ca]], dtype=jnp.float64)
        x0 = exb.sample_ring(keys[2 * i], n) @ rot.T
        x1 = exb.sample_four_lobes(keys[2 * i + 1], n) @ rot.T
        p0 = observable_values(model.A, model.standardization, x0)
        p1 = observable_values(model.A, model.standardization, x1)
        target = 0.5 * (jnp.mean(p0, axis=0) + jnp.mean(p1, axis=0))
        zeros0, zeros1 = jnp.zeros_like(x0), jnp.zeros_like(x1)
        r0 = observable_rate(model.A, model.standardization, x0, zeros0)
        r1 = observable_rate(model.A, model.standardization, x1, zeros1)
        f0 = exb.core.empirical_fiber_state(x0, zeros0, target, ph=p0, jphi_u=r0)
        f1 = exb.core.empirical_fiber_state(x1, zeros1, target, ph=p1, jphi_u=r1)
        rows.append({"angle": float(angle), "target": np.asarray(target).tolist(),
                     "endpoint_gap_norm": float(jnp.linalg.norm(jnp.mean(p0, axis=0) - jnp.mean(p1, axis=0))),
                     "min_endpoint_ess": float(jnp.minimum(f0.ess_fraction, f1.ess_fraction)),
                     "max_endpoint_condition": float(jnp.maximum(f0.covariance_condition, f1.covariance_condition)),
                     "max_calibration_residual": float(jnp.maximum(f0.calibration_residual, f1.calibration_residual))})
    return rows


def save_observable(path: Path, objective: str, model: ObservableModel, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, objective=np.asarray(objective), A=np.asarray(model.A),
             center=np.asarray(model.standardization.center), whitening=np.asarray(model.standardization.whitening),
             covariance_eigenvalues=np.asarray(model.standardization.covariance_eigenvalues),
             raw_coefficients=np.asarray(model.raw_coefficients), raw_intercept=np.asarray(model.raw_intercept),
             singular_values=np.asarray(jnp.linalg.svd(model.A, compute_uv=False)),
             metadata_json=np.asarray(json.dumps(metadata)))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def make_figures(out: Path, results: dict[str, Any]) -> None:
    """Generate the seven prespecified figure families when data are available."""
    names = [n for n in OBJECTIVES if n in results["objectives"]]
    if not names:
        return
    coefficients = np.asarray([results["objectives"][n]["raw_coefficients"] for n in names])
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), squeeze=False)
    for ax, name, coeff in zip(axes[0], names, coefficients):
        im = ax.imshow(coeff, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(RAW_DIM), BASIS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(coeff.shape[0]), [f"row {i+1}" for i in range(coeff.shape[0])])
        ax.set_title(name.upper()); fig.colorbar(im, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(out / "figure1_learned_observables.png", dpi=180); plt.close(fig)

    # Endpoint equivalence / ambiguity.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(names, [results["objectives"][n]["endpoint"]["expectation_gap_norm"] for n in names])
    axes[0].set_title("Endpoint expectation gap")
    axes[1].bar(names, [results["objectives"][n]["endpoint_classifier"]["auroc"] for n in names])
    axes[1].set_ylim(.45, 1.0); axes[1].set_title("Frozen-Phi endpoint AUROC")
    axes[2].bar(names, [results["objectives"][n]["endpoint"]["hidden_angular_gap_norm"] for n in names])
    axes[2].set_title("Held-out angular gap")
    fig.tight_layout(); fig.savefig(out / "figure2_endpoint_ambiguity.png", dpi=180); plt.close(fig)

    for figure, key, title, filename in (
        (3, "projection", "Fiber geometry", "figure3_fiber_geometry.png"),
        (4, "local", "Local law closure", "figure4_local_law_closure.png"),
    ):
        del figure
        fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), squeeze=False)
        for ax, name in zip(axes[0], names):
            if key == "projection":
                rows = results["objectives"][name][key]
                ax.plot([r["t"] for r in rows], [r["ess_fraction"] for r in rows], label="ESS")
                ax.plot([r["t"] for r in rows], [r["projection_distortion"] for r in rows], label="KL distortion")
            else:
                rows = results["objectives"][name]["downstream"][key]
                ax.plot([r["t"] for r in rows], [r["tangent_next_mmd"] for r in rows], label="tangent")
                ax.plot([r["t"] for r in rows], [r["mfsi_next_mmd"] for r in rows], label="MFSI")
            ax.set_title(name.upper()); ax.set_xlabel("t"); ax.legend(fontsize=7)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(out / filename, dpi=180); plt.close(fig)

    # Figures 5 and 6 use the stored path metrics rather than large particle clouds.
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        pm = results["objectives"][name]["downstream"]["per_method"]
        for method in ("moment_tangent", "mfsi_learned_safe"):
            ax.plot([r["t"] for r in pm[method]], [r["mmd"] for r in pm[method]], marker="o", label=method)
        ax.set_title(name.upper()); ax.set_xlabel("t"); ax.set_ylabel("projected-law MMD"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "figure5_full_rollout.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for method in ("moment_tangent", "mfsi_learned_safe"):
        axes[0].plot(names, [results["objectives"][n]["downstream"]["summary"][method]["mean_interior_mmd"] for n in names], marker="o", label=method)
    axes[0].set_title("Interior law MMD"); axes[0].legend(fontsize=7)
    axes[1].bar(names, [results["objectives"][n]["downstream"]["summary"]["mfsi_learned_safe"]["max_moment_error"] for n in names])
    axes[1].set_title("Maximum moment error")
    axes[2].bar(names, [results["objectives"][n]["downstream"]["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"] for n in names])
    axes[2].set_title("Held-out angular error")
    fig.tight_layout(); fig.savefig(out / "figure6_full_law_metrics.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for name in names:
        rows = results["objectives"][name]["robustness"]
        ax.plot([r["angle"] for r in rows], [r["min_endpoint_ess"] for r in rows], marker="o", label=name.upper())
    ax.set_xlabel("shared endpoint rotation (radians)"); ax.set_ylabel("minimum endpoint ESS"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "figure7_robustness.png", dpi=180); plt.close(fig)
