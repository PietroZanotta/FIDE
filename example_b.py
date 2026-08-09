#!/usr/bin/env python3
"""MFSI Experiment B: equal-covariance ring -> four-lobed transport in 2D.

This is the first non-oracle law-path experiment from the draft.  The measured
observables are first and second raw moments,

    Phi(x,y) = (x, y, x^2, xy, y^2),

with target c=(0,0,1,0,1).  The endpoints are smooth distributions with exactly
these population moments:

* Q_minus: centers uniformly distributed on a ring + isotropic Gaussian noise;
* Q_plus: four equally weighted centers on the coordinate axes + the same noise.

No symmetry assumption is used by either neural network or by the MFSI solver.
The four-fold structure is used only as a *held-out diagnostic* (cos 4 theta,
etc.), exactly as intended in the paper draft.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

import mfsi_components as core
from backend_runtime import TesseractRESTBackend, normalize_backend

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "example_b"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "learned_mfsi_example_b.npz"

STATE_DIM = 2
TARGET = jnp.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float64)
RING_RADIUS = 1.30
NOISE_STD = float(np.sqrt(1.0 - 0.5 * RING_RADIUS**2))
REFERENCE_HIDDEN = core.REFERENCE_HIDDEN
RITZ_HIDDEN = core.RITZ_HIDDEN
TIME_FREQ = core.TIME_FOURIER_FREQUENCIES

HELDOUT_TIMES = np.linspace(0.0, 1.0, 11)


# -----------------------------------------------------------------------------
# Endpoints and observables
# -----------------------------------------------------------------------------

def sample_ring(key, n):
    """Smooth ring-like Q_minus with covariance exactly I at population level."""
    ka, kz = jax.random.split(key)
    theta = 2.0 * jnp.pi * jax.random.uniform(ka, (n,), dtype=jnp.float64)
    centers = RING_RADIUS * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return centers + NOISE_STD * jax.random.normal(kz, (n, 2), dtype=jnp.float64)


def sample_four_lobes(key, n):
    """Smooth four-lobed Q_plus with the same population mean/covariance."""
    kk, kz = jax.random.split(key)
    labels = jax.random.randint(kk, (n,), 0, 4)
    theta = 0.5 * jnp.pi * labels.astype(jnp.float64)
    centers = RING_RADIUS * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    return centers + NOISE_STD * jax.random.normal(kz, (n, 2), dtype=jnp.float64)


def sample_bridge(key, t, n):
    k0, k1 = jax.random.split(key)
    x0 = sample_ring(k0, n)
    x1 = sample_four_lobes(k1, n)
    x = (1.0 - t) * x0 + t * x1
    return x, x1 - x0


def sample_bridge_times(key, times):
    """One independent endpoint pair for every supplied time."""
    n = times.shape[0]
    k0, k1 = jax.random.split(key)
    x0 = sample_ring(k0, n)
    x1 = sample_four_lobes(k1, n)
    return (1.0 - times[:, None]) * x0 + times[:, None] * x1, x1 - x0


def phi(x):
    xx, yy = x[..., 0], x[..., 1]
    return jnp.stack([xx, yy, xx * xx, xx * yy, yy * yy], axis=-1)


def jphi(x):
    xx, yy = x[..., 0], x[..., 1]
    z = jnp.zeros_like(xx)
    o = jnp.ones_like(xx)
    # (..., R=5, d=2)
    return jnp.stack([
        jnp.stack([o, z], axis=-1),
        jnp.stack([z, o], axis=-1),
        jnp.stack([2.0 * xx, z], axis=-1),
        jnp.stack([yy, xx], axis=-1),
        jnp.stack([z, 2.0 * yy], axis=-1),
    ], axis=-2)


def jphi_u(x, u):
    return jnp.einsum("...rd,...d->...r", jphi(x), u)


def empirical_moments(x):
    return jnp.mean(phi(x), axis=0)


def whiten_empirical(x):
    """Affine finite-ensemble normalization; not used in endpoint training data."""
    x = x - jnp.mean(x, axis=0)
    cov = (x.T @ x) / x.shape[0]
    vals, vecs = jnp.linalg.eigh(0.5 * (cov + cov.T))
    invsqrt = vecs @ jnp.diag(1.0 / jnp.sqrt(jnp.maximum(vals, 1e-12))) @ vecs.T
    return x @ invsqrt.T


# -----------------------------------------------------------------------------
# Dimension-generic neural fields
# -----------------------------------------------------------------------------

def features(t, x):
    x = jnp.asarray(x)
    batch_shape = x.shape[:-1]
    tt = jnp.broadcast_to(jnp.asarray(t, dtype=x.dtype), batch_shape)
    return jnp.concatenate([x, core.time_fourier_features(tt, TIME_FREQ)], axis=-1)


def reference_velocity(params, t, x):
    return core.mlp_apply(params, features(t, x))


def potential(params, t, x):
    return core.mlp_apply(params, features(t, x))[..., 0]


def potential_single(params, t, x):
    return potential(params, t, x[None, :])[0]


def potential_grad(params, t, x):
    # The MLP is pointwise across rows. Differentiating the summed scalar output
    # therefore returns one spatial gradient per row without materializing a
    # batch Jacobian or launching a vmap of reverse passes.
    return jax.grad(lambda xx: jnp.sum(potential(params, t, xx)))(x)


def learned_velocity(model, t, x):
    return reference_velocity(model[0], t, x) - potential_grad(model[1], t, x)


def _global_clip(tree, max_norm=5.0):
    leaves = jax.tree.leaves(tree)
    norm = jnp.sqrt(sum(jnp.sum(z * z) for z in leaves))
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-30))
    return jax.tree.map(lambda z: z * scale, tree)


def stratified_times(key, n, lo=0.01, hi=0.99):
    jitter = jax.random.uniform(key, (n,), dtype=jnp.float64)
    return lo + (hi - lo) * (jnp.arange(n, dtype=jnp.float64) + jitter) / n


# -----------------------------------------------------------------------------
# Learned reference flow
# -----------------------------------------------------------------------------

def train_reference(key, *, steps=1800, batch_size=3072, eval_every=100):
    input_dim = STATE_DIM + 1 + 2 * TIME_FREQ
    key, ki, kh_t, kh_b = jax.random.split(key, 4)
    params = core.init_mlp(ki, input_dim, REFERENCE_HIDDEN, output_dim=STATE_DIM)
    m = core._tree_zeros_like(params)
    v = core._tree_zeros_like(params)

    hold_t = stratified_times(kh_t, 8192)
    hold_x, hold_target = sample_bridge_times(kh_b, hold_t)

    def loss_fn(p, k):
        kt, kb = jax.random.split(k)
        t = stratified_times(kt, batch_size)
        x, target = sample_bridge_times(kb, t)
        pred = reference_velocity(p, t, x)
        return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))

    def hold_loss(p):
        pred = reference_velocity(p, hold_t, hold_x)
        return jnp.mean(jnp.sum((pred - hold_target) ** 2, axis=-1))

    vg = jax.jit(jax.value_and_grad(loss_fn))
    hfun = jax.jit(hold_loss)
    best_params = params
    best_hold = float("inf")
    hist = []
    for i in range(1, steps + 1):
        key, sub = jax.random.split(key)
        loss, grads = vg(params, sub)
        grads = _global_clip(grads, 5.0)
        lr = core.cosine_lr(i - 1, steps, 2e-3, 7e-5)
        params, m, v = core._adamw_update(params, grads, m, v, i, lr, 1e-6)
        if i == 1 or i % eval_every == 0 or i == steps:
            hv = float(hfun(params))
            if hv < best_hold:
                best_hold = hv
                best_params = params
            hist.append({"step": i, "train_loss": float(loss), "holdout_loss": hv, "lr": float(lr)})
    zero = float(jnp.mean(jnp.sum(hold_target**2, axis=-1)))
    return best_params, hist, {"fm_mse": best_hold, "zero_predictor_mse": zero, "ratio": best_hold / zero}


# -----------------------------------------------------------------------------
# Empirical projection and Deep-Ritz
# -----------------------------------------------------------------------------

def projected_batch(key, t, n, reference_params):
    x, _ = sample_bridge(key, t, n)
    u = reference_velocity(reference_params, t, x)
    fib = core.empirical_fiber_state(x, u, TARGET, ph=phi(x), jphi_u=jphi_u(x, u))
    return x, u, fib


def ritz_state_loss(params, t, x, w, h):
    psi = potential(params, t, x)
    psi = psi - w @ psi
    gp = potential_grad(params, t, x)
    h = h - w @ h
    return 0.5 * jnp.sum(w * jnp.sum(gp * gp, axis=-1)) + jnp.sum(w * h * psi)


def build_bank(key, reference_params, n_times, particles_per_time, lo=0.03, hi=0.97):
    kt, kb = jax.random.split(key)
    times = stratified_times(kt, n_times, lo=lo, hi=hi)
    keys = jax.random.split(kb, n_times)

    def one(k, t):
        x, u, fib = projected_batch(k, t, particles_per_time, reference_params)
        return (x, u, fib.projected_weights, fib.forcing, fib.calibration_residual,
                fib.ess_fraction, fib.covariance_rank, fib.covariance_condition)

    x, u, w, h, cres, ess, rank, cond = jax.vmap(one)(keys, times)
    return {"times": times, "x": x, "u": u, "weights": w, "h": h,
            "calibration_residual": cres, "ess": ess, "rank": rank, "condition": cond}


def bank_ritz_loss(params, bank):
    vals = jax.vmap(ritz_state_loss, in_axes=(None, 0, 0, 0, 0))(
        params, bank["times"], bank["x"], bank["weights"], bank["h"]
    )
    return jnp.mean(vals)


def train_ritz(
    key, reference_params, *, steps=1600, n_times=10, particles_per_time=384,
    pool_size=8, refresh_every=250, eval_every=100, lbfgs_maxiter=2,
    init_params=None, endpoint_penalty=0.05, endpoint_particles=384,
):
    input_dim = STATE_DIM + 1 + 2 * TIME_FREQ
    key, ki, kh, ke0, ke1 = jax.random.split(key, 5)
    params = core.init_mlp(ki, input_dim, RITZ_HIDDEN, output_dim=1) if init_params is None else init_params
    endpoint_x0 = sample_ring(ke0, endpoint_particles)
    endpoint_x1 = sample_four_lobes(ke1, endpoint_particles)

    def endpoint_loss(p):
        g0 = potential_grad(p, jnp.asarray(0.0), endpoint_x0)
        g1 = potential_grad(p, jnp.asarray(1.0), endpoint_x1)
        return 0.5 * (jnp.mean(jnp.sum(g0 * g0, axis=-1)) + jnp.mean(jnp.sum(g1 * g1, axis=-1)))
    m = core._tree_zeros_like(params)
    v = core._tree_zeros_like(params)

    build = jax.jit(lambda k: build_bank(k, reference_params, n_times, particles_per_time))
    hold_bank = build_bank(kh, reference_params, 12, 512, lo=0.05, hi=0.95)
    hold_loss = jax.jit(lambda p: bank_ritz_loss(p, hold_bank) + endpoint_penalty * endpoint_loss(p))

    def make_pool(k):
        ks = jax.random.split(k, pool_size)
        banks = [build(kk) for kk in ks]
        for b in banks:
            jax.tree.map(lambda z: z.block_until_ready() if hasattr(z, "block_until_ready") else z, b)
        return jax.tree.map(lambda *zs: jnp.stack(zs, axis=0), *banks)

    key, kp = jax.random.split(key)
    pool = make_pool(kp)

    def pooled_loss(p, pool_data, idx):
        bank = jax.tree.map(lambda z: z[idx], pool_data)
        return bank_ritz_loss(p, bank) + endpoint_penalty * endpoint_loss(p)

    vg = jax.jit(jax.value_and_grad(pooled_loss))
    best_params = params
    best_hold = float(hold_loss(params))
    hist = []
    for i in range(1, steps + 1):
        if refresh_every > 0 and i > 1 and (i - 1) % refresh_every == 0:
            key, kp = jax.random.split(key)
            pool = make_pool(kp)
        key, kidx = jax.random.split(key)
        idx = jax.random.randint(kidx, (), 0, pool_size)
        loss, grads = vg(params, pool, idx)
        grads = _global_clip(grads, 5.0)
        lr = core.cosine_lr(i - 1, steps, 1.5e-3, 4e-5)
        params, m, v = core._adamw_update(params, grads, m, v, i, lr, 1e-7)
        if i == 1 or i % eval_every == 0 or i == steps:
            hv = float(hold_loss(params))
            if hv < best_hold:
                best_hold = hv
                best_params = params
            hist.append({"step": i, "train_loss": float(loss), "holdout_loss": hv, "lr": float(lr)})

    polish = {"used": False}
    if lbfgs_maxiter > 0:
        import scipy.optimize
        from jax.flatten_util import ravel_pytree
        key, kpol, kval = jax.random.split(key, 3)
        polish_bank = build_bank(kpol, reference_params, 14, 512, lo=0.04, hi=0.96)
        validation_bank = build_bank(kval, reference_params, 14, 512, lo=0.04, hi=0.96)
        flat0, unravel = ravel_pytree(best_params)
        fvg = jax.jit(jax.value_and_grad(lambda z: bank_ritz_loss(unravel(z), polish_bank) + endpoint_penalty * endpoint_loss(unravel(z))))
        before_val = float(bank_ritz_loss(best_params, validation_bank) + endpoint_penalty * endpoint_loss(best_params))

        def fun(z):
            val, grad = fvg(jnp.asarray(z))
            return float(val), np.asarray(grad, dtype=np.float64)

        res = scipy.optimize.minimize(fun, np.asarray(flat0), jac=True, method="L-BFGS-B",
                                      options={"maxiter": lbfgs_maxiter, "maxls": 15, "ftol": 1e-11})
        candidate = unravel(jnp.asarray(res.x))
        after_val = float(bank_ritz_loss(candidate, validation_bank) + endpoint_penalty * endpoint_loss(candidate))
        accepted = after_val < before_val
        if accepted:
            best_params = candidate
        polish = {"used": True, "accepted": bool(accepted), "iterations": int(res.nit),
                  "validation_before": before_val, "validation_after": after_val,
                  "message": str(res.message)}
    return best_params, hist, {"heldout_ritz_loss": best_hold, "polish": polish}


# -----------------------------------------------------------------------------
# Generic diagnostics and dynamics
# -----------------------------------------------------------------------------

def weak_form_residual(params, t, x, w, h, n_tests=24):
    gp = potential_grad(params, t, x)
    h = h - w @ h
    mean = w @ x
    xc = x - mean
    scale = jnp.sqrt(jnp.sum(w * jnp.sum(xc * xc, axis=-1)) / STATE_DIM + 1e-8)
    angles = jnp.linspace(0.0, jnp.pi, n_tests, endpoint=False)
    freq = jnp.exp(jnp.linspace(jnp.log(0.35), jnp.log(3.5), n_tests)) / scale
    direction = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    avec = freq[:, None] * direction
    phase = jnp.linspace(-1.0, 1.0, n_tests)
    z = x @ avec.T + phase[None, :]
    test = jnp.tanh(z)
    sech2 = 1.0 - test * test
    grad_test = sech2[..., None] * avec[None, :, :]
    lhs = jnp.einsum("nd,nkd->nk", gp, grad_test)
    residual = jnp.sum(w[:, None] * (lhs + h[:, None] * test), axis=0)
    a = jnp.sqrt(jnp.sum(w[:, None] * lhs * lhs, axis=0))
    b = jnp.sqrt(jnp.sum(w[:, None] * (h[:, None] * test) ** 2, axis=0))
    nr = residual / (a + b + 1e-10)
    return jnp.sqrt(jnp.mean(nr * nr))


def weighted_mmd(x, wx, y):
    wy = jnp.ones((y.shape[0],), dtype=y.dtype) / y.shape[0]
    mean = wx @ x
    var = jnp.sum(wx * jnp.sum((x - mean) ** 2, axis=-1)) / STATE_DIM + 1e-6
    base = jnp.sqrt(var)
    bws = base * jnp.array([0.5, 1.0, 2.0], dtype=x.dtype)
    dxx = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    dyy = jnp.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    dxy = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    def one(s):
        kxx = jnp.exp(-0.5 * dxx / (s * s))
        kyy = jnp.exp(-0.5 * dyy / (s * s))
        kxy = jnp.exp(-0.5 * dxy / (s * s))
        return wx @ (kxx @ wx) + wy @ (kyy @ wy) - 2.0 * wx @ (kxy @ wy)
    return jnp.sqrt(jnp.maximum(jnp.mean(jax.vmap(one)(bws)), 0.0))


def angular_features(x):
    th = jnp.arctan2(x[..., 1], x[..., 0])
    return jnp.stack([jnp.cos(4 * th), jnp.sin(4 * th),
                      jnp.cos(8 * th), jnp.sin(8 * th)], axis=-1)


def safety_velocity(x, velocity):
    jp = jphi(x)  # N,R,D
    rate = jnp.mean(jnp.einsum("nrd,nd->nr", jp, velocity), axis=0)
    G = jnp.einsum("nrd,nsd->rs", jp, jp) / x.shape[0]
    coeff, _, _ = core._stable_cov_solve(G, rate, damping=1e-10)
    return velocity - jnp.einsum("nrd,r->nd", jp, coeff)


def integrate_field(x0, velocity_fn, n_steps=160):
    times = jnp.linspace(0.0, 1.0, n_steps + 1)
    dt = 1.0 / n_steps
    def step(x, i):
        t0, t1 = times[i], times[i + 1]
        v0 = velocity_fn(t0, x)
        xp = x + dt * v0
        v1 = velocity_fn(t1, xp)
        xn = x + 0.5 * dt * (v0 + v1)
        return xn, xn
    _, tail = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return times, jnp.concatenate([x0[None, ...], tail], axis=0)


def simulate_mgd_style(x0, key, n_steps=800, sigma=1.0):
    """MGD-style constant-moment predictor/corrector in 2D.

    Example B is endpoint-to-endpoint transport, whereas MGD is not.  This
    baseline therefore uses the draft's intended comparison object: MGD-style
    moment guidance with the constant first/second-moment trajectory c.
    """
    dt = jnp.asarray(1.0 / n_steps, dtype=x0.dtype)
    sigma = jnp.asarray(sigma, dtype=x0.dtype)
    keys = jax.random.split(key, n_steps)

    def step(x, k):
        noise = jax.random.normal(k, x.shape, dtype=x.dtype)
        y = x + jnp.sqrt(2.0 * dt) * sigma * noise
        jp = jphi(y)
        G = jnp.einsum("nrd,nsd->rs", jp, jp) / y.shape[0]
        err = empirical_moments(y) - TARGET
        theta, _, _ = core._stable_cov_solve(G, err / (dt * sigma * sigma), damping=1e-7)
        xn = y - dt * sigma * sigma * jnp.einsum("nrd,r->nd", jp, theta)
        return xn, xn

    _, tail = jax.lax.scan(step, x0, keys)
    times = jnp.linspace(0.0, 1.0, n_steps + 1)
    return times, jnp.concatenate([x0[None, ...], tail], axis=0)


# -----------------------------------------------------------------------------
# Tesseract kernel parity for 2D payloads
# -----------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def tesseract_parity(model):
    ref_api = _load_module("exb_ref_api", ROOT / "tesseracts/reference_transport/tesseract_api.py")
    fib_api = _load_module("exb_fib_api", ROOT / "tesseracts/moment_fiber_realizer/tesseract_api.py")
    key = jax.random.PRNGKey(811)
    t = jnp.asarray(0.43)
    x, _ = sample_bridge(key, t, 384)
    u = reference_velocity(model[0], t, x)
    ph = phi(x); jpu = jphi_u(x, u)
    p1 = {"x": x, "t": t, "velocity_params": core.flatten_mlp(model[0])}
    p2 = {"x": x, "t": t, "velocity": u, "phi_values": ph, "jphi_u": jpu,
          "target": TARGET, "log_base_weights": jnp.zeros((x.shape[0],), dtype=x.dtype),
          "potential_params": core.flatten_mlp(model[1])}
    o1 = ref_api.apply_jax(p1)
    o2 = fib_api.apply_jax(p2)
    fib = core.empirical_fiber_state(x, u, TARGET, ph=ph, jphi_u=jpu)
    corr = -potential_grad(model[1], t, x)
    diffs = {
        "reference_velocity": float(jnp.max(jnp.abs(o1["velocity"] - u))),
        "lambda": float(jnp.max(jnp.abs(o2["lambda_value"] - fib.lambda_))),
        "forcing": float(jnp.max(jnp.abs(o2["forcing"] - fib.forcing))),
        "correction": float(jnp.max(jnp.abs(o2["correction"] - corr))),
        "velocity": float(jnp.max(jnp.abs(o2["velocity"] - (u + corr)))),
    }
    return diffs


# -----------------------------------------------------------------------------
# Training, benchmark, outputs
# -----------------------------------------------------------------------------

def save_model(model):
    np.savez(MODEL_PATH,
             reference_params=np.asarray(core.flatten_mlp(model[0])),
             potential_params=np.asarray(core.flatten_mlp(model[1])),
             state_dim=np.array(STATE_DIM, dtype=np.int32))


def unflatten(flat, hidden, output_dim):
    input_dim = STATE_DIM + 1 + 2 * TIME_FREQ
    dims = (input_dim, *tuple(hidden), output_dim)
    params, off = [], 0
    for din, dout in zip(dims[:-1], dims[1:]):
        nw = din * dout
        W = flat[off:off + nw].reshape(din, dout); off += nw
        b = flat[off:off + dout]; off += dout
        params.append((W, b))
    if off != flat.size:
        raise ValueError(f"parameter length mismatch: consumed {off}, have {flat.size}")
    return tuple(params)


def load_model():
    d = np.load(MODEL_PATH)
    return (unflatten(jnp.asarray(d["reference_params"]), REFERENCE_HIDDEN, STATE_DIM),
            unflatten(jnp.asarray(d["potential_params"]), RITZ_HIDDEN, 1))


def endpoint_check(key, n=200000):
    k0, k1 = jax.random.split(key)
    x0, x1 = sample_ring(k0, n), sample_four_lobes(k1, n)
    return {
        "ring_moments": np.asarray(jnp.mean(phi(x0), axis=0)).tolist(),
        "four_lobe_moments": np.asarray(jnp.mean(phi(x1), axis=0)).tolist(),
        "target": np.asarray(TARGET).tolist(),
        "ring_angular": np.asarray(jnp.mean(angular_features(x0), axis=0)).tolist(),
        "four_lobe_angular": np.asarray(jnp.mean(angular_features(x1), axis=0)).tolist(),
    }


def reference_holdout(key, params, n=12000):
    kt, kb = jax.random.split(key)
    t = stratified_times(kt, n)
    x, tv = sample_bridge_times(kb, t)
    pred = reference_velocity(params, t, x)
    mse = jnp.mean(jnp.sum((pred - tv) ** 2, axis=-1))
    zero = jnp.mean(jnp.sum(tv * tv, axis=-1))
    return {"fm_mse": float(mse), "zero_predictor_mse": float(zero), "ratio": float(mse / zero)}


def projection_diagnostics(key, model, times, n_particles=4096):
    keys = jax.random.split(key, len(times))
    rows = []
    for k, tf in zip(keys, times):
        t = jnp.asarray(tf)
        x, _, fib = projected_batch(k, t, n_particles, model[0])
        weak = weak_form_residual(model[1], t, x, fib.projected_weights, fib.forcing)
        rows.append({
            "t": float(tf), "calibration_residual": float(fib.calibration_residual),
            "ess_fraction": float(fib.ess_fraction), "rank": int(fib.covariance_rank),
            "condition": float(fib.covariance_condition), "weak_form_residual": float(weak),
        })
    return rows


def _select_trajectory(traj, times_nodes, t):
    idx = int(np.argmin(np.abs(np.asarray(times_nodes) - float(t))))
    return traj[idx]


def _integrate_backend_field(x0, field_fn, n_steps):
    """Heun integration for an external (REST/Tesseract) velocity field."""
    times = np.linspace(0.0, 1.0, n_steps + 1)
    dt = 1.0 / n_steps
    x = np.asarray(x0, dtype=np.float64)
    traj = [x.copy()]
    for i in range(n_steps):
        v0 = np.asarray(field_fn(float(times[i]), x), dtype=np.float64)
        xp = x + dt * v0
        v1 = np.asarray(field_fn(float(times[i + 1]), xp), dtype=np.float64)
        x = x + 0.5 * dt * (v0 + v1)
        traj.append(x.copy())
    return times, np.stack(traj, axis=0)


def _remote_fiber_state(client, model, t, x, u):
    ph = np.asarray(phi(jnp.asarray(x)))
    jpu = np.asarray(jphi_u(jnp.asarray(x), jnp.asarray(u)))
    out = client.fiber_apply(
        x=x, t=float(t), velocity=u, phi_values=ph, jphi_u=jpu,
        target=np.asarray(TARGET), log_base_weights=np.zeros(x.shape[0]),
        potential_params=np.asarray(core.flatten_mlp(model[1])),
    )
    return SimpleNamespace(
        projected_weights=np.asarray(out["projected_weights"]),
        calibration_residual=float(out["calibration_residual"]),
        ess_fraction=float(out["ess_fraction"]),
        covariance_rank=int(out["covariance_rank"]),
        covariance_condition=float(out["covariance_condition"]),
        velocity=np.asarray(out["velocity"]),
        correction=np.asarray(out["correction"]),
    )


def benchmark(key, model, *, n_particles=3072, flow_steps=160, mgd_steps=800, target_bank=4096, backend="jax"):
    backend = normalize_backend(backend)
    k0, kmgd, ktarg = jax.random.split(key, 3)
    x0 = whiten_empirical(sample_ring(k0, n_particles))

    runs = {}
    if backend == "jax":
        raw_fn = lambda t, x: reference_velocity(model[0], t, x)
        tan_fn = lambda t, x: safety_velocity(x, reference_velocity(model[0], t, x))
        mfsi_fn = lambda t, x: learned_velocity(model, t, x)
        safe_fn = lambda t, x: safety_velocity(x, learned_velocity(model, t, x))
        for name, fn in (("raw_si", raw_fn), ("moment_tangent", tan_fn),
                         ("mfsi_learned", mfsi_fn), ("mfsi_learned_safe", safe_fn)):
            start = perf_counter()
            tn, tr = jax.jit(lambda z: integrate_field(z, fn, flow_steps))(x0)
            tr.block_until_ready()
            runs[name] = {"times": tn, "traj": tr, "runtime_s": perf_counter() - start}
    else:
        client = TesseractRESTBackend.from_env()
        ref_flat = np.asarray(core.flatten_mlp(model[0]))

        def ref_remote(t, x):
            return client.reference_velocity(ref_flat, t, x)

        def tangent_remote(t, x):
            u = ref_remote(t, x)
            return np.asarray(safety_velocity(jnp.asarray(x), jnp.asarray(u)))

        def mfsi_remote(t, x):
            u = ref_remote(t, x)
            return _remote_fiber_state(client, model, t, x, u).velocity

        def safe_remote(t, x):
            v = mfsi_remote(t, x)
            return np.asarray(safety_velocity(jnp.asarray(x), jnp.asarray(v)))

        for name, fn in (("raw_si", ref_remote), ("moment_tangent", tangent_remote),
                         ("mfsi_learned", mfsi_remote), ("mfsi_learned_safe", safe_remote)):
            start = perf_counter()
            tn, tr = _integrate_backend_field(np.asarray(x0), fn, flow_steps)
            runs[name] = {"times": tn, "traj": tr, "runtime_s": perf_counter() - start}

    start = perf_counter()
    tm, mgd_tr = jax.jit(lambda z, k: simulate_mgd_style(z, k, mgd_steps, 1.0))(x0, kmgd)
    mgd_tr.block_until_ready()
    runs["mgd_style"] = {"times": tm, "traj": mgd_tr, "runtime_s": perf_counter() - start}

    keys = jax.random.split(ktarg, len(HELDOUT_TIMES))
    target_states = []
    for k, tf in zip(keys, HELDOUT_TIMES):
        if backend == "jax":
            x, _, fib = projected_batch(k, jnp.asarray(tf), target_bank, model[0])
        else:
            x, _ = sample_bridge(k, jnp.asarray(tf), target_bank)
            x = np.asarray(x)
            u = client.reference_velocity(ref_flat, float(tf), x)
            fib = _remote_fiber_state(client, model, float(tf), x, u)
        target_states.append((x, fib))

    # deterministic subsampling keeps the O(N^2) MMD affordable and matched
    mmd_n = min(512, n_particles, target_bank)
    per_method = {name: [] for name in runs}
    target_rows = []
    for j, tf in enumerate(HELDOUT_TIMES):
        tx, fib = target_states[j]
        ti = jnp.linspace(0, tx.shape[0] - 1, mmd_n).astype(jnp.int32)
        xt = tx[ti]
        wt = fib.projected_weights[ti]; wt = wt / jnp.sum(wt)
        target_ang = wt @ angular_features(xt)
        wfull = jnp.asarray(fib.projected_weights)
        if backend == "jax":
            corr_full = -potential_grad(model[1], jnp.asarray(tf), jnp.asarray(tx))
        else:
            corr_full = jnp.asarray(fib.correction)
        correction_energy = 0.5 * jnp.sum(wfull * jnp.sum(corr_full * corr_full, axis=-1))
        # The reference bank is sampled uniformly, so KL(projected || empirical reference)
        # is the discrete tilt distortion sum_i w_i log(N w_i).
        projection_distortion = jnp.sum(wfull * jnp.log(jnp.maximum(wfull * tx.shape[0], 1e-300)))
        target_rows.append({
            "t": float(tf), "ess_fraction": float(fib.ess_fraction),
            "calibration_residual": float(fib.calibration_residual),
            "rank": int(fib.covariance_rank), "condition": float(fib.covariance_condition),
            "correction_energy": float(correction_energy),
            "projection_distortion": float(projection_distortion),
            "angular": np.asarray(target_ang).tolist(),
        })
        for name, run in runs.items():
            yfull = _select_trajectory(run["traj"], run["times"], tf)
            yi = jnp.linspace(0, yfull.shape[0] - 1, mmd_n).astype(jnp.int32)
            y = yfull[yi]
            momerr = jnp.linalg.norm(empirical_moments(yfull) - TARGET)
            mmd = weighted_mmd(xt, wt, y)
            aang = jnp.mean(angular_features(yfull), axis=0)
            aerr = jnp.linalg.norm(aang - target_ang)
            per_method[name].append({
                "t": float(tf), "projected_mmd": float(mmd),
                "moment_error": float(momerr), "angular_error": float(aerr),
                "moments": np.asarray(empirical_moments(yfull)).tolist(),
                "angular": np.asarray(aang).tolist(),
            })

    summary = {}
    interior = slice(1, -1)
    for name, rows in per_method.items():
        mmd = np.array([r["projected_mmd"] for r in rows])
        me = np.array([r["moment_error"] for r in rows])
        ae = np.array([r["angular_error"] for r in rows])
        deterministic = name != "mgd_style"
        summary[name] = {
            "mean_interior_mmd": float(np.mean(mmd[interior])),
            "max_interior_mmd": float(np.max(mmd[interior])),
            "max_moment_error": float(np.max(me)),
            "mean_interior_angular_error": float(np.mean(ae[interior])),
            "endpoint_t1_mmd": float(mmd[-1]),
            "runtime_s": float(runs[name]["runtime_s"]),
            "nfe": int(2 * flow_steps if deterministic else 0),
            "integration_steps": int(flow_steps if deterministic else mgd_steps),
            "component_calls": {
                "reference_transport": int(2 * flow_steps if deterministic else 0),
                "moment_fiber_realizer": int(2 * flow_steps if name in {"mfsi_learned", "mfsi_learned_safe"} else 0),
                "mgd_guidance_steps": int(mgd_steps if name == "mgd_style" else 0),
            },
        }
    return summary, per_method, target_rows, runs


def make_plot(per_method, target_rows, runs):
    # path metric curves
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, rows in per_method.items():
        ax.plot([r["t"] for r in rows], [r["projected_mmd"] for r in rows], marker="o", label=name)
    ax.set_xlabel("t"); ax.set_ylabel("MMD to independently projected target")
    ax.set_title("Example B: projected-law path discrepancy")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "path_mmd.png", dpi=180); plt.close(fig)

    # representative generated clouds, plus endpoint target structure
    names = ["raw_si", "moment_tangent", "mgd_style", "mfsi_learned_safe"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, name in zip(axes.ravel(), names):
        y = np.asarray(_select_trajectory(runs[name]["traj"], runs[name]["times"], 0.75))
        ax.scatter(y[::max(len(y)//1200,1), 0], y[::max(len(y)//1200,1), 1], s=3, alpha=.45)
        ax.set_title(f"{name}, t=0.75"); ax.set_aspect("equal"); ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    fig.tight_layout(); fig.savefig(OUT / "snapshots_t075.png", dpi=180); plt.close(fig)

    # Fiber profile: measured coordinates remain flat while held-out angular
    # structure follows the independently projected path.
    rows = per_method["mfsi_learned_safe"]
    tt = np.asarray([r["t"] for r in rows])
    mm = np.asarray([r["moments"] for r in rows]) - np.asarray(TARGET)[None, :]
    targ_ang = np.asarray([r["angular"] for r in target_rows])
    gen_ang = np.asarray([r["angular"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    labels_phi = ["E[x]", "E[y]", "E[x²]-1", "E[xy]", "E[y²]-1"]
    for k, lab in enumerate(labels_phi):
        axes[0].plot(tt, mm[:, k], marker='o', ms=3, label=lab)
    axes[0].axhline(0.0, linewidth=1, linestyle='--')
    axes[0].set_ylabel("measured-moment deviation")
    axes[0].set_title("Example B fiber profile: measured observables stay flat")
    axes[0].legend(fontsize=7, ncol=3)
    labels_ang = ["cos 4θ", "sin 4θ", "cos 8θ", "sin 8θ"]
    for k, lab in enumerate(labels_ang):
        axes[1].plot(tt, targ_ang[:, k], linewidth=2, label=f"target {lab}")
        axes[1].plot(tt, gen_ang[:, k], 'o--', ms=3, linewidth=1, label=f"MFSI {lab}")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("held-out angular descriptor")
    axes[1].set_title("Hidden angular structure moves along the fiber")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fiber_profile.png", dpi=180); plt.close(fig)

    # Projection / correction diagnostics requested by the paper metrics section.
    ce = np.asarray([r["correction_energy"] for r in target_rows])
    dp = np.asarray([r["projection_distortion"] for r in target_rows])
    ess = np.asarray([r["ess_fraction"] for r in target_rows])
    cond = np.asarray([r["condition"] for r in target_rows])
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    axes[0,0].plot(tt, ce, marker='o'); axes[0,0].set_title("Correction energy density")
    axes[0,1].plot(tt, dp, marker='o'); axes[0,1].set_title("Projection distortion")
    axes[1,0].plot(tt, ess, marker='o'); axes[1,0].set_title("ESS fraction")
    axes[1,1].plot(tt, cond, marker='o'); axes[1,1].set_title("Whitened covariance condition")
    for ax in axes.ravel(): ax.set_xlabel("t")
    fig.tight_layout(); fig.savefig(OUT / "projection_diagnostics.png", dpi=180); plt.close(fig)


def write_csv(summary):
    with (OUT / "benchmark_summary.csv").open("w", newline="") as f:
        fields = ["method", "mean_interior_mmd", "max_interior_mmd", "max_moment_error",
                  "mean_interior_angular_error", "endpoint_t1_mmd", "runtime_s", "nfe", "integration_steps"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for method, vals in summary.items():
            w.writerow({"method": method, **{k: v for k, v in vals.items() if k in fields}})


def run(args):
    master = jax.random.PRNGKey(args.seed)
    kend, kref, kritz, kval, kbench = jax.random.split(master, 5)

    ep = endpoint_check(kend, n=12000 if args.quick else 180000)
    if args.retrain or not MODEL_PATH.exists():
        # --quick is intentionally a smoke-test budget. Full paper runs use the
        # unchanged non-quick settings below.
        ref_steps = 120 if args.quick else 1800
        ritz_steps = 100 if args.quick else 1600
        ref, rh, reftrain = train_reference(kref, steps=ref_steps, batch_size=768 if args.quick else 3072)
        pot, phist, ritztrain = train_ritz(
            kritz, ref, steps=ritz_steps,
            n_times=4 if args.quick else 10,
            particles_per_time=128 if args.quick else 384,
            pool_size=3 if args.quick else 8,
            refresh_every=50 if args.quick else 250,
            lbfgs_maxiter=0 if args.quick else 2,
            endpoint_particles=128 if args.quick else 384,
        )
        model = (ref, pot); save_model(model)
    else:
        model = load_model(); rh, phist = [], []
        reftrain = {"loaded": True}; ritztrain = {"loaded": True}

    ref_hold = reference_holdout(kval, model[0], n=2000 if args.quick else 12000)
    kproj, ktess = jax.random.split(kval)
    proj = projection_diagnostics(
        kproj, model, [0.1,0.3,0.5,0.7,0.9],
        n_particles=1024 if args.quick else 4096,
    )
    tess = tesseract_parity(model)
    summary, per_method, target_rows, runs = benchmark(
        kbench, model,
        n_particles=512 if args.quick else 3072,
        flow_steps=40 if args.quick else 160,
        mgd_steps=120 if args.quick else 800,
        target_bank=768 if args.quick else 4096,
        backend=args.backend,
    )
    if not args.no_plots:
        make_plot(per_method, target_rows, runs)
    write_csv(summary)

    times_metric = np.asarray([r["t"] for r in target_rows], dtype=float)
    corr_metric = np.asarray([r["correction_energy"] for r in target_rows], dtype=float)
    dist_metric = np.asarray([r["projection_distortion"] for r in target_rows], dtype=float)
    path_functionals = {
        "integrated_correction_energy": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(corr_metric, times_metric)),
        "integrated_projection_distortion": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(dist_metric, times_metric)),
        "min_ess_fraction": float(min(r["ess_fraction"] for r in target_rows)),
        "median_ess_fraction": float(np.median([r["ess_fraction"] for r in target_rows])),
    }

    result = {
        "setup": {"backend": args.backend, "ring_radius": RING_RADIUS, "noise_std": NOISE_STD,
                  "target": np.asarray(TARGET).tolist(),
                  "endpoint_definition": "uniform ring centers vs four axis-aligned centers, shared isotropic Gaussian thickness"},
        "endpoint_population_check": ep,
        "reference_holdout": ref_hold,
        "projection_and_ritz_holdout": proj,
        "tesseract_kernel_parity": tess,
        "training": {"reference": reftrain, "ritz": ritztrain,
                     "reference_history": rh, "ritz_history": phist},
        "benchmark_summary": summary,
        "benchmark_per_time": per_method,
        "projected_target_diagnostics": target_rows,
        "path_functionals": path_functionals,
    }
    (OUT / "example_b_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "endpoint_population_check": ep,
        "reference_holdout": ref_hold,
        "projection_holdout": proj,
        "tesseract_kernel_parity": tess,
        "benchmark_summary": summary,
        "path_functionals": path_functionals,
    }, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--quick", action="store_true", help="small smoke-test budgets; not for paper metrics")
    p.add_argument("--no-plots", action="store_true", help="skip plot generation (useful for seed sweeps)")
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--backend", choices=("tesseract", "jax"), default=normalize_backend(None),
                   help="component execution backend; default: tesseract")
    run(p.parse_args())
