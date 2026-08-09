"""MFSI Example A: exact oracle + learned finite-particle implementation.

The module deliberately keeps two layers side by side:

1. exact 1D quadrature/Poisson routines used only as a development oracle;
2. the paper-facing learned path: flow-matched reference velocity, empirical
   I-projection, closed-form fiber forcing, and a time-conditioned Deep-Ritz
   potential.

The learned path does *not* use Example-A symmetry in its networks or losses.
Its training/validation criteria are intended to transfer to later examples:
fresh-bank calibration, ESS/rank diagnostics, held-out weak-form residuals,
generated-vs-projected two-sample discrepancy, and population moment drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

Array = jax.Array
DiffMode = Literal["unrolled", "implicit", "stop"]
DEFAULT_NEWTON_ITERS = 20
DEFAULT_DAMPING = 1e-10
DEFAULT_RCOND = 1e-9
TARGET = jnp.array([0.0, 1.0], dtype=jnp.float64)

TIME_FOURIER_FREQUENCIES = 4
REFERENCE_HIDDEN = (64, 64, 64)
RITZ_HIDDEN = (96, 96, 96, 96)


@dataclass(frozen=True)
class Grid:
    x: Array
    w: Array


class ReferenceState(NamedTuple):
    log_q_ref: Array
    q_ref: Array
    velocity: Array


class FiberState(NamedTuple):
    lambda_: Array
    q: Array
    moments: Array
    covariance: Array
    lambda_dot: Array
    forcing: Array
    correction: Array
    velocity: Array
    correction_energy: Array
    projection_distortion: Array
    ess_fraction: Array


class EmpiricalFiberState(NamedTuple):
    lambda_: Array
    projected_weights: Array
    moments: Array
    covariance: Array
    lambda_dot: Array
    forcing: Array
    ess_fraction: Array
    calibration_residual: Array
    covariance_rank: Array
    covariance_condition: Array


class LearnedMFSIModel(NamedTuple):
    reference_params: object
    potential_params: object


# ---------------------------------------------------------------------------
# Generic numerical helpers
# ---------------------------------------------------------------------------

def make_grid(xmax: float = 7.0, n: int = 2001) -> Grid:
    x = jnp.linspace(-xmax, xmax, n)
    dx = x[1] - x[0]
    w = jnp.ones_like(x) * dx
    w = w.at[0].set(0.5 * dx)
    w = w.at[-1].set(0.5 * dx)
    return Grid(x=x, w=w)


def phi(x: Array) -> Array:
    """Measured observables for Example A. Network architecture does not use symmetry."""
    return jnp.stack([x, x * x], axis=-1)


def jphi(x: Array) -> Array:
    return jnp.stack([jnp.ones_like(x), 2.0 * x], axis=-1)


def jphi_times_velocity(x: Array, velocity: Array) -> Array:
    return jphi(x) * velocity[..., None]


def inverse_softplus(y: float) -> float:
    return float(jnp.log(jnp.expm1(y)))


def _tree_zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def _tree_add(a, b):
    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_mul(a, scalar):
    return jax.tree.map(lambda x: scalar * x, a)


def _tree_square(a):
    return jax.tree.map(lambda x: x * x, a)


def _tree_div(a, b, eps=1e-8):
    return jax.tree.map(lambda x, y: x / (jnp.sqrt(y) + eps), a, b)


def _tree_weight_decay(params, grads, weight_decay):
    if weight_decay == 0.0:
        return grads
    return jax.tree.map(lambda g, p: g + weight_decay * p, grads, params)


def _tree_global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)))


def _clip_tree_by_global_norm(tree, max_norm: float):
    if max_norm <= 0.0:
        return tree
    norm = _tree_global_norm(tree)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-30))
    return jax.tree.map(lambda x: scale * x, tree)


def _adamw_update(params, grads, m, v, step, lr, weight_decay=0.0):
    grads = _tree_weight_decay(params, grads, weight_decay)
    m = jax.tree.map(lambda mm, g: 0.9 * mm + 0.1 * g, m, grads)
    v = jax.tree.map(lambda vv, g: 0.999 * vv + 0.001 * g * g, v, grads)
    mh = jax.tree.map(lambda mm: mm / (1.0 - 0.9**step), m)
    vh = jax.tree.map(lambda vv: vv / (1.0 - 0.999**step), v)
    update = _tree_div(mh, vh)
    params = jax.tree.map(lambda p, u: p - lr * u, params, update)
    return params, m, v


def cosine_lr(step: int | Array, total_steps: int, lr_start: float, lr_end: float) -> Array:
    z = jnp.minimum(jnp.asarray(step, dtype=jnp.float64) / max(total_steps, 1), 1.0)
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + jnp.cos(jnp.pi * z))


# ---------------------------------------------------------------------------
# Exact Example-A reference bridge (development oracle + training data source)
# ---------------------------------------------------------------------------

def beta_schedule(t: Array, params: Array | None = None) -> Array:
    if params is None:
        return jnp.zeros_like(jnp.asarray(t, dtype=jnp.float64))
    t = jnp.asarray(t, dtype=jnp.float64)
    z = 2.0 * t - 1.0
    powers = jnp.arange(params.shape[0])
    raw = jnp.sum(params * z[..., None] ** powers, axis=-1)
    return jax.nn.softplus(raw)


def gamma_schedule(t: Array, params: Array | None = None) -> Array:
    t = jnp.asarray(t, dtype=jnp.float64)
    return jnp.sqrt(jnp.maximum(2.0 * t * (1.0 - t), 0.0)) * beta_schedule(t, params)


def _normal_logpdf(x: Array, mean: Array, var: Array) -> Array:
    return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)


def reference_component_variance(t: Array, a: float, params: Array | None = None) -> Array:
    vp = 1.0 - a * a
    b = beta_schedule(t, params)
    # Use gamma^2 directly. This is algebraically identical to gamma_schedule(t)^2
    # but stays differentiable at t=0,1 where d/dt sqrt(t(1-t)) is singular.
    gamma2 = 2.0 * t * (1.0 - t) * b * b
    return (1.0 - t) ** 2 + t * t * vp + gamma2


def reference_component_variance_dt(t: Array, a: float, params: Array | None = None) -> Array:
    # AD keeps this correct for arbitrary differentiable schedule parameters.
    if jnp.ndim(t) == 0:
        return jax.grad(lambda s: reference_component_variance(s, a, params))(t)
    return jax.vmap(lambda s: jax.grad(lambda r: reference_component_variance(r, a, params))(s))(t)


def reference_logpdf_raw(x: Array, t: Array, a: float, params: Array | None = None) -> Array:
    var = reference_component_variance(t, a, params)
    mu = t * a
    lp = _normal_logpdf(x, mu, var)
    lm = _normal_logpdf(x, -mu, var)
    return jax.scipy.special.logsumexp(jnp.stack([lp, lm], axis=0), axis=0) - jnp.log(2.0)


def reference_velocity(x: Array, t: Array, a: float, params: Array | None = None) -> Array:
    """Exact SI velocity u_t=E[d_t I_t|I_t=x] for Example A."""
    var = reference_component_variance(t, a, params)
    dvar_dt = reference_component_variance_dt(t, a, params)
    k = 0.5 * dvar_dt / var
    label_mean = jnp.tanh((t * a * x) / var)
    return k * x + a * (1.0 - k * t) * label_mean


def _normalize_logdensity(logu: Array, weights: Array) -> tuple[Array, Array]:
    logz = jax.scipy.special.logsumexp(logu + jnp.log(weights))
    return logu - logz, logz


def reference_transport(params: Array | None, grid: Grid, t: Array, a: float) -> ReferenceState:
    raw = reference_logpdf_raw(grid.x, t, a, params)
    log_q_ref, _ = _normalize_logdensity(raw, grid.w)
    q_ref = jnp.exp(log_q_ref)
    u = reference_velocity(grid.x, t, a, params)
    return ReferenceState(log_q_ref=log_q_ref, q_ref=q_ref, velocity=u)


def sample_reference_bridge(
    key: Array,
    t: Array,
    n: int,
    a: float,
    bridge_params: Array | None = None,
) -> tuple[Array, Array]:
    """Draw ordinary SI samples and their conditional-regression targets.

    I_t=(1-t)X_- + t X_+ + gamma(t) Z.  The default bridge has gamma=0.
    This function is only the toy data source; learned models never receive the
    mixture label or analytic reference velocity.
    """
    k0, k1, ks, kz = jax.random.split(key, 4)
    x0 = jax.random.normal(k0, (n,), dtype=jnp.float64)
    sign = jnp.where(jax.random.bernoulli(ks, 0.5, (n,)), 1.0, -1.0)
    x1 = sign * a + jnp.sqrt(1.0 - a * a) * jax.random.normal(k1, (n,), dtype=jnp.float64)
    z = jax.random.normal(kz, (n,), dtype=jnp.float64)
    g = gamma_schedule(t, bridge_params)
    if bridge_params is None:
        gd = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        gd = jax.grad(lambda s: gamma_schedule(s, bridge_params))(t)
    xt = (1.0 - t) * x0 + t * x1 + g * z
    dxt = x1 - x0 + gd * z
    return xt, dxt


def sample_reference_bridge_times(
    key: Array,
    times: Array,
    a: float,
    bridge_params: Array | None = None,
) -> tuple[Array, Array]:
    """Vectorized bridge samples with one independently drawn tuple per time."""
    times = jnp.asarray(times, dtype=jnp.float64)
    n = times.shape[0]
    k0, k1, ks, kz = jax.random.split(key, 4)
    x0 = jax.random.normal(k0, (n,), dtype=jnp.float64)
    sign = jnp.where(jax.random.bernoulli(ks, 0.5, (n,)), 1.0, -1.0)
    x1 = sign * a + jnp.sqrt(1.0 - a * a) * jax.random.normal(k1, (n,), dtype=jnp.float64)
    z = jax.random.normal(kz, (n,), dtype=jnp.float64)
    g = gamma_schedule(times, bridge_params)
    if bridge_params is None:
        gd = jnp.zeros_like(times)
    else:
        gd = jax.vmap(lambda tt: jax.grad(lambda ss: gamma_schedule(ss, bridge_params))(tt))(times)
    return (1.0 - times) * x0 + times * x1 + g * z, x1 - x0 + gd * z


# ---------------------------------------------------------------------------
# Exact quadrature I-projection and Poisson oracle
# ---------------------------------------------------------------------------

def projected_density_from_lambda(lam, log_q_ref, ph, weights):
    log_q, logz_tilt = _normalize_logdensity(log_q_ref + ph @ lam, weights)
    q = jnp.exp(log_q)
    moments = jnp.sum(weights[:, None] * q[:, None] * ph, axis=0)
    centered = ph - moments
    cov = (centered.T * (weights * q)) @ centered
    return q, moments, cov, logz_tilt


def _stable_cov_solve(cov: Array, rhs: Array, rcond: float = DEFAULT_RCOND, damping: float = 0.0):
    """Diagonal whitening + eigenvalue rank truncation on the identifiable span."""
    diag = jnp.maximum(jnp.diag(cov), 1e-30)
    scale = jnp.sqrt(diag)
    cw = cov / (scale[:, None] * scale[None, :])
    cw = 0.5 * (cw + cw.T) + damping * jnp.eye(cov.shape[0], dtype=cov.dtype)
    vals, vecs = jnp.linalg.eigh(cw)
    vmax = jnp.maximum(jnp.max(vals), 1e-30)
    keep = vals > rcond * vmax
    inv = jnp.where(keep, 1.0 / jnp.maximum(vals, 1e-30), 0.0)
    rhsw = rhs / scale
    solw = vecs @ (inv * (vecs.T @ rhsw))
    sol = solw / scale
    rank = jnp.sum(keep.astype(jnp.int32))
    vmin = jnp.min(jnp.where(keep, vals, jnp.inf))
    cond = jnp.where(rank > 0, vmax / vmin, jnp.inf)
    return sol, rank, cond


def _newton_step_arrays(_, lam, *, log_q_ref, ph, weights, target, damping):
    _, moments, cov, _ = projected_density_from_lambda(lam, log_q_ref, ph, weights)
    step = jnp.linalg.solve(cov + damping * jnp.eye(cov.shape[0], dtype=cov.dtype), moments - target)
    norm = jnp.linalg.norm(step)
    step = step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-30))
    return lam - step


def calibrate_lambda_unrolled(log_q_ref, ph, weights, target=TARGET, *, iterations=DEFAULT_NEWTON_ITERS, damping=DEFAULT_DAMPING):
    lam0 = jnp.zeros(target.shape[0], dtype=log_q_ref.dtype)
    body = partial(_newton_step_arrays, log_q_ref=log_q_ref, ph=ph, weights=weights, target=target, damping=damping)
    return jax.lax.fori_loop(0, iterations, body, lam0)


@jax.custom_vjp
def calibrate_lambda_implicit(log_q_ref, ph, weights, target):
    return calibrate_lambda_unrolled(log_q_ref, ph, weights, target)


def _calibrate_implicit_fwd(log_q_ref, ph, weights, target):
    lam = calibrate_lambda_unrolled(log_q_ref, ph, weights, target)
    q, moments, cov, _ = projected_density_from_lambda(lam, log_q_ref, ph, weights)
    probs = weights * q
    return lam, (probs, ph, moments, cov, lam)


def _calibrate_implicit_bwd(res, lambda_bar):
    probs, ph, moments, cov, lam = res
    a = jnp.linalg.solve(cov.T + DEFAULT_DAMPING * jnp.eye(cov.shape[0], dtype=cov.dtype), lambda_bar)
    centered = ph - moments
    log_q_ref_bar = -probs * (centered @ a)
    # Unlike the old toy VJP, this propagates through Phi as well.
    ph_bar = -probs[:, None] * (a[None, :] + (centered @ a)[:, None] * lam[None, :])
    target_bar = a
    # Quadrature weights remain numerical integration definitions in this oracle path.
    return log_q_ref_bar, ph_bar, jnp.zeros_like(probs), target_bar


calibrate_lambda_implicit.defvjp(_calibrate_implicit_fwd, _calibrate_implicit_bwd)


def calibrate_lambda(log_q_ref, ph, weights, target=TARGET, *, differentiation: DiffMode = "implicit"):
    if differentiation == "implicit":
        return calibrate_lambda_implicit(log_q_ref, ph, weights, target)
    lam = calibrate_lambda_unrolled(log_q_ref, ph, weights, target)
    if differentiation == "stop":
        return jax.lax.stop_gradient(lam)
    if differentiation == "unrolled":
        return lam
    raise ValueError(differentiation)


def cumulative_trapezoid(y: Array, x: Array) -> Array:
    dx = x[1:] - x[:-1]
    increments = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.zeros(1, dtype=y.dtype), jnp.cumsum(increments)])


def exact_poisson_correction(q: Array, h: Array, grid: Grid, q_floor: float = 1e-12) -> Array:
    flux = cumulative_trapezoid(q * h, grid.x)
    total = flux[-1]
    cdf_q = cumulative_trapezoid(q, grid.x)
    flux = flux - total * cdf_q
    delta = -flux / jnp.maximum(q, q_floor)
    return jnp.where(q > q_floor, delta, 0.0)


def moment_fiber_realizer(reference: ReferenceState, grid: Grid, target: Array = TARGET, *, differentiation: DiffMode = "implicit") -> FiberState:
    ph = phi(grid.x)
    lam = calibrate_lambda(reference.log_q_ref, ph, grid.w, target, differentiation=differentiation)
    q, moments, cov, logz_tilt = projected_density_from_lambda(lam, reference.log_q_ref, ph, grid.w)
    m = jphi_times_velocity(grid.x, reference.velocity)
    probs = grid.w * q
    em = jnp.sum(probs[:, None] * m, axis=0)
    scalar = m @ lam
    cov_term = jnp.sum(probs[:, None] * (ph - target) * scalar[:, None], axis=0)
    lambda_dot = jnp.linalg.solve(cov + DEFAULT_DAMPING * jnp.eye(cov.shape[0], dtype=cov.dtype), -em - cov_term)
    h = (ph - target) @ lambda_dot + (m - em) @ lam
    # Exact centering enforces the Poisson solvability condition numerically.
    h = h - jnp.sum(probs * h)
    delta = exact_poisson_correction(q, h, grid)
    v = reference.velocity + delta
    energy = jnp.sum(probs * delta * delta)
    distortion = lam @ target - logz_tilt
    ess = 1.0 / jnp.sum(grid.w * q * q / jnp.maximum(reference.q_ref, 1e-300))
    return FiberState(lam, q, moments, cov, lambda_dot, h, delta, v, energy, distortion, ess)


def mfsi_pipeline(params, grid, t, a, *, differentiation: DiffMode = "implicit", target: Array = TARGET):
    ref = reference_transport(params, grid, t, a)
    fib = moment_fiber_realizer(ref, grid, target, differentiation=differentiation)
    return ref, fib


# ---------------------------------------------------------------------------
# Empirical I-projection used by the learned algorithm
# ---------------------------------------------------------------------------

def empirical_tilt_from_lambda(lam: Array, log_base_weights: Array, ph: Array):
    logits = log_base_weights + ph @ lam
    weights = jax.nn.softmax(logits)
    moments = weights @ ph
    centered = ph - moments
    cov = (centered.T * weights) @ centered
    return weights, moments, cov


def _calibrate_empirical_primal(log_base_weights: Array, ph: Array, target: Array, iterations: int = DEFAULT_NEWTON_ITERS):
    lam0 = jnp.zeros(target.shape[0], dtype=ph.dtype)
    def body(_, lam):
        _, moments, cov = empirical_tilt_from_lambda(lam, log_base_weights, ph)
        step, _, _ = _stable_cov_solve(cov, moments - target, damping=DEFAULT_DAMPING)
        norm = jnp.linalg.norm(step)
        step = step * jnp.minimum(1.0, 2.0 / jnp.maximum(norm, 1e-30))
        return lam - step
    return jax.lax.fori_loop(0, iterations, body, lam0)


@jax.custom_vjp
def calibrate_empirical_implicit(log_base_weights: Array, ph: Array, target: Array) -> Array:
    """Particle-aware implicit calibration VJP.

    The backward pass propagates through both base log-weights and Phi values,
    so upstream particle positions/observable parameters remain differentiable.
    """
    return _calibrate_empirical_primal(log_base_weights, ph, target)


def _empirical_fwd(log_base_weights, ph, target):
    lam = _calibrate_empirical_primal(log_base_weights, ph, target)
    w, moments, cov = empirical_tilt_from_lambda(lam, log_base_weights, ph)
    return lam, (w, ph, moments, cov, lam)


def _empirical_bwd(res, lambda_bar):
    w, ph, moments, cov, lam = res
    a, _, _ = _stable_cov_solve(cov.T, lambda_bar, damping=DEFAULT_DAMPING)
    centered = ph - moments
    proj = centered @ a
    logw_bar = -w * proj
    ph_bar = -w[:, None] * (a[None, :] + proj[:, None] * lam[None, :])
    target_bar = a
    return logw_bar, ph_bar, target_bar


calibrate_empirical_implicit.defvjp(_empirical_fwd, _empirical_bwd)


def empirical_fiber_state(
    x: Array,
    velocity: Array,
    target: Array = TARGET,
    *,
    log_base_weights: Array | None = None,
    ph: Array | None = None,
    jphi_u: Array | None = None,
) -> EmpiricalFiberState:
    n = x.shape[0]
    if log_base_weights is None:
        log_base_weights = jnp.zeros((n,), dtype=x.dtype)
    if ph is None:
        ph = phi(x)
    if jphi_u is None:
        jphi_u = jphi_times_velocity(x, velocity)
    lam = calibrate_empirical_implicit(log_base_weights, ph, target)
    w, moments, cov = empirical_tilt_from_lambda(lam, log_base_weights, ph)
    em = w @ jphi_u
    scalar = jphi_u @ lam
    cov_term = jnp.sum(w[:, None] * (ph - target) * scalar[:, None], axis=0)
    lambda_dot, rank, cond = _stable_cov_solve(cov, -em - cov_term, damping=DEFAULT_DAMPING)
    h = (ph - target) @ lambda_dot + (jphi_u - em) @ lam
    # Important generic stabilization: exact empirical solvability of weighted Poisson.
    h = h - w @ h
    residual = jnp.linalg.norm(moments - target)
    ess_fraction = 1.0 / (n * jnp.sum(w * w))
    return EmpiricalFiberState(lam, w, moments, cov, lambda_dot, h, ess_fraction, residual, rank, cond)


# ---------------------------------------------------------------------------
# Small generic MLPs for u_theta(t,x) and psi_omega(t,x)
# ---------------------------------------------------------------------------

def time_fourier_features(t: Array, n_freq: int = TIME_FOURIER_FREQUENCIES) -> Array:
    t = jnp.asarray(t)
    k = 2.0 ** jnp.arange(n_freq, dtype=t.dtype)
    angles = 2.0 * jnp.pi * t[..., None] * k
    return jnp.concatenate([t[..., None], jnp.sin(angles), jnp.cos(angles)], axis=-1)


def network_features(t: Array, x: Array, n_freq: int = TIME_FOURIER_FREQUENCIES) -> Array:
    x = jnp.asarray(x)
    t = jnp.broadcast_to(jnp.asarray(t, dtype=x.dtype), x.shape)
    return jnp.concatenate([x[..., None], time_fourier_features(t, n_freq)], axis=-1)


def init_mlp(key: Array, input_dim: int, hidden: Sequence[int], output_dim: int = 1):
    dims = (input_dim, *tuple(hidden), output_dim)
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        lim = jnp.sqrt(6.0 / (din + dout))
        W = jax.random.uniform(k, (din, dout), dtype=jnp.float64, minval=-lim, maxval=lim)
        b = jnp.zeros((dout,), dtype=jnp.float64)
        params.append((W, b))
    return tuple(params)


def mlp_apply(params, features: Array) -> Array:
    z = features
    for W, b in params[:-1]:
        z = jax.nn.silu(z @ W + b)
    W, b = params[-1]
    return z @ W + b


def reference_velocity_net(params, t: Array, x: Array) -> Array:
    return mlp_apply(params, network_features(t, x))[..., 0]


def potential_net(params, t: Array, x: Array) -> Array:
    return mlp_apply(params, network_features(t, x))[..., 0]


def potential_x(params, t: Array, x: Array) -> Array:
    # The potential network is pointwise in x.  A JVP in the all-ones direction
    # therefore returns the diagonal partials d psi(t,x_i)/d x_i without
    # materializing a Jacobian; this also compiles much more cheaply inside ODE scans.
    x = jnp.asarray(x)
    _, tangent = jax.jvp(lambda xx: potential_net(params, t, xx), (x,), (jnp.ones_like(x),))
    return tangent


def learned_correction(params, t: Array, x: Array) -> Array:
    return -potential_x(params, t, x)


def mlp_flat_size(hidden: Sequence[int], n_freq: int = TIME_FOURIER_FREQUENCIES) -> int:
    input_dim = 1 + 1 + 2 * n_freq
    dims = (input_dim, *tuple(hidden), 1)
    return int(sum(din * dout + dout for din, dout in zip(dims[:-1], dims[1:])))


def flatten_mlp(params) -> Array:
    return jnp.concatenate([jnp.concatenate([W.reshape(-1), b.reshape(-1)]) for W, b in params])


def unflatten_mlp(flat: Array, hidden: Sequence[int], n_freq: int = TIME_FOURIER_FREQUENCIES):
    input_dim = 1 + 1 + 2 * n_freq
    dims = (input_dim, *tuple(hidden), 1)
    params = []
    off = 0
    for din, dout in zip(dims[:-1], dims[1:]):
        nw = din * dout
        W = flat[off:off + nw].reshape((din, dout)); off += nw
        b = flat[off:off + dout]; off += dout
        params.append((W, b))
    return tuple(params)


# ---------------------------------------------------------------------------
# Flow-matching reference training
# ---------------------------------------------------------------------------

def _stratified_times(key: Array, n: int, lo: float = 0.001, hi: float = 0.999) -> Array:
    jitter = jax.random.uniform(key, (n,), dtype=jnp.float64)
    t = (jnp.arange(n, dtype=jnp.float64) + jitter) / n
    return lo + (hi - lo) * t


def train_reference_flow_matching(
    key: Array,
    a: float,
    *,
    steps: int = 1800,
    batch_size: int = 4096,
    hidden: Sequence[int] = REFERENCE_HIDDEN,
    lr_start: float = 1.5e-3,
    lr_end: float = 3e-5,
    weight_decay: float = 1e-6,
    grad_clip: float = 5.0,
    validation_size: int = 8192,
    eval_every: int = 100,
    bridge_params: Array | None = None,
    initial_params=None,
):
    """Train u_theta by standard SI velocity regression.

    The checkpoint is selected on a fixed independent SI regression bank, so
    neither training nor model selection uses the analytic Example-A velocity.
    ``initial_params`` permits generic continuation training of a checkpoint.
    """
    input_dim = 1 + 1 + 2 * TIME_FOURIER_FREQUENCIES
    key, init_key, kval_t, kval_b = jax.random.split(key, 4)
    params = init_mlp(init_key, input_dim, hidden) if initial_params is None else initial_params
    m, v = _tree_zeros_like(params), _tree_zeros_like(params)

    t_val = _stratified_times(kval_t, validation_size)
    x_val, target_val = sample_reference_bridge_times(kval_b, t_val, a, bridge_params)

    def loss_fn(p, k):
        kt, kb = jax.random.split(k)
        t = _stratified_times(kt, batch_size)
        x, target_v = sample_reference_bridge_times(kb, t, a, bridge_params)
        pred = reference_velocity_net(p, t, x)
        return jnp.mean((pred - target_v) ** 2)

    val_jit = jax.jit(lambda p: jnp.mean((reference_velocity_net(p, t_val, x_val) - target_val) ** 2))
    vg = jax.jit(jax.value_and_grad(loss_fn))
    best_params = params
    best_val = float(val_jit(params))
    history = []
    for i in range(1, steps + 1):
        key, sub = jax.random.split(key)
        loss, grads = vg(params, sub)
        grads = _clip_tree_by_global_norm(grads, grad_clip)
        lr = cosine_lr(i - 1, steps, lr_start, lr_end)
        params, m, v = _adamw_update(params, grads, m, v, i, lr, weight_decay)
        if i == 1 or i % eval_every == 0 or i == steps:
            val = float(val_jit(params))
            if val < best_val:
                best_val, best_params = val, params
            history.append((i, float(loss), float(lr), val))
    return best_params, history


# ---------------------------------------------------------------------------
# Deep-Ritz with generic stabilizations
# ---------------------------------------------------------------------------

def projected_batch(key: Array, t: Array, n: int, a: float, reference_params, target: Array = TARGET):
    x, _ = sample_reference_bridge(key, t, n, a, None)
    u = reference_velocity_net(reference_params, t, x)
    fib = empirical_fiber_state(x, u, target)
    return x, u, fib


def ritz_objective_on_state(potential_params, t: Array, x: Array, weights: Array, h: Array):
    """Weighted Ritz loss with exact gauge fixing on each empirical state."""
    psi = potential_net(potential_params, t, x)
    # Gauge fixing prevents the null constant mode from drifting under finite-batch noise.
    psi = psi - weights @ psi
    px = potential_x(potential_params, t, x)
    h = h - weights @ h
    return 0.5 * jnp.sum(weights * px * px) + jnp.sum(weights * h * psi)


def fresh_ritz_loss(
    potential_params,
    key: Array,
    reference_params,
    a: float,
    target: Array,
    n_times: int,
    particles_per_time: int,
):
    kt, kb = jax.random.split(key)
    times = _stratified_times(kt, n_times)
    keys = jax.random.split(kb, n_times)
    def one(k, t):
        x, _, fib = projected_batch(k, t, particles_per_time, a, reference_params, target)
        return ritz_objective_on_state(potential_params, t, x, fib.projected_weights, fib.forcing)
    return jnp.mean(jax.vmap(one)(keys, times))


def build_fixed_projected_bank(key, reference_params, a: float, target: Array, n_times: int, particles_per_time: int):
    kt, kb = jax.random.split(key)
    times = _stratified_times(kt, n_times)
    keys = jax.random.split(kb, n_times)
    def one(k, t):
        x, u, fib = projected_batch(k, t, particles_per_time, a, reference_params, target)
        return x, u, fib.projected_weights, fib.forcing, fib.calibration_residual, fib.ess_fraction, fib.covariance_rank, fib.covariance_condition
    x, u, w, h, resid, ess, rank, cond = jax.vmap(one)(keys, times)
    return {
        "times": times,
        "x": jax.lax.stop_gradient(x),
        "u": jax.lax.stop_gradient(u),
        "weights": jax.lax.stop_gradient(w),
        "h": jax.lax.stop_gradient(h),
        "calibration_residual": resid,
        "ess_fraction": ess,
        "rank": rank,
        "condition": cond,
    }


def fixed_bank_ritz_loss(potential_params, bank):
    vals = jax.vmap(lambda t, x, w, h: ritz_objective_on_state(potential_params, t, x, w, h))(
        bank["times"], bank["x"], bank["weights"], bank["h"]
    )
    return jnp.mean(vals)


def train_deep_ritz(
    key: Array,
    reference_params,
    a: float,
    *,
    target: Array = TARGET,
    steps: int = 1600,
    n_times: int = 12,
    particles_per_time: int = 256,
    hidden: Sequence[int] = RITZ_HIDDEN,
    lr_start: float = 1.0e-3,
    lr_end: float = 2e-5,
    weight_decay: float = 1e-7,
    grad_clip: float = 5.0,
    bank_pool_size: int = 16,
    bank_refresh_every: int = 200,
    eval_every: int = 100,
    validation_times: int = 16,
    validation_particles: int = 384,
    lbfgs_maxiter: int = 4,
    polish_times: int = 16,
    polish_particles: int = 256,
    initial_params=None,
):
    """Train one time-conditioned Deep-Ritz potential.

    The loss is unchanged from the paper.  Improvements are optimizer-side:
    rotating fresh projected banks, gradient clipping, independent held-out
    Ritz checkpoint selection, and an L-BFGS candidate accepted only when it
    improves that held-out objective.  No analytic Poisson oracle is used.
    """
    input_dim = 1 + 1 + 2 * TIME_FOURIER_FREQUENCIES
    key, init_key, kval = jax.random.split(key, 3)
    params = init_mlp(init_key, input_dim, hidden) if initial_params is None else initial_params
    m, v = _tree_zeros_like(params), _tree_zeros_like(params)

    validation_bank = build_fixed_projected_bank(
        kval, reference_params, a, target, validation_times, validation_particles
    )
    validation_loss = jax.jit(lambda p: fixed_bank_ritz_loss(p, validation_bank))

    build_one = jax.jit(lambda k: build_fixed_projected_bank(
        k, reference_params, a, target, n_times, particles_per_time
    ))
    def make_pool(master_key):
        ks = jax.random.split(master_key, bank_pool_size)
        banks = [build_one(k) for k in ks]
        for b in banks:
            jax.tree.map(lambda z: z.block_until_ready() if hasattr(z, "block_until_ready") else z, b)
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *banks)

    key, kpool = jax.random.split(key)
    pool = make_pool(kpool)
    def pooled_loss(p, pool_data, idx):
        return fixed_bank_ritz_loss(p, jax.tree.map(lambda z: z[idx], pool_data))
    loss_vg = jax.jit(jax.value_and_grad(pooled_loss))

    best_params = params
    best_val = float(validation_loss(params))
    history = []
    for i in range(1, steps + 1):
        if bank_refresh_every > 0 and i > 1 and (i - 1) % bank_refresh_every == 0:
            key, kpool = jax.random.split(key)
            pool = make_pool(kpool)
        key, kidx = jax.random.split(key)
        idx = jax.random.randint(kidx, (), 0, bank_pool_size)
        loss, grads = loss_vg(params, pool, idx)
        grads = _clip_tree_by_global_norm(grads, grad_clip)
        lr = cosine_lr(i - 1, steps, lr_start, lr_end)
        params, m, v = _adamw_update(params, grads, m, v, i, lr, weight_decay)
        if i == 1 or i % eval_every == 0 or i == steps:
            val = float(validation_loss(params))
            if val < best_val:
                best_val, best_params = val, params
            history.append((i, float(loss), float(lr), val))
    params = best_params

    polish_info = {"used": False, "accepted": False, "success": False, "iterations": 0,
                   "initial_loss": None, "final_loss": None,
                   "validation_before": best_val, "validation_after": best_val}
    if lbfgs_maxiter > 0:
        import scipy.optimize
        from jax.flatten_util import ravel_pytree
        key, kpolish = jax.random.split(key)
        bank = build_fixed_projected_bank(kpolish, reference_params, a, target, polish_times, polish_particles)
        flat0, unravel = ravel_pytree(params)
        valgrad = jax.jit(jax.value_and_grad(lambda z: fixed_bank_ritz_loss(unravel(z), bank)))
        def scipy_fun(z_np):
            val, grad = valgrad(jnp.asarray(z_np))
            return float(val), np.asarray(grad, dtype=np.float64)
        initial = float(fixed_bank_ritz_loss(params, bank))
        res = scipy.optimize.minimize(
            scipy_fun, np.asarray(flat0), jac=True, method="L-BFGS-B",
            options={"maxiter": int(lbfgs_maxiter), "ftol": 1e-12, "gtol": 1e-8, "maxls": 12},
        )
        candidate = unravel(jnp.asarray(res.x))
        final = float(fixed_bank_ritz_loss(candidate, bank))
        val_after = float(validation_loss(candidate))
        accepted = val_after < best_val
        if accepted:
            params = candidate
            best_val = val_after
        polish_info = {
            "used": True, "accepted": bool(accepted), "success": bool(res.success),
            "iterations": int(res.nit), "initial_loss": initial, "final_loss": final,
            "validation_before": float(polish_info["validation_before"]),
            "validation_after": val_after, "message": str(res.message),
        }
    return params, history, polish_info


# ---------------------------------------------------------------------------
# Future-compatible held-out diagnostics
# ---------------------------------------------------------------------------

def weak_form_residual(potential_params, t: Array, x: Array, weights: Array, h: Array, n_tests: int = 16) -> Array:
    """Normalized RMS residual of the weighted Poisson weak form.

    Uses smooth tanh test functions; unlike the analytic 1D oracle correction,
    this diagnostic only requires projected samples, weights, h, and first
    derivatives of the learned potential.
    """
    px = potential_x(potential_params, t, x)
    h = h - weights @ h
    # Scale test functions to the empirical cloud; deterministic phases make the
    # metric repeatable without using Example-A symmetry.
    mean = weights @ x
    std = jnp.sqrt(jnp.sum(weights * (x - mean) ** 2) + 1e-8)
    freqs = jnp.exp(jnp.linspace(jnp.log(0.35), jnp.log(4.0), n_tests)) / std
    phases = jnp.linspace(-1.2, 1.2, n_tests)
    z = x[:, None] * freqs[None, :] + phases[None, :]
    test = jnp.tanh(z)
    dtest = freqs[None, :] * (1.0 - test * test)
    residual = jnp.sum(weights[:, None] * (px[:, None] * dtest + h[:, None] * test), axis=0)
    a = jnp.sqrt(jnp.sum(weights[:, None] * (px[:, None] * dtest) ** 2, axis=0))
    b = jnp.sqrt(jnp.sum(weights[:, None] * (h[:, None] * test) ** 2, axis=0))
    normalized = residual / (a + b + 1e-10)
    return jnp.sqrt(jnp.mean(normalized * normalized))


def weighted_mmd_rbf(x: Array, wx: Array, y: Array) -> Array:
    """Multi-bandwidth weighted MMD; available without an analytic target law."""
    wy = jnp.ones((y.shape[0],), dtype=y.dtype) / y.shape[0]
    mean = wx @ x
    var = jnp.sum(wx * (x - mean) ** 2) + 1e-6
    base = jnp.sqrt(var)
    bandwidths = base * jnp.array([0.5, 1.0, 2.0], dtype=x.dtype)
    dxx = (x[:, None] - x[None, :]) ** 2
    dyy = (y[:, None] - y[None, :]) ** 2
    dxy = (x[:, None] - y[None, :]) ** 2
    def one_bw(s):
        kxx = jnp.exp(-0.5 * dxx / (s * s))
        kyy = jnp.exp(-0.5 * dyy / (s * s))
        kxy = jnp.exp(-0.5 * dxy / (s * s))
        return wx @ (kxx @ wx) + wy @ (kyy @ wy) - 2.0 * wx @ (kxy @ wy)
    return jnp.sqrt(jnp.maximum(jnp.mean(jax.vmap(one_bw)(bandwidths)), 0.0))


def learned_velocity(model: LearnedMFSIModel, t: Array, x: Array) -> Array:
    return reference_velocity_net(model.reference_params, t, x) + learned_correction(model.potential_params, t, x)


def ensemble_safety_velocity(x: Array, velocity: Array, target_rate: Array | None = None, *, jphi_fn=jphi) -> Array:
    """Generic Eq.-45-style population-rate safeguard in the span of J_Phi^T."""
    jp = jphi_fn(x)
    if target_rate is None:
        target_rate = jnp.zeros((jp.shape[-1],), dtype=x.dtype)
    current_rate = jnp.mean(jp * velocity[:, None], axis=0)
    residual = current_rate - target_rate
    G = (jp.T @ jp) / x.shape[0]
    coeff, _, _ = _stable_cov_solve(G, residual, damping=1e-10)
    return velocity - jp @ coeff


def integrate_learned_flow(model: LearnedMFSIModel, x0: Array, n_steps: int = 240, *, safety: bool = False):
    dt = 1.0 / n_steps
    times = jnp.linspace(0.0, 1.0, n_steps + 1)
    def step(x, i):
        t = times[i]
        v0 = learned_velocity(model, t, x)
        if safety:
            v0 = ensemble_safety_velocity(x, v0)
        xp = x + dt * v0
        v1 = learned_velocity(model, times[i + 1], xp)
        if safety:
            v1 = ensemble_safety_velocity(xp, v1)
        xn = x + 0.5 * dt * (v0 + v1)
        return xn, xn
    _, tail = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return times, jnp.concatenate([x0[None, :], tail], axis=0)


def heldout_learned_diagnostics(
    key: Array,
    model: LearnedMFSIModel,
    a: float,
    times: Array,
    generated_trajectory: Array | None = None,
    generated_time_nodes: Array | None = None,
    *,
    bank_particles: int = 1024,
    mmd_particles: int = 512,
):
    """Evaluate fresh projected banks using only future-available diagnostics."""
    keys = jax.random.split(key, len(times))

    @jax.jit
    def state_metrics(k, t):
        x, _, fib = projected_batch(k, t, bank_particles, a, model.reference_params, TARGET)
        weak = weak_form_residual(model.potential_params, t, x, fib.projected_weights, fib.forcing)
        return (x, fib.projected_weights, fib.calibration_residual, fib.ess_fraction,
                fib.covariance_rank, fib.covariance_condition, weak)

    @jax.jit
    def mmd_metric(x, w, y):
        return weighted_mmd_rbf(x, w, y)

    rows = []
    for k, t in zip(keys, np.asarray(times)):
        x, w, cres, ess, rank, cond, weak = state_metrics(k, jnp.asarray(t))
        row = {
            "t": float(t),
            "calibration_residual": float(cres),
            "ess_fraction": float(ess),
            "covariance_rank": int(rank),
            "covariance_condition": float(cond),
            "weak_form_residual": float(weak),
        }
        if generated_trajectory is not None:
            idx = int(np.argmin(np.abs(np.asarray(generated_time_nodes) - float(t))))
            yg_full = generated_trajectory[idx]
            mom = jnp.mean(phi(yg_full), axis=0)
            row["generated_moment_error"] = float(jnp.linalg.norm(mom - TARGET))
            xp = x[:mmd_particles]
            wp = w[:mmd_particles]
            wp = wp / jnp.sum(wp)
            yi = jnp.linspace(0, yg_full.shape[0] - 1, mmd_particles).astype(jnp.int32)
            yg = yg_full[yi]
            row["projected_mmd"] = float(mmd_metric(xp, wp, yg))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Existing oracle validation/ablation helpers
# ---------------------------------------------------------------------------

def pipeline_objective(params, grid, times, a, *, kind: Literal["correction_energy", "distortion"], differentiation: DiffMode = "implicit"):
    def one_t(t):
        _, fib = mfsi_pipeline(params, grid, t, a, differentiation=differentiation)
        return fib.correction_energy if kind == "correction_energy" else fib.projection_distortion
    return jnp.mean(jax.vmap(one_t)(times))


def fourth_moment(q: Array, grid: Grid) -> Array:
    return jnp.sum(grid.w * q * grid.x**4)


def density_time_derivative(params, grid, t, a, *, differentiation: DiffMode = "unrolled"):
    if differentiation == "implicit": differentiation = "unrolled"
    return jax.jacfwd(lambda s: mfsi_pipeline(params, grid, s, a, differentiation=differentiation)[1].q)(t)


def spatial_derivative(y: Array, x: Array) -> Array:
    dx = x[1] - x[0]
    interior = (y[2:] - y[:-2]) / (2.0 * dx)
    left = (y[1] - y[0]) / dx
    right = (y[-1] - y[-2]) / dx
    return jnp.concatenate([left[None], interior, right[None]])


def continuity_residual(q: Array, dqdt: Array, velocity: Array, grid: Grid) -> Array:
    return dqdt + spatial_derivative(q * velocity, grid.x)


def weighted_l2(y: Array, q: Array, grid: Grid) -> Array:
    return jnp.sqrt(jnp.sum(grid.w * q * y * y))


def tangent_only_velocity(reference: ReferenceState, fiber: FiberState, grid: Grid) -> Array:
    x, q, u = grid.x, fiber.q, reference.velocity
    jp = jphi(x)
    G = (jp.T * (grid.w * q)) @ jp
    m = jphi_times_velocity(x, u)
    r = jnp.sum(grid.w[:, None] * q[:, None] * m, axis=0)
    coeff, _, _ = _stable_cov_solve(G, r, damping=DEFAULT_DAMPING)
    return u - jp @ coeff


def save_learned_model(path, model: LearnedMFSIModel):
    """Persist only flat network weights; architecture is versioned in this module."""
    np.savez(
        path,
        reference_params=np.asarray(flatten_mlp(model.reference_params)),
        potential_params=np.asarray(flatten_mlp(model.potential_params)),
    )


def load_learned_model(path) -> LearnedMFSIModel:
    data = np.load(path)
    rp = unflatten_mlp(jnp.asarray(data["reference_params"]), REFERENCE_HIDDEN)
    pp = unflatten_mlp(jnp.asarray(data["potential_params"]), RITZ_HIDDEN)
    return LearnedMFSIModel(rp, pp)
