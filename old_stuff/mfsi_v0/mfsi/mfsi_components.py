"""Two-component JAX prototype for MFSI Example A.

There is deliberately NO Tesseract dependency in this file.  The public API is
already split along the two future Tesseract boundaries:

    ReferenceTransport(params, t) -> ReferenceState
    MomentFiberRealizer(ReferenceState) -> FiberState

The second component supports three differentiation modes for the inner
I-projection calibration:

    "unrolled" : differentiate through fixed Newton iterations;
    "implicit" : same primal Newton solve, custom implicit VJP;
    "stop"     : stop gradients through lambda* (negative-control ablation).

A future Tesseract implementation should preserve these component contracts.
Tesseract 1 can use ordinary JAX AD.  Tesseract 2 can expose the implicit VJP
implemented here rather than unrolling the nonlinear solve in its backward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jax.Array
DiffMode = Literal["unrolled", "implicit", "stop"]
DEFAULT_NEWTON_ITERS = 14
DEFAULT_DAMPING = 1e-10


@dataclass(frozen=True)
class Grid:
    x: Array
    w: Array


class ReferenceState(NamedTuple):
    """Output contract of future Tesseract 1: ReferenceTransport."""

    log_q_ref: Array  # normalized log-density on the quadrature grid
    q_ref: Array      # normalized density on the quadrature grid
    velocity: Array   # u_t(x) on the same grid


class FiberState(NamedTuple):
    """Output contract of future Tesseract 2: MomentFiberRealizer."""

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


TARGET = jnp.array([0.0, 1.0], dtype=jnp.float64)


def make_grid(xmax: float = 7.0, n: int = 2001) -> Grid:
    x = jnp.linspace(-xmax, xmax, n)
    dx = x[1] - x[0]
    w = jnp.ones_like(x) * dx
    w = w.at[0].set(0.5 * dx)
    w = w.at[-1].set(0.5 * dx)
    return Grid(x=x, w=w)


def phi(x: Array) -> Array:
    return jnp.stack([x, x * x], axis=-1)


def jphi_times_velocity(x: Array, velocity: Array) -> Array:
    """J_Phi(x) u(x) for Phi=(x,x^2), without materializing a Jacobian."""
    return jnp.stack([velocity, 2.0 * x * velocity], axis=-1)


def inverse_softplus(y: float) -> float:
    return float(jnp.log(jnp.expm1(y)))


# ---------------------------------------------------------------------------
# TESSERACT 1 CANDIDATE: REFERENCE TRANSPORT
# ---------------------------------------------------------------------------

def beta_schedule(t: Array, params: Array | None = None) -> Array:
    """Positive bridge-noise amplitude beta_xi(t).

    params=None gives the plain independent linear bridge (beta=0).  Otherwise
    beta is a softplus polynomial in z=2t-1.  Constant beta=1 is the known
    fiber-adapted solution for the first two moments in Example A.
    """
    if params is None:
        return jnp.asarray(0.0, dtype=jnp.float64)
    z = 2.0 * t - 1.0
    powers = jnp.arange(params.shape[0], dtype=jnp.float64)
    return jax.nn.softplus(jnp.sum(params * z**powers))


def gamma_schedule(t: Array, params: Array | None = None) -> Array:
    return jnp.sqrt(jnp.maximum(2.0 * t * (1.0 - t), 0.0)) * beta_schedule(t, params)


def _normal_logpdf(x: Array, mean: Array, var: Array) -> Array:
    return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)


def reference_component_variance(t: Array, a: float, params: Array | None = None) -> Array:
    vp = 1.0 - a * a
    g = gamma_schedule(t, params)
    return (1.0 - t) ** 2 + t * t * vp + g * g


def reference_component_variance_dt(t: Array, a: float, params: Array | None = None) -> Array:
    vp = 1.0 - a * a
    base_dt = -2.0 * (1.0 - t) + 2.0 * t * vp
    if params is None:
        return base_dt

    z = 2.0 * t - 1.0
    powers = jnp.arange(params.shape[0], dtype=jnp.float64)
    raw = jnp.sum(params * z**powers)
    beta = jax.nn.softplus(raw)
    k = jnp.arange(params.shape[0], dtype=jnp.float64)
    raw_dt = jnp.sum(
        jnp.where(k == 0, 0.0, 2.0 * k * params * z ** jnp.maximum(k - 1.0, 0.0))
    )
    beta_dt = jax.nn.sigmoid(raw) * raw_dt
    s = 2.0 * t * (1.0 - t)
    s_dt = 2.0 * (1.0 - 2.0 * t)
    gamma2_dt = s_dt * beta * beta + 2.0 * s * beta * beta_dt
    return base_dt + gamma2_dt


def reference_logpdf_raw(x: Array, t: Array, a: float, params: Array | None = None) -> Array:
    var = reference_component_variance(t, a, params)
    mu = t * a
    lp = _normal_logpdf(x, mu, var)
    lm = _normal_logpdf(x, -mu, var)
    return jax.scipy.special.logsumexp(jnp.stack([lp, lm], axis=0), axis=0) - jnp.log(2.0)


def reference_velocity(x: Array, t: Array, a: float, params: Array | None = None) -> Array:
    """Exact SI velocity u_t(x)=E[d_t I_t | X_t=x] for Example A."""
    var = reference_component_variance(t, a, params)
    dvar_dt = reference_component_variance_dt(t, a, params)
    k = 0.5 * dvar_dt / var
    label_mean = jnp.tanh((t * a * x) / var)
    return k * x + a * (1.0 - k * t) * label_mean


def _normalize_logdensity(logu: Array, weights: Array) -> tuple[Array, Array]:
    logz = jax.scipy.special.logsumexp(logu + jnp.log(weights))
    return logu - logz, logz


def reference_transport(
    params: Array | None,
    grid: Grid,
    t: Array,
    a: float,
) -> ReferenceState:
    """Future Tesseract 1 boundary.

    Inputs are bridge design parameters and time; outputs contain everything
    Tesseract 2 needs.  No moment-fiber logic occurs here.
    """
    raw = reference_logpdf_raw(grid.x, t, a, params)
    log_q_ref, _ = _normalize_logdensity(raw, grid.w)
    q_ref = jnp.exp(log_q_ref)
    u = reference_velocity(grid.x, t, a, params)
    return ReferenceState(log_q_ref=log_q_ref, q_ref=q_ref, velocity=u)


# ---------------------------------------------------------------------------
# TESSERACT 2 CANDIDATE: MOMENT-FIBER REALIZER
# ---------------------------------------------------------------------------

def projected_density_from_lambda(
    lam: Array,
    log_q_ref: Array,
    ph: Array,
    weights: Array,
) -> tuple[Array, Array, Array, Array]:
    log_q, logz_tilt = _normalize_logdensity(log_q_ref + ph @ lam, weights)
    q = jnp.exp(log_q)
    moments = jnp.sum(weights[:, None] * q[:, None] * ph, axis=0)
    centered = ph - moments
    cov = (centered.T * (weights * q)) @ centered
    return q, moments, cov, logz_tilt


def _newton_step_arrays(
    _: int,
    lam: Array,
    *,
    log_q_ref: Array,
    ph: Array,
    weights: Array,
    target: Array,
    damping: float,
) -> Array:
    _, moments, cov, _ = projected_density_from_lambda(lam, log_q_ref, ph, weights)
    step = jnp.linalg.solve(cov + damping * jnp.eye(target.size), moments - target)
    step = jnp.clip(step, -1.0, 1.0)
    return lam - step


def calibrate_lambda_unrolled(
    log_q_ref: Array,
    ph: Array,
    weights: Array,
    target: Array = TARGET,
    *,
    iterations: int = DEFAULT_NEWTON_ITERS,
    damping: float = DEFAULT_DAMPING,
) -> Array:
    """Fixed-iteration Newton solve; reverse mode differentiates the iterations."""
    lam0 = jnp.zeros(target.shape[0], dtype=log_q_ref.dtype)
    body = partial(
        _newton_step_arrays,
        log_q_ref=log_q_ref,
        ph=ph,
        weights=weights,
        target=target,
        damping=damping,
    )
    return jax.lax.fori_loop(0, iterations, body, lam0)


@jax.custom_vjp
def calibrate_lambda_implicit(
    log_q_ref: Array,
    ph: Array,
    weights: Array,
    target: Array,
) -> Array:
    """Same primal solve as unrolled Newton, but with an implicit backward pass.

    For F(lambda,log q_ref)=E_q[Phi]-target=0,
        d lambda = - C^{-1} dF.

    This is the local JAX prototype for the future solver-aware VJP of
    MomentFiberRealizer.  Phi and quadrature weights are treated as fixed
    scientific definitions in this toy example.
    """
    return calibrate_lambda_unrolled(log_q_ref, ph, weights, target)


def _calibrate_implicit_fwd(log_q_ref, ph, weights, target):
    lam = calibrate_lambda_unrolled(log_q_ref, ph, weights, target)
    q, moments, cov, _ = projected_density_from_lambda(lam, log_q_ref, ph, weights)
    probs = weights * q  # discrete probability mass on quadrature nodes
    return lam, (probs, ph, moments, cov)


def _calibrate_implicit_bwd(res, lambda_bar):
    probs, ph, moments, cov = res
    a = jnp.linalg.solve(cov.T + DEFAULT_DAMPING * jnp.eye(cov.shape[0]), lambda_bar)
    centered = ph - moments
    log_q_ref_bar = -probs * (centered @ a)
    target_bar = a
    # Phi and quadrature weights are fixed in Example A.  Returning zeros makes
    # that design decision explicit and keeps the future component contract clean.
    return (
        log_q_ref_bar,
        jnp.zeros_like(ph),
        jnp.zeros_like(probs),
        target_bar,
    )


calibrate_lambda_implicit.defvjp(_calibrate_implicit_fwd, _calibrate_implicit_bwd)


def calibrate_lambda(
    log_q_ref: Array,
    ph: Array,
    weights: Array,
    target: Array = TARGET,
    *,
    differentiation: DiffMode = "implicit",
) -> Array:
    if differentiation == "implicit":
        return calibrate_lambda_implicit(log_q_ref, ph, weights, target)
    lam = calibrate_lambda_unrolled(log_q_ref, ph, weights, target)
    if differentiation == "stop":
        return jax.lax.stop_gradient(lam)
    if differentiation == "unrolled":
        return lam
    raise ValueError(f"unknown differentiation mode: {differentiation}")


def cumulative_trapezoid(y: Array, x: Array) -> Array:
    dx = x[1:] - x[:-1]
    increments = 0.5 * (y[1:] + y[:-1]) * dx
    return jnp.concatenate([jnp.zeros(1, dtype=y.dtype), jnp.cumsum(increments)])


def exact_poisson_correction(q: Array, h: Array, grid: Grid, q_floor: float = 1e-12) -> Array:
    """Exact 1D solution delta=-psi' of d_x(q psi')=q h."""
    flux = cumulative_trapezoid(q * h, grid.x)
    # Correct tiny accumulated quadrature drift so both boundary fluxes vanish.
    total = flux[-1]
    cdf_q = cumulative_trapezoid(q, grid.x)
    flux = flux - total * cdf_q
    delta = -flux / jnp.maximum(q, q_floor)
    return jnp.where(q > q_floor, delta, 0.0)


def moment_fiber_realizer(
    reference: ReferenceState,
    grid: Grid,
    target: Array = TARGET,
    *,
    differentiation: DiffMode = "implicit",
) -> FiberState:
    """Future Tesseract 2 boundary: project and realize the complete law path.

    Importantly, this function knows nothing about `a`, the bridge schedule, or
    upstream parameters.  It consumes only Tesseract 1 outputs.
    """
    ph = phi(grid.x)
    lam = calibrate_lambda(
        reference.log_q_ref,
        ph,
        grid.w,
        target,
        differentiation=differentiation,
    )
    q, moments, cov, logz_tilt = projected_density_from_lambda(
        lam, reference.log_q_ref, ph, grid.w
    )

    m = jphi_times_velocity(grid.x, reference.velocity)
    em = jnp.sum(grid.w[:, None] * q[:, None] * m, axis=0)
    scalar = m @ lam
    cov_term = jnp.sum(
        grid.w[:, None] * q[:, None] * (ph - target) * scalar[:, None], axis=0
    )
    lambda_dot = jnp.linalg.solve(
        cov + DEFAULT_DAMPING * jnp.eye(target.size), -em - cov_term
    )
    h = (ph - target) @ lambda_dot + (m - em) @ lam
    delta = exact_poisson_correction(q, h, grid)
    v = reference.velocity + delta

    energy = jnp.sum(grid.w * q * delta * delta)
    # reference.log_q_ref is normalized, hence A(lambda)=log integral ref*exp(...)
    distortion = lam @ target - logz_tilt
    ess = 1.0 / jnp.sum(grid.w * q * q / jnp.maximum(reference.q_ref, 1e-300))

    return FiberState(
        lambda_=lam,
        q=q,
        moments=moments,
        covariance=cov,
        lambda_dot=lambda_dot,
        forcing=h,
        correction=delta,
        velocity=v,
        correction_energy=energy,
        projection_distortion=distortion,
        ess_fraction=ess,
    )


# ---------------------------------------------------------------------------
# COMPOSED PIPELINE + VALIDATION HELPERS
# ---------------------------------------------------------------------------

def mfsi_pipeline(
    params: Array | None,
    grid: Grid,
    t: Array,
    a: float,
    *,
    differentiation: DiffMode = "implicit",
    target: Array = TARGET,
) -> tuple[ReferenceState, FiberState]:
    ref = reference_transport(params, grid, t, a)
    fib = moment_fiber_realizer(ref, grid, target, differentiation=differentiation)
    return ref, fib


def pipeline_objective(
    params: Array,
    grid: Grid,
    times: Array,
    a: float,
    *,
    kind: Literal["correction_energy", "distortion"],
    differentiation: DiffMode = "implicit",
) -> Array:
    def one_t(t):
        _, fib = mfsi_pipeline(params, grid, t, a, differentiation=differentiation)
        if kind == "correction_energy":
            return fib.correction_energy
        if kind == "distortion":
            return fib.projection_distortion
        raise ValueError(kind)

    return jnp.mean(jax.vmap(one_t)(times))


def fourth_moment(q: Array, grid: Grid) -> Array:
    return jnp.sum(grid.w * q * grid.x**4)


def density_time_derivative(
    params: Array | None,
    grid: Grid,
    t: Array,
    a: float,
    *,
    differentiation: DiffMode = "unrolled",
) -> Array:
    """Forward-mode oracle for d_t q_t.

    Use ``unrolled`` here: JAX ``custom_vjp`` rules are reverse-mode only and
    therefore do not support ``jacfwd``. The primal projected density is the
    same for implicit and unrolled calibration; this routine is only an
    independent validation oracle.
    """
    if differentiation == "implicit":
        differentiation = "unrolled"

    def q_at_time(s):
        return mfsi_pipeline(params, grid, s, a, differentiation=differentiation)[1].q

    return jax.jacfwd(q_at_time)(t)


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
    """Eq. 45-style moment-rate correction; deliberately not a full-law solver."""
    x, q, u = grid.x, fiber.q, reference.velocity
    jphi = jnp.stack([jnp.ones_like(x), 2.0 * x], axis=-1)
    G = (jphi.T * (grid.w * q)) @ jphi
    m = jphi_times_velocity(x, u)
    r = jnp.sum(grid.w[:, None] * q[:, None] * m, axis=0)
    coeff = jnp.linalg.solve(G + DEFAULT_DAMPING * jnp.eye(2), r)
    return u - jphi @ coeff
