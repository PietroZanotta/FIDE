#!/usr/bin/env python3
"""
Stage A: Differentiable transport-conditioned experimental design on a moment fiber.

Analytic 2D Gaussian validation problem implemented in JAX.

Scientific setup
----------------
Reference stochastic interpolant:
    Q~_t = N(mu_ref(t), I),
    mu_ref(t) = ((2t-1)d, 0).

External law:
    P_t = N(mu_ref(t) + h(t), I),
    h(t) = (a sin(pi t), b sin(3 pi t)).

One physical directional population sensor:
    Phi_theta(x) = e_theta^T x,
    e_theta = (cos theta, sin theta).

The measurement is c_theta(t) = E_{P_t}[Phi_theta].  The I-projection
of Q~_t onto this moment fiber is computed by a scalar exponential tilt.
The scalar tilt lambda is solved numerically, but differentiated using an
implicit custom VJP (i.e. the Newton iterations are not unrolled).

For this Gaussian-linear Stage A toy, the weighted-Poisson MFSI correction
has an analytic solution.  We still compute it from the implicit calibration
time derivative, so the numerical pipeline mirrors the intended project:

    theta -> measurement -> lambda* -> projected law
          -> continuity forcing -> Poisson correction -> action.

The script validates every stage against closed-form expressions, compares
baselines, runs a differentiable constrained-design optimizer, checks the
dense oracle, prints a compact report, and writes JSON.

Run:
    python stage_a_transport_conditioned_design.py

Typical GPU-enabled JAX execution is automatic if a GPU-enabled jaxlib is
installed; the script prints the backend/devices it actually sees.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import jax

# Stage A is a numerical validation experiment, so use double precision.
# This must be configured before creating JAX arrays.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

Array = jax.Array
PI = jnp.pi
DTYPE = jnp.float64


# -----------------------------------------------------------------------------
# Core scientific model
# -----------------------------------------------------------------------------

def sensor_direction(theta: Array) -> Array:
    """Unit vector e_theta defining the physical directional sensor."""
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)])


def sensor_direction_derivative(theta: Array) -> Array:
    """d e_theta / d theta."""
    return jnp.stack([-jnp.sin(theta), jnp.cos(theta)])


def reference_mean(t: Array, d: Array) -> Array:
    """Mean of the analytic reference stochastic-interpolant marginal."""
    return jnp.stack([(2.0 * t - 1.0) * d, jnp.zeros_like(t)], axis=-1)


def reference_mean_dot(t: Array, d: Array) -> Array:
    """Time derivative of the reference mean."""
    return jnp.broadcast_to(jnp.array([2.0 * d, 0.0], dtype=DTYPE), t.shape + (2,))


def hidden_displacement(t: Array, a: Array, b: Array) -> Array:
    """External-law displacement relative to the reference SI."""
    return jnp.stack(
        [a * jnp.sin(PI * t), b * jnp.sin(3.0 * PI * t)], axis=-1
    )


def hidden_displacement_dot(t: Array, a: Array, b: Array) -> Array:
    """Time derivative of the external-law displacement."""
    return jnp.stack(
        [a * PI * jnp.cos(PI * t), 3.0 * b * PI * jnp.cos(3.0 * PI * t)],
        axis=-1,
    )


def external_mean(t: Array, a: Array, b: Array, d: Array) -> Array:
    return reference_mean(t, d) + hidden_displacement(t, a, b)


def external_mean_dot(t: Array, a: Array, b: Array, d: Array) -> Array:
    return reference_mean_dot(t, d) + hidden_displacement_dot(t, a, b)


def sensor_moment_from_mean(theta: Array, mean: Array) -> Array:
    """E[Phi_theta] for any distribution whose mean is `mean`."""
    return jnp.einsum("...i,i->...", mean, sensor_direction(theta))


def target_moment(theta: Array, t: Array, a: Array, b: Array, d: Array) -> Array:
    """Population measurement c_theta(t) inherited from the external law."""
    return sensor_moment_from_mean(theta, external_mean(t, a, b, d))


# -----------------------------------------------------------------------------
# Moment-fiber I-projection: numerical forward solve + implicit VJP
# -----------------------------------------------------------------------------

def calibration_residual(
    lam: Array, theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """
    Scalar calibration equation F(lambda, theta, t) = 0.

    Tilting N(mu_ref, I) by exp(lambda * e_theta^T x) gives
        N(mu_ref + lambda e_theta, I).
    F is the difference between its predicted sensor moment and c_theta(t).
    """
    e = sensor_direction(theta)
    projected_mu = reference_mean(t, d) + lam[..., None] * e
    return sensor_moment_from_mean(theta, projected_mu) - target_moment(
        theta, t, a, b, d
    )


def _newton_solve_lambda(
    theta: Array, t: Array, a: Array, b: Array, d: Array, n_iter: int = 6
) -> Array:
    """Primal calibration solve. Iterations are deliberately not differentiated."""
    lam0 = jnp.zeros_like(t, dtype=DTYPE)

    def body(_, lam):
        f = calibration_residual(lam, theta, t, a, b, d)
        # In this problem dF/dlambda = Var(Phi_theta) = 1 exactly.
        # Keep the formula explicit so Stage A exposes the moment geometry.
        f_lam = jnp.ones_like(f)
        return lam - f / f_lam

    return jax.lax.fori_loop(0, n_iter, body, lam0)


@jax.custom_vjp
def solve_lambda(theta: Array, t: Array, a: Array, b: Array, d: Array) -> Array:
    """Calibrated I-projection multiplier with implicit reverse-mode derivative."""
    return _newton_solve_lambda(theta, t, a, b, d)


def _solve_lambda_fwd(theta, t, a, b, d):
    lam = _newton_solve_lambda(theta, t, a, b, d)
    return lam, (lam, theta, t, a, b, d)


def _solve_lambda_bwd(res, cotangent):
    lam, theta, t, a, b, d = res

    # Implicit-function theorem:
    #   d lambda*/d y = - F_y / F_lambda.
    # `t` may be a vector in some calls, while theta/a/b/d are scalars.
    # We form vector-Jacobian products directly to avoid materializing a Jacobian.
    # Each time point is an independent scalar root.  Form the diagonal
    # F_lambda explicitly so the backward rule is the actual implicit VJP, not
    # an identity-Jacobian shortcut.
    if t.ndim == 0:
        f_lam = jax.grad(calibration_residual, argnums=0)(
            lam, theta, t, a, b, d
        )
    else:
        f_lam = jax.vmap(
            lambda ll, tt: jax.grad(calibration_residual, argnums=0)(
                ll, theta, tt, a, b, d
            )
        )(lam, t)

    # Solve F_lambda^T alpha = cotangent.  It is elementwise because the
    # calibrations are independent across time points.
    alpha = cotangent / f_lam

    def weighted_F(th, tt, aa, bb, dd):
        return jnp.sum(calibration_residual(lam, th, tt, aa, bb, dd) * alpha)

    g_theta, g_t, g_a, g_b, g_d = jax.grad(
        weighted_F, argnums=(0, 1, 2, 3, 4)
    )(theta, t, a, b, d)

    # The above gradients are alpha^T F_y. Apply the minus sign from IFT.
    return (-g_theta, -g_t, -g_a, -g_b, -g_d)


solve_lambda.defvjp(_solve_lambda_fwd, _solve_lambda_bwd)


def projected_mean(
    theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Tuple[Array, Array]:
    """Return (mu_I, lambda*) for the I-projected law."""
    lam = solve_lambda(theta, t, a, b, d)
    mu_i = reference_mean(t, d) + lam[..., None] * sensor_direction(theta)
    return mu_i, lam


def implicit_lambda_dot(
    lam: Array, theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """
    Compute d lambda*/dt from the calibration equation rather than by unrolling.

        F_lambda lambda_dot + F_t = 0.
    """
    # Vectorized scalar-root derivatives.
    def one(lam_i, t_i):
        f_lam = jax.grad(calibration_residual, argnums=0)(
            lam_i, theta, t_i, a, b, d
        )
        f_t = jax.grad(calibration_residual, argnums=2)(
            lam_i, theta, t_i, a, b, d
        )
        return -f_t / f_lam

    if t.ndim == 0:
        return one(lam, t)
    return jax.vmap(one)(lam, t)


# -----------------------------------------------------------------------------
# Exact Stage-A weighted-Poisson realization
# -----------------------------------------------------------------------------

def correction_velocity_from_poisson(
    theta: Array, lam_dot: Array
) -> Array:
    """
    Exact minimum-energy MFSI correction for Stage A.

    The continuity forcing is
        h_res(x) = lambda_dot * e_theta^T (x - mu_I).
    The weighted-Poisson potential can be chosen as
        psi(x) = -lambda_dot e_theta^T x,
    hence delta = -grad psi = lambda_dot e_theta.
    """
    return lam_dot[..., None] * sensor_direction(theta)


def forcing_value(
    x: Array, theta: Array, mu_i: Array, lam_dot: Array
) -> Array:
    centered_sensor = jnp.einsum("...i,i->...", x - mu_i, sensor_direction(theta))
    return lam_dot * centered_sensor


def poisson_lhs_over_q(
    x: Array, theta: Array, mu_i: Array, lam_dot: Array
) -> Array:
    """
    Evaluate [div(q grad psi)] / q for the analytic linear potential.
    For q=N(mu_I,I), grad log q = -(x-mu_I), and Delta psi=0.
    """
    grad_psi = -lam_dot[..., None] * sensor_direction(theta)
    grad_log_q = -(x - mu_i)
    return jnp.einsum("...i,...i->...", grad_log_q, grad_psi)


# -----------------------------------------------------------------------------
# Quadrature and objectives
# -----------------------------------------------------------------------------

def trapz_uniform(values: Array) -> Array:
    """Trapezoidal integral on [0,1] along the last axis."""
    n = values.shape[-1]
    dt = 1.0 / (n - 1)
    return dt * (0.5 * values[..., 0] + jnp.sum(values[..., 1:-1], axis=-1) + 0.5 * values[..., -1])


def time_grid(n_time: int) -> Array:
    return jnp.linspace(0.0, 1.0, n_time, dtype=DTYPE)


def lift_loss_pipeline(
    theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """Integrated W2^2 between I-projected and external Gaussians."""
    mu_i, _ = projected_mean(theta, t, a, b, d)
    mu_p = external_mean(t, a, b, d)
    pointwise = jnp.sum((mu_i - mu_p) ** 2, axis=-1)
    return trapz_uniform(pointwise)


def correction_action_pipeline(
    theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """Integrated exact MFSI correction action through the inverse pipeline."""
    _, lam = projected_mean(theta, t, a, b, d)
    lam_dot = implicit_lambda_dot(lam, theta, t, a, b, d)
    delta = correction_velocity_from_poisson(theta, lam_dot)
    energy = jnp.sum(delta**2, axis=-1)
    return trapz_uniform(energy)


def local_tangent_action_pipeline(
    theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """
    Minimum moment-rate correction using the kinetic tangent Gram formula.
    For a linear unit sensor, G=1 and this should exactly match Stage-A Poisson action.
    """
    e = sensor_direction(theta)
    u_ref = reference_mean_dot(t, d)
    c_dot = jnp.einsum("...i,i->...", external_mean_dot(t, a, b, d), e)
    reference_moment_rate = jnp.einsum("...i,i->...", u_ref, e)
    residual = reference_moment_rate - c_dot
    # delta_tangent = - e * residual because G = e^T e = 1.
    delta = -residual[..., None] * e
    return trapz_uniform(jnp.sum(delta**2, axis=-1))


def information_score_pipeline(
    theta: Array, t: Array, a: Array, b: Array, d: Array
) -> Array:
    """
    Population-signal score relative to the reference mean, with unit noise variance.
    Larger is better.  This is the relevant Stage-A Info-Population analogue.
    """
    signal = target_moment(theta, t, a, b, d) - sensor_moment_from_mean(
        theta, reference_mean(t, d)
    )
    return trapz_uniform(signal**2)


# Closed-form objectives used only for validation/oracle interpretation.
def lift_loss_analytic(theta: Array, a: Array, b: Array) -> Array:
    return 0.5 * (a**2 * jnp.sin(theta) ** 2 + b**2 * jnp.cos(theta) ** 2)

def action_analytic(theta: Array, a: Array, b: Array) -> Array:
    return 0.5 * PI**2 * (
        a**2 * jnp.cos(theta) ** 2 + 9.0 * b**2 * jnp.sin(theta) ** 2
    )


def info_analytic(theta: Array, a: Array, b: Array) -> Array:
    return 0.5 * (a**2 * jnp.cos(theta) ** 2 + b**2 * jnp.sin(theta) ** 2)


def grad_lift_analytic(theta: Array, a: Array, b: Array) -> Array:
    return 0.5 * (a**2 - b**2) * jnp.sin(2.0 * theta)


def grad_action_analytic(theta: Array, a: Array, b: Array) -> Array:
    return 0.5 * PI**2 * (9.0 * b**2 - a**2) * jnp.sin(2.0 * theta)


# -----------------------------------------------------------------------------
# Differentiable optimization
# -----------------------------------------------------------------------------

def wrap_theta(theta: Array) -> Array:
    """Sensor orientation is pi-periodic."""
    return jnp.mod(theta, PI)


def adam_optimize_scalar(
    objective,
    theta0: float,
    steps: int,
    lr: float,
) -> Tuple[Array, Array]:
    """Small dependency-free Adam optimizer for a scalar design variable."""
    theta = jnp.asarray(theta0, dtype=DTYPE)
    m = jnp.asarray(0.0, dtype=DTYPE)
    v = jnp.asarray(0.0, dtype=DTYPE)
    beta1 = jnp.asarray(0.9, dtype=DTYPE)
    beta2 = jnp.asarray(0.999, dtype=DTYPE)
    eps = jnp.asarray(1e-10, dtype=DTYPE)
    lr_j = jnp.asarray(lr, dtype=DTYPE)

    value_and_grad = jax.value_and_grad(objective)

    def step(carry, i):
        th, m_i, v_i = carry
        value, grad = value_and_grad(th)
        m_i = beta1 * m_i + (1.0 - beta1) * grad
        v_i = beta2 * v_i + (1.0 - beta2) * grad**2
        i1 = i.astype(DTYPE) + 1.0
        m_hat = m_i / (1.0 - beta1**i1)
        v_hat = v_i / (1.0 - beta2**i1)
        th = wrap_theta(th - lr_j * m_hat / (jnp.sqrt(v_hat) + eps))
        return (th, m_i, v_i), value

    (theta, _, _), values = jax.lax.scan(
        step, (theta, m, v), jnp.arange(steps)
    )
    return theta, values[-1]


def multistart_optimize(
    objective,
    starts: Iterable[float],
    steps: int,
    lr: float,
) -> Tuple[float, float]:
    """Run deterministic multi-start optimization and return the best result."""
    candidates = []
    compiled = jax.jit(lambda th0: adam_optimize_scalar(objective, th0, steps, lr))
    for s in starts:
        th, val = compiled(jnp.asarray(s, dtype=DTYPE))
        th_f = float(wrap_theta(th))
        val_f = float(objective(jnp.asarray(th_f, dtype=DTYPE)))
        candidates.append((val_f, th_f))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def central_difference(fun, x: float, eps: float = 1e-5) -> float:
    return float((fun(x + eps) - fun(x - eps)) / (2.0 * eps))


def rel_error(x: float, y: float, floor: float = 1e-12) -> float:
    return abs(x - y) / max(abs(y), floor)


def _float(x: Any) -> float:
    return float(jax.device_get(x))


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    try:
        arr = jax.device_get(x)
        if getattr(arr, "ndim", None) == 0:
            return float(arr)
        return arr.tolist()
    except Exception:
        return str(x)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    a = jnp.asarray(args.a, dtype=DTYPE)
    b = jnp.asarray(args.b, dtype=DTYPE)
    d = jnp.asarray(args.d, dtype=DTYPE)
    tau = float(args.tau)
    t = time_grid(args.n_time)

    lift_fn = jax.jit(lambda th: lift_loss_pipeline(th, t, a, b, d))
    action_fn = jax.jit(lambda th: correction_action_pipeline(th, t, a, b, d))
    tangent_fn = jax.jit(lambda th: local_tangent_action_pipeline(th, t, a, b, d))
    info_fn = jax.jit(lambda th: information_score_pipeline(th, t, a, b, d))

    # Trigger compilation before timing/printing and make device placement explicit.
    probe = jnp.asarray(args.probe_theta, dtype=DTYPE)
    _ = (lift_fn(probe), action_fn(probe), tangent_fn(probe), info_fn(probe))
    jax.block_until_ready(action_fn(probe))

    # ------------------------------------------------------------------
    # Exact-vs-pipeline validation
    # ------------------------------------------------------------------
    probe_t = jnp.asarray(args.probe_time, dtype=DTYPE)
    lam_probe = solve_lambda(probe, probe_t, a, b, d)
    lam_expected = jnp.dot(sensor_direction(probe), hidden_displacement(probe_t, a, b))
    dlam_dtheta_ad = jax.grad(lambda th: solve_lambda(th, probe_t, a, b, d))(probe)
    dlam_dtheta_expected = jnp.dot(
        sensor_direction_derivative(probe), hidden_displacement(probe_t, a, b)
    )

    lift_num = lift_fn(probe)
    lift_exact = lift_loss_analytic(probe, a, b)
    action_num = action_fn(probe)
    action_exact = action_analytic(probe, a, b)
    tangent_num = tangent_fn(probe)
    info_num = info_fn(probe)
    info_exact = info_analytic(probe, a, b)

    grad_lift_ad = jax.grad(lift_fn)(probe)
    grad_action_ad = jax.grad(action_fn)(probe)
    grad_lift_exact = grad_lift_analytic(probe, a, b)
    grad_action_exact = grad_action_analytic(probe, a, b)

    grad_lift_fd = central_difference(lambda x: _float(lift_fn(jnp.asarray(x, dtype=DTYPE))), args.probe_theta)
    grad_action_fd = central_difference(lambda x: _float(action_fn(jnp.asarray(x, dtype=DTYPE))), args.probe_theta)

    # Weighted-Poisson identity on deterministic test points.
    test_x = jnp.asarray(
        [[-1.1, 0.4], [0.0, -0.7], [0.8, 1.3], [1.5, -1.2]], dtype=DTYPE
    )
    mu_i_probe, lam_probe_2 = projected_mean(probe, probe_t, a, b, d)
    lam_dot_probe = implicit_lambda_dot(lam_probe_2, probe, probe_t, a, b, d)
    forcing = forcing_value(test_x, probe, mu_i_probe, lam_dot_probe)
    poisson_lhs = poisson_lhs_over_q(test_x, probe, mu_i_probe, lam_dot_probe)
    poisson_residual_max = jnp.max(jnp.abs(forcing - poisson_lhs))

    # ------------------------------------------------------------------
    # Dense design oracle
    # ------------------------------------------------------------------
    thetas = jnp.linspace(0.0, PI, args.n_theta + 1, dtype=DTYPE)[:-1]
    # Use analytic vectorized objectives for the dense oracle.  The pipeline has
    # already been validated above; this makes an 8k+ point oracle instantaneous.
    lift_land = jax.jit(jax.vmap(lambda th: lift_loss_analytic(th, a, b)))(thetas)
    action_land = jax.jit(jax.vmap(lambda th: action_analytic(th, a, b)))(thetas)
    info_land = jax.jit(jax.vmap(lambda th: info_analytic(th, a, b)))(thetas)

    idx_lift = int(jnp.argmin(lift_land))
    theta_lift_oracle = _float(thetas[idx_lift])
    lift_star = _float(lift_land[idx_lift])
    lift_bound = (1.0 + tau) * lift_star

    feasible = lift_land <= lift_bound + args.feasibility_eps
    masked_action = jnp.where(feasible, action_land, jnp.inf)
    idx_tc = int(jnp.argmin(masked_action))
    theta_tc_oracle = _float(thetas[idx_tc])

    idx_info = int(jnp.argmax(info_land))
    theta_info_oracle = _float(thetas[idx_info])

    idx_action = int(jnp.argmin(action_land))
    theta_action_oracle = _float(thetas[idx_action])

    # ------------------------------------------------------------------
    # Differentiable optimizers
    # ------------------------------------------------------------------
    starts = [
        (k + 0.37) * math.pi / args.n_starts for k in range(args.n_starts)
    ]

    theta_lift_opt, _ = multistart_optimize(
        lift_fn, starts, args.opt_steps, args.lr
    )

    theta_info_opt, _ = multistart_optimize(
        lambda th: -info_fn(th), starts, args.opt_steps, args.lr
    )

    # Transport-conditioned objective.  The Lift constraint is normalized so
    # penalty_weight is dimensionless and portable across modest parameter changes.
    lift_bound_j = jnp.asarray(lift_bound, dtype=DTYPE)

    def tc_penalized(th):
        l = lift_fn(th)
        g = (l - lift_bound_j) / jnp.maximum(lift_bound_j, jnp.asarray(1e-12, DTYPE))
        return action_fn(th) + args.penalty_weight * jax.nn.relu(g) ** 2

    theta_tc_opt, _ = multistart_optimize(
        tc_penalized, starts, args.opt_steps, args.lr
    )

    # If penalty optimization lands microscopically outside the constraint, use
    # the dense feasible oracle as the safe result and flag it explicitly.
    tc_opt_lift = _float(lift_fn(jnp.asarray(theta_tc_opt, DTYPE)))
    tc_optimizer_feasible = tc_opt_lift <= lift_bound + args.feasibility_eps

    # ------------------------------------------------------------------
    # Baseline summaries
    # ------------------------------------------------------------------
    def summarize_theta(theta_value: float) -> Dict[str, float]:
        th = jnp.asarray(theta_value, dtype=DTYPE)
        return {
            "theta_rad": theta_value,
            "theta_deg": theta_value * 180.0 / math.pi,
            "lift_loss": _float(lift_fn(th)),
            "correction_action": _float(action_fn(th)),
            "local_tangent_action": _float(tangent_fn(th)),
            "information_score": _float(info_fn(th)),
        }

    lift_summary = summarize_theta(theta_lift_oracle)
    tc_summary = summarize_theta(theta_tc_oracle)
    info_summary = summarize_theta(theta_info_oracle)
    action_summary = summarize_theta(theta_action_oracle)

    headroom = 1.0 - tc_summary["correction_action"] / lift_summary["correction_action"]
    lift_sacrifice = tc_summary["lift_loss"] / lift_star - 1.0

    # Small downsampled landscape in JSON, enough for later plotting without
    # making the result file unwieldy.
    n_json = min(args.landscape_points_json, args.n_theta)
    json_idx = jnp.linspace(0, args.n_theta - 1, n_json).astype(jnp.int32)
    landscape = {
        "theta_rad": jax.device_get(thetas[json_idx]).tolist(),
        "lift_loss": jax.device_get(lift_land[json_idx]).tolist(),
        "correction_action": jax.device_get(action_land[json_idx]).tolist(),
        "information_score": jax.device_get(info_land[json_idx]).tolist(),
    }

    validation = {
        "lambda_abs_error": abs(_float(lam_probe) - _float(lam_expected)),
        "lambda_theta_grad_abs_error": abs(_float(dlam_dtheta_ad) - _float(dlam_dtheta_expected)),
        "lift_abs_error": abs(_float(lift_num) - _float(lift_exact)),
        "action_abs_error": abs(_float(action_num) - _float(action_exact)),
        "info_abs_error": abs(_float(info_num) - _float(info_exact)),
        "poisson_residual_max_abs": _float(poisson_residual_max),
        "tangent_vs_poisson_action_abs_error": abs(_float(tangent_num) - _float(action_num)),
        "lift_grad_ad": _float(grad_lift_ad),
        "lift_grad_analytic": _float(grad_lift_exact),
        "lift_grad_fd": grad_lift_fd,
        "lift_grad_rel_error_vs_analytic": rel_error(_float(grad_lift_ad), _float(grad_lift_exact)),
        "lift_grad_rel_error_vs_fd": rel_error(_float(grad_lift_ad), grad_lift_fd),
        "action_grad_ad": _float(grad_action_ad),
        "action_grad_analytic": _float(grad_action_exact),
        "action_grad_fd": grad_action_fd,
        "action_grad_rel_error_vs_analytic": rel_error(_float(grad_action_ad), _float(grad_action_exact)),
        "action_grad_rel_error_vs_fd": rel_error(_float(grad_action_ad), grad_action_fd),
    }

    expected_action_ratio = 9.0 * args.b**2 / args.a**2
    expected_lift_sacrifice_x = args.b**2 / args.a**2 - 1.0

    result: Dict[str, Any] = {
        "experiment": "stage_a_transport_conditioned_moment_fiber_design",
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(x) for x in jax.devices()],
            "x64_enabled": bool(jax.config.jax_enable_x64),
        },
        "config": {
            "a": args.a,
            "b": args.b,
            "d": args.d,
            "tau": tau,
            "n_time": args.n_time,
            "n_theta": args.n_theta,
            "n_starts": args.n_starts,
            "opt_steps": args.opt_steps,
            "lr": args.lr,
            "penalty_weight": args.penalty_weight,
            "probe_theta": args.probe_theta,
            "probe_time": args.probe_time,
        },
        "validation": validation,
        "dense_oracle": {
            "lift_star": lift_star,
            "lift_bound": lift_bound,
            "transport_headroom_fraction": headroom,
            "transport_headroom_percent": 100.0 * headroom,
            "lift_sacrifice_fraction": lift_sacrifice,
            "lift_sacrifice_percent": 100.0 * lift_sacrifice,
            "expected_endpoint_direction_action_ratio_fast_over_slow": expected_action_ratio,
            "expected_slow_direction_lift_sacrifice_fraction": expected_lift_sacrifice_x,
            "lift": lift_summary,
            "transport_conditioned": tc_summary,
            "info_population": info_summary,
            "action_only": action_summary,
        },
        "differentiable_optimizer": {
            "lift": summarize_theta(theta_lift_opt),
            "info_population": summarize_theta(theta_info_opt),
            "transport_conditioned": summarize_theta(theta_tc_opt),
            "transport_conditioned_feasible": bool(tc_optimizer_feasible),
            "theta_error_to_dense_oracle_rad": {
                "lift": min(
                    abs(theta_lift_opt - theta_lift_oracle),
                    math.pi - abs(theta_lift_opt - theta_lift_oracle),
                ),
                "info_population": min(
                    abs(theta_info_opt - theta_info_oracle),
                    math.pi - abs(theta_info_opt - theta_info_oracle),
                ),
                "transport_conditioned": min(
                    abs(theta_tc_opt - theta_tc_oracle),
                    math.pi - abs(theta_tc_opt - theta_tc_oracle),
                ),
            },
        },
        "landscape_downsampled": landscape,
    }

    # Pass/fail checks use tolerances appropriate for x64 and trapezoidal quadrature.
    checks = {
        "lambda": validation["lambda_abs_error"] < 1e-10,
        "implicit_lambda_vjp": validation["lambda_theta_grad_abs_error"] < 1e-9,
        "lift_closed_form": validation["lift_abs_error"] < 5e-7,
        "action_closed_form": validation["action_abs_error"] < 2e-5,
        "poisson_identity": validation["poisson_residual_max_abs"] < 1e-10,
        "tangent_positive_control": validation["tangent_vs_poisson_action_abs_error"] < 2e-5,
        "lift_gradient": validation["lift_grad_rel_error_vs_analytic"] < 2e-6,
        "action_gradient": validation["action_grad_rel_error_vs_analytic"] < 2e-6,
        "tc_dense_feasible": tc_summary["lift_loss"] <= lift_bound + args.feasibility_eps,
        "tc_optimizer_feasible": bool(tc_optimizer_feasible),
    }
    result["checks"] = checks
    result["all_checks_pass"] = all(checks.values())

    return result


def print_report(result: Dict[str, Any]) -> None:
    cfg = result["config"]
    val = result["validation"]
    oracle = result["dense_oracle"]
    opt = result["differentiable_optimizer"]

    print("=" * 78)
    print("STAGE A — DIFFERENTIABLE TRANSPORT-CONDITIONED MOMENT-FIBER DESIGN")
    print("=" * 78)
    print(f"JAX {result['software']['jax_version']} | backend={result['software']['jax_backend']}")
    print("devices:", ", ".join(result["software"]["jax_devices"]))
    print(f"x64={result['software']['x64_enabled']}")
    print()
    print("Configuration")
    print(f"  a={cfg['a']:.6g}, b={cfg['b']:.6g}, d={cfg['d']:.6g}, tau={100*cfg['tau']:.2f}%")
    print(f"  n_time={cfg['n_time']}, n_theta={cfg['n_theta']}, starts={cfg['n_starts']}")
    print()
    print("Validation against closed form")
    print(f"  |lambda - analytic|                  = {val['lambda_abs_error']:.3e}")
    print(f"  |d_lambda/dtheta implicit - exact|   = {val['lambda_theta_grad_abs_error']:.3e}")
    print(f"  |Lift pipeline - analytic|           = {val['lift_abs_error']:.3e}")
    print(f"  |Action pipeline - analytic|         = {val['action_abs_error']:.3e}")
    print(f"  max weighted-Poisson residual        = {val['poisson_residual_max_abs']:.3e}")
    print(f"  |Tangent action - Poisson action|    = {val['tangent_vs_poisson_action_abs_error']:.3e}")
    print(f"  Lift grad rel. error (AD vs exact)   = {val['lift_grad_rel_error_vs_analytic']:.3e}")
    print(f"  Action grad rel. error (AD vs exact) = {val['action_grad_rel_error_vs_analytic']:.3e}")
    print()
    print("Dense-oracle designs")
    for key, label in [
        ("lift", "Lift"),
        ("info_population", "Info-Population"),
        ("transport_conditioned", "Transport-conditioned"),
        ("action_only", "Action-only"),
    ]:
        s = oracle[key]
        print(
            f"  {label:23s} theta={s['theta_deg']:8.3f} deg | "
            f"Lift={s['lift_loss']:.8f} | Action={s['correction_action']:.8f} | "
            f"Info={s['information_score']:.8f}"
        )
    print()
    print(f"  Lift optimum L*                     = {oracle['lift_star']:.8f}")
    print(f"  Allowed Lift bound                  = {oracle['lift_bound']:.8f}")
    print(f"  TC Lift sacrifice                   = {oracle['lift_sacrifice_percent']:.3f}%")
    print(f"  TC transport headroom               = {oracle['transport_headroom_percent']:.3f}%")
    print(
        "  Expected fast/slow action ratio     = "
        f"{oracle['expected_endpoint_direction_action_ratio_fast_over_slow']:.6f}x"
    )
    print()
    print("Differentiable optimizer vs dense oracle")
    for key, label in [
        ("lift", "Lift"),
        ("info_population", "Info-Population"),
        ("transport_conditioned", "Transport-conditioned"),
    ]:
        s = opt[key]
        err = opt["theta_error_to_dense_oracle_rad"][key]
        print(
            f"  {label:23s} theta={s['theta_deg']:8.3f} deg | "
            f"oracle angle error={err:.3e} rad"
        )
    print(f"  TC optimizer feasible: {opt['transport_conditioned_feasible']}")
    print()
    print("Checks")
    for name, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()
    print("ALL CHECKS PASS:", result["all_checks_pass"])
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=float, default=1.0, help="Slow-mode amplitude.")
    p.add_argument("--b", type=float, default=1.02, help="Fast-mode amplitude.")
    p.add_argument("--d", type=float, default=1.0, help="Endpoint half-separation.")
    p.add_argument("--tau", type=float, default=0.05, help="Relative Lift tolerance, e.g. 0.05 = 5%%.")
    p.add_argument("--n-time", type=int, default=4097, help="Time quadrature points.")
    p.add_argument("--n-theta", type=int, default=8192, help="Dense oracle angle points on [0,pi).")
    p.add_argument("--n-starts", type=int, default=12, help="Deterministic optimizer restarts.")
    p.add_argument("--opt-steps", type=int, default=900, help="Adam steps per restart.")
    p.add_argument("--lr", type=float, default=0.03, help="Adam learning rate.")
    p.add_argument(
        "--penalty-weight",
        type=float,
        default=2.0e4,
        help="Dimensionless squared-hinge penalty for Lift-constraint violation.",
    )
    p.add_argument("--feasibility-eps", type=float, default=2e-6)
    p.add_argument("--probe-theta", type=float, default=0.37)
    p.add_argument("--probe-time", type=float, default=0.413)
    p.add_argument("--landscape-points-json", type=int, default=257)
    p.add_argument(
        "--output", type=Path, default=Path("stage_a_results.json"), help="JSON output path."
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.n_time < 3 or args.n_time % 2 == 0:
        print("Note: an odd --n-time >= 3 is recommended; continuing with trapezoidal quadrature.")
    if args.n_theta < 128:
        raise ValueError("--n-theta should be >= 128 for a useful dense oracle.")
    if not (0.0 <= args.tau):
        raise ValueError("--tau must be nonnegative.")

    result = run(args)
    print_report(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(f"Results written to: {args.output.resolve()}")

    # Nonzero exit status makes validation failures visible in scripts/CI.
    if not result["all_checks_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
