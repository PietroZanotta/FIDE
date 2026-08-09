"""Faithful JAX implementation of Moment Guided Diffusion (MGD) for Example A.

This module implements the actual interacting-particle MGD dynamics from
Lempereur et al. (2026), rather than substituting the analytic Gaussian law.

Two numerical implementations are provided for cross-validation:

1. ``simulate_mgd_predictor_corrector`` follows Sec. 3.2, Eqs. (18)-(21):
   predictor with moment-transport drift + Brownian noise, then the linearized
   moment corrector.  The corrector update here follows Eq. (19),

       X_{t+h} = Y - h sigma^2 theta^T grad phi(Y),

   together with Eq. (21).  This sign/scale is also the one consistent with
   the continuous SDE in Theorem 3.1.

2. ``simulate_mgd_theorem_euler`` directly discretizes Theorem 3.1 using
   Euler-Maruyama and the explicit Laplacian term.  It is not the primary
   benchmark implementation; it is an independent numerical cross-check.

For Example A, phi(x)=(x,x^2), base N(0,1), and target Q_+ also has mean zero
and variance one.  With the variance-preserving MGD interpolant, m_t=(0,1) for
all t.  Appendix E of the MGD paper then gives an analytic oracle: the MGD law
is N(0,1) at every time for every sigma >= 0.  We use that fact only to test
this implementation, not to generate the benchmark trajectory.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class MGDResult(NamedTuple):
    trajectory: Array
    eta: Array
    theta: Array
    gram_condition: Array
    predictor_moment_error: Array
    corrected_moment_error: Array


def phi_1d(x: Array) -> Array:
    """phi=(x,x^2), shape (...,2)."""
    return jnp.stack([x, x * x], axis=-1)


def grad_phi_1d(x: Array) -> Array:
    """d phi / dx = (1,2x), shape (...,2)."""
    return jnp.stack([jnp.ones_like(x), 2.0 * x], axis=-1)


def laplacian_phi_1d(x: Array) -> Array:
    """Delta phi=(0,2) in one dimension, broadcast over particles."""
    return jnp.stack([jnp.zeros_like(x), 2.0 * jnp.ones_like(x)], axis=-1)


def empirical_moments(x: Array) -> Array:
    return jnp.mean(phi_1d(x), axis=0)


def gram_matrix(x: Array) -> Array:
    grad = grad_phi_1d(x)
    return (grad.T @ grad) / x.shape[0]


def _normalized_gram_solve(G: Array, rhs: Array, ridge: float = 1e-7) -> Array:
    """Solve G c = rhs using the paper's diagonal normalization + ridge.

    Appendix D.1 normalizes potentials so the Gram diagonal is one, then adds
    delta I before inversion.  Algebraically, if D=sqrt(diag G), we solve

        (D^-1 G D^-1 + delta I) c_norm = D^-1 rhs,
        c = D^-1 c_norm.

    This is preferable to adding a ridge in the raw, scale-dependent basis.
    """
    diag = jnp.maximum(jnp.diag(G), 1e-30)
    scale = jnp.sqrt(diag)
    Gn = G / (scale[:, None] * scale[None, :])
    rhsn = rhs / scale
    cn = jnp.linalg.solve(Gn + ridge * jnp.eye(G.shape[0], dtype=G.dtype), rhsn)
    return cn / scale


def _gram_condition_normalized(G: Array) -> Array:
    diag = jnp.maximum(jnp.diag(G), 1e-30)
    scale = jnp.sqrt(diag)
    Gn = G / (scale[:, None] * scale[None, :])
    s = jnp.linalg.svd(Gn, compute_uv=False)
    return s[0] / jnp.maximum(s[-1], 1e-30)


def example_a_moment_path(n_steps: int, dtype=jnp.float64) -> tuple[Array, Array]:
    """Population MGD moment path for the VP interpolant in Example A.

    MGD uses I_t=cos(alpha_t) Z + sin(alpha_t) X with independent endpoints.
    Both Z~N(0,1) and Q_+ have mean 0 and second moment 1, hence for
    phi=(x,x^2): m_t=(0,1) and dm_t/dt=(0,0) for every schedule alpha_t.
    """
    moments = jnp.broadcast_to(jnp.array([0.0, 1.0], dtype=dtype), (n_steps + 1, 2))
    moment_dot = jnp.zeros((n_steps, 2), dtype=dtype)
    return moments, moment_dot


def simulate_mgd_predictor_corrector(
    x0: Array,
    key: Array,
    sigma: float | Array,
    moments: Array,
    moment_dot: Array,
    *,
    ridge: float = 1e-7,
) -> MGDResult:
    """Interacting-particle MGD predictor/corrector (paper Sec. 3.2).

    Parameters
    ----------
    x0:
        Initial replicas, shape (n_rep,).  They should approximate the MGD
        Gaussian base and ideally match m_0 accurately.
    key:
        JAX PRNG key.
    sigma:
        MGD volatility.  Must be positive for the predictor/corrector form.
    moments:
        Target m_t on all n_steps+1 time nodes, shape (n_steps+1,r).
    moment_dot:
        dm_t/dt on the left nodes, shape (n_steps,r).
    ridge:
        Normalized Gram ridge; default 1e-7 as reported in MGD Appendix D.1.

    Returns the complete particle trajectory plus diagnostics.  The primary
    benchmark uses this function.
    """
    sigma = jnp.asarray(sigma, dtype=x0.dtype)
    n_steps = moment_dot.shape[0]
    h = jnp.asarray(1.0 / n_steps, dtype=x0.dtype)

    def step(carry, inputs):
        x, rng = carry
        m_next, mdot = inputs

        # Predictor: Eqs. (14),(18), with empirical Gram matrix.
        G = gram_matrix(x)
        eta = _normalized_gram_solve(G, mdot, ridge)
        grad = grad_phi_1d(x)
        predictor_drift = grad @ eta
        rng, sub = jax.random.split(rng)
        xi = jax.random.normal(sub, shape=x.shape, dtype=x.dtype)
        y = x + h * predictor_drift + jnp.sqrt(2.0 * h) * sigma * xi

        # Corrector: Eqs. (19)-(21).  Equation (21) is solved for theta,
        # then Eq. (19) applies -h*sigma^2 theta^T grad phi(Y).
        Gp = gram_matrix(y)
        pred_error = empirical_moments(y) - m_next
        theta_rhs = pred_error / (h * sigma * sigma)
        theta = _normalized_gram_solve(Gp, theta_rhs, ridge)
        grad_y = grad_phi_1d(y)
        x_next = y - h * sigma * sigma * (grad_y @ theta)

        corr_error = empirical_moments(x_next) - m_next
        diagnostics = (
            x_next,
            eta,
            theta,
            _gram_condition_normalized(Gp),
            jnp.linalg.norm(pred_error),
            jnp.linalg.norm(corr_error),
        )
        return (x_next, rng), diagnostics

    (_, _), outs = jax.lax.scan(step, (x0, key), (moments[1:], moment_dot))
    tail, eta, theta, cond, pred_err, corr_err = outs
    traj = jnp.concatenate([x0[None, :], tail], axis=0)
    return MGDResult(
        trajectory=traj,
        eta=eta,
        theta=theta,
        gram_condition=cond,
        predictor_moment_error=pred_err,
        corrected_moment_error=corr_err,
    )


def simulate_mgd_theorem_euler(
    x0: Array,
    key: Array,
    sigma: float | Array,
    moments: Array,
    moment_dot: Array,
    *,
    ridge: float = 1e-7,
) -> MGDResult:
    """Direct Euler-Maruyama discretization of MGD Theorem 3.1.

    This computes theta from G theta = E[Delta phi(X_t)] exactly as in Eq. (15).
    It is included to cross-check the predictor/corrector implementation on
    smooth observables.  It is *not* used as the headline MGD benchmark.
    """
    del moments  # The continuous theorem needs m_dot; moments are for API symmetry.
    sigma = jnp.asarray(sigma, dtype=x0.dtype)
    n_steps = moment_dot.shape[0]
    h = jnp.asarray(1.0 / n_steps, dtype=x0.dtype)

    def step(carry, mdot):
        x, rng = carry
        G = gram_matrix(x)
        eta = _normalized_gram_solve(G, mdot, ridge)
        lap_mean = jnp.mean(laplacian_phi_1d(x), axis=0)
        theta = _normalized_gram_solve(G, lap_mean, ridge)
        grad = grad_phi_1d(x)
        drift = grad @ (eta - sigma * sigma * theta)
        rng, sub = jax.random.split(rng)
        xi = jax.random.normal(sub, shape=x.shape, dtype=x.dtype)
        x_next = x + h * drift + jnp.sqrt(2.0 * h) * sigma * xi
        err = empirical_moments(x_next) - jnp.array([0.0, 1.0], dtype=x.dtype)
        diagnostics = (
            x_next,
            eta,
            theta,
            _gram_condition_normalized(G),
            jnp.asarray(0.0, dtype=x.dtype),
            jnp.linalg.norm(err),
        )
        return (x_next, rng), diagnostics

    (_, _), outs = jax.lax.scan(step, (x0, key), moment_dot)
    tail, eta, theta, cond, pred_err, corr_err = outs
    traj = jnp.concatenate([x0[None, :], tail], axis=0)
    return MGDResult(
        trajectory=traj,
        eta=eta,
        theta=theta,
        gram_condition=cond,
        predictor_moment_error=pred_err,
        corrected_moment_error=corr_err,
    )
