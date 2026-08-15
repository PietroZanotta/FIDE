from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array
BasisFn = Callable[[Array], Array]


@dataclass(frozen=True)
class AnchoredGLSConfig:
    """Numerics for endpoint-anchored linear-basis GLS reconstruction."""

    ridge_rel: float = 1.0e-12
    variance_floor: float = 1.0e-10


class AnchoredGLSResult(NamedTuple):
    coefficients: Array          # [n_basis, n_observables]
    coefficient_covariance: Array
    c: Array
    c_dot: Array
    information: Array
    score: Array


@dataclass(frozen=True)
class QuadraticBridgeConfig(AnchoredGLSConfig):
    pass


class QuadraticBridgeResult(NamedTuple):
    beta: Array
    beta_covariance: Array
    c: Array
    c_dot: Array
    information: Array


def quadratic_basis(times: Array) -> Array:
    times = jnp.asarray(times, dtype=jnp.float64)
    return (times * (1.0 - times))[:, None]


def quadratic_basis_derivative(times: Array) -> Array:
    times = jnp.asarray(times, dtype=jnp.float64)
    return (1.0 - 2.0 * times)[:, None]


def anchored_linear_part(c0: Array, c1: Array, times: Array) -> tuple[Array, Array]:
    c0 = jnp.asarray(c0, dtype=jnp.float64)
    c1 = jnp.asarray(c1, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    c = (1.0 - times[:, None]) * c0[None, :] + times[:, None] * c1[None, :]
    c_dot = jnp.broadcast_to((c1 - c0)[None, :], c.shape)
    return c, c_dot


def evaluate_anchored_basis_curve(
    coefficients: Array,
    c0: Array,
    c1: Array,
    times: Array,
    basis: BasisFn,
    basis_derivative: BasisFn,
) -> tuple[Array, Array]:
    """Evaluate an anchored curve whose basis vanishes at both endpoints."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    linear, linear_dot = anchored_linear_part(c0, c1, times)
    B = jnp.asarray(basis(times), dtype=jnp.float64)
    Bd = jnp.asarray(basis_derivative(times), dtype=jnp.float64)
    return linear + B @ coefficients, linear_dot + Bd @ coefficients


def fit_anchored_basis_gls(
    t_obs: Array,
    y_obs: Array,
    covariance_obs: Array,
    c0: Array,
    c1: Array,
    t_eval: Array,
    *,
    basis: BasisFn,
    basis_derivative: BasisFn,
    penalty: Array | None = None,
    cfg: AnchoredGLSConfig = AnchoredGLSConfig(),
) -> AnchoredGLSResult:
    """Differentiable full-covariance GLS for an endpoint-anchored basis.

    The model is

        c(t) = (1-t)c0 + t c1 + B(t) @ coefficients,

    where every column of B vanishes at t=0 and t=1.  ``penalty`` is an optional
    positive-semidefinite roughness matrix on the basis coefficients. This is the
    extension point for anchored cubic smoothing splines: supply an endpoint-
    vanishing spline basis and its integrated squared-curvature penalty.
    """
    t_obs = jnp.asarray(t_obs, dtype=jnp.float64)
    y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
    covariance_obs = jnp.asarray(covariance_obs, dtype=jnp.float64)
    c0 = jnp.asarray(c0, dtype=jnp.float64)
    c1 = jnp.asarray(c1, dtype=jnp.float64)
    t_eval = jnp.asarray(t_eval, dtype=jnp.float64)

    B = jnp.asarray(basis(t_obs), dtype=jnp.float64)  # [n,p]
    p = B.shape[1]
    m = c0.shape[0]
    eye_m = jnp.eye(m, dtype=jnp.float64)

    linear_obs, _ = anchored_linear_part(c0, c1, t_obs)
    residual = y_obs - linear_obs

    covariance_reg = 0.5 * (covariance_obs + jnp.swapaxes(covariance_obs, -1, -2))
    covariance_reg = covariance_reg + cfg.variance_floor * eye_m[None, :, :]
    precision = jax.vmap(lambda v: jnp.linalg.solve(v, eye_m))(covariance_reg)

    # Vectorized GLS normal equations for vec(coefficients), ordered [basis, observable].
    # H[(a,i),(b,j)] = sum_n B[n,a] B[n,b] Precision[n,i,j].
    H4 = jnp.einsum("na,nb,nij->aibj", B, B, precision)
    information = H4.reshape((p * m, p * m))
    score = jnp.einsum("na,nij,nj->ai", B, precision, residual).reshape(-1)

    scale = jnp.maximum(jnp.trace(information) / jnp.maximum(float(p * m), 1.0), 1.0)
    Hreg = information + cfg.ridge_rel * scale * jnp.eye(p * m, dtype=jnp.float64)
    if penalty is not None:
        P = jnp.asarray(penalty, dtype=jnp.float64)
        if P.shape != (p, p):
            raise ValueError(f"penalty must have shape {(p, p)}, got {P.shape}")
        Hreg = Hreg + jnp.kron(P, eye_m)

    coef_flat = jnp.linalg.solve(Hreg, score)
    coefficients = coef_flat.reshape((p, m))

    Hreg_inv = jnp.linalg.solve(Hreg, jnp.eye(p * m, dtype=jnp.float64))
    coef_cov = Hreg_inv @ information @ Hreg_inv.T
    coef_cov = 0.5 * (coef_cov + coef_cov.T)

    c, c_dot = evaluate_anchored_basis_curve(
        coefficients, c0, c1, t_eval, basis, basis_derivative
    )
    return AnchoredGLSResult(
        coefficients=coefficients,
        coefficient_covariance=coef_cov,
        c=c,
        c_dot=c_dot,
        information=information,
        score=score,
    )


def evaluate_quadratic_bridge(beta: Array, c0: Array, c1: Array, times: Array) -> tuple[Array, Array]:
    beta = jnp.asarray(beta, dtype=jnp.float64)
    return evaluate_anchored_basis_curve(
        beta[None, :], c0, c1, times, quadratic_basis, quadratic_basis_derivative
    )


def fit_quadratic_bridge_gls(
    t_obs: Array,
    y_obs: Array,
    covariance_obs: Array,
    c0: Array,
    c1: Array,
    t_eval: Array,
    cfg: QuadraticBridgeConfig = QuadraticBridgeConfig(),
) -> QuadraticBridgeResult:
    fit = fit_anchored_basis_gls(
        t_obs,
        y_obs,
        covariance_obs,
        c0,
        c1,
        t_eval,
        basis=quadratic_basis,
        basis_derivative=quadratic_basis_derivative,
        cfg=cfg,
    )
    m = jnp.asarray(c0).shape[0]
    return QuadraticBridgeResult(
        beta=fit.coefficients[0],
        beta_covariance=fit.coefficient_covariance.reshape((m, m)),
        c=fit.c,
        c_dot=fit.c_dot,
        information=fit.information.reshape((m, m)),
    )


def bridge_halfspace_constraints(
    c0: Array,
    c1: Array,
    times: Array,
    hull_equations: Array,
    *,
    margin: float = 0.0,
) -> tuple[Array, Array]:
    """Convert moment-hull facets into linear constraints ``A beta <= b``."""
    c0 = jnp.asarray(c0, dtype=jnp.float64)
    c1 = jnp.asarray(c1, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)
    equations = jnp.asarray(hull_equations, dtype=jnp.float64)

    normals = equations[:, :-1]
    offsets = equations[:, -1]
    z = times * (1.0 - times)
    bridge = (1.0 - times[:, None]) * c0[None, :] + times[:, None] * c1[None, :]

    A = z[:, None, None] * normals[None, :, :]
    b = -offsets[None, :] - jnp.einsum("fm,tm->tf", normals, bridge) - margin
    return A.reshape((-1, c0.shape[0])), b.reshape((-1,))


def max_constraint_violation(beta: Array, A: Array, b: Array) -> Array:
    beta = jnp.asarray(beta, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return jnp.maximum(0.0, jnp.max(A @ beta - b, initial=-jnp.inf))


class AnchoredBasisGLSReconstructor:
    def __init__(
        self,
        basis: BasisFn,
        basis_derivative: BasisFn,
        *,
        penalty: Array | None = None,
        cfg: AnchoredGLSConfig = AnchoredGLSConfig(),
    ):
        self.basis = basis
        self.basis_derivative = basis_derivative
        self.penalty = penalty
        self.cfg = cfg

    def reconstruct(self, t_obs, y_obs, covariance_obs, c0, c1, t_eval):
        return fit_anchored_basis_gls(
            t_obs,
            y_obs,
            covariance_obs,
            c0,
            c1,
            t_eval,
            basis=self.basis,
            basis_derivative=self.basis_derivative,
            penalty=self.penalty,
            cfg=self.cfg,
        )


class QuadraticBridgeReconstructor:
    def __init__(self, cfg: QuadraticBridgeConfig = QuadraticBridgeConfig()):
        self.cfg = cfg

    def reconstruct(self, t_obs, y_obs, covariance_obs, c0, c1, t_eval) -> QuadraticBridgeResult:
        return fit_quadratic_bridge_gls(
            t_obs, y_obs, covariance_obs, c0, c1, t_eval, self.cfg
        )
