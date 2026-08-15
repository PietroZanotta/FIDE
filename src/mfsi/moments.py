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


# -----------------------------------------------------------------------------
# Endpoint-anchored cubic smoothing spline regression (ordinary penalized LS)
# -----------------------------------------------------------------------------

import numpy as np


@dataclass(frozen=True)
class AnchoredCubicSplineConfig:
    """Fixed smoothing model for endpoint-anchored cubic B-spline regression.

    This is deliberately separate from ``AnchoredGLSConfig``: the default vortex
    observation model uses ordinary penalized least squares rather than carrying a
    full observation covariance through the regression.  ``smoothing`` and knot
    count are intended to be selected on pilot/design-only data and then frozen.
    """

    internal_knots: int = 3
    smoothing: float = 1.0e-4
    ridge_rel: float = 1.0e-10
    roughness_quadrature_order: int = 8

    def __post_init__(self) -> None:
        if int(self.internal_knots) < 0:
            raise ValueError("internal_knots must be >= 0")
        if float(self.smoothing) < 0.0:
            raise ValueError("smoothing must be nonnegative")
        if float(self.ridge_rel) < 0.0:
            raise ValueError("ridge_rel must be nonnegative")
        if int(self.roughness_quadrature_order) < 2:
            raise ValueError("roughness_quadrature_order must be >= 2")


class AnchoredCubicSplineResult(NamedTuple):
    coefficients: Array
    c: Array
    c_dot: Array
    normal_matrix: Array
    residual_sum_squares: Array
    roughness: Array


def _open_uniform_cubic_knots(internal_knots: int) -> np.ndarray:
    degree = 3
    if int(internal_knots) > 0:
        interior = np.linspace(0.0, 1.0, int(internal_knots) + 2, dtype=np.float64)[1:-1]
    else:
        interior = np.empty((0,), dtype=np.float64)
    return np.concatenate([
        np.zeros(degree + 1, dtype=np.float64),
        interior,
        np.ones(degree + 1, dtype=np.float64),
    ])


def _bspline_basis_numpy(times: np.ndarray, knots: np.ndarray, derivative: int = 0) -> np.ndarray:
    # Lazy import preserves the historical ability to import mfsi.moments without
    # requiring SciPy unless the new spline API is actually instantiated.
    from scipy.interpolate import BSpline

    times = np.asarray(times, dtype=np.float64)
    degree = 3
    n_full = len(knots) - degree - 1
    eye = np.eye(n_full, dtype=np.float64)
    cols = []
    for j in range(n_full):
        spline = BSpline(knots, eye[j], degree, extrapolate=False)
        if derivative:
            spline = spline.derivative(int(derivative))
        values = np.asarray(spline(times), dtype=np.float64)
        cols.append(values)
    full = np.stack(cols, axis=-1)
    # For a clamped/open cubic B-spline basis, only the first basis is nonzero at
    # t=0 and only the last at t=1. Dropping them produces a minimal basis whose
    # columns vanish exactly at both endpoints.
    return full[..., 1:-1]


def _cubic_roughness_numpy(knots: np.ndarray, order: int) -> np.ndarray:
    spans = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    n_basis = _bspline_basis_numpy(np.asarray([0.5]), knots, derivative=0).shape[-1]
    omega = np.zeros((n_basis, n_basis), dtype=np.float64)
    for a, b in zip(spans[:-1], spans[1:]):
        if b <= a:
            continue
        tq = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        wq = 0.5 * (b - a) * weights
        b2 = _bspline_basis_numpy(tq, knots, derivative=2)
        omega += np.einsum("q,qa,qb->ab", wq, b2, b2)
    return 0.5 * (omega + omega.T)


class AnchoredCubicSplineReconstructor:
    """JAX-differentiable anchored cubic regression with precomputed spline geometry.

    Times and the roughness matrix are constructed once on the host.  Reconstruction
    itself is a small JAX linear solve, so gradients propagate through observations,
    endpoint moments, sensor positions, and all downstream MFSI quantities.
    """

    def __init__(
        self,
        t_obs,
        t_eval,
        cfg: AnchoredCubicSplineConfig = AnchoredCubicSplineConfig(),
    ):
        t_obs_np = np.asarray(t_obs, dtype=np.float64)
        t_eval_np = np.asarray(t_eval, dtype=np.float64)
        if t_obs_np.ndim != 1 or t_eval_np.ndim != 1:
            raise ValueError("t_obs and t_eval must be one-dimensional")
        if len(t_obs_np) < 2:
            raise ValueError("at least two observation times are required")
        if np.any(np.diff(t_obs_np) <= 0.0) or np.any(np.diff(t_eval_np) < 0.0):
            raise ValueError("times must be increasing")
        if t_obs_np[0] < -1.0e-14 or t_obs_np[-1] > 1.0 + 1.0e-14:
            raise ValueError("anchored spline times must lie in [0, 1]")
        if t_eval_np[0] < -1.0e-14 or t_eval_np[-1] > 1.0 + 1.0e-14:
            raise ValueError("anchored spline evaluation times must lie in [0, 1]")

        knots = _open_uniform_cubic_knots(cfg.internal_knots)
        B_obs = _bspline_basis_numpy(t_obs_np, knots, derivative=0)
        B_eval = _bspline_basis_numpy(t_eval_np, knots, derivative=0)
        Bd_eval = _bspline_basis_numpy(t_eval_np, knots, derivative=1)
        omega = _cubic_roughness_numpy(knots, cfg.roughness_quadrature_order)

        self.cfg = cfg
        self.knots = knots
        self.t_obs = jnp.asarray(t_obs_np, dtype=jnp.float64)
        self.t_eval = jnp.asarray(t_eval_np, dtype=jnp.float64)
        self.B_obs = jnp.asarray(B_obs, dtype=jnp.float64)
        self.B_eval = jnp.asarray(B_eval, dtype=jnp.float64)
        self.Bd_eval = jnp.asarray(Bd_eval, dtype=jnp.float64)
        self.roughness_matrix = jnp.asarray(omega, dtype=jnp.float64)

    @property
    def n_basis(self) -> int:
        return int(self.B_obs.shape[1])

    def reconstruct(self, y_obs, c0, c1) -> AnchoredCubicSplineResult:
        y_obs = jnp.asarray(y_obs, dtype=jnp.float64)
        c0 = jnp.asarray(c0, dtype=jnp.float64)
        c1 = jnp.asarray(c1, dtype=jnp.float64)
        if y_obs.ndim != 2:
            raise ValueError("y_obs must have shape [n_observations, n_observables]")
        if y_obs.shape[0] != self.B_obs.shape[0]:
            raise ValueError("y_obs time dimension does not match constructor t_obs")
        if c0.shape != c1.shape or y_obs.shape[1] != c0.shape[0]:
            raise ValueError("endpoint and observation observable dimensions do not match")

        linear_obs, _ = anchored_linear_part(c0, c1, self.t_obs)
        residual = y_obs - linear_obs

        gram = self.B_obs.T @ self.B_obs
        scale = jnp.maximum(jnp.trace(gram) / jnp.maximum(float(self.n_basis), 1.0), 1.0)
        normal = (
            gram
            + float(self.cfg.smoothing) * self.roughness_matrix
            + float(self.cfg.ridge_rel) * scale * jnp.eye(self.n_basis, dtype=jnp.float64)
        )
        coefficients = jnp.linalg.solve(normal, self.B_obs.T @ residual)

        linear_eval, linear_dot = anchored_linear_part(c0, c1, self.t_eval)
        c = linear_eval + self.B_eval @ coefficients
        c_dot = linear_dot + self.Bd_eval @ coefficients
        fitted_obs = linear_obs + self.B_obs @ coefficients
        rss = jnp.sum((y_obs - fitted_obs) ** 2)
        roughness = jnp.einsum(
            "am,ab,bm->", coefficients, self.roughness_matrix, coefficients
        )
        return AnchoredCubicSplineResult(
            coefficients=coefficients,
            c=c,
            c_dot=c_dot,
            normal_matrix=normal,
            residual_sum_squares=rss,
            roughness=roughness,
        )


def fit_anchored_cubic_spline(
    t_obs,
    y_obs,
    c0,
    c1,
    t_eval,
    cfg: AnchoredCubicSplineConfig = AnchoredCubicSplineConfig(),
) -> AnchoredCubicSplineResult:
    """Convenience non-cached entry point; prefer the reconstructor inside optimizers."""
    return AnchoredCubicSplineReconstructor(t_obs, t_eval, cfg).reconstruct(y_obs, c0, c1)
