from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import logsumexp


class ExactFeasibilityError(RuntimeError):
    """Expected scientific infeasibility in authoritative 2-D hull checks.

    This is distinct from a programming error: callers should normally reject the
    candidate and continue the search rather than aborting the whole experiment.
    """

    def __init__(self, message: str, *, reason: str, violation: float = float("nan")):
        super().__init__(message)
        self.reason = str(reason)
        self.violation = float(violation)


@dataclass(frozen=True)
class ExactBetaPolytope:
    """Exact 2-D beta feasibility polytope for authoritative evaluation.

    This module is intentionally NumPy/SciPy-only.  It is never used inside an
    autodiff objective; it is the non-differentiated scientific acceptance boundary.
    """

    A: np.ndarray
    b: np.ndarray
    feasible_beta: np.ndarray
    physical_equations: np.ndarray
    particle_equations: tuple[np.ndarray, ...]
    endpoint_max_violation: float


@dataclass(frozen=True)
class ExactBetaProjection:
    beta: np.ndarray
    active: bool
    distance: float
    max_unconstrained_violation: float


@dataclass(frozen=True)
class ExactTiltState:
    lam: np.ndarray
    weights: np.ndarray
    moments: np.ndarray
    covariance: np.ndarray
    residual: np.ndarray
    residual_norm: float
    ess_fraction: float
    success: bool


def hull_equations_2d(points: np.ndarray) -> np.ndarray:
    """Return exact 2-D ConvexHull equations ``normal @ x + offset <= 0``."""
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 14), axis=0)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected [n,2] points, got {pts.shape}")
    if pts.shape[0] < 3 or np.linalg.matrix_rank(pts - np.mean(pts, axis=0)) < 2:
        raise ExactFeasibilityError(
            "Moment set is rank-deficient; cannot construct a 2-D hull.",
            reason="rank_deficient_hull",
        )
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise ExactFeasibilityError(
            f"Could not construct moment hull: {exc}",
            reason="qhull_failure",
        ) from exc
    return np.asarray(hull.equations, dtype=np.float64)


def build_common_quadratic_beta_polytope_2d(
    *,
    times: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    physical_features: np.ndarray,
    particle_features_by_time: np.ndarray,
    particle_mask_by_time: np.ndarray,
    margin: float = 0.0,
    endpoint_tol: float = 2.0e-8,
    physical_equations: np.ndarray | None = None,
    particle_equations: Sequence[np.ndarray] | None = None,
) -> ExactBetaPolytope:
    """Exact common physical/particle hull constraints for quadratic beta.

    For c(t)=(1-t)c0+t c1+t(1-t) beta, each hull facet yields one linear
    inequality in beta.  We use all exact ConvexHull facets, not a directional
    support approximation.
    """
    times = np.asarray(times, dtype=np.float64)
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    physical_features = np.asarray(physical_features, dtype=np.float64)
    particle_features_by_time = np.asarray(particle_features_by_time, dtype=np.float64)
    particle_mask_by_time = np.asarray(particle_mask_by_time, dtype=bool)

    physical_eq = (
        hull_equations_2d(physical_features)
        if physical_equations is None
        else np.asarray(physical_equations, dtype=np.float64)
    )
    particle_eqs: list[np.ndarray] = []
    A_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    endpoint_max = 0.0

    for k, t in enumerate(times):
        if particle_equations is None:
            pts = particle_features_by_time[k][particle_mask_by_time[k]]
            particle_eq = hull_equations_2d(pts)
        else:
            particle_eq = np.asarray(particle_equations[k], dtype=np.float64)
        particle_eqs.append(particle_eq)

        z = float(t * (1.0 - t))
        bridge = (1.0 - t) * c0 + t * c1
        for eq in (physical_eq, particle_eq):
            normals = eq[:, :2]
            offsets = eq[:, 2]
            if abs(z) < 1.0e-14:
                violation = normals @ bridge + offsets + float(margin)
                endpoint_max = max(endpoint_max, float(np.max(violation)))
            else:
                A_rows.append(z * normals)
                b_rows.append(-offsets - normals @ bridge - float(margin))

    if endpoint_max > endpoint_tol:
        raise ExactFeasibilityError(
            "Endpoint moment lies outside the exact common physical/particle hull: "
            f"violation={endpoint_max:.3e}.",
            reason="endpoint_outside_common_hull",
            violation=endpoint_max,
        )

    A = np.concatenate(A_rows, axis=0) if A_rows else np.zeros((0, 2), dtype=np.float64)
    b = np.concatenate(b_rows, axis=0) if b_rows else np.zeros((0,), dtype=np.float64)

    if A.shape[0]:
        feas = linprog(
            c=np.zeros(2, dtype=np.float64),
            A_ub=A,
            b_ub=b,
            bounds=[(None, None), (None, None)],
            method="highs",
        )
        if not feas.success:
            raise ExactFeasibilityError(
                "Exact common quadratic-beta feasibility intersection is empty.",
                reason="empty_common_beta_polytope",
            )
        feasible_beta = np.asarray(feas.x, dtype=np.float64)
    else:
        feasible_beta = np.zeros(2, dtype=np.float64)

    return ExactBetaPolytope(
        A=A,
        b=b,
        feasible_beta=feasible_beta,
        physical_equations=physical_eq,
        particle_equations=tuple(particle_eqs),
        endpoint_max_violation=max(0.0, endpoint_max),
    )


def project_metric_polytope_exact_2d(
    beta_unconstrained: np.ndarray,
    information: np.ndarray,
    polytope: ExactBetaPolytope,
    *,
    tol: float = 1.0e-10,
) -> ExactBetaProjection:
    """Exact final 2-D metric projection using SLSQP on all hull facets."""
    beta_u = np.asarray(beta_unconstrained, dtype=np.float64)
    H = np.asarray(information, dtype=np.float64)
    A = polytope.A
    b = polytope.b
    if A.shape[0] == 0:
        return ExactBetaProjection(beta_u.copy(), False, 0.0, 0.0)

    violation = float(max(0.0, np.max(A @ beta_u - b)))
    if violation <= tol:
        return ExactBetaProjection(beta_u.copy(), False, 0.0, violation)

    H = 0.5 * (H + H.T)
    scale = max(float(np.trace(H)) / 2.0, 1.0)
    H = H + 1.0e-14 * scale * np.eye(2)

    def fun(x):
        d = x - beta_u
        return 0.5 * float(d @ H @ d)

    def jac(x):
        return H @ (x - beta_u)

    constraints = {
        "type": "ineq",
        "fun": lambda x: b - A @ x,
        "jac": lambda x: -A,
    }
    res = minimize(
        fun,
        x0=polytope.feasible_beta,
        jac=jac,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1.0e-13, "maxiter": 500, "disp": False},
    )
    if not res.success:
        raise ExactFeasibilityError(
            f"Exact beta projection failed: {res.message}",
            reason="beta_projection_failure",
            violation=violation,
        )
    beta = np.asarray(res.x, dtype=np.float64)
    post_violation = float(np.max(A @ beta - b))
    if post_violation > max(5.0e-9, 10.0 * tol):
        raise ExactFeasibilityError(
            f"Exact beta projection returned infeasible point: violation={post_violation:.3e}",
            reason="beta_projection_infeasible",
            violation=post_violation,
        )
    d = beta - beta_u
    distance = float(np.sqrt(max(float(d @ H @ d), 0.0)))
    return ExactBetaProjection(beta, True, distance, violation)


def max_hull_violation(equations: np.ndarray, target: np.ndarray) -> float:
    eq = np.asarray(equations, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.max(eq[:, :2] @ target + eq[:, 2]))


def _tilt_state(phi: np.ndarray, base_weights: np.ndarray, target: np.ndarray, lam: np.ndarray):
    base = np.asarray(base_weights, dtype=np.float64)
    base = base / np.sum(base)
    phi = np.asarray(phi, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    log_base = np.full_like(base, -np.inf, dtype=np.float64)
    positive = base > 0.0
    log_base[positive] = np.log(base[positive])
    logits = log_base + phi @ lam
    logz = logsumexp(logits)
    w = np.exp(logits - logz)
    moment = np.sum(w[:, None] * phi, axis=0)
    centered = phi - moment[None, :]
    covariance = centered.T @ (w[:, None] * centered)
    residual = moment - target
    dual = float(logz - lam @ target)
    return w, moment, covariance, residual, dual


def robust_empirical_tilt_exact(
    phi: np.ndarray,
    base_weights: np.ndarray,
    target: np.ndarray,
    *,
    lam0: np.ndarray | None = None,
    newton_steps: int = 300,
    newton_ridge: float = 1.0e-7,
    step_cap: float = 20.0,
    lambda_clip: float = 1000.0,
    accept_tol: float = 2.0e-6,
    lbfgs_maxiter: int = 800,
    retry_multiplier: float = 2.0,
    retries: int = 2,
) -> ExactTiltState:
    """Robust non-differentiated hard empirical I-projection for final scoring.

    Newton is attempted first with warm starts.  If it does not reach the declared
    acceptance tolerance, exact-gradient L-BFGS-B minimizes the convex dual.  This
    mirrors the robust final-evaluation logic of the learned-reference benchmark.
    """
    phi = np.asarray(phi, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    base = base / np.sum(base)
    target = np.asarray(target, dtype=np.float64)
    m = phi.shape[1]
    start = np.zeros(m, dtype=np.float64) if lam0 is None else np.asarray(lam0, dtype=np.float64)

    best = None
    best_norm = np.inf
    for retry in range(int(retries) + 1):
        clip = float(lambda_clip) * (float(retry_multiplier) ** retry)
        lam = np.clip(start, -clip, clip).copy()

        for _ in range(int(newton_steps)):
            w, moment, C, F, dual = _tilt_state(phi, base, target, lam)
            norm = float(np.linalg.norm(F))
            if norm < best_norm:
                best = (lam.copy(), w.copy(), moment.copy(), C.copy(), F.copy())
                best_norm = norm
            if norm <= accept_tol:
                break
            H = C + float(newton_ridge) * np.eye(m)
            try:
                delta = np.linalg.solve(H, F)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, F, rcond=None)[0]
            dn = float(np.linalg.norm(delta))
            if dn > step_cap:
                delta *= float(step_cap) / max(dn, 1.0e-300)

            # Convex-dual backtracking.
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                trial = np.clip(lam - scale * delta, -clip, clip)
                *_, trial_dual = _tilt_state(phi, base, target, trial)
                if trial_dual <= dual + 1.0e-14:
                    lam = trial
                    accepted = True
                    break
            if not accepted:
                lam = np.clip(lam - 0.01 * delta, -clip, clip)

        w, moment, C, F, _ = _tilt_state(phi, base, target, lam)
        norm = float(np.linalg.norm(F))
        if norm <= accept_tol:
            best = (lam.copy(), w.copy(), moment.copy(), C.copy(), F.copy())
            best_norm = norm
            break

        log_base = np.full_like(base, -np.inf, dtype=np.float64)
        positive = base > 0.0
        log_base[positive] = np.log(base[positive])

        def dual_and_grad(x):
            logits = log_base + phi @ x
            logz = logsumexp(logits)
            ww = np.exp(logits - logz)
            moment_x = np.sum(ww[:, None] * phi, axis=0)
            return float(logz - x @ target), moment_x - target

        res = minimize(
            lambda x: dual_and_grad(x)[0],
            x0=lam,
            jac=lambda x: dual_and_grad(x)[1],
            method="L-BFGS-B",
            bounds=[(-clip, clip)] * m,
            options={"maxiter": int(lbfgs_maxiter), "ftol": 1.0e-15, "gtol": 1.0e-12},
        )
        lam = np.asarray(res.x, dtype=np.float64)
        w, moment, C, F, _ = _tilt_state(phi, base, target, lam)
        norm = float(np.linalg.norm(F))
        if norm < best_norm:
            best = (lam.copy(), w.copy(), moment.copy(), C.copy(), F.copy())
            best_norm = norm
        if norm <= accept_tol:
            break
        start = lam

    if best is None:
        raise RuntimeError("Exact empirical I-projection failed without producing a state.")
    lam, w, moment, C, F = best
    ess_proj = 1.0 / max(float(np.sum(w * w)), 1.0e-300)
    ess_base = 1.0 / max(float(np.sum(base * base)), 1.0e-300)
    ess_fraction = ess_proj / ess_base
    return ExactTiltState(
        lam=lam,
        weights=w,
        moments=moment,
        covariance=C,
        residual=F,
        residual_norm=best_norm,
        ess_fraction=float(ess_fraction),
        success=bool(best_norm <= accept_tol),
    )
