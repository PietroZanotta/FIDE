"""Exact endpoint-hull feasibility and instrumented empirical I-projection."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp

import mfsi_components as core


def _validate_pair(minus: np.ndarray, plus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minus = np.asarray(minus, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.float64)
    if minus.ndim != 2 or plus.ndim != 2 or minus.shape[1] != plus.shape[1]:
        raise ValueError("endpoint features must have matching [bank, R] shapes")
    if not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise ValueError("endpoint features must be finite")
    return minus, plus


def common_hull_equalities(minus: np.ndarray, plus: np.ndarray, *, slack_column: bool = False):
    minus, plus = _validate_pair(minus, plus)
    n_minus, dimension = minus.shape
    n_plus = plus.shape[0]
    columns = n_minus + n_plus + int(slack_column)
    matrix = np.zeros((dimension + 2, columns), dtype=np.float64)
    matrix[0, :n_minus] = 1.0
    matrix[1, n_minus:n_minus + n_plus] = 1.0
    matrix[2:, :n_minus] = minus.T
    matrix[2:, n_minus:n_minus + n_plus] = -plus.T
    rhs = np.concatenate(([1.0, 1.0], np.zeros(dimension, dtype=np.float64)))
    return matrix, rhs


def _weight_diagnostics(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights[weights > 0.0]
    return {
        "ess_fraction": float(1.0 / (len(weights) * np.sum(weights * weights))),
        "maximum_weight": float(weights.max()),
        "minimum_weight": float(weights.min()),
        "positive_weight_count": int(len(positive)),
        "entropy_fraction": float(
            -np.sum(positive * np.log(positive)) / np.log(len(weights))
        ),
    }


def solve_common_hull_lp(minus: np.ndarray, plus: np.ndarray) -> dict:
    """Find any exact common empirical convex-hull point using HiGHS."""
    minus, plus = _validate_pair(minus, plus)
    n_minus, n_plus = len(minus), len(plus)
    equality, rhs = common_hull_equalities(minus, plus)
    result = linprog(
        np.zeros(n_minus + n_plus), A_eq=equality, b_eq=rhs,
        bounds=(0.0, None), method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    output = {
        "solver": "scipy.optimize.linprog(method='highs')",
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
    }
    if not result.success:
        output.update({"maximum_equality_residual": None, "target": None})
        return output
    a, b = result.x[:n_minus], result.x[n_minus:]
    minus_target, plus_target = a @ minus, b @ plus
    output.update({
        "maximum_equality_residual": float(np.max(np.abs(equality @ result.x - rhs))),
        "endpoint_target_disagreement": float(np.max(np.abs(minus_target - plus_target))),
        "target": 0.5 * (minus_target + plus_target),
        "minus_weights": a, "plus_weights": b,
        "minus_weight_diagnostics": _weight_diagnostics(a),
        "plus_weight_diagnostics": _weight_diagnostics(b),
    })
    return output


def solve_target_hull_lp(features: np.ndarray, target: np.ndarray) -> dict:
    """Test whether a fixed target belongs to one empirical feature hull."""
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    equality = np.vstack([np.ones((1, len(features))), features.T])
    rhs = np.concatenate(([1.0], target))
    result = linprog(
        np.zeros(len(features)), A_eq=equality, b_eq=rhs, bounds=(0.0, None),
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    output = {
        "solver": "scipy.optimize.linprog fixed-target hull membership (HiGHS)",
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
    }
    if result.success:
        output.update({
            "maximum_equality_residual": float(np.max(np.abs(equality @ result.x - rhs))),
            "weights": result.x, "weight_diagnostics": _weight_diagnostics(result.x),
        })
    else:
        output["maximum_equality_residual"] = None
    return output


def solve_maximum_minimum_weight_lp(minus: np.ndarray, plus: np.ndarray) -> dict:
    """Maximize a common lower bound on every endpoint probability weight."""
    minus, plus = _validate_pair(minus, plus)
    n_minus, n_plus = len(minus), len(plus)
    equality, rhs = common_hull_equalities(minus, plus, slack_column=True)
    objective = np.zeros(n_minus + n_plus + 1, dtype=np.float64)
    objective[-1] = -1.0
    inequality = np.zeros((n_minus + n_plus, len(objective)), dtype=np.float64)
    inequality[np.arange(n_minus), np.arange(n_minus)] = -1.0
    inequality[np.arange(n_minus), -1] = 1.0
    rows = n_minus + np.arange(n_plus)
    inequality[rows, n_minus + np.arange(n_plus)] = -1.0
    inequality[rows, -1] = 1.0
    result = linprog(
        objective, A_ub=inequality, b_ub=np.zeros(n_minus + n_plus),
        A_eq=equality, b_eq=rhs, bounds=(0.0, None), method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    output = {
        "solver": "scipy.optimize.linprog maximum-minimum-weight (HiGHS)",
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "iterations": int(result.nit),
    }
    if not result.success:
        output.update({"maximum_equality_residual": None, "maximum_minimum_weight": None, "target": None})
        return output
    a, b, minimum_weight = result.x[:n_minus], result.x[n_minus:n_minus + n_plus], result.x[-1]
    minus_target, plus_target = a @ minus, b @ plus
    output.update({
        "maximum_equality_residual": float(np.max(np.abs(equality @ result.x - rhs))),
        "endpoint_target_disagreement": float(np.max(np.abs(minus_target - plus_target))),
        "maximum_minimum_weight": float(minimum_weight),
        "maximum_minimum_weight_fraction_of_uniform": float(
            minimum_weight / min(1.0 / n_minus, 1.0 / n_plus)
        ),
        "target": 0.5 * (minus_target + plus_target),
        "minus_weights": a, "plus_weights": b,
        "minus_weight_diagnostics": _weight_diagnostics(a),
        "plus_weight_diagnostics": _weight_diagnostics(b),
    })
    return output


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _moments_covariance(features: np.ndarray, weights: np.ndarray):
    moments = weights @ features
    centered = features - moments
    covariance = (centered.T * weights) @ centered
    return moments, covariance


def _stable_newton_direction(covariance: np.ndarray, gradient: np.ndarray, damping: float):
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.maximum(np.diag(covariance), 1e-30)
    scale = np.sqrt(diagonal)
    normalized = covariance / (scale[:, None] * scale[None, :])
    normalized += damping * np.eye(len(gradient))
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    cutoff = max(float(eigenvalues.max()), 1e-30) * 1e-12
    inverse = np.where(eigenvalues > cutoff, 1.0 / np.maximum(eigenvalues, 1e-30), 0.0)
    direction = -(eigenvectors @ (inverse * (eigenvectors.T @ (gradient / scale)))) / scale
    return direction


def _line_search(objective, current: float, gradient: np.ndarray, direction: np.ndarray):
    directional = float(gradient @ direction)
    if not np.isfinite(directional) or directional >= 0.0:
        direction = -gradient
        directional = -float(gradient @ gradient)
    step = 1.0
    for backtrack in range(40):
        candidate = objective(step, direction)
        if np.isfinite(candidate) and candidate <= current + 1e-4 * step * directional:
            return step, direction, candidate, backtrack
        step *= 0.5
    return None, direction, current, 40


def solve_maximum_entropy_common_target(
    minus: np.ndarray,
    plus: np.ndarray,
    *,
    tolerance: float = 1e-11,
    max_iterations: int = 500,
) -> dict:
    """Maximum-total-entropy common point via its convex low-dimensional dual."""
    minus, plus = _validate_pair(minus, plus)
    lam = np.zeros(minus.shape[1], dtype=np.float64)
    trace = []

    def objective_at(candidate):
        return float(logsumexp(minus @ candidate) - np.log(len(minus))
                     + logsumexp(-(plus @ candidate)) - np.log(len(plus)))

    reason = "maximum_iterations"
    for iteration in range(max_iterations + 1):
        wm, wp = _softmax(minus @ lam), _softmax(-(plus @ lam))
        mean_m, covariance_m = _moments_covariance(minus, wm)
        mean_p, covariance_p = _moments_covariance(plus, wp)
        gradient = mean_m - mean_p
        objective = objective_at(lam)
        trace.append({
            "iteration": iteration, "dual_objective": objective,
            "gradient_l2_norm": float(np.linalg.norm(gradient)),
            "maximum_absolute_gradient": float(np.max(np.abs(gradient))),
            "lambda_norm": float(np.linalg.norm(lam)),
        })
        if np.max(np.abs(gradient)) <= tolerance:
            reason = "residual_tolerance"
            break
        if iteration == max_iterations:
            break
        direction = _stable_newton_direction(covariance_m + covariance_p, gradient, 1e-12)
        step, direction, _, backtracks = _line_search(
            lambda alpha, vector: objective_at(lam + alpha * vector),
            objective, gradient, direction,
        )
        trace[-1]["line_search_backtracks"] = backtracks
        if step is None:
            reason = "line_search_failed"
            break
        trace[-1]["accepted_step"] = step
        lam = lam + step * direction
        if np.linalg.norm(lam) > 1e8:
            reason = "diverging_multiplier_boundary_solution"
            break
    wm, wp = _softmax(minus @ lam), _softmax(-(plus @ lam))
    mean_m, _ = _moments_covariance(minus, wm)
    mean_p, _ = _moments_covariance(plus, wp)
    residual = mean_m - mean_p
    converged = bool(np.max(np.abs(residual)) <= tolerance)
    return {
        "criterion": "maximize H(a)+H(b) subject to a,b simplex and matched endpoint moments",
        "solver": "damped Newton on convex log-sum-exp dual with Armijo line search",
        "converged": converged, "convergence_reason": reason,
        "iterations": len(trace) - 1, "trace": trace,
        "initial_dual_objective": trace[0]["dual_objective"],
        "final_dual_objective": trace[-1]["dual_objective"],
        "maximum_equality_residual": float(np.max(np.abs(residual))),
        "target": 0.5 * (mean_m + mean_p) if converged else None,
        "lambda": lam, "minus_weights": wm, "plus_weights": wp,
        "minus_weight_diagnostics": _weight_diagnostics(wm),
        "plus_weight_diagnostics": _weight_diagnostics(wp),
    }


def calibrate_iprojection_instrumented(
    features: np.ndarray,
    target: np.ndarray,
    *,
    initial_lambda: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> dict:
    """Residual-driven Newton wrapper for the unchanged empirical dual.

    The objective is log(mean(exp(Phi lambda))) - lambda^T c. Final weights
    and moments are independently recomputed by the repository implementation.
    """
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    lam = np.zeros(features.shape[1], dtype=np.float64) if initial_lambda is None else np.asarray(initial_lambda, dtype=np.float64).copy()
    trace = []

    def evaluate(candidate):
        logits = features @ candidate
        weights = _softmax(logits)
        moments, covariance = _moments_covariance(features, weights)
        objective = float(logsumexp(logits) - np.log(len(features)) - candidate @ target)
        return objective, weights, moments, covariance

    reason = "maximum_iterations"
    initial_objective = evaluate(lam)[0]
    for iteration in range(max_iterations + 1):
        objective, weights, moments, covariance = evaluate(lam)
        gradient = moments - target
        eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
        trace.append({
            "iteration": iteration, "dual_objective": objective,
            "gradient_l2_norm": float(np.linalg.norm(gradient)),
            "maximum_absolute_residual": float(np.max(np.abs(gradient))),
            "lambda_norm": float(np.linalg.norm(lam)),
            "covariance_minimum_eigenvalue": float(eigenvalues.min()),
            "covariance_maximum_eigenvalue": float(eigenvalues.max()),
        })
        if np.max(np.abs(gradient)) <= tolerance:
            reason = "residual_tolerance"
            break
        if iteration == max_iterations:
            break
        accepted = False
        for damping in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            direction = _stable_newton_direction(covariance, gradient, damping)
            step, direction, _, backtracks = _line_search(
                lambda alpha, vector: evaluate(lam + alpha * vector)[0],
                objective, gradient, direction,
            )
            if step is not None:
                trace[-1].update({
                    "accepted_step": step, "damping": damping,
                    "line_search_backtracks": backtracks,
                })
                lam = lam + step * direction
                accepted = True
                break
        if not accepted:
            reason = "line_search_failed"
            break
        if np.linalg.norm(lam) > 1e10:
            reason = "diverging_multiplier_boundary_target"
            break

    objective, weights, moments, covariance = evaluate(lam)
    direct_residual = moments - target
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    cutoff = max(float(eigenvalues.max()), 1e-30) * core.DEFAULT_RCOND
    retained = eigenvalues[eigenvalues > cutoff]
    rank = int(len(retained))

    repository_weights, repository_moments, repository_covariance = core.empirical_tilt_from_lambda(
        jnp.asarray(lam, dtype=jnp.float64),
        jnp.zeros((len(features),), dtype=jnp.float64),
        jnp.asarray(features, dtype=jnp.float64),
    )
    repository_weights = np.asarray(repository_weights)
    repository_moments = np.asarray(repository_moments)
    repository_covariance = np.asarray(repository_covariance)
    reported_residual = repository_moments - target
    convergence = bool(np.max(np.abs(reported_residual)) <= tolerance)
    if convergence and reason != "residual_tolerance":
        reason = "repository_residual_tolerance"
    return {
        "objective": "log(mean(exp(Phi @ lambda))) - lambda @ target",
        "solver": "residual-driven damped Newton/Armijo wrapper; final evaluation by mfsi_components.empirical_tilt_from_lambda",
        "converged": convergence, "convergence_reason": reason,
        "iterations": len(trace) - 1, "trace": trace,
        "initial_dual_objective": initial_objective, "final_dual_objective": objective,
        "lambda": lam, "lambda_norm": float(np.linalg.norm(lam)),
        "weights": repository_weights, "moments": repository_moments,
        "covariance": repository_covariance,
        "direct_weighted_mean_residual": direct_residual,
        "reported_residual": reported_residual,
        "maximum_absolute_standardized_residual": float(np.max(np.abs(reported_residual))),
        "residual_identity_maximum_difference": float(np.max(np.abs(direct_residual - reported_residual))),
        "numpy_repository_weight_maximum_difference": float(np.max(np.abs(weights - repository_weights))),
        "covariance_eigenvalues": eigenvalues,
        "covariance_rank": rank,
        "covariance_condition": float(retained.max() / retained.min()) if rank else float("inf"),
        **_weight_diagnostics(repository_weights),
    }
