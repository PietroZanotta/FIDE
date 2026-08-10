"""Shared-target endpoint calibration using the repository I-projection."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import mfsi_components as core


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _weighted_moments(features: np.ndarray, weights: np.ndarray):
    mean = weights @ features
    centered = features - mean
    covariance = (centered.T * weights) @ centered
    return mean, covariance


def select_central_common_target(
    minus_features: np.ndarray,
    plus_features: np.ndarray,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-11,
) -> dict:
    """Minimize total endpoint projection KL over a shared feasible target.

    At the central optimum the endpoint multipliers are opposite. Solving
    E_minus,lambda[Phi] = E_plus,-lambda[Phi] therefore finds the common target
    without using any morphology or learned-method result.
    """
    minus = np.asarray(minus_features, dtype=np.float64)
    plus = np.asarray(plus_features, dtype=np.float64)
    if minus.ndim != 2 or plus.ndim != 2 or minus.shape[1] != plus.shape[1]:
        raise ValueError("endpoint features must have matching [bank, R] shapes")
    lam = np.zeros(minus.shape[1], dtype=np.float64)
    converged = False
    for iteration in range(max_iterations):
        wm = _softmax(minus @ lam)
        wp = _softmax(-(plus @ lam))
        mean_m, covariance_m = _weighted_moments(minus, wm)
        mean_p, covariance_p = _weighted_moments(plus, wp)
        residual = mean_m - mean_p
        if np.max(np.abs(residual)) <= tolerance:
            converged = True
            break
        covariance = covariance_m + covariance_p
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
        normalized = covariance / (scale[:, None] * scale[None, :])
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (normalized + normalized.T))
        cutoff = max(float(eigenvalues.max()), 1e-30) * 1e-11
        inverse = np.where(eigenvalues > cutoff, 1.0 / np.maximum(eigenvalues, 1e-30), 0.0)
        step = (eigenvectors @ (inverse * (eigenvectors.T @ (residual / scale)))) / scale
        step_norm = np.linalg.norm(step)
        if step_norm > 2.0:
            step *= 2.0 / step_norm
        base_error = np.linalg.norm(residual)
        accepted = False
        for power in range(14):
            candidate = lam - step * (0.5 ** power)
            cm = _softmax(minus @ candidate) @ minus
            cp = _softmax(-(plus @ candidate)) @ plus
            if np.linalg.norm(cm - cp) < base_error:
                lam = candidate
                accepted = True
                break
        if not accepted:
            break
    wm = _softmax(minus @ lam)
    wp = _softmax(-(plus @ lam))
    mean_m, _ = _weighted_moments(minus, wm)
    mean_p, _ = _weighted_moments(plus, wp)
    return {
        "target": 0.5 * (mean_m + mean_p),
        "selection_lambda": lam,
        "selection_residual": float(np.max(np.abs(mean_m - mean_p))),
        "iterations": iteration + 1,
        "converged": converged,
    }


def calibrate_endpoint(features: np.ndarray, target: np.ndarray) -> dict:
    """Call the existing validated empirical I-projection and report diagnostics."""
    ph = jnp.asarray(features, dtype=jnp.float64)
    target_jax = jnp.asarray(target, dtype=jnp.float64)
    log_base = jnp.zeros((len(features),), dtype=jnp.float64)
    lam = core.calibrate_empirical_implicit(log_base, ph, target_jax)
    weights, moments, covariance = core.empirical_tilt_from_lambda(lam, log_base, ph)
    lam, weights, moments, covariance = map(np.asarray, (lam, weights, moments, covariance))
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    cutoff = max(float(eigenvalues.max()), 1e-30) * core.DEFAULT_RCOND
    retained = eigenvalues[eigenvalues > cutoff]
    rank = int(len(retained))
    condition = float(retained.max() / retained.min()) if rank else float("inf")
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1e-300))) / np.log(len(weights)))
    return {
        "solver": "mfsi_components.calibrate_empirical_implicit",
        "lambda": lam,
        "weights": weights,
        "moments": moments,
        "covariance": covariance,
        "max_abs_residual": float(np.max(np.abs(moments - target))),
        "ess_fraction": float(1.0 / (len(weights) * np.sum(weights * weights))),
        "weight_entropy_fraction": entropy,
        "max_weight": float(weights.max()),
        "lambda_norm": float(np.linalg.norm(lam)),
        "covariance_rank": rank,
        "covariance_condition": condition,
    }


def _feasibility_diagnostics(
    features: np.ndarray, target: np.ndarray, lam: np.ndarray, weights: np.ndarray
) -> dict:
    moments, covariance = _weighted_moments(features, weights)
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    cutoff = max(float(eigenvalues.max()), 1e-30) * core.DEFAULT_RCOND
    retained = eigenvalues[eigenvalues > cutoff]
    rank = int(len(retained))
    return {
        "solver": "symmetric_common_target_feasibility",
        "lambda": lam, "weights": weights, "moments": moments, "covariance": covariance,
        "max_abs_residual": float(np.max(np.abs(moments - target))),
        "ess_fraction": float(1.0 / (len(weights) * np.sum(weights * weights))),
        "weight_entropy_fraction": float(
            -np.sum(weights * np.log(np.maximum(weights, 1e-300))) / np.log(len(weights))
        ),
        "max_weight": float(weights.max()), "lambda_norm": float(np.linalg.norm(lam)),
        "covariance_rank": rank,
        "covariance_condition": (
            float(retained.max() / retained.min()) if rank else float("inf")
        ),
    }


def calibrate_shared_target(minus_features: np.ndarray, plus_features: np.ndarray) -> dict:
    selected = select_central_common_target(minus_features, plus_features)
    target = selected["target"]
    if selected["selection_residual"] <= 1e-5:
        # Only a feasible common target is passed to the repository's validated
        # I-projection. Infeasible pairs are explicitly rejected without asking
        # its finite Newton loop to chase a target outside the empirical hull.
        minus = calibrate_endpoint(minus_features, target)
        plus = calibrate_endpoint(plus_features, target)
    else:
        lam = selected["selection_lambda"]
        minus_weights = _softmax(np.asarray(minus_features) @ lam)
        plus_weights = _softmax(-(np.asarray(plus_features) @ lam))
        minus = _feasibility_diagnostics(minus_features, target, lam, minus_weights)
        plus = _feasibility_diagnostics(plus_features, target, -lam, plus_weights)
    return {"selection": selected, "target": target, "minus": minus, "plus": plus}
