"""Tilted-particle and dual-calibration scientific solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .energy import prior_probabilities
from .homometric import PopulationSupport
from .network import PriorParameters


@dataclass
class TiltedEnsembleResult:
    indices: np.ndarray
    weights: np.ndarray
    atom_probabilities: np.ndarray
    moment_mean: float
    moment_covariance: float
    hidden_mean: float
    mode_plus_probability: float
    effective_sample_size: float
    log_normalizer_increment: float
    resampling_count: int
    acceptance_rate: float
    dual: float


@dataclass
class CalibrationResult:
    dual: float
    status: str
    converged: bool
    iterations: int
    sampler_calls: int
    fit_trace: list[dict[str, float]]
    final_ensemble: TiltedEnsembleResult
    residual: float
    residual_standard_error: float


def _normalize_logweights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("invalid particle weights")
    return weights / total


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def tilted_ensemble(
    params: PriorParameters,
    support: PopulationSupport,
    dual: float,
    *,
    particles: int,
    tempering_steps: int,
    rejuvenation_steps: int,
    resample_threshold: float,
    seed: int,
) -> TiltedEnsembleResult:
    """Approximate p(y) exp(dual * Phi(y)) using tempered particles.

    Rejuvenation uses an independence Metropolis proposal from the population
    prior.  The prior proposal cancels from the acceptance ratio, leaving only
    the current tempered likelihood factor.
    """
    if particles < 2:
        raise ValueError("particles must be at least 2")
    if tempering_steps < 1:
        raise ValueError("tempering_steps must be positive")
    rng = np.random.default_rng(int(seed))
    prior = prior_probabilities(params, support)
    indices = rng.choice(support.size, size=particles, replace=True, p=prior).astype(np.int64)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    log_normalizer = 0.0
    resampling_count = 0
    accepted = 0
    proposed = 0

    beta_previous = 0.0
    for beta in np.linspace(1.0 / tempering_steps, 1.0, tempering_steps):
        delta = float(beta - beta_previous)
        increments = np.exp(np.clip(delta * float(dual) * support.pair[indices], -700.0, 700.0))
        z_increment = float(np.sum(weights * increments))
        if not np.isfinite(z_increment) or z_increment <= 0.0:
            raise FloatingPointError("non-finite SMC normalizer increment")
        log_normalizer += math.log(z_increment)
        weights = weights * increments / z_increment

        if _effective_sample_size(weights) < resample_threshold * particles:
            selected = _systematic_resample(weights, rng)
            indices = indices[selected]
            weights.fill(1.0 / particles)
            resampling_count += 1

        for _ in range(rejuvenation_steps):
            proposals = rng.choice(support.size, size=particles, replace=True, p=prior).astype(np.int64)
            log_ratio = float(beta) * float(dual) * (
                support.pair[proposals] - support.pair[indices]
            )
            accept = np.log(rng.random(particles)) < np.minimum(log_ratio, 0.0)
            indices[accept] = proposals[accept]
            accepted += int(np.sum(accept))
            proposed += particles
        beta_previous = float(beta)

    moment = float(np.sum(weights * support.pair[indices]))
    covariance = float(np.sum(weights * np.square(support.pair[indices] - moment)))
    hidden = float(np.sum(weights * support.triplet[indices]))
    mode_plus = float(np.sum(weights * (support.labels[indices] > 0)))
    atom_probabilities = np.bincount(
        indices, weights=weights, minlength=support.size
    ).astype(np.float64)
    atom_probabilities /= atom_probabilities.sum()
    acceptance_rate = float(accepted / proposed) if proposed else 1.0
    return TiltedEnsembleResult(
        indices=indices,
        weights=weights,
        atom_probabilities=atom_probabilities,
        moment_mean=moment,
        moment_covariance=covariance,
        hidden_mean=hidden,
        mode_plus_probability=mode_plus,
        effective_sample_size=_effective_sample_size(weights),
        log_normalizer_increment=log_normalizer,
        resampling_count=resampling_count,
        acceptance_rate=acceptance_rate,
        dual=float(dual),
    )


def calibrate_dual(
    params: PriorParameters,
    support: PopulationSupport,
    target_moment: float,
    *,
    sampler_options: dict,
    calibration_options: dict,
    seed: int,
) -> CalibrationResult:
    target = float(target_moment)
    if target < float(support.pair.min()) or target > float(support.pair.max()):
        fallback = tilted_ensemble(
            params,
            support,
            0.0,
            seed=seed,
            **sampler_options,
        )
        return CalibrationResult(
            dual=0.0,
            status="incompatible",
            converged=False,
            iterations=0,
            sampler_calls=1,
            fit_trace=[],
            final_ensemble=fallback,
            residual=fallback.moment_mean - target,
            residual_standard_error=math.inf,
        )

    max_iterations = int(calibration_options["max_iterations"])
    tolerance = float(calibration_options["tolerance"])
    ridge = float(calibration_options["ridge"])
    damping = float(calibration_options["damping"])
    max_step = float(calibration_options["max_step"])
    weak_covariance = float(calibration_options["weak_support_covariance"])
    max_dual_norm = float(calibration_options["max_dual_norm"])

    dual = 0.0
    trace: list[dict[str, float]] = []
    converged = False
    status = "max_iterations"
    for iteration in range(max_iterations):
        ensemble = tilted_ensemble(
            params,
            support,
            dual,
            seed=seed,  # common random numbers across nearby dual iterates
            **sampler_options,
        )
        residual = ensemble.moment_mean - target
        standard_error = math.sqrt(
            max(ensemble.moment_covariance, 0.0)
            / max(ensemble.effective_sample_size, 1.0)
        )
        trace.append(
            {
                "iteration": float(iteration),
                "dual": dual,
                "moment": ensemble.moment_mean,
                "residual": residual,
                "covariance": ensemble.moment_covariance,
                "effective_sample_size": ensemble.effective_sample_size,
                "standard_error": standard_error,
            }
        )
        if abs(residual) <= tolerance + 1.96 * standard_error:
            converged = True
            status = "converged"
            break
        if ensemble.moment_covariance < weak_covariance:
            status = "weak_support"
            break
        step = damping * residual / (ensemble.moment_covariance + ridge)
        dual -= float(np.clip(step, -max_step, max_step))
        if abs(dual) > max_dual_norm:
            status = "weak_support"
            break

    final = tilted_ensemble(
        params,
        support,
        dual,
        seed=seed + 1_000_003,
        **sampler_options,
    )
    residual = final.moment_mean - target
    standard_error = math.sqrt(
        max(final.moment_covariance, 0.0) / max(final.effective_sample_size, 1.0)
    )
    if converged and abs(residual) > tolerance + 2.58 * standard_error:
        status = "monte_carlo_limited"
        converged = False
    return CalibrationResult(
        dual=dual,
        status=status,
        converged=converged,
        iterations=len(trace),
        sampler_calls=len(trace) + 1,
        fit_trace=trace,
        final_ensemble=final,
        residual=residual,
        residual_standard_error=standard_error,
    )
