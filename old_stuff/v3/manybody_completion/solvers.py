"""Tilted-particle and dual-calibration scientific solvers.

Tesseract 1 supports two proposal modes:

* the population prior itself (the original implementation), or
* a learned tilt-conditioned defensive proposal.

When the learned proposal is active, the sampler bridges from the proposal
``r`` to the requested unnormalized target ``p exp(lambda Phi)`` along

``pi_beta(y) ∝ r(y) [p(y) exp(lambda Phi(y)) / r(y)]**beta``.

Importance weights and independence-Metropolis corrections therefore preserve
the same final target.  The learned proposal changes efficiency, not the law.

Tesseract 2 optionally receives a learned dual warm start.  It always follows
that prediction with covariance-Newton corrections and an independent final
Tesseract 1 call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .adaptive_components import (
    ProposalModel,
    WarmStartModel,
    exact_tilt_probabilities_jax,
    importance_ess_fraction,
    proposal_probabilities,
    warm_start_dual,
)
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
    proposal_used: bool = False
    proposal_defensive_mixture: float = 1.0
    proposal_expected_ess_fraction: float = 0.0


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
    initial_dual: float = 0.0
    warm_start_used: bool = False


def _normalized_prior(probabilities: np.ndarray, support: PopulationSupport) -> np.ndarray:
    prior = np.asarray(probabilities, dtype=np.float64)
    if prior.shape != (support.size,):
        raise ValueError(f"expected prior shape {(support.size,)}, got {prior.shape}")
    if np.any(prior < 0.0) or not np.all(np.isfinite(prior)):
        raise ValueError("prior probabilities must be finite and nonnegative")
    prior = np.maximum(prior, 1e-15)
    return prior / prior.sum()


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def tilted_ensemble_from_probabilities(
    prior_probabilities_array: np.ndarray,
    support: PopulationSupport,
    dual: float,
    *,
    particles: int,
    tempering_steps: int,
    rejuvenation_steps: int,
    resample_threshold: float,
    seed: int,
    final_particles: int | None = None,
    proposal_model: ProposalModel | None = None,
) -> TiltedEnsembleResult:
    """Approximate ``p(y) exp(dual * Phi(y))`` with corrected particles.

    A learned proposal is optional.  Its probabilities are defensively mixed
    with the prior and all proposal terms are included in the annealed
    importance ratio and Metropolis acceptance ratio.
    """
    _ = final_particles  # calibration-level option accepted for API symmetry
    if particles < 2:
        raise ValueError("particles must be at least 2")
    if tempering_steps < 1:
        raise ValueError("tempering_steps must be positive")
    prior = _normalized_prior(prior_probabilities_array, support)
    if proposal_model is None:
        proposal = prior
        defensive_mixture = 1.0
    else:
        proposal = proposal_probabilities(proposal_model, prior, support, float(dual))
        defensive_mixture = proposal_model.defensive_mixture
    proposal = _normalized_prior(proposal, support)

    exact_target = np.asarray(
        exact_tilt_probabilities_jax(
            np.asarray(prior), np.asarray(support.pair), float(dual)
        ),
        dtype=np.float64,
    )
    expected_ess_fraction = importance_ess_fraction(exact_target, proposal)

    rng = np.random.default_rng(int(seed))
    indices = rng.choice(
        support.size, size=particles, replace=True, p=proposal
    ).astype(np.int64)
    weights = np.full(particles, 1.0 / particles, dtype=np.float64)
    log_normalizer = 0.0
    resampling_count = 0
    accepted = 0
    proposed = 0

    log_target_over_proposal = (
        np.log(np.maximum(prior, 1e-300))
        + float(dual) * support.pair
        - np.log(np.maximum(proposal, 1e-300))
    )
    beta_previous = 0.0
    for beta in np.linspace(1.0 / tempering_steps, 1.0, tempering_steps):
        delta = float(beta - beta_previous)
        increments = np.exp(
            np.clip(delta * log_target_over_proposal[indices], -700.0, 700.0)
        )
        z_increment = float(np.sum(weights * increments))
        if not np.isfinite(z_increment) or z_increment <= 0.0:
            raise FloatingPointError("non-finite annealed importance normalizer increment")
        log_normalizer += math.log(z_increment)
        weights = weights * increments / z_increment

        if _effective_sample_size(weights) < resample_threshold * particles:
            selected = _systematic_resample(weights, rng)
            indices = indices[selected]
            weights.fill(1.0 / particles)
            resampling_count += 1

        for _ in range(rejuvenation_steps):
            proposals = rng.choice(
                support.size, size=particles, replace=True, p=proposal
            ).astype(np.int64)
            log_ratio = float(beta) * (
                log_target_over_proposal[proposals]
                - log_target_over_proposal[indices]
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
        proposal_used=proposal_model is not None,
        proposal_defensive_mixture=float(defensive_mixture),
        proposal_expected_ess_fraction=float(expected_ess_fraction),
    )


def tilted_ensemble(
    params: PriorParameters,
    support: PopulationSupport,
    dual: float,
    **options,
) -> TiltedEnsembleResult:
    """Compatibility wrapper using the exact parametric oracle prior."""
    return tilted_ensemble_from_probabilities(
        prior_probabilities(params, support), support, dual, **options
    )


def calibrate_dual_from_probabilities(
    prior_probabilities_array: np.ndarray,
    support: PopulationSupport,
    target_moment: float,
    *,
    sampler_options: dict,
    calibration_options: dict,
    seed: int,
    proposal_model: ProposalModel | None = None,
    warm_start_model: WarmStartModel | None = None,
) -> CalibrationResult:
    prior = _normalized_prior(prior_probabilities_array, support)
    target = float(target_moment)
    fit_sampler_options = dict(sampler_options)
    final_particles = int(
        fit_sampler_options.pop("final_particles", fit_sampler_options["particles"])
    )
    if final_particles < 2:
        raise ValueError("final_particles must be at least 2")
    if target < float(support.pair.min()) or target > float(support.pair.max()):
        fallback = tilted_ensemble_from_probabilities(
            prior,
            support,
            0.0,
            seed=seed,
            proposal_model=proposal_model,
            **fit_sampler_options,
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
            initial_dual=0.0,
            warm_start_used=warm_start_model is not None,
        )

    max_iterations = int(calibration_options["max_iterations"])
    tolerance = float(calibration_options["tolerance"])
    ridge = float(calibration_options["ridge"])
    damping = float(calibration_options["damping"])
    max_step = float(calibration_options["max_step"])
    weak_covariance = float(calibration_options["weak_support_covariance"])
    max_dual_norm = float(calibration_options["max_dual_norm"])

    initial_dual = (
        warm_start_dual(warm_start_model, prior, support, target)
        if warm_start_model is not None
        else float(calibration_options.get("initial_dual", 0.0))
    )
    dual = float(np.clip(initial_dual, -max_dual_norm, max_dual_norm))
    trace: list[dict[str, float]] = []
    converged = False
    status = "max_iterations"
    for iteration in range(max_iterations):
        ensemble = tilted_ensemble_from_probabilities(
            prior,
            support,
            dual,
            seed=seed,  # common random numbers across nearby dual iterates
            proposal_model=proposal_model,
            **fit_sampler_options,
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
                "ess_fraction": ensemble.effective_sample_size
                / max(float(fit_sampler_options["particles"]), 1.0),
                "proposal_expected_ess_fraction": ensemble.proposal_expected_ess_fraction,
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

    final_sampler_options = dict(fit_sampler_options)
    final_sampler_options["particles"] = final_particles
    final = tilted_ensemble_from_probabilities(
        prior,
        support,
        dual,
        seed=seed + 1_000_003,
        proposal_model=proposal_model,
        **final_sampler_options,
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
        initial_dual=initial_dual,
        warm_start_used=warm_start_model is not None,
    )


def calibrate_dual(
    params: PriorParameters,
    support: PopulationSupport,
    target_moment: float,
    *,
    sampler_options: dict,
    calibration_options: dict,
    seed: int,
    proposal_model: ProposalModel | None = None,
    warm_start_model: WarmStartModel | None = None,
) -> CalibrationResult:
    """Compatibility wrapper using the exact parametric oracle prior."""
    return calibrate_dual_from_probabilities(
        prior_probabilities(params, support),
        support,
        target_moment,
        sampler_options=sampler_options,
        calibration_options=calibration_options,
        seed=seed,
        proposal_model=proposal_model,
        warm_start_model=warm_start_model,
    )
