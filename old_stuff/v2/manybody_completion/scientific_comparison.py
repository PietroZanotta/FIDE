"""Matched scientific comparison for the finite DiffPOP example."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import platform
import numpy as np

from .classical_baselines import maxent_uniform, one_shot_reweight, prior_only
from .energy import conditioned_probabilities, distribution_summaries
from .fast_training import train_variants
from .homometric import build_population_support, certify_pair_ambiguity
from .metrics import (
    energy_distance_discrete,
    energy_score_discrete,
    smoothed_kl,
    total_variation,
)
from .network import PriorParameters
from .solver_factory import create_solver_backend
from .uq import summarize_higher_order


def _evaluate_distribution(
    name: str,
    probabilities: np.ndarray,
    support,
    reference_probabilities: np.ndarray,
    target_moment: float,
    *,
    effective_sample_size: float,
    particle_count: int | None,
    diagnostics: dict,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    reference_probabilities = np.asarray(reference_probabilities, dtype=np.float64)
    reference_probabilities /= reference_probabilities.sum()
    summaries = distribution_summaries(probabilities, support)
    reference = distribution_summaries(reference_probabilities, support)
    ess_denominator = particle_count if particle_count is not None else support.size
    uq = summarize_higher_order(support.triplet, support.labels, probabilities)
    uq["mode_probability_error"] = abs(
        uq["mode_plus_probability"] - reference["mode_plus_probability"]
    )
    return {
        "name": name,
        "pair_mean": summaries["pair_mean"],
        "moment_error": abs(summaries["pair_mean"] - target_moment),
        "pair_variance": summaries["pair_variance"],
        "hidden_mean_error": abs(summaries["triplet_mean"] - reference["triplet_mean"]),
        "mode_probability_error": abs(
            summaries["mode_plus_probability"] - reference["mode_plus_probability"]
        ),
        "hidden_energy_score": energy_score_discrete(
            support.triplet,
            probabilities,
            support.triplet,
            reference_probabilities,
        ),
        "hidden_energy_distance": energy_distance_discrete(
            support.triplet, probabilities, reference_probabilities
        ),
        "joint_total_variation": total_variation(probabilities, reference_probabilities),
        "reference_to_method_kl": smoothed_kl(reference_probabilities, probabilities),
        "effective_sample_size": float(effective_sample_size),
        "ess_fraction": float(effective_sample_size / max(ess_denominator, 1)),
        "higher_order_conditional_uq": uq,
        "diagnostics": diagnostics,
    }


def run_scientific_comparison(config: dict) -> tuple[dict, dict[str, np.ndarray]]:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    support = build_population_support(int(config["system"]["n_spins"]))
    true_params = PriorParameters.from_mapping(config["true_prior"])
    initial_params = PriorParameters.from_mapping(config["learned_initial"])

    true_dual = float(config["target"]["true_tilt"])
    reference_probabilities = conditioned_probabilities(true_params, support, true_dual)
    target_summary = distribution_summaries(reference_probabilities, support)
    target_moment = target_summary["pair_mean"]

    trained = train_variants(config, support, true_params, initial_params, rng)
    backend = create_solver_backend(config.get("solver_backend"))
    sampler_options = dict(config["sampler"])
    calibration_options = dict(config["calibration"])
    particle_count = int(sampler_options["particles"])

    methods: dict[str, dict] = {}

    maxent = maxent_uniform(support, target_moment)
    methods[maxent.name] = _evaluate_distribution(
        maxent.name,
        maxent.probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=maxent.effective_sample_size,
        particle_count=None,
        diagnostics={"dual": maxent.dual, "information_budget": "target moment only"},
    )

    prior = prior_only(trained["mle"], support)
    methods[prior.name] = _evaluate_distribution(
        prior.name,
        prior.probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=prior.effective_sample_size,
        particle_count=None,
        diagnostics={"dual": 0.0, "information_budget": "population samples"},
    )

    one_shot, one_indices, one_weights = one_shot_reweight(
        trained["mle"],
        support,
        target_moment,
        particles=particle_count,
        seed=seed + 200,
    )
    methods[one_shot.name] = _evaluate_distribution(
        one_shot.name,
        one_shot.probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=one_shot.effective_sample_size,
        particle_count=particle_count,
        diagnostics={
            "dual": one_shot.dual,
            "rejuvenation_steps": 0,
            "information_budget": "population samples and target moment",
        },
    )

    calibrated_arrays: dict[str, np.ndarray] = {
        "reference_probabilities": reference_probabilities,
        "support_pair": support.pair,
        "support_triplet": support.triplet,
        "support_labels": support.labels,
        "one_shot_indices": one_indices,
        "one_shot_weights": one_weights,
    }
    calibration_results = {}
    for method_name, key, method_seed in [
        ("Calibrated-StopGrad", "stopgrad", seed + 400),
        ("Full-E2E", "full_e2e", seed + 600),
    ]:
        result = backend.run_dual_calibration(
            trained[key],
            support,
            target_moment,
            sampler_options,
            calibration_options,
            method_seed,
        )
        calibration_results[method_name] = result
        ensemble = result.final_ensemble
        methods[method_name] = _evaluate_distribution(
            method_name,
            ensemble.atom_probabilities,
            support,
            reference_probabilities,
            target_moment,
            effective_sample_size=ensemble.effective_sample_size,
            particle_count=particle_count,
            diagnostics={
                "dual": result.dual,
                "status": result.status,
                "converged": result.converged,
                "iterations": result.iterations,
                "sampler_calls": result.sampler_calls,
                "residual": result.residual,
                "residual_standard_error": result.residual_standard_error,
                "moment_covariance": ensemble.moment_covariance,
                "acceptance_rate": ensemble.acceptance_rate,
                "resampling_count": ensemble.resampling_count,
                "log_normalizer_increment": ensemble.log_normalizer_increment,
                "fit_trace": result.fit_trace,
                "information_budget": "population samples and target moment",
            },
        )
        calibrated_arrays[f"{key}_probabilities"] = ensemble.atom_probabilities
        calibrated_arrays[f"{key}_indices"] = ensemble.indices
        calibrated_arrays[f"{key}_weights"] = ensemble.weights

    methods["Exact-Reference"] = _evaluate_distribution(
        "Exact-Reference",
        reference_probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=float(support.size),
        particle_count=None,
        diagnostics={"dual": true_dual, "oracle": True},
    )

    full = methods["Full-E2E"]
    stop = methods["Calibrated-StopGrad"]
    decision = {
        "exploratory": True,
        "full_minus_stopgrad_ess_fraction": full["ess_fraction"] - stop["ess_fraction"],
        "full_minus_stopgrad_moment_error": full["moment_error"] - stop["moment_error"],
        "full_minus_stopgrad_hidden_energy_score": (
            full["hidden_energy_score"] - stop["hidden_energy_score"]
        ),
        "full_e2e_improves_any_finite_budget_endpoint": bool(
            full["ess_fraction"] > stop["ess_fraction"]
            or full["hidden_energy_score"] < stop["hidden_energy_score"]
            or full["diagnostics"].get("sampler_calls", 10**9)
            < stop["diagnostics"].get("sampler_calls", 10**9)
        ),
        "calibration_gate": bool(
            full["moment_error"]
            <= stop["moment_error"] + 3.0 * float(config["calibration"]["tolerance"])
        ),
        "mode_gate": bool(
            full["mode_probability_error"] <= stop["mode_probability_error"] + 0.08
        ),
    }

    report = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "python": platform.python_version(),
            "support_size": support.size,
            "n_spins": support.n_spins,
            "solver_backend": backend.kind,
        },
        "configuration": config,
        "target": {
            "true_dual": true_dual,
            "target_moment": target_moment,
            "exact_summaries": target_summary,
            "ambiguity_certificate": certify_pair_ambiguity(
                support, conditioned_probabilities(true_params, support, 0.0)
            ),
        },
        "training": {
            "true_parameters": true_params.to_mapping(),
            "initial_parameters": initial_params.to_mapping(),
            "mle_parameters": trained["mle"].to_mapping(),
            "stopgrad_parameters": trained["stopgrad"].to_mapping(),
            "full_e2e_parameters": trained["full_e2e"].to_mapping(),
            "loss_traces": trained["traces"],
        },
        "methods": methods,
        "decision_summary": decision,
    }
    return report, calibrated_arrays
