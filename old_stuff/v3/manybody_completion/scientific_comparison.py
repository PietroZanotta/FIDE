"""Matched flow-matching and DiffPOP comparison on the finite benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import numpy as np

from .adaptive_components import flatten_adaptive_parameters
from .classical_baselines import maxent_uniform, one_shot_reweight_probabilities
from .energy import (
    conditioned_from_reference,
    conditioned_probabilities,
    distribution_summaries,
)
from .fast_training import train_variants
from .flow import flatten_flow_parameters, flow_distribution_from_base
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
    exact_uq: bool = False,
) -> dict:
    probabilities = np.array(probabilities, dtype=np.float64, copy=True)
    probabilities /= probabilities.sum()
    reference_probabilities = np.array(reference_probabilities, dtype=np.float64, copy=True)
    reference_probabilities /= reference_probabilities.sum()
    summaries = distribution_summaries(probabilities, support)
    reference = distribution_summaries(reference_probabilities, support)
    ess_denominator = particle_count if particle_count is not None else support.size
    uq = summarize_higher_order(
        support.triplet,
        support.labels,
        probabilities,
        reference_probabilities=reference_probabilities,
        effective_sample_size=None if exact_uq else effective_sample_size,
    )
    moment_error = abs(summaries["pair_mean"] - target_moment)
    mode_error = abs(
        summaries["mode_plus_probability"] - reference["mode_plus_probability"]
    )
    hidden_score = energy_score_discrete(
        support.triplet,
        probabilities,
        support.triplet,
        reference_probabilities,
    )
    hidden_distance = energy_distance_discrete(
        support.triplet, probabilities, reference_probabilities
    )
    joint_tv = total_variation(probabilities, reference_probabilities)
    return {
        "name": name,
        "pair_mean": summaries["pair_mean"],
        "moment_error": moment_error,
        "model_moment_error": moment_error,
        "particle_moment_error": moment_error,
        "pair_variance": summaries["pair_variance"],
        "hidden_mean_error": abs(summaries["triplet_mean"] - reference["triplet_mean"]),
        "mode_probability_error": mode_error,
        "particle_mode_probability_error": mode_error,
        "hidden_energy_score": hidden_score,
        "particle_hidden_energy_score": hidden_score,
        "hidden_energy_distance": hidden_distance,
        "particle_hidden_energy_distance": hidden_distance,
        "joint_total_variation": joint_tv,
        "particle_joint_total_variation": joint_tv,
        "reference_to_method_kl": smoothed_kl(reference_probabilities, probabilities),
        "effective_sample_size": float(effective_sample_size),
        "ess_fraction": float(effective_sample_size / max(ess_denominator, 1)),
        "higher_order_conditional_uq": uq,
        "diagnostics": diagnostics,
    }


def _flow_diagnostics(distribution, information_budget: str, condition: float | None) -> dict:
    return {
        "status": "generated",
        "information_budget": information_budget,
        "condition": condition,
        "sampling_steps": distribution.sampling_steps,
        "assignment_temperature": distribution.assignment_temperature,
        "quantization_rmse": distribution.quantization_rmse,
        "mean_assignment_entropy": distribution.mean_assignment_entropy,
    }


def _evaluate_calibrated_flow(
    *,
    method_name: str,
    prior_probabilities_array: np.ndarray,
    support,
    reference_probabilities: np.ndarray,
    target_moment: float,
    backend,
    sampler_options: dict,
    calibration_options: dict,
    seed: int,
    flow_diagnostics: dict,
    proposal_model=None,
    warm_start_model=None,
) -> tuple[dict, object, np.ndarray]:
    result = backend.run_dual_calibration_probabilities(
        prior_probabilities_array,
        support,
        target_moment,
        sampler_options,
        calibration_options,
        seed,
        proposal_model=proposal_model,
        warm_start_model=warm_start_model,
    )
    exact_probabilities, exact_dual = conditioned_from_reference(
        prior_probabilities_array, support, target_moment
    )
    ensemble = result.final_ensemble
    final_particle_count = int(ensemble.indices.size)
    diagnostics = {
        **flow_diagnostics,
        "dual": result.dual,
        "initial_dual": result.initial_dual,
        "warm_start_used": result.warm_start_used,
        "warm_start_absolute_error": abs(result.initial_dual - exact_dual),
        "exact_finite_support_dual": exact_dual,
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
        "proposal_used": ensemble.proposal_used,
        "proposal_defensive_mixture": ensemble.proposal_defensive_mixture,
        "proposal_expected_ess_fraction": ensemble.proposal_expected_ess_fraction,
        "fit_particle_count": int(sampler_options["particles"]),
        "final_particle_count": final_particle_count,
        "particle_to_exact_tilt_tv": total_variation(
            ensemble.atom_probabilities, exact_probabilities
        ),
        "fit_trace": result.fit_trace,
    }
    model_evaluated = _evaluate_distribution(
        method_name,
        exact_probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=float(support.size),
        particle_count=None,
        diagnostics=diagnostics,
        exact_uq=True,
    )
    particle_evaluated = _evaluate_distribution(
        method_name,
        ensemble.atom_probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=ensemble.effective_sample_size,
        particle_count=final_particle_count,
        diagnostics=diagnostics,
    )
    # Distributional fidelity is evaluated on the exact finite-support tilt in this
    # toy benchmark.  Fresh-particle quantities remain explicit because they are
    # the operational calibration and Monte Carlo diagnostics available in a
    # non-enumerable many-body system.
    evaluated = model_evaluated
    evaluated["model_moment_error"] = model_evaluated["moment_error"]
    evaluated["moment_error"] = particle_evaluated["moment_error"]
    evaluated["particle_moment_error"] = particle_evaluated["moment_error"]
    evaluated["particle_pair_mean"] = particle_evaluated["pair_mean"]
    evaluated["particle_mode_probability_error"] = particle_evaluated[
        "mode_probability_error"
    ]
    evaluated["particle_hidden_energy_score"] = particle_evaluated[
        "hidden_energy_score"
    ]
    evaluated["particle_hidden_energy_distance"] = particle_evaluated[
        "hidden_energy_distance"
    ]
    evaluated["particle_joint_total_variation"] = particle_evaluated[
        "joint_total_variation"
    ]
    evaluated["particle_higher_order_conditional_uq"] = particle_evaluated[
        "higher_order_conditional_uq"
    ]
    evaluated["effective_sample_size"] = particle_evaluated["effective_sample_size"]
    evaluated["ess_fraction"] = particle_evaluated["ess_fraction"]
    return evaluated, result, exact_probabilities


def run_scientific_comparison(config: dict) -> tuple[dict, dict[str, np.ndarray]]:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    support = build_population_support(int(config["system"]["n_spins"]))
    true_params = PriorParameters.from_mapping(config["true_prior"])
    initial_mapping = config.get("learned_initial", config["true_prior"])
    initial_params = PriorParameters.from_mapping(initial_mapping)

    true_dual = float(config["target"]["true_tilt"])
    reference_probabilities = conditioned_probabilities(true_params, support, true_dual)
    target_summary = distribution_summaries(reference_probabilities, support)
    target_moment = target_summary["pair_mean"]

    trained = train_variants(config, support, true_params, initial_params, rng)
    backend = create_solver_backend(config.get("solver_backend"))
    sampler_options = dict(config["sampler"])
    calibration_options = dict(config["calibration"])
    particle_count = int(sampler_options["particles"])
    flow_config = config["flow"]
    evaluation_sample_count = int(flow_config["evaluation_samples"])
    state_dim = support.n_spins + 1
    common_base = np.random.default_rng(seed + 1701).normal(
        size=(evaluation_sample_count, state_dim)
    )

    population_distribution = flow_distribution_from_base(
        trained["population_flow"],
        support,
        common_base,
        condition=None,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    )
    direct_distribution = flow_distribution_from_base(
        trained["direct_flow"],
        support,
        common_base,
        condition=target_moment,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    )
    stopgrad_distribution = flow_distribution_from_base(
        trained["stopgrad"],
        support,
        common_base,
        condition=None,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    )
    full_distribution = flow_distribution_from_base(
        trained["full_e2e"],
        support,
        common_base,
        condition=None,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    )
    synergy_distribution = flow_distribution_from_base(
        trained["synergy_e2e"],
        support,
        common_base,
        condition=None,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    )

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
        diagnostics={
            "dual": maxent.dual,
            "status": "exact",
            "information_budget": "target pair moment only",
        },
    )

    methods["Population-Flow"] = _evaluate_distribution(
        "Population-Flow",
        population_distribution.probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=evaluation_sample_count,
        particle_count=evaluation_sample_count,
        diagnostics=_flow_diagnostics(
            population_distribution, "unconditional microscopic population samples", None
        ),
    )
    methods["Direct-Conditional-Flow"] = _evaluate_distribution(
        "Direct-Conditional-Flow",
        direct_distribution.probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=evaluation_sample_count,
        particle_count=evaluation_sample_count,
        diagnostics=_flow_diagnostics(
            direct_distribution,
            "conditional microscopic samples paired with target moments",
            target_moment,
        ),
    )

    one_shot, one_indices, one_weights = one_shot_reweight_probabilities(
        population_distribution.probabilities,
        support,
        target_moment,
        particles=particle_count,
        seed=seed + 200,
        name="Flow-One-Shot-Reweight",
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
            **_flow_diagnostics(
                population_distribution,
                "unconditional population flow plus target moment",
                None,
            ),
            "dual": one_shot.dual,
            "status": "reweighted",
            "rejuvenation_steps": 0,
        },
    )

    arrays: dict[str, np.ndarray] = {
        "reference_probabilities": reference_probabilities,
        "support_pair": support.pair,
        "support_triplet": support.triplet,
        "support_labels": support.labels,
        "evaluation_base_samples": common_base,
        "population_flow_probabilities": population_distribution.probabilities,
        "population_flow_samples": population_distribution.samples,
        "direct_flow_probabilities": direct_distribution.probabilities,
        "direct_flow_samples": direct_distribution.samples,
        "stopgrad_flow_probabilities": stopgrad_distribution.probabilities,
        "full_e2e_flow_probabilities": full_distribution.probabilities,
        "synergy_e2e_flow_probabilities": synergy_distribution.probabilities,
        "one_shot_indices": one_indices,
        "one_shot_weights": one_weights,
    }

    calibrated_specs = [
        (
            "Flow-DiffPOP-PostHoc",
            population_distribution,
            seed + 400,
            "unconditional population flow plus target moment; frozen flow",
            "posthoc",
            None,
            None,
        ),
        (
            "Flow-DiffPOP-StopGrad",
            stopgrad_distribution,
            seed + 500,
            "population and conditional samples; dual derivative stopped",
            "stopgrad",
            None,
            None,
        ),
        (
            "Flow-DiffPOP-FullE2E",
            full_distribution,
            seed + 600,
            "population and conditional samples; gradients through flow and dual",
            "full_e2e",
            None,
            None,
        ),
        (
            "Flow-DiffPOP-SynergyE2E",
            synergy_distribution,
            seed + 700,
            "jointly trained flow, defensive tilt proposal, and corrected dual warm start",
            "synergy_e2e",
            trained["synergy_proposal"],
            trained["synergy_warm_start"],
        ),
    ]
    for (
        method_name,
        distribution,
        method_seed,
        budget,
        key,
        proposal_model,
        warm_start_model,
    ) in calibrated_specs:
        flow_diagnostics = _flow_diagnostics(distribution, budget, None)
        evaluated, result, exact_tilt = _evaluate_calibrated_flow(
            method_name=method_name,
            prior_probabilities_array=distribution.probabilities,
            support=support,
            reference_probabilities=reference_probabilities,
            target_moment=target_moment,
            backend=backend,
            sampler_options=sampler_options,
            calibration_options=calibration_options,
            seed=method_seed,
            flow_diagnostics=flow_diagnostics,
            proposal_model=proposal_model,
            warm_start_model=warm_start_model,
        )
        methods[method_name] = evaluated
        arrays[f"{key}_conditioned_probabilities"] = result.final_ensemble.atom_probabilities
        arrays[f"{key}_exact_tilt_probabilities"] = exact_tilt
        arrays[f"{key}_indices"] = result.final_ensemble.indices
        arrays[f"{key}_weights"] = result.final_ensemble.weights

    methods["Exact-Reference"] = _evaluate_distribution(
        "Exact-Reference",
        reference_probabilities,
        support,
        reference_probabilities,
        target_moment,
        effective_sample_size=float(support.size),
        particle_count=None,
        diagnostics={"dual": true_dual, "status": "oracle", "oracle": True},
    )

    for model_name in (
        "population_flow",
        "direct_flow",
        "stopgrad",
        "full_e2e",
        "synergy_e2e",
    ):
        for array_name, values in flatten_flow_parameters(trained[model_name]).items():
            arrays[f"parameters_{model_name}_{array_name}"] = values
    for array_name, values in flatten_adaptive_parameters(
        trained["synergy_proposal"], trained["synergy_warm_start"]
    ).items():
        arrays[f"parameters_synergy_{array_name}"] = values

    direct = methods["Direct-Conditional-Flow"]
    posthoc = methods["Flow-DiffPOP-PostHoc"]
    stopgrad = methods["Flow-DiffPOP-StopGrad"]
    full = methods["Flow-DiffPOP-FullE2E"]
    synergy = methods["Flow-DiffPOP-SynergyE2E"]
    mode_margin = float(config["evaluation"].get("mode_noninferiority_margin", 0.08))
    score_margin = float(config["evaluation"].get("score_noninferiority_margin", 0.02))
    calibration_margin = float(config["evaluation"].get("calibration_noninferiority_margin", 0.02))
    decision = {
        "exploratory": True,
        "synergy_minus_full_sampler_calls": (
            synergy["diagnostics"]["sampler_calls"] - full["diagnostics"]["sampler_calls"]
        ),
        "synergy_minus_posthoc_sampler_calls": (
            synergy["diagnostics"]["sampler_calls"] - posthoc["diagnostics"]["sampler_calls"]
        ),
        "synergy_minus_full_ess_fraction": synergy["ess_fraction"] - full["ess_fraction"],
        "synergy_minus_posthoc_ess_fraction": synergy["ess_fraction"] - posthoc["ess_fraction"],
        "synergy_minus_full_hidden_energy_score": (
            synergy["hidden_energy_score"] - full["hidden_energy_score"]
        ),
        "synergy_minus_posthoc_hidden_energy_score": (
            synergy["hidden_energy_score"] - posthoc["hidden_energy_score"]
        ),
        "synergy_minus_full_moment_error": synergy["moment_error"] - full["moment_error"],
        "synergy_minus_posthoc_moment_error": (
            synergy["moment_error"] - posthoc["moment_error"]
        ),
        "synergy_improves_a_finite_budget_endpoint": bool(
            synergy["ess_fraction"] > max(full["ess_fraction"], posthoc["ess_fraction"])
            or synergy["diagnostics"]["sampler_calls"]
            < min(full["diagnostics"]["sampler_calls"], posthoc["diagnostics"]["sampler_calls"])
            or synergy["hidden_energy_score"]
            < min(full["hidden_energy_score"], posthoc["hidden_energy_score"])
        ),
        "full_minus_direct_flow_moment_error": full["moment_error"] - direct["moment_error"],
        "full_minus_direct_flow_hidden_energy_score": (
            full["hidden_energy_score"] - direct["hidden_energy_score"]
        ),
        "full_minus_direct_flow_mode_error": (
            full["mode_probability_error"] - direct["mode_probability_error"]
        ),
        "posthoc_minus_direct_flow_moment_error": (
            posthoc["moment_error"] - direct["moment_error"]
        ),
        "posthoc_minus_direct_flow_hidden_energy_score": (
            posthoc["hidden_energy_score"] - direct["hidden_energy_score"]
        ),
        "full_minus_stopgrad_ess_fraction": full["ess_fraction"] - stopgrad["ess_fraction"],
        "full_minus_stopgrad_hidden_energy_score": (
            full["hidden_energy_score"] - stopgrad["hidden_energy_score"]
        ),
        "full_minus_posthoc_ess_fraction": full["ess_fraction"] - posthoc["ess_fraction"],
        "full_minus_posthoc_hidden_energy_score": (
            full["hidden_energy_score"] - posthoc["hidden_energy_score"]
        ),
        "diffpop_full_calibration_noninferior_to_direct": bool(
            full["moment_error"] <= direct["moment_error"] + calibration_margin
        ),
        "diffpop_full_mode_noninferior_to_direct": bool(
            full["mode_probability_error"] <= direct["mode_probability_error"] + mode_margin
        ),
        "diffpop_full_score_noninferior_to_direct": bool(
            full["hidden_energy_score"] <= direct["hidden_energy_score"] + score_margin
        ),
        "diffpop_full_improves_direct_flow_on_any_primary_metric": bool(
            full["moment_error"] < direct["moment_error"]
            or full["hidden_energy_score"] < direct["hidden_energy_score"]
            or full["mode_probability_error"] < direct["mode_probability_error"]
        ),
        "diffpop_posthoc_supported_in_this_run": bool(
            posthoc["moment_error"] <= direct["moment_error"] + calibration_margin
            and posthoc["mode_probability_error"] <= direct["mode_probability_error"] + mode_margin
            and posthoc["hidden_energy_score"] < direct["hidden_energy_score"]
        ),
        "diffpop_full_supported_in_this_run": bool(
            full["moment_error"] <= direct["moment_error"] + calibration_margin
            and full["mode_probability_error"] <= direct["mode_probability_error"] + mode_margin
            and full["hidden_energy_score"] < direct["hidden_energy_score"]
        ),
        "diffpop_synergy_supported_in_this_run": bool(
            synergy["moment_error"] <= direct["moment_error"] + calibration_margin
            and synergy["mode_probability_error"]
            <= direct["mode_probability_error"] + mode_margin
            and synergy["hidden_energy_score"] <= direct["hidden_energy_score"] + score_margin
            and (
                synergy["ess_fraction"] > full["ess_fraction"]
                or synergy["diagnostics"]["sampler_calls"]
                < full["diagnostics"]["sampler_calls"]
                or synergy["hidden_energy_score"] < full["hidden_energy_score"]
            )
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
            "learned_model": "continuous conditional flow matching",
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
            "flow_architecture": {
                "state_dim": state_dim,
                "hidden_width": int(flow_config["hidden_width"]),
                "hidden_layers": int(flow_config["hidden_layers"]),
                "label_scale": float(flow_config["label_scale"]),
            },
            "parameter_counts": trained["parameter_counts"],
            "loss_traces": trained["traces"],
            "gradient_norm_traces": trained["gradient_norms"],
            "synergy_diagnostics": trained["synergy_diagnostics"],
            "synergy_component_architecture": {
                "proposal": trained["synergy_proposal"].to_mapping() | {"parameters": "stored in NPZ"},
                "warm_start": trained["synergy_warm_start"].to_mapping() | {"parameters": "stored in NPZ"},
            },
            "training_task_targets": [task.target_moment for task in trained["tasks"]],
            "training_task_generating_duals": [task.generating_dual for task in trained["tasks"]],
        },
        "methods": methods,
        "decision_summary": decision,
    }
    return report, arrays
