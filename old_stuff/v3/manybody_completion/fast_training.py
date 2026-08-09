"""Bounded flow-matching and DiffPOP training orchestration."""

from __future__ import annotations

import numpy as np

from .adaptive_components import (
    ProposalArchitecture,
    WarmStartArchitecture,
    initialize_proposal_model,
    initialize_warm_start_model,
)
from .flow import (
    FlowArchitecture,
    fine_tune_diffpop_flow,
    fit_flow_matching,
    initialize_flow_model,
    sample_flow_distribution,
    support_embeddings,
)
from .homometric import PopulationSupport
from .network import PriorParameters
from .synergy_training import pretrain_adaptive_components, fine_tune_synergy_system
from .training import generate_training_samples, make_conditional_tasks


def train_variants(
    config: dict,
    support: PopulationSupport,
    true_params: PriorParameters,
    initial_params: PriorParameters | None,
    rng: np.random.Generator,
) -> dict:
    """Train an unconditional population flow, a direct conditional flow, and DiffPOP routes."""
    _ = initial_params  # retained in the public signature for old callers/configurations
    training = config["training"]
    flow_config = config["flow"]
    embeddings = support_embeddings(support, float(flow_config["label_scale"]))

    prior_ids = generate_training_samples(
        true_params, support, int(training["prior_samples"]), rng
    )
    tasks = make_conditional_tasks(
        true_params,
        support,
        training["task_tilts"],
        int(training["samples_per_task"]),
        rng,
    )

    population_architecture = FlowArchitecture(
        state_dim=embeddings.shape[1],
        hidden_width=int(flow_config["hidden_width"]),
        hidden_layers=int(flow_config["hidden_layers"]),
        condition_dim=0,
    )
    initial_population = initialize_flow_model(
        population_architecture,
        seed=int(config["seed"]) + 11,
        label_scale=float(flow_config["label_scale"]),
    )
    population_flow, population_trace, population_gradient_norms = fit_flow_matching(
        initial_population,
        embeddings[prior_ids],
        conditions=None,
        steps=int(training["flow_steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["flow_learning_rate"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(config["seed"]) + 101,
    )

    direct_architecture = FlowArchitecture(
        state_dim=embeddings.shape[1],
        hidden_width=int(flow_config["hidden_width"]),
        hidden_layers=int(flow_config["hidden_layers"]),
        condition_dim=1,
    )
    direct_initial = initialize_flow_model(
        direct_architecture,
        seed=int(config["seed"]) + 13,
        label_scale=float(flow_config["label_scale"]),
    )
    direct_ids = np.concatenate([task.sample_indices for task in tasks])
    direct_conditions = np.concatenate(
        [
            np.full(task.sample_indices.size, task.target_moment, dtype=np.float64)
            for task in tasks
        ]
    )
    direct_flow, direct_trace, direct_gradient_norms = fit_flow_matching(
        direct_initial,
        embeddings[direct_ids],
        conditions=direct_conditions,
        steps=int(training["direct_flow_steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["flow_learning_rate"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(config["seed"]) + 103,
    )

    fine_tune_arguments = {
        "support": support,
        "task_targets": [task.target_moment for task in tasks],
        "task_sample_indices": [task.sample_indices for task in tasks],
        "steps": int(training["conditional_steps"]),
        "learning_rate": float(training["conditional_learning_rate"]),
        "flow_particles": int(training["fine_tune_particles"]),
        "sampling_steps": int(flow_config["training_sampling_steps"]),
        "assignment_temperature": float(flow_config["assignment_temperature"]),
        "dual_iterations": int(training["differentiable_dual_iterations"]),
        "ess_weight": float(training["ess_weight"]),
        "anchor_weight": float(training["anchor_weight"]),
        "gradient_clip": float(training["gradient_clip"]),
    }
    stopgrad, stopgrad_trace, stopgrad_gradient_norms = fine_tune_diffpop_flow(
        population_flow,
        differentiate_dual=False,
        seed=int(config["seed"]) + 107,
        **fine_tune_arguments,
    )
    full_e2e, full_trace, full_gradient_norms = fine_tune_diffpop_flow(
        population_flow,
        differentiate_dual=True,
        seed=int(config["seed"]) + 107,  # matched base noise and initialization
        **fine_tune_arguments,
    )

    synergy_config = config.get("synergy", {})
    proposal_architecture = ProposalArchitecture(
        hidden_width=int(synergy_config.get("proposal_hidden_width", 32)),
        hidden_layers=int(synergy_config.get("proposal_hidden_layers", 2)),
    )
    warm_architecture = WarmStartArchitecture(
        hidden_width=int(synergy_config.get("warm_hidden_width", 24)),
        hidden_layers=int(synergy_config.get("warm_hidden_layers", 2)),
    )
    proposal_initial = initialize_proposal_model(
        proposal_architecture,
        seed=int(config["seed"]) + 211,
        defensive_mixture=float(synergy_config.get("defensive_mixture", 0.10)),
        max_logit_correction=float(synergy_config.get("max_logit_correction", 8.0)),
    )
    warm_initial = initialize_warm_start_model(
        warm_architecture,
        seed=int(config["seed"]) + 223,
        max_abs_dual=float(config["calibration"]["max_dual_norm"]),
    )
    component_prior = sample_flow_distribution(
        population_flow,
        support,
        sample_count=int(synergy_config.get("component_prior_samples", 1024)),
        seed=int(config["seed"]) + 227,
        sampling_steps=int(flow_config["evaluation_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        integration_method=str(flow_config.get("integration_method", "heun")),
    ).probabilities
    component_pretraining = pretrain_adaptive_components(
        proposal_initial,
        warm_initial,
        support,
        component_prior,
        synergy_config.get("pretraining_duals", training["task_tilts"]),
        steps=int(synergy_config.get("component_pretrain_steps", 120)),
        learning_rate=float(synergy_config.get("component_learning_rate", 0.002)),
        proposal_ess_weight=float(synergy_config.get("pretrain_ess_weight", 0.20)),
        warm_start_weight=float(synergy_config.get("pretrain_warm_weight", 0.25)),
        gradient_clip=float(training["gradient_clip"]),
    )
    synergy = fine_tune_synergy_system(
        population_flow,
        component_pretraining.proposal_model,
        component_pretraining.warm_start_model,
        support,
        [task.target_moment for task in tasks],
        [task.sample_indices for task in tasks],
        steps=int(synergy_config.get("joint_steps", training["conditional_steps"])),
        flow_learning_rate=float(
            synergy_config.get("joint_flow_learning_rate", training["conditional_learning_rate"])
        ),
        component_learning_rate=float(
            synergy_config.get("joint_component_learning_rate", 0.0005)
        ),
        flow_particles=int(synergy_config.get("joint_flow_particles", training["fine_tune_particles"])),
        sampling_steps=int(flow_config["training_sampling_steps"]),
        assignment_temperature=float(flow_config["assignment_temperature"]),
        limited_dual_iterations=int(synergy_config.get("limited_dual_iterations", 3)),
        reference_dual_iterations=int(
            synergy_config.get("reference_dual_iterations", training["differentiable_dual_iterations"])
        ),
        dual_ridge=float(config["calibration"]["ridge"]),
        dual_damping=float(config["calibration"]["damping"]),
        dual_max_step=float(config["calibration"]["max_step"]),
        conditional_score_weight=float(synergy_config.get("conditional_score_weight", 1.0)),
        proposal_ess_weight=float(synergy_config.get("proposal_ess_weight", 0.15)),
        fresh_residual_weight=float(synergy_config.get("fresh_residual_weight", 12.0)),
        warm_start_weight=float(synergy_config.get("warm_start_weight", 0.15)),
        dual_path_weight=float(synergy_config.get("dual_path_weight", 2.0)),
        anchor_weight=float(synergy_config.get("anchor_weight", training["anchor_weight"])),
        proposal_reference_weight=float(
            synergy_config.get("proposal_reference_weight", 0.30)
        ),
        seed=int(config["seed"]) + 229,
        gradient_clip=float(training["gradient_clip"]),
    )

    return {
        "population_flow": population_flow,
        "direct_flow": direct_flow,
        "stopgrad": stopgrad,
        "full_e2e": full_e2e,
        "synergy_e2e": synergy.flow_model,
        "synergy_proposal": synergy.proposal_model,
        "synergy_warm_start": synergy.warm_start_model,
        "tasks": tasks,
        "prior_training_indices": prior_ids,
        "traces": {
            "population_flow_matching": population_trace,
            "direct_conditional_flow_matching": direct_trace,
            "diffpop_stopgrad": stopgrad_trace,
            "diffpop_full_e2e": full_trace,
            "synergy_component_pretraining": component_pretraining.loss_trace,
            "diffpop_synergy_e2e": synergy.loss_trace,
        },
        "gradient_norms": {
            "population_flow_matching": population_gradient_norms,
            "direct_conditional_flow_matching": direct_gradient_norms,
            "diffpop_stopgrad": stopgrad_gradient_norms,
            "diffpop_full_e2e": full_gradient_norms,
            "diffpop_synergy_e2e": synergy.gradient_norm_trace,
        },
        "synergy_diagnostics": {
            "component_pretraining_proposal_ess_fraction": component_pretraining.proposal_ess_trace,
            "component_pretraining_warm_rmse": component_pretraining.warm_rmse_trace,
            "joint_training": synergy.diagnostics_trace,
        },
        "parameter_counts": {
            "population_flow": population_architecture.parameter_count,
            "direct_conditional_flow": direct_architecture.parameter_count,
            "diffpop_stopgrad": population_architecture.parameter_count,
            "diffpop_full_e2e": population_architecture.parameter_count,
            "diffpop_synergy_flow": population_architecture.parameter_count,
            "diffpop_synergy_proposal": proposal_architecture.parameter_count,
            "diffpop_synergy_warm_start": warm_architecture.parameter_count,
        },
    }
