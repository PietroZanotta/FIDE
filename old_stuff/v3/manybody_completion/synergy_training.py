"""Staged co-training of the flow prior and the two adaptive Tesseract components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .adaptive_components import (
    ProposalModel,
    WarmStartModel,
    exact_tilt_probabilities_jax,
    importance_ess_fraction_jax,
    proposal_probabilities_jax,
    warm_start_dual_jax,
)
from .flow import (
    FlowModel,
    integrate_flow,
    soft_atom_probabilities_jax,
    support_embeddings,
)
from .homometric import PopulationSupport

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AdaptivePretrainingResult:
    proposal_model: ProposalModel
    warm_start_model: WarmStartModel
    loss_trace: list[float]
    proposal_ess_trace: list[float]
    warm_rmse_trace: list[float]


@dataclass(frozen=True)
class SynergyTrainingResult:
    flow_model: FlowModel
    proposal_model: ProposalModel
    warm_start_model: WarmStartModel
    loss_trace: list[float]
    diagnostics_trace: list[dict[str, float]]
    gradient_norm_trace: list[float]


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_global_norm(tree) -> jax.Array:
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
    )


def _adam_update(
    parameters,
    gradients,
    first_moment,
    second_moment,
    iteration: int,
    learning_rates,
    gradient_clip: float,
):
    norm = _tree_global_norm(gradients)
    scale = jnp.minimum(1.0, gradient_clip / jnp.maximum(norm, 1e-12))
    gradients = jax.tree_util.tree_map(lambda value: value * scale, gradients)
    first_moment = jax.tree_util.tree_map(
        lambda old, value: 0.9 * old + 0.1 * value, first_moment, gradients
    )
    second_moment = jax.tree_util.tree_map(
        lambda old, value: 0.999 * old + 0.001 * jnp.square(value),
        second_moment,
        gradients,
    )
    first_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - 0.9**iteration), first_moment
    )
    second_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - 0.999**iteration), second_moment
    )
    parameters = jax.tree_util.tree_map(
        lambda parameter, first, second, learning_rate: parameter
        - learning_rate * first / (jnp.sqrt(second) + 1e-8),
        parameters,
        first_hat,
        second_hat,
        learning_rates,
    )
    return parameters, first_moment, second_moment, float(norm)


def _newton_dual_jax(
    prior: jax.Array,
    pair_values: jax.Array,
    target: jax.Array,
    initial_dual: jax.Array,
    *,
    iterations: int,
    ridge: float,
    damping: float,
    max_step: float,
) -> tuple[jax.Array, jax.Array]:
    dual = initial_dual
    residuals = []
    for _ in range(int(iterations)):
        conditioned = exact_tilt_probabilities_jax(prior, pair_values, dual)
        mean = jnp.sum(conditioned * pair_values)
        covariance = jnp.sum(conditioned * jnp.square(pair_values - mean))
        residual = mean - target
        residuals.append(residual)
        step = damping * residual / (covariance + ridge)
        dual = dual - jnp.clip(step, -max_step, max_step)
    return dual, jnp.stack(residuals) if residuals else jnp.zeros((0,), dtype=prior.dtype)


def pretrain_adaptive_components(
    proposal_model: ProposalModel,
    warm_start_model: WarmStartModel,
    support: PopulationSupport,
    prior_probabilities: np.ndarray,
    training_duals: Sequence[float],
    *,
    steps: int,
    learning_rate: float,
    proposal_ess_weight: float,
    warm_start_weight: float,
    gradient_clip: float,
) -> AdaptivePretrainingResult:
    """Teach the proposal and warm start from exact finite-support tilt examples.

    This stage does not change the flow prior.  It gives both Tesseracts a stable
    initialization before joint optimization starts.
    """
    prior = jnp.asarray(prior_probabilities, dtype=jnp.float64)
    prior = prior / jnp.sum(prior)
    pair = jnp.asarray(support.pair, dtype=jnp.float64)
    triplet = jnp.asarray(support.triplet, dtype=jnp.float64)
    labels = jnp.asarray(support.labels, dtype=jnp.float64)
    duals = jnp.asarray(training_duals, dtype=jnp.float64)
    exact_targets = jax.vmap(
        lambda dual: jnp.sum(exact_tilt_probabilities_jax(prior, pair, dual) * pair)
    )(duals)

    initial_parameters = {
        "proposal": proposal_model.parameters,
        "warm": warm_start_model.parameters,
    }

    def objective(parameters):
        losses = []
        ess_values = []
        warm_errors = []
        for dual, target in zip(duals, exact_targets):
            exact = exact_tilt_probabilities_jax(prior, pair, dual)
            proposal = proposal_probabilities_jax(
                parameters["proposal"],
                proposal_model.architecture,
                prior,
                pair,
                triplet,
                labels,
                dual,
                defensive_mixture=proposal_model.defensive_mixture,
                max_logit_correction=proposal_model.max_logit_correction,
            )
            ess_fraction = importance_ess_fraction_jax(exact, proposal)
            proposal_kl = jnp.sum(
                exact
                * (
                    jnp.log(jnp.maximum(exact, 1e-15))
                    - jnp.log(jnp.maximum(proposal, 1e-15))
                )
            )
            predicted_dual = warm_start_dual_jax(
                parameters["warm"],
                warm_start_model.architecture,
                prior,
                pair,
                triplet,
                labels,
                target,
                max_abs_dual=warm_start_model.max_abs_dual,
            )
            warm_error = predicted_dual - dual
            losses.append(
                proposal_kl
                - proposal_ess_weight * jnp.log(jnp.maximum(ess_fraction, 1e-12))
                + warm_start_weight * jnp.square(warm_error)
            )
            ess_values.append(ess_fraction)
            warm_errors.append(jnp.square(warm_error))
        return (
            jnp.mean(jnp.stack(losses)),
            (
                jnp.mean(jnp.stack(ess_values)),
                jnp.sqrt(jnp.mean(jnp.stack(warm_errors))),
            ),
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    parameters = initial_parameters
    first = _tree_zeros_like(parameters)
    second = _tree_zeros_like(parameters)
    learning_rates = jax.tree_util.tree_map(lambda _: float(learning_rate), parameters)
    loss_trace: list[float] = []
    ess_trace: list[float] = []
    warm_trace: list[float] = []
    for iteration in range(1, int(steps) + 1):
        (value, (mean_ess, warm_rmse)), gradients = value_and_grad(parameters)
        parameters, first, second, _ = _adam_update(
            parameters,
            gradients,
            first,
            second,
            iteration,
            learning_rates,
            float(gradient_clip),
        )
        loss_trace.append(float(value))
        ess_trace.append(float(mean_ess))
        warm_trace.append(float(warm_rmse))
    return AdaptivePretrainingResult(
        proposal_model=ProposalModel(
            proposal_model.architecture,
            parameters["proposal"],
            proposal_model.defensive_mixture,
            proposal_model.max_logit_correction,
        ),
        warm_start_model=WarmStartModel(
            warm_start_model.architecture,
            parameters["warm"],
            warm_start_model.max_abs_dual,
        ),
        loss_trace=loss_trace,
        proposal_ess_trace=ess_trace,
        warm_rmse_trace=warm_trace,
    )


def fine_tune_synergy_system(
    initial_flow: FlowModel,
    initial_proposal: ProposalModel,
    initial_warm_start: WarmStartModel,
    support: PopulationSupport,
    task_targets: Sequence[float],
    task_sample_indices: Sequence[np.ndarray],
    *,
    steps: int,
    flow_learning_rate: float,
    component_learning_rate: float,
    flow_particles: int,
    sampling_steps: int,
    assignment_temperature: float,
    limited_dual_iterations: int,
    reference_dual_iterations: int,
    dual_ridge: float,
    dual_damping: float,
    dual_max_step: float,
    conditional_score_weight: float,
    proposal_ess_weight: float,
    fresh_residual_weight: float,
    warm_start_weight: float,
    dual_path_weight: float,
    anchor_weight: float,
    proposal_reference_weight: float,
    seed: int,
    gradient_clip: float,
) -> SynergyTrainingResult:
    """Jointly optimize the flow, learned sampler proposal, and dual warm start.

    The loss deliberately represents a *finite-budget* computation.  The warm
    start is followed by only ``limited_dual_iterations`` Newton corrections.
    The proposal is rewarded for high expected importance ESS for the resulting
    tilt.  An independently interpreted population residual penalizes solutions
    that only look calibrated because of a favorable solver trajectory.
    """
    if initial_flow.architecture.condition_dim != 0:
        raise ValueError("the synergistic DiffPOP prior must be unconditional")
    if not task_targets or len(task_targets) != len(task_sample_indices):
        raise ValueError("task targets and samples must be non-empty and aligned")
    sample_lengths = {np.asarray(indices).size for indices in task_sample_indices}
    if len(sample_lengths) != 1:
        raise ValueError("all conditional tasks must contain the same sample count")

    rng = np.random.default_rng(int(seed))
    base_samples = jnp.asarray(
        rng.normal(
            size=(
                len(task_targets),
                int(flow_particles),
                initial_flow.architecture.state_dim,
            )
        ),
        dtype=jnp.float64,
    )
    embeddings = jnp.asarray(support_embeddings(support, initial_flow.label_scale))
    pair = jnp.asarray(support.pair, dtype=jnp.float64)
    triplet = jnp.asarray(support.triplet, dtype=jnp.float64)
    labels = jnp.asarray(support.labels, dtype=jnp.float64)
    targets = jnp.asarray(task_targets, dtype=jnp.float64)
    ids = jnp.asarray(np.stack(task_sample_indices), dtype=jnp.int32)

    def flow_priors(flow_parameters):
        outputs = []
        for task_index in range(len(task_targets)):
            generated = integrate_flow(
                flow_parameters,
                initial_flow.architecture,
                base_samples[task_index],
                steps=int(sampling_steps),
                method="heun",
            )
            prior, _ = soft_atom_probabilities_jax(
                generated, embeddings, float(assignment_temperature)
            )
            outputs.append(prior)
        return jnp.stack(outputs)

    anchor_priors = jax.lax.stop_gradient(flow_priors(initial_flow.parameters))
    parameters = {
        "flow": initial_flow.parameters,
        "proposal": initial_proposal.parameters,
        "warm": initial_warm_start.parameters,
    }

    def objective(current_parameters):
        priors = flow_priors(current_parameters["flow"])
        task_losses = []
        ess_values = []
        final_residuals = []
        warm_errors = []
        for task_index in range(len(task_targets)):
            prior = priors[task_index]
            target = targets[task_index]
            reference_dual, _ = _newton_dual_jax(
                prior,
                pair,
                target,
                jnp.asarray(0.0, dtype=prior.dtype),
                iterations=int(reference_dual_iterations),
                ridge=float(dual_ridge),
                damping=1.0,
                max_step=float(dual_max_step),
            )
            reference_dual = jax.lax.stop_gradient(reference_dual)
            initial_dual = warm_start_dual_jax(
                current_parameters["warm"],
                initial_warm_start.architecture,
                prior,
                pair,
                triplet,
                labels,
                target,
                max_abs_dual=initial_warm_start.max_abs_dual,
            )
            final_dual, residual_path = _newton_dual_jax(
                prior,
                pair,
                target,
                initial_dual,
                iterations=int(limited_dual_iterations),
                ridge=float(dual_ridge),
                damping=float(dual_damping),
                max_step=float(dual_max_step),
            )
            conditioned = exact_tilt_probabilities_jax(prior, pair, final_dual)
            conditioned_mean = jnp.sum(conditioned * pair)
            final_residual = conditioned_mean - target
            proposal = proposal_probabilities_jax(
                current_parameters["proposal"],
                initial_proposal.architecture,
                prior,
                pair,
                triplet,
                labels,
                final_dual,
                defensive_mixture=initial_proposal.defensive_mixture,
                max_logit_correction=initial_proposal.max_logit_correction,
            )
            expected_ess = importance_ess_fraction_jax(conditioned, proposal)
            reference_conditioned = exact_tilt_probabilities_jax(
                prior, pair, reference_dual
            )
            reference_proposal = proposal_probabilities_jax(
                current_parameters["proposal"],
                initial_proposal.architecture,
                prior,
                pair,
                triplet,
                labels,
                reference_dual,
                defensive_mixture=initial_proposal.defensive_mixture,
                max_logit_correction=initial_proposal.max_logit_correction,
            )
            proposal_reference_kl = jnp.sum(
                reference_conditioned
                * (
                    jnp.log(jnp.maximum(reference_conditioned, 1e-15))
                    - jnp.log(jnp.maximum(reference_proposal, 1e-15))
                )
            )
            log_conditioned = jnp.log(jnp.maximum(conditioned, 1e-15))
            conditional_score = -jnp.mean(log_conditioned[ids[task_index]])
            anchor = anchor_priors[task_index]
            anchor_kl = jnp.sum(
                anchor
                * (
                    jnp.log(jnp.maximum(anchor, 1e-15))
                    - jnp.log(jnp.maximum(prior, 1e-15))
                )
            )
            path_cost = (
                jnp.mean(jnp.square(residual_path))
                if residual_path.size
                else jnp.square(final_residual)
            )
            warm_error = initial_dual - reference_dual
            task_loss = (
                conditional_score_weight * conditional_score
                - proposal_ess_weight * jnp.log(jnp.maximum(expected_ess, 1e-12))
                + fresh_residual_weight * jnp.square(final_residual)
                + warm_start_weight * jnp.square(warm_error)
                + dual_path_weight * path_cost
                + anchor_weight * anchor_kl
                + proposal_reference_weight * proposal_reference_kl
            )
            task_losses.append(task_loss)
            ess_values.append(expected_ess)
            final_residuals.append(jnp.abs(final_residual))
            warm_errors.append(jnp.abs(warm_error))
        return (
            jnp.mean(jnp.stack(task_losses)),
            (
                jnp.mean(jnp.stack(ess_values)),
                jnp.mean(jnp.stack(final_residuals)),
                jnp.mean(jnp.stack(warm_errors)),
            ),
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    first = _tree_zeros_like(parameters)
    second = _tree_zeros_like(parameters)
    learning_rates = {
        "flow": jax.tree_util.tree_map(lambda _: float(flow_learning_rate), parameters["flow"]),
        "proposal": jax.tree_util.tree_map(
            lambda _: float(component_learning_rate), parameters["proposal"]
        ),
        "warm": jax.tree_util.tree_map(
            lambda _: float(component_learning_rate), parameters["warm"]
        ),
    }
    loss_trace: list[float] = []
    diagnostics_trace: list[dict[str, float]] = []
    gradient_norm_trace: list[float] = []
    for iteration in range(1, int(steps) + 1):
        (value, (mean_ess, mean_residual, mean_warm_error)), gradients = value_and_grad(
            parameters
        )
        parameters, first, second, gradient_norm = _adam_update(
            parameters,
            gradients,
            first,
            second,
            iteration,
            learning_rates,
            float(gradient_clip),
        )
        loss_trace.append(float(value))
        gradient_norm_trace.append(gradient_norm)
        diagnostics_trace.append(
            {
                "mean_expected_proposal_ess_fraction": float(mean_ess),
                "mean_limited_solver_residual": float(mean_residual),
                "mean_warm_start_absolute_error": float(mean_warm_error),
            }
        )

    return SynergyTrainingResult(
        flow_model=FlowModel(
            initial_flow.architecture, parameters["flow"], initial_flow.label_scale
        ),
        proposal_model=ProposalModel(
            initial_proposal.architecture,
            parameters["proposal"],
            initial_proposal.defensive_mixture,
            initial_proposal.max_logit_correction,
        ),
        warm_start_model=WarmStartModel(
            initial_warm_start.architecture,
            parameters["warm"],
            initial_warm_start.max_abs_dual,
        ),
        loss_trace=loss_trace,
        diagnostics_trace=diagnostics_trace,
        gradient_norm_trace=gradient_norm_trace,
    )
