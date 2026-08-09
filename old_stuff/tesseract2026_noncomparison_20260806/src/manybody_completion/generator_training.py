"""End-to-end training utilities for the native equivariant generator.

The implementation intentionally avoids optimizer-framework dependencies. A
small explicit Adam optimizer keeps parameter updates auditable and limits the
host environment to JAX. The objective supports the four methodology ablations
through an explicit stage-routing contract rather than an implicit or
straight-through gradient convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .ablation import (
    AblationMode,
    AblationSpec,
    CompletionStage,
    get_ablation_spec,
    stage_key,
)
from .composition import CompletionOptions, periodic_correction_mse, run_local_completion
from .energies import soft_repulsive_energy_per_configuration
from .generator import (
    EquivariantGeneratorConfig,
    GeneratorParameters,
    apply_equivariant_generator,
)
from .observables import PairBasis, ensemble_pair_moments
from .projection import project_ensemble_moments
from .relaxation import relax_proximal


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GeneratorBatch:
    """Fixed latent inputs and reduced-statistic targets for one training batch."""

    anchor_coordinates: Array
    node_latents: Array
    conditions: Array
    target_moments: Array
    box: Array
    basis: PairBasis
    moment_scales: Array
    basis_mask: Array

    def validate(self, config: EquivariantGeneratorConfig) -> None:
        if self.anchor_coordinates.ndim != 4 or self.anchor_coordinates.shape[-1] != 2:
            raise ValueError(
                "anchor_coordinates must have shape (B, M, N, 2); "
                f"got {self.anchor_coordinates.shape}"
            )
        expected_latent_shape = self.anchor_coordinates.shape[:-1] + (config.latent_dim,)
        if self.node_latents.shape != expected_latent_shape:
            raise ValueError(
                f"node_latents must have shape {expected_latent_shape}; "
                f"got {self.node_latents.shape}"
            )
        batch_size = self.anchor_coordinates.shape[0]
        if self.conditions.ndim != 2 or self.conditions.shape[0] != batch_size:
            raise ValueError("conditions must have shape (B, C)")
        if self.target_moments.ndim != 2 or self.target_moments.shape[0] != batch_size:
            raise ValueError("target_moments must have shape (B, R)")
        if self.target_moments.shape[1] != self.basis.centers.shape[0]:
            raise ValueError("target moment dimension must match pair basis size")
        if self.box.shape != (2,):
            raise ValueError("box must have shape (2,)")
        if self.moment_scales.shape != (self.target_moments.shape[1],):
            raise ValueError("moment_scales must have shape (R,)")
        if self.basis_mask.shape != (self.target_moments.shape[1],):
            raise ValueError("basis_mask must have shape (R,)")

    def tree_flatten(self):
        """Represent the batch as a JAX pytree for compiled minibatch steps."""
        children = (
            self.anchor_coordinates,
            self.node_latents,
            self.conditions,
            self.target_moments,
            self.box,
            self.basis.centers,
            self.basis.widths,
            self.moment_scales,
            self.basis_mask,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        (
            anchor_coordinates,
            node_latents,
            conditions,
            target_moments,
            box,
            basis_centers,
            basis_widths,
            moment_scales,
            basis_mask,
        ) = children
        return cls(
            anchor_coordinates=anchor_coordinates,
            node_latents=node_latents,
            conditions=conditions,
            target_moments=target_moments,
            box=box,
            basis=PairBasis(centers=basis_centers, widths=basis_widths),
            moment_scales=moment_scales,
            basis_mask=basis_mask,
        )

    def validate_numerics(self) -> None:
        """Validate array values outside JAX transformations."""
        moment_scales = np.asarray(jax.device_get(self.moment_scales))
        basis_mask = np.asarray(jax.device_get(self.basis_mask))
        if not np.all(np.isfinite(moment_scales)) or np.any(moment_scales <= 0):
            raise ValueError("moment_scales must be finite and strictly positive")
        if not np.all(np.isfinite(basis_mask)) or np.any((basis_mask < 0) | (basis_mask > 1)):
            raise ValueError("basis_mask entries must be finite and lie in [0, 1]")


def subset_generator_batch(batch: GeneratorBatch, indices: Array | np.ndarray) -> GeneratorBatch:
    """Select samples while preserving the shared geometry and basis metadata."""
    indices = jnp.asarray(indices, dtype=jnp.int32)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("indices must be a nonempty one-dimensional array")
    return GeneratorBatch(
        anchor_coordinates=batch.anchor_coordinates[indices],
        node_latents=batch.node_latents[indices],
        conditions=batch.conditions[indices],
        target_moments=batch.target_moments[indices],
        box=batch.box,
        basis=batch.basis,
        moment_scales=batch.moment_scales,
        basis_mask=batch.basis_mask,
    )


@dataclass(frozen=True)
class GeneratorObjectiveWeights:
    """Weights for the staged native-generator objective."""

    observed: float = 1000.0
    physical: float = 1.0
    correction: float = 1000.0
    preprojection_observed: float = 0.0

    def validate(self) -> None:
        values = (
            self.observed,
            self.physical,
            self.correction,
            self.preprojection_observed,
        )
        if any(value < 0 for value in values):
            raise ValueError("objective weights must be nonnegative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one objective weight must be positive")


@dataclass(frozen=True)
class AdamOptions:
    """Explicit Adam settings for deterministic smoke training."""

    num_steps: int = 20
    learning_rate: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    jit_objective: bool = True

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True)
class GeneratorTrainingResult:
    """Trained parameters and full optimization history."""

    parameters: GeneratorParameters
    final_loss: Array
    final_metrics: Mapping[str, Array]
    history: Mapping[str, Array]


GENERATOR_METRIC_NAMES = (
    "observed_loss",
    "preprojection_loss",
    "physical_loss",
    "correction_loss",
    "moment_error_initial",
    "moment_error_relaxed",
    "moment_error_projected",
    "moment_error_training",
    "moment_error_serving",
    "physical_energy",
    "physical_energy_initial",
    "physical_energy_relaxed",
    "physical_energy_projected",
    "physical_energy_training",
    "physical_energy_serving",
    "training_correction_rms",
    "serving_correction_rms",
    "total_correction_rms",
    "relaxation_displacement",
    "projection_correction",
    "relaxation_converged",
    "projection_converged",
    "projection_rank_deficient",
    "loss_std",
)


def _whitened_error(moments: Array, target: Array, scales: Array, mask: Array) -> Array:
    return mask * (moments - target) / scales


def _mean_physical_energy(
    coordinates: Array,
    box: Array,
    completion_options: CompletionOptions,
) -> Array:
    return jnp.mean(
        soft_repulsive_energy_per_configuration(
            coordinates,
            box,
            completion_options.physical.r0,
            completion_options.physical.kappa,
        )
    )


def _stage_value(stages: Mapping[str, object], prefix: str, stage: CompletionStage) -> Array:
    value = stages[stage_key(prefix, stage)]
    return jnp.asarray(value)


def _single_sample_ablation_objective(
    parameters: GeneratorParameters,
    anchor_coordinates: Array,
    node_latents: Array,
    condition: Array,
    target_moments: Array,
    *,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    basis_mask: Array,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
    weights: GeneratorObjectiveWeights,
    ablation: AblationSpec,
) -> tuple[Array, dict[str, Array]]:
    generated = apply_equivariant_generator(
        parameters,
        anchor_coordinates,
        node_latents,
        condition,
        box,
        generator_config,
    )
    stages = run_local_completion(
        initial_coordinates=generated,
        target_moments=target_moments,
        box=box,
        basis=basis,
        moment_scales=moment_scales,
        basis_mask=basis_mask,
        options=completion_options,
    )

    coordinates = {
        stage: _stage_value(stages, "coordinates", stage) for stage in CompletionStage
    }
    moments = {stage: _stage_value(stages, "moments", stage) for stage in CompletionStage}
    errors = {
        stage: _whitened_error(
            moments[stage], target_moments, moment_scales, basis_mask
        )
        for stage in CompletionStage
    }
    physical_energies = {
        stage: _mean_physical_energy(coordinates[stage], box, completion_options)
        for stage in CompletionStage
    }

    training_stage = ablation.training_stage
    serving_stage = ablation.serving_stage
    observed_loss = weights.observed * jnp.sum(errors[training_stage] ** 2)
    preprojection_loss = weights.preprojection_observed * jnp.sum(
        errors[CompletionStage.INITIAL] ** 2
    )
    physical_loss = weights.physical * physical_energies[training_stage]
    training_correction_mse = periodic_correction_mse(
        coordinates[training_stage],
        coordinates[CompletionStage.INITIAL],
        box,
    )
    correction_loss = weights.correction * training_correction_mse
    loss = observed_loss + preprojection_loss + physical_loss + correction_loss

    serving_correction_mse = periodic_correction_mse(
        coordinates[serving_stage],
        coordinates[CompletionStage.INITIAL],
        box,
    )
    total_correction_mse = periodic_correction_mse(
        coordinates[CompletionStage.PROJECTED],
        coordinates[CompletionStage.INITIAL],
        box,
    )
    relaxation_diagnostics = stages["relaxation"]
    projection_diagnostics = stages["projection"]
    metrics = {
        "observed_loss": observed_loss,
        "preprojection_loss": preprojection_loss,
        "physical_loss": physical_loss,
        "correction_loss": correction_loss,
        "moment_error_initial": jnp.linalg.norm(errors[CompletionStage.INITIAL]),
        "moment_error_relaxed": jnp.linalg.norm(errors[CompletionStage.RELAXED]),
        "moment_error_projected": jnp.linalg.norm(errors[CompletionStage.PROJECTED]),
        "moment_error_training": jnp.linalg.norm(errors[training_stage]),
        "moment_error_serving": jnp.linalg.norm(errors[serving_stage]),
        # Backward-compatible alias: the energy used by the current objective.
        "physical_energy": physical_energies[training_stage],
        "physical_energy_initial": physical_energies[CompletionStage.INITIAL],
        "physical_energy_relaxed": physical_energies[CompletionStage.RELAXED],
        "physical_energy_projected": physical_energies[CompletionStage.PROJECTED],
        "physical_energy_training": physical_energies[training_stage],
        "physical_energy_serving": physical_energies[serving_stage],
        "training_correction_rms": jnp.sqrt(training_correction_mse),
        "serving_correction_rms": jnp.sqrt(serving_correction_mse),
        "total_correction_rms": jnp.sqrt(total_correction_mse),
        "relaxation_displacement": relaxation_diagnostics["prox_displacement"],
        "projection_correction": projection_diagnostics["correction_norm"],
        "relaxation_converged": relaxation_diagnostics["converged"].astype(
            generated.dtype
        ),
        "projection_converged": projection_diagnostics["converged"].astype(
            generated.dtype
        ),
        "projection_rank_deficient": projection_diagnostics["rank_deficient"].astype(
            generated.dtype
        ),
    }
    return loss, metrics




def _single_sample_training_objective(
    parameters: GeneratorParameters,
    anchor_coordinates: Array,
    node_latents: Array,
    condition: Array,
    target_moments: Array,
    *,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    basis_mask: Array,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
    weights: GeneratorObjectiveWeights,
    ablation: AblationSpec,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate only the solver stages needed by the selected training path."""
    generated = apply_equivariant_generator(
        parameters,
        anchor_coordinates,
        node_latents,
        condition,
        box,
        generator_config,
    )
    initial_moments = ensemble_pair_moments(generated, box, basis)
    initial_error = _whitened_error(
        initial_moments, target_moments, moment_scales, basis_mask
    )
    initial_energy = _mean_physical_energy(generated, box, completion_options)
    dtype = generated.dtype
    relaxation_used = ablation.requires_training_relaxation
    projection_used = ablation.requires_training_projection

    if ablation.training_stage is CompletionStage.INITIAL:
        training_coordinates = generated
        training_moments = initial_moments
        relaxation_converged = jnp.asarray(True)
        projection_converged = jnp.asarray(True)
        projection_rank_deficient = jnp.asarray(False)
    elif ablation.training_stage is CompletionStage.RELAXED:
        training_coordinates, relaxation_diagnostics = relax_proximal(
            initial_coordinates=generated,
            box=box,
            r0=completion_options.physical.r0,
            kappa=completion_options.physical.kappa,
            prox_strength=completion_options.physical.prox_strength,
            options=completion_options.relaxation,
        )
        training_moments = ensemble_pair_moments(training_coordinates, box, basis)
        relaxation_converged = relaxation_diagnostics["converged"]
        projection_converged = jnp.asarray(True)
        projection_rank_deficient = jnp.asarray(False)
    else:
        relaxed_coordinates, relaxation_diagnostics = relax_proximal(
            initial_coordinates=generated,
            box=box,
            r0=completion_options.physical.r0,
            kappa=completion_options.physical.kappa,
            prox_strength=completion_options.physical.prox_strength,
            options=completion_options.relaxation,
        )
        training_coordinates, projection_diagnostics = project_ensemble_moments(
            coordinates=relaxed_coordinates,
            target_moments=target_moments,
            box=box,
            basis=basis,
            moment_scales=moment_scales,
            basis_mask=basis_mask,
            options=completion_options.projection,
        )
        training_moments = ensemble_pair_moments(training_coordinates, box, basis)
        relaxation_converged = relaxation_diagnostics["converged"]
        projection_converged = projection_diagnostics["converged"]
        projection_rank_deficient = projection_diagnostics["rank_deficient"]

    training_error = _whitened_error(
        training_moments, target_moments, moment_scales, basis_mask
    )
    training_energy = _mean_physical_energy(
        training_coordinates, box, completion_options
    )
    correction_mse = periodic_correction_mse(training_coordinates, generated, box)
    observed_loss = weights.observed * jnp.sum(training_error**2)
    preprojection_loss = weights.preprojection_observed * jnp.sum(initial_error**2)
    physical_loss = weights.physical * training_energy
    correction_loss = weights.correction * correction_mse
    loss = observed_loss + preprojection_loss + physical_loss + correction_loss
    metrics = {
        "observed_loss": observed_loss,
        "preprojection_loss": preprojection_loss,
        "physical_loss": physical_loss,
        "correction_loss": correction_loss,
        "moment_error_initial": jnp.linalg.norm(initial_error),
        "moment_error_training": jnp.linalg.norm(training_error),
        "physical_energy_initial": initial_energy,
        "physical_energy_training": training_energy,
        "training_correction_rms": jnp.sqrt(correction_mse),
        "training_relaxation_used": jnp.asarray(relaxation_used, dtype=dtype),
        "training_projection_used": jnp.asarray(projection_used, dtype=dtype),
        "relaxation_converged": relaxation_converged.astype(dtype),
        "projection_converged": projection_converged.astype(dtype),
        "projection_rank_deficient": projection_rank_deficient.astype(dtype),
    }
    return loss, metrics


def ablation_training_objective(
    parameters: GeneratorParameters,
    batch: GeneratorBatch,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
    weights: GeneratorObjectiveWeights | None = None,
    mode: AblationMode | str = AblationMode.FULL_E2E,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate a compute-minimal objective for one ablation mode.

    Base and Post-hoc execute no scientific solver during training. Relax-E2E
    executes only relaxation. Full-E2E executes both stages. Use
    :func:`ablation_generator_objective` for complete post-training diagnostics.
    """
    if weights is None:
        weights = GeneratorObjectiveWeights()
    weights.validate()
    batch.validate(generator_config)
    ablation = get_ablation_spec(mode)
    sample_fn = lambda anchors, latents, condition, target: (
        _single_sample_training_objective(
            parameters,
            anchors,
            latents,
            condition,
            target,
            box=batch.box,
            basis=batch.basis,
            moment_scales=batch.moment_scales,
            basis_mask=batch.basis_mask,
            generator_config=generator_config,
            completion_options=completion_options,
            weights=weights,
            ablation=ablation,
        )
    )
    losses, metrics = jax.vmap(sample_fn)(
        batch.anchor_coordinates,
        batch.node_latents,
        batch.conditions,
        batch.target_moments,
    )
    mean_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    mean_metrics = {**mean_metrics, "loss_std": jnp.std(losses)}
    return jnp.mean(losses), jax.tree_util.tree_map(jax.lax.stop_gradient, mean_metrics)

def ablation_generator_objective(
    parameters: GeneratorParameters,
    batch: GeneratorBatch,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
    weights: GeneratorObjectiveWeights | None = None,
    mode: AblationMode | str = AblationMode.FULL_E2E,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate one ablation objective on a complete fixed batch.

    Solver outputs not selected as the training stage are used only in detached
    diagnostics. In particular, ``post_hoc`` has the same optimization objective
    as ``base`` and differs only in its serving stage.
    """
    if weights is None:
        weights = GeneratorObjectiveWeights()
    weights.validate()
    batch.validate(generator_config)
    ablation = get_ablation_spec(mode)

    sample_fn = lambda anchors, latents, condition, target: (
        _single_sample_ablation_objective(
            parameters,
            anchors,
            latents,
            condition,
            target,
            box=batch.box,
            basis=batch.basis,
            moment_scales=batch.moment_scales,
            basis_mask=batch.basis_mask,
            generator_config=generator_config,
            completion_options=completion_options,
            weights=weights,
            ablation=ablation,
        )
    )
    losses, metrics = jax.vmap(sample_fn)(
        batch.anchor_coordinates,
        batch.node_latents,
        batch.conditions,
        batch.target_moments,
    )
    mean_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    mean_metrics = {**mean_metrics, "loss_std": jnp.std(losses)}
    return jnp.mean(losses), jax.tree_util.tree_map(jax.lax.stop_gradient, mean_metrics)


def local_generator_objective(
    parameters: GeneratorParameters,
    batch: GeneratorBatch,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
    weights: GeneratorObjectiveWeights | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Backward-compatible alias for the full end-to-end objective."""
    return ablation_generator_objective(
        parameters,
        batch,
        generator_config,
        completion_options,
        weights,
        mode=AblationMode.FULL_E2E,
    )


def generate_batch_coordinates(
    parameters: GeneratorParameters,
    batch: GeneratorBatch,
    generator_config: EquivariantGeneratorConfig,
) -> Array:
    """Generate every condition in ``batch`` with shared parameters."""
    batch.validate(generator_config)
    return jax.vmap(
        lambda anchors, latents, condition: apply_equivariant_generator(
            parameters,
            anchors,
            latents,
            condition,
            batch.box,
            generator_config,
        )
    )(batch.anchor_coordinates, batch.node_latents, batch.conditions)


def evaluate_generator_completion(
    parameters: GeneratorParameters,
    batch: GeneratorBatch,
    generator_config: EquivariantGeneratorConfig,
    completion_options: CompletionOptions,
) -> tuple[Array, Mapping[str, object]]:
    """Run all completion stages for reporting, independent of training mode."""
    generated = generate_batch_coordinates(parameters, batch, generator_config)
    stages = jax.vmap(
        lambda coordinates, target: run_local_completion(
            coordinates,
            target,
            batch.box,
            batch.basis,
            batch.moment_scales,
            batch.basis_mask,
            completion_options,
        )
    )(generated, batch.target_moments)
    return generated, stages


def _tree_global_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))


def _tree_dot(left: Any, right: Any) -> Array:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        raise ValueError("pytrees must have identical structure")
    return sum(jnp.vdot(a, b) for a, b in zip(left_leaves, right_leaves))


def _tree_add_scaled(tree: Any, direction: Any, scale: Array | float) -> Any:
    return jax.tree_util.tree_map(
        lambda value, delta: value + scale * delta,
        tree,
        direction,
    )


def parameter_directional_derivative_sweep(
    objective: Callable[[GeneratorParameters], tuple[Array, Mapping[str, Array]]],
    parameters: GeneratorParameters,
    direction_key: Array,
    epsilons: Sequence[float] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
    *,
    jit_objective: bool = True,
) -> dict[str, Array]:
    """Check one normalized parameter-space directional derivative."""
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    if not leaves:
        raise ValueError("parameters must contain at least one array leaf")
    keys = jax.random.split(direction_key, len(leaves))
    direction_leaves = [
        jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
        for key, leaf in zip(keys, leaves)
    ]
    direction = jax.tree_util.tree_unflatten(structure, direction_leaves)
    direction_norm = _tree_global_norm(direction)
    direction = jax.tree_util.tree_map(lambda value: value / direction_norm, direction)

    def scalar_only(model_parameters: GeneratorParameters) -> Array:
        return objective(model_parameters)[0]

    scalar_fn = jax.jit(scalar_only) if jit_objective else scalar_only
    gradient_fn = jax.jit(jax.grad(scalar_only)) if jit_objective else jax.grad(scalar_only)
    gradient = gradient_fn(parameters)
    autodiff = _tree_dot(gradient, direction)
    epsilon_array = jnp.asarray(tuple(epsilons), dtype=leaves[0].dtype)
    finite_differences = jnp.stack(
        [
            (
                scalar_fn(_tree_add_scaled(parameters, direction, epsilon))
                - scalar_fn(_tree_add_scaled(parameters, direction, -epsilon))
            )
            / (2.0 * epsilon)
            for epsilon in epsilon_array
        ]
    )
    absolute_errors = jnp.abs(finite_differences - autodiff)
    relative_errors = absolute_errors / jnp.maximum(jnp.abs(autodiff), 1e-15)
    return {
        "autodiff": autodiff,
        "gradient_norm": _tree_global_norm(gradient),
        "epsilons": epsilon_array,
        "finite_differences": finite_differences,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "best_absolute_error": jnp.min(absolute_errors),
        "best_relative_error": jnp.min(relative_errors),
    }


def _adam_update(
    parameters: GeneratorParameters,
    gradients: GeneratorParameters,
    first_moment: GeneratorParameters,
    second_moment: GeneratorParameters,
    step: int,
    options: AdamOptions,
) -> tuple[GeneratorParameters, GeneratorParameters, GeneratorParameters, Array, Array]:
    gradient_norm = _tree_global_norm(gradients)
    clip_factor = jnp.minimum(
        1.0,
        jnp.asarray(options.gradient_clip_norm, dtype=gradient_norm.dtype)
        / jnp.maximum(gradient_norm, 1e-15),
    )
    gradients = jax.tree_util.tree_map(lambda gradient: gradient * clip_factor, gradients)
    if options.weight_decay > 0:
        gradients = jax.tree_util.tree_map(
            lambda gradient, parameter: gradient + options.weight_decay * parameter,
            gradients,
            parameters,
        )

    first_moment = jax.tree_util.tree_map(
        lambda moment, gradient: options.beta1 * moment + (1.0 - options.beta1) * gradient,
        first_moment,
        gradients,
    )
    second_moment = jax.tree_util.tree_map(
        lambda moment, gradient: options.beta2 * moment
        + (1.0 - options.beta2) * gradient * gradient,
        second_moment,
        gradients,
    )
    bias1 = 1.0 - options.beta1**step
    bias2 = 1.0 - options.beta2**step
    updates = jax.tree_util.tree_map(
        lambda first, second: options.learning_rate
        * (first / bias1)
        / (jnp.sqrt(second / bias2) + options.epsilon),
        first_moment,
        second_moment,
    )
    parameters = jax.tree_util.tree_map(
        lambda parameter, update: parameter - update,
        parameters,
        updates,
    )
    update_norm = _tree_global_norm(updates)
    return parameters, first_moment, second_moment, gradient_norm, update_norm


def train_equivariant_generator(
    objective: Callable[[GeneratorParameters], tuple[Array, Mapping[str, Array]]],
    initial_parameters: GeneratorParameters,
    options: AdamOptions | None = None,
) -> GeneratorTrainingResult:
    """Train the native generator with deterministic full-batch Adam."""
    if options is None:
        options = AdamOptions()
    options.validate()

    parameters = initial_parameters
    first_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    second_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    evaluator = jax.jit(value_and_grad) if options.jit_objective else value_and_grad
    history: dict[str, list[Array]] | None = None
    metric_names: tuple[str, ...] | None = None

    for step in range(1, options.num_steps + 1):
        (loss, metrics), gradients = evaluator(parameters)
        current_metric_names = tuple(metrics.keys())
        if history is None:
            metric_names = current_metric_names
            history = {
                "loss": [],
                "gradient_norm": [],
                "update_norm": [],
                **{name: [] for name in metric_names},
            }
        elif current_metric_names != metric_names:
            raise KeyError(
                "objective metric keys changed during training: "
                f"expected {metric_names}, got {current_metric_names}"
            )
        parameters, first_moment, second_moment, gradient_norm, update_norm = _adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            step,
            options,
        )
        history["loss"].append(loss)
        history["gradient_norm"].append(gradient_norm)
        history["update_norm"].append(update_norm)
        for name in metric_names:
            history[name].append(metrics[name])

    if history is None or metric_names is None:
        raise RuntimeError("training loop produced no optimization steps")
    (final_loss, final_metrics), final_gradients = evaluator(parameters)
    if tuple(final_metrics.keys()) != metric_names:
        raise KeyError("objective metric keys changed at final evaluation")
    history["loss"].append(final_loss)
    history["gradient_norm"].append(_tree_global_norm(final_gradients))
    history["update_norm"].append(jnp.asarray(0.0, dtype=final_loss.dtype))
    for name in metric_names:
        history[name].append(final_metrics[name])

    return GeneratorTrainingResult(
        parameters=parameters,
        final_loss=final_loss,
        final_metrics=final_metrics,
        history={name: jnp.stack(values) for name, values in history.items()},
    )


def train_equivariant_generator_minibatches(
    objective: Callable[[GeneratorParameters, GeneratorBatch], tuple[Array, Mapping[str, Array]]],
    initial_parameters: GeneratorParameters,
    minibatches: Sequence[GeneratorBatch],
    evaluation_batch: GeneratorBatch,
    options: AdamOptions | None = None,
) -> GeneratorTrainingResult:
    """Train with deterministic, fixed-shape minibatches and persistent Adam state.

    ``minibatches`` defines the complete update schedule.  Every batch must have
    the same array shapes so one compiled update function can be reused.  The
    final loss and metrics are evaluated on ``evaluation_batch`` rather than on
    the final optimization minibatch.
    """
    if options is None:
        options = AdamOptions(num_steps=len(minibatches))
    options.validate()
    if len(minibatches) != options.num_steps:
        raise ValueError(
            "options.num_steps must equal the number of supplied minibatches; "
            f"got {options.num_steps} and {len(minibatches)}"
        )
    if not minibatches:
        raise ValueError("at least one minibatch is required")
    expected_shape = minibatches[0].anchor_coordinates.shape
    for index, batch in enumerate(minibatches):
        if batch.anchor_coordinates.shape != expected_shape:
            raise ValueError(
                "all minibatches must have the same anchor shape; "
                f"batch 0 has {expected_shape}, batch {index} has "
                f"{batch.anchor_coordinates.shape}"
            )

    parameters = initial_parameters
    first_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    second_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    value_and_grad = jax.value_and_grad(objective, argnums=0, has_aux=True)
    evaluator = jax.jit(value_and_grad) if options.jit_objective else value_and_grad
    history: dict[str, list[Array]] | None = None
    metric_names: tuple[str, ...] | None = None

    for step, batch in enumerate(minibatches, start=1):
        (loss, metrics), gradients = evaluator(parameters, batch)
        current_metric_names = tuple(metrics.keys())
        if history is None:
            metric_names = current_metric_names
            history = {
                "loss": [],
                "gradient_norm": [],
                "update_norm": [],
                **{name: [] for name in metric_names},
            }
        elif current_metric_names != metric_names:
            raise KeyError(
                "objective metric keys changed during minibatch training: "
                f"expected {metric_names}, got {current_metric_names}"
            )
        parameters, first_moment, second_moment, gradient_norm, update_norm = _adam_update(
            parameters, gradients, first_moment, second_moment, step, options
        )
        history["loss"].append(loss)
        history["gradient_norm"].append(gradient_norm)
        history["update_norm"].append(update_norm)
        for name in metric_names:
            history[name].append(metrics[name])

    if history is None or metric_names is None:
        raise RuntimeError("minibatch training produced no optimization steps")
    # Evaluate aggregate train metrics without differentiating through the much
    # larger evaluation batch.  The final gradient norm is measured on the last
    # fixed-shape minibatch, reusing the already compiled update graph.
    value_evaluator = jax.jit(objective) if options.jit_objective else objective
    final_loss, final_metrics = value_evaluator(parameters, evaluation_batch)
    (_, _), final_gradients = evaluator(parameters, minibatches[-1])
    if tuple(final_metrics.keys()) != metric_names:
        raise KeyError("objective metric keys changed at final evaluation")
    history["loss"].append(final_loss)
    history["gradient_norm"].append(_tree_global_norm(final_gradients))
    history["update_norm"].append(jnp.asarray(0.0, dtype=final_loss.dtype))
    for name in metric_names:
        history[name].append(final_metrics[name])

    return GeneratorTrainingResult(
        parameters=parameters,
        final_loss=final_loss,
        final_metrics=final_metrics,
        history={name: jnp.stack(values) for name, values in history.items()},
    )
