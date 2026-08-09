"""Deterministic training and evaluation utilities for conditional flow matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from .flow_matching import (
    ConditionalFlowConfig,
    FlowParameters,
    FlowSamplingOptions,
    conditional_flow_matching_loss,
    sample_conditional_flow,
    sample_uniform_torus,
)
from .geometry import translation_gauge_fixed_displacement


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FlowTrainingBatch:
    """One fixed-shape minibatch of reference ensembles and conditions."""

    target_coordinates: Array
    conditions: Array
    box: Array

    def validate(self) -> None:
        if self.target_coordinates.ndim != 4 or self.target_coordinates.shape[-1] != 2:
            raise ValueError(
                "target_coordinates must have shape (B, M, N, 2); "
                f"got {self.target_coordinates.shape}"
            )
        if self.conditions.ndim != 2:
            raise ValueError(f"conditions must have shape (B, C); got {self.conditions.shape}")
        if self.conditions.shape[0] != self.target_coordinates.shape[0]:
            raise ValueError("conditions and target_coordinates must share batch size")
        if self.box.shape != (2,):
            raise ValueError(f"box must have shape (2,), got {self.box.shape}")
        if self.target_coordinates.shape[-2] < 2:
            raise ValueError("at least two particles are required")

    def tree_flatten(self):
        return (self.target_coordinates, self.conditions, self.box), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        del auxiliary_data
        return cls(*children)


@dataclass(frozen=True)
class FlowAdamOptions:
    """Adam settings for flow-matching pretraining."""

    learning_rate: float = 2e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: float = 5.0
    weight_decay: float = 0.0
    jit_step: bool = True

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")


@dataclass(frozen=True)
class FlowTrainingResult:
    parameters: FlowParameters
    history: Mapping[str, Array]
    final_loss: Array
    final_metrics: Mapping[str, Array]


def subset_flow_batch(batch: FlowTrainingBatch, indices: Array) -> FlowTrainingBatch:
    """Select examples without altering the shared periodic box."""
    batch.validate()
    return FlowTrainingBatch(
        target_coordinates=batch.target_coordinates[indices],
        conditions=batch.conditions[indices],
        box=batch.box,
    )


def _tree_global_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))


def _particle_permutations(key: Array, coordinates: Array) -> Array:
    """Draw independent particle permutations for each sample and replica."""
    batch_size, num_replicas, num_particles = coordinates.shape[:3]
    keys = jax.random.split(key, batch_size * num_replicas)
    return jax.vmap(
        lambda permutation_key: jax.random.permutation(permutation_key, num_particles)
    )(keys).reshape((batch_size, num_replicas, num_particles))


def _apply_particle_permutations(values: Array, permutations: Array) -> Array:
    """Apply a shared label permutation to arrays with a particle axis at ``-2``."""
    return jnp.take_along_axis(values, permutations[..., None], axis=-2)


def exhaustive_particle_match_targets(
    source: Array,
    targets: Array,
    box: Array,
) -> Array:
    """Reorder small target point sets by minimum gauge-fixed torus cost.

    This is a coupling choice, not a learned operation.  It removes arbitrary
    exchangeable label noise from the conditional flow target.  Exhaustive
    matching is deliberately restricted to at most eight particles; the exact
    homometric benchmark uses four, for which only 24 permutations are needed.
    """
    source = jnp.asarray(source)
    targets = jnp.asarray(targets, dtype=source.dtype)
    if source.shape != targets.shape or source.ndim != 4 or source.shape[-1] != 2:
        raise ValueError("source and targets must have the same shape (B, M, N, 2)")
    num_particles = source.shape[-2]
    if num_particles > 8:
        raise ValueError("exhaustive particle matching supports at most eight particles")
    permutation_table = jnp.asarray(
        tuple(permutations(range(num_particles))),
        dtype=jnp.int32,
    )
    box = jnp.asarray(box, dtype=source.dtype)

    def match_replica(replica_source: Array, replica_target: Array) -> Array:
        candidates = replica_target[permutation_table]
        displacement = jax.vmap(
            lambda candidate: translation_gauge_fixed_displacement(
                replica_source, candidate, box
            )
        )(candidates)
        normalized = displacement / box
        costs = jnp.mean(normalized * normalized, axis=(-2, -1))
        return candidates[jnp.argmin(costs)]

    return jax.vmap(jax.vmap(match_replica))(source, targets)


def stochastic_flow_matching_objective(
    parameters: FlowParameters,
    batch: FlowTrainingBatch,
    key: Array,
    config: ConditionalFlowConfig,
) -> tuple[Array, dict[str, Array]]:
    """Sample path couplings and evaluate a minibatch flow-matching loss."""
    batch.validate()
    source_key, latent_key, time_key, permutation_key = jax.random.split(key, 4)
    targets = batch.target_coordinates
    source = sample_uniform_torus(
        source_key,
        targets.shape,
        batch.box,
        dtype=targets.dtype,
    )
    node_latents = jax.random.normal(
        latent_key,
        targets.shape[:-1] + (config.network.latent_dim,),
        dtype=targets.dtype,
    )
    if config.particle_matching == "exhaustive":
        targets = exhaustive_particle_match_targets(source, targets, batch.box)
    # Exchangeability augmentation must preserve the source-target coupling.
    # Permuting the target alone would average over incompatible matchings and
    # collapse the conditional vector field toward zero.
    permutations = _particle_permutations(permutation_key, targets)
    source = _apply_particle_permutations(source, permutations)
    targets = _apply_particle_permutations(targets, permutations)
    node_latents = _apply_particle_permutations(node_latents, permutations)
    times = jax.random.uniform(
        time_key,
        (targets.shape[0],),
        minval=0.0,
        maxval=1.0,
        dtype=targets.dtype,
    )

    losses, metrics = jax.vmap(
        lambda sample_source, sample_target, sample_latents, condition, time: (
            conditional_flow_matching_loss(
                parameters,
                sample_source,
                sample_target,
                sample_latents,
                condition,
                time,
                batch.box,
                config,
            )
        )
    )(source, targets, node_latents, batch.conditions, times)
    mean_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return jnp.mean(losses), {**mean_metrics, "loss_std": jnp.std(losses)}


def fixed_flow_matching_objective(
    parameters: FlowParameters,
    batch: FlowTrainingBatch,
    key: Array,
    config: ConditionalFlowConfig,
) -> tuple[Array, dict[str, Array]]:
    """Alias emphasizing that a fixed key makes evaluation reproducible."""
    return stochastic_flow_matching_objective(parameters, batch, key, config)


def _adam_update(
    parameters: FlowParameters,
    gradients: FlowParameters,
    first_moment: FlowParameters,
    second_moment: FlowParameters,
    step: int,
    options: FlowAdamOptions,
) -> tuple[FlowParameters, FlowParameters, FlowParameters, Array, Array]:
    gradient_norm = _tree_global_norm(gradients)
    clip_factor = jnp.minimum(
        1.0,
        jnp.asarray(options.gradient_clip_norm, dtype=gradient_norm.dtype)
        / jnp.maximum(gradient_norm, 1e-15),
    )
    gradients = jax.tree_util.tree_map(lambda value: value * clip_factor, gradients)
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
    return (
        parameters,
        first_moment,
        second_moment,
        gradient_norm,
        _tree_global_norm(updates),
    )


def train_conditional_flow(
    initial_parameters: FlowParameters,
    minibatches: Sequence[FlowTrainingBatch],
    key: Array,
    config: ConditionalFlowConfig,
    options: FlowAdamOptions | None = None,
) -> FlowTrainingResult:
    """Train over a predetermined sequence of fixed-shape minibatches."""
    if options is None:
        options = FlowAdamOptions()
    config.validate()
    options.validate()
    if not minibatches:
        raise ValueError("minibatches must be nonempty")
    for batch in minibatches:
        batch.validate()

    parameters = initial_parameters
    first_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    second_moment = jax.tree_util.tree_map(jnp.zeros_like, parameters)

    def value_and_grad(model_parameters, batch, step_key):
        return jax.value_and_grad(stochastic_flow_matching_objective, has_aux=True)(
            model_parameters, batch, step_key, config
        )

    step_fn = jax.jit(value_and_grad) if options.jit_step else value_and_grad
    keys = jax.random.split(key, len(minibatches))
    history: dict[str, list[Array]] = {
        "loss": [],
        "velocity_rmse": [],
        "target_velocity_rms": [],
        "predicted_velocity_rms": [],
        "mean_velocity_norm": [],
        "loss_std": [],
        "gradient_norm": [],
        "update_norm": [],
    }
    final_metrics: Mapping[str, Array] = {}
    final_loss = jnp.asarray(jnp.nan)
    for step, (batch, step_key) in enumerate(zip(minibatches, keys), start=1):
        (loss, metrics), gradients = step_fn(parameters, batch, step_key)
        parameters, first_moment, second_moment, gradient_norm, update_norm = _adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            step,
            options,
        )
        final_loss = loss
        final_metrics = metrics
        history["loss"].append(loss)
        for name in (
            "velocity_rmse",
            "target_velocity_rms",
            "predicted_velocity_rms",
            "mean_velocity_norm",
            "loss_std",
        ):
            history[name].append(metrics[name])
        history["gradient_norm"].append(gradient_norm)
        history["update_norm"].append(update_norm)

    stacked_history = {name: jnp.stack(values) for name, values in history.items()}
    return FlowTrainingResult(
        parameters=parameters,
        history=stacked_history,
        final_loss=final_loss,
        final_metrics=final_metrics,
    )


def sample_flow_conditions(
    parameters: FlowParameters,
    key: Array,
    conditions: Array,
    *,
    num_samples_per_condition: int,
    num_replicas: int,
    num_particles: int,
    box: Array,
    config: ConditionalFlowConfig,
    sampling_options: FlowSamplingOptions,
    dtype: jnp.dtype,
) -> Array:
    """Return samples with shape ``(B, K, M, N, 2)``."""
    conditions = jnp.asarray(conditions, dtype=dtype)
    if conditions.ndim != 2:
        raise ValueError("conditions must have shape (B, C)")
    if num_samples_per_condition < 1:
        raise ValueError("num_samples_per_condition must be positive")
    keys = jax.random.split(key, conditions.shape[0] * num_samples_per_condition)
    keys = keys.reshape((conditions.shape[0], num_samples_per_condition, 2))

    def sample_one(sample_key, condition):
        return sample_conditional_flow(
            parameters,
            sample_key,
            condition,
            num_replicas=num_replicas,
            num_particles=num_particles,
            box=box,
            config=config,
            options=sampling_options,
            dtype=dtype,
        )

    return jax.vmap(
        lambda condition_keys, condition: jax.vmap(
            lambda sample_key: sample_one(sample_key, condition)
        )(condition_keys)
    )(keys, conditions)


def sample_flow_conditions_chunked(
    parameters: FlowParameters,
    key: Array,
    conditions: Array,
    *,
    num_samples_per_condition: int,
    chunk_size: int,
    num_replicas: int,
    num_particles: int,
    box: Array,
    config: ConditionalFlowConfig,
    sampling_options: FlowSamplingOptions,
    dtype: jnp.dtype,
) -> Array:
    """Sample in fixed-size chunks to bound XLA compilation and memory use.

    The output is identical in shape to :func:`sample_flow_conditions`.  A
    divisible sample count is required so every chunk reuses one compiled
    executable without padding or dropping random draws.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if num_samples_per_condition % chunk_size != 0:
        raise ValueError("num_samples_per_condition must be divisible by chunk_size")
    num_chunks = num_samples_per_condition // chunk_size
    chunk_keys = jax.random.split(key, num_chunks)
    chunks = [
        sample_flow_conditions(
            parameters,
            chunk_key,
            conditions,
            num_samples_per_condition=chunk_size,
            num_replicas=num_replicas,
            num_particles=num_particles,
            box=box,
            config=config,
            sampling_options=sampling_options,
            dtype=dtype,
        )
        for chunk_key in chunk_keys
    ]
    return jnp.concatenate(chunks, axis=1)


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


def flow_parameter_directional_derivative_sweep(
    parameters: FlowParameters,
    batch: FlowTrainingBatch,
    objective_key: Array,
    direction_key: Array,
    config: ConditionalFlowConfig,
    epsilons: Sequence[float] = (3e-3, 1e-3, 3e-4),
    *,
    jit_objective: bool = True,
) -> dict[str, Array]:
    """Check one fixed-randomness parameter directional derivative."""
    leaves, structure = jax.tree_util.tree_flatten(parameters)
    if not leaves:
        raise ValueError("parameters must contain array leaves")
    keys = jax.random.split(direction_key, len(leaves))
    direction = jax.tree_util.tree_unflatten(
        structure,
        [
            jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
            for key, leaf in zip(keys, leaves)
        ],
    )
    norm = _tree_global_norm(direction)
    direction = jax.tree_util.tree_map(lambda value: value / norm, direction)

    def scalar_objective(model_parameters):
        return fixed_flow_matching_objective(
            model_parameters, batch, objective_key, config
        )[0]

    scalar_fn = jax.jit(scalar_objective) if jit_objective else scalar_objective
    gradient_fn = jax.jit(jax.grad(scalar_objective)) if jit_objective else jax.grad(
        scalar_objective
    )
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
