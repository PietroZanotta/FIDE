"""Deterministic high-throughput pretraining and route-specific fine-tuning.

The public API is unchanged, but optimizer execution is fused into compiled
multi-update chunks using :mod:`manybody_completion.fast_training`.

Performance properties
----------------------
* Minibatch schedules and random keys are materialized once and transferred to
  the accelerator once.
* Loss evaluation, reverse-mode differentiation, global-norm clipping, and
  Adam are compiled into one update.
* Several updates execute inside one ``jax.lax.scan`` call, reducing Python and
  accelerator-dispatch overhead.
* Metrics stay on device until training completes; there is no per-step
  ``device_get`` synchronization.
* Route objectives retain one complete leading minibatch, so a Tesseract-backed
  solver can process one coarse batched call per stage and optimizer update.

For Tesseract-backed Full-E2E training, keep each Tesseract service alive for
all updates and make its API accept coordinates shaped ``(B, M, N, 2)``.  All
inner solver iterations must remain inside the Tesseract ``apply`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .energy import mean_repulsive_energy
from .fast_training import FastAdamOptions, train_in_chunks
from .flow import (
    SamplingOptions,
    flow_matching_loss,
    sample_conditional_flow,
    sample_uniform_torus,
)
from .geometry import periodic_mean_squared_displacement
from .network import FlowNetworkConfig, Parameters
from .observables import ensemble_pair_moments
from .routing import AblationMode, training_stage
from .solvers import SolverBackend


@dataclass(frozen=True)
class AdamOptions:
    """Optimizer and dispatch settings.

    ``chunk_size`` is the requested number of optimizer updates per host
    dispatch.  The implementation automatically chooses the largest divisor of
    ``num_steps`` not exceeding this value, so existing configurations do not
    need to make ``num_steps`` divisible by ``chunk_size``.

    Solver-aware routes are capped more conservatively:

    * Base/native route: up to ``chunk_size`` updates per dispatch;
    * Relax-E2E: up to eight updates;
    * Full-E2E: up to four updates.

    These caps avoid creating an excessively large compiled graph around
    external Tesseract calls while still eliminating most Python dispatch.
    """

    learning_rate: float
    num_steps: int
    batch_size: int
    gradient_clip_norm: float = 5.0
    chunk_size: int = 32
    synchronize_each_chunk: bool = False

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")


@dataclass(frozen=True)
class FineTuneWeights:
    flow_matching: float = 1.0
    observed: float = 20.0
    physical: float = 1.0
    correction: float = 20.0


@dataclass(frozen=True)
class TrainingResult:
    parameters: Parameters
    history: dict[str, np.ndarray]


def tree_l2_norm(tree: Any) -> Array:
    """Global Euclidean norm of a parameter pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves) + 1e-24)


def make_minibatch_schedule(
    num_samples: int,
    options: AdamOptions,
    seed: int,
) -> np.ndarray:
    """Create a deterministic fixed-shape schedule with wraparound sampling."""
    options.validate()
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if options.batch_size > num_samples:
        raise ValueError(
            "batch_size cannot exceed num_samples for deterministic "
            "without-replacement schedule construction"
        )

    rng = np.random.default_rng(seed)
    schedule = np.empty(
        (options.num_steps, options.batch_size),
        dtype=np.int32,
    )
    pool = np.arange(num_samples, dtype=np.int32)
    cursor = num_samples
    shuffled = pool
    for step in range(options.num_steps):
        if cursor + options.batch_size > num_samples:
            shuffled = rng.permutation(pool)
            cursor = 0
        schedule[step] = shuffled[cursor : cursor + options.batch_size]
        cursor += options.batch_size
    return schedule


def _largest_divisor_at_most(value: int, limit: int) -> int:
    """Return the largest positive divisor of ``value`` not above ``limit``."""
    upper = min(value, max(limit, 1))
    for candidate in range(upper, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _route_chunk_size(mode: AblationMode | None, options: AdamOptions) -> int:
    """Choose a fixed scan length appropriate for native or solver training."""
    requested = options.chunk_size
    if mode is AblationMode.RELAX_E2E:
        requested = min(requested, 8)
    elif mode is AblationMode.FULL_E2E:
        requested = min(requested, 4)
    return _largest_divisor_at_most(options.num_steps, requested)


def _folded_training_keys(base_key: Array, num_steps: int) -> Array:
    """Match the previous ``fold_in(base_key, step)`` key sequence exactly."""
    step_indices = jnp.arange(num_steps, dtype=jnp.uint32)
    return jax.vmap(lambda step: jax.random.fold_in(base_key, step))(step_indices)


def _history_to_numpy(history: Any) -> dict[str, np.ndarray]:
    """Perform one synchronized device-to-host transfer after training."""
    host_history = jax.device_get(history)
    return {
        name: np.asarray(values)
        for name, values in host_history.items()
    }


def _run_fast_training(
    parameters: Parameters,
    batches: dict[str, Array],
    keys: Array,
    loss_function: Any,
    options: AdamOptions,
    *,
    mode: AblationMode | None,
) -> TrainingResult:
    """Execute a complete schedule with fused, chunked Adam updates."""
    chunk_size = _route_chunk_size(mode, options)
    result = train_in_chunks(
        parameters,
        batches,
        keys,
        loss_function,
        FastAdamOptions(
            learning_rate=options.learning_rate,
            num_steps=options.num_steps,
            chunk_size=chunk_size,
            gradient_clip_norm=options.gradient_clip_norm,
        ),
        synchronize_each_chunk=options.synchronize_each_chunk,
    )
    return TrainingResult(
        parameters=result.parameters,
        history=_history_to_numpy(result.history),
    )


def pretrain_flow(
    parameters: Parameters,
    targets: Array,
    conditions: Array,
    box: Array,
    network_config: FlowNetworkConfig,
    options: AdamOptions,
    *,
    schedule_seed: int,
    random_seed: int,
) -> TrainingResult:
    """Shared simulation-free flow-matching pretraining.

    The complete deterministic schedule is gathered once into fixed-shape
    device batches.  This avoids Python-side indexing and host-to-device copies
    during the optimizer loop.
    """
    options.validate()
    targets = jax.device_put(jnp.asarray(targets))
    conditions = jax.device_put(jnp.asarray(conditions))
    box = jax.device_put(jnp.asarray(box))

    schedule = make_minibatch_schedule(
        targets.shape[0],
        options,
        schedule_seed,
    )
    schedule_device = jax.device_put(jnp.asarray(schedule, dtype=jnp.int32))
    scheduled_batches = {
        "targets": targets[schedule_device],
        "conditions": conditions[schedule_device],
    }
    keys = _folded_training_keys(
        jax.random.PRNGKey(random_seed),
        options.num_steps,
    )

    def loss_function(
        model: Parameters,
        batch: dict[str, Array],
        key: Array,
    ) -> tuple[Array, dict[str, Array]]:
        return flow_matching_loss(
            model,
            batch["targets"],
            batch["conditions"],
            key,
            box,
            network_config,
        )

    return _run_fast_training(
        parameters,
        scheduled_batches,
        keys,
        loss_function,
        options,
        mode=None,
    )


def route_objective(
    parameters: Parameters,
    mode: AblationMode,
    target_batch: Array,
    condition_batch: Array,
    moment_batch: Array,
    key: Array,
    backend: SolverBackend,
    network_config: FlowNetworkConfig,
    sampling_options: SamplingOptions,
    weights: FineTuneWeights,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the common solver-aware stochastic objective for one batch.

    ``target_batch`` has a leading minibatch dimension.  Consequently,
    ``training_stage`` receives the complete batch and a Tesseract-backed
    implementation can issue one coarse relaxation request and one coarse
    projection request per optimizer update.  Do not place per-sample RPCs or
    inner solver iterations in this function.
    """
    flow_key, source_key = jax.random.split(key)
    flow_loss, flow_metrics = flow_matching_loss(
        parameters,
        target_batch,
        condition_batch,
        flow_key,
        backend.box,
        network_config,
    )
    source = sample_uniform_torus(
        source_key,
        target_batch.shape,
        backend.box,
        target_batch.dtype,
    )
    generated = sample_conditional_flow(
        parameters,
        source,
        condition_batch,
        backend.box,
        network_config,
        sampling_options,
    )
    stage, route_diagnostics = training_stage(
        mode,
        generated,
        moment_batch,
        backend,
    )
    stage_moments = jax.vmap(
        lambda ensemble: ensemble_pair_moments(
            ensemble,
            backend.box,
            backend.basis,
        )
    )(stage)
    moment_scale = jnp.maximum(backend.moment_scales, 1e-12)
    observed_error = (stage_moments - moment_batch) / moment_scale
    observed_loss = jnp.mean(observed_error * observed_error)
    physical_loss = jnp.mean(
        jax.vmap(
            lambda ensemble: mean_repulsive_energy(
                ensemble,
                backend.box,
                backend.physical,
            )
        )(stage)
    )
    correction_squared = jax.vmap(
        lambda final, initial: periodic_mean_squared_displacement(
            final,
            initial,
            backend.box,
        )
    )(stage, generated)
    correction_loss = jnp.mean(correction_squared)
    correction = jnp.sqrt(jnp.maximum(correction_squared, 0.0))
    total = (
        weights.flow_matching * flow_loss
        + weights.observed * observed_loss
        + weights.physical * physical_loss
        + weights.correction * correction_loss
    )
    metrics = {
        **flow_metrics,
        "observed_loss": observed_loss,
        "physical_loss": physical_loss,
        "correction_loss": correction_loss,
        "training_correction_rms": jnp.mean(correction),
        "relaxation_used": route_diagnostics["relaxation_used"],
        "projection_used": route_diagnostics["projection_used"],
    }
    return total, metrics


def fine_tune_route(
    parameters: Parameters,
    mode: AblationMode,
    targets: Array,
    conditions: Array,
    target_moments: Array,
    backend: SolverBackend,
    network_config: FlowNetworkConfig,
    sampling_options: SamplingOptions,
    optimizer_options: AdamOptions,
    weights: FineTuneWeights,
    *,
    schedule_seed: int,
    random_seed: int,
) -> TrainingResult:
    """Fine-tune one route from the same shared pretrained parameters.

    Base uses the largest configured scan chunk.  Relax-E2E and Full-E2E use
    automatically capped chunks because each update may cross a Tesseract
    boundary.  The Tesseract services themselves must be started once outside
    this function and reused by ``backend`` for the complete call.
    """
    if mode is AblationMode.POST_HOC:
        raise ValueError(
            "Post-hoc reuses Base parameters and is not trained separately"
        )
    optimizer_options.validate()

    targets = jax.device_put(jnp.asarray(targets))
    conditions = jax.device_put(jnp.asarray(conditions))
    target_moments = jax.device_put(jnp.asarray(target_moments))

    schedule = make_minibatch_schedule(
        targets.shape[0],
        optimizer_options,
        schedule_seed,
    )
    schedule_device = jax.device_put(jnp.asarray(schedule, dtype=jnp.int32))
    scheduled_batches = {
        "targets": targets[schedule_device],
        "conditions": conditions[schedule_device],
        "target_moments": target_moments[schedule_device],
    }
    keys = _folded_training_keys(
        jax.random.PRNGKey(random_seed),
        optimizer_options.num_steps,
    )

    def loss_function(
        model: Parameters,
        batch: dict[str, Array],
        key: Array,
    ) -> tuple[Array, dict[str, Array]]:
        return route_objective(
            model,
            mode,
            batch["targets"],
            batch["conditions"],
            batch["target_moments"],
            key,
            backend,
            network_config,
            sampling_options,
            weights,
        )

    return _run_fast_training(
        parameters,
        scheduled_batches,
        keys,
        loss_function,
        optimizer_options,
        mode=mode,
    )
