"""Conditional equivariant flow matching on a periodic particle box.

The module uses a translation-gauge-fixed shortest torus path.  Given an
independent uniform source ensemble ``X0`` and a target ensemble ``X1``, the
path velocity is

    U = minimum_image(X1 - X0) - mean_particles(minimum_image(X1 - X0)).

The subtraction removes the unidentifiable global translation independently in
each replica.  At ``t=1`` the path therefore reaches a global translation of
the target, which preserves every observable used by this project.

The velocity network reuses the audited equivariant message-passing dynamics
from :mod:`manybody_completion.generator`.  Time and reduced statistics enter
only as scalar node features, while coordinate vectors are assembled from
periodic relative directions and symmetric scalar edge weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

from .generator import (
    EquivariantGeneratorConfig,
    GeneratorParameters,
    apply_equivariant_displacement_field,
    initialize_equivariant_generator,
)
from .geometry import translation_gauge_fixed_displacement, wrap_positions

FlowParameters: TypeAlias = GeneratorParameters
IntegratorName: TypeAlias = Literal["euler", "heun"]


@dataclass(frozen=True)
class ConditionalFlowConfig:
    """Architecture and numerical settings for the conditional velocity field."""

    network: EquivariantGeneratorConfig
    time_frequencies: int = 4
    velocity_scale: float = 2.0

    def validate(self) -> None:
        self.network.validate()
        if self.time_frequencies < 1:
            raise ValueError("time_frequencies must be positive")
        if self.velocity_scale <= 0:
            raise ValueError("velocity_scale must be positive")

    @property
    def time_embedding_dim(self) -> int:
        return 1 + 2 * self.time_frequencies


@dataclass(frozen=True)
class FlowSamplingOptions:
    """Fixed-step periodic ODE sampling settings."""

    num_steps: int = 16
    integrator: IntegratorName = "heun"

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.integrator not in ("euler", "heun"):
            raise ValueError("integrator must be 'euler' or 'heun'")


def sinusoidal_time_embedding(
    time: Array | float,
    num_frequencies: int,
    *,
    dtype: jnp.dtype | None = None,
) -> Array:
    """Embed scalar time in ``[0, 1]`` using Fourier features plus raw time."""
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be positive")
    time = jnp.asarray(time, dtype=dtype)
    if time.ndim != 0:
        raise ValueError(f"time must be scalar; got shape {time.shape}")
    frequencies = 2.0 ** jnp.arange(num_frequencies, dtype=time.dtype)
    phases = 2.0 * jnp.pi * frequencies * time
    return jnp.concatenate(
        (time[None], jnp.sin(phases), jnp.cos(phases)), axis=0
    )


def initialize_conditional_flow(
    key: Array,
    condition_dim: int,
    config: ConditionalFlowConfig,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> FlowParameters:
    """Initialize the time-conditioned equivariant velocity network."""
    config.validate()
    if condition_dim < 1:
        raise ValueError("condition_dim must be positive")
    return initialize_equivariant_generator(
        key,
        condition_dim=condition_dim + config.time_embedding_dim,
        config=config.network,
        dtype=dtype,
    )


def apply_conditional_velocity(
    parameters: FlowParameters,
    coordinates: Array,
    node_latents: Array,
    condition: Array,
    time: Array | float,
    box: Array,
    config: ConditionalFlowConfig,
) -> Array:
    """Evaluate one translation-gauge-fixed equivariant velocity field.

    Args:
        coordinates: One ensemble with shape ``(M, N, 2)``.
        node_latents: Scalar node features with shape ``(M, N, latent_dim)``.
        condition: Reduced-statistic condition vector with shape ``(C,)``.
        time: Scalar ODE time.
        box: Periodic side lengths with shape ``(2,)``.
        config: Flow architecture settings.
    """
    config.validate()
    coordinates = jnp.asarray(coordinates)
    condition = jnp.asarray(condition, dtype=coordinates.dtype)
    time_features = sinusoidal_time_embedding(
        time, config.time_frequencies, dtype=coordinates.dtype
    )
    augmented_condition = jnp.concatenate((condition, time_features), axis=0)
    displacement = apply_equivariant_displacement_field(
        parameters,
        coordinates,
        node_latents,
        augmented_condition,
        box,
        config.network,
        remove_mean_velocity=True,
    )
    return jnp.asarray(config.velocity_scale, dtype=coordinates.dtype) * displacement


def sample_uniform_torus(
    key: Array,
    shape: tuple[int, ...],
    box: Array,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Draw independent uniform coordinates in a periodic rectangular box."""
    if len(shape) < 2 or shape[-1] != 2:
        raise ValueError("shape must end in (..., N, 2)")
    box = jnp.asarray(box, dtype=dtype)
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    unit = jax.random.uniform(key, shape, minval=0.0, maxval=1.0, dtype=dtype)
    return unit * box


def flow_matching_path(
    source: Array,
    target: Array,
    time: Array | float,
    box: Array,
) -> tuple[Array, Array]:
    """Return the gauge-fixed torus interpolation and its constant velocity."""
    source = jnp.asarray(source)
    target = jnp.asarray(target, dtype=source.dtype)
    time = jnp.asarray(time, dtype=source.dtype)
    if time.ndim != 0:
        raise ValueError(f"time must be scalar; got shape {time.shape}")
    velocity = translation_gauge_fixed_displacement(source, target, box)
    coordinates = wrap_positions(source + time * velocity, box)
    return coordinates, velocity


def conditional_flow_matching_loss(
    parameters: FlowParameters,
    source: Array,
    target: Array,
    node_latents: Array,
    condition: Array,
    time: Array | float,
    box: Array,
    config: ConditionalFlowConfig,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate one conditional flow-matching regression objective."""
    path_coordinates, target_velocity = flow_matching_path(source, target, time, box)
    predicted_velocity = apply_conditional_velocity(
        parameters,
        path_coordinates,
        node_latents,
        condition,
        time,
        box,
        config,
    )
    box = jnp.asarray(box, dtype=path_coordinates.dtype)
    normalized_error = (predicted_velocity - target_velocity) / box
    loss = jnp.mean(normalized_error * normalized_error)
    target_rms = jnp.sqrt(jnp.mean((target_velocity / box) ** 2))
    predicted_rms = jnp.sqrt(jnp.mean((predicted_velocity / box) ** 2))
    return loss, {
        "velocity_rmse": jnp.sqrt(loss),
        "target_velocity_rms": target_rms,
        "predicted_velocity_rms": predicted_rms,
        "mean_velocity_norm": jnp.sqrt(
            jnp.mean(jnp.mean(predicted_velocity, axis=-2) ** 2)
        ),
    }


def _single_ode_step(
    parameters: FlowParameters,
    coordinates: Array,
    node_latents: Array,
    condition: Array,
    time: Array,
    step_size: Array,
    box: Array,
    config: ConditionalFlowConfig,
    integrator: IntegratorName,
) -> Array:
    velocity = apply_conditional_velocity(
        parameters, coordinates, node_latents, condition, time, box, config
    )
    if integrator == "euler":
        return wrap_positions(coordinates + step_size * velocity, box)
    predictor = wrap_positions(coordinates + step_size * velocity, box)
    next_velocity = apply_conditional_velocity(
        parameters,
        predictor,
        node_latents,
        condition,
        time + step_size,
        box,
        config,
    )
    return wrap_positions(
        coordinates + 0.5 * step_size * (velocity + next_velocity), box
    )


def sample_conditional_flow_from_prior(
    parameters: FlowParameters,
    source: Array,
    node_latents: Array,
    condition: Array,
    box: Array,
    config: ConditionalFlowConfig,
    options: FlowSamplingOptions | None = None,
    *,
    return_trajectory: bool = False,
) -> Array | tuple[Array, Array]:
    """Integrate the conditional periodic ODE from a supplied prior sample."""
    if options is None:
        options = FlowSamplingOptions()
    config.validate()
    options.validate()
    source = wrap_positions(jnp.asarray(source), box)
    dtype = source.dtype
    step_size = jnp.asarray(1.0 / options.num_steps, dtype=dtype)
    times = jnp.arange(options.num_steps, dtype=dtype) * step_size

    def scan_step(coordinates: Array, time: Array) -> tuple[Array, Array]:
        next_coordinates = _single_ode_step(
            parameters,
            coordinates,
            node_latents,
            condition,
            time,
            step_size,
            box,
            config,
            options.integrator,
        )
        return next_coordinates, next_coordinates

    final_coordinates, trajectory = jax.lax.scan(scan_step, source, times)
    if return_trajectory:
        return final_coordinates, jnp.concatenate((source[None], trajectory), axis=0)
    return final_coordinates


def sample_conditional_flow(
    parameters: FlowParameters,
    key: Array,
    condition: Array,
    *,
    num_replicas: int,
    num_particles: int,
    box: Array,
    config: ConditionalFlowConfig,
    options: FlowSamplingOptions | None = None,
    dtype: jnp.dtype = jnp.float64,
    return_trajectory: bool = False,
) -> Array | tuple[Array, Array]:
    """Draw prior coordinates and latent node features, then solve the ODE."""
    if num_replicas < 1 or num_particles < 2:
        raise ValueError("num_replicas must be positive and num_particles at least two")
    source_key, latent_key = jax.random.split(key)
    source = sample_uniform_torus(
        source_key,
        (num_replicas, num_particles, 2),
        box,
        dtype=dtype,
    )
    node_latents = jax.random.normal(
        latent_key,
        (num_replicas, num_particles, config.network.latent_dim),
        dtype=dtype,
    )
    return sample_conditional_flow_from_prior(
        parameters,
        source,
        node_latents,
        condition,
        box,
        config,
        options,
        return_trajectory=return_trajectory,
    )
