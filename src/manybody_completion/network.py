"""Compact conditional equivariant velocity network with plain JAX pytrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances, periodic_direction_displacements

Parameters = dict[str, Any]


@dataclass(frozen=True)
class FlowNetworkConfig:
    hidden_dim: int = 32
    message_dim: int = 32
    num_layers: int = 2
    radial_basis_size: int = 8
    radial_max: float = 0.5
    radial_width: float = 0.08
    time_frequencies: int = 4
    velocity_scale: float = 1.0

    def validate(self) -> None:
        if min(self.hidden_dim, self.message_dim, self.num_layers) < 1:
            raise ValueError("network dimensions and layer count must be positive")
        if self.radial_basis_size < 1:
            raise ValueError("radial_basis_size must be positive")
        if min(self.radial_max, self.radial_width, self.velocity_scale) <= 0:
            raise ValueError("radial and velocity scales must be positive")
        if self.time_frequencies < 1:
            raise ValueError("time_frequencies must be positive")


def _init_linear(key: Array, input_dim: int, output_dim: int, dtype: jnp.dtype) -> dict[str, Array]:
    scale = jnp.sqrt(jnp.asarray(2.0 / max(input_dim, 1), dtype=dtype))
    weight = scale * jax.random.normal(key, (input_dim, output_dim), dtype=dtype)
    bias = jnp.zeros((output_dim,), dtype=dtype)
    return {"weight": weight, "bias": bias}


def _init_mlp(
    key: Array,
    dimensions: tuple[int, ...],
    dtype: jnp.dtype,
    *,
    zero_final: bool = False,
) -> tuple[dict[str, Array], ...]:
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = tuple(
        _init_linear(layer_key, input_dim, output_dim, dtype)
        for layer_key, input_dim, output_dim in zip(keys, dimensions[:-1], dimensions[1:])
    )
    if zero_final:
        final = dict(layers[-1])
        final["weight"] = jnp.zeros_like(final["weight"])
        final["bias"] = jnp.zeros_like(final["bias"])
        layers = (*layers[:-1], final)
    return layers


def _apply_mlp(layers: tuple[dict[str, Array], ...], values: Array) -> Array:
    output = values
    for index, layer in enumerate(layers):
        output = output @ layer["weight"] + layer["bias"]
        if index + 1 < len(layers):
            output = jax.nn.silu(output)
    return output


def time_embedding(time: Array, frequencies: int) -> Array:
    """Fourier time embedding with shape ``(..., 1 + 2F)``."""
    time = jnp.asarray(time)
    harmonics = jnp.arange(1, frequencies + 1, dtype=time.dtype)
    phase = 2.0 * jnp.pi * time[..., None] * harmonics
    return jnp.concatenate((time[..., None], jnp.sin(phase), jnp.cos(phase)), axis=-1)


def initialize_flow_network(
    key: Array,
    condition_dim: int,
    config: FlowNetworkConfig,
    *,
    dtype: jnp.dtype,
) -> Parameters:
    """Initialize the equivariant velocity network."""
    config.validate()
    time_dim = 1 + 2 * config.time_frequencies
    keys = iter(jax.random.split(key, 1 + 3 * config.num_layers))
    encoder = _init_mlp(
        next(keys),
        (condition_dim + time_dim, config.hidden_dim, config.hidden_dim),
        dtype,
    )
    layers: list[dict[str, Any]] = []
    for _ in range(config.num_layers):
        message = _init_mlp(
            next(keys),
            (
                2 * config.hidden_dim + config.radial_basis_size,
                config.message_dim,
                config.message_dim,
            ),
            dtype,
        )
        node_update = _init_mlp(
            next(keys),
            (config.hidden_dim + config.message_dim, config.hidden_dim, config.hidden_dim),
            dtype,
        )
        coordinate_weight = _init_mlp(
            next(keys),
            (
                2 * config.hidden_dim + config.radial_basis_size,
                config.message_dim,
                1,
            ),
            dtype,
            zero_final=True,
        )
        layers.append(
            {
                "message": message,
                "node_update": node_update,
                "coordinate_weight": coordinate_weight,
            }
        )
    return {"encoder": encoder, "layers": tuple(layers)}


def _radial_features(distances: Array, config: FlowNetworkConfig) -> Array:
    centers = jnp.linspace(
        0.0,
        config.radial_max,
        config.radial_basis_size,
        dtype=distances.dtype,
    )
    return jnp.exp(-0.5 * ((distances[..., None] - centers) / config.radial_width) ** 2)


def flow_velocity(
    parameters: Parameters,
    coordinates: Array,
    time: Array,
    condition: Array,
    box: Array,
    config: FlowNetworkConfig,
) -> Array:
    """Evaluate a permutation/translation/D4-equivariant velocity field.

    Parameters
    ----------
    coordinates:
        Shape ``(B, M, N, 2)``.
    time:
        Scalar or shape ``(B,)``.
    condition:
        Shape ``(B, R)``.
    """
    config.validate()
    coordinates = jnp.asarray(coordinates)
    condition = jnp.asarray(condition, dtype=coordinates.dtype)
    time = jnp.asarray(time, dtype=coordinates.dtype)
    if time.ndim == 0:
        time = jnp.broadcast_to(time, (coordinates.shape[0],))
    features = jnp.concatenate(
        (condition, time_embedding(time, config.time_frequencies)), axis=-1
    )
    node_state = _apply_mlp(parameters["encoder"], features)
    node_state = jnp.broadcast_to(
        node_state[:, None, None, :],
        coordinates.shape[:-1] + (config.hidden_dim,),
    )
    distances = chord_distances(coordinates, box)
    radial = _radial_features(distances, config)
    directions = periodic_direction_displacements(coordinates, box)
    num_particles = coordinates.shape[-2]
    pair_mask = 1.0 - jnp.eye(num_particles, dtype=coordinates.dtype)
    normalizer = jnp.asarray(max(num_particles - 1, 1), coordinates.dtype)
    velocity = jnp.zeros_like(coordinates)

    for layer in parameters["layers"]:
        state_i = jnp.broadcast_to(
            node_state[..., :, None, :],
            node_state.shape[:-2]
            + (num_particles, num_particles, config.hidden_dim),
        )
        state_j = jnp.broadcast_to(
            node_state[..., None, :, :],
            node_state.shape[:-2]
            + (num_particles, num_particles, config.hidden_dim),
        )
        directed = jnp.concatenate((state_i, state_j, radial), axis=-1)
        messages = _apply_mlp(layer["message"], directed) * pair_mask[..., None]
        aggregate = jnp.sum(messages, axis=-2) / normalizer
        update = _apply_mlp(
            layer["node_update"], jnp.concatenate((node_state, aggregate), axis=-1)
        )
        node_state = (node_state + update) / jnp.sqrt(
            jnp.asarray(2.0, coordinates.dtype)
        )
        symmetric = jnp.concatenate(
            (state_i + state_j, jnp.abs(state_i - state_j), radial), axis=-1
        )
        weight = _apply_mlp(layer["coordinate_weight"], symmetric)[..., 0]
        weight = jnp.tanh(weight) * pair_mask
        velocity = velocity + jnp.sum(weight[..., None] * directions, axis=-2) / normalizer

    velocity = config.velocity_scale * velocity
    # Fix the unidentifiable global translation gauge per replica.
    return velocity - jnp.mean(velocity, axis=-2, keepdims=True)
