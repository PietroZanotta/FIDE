"""Compact native equivariant generator for periodic particle ensembles.

The generator is deliberately implemented in pure JAX rather than depending on
an additional neural-network framework.  It treats latent node features and
reduced statistics as scalars, and constructs coordinate updates from periodic
relative direction vectors multiplied by learned scalar weights.  Consequently
it is:

* permutation equivariant over particle labels;
* equivariant to global translations on the torus;
* equivariant to square-box D4 transformations when latent features are scalars.

The public functions accept and return plain JAX pytrees, which keeps them easy
to serialize and compatible with ``jax.jit``, ``jax.grad``, and later training
through Tesseract-backed solver calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

import jax
import jax.numpy as jnp
from jax import Array

from .geometry import chord_distances, periodic_direction_displacements, wrap_positions

GeneratorParameters: TypeAlias = dict[str, object]
MLPParameters: TypeAlias = tuple[dict[str, Array], ...]


@dataclass(frozen=True)
class EquivariantGeneratorConfig:
    """Architecture and numerical limits for the native generator."""

    latent_dim: int = 4
    hidden_dim: int = 32
    message_dim: int = 32
    num_message_passing_steps: int = 2
    radial_basis_size: int = 8
    radial_min: float = 0.0
    radial_max: float = 0.48
    radial_width: float = 0.10
    max_coordinate_update: float = 0.05

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.hidden_dim < 1 or self.message_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        if self.num_message_passing_steps < 1:
            raise ValueError("num_message_passing_steps must be positive")
        if self.radial_basis_size < 1:
            raise ValueError("radial_basis_size must be positive")
        if self.radial_min < 0 or self.radial_max <= self.radial_min:
            raise ValueError("radial range must satisfy 0 <= radial_min < radial_max")
        if self.radial_width <= 0:
            raise ValueError("radial_width must be positive")
        if self.max_coordinate_update <= 0:
            raise ValueError("max_coordinate_update must be positive")


def _glorot_uniform(
    key: Array,
    input_dim: int,
    output_dim: int,
    dtype: jnp.dtype,
) -> Array:
    limit = jnp.sqrt(jnp.asarray(6.0 / (input_dim + output_dim), dtype=dtype))
    return jax.random.uniform(
        key,
        (input_dim, output_dim),
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )


def _init_mlp(
    key: Array,
    layer_sizes: tuple[int, ...],
    dtype: jnp.dtype,
    *,
    final_scale: float = 1.0,
) -> MLPParameters:
    if len(layer_sizes) < 2:
        raise ValueError("an MLP requires at least an input and output size")
    keys = jax.random.split(key, len(layer_sizes) - 1)
    layers: list[dict[str, Array]] = []
    for index, (input_dim, output_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weight = _glorot_uniform(keys[index], input_dim, output_dim, dtype)
        if index == len(layer_sizes) - 2:
            weight = weight * jnp.asarray(final_scale, dtype=dtype)
        layers.append(
            {
                "weight": weight,
                "bias": jnp.zeros((output_dim,), dtype=dtype),
            }
        )
    return tuple(layers)


def _apply_mlp(parameters: MLPParameters, inputs: Array) -> Array:
    output = inputs
    for index, layer in enumerate(parameters):
        output = output @ layer["weight"] + layer["bias"]
        if index + 1 < len(parameters):
            output = jax.nn.silu(output)
    return output


def _radial_features(distances: Array, config: EquivariantGeneratorConfig) -> Array:
    centers = jnp.linspace(
        config.radial_min,
        config.radial_max,
        config.radial_basis_size,
        dtype=distances.dtype,
    )
    scaled = (distances[..., None] - centers) / jnp.asarray(
        config.radial_width, dtype=distances.dtype
    )
    return jnp.exp(-0.5 * scaled * scaled)


def _bounded_vector_update(raw_update: Array, maximum_norm: float) -> Array:
    """Smoothly bound each particle update without changing its direction."""
    norm = jnp.linalg.norm(raw_update, axis=-1, keepdims=True)
    maximum = jnp.asarray(maximum_norm, dtype=raw_update.dtype)
    scale = maximum * jnp.tanh(norm / maximum) / jnp.maximum(norm, 1e-15)
    return raw_update * scale


def initialize_equivariant_generator(
    key: Array,
    condition_dim: int,
    config: EquivariantGeneratorConfig | None = None,
    dtype: jnp.dtype = jnp.float64,
) -> GeneratorParameters:
    """Initialize a compact message-passing generator parameter pytree."""
    if config is None:
        config = EquivariantGeneratorConfig()
    config.validate()
    if condition_dim < 1:
        raise ValueError("condition_dim must be positive")

    num_networks = 1 + 3 * config.num_message_passing_steps
    keys = iter(jax.random.split(key, num_networks))
    encoder_input = config.latent_dim + condition_dim
    parameters: GeneratorParameters = {
        "encoder": _init_mlp(
            next(keys),
            (encoder_input, config.hidden_dim, config.hidden_dim),
            dtype,
        ),
        "layers": [],
    }

    directed_pair_input = 2 * config.hidden_dim + config.radial_basis_size
    symmetric_pair_input = 2 * config.hidden_dim + config.radial_basis_size
    node_update_input = config.hidden_dim + config.message_dim
    layers: list[dict[str, MLPParameters]] = []
    for _ in range(config.num_message_passing_steps):
        layers.append(
            {
                "message": _init_mlp(
                    next(keys),
                    (directed_pair_input, config.message_dim, config.message_dim),
                    dtype,
                ),
                "node_update": _init_mlp(
                    next(keys),
                    (node_update_input, config.hidden_dim, config.hidden_dim),
                    dtype,
                    final_scale=0.25,
                ),
                "coordinate_weight": _init_mlp(
                    next(keys),
                    (symmetric_pair_input, config.message_dim, 1),
                    dtype,
                    final_scale=0.05,
                ),
            }
        )
    parameters["layers"] = tuple(layers)
    return parameters


def _validate_generator_inputs(
    anchor_coordinates: Array,
    node_latents: Array,
    condition: Array,
    box: Array,
    config: EquivariantGeneratorConfig,
) -> None:
    if anchor_coordinates.ndim != 3 or anchor_coordinates.shape[-1] != 2:
        raise ValueError(
            "anchor_coordinates must have shape (M, N, 2); "
            f"got {anchor_coordinates.shape}"
        )
    if node_latents.shape != anchor_coordinates.shape[:-1] + (config.latent_dim,):
        raise ValueError(
            "node_latents must have shape (M, N, latent_dim); "
            f"got {node_latents.shape}"
        )
    if condition.ndim != 1:
        raise ValueError(f"condition must have shape (C,), got {condition.shape}")
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")
    if anchor_coordinates.shape[-2] < 2:
        raise ValueError("at least two particles are required")


def apply_equivariant_generator(
    parameters: GeneratorParameters,
    anchor_coordinates: Array,
    node_latents: Array,
    condition: Array,
    box: Array,
    config: EquivariantGeneratorConfig | None = None,
) -> Array:
    """Generate one periodic ensemble with shape ``(M, N, 2)``.

    ``anchor_coordinates`` are latent torus positions, not observed target
    coordinates.  A global translation of these anchors produces the same
    translation of the generated ensemble.  ``node_latents`` must be permuted
    together with particles when checking particle-label equivariance.
    """
    if config is None:
        config = EquivariantGeneratorConfig()
    config.validate()

    coordinates = jnp.asarray(anchor_coordinates)
    dtype = coordinates.dtype
    node_latents = jnp.asarray(node_latents, dtype=dtype)
    condition = jnp.asarray(condition, dtype=dtype)
    box = jnp.asarray(box, dtype=dtype)
    _validate_generator_inputs(coordinates, node_latents, condition, box, config)

    coordinates = wrap_positions(coordinates, box)
    condition_features = jnp.broadcast_to(
        condition,
        coordinates.shape[:-1] + (condition.shape[0],),
    )
    node_state = _apply_mlp(
        parameters["encoder"],
        jnp.concatenate((node_latents, condition_features), axis=-1),
    )

    num_particles = coordinates.shape[-2]
    pair_mask = 1.0 - jnp.eye(num_particles, dtype=dtype)
    pair_normalizer = jnp.asarray(max(num_particles - 1, 1), dtype=dtype)

    for layer in parameters["layers"]:
        distances = chord_distances(coordinates, box)
        radial = _radial_features(distances, config)
        directions = periodic_direction_displacements(coordinates, box)

        state_i = jnp.broadcast_to(
            node_state[..., :, None, :],
            node_state.shape[:-2] + (num_particles, num_particles, config.hidden_dim),
        )
        state_j = jnp.broadcast_to(
            node_state[..., None, :, :],
            node_state.shape[:-2] + (num_particles, num_particles, config.hidden_dim),
        )

        directed_features = jnp.concatenate((state_i, state_j, radial), axis=-1)
        messages = _apply_mlp(layer["message"], directed_features)
        messages = messages * pair_mask[..., None]
        aggregate = jnp.sum(messages, axis=-2) / pair_normalizer
        node_delta = _apply_mlp(
            layer["node_update"],
            jnp.concatenate((node_state, aggregate), axis=-1),
        )
        node_state = (node_state + node_delta) / jnp.sqrt(
            jnp.asarray(2.0, dtype=dtype)
        )

        # Symmetric scalar weights guarantee pairwise antisymmetric coordinate
        # updates because the periodic direction vectors satisfy d_ji = -d_ij.
        symmetric_features = jnp.concatenate(
            (state_i + state_j, jnp.abs(state_i - state_j), radial), axis=-1
        )
        scalar_weight = jnp.tanh(
            _apply_mlp(layer["coordinate_weight"], symmetric_features)[..., 0]
        )
        scalar_weight = scalar_weight * pair_mask
        raw_update = jnp.sum(scalar_weight[..., None] * directions, axis=-2) / pair_normalizer
        coordinate_update = _bounded_vector_update(
            raw_update, config.max_coordinate_update
        )
        coordinates = wrap_positions(coordinates + coordinate_update, box)

    return coordinates


def count_generator_parameters(parameters: GeneratorParameters) -> int:
    """Return the number of scalar trainable parameters in a generator pytree."""
    leaves = jax.tree_util.tree_leaves(parameters)
    return int(sum(leaf.size for leaf in leaves))


def make_periodic_grid_anchors(
    key: Array,
    batch_size: int,
    num_replicas: int,
    grid_shape: tuple[int, int],
    box: Array,
    *,
    jitter_scale: float = 0.0,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Create target-independent periodic latent anchors on a rectangular grid.

    Each replica receives an independent global torus translation and optional
    Gaussian jitter.  The construction is useful for deterministic smoke tests
    because it avoids severe initial overlaps without injecting any microscopic
    target configuration into the generator input.
    """
    if batch_size < 1 or num_replicas < 1:
        raise ValueError("batch_size and num_replicas must be positive")
    nx, ny = grid_shape
    if nx < 1 or ny < 1:
        raise ValueError("grid_shape entries must be positive")
    if jitter_scale < 0:
        raise ValueError("jitter_scale cannot be negative")
    box = jnp.asarray(box, dtype=dtype)
    if box.shape != (2,):
        raise ValueError(f"box must have shape (2,), got {box.shape}")

    x = (jnp.arange(nx, dtype=dtype) + 0.5) / nx * box[0]
    y = (jnp.arange(ny, dtype=dtype) + 0.5) / ny * box[1]
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    grid = jnp.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)

    translation_key, jitter_key = jax.random.split(key)
    translations = jax.random.uniform(
        translation_key,
        (batch_size, num_replicas, 1, 2),
        minval=0.0,
        maxval=1.0,
        dtype=dtype,
    ) * box
    jitter = jitter_scale * jax.random.normal(
        jitter_key,
        (batch_size, num_replicas, nx * ny, 2),
        dtype=dtype,
    )
    return wrap_positions(grid[None, None, :, :] + translations + jitter, box)


def flatten_generator_parameters(
    parameters: GeneratorParameters,
    prefix: str = "parameters",
) -> dict[str, Array]:
    """Flatten the parameter pytree into stable path-keyed arrays.

    The returned mapping is suitable for ``numpy.savez``.  Paths encode nested
    dictionary keys and tuple indices, making parameter archives inspectable
    without relying on Python pickle serialization.
    """
    flattened: dict[str, Array] = {}

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}")
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
            return
        flattened[path] = jnp.asarray(value)

    visit(parameters, prefix)
    return flattened


def restore_generator_parameters(
    template: GeneratorParameters,
    arrays: Mapping[str, Array],
    prefix: str = "parameters",
) -> GeneratorParameters:
    """Restore a parameter pytree from a path-keyed array mapping.

    ``template`` defines the expected architecture and dtypes.  This makes a
    saved archive fail loudly when loaded into an incompatible model rather
    than silently reshaping or dropping parameters.
    """
    available = set(arrays.keys())
    consumed: set[str] = set()

    def restore(value: object, path: str) -> object:
        if isinstance(value, dict):
            return {key: restore(value[key], f"{path}.{key}") for key in sorted(value)}
        if isinstance(value, tuple):
            return tuple(restore(item, f"{path}.{index}") for index, item in enumerate(value))
        if isinstance(value, list):
            return [restore(item, f"{path}.{index}") for index, item in enumerate(value)]
        if path not in available:
            raise KeyError(f"missing generator parameter array: {path}")
        reference = jnp.asarray(value)
        restored = jnp.asarray(arrays[path], dtype=reference.dtype)
        if restored.shape != reference.shape:
            raise ValueError(
                f"parameter shape mismatch for {path}: expected {reference.shape}, "
                f"got {restored.shape}"
            )
        consumed.add(path)
        return restored

    result = restore(template, prefix)
    extras = available - consumed
    if extras:
        raise KeyError(f"unexpected generator parameter arrays: {sorted(extras)}")
    return cast(GeneratorParameters, result)
