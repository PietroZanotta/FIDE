"""Configuration-driven setup for native-generator experiments.

This module contains deterministic data/latent construction shared by the
single-mode smoke run and the four-way ablation. Keeping setup in the library
prevents subtle differences in initialization, normalization, or solver options
from contaminating comparisons between training modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .composition import CompletionOptions, PhysicalParameters
from .generator import (
    EquivariantGeneratorConfig,
    GeneratorParameters,
    initialize_equivariant_generator,
    make_periodic_grid_anchors,
)
from .generator_training import (
    AdamOptions,
    GeneratorBatch,
    GeneratorObjectiveWeights,
)
from .observables import PairBasis
from .projection import ProjectionOptions
from .relaxation import RelaxationOptions


@dataclass(frozen=True)
class GeneratorExperimentProblem:
    """All immutable inputs required for a fair generator-training comparison."""

    initial_parameters: GeneratorParameters
    batch: GeneratorBatch
    model_config: EquivariantGeneratorConfig
    completion_options: CompletionOptions
    objective_weights: GeneratorObjectiveWeights
    training_options: AdamOptions
    condition_mean: Array
    condition_scale: Array
    selected_indices: Array


def _resolve_dtype(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.float64
    if name == "float32":
        return jnp.float32
    raise ValueError("dtype must be 'float32' or 'float64'")


def build_generator_experiment_problem(
    configuration: dict[str, Any],
    repository_root: Path,
) -> GeneratorExperimentProblem:
    """Build deterministic model, batch, solver, and optimizer state from YAML data."""
    dtype = _resolve_dtype(configuration["dtype"])
    archive_path = repository_root / configuration["dataset"]
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive does not exist: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        indices = np.asarray(configuration["target_indices"], dtype=np.int64)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("target_indices must be a nonempty one-dimensional sequence")
        coordinates_shape = archive["coordinates"].shape
        if len(coordinates_shape) != 4 or coordinates_shape[-1] != 2:
            raise ValueError("dataset coordinates must have shape (S, M, N, 2)")
        num_samples, num_replicas, num_particles = coordinates_shape[:3]
        if np.any(indices < 0) or np.any(indices >= num_samples):
            raise IndexError("target_indices contain an out-of-range dataset index")
        target_moments = jnp.asarray(archive["pair_moments"][indices], dtype=dtype)
        all_moments = jnp.asarray(archive["pair_moments"], dtype=dtype)
        box = jnp.asarray(archive["box"], dtype=dtype)
        basis = PairBasis(
            centers=jnp.asarray(archive["pair_basis_centers"], dtype=dtype),
            widths=jnp.asarray(archive["pair_basis_widths"], dtype=dtype),
        )

    grid_shape = tuple(int(value) for value in configuration["latent_anchors"]["grid_shape"])
    if len(grid_shape) != 2 or grid_shape[0] * grid_shape[1] != num_particles:
        raise ValueError("latent grid_shape must contain exactly num_particles sites")

    model_config = EquivariantGeneratorConfig(**configuration["model"])
    model_config.validate()
    key = jax.random.PRNGKey(int(configuration["seed"]))
    parameter_key, anchor_key, latent_key = jax.random.split(key, 3)
    parameters = initialize_equivariant_generator(
        parameter_key,
        condition_dim=target_moments.shape[-1],
        config=model_config,
        dtype=dtype,
    )
    batch_size = int(indices.size)
    anchors = make_periodic_grid_anchors(
        anchor_key,
        batch_size=batch_size,
        num_replicas=num_replicas,
        grid_shape=grid_shape,
        box=box,
        jitter_scale=float(configuration["latent_anchors"]["jitter_scale"]),
        dtype=dtype,
    )
    node_latents = jax.random.normal(
        latent_key,
        (batch_size, num_replicas, num_particles, model_config.latent_dim),
        dtype=dtype,
    )
    condition_mean = jnp.mean(all_moments, axis=0)
    condition_scale = jnp.maximum(
        jnp.std(all_moments, axis=0),
        jnp.asarray(configuration["condition_scale_floor"], dtype=dtype),
    )
    conditions = (target_moments - condition_mean) / condition_scale

    scale_mode = configuration.get("projection_moment_scales", "unit")
    if scale_mode == "unit":
        moment_scales = jnp.ones((target_moments.shape[-1],), dtype=dtype)
    elif scale_mode == "dataset_std":
        moment_scales = condition_scale
    else:
        raise ValueError("projection_moment_scales must be 'unit' or 'dataset_std'")
    basis_mask = jnp.ones_like(moment_scales)
    batch = GeneratorBatch(
        anchor_coordinates=anchors,
        node_latents=node_latents,
        conditions=conditions,
        target_moments=target_moments,
        box=box,
        basis=basis,
        moment_scales=moment_scales,
        basis_mask=basis_mask,
    )
    batch.validate(model_config)
    batch.validate_numerics()

    completion = CompletionOptions(
        physical=PhysicalParameters(**configuration["physical"]),
        relaxation=RelaxationOptions(**configuration["relaxation"]),
        projection=ProjectionOptions(**configuration["projection"]),
    )
    weights = GeneratorObjectiveWeights(**configuration["objective"])
    weights.validate()
    training_config = dict(configuration["training"])
    training_config.setdefault("jit_objective", True)
    training = AdamOptions(**training_config)
    training.validate()
    return GeneratorExperimentProblem(
        initial_parameters=parameters,
        batch=batch,
        model_config=model_config,
        completion_options=completion,
        objective_weights=weights,
        training_options=training,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        selected_indices=jnp.asarray(indices),
    )
