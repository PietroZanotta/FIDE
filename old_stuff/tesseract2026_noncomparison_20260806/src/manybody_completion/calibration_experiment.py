"""Dataset-backed setup for calibration-scale generator ablations.

The calibration experiment differs from the earlier fixed two-condition smoke
problem in three important ways:

* train/validation indices are stratified by the hidden regime label;
* condition, projection, and held-out descriptor scales are fitted on training
  data only;
* the optimizer receives a deterministic fixed-shape minibatch schedule.

Hidden labels and angular descriptors are retained exclusively for evaluation.
They are never placed in :class:`~manybody_completion.generator_training.GeneratorBatch`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    subset_generator_batch,
)
from .observables import (
    PairBasis,
    ensemble_angular_cosine_moments,
    ensemble_pair_moments,
)
from .projection import ProjectionOptions
from .relaxation import RelaxationOptions


@dataclass(frozen=True)
class CalibrationSplit:
    """Deterministic stratified train/validation partition."""

    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class CalibrationExperimentProblem:
    """Immutable inputs and metadata for calibration-scale training."""

    initial_parameters: GeneratorParameters
    full_batch: GeneratorBatch
    train_batch: GeneratorBatch
    validation_batch: GeneratorBatch
    minibatches: tuple[GeneratorBatch, ...]
    minibatch_indices: np.ndarray
    model_config: EquivariantGeneratorConfig
    completion_options: CompletionOptions
    objective_weights: GeneratorObjectiveWeights
    training_options: AdamOptions
    split: CalibrationSplit
    reference_coordinates: Array
    reference_angular_moments: Array
    regime_labels: Array
    regime_names: tuple[str, ...]
    angular_orders: Array
    angular_neighbor_scale: float
    angular_mean: Array
    angular_scale: Array
    condition_mean: Array
    condition_scale: Array


def _resolve_dtype(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.float64
    if name == "float32":
        return jnp.float32
    raise ValueError("dtype must be 'float32' or 'float64'")


def stratified_train_validation_split(
    labels: np.ndarray,
    *,
    validation_per_regime: int,
    seed: int,
) -> CalibrationSplit:
    """Create a reproducible split with equal validation counts per regime."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError("labels must be a one-dimensional array with at least two samples")
    if validation_per_regime < 1:
        raise ValueError("validation_per_regime must be positive")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if members.size <= validation_per_regime:
            raise ValueError(
                f"regime {label!r} has {members.size} samples; it needs more than "
                f"validation_per_regime={validation_per_regime}"
            )
        shuffled = rng.permutation(members)
        validation_parts.append(np.sort(shuffled[:validation_per_regime]))
        train_parts.append(np.sort(shuffled[validation_per_regime:]))

    train = np.sort(np.concatenate(train_parts)).astype(np.int32)
    validation = np.sort(np.concatenate(validation_parts)).astype(np.int32)
    if np.intersect1d(train, validation).size:
        raise RuntimeError("train and validation indices overlap")
    if train.size + validation.size != labels.size:
        raise RuntimeError("split does not cover every sample exactly once")
    return CalibrationSplit(train_indices=train, validation_indices=validation)


def make_minibatch_schedule(
    train_indices: np.ndarray,
    *,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> np.ndarray:
    """Return ``(num_steps, batch_size)`` shuffled indices.

    A divisible training size is required deliberately.  It keeps every JAX
    update shape identical and avoids either dropping examples or silently
    duplicating them at epoch boundaries.
    """
    indices = np.asarray(train_indices, dtype=np.int32)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("train_indices must be a nonempty one-dimensional array")
    if batch_size < 1 or num_epochs < 1:
        raise ValueError("batch_size and num_epochs must be positive")
    if indices.size % batch_size != 0:
        raise ValueError(
            "training-set size must be divisible by batch_size for fixed-shape "
            f"minibatches; got {indices.size} and {batch_size}"
        )

    rng = np.random.default_rng(seed)
    epochs = [rng.permutation(indices).reshape((-1, batch_size)) for _ in range(num_epochs)]
    return np.concatenate(epochs, axis=0).astype(np.int32)


def _shared_angular_neighbor_scale(metadata: dict[str, Any]) -> float:
    regimes = metadata.get("source_config", {}).get("regimes", [])
    scales = {
        float(regime["energy"]["angular_neighbor_scale"])
        for regime in regimes
        if "energy" in regime and "angular_neighbor_scale" in regime["energy"]
    }
    if len(scales) != 1:
        raise ValueError(
            "dataset metadata must specify one shared angular_neighbor_scale; "
            f"found {sorted(scales)}"
        )
    return scales.pop()


def _validate_archive_descriptors(
    coordinates: Array,
    pair_moments: Array,
    angular_moments: Array,
    box: Array,
    basis: PairBasis,
    angular_orders: Array,
    angular_neighbor_scale: float,
    tolerance: float,
) -> None:
    recomputed_pair = jax.vmap(
        lambda sample: ensemble_pair_moments(sample, box, basis)
    )(coordinates)
    recomputed_angular = jax.vmap(
        lambda sample: ensemble_angular_cosine_moments(
            sample, box, angular_orders, angular_neighbor_scale
        )
    )(coordinates)
    pair_error = float(
        jax.device_get(jnp.max(jnp.abs(recomputed_pair - pair_moments)))
    )
    angular_error = float(
        jax.device_get(jnp.max(jnp.abs(recomputed_angular - angular_moments)))
    )
    if pair_error > tolerance or angular_error > tolerance:
        raise ValueError(
            "dataset descriptors do not match coordinate recomputation: "
            f"pair_error={pair_error:.3e}, angular_error={angular_error:.3e}, "
            f"tolerance={tolerance:.3e}"
        )


def build_calibration_experiment_problem(
    configuration: dict[str, Any],
    repository_root: Path,
) -> CalibrationExperimentProblem:
    """Build a leakage-safe deterministic calibration experiment."""
    dtype = _resolve_dtype(configuration["dtype"])
    archive_path = repository_root / configuration["dataset"]
    metadata_path = repository_root / configuration["dataset_metadata"]
    if not archive_path.is_file():
        raise FileNotFoundError(f"dataset archive does not exist: {archive_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"dataset metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(archive_path, allow_pickle=False) as archive:
        coordinates_np = np.asarray(archive["coordinates"])
        pair_moments_np = np.asarray(archive["pair_moments"])
        angular_moments_np = np.asarray(archive["angular_moments"])
        labels_np = np.asarray(archive["regime_label"], dtype=np.int32)
        regime_names_np = np.asarray(archive["regime_name"])
        box = jnp.asarray(archive["box"], dtype=dtype)
        basis = PairBasis(
            centers=jnp.asarray(archive["pair_basis_centers"], dtype=dtype),
            widths=jnp.asarray(archive["pair_basis_widths"], dtype=dtype),
        )
        angular_orders = jnp.asarray(archive["angular_orders"], dtype=dtype)

    if coordinates_np.ndim != 4 or coordinates_np.shape[-1] != 2:
        raise ValueError("dataset coordinates must have shape (S, M, N, 2)")
    num_samples, num_replicas, num_particles = coordinates_np.shape[:3]
    expected_pair_shape = (num_samples, basis.centers.shape[0])
    expected_angular_shape = (num_samples, angular_orders.shape[0])
    if pair_moments_np.shape != expected_pair_shape:
        raise ValueError(
            f"pair_moments must have shape {expected_pair_shape}; got {pair_moments_np.shape}"
        )
    if angular_moments_np.shape != expected_angular_shape:
        raise ValueError(
            "angular_moments must have shape "
            f"{expected_angular_shape}; got {angular_moments_np.shape}"
        )
    if labels_np.shape != (num_samples,) or regime_names_np.shape != (num_samples,):
        raise ValueError("regime labels and names must have shape (S,)")

    coordinates = jnp.asarray(coordinates_np, dtype=dtype)
    pair_moments = jnp.asarray(pair_moments_np, dtype=dtype)
    angular_moments = jnp.asarray(angular_moments_np, dtype=dtype)
    angular_neighbor_scale = _shared_angular_neighbor_scale(metadata)
    integrity = configuration.get("integrity", {})
    if integrity.get("recompute_descriptors", True):
        _validate_archive_descriptors(
            coordinates,
            pair_moments,
            angular_moments,
            box,
            basis,
            angular_orders,
            angular_neighbor_scale,
            float(integrity.get("descriptor_tolerance", 1e-10)),
        )

    split = stratified_train_validation_split(
        labels_np,
        validation_per_regime=int(configuration["split"]["validation_per_regime"]),
        seed=int(configuration["split"]["seed"]),
    )
    train_indices = jnp.asarray(split.train_indices)
    pair_train = pair_moments[train_indices]
    angular_train = angular_moments[train_indices]
    condition_floor = jnp.asarray(configuration["condition_scale_floor"], dtype=dtype)
    angular_floor = jnp.asarray(configuration["angular_scale_floor"], dtype=dtype)
    condition_mean = jnp.mean(pair_train, axis=0)
    condition_scale = jnp.maximum(jnp.std(pair_train, axis=0), condition_floor)
    angular_mean = jnp.mean(angular_train, axis=0)
    angular_scale = jnp.maximum(jnp.std(angular_train, axis=0), angular_floor)
    conditions = (pair_moments - condition_mean) / condition_scale

    projection_scale_mode = configuration.get("projection_moment_scales", "unit")
    if projection_scale_mode == "unit":
        moment_scales = jnp.ones_like(condition_scale)
    elif projection_scale_mode == "training_std":
        moment_scales = condition_scale
    else:
        raise ValueError(
            "projection_moment_scales must be 'unit' or 'training_std'"
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
        condition_dim=pair_moments.shape[-1],
        config=model_config,
        dtype=dtype,
    )
    anchors = make_periodic_grid_anchors(
        anchor_key,
        batch_size=num_samples,
        num_replicas=num_replicas,
        grid_shape=grid_shape,
        box=box,
        jitter_scale=float(configuration["latent_anchors"]["jitter_scale"]),
        dtype=dtype,
    )
    node_latents = jax.random.normal(
        latent_key,
        (num_samples, num_replicas, num_particles, model_config.latent_dim),
        dtype=dtype,
    )
    full_batch = GeneratorBatch(
        anchor_coordinates=anchors,
        node_latents=node_latents,
        conditions=conditions,
        target_moments=pair_moments,
        box=box,
        basis=basis,
        moment_scales=moment_scales,
        basis_mask=jnp.ones_like(moment_scales),
    )
    full_batch.validate(model_config)
    full_batch.validate_numerics()
    train_batch = subset_generator_batch(full_batch, split.train_indices)
    validation_batch = subset_generator_batch(full_batch, split.validation_indices)

    minibatch_config = configuration["minibatching"]
    schedule = make_minibatch_schedule(
        split.train_indices,
        batch_size=int(minibatch_config["batch_size"]),
        num_epochs=int(minibatch_config["num_epochs"]),
        seed=int(minibatch_config["shuffle_seed"]),
    )
    minibatches = tuple(subset_generator_batch(full_batch, row) for row in schedule)

    completion = CompletionOptions(
        physical=PhysicalParameters(**configuration["physical"]),
        relaxation=RelaxationOptions(**configuration["relaxation"]),
        projection=ProjectionOptions(**configuration["projection"]),
    )
    weights = GeneratorObjectiveWeights(**configuration["objective"])
    weights.validate()
    training_config = dict(configuration["training"])
    training_config["num_steps"] = int(schedule.shape[0])
    training_config.setdefault("jit_objective", True)
    training = AdamOptions(**training_config)
    training.validate()

    unique_names = tuple(
        str(regime_names_np[np.flatnonzero(labels_np == label)[0]])
        for label in np.unique(labels_np)
    )
    return CalibrationExperimentProblem(
        initial_parameters=parameters,
        full_batch=full_batch,
        train_batch=train_batch,
        validation_batch=validation_batch,
        minibatches=minibatches,
        minibatch_indices=schedule,
        model_config=model_config,
        completion_options=completion,
        objective_weights=weights,
        training_options=training,
        split=split,
        reference_coordinates=coordinates,
        reference_angular_moments=angular_moments,
        regime_labels=jnp.asarray(labels_np),
        regime_names=unique_names,
        angular_orders=angular_orders,
        angular_neighbor_scale=angular_neighbor_scale,
        angular_mean=angular_mean,
        angular_scale=angular_scale,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
    )
