"""Dataset-backed conditional flow-matching experiment construction.

The experiment uses the calibration archive only as a compact stochastic
sampler benchmark.  Conditions are pair moments normalized from the training
split.  Hidden regime labels and angular descriptors remain evaluation-only.
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

from .calibration_experiment import (
    CalibrationSplit,
    make_minibatch_schedule,
    stratified_train_validation_split,
)
from .flow_matching import (
    ConditionalFlowConfig,
    FlowParameters,
    FlowSamplingOptions,
    initialize_conditional_flow,
)
from .flow_training import (
    FlowAdamOptions,
    FlowTrainingBatch,
    subset_flow_batch,
)
from .observables import PairBasis


@dataclass(frozen=True)
class FlowExperimentProblem:
    initial_parameters: FlowParameters
    full_batch: FlowTrainingBatch
    train_batch: FlowTrainingBatch
    validation_batch: FlowTrainingBatch
    minibatches: tuple[FlowTrainingBatch, ...]
    minibatch_indices: np.ndarray
    split: CalibrationSplit
    flow_config: ConditionalFlowConfig
    optimizer_options: FlowAdamOptions
    sampling_options: FlowSamplingOptions
    pair_basis: PairBasis
    target_pair_moments: Array
    target_angular_moments: Array
    angular_orders: Array
    angular_neighbor_scale: float
    angular_scale: Array
    regime_labels: Array
    condition_mean: Array
    condition_scale: Array


def _resolve_dtype(name: str) -> jnp.dtype:
    if name == "float64":
        return jnp.float64
    if name == "float32":
        return jnp.float32
    raise ValueError("dtype must be 'float32' or 'float64'")


def _shared_angular_neighbor_scale(metadata: dict[str, Any]) -> float:
    regimes = metadata.get("source_config", {}).get("regimes", [])
    values = {
        float(regime["energy"]["angular_neighbor_scale"])
        for regime in regimes
        if "energy" in regime and "angular_neighbor_scale" in regime["energy"]
    }
    if len(values) != 1:
        raise ValueError("metadata must contain one shared angular_neighbor_scale")
    return values.pop()


def build_flow_experiment_problem(
    configuration: dict[str, Any],
    repository_root: Path,
) -> FlowExperimentProblem:
    """Build a deterministic leakage-safe flow-matching experiment."""
    dtype = _resolve_dtype(str(configuration["dtype"]))
    archive_path = repository_root / str(configuration["dataset"])
    metadata_path = repository_root / str(configuration["dataset_metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(archive_path, allow_pickle=False) as archive:
        coordinates = jnp.asarray(archive["coordinates"], dtype=dtype)
        pair_moments = jnp.asarray(archive["pair_moments"], dtype=dtype)
        angular_moments = jnp.asarray(archive["angular_moments"], dtype=dtype)
        labels_np = np.asarray(archive["regime_label"], dtype=np.int32)
        box = jnp.asarray(archive["box"], dtype=dtype)
        pair_basis = PairBasis(
            centers=jnp.asarray(archive["pair_basis_centers"], dtype=dtype),
            widths=jnp.asarray(archive["pair_basis_widths"], dtype=dtype),
        )
        angular_orders = jnp.asarray(archive["angular_orders"], dtype=dtype)

    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (S, M, N, 2)")
    if pair_moments.shape[0] != coordinates.shape[0]:
        raise ValueError("pair moments and coordinates must share sample count")
    if angular_moments.shape[0] != coordinates.shape[0]:
        raise ValueError("angular moments and coordinates must share sample count")

    split_config = configuration["split"]
    split = stratified_train_validation_split(
        labels_np,
        validation_per_regime=int(split_config["validation_per_regime"]),
        seed=int(split_config["seed"]),
    )
    train_pair = pair_moments[split.train_indices]
    condition_mean = jnp.mean(train_pair, axis=0)
    condition_scale = jnp.maximum(
        jnp.std(train_pair, axis=0),
        jnp.asarray(float(configuration["condition_scale_floor"]), dtype=dtype),
    )
    conditions = (pair_moments - condition_mean) / condition_scale
    train_angular = angular_moments[split.train_indices]
    angular_scale = jnp.maximum(
        jnp.std(train_angular, axis=0),
        jnp.asarray(float(configuration["angular_scale_floor"]), dtype=dtype),
    )

    full_batch = FlowTrainingBatch(
        target_coordinates=coordinates,
        conditions=conditions,
        box=box,
    )
    full_batch.validate()
    train_batch = subset_flow_batch(full_batch, jnp.asarray(split.train_indices))
    validation_batch = subset_flow_batch(
        full_batch, jnp.asarray(split.validation_indices)
    )

    minibatch_config = configuration["minibatching"]
    schedule = make_minibatch_schedule(
        split.train_indices,
        batch_size=int(minibatch_config["batch_size"]),
        num_epochs=int(minibatch_config["num_epochs"]),
        seed=int(minibatch_config["shuffle_seed"]),
    )
    minibatches = tuple(
        subset_flow_batch(full_batch, jnp.asarray(indices)) for indices in schedule
    )

    from .generator import EquivariantGeneratorConfig

    network_config = EquivariantGeneratorConfig(**configuration["model"])
    flow_config = ConditionalFlowConfig(
        network=network_config,
        **configuration["flow"],
    )
    flow_config.validate()
    key = jax.random.PRNGKey(int(configuration["seed"]))
    initial_parameters = initialize_conditional_flow(
        key,
        condition_dim=conditions.shape[-1],
        config=flow_config,
        dtype=dtype,
    )
    optimizer_options = FlowAdamOptions(**configuration["training"])
    optimizer_options.validate()
    sampling_options = FlowSamplingOptions(**configuration["sampling"])
    sampling_options.validate()

    return FlowExperimentProblem(
        initial_parameters=initial_parameters,
        full_batch=full_batch,
        train_batch=train_batch,
        validation_batch=validation_batch,
        minibatches=minibatches,
        minibatch_indices=schedule,
        split=split,
        flow_config=flow_config,
        optimizer_options=optimizer_options,
        sampling_options=sampling_options,
        pair_basis=pair_basis,
        target_pair_moments=pair_moments,
        target_angular_moments=angular_moments,
        angular_orders=angular_orders,
        angular_neighbor_scale=_shared_angular_neighbor_scale(metadata),
        angular_scale=angular_scale,
        regime_labels=jnp.asarray(labels_np),
        condition_mean=condition_mean,
        condition_scale=condition_scale,
    )
