"""Train and validate the first conditional equivariant flow-matching sampler."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.config import load_yaml
from manybody_completion.energies import soft_repulsive_energy_per_configuration
from manybody_completion.flow_experiment import build_flow_experiment_problem
from manybody_completion.flow_training import (
    fixed_flow_matching_objective,
    flow_parameter_directional_derivative_sweep,
    sample_flow_conditions,
    train_conditional_flow,
)
from manybody_completion.generator import (
    count_generator_parameters,
    flatten_generator_parameters,
)
from manybody_completion.heldout_evaluation import (
    angular_distribution_diagnostics,
    batch_ensemble_angular_moments,
    median_reference_bandwidth,
)
from manybody_completion.observables import (
    ensemble_pair_moments,
    pair_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_python(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sample_metrics(
    samples: jax.Array,
    target_pair_moments: jax.Array,
    target_angular_moments: jax.Array,
    labels: jax.Array,
    *,
    box: jax.Array,
    pair_basis,
    condition_scale: jax.Array,
    angular_orders: jax.Array,
    angular_neighbor_scale: float,
    angular_scale: jax.Array,
    overlap_threshold: float,
    physical_r0: float,
    physical_kappa: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Return scalar diagnostics and descriptor arrays for ``(B,K,M,N,2)`` samples."""
    batch_size, samples_per_condition = samples.shape[:2]
    pair_moments = jax.vmap(
        lambda condition_samples: jax.vmap(
            lambda ensemble: ensemble_pair_moments(ensemble, box, pair_basis)
        )(condition_samples)
    )(samples)
    target_pair = target_pair_moments[:, None, :]
    pair_error = (pair_moments - target_pair) / condition_scale
    pair_error_norm = jnp.linalg.norm(pair_error, axis=-1)

    flat_samples = samples.reshape((-1,) + samples.shape[2:])
    angular_moments = batch_ensemble_angular_moments(
        flat_samples,
        box,
        angular_orders,
        angular_neighbor_scale,
    )
    repeated_reference = jnp.repeat(
        target_angular_moments, samples_per_condition, axis=0
    )
    repeated_labels = jnp.repeat(labels, samples_per_condition, axis=0)
    reference_white = repeated_reference / angular_scale
    bandwidth = median_reference_bandwidth(reference_white)
    angular = angular_distribution_diagnostics(
        angular_moments,
        repeated_reference,
        repeated_labels,
        angular_scale,
        bandwidth=bandwidth,
    )

    energy = soft_repulsive_energy_per_configuration(
        flat_samples,
        box,
        physical_r0,
        physical_kappa,
    )
    pair_diag = pair_diagnostics(flat_samples, box, overlap_threshold)
    angular_by_condition = angular_moments.reshape(
        (batch_size, samples_per_condition, -1)
    )
    pair_std = jnp.std(pair_moments, axis=1)
    angular_std = jnp.std(angular_by_condition, axis=1)
    metrics = {
        "pair_moment_error_mean": jnp.mean(pair_error_norm),
        "pair_moment_error_max": jnp.max(pair_error_norm),
        "physical_energy_mean": jnp.mean(energy),
        "minimum_pair_distance_mean": jnp.mean(pair_diag["minimum_pair_distance"]),
        "overlap_fraction_mean": jnp.mean(pair_diag["overlap_fraction"]),
        "pair_descriptor_sample_std": jnp.mean(jnp.linalg.norm(pair_std, axis=-1)),
        "angular_descriptor_sample_std": jnp.mean(
            jnp.linalg.norm(angular_std / angular_scale, axis=-1)
        ),
        **angular,
    }
    arrays = {
        "samples": samples,
        "pair_moments": pair_moments,
        "angular_moments": angular_by_condition,
    }
    return metrics, arrays


def _write_trace(path: Path, history: dict[str, Any]) -> None:
    columns = list(history)
    length = len(history[columns[0]])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *columns])
        writer.writeheader()
        for step in range(length):
            writer.writerow(
                {"step": step, **{name: history[name][step] for name in columns}}
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "flow_matching_toy.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_toy.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_toy_trace.csv",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_toy_outputs.npz",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_toy_parameters.npz",
    )
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem = build_flow_experiment_problem(configuration, REPO_ROOT)
    reporting = configuration["reporting"]
    evaluation_seed = int(reporting["evaluation_seed"])
    train_evaluation_key = jax.random.PRNGKey(evaluation_seed)
    validation_evaluation_key = jax.random.PRNGKey(evaluation_seed + 1)

    initial_train_loss, initial_train_metrics = fixed_flow_matching_objective(
        problem.initial_parameters,
        problem.train_batch,
        train_evaluation_key,
        problem.flow_config,
    )
    initial_validation_loss, initial_validation_metrics = fixed_flow_matching_objective(
        problem.initial_parameters,
        problem.validation_batch,
        validation_evaluation_key,
        problem.flow_config,
    )

    result = train_conditional_flow(
        problem.initial_parameters,
        problem.minibatches,
        jax.random.PRNGKey(int(configuration["seed"]) + 1),
        problem.flow_config,
        problem.optimizer_options,
    )
    final_train_loss, final_train_metrics = fixed_flow_matching_objective(
        result.parameters,
        problem.train_batch,
        train_evaluation_key,
        problem.flow_config,
    )
    final_validation_loss, final_validation_metrics = fixed_flow_matching_objective(
        result.parameters,
        problem.validation_batch,
        validation_evaluation_key,
        problem.flow_config,
    )

    finite_difference = configuration["finite_difference"]
    gradient_check = flow_parameter_directional_derivative_sweep(
        result.parameters,
        problem.validation_batch,
        jax.random.PRNGKey(int(finite_difference["objective_seed"])),
        jax.random.PRNGKey(int(finite_difference["direction_seed"])),
        problem.flow_config,
        epsilons=tuple(float(value) for value in finite_difference["epsilons"]),
        jit_objective=bool(finite_difference.get("jit_objective", True)),
    )

    validation_indices = jnp.asarray(problem.split.validation_indices)
    validation_conditions = problem.validation_batch.conditions
    sample_key = jax.random.PRNGKey(int(reporting["sampling_seed"]))
    num_samples = int(reporting["num_samples_per_condition"])
    sample_kwargs = dict(
        conditions=validation_conditions,
        num_samples_per_condition=num_samples,
        num_replicas=problem.validation_batch.target_coordinates.shape[1],
        num_particles=problem.validation_batch.target_coordinates.shape[2],
        box=problem.validation_batch.box,
        config=problem.flow_config,
        sampling_options=problem.sampling_options,
        dtype=problem.validation_batch.target_coordinates.dtype,
    )
    initial_samples = sample_flow_conditions(
        problem.initial_parameters, sample_key, **sample_kwargs
    )
    final_samples = sample_flow_conditions(result.parameters, sample_key, **sample_kwargs)
    metric_kwargs = dict(
        target_pair_moments=problem.target_pair_moments[validation_indices],
        target_angular_moments=problem.target_angular_moments[validation_indices],
        labels=problem.regime_labels[validation_indices],
        box=problem.validation_batch.box,
        pair_basis=problem.pair_basis,
        condition_scale=problem.condition_scale,
        angular_orders=problem.angular_orders,
        angular_neighbor_scale=problem.angular_neighbor_scale,
        angular_scale=problem.angular_scale,
        overlap_threshold=float(reporting["overlap_threshold"]),
        physical_r0=float(reporting["physical_r0"]),
        physical_kappa=float(reporting["physical_kappa"]),
    )
    initial_sample_metrics, initial_arrays = _sample_metrics(
        initial_samples, **metric_kwargs
    )
    final_sample_metrics, final_arrays = _sample_metrics(final_samples, **metric_kwargs)

    report = {
        "schema_version": 1,
        "backend": "local-jax",
        "jax_version": jax.__version__,
        "configuration": configuration,
        "dataset_shape": list(problem.full_batch.target_coordinates.shape),
        "train_indices": problem.split.train_indices.tolist(),
        "validation_indices": problem.split.validation_indices.tolist(),
        "minibatch_indices": problem.minibatch_indices.tolist(),
        "parameter_count": count_generator_parameters(result.parameters),
        "flow_matching": {
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "train_loss_reduction": initial_train_loss / jnp.maximum(final_train_loss, 1e-15),
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "validation_loss_reduction": initial_validation_loss
            / jnp.maximum(final_validation_loss, 1e-15),
            "initial_train_metrics": initial_train_metrics,
            "final_train_metrics": final_train_metrics,
            "initial_validation_metrics": initial_validation_metrics,
            "final_validation_metrics": final_validation_metrics,
        },
        "gradient_check": gradient_check,
        "sampling": {
            "initial": initial_sample_metrics,
            "final": final_sample_metrics,
        },
        "history": result.history,
    }
    serializable = _to_python(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_trace(args.trace_output, serializable["history"])
    np.savez_compressed(
        args.arrays_output,
        initial_samples=np.asarray(initial_arrays["samples"]),
        final_samples=np.asarray(final_arrays["samples"]),
        initial_pair_moments=np.asarray(initial_arrays["pair_moments"]),
        final_pair_moments=np.asarray(final_arrays["pair_moments"]),
        initial_angular_moments=np.asarray(initial_arrays["angular_moments"]),
        final_angular_moments=np.asarray(final_arrays["angular_moments"]),
        validation_indices=np.asarray(problem.split.validation_indices),
    )
    flattened = flatten_generator_parameters(result.parameters, prefix="flow_parameters")
    np.savez_compressed(
        args.parameters_output,
        **{name: np.asarray(value) for name, value in flattened.items()},
    )

    acceptance = configuration["acceptance"]
    failures: list[str] = []
    if serializable["flow_matching"]["train_loss_reduction"] < float(
        acceptance["minimum_training_loss_reduction"]
    ):
        failures.append("training loss did not decrease enough")
    if serializable["gradient_check"]["best_relative_error"] > float(
        acceptance["maximum_gradient_relative_error"]
    ):
        failures.append("parameter gradient failed finite differences")
    if serializable["flow_matching"]["final_validation_metrics"][
        "mean_velocity_norm"
    ] > float(acceptance["maximum_mean_velocity_norm"]):
        failures.append("velocity field violated the translation gauge")
    if failures:
        raise SystemExit("; ".join(failures))

    console = {
        "parameter_count": serializable["parameter_count"],
        "num_updates": len(serializable["history"]["loss"]),
        "train_loss_reduction": serializable["flow_matching"]["train_loss_reduction"],
        "validation_loss_reduction": serializable["flow_matching"][
            "validation_loss_reduction"
        ],
        "best_gradient_relative_error": serializable["gradient_check"][
            "best_relative_error"
        ],
        "initial_sampling": serializable["sampling"]["initial"],
        "final_sampling": serializable["sampling"]["final"],
        "report": str(args.output),
    }
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
