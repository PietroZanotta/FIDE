"""Train the conditional equivariant flow on the exact homometric benchmark."""

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
    sample_flow_conditions_chunked,
    subset_flow_batch,
    train_conditional_flow,
)
from manybody_completion.generator import (
    count_generator_parameters,
    flatten_generator_parameters,
)
from manybody_completion.heldout_evaluation import (
    biased_rbf_mmd_squared,
    median_reference_bandwidth,
)
from manybody_completion.homometric import (
    classify_homometric_configurations,
    homometric_mode_metrics,
)
from manybody_completion.observables import (
    angular_cosine_moments,
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


def _evaluate_samples(
    samples: jax.Array,
    *,
    target_pair_moments: jax.Array,
    condition_scale: jax.Array,
    reference_coordinates: jax.Array,
    reference_angular_a: jax.Array,
    reference_angular_b: jax.Array,
    box: jax.Array,
    pair_basis,
    angular_orders: jax.Array,
    angular_neighbor_scale: float,
    angular_scale: jax.Array,
    overlap_threshold: float,
    physical_r0: float,
    physical_kappa: float,
    ambiguous_distance_threshold: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Evaluate pair feasibility and unresolved-mode recovery.

    ``samples`` has shape ``(1, K, M, N, 2)`` because the benchmark has one
    unique observed condition.  Mode classification is performed per replica,
    yielding ``K*M`` independent microscopic draws.
    """
    if samples.ndim != 5 or samples.shape[0] != 1 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape (1, K, M, N, 2)")
    ensembles = samples[0]
    pair_moments = jax.vmap(
        lambda ensemble: ensemble_pair_moments(ensemble, box, pair_basis)
    )(ensembles)
    pair_delta = pair_moments - target_pair_moments
    pair_error = jnp.linalg.norm(pair_delta, axis=-1)
    pair_error_white = jnp.linalg.norm(pair_delta / condition_scale, axis=-1)

    classification = classify_homometric_configurations(
        ensembles,
        box,
        angular_orders,
        angular_neighbor_scale,
        reference_angular_a,
        reference_angular_b,
        angular_scale=None,
    )
    mode_metrics = homometric_mode_metrics(
        classification,
        ambiguous_distance_threshold=ambiguous_distance_threshold,
    )

    generated_angular = classification["angular_descriptor"].reshape(
        (-1, angular_orders.shape[0])
    )
    reference_flat = reference_coordinates.reshape(
        (-1,) + reference_coordinates.shape[-2:]
    )
    reference_angular = angular_cosine_moments(
        reference_flat,
        box,
        angular_orders,
        angular_neighbor_scale,
    )
    generated_white = generated_angular / angular_scale
    reference_white = reference_angular / angular_scale
    bandwidth = median_reference_bandwidth(reference_white)
    angular_mmd2 = biased_rbf_mmd_squared(
        generated_white,
        reference_white,
        bandwidth,
    )

    flat_configurations = ensembles.reshape((-1,) + ensembles.shape[-2:])
    physical_energy = soft_repulsive_energy_per_configuration(
        flat_configurations,
        box,
        physical_r0,
        physical_kappa,
    )
    diagnostics = pair_diagnostics(
        flat_configurations,
        box,
        overlap_threshold,
    )
    metrics = {
        "pair_moment_error_mean": jnp.mean(pair_error),
        "pair_moment_error_max": jnp.max(pair_error),
        "whitened_pair_moment_error_mean": jnp.mean(pair_error_white),
        "whitened_pair_moment_error_max": jnp.max(pair_error_white),
        "physical_energy_mean": jnp.mean(physical_energy),
        "minimum_pair_distance_mean": jnp.mean(
            diagnostics["minimum_pair_distance"]
        ),
        "overlap_fraction_mean": jnp.mean(diagnostics["overlap_fraction"]),
        "angular_mmd2": angular_mmd2,
        "angular_mmd_bandwidth": bandwidth,
        **mode_metrics,
    }
    arrays = {
        "samples": samples,
        "pair_moments": pair_moments,
        "angular_descriptors": classification["angular_descriptor"],
        "mode_labels": classification["label"],
        "mode_reference_distances": classification["minimum_distance"],
    }
    return metrics, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "flow_matching_homometric.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_homometric.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_homometric_trace.csv",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_homometric_outputs.npz",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "flow_matching_homometric_parameters.npz",
    )
    args = parser.parse_args()

    print("[homometric-flow] loading problem", flush=True)
    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem = build_flow_experiment_problem(configuration, REPO_ROOT)
    with np.load(REPO_ROOT / configuration["dataset"], allow_pickle=False) as archive:
        reference_angular_a = jnp.asarray(
            archive["reference_angular_a"],
            dtype=problem.train_batch.target_coordinates.dtype,
        )
        reference_angular_b = jnp.asarray(
            archive["reference_angular_b"],
            dtype=problem.train_batch.target_coordinates.dtype,
        )
        target_pair_moments = jnp.asarray(
            archive["reference_pair_moments"],
            dtype=problem.train_batch.target_coordinates.dtype,
        )

    reporting = configuration["reporting"]
    evaluation_seed = int(reporting["evaluation_seed"])
    print("[homometric-flow] fixed initial objectives", flush=True)
    initial_train_loss, initial_train_metrics = fixed_flow_matching_objective(
        problem.initial_parameters,
        problem.train_batch,
        jax.random.PRNGKey(evaluation_seed),
        problem.flow_config,
    )
    initial_validation_loss, initial_validation_metrics = fixed_flow_matching_objective(
        problem.initial_parameters,
        problem.validation_batch,
        jax.random.PRNGKey(evaluation_seed + 1),
        problem.flow_config,
    )
    print("[homometric-flow] training", flush=True)
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
        jax.random.PRNGKey(evaluation_seed),
        problem.flow_config,
    )
    final_validation_loss, final_validation_metrics = fixed_flow_matching_objective(
        result.parameters,
        problem.validation_batch,
        jax.random.PRNGKey(evaluation_seed + 1),
        problem.flow_config,
    )

    print("[homometric-flow] materializing parameters", flush=True)
    # Materialize the trained pytree on the host before sampling.  Clearing the
    # training executables prevents reverse-mode compilation state from
    # inflating the subsequent ODE sampler on CPU validation hosts.
    initial_parameters_host = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        problem.initial_parameters,
    )
    trained_parameters_host = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        result.parameters,
    )
    history_host = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)),
        result.history,
    )
    del result
    jax.clear_caches()
    initial_parameters = jax.tree_util.tree_map(jnp.asarray, initial_parameters_host)
    trained_parameters = jax.tree_util.tree_map(jnp.asarray, trained_parameters_host)

    # Every dataset row has the same reduced condition.  Sample one normalized
    # condition repeatedly rather than pretending validation rows are distinct.
    condition = problem.validation_batch.conditions[:1]
    sample_kwargs = dict(
        conditions=condition,
        num_samples_per_condition=int(reporting["num_samples"]),
        chunk_size=int(reporting["sampling_chunk_size"]),
        num_replicas=problem.validation_batch.target_coordinates.shape[1],
        num_particles=problem.validation_batch.target_coordinates.shape[2],
        box=problem.validation_batch.box,
        config=problem.flow_config,
        sampling_options=problem.sampling_options,
        dtype=problem.validation_batch.target_coordinates.dtype,
    )
    sampling_key = jax.random.PRNGKey(int(reporting["sampling_seed"]))
    print("[homometric-flow] sampling initial model", flush=True)
    initial_samples = sample_flow_conditions_chunked(
        initial_parameters,
        sampling_key,
        **sample_kwargs,
    )
    initial_samples_host = np.asarray(jax.device_get(initial_samples))
    del initial_samples
    print("[homometric-flow] sampling trained model", flush=True)
    final_samples = sample_flow_conditions_chunked(
        trained_parameters,
        sampling_key,
        **sample_kwargs,
    )
    final_samples_host = np.asarray(jax.device_get(final_samples))
    del final_samples
    initial_samples = jnp.asarray(initial_samples_host)
    final_samples = jnp.asarray(final_samples_host)
    metric_kwargs = dict(
        target_pair_moments=target_pair_moments,
        condition_scale=problem.condition_scale,
        reference_coordinates=problem.validation_batch.target_coordinates,
        reference_angular_a=reference_angular_a,
        reference_angular_b=reference_angular_b,
        box=problem.validation_batch.box,
        pair_basis=problem.pair_basis,
        angular_orders=problem.angular_orders,
        angular_neighbor_scale=problem.angular_neighbor_scale,
        angular_scale=problem.angular_scale,
        overlap_threshold=float(reporting["overlap_threshold"]),
        physical_r0=float(reporting["physical_r0"]),
        physical_kappa=float(reporting["physical_kappa"]),
        ambiguous_distance_threshold=float(
            reporting["ambiguous_distance_threshold"]
        ),
    )
    print("[homometric-flow] evaluating samples", flush=True)
    initial_sampling_metrics, initial_arrays = _evaluate_samples(
        initial_samples,
        **metric_kwargs,
    )
    final_sampling_metrics, final_arrays = _evaluate_samples(
        final_samples,
        **metric_kwargs,
    )

    # Derivative verification uses a small fixed validation subset in a clean
    # compilation cache.  It checks the same parameter objective while keeping
    # the finite-difference executable bounded.
    print("[homometric-flow] gradient check", flush=True)
    jax.clear_caches()
    finite_difference = configuration["finite_difference"]
    gradient_batch = subset_flow_batch(
        problem.validation_batch,
        jnp.arange(min(4, problem.validation_batch.target_coordinates.shape[0])),
    )
    gradient_check = flow_parameter_directional_derivative_sweep(
        trained_parameters,
        gradient_batch,
        jax.random.PRNGKey(int(finite_difference["objective_seed"])),
        jax.random.PRNGKey(int(finite_difference["direction_seed"])),
        problem.flow_config,
        epsilons=tuple(float(value) for value in finite_difference["epsilons"]),
        jit_objective=bool(finite_difference.get("jit_objective", True)),
    )

    print("[homometric-flow] writing artifacts", flush=True)
    report = {
        "schema_version": 1,
        "backend": "local-jax",
        "jax_version": jax.__version__,
        "configuration": configuration,
        "dataset_shape": list(problem.full_batch.target_coordinates.shape),
        "train_indices": problem.split.train_indices,
        "validation_indices": problem.split.validation_indices,
        "minibatch_indices": problem.minibatch_indices,
        "unique_condition_count": int(
            np.unique(
                np.asarray(problem.full_batch.conditions),
                axis=0,
            ).shape[0]
        ),
        "parameter_count": count_generator_parameters(trained_parameters),
        "flow_matching": {
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "train_loss_reduction": initial_train_loss
            / jnp.maximum(final_train_loss, 1e-15),
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
            "initial": initial_sampling_metrics,
            "final": final_sampling_metrics,
        },
        "history": history_host,
    }
    serializable = _to_python(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_trace(args.trace_output, serializable["history"])
    np.savez_compressed(
        args.arrays_output,
        initial_samples=np.asarray(initial_arrays["samples"]),
        final_samples=np.asarray(final_arrays["samples"]),
        initial_pair_moments=np.asarray(initial_arrays["pair_moments"]),
        final_pair_moments=np.asarray(final_arrays["pair_moments"]),
        initial_angular_descriptors=np.asarray(initial_arrays["angular_descriptors"]),
        final_angular_descriptors=np.asarray(final_arrays["angular_descriptors"]),
        initial_mode_labels=np.asarray(initial_arrays["mode_labels"]),
        final_mode_labels=np.asarray(final_arrays["mode_labels"]),
        initial_mode_reference_distances=np.asarray(
            initial_arrays["mode_reference_distances"]
        ),
        final_mode_reference_distances=np.asarray(
            final_arrays["mode_reference_distances"]
        ),
    )
    flattened = flatten_generator_parameters(
        trained_parameters, prefix="flow_parameters"
    )
    np.savez_compressed(
        args.parameters_output,
        **{name: np.asarray(value) for name, value in flattened.items()},
    )

    acceptance = configuration["acceptance"]
    failures: list[str] = []
    if serializable["flow_matching"]["validation_loss_reduction"] < float(
        acceptance["minimum_validation_loss_reduction"]
    ):
        failures.append("validation flow loss did not decrease enough")
    if serializable["gradient_check"]["best_relative_error"] > float(
        acceptance["maximum_gradient_relative_error"]
    ):
        failures.append("parameter gradient failed finite differences")
    if serializable["flow_matching"]["final_validation_metrics"][
        "mean_velocity_norm"
    ] > float(acceptance["maximum_mean_velocity_norm"]):
        failures.append("velocity field violated the translation gauge")
    if serializable["sampling"]["final"]["normalized_mode_entropy"] < float(
        acceptance["minimum_normalized_mode_entropy"]
    ):
        failures.append("the sampler collapsed to one homometric mode")
    if serializable["sampling"]["final"]["ambiguous_fraction"] > float(
        acceptance["maximum_ambiguous_fraction"]
    ):
        failures.append("too many samples lie far from both reference modes")
    if failures:
        raise SystemExit("; ".join(failures))

    print(
        json.dumps(
            {
                "parameter_count": serializable["parameter_count"],
                "num_updates": len(serializable["history"]["loss"]),
                "validation_loss_reduction": serializable["flow_matching"][
                    "validation_loss_reduction"
                ],
                "gradient_relative_error": serializable["gradient_check"][
                    "best_relative_error"
                ],
                "initial_sampling": serializable["sampling"]["initial"],
                "final_sampling": serializable["sampling"]["final"],
                "report": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
