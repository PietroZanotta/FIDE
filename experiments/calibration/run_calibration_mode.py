"""Train or evaluate one calibration ablation in an isolated JAX process."""

from __future__ import annotations

import argparse
import csv
import gc
from functools import partial
import json
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.ablation import AblationMode, CompletionStage, get_ablation_spec
from manybody_completion.calibration_experiment import (
    CalibrationExperimentProblem,
    build_calibration_experiment_problem,
)
from manybody_completion.config import load_yaml
from manybody_completion.generator import (
    flatten_generator_parameters,
    restore_generator_parameters,
)
from manybody_completion.generator_training import (
    GeneratorTrainingResult,
    ablation_generator_objective,
    ablation_training_objective,
    evaluate_generator_completion,
    train_equivariant_generator_minibatches,
)
from manybody_completion.heldout_evaluation import (
    evaluate_angular_stages,
    median_reference_bandwidth,
)
from manybody_completion.scalar_training import arrays_to_python

REPO_ROOT = Path(__file__).resolve().parents[2]


def _materialize_and_clear(parameters: object) -> object:
    """Detach parameters from compiled training executables before evaluation."""
    host_parameters = jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value)).copy(), parameters
    )
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()
    gc.collect()
    return jax.tree_util.tree_map(jnp.asarray, host_parameters)


def _tree_l2_distance(left: object, right: object) -> jax.Array:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        raise ValueError("parameter trees have different structures")
    return jnp.sqrt(
        sum(jnp.sum((a - b) ** 2) for a, b in zip(left_leaves, right_leaves))
    )


def _training_objective(problem: CalibrationExperimentProblem, mode: AblationMode):
    return partial(
        ablation_training_objective,
        generator_config=problem.model_config,
        completion_options=problem.completion_options,
        weights=problem.objective_weights,
        mode=mode,
    )


def _split_payload(problem: CalibrationExperimentProblem, split_name: str):
    if split_name == "train":
        indices = problem.split.train_indices
        batch = problem.train_batch
    elif split_name == "validation":
        indices = problem.split.validation_indices
        batch = problem.validation_batch
    else:
        raise ValueError(f"unknown split {split_name!r}")
    index_array = jnp.asarray(indices)
    return (
        batch,
        problem.reference_angular_moments[index_array],
        problem.regime_labels[index_array],
    )


def _evaluate_mode_split(
    problem: CalibrationExperimentProblem,
    parameters: object,
    mode: AblationMode,
    split_name: str,
    bandwidth: jax.Array,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    batch, reference_angular, labels = _split_payload(problem, split_name)
    loss, scalar_metrics = ablation_generator_objective(
        parameters,
        batch,
        problem.model_config,
        problem.completion_options,
        problem.objective_weights,
        mode,
    )
    generated, stages = evaluate_generator_completion(
        parameters,
        batch,
        problem.model_config,
        problem.completion_options,
    )
    coordinates_by_stage = {
        CompletionStage.INITIAL.value: generated,
        CompletionStage.RELAXED.value: stages["relaxed_coordinates"],
        CompletionStage.PROJECTED.value: stages["projected_coordinates"],
    }
    angular = evaluate_angular_stages(
        coordinates_by_stage,
        reference_angular,
        labels,
        batch.box,
        problem.angular_orders,
        problem.angular_neighbor_scale,
        problem.angular_scale,
        bandwidth=bandwidth,
    )
    serving_stage = get_ablation_spec(mode).serving_stage.value
    print(f"[calibration:{mode.value}] assembled {split_name}", flush=True)
    report = {
        "loss": loss,
        "pipeline_metrics": scalar_metrics,
        "angular": {
            stage: value["diagnostics"] for stage, value in angular.items()
        },
        "serving_stage": serving_stage,
        "serving_angular": angular[serving_stage]["diagnostics"],
    }
    arrays: dict[str, np.ndarray] = {
        "generated_coordinates": np.asarray(generated),
        "relaxed_coordinates": np.asarray(stages["relaxed_coordinates"]),
        "projected_coordinates": np.asarray(stages["projected_coordinates"]),
        "moments_initial": np.asarray(stages["moments_initial"]),
        "moments_relaxed": np.asarray(stages["moments_relaxed"]),
        "moments_projected": np.asarray(stages["moments_projected"]),
    }
    for stage, value in angular.items():
        arrays[f"angular_{stage}"] = np.asarray(value["moments"])
    return report, arrays


def _save_trace(path: Path, history: dict[str, jax.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = arrays_to_python(history)
    metric_names = tuple(values.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *metric_names])
        writer.writeheader()
        for step in range(len(values["loss"])):
            writer.writerow(
                {"step": step, **{name: values[name][step] for name in metric_names}}
            )


def _load_parameters(problem: CalibrationExperimentProblem, path: Path) -> object:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return restore_generator_parameters(
        problem.initial_parameters, arrays, prefix="parameters"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in AblationMode])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--arrays-output", type=Path, required=True)
    parser.add_argument("--parameters-output", type=Path, required=True)
    parser.add_argument("--parameters-input", type=Path)
    parser.add_argument("--source-report", type=Path)
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem = build_calibration_experiment_problem(configuration, REPO_ROOT)
    mode = AblationMode.parse(args.mode)
    print(f"[calibration:{mode.value}] problem built", flush=True)

    if mode is AblationMode.POST_HOC:
        if args.parameters_input is None or args.source_report is None:
            raise ValueError("Post-hoc evaluation requires Base parameters and report")
        parameters = _load_parameters(problem, args.parameters_input)
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
        training_final_loss = source_report["training_final_loss"]
        parameter_displacement = source_report["parameter_displacement"]
        history = None
    else:
        objective = _training_objective(problem, mode)
        print(f"[calibration:{mode.value}] training", flush=True)
        result = train_equivariant_generator_minibatches(
            objective,
            problem.initial_parameters,
            problem.minibatches,
            problem.train_batch,
            problem.training_options,
        )
        parameters = result.parameters
        training_final_loss = result.final_loss
        parameter_displacement = _tree_l2_distance(
            parameters, problem.initial_parameters
        )
        history = result.history

    parameters = _materialize_and_clear(parameters)
    reference_train = problem.reference_angular_moments[
        jnp.asarray(problem.split.train_indices)
    ]
    bandwidth = median_reference_bandwidth(
        (reference_train - problem.angular_mean) / problem.angular_scale
    )
    finite_difference: dict[str, object] = {}

    split_reports: dict[str, Any] = {}
    output_arrays: dict[str, np.ndarray] = {}
    print(f"[calibration:{mode.value}] evaluating", flush=True)
    for split_name in ("train", "validation"):
        print(f"[calibration:{mode.value}] evaluating {split_name}", flush=True)
        split_report, split_arrays = _evaluate_mode_split(
            problem, parameters, mode, split_name, bandwidth
        )
        split_reports[split_name] = split_report
        for name, value in split_arrays.items():
            output_arrays[f"{split_name}.{name}"] = value

    report = {
        "schema_version": 1,
        "mode": mode.value,
        "training_stage": get_ablation_spec(mode).training_stage.value,
        "serving_stage": get_ablation_spec(mode).serving_stage.value,
        "training_final_loss": training_final_loss,
        "parameter_displacement": parameter_displacement,
        "train": split_reports["train"],
        "validation": split_reports["validation"],
        "finite_difference": finite_difference,
        "shared": {
            "configuration": configuration,
            "jax_version": jax.__version__,
            "dataset_shape": list(problem.reference_coordinates.shape),
            "train_indices": problem.split.train_indices,
            "validation_indices": problem.split.validation_indices,
            "minibatch_indices": problem.minibatch_indices,
            "regime_names": problem.regime_names,
            "condition_mean": problem.condition_mean,
            "condition_scale": problem.condition_scale,
            "angular_mean": problem.angular_mean,
            "angular_scale": problem.angular_scale,
            "angular_mmd_bandwidth": bandwidth,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[calibration:{mode.value}] writing report", flush=True)
    args.output.write_text(
        json.dumps(arrays_to_python(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[calibration:{mode.value}] writing arrays", flush=True)
    np.savez_compressed(args.arrays_output, **output_arrays)
    parameter_arrays = {
        name: np.asarray(value)
        for name, value in flatten_generator_parameters(
            parameters, prefix="parameters"
        ).items()
    }
    np.savez_compressed(args.parameters_output, **parameter_arrays)
    if history is not None:
        _save_trace(args.trace_output, history)
    print(f"[calibration:{mode.value}] complete", flush=True)
    # Isolated workers intentionally bypass interpreter teardown after every
    # artifact has been closed. Some CPU XLA builds retain compilation threads
    # during normal shutdown, which can otherwise stall the orchestrator.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
