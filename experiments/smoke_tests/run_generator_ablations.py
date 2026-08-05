"""Train and compare Base, Post-hoc, Relax-E2E, and Full-E2E modes."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.ablation import AblationMode, get_ablation_spec
from manybody_completion.config import load_yaml
from manybody_completion.generator import (
    count_generator_parameters,
    flatten_generator_parameters,
)
from manybody_completion.generator_experiment import build_generator_experiment_problem
from manybody_completion.generator_training import (
    GeneratorTrainingResult,
    ablation_generator_objective,
    ablation_training_objective,
    evaluate_generator_completion,
    parameter_directional_derivative_sweep,
    train_equivariant_generator,
)
from manybody_completion.scalar_training import arrays_to_python

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINED_MODES = (
    AblationMode.BASE,
    AblationMode.RELAX_E2E,
    AblationMode.FULL_E2E,
)
ALL_MODES = (
    AblationMode.BASE,
    AblationMode.POST_HOC,
    AblationMode.RELAX_E2E,
    AblationMode.FULL_E2E,
)


def _tree_l2_distance(left: object, right: object) -> jax.Array:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    if left_structure != right_structure:
        raise ValueError("parameter trees have different structures")
    return jnp.sqrt(
        sum(jnp.sum((a - b) ** 2) for a, b in zip(left_leaves, right_leaves))
    )


def _build_objective(problem, mode: AblationMode):
    return partial(
        ablation_training_objective,
        batch=problem.batch,
        generator_config=problem.model_config,
        completion_options=problem.completion_options,
        weights=problem.objective_weights,
        mode=mode,
    )




def _summarize_result(
    mode: AblationMode,
    result: GeneratorTrainingResult,
    initial_parameters: object,
    initial_evaluation_metrics: dict[str, jax.Array],
    final_evaluation_metrics: dict[str, jax.Array],
) -> dict[str, Any]:
    initial_loss = result.history["loss"][0]
    initial_correction = initial_evaluation_metrics["total_correction_rms"]
    final_correction = final_evaluation_metrics["total_correction_rms"]
    return {
        "mode": mode.value,
        "training_stage": get_ablation_spec(mode).training_stage.value,
        "serving_stage": get_ablation_spec(mode).serving_stage.value,
        "initial_loss": initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction_factor": initial_loss / jnp.maximum(result.final_loss, 1e-15),
        "initial_total_correction_rms": initial_correction,
        "final_total_correction_rms": final_correction,
        "total_correction_reduction_fraction": 1.0
        - final_correction / jnp.maximum(initial_correction, 1e-15),
        "parameter_displacement": _tree_l2_distance(
            result.parameters, initial_parameters
        ),
        "final_metrics": final_evaluation_metrics,
        "training_final_metrics": result.final_metrics,
    }


def _save_trace(path: Path, results: dict[AblationMode, GeneratorTrainingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(results[AblationMode.BASE].history.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "step", *metric_names])
        writer.writeheader()
        for mode in ALL_MODES:
            history = arrays_to_python(results[mode].history)
            for step in range(len(history["loss"])):
                writer.writerow(
                    {
                        "mode": mode.value,
                        "step": step,
                        **{name: history[name][step] for name in metric_names},
                    }
                )


def _save_arrays_and_parameters(
    arrays_path: Path,
    parameters_path: Path,
    problem,
    results: dict[AblationMode, GeneratorTrainingResult],
) -> None:
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    parameters_path.parent.mkdir(parents=True, exist_ok=True)
    array_payload: dict[str, np.ndarray] = {
        "anchor_coordinates": np.asarray(problem.batch.anchor_coordinates),
        "node_latents": np.asarray(problem.batch.node_latents),
        "conditions": np.asarray(problem.batch.conditions),
        "target_moments": np.asarray(problem.batch.target_moments),
        "box": np.asarray(problem.batch.box),
        "pair_basis_centers": np.asarray(problem.batch.basis.centers),
        "pair_basis_widths": np.asarray(problem.batch.basis.widths),
    }
    parameter_payload: dict[str, np.ndarray] = {}
    for mode in ALL_MODES:
        parameters = results[mode].parameters
        generated, stages = evaluate_generator_completion(
            parameters,
            problem.batch,
            problem.model_config,
            problem.completion_options,
        )
        prefix = mode.value
        array_payload[f"{prefix}.generated_coordinates"] = np.asarray(generated)
        array_payload[f"{prefix}.relaxed_coordinates"] = np.asarray(
            stages["relaxed_coordinates"]
        )
        array_payload[f"{prefix}.projected_coordinates"] = np.asarray(
            stages["projected_coordinates"]
        )
        array_payload[f"{prefix}.moments_initial"] = np.asarray(stages["moments_initial"])
        array_payload[f"{prefix}.moments_relaxed"] = np.asarray(stages["moments_relaxed"])
        array_payload[f"{prefix}.moments_projected"] = np.asarray(
            stages["moments_projected"]
        )
        for name, value in flatten_generator_parameters(
            parameters, prefix=f"{prefix}.parameters"
        ).items():
            parameter_payload[name] = np.asarray(value)
    np.savez_compressed(arrays_path, **array_payload)
    np.savez_compressed(parameters_path, **parameter_payload)


def _validate_acceptance(report: dict[str, Any], acceptance: dict[str, Any]) -> None:
    failures: list[str] = []
    modes = report["modes"]
    for mode in ALL_MODES:
        summary = modes[mode.value]
        if summary["loss_reduction_factor"] < acceptance["minimum_loss_reduction_factor"]:
            failures.append(f"{mode.value}: loss reduction is below threshold")
        if not np.isfinite(summary["final_loss"]):
            failures.append(f"{mode.value}: final loss is non-finite")
        metrics = summary["final_metrics"]
        if metrics["relaxation_converged"] < acceptance["minimum_relaxation_convergence_rate"]:
            failures.append(f"{mode.value}: relaxation convergence rate is too low")
        if metrics["projection_converged"] < acceptance["minimum_projection_convergence_rate"]:
            failures.append(f"{mode.value}: projection convergence rate is too low")
        if metrics["projection_rank_deficient"] > acceptance["maximum_rank_deficient_rate"]:
            failures.append(f"{mode.value}: projection rank-deficient rate is too high")
    if report["base_post_hoc_parameter_distance"] > acceptance[
        "maximum_base_post_hoc_parameter_distance"
    ]:
        failures.append("Base and Post-hoc no longer share an identical trained generator")
    if modes["full_e2e"]["final_metrics"]["moment_error_projected"] > acceptance[
        "maximum_full_projected_moment_error"
    ]:
        failures.append("Full-E2E projected moment error is too large")
    if report["full_vs_post_hoc_correction_ratio"] > acceptance[
        "maximum_full_vs_post_hoc_correction_ratio"
    ]:
        failures.append("Full-E2E correction burden is unexpectedly worse than Post-hoc")
    for mode, check in report["finite_difference"].items():
        if check["best_relative_error"] > acceptance["maximum_gradient_relative_error"]:
            failures.append(f"{mode}: parameter gradient failed finite differences")
    if failures:
        raise SystemExit("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "generator_ablation_smoke.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "generator_ablation_smoke.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "generator_ablation_smoke_trace.csv",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "generator_ablation_smoke_outputs.npz",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "generator_ablation_smoke_parameters.npz",
    )
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem = build_generator_experiment_problem(configuration, REPO_ROOT)

    finite_difference: dict[str, object] = {}
    finite_difference_modes = [
        AblationMode.parse(value)
        for value in configuration["finite_difference"]["modes"]
    ]
    for index, mode in enumerate(finite_difference_modes):
        finite_difference[mode.value] = parameter_directional_derivative_sweep(
            _build_objective(problem, mode),
            problem.initial_parameters,
            jax.random.PRNGKey(
                int(configuration["finite_difference"]["direction_seed"]) + index
            ),
            configuration["finite_difference"]["epsilons"],
            jit_objective=True,
        )

    results: dict[AblationMode, GeneratorTrainingResult] = {}
    for mode in TRAINED_MODES:
        results[mode] = train_equivariant_generator(
            _build_objective(problem, mode),
            problem.initial_parameters,
            problem.training_options,
        )
    # By definition Post-hoc has the same native optimization as Base. Reuse the
    # exact trained parameters/history and vary only the serving-stage metrics.
    base_result = results[AblationMode.BASE]
    _, post_hoc_metrics = _build_objective(problem, AblationMode.POST_HOC)(
        base_result.parameters
    )
    results[AblationMode.POST_HOC] = GeneratorTrainingResult(
        parameters=base_result.parameters,
        final_loss=base_result.final_loss,
        final_metrics=post_hoc_metrics,
        history=base_result.history,
    )

    initial_evaluation_metrics = {
        mode: ablation_generator_objective(
            problem.initial_parameters,
            problem.batch,
            problem.model_config,
            problem.completion_options,
            problem.objective_weights,
            mode,
        )[1]
        for mode in ALL_MODES
    }
    final_evaluation_metrics = {
        mode: ablation_generator_objective(
            results[mode].parameters,
            problem.batch,
            problem.model_config,
            problem.completion_options,
            problem.objective_weights,
            mode,
        )[1]
        for mode in ALL_MODES
    }
    summaries = {
        mode.value: _summarize_result(
            mode,
            results[mode],
            problem.initial_parameters,
            initial_evaluation_metrics[mode],
            final_evaluation_metrics[mode],
        )
        for mode in ALL_MODES
    }
    post_hoc_correction = summaries["post_hoc"]["final_total_correction_rms"]
    full_correction = summaries["full_e2e"]["final_total_correction_rms"]
    report = {
        "schema_version": 1,
        "backend": "local-jax",
        "jax_version": jax.__version__,
        "configuration": configuration,
        "parameter_count": count_generator_parameters(problem.initial_parameters),
        "batch_shape": list(problem.batch.anchor_coordinates.shape),
        "selected_indices": problem.selected_indices,
        "condition_mean": problem.condition_mean,
        "condition_scale": problem.condition_scale,
        "modes": summaries,
        "finite_difference": finite_difference,
        "base_post_hoc_parameter_distance": _tree_l2_distance(
            results[AblationMode.BASE].parameters,
            results[AblationMode.POST_HOC].parameters,
        ),
        "full_vs_post_hoc_correction_ratio": full_correction
        / jnp.maximum(post_hoc_correction, 1e-15),
        "full_vs_post_hoc_correction_improvement_fraction": 1.0
        - full_correction / jnp.maximum(post_hoc_correction, 1e-15),
    }
    serializable = arrays_to_python(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _save_trace(args.trace_output, results)
    _save_arrays_and_parameters(
        args.arrays_output,
        args.parameters_output,
        problem,
        results,
    )

    console = {
        "parameter_count": serializable["parameter_count"],
        "batch_shape": serializable["batch_shape"],
        "full_vs_post_hoc_correction_ratio": serializable[
            "full_vs_post_hoc_correction_ratio"
        ],
        "modes": {
            mode: {
                "initial_loss": summary["initial_loss"],
                "final_loss": summary["final_loss"],
                "loss_reduction_factor": summary["loss_reduction_factor"],
                "final_total_correction_rms": summary["final_total_correction_rms"],
                "moment_error_initial": summary["final_metrics"][
                    "moment_error_initial"
                ],
                "moment_error_projected": summary["final_metrics"][
                    "moment_error_projected"
                ],
            }
            for mode, summary in serializable["modes"].items()
        },
        "report": str(args.output),
    }
    print(json.dumps(console, indent=2, sort_keys=True))
    _validate_acceptance(serializable, configuration["acceptance"])


if __name__ == "__main__":
    main()
