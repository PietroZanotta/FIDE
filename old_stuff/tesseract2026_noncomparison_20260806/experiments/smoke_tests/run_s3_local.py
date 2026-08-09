"""Run S3 locally: compose both solvers and recover the scalar generator parameter."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from manybody_completion.composition import (
    CompletionOptions,
    PhysicalParameters,
)
from manybody_completion.observables import PairBasis
from manybody_completion.projection import ProjectionOptions
from manybody_completion.relaxation import RelaxationOptions
from manybody_completion.scalar_training import (
    ScalarGeneratorProblem,
    ScalarObjectiveWeights,
    ScalarTrainingOptions,
    arrays_to_python,
    local_s3_objective,
    scalar_gradient_sweep,
    train_scalar_parameter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_configuration(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build_components(configuration: dict):
    archive = REPO_ROOT / configuration["problem_archive"]
    with np.load(archive, allow_pickle=False) as data:
        dtype = jnp.float64 if configuration["dtype"] == "float64" else jnp.float32
        target = jnp.asarray(data["s3_target_moments"], dtype=dtype)
        problem = ScalarGeneratorProblem(
            base_coordinates=jnp.asarray(data["s3_base_coordinates"], dtype=dtype),
            latent_displacements=jnp.asarray(data["s3_latent_displacements"], dtype=dtype),
            target_moments=target,
            box=jnp.asarray(data["box"], dtype=dtype),
            basis=PairBasis(
                centers=jnp.asarray(data["s3_basis_centers"], dtype=dtype),
                widths=jnp.asarray(data["s3_basis_widths"], dtype=dtype),
            ),
            moment_scales=jnp.ones_like(target),
            basis_mask=jnp.ones_like(target),
            target_parameter=float(data["s3_a_star"]),
        )
        initial_parameter = jnp.asarray(data["s3_a_initial"], dtype=dtype)

    completion = CompletionOptions(
        physical=PhysicalParameters(**configuration["physical"]),
        relaxation=RelaxationOptions(**configuration["relaxation"]),
        projection=ProjectionOptions(**configuration["projection"]),
    )
    weights = ScalarObjectiveWeights(**configuration["objective"])
    training = ScalarTrainingOptions(
        **configuration["training"],
        jit_objective=True,
    )
    return problem, initial_parameter, completion, weights, training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "s3_scalar.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "s3_scalar_local.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "s3_scalar_trace.csv",
    )
    args = parser.parse_args()

    configuration = _load_configuration(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    problem, initial_parameter, completion, weights, training = _build_components(
        configuration
    )
    objective = partial(
        local_s3_objective,
        problem=problem,
        completion_options=completion,
        weights=weights,
    )

    gradient_check = scalar_gradient_sweep(
        objective,
        initial_parameter,
        configuration["gradient_check"]["epsilons"],
        jit_objective=True,
    )
    result = train_scalar_parameter(objective, initial_parameter, training)

    initial_loss = result.history["loss"][0]
    final_loss = result.final_loss
    target_parameter = jnp.asarray(problem.target_parameter, dtype=result.final_parameter.dtype)
    summary = {
        "schema_version": 1,
        "backend": "local-jax",
        "jax_version": jax.__version__,
        "configuration": configuration,
        "initial_parameter": initial_parameter,
        "target_parameter": target_parameter,
        "final_parameter": result.final_parameter,
        "parameter_error": jnp.abs(result.final_parameter - target_parameter),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_factor": initial_loss / jnp.maximum(final_loss, 1e-15),
        "initial_gradient": result.history["gradient"][0],
        "final_gradient": result.final_gradient,
        "gradient_check": gradient_check,
        "final_metrics": result.final_metrics,
        "history": result.history,
    }
    serializable = arrays_to_python(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace = serializable["history"]
    trace_columns = list(trace)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with args.trace_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *trace_columns])
        writer.writeheader()
        for step in range(len(trace["loss"])):
            writer.writerow(
                {
                    "step": step,
                    **{column: trace[column][step] for column in trace_columns},
                }
            )

    console_summary = {
        key: serializable[key]
        for key in (
            "initial_parameter",
            "target_parameter",
            "final_parameter",
            "parameter_error",
            "initial_loss",
            "final_loss",
            "loss_reduction_factor",
            "initial_gradient",
            "final_gradient",
        )
    }
    console_summary["best_gradient_relative_error"] = serializable["gradient_check"][
        "best_relative_error"
    ]
    console_summary["final_metrics"] = serializable["final_metrics"]
    console_summary["report"] = str(args.output)
    console_summary["trace"] = str(args.trace_output)
    print(json.dumps(console_summary, indent=2, sort_keys=True))

    acceptance = configuration["acceptance"]
    failures = []
    if not np.isfinite(serializable["initial_gradient"]) or serializable["initial_gradient"] == 0:
        failures.append("the initial gradient is non-finite or zero")
    if serializable["parameter_error"] > acceptance["maximum_parameter_error"]:
        failures.append("the trained parameter did not recover the target neighborhood")
    if serializable["loss_reduction_factor"] < acceptance["minimum_loss_reduction_factor"]:
        failures.append("the outer loss did not decrease enough")
    if (
        serializable["final_metrics"]["moment_error_projected"]
        > acceptance["maximum_projected_moment_error"]
    ):
        failures.append("the projected moment error is too large")
    if (
        serializable["gradient_check"]["best_relative_error"]
        > acceptance["maximum_gradient_relative_error"]
    ):
        failures.append("the composed gradient failed the finite-difference sweep")
    if not serializable["final_metrics"]["relaxation_converged"]:
        failures.append("the S3 relaxation did not reach its configured stopping criterion")
    if not serializable["final_metrics"]["projection_converged"]:
        failures.append("the S3 projection did not reach its configured stopping criterion")
    if serializable["final_metrics"]["projection_rank_deficient"]:
        failures.append("the S3 projection unexpectedly became rank deficient")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
