"""Train the compact native generator through both local scientific solvers."""

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

from manybody_completion.composition import (
    CompletionOptions,
    PhysicalParameters,
    run_local_completion,
)
from manybody_completion.config import load_yaml
from manybody_completion.generator import (
    EquivariantGeneratorConfig,
    apply_equivariant_generator,
    count_generator_parameters,
    flatten_generator_parameters,
    initialize_equivariant_generator,
    make_periodic_grid_anchors,
)
from manybody_completion.generator_training import (
    AdamOptions,
    GeneratorBatch,
    GeneratorObjectiveWeights,
    local_generator_objective,
    parameter_directional_derivative_sweep,
    train_equivariant_generator,
)
from manybody_completion.observables import PairBasis
from manybody_completion.projection import ProjectionOptions
from manybody_completion.relaxation import RelaxationOptions
from manybody_completion.scalar_training import arrays_to_python

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_problem(configuration: dict[str, Any]):
    dtype = jnp.float64 if configuration["dtype"] == "float64" else jnp.float32
    archive_path = REPO_ROOT / configuration["dataset"]
    with np.load(archive_path, allow_pickle=False) as archive:
        indices = np.asarray(configuration["target_indices"], dtype=np.int64)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError("target_indices must be a nonempty one-dimensional sequence")
        num_samples, num_replicas, num_particles = archive["coordinates"].shape[:3]
        if np.any(indices < 0) or np.any(indices >= num_samples):
            raise IndexError("target_indices contain an out-of-range dataset index")
        target_moments = jnp.asarray(archive["pair_moments"][indices], dtype=dtype)
        all_moments = jnp.asarray(archive["pair_moments"], dtype=dtype)
        box = jnp.asarray(archive["box"], dtype=dtype)
        basis = PairBasis(
            centers=jnp.asarray(archive["pair_basis_centers"], dtype=dtype),
            widths=jnp.asarray(archive["pair_basis_widths"], dtype=dtype),
        )

    batch_size = int(indices.size)
    grid_shape = tuple(int(value) for value in configuration["latent_anchors"]["grid_shape"])
    if grid_shape[0] * grid_shape[1] != num_particles:
        raise ValueError("latent grid_shape must contain exactly num_particles sites")

    model_config = EquivariantGeneratorConfig(**configuration["model"])
    key = jax.random.PRNGKey(configuration["seed"])
    parameter_key, anchor_key, latent_key = jax.random.split(key, 3)
    parameters = initialize_equivariant_generator(
        parameter_key,
        condition_dim=target_moments.shape[-1],
        config=model_config,
        dtype=dtype,
    )
    anchors = make_periodic_grid_anchors(
        anchor_key,
        batch_size=batch_size,
        num_replicas=num_replicas,
        grid_shape=grid_shape,
        box=box,
        jitter_scale=configuration["latent_anchors"]["jitter_scale"],
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
    if configuration["projection_moment_scales"] != "unit":
        raise ValueError("only unit projection_moment_scales are supported in this smoke test")
    moment_scales = jnp.ones((target_moments.shape[-1],), dtype=dtype)
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
    completion = CompletionOptions(
        physical=PhysicalParameters(**configuration["physical"]),
        relaxation=RelaxationOptions(**configuration["relaxation"]),
        projection=ProjectionOptions(**configuration["projection"]),
    )
    weights = GeneratorObjectiveWeights(**configuration["objective"])
    training = AdamOptions(**configuration["training"], jit_objective=True)
    return (
        parameters,
        batch,
        model_config,
        completion,
        weights,
        training,
        condition_mean,
        condition_scale,
    )


def _evaluate_stages(parameters, batch, model_config, completion):
    generated = jax.vmap(
        lambda anchors, latents, condition: apply_equivariant_generator(
            parameters, anchors, latents, condition, batch.box, model_config
        )
    )(batch.anchor_coordinates, batch.node_latents, batch.conditions)
    stages = jax.vmap(
        lambda coordinates, target: run_local_completion(
            coordinates,
            target,
            batch.box,
            batch.basis,
            batch.moment_scales,
            batch.basis_mask,
            completion,
        )
    )(generated, batch.target_moments)
    return generated, stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "native_generator_smoke.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "native_generator_smoke.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "native_generator_smoke_trace.csv",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "native_generator_smoke_outputs.npz",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "native_generator_smoke_parameters.npz",
    )
    args = parser.parse_args()

    configuration = load_yaml(args.config)
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    (
        initial_parameters,
        batch,
        model_config,
        completion,
        weights,
        training,
        condition_mean,
        condition_scale,
    ) = _build_problem(configuration)
    objective = partial(
        local_generator_objective,
        batch=batch,
        generator_config=model_config,
        completion_options=completion,
        weights=weights,
    )

    finite_difference = parameter_directional_derivative_sweep(
        objective,
        initial_parameters,
        jax.random.PRNGKey(configuration["finite_difference"]["direction_seed"]),
        configuration["finite_difference"]["epsilons"],
        jit_objective=True,
    )
    result = train_equivariant_generator(objective, initial_parameters, training)
    generated, stages = _evaluate_stages(result.parameters, batch, model_config, completion)

    initial_loss = result.history["loss"][0]
    initial_correction = result.history["total_correction_rms"][0]
    final_correction = result.final_metrics["total_correction_rms"]
    correction_reduction_fraction = 1.0 - final_correction / jnp.maximum(
        initial_correction, 1e-15
    )
    report = {
        "schema_version": 1,
        "backend": "local-jax",
        "jax_version": jax.__version__,
        "configuration": configuration,
        "parameter_count": count_generator_parameters(initial_parameters),
        "batch_shape": list(batch.anchor_coordinates.shape),
        "initial_loss": initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction_factor": initial_loss / jnp.maximum(result.final_loss, 1e-15),
        "initial_correction_rms": initial_correction,
        "final_correction_rms": final_correction,
        "correction_reduction_fraction": correction_reduction_fraction,
        "finite_difference": finite_difference,
        "final_metrics": result.final_metrics,
        "history": result.history,
        "condition_mean": condition_mean,
        "condition_scale": condition_scale,
    }
    serializable = arrays_to_python(report)

    for path in (args.output, args.trace_output, args.arrays_output, args.parameters_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    trace = serializable["history"]
    columns = list(trace)
    with args.trace_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *columns])
        writer.writeheader()
        for step in range(len(trace["loss"])):
            writer.writerow({"step": step, **{name: trace[name][step] for name in columns}})

    np.savez_compressed(
        args.arrays_output,
        anchor_coordinates=np.asarray(batch.anchor_coordinates),
        node_latents=np.asarray(batch.node_latents),
        conditions=np.asarray(batch.conditions),
        target_moments=np.asarray(batch.target_moments),
        generated_coordinates=np.asarray(generated),
        relaxed_coordinates=np.asarray(stages["relaxed_coordinates"]),
        projected_coordinates=np.asarray(stages["projected_coordinates"]),
        moments_initial=np.asarray(stages["moments_initial"]),
        moments_relaxed=np.asarray(stages["moments_relaxed"]),
        moments_projected=np.asarray(stages["moments_projected"]),
        box=np.asarray(batch.box),
        pair_basis_centers=np.asarray(batch.basis.centers),
        pair_basis_widths=np.asarray(batch.basis.widths),
    )
    parameter_arrays = {
        name: np.asarray(value)
        for name, value in flatten_generator_parameters(result.parameters).items()
    }
    np.savez_compressed(args.parameters_output, **parameter_arrays)

    console = {
        key: serializable[key]
        for key in (
            "parameter_count",
            "batch_shape",
            "initial_loss",
            "final_loss",
            "loss_reduction_factor",
            "initial_correction_rms",
            "final_correction_rms",
            "correction_reduction_fraction",
        )
    }
    console["best_gradient_relative_error"] = serializable["finite_difference"][
        "best_relative_error"
    ]
    console["final_metrics"] = serializable["final_metrics"]
    console["report"] = str(args.output)
    print(json.dumps(console, indent=2, sort_keys=True))

    acceptance = configuration["acceptance"]
    failures: list[str] = []
    if serializable["loss_reduction_factor"] < acceptance["minimum_loss_reduction_factor"]:
        failures.append("loss reduction is below the configured threshold")
    if (
        serializable["correction_reduction_fraction"]
        < acceptance["minimum_correction_reduction_fraction"]
    ):
        failures.append("correction burden did not decrease enough")
    if (
        serializable["final_metrics"]["moment_error_projected"]
        > acceptance["maximum_projected_moment_error"]
    ):
        failures.append("projected moment error is too large")
    if (
        serializable["finite_difference"]["best_relative_error"]
        > acceptance["maximum_gradient_relative_error"]
    ):
        failures.append("parameter gradient failed the finite-difference sweep")
    if (
        serializable["final_metrics"]["relaxation_converged"]
        < acceptance["minimum_final_relaxation_convergence_rate"]
    ):
        failures.append("not all final relaxation solves converged")
    if (
        serializable["final_metrics"]["projection_converged"]
        < acceptance["minimum_final_projection_convergence_rate"]
    ):
        failures.append("not all final projection solves converged")
    if (
        serializable["final_metrics"]["projection_rank_deficient"]
        > acceptance["maximum_rank_deficient_rate"]
    ):
        failures.append("a final projection was rank deficient")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
