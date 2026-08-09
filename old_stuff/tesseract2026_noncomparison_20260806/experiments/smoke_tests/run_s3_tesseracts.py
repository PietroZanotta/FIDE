"""Train the S3 scalar through both built Tesseract containers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from manybody_completion.composition import periodic_correction_mse, scalar_generator
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.scalar_training import (
    ScalarTrainingOptions,
    arrays_to_python,
    scalar_gradient_sweep,
    train_scalar_parameter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        default=REPO_ROOT / "artifacts" / "s3_scalar_tesseracts.json",
    )
    args = parser.parse_args()

    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    jax.config.update("jax_enable_x64", configuration["dtype"] == "float64")
    dtype = jnp.float64 if configuration["dtype"] == "float64" else jnp.float32
    with np.load(REPO_ROOT / configuration["problem_archive"], allow_pickle=False) as data:
        base = jnp.asarray(data["s3_base_coordinates"], dtype=dtype)
        latent = jnp.asarray(data["s3_latent_displacements"], dtype=dtype)
        target = jnp.asarray(data["s3_target_moments"], dtype=dtype)
        box = jnp.asarray(data["box"], dtype=dtype)
        target_parameter = jnp.asarray(data["s3_a_star"], dtype=dtype)
        initial_parameter = jnp.asarray(data["s3_a_initial"], dtype=dtype)
        basis = PairBasis(
            centers=jnp.asarray(data["s3_basis_centers"], dtype=dtype),
            widths=jnp.asarray(data["s3_basis_widths"], dtype=dtype),
        )
    scales = jnp.ones_like(target)
    mask = jnp.ones_like(target)

    relaxation_inputs = {
        "box": box,
        **{
            key: jnp.asarray(value, dtype=dtype)
            for key, value in configuration["physical"].items()
        },
        **configuration["relaxation"],
    }
    projection_inputs = {
        "target_moments": target,
        "box": box,
        "basis_centers": basis.centers,
        "basis_widths": basis.widths,
        "moment_scales": scales,
        "basis_mask": mask,
        **configuration["projection"],
    }
    observed_weight = configuration["objective"]["observed"]
    correction_weight = configuration["objective"]["correction"]
    training = ScalarTrainingOptions(
        **configuration["training"],
        jit_objective=False,
    )

    relaxation_tesseract = Tesseract.from_image("manybody-physical-relaxation")
    projection_tesseract = Tesseract.from_image("manybody-moment-projection")
    # Tesseract Core 1.10.0 hardcodes debug=True in from_image(), which adds
    # unnecessary debugpy ports and can fail under Docker Desktop forwarding.
    relaxation_tesseract._spawn_config["debug"] = False
    projection_tesseract._spawn_config["debug"] = False

    with relaxation_tesseract, projection_tesseract:

        def objective(parameter):
            initial = scalar_generator(parameter, base, latent, box)
            current_relaxation_inputs = dict(relaxation_inputs)
            current_relaxation_inputs["coordinates"] = initial
            relaxation_output = apply_tesseract(
                relaxation_tesseract,
                current_relaxation_inputs,
            )
            current_projection_inputs = dict(projection_inputs)
            current_projection_inputs["coordinates"] = relaxation_output[
                "relaxed_coordinates"
            ]
            projection_output = apply_tesseract(
                projection_tesseract,
                current_projection_inputs,
            )
            projected = projection_output["projected_coordinates"]
            initial_moments = ensemble_pair_moments(initial, box, basis)
            relaxed_moments = ensemble_pair_moments(
                relaxation_output["relaxed_coordinates"], box, basis
            )
            projected_moments = ensemble_pair_moments(projected, box, basis)
            initial_error = mask * (initial_moments - target) / scales
            relaxed_error = mask * (relaxed_moments - target) / scales
            projected_error = mask * (projected_moments - target) / scales
            correction_mse = periodic_correction_mse(projected, initial, box)
            observed_loss = observed_weight * jnp.sum(projected_error**2)
            correction_loss = correction_weight * correction_mse
            loss = observed_loss + correction_loss
            metrics = {
                "observed_loss": observed_loss,
                "correction_loss": correction_loss,
                "moment_error_initial": jnp.linalg.norm(initial_error),
                "moment_error_relaxed": jnp.linalg.norm(relaxed_error),
                "moment_error_projected": jnp.linalg.norm(projected_error),
                "total_correction_rms": jnp.sqrt(correction_mse),
                "relaxation_displacement": relaxation_output["prox_displacement"],
                "projection_correction": projection_output["correction_norm"],
                "projection_residual": projection_output["constraint_residual"],
                "relaxation_converged": relaxation_output["converged"],
                "projection_converged": projection_output["converged"],
                "projection_rank_deficient": projection_output["rank_deficient"],
            }
            return loss, metrics

        gradient_check = scalar_gradient_sweep(
            objective,
            initial_parameter,
            configuration["gradient_check"]["epsilons"],
            jit_objective=False,
        )
        result = train_scalar_parameter(objective, initial_parameter, training)

    summary = {
        "schema_version": 1,
        "backend": "tesseract-jax",
        "jax_version": jax.__version__,
        "initial_parameter": initial_parameter,
        "target_parameter": target_parameter,
        "final_parameter": result.final_parameter,
        "parameter_error": jnp.abs(result.final_parameter - target_parameter),
        "initial_loss": result.history["loss"][0],
        "final_loss": result.final_loss,
        "loss_reduction_factor": result.history["loss"][0]
        / jnp.maximum(result.final_loss, 1e-15),
        "initial_gradient": result.history["gradient"][0],
        "final_gradient": result.final_gradient,
        "gradient_check": gradient_check,
        "final_metrics": result.final_metrics,
        "history": result.history,
    }
    serializable = arrays_to_python(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
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
    print(json.dumps(console_summary, indent=2, sort_keys=True))

    acceptance = configuration["acceptance"]
    failures = []
    if not np.isfinite(serializable["initial_gradient"]) or serializable["initial_gradient"] == 0:
        failures.append("the initial Tesseract gradient is non-finite or zero")
    if serializable["parameter_error"] > acceptance["maximum_parameter_error"]:
        failures.append("the Tesseract composition did not recover the target neighborhood")
    if serializable["loss_reduction_factor"] < acceptance["minimum_loss_reduction_factor"]:
        failures.append("the Tesseract-composed loss did not decrease enough")
    if (
        serializable["final_metrics"]["moment_error_projected"]
        > acceptance["maximum_projected_moment_error"]
    ):
        failures.append("the Tesseract-composed projected moment error is too large")
    if (
        serializable["gradient_check"]["best_relative_error"]
        > acceptance["maximum_gradient_relative_error"]
    ):
        failures.append("the Tesseract-composed gradient failed finite differences")
    if serializable["final_metrics"]["projection_rank_deficient"]:
        failures.append("the Tesseract projection unexpectedly became rank deficient")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
