"""S3 scalar-generator objective, gradient checks, and deterministic training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from .composition import (
    CompletionOptions,
    periodic_correction_mse,
    run_local_completion,
    scalar_generator,
    stop_gradient_diagnostics,
)
from .observables import PairBasis


@dataclass(frozen=True)
class ScalarGeneratorProblem:
    """All fixed arrays defining one deterministic S3 problem."""

    base_coordinates: Array
    latent_displacements: Array
    target_moments: Array
    box: Array
    basis: PairBasis
    moment_scales: Array
    basis_mask: Array
    target_parameter: float | None = None


@dataclass(frozen=True)
class ScalarObjectiveWeights:
    """Weights for the minimal S3 objective."""

    observed: float = 1000.0
    correction: float = 1000.0


@dataclass(frozen=True)
class ScalarTrainingOptions:
    """Projected gradient-descent settings for the one-parameter model."""

    num_steps: int = 30
    learning_rate: float = 0.1
    gradient_clip: float = 1.0
    parameter_min: float = 0.0
    parameter_max: float = 1.2
    jit_objective: bool = True


@dataclass(frozen=True)
class ScalarTrainingResult:
    """Final parameter, metrics, and complete scalar optimization trace."""

    final_parameter: Array
    final_loss: Array
    final_gradient: Array
    final_metrics: Mapping[str, Array]
    history: Mapping[str, Array]


def _whitened_error(
    moments: Array,
    target: Array,
    scales: Array,
    mask: Array,
) -> Array:
    return mask * (moments - target) / scales


def local_s3_objective(
    parameter: Array | float,
    problem: ScalarGeneratorProblem,
    completion_options: CompletionOptions | None = None,
    weights: ScalarObjectiveWeights | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the paper's minimal scalar objective through both local solvers."""
    if completion_options is None:
        completion_options = CompletionOptions()
    if weights is None:
        weights = ScalarObjectiveWeights()

    initial = scalar_generator(
        parameter,
        problem.base_coordinates,
        problem.latent_displacements,
        problem.box,
    )
    stages = run_local_completion(
        initial_coordinates=initial,
        target_moments=problem.target_moments,
        box=problem.box,
        basis=problem.basis,
        moment_scales=problem.moment_scales,
        basis_mask=problem.basis_mask,
        options=completion_options,
    )

    initial_error = _whitened_error(
        stages["moments_initial"],
        problem.target_moments,
        problem.moment_scales,
        problem.basis_mask,
    )
    relaxed_error = _whitened_error(
        stages["moments_relaxed"],
        problem.target_moments,
        problem.moment_scales,
        problem.basis_mask,
    )
    projected_error = _whitened_error(
        stages["moments_projected"],
        problem.target_moments,
        problem.moment_scales,
        problem.basis_mask,
    )
    observed_loss = weights.observed * jnp.sum(projected_error * projected_error)
    correction_mse = periodic_correction_mse(
        stages["projected_coordinates"],
        stages["initial_coordinates"],
        problem.box,
    )
    correction_loss = weights.correction * correction_mse
    loss = observed_loss + correction_loss

    relaxation_diagnostics = stages["relaxation"]
    projection_diagnostics = stages["projection"]
    metrics = {
        "observed_loss": observed_loss,
        "correction_loss": correction_loss,
        "moment_error_initial": jnp.linalg.norm(initial_error),
        "moment_error_relaxed": jnp.linalg.norm(relaxed_error),
        "moment_error_projected": jnp.linalg.norm(projected_error),
        "total_correction_rms": jnp.sqrt(correction_mse),
        "relaxation_displacement": relaxation_diagnostics["prox_displacement"],
        "projection_correction": projection_diagnostics["correction_norm"],
        "physical_energy_initial": relaxation_diagnostics["physical_energy_before"],
        "physical_energy_relaxed": relaxation_diagnostics["physical_energy_after"],
        "projection_residual": projection_diagnostics["constraint_residual"],
        "projection_effective_rank": projection_diagnostics["effective_rank"],
        "relaxation_converged": relaxation_diagnostics["converged"],
        "projection_converged": projection_diagnostics["converged"],
        "projection_rank_deficient": projection_diagnostics["rank_deficient"],
    }
    return loss, stop_gradient_diagnostics(metrics)


def scalar_gradient_sweep(
    objective: Callable[[Array], tuple[Array, Mapping[str, Array]]],
    parameter: Array | float,
    epsilons: Sequence[float] = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    *,
    jit_objective: bool = True,
) -> dict[str, Array]:
    """Compare the composed scalar gradient with centered finite differences."""
    parameter = jnp.asarray(parameter)

    def scalar_only(value: Array) -> Array:
        return objective(value)[0]

    scalar_fn = jax.jit(scalar_only) if jit_objective else scalar_only
    gradient_fn = jax.jit(jax.grad(scalar_only)) if jit_objective else jax.grad(scalar_only)
    autodiff = gradient_fn(parameter)
    epsilon_array = jnp.asarray(tuple(epsilons), dtype=parameter.dtype)
    finite_differences = jnp.stack(
        [
            (scalar_fn(parameter + epsilon) - scalar_fn(parameter - epsilon))
            / (2.0 * epsilon)
            for epsilon in epsilon_array
        ]
    )
    absolute_errors = jnp.abs(finite_differences - autodiff)
    relative_errors = absolute_errors / jnp.maximum(jnp.abs(autodiff), 1e-15)
    return {
        "autodiff": autodiff,
        "epsilons": epsilon_array,
        "finite_differences": finite_differences,
        "absolute_errors": absolute_errors,
        "relative_errors": relative_errors,
        "best_absolute_error": jnp.min(absolute_errors),
        "best_relative_error": jnp.min(relative_errors),
    }


def _validate_training_options(options: ScalarTrainingOptions) -> None:
    if options.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if options.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if options.gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    if options.parameter_min >= options.parameter_max:
        raise ValueError("parameter_min must be smaller than parameter_max")


def train_scalar_parameter(
    objective: Callable[[Array], tuple[Array, Mapping[str, Array]]],
    initial_parameter: Array | float,
    options: ScalarTrainingOptions | None = None,
) -> ScalarTrainingResult:
    """Train a scalar with projected gradient descent through an arbitrary pipeline.

    The objective contract is deliberately solver-agnostic.  Local JAX solvers
    and Tesseract-backed solvers can therefore share this training loop.
    """
    if options is None:
        options = ScalarTrainingOptions()
    _validate_training_options(options)

    parameter = jnp.asarray(initial_parameter)
    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    evaluator = jax.jit(value_and_grad) if options.jit_objective else value_and_grad

    history: dict[str, list[Array]] = {
        "parameter": [],
        "loss": [],
        "gradient": [],
        "moment_error_initial": [],
        "moment_error_relaxed": [],
        "moment_error_projected": [],
        "total_correction_rms": [],
        "relaxation_displacement": [],
        "projection_correction": [],
    }

    for _ in range(options.num_steps):
        (loss, metrics), gradient = evaluator(parameter)
        history["parameter"].append(parameter)
        history["loss"].append(loss)
        history["gradient"].append(gradient)
        for key in history:
            if key not in {"parameter", "loss", "gradient"}:
                history[key].append(metrics[key])
        clipped_gradient = jnp.clip(
            gradient,
            -options.gradient_clip,
            options.gradient_clip,
        )
        parameter = jnp.clip(
            parameter - options.learning_rate * clipped_gradient,
            options.parameter_min,
            options.parameter_max,
        )

    (final_loss, final_metrics), final_gradient = evaluator(parameter)
    history["parameter"].append(parameter)
    history["loss"].append(final_loss)
    history["gradient"].append(final_gradient)
    for key in history:
        if key not in {"parameter", "loss", "gradient"}:
            history[key].append(final_metrics[key])

    stacked_history = {key: jnp.stack(values) for key, values in history.items()}
    return ScalarTrainingResult(
        final_parameter=parameter,
        final_loss=final_loss,
        final_gradient=final_gradient,
        final_metrics=final_metrics,
        history=stacked_history,
    )


def arrays_to_python(tree: Any) -> Any:
    """Convert a JAX/NumPy pytree into JSON-compatible Python values."""
    if isinstance(tree, Mapping):
        return {key: arrays_to_python(value) for key, value in tree.items()}
    if isinstance(tree, (tuple, list)):
        return [arrays_to_python(value) for value in tree]
    value = jax.device_get(tree)
    if hasattr(value, "shape"):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value
