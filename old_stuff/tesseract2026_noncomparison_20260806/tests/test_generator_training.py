import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.generator_training import (
    AdamOptions,
    parameter_directional_derivative_sweep,
    train_equivariant_generator,
)

jax.config.update("jax_enable_x64", True)


_METRIC_NAMES = (
    "observed_loss",
    "preprojection_loss",
    "physical_loss",
    "correction_loss",
    "moment_error_initial",
    "moment_error_relaxed",
    "moment_error_projected",
    "physical_energy",
    "total_correction_rms",
    "relaxation_displacement",
    "projection_correction",
    "relaxation_converged",
    "projection_converged",
    "projection_rank_deficient",
    "loss_std",
)


def _quadratic_objective(parameters):
    vector = parameters["vector"]
    loss = 0.5 * jnp.sum(vector * vector)
    zero = jnp.asarray(0.0, dtype=vector.dtype)
    metrics = {name: zero for name in _METRIC_NAMES}
    metrics["correction_loss"] = loss
    metrics["total_correction_rms"] = jnp.sqrt(jnp.mean(vector * vector))
    metrics["relaxation_converged"] = jnp.asarray(1.0, dtype=vector.dtype)
    metrics["projection_converged"] = jnp.asarray(1.0, dtype=vector.dtype)
    return loss, metrics


def test_parameter_directional_derivative_utility_on_quadratic():
    parameters = {"vector": jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float64)}
    check = parameter_directional_derivative_sweep(
        _quadratic_objective,
        parameters,
        jax.random.PRNGKey(9),
        epsilons=(1e-3, 3e-4, 1e-4),
    )
    assert float(check["gradient_norm"]) > 0.0
    assert float(check["best_relative_error"]) < 1e-8


def test_explicit_adam_reduces_a_quadratic_objective():
    parameters = {"vector": jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float64)}
    result = train_equivariant_generator(
        _quadratic_objective,
        parameters,
        AdamOptions(num_steps=25, learning_rate=0.05, gradient_clip_norm=10.0),
    )
    assert float(result.final_loss) < 0.25 * float(result.history["loss"][0])
    assert np.linalg.norm(np.asarray(result.parameters["vector"])) < np.linalg.norm(
        np.asarray(parameters["vector"])
    )
