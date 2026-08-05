from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.composition import CompletionOptions, PhysicalParameters, scalar_generator
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.problem_instances import build_smoke_problem_instances
from manybody_completion.projection import ProjectionOptions
from manybody_completion.relaxation import RelaxationOptions
from manybody_completion.scalar_training import (
    ScalarGeneratorProblem,
    ScalarObjectiveWeights,
    ScalarTrainingOptions,
    local_s3_objective,
    scalar_gradient_sweep,
    train_scalar_parameter,
)

jax.config.update("jax_enable_x64", True)


def _s3_setup():
    arrays, _ = build_smoke_problem_instances()
    target = jnp.asarray(arrays["s3_target_moments"])
    problem = ScalarGeneratorProblem(
        base_coordinates=jnp.asarray(arrays["s3_base_coordinates"]),
        latent_displacements=jnp.asarray(arrays["s3_latent_displacements"]),
        target_moments=target,
        box=jnp.asarray(arrays["box"]),
        basis=PairBasis(
            centers=jnp.asarray(arrays["s3_basis_centers"]),
            widths=jnp.asarray(arrays["s3_basis_widths"]),
        ),
        moment_scales=jnp.ones_like(target),
        basis_mask=jnp.ones_like(target),
        target_parameter=float(arrays["s3_a_star"]),
    )
    completion = CompletionOptions(
        physical=PhysicalParameters(r0=0.08, kappa=30.0, prox_strength=0.05),
        relaxation=RelaxationOptions(
            num_steps=64,
            step_size=0.05,
            tolerance=1e-3,
            max_update_norm=0.04,
            line_search_steps=12,
        ),
        projection=ProjectionOptions(
            num_steps=20,
            tolerance=1e-6,
            kkt_tolerance=2e-4,
            ridge=1e-8,
            max_step_norm=0.05,
            max_correction_norm=0.25,
            line_search_steps=8,
        ),
    )
    return problem, jnp.asarray(arrays["s3_a_initial"]), completion


def test_s3_target_is_generated_at_known_scalar():
    problem, _, _ = _s3_setup()
    coordinates = scalar_generator(
        problem.target_parameter,
        problem.base_coordinates,
        problem.latent_displacements,
        problem.box,
    )
    moments = ensemble_pair_moments(coordinates, problem.box, problem.basis)
    np.testing.assert_allclose(moments, problem.target_moments, atol=2e-12, rtol=2e-12)


def test_s3_composed_gradient_and_scalar_recovery():
    problem, initial_parameter, completion = _s3_setup()
    objective = partial(
        local_s3_objective,
        problem=problem,
        completion_options=completion,
        weights=ScalarObjectiveWeights(observed=1000.0, correction=1000.0),
    )

    gradient_check = scalar_gradient_sweep(objective, initial_parameter)
    assert np.isfinite(float(gradient_check["autodiff"]))
    assert abs(float(gradient_check["autodiff"])) > 1e-3
    assert float(gradient_check["best_relative_error"]) < 1e-6

    result = train_scalar_parameter(
        objective,
        initial_parameter,
        ScalarTrainingOptions(
            num_steps=30,
            learning_rate=0.1,
            gradient_clip=1.0,
            parameter_min=0.0,
            parameter_max=1.2,
        ),
    )
    initial_loss = float(result.history["loss"][0])
    final_loss = float(result.final_loss)
    assert final_loss < initial_loss / 10.0
    assert abs(float(result.final_parameter) - problem.target_parameter) < 0.01
    assert float(result.final_metrics["moment_error_projected"]) < 1e-5
    assert bool(result.final_metrics["relaxation_converged"])
    assert bool(result.final_metrics["projection_converged"])
    assert not bool(result.final_metrics["projection_rank_deficient"])
