from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.config import load_yaml
from manybody_completion.flow_experiment import build_flow_experiment_problem
from manybody_completion.flow_training import (
    fixed_flow_matching_objective,
    flow_parameter_directional_derivative_sweep,
    train_conditional_flow,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]


def _small_configuration():
    configuration = load_yaml(ROOT / "configs" / "flow_matching_toy.yaml")
    configuration["minibatching"]["num_epochs"] = 1
    configuration["model"]["hidden_dim"] = 8
    configuration["model"]["message_dim"] = 8
    return configuration


def test_flow_experiment_uses_train_only_normalization_and_fixed_batches():
    problem = build_flow_experiment_problem(_small_configuration(), ROOT)
    train_conditions = np.asarray(problem.train_batch.conditions)
    np.testing.assert_allclose(train_conditions.mean(axis=0), 0.0, atol=3e-14)
    assert problem.minibatch_indices.shape == (3, 4)
    assert all(batch.target_coordinates.shape[0] == 4 for batch in problem.minibatches)
    assert set(problem.split.train_indices).isdisjoint(problem.split.validation_indices)


def test_flow_parameter_gradient_matches_finite_differences():
    problem = build_flow_experiment_problem(_small_configuration(), ROOT)
    result = flow_parameter_directional_derivative_sweep(
        problem.initial_parameters,
        problem.minibatches[0],
        jax.random.PRNGKey(7),
        jax.random.PRNGKey(8),
        problem.flow_config,
        epsilons=(3e-3, 1e-3, 3e-4),
        jit_objective=False,
    )
    assert float(result["gradient_norm"]) > 1e-8
    assert float(result["best_relative_error"]) < 2e-5


def test_flow_training_updates_parameters_and_keeps_metrics_finite():
    problem = build_flow_experiment_problem(_small_configuration(), ROOT)
    initial_loss, _ = fixed_flow_matching_objective(
        problem.initial_parameters,
        problem.train_batch,
        jax.random.PRNGKey(200),
        problem.flow_config,
    )
    result = train_conditional_flow(
        problem.initial_parameters,
        problem.minibatches,
        jax.random.PRNGKey(201),
        problem.flow_config,
        problem.optimizer_options,
    )
    final_loss, metrics = fixed_flow_matching_objective(
        result.parameters,
        problem.train_batch,
        jax.random.PRNGKey(200),
        problem.flow_config,
    )
    assert np.isfinite(float(initial_loss))
    assert np.isfinite(float(final_loss))
    assert all(np.all(np.isfinite(np.asarray(value))) for value in metrics.values())
    initial_leaves = jax.tree_util.tree_leaves(problem.initial_parameters)
    final_leaves = jax.tree_util.tree_leaves(result.parameters)
    parameter_change = sum(
        float(jnp.sum((after - before) ** 2))
        for before, after in zip(initial_leaves, final_leaves)
    )
    assert parameter_change > 0.0


def test_exhaustive_particle_matching_removes_label_noise() -> None:
    from manybody_completion.flow_training import exhaustive_particle_match_targets
    from manybody_completion.geometry import translation_gauge_fixed_displacement

    box = jnp.ones((2,), dtype=jnp.float64)
    source = jnp.asarray(
        [[[[0.05, 0.10], [0.30, 0.15], [0.62, 0.52], [0.81, 0.77]]]],
        dtype=jnp.float64,
    )
    target = source[..., jnp.asarray([2, 0, 3, 1]), :]
    matched = exhaustive_particle_match_targets(source, target, box)
    matched_cost = jnp.mean(
        translation_gauge_fixed_displacement(source, matched, box) ** 2
    )
    identity_cost = jnp.mean(
        translation_gauge_fixed_displacement(source, target, box) ** 2
    )
    np.testing.assert_allclose(matched_cost, 0.0, atol=1e-15)
    assert float(matched_cost) < float(identity_cost)


def test_chunked_sampling_is_deterministic_and_has_requested_shape() -> None:
    from manybody_completion.flow_matching import FlowSamplingOptions
    from manybody_completion.flow_training import sample_flow_conditions_chunked

    problem = build_flow_experiment_problem(_small_configuration(), ROOT)
    options = FlowSamplingOptions(num_steps=2, integrator="euler")
    kwargs = dict(
        conditions=problem.validation_batch.conditions[:1],
        num_samples_per_condition=4,
        chunk_size=2,
        num_replicas=2,
        num_particles=6,
        box=problem.validation_batch.box,
        config=problem.flow_config,
        sampling_options=options,
        dtype=jnp.float64,
    )
    first = sample_flow_conditions_chunked(
        problem.initial_parameters,
        jax.random.PRNGKey(901),
        **kwargs,
    )
    second = sample_flow_conditions_chunked(
        problem.initial_parameters,
        jax.random.PRNGKey(901),
        **kwargs,
    )
    assert first.shape == (1, 4, 2, 6, 2)
    np.testing.assert_array_equal(first, second)


def test_homometric_problem_coalesces_one_exact_condition() -> None:
    configuration = load_yaml(ROOT / "configs" / "flow_matching_homometric.yaml")
    configuration["minibatching"]["num_epochs"] = 1
    problem = build_flow_experiment_problem(configuration, ROOT)
    conditions = np.asarray(problem.full_batch.conditions)
    np.testing.assert_array_equal(conditions, np.zeros_like(conditions))
    assert np.unique(conditions, axis=0).shape[0] == 1
