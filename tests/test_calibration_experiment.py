from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.ablation import AblationMode
from manybody_completion.calibration_experiment import (
    build_calibration_experiment_problem,
    make_minibatch_schedule,
    stratified_train_validation_split,
)
from manybody_completion.config import load_yaml
from manybody_completion.generator_training import (
    ablation_training_objective,
    subset_generator_batch,
    train_equivariant_generator_minibatches,
)

jax.config.update("jax_enable_x64", True)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stratified_split_is_balanced_disjoint_and_reproducible():
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int32)
    first = stratified_train_validation_split(
        labels, validation_per_regime=2, seed=17
    )
    second = stratified_train_validation_split(
        labels, validation_per_regime=2, seed=17
    )
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    assert np.intersect1d(first.train_indices, first.validation_indices).size == 0
    assert np.bincount(labels[first.train_indices]).tolist() == [6, 6]
    assert np.bincount(labels[first.validation_indices]).tolist() == [2, 2]


def test_minibatch_schedule_covers_each_training_sample_once_per_epoch():
    indices = np.arange(12, dtype=np.int32)
    schedule = make_minibatch_schedule(
        indices, batch_size=4, num_epochs=3, seed=9
    )
    assert schedule.shape == (9, 4)
    for epoch in range(3):
        values = schedule[epoch * 3 : (epoch + 1) * 3].reshape(-1)
        np.testing.assert_array_equal(np.sort(values), indices)


def test_calibration_builder_uses_training_only_normalization_and_hides_labels():
    configuration = load_yaml(REPO_ROOT / "configs" / "calibration_ablations.yaml")
    problem = build_calibration_experiment_problem(configuration, REPO_ROOT)
    with np.load(REPO_ROOT / configuration["dataset"], allow_pickle=False) as archive:
        pair = archive["pair_moments"]
        expected_mean = pair[problem.split.train_indices].mean(axis=0)
        expected_scale = np.maximum(
            pair[problem.split.train_indices].std(axis=0),
            configuration["condition_scale_floor"],
        )
    np.testing.assert_allclose(problem.condition_mean, expected_mean, atol=1e-13)
    np.testing.assert_allclose(problem.condition_scale, expected_scale, atol=1e-13)
    assert not hasattr(problem.full_batch, "regime_labels")
    assert not hasattr(problem.full_batch, "reference_angular_moments")
    assert problem.train_batch.anchor_coordinates.shape[0] == 12
    assert problem.validation_batch.anchor_coordinates.shape[0] == 4


def test_generator_batch_is_a_valid_dynamic_jax_pytree():
    configuration = load_yaml(REPO_ROOT / "configs" / "calibration_ablations.yaml")
    problem = build_calibration_experiment_problem(configuration, REPO_ROOT)
    small = subset_generator_batch(problem.train_batch, np.asarray([0, 1], dtype=np.int32))

    @jax.jit
    def total_target(batch):
        return jnp.sum(batch.target_moments) + jnp.sum(batch.basis.centers)

    observed = total_target(small)
    expected = jnp.sum(small.target_moments) + jnp.sum(small.basis.centers)
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


def test_minibatch_optimizer_persists_state_across_base_updates():
    configuration = load_yaml(REPO_ROOT / "configs" / "calibration_ablations.yaml")
    configuration["minibatching"]["num_epochs"] = 1
    configuration["training"]["jit_objective"] = False
    problem = build_calibration_experiment_problem(configuration, REPO_ROOT)

    def objective(parameters, batch):
        return ablation_training_objective(
            parameters,
            batch,
            problem.model_config,
            problem.completion_options,
            problem.objective_weights,
            AblationMode.BASE,
        )

    result = train_equivariant_generator_minibatches(
        objective,
        problem.initial_parameters,
        problem.minibatches,
        problem.train_batch,
        problem.training_options,
    )
    assert result.history["loss"].shape[0] == len(problem.minibatches) + 1
    assert np.isfinite(float(result.final_loss))
    assert float(result.history["update_norm"][0]) > 0.0
    assert float(result.history["update_norm"][-1]) == 0.0
