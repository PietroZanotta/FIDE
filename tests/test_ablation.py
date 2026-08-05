from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.ablation import (
    AblationMode,
    CompletionStage,
    get_ablation_spec,
)
from manybody_completion.composition import CompletionOptions, PhysicalParameters
from manybody_completion.generator import (
    EquivariantGeneratorConfig,
    initialize_equivariant_generator,
    make_periodic_grid_anchors,
)
from manybody_completion.generator_training import (
    GeneratorBatch,
    GeneratorObjectiveWeights,
    ablation_generator_objective,
    ablation_training_objective,
)
from manybody_completion.observables import PairBasis, ensemble_pair_moments
from manybody_completion.projection import ProjectionOptions
from manybody_completion.relaxation import RelaxationOptions

jax.config.update("jax_enable_x64", True)


def _setup():
    dtype = jnp.float64
    box = jnp.asarray([1.0, 1.0], dtype=dtype)
    basis = PairBasis(
        centers=jnp.asarray([0.18, 0.36], dtype=dtype),
        widths=jnp.asarray([0.09, 0.11], dtype=dtype),
    )
    config = EquivariantGeneratorConfig(
        latent_dim=2,
        hidden_dim=5,
        message_dim=5,
        num_message_passing_steps=1,
        radial_basis_size=3,
        max_coordinate_update=0.03,
    )
    parameter_key, anchor_key, latent_key, target_key = jax.random.split(
        jax.random.PRNGKey(41), 4
    )
    parameters = initialize_equivariant_generator(
        parameter_key,
        condition_dim=2,
        config=config,
        dtype=dtype,
    )
    anchors = make_periodic_grid_anchors(
        anchor_key,
        batch_size=1,
        num_replicas=3,
        grid_shape=(2, 2),
        box=box,
        jitter_scale=0.015,
        dtype=dtype,
    )
    latents = jax.random.normal(latent_key, (1, 3, 4, 2), dtype=dtype)
    reference = jax.random.uniform(target_key, (3, 4, 2), dtype=dtype)
    target_moments = ensemble_pair_moments(reference, box, basis)[None, :]
    batch = GeneratorBatch(
        anchor_coordinates=anchors,
        node_latents=latents,
        conditions=jnp.asarray([[0.2, -0.3]], dtype=dtype),
        target_moments=target_moments,
        box=box,
        basis=basis,
        moment_scales=jnp.ones((2,), dtype=dtype),
        basis_mask=jnp.ones((2,), dtype=dtype),
    )
    completion = CompletionOptions(
        physical=PhysicalParameters(r0=0.12, kappa=25.0, prox_strength=0.05),
        relaxation=RelaxationOptions(
            num_steps=8,
            step_size=0.04,
            tolerance=1e-3,
            max_update_norm=0.03,
            line_search_steps=5,
        ),
        projection=ProjectionOptions(
            num_steps=8,
            tolerance=2e-5,
            kkt_tolerance=5e-3,
            ridge=1e-7,
            max_step_norm=0.05,
            max_correction_norm=0.3,
            line_search_steps=6,
        ),
    )
    weights = GeneratorObjectiveWeights(
        observed=100.0,
        physical=0.1,
        correction=100.0,
    )
    return parameters, batch, config, completion, weights


def _gradient(objective, parameters):
    return jax.grad(lambda values: objective(values)[0])(parameters)


def _tree_allclose(left, right, atol=1e-11, rtol=1e-11):
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _tree_distance(left, right):
    return np.sqrt(
        sum(
            float(jnp.sum((a - b) ** 2))
            for a, b in zip(
                jax.tree_util.tree_leaves(left),
                jax.tree_util.tree_leaves(right),
            )
        )
    )


def test_ablation_specs_are_explicit_and_parse_aliases():
    assert AblationMode.parse("post-hoc") is AblationMode.POST_HOC
    assert AblationMode.parse("full") is AblationMode.FULL_E2E
    assert get_ablation_spec("base").training_stage is CompletionStage.INITIAL
    assert get_ablation_spec("post_hoc").serving_stage is CompletionStage.PROJECTED
    assert get_ablation_spec("relax_e2e").training_stage is CompletionStage.RELAXED
    assert get_ablation_spec("full_e2e").training_stage is CompletionStage.PROJECTED


def test_base_and_posthoc_have_identical_losses_and_parameter_gradients():
    parameters, batch, config, completion, weights = _setup()
    base = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=completion,
        weights=weights,
        mode=AblationMode.BASE,
    )
    post_hoc = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=completion,
        weights=weights,
        mode=AblationMode.POST_HOC,
    )
    base_loss, base_metrics = base(parameters)
    post_loss, post_metrics = post_hoc(parameters)
    np.testing.assert_allclose(base_loss, post_loss, atol=0.0, rtol=0.0)
    _tree_allclose(_gradient(base, parameters), _gradient(post_hoc, parameters))
    np.testing.assert_allclose(
        base_metrics["moment_error_training"],
        base_metrics["moment_error_initial"],
    )
    np.testing.assert_allclose(
        post_metrics["moment_error_serving"],
        post_metrics["moment_error_projected"],
    )


def test_relax_e2e_loss_and_gradient_do_not_depend_on_projection_options():
    parameters, batch, config, completion, weights = _setup()
    alternative_completion = CompletionOptions(
        physical=completion.physical,
        relaxation=completion.relaxation,
        projection=ProjectionOptions(
            num_steps=2,
            tolerance=1e-2,
            kkt_tolerance=1e-1,
            ridge=1e-3,
            max_step_norm=0.01,
            max_correction_norm=0.02,
            line_search_steps=2,
        ),
    )
    objective_a = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=completion,
        weights=weights,
        mode=AblationMode.RELAX_E2E,
    )
    objective_b = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=alternative_completion,
        weights=weights,
        mode=AblationMode.RELAX_E2E,
    )
    np.testing.assert_allclose(objective_a(parameters)[0], objective_b(parameters)[0])
    _tree_allclose(_gradient(objective_a, parameters), _gradient(objective_b, parameters))


def test_full_e2e_uses_a_distinct_solver_gradient_path():
    parameters, batch, config, completion, weights = _setup()
    relax = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=completion,
        weights=weights,
        mode=AblationMode.RELAX_E2E,
    )
    full = partial(
        ablation_generator_objective,
        batch=batch,
        generator_config=config,
        completion_options=completion,
        weights=weights,
        mode=AblationMode.FULL_E2E,
    )
    relax_gradient = _gradient(relax, parameters)
    full_gradient = _gradient(full, parameters)
    assert _tree_distance(relax_gradient, full_gradient) > 1e-6
    assert np.isfinite(float(full(parameters)[0]))


def test_training_objective_reports_only_the_required_solver_path():
    parameters, batch, config, completion, weights = _setup()
    expected = {
        AblationMode.BASE: (0.0, 0.0),
        AblationMode.POST_HOC: (0.0, 0.0),
        AblationMode.RELAX_E2E: (1.0, 0.0),
        AblationMode.FULL_E2E: (1.0, 1.0),
    }
    for mode, flags in expected.items():
        _, metrics = ablation_training_objective(
            parameters,
            batch,
            config,
            completion,
            weights,
            mode,
        )
        assert float(metrics["training_relaxation_used"]) == flags[0]
        assert float(metrics["training_projection_used"]) == flags[1]


def test_base_training_does_not_execute_relaxation(monkeypatch):
    import manybody_completion.generator_training as training_module

    parameters, batch, config, completion, weights = _setup()

    def fail_relaxation(*args, **kwargs):
        raise AssertionError("Base training must not execute relaxation")

    monkeypatch.setattr(training_module, "relax_proximal", fail_relaxation)
    loss, _ = ablation_training_objective(
        parameters,
        batch,
        config,
        completion,
        weights,
        AblationMode.BASE,
    )
    assert np.isfinite(float(loss))


def test_relax_training_does_not_execute_projection(monkeypatch):
    import manybody_completion.generator_training as training_module

    parameters, batch, config, completion, weights = _setup()

    def fail_projection(*args, **kwargs):
        raise AssertionError("Relax-E2E training must not execute projection")

    monkeypatch.setattr(training_module, "project_ensemble_moments", fail_projection)
    loss, _ = ablation_training_objective(
        parameters,
        batch,
        config,
        completion,
        weights,
        AblationMode.RELAX_E2E,
    )
    assert np.isfinite(float(loss))
