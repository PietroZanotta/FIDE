import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.flow_matching import (
    ConditionalFlowConfig,
    FlowSamplingOptions,
    apply_conditional_velocity,
    conditional_flow_matching_loss,
    flow_matching_path,
    initialize_conditional_flow,
    sample_conditional_flow,
)
from manybody_completion.generator import EquivariantGeneratorConfig
from manybody_completion.geometry import wrap_positions
from manybody_completion.observables import PairBasis, ensemble_pair_moments

jax.config.update("jax_enable_x64", True)


def _setup():
    network = EquivariantGeneratorConfig(
        latent_dim=3,
        hidden_dim=12,
        message_dim=10,
        num_message_passing_steps=2,
        radial_basis_size=5,
        radial_min=0.0,
        radial_max=0.5,
        radial_width=0.12,
        max_coordinate_update=0.18,
    )
    config = ConditionalFlowConfig(
        network=network,
        time_frequencies=3,
        velocity_scale=2.0,
    )
    key = jax.random.PRNGKey(90)
    parameter_key, source_key, target_key, latent_key = jax.random.split(key, 4)
    parameters = initialize_conditional_flow(
        parameter_key, condition_dim=4, config=config, dtype=jnp.float64
    )
    source = jax.random.uniform(source_key, (2, 6, 2), dtype=jnp.float64)
    target = jax.random.uniform(target_key, (2, 6, 2), dtype=jnp.float64)
    latents = jax.random.normal(latent_key, (2, 6, 3), dtype=jnp.float64)
    condition = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float64)
    box = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    return parameters, source, target, latents, condition, box, config


def test_flow_path_reaches_target_up_to_replica_translation():
    _, source, target, _, _, box, _ = _setup()
    start, velocity = flow_matching_path(source, target, 0.0, box)
    end, _ = flow_matching_path(source, target, 1.0, box)
    np.testing.assert_allclose(np.asarray(start), np.asarray(source), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(jnp.mean(velocity, axis=-2)),
        np.zeros((2, 2)),
        atol=2e-15,
    )
    basis = PairBasis.uniform(5, 0.0, 0.5, width=0.12, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(ensemble_pair_moments(end, box, basis)),
        np.asarray(ensemble_pair_moments(target, box, basis)),
        atol=2e-12,
        rtol=2e-12,
    )


def test_flow_velocity_is_permutation_and_translation_equivariant():
    parameters, source, _, latents, condition, box, config = _setup()
    reference = apply_conditional_velocity(
        parameters, source, latents, condition, 0.37, box, config
    )
    permutation = jnp.asarray([4, 0, 5, 1, 3, 2])
    inverse = jnp.argsort(permutation)
    permuted = apply_conditional_velocity(
        parameters,
        source[:, permutation],
        latents[:, permutation],
        condition,
        0.37,
        box,
        config,
    )
    np.testing.assert_allclose(
        np.asarray(permuted[:, inverse]), np.asarray(reference), atol=3e-12, rtol=3e-12
    )
    shift = jnp.asarray([0.71, -0.44], dtype=jnp.float64)
    translated = apply_conditional_velocity(
        parameters,
        wrap_positions(source + shift, box),
        latents,
        condition,
        0.37,
        box,
        config,
    )
    np.testing.assert_allclose(
        np.asarray(translated), np.asarray(reference), atol=3e-12, rtol=3e-12
    )
    np.testing.assert_allclose(
        np.asarray(jnp.mean(reference, axis=-2)), np.zeros((2, 2)), atol=3e-15
    )


def test_flow_velocity_has_square_box_d4_equivariance():
    parameters, source, _, latents, condition, box, config = _setup()

    def rotate(values):
        return wrap_positions(jnp.stack((-values[..., 1], values[..., 0]), axis=-1), box)

    def rotate_vectors(values):
        return jnp.stack((-values[..., 1], values[..., 0]), axis=-1)

    reference = apply_conditional_velocity(
        parameters, source, latents, condition, 0.51, box, config
    )
    rotated = apply_conditional_velocity(
        parameters, rotate(source), latents, condition, 0.51, box, config
    )
    np.testing.assert_allclose(
        np.asarray(rotated), np.asarray(rotate_vectors(reference)), atol=4e-12, rtol=4e-12
    )


def test_flow_loss_has_finite_nonzero_parameter_gradient():
    parameters, source, target, latents, condition, box, config = _setup()

    def loss_fn(model_parameters):
        return conditional_flow_matching_loss(
            model_parameters,
            source,
            target,
            latents,
            condition,
            0.43,
            box,
            config,
        )[0]

    gradients = jax.grad(loss_fn)(parameters)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    norm = np.sqrt(sum(float(jnp.sum(leaf * leaf)) for leaf in leaves))
    assert norm > 1e-8


def test_flow_sampler_is_reproducible_wrapped_and_returns_trajectory():
    parameters, _, _, _, condition, box, config = _setup()
    key = jax.random.PRNGKey(123)
    options = FlowSamplingOptions(num_steps=5, integrator="heun")
    first, trajectory = sample_conditional_flow(
        parameters,
        key,
        condition,
        num_replicas=2,
        num_particles=6,
        box=box,
        config=config,
        options=options,
        dtype=jnp.float64,
        return_trajectory=True,
    )
    second = sample_conditional_flow(
        parameters,
        key,
        condition,
        num_replicas=2,
        num_particles=6,
        box=box,
        config=config,
        options=options,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(np.asarray(first), np.asarray(second), atol=0.0, rtol=0.0)
    assert trajectory.shape == (6, 2, 6, 2)
    assert np.all(np.asarray(first) >= 0.0)
    assert np.all(np.asarray(first) < 1.0)
