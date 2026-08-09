import jax
import jax.numpy as jnp
import numpy as np

from manybody_completion.generator import (
    EquivariantGeneratorConfig,
    apply_equivariant_generator,
    count_generator_parameters,
    flatten_generator_parameters,
    initialize_equivariant_generator,
    restore_generator_parameters,
)
from manybody_completion.geometry import wrap_positions

jax.config.update("jax_enable_x64", True)


def _setup():
    config = EquivariantGeneratorConfig(
        latent_dim=3,
        hidden_dim=12,
        message_dim=10,
        num_message_passing_steps=2,
        radial_basis_size=5,
        max_coordinate_update=0.04,
    )
    key = jax.random.PRNGKey(17)
    param_key, coordinate_key, latent_key = jax.random.split(key, 3)
    parameters = initialize_equivariant_generator(
        param_key, condition_dim=4, config=config, dtype=jnp.float64
    )
    coordinates = jax.random.uniform(
        coordinate_key, (3, 7, 2), minval=0.0, maxval=1.0, dtype=jnp.float64
    )
    latents = jax.random.normal(latent_key, (3, 7, 3), dtype=jnp.float64)
    condition = jnp.asarray([0.1, -0.2, 0.3, 0.4], dtype=jnp.float64)
    box = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    return parameters, coordinates, latents, condition, box, config


def test_generator_shapes_finiteness_and_parameter_count():
    parameters, coordinates, latents, condition, box, config = _setup()
    output = apply_equivariant_generator(
        parameters, coordinates, latents, condition, box, config
    )
    assert output.shape == coordinates.shape
    assert np.all(np.isfinite(np.asarray(output)))
    assert np.all(np.asarray(output) >= 0.0)
    assert np.all(np.asarray(output) < 1.0)
    assert count_generator_parameters(parameters) > 100


def test_generator_particle_permutation_equivariance():
    parameters, coordinates, latents, condition, box, config = _setup()
    permutation = jnp.asarray([4, 0, 6, 1, 5, 2, 3])
    inverse = jnp.argsort(permutation)
    reference = apply_equivariant_generator(
        parameters, coordinates, latents, condition, box, config
    )
    permuted = apply_equivariant_generator(
        parameters,
        coordinates[:, permutation],
        latents[:, permutation],
        condition,
        box,
        config,
    )
    np.testing.assert_allclose(
        np.asarray(permuted[:, inverse]), np.asarray(reference), atol=2e-12, rtol=2e-12
    )


def test_generator_toroidal_translation_equivariance_across_wraps():
    parameters, coordinates, latents, condition, box, config = _setup()
    shift = jnp.asarray([0.73, -0.41], dtype=jnp.float64)
    reference = apply_equivariant_generator(
        parameters, coordinates, latents, condition, box, config
    )
    translated = apply_equivariant_generator(
        parameters,
        wrap_positions(coordinates + shift, box),
        latents,
        condition,
        box,
        config,
    )
    expected = wrap_positions(reference + shift, box)
    periodic_error = jnp.mod(translated - expected + 0.5 * box, box) - 0.5 * box
    np.testing.assert_allclose(
        np.asarray(periodic_error), np.zeros_like(np.asarray(periodic_error)), atol=3e-12
    )



def test_generator_square_box_d4_rotation_equivariance():
    parameters, coordinates, latents, condition, box, config = _setup()

    def rotate_quarter_turn(values):
        return wrap_positions(
            jnp.stack((-values[..., 1], values[..., 0]), axis=-1), box
        )

    reference = apply_equivariant_generator(
        parameters, coordinates, latents, condition, box, config
    )
    rotated = apply_equivariant_generator(
        parameters, rotate_quarter_turn(coordinates), latents, condition, box, config
    )
    expected = rotate_quarter_turn(reference)
    periodic_error = jnp.mod(rotated - expected + 0.5 * box, box) - 0.5 * box
    np.testing.assert_allclose(
        np.asarray(periodic_error), np.zeros_like(np.asarray(periodic_error)), atol=3e-12
    )



def test_generator_parameter_archive_round_trip():
    parameters, coordinates, latents, condition, box, config = _setup()
    flattened = flatten_generator_parameters(parameters)
    restored = restore_generator_parameters(parameters, flattened)
    reference = apply_equivariant_generator(
        parameters, coordinates, latents, condition, box, config
    )
    recovered = apply_equivariant_generator(
        restored, coordinates, latents, condition, box, config
    )
    np.testing.assert_allclose(
        np.asarray(recovered), np.asarray(reference), atol=0.0, rtol=0.0
    )


def test_generator_has_finite_nonzero_parameter_gradient():
    parameters, coordinates, latents, condition, box, config = _setup()

    def loss_fn(model_parameters):
        generated = apply_equivariant_generator(
            model_parameters, coordinates, latents, condition, box, config
        )
        return jnp.mean(jnp.sin(2.0 * jnp.pi * generated) ** 2)

    gradients = jax.grad(loss_fn)(parameters)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    global_norm = np.sqrt(sum(float(jnp.sum(leaf * leaf)) for leaf in leaves))
    assert global_norm > 1e-8
