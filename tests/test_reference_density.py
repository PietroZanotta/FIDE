import jax.numpy as jnp

from mfsi.reference_density import (
    backward_latent_with_log_density_correction,
    latent_velocity_and_divergence,
    logistic_log_abs_det_jacobian,
)


def linear_velocity_params(a=0.3, b=-0.15):
    # A one-layer velocity model is exactly linear in the first two model
    # features, which are the two latent coordinates.
    weight = jnp.zeros((7, 2), dtype=jnp.float64)
    weight = weight.at[0, 0].set(a)
    weight = weight.at[1, 1].set(b)
    return ({"W": weight, "b": jnp.zeros(2, dtype=jnp.float64)},)


def test_exact_divergence_for_linear_latent_velocity():
    params = linear_velocity_params(0.3, -0.15)
    z = jnp.asarray([[0.2, -0.7], [1.1, 0.4]])
    velocity, divergence = latent_velocity_and_divergence(params, 0.4, z)
    assert jnp.allclose(velocity, z * jnp.asarray([0.3, -0.15]))
    assert jnp.allclose(divergence, 0.15)


def test_backward_cnf_correction_matches_linear_analytic_solution():
    a, b, time = 0.3, -0.15, 0.7
    params = linear_velocity_params(a, b)
    z_t = jnp.asarray([[0.8, -0.2], [-1.3, 0.9]])
    z0, correction = backward_latent_with_log_density_correction(
        params, z_t, time, steps=64
    )
    expected_z0 = z_t * jnp.exp(-jnp.asarray([a, b]) * time)
    expected_correction = -(a + b) * time
    assert jnp.allclose(z0, expected_z0, rtol=2e-10, atol=2e-10)
    assert jnp.allclose(correction, expected_correction, rtol=1e-12, atol=1e-12)


def test_logistic_jacobian_matches_closed_form_at_origin():
    bounds = jnp.asarray([-650.0, 3000.0, -950.0, 1000.0])
    value = logistic_log_abs_det_jacobian(jnp.zeros((3, 2)), bounds)
    expected = jnp.log((3650.0 / 4.0) * (1950.0 / 4.0))
    assert jnp.allclose(value, expected)
