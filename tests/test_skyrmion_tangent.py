from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from experiments.skyrmions_deep_ritz.measurements import LocalDensitySensors
from experiments.skyrmions_deep_ritz.tangent import (
    TangentCertificateConfig,
    audit_tangent_action,
    local_density_gradients,
)


def test_local_density_gradient_matches_autodiff() -> None:
    family = LocalDensitySensors(2, 0.18, min_separation=0.1)
    x = jnp.asarray(
        [[0.18, 0.27], [0.74, 0.61], [1.31, 0.83]], dtype=jnp.float64
    )
    eta = jnp.asarray([0.32, 0.22, 1.42, 0.71], dtype=jnp.float64)
    analytic = local_density_gradients(x, eta, family)
    automatic = jax.jacrev(lambda value: family.features(value[None], eta)[0])(x)
    automatic = jnp.moveaxis(automatic, 0, 1)
    assert analytic.shape == automatic.shape
    assert jnp.allclose(analytic, automatic, rtol=2e-12, atol=2e-12)


def test_tangent_action_is_nonnegative_exact_and_sensor_order_invariant() -> None:
    family = LocalDensitySensors(2, 0.2, min_separation=0.1)
    key = jax.random.PRNGKey(7)
    x = jax.random.uniform(key, (3, 24, 5, 2), dtype=jnp.float64)
    velocity = 0.04 * jnp.sin(3.0 * x)
    weights = jnp.ones((3, 24), dtype=jnp.float64) / 24.0
    eta = jnp.asarray([0.35, 0.24, 1.45, 0.76], dtype=jnp.float64)
    advective = family.jvp(x, velocity, eta)
    mean_advective = jnp.einsum("tn,tnr->tr", weights, advective)
    target_dot = mean_advective + jnp.asarray(
        [[0.01, -0.02], [0.015, 0.005], [-0.008, 0.012]], dtype=jnp.float64
    )
    config = TangentCertificateConfig(
        minimum_ess_fraction=0.0,
        maximum_moment_rate_residual=1e-11,
    )
    result = audit_tangent_action(
        x,
        velocity,
        weights,
        target_dot,
        eta,
        family,
        jnp.asarray([0.25, 0.5, 0.25]),
        cfg=config,
    )
    permutation = jnp.asarray([1, 0])
    permuted = audit_tangent_action(
        x,
        velocity,
        weights,
        target_dot[:, permutation],
        eta.reshape(2, 2)[permutation].reshape(-1),
        family,
        jnp.asarray([0.25, 0.5, 0.25]),
        cfg=config,
    )
    assert result["valid"]
    assert result["action"] >= 0.0
    assert result["maximum_moment_rate_residual"] < 1e-12
    assert np.isclose(result["action"], permuted["action"], rtol=2e-12, atol=2e-12)
