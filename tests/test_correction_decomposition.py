from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mfsi.decomposition import correction_decomposition, raster_tangent_projection
from mfsi.poisson import weighted_laplacian


def test_direct_field_decomposition_satisfies_discrete_polarization() -> None:
    full = jnp.asarray(
        [[[0.0, 1.0, 0.5], [0.2, -0.3, 0.7], [0.1, 0.4, -0.2]]],
        dtype=jnp.float64,
    )
    tangent = jnp.asarray(
        [[[0.1, 0.2, 0.3], [-0.1, 0.0, 0.1], [0.2, -0.2, 0.4]]],
        dtype=jnp.float64,
    )
    density = jnp.asarray(
        [[[0.8, 1.0, 0.9], [1.1, 0.7, 1.2], [0.6, 1.3, 0.8]]],
        dtype=jnp.float64,
    )
    result = correction_decomposition(
        full, tangent, density, dx=0.25, cell_area=0.25**2
    )
    np.testing.assert_allclose(result.discrete_polarization_residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        result.full_energy,
        result.tangent_energy
        + result.hidden_energy
        + 2.0 * result.tangent_hidden_inner_product,
        atol=1e-13,
    )


def test_raster_tangent_projection_is_feasible_orthogonal_and_pythagorean() -> None:
    density = jnp.asarray(
        [[0.8, 1.0, 0.9], [1.1, 0.7, 1.2], [0.6, 1.3, 0.8]],
        dtype=jnp.float64,
    )
    phi_1 = jnp.asarray(
        [[0.0, 0.2, 0.5], [0.1, 0.4, 0.9], [0.0, 0.3, 0.7]],
        dtype=jnp.float64,
    )
    phi_2 = jnp.asarray(
        [[0.8, 0.4, 0.0], [0.5, 0.2, -0.1], [0.9, 0.3, -0.2]],
        dtype=jnp.float64,
    )
    hidden = jnp.asarray(
        [[0.2, -0.1, 0.3], [-0.4, 0.1, 0.2], [0.0, 0.5, -0.2]],
        dtype=jnp.float64,
    )
    features = jnp.stack([phi_1, phi_2], axis=-1)
    dx = 0.25
    area = dx**2

    # Remove the feature-gradient component from an arbitrary hidden field so
    # the constructed Full potential has a known orthogonal decomposition.
    preliminary = raster_tangent_projection(
        hidden[None],
        density[None],
        weighted_laplacian(hidden, density, dx)[None] / density[None],
        features,
        dx=dx,
        cell_area=area,
    )
    hidden_orthogonal = hidden - jnp.einsum(
        "j,xyj->xy", preliminary.coefficients[0], features
    )
    tangent = 0.7 * phi_1 - 0.35 * phi_2
    full = tangent + hidden_orthogonal
    source = weighted_laplacian(full, density, dx) / density

    result = raster_tangent_projection(
        full[None],
        density[None],
        source[None],
        features,
        dx=dx,
        cell_area=area,
    )
    np.testing.assert_allclose(result.full_moment_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.tangent_moment_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.hidden_moment_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.tangent_hidden_inner_product, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(result.pythagorean_residual, 0.0, atol=1e-12)
    assert float(result.hierarchy_raw_violation[0]) <= 1e-12
    assert int(result.gram_rank[0]) == 2
