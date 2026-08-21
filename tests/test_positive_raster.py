from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mfsi.grid import CartesianGrid2D
from mfsi.raster import rasterize_projected_particles_positive


def test_positive_raster_uses_one_mass_preserving_kernel_for_density_and_source():
    grid = CartesianGrid2D(half_width=2.0, n=17)
    locations = jnp.asarray(
        [[-1.7, -1.5], [-0.3, 0.2], [0.8, -0.4], [1.75, 1.6]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray([0.1, 0.25, 0.4, 0.25], dtype=jnp.float64)
    forcing = jnp.asarray([0.7, -0.3, 0.2, -0.16], dtype=jnp.float64)
    forcing -= jnp.sum(weights * forcing)

    result = rasterize_projected_particles_positive(
        locations, weights, forcing, grid, bandwidth=0.45
    )

    assert float(jnp.min(result.q)) > 0.0
    np.testing.assert_allclose(jnp.sum(result.mass), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(
        jnp.sum(result.source) * grid.cell_area, 0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(result.q * result.h, result.source, rtol=2.0e-15)
    assert abs(float(result.source_mass_after_center)) <= 2.0e-15
