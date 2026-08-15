from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .grid import CartesianGrid2D

Array = jax.Array


@dataclass(frozen=True)
class RasterConfig:
    bandwidth: float = 0.0
    truncate: float = 4.0


class RasterResult(NamedTuple):
    q: Array
    mass: Array
    h: Array
    source_mass_before_center: Array
    source_mass_after_center: Array


def gaussian_kernel_2d(sigma_cells: float, truncate: float) -> Array:
    if sigma_cells <= 0.0:
        return jnp.ones((1, 1), dtype=jnp.float64)
    radius = max(1, int(math.ceil(float(truncate) * float(sigma_cells))))
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float64)
    k = jnp.exp(-0.5 * (x / float(sigma_cells)) ** 2)
    k = k / jnp.sum(k)
    return k[:, None] * k[None, :]


def rasterize_projected_particles(
    x: Array,
    weights: Array,
    forcing: Array,
    grid: CartesianGrid2D,
    cfg: RasterConfig = RasterConfig(),
) -> RasterResult:
    """Histogram + Gaussian anti-aliasing, JAX-differentiable in weights/forcing.

    Reference particle positions are frozen during measurement-design optimization,
    so hard cell assignment does not obstruct eta gradients. All eta-dependent
    quantities (I-projection weights and forcing) remain in the JAX graph.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)

    idx = grid.flat_bin_index(x)
    mass_flat = jnp.zeros(grid.n * grid.n, dtype=jnp.float64).at[idx].add(weights)
    source_flat = jnp.zeros(grid.n * grid.n, dtype=jnp.float64).at[idx].add(weights * forcing)
    mass = mass_flat.reshape((grid.n, grid.n))
    source = source_flat.reshape((grid.n, grid.n))

    bandwidth = cfg.bandwidth if cfg.bandwidth > 0.0 else 0.35 * grid.dx
    kernel = gaussian_kernel_2d(bandwidth / grid.dx, cfg.truncate)
    mass = jsp.signal.convolve2d(mass, kernel, mode="same")
    source = jsp.signal.convolve2d(source, kernel, mode="same")

    norm = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / norm
    source = source / norm
    source_before = jnp.sum(source)
    source = source - mass * source_before
    source_after = jnp.sum(source)

    q = mass / grid.cell_area
    # Avoid differentiating through 0/0 in empty histogram cells.  ``jnp.where``
    # around ``source / mass`` is not sufficient because autodiff may still trace
    # the undefined division in the inactive branch.
    occupied = mass > 1.0e-300
    safe_mass = jnp.where(occupied, mass, 1.0)
    h = jnp.where(occupied, source / safe_mass, 0.0)
    return RasterResult(q, mass, h, source_before, source_after)
