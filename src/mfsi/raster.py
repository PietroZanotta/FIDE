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
    source: Array | None = None


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
    return RasterResult(q, mass, h, source_before, source_after, source / grid.cell_area)


def _bilinear_cell_center_deposition(
    x: Array,
    values: Array,
    grid: CartesianGrid2D,
) -> Array:
    """Deposit values onto cell centers with mass-preserving bilinear weights."""
    coordinate = (x + float(grid.half_width)) / float(grid.dx) - 0.5
    lower = jnp.floor(coordinate).astype(jnp.int32)
    fraction = coordinate - lower.astype(jnp.float64)
    ix0 = jnp.clip(lower[:, 0], 0, grid.n - 1)
    iy0 = jnp.clip(lower[:, 1], 0, grid.n - 1)
    ix1 = jnp.clip(lower[:, 0] + 1, 0, grid.n - 1)
    iy1 = jnp.clip(lower[:, 1] + 1, 0, grid.n - 1)
    fx = fraction[:, 0]
    fy = fraction[:, 1]
    flat = jnp.zeros(grid.n * grid.n, dtype=jnp.float64)
    flat = flat.at[iy0 * grid.n + ix0].add(values * (1.0 - fx) * (1.0 - fy))
    flat = flat.at[iy0 * grid.n + ix1].add(values * fx * (1.0 - fy))
    flat = flat.at[iy1 * grid.n + ix0].add(values * (1.0 - fx) * fy)
    flat = flat.at[iy1 * grid.n + ix1].add(values * fx * fy)
    return flat.reshape((grid.n, grid.n))


def _full_support_gaussian_matrix(grid: CartesianGrid2D, bandwidth: float) -> Array:
    """Boundary-normalized 1-D Gaussian map from source to target cell centers."""
    if not float(bandwidth) > 0.0:
        raise ValueError("positive-support rasterization requires bandwidth > 0")
    centers = grid.centers_1d()
    displacement = centers[:, None] - centers[None, :]
    kernel = jnp.exp(-0.5 * (displacement / float(bandwidth)) ** 2)
    # Each source-cell column integrates to one on the truncated computational
    # domain.  The 2-D separable map therefore preserves deposited mass exactly.
    return kernel / jnp.maximum(jnp.sum(kernel, axis=0, keepdims=True), 1.0e-300)


def rasterize_projected_particles_positive(
    x: Array,
    weights: Array,
    forcing: Array,
    grid: CartesianGrid2D,
    *,
    bandwidth: float,
) -> RasterResult:
    """Positive full-support common-kernel density/source deposition.

    Particle locations are first represented to subcell accuracy by a
    mass-preserving bilinear deposit.  A full-domain Gaussian map is then applied
    identically to the mass and signed source deposits.  Column normalization is
    the declared boundary treatment and preserves both totals before the optional
    global floating-point source centering.  No density floor is introduced.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    raw_mass = _bilinear_cell_center_deposition(x, weights, grid)
    raw_source = _bilinear_cell_center_deposition(x, weights * forcing, grid)
    kernel = _full_support_gaussian_matrix(grid, float(bandwidth))
    mass = kernel @ raw_mass @ kernel.T
    source_mass = kernel @ raw_source @ kernel.T

    normalization = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / normalization
    source_mass = source_mass / normalization
    source_before = jnp.sum(source_mass)
    # The particle forcing is analytically centered.  Remove only its residual
    # global floating-point mean, distributed in the already deposited density.
    source_mass = source_mass - mass * source_before
    source_after = jnp.sum(source_mass)

    q = mass / float(grid.cell_area)
    source = source_mass / float(grid.cell_area)
    h = source / q
    return RasterResult(q, mass, h, source_before, source_after, source)


def gaussian_kernel_2d_rect(
    sigma_y_cells: float,
    sigma_x_cells: float,
    truncate: float,
) -> Array:
    """Separable anisotropic Gaussian antialiasing kernel in grid-cell units."""
    sy = float(sigma_y_cells)
    sx = float(sigma_x_cells)
    if sy <= 0.0 and sx <= 0.0:
        return jnp.ones((1, 1), dtype=jnp.float64)
    sy = max(sy, 1.0e-15)
    sx = max(sx, 1.0e-15)
    ry = max(1, int(math.ceil(float(truncate) * sy)))
    rx = max(1, int(math.ceil(float(truncate) * sx)))
    yy = jnp.arange(-ry, ry + 1, dtype=jnp.float64)
    xx = jnp.arange(-rx, rx + 1, dtype=jnp.float64)
    ky = jnp.exp(-0.5 * (yy / sy) ** 2)
    kx = jnp.exp(-0.5 * (xx / sx) ** 2)
    ky = ky / jnp.sum(ky)
    kx = kx / jnp.sum(kx)
    return ky[:, None] * kx[None, :]


def rasterize_projected_particles_rect(
    x: Array,
    weights: Array,
    forcing: Array,
    grid,
    cfg: RasterConfig = RasterConfig(),
) -> RasterResult:
    """Rectangular-grid sibling of ``rasterize_projected_particles``.

    The historical square-grid function is intentionally unchanged.  ``grid`` is
    expected to provide ``size``, ``shape``, ``dx``, ``dy``, ``cell_area`` and
    ``flat_bin_index`` (e.g. ``RectangularGrid2D``).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)

    idx = grid.flat_bin_index(x)
    mass_flat = jnp.zeros(int(grid.size), dtype=jnp.float64).at[idx].add(weights)
    source_flat = jnp.zeros(int(grid.size), dtype=jnp.float64).at[idx].add(weights * forcing)
    mass = mass_flat.reshape(grid.shape)
    source = source_flat.reshape(grid.shape)

    bandwidth = float(cfg.bandwidth) if cfg.bandwidth > 0.0 else 0.35 * min(grid.dx, grid.dy)
    kernel = gaussian_kernel_2d_rect(bandwidth / grid.dy, bandwidth / grid.dx, cfg.truncate)
    mass = jsp.signal.convolve2d(mass, kernel, mode="same")
    source = jsp.signal.convolve2d(source, kernel, mode="same")

    norm = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / norm
    source = source / norm
    source_before = jnp.sum(source)
    source = source - mass * source_before
    source_after = jnp.sum(source)

    q = mass / float(grid.cell_area)
    occupied = mass > 1.0e-300
    safe_mass = jnp.where(occupied, mass, 1.0)
    h = jnp.where(occupied, source / safe_mass, 0.0)
    return RasterResult(q, mass, h, source_before, source_after, source / float(grid.cell_area))
