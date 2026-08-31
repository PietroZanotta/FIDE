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


def _full_support_gaussian_matrix_1d(centers: Array, bandwidth: float) -> Array:
    """Boundary-normalized Gaussian map on an arbitrary 1-D cell-center axis."""
    if not float(bandwidth) > 0.0:
        raise ValueError("positive-support rasterization requires bandwidth > 0")
    centers = jnp.asarray(centers, dtype=jnp.float64)
    displacement = centers[:, None] - centers[None, :]
    kernel = jnp.exp(-0.5 * (displacement / float(bandwidth)) ** 2)
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


def _bilinear_cell_center_deposition_rect(
    x: Array,
    values: Array,
    grid,
) -> Array:
    """Mass-preserving bilinear deposit on a ``RectangularGrid2D``."""
    coordinate_x = (x[:, 0] - float(grid.x_min)) / float(grid.dx) - 0.5
    coordinate_y = (x[:, 1] - float(grid.y_min)) / float(grid.dy) - 0.5
    lower_x = jnp.floor(coordinate_x).astype(jnp.int32)
    lower_y = jnp.floor(coordinate_y).astype(jnp.int32)
    fraction_x = coordinate_x - lower_x.astype(jnp.float64)
    fraction_y = coordinate_y - lower_y.astype(jnp.float64)
    ix0 = jnp.clip(lower_x, 0, grid.nx - 1)
    iy0 = jnp.clip(lower_y, 0, grid.ny - 1)
    ix1 = jnp.clip(lower_x + 1, 0, grid.nx - 1)
    iy1 = jnp.clip(lower_y + 1, 0, grid.ny - 1)
    flat = jnp.zeros(int(grid.size), dtype=jnp.float64)
    flat = flat.at[iy0 * grid.nx + ix0].add(
        values * (1.0 - fraction_x) * (1.0 - fraction_y)
    )
    flat = flat.at[iy0 * grid.nx + ix1].add(
        values * fraction_x * (1.0 - fraction_y)
    )
    flat = flat.at[iy1 * grid.nx + ix0].add(
        values * (1.0 - fraction_x) * fraction_y
    )
    flat = flat.at[iy1 * grid.nx + ix1].add(
        values * fraction_x * fraction_y
    )
    return flat.reshape(grid.shape)


def rasterize_projected_particles_positive_rect(
    x: Array,
    weights: Array,
    forcing: Array,
    grid,
    *,
    bandwidth: float,
) -> RasterResult:
    """Positive common-kernel rasterization on a rectangular physical grid.

    This is the rectangular-domain counterpart of
    :func:`rasterize_projected_particles_positive`.  Independent,
    boundary-normalized Gaussian maps are applied along the physical x and y
    axes.  The same map is used for density and signed source, so deposited mass,
    global source compatibility, and ``q * h == source`` are preserved without
    adding a density floor to the scientific Poisson operator.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    raw_mass = _bilinear_cell_center_deposition_rect(x, weights, grid)
    raw_source = _bilinear_cell_center_deposition_rect(x, weights * forcing, grid)
    kernel_x = _full_support_gaussian_matrix_1d(grid.x_centers(), float(bandwidth))
    kernel_y = _full_support_gaussian_matrix_1d(grid.y_centers(), float(bandwidth))
    mass = kernel_y @ raw_mass @ kernel_x.T
    source_mass = kernel_y @ raw_source @ kernel_x.T

    normalization = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / normalization
    source_mass = source_mass / normalization
    source_before = jnp.sum(source_mass)
    source_mass = source_mass - mass * source_before
    source_after = jnp.sum(source_mass)

    q = mass / float(grid.cell_area)
    source = source_mass / float(grid.cell_area)
    h = source / q
    return RasterResult(q, mass, h, source_before, source_after, source)


def _normal_cdf_difference(lower: Array, upper: Array) -> Array:
    """Stable ``Phi(upper)-Phi(lower)`` including far positive tails."""
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)
    direct = jsp.special.ndtr(upper) - jsp.special.ndtr(lower)
    survival = jsp.special.ndtr(-lower) - jsp.special.ndtr(-upper)
    return jnp.where(lower > 0.0, survival, direct)


def reflected_gaussian_cell_mass_matrix_1d(
    edges: Array,
    sources: Array,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> Array:
    """Cell-integrated even-reflection Gaussian kernel on an interval.

    Columns correspond to source locations and rows to target cells.  The
    method-of-images construction conserves each source column without an
    ``x``-dependent normalization.  ``image_pairs`` counts translated image
    pairs on either side of the central image.
    """
    if not float(bandwidth) > 0.0:
        raise ValueError("reflected rasterization requires bandwidth > 0")
    if int(image_pairs) < 0:
        raise ValueError("image_pairs must be nonnegative")
    edges = jnp.asarray(edges, dtype=jnp.float64)
    sources = jnp.asarray(sources, dtype=jnp.float64)
    lower_bound = edges[0]
    upper_bound = edges[-1]
    length = upper_bound - lower_bound
    left = edges[:-1, None]
    right = edges[1:, None]
    result = jnp.zeros((edges.shape[0] - 1, sources.shape[0]), dtype=jnp.float64)
    h = float(bandwidth)
    for image_index in range(-int(image_pairs), int(image_pairs) + 1):
        shift = 2.0 * image_index * length
        first_left = (left - sources[None, :] + shift) / h
        first_right = (right - sources[None, :] + shift) / h
        second_left = (left + sources[None, :] - 2.0 * lower_bound + shift) / h
        second_right = (right + sources[None, :] - 2.0 * lower_bound + shift) / h
        result = result + _normal_cdf_difference(first_left, first_right)
        result = result + _normal_cdf_difference(second_left, second_right)
    return result


def reflected_gaussian_face_flux_matrix_1d(
    faces: Array,
    sources: Array,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> Array:
    """Odd-reflection Gaussian kernel on faces for the normal flux.

    This is the Dirichlet image kernel paired with the even scalar kernel.
    Boundary rows are set to exact zero, matching their analytic values.
    """
    if not float(bandwidth) > 0.0:
        raise ValueError("reflected rasterization requires bandwidth > 0")
    if int(image_pairs) < 0:
        raise ValueError("image_pairs must be nonnegative")
    faces = jnp.asarray(faces, dtype=jnp.float64)
    sources = jnp.asarray(sources, dtype=jnp.float64)
    lower_bound = faces[0]
    upper_bound = faces[-1]
    length = upper_bound - lower_bound
    h = float(bandwidth)
    normalizer = math.sqrt(2.0 * math.pi) * h
    result = jnp.zeros((faces.shape[0], sources.shape[0]), dtype=jnp.float64)
    for image_index in range(-int(image_pairs), int(image_pairs) + 1):
        shift = 2.0 * image_index * length
        direct = (faces[:, None] - sources[None, :] + shift) / h
        reflected = (
            faces[:, None] + sources[None, :] - 2.0 * lower_bound + shift
        ) / h
        result = result + jnp.exp(-0.5 * direct**2) / normalizer
        result = result - jnp.exp(-0.5 * reflected**2) / normalizer
    result = result.at[0].set(jnp.zeros_like(result[0]))
    result = result.at[-1].set(jnp.zeros_like(result[-1]))
    return result


def rasterize_projected_particles_reflected_rect(
    x: Array,
    weights: Array,
    forcing: Array,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> RasterResult:
    """Direct even-reflection density/source raster on a rectangular domain.

    The scalar Gaussian is integrated over target cells and evaluated directly
    at the particle source locations.  Density and signed defect use identical
    kernels.  There is no CIC step, no per-source normalization, and no density
    floor.  Only the residual global floating-point source mean is removed.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    x_edges = jnp.linspace(
        float(grid.x_min), float(grid.x_max), int(grid.nx) + 1, dtype=jnp.float64
    )
    y_edges = jnp.linspace(
        float(grid.y_min), float(grid.y_max), int(grid.ny) + 1, dtype=jnp.float64
    )
    kernel_x = reflected_gaussian_cell_mass_matrix_1d(
        x_edges, x[:, 0], bandwidth=bandwidth, image_pairs=image_pairs
    )
    kernel_y = reflected_gaussian_cell_mass_matrix_1d(
        y_edges, x[:, 1], bandwidth=bandwidth, image_pairs=image_pairs
    )
    mass = (kernel_y * weights[None, :]) @ kernel_x.T
    source_mass = (kernel_y * (weights * forcing)[None, :]) @ kernel_x.T
    source_before = jnp.sum(source_mass)
    total_mass = jnp.sum(mass)
    source_mass = source_mass - mass * source_before / jnp.maximum(total_mass, 1.0e-300)
    source_after = jnp.sum(source_mass)
    q = mass / float(grid.cell_area)
    source = source_mass / float(grid.cell_area)
    h = source / q
    return RasterResult(q, mass, h, source_before, source_after, source)


def reflected_particle_flux_rect(
    x: Array,
    weights: Array,
    velocity: Array,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> tuple[Array, Array]:
    """Matched reflected ``S(q u)`` flux on rectangular cell faces.

    The normal coordinate uses the odd face kernel and the tangential
    coordinate uses the even cell-integrated scalar kernel.  Returned arrays
    have shapes ``(ny,nx+1)`` and ``(ny+1,nx)``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    x_edges = jnp.linspace(
        float(grid.x_min), float(grid.x_max), int(grid.nx) + 1, dtype=jnp.float64
    )
    y_edges = jnp.linspace(
        float(grid.y_min), float(grid.y_max), int(grid.ny) + 1, dtype=jnp.float64
    )
    kernel_x_mass = reflected_gaussian_cell_mass_matrix_1d(
        x_edges, x[:, 0], bandwidth=bandwidth, image_pairs=image_pairs
    )
    kernel_y_mass = reflected_gaussian_cell_mass_matrix_1d(
        y_edges, x[:, 1], bandwidth=bandwidth, image_pairs=image_pairs
    )
    kernel_x_face = reflected_gaussian_face_flux_matrix_1d(
        x_edges, x[:, 0], bandwidth=bandwidth, image_pairs=image_pairs
    )
    kernel_y_face = reflected_gaussian_face_flux_matrix_1d(
        y_edges, x[:, 1], bandwidth=bandwidth, image_pairs=image_pairs
    )
    flux_x = (
        (kernel_y_mass * (weights * velocity[:, 0])[None, :]) @ kernel_x_face.T
    ) / float(grid.dy)
    flux_y = (
        (kernel_y_face * (weights * velocity[:, 1])[None, :]) @ kernel_x_mass.T
    ) / float(grid.dx)
    return flux_x, flux_y


def reflected_flux_divergence_rect(
    flux_x_faces: Array,
    flux_y_faces: Array,
    grid,
) -> Array:
    """Finite-volume divergence of matched reflected face fluxes."""
    return (
        jnp.diff(jnp.asarray(flux_x_faces, dtype=jnp.float64), axis=1)
        / float(grid.dx)
        + jnp.diff(jnp.asarray(flux_y_faces, dtype=jnp.float64), axis=0)
        / float(grid.dy)
    )


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
