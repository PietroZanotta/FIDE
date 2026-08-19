"""Experiment-local periodic rasterization and full-action Poisson solves.

These routines deliberately do not alter the shared rectangular/Neumann path
used by vortices and ocean drifters. Position-only states use the spatial torus;
polarity-aware states use the product torus ``(x, y, r_theta * theta)``. The
explicit ``r_theta`` makes the metric relating translation and rotation part of
the experiment rather than an accidental grid-resolution choice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mfsi.linear import implicit_cg

Array = jax.Array


@dataclass(frozen=True)
class PeriodicGrid2D:
    box_size: float
    n: int

    def __post_init__(self) -> None:
        if self.box_size <= 0.0 or int(self.n) < 4:
            raise ValueError("box_size must be positive and n >= 4")

    @property
    def dx(self) -> float:
        return self.box_size / self.n

    @property
    def cell_area(self) -> float:
        return self.dx**2

    def flat_bin_index(self, x: Array) -> Array:
        position = jnp.mod(jnp.asarray(x, dtype=jnp.float64)[..., :2], self.box_size)
        ij = jnp.floor(position / self.dx).astype(jnp.int32)
        return ij[..., 1] * self.n + ij[..., 0]


@dataclass(frozen=True)
class PeriodicGrid3D:
    """An anisotropic product torus for position and polarity angle."""

    box_size: float
    shape: tuple[int, int, int]
    polarity_metric_radius: float = 1.0

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if self.box_size <= 0.0:
            raise ValueError("box_size must be positive")
        if len(shape) != 3 or any(value < 3 for value in shape):
            raise ValueError("shape must contain three periodic dimensions >= 3")
        if self.polarity_metric_radius <= 0.0:
            raise ValueError("polarity_metric_radius must be positive")
        object.__setattr__(self, "shape", shape)

    @property
    def dx(self) -> float:
        return self.box_size / self.shape[0]

    @property
    def dy(self) -> float:
        return self.box_size / self.shape[1]

    @property
    def dtheta_metric(self) -> float:
        return self.polarity_metric_radius * 2.0 * jnp.pi / self.shape[2]

    @property
    def spacings(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, float(self.dtheta_metric))

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * float(self.dtheta_metric)

    def flat_bin_index(self, x: Array) -> Array:
        state = jnp.asarray(x, dtype=jnp.float64)
        if state.shape[-1] != 3:
            raise ValueError("polarity rasterization requires state shape [...,3]")
        nx, ny, ntheta = self.shape
        ix = jnp.floor(jnp.mod(state[..., 0], self.box_size) / self.dx).astype(jnp.int32)
        iy = jnp.floor(jnp.mod(state[..., 1], self.box_size) / self.dy).astype(jnp.int32)
        itheta = jnp.floor(
            jnp.mod(state[..., 2], 2.0 * jnp.pi) * ntheta / (2.0 * jnp.pi)
        ).astype(jnp.int32)
        return (ix * ny + iy) * ntheta + itheta


@dataclass(frozen=True)
class PeriodicPoissonConfig:
    operator_floor_rel: float = 2.0e-5
    cg_tol: float = 1.0e-8
    cg_maxiter: int = 520
    gauge_strength: float = 1.0


class PeriodicRasterResult(NamedTuple):
    q: Array
    mass: Array
    h: Array


class PeriodicPoissonResult(NamedTuple):
    action: Array
    potential: Array
    relative_residual: Array
    weighted_mean_potential: Array
    operator_floor: Array


class PeriodicPoissonBatchResult(NamedTuple):
    action: Array
    potential: Array
    relative_residual: Array
    weighted_mean_potential: Array
    operator_floor: Array


def stable_relative_residual(
    residual: Array, rhs: Array, *, axes: tuple[int, ...]
) -> Array:
    """Relative L2 residual with a finite derivative for an exact zero system."""
    residual_squared = jnp.sum(jnp.asarray(residual) ** 2, axis=axes)
    rhs_squared = jnp.sum(jnp.asarray(rhs) ** 2, axis=axes)
    ratio = jnp.sqrt(residual_squared + 1.0e-30) / jnp.sqrt(
        rhs_squared + 1.0e-30
    )
    return jnp.where(rhs_squared > 1.0e-28, ratio, 0.0)


def _periodic_gaussian_blur(field: Array, sigma: float, grid: PeriodicGrid2D) -> Array:
    if sigma <= 0.0:
        return field
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.n, d=grid.dx)
    kx, ky = jnp.meshgrid(k, k, indexing="xy")
    transfer = jnp.exp(-0.5 * float(sigma) ** 2 * (kx**2 + ky**2))
    return jnp.fft.ifft2(jnp.fft.fft2(field) * transfer).real


def _periodic_gaussian_blur3d(
    field: Array, sigma: float, grid: PeriodicGrid3D
) -> Array:
    if sigma <= 0.0:
        return field
    nx, ny, ntheta = grid.shape
    # A sampled positive kernel stays well behaved even when sigma is smaller
    # than one cell. A continuous spectral multiplier can ring in that regime
    # and create small negative density values.
    def periodic_distance(count: int, spacing: float) -> Array:
        index = jnp.arange(count, dtype=jnp.float64)
        return jnp.minimum(index, count - index) * spacing

    x = periodic_distance(nx, grid.dx)
    y = periodic_distance(ny, grid.dy)
    theta = periodic_distance(ntheta, float(grid.dtheta_metric))
    squared_distance = (
        x[:, None, None] ** 2
        + y[None, :, None] ** 2
        + theta[None, None, :] ** 2
    )
    kernel = jnp.exp(-0.5 * squared_distance / float(sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return jnp.fft.ifftn(jnp.fft.fftn(field) * jnp.fft.fftn(kernel)).real


def rasterize_periodic_particles(
    x: Array,
    weights: Array,
    forcing: Array,
    grid: PeriodicGrid2D,
    *,
    bandwidth: float = 0.0,
) -> PeriodicRasterResult:
    """Rasterize frozen particles while retaining gradients in weights/forcing."""
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    if x.shape[-1] != 2:
        raise ValueError("periodic full-action rasterization currently supports state_dim=2")
    index = grid.flat_bin_index(x)
    mass = jnp.zeros(grid.n * grid.n, dtype=jnp.float64).at[index].add(weights)
    source = jnp.zeros(grid.n * grid.n, dtype=jnp.float64).at[index].add(weights * forcing)
    mass = mass.reshape((grid.n, grid.n))
    source = source.reshape((grid.n, grid.n))
    sigma = float(bandwidth) if bandwidth > 0.0 else 0.35 * grid.dx
    mass = _periodic_gaussian_blur(mass, sigma, grid)
    source = _periodic_gaussian_blur(source, sigma, grid)
    norm = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / norm
    source = source / norm
    source = source - mass * jnp.sum(source)
    occupied = mass > 1.0e-300
    h = jnp.where(occupied, source / jnp.where(occupied, mass, 1.0), 0.0)
    return PeriodicRasterResult(mass / grid.cell_area, mass, h)


def rasterize_periodic_particles3d(
    x: Array,
    weights: Array,
    forcing: Array,
    grid: PeriodicGrid3D,
    *,
    bandwidth: float = 0.0,
) -> PeriodicRasterResult:
    """Rasterize particles on ``(x,y,theta)`` using the declared product metric."""
    x = jnp.asarray(x, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    if x.shape[-1] != 3:
        raise ValueError("polarity full-action rasterization requires state_dim=3")
    if weights.shape != x.shape[:-1] or forcing.shape != weights.shape:
        raise ValueError("weights and forcing must match the particle leading shape")
    index = grid.flat_bin_index(x)
    size = math.prod(grid.shape)
    mass = jnp.zeros(size, dtype=jnp.float64).at[index].add(weights).reshape(grid.shape)
    source = (
        jnp.zeros(size, dtype=jnp.float64)
        .at[index]
        .add(weights * forcing)
        .reshape(grid.shape)
    )
    sigma = (
        float(bandwidth)
        if bandwidth > 0.0
        else 0.35 * min(grid.dx, grid.dy, float(grid.dtheta_metric))
    )
    mass = _periodic_gaussian_blur3d(mass, sigma, grid)
    source = _periodic_gaussian_blur3d(source, sigma, grid)
    mass = jnp.maximum(mass, 0.0)
    norm = jnp.maximum(jnp.sum(mass), 1.0e-300)
    mass = mass / norm
    source = source / norm
    # Remove source values at numerical zero-density cells, then enforce
    # discrete compatibility exactly after FFT roundoff.
    source = jnp.where(mass > 0.0, source, 0.0)
    source = source - mass * jnp.sum(source)
    occupied = mass > 0.0
    h = jnp.where(occupied, source / jnp.where(occupied, mass, 1.0), 0.0)
    return PeriodicRasterResult(mass / grid.cell_volume, mass, h)


def periodic_weighted_laplacian(psi: Array, q: Array, dx: float) -> Array:
    """Discrete ``-div(q grad psi)`` with wraparound edge fluxes."""
    out = jnp.zeros_like(psi)
    dx2 = float(dx) ** 2
    for axis in (0, 1):
        neighbor = jnp.roll(psi, -1, axis=axis)
        q_edge = 0.5 * (q + jnp.roll(q, -1, axis=axis))
        flux = q_edge * (psi - neighbor) / dx2
        out = out + flux - jnp.roll(flux, 1, axis=axis)
    return out


def periodic_weighted_laplacian_diag(q: Array, dx: float) -> Array:
    dx2 = float(dx) ** 2
    diag = jnp.zeros_like(q)
    for axis in (0, 1):
        edge = 0.5 * (q + jnp.roll(q, -1, axis=axis)) / dx2
        diag = diag + edge + jnp.roll(edge, 1, axis=axis)
    return diag


def periodic_weighted_laplacian3d(
    potential: Array,
    q: Array,
    spacings: tuple[float, float, float],
) -> Array:
    """Seven-point ``-div(q grad potential)`` on the anisotropic product torus."""
    potential = jnp.asarray(potential, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    if potential.shape != q.shape or potential.ndim < 3:
        raise ValueError("potential and q must have identical rank >= 3 shapes")
    out = jnp.zeros_like(potential)
    for axis, spacing in zip((-3, -2, -1), spacings, strict=True):
        neighbor = jnp.roll(potential, -1, axis=axis)
        q_edge = 0.5 * (q + jnp.roll(q, -1, axis=axis))
        flux = q_edge * (potential - neighbor) / float(spacing) ** 2
        out = out + flux - jnp.roll(flux, 1, axis=axis)
    return out


def periodic_weighted_laplacian_diag3d(
    q: Array, spacings: tuple[float, float, float]
) -> Array:
    q = jnp.asarray(q, dtype=jnp.float64)
    if q.ndim < 3:
        raise ValueError("q must have rank >= 3")
    diagonal = jnp.zeros_like(q)
    for axis, spacing in zip((-3, -2, -1), spacings, strict=True):
        edge = 0.5 * (q + jnp.roll(q, -1, axis=axis)) / float(spacing) ** 2
        diagonal = diagonal + edge + jnp.roll(edge, 1, axis=axis)
    return diagonal


def prepare_periodic_poisson3d_batch(
    q: Array,
    h: Array,
    cfg: PeriodicPoissonConfig,
) -> tuple[Array, Array, Array, Array]:
    """Construct the stabilized coefficient, compatible RHS, gauge, and floor."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    if q.ndim != 4 or h.shape != q.shape:
        raise ValueError("q and h must have identical [B,Nx,Ny,Ntheta] shapes")
    spatial_axes = (-3, -2, -1)
    floor = cfg.operator_floor_rel * jnp.max(q, axis=spatial_axes, keepdims=True)
    q_operator = q + floor
    rhs = -(q * h)
    flat_q = q.reshape((q.shape[0], -1))
    gauge = flat_q / jnp.maximum(
        jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300
    )
    return q_operator, rhs, gauge.reshape(q.shape), floor.reshape((q.shape[0],))


def solve_periodic_weighted_poisson3d_batch_jax(
    q: Array,
    h: Array,
    grid: PeriodicGrid3D,
    cfg: PeriodicPoissonConfig = PeriodicPoissonConfig(),
) -> PeriodicPoissonBatchResult:
    """Sound JAX reference for a batch of anisotropic periodic systems."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    if q.shape[1:] != grid.shape:
        raise ValueError(f"q batch grid must have shape {grid.shape}, got {q.shape[1:]}")
    q_operator, rhs, gauge, floor = prepare_periodic_poisson3d_batch(q, h, cfg)
    shape = grid.shape
    spacings = grid.spacings

    def solve_one(qi: Array, ri: Array, gi: Array) -> Array:
        def matvec(flat: Array) -> Array:
            potential = flat.reshape(shape)
            value = periodic_weighted_laplacian3d(
                potential, qi, spacings
            ).reshape(-1)
            gauge_flat = gi.reshape(-1)
            return value + cfg.gauge_strength * gauge_flat * jnp.dot(
                gauge_flat, flat
            )

        gauge_flat = gi.reshape(-1)
        diagonal = periodic_weighted_laplacian_diag3d(qi, spacings).reshape(-1)
        diagonal = diagonal + cfg.gauge_strength * gauge_flat**2
        return implicit_cg(
            matvec,
            ri.reshape(-1),
            tol=cfg.cg_tol,
            maxiter=cfg.cg_maxiter,
            preconditioner=lambda residual: residual / jnp.maximum(diagonal, 1.0e-12),
        ).reshape(shape)

    potential = jax.vmap(solve_one)(q_operator, rhs, gauge)
    stabilized = periodic_weighted_laplacian3d(potential, q_operator, spacings)
    gauge_dot = jnp.sum(gauge * potential, axis=(-3, -2, -1), keepdims=True)
    residual = stabilized + cfg.gauge_strength * gauge * gauge_dot - rhs
    relative = stable_relative_residual(
        residual, rhs, axes=(-3, -2, -1)
    )
    physical = periodic_weighted_laplacian3d(potential, q, spacings)
    action = grid.cell_volume * jnp.sum(
        potential * physical, axis=(-3, -2, -1)
    )
    weighted_mean = grid.cell_volume * jnp.sum(
        q * potential, axis=(-3, -2, -1)
    )
    return PeriodicPoissonBatchResult(
        action, potential, relative, weighted_mean, floor
    )


def solve_periodic_weighted_poisson(
    q: Array,
    h: Array,
    grid: PeriodicGrid2D,
    cfg: PeriodicPoissonConfig = PeriodicPoissonConfig(),
) -> PeriodicPoissonResult:
    """Solve the position-only weighted Poisson problem on a spatial torus."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    if q.shape != (grid.n, grid.n) or h.shape != q.shape:
        raise ValueError(f"q and h must both have shape {(grid.n, grid.n)}")
    floor = cfg.operator_floor_rel * jnp.max(q)
    q_operator = q + floor
    rhs = -(q * h).reshape(-1)
    gauge = q.reshape(-1)
    gauge = gauge / jnp.maximum(jnp.linalg.norm(gauge), 1.0e-300)

    def matvec(flat: Array) -> Array:
        psi = flat.reshape(q.shape)
        value = periodic_weighted_laplacian(psi, q_operator, grid.dx).reshape(-1)
        return value + cfg.gauge_strength * gauge * jnp.dot(gauge, flat)

    diag = periodic_weighted_laplacian_diag(q_operator, grid.dx).reshape(-1)
    diag = diag + cfg.gauge_strength * gauge**2
    potential_flat = implicit_cg(
        matvec,
        rhs,
        tol=cfg.cg_tol,
        maxiter=cfg.cg_maxiter,
        preconditioner=lambda residual: residual / jnp.maximum(diag, 1.0e-10),
    )
    potential = potential_flat.reshape(q.shape)
    physical = periodic_weighted_laplacian(potential, q, grid.dx)
    action = grid.cell_area * jnp.sum(potential * physical)
    residual = matvec(potential_flat) - rhs
    relative = stable_relative_residual(residual, rhs, axes=(0,))
    return PeriodicPoissonResult(
        action,
        potential,
        relative,
        grid.cell_area * jnp.sum(q * potential),
        floor,
    )
