"""Experiment-local periodic rasterization and 2-D full-action solve.

These routines deliberately do not alter the shared rectangular/Neumann path
used by vortices and ocean drifters.  They provide the position-only fallback
on the active-nematic spatial torus.  Extending this discretization with a third
periodic polarity axis is the remaining blocker for polarity-aware full action.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _periodic_gaussian_blur(field: Array, sigma: float, grid: PeriodicGrid2D) -> Array:
    if sigma <= 0.0:
        return field
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.n, d=grid.dx)
    kx, ky = jnp.meshgrid(k, k, indexing="xy")
    transfer = jnp.exp(-0.5 * float(sigma) ** 2 * (kx**2 + ky**2))
    return jnp.fft.ifft2(jnp.fft.fft2(field) * transfer).real


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
    relative = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1.0e-14)
    return PeriodicPoissonResult(
        action,
        potential,
        relative,
        grid.cell_area * jnp.sum(q * potential),
        floor,
    )
