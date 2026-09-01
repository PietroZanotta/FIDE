"""Fast reflected/Neumann raster plans for prospective vortices.

This module carries the final ``vortices_percentage`` scalar discretization
into the prospective optimizer.  Density and signed continuity source use the
same direct, cell-integrated even-reflection Gaussian.  The bandwidth is fixed
in physical units from frozen reference rollouts and never changes with the
PDE grid.

The generic primitive in :mod:`mfsi.raster` is deliberately left untouched.
For optimization, the particle locations and PDE fidelity are frozen, so the
expensive reflection matrices are built once and reused for every CRN trial,
objective call, and gradient step.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.raster import (
    reflected_gaussian_cell_mass_matrix_1d,
    rasterize_projected_particles_reflected_rect,
)

jax.config.update("jax_enable_x64", True)


class ReflectedRasterBatch(NamedTuple):
    q: jax.Array
    mass: jax.Array
    h: jax.Array
    source_mass_before_center: jax.Array
    source_mass_after_center: jax.Array
    source: jax.Array


@dataclass(frozen=True)
class ReflectedRasterPlan:
    """Precomputed reflection matrices for selected times of one rollout."""

    kernel_x: jax.Array  # [T, nx, particles]
    kernel_y: jax.Array  # [T, ny, particles]
    cell_area: float
    bandwidth: float
    image_pairs: int

    @property
    def estimated_bytes(self) -> int:
        return int(self.kernel_x.size + self.kernel_y.size) * 8


def reference_scott_bandwidth(nodes, weights) -> tuple[float, np.ndarray]:
    """Weighted 2-D Scott rule used by the corrected percentage experiment."""
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 2 or w.shape != x.shape[:2]:
        raise ValueError("nodes/weights must have shapes [time,particle,2]/[time,particle]")
    w = w / np.sum(w, axis=1, keepdims=True)
    mean = np.sum(w[..., None] * x, axis=1)
    variance = np.sum(w[..., None] * (x - mean[:, None, :]) ** 2, axis=1)
    scale = np.sqrt(np.mean(variance, axis=1))
    effective_n = 1.0 / np.sum(w * w, axis=1)
    by_time = scale * effective_n ** (-1.0 / 6.0)
    return float(np.median(by_time)), by_time


def common_reference_scott_bandwidth(
    rollout_paths: Iterable[str | Path],
) -> tuple[float, np.ndarray]:
    """Freeze one median bandwidth across an ordered reference ensemble."""
    values: list[float] = []
    for path in rollout_paths:
        with np.load(Path(path), allow_pickle=False) as rollout:
            values.append(
                reference_scott_bandwidth(rollout["nodes"], rollout["weights"])[0]
            )
    if not values:
        raise ValueError("at least one frozen reference rollout is required")
    per_reference = np.asarray(values, dtype=np.float64)
    return float(np.median(per_reference)), per_reference


def build_reflected_raster_plan(
    nodes,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
) -> ReflectedRasterPlan:
    """Build reusable matrices for a small optimization fidelity.

    Authoritative 128x64 evaluation should stream one time node instead of
    retaining a multi-gigabyte all-time plan.  The optimizer's 24x12 and 32x16
    fidelities are intentionally small enough to keep these plans resident.
    """
    x = jnp.asarray(nodes, dtype=jnp.float64)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError("nodes must have shape [time, particle, 2]")
    x_edges = jnp.linspace(grid.x_min, grid.x_max, grid.nx + 1, dtype=jnp.float64)
    y_edges = jnp.linspace(grid.y_min, grid.y_max, grid.ny + 1, dtype=jnp.float64)
    make_x = lambda row: reflected_gaussian_cell_mass_matrix_1d(
        x_edges, row[:, 0], bandwidth=float(bandwidth), image_pairs=int(image_pairs)
    )
    make_y = lambda row: reflected_gaussian_cell_mass_matrix_1d(
        y_edges, row[:, 1], bandwidth=float(bandwidth), image_pairs=int(image_pairs)
    )
    return ReflectedRasterPlan(
        kernel_x=jax.vmap(make_x)(x),
        kernel_y=jax.vmap(make_y)(x),
        cell_area=float(grid.cell_area),
        bandwidth=float(bandwidth),
        image_pairs=int(image_pairs),
    )


def rasterize_reflected_with_plan(
    weights,
    forcing,
    plan: ReflectedRasterPlan,
) -> ReflectedRasterBatch:
    """Raster a ``[trial,time,particle]`` bank with cached reflection matrices."""
    w = jnp.asarray(weights, dtype=jnp.float64)
    f = jnp.asarray(forcing, dtype=jnp.float64)
    if w.shape != f.shape or w.ndim != 3:
        raise ValueError("weights and forcing must share [trial,time,particle] shape")
    if w.shape[1] != plan.kernel_x.shape[0] or w.shape[2] != plan.kernel_x.shape[2]:
        raise ValueError("raster plan does not match trajectory shape")

    # Vmap over time keeps the contraction intermediates at [trial,ny,nx]
    # rather than materializing a particle-by-grid tensor.
    time_weights = jnp.swapaxes(w, 0, 1)
    time_source_weights = jnp.swapaxes(w * f, 0, 1)

    def contract(kernel_x, kernel_y, values):
        return jnp.einsum("yn,bn,xn->byx", kernel_y, values, kernel_x)

    mass = jnp.swapaxes(
        jax.vmap(contract)(plan.kernel_x, plan.kernel_y, time_weights), 0, 1
    )
    source_mass = jnp.swapaxes(
        jax.vmap(contract)(plan.kernel_x, plan.kernel_y, time_source_weights), 0, 1
    )
    total_mass = jnp.sum(mass, axis=(-2, -1), keepdims=True)
    source_before = jnp.sum(source_mass, axis=(-2, -1))
    source_mass = source_mass - mass * (
        source_before[..., None, None] / jnp.maximum(total_mass, 1.0e-300)
    )
    source_after = jnp.sum(source_mass, axis=(-2, -1))
    q = mass / float(plan.cell_area)
    source = source_mass / float(plan.cell_area)
    return ReflectedRasterBatch(
        q=q,
        mass=mass,
        h=source / q,
        source_mass_before_center=source_before,
        source_mass_after_center=source_after,
        source=source,
    )


def rasterize_reflected_single_time(
    nodes,
    weights,
    forcing,
    grid,
    *,
    bandwidth: float,
    image_pairs: int = 4,
):
    """Authoritative streaming primitive: one kernel pair, all trials."""
    one = lambda w, f: rasterize_projected_particles_reflected_rect(
        nodes,
        w,
        f,
        grid,
        bandwidth=float(bandwidth),
        image_pairs=int(image_pairs),
    )
    return jax.vmap(one)(weights, forcing)


__all__ = [
    "ReflectedRasterBatch",
    "ReflectedRasterPlan",
    "build_reflected_raster_plan",
    "common_reference_scott_bandwidth",
    "rasterize_reflected_single_time",
    "rasterize_reflected_with_plan",
    "reference_scott_bandwidth",
]
