"""Screened full correction for finite active-nematic defect measures.

Continuity convention
---------------------
The normalized shape machinery returns ``h_shape`` satisfying

    partial_t q + div(q u) = q h_shape.

For ``mu=M q`` and reference source ``g_ref``, the residual is

    h_ub = h_shape + dot(M)/M - g_ref.

We impose the correction constraint

    div(mu delta) - mu alpha = -mu h_ub

and minimize ``integral mu (|delta|^2 + kappa alpha^2)``.  With
``delta=grad(psi)`` and ``alpha=psi/kappa``, stationarity gives the symmetric
positive-definite screened problem

    -div(q grad(psi)) + (q/kappa) psi = q h_ub.

The spatially constant mass ``M`` cancels from the PDE but multiplies both
reported action terms.  The reaction term removes the constant nullspace, so
this solver intentionally has no gauge constraint.  As ``kappa -> infinity``,
``psi`` is the negative of the legacy balanced solver's potential under its
RHS convention; the physical correction velocity and Dirichlet action agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mfsi.linear import implicit_cg

try:
    from .periodic_numerics import (
        PeriodicGrid2D,
        PeriodicGrid3D,
        periodic_weighted_laplacian,
        periodic_weighted_laplacian3d,
        periodic_weighted_laplacian_diag,
        periodic_weighted_laplacian_diag3d,
        stable_relative_residual,
    )
except ImportError:  # pragma: no cover
    from periodic_numerics import (
        PeriodicGrid2D,
        PeriodicGrid3D,
        periodic_weighted_laplacian,
        periodic_weighted_laplacian3d,
        periodic_weighted_laplacian_diag,
        periodic_weighted_laplacian_diag3d,
        stable_relative_residual,
    )


Array = jax.Array


@dataclass(frozen=True)
class UnbalancedCorrectionConfig:
    reaction_kappa: float = 1.0
    operator_floor_rel: float = 0.0
    cg_tol: float = 1.0e-8
    cg_maxiter: int = 800

    def __post_init__(self) -> None:
        if self.reaction_kappa <= 0.0:
            raise ValueError("reaction_kappa must be positive")
        if self.operator_floor_rel < 0.0:
            raise ValueError("operator_floor_rel must be nonnegative")
        if self.cg_tol <= 0.0 or self.cg_maxiter < 1:
            raise ValueError("invalid screened-CG tolerance or iteration count")


class UnbalancedCorrectionResult(NamedTuple):
    total_action: Array
    move_action: Array
    reaction_action: Array
    reaction_fraction: Array
    potential: Array
    source_correction: Array
    relative_residual: Array
    stabilized_relative_residual: Array
    operator_floor: Array


def solve_unbalanced_screened_poisson_batch_jax(
    q: Array,
    h_ub: Array,
    *,
    mass: Array,
    grid: PeriodicGrid3D,
    config: UnbalancedCorrectionConfig = UnbalancedCorrectionConfig(),
) -> UnbalancedCorrectionResult:
    """Vectorized JAX fallback for a trajectory of independent 3D systems."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h_ub = jnp.asarray(h_ub, dtype=jnp.float64)
    mass = jnp.asarray(mass, dtype=jnp.float64)
    if q.ndim != 4 or q.shape[1:] != grid.shape or h_ub.shape != q.shape:
        raise ValueError(f"q and h_ub must have shape [B,{','.join(map(str, grid.shape))}]")
    if mass.shape != (q.shape[0],):
        raise ValueError("mass must have one scalar per batch system")
    return jax.vmap(
        lambda q_row, h_row, mass_row: solve_unbalanced_screened_poisson(
            q_row, h_row, mass=mass_row, grid=grid, config=config
        )
    )(q, h_ub, mass)


def unbalanced_residual(
    shape_residual: Array,
    target_relative_mass_rate: Array,
    reference_source_rate: Array,
) -> Array:
    """Return ``h_shape + dot(M)/M - g_ref`` with explicit naming."""
    return (
        jnp.asarray(shape_residual, dtype=jnp.float64)
        + jnp.asarray(target_relative_mass_rate, dtype=jnp.float64)
        - jnp.asarray(reference_source_rate, dtype=jnp.float64)
    )


def _operator_and_diagonal(
    q: Array,
    grid: PeriodicGrid2D | PeriodicGrid3D,
):
    if isinstance(grid, PeriodicGrid2D):
        return (
            lambda psi, density: periodic_weighted_laplacian(
                psi, density, grid.dx
            ),
            lambda density: periodic_weighted_laplacian_diag(density, grid.dx),
            grid.cell_area,
            (0, 1),
        )
    return (
        lambda psi, density: periodic_weighted_laplacian3d(
            psi, density, grid.spacings
        ),
        lambda density: periodic_weighted_laplacian_diag3d(
            density, grid.spacings
        ),
        grid.cell_volume,
        (0, 1, 2),
    )


def solve_unbalanced_screened_poisson(
    q: Array,
    h_ub: Array,
    *,
    mass: Array | float,
    grid: PeriodicGrid2D | PeriodicGrid3D,
    config: UnbalancedCorrectionConfig = UnbalancedCorrectionConfig(),
) -> UnbalancedCorrectionResult:
    """Solve one time-local normalized-shape screened correction system."""
    q = jnp.asarray(q, dtype=jnp.float64)
    h_ub = jnp.asarray(h_ub, dtype=jnp.float64)
    mass = jnp.asarray(mass, dtype=jnp.float64)
    expected = (grid.n, grid.n) if isinstance(grid, PeriodicGrid2D) else grid.shape
    if q.shape != expected or h_ub.shape != expected:
        raise ValueError(f"q and h_ub must both have grid shape {expected}")
    if mass.ndim != 0:
        raise ValueError("species mass must be scalar at a fixed time")

    apply_laplacian, diagonal_laplacian, cell_volume, axes = _operator_and_diagonal(
        q, grid
    )
    floor = float(config.operator_floor_rel) * jnp.max(q)
    q_operator = q + floor
    inverse_kappa = 1.0 / float(config.reaction_kappa)
    rhs = q * h_ub

    def apply(psi: Array, density: Array) -> Array:
        return apply_laplacian(psi, density) + inverse_kappa * density * psi

    def matvec(flat: Array) -> Array:
        return apply(flat.reshape(expected), q_operator).reshape(-1)

    diagonal = diagonal_laplacian(q_operator) + inverse_kappa * q_operator
    solution = implicit_cg(
        matvec,
        rhs.reshape(-1),
        tol=float(config.cg_tol),
        maxiter=int(config.cg_maxiter),
        preconditioner=lambda row: row
        / jnp.maximum(diagonal.reshape(-1), 1.0e-12),
    ).reshape(expected)

    physical_laplacian = apply_laplacian(solution, q)
    move = mass * cell_volume * jnp.sum(solution * physical_laplacian)
    reaction = (
        mass
        * cell_volume
        * inverse_kappa
        * jnp.sum(q * solution**2)
    )
    total = move + reaction
    physical_residual = apply(solution, q) - rhs
    stabilized_residual = apply(solution, q_operator) - rhs
    relative = stable_relative_residual(physical_residual, rhs, axes=axes)
    stabilized_relative = stable_relative_residual(
        stabilized_residual, rhs, axes=axes
    )
    return UnbalancedCorrectionResult(
        total_action=total,
        move_action=move,
        reaction_action=reaction,
        reaction_fraction=jnp.where(total > 0.0, reaction / total, 0.0),
        potential=solution,
        source_correction=solution * inverse_kappa,
        relative_residual=relative,
        stabilized_relative_residual=stabilized_relative,
        operator_floor=floor,
    )
