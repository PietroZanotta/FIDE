from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .linear import implicit_cg

Array = jax.Array


@dataclass(frozen=True)
class PoissonConfig:
    dx: float
    operator_floor_rel: float = 2.0e-5
    cg_tol: float = 1.0e-8
    cg_maxiter: int = 520
    gauge_strength: float = 1.0

    @property
    def cell_area(self) -> float:
        return self.dx * self.dx


class PoissonResult(NamedTuple):
    action: Array
    potential: Array
    relative_residual: Array
    weighted_mean_potential: Array
    operator_floor: Array


def weighted_laplacian(psi: Array, q: Array, dx: float) -> Array:
    """Discrete ``-div(q grad psi)`` with edge-averaged q on a Cartesian grid."""
    dx2 = float(dx) * float(dx)
    qx = 0.5 * (q[:, :-1] + q[:, 1:])
    qy = 0.5 * (q[:-1, :] + q[1:, :])
    out = jnp.zeros_like(psi)

    diffx = psi[:, :-1] - psi[:, 1:]
    out = out.at[:, :-1].add(qx * diffx / dx2)
    out = out.at[:, 1:].add(-qx * diffx / dx2)

    diffy = psi[:-1, :] - psi[1:, :]
    out = out.at[:-1, :].add(qy * diffy / dx2)
    out = out.at[1:, :].add(-qy * diffy / dx2)
    return out


def weighted_laplacian_diag(q: Array, dx: float) -> Array:
    dx2 = float(dx) * float(dx)
    qx = 0.5 * (q[:, :-1] + q[:, 1:]) / dx2
    qy = 0.5 * (q[:-1, :] + q[1:, :]) / dx2
    diag = jnp.zeros_like(q)
    diag = diag.at[:, :-1].add(qx)
    diag = diag.at[:, 1:].add(qx)
    diag = diag.at[:-1, :].add(qy)
    diag = diag.at[1:, :].add(qy)
    return diag


def solve_weighted_poisson(
    q: Array,
    h: Array,
    cfg: PoissonConfig,
) -> PoissonResult:
    """Solve the weighted Poisson problem with implicit linear-solve gradients.

    The reverse pass propagates through both the RHS ``b = -q h`` and the
    q-dependent operator ``K(q)``. CG iterations themselves are not unrolled.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)

    q_floor = cfg.operator_floor_rel * jnp.max(q)
    q_operator = q + q_floor
    rhs = -(q * h).reshape(-1)

    gauge_vector = q.reshape(-1)
    gauge_vector = gauge_vector / jnp.maximum(jnp.linalg.norm(gauge_vector), 1.0e-300)

    def matvec(z_flat: Array) -> Array:
        z = z_flat.reshape(q.shape)
        kz = weighted_laplacian(z, q_operator, cfg.dx).reshape(-1)
        return kz + cfg.gauge_strength * gauge_vector * jnp.dot(gauge_vector, z_flat)

    diag = weighted_laplacian_diag(q_operator, cfg.dx).reshape(-1)
    diag = diag + cfg.gauge_strength * gauge_vector * gauge_vector

    def preconditioner(r: Array) -> Array:
        return r / jnp.maximum(diag, 1.0e-10)

    psi_flat = implicit_cg(
        matvec,
        rhs,
        tol=cfg.cg_tol,
        maxiter=cfg.cg_maxiter,
        preconditioner=preconditioner,
    )
    psi = psi_flat.reshape(q.shape)
    kpsi = weighted_laplacian(psi, q_operator, cfg.dx)

    # The floor stabilizes the linear solve, but the scientific action is the
    # physical q-weighted Dirichlet energy.  Reporting energy with q_operator would
    # silently change the MFSI variational objective by adding a uniform background.
    kpsi_physical = weighted_laplacian(psi, q, cfg.dx)
    action = cfg.cell_area * jnp.sum(psi * kpsi_physical)
    residual = matvec(psi_flat) - rhs
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1.0e-14)
    weighted_mean = jnp.sum(q * cfg.cell_area * psi)

    return PoissonResult(
        action=action,
        potential=psi,
        relative_residual=relative_residual,
        weighted_mean_potential=weighted_mean,
        operator_floor=q_floor,
    )
