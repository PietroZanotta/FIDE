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


class PhysicalPoissonBatchResult(NamedTuple):
    """Host-side equation-preserving authoritative batch solve."""

    action: object
    potential: object
    relative_residual: object
    absolute_residual: object
    maximum_component_compatibility_residual: object
    component_count: object
    compatible: object
    solver_converged: object


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

    The reverse pass propagates through both the RHS ``b = q h`` and the
    q-dependent operator ``K(q)``. CG iterations themselves are not unrolled.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)

    q_floor = cfg.operator_floor_rel * jnp.max(q)
    rhs = (q * h).reshape(-1)

    gauge_vector = q.reshape(-1)
    gauge_vector = gauge_vector / jnp.maximum(jnp.linalg.norm(gauge_vector), 1.0e-300)

    def matvec(z_flat: Array) -> Array:
        z = z_flat.reshape(q.shape)
        kz = weighted_laplacian(z, q, cfg.dx).reshape(-1)
        return kz + cfg.gauge_strength * gauge_vector * jnp.dot(gauge_vector, z_flat)

    # The density floor is preconditioning-only.  It improves diagonal scaling
    # without changing the scientific equation K(q) psi = q h.
    q_preconditioner = q + q_floor
    diag = weighted_laplacian_diag(q_preconditioner, cfg.dx).reshape(-1)
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


def solve_weighted_poisson_physical_direct_batch(
    q: object,
    h: object,
    cfg: PoissonConfig,
    *,
    compatibility_tolerance: float = 1.0e-10,
    reject_incompatible: bool = False,
) -> PhysicalPoissonBatchResult:
    """Solve ``K(q) psi = q h`` exactly on each conductive component.

    This host-side authoritative path never adds density to the operator.  Exact
    zero-conductance regions are removed, each connected conductive component is
    pinned once, and a sparse direct solve (with an equation-preserving PCG
    fallback) is used.  Component compatibility is checked before solving; an
    incompatible physical equation is reported rather than regularized.  Pinning
    changes only componentwise constants, hence neither the correction field nor
    its physical Dirichlet action.  ``operator_floor_rel`` is used only to scale
    the fallback preconditioner and never enters the scientific matrix or action.
    """
    import numpy as np

    q_array = np.asarray(q, dtype=np.float64)
    h_array = np.asarray(h, dtype=np.float64)
    if q_array.ndim != 3 or h_array.shape != q_array.shape:
        raise ValueError("q and h must have identical [B,H,W] shapes")
    return _solve_weighted_poisson_physical_source_batch(
        q_array,
        q_array * h_array,
        cfg,
        compatibility_tolerance=compatibility_tolerance,
        reject_incompatible=reject_incompatible,
    )


def solve_weighted_poisson_source_physical_direct_batch(
    q: object,
    source: object,
    cfg: PoissonConfig,
    *,
    compatibility_tolerance: float = 1.0e-10,
    reject_incompatible: bool = False,
) -> PhysicalPoissonBatchResult:
    """Solve ``K(q) psi = source`` without reconstructing ``source`` as ``q*h``."""
    return _solve_weighted_poisson_physical_source_batch(
        q,
        source,
        cfg,
        compatibility_tolerance=compatibility_tolerance,
        reject_incompatible=reject_incompatible,
    )


def _solve_weighted_poisson_physical_source_batch(
    q: object,
    source: object,
    cfg: PoissonConfig,
    *,
    compatibility_tolerance: float,
    reject_incompatible: bool,
) -> PhysicalPoissonBatchResult:
    """Shared implementation for an explicitly deposited physical source."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import LinearOperator, cg, splu

    q_batch = np.asarray(q, dtype=np.float64)
    source_batch = np.asarray(source, dtype=np.float64)
    if q_batch.ndim != 3 or source_batch.shape != q_batch.shape:
        raise ValueError("q and source must have identical [B,H,W] shapes")
    if np.any(q_batch < 0.0) or not np.all(np.isfinite(q_batch)):
        raise ValueError("q must be finite and nonnegative")
    if not np.all(np.isfinite(source_batch)):
        raise ValueError("source must be finite")

    batch, ny, nx = q_batch.shape
    size = ny * nx
    indices = np.arange(size, dtype=np.int64).reshape((ny, nx))
    potentials = np.zeros_like(q_batch)
    actions = np.empty(batch, dtype=np.float64)
    relative = np.empty(batch, dtype=np.float64)
    absolute = np.empty(batch, dtype=np.float64)
    compatibility = np.empty(batch, dtype=np.float64)
    counts = np.empty(batch, dtype=np.int32)
    compatible = np.ones(batch, dtype=bool)
    solver_converged = np.ones(batch, dtype=bool)
    inv_dx2 = 1.0 / (float(cfg.dx) ** 2)

    for batch_index in range(batch):
        q_one = q_batch[batch_index]
        q_max = float(np.max(q_one))
        if not q_max > 0.0:
            raise ValueError(f"q[{batch_index}] has no positive mass")
        q_scaled = q_one / q_max
        rhs = (source_batch[batch_index] / q_max).reshape(-1)

        horizontal = 0.5 * (q_scaled[:, :-1] + q_scaled[:, 1:]) * inv_dx2
        vertical = 0.5 * (q_scaled[:-1, :] + q_scaled[1:, :]) * inv_dx2
        edge_a = np.concatenate(
            [indices[:, :-1].ravel(), indices[:-1, :].ravel()]
        )
        edge_b = np.concatenate(
            [indices[:, 1:].ravel(), indices[1:, :].ravel()]
        )
        edge_weight = np.concatenate([horizontal.ravel(), vertical.ravel()])
        conductive = edge_weight > 0.0
        edge_a = edge_a[conductive]
        edge_b = edge_b[conductive]
        edge_weight = edge_weight[conductive]
        active = np.zeros(size, dtype=bool)
        active[edge_a] = True
        active[edge_b] = True
        active |= rhs != 0.0
        active_indices = np.flatnonzero(active)
        local_index = np.full(size, -1, dtype=np.int64)
        local_index[active_indices] = np.arange(len(active_indices))
        local_a = local_index[edge_a]
        local_b = local_index[edge_b]

        rows = np.concatenate([local_a, local_b, local_a, local_b])
        columns = np.concatenate([local_a, local_b, local_b, local_a])
        values = np.concatenate(
            [edge_weight, edge_weight, -edge_weight, -edge_weight]
        )
        matrix = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(active_indices), len(active_indices)),
        ).tocsr()
        adjacency = sparse.coo_matrix(
            (
                np.ones(2 * len(local_a), dtype=np.int8),
                (
                    np.concatenate([local_a, local_b]),
                    np.concatenate([local_b, local_a]),
                ),
            ),
            shape=matrix.shape,
        ).tocsr()
        component_count, labels = connected_components(
            adjacency, directed=False, return_labels=True
        )
        counts[batch_index] = int(component_count)
        rhs_active = rhs[active_indices]
        solution_active = np.zeros_like(rhs_active)
        rhs_scale = max(float(np.linalg.norm(rhs_active)), 1.0e-14)
        max_compatibility = 0.0

        for component in range(component_count):
            local_nodes = np.flatnonzero(labels == component)
            component_rhs = rhs_active[local_nodes]
            component_error = abs(float(np.sum(component_rhs))) / rhs_scale
            max_compatibility = max(max_compatibility, component_error)
            if component_error > float(compatibility_tolerance):
                compatible[batch_index] = False
                if reject_incompatible:
                    raise RuntimeError(
                        "physical-q Poisson RHS is incompatible on conductive "
                        f"component {component} of batch {batch_index}: relative "
                        f"mass residual {component_error:.6e} exceeds "
                        f"{float(compatibility_tolerance):.6e}"
                    )
            if len(local_nodes) == 1:
                continue
            component_matrix = matrix[local_nodes][:, local_nodes].tocsc()
            # Pin the best-scaled node, rather than whichever node happens to be
            # last in raster order.  This is only a gauge choice and therefore
            # leaves every physical edge equation and the action unchanged.
            component_diagonal = np.asarray(
                component_matrix.diagonal(), dtype=np.float64
            )
            pin = int(np.argmax(component_diagonal))
            free = np.arange(len(local_nodes), dtype=np.int64) != pin
            reduced = component_matrix[free][:, free].tocsc()
            reduced_rhs = component_rhs[free]
            diagonal = np.asarray(reduced.diagonal(), dtype=np.float64)
            # Symmetric Jacobi scaling is an equation-equivalent change of
            # variables.  It prevents an accepted-but-inaccurate SuperLU result
            # on physical q fields spanning many orders of magnitude.
            inverse_sqrt_diagonal = 1.0 / np.sqrt(
                np.maximum(diagonal, np.finfo(np.float64).tiny)
            )
            scaling = sparse.diags(inverse_sqrt_diagonal, format="csc")
            scaled_reduced = (scaling @ reduced @ scaling).tocsc()
            scaled_rhs = inverse_sqrt_diagonal * reduced_rhs
            try:
                factor = splu(scaled_reduced)

                def direct_solve(vector: np.ndarray) -> np.ndarray:
                    return inverse_sqrt_diagonal * factor.solve(
                        inverse_sqrt_diagonal * vector
                    )

                component_solution = direct_solve(reduced_rhs)
                # SuperLU can finish without an exception while losing accuracy
                # on a severely ill-scaled component.  Residual-based refinement
                # uses the same factorization and does not alter the equation.
                reduced_scale = max(float(np.linalg.norm(reduced_rhs)), 1.0e-14)
                for _ in range(6):
                    refinement_residual = reduced_rhs - reduced @ component_solution
                    if float(np.linalg.norm(refinement_residual)) <= 1.0e-12 * reduced_scale:
                        break
                    component_solution += direct_solve(refinement_residual)
            except RuntimeError:
                # SuperLU can reject an exactly representable but extremely
                # ill-scaled physical component.  Fall back to equation-preserving
                # PCG; the density floor enters only its diagonal preconditioner.
                floor_diagonal = float(cfg.operator_floor_rel) * inv_dx2
                preconditioner_diagonal = diagonal + floor_diagonal
                preconditioner = LinearOperator(
                    reduced.shape,
                    matvec=lambda vector, d=preconditioner_diagonal: vector
                    / np.maximum(d, 1.0e-300),
                    dtype=np.float64,
                )
                component_solution, info = cg(
                    reduced,
                    reduced_rhs,
                    M=preconditioner,
                    rtol=float(cfg.cg_tol),
                    atol=0.0,
                    maxiter=max(2000, 10 * int(cfg.cg_maxiter)),
                )
                if info != 0 or not np.all(np.isfinite(component_solution)):
                    solver_converged[batch_index] = False
                    component_solution = np.nan_to_num(
                        component_solution, nan=0.0, posinf=0.0, neginf=0.0
                    )
            solution_active[local_nodes[free]] = component_solution

        psi_flat = np.zeros(size, dtype=np.float64)
        psi_flat[active_indices] = solution_active
        psi = psi_flat.reshape((ny, nx))
        q_mass = float(np.sum(q_one))
        psi -= float(np.sum(q_one * psi)) / max(q_mass, 1.0e-300)
        potentials[batch_index] = psi

        operator_active = np.asarray(matrix @ solution_active, dtype=np.float64)
        residual_active = operator_active - rhs_active
        absolute[batch_index] = float(np.linalg.norm(residual_active))
        relative[batch_index] = absolute[batch_index] / rhs_scale
        compatibility[batch_index] = max_compatibility
        actions[batch_index] = float(cfg.cell_area) * q_max * float(
            np.dot(solution_active, operator_active)
        )

    return PhysicalPoissonBatchResult(
        action=actions,
        potential=potentials,
        relative_residual=relative,
        absolute_residual=absolute,
        maximum_component_compatibility_residual=compatibility,
        component_count=counts,
        compatible=compatible,
        solver_converged=solver_converged,
    )
