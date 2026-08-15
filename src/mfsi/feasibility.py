from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class SupportPolytopeConfig:
    directions: int = 96
    margin: float = 0.0
    feasibility_tol: float = 1.0e-9


class ProjectionResult(NamedTuple):
    beta: Array
    active: Array
    distance: Array
    max_unconstrained_violation: Array


class Polytope2D(NamedTuple):
    A: Array
    b: Array
    vertices: Array
    vertex_feasible: Array


def unit_directions_2d(count: int) -> Array:
    if count < 8:
        raise ValueError("support direction count must be >= 8")
    theta = 2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float64) / float(count)
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


def common_beta_support_polytope_2d(
    *,
    directions: Array,
    times: Array,
    c0: Array,
    c1: Array,
    physical_features: Array,
    particle_features_by_time: Array,
    particle_mask_by_time: Array,
    margin: float = 0.0,
) -> tuple[Array, Array, Array]:
    """Build ``A beta <= b`` from support functions of physical and particle hulls.

    Fixed moment-space directions make this construction JAX differentiable with
    respect to the observable parameters. For each direction we retain the tightest
    time constraint, yielding a compact 2-D polygon in beta-space.
    """
    d = jnp.asarray(directions, dtype=jnp.float64)                   # [D,2]
    times = jnp.asarray(times, dtype=jnp.float64)
    c0 = jnp.asarray(c0, dtype=jnp.float64)
    c1 = jnp.asarray(c1, dtype=jnp.float64)
    physical_features = jnp.asarray(physical_features, dtype=jnp.float64)  # [P,2]
    particle_features_by_time = jnp.asarray(particle_features_by_time, dtype=jnp.float64)  # [T,N,2]
    particle_mask_by_time = jnp.asarray(particle_mask_by_time, dtype=bool)

    physical_support = jnp.max(physical_features @ d.T, axis=0)  # [D]
    particle_scores = jnp.einsum("tnm,dm->tnd", particle_features_by_time, d)
    particle_scores = jnp.where(particle_mask_by_time[..., None], particle_scores, -jnp.inf)
    particle_support = jnp.max(particle_scores, axis=1)           # [T,D]
    joint_support = jnp.minimum(physical_support[None, :], particle_support)

    z = times * (1.0 - times)
    bridge = (1.0 - times[:, None]) * c0[None, :] + times[:, None] * c1[None, :]
    bridge_support = bridge @ d.T
    slack = joint_support - bridge_support - float(margin)         # [T,D]

    interior = z > 1.0e-12
    beta_bounds = jnp.where(interior[:, None], slack / jnp.maximum(z[:, None], 1.0e-12), jnp.inf)
    b = jnp.min(beta_bounds, axis=0)
    endpoint_violation = jnp.max(jnp.where(interior[:, None], -jnp.inf, -slack))
    return d, b, jnp.maximum(endpoint_violation, 0.0)


def _vertex_candidates(A: Array, b: Array) -> tuple[Array, Array]:
    D = A.shape[0]
    i, j = jnp.triu_indices(D, k=1)
    ai, aj = A[i], A[j]
    det = ai[:, 0] * aj[:, 1] - ai[:, 1] * aj[:, 0]
    safe = jnp.abs(det) > 1.0e-10
    det_safe = jnp.where(safe, det, 1.0)
    x = (b[i] * aj[:, 1] - ai[:, 1] * b[j]) / det_safe
    y = (ai[:, 0] * b[j] - b[i] * aj[:, 0]) / det_safe
    v = jnp.stack([x, y], axis=-1)
    return v, safe




def prepare_polytope_2d(A: Array, b: Array, *, tol: float = 1.0e-9) -> Polytope2D:
    A = jnp.asarray(A, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    vertices, nonparallel = _vertex_candidates(A, b)
    vertex_feasible = nonparallel & jnp.all(A @ vertices.T <= b[:, None] + tol, axis=0)
    return Polytope2D(A, b, vertices, vertex_feasible)

def project_metric_polytope_2d(
    beta_unconstrained: Array,
    H: Array,
    A: Array | None = None,
    b: Array | None = None,
    *,
    polytope: Polytope2D | None = None,
    tol: float = 1.0e-9,
) -> ProjectionResult:
    """Exact 2-D metric projection onto a polygon represented by halfspaces.

    The optimum of a strictly convex quadratic over a polygon is either the
    unconstrained point, a projection onto one facet, or a feasible vertex. We
    enumerate those finite candidates; no iterative optimizer or SciPy call is
    needed. Gradients are piecewise exact through the selected candidate.
    """
    beta_u = jnp.asarray(beta_unconstrained, dtype=jnp.float64)
    H = jnp.asarray(H, dtype=jnp.float64)
    if polytope is None:
        if A is None or b is None:
            raise ValueError("pass A,b or a prepared polytope")
        polytope = prepare_polytope_2d(A, b, tol=tol)
    A, b = polytope.A, polytope.b
    Hinv = jnp.linalg.inv(H)

    def feasible(x):
        return jnp.all(A @ x <= b + tol)

    violation = jnp.maximum(0.0, jnp.max(A @ beta_u - b))

    # H-metric projection onto each single facet.
    Hinv_a = A @ Hinv.T
    denom = jnp.einsum("di,di->d", A, Hinv_a)
    step = (A @ beta_u - b) / jnp.maximum(denom, 1.0e-18)
    facets = beta_u[None, :] - step[:, None] * Hinv_a
    facet_feasible = jax.vmap(feasible)(facets)

    vertices = polytope.vertices
    vertex_feasible = polytope.vertex_feasible

    candidates = jnp.concatenate([beta_u[None, :], facets, vertices, jnp.zeros((1, 2), dtype=beta_u.dtype)], axis=0)
    mask = jnp.concatenate([
        jnp.asarray([feasible(beta_u)]),
        facet_feasible,
        vertex_feasible,
        jnp.asarray([feasible(jnp.zeros(2, dtype=beta_u.dtype))]),
    ])
    delta = candidates - beta_u[None, :]
    costs = 0.5 * jnp.einsum("ni,ij,nj->n", delta, H, delta)
    costs = jnp.where(mask, costs, jnp.inf)
    best = jnp.argmin(costs)
    beta = candidates[best]
    distance = jnp.sqrt(jnp.maximum(2.0 * costs[best], 0.0))
    return ProjectionResult(beta, violation > tol, distance, violation)
