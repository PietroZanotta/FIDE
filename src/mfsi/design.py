from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

Array = jax.Array
Objective = Callable[[Array], Array]
Constraint = tuple[Objective, float]


@dataclass(frozen=True)
class OptimizerConfig:
    steps: int = 250
    learning_rate: float = 2.0e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    constraint_penalty: float = 1.0e4
    feasibility_tol: float = 1.0e-6


@dataclass(frozen=True)
class OptimizeResult:
    eta: Array
    value: float
    feasible: bool
    violations: tuple[float, ...]


def projective_separation_violation(min_sep_rad: float) -> Objective:
    """Compatibility name: circular separation for physical point sensors.

    GaussianSensor2D places a sensor at r(cos eta, sin eta), hence eta and eta+pi
    are antipodal locations, not the same design.  The physical angle is therefore
    2*pi-periodic.
    """
    min_sep_rad = float(min_sep_rad)

    def violation(eta: Array) -> Array:
        eta = jnp.mod(eta, 2.0 * jnp.pi)
        d = jnp.abs(eta[0] - eta[1])
        sep = jnp.minimum(d, 2.0 * jnp.pi - d)
        return min_sep_rad - sep

    return violation


def random_projective_starts(
    key: Array,
    count: int,
    *,
    min_sep_rad: float = 0.0,
    oversample: int = 32,
) -> Array:
    """Broad deterministic random starts on [0, 2*pi), with no known optima."""
    if count < 1:
        raise ValueError("count must be >= 1")

    n = max(count * oversample, count)
    candidates = jax.random.uniform(
        key,
        shape=(n, 2),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=jnp.float64,
    )
    candidates = jnp.sort(candidates, axis=1)
    d = candidates[:, 1] - candidates[:, 0]
    sep = jnp.minimum(d, 2.0 * jnp.pi - d)

    # Static-shape selection: sort valid candidates first rather than boolean-indexing.
    valid = sep >= float(min_sep_rad)
    order = jnp.argsort(~valid)
    chosen = candidates[order[:count]]
    if int(jnp.sum(valid)) < count:
        raise ValueError("Could not generate enough separated starts; increase oversample")
    return chosen


def _penalized(
    primary: Objective,
    constraints: Sequence[Constraint],
    penalty: float,
) -> Objective:
    def objective(eta: Array) -> Array:
        loss = primary(eta)
        for fn, upper in constraints:
            v = jax.nn.relu(fn(eta) - upper)
            loss = loss + penalty * v * v
        return loss

    return objective


def _adam_single(objective: Objective, eta0: Array, cfg: OptimizerConfig) -> Array:
    """JAX-loop Adam for one start."""
    eta0 = jnp.asarray(eta0, dtype=jnp.float64)
    value_and_grad = jax.value_and_grad(objective)

    def step(i, state):
        eta, m, v = state
        _, grad = value_and_grad(eta)
        t = i + 1
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grad * grad)
        mhat = m / (1.0 - cfg.beta1**t)
        vhat = v / (1.0 - cfg.beta2**t)
        eta = eta - cfg.learning_rate * mhat / (jnp.sqrt(vhat) + cfg.eps)
        return eta, m, v

    eta, _, _ = jax.lax.fori_loop(
        0,
        int(cfg.steps),
        step,
        (eta0, jnp.zeros_like(eta0), jnp.zeros_like(eta0)),
    )
    return eta


def optimize_multistart_candidates(
    primary: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
    vectorize_starts: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[OptimizeResult]:
    """Optimize every start and return all seed/optimized candidates.

    The expensive objective graph is compiled once and vectorized over starts.
    Seed incumbents are retained so a constrained phase never loses an already
    feasible design merely because an Adam step exits the feasible set.
    """
    starts = jnp.asarray(starts, dtype=jnp.float64)
    objective = _penalized(primary, constraints, cfg.constraint_penalty)

    # Heavy objectives such as the stage-4 reverse-CG action can exceed device
    # memory when all starts are vmapped together.  Keep vectorization for the cheap
    # law/tangent stages, but allow sequential execution with one shared compiled
    # graph for memory-bounded objectives.
    if vectorize_starts:
        optimize_batch = jax.jit(jax.vmap(lambda eta0: _adam_single(objective, eta0, cfg)))
        optimized = optimize_batch(starts)
        # Synchronize before reporting completion: JAX dispatch is asynchronous on
        # accelerators, so merely constructing ``optimized`` is not a useful timing
        # boundary for callers.
        if progress_callback is not None:
            jax.block_until_ready(optimized)
            progress_callback(int(starts.shape[0]), int(starts.shape[0]))
    else:
        optimize_one = jax.jit(lambda eta0: _adam_single(objective, eta0, cfg))
        optimized_rows = []
        total = int(starts.shape[0])
        for i in range(total):
            optimized_eta = optimize_one(starts[i])
            if progress_callback is not None:
                jax.block_until_ready(optimized_eta)
                progress_callback(i + 1, total)
            optimized_rows.append(optimized_eta)
        optimized = jnp.stack(optimized_rows)

    primary_eval = jax.jit(primary)
    out: list[OptimizeResult] = []

    def add_candidate(eta: Array) -> None:
        if canonicalize is not None:
            eta = canonicalize(eta)
        violations = tuple(float(fn(eta) - upper) for fn, upper in constraints)
        feasible = all(v <= cfg.feasibility_tol for v in violations)
        out.append(
            OptimizeResult(
                eta=eta,
                value=float(primary_eval(eta)),
                feasible=feasible,
                violations=violations,
            )
        )

    for eta in starts:
        add_candidate(eta)
    for eta in optimized:
        add_candidate(eta)

    return out


def optimize_multistart(
    primary: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
) -> OptimizeResult:
    """Return the best feasible result across all starts."""
    candidates = optimize_multistart_candidates(
        primary,
        starts,
        cfg,
        constraints=constraints,
        canonicalize=canonicalize,
    )
    feasible = [r for r in candidates if r.feasible]
    if constraints and not feasible:
        worst = min(candidates, key=lambda r: max(r.violations, default=0.0))
        raise RuntimeError(
            "Gradient optimizer found no feasible candidate; "
            f"best constraint violations={worst.violations}."
        )
    return min(feasible or candidates, key=lambda r: r.value)


# Compatibility name used by one intermediate cleanup revision.
# Keep this alias so old orchestration scripts do not break.
optimize_all_starts = optimize_multistart_candidates


def lexicographic_optimize(
    population_loss: Objective,
    finite_risk: Objective,
    action: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    epsilon_l: float = 0.0,
    epsilon_r: float = 0.0,
    geometry_constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
) -> dict[str, OptimizeResult | float]:
    population = optimize_multistart(
        population_loss,
        starts,
        cfg,
        constraints=geometry_constraints,
        canonicalize=canonicalize,
    )
    l_max = population.value + float(epsilon_l)

    law_constraints = tuple(geometry_constraints) + ((population_loss, l_max),)
    law_starts = jnp.concatenate([population.eta[None, :], starts], axis=0)
    law = optimize_multistart(
        finite_risk,
        law_starts,
        cfg,
        constraints=law_constraints,
        canonicalize=canonicalize,
    )
    r_max = law.value + float(epsilon_r)

    conditioned_constraints = law_constraints + ((finite_risk, r_max),)
    conditioned_starts = jnp.concatenate(
        [law.eta[None, :], population.eta[None, :], starts], axis=0
    )
    conditioned = optimize_multistart(
        action,
        conditioned_starts,
        cfg,
        constraints=conditioned_constraints,
        canonicalize=canonicalize,
    )

    return {
        "population_optimum": population,
        "law": law,
        "conditioned": conditioned,
        "L_max": l_max,
        "R_max": r_max,
    }


def point_box_violation(
    *,
    n_sensors: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> Objective:
    """Return ``max(bound violation)`` for flat 2-D point-sensor designs."""
    n_sensors = int(n_sensors)
    x0, x1 = map(float, x_bounds)
    y0, y1 = map(float, y_bounds)
    if n_sensors < 1:
        raise ValueError("n_sensors must be >= 1")
    if not x1 > x0 or not y1 > y0:
        raise ValueError("invalid point-sensor bounds")

    def violation(eta: Array) -> Array:
        centers = jnp.asarray(eta, dtype=jnp.float64).reshape((n_sensors, 2))
        vx = jnp.maximum(x0 - centers[:, 0], centers[:, 0] - x1)
        vy = jnp.maximum(y0 - centers[:, 1], centers[:, 1] - y1)
        return jnp.maximum(jnp.max(vx), jnp.max(vy))

    return violation


def point_separation_violation(min_sep: float, *, n_sensors: int) -> Objective:
    """Return ``min_sep - min_{i<j} ||s_i-s_j||`` for point sensors."""
    min_sep = float(min_sep)
    n_sensors = int(n_sensors)
    if min_sep < 0.0:
        raise ValueError("min_sep must be nonnegative")
    if n_sensors < 2:
        return lambda eta: jnp.asarray(-jnp.inf, dtype=jnp.float64)

    def violation(eta: Array) -> Array:
        centers = jnp.asarray(eta, dtype=jnp.float64).reshape((n_sensors, 2))
        diff = centers[:, None, :] - centers[None, :, :]
        dist = jnp.sqrt(jnp.maximum(jnp.sum(diff * diff, axis=-1), 1.0e-300))
        diag = jnp.eye(n_sensors, dtype=bool)
        dist = jnp.where(diag, jnp.inf, dist)
        return min_sep - jnp.min(dist)

    return violation


def random_point_sensor_starts(
    key: Array,
    count: int,
    *,
    n_sensors: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    min_sep: float = 0.0,
    oversample: int = 64,
) -> Array:
    """Generate broad CRN-friendly starts for free 2-D point sensors.

    The return shape is ``[count, 2*n_sensors]``.  No known or historical optimum
    is encoded in the generator.
    """
    count = int(count)
    n_sensors = int(n_sensors)
    if count < 1 or n_sensors < 1:
        raise ValueError("count and n_sensors must be >= 1")
    x0, x1 = map(float, x_bounds)
    y0, y1 = map(float, y_bounds)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("invalid point-sensor bounds")

    total = max(count * int(oversample), count)
    raw = jax.random.uniform(key, (total, n_sensors, 2), dtype=jnp.float64)
    x = x0 + (x1 - x0) * raw[..., 0]
    y = y0 + (y1 - y0) * raw[..., 1]
    candidates = jnp.stack([x, y], axis=-1)

    if n_sensors >= 2 and float(min_sep) > 0.0:
        diff = candidates[:, :, None, :] - candidates[:, None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        diag = jnp.eye(n_sensors, dtype=bool)[None, :, :]
        d2 = jnp.where(diag, jnp.inf, d2)
        valid = jnp.sqrt(jnp.min(d2, axis=(1, 2))) >= float(min_sep)
    else:
        valid = jnp.ones((total,), dtype=bool)

    order = jnp.argsort(~valid)
    if int(jnp.sum(valid)) < count:
        raise ValueError(
            "Could not generate enough separated point-sensor starts; "
            "increase oversample or relax min_sep"
        )
    return candidates[order[:count]].reshape((count, 2 * n_sensors))
