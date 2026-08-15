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


def random_projective_starts(
    key: Array,
    count: int,
    *,
    min_sep_rad: float,
) -> Array:
    """Draw two-angle starts on [0, pi) with the requested projective separation."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if not 0.0 <= min_sep_rad < 0.5 * jnp.pi:
        raise ValueError("min_sep_rad must lie in [0, pi/2)")
    k0, kd = jax.random.split(key)
    first = jax.random.uniform(k0, (count,), minval=0.0, maxval=jnp.pi, dtype=jnp.float64)
    delta = jax.random.uniform(
        kd,
        (count,),
        minval=min_sep_rad,
        maxval=jnp.pi - min_sep_rad,
        dtype=jnp.float64,
    )
    second = jnp.mod(first + delta, jnp.pi)
    return jnp.sort(jnp.stack([first, second], axis=-1), axis=-1)


def projective_separation_violation(min_sep_rad: float) -> Objective:
    """Return g(eta) <= 0 for the two-sensor projective separation constraint."""
    min_sep_rad = float(min_sep_rad)

    def violation(eta: Array) -> Array:
        eta = jnp.mod(eta, jnp.pi)
        d = jnp.abs(eta[0] - eta[1])
        sep = jnp.minimum(d, jnp.pi - d)
        return min_sep_rad - sep

    return violation


def _penalized(primary: Objective, constraints: Sequence[Constraint], penalty: float) -> Objective:
    def objective(eta: Array) -> Array:
        loss = primary(eta)
        for fn, upper in constraints:
            violation = jax.nn.relu(fn(eta) - upper)
            loss = loss + penalty * violation * violation
        return loss

    return objective


def _adam(objective: Objective, eta0: Array, cfg: OptimizerConfig) -> Array:
    """Tiny dependency-free Adam loop using JAX autodiff."""
    eta = jnp.asarray(eta0, dtype=jnp.float64)
    m = jnp.zeros_like(eta)
    v = jnp.zeros_like(eta)
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    for step in range(1, cfg.steps + 1):
        _, grad = value_and_grad(eta)
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grad * grad)
        mhat = m / (1.0 - cfg.beta1**step)
        vhat = v / (1.0 - cfg.beta2**step)
        eta = eta - cfg.learning_rate * mhat / (jnp.sqrt(vhat) + cfg.eps)

    return eta




def optimize_all_starts(
    primary: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
) -> list[OptimizeResult]:
    """Run every gradient start and return all terminal candidates."""
    objective = _penalized(primary, constraints, cfg.constraint_penalty)
    out: list[OptimizeResult] = []
    for eta0 in jnp.asarray(starts, dtype=jnp.float64):
        seed = canonicalize(eta0) if canonicalize is not None else eta0
        seed_violations = tuple(float(fn(seed) - upper) for fn, upper in constraints)
        out.append(OptimizeResult(
            eta=seed, value=float(primary(seed)),
            feasible=all(v <= cfg.feasibility_tol for v in seed_violations),
            violations=seed_violations,
        ))

        eta = _adam(objective, eta0, cfg)
        if canonicalize is not None:
            eta = canonicalize(eta)
        violations = tuple(float(fn(eta) - upper) for fn, upper in constraints)
        feasible = all(v <= cfg.feasibility_tol for v in violations)
        out.append(OptimizeResult(eta=eta, value=float(primary(eta)), feasible=feasible, violations=violations))
    return out

def optimize_multistart(
    primary: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
) -> OptimizeResult:
    objective = _penalized(primary, constraints, cfg.constraint_penalty)
    candidates: list[OptimizeResult] = []

    for eta0 in jnp.asarray(starts, dtype=jnp.float64):
        seed = canonicalize(eta0) if canonicalize is not None else eta0
        seed_violations = tuple(float(fn(seed) - upper) for fn, upper in constraints)
        candidates.append(OptimizeResult(
            eta=seed,
            value=float(primary(seed)),
            feasible=all(v <= cfg.feasibility_tol for v in seed_violations),
            violations=seed_violations,
        ))

        eta = _adam(objective, eta0, cfg)
        if canonicalize is not None:
            eta = canonicalize(eta)

        violations = tuple(float(fn(eta) - upper) for fn, upper in constraints)
        feasible = all(v <= cfg.feasibility_tol for v in violations)
        candidates.append(
            OptimizeResult(
                eta=eta,
                value=float(primary(eta)),
                feasible=feasible,
                violations=violations,
            )
        )

    feasible = [r for r in candidates if r.feasible]
    if constraints and not feasible:
        worst = min(candidates, key=lambda r: max(r.violations, default=0.0))
        raise RuntimeError(
            "Gradient optimizer found no feasible candidate; "
            f"best constraint violations={worst.violations}. "
            "Add starts or adjust optimizer settings in config."
        )
    return min(feasible or candidates, key=lambda r: r.value)


def lexicographic_optimize(
    population_loss: Objective,
    finite_risk: Objective,
    action: Objective,
    starts: Array,
    cfg: OptimizerConfig,
    *,
    tau_l: float = 0.05,
    tau_r: float = 0.01,
    geometry_constraints: Sequence[Constraint] = (),
    canonicalize: Callable[[Array], Array] | None = None,
) -> dict[str, OptimizeResult | float]:
    """Law-first lexicographic design with gradient-based eta optimization.

    1. Find the population-law optimum L*.
    2. Minimize finite-resource risk under L <= (1+tau_l)L*.
    3. Minimize the requested action under both law screens.

    Penalties guide the gradient solve; returned candidates must pass explicit hard
    feasibility checks. Multiple starts replace a dense eta grid.
    """
    population = optimize_multistart(
        population_loss,
        starts,
        cfg,
        constraints=geometry_constraints,
        canonicalize=canonicalize,
    )
    l_max = (1.0 + tau_l) * population.value

    law_constraints = tuple(geometry_constraints) + ((population_loss, l_max),)
    law = optimize_multistart(
        finite_risk,
        starts,
        cfg,
        constraints=law_constraints,
        canonicalize=canonicalize,
    )
    r_max = (1.0 + tau_r) * law.value

    conditioned_constraints = law_constraints + ((finite_risk, r_max),)
    conditioned = optimize_multistart(
        action,
        starts,
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
