"""Differentiable proximal physical relaxation on a smooth periodic torus.

This module is the reusable numerical core for the physical-relaxation
Tesseract.  The baseline derivative strategy is reverse-mode differentiation
through a fixed number of line-searched gradient steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .energies import soft_repulsive_energy_per_configuration
from .geometry import chord_distances, off_diagonal_mask, wrap_positions


@dataclass(frozen=True)
class RelaxationOptions:
    """Options for the unrolled proximal relaxation solver."""

    num_steps: int = 128
    step_size: float = 2.5e-3
    tolerance: float = 1e-7
    max_update_norm: float | None = 0.04
    line_search_steps: int = 12
    line_search_shrink: float = 0.5
    armijo_coefficient: float = 1e-4


def smooth_periodic_displacement(target: Array, reference: Array, box: Array) -> Array:
    """Smooth componentwise displacement on the periodic box.

    This uses the same sinusoidal embedding as the pair geometry.  It is zero
    for periodically equivalent coordinates and smooth at wrapping boundaries.
    """
    target = jnp.asarray(target)
    reference = jnp.asarray(reference, dtype=target.dtype)
    box = jnp.asarray(box, dtype=target.dtype)
    return (box / jnp.pi) * jnp.sin(jnp.pi * (target - reference) / box)


def proximal_objective(
    coordinates: Array,
    initial_coordinates: Array,
    box: Array,
    r0: Array | float,
    kappa: Array | float,
    prox_strength: Array | float,
) -> Array:
    """Size-normalized proximal objective corresponding to Eq. (9)."""
    coordinates = jnp.asarray(coordinates)
    initial_coordinates = jnp.asarray(initial_coordinates, dtype=coordinates.dtype)
    physical = jnp.mean(
        soft_repulsive_energy_per_configuration(coordinates, box, r0, kappa)
    )
    displacement = smooth_periodic_displacement(coordinates, initial_coordinates, box)
    proximity = 0.5 * jnp.mean(jnp.sum(displacement * displacement, axis=-1)) / prox_strength
    return physical + proximity


def _clip_updates(updates: Array, max_norm: float | None) -> Array:
    if max_norm is None:
        return updates
    norms = jnp.linalg.norm(updates, axis=-1, keepdims=True)
    factors = jnp.minimum(1.0, max_norm / jnp.maximum(norms, 1e-15))
    return updates * factors


def _minimum_pair_distance(coordinates: Array, box: Array) -> Array:
    n = coordinates.shape[-2]
    distances = chord_distances(coordinates, box)
    mask = off_diagonal_mask(n, dtype=bool)
    mask = jnp.broadcast_to(mask, distances.shape)
    return jnp.min(jnp.where(mask, distances, jnp.inf))


@partial(
    jax.jit,
    static_argnames=(
        "num_steps",
        "max_update_norm",
        "line_search_steps",
    ),
)
def _relax_kernel(
    initial_coordinates: Array,
    box: Array,
    r0: Array,
    kappa: Array,
    prox_strength: Array,
    *,
    num_steps: int,
    step_size: float,
    tolerance: float,
    max_update_norm: float | None,
    line_search_steps: int,
    line_search_shrink: float,
    armijo_coefficient: float,
) -> tuple[Array, dict[str, Array]]:
    initial_coordinates = wrap_positions(initial_coordinates, box)

    objective_fn = lambda y: proximal_objective(
        y, initial_coordinates, box, r0, kappa, prox_strength
    )
    value_and_grad = jax.value_and_grad(objective_fn)

    initial_objective, initial_gradient = value_and_grad(initial_coordinates)
    initial_stationarity = jnp.max(jnp.linalg.norm(initial_gradient, axis=-1))

    # State: coordinates, objective, gradient, iteration count, line-search failures, converged.
    state = (
        initial_coordinates,
        initial_objective,
        initial_gradient,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        initial_stationarity <= tolerance,
    )

    def solver_step(_, state):
        coordinates, objective, gradient, iterations, failures, converged = state
        gradient_norm_sq = jnp.sum(gradient * gradient)
        raw_update = _clip_updates(-step_size * gradient, max_update_norm)

        # Backtracking is expressed as a fixed loop for JIT and reverse-mode AD.
        ls_state = (
            jnp.asarray(1.0, dtype=coordinates.dtype),
            coordinates,
            objective,
            jnp.asarray(False),
        )

        def line_search_step(_, ls_state):
            scale, best_coordinates, best_objective, accepted = ls_state
            candidate = wrap_positions(coordinates + scale * raw_update, box)
            candidate_objective = objective_fn(candidate)
            # The Armijo term uses the actual clipped update's directional derivative.
            directional = jnp.sum(gradient * (scale * raw_update))
            sufficient_decrease = candidate_objective <= (
                objective + armijo_coefficient * directional
            )
            take = (~accepted) & sufficient_decrease
            best_coordinates = jnp.where(take, candidate, best_coordinates)
            best_objective = jnp.where(take, candidate_objective, best_objective)
            accepted = accepted | sufficient_decrease
            scale = jnp.where(accepted, scale, scale * line_search_shrink)
            return scale, best_coordinates, best_objective, accepted

        scale, candidate_coordinates, candidate_objective, accepted = jax.lax.fori_loop(
            0, line_search_steps, line_search_step, ls_state
        )
        del scale, gradient_norm_sq

        # If every line-search trial fails, retain the previous iterate.
        new_coordinates = jnp.where(accepted, candidate_coordinates, coordinates)
        new_objective, new_gradient = value_and_grad(new_coordinates)
        stationarity = jnp.max(jnp.linalg.norm(new_gradient, axis=-1))
        newly_converged = stationarity <= tolerance

        active = ~converged
        coordinates_out = jnp.where(active, new_coordinates, coordinates)
        objective_out = jnp.where(active, new_objective, objective)
        gradient_out = jnp.where(active, new_gradient, gradient)
        iterations_out = iterations + active.astype(jnp.int32)
        failures_out = failures + (active & (~accepted)).astype(jnp.int32)
        converged_out = converged | (active & newly_converged)
        return (
            coordinates_out,
            objective_out,
            gradient_out,
            iterations_out,
            failures_out,
            converged_out,
        )

    coordinates, final_objective, final_gradient, iterations, failures, converged = (
        jax.lax.fori_loop(0, num_steps, solver_step, state)
    )

    physical_before = jnp.mean(
        soft_repulsive_energy_per_configuration(initial_coordinates, box, r0, kappa)
    )
    physical_after = jnp.mean(
        soft_repulsive_energy_per_configuration(coordinates, box, r0, kappa)
    )
    physical_gradient = jax.grad(
        lambda y: jnp.mean(soft_repulsive_energy_per_configuration(y, box, r0, kappa))
    )(coordinates)
    periodic_displacement = smooth_periodic_displacement(
        coordinates, initial_coordinates, box
    )
    diagnostics = {
        "physical_energy_before": physical_before,
        "physical_energy_after": physical_after,
        "proximal_objective_before": initial_objective,
        "proximal_objective_after": final_objective,
        "max_force": jnp.max(jnp.linalg.norm(physical_gradient, axis=-1)),
        "stationarity_norm": jnp.max(jnp.linalg.norm(final_gradient, axis=-1)),
        "prox_displacement": jnp.sqrt(
            jnp.mean(jnp.sum(periodic_displacement * periodic_displacement, axis=-1))
        ),
        "minimum_pair_distance_before": _minimum_pair_distance(initial_coordinates, box),
        "minimum_pair_distance_after": _minimum_pair_distance(coordinates, box),
        "iterations": iterations,
        "line_search_failures": failures,
        "converged": converged,
    }
    return coordinates, diagnostics


def relax_proximal(
    initial_coordinates: Array,
    box: Array,
    r0: Array | float,
    kappa: Array | float,
    prox_strength: Array | float,
    options: RelaxationOptions | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Apply fixed-iteration differentiable proximal physical relaxation."""
    if options is None:
        options = RelaxationOptions()
    if options.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if options.step_size <= 0:
        raise ValueError("step_size must be positive")
    if options.tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if options.line_search_steps < 1:
        raise ValueError("line_search_steps must be positive")
    if not 0 < options.line_search_shrink < 1:
        raise ValueError("line_search_shrink must be in (0, 1)")
    if not 0 < options.armijo_coefficient < 1:
        raise ValueError("armijo_coefficient must be in (0, 1)")

    initial_coordinates = jnp.asarray(initial_coordinates)
    dtype = initial_coordinates.dtype
    return _relax_kernel(
        initial_coordinates,
        jnp.asarray(box, dtype=dtype),
        jnp.asarray(r0, dtype=dtype),
        jnp.asarray(kappa, dtype=dtype),
        jnp.asarray(prox_strength, dtype=dtype),
        num_steps=options.num_steps,
        step_size=options.step_size,
        tolerance=options.tolerance,
        max_update_norm=options.max_update_norm,
        line_search_steps=options.line_search_steps,
        line_search_shrink=options.line_search_shrink,
        armijo_coefficient=options.armijo_coefficient,
    )
