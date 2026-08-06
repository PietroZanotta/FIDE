"""Differentiable local solver backends used by the scientific runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array

from .energy import PhysicalParameters, mean_repulsive_energy
from .geometry import periodic_rms_displacement, wrap_positions
from .observables import PairBasis, ensemble_pair_moments


@dataclass(frozen=True)
class RelaxationOptions:
    num_steps: int = 6
    step_size: float = 0.08
    prox_strength: float = 0.15
    max_particle_step: float = 0.05
    tolerance: float = 5e-3


@dataclass(frozen=True)
class ProjectionOptions:
    num_steps: int = 5
    ridge: float = 1e-6
    max_particle_step: float = 0.08
    tolerance: float = 2e-4
    rank_tolerance: float = 1e-7


class SolverBackend(Protocol):
    """Coarse solver contract shared by local and future Tesseract backends."""

    def relax(self, coordinates: Array) -> tuple[Array, dict[str, Array]]: ...

    def project(
        self, coordinates: Array, target_moments: Array
    ) -> tuple[Array, dict[str, Array]]: ...


def _smooth_periodic_difference(left: Array, right: Array, box: Array) -> Array:
    delta = left - right
    return (box / jnp.pi) * jnp.sin(jnp.pi * delta / box)


def relax_ensemble(
    coordinates: Array,
    box: Array,
    physical: PhysicalParameters,
    options: RelaxationOptions,
) -> tuple[Array, dict[str, Array]]:
    """Fixed-step proximal physical relaxation for one ensemble ``(M,N,2)``."""
    initial = wrap_positions(coordinates, box)

    def objective(values: Array) -> Array:
        proximity = _smooth_periodic_difference(values, initial, box)
        proximity_penalty = 0.5 * options.prox_strength * jnp.mean(
            proximity * proximity
        )
        return mean_repulsive_energy(values, box, physical) + proximity_penalty

    gradient = jax.grad(objective)

    def step(_: int, values: Array) -> Array:
        raw = -options.step_size * gradient(values)
        norm = jnp.sqrt(jnp.sum(raw * raw, axis=-1, keepdims=True) + 1e-18)
        scale = jnp.minimum(1.0, options.max_particle_step / norm)
        return wrap_positions(values + raw * scale, box)

    relaxed = jax.lax.fori_loop(0, options.num_steps, step, initial)
    final_gradient = gradient(relaxed)
    gradient_rms = jnp.sqrt(jnp.mean(final_gradient * final_gradient))
    return relaxed, {
        "energy_before": mean_repulsive_energy(initial, box, physical),
        "energy_after": mean_repulsive_energy(relaxed, box, physical),
        "correction_rms": periodic_rms_displacement(relaxed, initial, box),
        "stationarity_rms": gradient_rms,
        "converged": gradient_rms <= options.tolerance,
    }


def project_ensemble(
    coordinates: Array,
    target_moments: Array,
    box: Array,
    basis: PairBasis,
    moment_scales: Array,
    options: ProjectionOptions,
) -> tuple[Array, dict[str, Array]]:
    """Ridge-regularized ensemble moment projection for one ensemble."""
    initial = wrap_positions(coordinates, box)
    shape = initial.shape
    scales = jnp.maximum(jnp.asarray(moment_scales, initial.dtype), 1e-12)
    target = jnp.asarray(target_moments, initial.dtype)

    def moments(flattened: Array) -> Array:
        return ensemble_pair_moments(flattened.reshape(shape), box, basis)

    def residual(flattened: Array) -> Array:
        return (moments(flattened) - target) / scales

    def step(_: int, flattened: Array) -> Array:
        value = residual(flattened)
        jacobian = jax.jacrev(residual)(flattened)
        gram = jacobian @ jacobian.T + options.ridge * jnp.eye(
            value.shape[0], dtype=flattened.dtype
        )
        correction = -jacobian.T @ jnp.linalg.solve(gram, value)
        correction = correction.reshape(shape)
        norm = jnp.sqrt(jnp.sum(correction * correction, axis=-1, keepdims=True) + 1e-18)
        correction = correction * jnp.minimum(1.0, options.max_particle_step / norm)
        return wrap_positions(flattened.reshape(shape) + correction, box).reshape(-1)

    projected_flat = jax.lax.fori_loop(0, options.num_steps, step, initial.reshape(-1))
    projected = projected_flat.reshape(shape)
    final_residual = residual(projected_flat)
    final_jacobian = jax.jacrev(residual)(projected_flat)
    singular_values = jnp.linalg.svd(final_jacobian, compute_uv=False)
    effective_rank = jnp.sum(singular_values > options.rank_tolerance)
    return projected, {
        "moments_before": ensemble_pair_moments(initial, box, basis),
        "moments_after": ensemble_pair_moments(projected, box, basis),
        "constraint_residual": jnp.linalg.norm(final_residual),
        "correction_rms": periodic_rms_displacement(projected, initial, box),
        "singular_values": singular_values,
        "effective_rank": effective_rank,
        "rank_deficient": effective_rank < target.shape[0],
        "converged": jnp.linalg.norm(final_residual) <= options.tolerance,
    }


@dataclass(frozen=True)
class LocalJaxBackend:
    """Bound local-JAX implementation of both scientific solver interfaces."""

    box: Array
    basis: PairBasis
    moment_scales: Array
    physical: PhysicalParameters
    relaxation_options: RelaxationOptions
    projection_options: ProjectionOptions

    def relax(self, coordinates: Array) -> tuple[Array, dict[str, Array]]:
        return relax_ensemble(
            coordinates,
            self.box,
            self.physical,
            self.relaxation_options,
        )

    def project(
        self, coordinates: Array, target_moments: Array
    ) -> tuple[Array, dict[str, Array]]:
        return project_ensemble(
            coordinates,
            target_moments,
            self.box,
            self.basis,
            self.moment_scales,
            self.projection_options,
        )


def batch_relax(
    backend: LocalJaxBackend, coordinates: Array
) -> tuple[Array, dict[str, Array]]:
    """Vectorize relaxation over independent ensembles."""
    return jax.vmap(backend.relax)(coordinates)


def batch_project(
    backend: LocalJaxBackend,
    coordinates: Array,
    target_moments: Array,
) -> tuple[Array, dict[str, Array]]:
    """Vectorize projection over independent ensembles."""
    return jax.vmap(backend.project)(coordinates, target_moments)
