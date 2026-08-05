"""Reusable differentiable periodic stochastic simulators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .energies import EnergyFamily, EnergyParameters, total_energy
from .geometry import wrap_positions


@dataclass(frozen=True)
class LangevinConfig:
    """Fixed-step overdamped Langevin integration options."""

    num_steps: int = 1000
    time_step: float = 2.5e-4
    temperature: float = 0.02
    max_drift_norm: float | None = 0.05
    record_every: int = 0


def random_uniform_ensemble(
    key: Array,
    num_replicas: int,
    num_particles: int,
    box: Array,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Sample independent uniform configurations in the periodic box."""
    box = jnp.asarray(box, dtype=dtype)
    return jax.random.uniform(key, (num_replicas, num_particles, 2), dtype=dtype) * box


def _clip_vectors(vectors: Array, max_norm: float | None) -> Array:
    if max_norm is None:
        return vectors
    norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    factor = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))
    return vectors * factor


@partial(jax.jit, static_argnames=("family", "num_steps", "record_every", "max_drift_norm"))
def _simulate_kernel(
    key: Array,
    initial_coordinates: Array,
    box: Array,
    params: EnergyParameters,
    *,
    family: EnergyFamily,
    num_steps: int,
    time_step: float,
    temperature: float,
    max_drift_norm: float | None,
    record_every: int,
) -> tuple[Array, Array]:
    coordinates = wrap_positions(initial_coordinates, box)
    noise_scale = jnp.sqrt(2.0 * temperature * time_step)

    def step(carry: tuple[Array, Array], _: Array) -> tuple[tuple[Array, Array], Array]:
        coordinates, step_key = carry
        step_key, noise_key = jax.random.split(step_key)
        energy_for_coordinates = lambda value: total_energy(value, box, params, family)
        gradient = jax.grad(energy_for_coordinates)(coordinates)
        drift = _clip_vectors(-time_step * gradient, max_drift_norm)
        noise = noise_scale * jax.random.normal(noise_key, coordinates.shape, dtype=coordinates.dtype)
        updated = wrap_positions(coordinates + drift + noise, box)
        return (updated, step_key), updated

    (final_coordinates, _), trajectory = jax.lax.scan(
        step,
        (coordinates, key),
        xs=jnp.arange(num_steps),
    )
    if record_every > 0:
        trajectory = trajectory[record_every - 1 :: record_every]
    else:
        trajectory = trajectory[-1:]
    return final_coordinates, trajectory


def simulate_overdamped_langevin(
    key: Array,
    initial_coordinates: Array,
    box: Array,
    params: EnergyParameters,
    config: LangevinConfig,
    family: EnergyFamily,
) -> tuple[Array, Array]:
    """Run fixed-step overdamped Langevin dynamics.

    Returns the final ensemble and either the final state only or a thinned
    trajectory, depending on ``record_every``.
    """
    if config.num_steps < 1:
        raise ValueError("num_steps must be positive")
    if config.time_step <= 0:
        raise ValueError("time_step must be positive")
    if config.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if config.record_every < 0:
        raise ValueError("record_every cannot be negative")
    return _simulate_kernel(
        key,
        jnp.asarray(initial_coordinates),
        jnp.asarray(box, dtype=initial_coordinates.dtype),
        params,
        family=family,
        num_steps=config.num_steps,
        time_step=config.time_step,
        temperature=config.temperature,
        max_drift_norm=config.max_drift_norm,
        record_every=config.record_every,
    )
