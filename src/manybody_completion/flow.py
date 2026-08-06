"""Torus-aware conditional flow matching and differentiable ODE sampling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .geometry import gauge_fixed_displacement, wrap_positions
from .network import FlowNetworkConfig, Parameters, flow_velocity


@dataclass(frozen=True)
class SamplingOptions:
    num_steps: int = 8
    method: str = "heun"

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'")


@lru_cache(maxsize=8)
def _permutation_table(num_particles: int) -> np.ndarray:
    if num_particles > 8:
        raise ValueError("exhaustive matching supports at most eight particles")
    return np.asarray(tuple(permutations(range(num_particles))), dtype=np.int32)


def exhaustive_match_targets(source: Array, target: Array, box: Array) -> Array:
    """Minimum-cost exchangeable source-target coupling for small systems."""
    table = jnp.asarray(_permutation_table(source.shape[-2]))

    def match_replica(replica_source: Array, replica_target: Array) -> Array:
        candidates = replica_target[table]
        displacement = jax.vmap(
            lambda candidate: gauge_fixed_displacement(
                replica_source, candidate, box
            )
        )(candidates)
        normalized = displacement / box
        cost = jnp.mean(normalized * normalized, axis=(-2, -1))
        return candidates[jnp.argmin(cost)]

    return jax.vmap(jax.vmap(match_replica))(source, target)


def sample_uniform_torus(key: Array, shape: tuple[int, ...], box: Array, dtype: jnp.dtype) -> Array:
    """Sample exchangeable coordinates from the uniform torus prior."""
    return jax.random.uniform(key, shape, dtype=dtype) * jnp.asarray(box, dtype=dtype)


def flow_matching_loss(
    parameters: Parameters,
    target: Array,
    condition: Array,
    key: Array,
    box: Array,
    network_config: FlowNetworkConfig,
) -> tuple[Array, dict[str, Array]]:
    """Simulation-free conditional flow-matching regression loss."""
    source_key, time_key = jax.random.split(key)
    source = sample_uniform_torus(source_key, target.shape, box, target.dtype)
    matched = exhaustive_match_targets(source, target, box)
    displacement = gauge_fixed_displacement(source, matched, box)
    time = jax.random.uniform(time_key, (target.shape[0],), dtype=target.dtype)
    path = wrap_positions(source + time[:, None, None, None] * displacement, box)
    predicted = flow_velocity(
        parameters, path, time, condition, box, network_config
    )
    normalized_error = (predicted - displacement) / box
    loss = jnp.mean(normalized_error * normalized_error)
    return loss, {
        "flow_loss": loss,
        "target_velocity_rms": jnp.sqrt(
            jnp.mean(jnp.sum((displacement / box) ** 2, axis=-1))
        ),
        "predicted_velocity_rms": jnp.sqrt(
            jnp.mean(jnp.sum((predicted / box) ** 2, axis=-1))
        ),
    }


def sample_conditional_flow(
    parameters: Parameters,
    source: Array,
    condition: Array,
    box: Array,
    network_config: FlowNetworkConfig,
    options: SamplingOptions,
) -> Array:
    """Integrate the learned wrapped ODE from ``t=0`` to ``t=1``."""
    options.validate()
    step_size = jnp.asarray(1.0 / options.num_steps, dtype=source.dtype)

    def step(coordinates: Array, index: Array) -> tuple[Array, None]:
        time = jnp.full((coordinates.shape[0],), index * step_size, source.dtype)
        first = flow_velocity(
            parameters, coordinates, time, condition, box, network_config
        )
        if options.method == "euler":
            updated = coordinates + step_size * first
        else:
            proposal = wrap_positions(coordinates + step_size * first, box)
            second = flow_velocity(
                parameters,
                proposal,
                time + step_size,
                condition,
                box,
                network_config,
            )
            updated = coordinates + 0.5 * step_size * (first + second)
        return wrap_positions(updated, box), None

    result, _ = jax.lax.scan(
        step,
        wrap_positions(source, box),
        jnp.arange(options.num_steps, dtype=source.dtype),
    )
    return result
