"""Periodic endpoint-trained stochastic interpolant for defect states.

This is a narrow geometry adapter around the vortices reference-flow pattern:
the same ``FlowMatchingConfig``, Adam schedule, MLP checkpoint format, and
``velocity/rollout`` public API are retained.  State features are sine/cosine
coordinates, endpoint bridges follow shortest periodic arcs, and every RK4
substep is wrapped back to the torus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from mfsi.flow_matching import FlowMatchingConfig, cosine_learning_rate, init_mlp
from mfsi.reference import Params, load_npz_checkpoint, save_npz_checkpoint, time_features

Array = jax.Array


def periodic_model_features(t: Array, x: Array, periods: Array) -> Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    periods = jnp.asarray(periods, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * x / periods
    t = jnp.asarray(t, dtype=jnp.float64)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, x.shape[:-1])
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase), time_features(t)], axis=-1)


def periodic_velocity_mlp(params: Params, t: Array, x: Array, periods: Array) -> Array:
    h = periodic_model_features(t, x, periods)
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    return h @ params[-1]["W"] + params[-1]["b"]


def periodic_delta(x1: Array, x0: Array, periods: Array) -> Array:
    return jnp.mod(x1 - x0 + 0.5 * periods, periods) - 0.5 * periods


@dataclass(frozen=True)
class PeriodicReferenceFlow:
    params: Params
    periods: Array
    substeps_per_interval: int = 16
    metadata: Mapping[str, Any] | None = None

    def velocity(self, x: Array, t: Array) -> Array:
        return periodic_velocity_mlp(self.params, t, x, self.periods)

    def rollout(self, x0: Array, times: Array) -> Array:
        x0 = jnp.mod(jnp.asarray(x0, dtype=jnp.float64), self.periods)
        times = jnp.asarray(times, dtype=jnp.float64)

        def interval(state: Array, pair: tuple[Array, Array]):
            t0, t1 = pair
            dt = (t1 - t0) / float(self.substeps_per_interval)

            def substep(index: int, x: Array) -> Array:
                t = t0 + index.astype(jnp.float64) * dt
                k1 = self.velocity(x, t)
                k2 = self.velocity(jnp.mod(x + 0.5 * dt * k1, self.periods), t + 0.5 * dt)
                k3 = self.velocity(jnp.mod(x + 0.5 * dt * k2, self.periods), t + 0.5 * dt)
                k4 = self.velocity(jnp.mod(x + dt * k3, self.periods), t + dt)
                return jnp.mod(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), self.periods)

            next_state = jax.lax.fori_loop(0, int(self.substeps_per_interval), substep, state)
            return next_state, next_state

        _, nodes = jax.lax.scan(interval, x0, (times[:-1], times[1:]))
        return jnp.concatenate([x0[None], nodes], axis=0)

    def save(self, path: str | Path) -> None:
        metadata = dict(self.metadata or {})
        metadata["periodic_geometry"] = {"periods": [float(v) for v in self.periods]}
        save_npz_checkpoint(path, self.params, metadata)

    @classmethod
    def from_npz(cls, path: str | Path, *, substeps_per_interval: int = 16):
        params, metadata = load_npz_checkpoint(path)
        geometry = metadata.get("periodic_geometry", {})
        if "periods" not in geometry:
            raise ValueError("checkpoint does not declare periodic state geometry")
        return cls(params, jnp.asarray(geometry["periods"]), substeps_per_interval, metadata)


class _AdamState(NamedTuple):
    m: Any
    v: Any
    step: Array


def train_periodic_reference_flow(
    source,
    cfg: FlowMatchingConfig,
    *,
    periods: Array,
    substeps_per_interval: int,
) -> tuple[PeriodicReferenceFlow, list[dict[str, float]]]:
    """Train from endpoint population samples only using shortest-arc bridges."""
    periods = jnp.asarray(periods, dtype=jnp.float64)
    state_dim = int(periods.shape[0])
    if state_dim not in (2, 3) or bool(jnp.any(periods <= 0.0)):
        raise ValueError("periods must contain two or three positive values")
    if cfg.bridge_schedule != "linear":
        raise ValueError("the periodic shortest-arc adapter currently requires bridge_schedule='linear'")
    key = jax.random.PRNGKey(cfg.seed)
    init_key, key = jax.random.split(key)
    params = init_mlp(
        init_key,
        input_dim=2 * state_dim + 5,
        hidden_width=cfg.hidden_width,
        hidden_layers=cfg.hidden_layers,
        output_dim=state_dim,
    )
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = _AdamState(zeros, zeros, jnp.asarray(0, dtype=jnp.int32))

    def batch(batch_key: Array):
        kt, k0, k1, kz = jax.random.split(batch_key, 4)
        t = jax.random.uniform(kt, (cfg.batch_size,), dtype=jnp.float64)
        x0 = source.sample(k0, cfg.batch_size, endpoint=0)
        x1 = source.sample(k1, cfg.batch_size, endpoint=1)
        z = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        # The periodic bridge starts at x0 and follows the selected shortest lift.
        displacement = periodic_delta(x1, x0, periods)
        gamma = float(cfg.bridge_noise_std) * jnp.sin(jnp.pi * t)
        gamma_dot = float(cfg.bridge_noise_std) * jnp.pi * jnp.cos(jnp.pi * t)
        while gamma.ndim < x0.ndim:
            gamma = gamma[..., None]
            gamma_dot = gamma_dot[..., None]
        xt = jnp.mod(x0 + t[:, None] * displacement + gamma * z, periods)
        target = displacement + gamma_dot * z
        return t, xt, target

    def loss_fn(network, t, x, target):
        error = periodic_velocity_mlp(network, t, x, periods) - target
        return jnp.mean(jnp.sum(error**2, axis=-1))

    @jax.jit
    def step(network, adam, step_key):
        t, x, target = batch(step_key)
        loss, grads = jax.value_and_grad(loss_fn)(network, t, x, target)
        norm = jnp.sqrt(sum(jnp.sum(value**2) for value in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, cfg.grad_clip_norm / jnp.maximum(norm, 1.0e-30))
        grads = jax.tree_util.tree_map(lambda value: scale * value, grads)
        count = adam.step + 1
        m = jax.tree_util.tree_map(lambda old, grad: cfg.adam_beta1 * old + (1.0 - cfg.adam_beta1) * grad, adam.m, grads)
        v = jax.tree_util.tree_map(lambda old, grad: cfg.adam_beta2 * old + (1.0 - cfg.adam_beta2) * grad**2, adam.v, grads)
        mhat = jax.tree_util.tree_map(lambda value: value / (1.0 - cfg.adam_beta1**count), m)
        vhat = jax.tree_util.tree_map(lambda value: value / (1.0 - cfg.adam_beta2**count), v)
        learning_rate = cosine_learning_rate(count, cfg.train_steps, cfg.learning_rate, cfg.min_learning_rate_ratio)
        network = jax.tree_util.tree_map(
            lambda value, first, second: value - learning_rate * first / (jnp.sqrt(second) + cfg.adam_eps),
            network, mhat, vhat,
        )
        return network, _AdamState(m, v, count), loss, norm, learning_rate

    history = []
    for iteration in range(1, cfg.train_steps + 1):
        key, step_key = jax.random.split(key)
        params, state, loss, grad_norm, learning_rate = step(params, state, step_key)
        if iteration == 1 or iteration % cfg.log_every == 0 or iteration == cfg.train_steps:
            history.append({
                "step": iteration,
                "conditional_fm_loss": float(loss),
                "grad_norm_preclip": float(grad_norm),
                "learning_rate": float(learning_rate),
            })
    metadata = {
        "network": {"hidden_width": cfg.hidden_width, "hidden_layers": cfg.hidden_layers},
        "bridge": {"schedule": cfg.bridge_schedule, "noise_std": cfg.bridge_noise_std, "shortest_periodic_arc": True},
        "training": {"seed": cfg.seed, "steps": cfg.train_steps, "batch_size": cfg.batch_size, "endpoint_only": True},
        "periodic_geometry": {"periods": [float(value) for value in periods]},
    }
    return PeriodicReferenceFlow(params, periods, substeps_per_interval, metadata), history
