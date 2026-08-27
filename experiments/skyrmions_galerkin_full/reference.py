from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .domain import minimum_image

Array = jax.Array
Layers = tuple[dict[str, Array], ...]
ReferenceParams = dict[str, Layers]


def _init_layers(key: Array, dims: list[int]) -> Layers:
    keys = jax.random.split(key, len(dims) - 1)
    return tuple(
        {
            "W": jax.random.normal(k, (din, dout), dtype=jnp.float64)
            * jnp.sqrt(2.0 / float(din + dout)),
            "b": jnp.zeros((dout,), dtype=jnp.float64),
        }
        for k, din, dout in zip(keys, dims[:-1], dims[1:])
    )


def _mlp(layers: Layers, values: Array, *, final_activation: bool = False) -> Array:
    h = values
    for layer in layers[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    h = h @ layers[-1]["W"] + layers[-1]["b"]
    return jax.nn.silu(h) if final_activation else h


def time_features(t: Array) -> Array:
    t = jnp.asarray(t, dtype=jnp.float64)
    return jnp.stack(
        [t, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t), jnp.sin(2 * jnp.pi * t), jnp.cos(2 * jnp.pi * t)],
        axis=-1,
    )


def periodic_position_features(x: Array, box: tuple[float, float]) -> Array:
    phase = 2.0 * jnp.pi * x / jnp.asarray(box, dtype=x.dtype)
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


def init_equivariant_reference(
    key: Array, *, hidden_width: int, hidden_layers: int
) -> ReferenceParams:
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    ke, ko = jax.random.split(key)
    embed_dims = [9] + [int(hidden_width)] * int(hidden_layers)
    output_dims = [2 * int(hidden_width) + 5, int(hidden_width), 2]
    return {"embed": _init_layers(ke, embed_dims), "output": _init_layers(ko, output_dims)}


def equivariant_velocity(
    params: ReferenceParams,
    t: Array,
    configurations: Array,
    *,
    box: tuple[float, float],
) -> Array:
    """Permutation-equivariant continuous-time velocity field."""

    x = jnp.asarray(configurations, dtype=jnp.float64)
    tf = time_features(t)
    while tf.ndim < x.ndim:
        tf = tf[..., None, :]
    tf = jnp.broadcast_to(tf, x.shape[:-1] + (5,))
    local = jnp.concatenate([periodic_position_features(x, box), tf], axis=-1)
    embedded = _mlp(params["embed"], local, final_activation=True)
    pooled = jnp.mean(embedded, axis=-2, keepdims=True)
    pooled = jnp.broadcast_to(pooled, embedded.shape)
    return _mlp(params["output"], jnp.concatenate([embedded, pooled, tf], axis=-1))


@dataclass(frozen=True)
class ReferenceTrainingConfig:
    seed: int = 20260822
    hidden_width: int = 48
    hidden_layers: int = 2
    train_steps: int = 2500
    batch_size: int = 256
    learning_rate: float = 8.0e-4
    min_learning_rate_ratio: float = 0.08
    grad_clip_norm: float = 8.0
    bridge_noise_std: float = 0.01
    log_every: int = 250


class _AdamState(NamedTuple):
    m: Any
    v: Any
    step: Array


def _tree_norm(tree) -> Array:
    return jnp.sqrt(sum(jnp.sum(v * v) for v in jax.tree_util.tree_leaves(tree)))


def train_endpoint_reference(
    endpoint0: Array,
    endpoint1: Array,
    cfg: ReferenceTrainingConfig,
    *,
    box: tuple[float, float],
) -> tuple["EquivariantReferenceFlow", list[dict[str, float]]]:
    """Train from endpoint ensembles only; intermediate truth is never accepted."""

    endpoint0 = jnp.asarray(endpoint0, dtype=jnp.float64)
    endpoint1 = jnp.asarray(endpoint1, dtype=jnp.float64)
    if endpoint0.ndim != 3 or endpoint0.shape != endpoint1.shape or endpoint0.shape[-1] != 2:
        raise ValueError("endpoint arrays must have matching [sample,particle,2] shapes")
    key = jax.random.PRNGKey(int(cfg.seed))
    key, init_key = jax.random.split(key)
    params = init_equivariant_reference(
        init_key, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers
    )
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = _AdamState(zeros, zeros, jnp.asarray(0, dtype=jnp.int32))

    def sample_batch(batch_key):
        kt, k0, k1, kz = jax.random.split(batch_key, 4)
        batch = int(cfg.batch_size)
        idx0 = jax.random.randint(k0, (batch,), 0, endpoint0.shape[0])
        idx1 = jax.random.randint(k1, (batch,), 0, endpoint1.shape[0])
        x0, x1 = endpoint0[idx0], endpoint1[idx1]
        t = jax.random.uniform(kt, (batch,), dtype=jnp.float64)
        displacement = minimum_image(x1 - x0, jnp.asarray(box))
        noise = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        gamma = float(cfg.bridge_noise_std) * jnp.sin(jnp.pi * t)[:, None, None]
        gamma_dot = float(cfg.bridge_noise_std) * jnp.pi * jnp.cos(jnp.pi * t)[:, None, None]
        xt = jnp.mod(x0 + t[:, None, None] * displacement + gamma * noise, jnp.asarray(box))
        target = displacement + gamma_dot * noise
        return t, xt, target

    def loss_fn(p, t, x, target):
        predicted = equivariant_velocity(p, t, x, box=box)
        return jnp.mean(jnp.sum((predicted - target) ** 2, axis=(-2, -1)))

    @jax.jit
    def step(p, adam, step_key):
        t, x, target = sample_batch(step_key)
        loss, grads = jax.value_and_grad(loss_fn)(p, t, x, target)
        norm = _tree_norm(grads)
        scale = jnp.minimum(1.0, float(cfg.grad_clip_norm) / jnp.maximum(norm, 1.0e-30))
        grads = jax.tree_util.tree_map(lambda g: scale * g, grads)
        count = adam.step + 1
        beta1, beta2 = 0.9, 0.999
        m = jax.tree_util.tree_map(lambda old, g: beta1 * old + (1 - beta1) * g, adam.m, grads)
        v = jax.tree_util.tree_map(lambda old, g: beta2 * old + (1 - beta2) * g * g, adam.v, grads)
        mhat = jax.tree_util.tree_map(lambda z: z / (1 - beta1**count), m)
        vhat = jax.tree_util.tree_map(lambda z: z / (1 - beta2**count), v)
        fraction = jnp.clip(count / max(float(cfg.train_steps), 1.0), 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * fraction))
        lr = float(cfg.learning_rate) * (
            float(cfg.min_learning_rate_ratio) + (1.0 - float(cfg.min_learning_rate_ratio)) * cosine
        )
        p = jax.tree_util.tree_map(lambda q, a, b: q - lr * a / (jnp.sqrt(b) + 1.0e-8), p, mhat, vhat)
        return p, _AdamState(m, v, count), loss, norm, lr

    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for index in range(1, int(cfg.train_steps) + 1):
        key, step_key = jax.random.split(key)
        params, state, loss, norm, lr = step(params, state, step_key)
        if index == 1 or index % int(cfg.log_every) == 0 or index == int(cfg.train_steps):
            history.append({
                "step": index,
                "loss": float(loss),
                "gradient_norm": float(norm),
                "learning_rate": float(lr),
                "elapsed_seconds": time.perf_counter() - started,
            })
    metadata = {
        "kind": "permutation_equivariant_endpoint_cfm_v1",
        "endpoint_only": True,
        "box": list(box),
        "training": asdict(cfg),
    }
    return EquivariantReferenceFlow(params, box=box, metadata=metadata), history


@dataclass(frozen=True)
class EquivariantReferenceFlow:
    params: ReferenceParams
    box: tuple[float, float] = (2.0, 1.0)
    metadata: dict[str, Any] | None = None

    def velocity(self, x: Array, t: Array) -> Array:
        return equivariant_velocity(self.params, t, x, box=self.box)

    def rollout(self, x0: Array, times: Array, *, substeps_per_interval: int = 12) -> Array:
        x0 = jnp.asarray(x0, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        box = jnp.asarray(self.box, dtype=x0.dtype)

        def interval(x, pair):
            t0, t1 = pair
            dt = (t1 - t0) / float(substeps_per_interval)

            def one(i, state):
                t = t0 + i.astype(jnp.float64) * dt
                k1 = self.velocity(state, t)
                k2 = self.velocity(jnp.mod(state + 0.5 * dt * k1, box), t + 0.5 * dt)
                k3 = self.velocity(jnp.mod(state + 0.5 * dt * k2, box), t + 0.5 * dt)
                k4 = self.velocity(jnp.mod(state + dt * k3, box), t + dt)
                return jnp.mod(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, box)

            xn = jax.lax.fori_loop(0, int(substeps_per_interval), one, x)
            return xn, xn

        _, nodes = jax.lax.scan(interval, x0, (times[:-1], times[1:]))
        return jnp.concatenate([x0[None, ...], nodes], axis=0)


def save_reference(path: str | Path, flow: EquivariantReferenceFlow) -> None:
    arrays: dict[str, np.ndarray] = {}
    for group in ("embed", "output"):
        for index, layer in enumerate(flow.params[group]):
            arrays[f"{group}_W{index}"] = np.asarray(layer["W"])
            arrays[f"{group}_b{index}"] = np.asarray(layer["b"])
    arrays["metadata_json"] = np.asarray(json.dumps(flow.metadata or {}, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def load_reference(path: str | Path) -> EquivariantReferenceFlow:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        params: ReferenceParams = {}
        for group in ("embed", "output"):
            indices = sorted(
                int(name.removeprefix(f"{group}_W"))
                for name in data.files
                if name.startswith(f"{group}_W")
            )
            params[group] = tuple(
                {"W": jnp.asarray(data[f"{group}_W{i}"]), "b": jnp.asarray(data[f"{group}_b{i}"])}
                for i in indices
            )
    if metadata.get("kind") != "permutation_equivariant_endpoint_cfm_v1":
        raise RuntimeError("incompatible skyrmion reference checkpoint")
    return EquivariantReferenceFlow(params, box=tuple(metadata["box"]), metadata=metadata)

