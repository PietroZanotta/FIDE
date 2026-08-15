from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
Params = tuple[dict[str, Array], ...]


def time_features(t: Array) -> Array:
    """Time embedding used by the endpoint-trained velocity checkpoint."""
    t = jnp.asarray(t, dtype=jnp.float64)
    return jnp.stack(
        [
            t,
            jnp.sin(jnp.pi * t),
            jnp.cos(jnp.pi * t),
            jnp.sin(2.0 * jnp.pi * t),
            jnp.cos(2.0 * jnp.pi * t),
        ],
        axis=-1,
    )


def model_features(t: Array, x: Array) -> Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, x.shape[:-1])
    return jnp.concatenate([x, time_features(t)], axis=-1)


def velocity_mlp(params: Params, t: Array, x: Array) -> Array:
    """Evaluate the frozen velocity MLP."""
    h = model_features(t, x)
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def _layer_indices(keys: Sequence[str]) -> list[int]:
    out = []
    for key in keys:
        match = re.fullmatch(r"W(\d+)", key)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def load_npz_checkpoint(path: str | Path) -> tuple[Params, dict[str, Any]]:
    """Load the existing W0/b0, W1/b1, ... NPZ checkpoint format.

    Historical metadata is retained as opaque metadata; the core does not inspect
    or depend on historical stage names.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata: dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw = np.asarray(data["metadata_json"])
            metadata = json.loads(str(raw.item() if raw.ndim == 0 else raw))

        indices = _layer_indices(data.files)
        if not indices:
            raise ValueError(f"No W0/b0-style network layers found in {path}")
        expected = list(range(indices[-1] + 1))
        if indices != expected:
            raise ValueError(f"Checkpoint layers are not contiguous: {indices}")

        params = tuple(
            {
                "W": jnp.asarray(data[f"W{i}"], dtype=jnp.float64),
                "b": jnp.asarray(data[f"b{i}"], dtype=jnp.float64),
            }
            for i in indices
        )
    return params, metadata


def save_npz_checkpoint(
    path: str | Path,
    params: Params,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save the same simple checkpoint format used by the existing experiments."""
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(params):
        arrays[f"W{i}"] = np.asarray(layer["W"])
        arrays[f"b{i}"] = np.asarray(layer["b"])
    arrays["metadata_json"] = np.asarray(json.dumps(dict(metadata or {}), sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def rk4_rollout(
    params: Params,
    x0: Array,
    times: Array,
    *,
    substeps_per_interval: int,
) -> Array:
    """Roll particles through arbitrary increasing time nodes with fixed-step RK4.

    Returns shape ``[len(times), *x0.shape]`` and includes ``x0`` at ``times[0]``.
    ``substeps_per_interval`` is static and intentionally configured outside the CLI.
    """
    if substeps_per_interval < 1:
        raise ValueError("substeps_per_interval must be >= 1")

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)

    def interval_step(x: Array, pair: tuple[Array, Array]):
        t0, t1 = pair
        dt = (t1 - t0) / float(substeps_per_interval)

        def one_substep(i: int, state: Array) -> Array:
            t = t0 + i.astype(jnp.float64) * dt
            k1 = velocity_mlp(params, t, state)
            k2 = velocity_mlp(params, t + 0.5 * dt, state + 0.5 * dt * k1)
            k3 = velocity_mlp(params, t + 0.5 * dt, state + 0.5 * dt * k2)
            k4 = velocity_mlp(params, t + dt, state + dt * k3)
            return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        xn = jax.lax.fori_loop(0, substeps_per_interval, one_substep, x)
        return xn, xn

    pairs = (times[:-1], times[1:])
    _, nodes = jax.lax.scan(interval_step, x0, pairs)
    return jnp.concatenate([x0[None, ...], nodes], axis=0)


@dataclass(frozen=True)
class MLPReferenceFlow:
    """Frozen endpoint-trained reference flow."""

    params: Params
    substeps_per_interval: int = 16
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        substeps_per_interval: int = 16,
    ) -> "MLPReferenceFlow":
        params, metadata = load_npz_checkpoint(path)
        return cls(
            params=params,
            substeps_per_interval=substeps_per_interval,
            metadata=metadata,
        )

    def velocity(self, x: Array, t: Array) -> Array:
        return velocity_mlp(self.params, t, x)

    def rollout(self, x0: Array, times: Array) -> Array:
        return rk4_rollout(
            self.params,
            x0,
            times,
            substeps_per_interval=self.substeps_per_interval,
        )
