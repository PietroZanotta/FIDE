from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from .domain import minimum_image
from .measurements import LocalDensitySensors

Array = jax.Array
Layers = tuple[dict[str, Array], ...]
RitzParams = dict[str, Layers]


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


def _init_independent_layers(key: Array, dims: list[int], time_nodes: int) -> Layers:
    """Initialize one independent MLP per (uniform) scientific time node."""

    networks = tuple(_init_layers(k, dims) for k in jax.random.split(key, time_nodes))
    return tuple({
        "W": jnp.stack([network[layer]["W"] for network in networks]),
        "b": jnp.stack([network[layer]["b"] for network in networks]),
    } for layer in range(len(dims) - 1))


def _mlp(layers: Layers, x: Array, *, activate_last: bool = False) -> Array:
    h = x
    for layer in layers[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    h = h @ layers[-1]["W"] + layers[-1]["b"]
    return jax.nn.silu(h) if activate_last else h


def init_ritz_params(
    key: Array,
    *,
    hidden_width: int = 40,
    hidden_layers: int = 2,
    independent_time_nodes: int = 0,
) -> RitzParams:
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    ke, kh = jax.random.split(key)
    # Four periodic position coordinates plus five time coordinates.
    initializer = (
        lambda key_, dims: _init_independent_layers(
            key_, dims, int(independent_time_nodes)
        )
        if independent_time_nodes > 0 else _init_layers(key_, dims)
    )
    embed = initializer(ke, [9] + [int(hidden_width)] * int(hidden_layers))
    head = initializer(kh, [int(hidden_width) + 5, int(hidden_width), 1])
    return {"embed": embed, "head": head}


def promote_ritz_params_to_independent(
    params: RitzParams, time_nodes: int
) -> RitzParams:
    """Broadcast a shared solution into independent per-time warm starts."""

    if time_nodes < 1:
        return params
    existing = int(params["embed"][0]["W"].shape[0]) if params["embed"][0]["W"].ndim == 3 else 0
    if existing:
        if existing != time_nodes:
            raise ValueError(
                f"checkpoint has {existing} independent nodes, expected {time_nodes}"
            )
        return params
    return {
        group: tuple({
            name: jnp.broadcast_to(value, (time_nodes,) + value.shape).copy()
            for name, value in layer.items()
        } for layer in params[group])
        for group in ("embed", "head")
    }


def _layers_at_time(layers: Layers, t: Array) -> Layers:
    """Select an independent network at an exact uniform scientific node."""

    if layers[0]["W"].ndim == 2:
        return layers
    time_nodes = layers[0]["W"].shape[0]
    index = jnp.clip(jnp.rint(t * (time_nodes - 1)).astype(jnp.int32), 0, time_nodes - 1)
    return tuple({"W": layer["W"][index], "b": layer["b"][index]} for layer in layers)


def save_ritz_checkpoint(
    path: str | Path,
    params: RitzParams,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for group in ("embed", "head"):
        for index, layer in enumerate(params[group]):
            arrays[f"{group}_W{index}"] = np.asarray(layer["W"])
            arrays[f"{group}_b{index}"] = np.asarray(layer["b"])
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    np.savez_compressed(Path(path), **arrays)


def load_ritz_checkpoint(path: str | Path) -> tuple[RitzParams, dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as data:
        params: RitzParams = {}
        for group in ("embed", "head"):
            indices = sorted(
                int(name.removeprefix(f"{group}_W"))
                for name in data.files if name.startswith(f"{group}_W")
            )
            if not indices:
                raise ValueError(f"missing {group} layers in Deep Ritz checkpoint")
            params[group] = tuple({
                "W": jnp.asarray(data[f"{group}_W{i}"], dtype=jnp.float64),
                "b": jnp.asarray(data[f"{group}_b{i}"], dtype=jnp.float64),
            } for i in indices)
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return params, metadata


def _time_features(t: Array) -> Array:
    return jnp.asarray(
        [t, jnp.sin(jnp.pi * t), jnp.cos(jnp.pi * t), jnp.sin(2 * jnp.pi * t), jnp.cos(2 * jnp.pi * t)]
    )


def invariant_potential(
    params: RitzParams,
    configuration: Array,
    t: Array,
    *,
    box: tuple[float, float] = (2.0, 1.0),
) -> Array:
    """Compact time-conditioned DeepSets scalar potential."""

    x = jnp.asarray(configuration, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * x / jnp.asarray(box, dtype=x.dtype)
    tf = _time_features(t)
    embed_layers = _layers_at_time(params["embed"], t)
    head_layers = _layers_at_time(params["head"], t)
    local_tf = jnp.broadcast_to(tf, x.shape[:-1] + (5,))
    local = jnp.concatenate([jnp.sin(phase), jnp.cos(phase), local_tf], axis=-1)
    embedded = _mlp(embed_layers, local, activate_last=True)
    pooled = jnp.mean(embedded, axis=-2)
    return jnp.squeeze(_mlp(head_layers, jnp.concatenate([pooled, tf], axis=-1)), axis=-1)


def potential_values_and_gradients(
    params: RitzParams,
    configurations: Array,
    times: Array,
    *,
    box: tuple[float, float] = (2.0, 1.0),
) -> tuple[Array, Array]:
    one_value_grad = jax.value_and_grad(lambda x, t: invariant_potential(params, x, t, box=box), argnums=0)
    per_time = jax.vmap(lambda row, t: jax.vmap(lambda x: one_value_grad(x, t))(row))
    values, gradients = per_time(configurations, times)
    return values, gradients


def potential_values(
    params: RitzParams,
    configurations: Array,
    times: Array,
    *,
    box: tuple[float, float] = (2.0, 1.0),
) -> Array:
    per_time = jax.vmap(
        lambda row, t: jax.vmap(
            lambda x: invariant_potential(params, x, t, box=box)
        )(row)
    )
    return per_time(configurations, times)


def ritz_objective(
    params: RitzParams,
    configurations: Array,
    weights: Array,
    forcing: Array,
    times: Array,
    time_weights: Array,
    *,
    box: tuple[float, float] = (2.0, 1.0),
) -> Array:
    values, gradients = potential_values_and_gradients(params, configurations, times, box=box)
    centered = values - jnp.einsum("tn,tn->t", weights, values)[:, None]
    kinetic = jnp.sum(gradients * gradients, axis=(-2, -1))
    rows = 0.5 * jnp.einsum("tn,tn->t", weights, kinetic) + jnp.einsum(
        "tn,tn,tn->t", weights, forcing, centered
    )
    return jnp.sum(time_weights * rows)


def _ritz_objective_contribution(
    params: RitzParams,
    configurations: Array,
    weights: Array,
    forcing: Array,
    forcing_mean: Array,
    times: Array,
    time_weights: Array,
    *,
    box: tuple[float, float],
) -> Array:
    """Additive chunk of the exact globally gauge-centered Ritz objective."""

    values, gradients = potential_values_and_gradients(params, configurations, times, box=box)
    kinetic = jnp.sum(gradients * gradients, axis=(-2, -1))
    rows = (
        0.5 * jnp.einsum("tn,tn->t", weights, kinetic)
        + jnp.einsum("tn,tn,tn->t", weights, forcing, values)
        - forcing_mean * jnp.einsum("tn,tn->t", weights, values)
    )
    return jnp.sum(time_weights * rows)


@dataclass(frozen=True)
class DeepRitzConfig:
    seed: int = 20260822
    hidden_width: int = 40
    hidden_layers: int = 2
    independent_time_nodes: int = 0
    adam_steps: int = 1200
    adam_batch_size: int = 512
    adam_learning_rate: float = 8.0e-4
    adam_min_learning_rate_ratio: float = 0.05
    gradient_clip_norm: float = 20.0
    lbfgs_iterations: int = 120
    lbfgs_batch_size: int = 512
    lbfgs_history: int = 12
    lbfgs_gradient_tolerance: float = 2.0e-7
    lbfgs_line_search_steps: int = 16
    log_every: int = 100
    compiled_full_bank: bool = False
    box: tuple[float, float] = (2.0, 1.0)


class DeepRitzResult(NamedTuple):
    params: RitzParams
    history: list[dict[str, Any]]
    adam_final_objective: float
    lbfgs_final_objective: float
    adam_seconds: float
    lbfgs_seconds: float
    lbfgs_converged: bool
    finite: bool


def _tree_norm(tree) -> Array:
    return jnp.sqrt(sum(jnp.sum(v * v) for v in jax.tree_util.tree_leaves(tree)))


def _lbfgs_refine(
    params: RitzParams,
    objective: Callable[[RitzParams], Array],
    cfg: DeepRitzConfig,
    history: list[dict[str, Any]],
    flat_value_grad: Callable[[Array], tuple[Array, Array]] | None = None,
) -> tuple[RitzParams, float, bool]:
    flat, unravel = ravel_pytree(params)
    value_grad = flat_value_grad or jax.jit(
        jax.value_and_grad(lambda vector: objective(unravel(vector)))
    )
    value, gradient = value_grad(flat)
    s_history: list[Array] = []
    y_history: list[Array] = []
    rho_history: list[Array] = []
    converged = False

    for iteration in range(1, int(cfg.lbfgs_iterations) + 1):
        grad_norm = float(jnp.linalg.norm(gradient))
        if not np.isfinite(float(value)) or not np.isfinite(grad_norm):
            break
        if grad_norm <= float(cfg.lbfgs_gradient_tolerance):
            converged = True
            break

        q = gradient
        alphas: list[Array] = []
        for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha = rho * jnp.vdot(s, q)
            alphas.append(alpha)
            q = q - alpha * y
        if s_history:
            gamma = jnp.vdot(s_history[-1], y_history[-1]) / jnp.maximum(
                jnp.vdot(y_history[-1], y_history[-1]), 1.0e-30
            )
        else:
            gamma = jnp.asarray(1.0, dtype=flat.dtype)
        direction = gamma * q
        for s, y, rho, alpha in zip(s_history, y_history, rho_history, reversed(alphas)):
            direction = direction + s * (alpha - rho * jnp.vdot(y, direction))
        direction = -direction
        directional = float(jnp.vdot(gradient, direction))
        if not np.isfinite(directional) or directional >= 0.0:
            direction = -gradient
            directional = -float(jnp.vdot(gradient, gradient))

        step_size = 1.0
        accepted = False
        next_value, next_gradient, next_flat = value, gradient, flat
        for _ in range(int(cfg.lbfgs_line_search_steps)):
            candidate = flat + step_size * direction
            candidate_value, candidate_gradient = value_grad(candidate)
            if np.isfinite(float(candidate_value)) and float(candidate_value) <= float(value) + 1.0e-4 * step_size * directional:
                next_value, next_gradient, next_flat = candidate_value, candidate_gradient, candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break

        s = next_flat - flat
        y = next_gradient - gradient
        curvature = jnp.vdot(s, y)
        if float(curvature) > 1.0e-12 * max(float(jnp.linalg.norm(s) * jnp.linalg.norm(y)), 1.0):
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / curvature)
            if len(s_history) > int(cfg.lbfgs_history):
                s_history.pop(0), y_history.pop(0), rho_history.pop(0)
        flat, value, gradient = next_flat, next_value, next_gradient
        if iteration == 1 or iteration % int(cfg.log_every) == 0 or iteration == int(cfg.lbfgs_iterations):
            history.append({
                "phase": "lbfgs", "iteration": iteration, "objective": float(value),
                "gradient_norm": float(jnp.linalg.norm(gradient)), "step_size": step_size,
            })
    return unravel(flat), float(value), converged


def solve_deep_ritz(
    configurations: Array,
    weights: Array,
    forcing: Array,
    times: Array,
    time_weights: Array,
    cfg: DeepRitzConfig = DeepRitzConfig(),
    *,
    initial_params: RitzParams | None = None,
) -> DeepRitzResult:
    """Projected-weight minibatch Adam followed by full-bank JAX L-BFGS.

    Adam samples configurations from each time node's projected empirical law,
    giving an unbiased stochastic Ritz objective.  L-BFGS and all reported
    diagnostics remain deterministic full-bank calculations.
    """

    arrays = tuple(jnp.asarray(v, dtype=jnp.float64) for v in (configurations, weights, forcing, times, time_weights))
    configurations, weights, forcing, times, time_weights = arrays
    params = initial_params or init_ritz_params(
        jax.random.PRNGKey(int(cfg.seed)), hidden_width=cfg.hidden_width,
        hidden_layers=cfg.hidden_layers,
        independent_time_nodes=cfg.independent_time_nodes,
    )
    if int(cfg.independent_time_nodes) > 0:
        if int(cfg.independent_time_nodes) != int(times.shape[0]):
            raise ValueError(
                "independent_time_nodes must equal the number of scientific time nodes"
            )
        params = promote_ritz_params_to_independent(
            params, int(cfg.independent_time_nodes)
        )
    objective = lambda p: ritz_objective(
        p, configurations, weights, forcing, times, time_weights, box=cfg.box
    )
    _, unravel_full = ravel_pytree(params)
    lbfgs_batch_size = min(max(int(cfg.lbfgs_batch_size), 1), int(configurations.shape[1]))
    forcing_mean = jnp.einsum("tn,tn->t", weights, forcing)

    def raw_chunk_flat_value_grad(vector, chunk_x, chunk_w, chunk_h):
        return jax.value_and_grad(
            lambda flat: _ritz_objective_contribution(
                unravel_full(flat), chunk_x, chunk_w, chunk_h, forcing_mean,
                times, time_weights, box=cfg.box,
            )
        )(vector)

    chunk_flat_value_grad = jax.jit(raw_chunk_flat_value_grad)

    sample_count = int(configurations.shape[1])
    compiled_chunks = bool(
        cfg.compiled_full_bank and sample_count % lbfgs_batch_size == 0
    )
    if compiled_chunks:
        chunk_count = sample_count // lbfgs_batch_size
        chunked_configurations = jnp.swapaxes(configurations.reshape(
            (configurations.shape[0], chunk_count, lbfgs_batch_size)
            + configurations.shape[2:]
        ), 0, 1)
        chunked_weights = jnp.swapaxes(weights.reshape(
            weights.shape[0], chunk_count, lbfgs_batch_size
        ), 0, 1)
        chunked_forcing = jnp.swapaxes(forcing.reshape(
            forcing.shape[0], chunk_count, lbfgs_batch_size
        ), 0, 1)

        @jax.jit
        def compiled_exact_flat_value_grad(vector: Array) -> tuple[Array, Array]:
            initial = (
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.zeros_like(vector),
            )

            def accumulate(carry, batch):
                value, gradient = raw_chunk_flat_value_grad(vector, *batch)
                return (carry[0] + value, carry[1] + gradient), None

            return jax.lax.scan(
                accumulate, initial,
                (chunked_configurations, chunked_weights, chunked_forcing),
            )[0]

    def exact_flat_value_grad(vector: Array) -> tuple[Array, Array]:
        if compiled_chunks:
            return compiled_exact_flat_value_grad(vector)
        total_value = jnp.asarray(0.0, dtype=jnp.float64)
        total_gradient = jnp.zeros_like(vector)
        for start in range(0, int(configurations.shape[1]), lbfgs_batch_size):
            stop = min(start + lbfgs_batch_size, int(configurations.shape[1]))
            value, gradient = chunk_flat_value_grad(
                vector,
                configurations[:, start:stop],
                weights[:, start:stop],
                forcing[:, start:stop],
            )
            total_value = total_value + value
            total_gradient = total_gradient + gradient
        return total_value, total_gradient

    def exact_objective(p: RitzParams) -> Array:
        vector, _ = ravel_pytree(p)
        return exact_flat_value_grad(vector)[0]
    adam_batch_size = min(max(int(cfg.adam_batch_size), 1), int(configurations.shape[1]))
    use_minibatches = adam_batch_size < int(configurations.shape[1])
    uniform_batch_weights = jnp.full(
        (configurations.shape[0], adam_batch_size),
        1.0 / float(adam_batch_size),
        dtype=jnp.float64,
    )

    @jax.jit
    def minibatch_value_grad(p, batch_configurations, batch_forcing):
        return jax.value_and_grad(ritz_objective)(
            p,
            batch_configurations,
            uniform_batch_weights,
            batch_forcing,
            times,
            time_weights,
            box=cfg.box,
        )

    def draw_batch(step: int) -> tuple[Array, Array]:
        step_key = jax.random.fold_in(jax.random.PRNGKey(int(cfg.seed) + 7919), int(step))
        keys = jax.random.split(step_key, configurations.shape[0])
        indices = jax.vmap(
            lambda key, row: jax.random.categorical(
                key, jnp.log(jnp.maximum(row, 1.0e-300)), shape=(adam_batch_size,)
            )
        )(keys, weights)
        configuration_indices = jnp.broadcast_to(
            indices[..., None, None],
            indices.shape + configurations.shape[2:],
        )
        gathered_x = jnp.take_along_axis(configurations, configuration_indices, axis=1)
        gathered_h = jnp.take_along_axis(forcing, indices, axis=1)
        return gathered_x, gathered_h
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    m, v = zeros, zeros
    history: list[dict[str, Any]] = []
    adam_started = time.perf_counter()
    finite = True
    adam_final = float("nan")

    if bool(cfg.compiled_full_bank) and not use_minibatches and int(cfg.adam_steps) > 0:
        flat_params, _ = ravel_pytree(params)
        leaf_sizes = tuple(
            int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)
        )
        leaf_starts = tuple(np.cumsum((0,) + leaf_sizes[:-1]).tolist())
        steps_np = np.arange(1, int(cfg.adam_steps) + 1, dtype=np.int32)
        fractions = steps_np.astype(np.float64) / max(float(cfg.adam_steps), 1.0)
        cosines = 0.5 * (1.0 + np.cos(np.pi * fractions))
        learning_rates = jnp.asarray(
            float(cfg.adam_learning_rate) * (
                float(cfg.adam_min_learning_rate_ratio)
                + (1.0 - float(cfg.adam_min_learning_rate_ratio)) * cosines
            ), dtype=jnp.float64,
        )
        beta1_corrections = jnp.asarray(
            1.0 - np.power(0.9, steps_np), dtype=jnp.float64
        )
        beta2_corrections = jnp.asarray(
            1.0 - np.power(0.999, steps_np), dtype=jnp.float64
        )

        def flat_tree_norm(vector: Array) -> Array:
            squared = jnp.asarray(0.0, dtype=vector.dtype)
            for start, size in zip(leaf_starts, leaf_sizes, strict=True):
                squared = squared + jnp.sum(jax.lax.dynamic_slice_in_dim(
                    vector, start, size
                ) ** 2)
            return jnp.sqrt(squared)

        @jax.jit
        def compiled_adam(initial_flat: Array):
            initial = (
                initial_flat, jnp.zeros_like(initial_flat), jnp.zeros_like(initial_flat)
            )

            def update(carry, schedule):
                flat, first_moment, second_moment = carry
                lr, beta1_correction, beta2_correction = schedule
                value, gradient = exact_flat_value_grad(flat)
                norm = flat_tree_norm(gradient)
                scale = jnp.minimum(
                    1.0,
                    float(cfg.gradient_clip_norm) / jnp.maximum(norm, 1.0e-30),
                )
                clipped = scale * gradient
                first_moment = 0.9 * first_moment + 0.1 * clipped
                second_moment = 0.999 * second_moment + 0.001 * clipped * clipped
                first_hat = first_moment / beta1_correction
                second_hat = second_moment / beta2_correction
                flat = flat - lr * first_hat / (jnp.sqrt(second_hat) + 1.0e-8)
                return (flat, first_moment, second_moment), (value, norm)

            return jax.lax.scan(
                update, initial,
                (learning_rates, beta1_corrections, beta2_corrections),
            )

        (flat_params, _, _), (values, norms) = compiled_adam(flat_params)
        params = unravel_full(flat_params)
        values_np, norms_np = np.asarray(values), np.asarray(norms)
        log_steps = tuple(
            step for step in range(1, int(cfg.adam_steps) + 1)
            if step == 1 or step % int(cfg.log_every) == 0
            or step == int(cfg.adam_steps)
        )
        for step in log_steps:
            value_float = float(values_np[step - 1])
            norm_float = float(norms_np[step - 1])
            if not np.isfinite(value_float) or not np.isfinite(norm_float):
                finite = False
                break
            history.append({
                "phase": "adam", "step": step, "objective": value_float,
                "gradient_norm": norm_float,
                "learning_rate": float(np.asarray(learning_rates)[step - 1]),
                "batch_size": adam_batch_size,
            })
    else:
        for step in range(1, int(cfg.adam_steps) + 1):
            if use_minibatches:
                batch_configurations, batch_forcing = draw_batch(step)
                value, grads = minibatch_value_grad(params, batch_configurations, batch_forcing)
            else:
                vector, _ = ravel_pytree(params)
                value, flat_grads = exact_flat_value_grad(vector)
                grads = unravel_full(flat_grads)
            norm = _tree_norm(grads)
            scale = jnp.minimum(1.0, float(cfg.gradient_clip_norm) / jnp.maximum(norm, 1.0e-30))
            grads = jax.tree_util.tree_map(lambda g: scale * g, grads)
            m = jax.tree_util.tree_map(lambda old, g: 0.9 * old + 0.1 * g, m, grads)
            v = jax.tree_util.tree_map(lambda old, g: 0.999 * old + 0.001 * g * g, v, grads)
            mhat = jax.tree_util.tree_map(lambda z: z / (1.0 - 0.9**step), m)
            vhat = jax.tree_util.tree_map(lambda z: z / (1.0 - 0.999**step), v)
            fraction = step / max(float(cfg.adam_steps), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * fraction))
            lr = float(cfg.adam_learning_rate) * (
                float(cfg.adam_min_learning_rate_ratio)
                + (1.0 - float(cfg.adam_min_learning_rate_ratio)) * cosine
            )
            params = jax.tree_util.tree_map(
                lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1.0e-8), params, mhat, vhat
            )
            if step == 1 or step % int(cfg.log_every) == 0 or step == int(cfg.adam_steps):
                value_float, norm_float = float(value), float(norm)
                if not np.isfinite(value_float) or not np.isfinite(norm_float):
                    finite = False
                    break
                history.append({
                    "phase": "adam", "step": step, "objective": value_float,
                    "gradient_norm": norm_float, "learning_rate": float(lr),
                    "batch_size": adam_batch_size,
                })
    if finite:
        adam_final = float(exact_objective(params))
    adam_seconds = time.perf_counter() - adam_started

    lbfgs_started = time.perf_counter()
    if finite and int(cfg.lbfgs_iterations) > 0:
        params, lbfgs_final, converged = _lbfgs_refine(
            params, objective, cfg, history,
            flat_value_grad=exact_flat_value_grad,
        )
    else:
        lbfgs_final, converged = adam_final, False
    lbfgs_seconds = time.perf_counter() - lbfgs_started
    finite = finite and np.isfinite(lbfgs_final)
    return DeepRitzResult(
        params=params,
        history=history,
        adam_final_objective=adam_final,
        lbfgs_final_objective=lbfgs_final,
        adam_seconds=adam_seconds,
        lbfgs_seconds=lbfgs_seconds,
        lbfgs_converged=converged,
        finite=bool(finite),
    )


@dataclass(frozen=True)
class CertificateConfig:
    maximum_weak_residual: float = 0.12
    maximum_energy_residual: float = 0.08
    maximum_gauge_residual: float = 1.0e-9
    maximum_moment_rate_residual: float = 0.10


def _audit_features(configuration: Array, box: tuple[float, float]) -> Array:
    x = configuration
    probes = jnp.asarray([[0.21, 0.22], [0.58, 0.78], [1.18, 0.35], [1.72, 0.70]], dtype=x.dtype)
    delta = minimum_image(x[:, None, :] - probes, jnp.asarray(box, dtype=x.dtype))
    local = jnp.mean(jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1) / 0.14**2), axis=0)
    pair = minimum_image(x[:, None, :] - x[None, :, :], jnp.asarray(box, dtype=x.dtype))
    distance = jnp.sqrt(jnp.sum(pair * pair, axis=-1) + 1.0e-10)
    mask = 1.0 - jnp.eye(x.shape[0], dtype=x.dtype)
    pair_features = jnp.stack([
        jnp.sum(mask * jnp.exp(-0.5 * ((distance - center) / 0.07) ** 2))
        / float(x.shape[0] * (x.shape[0] - 1))
        for center in (0.16, 0.30, 0.48)
    ])
    phase_x = 2 * jnp.pi * x[:, 0] / box[0]
    phase_y = 2 * jnp.pi * x[:, 1] / box[1]
    structure = jnp.asarray([
        jnp.mean(jnp.cos(phase_x)) ** 2 + jnp.mean(jnp.sin(phase_x)) ** 2,
        jnp.mean(jnp.cos(phase_y)) ** 2 + jnp.mean(jnp.sin(phase_y)) ** 2,
    ])
    return jnp.concatenate([local, pair_features, structure])


def audit_deep_ritz(
    params: RitzParams,
    configurations: Array,
    weights: Array,
    forcing: Array,
    times: Array,
    time_weights: Array,
    *,
    family: LocalDensitySensors | None = None,
    eta: Array | None = None,
    reference_velocity: Array | None = None,
    target_derivatives: Array | None = None,
    cfg: CertificateConfig = CertificateConfig(),
    box: tuple[float, float] = (2.0, 1.0),
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Independent exact certificates accumulated in bounded-memory chunks."""

    configurations = jnp.asarray(configurations)
    weights = jnp.asarray(weights)
    forcing = jnp.asarray(forcing)
    times = jnp.asarray(times)
    time_weights = jnp.asarray(time_weights)
    sample_count = int(configurations.shape[1])
    chunk_size = min(max(int(chunk_size), 1), sample_count)
    feature_count = int(_audit_features(configurations[0, 0], box).shape[0])
    dtype = configurations.dtype

    feature_values = jax.vmap(jax.vmap(lambda x: _audit_features(x, box)))

    @jax.jit
    def first_pass(chunk_x, chunk_w):
        values = potential_values(params, chunk_x, times, box=box)
        tests = feature_values(chunk_x)
        return (
            jnp.einsum("tn,tn->t", chunk_w, values),
            jnp.einsum("tn,tnk->tk", chunk_w, tests),
        )

    value_means = jnp.zeros((len(times),), dtype=dtype)
    test_means = jnp.zeros((len(times), feature_count), dtype=dtype)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        value_part, test_part = first_pass(
            configurations[:, start:stop], weights[:, start:stop]
        )
        value_means = value_means + value_part
        test_means = test_means + test_part

    feature_fn = lambda x: _audit_features(x, box)
    per_sample = jax.vmap(lambda x: (feature_fn(x), jax.jacrev(feature_fn)(x)))
    feature_values_and_gradients = jax.vmap(per_sample)

    @jax.jit
    def second_pass(chunk_x, chunk_w, chunk_h):
        values, gradients = potential_values_and_gradients(
            params, chunk_x, times, box=box
        )
        tests, test_gradients = feature_values_and_gradients(chunk_x)
        centered_values = values - value_means[:, None]
        centered_tests = tests - test_means[:, None, :]
        energy_rows = jnp.sum(gradients * gradients, axis=(-2, -1))
        corrected_rate = (
            family.jvp(chunk_x, -gradients, eta)
            if family is not None and eta is not None
            else jnp.zeros((chunk_x.shape[0], chunk_x.shape[1], 0), dtype=dtype)
        )
        return (
            jnp.einsum("tn,tn->t", chunk_w, energy_rows),
            jnp.einsum("tn,tn->t", chunk_w, energy_rows * energy_rows),
            jnp.einsum("tn,tn,tn->t", chunk_w, chunk_h, centered_values),
            jnp.einsum("tn,tn->t", chunk_w, centered_values),
            jnp.einsum("tn,tnad,tnkad->tk", chunk_w, gradients, test_gradients),
            jnp.einsum("tn,tn,tnk->tk", chunk_w, chunk_h, centered_tests),
            jnp.einsum("tn,tnkad,tnkad->tk", chunk_w, test_gradients, test_gradients),
            jnp.einsum("tn,tn,tn->t", chunk_w, chunk_h, chunk_h),
            jnp.einsum("tn,tnk,tnk->tk", chunk_w, centered_tests, centered_tests),
            jnp.einsum("tn,tnr->tr", chunk_w, corrected_rate),
        )

    kinetic = jnp.zeros((len(times),), dtype=dtype)
    kinetic_second = jnp.zeros_like(kinetic)
    linear = jnp.zeros_like(kinetic)
    gauge = jnp.zeros_like(kinetic)
    weak_left = jnp.zeros((len(times), feature_count), dtype=dtype)
    weak_right = jnp.zeros_like(weak_left)
    grad_scale_sq = jnp.zeros_like(weak_left)
    h_scale_sq = jnp.zeros_like(kinetic)
    phi_scale_sq = jnp.zeros_like(weak_left)
    moment_columns = int(eta.shape[0] // 2) if family is not None and eta is not None else 0
    corrected_moment_rate = jnp.zeros((len(times), moment_columns), dtype=dtype)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        parts = second_pass(
            configurations[:, start:stop],
            weights[:, start:stop],
            forcing[:, start:stop],
        )
        kinetic = kinetic + parts[0]
        kinetic_second = kinetic_second + parts[1]
        linear = linear + parts[2]
        gauge = gauge + parts[3]
        weak_left = weak_left + parts[4]
        weak_right = weak_right + parts[5]
        grad_scale_sq = grad_scale_sq + parts[6]
        h_scale_sq = h_scale_sq + parts[7]
        phi_scale_sq = phi_scale_sq + parts[8]
        corrected_moment_rate = corrected_moment_rate + parts[9]

    kinetic_variance = jnp.maximum(kinetic_second - kinetic * kinetic, 0.0)
    effective_samples = 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=-1), 1.0e-300)
    action_standard_error = jnp.sqrt(jnp.sum(
        (time_weights * jnp.sqrt(kinetic_variance / jnp.maximum(effective_samples, 1.0))) ** 2
    ))
    energy_residual = jnp.abs(kinetic + linear) / jnp.maximum(
        kinetic + jnp.abs(linear), 1.0e-12
    )
    raw_weak = weak_left + weak_right
    psi_scale = jnp.sqrt(kinetic)[:, None] * jnp.sqrt(grad_scale_sq)
    weak_normalized = jnp.abs(raw_weak) / jnp.maximum(
        psi_scale + jnp.sqrt(h_scale_sq)[:, None] * jnp.sqrt(phi_scale_sq),
        1.0e-12,
    )

    moment_rate = jnp.asarray(0.0)
    if (
        family is not None and eta is not None and reference_velocity is not None
        and target_derivatives is not None
    ):
        advective_moment_rate = jnp.zeros_like(corrected_moment_rate)
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            advective = family.jvp(
                configurations[:, start:stop],
                reference_velocity[:, start:stop],
                eta,
            )
            advective_moment_rate = advective_moment_rate + jnp.einsum(
                "tn,tnr->tr", weights[:, start:stop], advective
            )
        rhs = target_derivatives - advective_moment_rate
        moment_rate = jnp.max(
            jnp.linalg.norm(corrected_moment_rate - rhs, axis=-1)
            / jnp.maximum(1.0, jnp.linalg.norm(rhs, axis=-1))
        )

    max_weak = float(jnp.max(weak_normalized))
    max_energy = float(jnp.max(energy_residual))
    max_gauge = float(jnp.max(jnp.abs(gauge)))
    max_moment = float(moment_rate)
    action = float(jnp.sum(time_weights * kinetic))
    finite = all(np.isfinite(value) for value in (max_weak, max_energy, max_gauge, max_moment, action))
    valid = (
        finite
        and max_weak <= cfg.maximum_weak_residual
        and max_energy <= cfg.maximum_energy_residual
        and max_gauge <= cfg.maximum_gauge_residual
        and max_moment <= cfg.maximum_moment_rate_residual
    )
    return {
        "action": action,
        "action_standard_error": float(action_standard_error),
        "kinetic_by_time": np.asarray(kinetic).tolist(),
        "maximum_weak_residual": max_weak,
        "weak_residual_by_time_and_feature": np.asarray(weak_normalized).tolist(),
        "maximum_energy_residual": max_energy,
        "energy_residual_by_time": np.asarray(energy_residual).tolist(),
        "maximum_gauge_residual": max_gauge,
        "maximum_moment_rate_residual": max_moment,
        "valid": valid,
        "thresholds": asdict(cfg),
    }


def manufactured_cosine_weak_residual(
    params: RitzParams,
    configurations: Array,
    weights: Array,
    *,
    t: float = 0.5,
    box: tuple[float, float] = (2.0, 1.0),
) -> Array:
    """Held-out residual for ``h=mean_a cos(2π x_a/Lx)`` on a uniform torus.

    The exact mean-zero solution is ``psi=-h/k²``.  This helper is an internal
    manufactured-problem check for optimizer tests, not a Stage-A experiment.
    """

    x = jnp.asarray(configurations, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    wave = 2.0 * jnp.pi / float(box[0])

    def test_feature(row):
        return jnp.mean(jnp.cos(wave * row[:, 0]))

    psi_grad = jax.vmap(
        jax.grad(lambda row: invariant_potential(params, row, jnp.asarray(t), box=box))
    )(x)
    values, test_grad = jax.vmap(jax.value_and_grad(test_feature))(x)
    centered = values - weights @ values
    residual = jnp.einsum("n,nad,nad->", weights, psi_grad, test_grad) + jnp.einsum(
        "n,n,n->", weights, values, centered
    )
    scale = jnp.sqrt(jnp.einsum("n,nad,nad->", weights, psi_grad, psi_grad)) * jnp.sqrt(
        jnp.einsum("n,nad,nad->", weights, test_grad, test_grad)
    ) + jnp.sqrt(jnp.einsum("n,n,n->", weights, values, values)) * jnp.sqrt(
        jnp.einsum("n,n,n->", weights, centered, centered)
    )
    return jnp.abs(residual) / jnp.maximum(scale, 1.0e-12)
