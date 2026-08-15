from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .interfaces import EndpointSource
from .reference import MLPReferenceFlow, Params, velocity_mlp

Array = jax.Array


@dataclass(frozen=True)
class FlowMatchingConfig:
    seed: int = 20260813
    hidden_width: int = 128
    hidden_layers: int = 4
    train_steps: int = 12000
    batch_size: int = 2048
    learning_rate: float = 1.0e-3
    min_learning_rate_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip_norm: float = 10.0
    bridge_schedule: str = "linear"
    bridge_noise_std: float = 0.15
    log_every: int = 500


class AdamState(NamedTuple):
    m: Any
    v: Any
    step: Array


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_norm(tree) -> Array:
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)))


def _tree_scale(tree, scale: Array):
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


def init_mlp(key: Array, input_dim: int, hidden_width: int, hidden_layers: int, output_dim: int = 2) -> Params:
    dims = [input_dim] + [hidden_width] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        std = jnp.sqrt(2.0 / float(din + dout))
        params.append({
            "W": std * jax.random.normal(k, (din, dout), dtype=jnp.float64),
            "b": jnp.zeros((dout,), dtype=jnp.float64),
        })
    return tuple(params)


def bridge_coefficients(t: Array, schedule: str) -> tuple[Array, Array, Array, Array]:
    t = jnp.asarray(t, dtype=jnp.float64)
    if schedule == "linear":
        return 1.0 - t, t, -jnp.ones_like(t), jnp.ones_like(t)
    if schedule == "trig":
        half = 0.5 * jnp.pi * t
        return (
            jnp.cos(half),
            jnp.sin(half),
            -0.5 * jnp.pi * jnp.sin(half),
            0.5 * jnp.pi * jnp.cos(half),
        )
    raise ValueError(f"unknown bridge schedule {schedule!r}")


def stochastic_interpolant(t: Array, x0: Array, x1: Array, z: Array, schedule: str, noise_std: float) -> tuple[Array, Array]:
    alpha, beta, alpha_dot, beta_dot = bridge_coefficients(t, schedule)
    gamma = float(noise_std) * jnp.sin(jnp.pi * t)
    gamma_dot = float(noise_std) * jnp.pi * jnp.cos(jnp.pi * t)
    while alpha.ndim < x0.ndim:
        alpha = alpha[..., None]
        beta = beta[..., None]
        alpha_dot = alpha_dot[..., None]
        beta_dot = beta_dot[..., None]
        gamma = gamma[..., None]
        gamma_dot = gamma_dot[..., None]
    xt = alpha * x0 + beta * x1 + gamma * z
    target = alpha_dot * x0 + beta_dot * x1 + gamma_dot * z
    return xt, target


def sample_cfm_batch(source: EndpointSource, key: Array, n: int, schedule: str, noise_std: float) -> tuple[Array, Array, Array]:
    kt, k0, k1, kz = jax.random.split(key, 4)
    t = jax.random.uniform(kt, (n,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    x0 = source.sample(k0, n, endpoint=0)
    x1 = source.sample(k1, n, endpoint=1)
    z = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
    xt, target = stochastic_interpolant(t, x0, x1, z, schedule, noise_std)
    return t, xt, target


def cosine_learning_rate(step: Array, total_steps: int, lr0: float, min_ratio: float) -> Array:
    frac = jnp.clip(step / max(float(total_steps), 1.0), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
    return lr0 * (min_ratio + (1.0 - min_ratio) * cosine)


def train_reference_flow(source: EndpointSource, cfg: FlowMatchingConfig, *, substeps_per_interval: int) -> tuple[MLPReferenceFlow, list[dict[str, float]]]:
    key = jax.random.PRNGKey(cfg.seed)
    kinit, key = jax.random.split(key)
    params = init_mlp(kinit, input_dim=7, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers)
    state = AdamState(_tree_zeros_like(params), _tree_zeros_like(params), jnp.asarray(0, dtype=jnp.int32))

    beta1, beta2, eps = cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps

    def loss_fn(p, t, x, target):
        err = velocity_mlp(p, t, x) - target
        return jnp.mean(jnp.sum(err * err, axis=-1))

    @jax.jit
    def step_fn(p, s, k):
        t, x, target = sample_cfm_batch(source, k, cfg.batch_size, cfg.bridge_schedule, cfg.bridge_noise_std)
        loss, grads = jax.value_and_grad(loss_fn)(p, t, x, target)
        gnorm = _tree_norm(grads)
        grads = _tree_scale(grads, jnp.minimum(1.0, cfg.grad_clip_norm / jnp.maximum(gnorm, 1.0e-30)))
        step = s.step + 1
        m = jax.tree_util.tree_map(lambda m0, g: beta1 * m0 + (1.0 - beta1) * g, s.m, grads)
        v = jax.tree_util.tree_map(lambda v0, g: beta2 * v0 + (1.0 - beta2) * g * g, s.v, grads)
        mhat = jax.tree_util.tree_map(lambda z: z / (1.0 - beta1 ** step), m)
        vhat = jax.tree_util.tree_map(lambda z: z / (1.0 - beta2 ** step), v)
        lr = cosine_learning_rate(step, cfg.train_steps, cfg.learning_rate, cfg.min_learning_rate_ratio)
        p = jax.tree_util.tree_map(lambda q, mh, vh: q - lr * mh / (jnp.sqrt(vh) + eps), p, mhat, vhat)
        return p, AdamState(m, v, step), loss, gnorm, lr

    history: list[dict[str, float]] = []
    for step in range(1, cfg.train_steps + 1):
        key, ks = jax.random.split(key)
        params, state, loss, gnorm, lr = step_fn(params, state, ks)
        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            row = {
                "step": step,
                "conditional_fm_loss": float(loss),
                "grad_norm_preclip": float(gnorm),
                "learning_rate": float(lr),
            }
            history.append(row)
            print(
                f"reference train {step:6d}/{cfg.train_steps} | "
                f"loss={row['conditional_fm_loss']:.6e} | grad={row['grad_norm_preclip']:.3e}",
                flush=True,
            )

    metadata = {
        "network": {
            "parameter_layers": cfg.hidden_layers + 1,
            "hidden_width": cfg.hidden_width,
            "hidden_layers": cfg.hidden_layers,
        },
        "bridge": {
            "schedule": cfg.bridge_schedule,
            "noise_std": cfg.bridge_noise_std,
            "independent_endpoint_pairing": True,
        },
        "training": {"seed": cfg.seed, "steps": cfg.train_steps, "batch_size": cfg.batch_size},
    }
    return MLPReferenceFlow(params, substeps_per_interval=substeps_per_interval, metadata=metadata), history
