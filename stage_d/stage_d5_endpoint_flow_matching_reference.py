#!/usr/bin/env python3
"""
Stage D.5: endpoint-trained stochastic-interpolant / conditional flow matching
reference (NO analytic path teacher, NO CNF).

Scientific purpose
------------------
Stages D.0--D.4 used a neural velocity that was trained to approximate the known
Stage-B analytic reference path.  D.5 removes that teacher.  The model sees only
samples from the two endpoint laws Q_0 and Q_1 and a generic, user-declared
stochastic-interpolant construction.

The default bridge is

    X_t = (1-t) X_0 + t X_1 + gamma(t) Z,
    gamma(t) = eps * sin(pi t),

with X_0 ~ Q_0, X_1 ~ Q_1 sampled independently and Z ~ N(0,I).  Its samplewise
conditional flow-matching target is

    dX_t/dt = X_1 - X_0 + eps*pi*cos(pi t) Z.

The neural field u_theta(t,x) is trained by

    E ||u_theta(t, X_t) - dX_t/dt||^2.

Crucially, this target does NOT use the Stage-B analytic A_t map, B_t matrix, or
B_t x velocity teacher.  The population velocity represented by the trained model
is the conditional expectation E[dX_t/dt | X_t=x].

Two endpoint-data modes are supported:

  1. synthetic Stage-B endpoints (default convenience mode): only the endpoint
     mixture parameters r and sigma are read from the Stage-B configuration;
     no intermediate analytic map or velocity is evaluated;
  2. external endpoint arrays via --x0-samples and --x1-samples (.npy or .npz).

Validation is teacher-free:

  * held-out conditional-FM regression loss (not expected to vanish because the
    conditional target has irreducible variance under independent endpoint pairing),
  * ODE-rollout marginals versus direct samples from the declared stochastic
    interpolant at held-out times, using MMD / mean / covariance diagnostics,
  * endpoint rollout Q_0 -> Q_1 distributional fidelity,
  * learned divergence magnitude as a diagnostic only (there is no analytic
    divergence target in D.5).

The checkpoint format and neural architecture intentionally match D.0 so existing
Stage-D utilities that only require velocity_mlp/load_checkpoint can read D.5
checkpoints.  Downstream science should nevertheless identify the checkpoint as
Stage D.5 and should not interpret any analytic-particle comparison as part of D.5
training.

Examples
--------
Synthetic Stage-B endpoint samples, smoke test:

    python stage_d5_endpoint_flow_matching_reference.py \\
        --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
        --preset quick

Main endpoint-trained reference:

    python stage_d5_endpoint_flow_matching_reference.py \\
        --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
        --preset reference \\
        --output-prefix stage_d5_endpoint_flow_matching_reference

External endpoint samples:

    python stage_d5_endpoint_flow_matching_reference.py \\
        --x0-samples q0_samples.npy \\
        --x1-samples q1_samples.npy \\
        --bridge-noise-std 0.10 \\
        --mmd-bandwidth 0.50 \\
        --preset reference

Evaluate an existing checkpoint (pass the same endpoint data/source metadata):

    python stage_d5_endpoint_flow_matching_reference.py \\
        --backend ../stage_b/stage_b2_transport_conditioned_design.py \\
        --eval-only \\
        --checkpoint stage_d5_endpoint_flow_matching_reference.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


Array = jax.Array
PI = math.pi


# -----------------------------------------------------------------------------
# Optional Stage-B loading: endpoint parameters only
# -----------------------------------------------------------------------------


def load_backend(path: Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Stage-B backend not found: {path}")
    spec = importlib.util.spec_from_file_location("stage_b2_backend_d5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Stage-B backend from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def autodetect_backend() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [
        Path("stage_b2_transport_conditioned_design.py"),
        Path("stage_b2_transport_conditioned_design(4).py"),
        here / "stage_b2_transport_conditioned_design.py",
        here / "stage_b2_transport_conditioned_design(4).py",
        Path("../stage_b/stage_b2_transport_conditioned_design.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class D5Config:
    preset: str = "quick"
    seed: int = 20260813

    # Same model interface as D.0: input (x1,x2,time features) -> R^2 velocity.
    hidden_width: int = 64
    hidden_layers: int = 3

    # Optimization.
    train_steps: int = 1000
    batch_size: int = 1024
    learning_rate: float = 2.0e-3
    min_learning_rate_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip_norm: float = 10.0
    log_every: int = 100

    # Generic bridge.  The default absolute noise level is inferred as
    # bridge_noise_multiplier * endpoint sigma when Stage-B endpoint parameters
    # are available.  For external-only data with no Stage-B backend, the default
    # is zero unless --bridge-noise-std is given explicitly.
    bridge_schedule: str = "linear"  # linear or trig
    bridge_noise_multiplier: float = 0.5

    # External-data split.  Synthetic endpoints generate fresh validation samples.
    validation_fraction: float = 0.20

    # Teacher-free regression validation.
    validation_size: int = 8192
    validation_time_bins: int = 10
    divergence_validation_size: int = 128

    # Teacher-free ODE/path validation.
    rollout_particles: int = 512
    rollout_steps: int = 100
    rollout_eval_times: int = 11

    # If None, use Stage-B bandwidth when available, otherwise median heuristic.
    mmd_bandwidth: float | None = None


def preset_d5_config(name: str) -> D5Config:
    if name == "quick":
        return D5Config()
    if name == "reference":
        return D5Config(
            preset="reference",
            hidden_width=128,
            hidden_layers=4,
            train_steps=12000,
            batch_size=2048,
            learning_rate=1.0e-3,
            log_every=500,
            validation_size=32768,
            validation_time_bins=20,
            divergence_validation_size=1024,
            rollout_particles=2048,
            rollout_steps=240,
            rollout_eval_times=21,
        )
    if name == "confirm":
        return D5Config(
            preset="confirm",
            hidden_width=192,
            hidden_layers=4,
            train_steps=24000,
            batch_size=4096,
            learning_rate=8.0e-4,
            log_every=1000,
            validation_size=65536,
            validation_time_bins=25,
            divergence_validation_size=2048,
            rollout_particles=4096,
            rollout_steps=400,
            rollout_eval_times=31,
        )
    raise ValueError(f"Unknown Stage D.5 preset {name!r}")


# -----------------------------------------------------------------------------
# Endpoint data
# -----------------------------------------------------------------------------


def _load_array(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = next((k for k in preferred_keys if k in data.files), None)
            if key is None:
                if len(data.files) != 1:
                    raise ValueError(
                        f"{path} contains keys {data.files}; expected one of {preferred_keys} "
                        "or a single-array npz."
                    )
                key = data.files[0]
            arr = data[key]
    else:
        raise ValueError(f"Endpoint samples must be .npy or .npz, got {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected endpoint array with shape [N,2], got {arr.shape} from {path}")
    if len(arr) < 8:
        raise ValueError(f"Need at least 8 endpoint samples, got {len(arr)} from {path}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {path}")
    return arr


def _split_array(x: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < fraction < 0.5):
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(1, int(round(fraction * len(x))))
    n_val = min(n_val, len(x) - 1)
    val = x[idx[:n_val]]
    train = x[idx[n_val:]]
    return train, val


class EndpointSampler:
    """
    Sample Q0/Q1 without evaluating any intermediate analytic Stage-B path.

    Synthetic mode uses only endpoint marginal parameters:
        Q0 = 0.5 N((+r,0), sigma^2 I) + 0.5 N((-r,0), sigma^2 I)
        Q1 = 0.5 N((0,+r), sigma^2 I) + 0.5 N((0,-r), sigma^2 I)

    External mode samples with replacement from train/validation arrays.
    """

    def __init__(
        self,
        *,
        r: float | None,
        sigma: float | None,
        x0_external: np.ndarray | None,
        x1_external: np.ndarray | None,
        validation_fraction: float,
        seed: int,
    ):
        if (x0_external is None) != (x1_external is None):
            raise ValueError("Pass both --x0-samples and --x1-samples, or neither.")

        self.r = None if r is None else float(r)
        self.sigma = None if sigma is None else float(sigma)
        self.mode = "external" if x0_external is not None else "synthetic_stage_b_endpoints"

        if self.mode == "synthetic_stage_b_endpoints":
            if self.r is None or self.sigma is None:
                raise ValueError(
                    "Synthetic endpoint mode requires Stage-B r and sigma. Pass --backend, "
                    "or provide --x0-samples and --x1-samples."
                )
            self.x0_train = self.x0_val = None
            self.x1_train = self.x1_val = None
        else:
            x0_train, x0_val = _split_array(x0_external, validation_fraction, seed + 11)
            x1_train, x1_val = _split_array(x1_external, validation_fraction, seed + 23)
            self.x0_train = jnp.asarray(x0_train, dtype=jnp.float64)
            self.x0_val = jnp.asarray(x0_val, dtype=jnp.float64)
            self.x1_train = jnp.asarray(x1_train, dtype=jnp.float64)
            self.x1_val = jnp.asarray(x1_val, dtype=jnp.float64)

    def _sample_synthetic(self, key: Array, n: int, endpoint: int) -> Array:
        ksign, knoise = jax.random.split(key)
        signs = jnp.where(jax.random.bernoulli(ksign, 0.5, (n,)), 1.0, -1.0)
        noise = self.sigma * jax.random.normal(knoise, (n, 2), dtype=jnp.float64)
        if endpoint == 0:
            mean = jnp.stack([self.r * signs, jnp.zeros_like(signs)], axis=-1)
        else:
            mean = jnp.stack([jnp.zeros_like(signs), self.r * signs], axis=-1)
        return mean + noise

    @staticmethod
    def _sample_array(key: Array, arr: Array, n: int) -> Array:
        idx = jax.random.randint(key, (n,), minval=0, maxval=arr.shape[0])
        return arr[idx]

    def sample_x0(self, key: Array, n: int, split: str = "train") -> Array:
        if self.mode == "synthetic_stage_b_endpoints":
            return self._sample_synthetic(key, n, endpoint=0)
        arr = self.x0_train if split == "train" else self.x0_val
        return self._sample_array(key, arr, n)

    def sample_x1(self, key: Array, n: int, split: str = "train") -> Array:
        if self.mode == "synthetic_stage_b_endpoints":
            return self._sample_synthetic(key, n, endpoint=1)
        arr = self.x1_train if split == "train" else self.x1_val
        return self._sample_array(key, arr, n)

    def metadata(self) -> Dict[str, Any]:
        if self.mode == "synthetic_stage_b_endpoints":
            return {
                "mode": self.mode,
                "r": self.r,
                "sigma": self.sigma,
                "analytic_intermediate_path_used": False,
            }
        return {
            "mode": self.mode,
            "train_sizes": {"x0": int(self.x0_train.shape[0]), "x1": int(self.x1_train.shape[0])},
            "validation_sizes": {"x0": int(self.x0_val.shape[0]), "x1": int(self.x1_val.shape[0])},
            "analytic_intermediate_path_used": False,
        }


# -----------------------------------------------------------------------------
# Stochastic interpolant / conditional-FM bridge
# -----------------------------------------------------------------------------


def bridge_coefficients(t: Array, schedule: str) -> Tuple[Array, Array, Array, Array]:
    """Return alpha, beta, alpha_dot, beta_dot."""
    t = jnp.asarray(t, dtype=jnp.float64)
    if schedule == "linear":
        alpha = 1.0 - t
        beta = t
        alpha_dot = -jnp.ones_like(t)
        beta_dot = jnp.ones_like(t)
    elif schedule == "trig":
        half = 0.5 * jnp.pi * t
        alpha = jnp.cos(half)
        beta = jnp.sin(half)
        alpha_dot = -0.5 * jnp.pi * jnp.sin(half)
        beta_dot = 0.5 * jnp.pi * jnp.cos(half)
    else:
        raise ValueError(f"Unknown bridge schedule {schedule!r}")
    return alpha, beta, alpha_dot, beta_dot


def bridge_noise(t: Array, noise_std: float) -> Tuple[Array, Array]:
    t = jnp.asarray(t, dtype=jnp.float64)
    gamma = float(noise_std) * jnp.sin(jnp.pi * t)
    gamma_dot = float(noise_std) * jnp.pi * jnp.cos(jnp.pi * t)
    return gamma, gamma_dot


def stochastic_interpolant(
    t: Array,
    x0: Array,
    x1: Array,
    z: Array,
    schedule: str,
    noise_std: float,
) -> Tuple[Array, Array]:
    """Sample X_t and its samplewise derivative dX_t/dt."""
    alpha, beta, alpha_dot, beta_dot = bridge_coefficients(t, schedule)
    gamma, gamma_dot = bridge_noise(t, noise_std)
    while alpha.ndim < x0.ndim:
        alpha = alpha[..., None]
        beta = beta[..., None]
        alpha_dot = alpha_dot[..., None]
        beta_dot = beta_dot[..., None]
        gamma = gamma[..., None]
        gamma_dot = gamma_dot[..., None]
    xt = alpha * x0 + beta * x1 + gamma * z
    dxt = alpha_dot * x0 + beta_dot * x1 + gamma_dot * z
    return xt, dxt


def sample_cfm_batch(
    sampler: EndpointSampler,
    key: Array,
    n: int,
    schedule: str,
    noise_std: float,
    split: str = "train",
):
    kt, k0, k1, kz = jax.random.split(key, 4)
    t = jax.random.uniform(kt, (n,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    # Independent endpoint draws are intentional: D.5 assumes no pairing oracle.
    x0 = sampler.sample_x0(k0, n, split=split)
    x1 = sampler.sample_x1(k1, n, split=split)
    z = jax.random.normal(kz, (n, 2), dtype=jnp.float64)
    xt, target = stochastic_interpolant(t, x0, x1, z, schedule, noise_std)
    return t, xt, target


# -----------------------------------------------------------------------------
# Neural velocity model -- checkpoint-compatible with D.0
# -----------------------------------------------------------------------------


def time_features(t: Array) -> Array:
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


def init_mlp(key: Array, input_dim: int, hidden_width: int, hidden_layers: int, output_dim: int = 2):
    dims = [input_dim] + [hidden_width] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        std = math.sqrt(2.0 / float(din + dout))
        W = std * jax.random.normal(k, (din, dout), dtype=jnp.float64)
        b = jnp.zeros((dout,), dtype=jnp.float64)
        params.append({"W": W, "b": b})
    return tuple(params)


def velocity_mlp(params, t: Array, x: Array) -> Array:
    h = model_features(t, x)
    for layer in params[:-1]:
        h = jax.nn.silu(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_global_norm(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def tree_scale(tree, scale: Array):
    return jax.tree_util.tree_map(lambda x: x * scale, tree)


class AdamState(NamedTuple):
    m: Any
    v: Any
    step: Array


def cosine_learning_rate(step: Array, total_steps: int, lr0: float, min_ratio: float) -> Array:
    frac = jnp.clip(step / max(float(total_steps), 1.0), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
    return lr0 * (min_ratio + (1.0 - min_ratio) * cosine)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def make_train_step(sampler: EndpointSampler, cfg: D5Config, noise_std: float):
    beta1 = float(cfg.adam_beta1)
    beta2 = float(cfg.adam_beta2)
    eps = float(cfg.adam_eps)

    def loss_fn(params, t, x, target):
        pred = velocity_mlp(params, t, x)
        err = pred - target
        return jnp.mean(jnp.sum(err * err, axis=-1))

    @jax.jit
    def train_step(params, state: AdamState, key: Array):
        t, x, target = sample_cfm_batch(
            sampler, key, cfg.batch_size, cfg.bridge_schedule, noise_std, split="train"
        )
        loss, grads = jax.value_and_grad(loss_fn)(params, t, x, target)

        gnorm = tree_global_norm(grads)
        clip_scale = jnp.minimum(1.0, cfg.grad_clip_norm / jnp.maximum(gnorm, 1e-30))
        grads = tree_scale(grads, clip_scale)

        step = state.step + 1
        m = jax.tree_util.tree_map(lambda m0, g: beta1 * m0 + (1.0 - beta1) * g, state.m, grads)
        v = jax.tree_util.tree_map(lambda v0, g: beta2 * v0 + (1.0 - beta2) * (g * g), state.v, grads)
        mhat = jax.tree_util.tree_map(lambda z: z / (1.0 - beta1 ** step), m)
        vhat = jax.tree_util.tree_map(lambda z: z / (1.0 - beta2 ** step), v)
        lr = cosine_learning_rate(step, cfg.train_steps, cfg.learning_rate, cfg.min_learning_rate_ratio)
        params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params, mhat, vhat,
        )
        return params, AdamState(m=m, v=v, step=step), loss, gnorm, lr

    return train_step


def train_model(sampler: EndpointSampler, cfg: D5Config, noise_std: float):
    key = jax.random.PRNGKey(cfg.seed)
    kinit, key = jax.random.split(key)
    params = init_mlp(kinit, input_dim=7, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers)
    state = AdamState(
        m=tree_zeros_like(params),
        v=tree_zeros_like(params),
        step=jnp.asarray(0, dtype=jnp.int32),
    )
    step_fn = make_train_step(sampler, cfg, noise_std)

    history: List[Dict[str, float]] = []
    t0 = time.time()
    for step in range(1, cfg.train_steps + 1):
        key, kstep = jax.random.split(key)
        params, state, loss, gnorm, lr = step_fn(params, state, kstep)
        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            row = {
                "step": int(step),
                "conditional_fm_loss": float(loss),
                "grad_norm_preclip": float(gnorm),
                "learning_rate": float(lr),
                "elapsed_seconds": float(time.time() - t0),
            }
            history.append(row)
            print(
                f"step {step:6d}/{cfg.train_steps} | CFM loss={row['conditional_fm_loss']:.6e} | "
                f"grad={row['grad_norm_preclip']:.3e} | lr={row['learning_rate']:.3e}",
                flush=True,
            )
    return params, history


# -----------------------------------------------------------------------------
# Validation utilities
# -----------------------------------------------------------------------------


def covariance_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xc = x - np.mean(x, axis=0, keepdims=True)
    return (xc.T @ xc) / max(x.shape[0] - 1, 1)


def rbf_mmd2_biased_np(x: np.ndarray, y: np.ndarray, bandwidth: float, block: int = 256) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h2 = max(float(bandwidth) ** 2, 1.0e-24)

    def kernel_mean(a, b):
        total = 0.0
        count = 0
        for i in range(0, len(a), block):
            aa = a[i : i + block]
            for j in range(0, len(b), block):
                bb = b[j : j + block]
                d2 = np.sum((aa[:, None, :] - bb[None, :, :]) ** 2, axis=-1)
                total += float(np.exp(-0.5 * d2 / h2).sum())
                count += int(d2.size)
        return total / max(count, 1)

    return max(kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y), 0.0)


def median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray, max_points: int = 2048) -> float:
    z = np.concatenate([np.asarray(x), np.asarray(y)], axis=0)
    if len(z) > max_points:
        rng = np.random.default_rng(9137)
        z = z[rng.choice(len(z), size=max_points, replace=False)]
    # Use upper-triangle pairwise distances.  For 2048 points this is manageable.
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    tri = d2[np.triu_indices(len(z), k=1)]
    positive = tri[tri > 0.0]
    if len(positive) == 0:
        return 1.0
    return float(math.sqrt(np.median(positive)))


def divergence_one(params, t: Array, x: Array) -> Array:
    jac = jax.jacfwd(lambda xx: velocity_mlp(params, t, xx))(x)
    return jnp.trace(jac)


def validate_regression(
    params,
    sampler: EndpointSampler,
    cfg: D5Config,
    noise_std: float,
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(cfg.seed + 100_003)
    split = "validation" if sampler.mode == "external" else "train"
    t, x, target = sample_cfm_batch(
        sampler, key, cfg.validation_size, cfg.bridge_schedule, noise_std, split=split
    )
    pred = jax.jit(velocity_mlp)(params, t, x)
    err = pred - target
    err2 = jnp.sum(err * err, axis=-1)
    tar2 = jnp.sum(target * target, axis=-1)
    mse = float(jnp.mean(err2))
    target_power = float(jnp.mean(tar2))
    normalized_rmse = math.sqrt(mse / max(target_power, 1e-30))

    t_np = np.asarray(t)
    err2_np = np.asarray(err2)
    tar2_np = np.asarray(tar2)
    bins = np.linspace(0.0, 1.0, cfg.validation_time_bins + 1)
    per_bin = []
    for i in range(cfg.validation_time_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (t_np >= lo) & ((t_np < hi) if i < cfg.validation_time_bins - 1 else (t_np <= hi))
        if not np.any(mask):
            continue
        per_bin.append({
            "t_lo": float(lo),
            "t_hi": float(hi),
            "conditional_fm_mse": float(np.mean(err2_np[mask])),
            "target_power": float(np.mean(tar2_np[mask])),
            "normalized_rmse": float(
                math.sqrt(float(np.mean(err2_np[mask])) / max(float(np.mean(tar2_np[mask])), 1e-30))
            ),
            "n": int(mask.sum()),
        })

    ndiv = min(int(cfg.divergence_validation_size), int(cfg.validation_size))
    if ndiv > 0:
        div = jax.jit(jax.vmap(lambda tt, xx: divergence_one(params, tt, xx)))(t[:ndiv], x[:ndiv])
        div_np = np.asarray(div, dtype=np.float64)
        div_mean = float(np.mean(div_np))
        div_rms = float(np.sqrt(np.mean(div_np ** 2)))
        div_max_abs = float(np.max(np.abs(div_np)))
    else:
        div_mean = float("nan")
        div_rms = float("nan")
        div_max_abs = float("nan")

    return {
        "size": int(cfg.validation_size),
        "split": split,
        "conditional_fm_mse": mse,
        "conditional_target_power": target_power,
        "normalized_target_rmse": float(normalized_rmse),
        "interpretation": (
            "Unlike D0 teacher regression, this loss has irreducible conditional variance under independent "
            "endpoint pairing; path-marginal rollout validation is the primary model check."
        ),
        "time_bins": per_bin,
        "divergence_subsample_size": int(ndiv),
        "divergence_mean": div_mean,
        "divergence_rms": div_rms,
        "divergence_max_abs": div_max_abs,
    }


def rk4_trajectory(params, x0: Array, steps: int) -> Array:
    """Fixed-step learned ODE trajectory, shape [steps+1, N, 2]."""
    dt = 1.0 / float(steps)

    def step_fn(x, i):
        t = i.astype(jnp.float64) * dt
        k1 = velocity_mlp(params, t, x)
        k2 = velocity_mlp(params, t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = velocity_mlp(params, t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = velocity_mlp(params, t + dt, x + dt * k3)
        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xn, xn

    idx = jnp.arange(steps, dtype=jnp.int32)
    _, xs = jax.lax.scan(step_fn, x0, idx)
    return jnp.concatenate([x0[None, ...], xs], axis=0)


def direct_interpolant_sample(
    sampler: EndpointSampler,
    key: Array,
    n: int,
    t: float,
    schedule: str,
    noise_std: float,
    split: str,
) -> Array:
    k0, k1, kz = jax.random.split(key, 3)
    x0 = sampler.sample_x0(k0, n, split=split)
    x1 = sampler.sample_x1(k1, n, split=split)
    z = jax.random.normal(kz, (n, 2), dtype=jnp.float64)
    tt = jnp.full((n,), float(t), dtype=jnp.float64)
    xt, _ = stochastic_interpolant(tt, x0, x1, z, schedule, noise_std)
    return xt


def validate_rollout(
    params,
    sampler: EndpointSampler,
    cfg: D5Config,
    noise_std: float,
    mmd_bandwidth: float,
) -> Dict[str, Any]:
    key = jax.random.PRNGKey(cfg.seed + 200_003)
    kx0, key = jax.random.split(key)
    split = "validation" if sampler.mode == "external" else "train"
    x0 = sampler.sample_x0(kx0, cfg.rollout_particles, split=split)

    traj = jax.jit(lambda z: rk4_trajectory(params, z, cfg.rollout_steps))(x0)
    traj_np = np.asarray(traj, dtype=np.float64)

    eval_idx = np.linspace(0, cfg.rollout_steps, cfg.rollout_eval_times).round().astype(int)
    eval_idx = np.unique(eval_idx)

    rows = []
    for j, idx in enumerate(eval_idx):
        t = idx / float(cfg.rollout_steps)
        key, kd = jax.random.split(key)
        direct = np.asarray(
            direct_interpolant_sample(
                sampler, kd, cfg.rollout_particles, t,
                cfg.bridge_schedule, noise_std, split,
            ),
            dtype=np.float64,
        )
        learned = traj_np[idx]

        mean_err = float(np.linalg.norm(np.mean(learned, axis=0) - np.mean(direct, axis=0)))
        cov_err = float(np.linalg.norm(covariance_np(learned) - covariance_np(direct), ord="fro"))
        mmd2 = rbf_mmd2_biased_np(learned, direct, mmd_bandwidth)
        rows.append({
            "t": float(t),
            "sample_mmd2_biased": float(mmd2),
            "mean_l2_error": mean_err,
            "covariance_fro_error": cov_err,
            "learned_mean": np.mean(learned, axis=0).tolist(),
            "direct_interpolant_mean": np.mean(direct, axis=0).tolist(),
        })

    interior = [r for r in rows if 0.0 < r["t"] < 1.0]
    endpoint = rows[-1]
    return {
        "particles": int(cfg.rollout_particles),
        "rk4_steps": int(cfg.rollout_steps),
        "mmd_bandwidth": float(mmd_bandwidth),
        "times": rows,
        "mean_interior_sample_mmd2_biased": float(
            np.mean([r["sample_mmd2_biased"] for r in interior]) if interior else 0.0
        ),
        "max_interior_sample_mmd2_biased": float(
            max([r["sample_mmd2_biased"] for r in interior], default=0.0)
        ),
        "endpoint_sample_mmd2_biased": float(endpoint["sample_mmd2_biased"]),
        "endpoint_mean_l2_error": float(endpoint["mean_l2_error"]),
        "endpoint_covariance_fro_error": float(endpoint["covariance_fro_error"]),
        "interpretation": (
            "Compares the learned ODE marginal against fresh direct samples from the declared endpoint-only "
            "stochastic interpolant; no Stage-B analytic intermediate marginal is used."
        ),
    }


# -----------------------------------------------------------------------------
# Checkpoint / JSON I/O -- same parameter convention as D.0
# -----------------------------------------------------------------------------


def jsonify(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return {k: jsonify(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, Mapping):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, jax.Array):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    return x


def save_checkpoint(path: Path, params, metadata: Dict[str, Any]) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for i, layer in enumerate(params):
        arrays[f"W{i}"] = np.asarray(layer["W"])
        arrays[f"b{i}"] = np.asarray(layer["b"])
    arrays["metadata_json"] = np.asarray(json.dumps(jsonify(metadata), sort_keys=True))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        n_layers = int(meta["network"]["parameter_layers"])
        params = tuple(
            {
                "W": jnp.asarray(data[f"W{i}"], dtype=jnp.float64),
                "b": jnp.asarray(data[f"b{i}"], dtype=jnp.float64),
            }
            for i in range(n_layers)
        )
    return params, meta


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_validation_summary(regression: Dict[str, Any], rollout: Dict[str, Any], noise_std: float) -> None:
    print("\n" + "=" * 78)
    print("Stage D.5 endpoint-trained flow-matching validation")
    print("=" * 78)
    print(f"bridge noise std: {noise_std:.6g}")
    print(
        f"held-out conditional-FM MSE={regression['conditional_fm_mse']:.6e}, "
        f"normalized target RMSE={regression['normalized_target_rmse']:.4f}"
    )
    print(
        f"rollout vs direct bridge: mean interior MMD^2={rollout['mean_interior_sample_mmd2_biased']:.6e}, "
        f"max interior MMD^2={rollout['max_interior_sample_mmd2_biased']:.6e}"
    )
    print(
        f"endpoint Q1: MMD^2={rollout['endpoint_sample_mmd2_biased']:.6e}, "
        f"mean err={rollout['endpoint_mean_l2_error']:.3e}, "
        f"cov err={rollout['endpoint_covariance_fro_error']:.3e}"
    )
    print(
        f"learned divergence diagnostic: RMS={regression['divergence_rms']:.3e}, "
        f"max abs={regression['divergence_max_abs']:.3e}"
    )
    print("No A_t, B_t, or analytic intermediate reference was used in training/validation.")
    print("=" * 78)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None, help="Stage-B backend; used only for endpoint parameters/metadata")
    p.add_argument("--x0-samples", type=str, default=None, help="External Q0 samples (.npy/.npz), shape [N,2]")
    p.add_argument("--x1-samples", type=str, default=None, help="External Q1 samples (.npy/.npz), shape [N,2]")
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--hidden-width", type=int, default=None)
    p.add_argument("--hidden-layers", type=int, default=None)
    p.add_argument("--bridge-schedule", choices=("linear", "trig"), default=None)
    p.add_argument("--bridge-noise-multiplier", type=float, default=None)
    p.add_argument("--bridge-noise-std", type=float, default=None, help="Absolute bridge noise std; overrides multiplier")
    p.add_argument("--validation-fraction", type=float, default=None)
    p.add_argument("--validation-size", type=int, default=None)
    p.add_argument("--divergence-validation-size", type=int, default=None)
    p.add_argument("--rollout-particles", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--rollout-eval-times", type=int, default=None)
    p.add_argument("--mmd-bandwidth", type=float, default=None)
    p.add_argument("--output-prefix", type=str, default="stage_d5_endpoint_flow_matching_reference")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = preset_d5_config(args.preset)
    overrides: Dict[str, Any] = {}
    for attr in (
        "seed", "train_steps", "batch_size", "learning_rate", "hidden_width", "hidden_layers",
        "bridge_schedule", "bridge_noise_multiplier", "validation_fraction", "validation_size",
        "divergence_validation_size",
        "rollout_particles", "rollout_steps", "rollout_eval_times", "mmd_bandwidth",
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides[attr] = value
    cfg = dataclasses.replace(cfg, **overrides)

    # Stage-B is optional when external endpoints are supplied.  It is never used
    # to construct intermediate states or training velocities in D.5.
    backend_path = Path(args.backend) if args.backend else autodetect_backend()
    backend = None
    stage_b_cfg = None
    if backend_path is not None:
        backend = load_backend(backend_path)
        stage_b_cfg = backend.preset_config("reference")

    x0_external = None
    x1_external = None
    if args.x0_samples is not None or args.x1_samples is not None:
        if args.x0_samples is None or args.x1_samples is None:
            raise ValueError("Pass both --x0-samples and --x1-samples.")
        x0_external = _load_array(Path(args.x0_samples), ("x0", "samples", "x"))
        x1_external = _load_array(Path(args.x1_samples), ("x1", "samples", "x"))

    r = float(stage_b_cfg.r) if stage_b_cfg is not None else None
    sigma = float(stage_b_cfg.sigma) if stage_b_cfg is not None else None
    sampler = EndpointSampler(
        r=r,
        sigma=sigma,
        x0_external=x0_external,
        x1_external=x1_external,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
    )

    if args.bridge_noise_std is not None:
        noise_std = float(args.bridge_noise_std)
    elif sigma is not None:
        noise_std = float(cfg.bridge_noise_multiplier * sigma)
    else:
        noise_std = 0.0
        print(
            "WARNING: no Stage-B sigma is available and --bridge-noise-std was not provided; "
            "using a deterministic endpoint interpolant (bridge noise std = 0).",
            flush=True,
        )
    if noise_std < 0.0:
        raise ValueError("bridge noise std must be nonnegative")

    # MMD bandwidth is chosen from explicit CLI/config, then Stage-B, then an
    # endpoint-data median heuristic.  This affects diagnostics only.
    if cfg.mmd_bandwidth is not None:
        mmd_bw = float(cfg.mmd_bandwidth)
    elif stage_b_cfg is not None:
        mmd_bw = float(stage_b_cfg.mmd_bandwidth)
    else:
        k0, k1 = jax.random.split(jax.random.PRNGKey(cfg.seed + 314159))
        split = "validation" if sampler.mode == "external" else "train"
        bx0 = np.asarray(sampler.sample_x0(k0, min(2048, cfg.rollout_particles), split=split))
        bx1 = np.asarray(sampler.sample_x1(k1, min(2048, cfg.rollout_particles), split=split))
        mmd_bw = median_heuristic_bandwidth(bx0, bx1)
    if not (mmd_bw > 0.0 and math.isfinite(mmd_bw)):
        raise ValueError(f"Invalid MMD bandwidth {mmd_bw}")

    output_prefix = Path(args.output_prefix)
    checkpoint_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    if args.eval_only:
        cp = Path(args.checkpoint) if args.checkpoint else checkpoint_path
        params, loaded_meta = load_checkpoint(cp)
        history: List[Dict[str, float]] = []
        print(f"Loaded checkpoint: {cp}")
    else:
        params, history = train_model(sampler, cfg, noise_std)
        loaded_meta = None

    regression = validate_regression(params, sampler, cfg, noise_std)
    rollout = validate_rollout(params, sampler, cfg, noise_std, mmd_bw)
    print_validation_summary(regression, rollout, noise_std)

    network_meta = {
        "input_dim": 7,
        "output_dim": 2,
        "hidden_width": int(cfg.hidden_width),
        "hidden_layers": int(cfg.hidden_layers),
        "parameter_layers": int(len(params)),
        "time_features": ["t", "sin(pi t)", "cos(pi t)", "sin(2 pi t)", "cos(2 pi t)"],
        "d0_checkpoint_parameter_compatible": True,
    }
    bridge_meta = {
        "construction": "endpoint stochastic interpolant / conditional flow matching",
        "endpoint_coupling": "independent product coupling",
        "schedule": cfg.bridge_schedule,
        "gamma": "noise_std * sin(pi t)",
        "noise_std": float(noise_std),
        "uses_analytic_A_t": False,
        "uses_analytic_B_t": False,
        "uses_analytic_velocity_teacher": False,
    }
    physical_meta: Dict[str, Any] = {
        "reference_learning_mode": "endpoint_only",
        "mmd_bandwidth": float(mmd_bw),
    }
    if stage_b_cfg is not None:
        # kappa is saved only as physical-system provenance/compatibility metadata.
        # It is never read by the D.5 training objective or bridge sampler.
        physical_meta.update({
            "r": float(stage_b_cfg.r),
            "sigma": float(stage_b_cfg.sigma),
            "kappa": float(stage_b_cfg.kappa),
            "kappa_used_in_training": False,
        })

    result = {
        "stage": "D.5",
        "purpose": (
            "Learn a reference flow from endpoint samples using conditional flow matching, without the analytic "
            "Stage-B intermediate path or velocity teacher."
        ),
        "backend_path": None if backend_path is None else str(Path(backend_path).resolve()),
        "config": jsonify(cfg),
        "endpoint_data": sampler.metadata(),
        "bridge": bridge_meta,
        "network": network_meta,
        "physical_system": physical_meta,
        "training_history": history,
        "heldout_conditional_fm_validation": regression,
        "teacher_free_rollout_validation": rollout,
        "evaluation_only": bool(args.eval_only),
        "loaded_checkpoint_metadata": loaded_meta,
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(json_path, result)

    if not args.eval_only:
        checkpoint_meta = {
            "stage": "D.5",
            "config": jsonify(cfg),
            "endpoint_data": sampler.metadata(),
            "bridge": bridge_meta,
            "network": network_meta,
            "physical_system": physical_meta,
        }
        save_checkpoint(checkpoint_path, params, checkpoint_meta)
        print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved diagnostics: {json_path}")


if __name__ == "__main__":
    main()
