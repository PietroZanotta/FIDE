#!/usr/bin/env python3
"""
Stage D.0: learned reference stochastic interpolant by flow matching.

Purpose
-------
This is the first learned-reference experiment.  It deliberately changes only one
object relative to Stage B/C: the analytic reference velocity is replaced by a
neural velocity trained with the standard flow-matching regression objective on
THE SAME analytic probability path used by Stage B.

The scientific population P_t^alpha, sensor family, moment-fiber projection, and
Poisson correction are not used or changed here.  Stage D.0 is only a controlled
reference-learning unit test.

Teacher path inherited from Stage B
-----------------------------------
Let X_0 be the symmetric two-lobe mixture G_0 used by Stage B and

    X_t = A_t X_0,

where

    A_t = R(pi t / 2) diag(exp(s_t), exp(-s_t)),
    s_t = kappa sin(pi t).

The exact reference velocity is

    u_*(t, x) = B_t x,      B_t = dot(A_t) A_t^{-1}.

We train a generic time-conditioned MLP u_theta(t,x) by

    L_FM(theta) = E ||u_theta(t, X_t) - u_*(t, X_t)||^2,

with t ~ Uniform[0,1].  Because the teacher path is deterministic and invertible,
this is a zero-Bayes-error flow-matching benchmark: approximation error comes
from finite optimization/model capacity, not from an ambiguous regression target.

Validation
----------
The script reports, on independent held-out samples:
  * velocity MSE and relative L2 error,
  * time-binned relative velocity error,
  * divergence RMSE versus the exact divergence (zero for this volume-preserving path),
  * learned-ODE rollout error against the exact A_t X_0 path,
  * sample Gaussian-kernel MMD^2, mean error, and covariance error over held-out times,
  * an analytic-velocity RK4 rollout floor using the identical ODE integrator.

The divergence diagnostic is included now because Stage D.1 can use the learned
velocity as a continuous normalizing flow to construct a density consistent with
that same learned velocity.

Outputs
-------
A single script is the only Stage-D.0 source file.  A run writes:
  <output-prefix>.npz   neural parameters + metadata
  <output-prefix>.json  training/validation diagnostics

No Optax/Flax dependency is required; Adam is implemented with JAX pytrees.

Examples
--------
Quick smoke test:
    python stage_d0_flow_matching_reference.py --preset quick

Main pilot:
    python stage_d0_flow_matching_reference.py --preset reference \
        --output-prefix stage_d0_flow_matching_reference

Evaluate a saved checkpoint without retraining:
    python stage_d0_flow_matching_reference.py --eval-only \
        --checkpoint stage_d0_flow_matching_reference.npz
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
# Dynamic loading of the validated Stage-B configuration
# -----------------------------------------------------------------------------


def load_backend(path: Path):
    """Load the Stage-B module only to inherit its physical configuration."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Stage B backend not found: {path}")
    spec = importlib.util.spec_from_file_location("stage_b2_backend_d0", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend module from {path}")
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
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# D.0 configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class D0Config:
    preset: str = "quick"
    seed: int = 20260812

    # MLP: input = (x1, x2, t, sin/cos pi t, sin/cos 2 pi t), output R^2.
    hidden_width: int = 64
    hidden_layers: int = 3

    # Optimization.
    train_steps: int = 600
    batch_size: int = 1024
    learning_rate: float = 2.0e-3
    min_learning_rate_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    grad_clip_norm: float = 10.0
    log_every: int = 100

    # Independent velocity/divergence validation.
    validation_size: int = 8192
    validation_time_bins: int = 10

    # ODE rollout validation.
    rollout_particles: int = 512
    rollout_steps: int = 100
    rollout_eval_times: int = 11

    # Uses the same MMD bandwidth as Stage B by default.
    mmd_bandwidth: float | None = None


def preset_d0_config(name: str) -> D0Config:
    if name == "quick":
        return D0Config()
    if name == "reference":
        return D0Config(
            preset="reference",
            hidden_width=128,
            hidden_layers=3,
            train_steps=6000,
            batch_size=2048,
            learning_rate=1.5e-3,
            log_every=500,
            validation_size=32768,
            validation_time_bins=20,
            rollout_particles=1024,
            rollout_steps=200,
            rollout_eval_times=21,
        )
    if name == "confirm":
        return D0Config(
            preset="confirm",
            hidden_width=128,
            hidden_layers=4,
            train_steps=12000,
            batch_size=4096,
            learning_rate=1.0e-3,
            log_every=1000,
            validation_size=65536,
            validation_time_bins=25,
            rollout_particles=2048,
            rollout_steps=400,
            rollout_eval_times=21,
        )
    raise ValueError(f"Unknown Stage D.0 preset {name!r}")


# -----------------------------------------------------------------------------
# Exact Stage-B teacher geometry
# -----------------------------------------------------------------------------


class AnalyticReferenceTeacher:
    """
    Lightweight copy of the Stage-B analytic reference geometry.

    The physical scalar parameters are read from the validated Stage-B Config;
    formulas match StageB.A_matrix and StageB.B_matrix exactly.  Keeping this
    class lightweight avoids constructing the Stage-B spatial inverse-problem grid
    during pure flow-matching training.
    """

    def __init__(self, stage_b_cfg):
        self.r = float(stage_b_cfg.r)
        self.sigma = float(stage_b_cfg.sigma)
        self.kappa = float(stage_b_cfg.kappa)
        self.mmd_bandwidth = float(stage_b_cfg.mmd_bandwidth)

    @staticmethod
    def rotation(theta: Array) -> Array:
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.stack(
            [jnp.stack([c, -s], axis=-1), jnp.stack([s, c], axis=-1)],
            axis=-2,
        )

    def A_matrix(self, t: Array) -> Array:
        omega = 0.5 * jnp.pi * t
        s = self.kappa * jnp.sin(jnp.pi * t)
        R = self.rotation(omega)
        # R @ diag(exp(s), exp(-s)); broadcasting works for scalar or batched t.
        scales = jnp.stack([jnp.exp(s), jnp.exp(-s)], axis=-1)
        return R * scales[..., None, :]

    def B_matrix(self, t: Array) -> Array:
        omega = 0.5 * jnp.pi * t
        R = self.rotation(omega)
        J = jnp.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)
        sdot = self.kappa * jnp.pi * jnp.cos(jnp.pi * t)
        S = jnp.zeros(R.shape, dtype=jnp.float64)
        S = S.at[..., 0, 0].set(sdot)
        S = S.at[..., 1, 1].set(-sdot)
        return 0.5 * jnp.pi * J + R @ S @ jnp.swapaxes(R, -1, -2)

    def sample_x0(self, key: Array, n: int) -> Array:
        key_sign, key_noise = jax.random.split(key)
        signs = jnp.where(jax.random.bernoulli(key_sign, 0.5, (n,)), 1.0, -1.0)
        means = jnp.stack([self.r * signs, jnp.zeros_like(signs)], axis=-1)
        return means + self.sigma * jax.random.normal(key_noise, (n, 2), dtype=jnp.float64)

    def pushforward(self, t: Array, x0: Array) -> Array:
        A = self.A_matrix(t)
        if jnp.ndim(t) == 0:
            return x0 @ A.T
        return jnp.einsum("nij,nj->ni", A, x0)

    def velocity(self, t: Array, x: Array) -> Array:
        B = self.B_matrix(t)
        if jnp.ndim(t) == 0:
            return x @ B.T
        return jnp.einsum("nij,nj->ni", B, x)


# -----------------------------------------------------------------------------
# Neural velocity model
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
    tf = time_features(t)
    return jnp.concatenate([x, tf], axis=-1)


def init_mlp(key: Array, input_dim: int, hidden_width: int, hidden_layers: int, output_dim: int = 2):
    dims = [input_dim] + [hidden_width] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        # Glorot normal initialization.
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
# Flow-matching training
# -----------------------------------------------------------------------------


def sample_training_batch(teacher: AnalyticReferenceTeacher, key: Array, n: int):
    kt, kx = jax.random.split(key)
    t = jax.random.uniform(kt, (n,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    x0 = teacher.sample_x0(kx, n)
    xt = teacher.pushforward(t, x0)
    ut = teacher.velocity(t, xt)
    return t, xt, ut


def make_train_step(teacher: AnalyticReferenceTeacher, cfg: D0Config):
    beta1 = float(cfg.adam_beta1)
    beta2 = float(cfg.adam_beta2)
    eps = float(cfg.adam_eps)

    def loss_fn(params, t, x, target):
        pred = velocity_mlp(params, t, x)
        err = pred - target
        return jnp.mean(jnp.sum(err * err, axis=-1))

    @jax.jit
    def train_step(params, state: AdamState, key: Array):
        t, x, target = sample_training_batch(teacher, key, cfg.batch_size)
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
            params,
            mhat,
            vhat,
        )
        return params, AdamState(m=m, v=v, step=step), loss, gnorm, lr

    return train_step


def train_model(teacher: AnalyticReferenceTeacher, cfg: D0Config):
    key = jax.random.PRNGKey(cfg.seed)
    kinit, key = jax.random.split(key)
    params = init_mlp(kinit, input_dim=7, hidden_width=cfg.hidden_width, hidden_layers=cfg.hidden_layers)
    state = AdamState(m=tree_zeros_like(params), v=tree_zeros_like(params), step=jnp.asarray(0, dtype=jnp.int32))
    step_fn = make_train_step(teacher, cfg)

    history: List[Dict[str, float]] = []
    t0 = time.time()
    for step in range(1, cfg.train_steps + 1):
        key, kstep = jax.random.split(key)
        params, state, loss, gnorm, lr = step_fn(params, state, kstep)
        if step == 1 or step % cfg.log_every == 0 or step == cfg.train_steps:
            row = {
                "step": int(step),
                "loss": float(loss),
                "grad_norm_preclip": float(gnorm),
                "learning_rate": float(lr),
                "elapsed_seconds": float(time.time() - t0),
            }
            history.append(row)
            print(
                f"step {step:6d}/{cfg.train_steps} | "
                f"FM loss={row['loss']:.6e} | grad={row['grad_norm_preclip']:.3e} | "
                f"lr={row['learning_rate']:.3e}",
                flush=True,
            )
    return params, history


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def covariance_np(x: np.ndarray) -> np.ndarray:
    xc = x - np.mean(x, axis=0, keepdims=True)
    return (xc.T @ xc) / max(x.shape[0] - 1, 1)


def rbf_mmd2_biased_np(x: np.ndarray, y: np.ndarray, bandwidth: float, block: int = 256) -> float:
    """Biased empirical MMD^2 with bounded memory."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h2 = float(bandwidth) ** 2

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


def divergence_one(params, t: Array, x: Array) -> Array:
    jac = jax.jacfwd(lambda xx: velocity_mlp(params, t, xx))(x)
    return jnp.trace(jac)


def validate_velocity(params, teacher: AnalyticReferenceTeacher, cfg: D0Config) -> Dict[str, Any]:
    key = jax.random.PRNGKey(cfg.seed + 100_003)
    t, x, target = sample_training_batch(teacher, key, cfg.validation_size)

    pred = jax.jit(velocity_mlp)(params, t, x)
    err = pred - target
    mse = float(jnp.mean(jnp.sum(err * err, axis=-1)))
    target_power = float(jnp.mean(jnp.sum(target * target, axis=-1)))
    rel_l2 = math.sqrt(mse / max(target_power, 1e-30))

    # Continuous-time bin diagnostics; all points come from an untouched bank.
    t_np = np.asarray(t)
    err2_np = np.asarray(jnp.sum(err * err, axis=-1))
    tar2_np = np.asarray(jnp.sum(target * target, axis=-1))
    bins = np.linspace(0.0, 1.0, cfg.validation_time_bins + 1)
    per_bin = []
    for i in range(cfg.validation_time_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (t_np >= lo) & ((t_np < hi) if i < cfg.validation_time_bins - 1 else (t_np <= hi))
        if not np.any(mask):
            continue
        ratio = math.sqrt(float(np.mean(err2_np[mask])) / max(float(np.mean(tar2_np[mask])), 1e-30))
        per_bin.append({"t_lo": float(lo), "t_hi": float(hi), "relative_l2": float(ratio), "n": int(mask.sum())})

    # Divergence is diagnostically important for the CNF density planned in D.1.
    # Subsample because jacfwd per point is more expensive than velocity evaluation.
    div_n = min(2048, cfg.validation_size)
    div_pred = jax.jit(jax.vmap(lambda tt, xx: divergence_one(params, tt, xx)))(t[:div_n], x[:div_n])
    # The exact B_t is traceless for this rotation + volume-preserving anisotropy.
    div_true = jax.vmap(lambda tt: jnp.trace(teacher.B_matrix(tt)))(t[:div_n])
    div_rmse = float(jnp.sqrt(jnp.mean((div_pred - div_true) ** 2)))
    div_max_abs = float(jnp.max(jnp.abs(div_pred - div_true)))

    return {
        "size": int(cfg.validation_size),
        "velocity_mse": mse,
        "velocity_target_power": target_power,
        "velocity_relative_l2": float(rel_l2),
        "max_time_bin_relative_l2": float(max(r["relative_l2"] for r in per_bin)),
        "time_bins": per_bin,
        "divergence_subsample_size": int(div_n),
        "divergence_rmse": div_rmse,
        "divergence_max_abs_error": div_max_abs,
    }


def rk4_trajectory(params, teacher: AnalyticReferenceTeacher, x0: Array, steps: int, learned: bool) -> Array:
    """Return states at all fixed RK4 nodes, shape [steps+1, n, 2]."""
    dt = 1.0 / float(steps)

    if learned:
        def vel(t, x):
            return velocity_mlp(params, t, x)
    else:
        def vel(t, x):
            return teacher.velocity(t, x)

    def step_fn(x, i):
        t = i.astype(jnp.float64) * dt
        k1 = vel(t, x)
        k2 = vel(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = vel(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = vel(t + dt, x + dt * k3)
        xn = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xn, xn

    idx = jnp.arange(steps, dtype=jnp.int32)
    _, xs = jax.lax.scan(step_fn, x0, idx)
    return jnp.concatenate([x0[None, ...], xs], axis=0)


def validate_rollout(params, teacher: AnalyticReferenceTeacher, cfg: D0Config) -> Dict[str, Any]:
    key = jax.random.PRNGKey(cfg.seed + 200_003)
    x0 = teacher.sample_x0(key, cfg.rollout_particles)

    learned_traj = jax.jit(lambda z: rk4_trajectory(params, teacher, z, cfg.rollout_steps, True))(x0)
    oracle_traj = jax.jit(lambda z: rk4_trajectory(params, teacher, z, cfg.rollout_steps, False))(x0)

    eval_idx = np.linspace(0, cfg.rollout_steps, cfg.rollout_eval_times).round().astype(int)
    eval_idx = np.unique(eval_idx)
    bw = teacher.mmd_bandwidth if cfg.mmd_bandwidth is None else float(cfg.mmd_bandwidth)

    x0_np = np.asarray(x0)
    learned_np = np.asarray(learned_traj)
    oracle_np = np.asarray(oracle_traj)

    rows = []
    for idx in eval_idx:
        t = idx / float(cfg.rollout_steps)
        exact = np.asarray(teacher.pushforward(jnp.asarray(t, dtype=jnp.float64), x0))
        learned = learned_np[idx]
        oracle = oracle_np[idx]

        lerr = learned - exact
        oerr = oracle - exact
        exact_power = max(float(np.mean(np.sum(exact * exact, axis=-1))), 1e-30)
        learned_rmse = math.sqrt(float(np.mean(np.sum(lerr * lerr, axis=-1))))
        oracle_rmse = math.sqrt(float(np.mean(np.sum(oerr * oerr, axis=-1))))

        mean_err = float(np.linalg.norm(np.mean(learned, axis=0) - np.mean(exact, axis=0)))
        cov_err = float(np.linalg.norm(covariance_np(learned) - covariance_np(exact), ord="fro"))
        mmd2 = rbf_mmd2_biased_np(learned, exact, bw)

        rows.append({
            "t": float(t),
            "learned_paired_rmse": float(learned_rmse),
            "learned_relative_state_l2": float(learned_rmse / math.sqrt(exact_power)),
            "oracle_rk4_paired_rmse": float(oracle_rmse),
            "sample_mmd2_biased": float(mmd2),
            "mean_l2_error": mean_err,
            "covariance_fro_error": cov_err,
        })

    interior = [r for r in rows if r["t"] > 0.0 and r["t"] < 1.0]
    endpoint = rows[-1]
    return {
        "particles": int(cfg.rollout_particles),
        "rk4_steps": int(cfg.rollout_steps),
        "mmd_bandwidth": float(bw),
        "times": rows,
        "mean_interior_sample_mmd2_biased": float(np.mean([r["sample_mmd2_biased"] for r in interior])) if interior else 0.0,
        "max_interior_sample_mmd2_biased": float(max([r["sample_mmd2_biased"] for r in interior], default=0.0)),
        "max_learned_relative_state_l2": float(max(r["learned_relative_state_l2"] for r in rows)),
        "max_oracle_rk4_paired_rmse": float(max(r["oracle_rk4_paired_rmse"] for r in rows)),
        "endpoint_sample_mmd2_biased": float(endpoint["sample_mmd2_biased"]),
        "endpoint_relative_state_l2": float(endpoint["learned_relative_state_l2"]),
    }


# -----------------------------------------------------------------------------
# Checkpoint / JSON I/O
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
# Self-checks and reporting
# -----------------------------------------------------------------------------


def teacher_consistency_checks(teacher: AnalyticReferenceTeacher) -> Dict[str, float]:
    """Verify B = dot(A) A^{-1} and det(A)=1 numerically before training."""
    ts = jnp.linspace(0.0, 1.0, 17, dtype=jnp.float64)

    def one(t):
        A = teacher.A_matrix(t)
        dA = jax.jacfwd(teacher.A_matrix)(t)
        B_from_A = dA @ jnp.linalg.inv(A)
        B = teacher.B_matrix(t)
        return jnp.linalg.norm(B - B_from_A), jnp.abs(jnp.linalg.det(A) - 1.0), jnp.abs(jnp.trace(B))

    vals = jax.vmap(one)(ts)
    return {
        "max_B_minus_dA_Ainv_fro": float(jnp.max(vals[0])),
        "max_abs_det_A_minus_1": float(jnp.max(vals[1])),
        "max_abs_trace_B": float(jnp.max(vals[2])),
    }


def print_validation_summary(velocity: Dict[str, Any], rollout: Dict[str, Any], checks: Dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("Stage D.0 validation summary")
    print("=" * 78)
    print(
        "teacher geometry: "
        f"max ||B-dA A^-1||_F={checks['max_B_minus_dA_Ainv_fro']:.3e}, "
        f"max |det(A)-1|={checks['max_abs_det_A_minus_1']:.3e}, "
        f"max |tr(B)|={checks['max_abs_trace_B']:.3e}"
    )
    print(
        f"held-out velocity: rel L2={velocity['velocity_relative_l2']:.4e}, "
        f"worst time-bin rel L2={velocity['max_time_bin_relative_l2']:.4e}"
    )
    print(
        f"held-out divergence: RMSE={velocity['divergence_rmse']:.4e}, "
        f"max abs={velocity['divergence_max_abs_error']:.4e}"
    )
    print(
        f"rollout: mean interior MMD^2={rollout['mean_interior_sample_mmd2_biased']:.4e}, "
        f"max interior MMD^2={rollout['max_interior_sample_mmd2_biased']:.4e}, "
        f"endpoint MMD^2={rollout['endpoint_sample_mmd2_biased']:.4e}"
    )
    print(
        f"rollout: max learned relative state L2={rollout['max_learned_relative_state_l2']:.4e}, "
        f"analytic RK4 floor (max paired RMSE)={rollout['max_oracle_rk4_paired_rmse']:.4e}"
    )
    print("=" * 78)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=str, default=None, help="Path to Stage B.2 backend")
    p.add_argument("--preset", choices=("quick", "reference", "confirm"), default="quick")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--hidden-width", type=int, default=None)
    p.add_argument("--hidden-layers", type=int, default=None)
    p.add_argument("--rollout-particles", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--validation-size", type=int, default=None)
    p.add_argument("--output-prefix", type=str, default="stage_d0_flow_matching_reference")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint used by --eval-only")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    backend_path = Path(args.backend) if args.backend else autodetect_backend()
    if backend_path is None:
        raise FileNotFoundError(
            "Could not find Stage-B backend. Pass --backend /path/to/stage_b2_transport_conditioned_design.py"
        )
    backend = load_backend(backend_path)
    stage_b_cfg = backend.preset_config("reference")
    teacher = AnalyticReferenceTeacher(stage_b_cfg)

    cfg = preset_d0_config(args.preset)
    overrides: Dict[str, Any] = {}
    for attr in (
        "seed", "train_steps", "batch_size", "learning_rate", "hidden_width",
        "hidden_layers", "rollout_particles", "rollout_steps", "validation_size",
    ):
        val = getattr(args, attr)
        if val is not None:
            overrides[attr] = val
    cfg = dataclasses.replace(cfg, **overrides)

    consistency = teacher_consistency_checks(teacher)
    if consistency["max_B_minus_dA_Ainv_fro"] > 1e-10:
        raise RuntimeError(f"Teacher B(t) consistency check failed: {consistency}")
    if consistency["max_abs_det_A_minus_1"] > 1e-10:
        raise RuntimeError(f"Teacher volume-preservation check failed: {consistency}")

    output_prefix = Path(args.output_prefix)
    checkpoint_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    if args.eval_only:
        cp = Path(args.checkpoint) if args.checkpoint else checkpoint_path
        params, loaded_meta = load_checkpoint(cp)
        history: List[Dict[str, float]] = []
        print(f"Loaded checkpoint: {cp}")
    else:
        params, history = train_model(teacher, cfg)
        loaded_meta = None

    velocity = validate_velocity(params, teacher, cfg)
    rollout = validate_rollout(params, teacher, cfg)
    print_validation_summary(velocity, rollout, consistency)

    network_meta = {
        "input_dim": 7,
        "output_dim": 2,
        "hidden_width": int(cfg.hidden_width),
        "hidden_layers": int(cfg.hidden_layers),
        "parameter_layers": int(len(params)),
        "time_features": ["t", "sin(pi t)", "cos(pi t)", "sin(2 pi t)", "cos(2 pi t)"],
    }
    physical_meta = {
        "r": teacher.r,
        "sigma": teacher.sigma,
        "kappa": teacher.kappa,
        "mmd_bandwidth": teacher.mmd_bandwidth,
        "teacher_path": "X_t = A_t X_0 from Stage B",
    }

    result = {
        "stage": "D.0",
        "purpose": "Learn the existing analytic Stage-B reference path by flow matching before changing MFSI.",
        "backend_path": str(Path(backend_path).resolve()),
        "config": jsonify(cfg),
        "network": network_meta,
        "physical_system": physical_meta,
        "teacher_consistency": consistency,
        "training_history": history,
        "heldout_velocity_validation": velocity,
        "rollout_validation": rollout,
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
            "stage": "D.0",
            "config": jsonify(cfg),
            "network": network_meta,
            "physical_system": physical_meta,
            "teacher_consistency": consistency,
        }
        save_checkpoint(checkpoint_path, params, checkpoint_meta)
        print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved diagnostics: {json_path}")


if __name__ == "__main__":
    main()
