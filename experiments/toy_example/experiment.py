from __future__ import annotations

import csv
from functools import partial
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain import ToyEndpointSource, ToyPopulation
from mfsi.law_objectives import TrialBank
from mfsi.selection import optimize_population_and_law

from mfsi.design import (
    OptimizerConfig,
    optimize_multistart_candidates,
    projective_separation_violation,
    random_projective_starts,
)
from mfsi.feasibility import (
    common_beta_support_polytope_2d,
    project_metric_polytope_2d,
    unit_directions_2d,
)
from mfsi.exact_feasibility import (
    ExactBetaPolytope,
    ExactFeasibilityError,
    build_common_quadratic_beta_polytope_2d,
    hull_equations_2d,
    max_hull_violation,
    project_metric_polytope_exact_2d,
    robust_empirical_tilt_exact,
)
from mfsi.grid import CartesianGrid2D
from mfsi.io import write_json
from mfsi.measurements import GaussianSensor2D
from mfsi.metrics import gaussian_mmd2_grid_mass, gaussian_mmd_kernel
from mfsi.moments import (
    QuadraticBridgeConfig,
    evaluate_quadratic_bridge,
    fit_quadratic_bridge_gls,
)
from mfsi.particles import ParticleMFSIConfig, particle_mfsi_state
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.raster import RasterConfig, rasterize_projected_particles
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint, velocity_mlp

try:
    from mfsi.poisson import PoissonConfig, solve_weighted_poisson, weighted_laplacian
except ImportError:  # compatibility with the earlier name used in the cleanup
    from mfsi.poisson import (
        WeightedPoissonConfig as PoissonConfig,
        solve_weighted_poisson,
        weighted_laplacian,
    )


Array = jax.Array


@partial(
    jax.jit,
    static_argnames=("dx", "operator_floor_rel", "gauge_strength"),
)
def _batched_poisson_diagnostics(
    psi: Array,
    q: Array,
    h: Array,
    *,
    dx: float,
    operator_floor_rel: float,
    gauge_strength: float,
) -> tuple[Array, Array]:
    """Physical actions and true residuals for an explicit Poisson batch."""
    q_floor = operator_floor_rel * jnp.max(q, axis=(-2, -1), keepdims=True)
    q_operator = q + q_floor
    rhs = -(q * h)
    flat_q = q.reshape((q.shape[0], -1))
    gauge = flat_q / jnp.maximum(
        jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300
    )
    gauge = gauge.reshape(q.shape)

    operator_psi = jax.vmap(lambda p, qo: weighted_laplacian(p, qo, dx))(
        psi, q_operator
    )
    gauge_projection = jnp.sum(gauge * psi, axis=(-2, -1), keepdims=True)
    residual = operator_psi + gauge_strength * gauge * gauge_projection - rhs
    relative_residual = jnp.linalg.norm(
        residual.reshape((residual.shape[0], -1)), axis=-1
    ) / jnp.maximum(
        jnp.linalg.norm(rhs.reshape((rhs.shape[0], -1)), axis=-1), 1.0e-14
    )

    physical_operator_psi = jax.vmap(
        lambda p, q_one: weighted_laplacian(p, q_one, dx)
    )(psi, q)
    actions = (dx * dx) * jnp.sum(
        psi * physical_operator_psi, axis=(-2, -1)
    )
    return actions, relative_residual


# -----------------------------------------------------------------------------
# Small data containers
# -----------------------------------------------------------------------------


class Reconstruction(NamedTuple):
    c: Array
    c_dot: Array
    beta: Array
    beta_unconstrained: Array
    projection_distance: Array
    max_unconstrained_violation: Array
    endpoint_feasibility_violation: Array


class TrialMetrics(NamedTuple):
    law_risk: Array
    tangent_action: Array
    full_action: Array
    max_calibration_residual: Array
    min_ess_fraction: Array
    max_projection_distance: Array
    max_poisson_relative_residual: Array
    valid: Array


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _trap_weights(n: int) -> Array:
    w = np.ones(int(n), dtype=np.float64)
    if n > 1:
        w[0] = 0.5
        w[-1] = 0.5
        w /= (n - 1)
    return jnp.asarray(w, dtype=jnp.float64)


def _trap_weights_at_nodes(times: Array) -> Array:
    """Trapezoid integration weights for an arbitrary increasing subset of nodes."""
    t = jnp.asarray(times, dtype=jnp.float64)
    n = int(t.shape[0])
    if n == 1:
        return jnp.ones((1,), dtype=jnp.float64)
    w = jnp.zeros((n,), dtype=jnp.float64)
    w = w.at[0].set(0.5 * (t[1] - t[0]))
    w = w.at[-1].set(0.5 * (t[-1] - t[-2]))
    if n > 2:
        w = w.at[1:-1].set(0.5 * (t[2:] - t[:-2]))
    return w


def _nested_acquisition_indices(time_n: int, k: int) -> np.ndarray:
    if k < 2 or k > time_n:
        raise ValueError(f"acquisition_k must satisfy 2 <= K <= time_n; got K={k}, T={time_n}")
    raw = np.rint(np.linspace(0, time_n - 1, k)).astype(int)
    raw[0] = 0
    raw[-1] = time_n - 1
    raw = np.unique(raw)
    if len(raw) != k:
        # Deterministic fallback that preserves both endpoints.
        interior = np.arange(1, time_n - 1)
        want = k - 2
        chosen = interior[np.rint(np.linspace(0, len(interior) - 1, want)).astype(int)] if want else np.array([], dtype=int)
        raw = np.concatenate([[0], chosen, [time_n - 1]])
    if len(np.unique(raw)) != k:
        raise ValueError(f"Could not construct {k} unique acquisition nodes from time_n={time_n}")
    return raw.astype(np.int32)


def _heldout_indices(time_n: int, acq_idx: np.ndarray) -> np.ndarray:
    acq = set(map(int, acq_idx.tolist()))
    held = [i for i in range(1, time_n - 1) if i not in acq]
    if not held:
        # For very small smoke configurations, use interior acquisition nodes as a
        # wiring check rather than creating an empty law metric.
        held = [i for i in range(1, time_n - 1)]
    return np.asarray(held, dtype=np.int32)


def _mean_se(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
    return {"mean": float(np.mean(x)), "se": se, "n": int(x.size)}


def _paired_reduction(num: list[float], den: list[float]) -> dict[str, float | int]:
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-14)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {
            "ratio_of_means_reduction": float("nan"),
            "mean_paired_reduction": float("nan"),
            "se_paired_reduction": float("nan"),
            "n": 0,
        }
    paired = 1.0 - a / b
    return {
        "ratio_of_means_reduction": float(1.0 - np.mean(a) / np.mean(b)),
        "mean_paired_reduction": float(np.mean(paired)),
        "se_paired_reduction": float(np.std(paired, ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan"),
        "n": int(a.size),
    }


def _bootstrap_ratio_reduction(
    num: list[float],
    den: list[float],
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-14)
    a, b = a[mask], b[mask]
    if a.size == 0 or reps <= 0:
        return {"lower": float("nan"), "upper": float("nan"), "reps": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(reps, a.size))
    vals = 1.0 - np.mean(a[idx], axis=1) / np.mean(b[idx], axis=1)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"lower": float(lo), "upper": float(hi), "reps": int(reps)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reference_metadata_compatible(metadata: dict[str, Any], endpoints: ToyEndpointSource, cfg: dict[str, Any]) -> bool:
    train = cfg.get("reference_training", {})
    saved = dict(metadata.get("training", {}))
    endpoint = dict(metadata.get("endpoint_source", {}))
    expected = {
        "seed": int(train.get("seed", cfg.get("seed", 0))),
        "hidden_width": int(train.get("hidden_width", 128)),
        "hidden_layers": int(train.get("hidden_layers", 4)),
        "train_steps": int(train.get("train_steps", 12000)),
        "batch_size": int(train.get("batch_size", 2048)),
        "learning_rate": float(train.get("learning_rate", 1.0e-3)),
        "min_learning_rate_ratio": float(train.get("min_learning_rate_ratio", 0.05)),
        "adam_beta1": float(train.get("adam_beta1", 0.9)),
        "adam_beta2": float(train.get("adam_beta2", 0.999)),
        "adam_eps": float(train.get("adam_eps", 1.0e-8)),
        "grad_clip_norm": float(train.get("grad_clip_norm", 10.0)),
        "bridge_schedule": str(train.get("bridge_schedule", "linear")),
        "bridge_noise_std": float(train.get("bridge_noise_std", 0.15)),
    }
    for key, value in expected.items():
        if key not in saved:
            return False
        got = saved[key]
        if isinstance(value, float):
            if not np.isclose(float(got), value, rtol=0.0, atol=1.0e-14):
                return False
        elif got != value:
            return False
    return (
        np.isclose(float(endpoint.get("radius", np.nan)), float(endpoints.radius), rtol=0.0, atol=1.0e-14)
        and np.isclose(float(endpoint.get("sigma", np.nan)), float(endpoints.sigma), rtol=0.0, atol=1.0e-14)
    )


def _optimizer_cfg(cfg: dict[str, Any], prefix: str) -> OptimizerConfig:
    opt = cfg["optimization"]
    return OptimizerConfig(
        steps=int(opt.get(f"{prefix}_steps", opt.get("steps", 250))),
        learning_rate=float(opt.get(f"{prefix}_learning_rate", opt.get("learning_rate", 0.02))),
        beta1=float(opt.get("beta1", 0.9)),
        beta2=float(opt.get("beta2", 0.999)),
        eps=float(opt.get("eps", 1e-8)),
        constraint_penalty=float(opt.get("constraint_penalty", 1e4)),
        feasibility_tol=float(opt.get("feasibility_tol", 1e-6)),
    )


def _canonical_deg(eta: Array) -> list[float]:
    return np.degrees(np.asarray(jnp.sort(jnp.mod(eta, 2.0 * jnp.pi)), dtype=np.float64)).tolist()


# -----------------------------------------------------------------------------
# Endpoint-only flow matching reference
# -----------------------------------------------------------------------------


def _init_mlp(key: Array, *, input_dim: int, hidden_width: int, hidden_layers: int, output_dim: int = 2):
    dims = [input_dim] + [hidden_width] * hidden_layers + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        scale = math.sqrt(2.0 / max(din, 1))
        W = scale * jax.random.normal(k, (din, dout), dtype=jnp.float64)
        b = jnp.zeros((dout,), dtype=jnp.float64)
        layers.append({"W": W, "b": b})
    return tuple(layers)


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_l2_norm(tree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def _train_reference(
    endpoints: ToyEndpointSource,
    cfg: dict[str, Any],
    checkpoint: Path,
) -> tuple[MLPReferenceFlow, dict[str, Any]]:
    ref_cfg = cfg["reference"]
    train_cfg = cfg.get("reference_training", ref_cfg)
    seed = int(train_cfg.get("seed", cfg["seed"]))
    hidden_width = int(train_cfg.get("hidden_width", 128))
    hidden_layers = int(train_cfg.get("hidden_layers", 4))
    train_steps = int(train_cfg.get("train_steps", 12000))
    batch_size = int(train_cfg.get("batch_size", 2048))
    lr0 = float(train_cfg.get("learning_rate", 1e-3))
    lr_ratio = float(train_cfg.get("min_learning_rate_ratio", 0.05))
    beta1 = float(train_cfg.get("adam_beta1", 0.9))
    beta2 = float(train_cfg.get("adam_beta2", 0.999))
    adam_eps = float(train_cfg.get("adam_eps", 1e-8))
    clip = float(train_cfg.get("grad_clip_norm", 10.0))
    log_every = int(train_cfg.get("log_every", max(train_steps // 20, 1)))
    bridge_noise_std = float(train_cfg.get("bridge_noise_std", 0.5 * endpoints.sigma))

    key = jax.random.PRNGKey(seed + 101)
    key, kinit = jax.random.split(key)
    params = _init_mlp(
        kinit,
        input_dim=7,  # x in R^2 + 5 time features
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
    )
    m = _tree_zeros_like(params)
    v = _tree_zeros_like(params)

    def sample_batch(key: Array):
        k0, k1, kt, kz = jax.random.split(key, 4)
        x0 = endpoints.sample(k0, batch_size, endpoint=0)
        x1 = endpoints.sample(k1, batch_size, endpoint=1)
        t = jax.random.uniform(kt, (batch_size,), minval=0.0, maxval=1.0, dtype=jnp.float64)
        z = jax.random.normal(kz, (batch_size, 2), dtype=jnp.float64)
        gamma = bridge_noise_std * jnp.sin(jnp.pi * t)
        gamma_dot = bridge_noise_std * jnp.pi * jnp.cos(jnp.pi * t)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1 + gamma[:, None] * z
        target = x1 - x0 + gamma_dot[:, None] * z
        return t, xt, target

    def loss_fn(p, t, x, target):
        pred = velocity_mlp(p, t, x)
        return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))

    @jax.jit
    def train_step(p, m, v, key, step):
        t, x, target = sample_batch(key)
        loss, grads = jax.value_and_grad(loss_fn)(p, t, x, target)
        norm = _tree_l2_norm(grads)
        scale = jnp.minimum(1.0, clip / jnp.maximum(norm, 1e-30))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        m = jax.tree_util.tree_map(lambda mm, g: beta1 * mm + (1.0 - beta1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda vv, g: beta2 * vv + (1.0 - beta2) * (g * g), v, grads)
        s = step.astype(jnp.float64) + 1.0
        mhat = jax.tree_util.tree_map(lambda mm: mm / (1.0 - beta1**s), m)
        vhat = jax.tree_util.tree_map(lambda vv: vv / (1.0 - beta2**s), v)
        frac = step.astype(jnp.float64) / jnp.maximum(float(train_steps - 1), 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
        lr = lr0 * (lr_ratio + (1.0 - lr_ratio) * cosine)
        p = jax.tree_util.tree_map(
            lambda pp, mm, vv: pp - lr * mm / (jnp.sqrt(vv) + adam_eps),
            p,
            mhat,
            vhat,
        )
        return p, m, v, loss, norm, lr

    history: list[dict[str, float | int]] = []
    for step in range(train_steps):
        key, kstep = jax.random.split(key)
        params, m, v, loss, grad_norm, lr = train_step(
            params, m, v, kstep, jnp.asarray(step, dtype=jnp.int32)
        )
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == train_steps:
            row = {
                "step": int(step + 1),
                "loss": float(loss),
                "grad_norm": float(grad_norm),
                "learning_rate": float(lr),
            }
            history.append(row)
            print(f"  reference step {step + 1:6d}/{train_steps}: loss={row['loss']:.6g}", flush=True)

    metadata = {
        "training": {
            "seed": seed,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "learning_rate": lr0,
            "min_learning_rate_ratio": lr_ratio,
            "adam_beta1": beta1,
            "adam_beta2": beta2,
            "adam_eps": adam_eps,
            "grad_clip_norm": clip,
            "bridge_schedule": str(train_cfg.get("bridge_schedule", "linear")),
            "bridge_noise_std": bridge_noise_std,
            "history": history,
        },
        "endpoint_source": {
            "radius": endpoints.radius,
            "sigma": endpoints.sigma,
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_npz_checkpoint(checkpoint, params, metadata)
    flow = MLPReferenceFlow(
        params=params,
        substeps_per_interval=int(ref_cfg["rk4_substeps_per_time_interval"]),
        metadata=metadata,
    )
    return flow, metadata


def ensure_reference(
    endpoints: ToyEndpointSource,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[MLPReferenceFlow, Path, dict[str, Any]]:
    checkpoint = output_dir / "reference.npz"
    if checkpoint.exists():
        flow = MLPReferenceFlow.from_npz(
            checkpoint,
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        metadata = dict(flow.metadata or {})
        if _reference_metadata_compatible(metadata, endpoints, cfg):
            print("[reference] reusing compatible reference.npz", flush=True)
            return flow, checkpoint, metadata
        print("[reference] cached reference.npz is incompatible with current config; retraining", flush=True)

    print("[reference] training endpoint-only flow-matching reference", flush=True)
    flow, metadata = _train_reference(endpoints, cfg, checkpoint)
    return flow, checkpoint, metadata


# -----------------------------------------------------------------------------
# Frozen reference bank and CRN trial banks
# -----------------------------------------------------------------------------


def build_reference_bank(
    flow: MLPReferenceFlow,
    endpoints: ToyEndpointSource,
    times: Array,
    cfg: dict[str, Any],
) -> tuple[Array, Array, Array, Array]:
    ref_cfg = cfg["reference"]
    mode = str(ref_cfg.get("bank_mode", "gauss-hermite"))
    if mode == "gauss-hermite":
        x0, base_weights = endpoints.gauss_hermite_bank(int(ref_cfg.get("gh_order", 36)))
    elif mode == "particles":
        n = int(ref_cfg.get("particles", 32768))
        x0 = endpoints.sample(jax.random.PRNGKey(int(cfg["seed"]) + 211), n, endpoint=0)
        base_weights = jnp.ones(n, dtype=jnp.float64) / float(n)
    else:
        raise ValueError(f"Unknown reference.bank_mode={mode!r}")

    nodes = flow.rollout(x0, times)
    velocity = jax.vmap(lambda t, x: flow.velocity(x, t))(times, nodes)
    base_weights_by_time = jnp.broadcast_to(base_weights[None, :], (len(times), len(base_weights)))
    return nodes, velocity, base_weights_by_time, x0


def make_trial_bank(
    population: ToyPopulation,
    times: Array,
    acq_idx: Array,
    *,
    finite_n: int,
    trials: int,
    seed: int,
    namespace: int,
) -> TrialBank:
    masses = []
    indices = []
    detector = []
    alphas = []
    alpha_min = float(population.alpha_min)
    alpha_max = float(population.alpha_max)

    for trial in range(int(trials)):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(namespace), int(trial)]))
        alpha = float(rng.uniform(alpha_min, alpha_max))
        mass = np.asarray(population.masses(times, jnp.asarray(alpha)), dtype=np.float64)
        flat = mass.reshape((mass.shape[0], -1))
        sample_rows = []
        noise_rows = []
        for idx in np.asarray(acq_idx, dtype=int):
            p = np.maximum(flat[idx], 0.0)
            p /= p.sum()
            sample_rows.append(rng.choice(p.size, size=int(finite_n), replace=True, p=p))
            noise_rows.append(rng.standard_normal(2))
        masses.append(flat)
        indices.append(np.stack(sample_rows, axis=0))
        detector.append(np.stack(noise_rows, axis=0))
        alphas.append(alpha)

    return TrialBank(
        masses=jnp.asarray(np.stack(masses), dtype=jnp.float64),
        sample_indices=jnp.asarray(np.stack(indices), dtype=jnp.int32),
        detector_z=jnp.asarray(np.stack(detector), dtype=jnp.float64),
        alphas=jnp.asarray(np.asarray(alphas), dtype=jnp.float64),
    )


def save_reference_bank(
    output_dir: Path,
    *,
    times: Array,
    nodes: Array,
    velocity: Array,
    base_weights: Array,
    in_domain_mask: Array,
    in_domain_base_mass: Array,
    signature: str,
) -> None:
    np.savez_compressed(
        output_dir / "reference_bank.npz",
        times=np.asarray(times),
        reference_particles=np.asarray(nodes),
        reference_velocity=np.asarray(velocity),
        base_weights=np.asarray(base_weights),
        in_domain_mask=np.asarray(in_domain_mask),
        in_domain_base_mass=np.asarray(in_domain_base_mass),
        signature=np.asarray(signature),
    )


def save_trial_bank(
    output_dir: Path, name: str, bank: TrialBank, *, acq_idx: Array, signature: str
) -> None:
    np.savez_compressed(
        output_dir / f"{name}_bank.npz",
        masses=np.asarray(bank.masses),
        sample_indices=np.asarray(bank.sample_indices),
        detector_z=np.asarray(bank.detector_z),
        alphas=np.asarray(bank.alphas),
        acquisition_indices=np.asarray(acq_idx),
        signature=np.asarray(signature),
    )


# -----------------------------------------------------------------------------
# Authoritative toy experiment evaluator
# -----------------------------------------------------------------------------


class ToyExperiment:
    def __init__(
        self,
        cfg: dict[str, Any],
        reference: MLPReferenceFlow,
        *,
        reference_nodes: Array | None = None,
        reference_velocity: Array | None = None,
        reference_weights: Array | None = None,
    ):
        self.cfg = cfg
        self.reference = reference

        pop_cfg = cfg["population"]
        meas_cfg = cfg["measurement"]
        pois_cfg = cfg["poisson"]
        mom_cfg = cfg.get("moment_reconstruction", {})
        proj_cfg = cfg["projection"]
        particle_cfg = cfg.get("particle_mfsi", {})
        raster_cfg = cfg.get("raster", {})

        self.grid = CartesianGrid2D(
            half_width=float(pop_cfg.get("domain_half_width", 3.2)),
            n=int(pois_cfg["grid_n"]),
        )
        self.times = jnp.linspace(0.0, 1.0, int(pois_cfg["time_n"]), dtype=jnp.float64)
        self.time_w = _trap_weights(len(self.times))
        self.acq_idx = jnp.asarray(
            _nested_acquisition_indices(len(self.times), int(meas_cfg["acquisition_k"])),
            dtype=jnp.int32,
        )
        self.heldout_idx = jnp.asarray(
            _heldout_indices(len(self.times), np.asarray(self.acq_idx)), dtype=jnp.int32
        )
        self.population_idx = jnp.arange(1, len(self.times) - 1, dtype=jnp.int32)

        self.endpoints = ToyEndpointSource(
            radius=float(pop_cfg["radius"]),
            sigma=float(pop_cfg["sigma"]),
        )
        self.population = ToyPopulation(
            grid=self.grid,
            radius=float(pop_cfg["radius"]),
            sigma=float(pop_cfg["sigma"]),
            alpha_min=math.radians(float(pop_cfg["alpha_min_deg"])),
            alpha_max=math.radians(float(pop_cfg["alpha_max_deg"])),
        )
        self.family = GaussianSensor2D(
            radius=float(meas_cfg["sensor_radius"]),
            width=float(meas_cfg["sensor_width"]),
        )

        self.projector = EmpiricalIProjector(
            IProjectionConfig(
                max_steps=int(proj_cfg.get("search_max_steps", proj_cfg.get("max_steps", 300))),
                residual_tol=float(proj_cfg.get("search_residual_tol", proj_cfg.get("residual_tol", 1e-10))),
                newton_ridge=float(proj_cfg.get("newton_ridge", 1e-7)),
                step_cap=float(proj_cfg.get("step_cap", 20.0)),
                lambda_clip=float(proj_cfg.get("lambda_clip", 1000.0)),
                line_search_steps=int(proj_cfg.get("search_line_search_steps", proj_cfg.get("line_search_steps", 8))),
                implicit_ridge=float(proj_cfg.get("implicit_ridge", 0.0)),
            )
        )
        self.moment_cfg = QuadraticBridgeConfig(
            ridge_rel=float(mom_cfg.get("ridge_rel", 1e-12)),
            variance_floor=float(mom_cfg.get("variance_floor", 1e-10)),
        )
        self.particle_cfg = ParticleMFSIConfig(
            covariance_ridge=float(particle_cfg.get("covariance_ridge", 1e-7)),
            tangent_ridge=float(particle_cfg.get("tangent_ridge", 1e-7)),
        )
        self.raster_cfg = RasterConfig(
            bandwidth=float(raster_cfg.get("bandwidth", 0.0)),
            truncate=float(raster_cfg.get("truncate", 4.0)),
        )
        self.poisson_cfg = PoissonConfig(
            dx=float(self.grid.dx),
            operator_floor_rel=float(pois_cfg.get("operator_floor_rel", 2e-5)),
            cg_tol=float(pois_cfg.get("cg_tol", 1e-8)),
            cg_maxiter=int(pois_cfg.get("cg_maxiter", 520)),
            gauge_strength=float(pois_cfg.get("gauge_strength", 1.0)),
        )

        # Stage 4 uses a lower-fidelity discretization of the *same* full MFSI
        # weighted-Poisson action for gradients.  Only numerical resolution changes:
        # fewer CRN trials/time nodes, a coarser Poisson grid, and a looser CG solve.
        # Every optimized endpoint is re-scored with ``self.poisson_cfg`` on the full
        # scientific grid/time/action bank before it can be selected.
        opt_cfg = cfg.get("optimization", {})
        grad_time_n = int(opt_cfg.get("full_gradient_time_n", min(5, len(self.times))))
        grad_time_n = max(3, min(grad_time_n, len(self.times)))
        grad_idx = np.rint(np.linspace(0, len(self.times) - 1, grad_time_n)).astype(np.int32)
        grad_idx = np.unique(grad_idx)
        if len(grad_idx) < 3:
            grad_idx = np.asarray([0, len(self.times) // 2, len(self.times) - 1], dtype=np.int32)
        self.full_gradient_time_idx = jnp.asarray(grad_idx, dtype=jnp.int32)
        self.full_gradient_time_w = _trap_weights_at_nodes(self.times[self.full_gradient_time_idx])

        grad_grid_n = int(opt_cfg.get("full_gradient_grid_n", min(31, self.grid.n)))
        grad_grid_n = max(9, min(grad_grid_n, self.grid.n))
        self.full_gradient_grid = CartesianGrid2D(
            half_width=float(pop_cfg.get("domain_half_width", 3.2)),
            n=grad_grid_n,
        )
        self.poisson_gradient_cfg = PoissonConfig(
            dx=float(self.full_gradient_grid.dx),
            operator_floor_rel=float(opt_cfg.get(
                "full_gradient_operator_floor_rel", pois_cfg.get("operator_floor_rel", 2e-5)
            )),
            cg_tol=float(opt_cfg.get(
                "full_gradient_cg_tol", max(float(pois_cfg.get("cg_tol", 1e-8)), 1e-5)
            )),
            cg_maxiter=int(opt_cfg.get(
                "full_gradient_cg_maxiter", min(int(pois_cfg.get("cg_maxiter", 520)), 80)
            )),
            gauge_strength=float(pois_cfg.get("gauge_strength", 1.0)),
        )
        self.full_gradient_poisson_backend = str(
            opt_cfg.get("full_gradient_poisson_backend", "jax")
        )
        if self.full_gradient_poisson_backend not in {"jax", "tesseract_cpp"}:
            raise ValueError(
                "optimization.full_gradient_poisson_backend must be 'jax' or "
                f"'tesseract_cpp'; got {self.full_gradient_poisson_backend!r}"
            )
        native_revision = "not-applicable"
        if self.full_gradient_poisson_backend == "tesseract_cpp":
            from mfsi.poisson_tesseract import (
                NATIVE_SOLVER_REVISION,
                TesseractPoissonUnavailable,
                is_tesseract_poisson_available,
            )

            if not is_tesseract_poisson_available():
                raise TesseractPoissonUnavailable(
                    "The toy stage-4 proxy explicitly requests tesseract_cpp, but "
                    "Tesseract-JAX or the native extension is unavailable."
                )
            native_revision = NATIVE_SOLVER_REVISION
        self.full_exact_poisson_backend = str(
            opt_cfg.get(
                "full_exact_poisson_backend", self.full_gradient_poisson_backend
            )
        )
        if self.full_exact_poisson_backend not in {"jax", "tesseract_cpp"}:
            raise ValueError(
                "optimization.full_exact_poisson_backend must be 'jax' or "
                f"'tesseract_cpp'; got {self.full_exact_poisson_backend!r}"
            )
        if (
            self.full_exact_poisson_backend == "tesseract_cpp"
            and self.full_gradient_poisson_backend != "tesseract_cpp"
        ):
            from mfsi.poisson_tesseract import (
                TesseractPoissonUnavailable,
                is_tesseract_poisson_available,
            )

            if not is_tesseract_poisson_available():
                raise TesseractPoissonUnavailable(
                    "The toy exact stage-4 evaluator explicitly requests "
                    "tesseract_cpp, but Tesseract-JAX or the native extension is "
                    "unavailable."
                )
        self.full_gradient_cache_signature = hashlib.sha256(
            json.dumps(
                {
                    "full_gradient_poisson_backend": self.full_gradient_poisson_backend,
                    "native_solver_revision": native_revision,
                    "gradient_grid_n": int(self.full_gradient_grid.n),
                    "gradient_trials": int(opt_cfg.get("full_gradient_trials", 4)),
                    "gradient_time_indices": np.asarray(
                        self.full_gradient_time_idx, dtype=np.int32
                    ).tolist(),
                    "gradient_cg_tol": float(self.poisson_gradient_cfg.cg_tol),
                    "gradient_cg_maxiter": int(self.poisson_gradient_cfg.cg_maxiter),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        self.mmd_kernel = gaussian_mmd_kernel(
            self.grid.n,
            self.grid.dx,
            float(cfg["law"]["mmd_bandwidth"]),
        )
        self.support_directions = unit_directions_2d(
            int(cfg.get("feasibility", {}).get("directions", 96))
        )
        self.feasibility_margin = float(
            cfg.get("feasibility", {}).get(
                "margin", cfg.get("moment_reconstruction", {}).get("feasibility_margin", 0.0)
            )
        )

        if reference_nodes is None or reference_velocity is None or reference_weights is None:
            nodes, velocity, weights, _ = build_reference_bank(reference, self.endpoints, self.times, cfg)
            reference_nodes = nodes
            reference_velocity = velocity
            reference_weights = weights

        self.reference_nodes = jnp.asarray(reference_nodes, dtype=jnp.float64)
        self.reference_velocity = jnp.asarray(reference_velocity, dtype=jnp.float64)
        self.reference_weights = jnp.asarray(reference_weights, dtype=jnp.float64)

        self.reference_in_domain = self.grid.in_domain(self.reference_nodes)
        self.reference_base_mass = jnp.sum(
            jnp.where(self.reference_in_domain, self.reference_weights, 0.0), axis=-1
        )
        masked = jnp.where(self.reference_in_domain, self.reference_weights, 0.0)
        self.reference_weights = masked / jnp.maximum(jnp.sum(masked, axis=-1, keepdims=True), 1e-300)

        # NumPy/SciPy exact geometry is cached only for authoritative rescoring and
        # validation. It never enters a differentiated objective.
        self._exact_geometry_cache: dict[tuple[float, float], dict[str, Any]] = {}
        self._exact_polytope_cache: dict[tuple[float, float], ExactBetaPolytope] = {}
        self._exact_polytope_error_cache: dict[tuple[float, float], tuple[str, float, str]] = {}
        self._exact_population_result_cache: dict[tuple[float, float], dict[str, Any]] = {}
        self._exact_finite_result_cache: dict[tuple[tuple[float, float], int], dict[str, Any]] = {}
        self._exact_tangent_result_cache: dict[tuple[tuple[float, float], int], dict[str, Any]] = {}
        self._exact_full_result_cache: dict[tuple[tuple[float, float], int, int], dict[str, Any]] = {}
        self._exact_reconstruct_cache: dict[tuple[tuple[float, float], int, int], dict[str, Any]] = {}
        self._exact_trial_result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Geometry / measurement reconstruction
    # ------------------------------------------------------------------

    def _geometry(self, eta: Array):
        eta = self.family.canonicalize(eta)
        phi_grid = self.family.features(self.grid.flat_points(), eta)
        phi_nodes = self.family.features(self.reference_nodes, eta)
        grad_nodes = self.family.feature_gradients(self.reference_nodes, eta)
        return phi_grid, phi_nodes, grad_nodes

    def _measurement_data(self, phi_grid: Array, bank: TrialBank, trial: int):
        masses = bank.masses[trial]
        idx = bank.sample_indices[trial]
        z = bank.detector_z[trial]
        acq_mass = masses[self.acq_idx]
        exact = acq_mass @ phi_grid

        second = jnp.einsum("kg,gi,gj->kij", acq_mass, phi_grid, phi_grid)
        cov = second - jnp.einsum("ki,kj->kij", exact, exact)
        eye = jnp.eye(phi_grid.shape[-1], dtype=jnp.float64)
        V = cov / float(self.cfg["measurement"]["finite_n"])
        V = V + (
            float(self.cfg["measurement"]["obs_noise_std"]) ** 2
            + float(self.moment_cfg.variance_floor)
        ) * eye

        samples = phi_grid[idx]
        y = jnp.mean(samples, axis=1) + float(self.cfg["measurement"]["obs_noise_std"]) * z
        endpoint = (self.acq_idx == 0) | (self.acq_idx == len(self.times) - 1)
        y = jnp.where(endpoint[:, None], exact, y)
        return y, V, exact[0], exact[-1]

    def _reconstruct_from_geometry(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        bank: TrialBank,
        trial: int | Array,
    ) -> Reconstruction:
        """Reconstruct one trial while reusing eta-dependent sensor geometry."""
        y, V, c0, c1 = self._measurement_data(phi_grid, bank, trial)
        fit = fit_quadratic_bridge_gls(
            self.times[self.acq_idx], y, V, c0, c1, self.times, self.moment_cfg
        )

        A, b, endpoint_violation = common_beta_support_polytope_2d(
            directions=self.support_directions,
            times=self.times,
            c0=c0,
            c1=c1,
            physical_features=phi_grid,
            particle_features_by_time=phi_nodes,
            particle_mask_by_time=self.reference_in_domain,
            margin=self.feasibility_margin,
        )
        proj = project_metric_polytope_2d(
            fit.beta,
            fit.information,
            A,
            b,
            tol=float(self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("tol", 1e-9))),
        )
        c, c_dot = evaluate_quadratic_bridge(proj.beta, c0, c1, self.times)
        return Reconstruction(
            c=c,
            c_dot=c_dot,
            beta=proj.beta,
            beta_unconstrained=fit.beta,
            projection_distance=proj.distance,
            max_unconstrained_violation=proj.max_unconstrained_violation,
            endpoint_feasibility_violation=endpoint_violation,
        )

    def _reconstruct(self, eta: Array, bank: TrialBank, trial: int) -> Reconstruction:
        phi_grid, phi_nodes, _ = self._geometry(eta)
        return self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)

    # ------------------------------------------------------------------
    # Authoritative non-differentiated geometry / evaluation
    # ------------------------------------------------------------------

    def _exact_key(self, eta: Array) -> tuple[float, float]:
        x = np.asarray(self.family.canonicalize(eta), dtype=np.float64)
        return tuple(np.round(x, 12))

    def _exact_geometry(self, eta: Array) -> dict[str, Any]:
        """Exact 2-D moment hulls, cached per design for final scoring only."""
        eta = self.family.canonicalize(eta)
        key = self._exact_key(eta)
        cached = self._exact_geometry_cache.get(key)
        if cached is not None:
            return cached

        phi_grid_j = self.family.features(self.grid.flat_points(), eta)
        phi_nodes_j = self.family.features(self.reference_nodes, eta)
        phi_grid = np.asarray(phi_grid_j, dtype=np.float64)
        phi_nodes = np.asarray(phi_nodes_j, dtype=np.float64)
        in_domain = np.asarray(self.reference_in_domain, dtype=bool)

        physical_eq = hull_equations_2d(phi_grid)
        particle_eq = tuple(
            hull_equations_2d(phi_nodes[k][in_domain[k]])
            for k in range(len(self.times))
        )

        alpha_mid = 0.5 * (self.population.alpha_min + self.population.alpha_max)
        mass0 = np.asarray(self.population.mass(jnp.asarray(0.0), jnp.asarray(alpha_mid)), dtype=np.float64).reshape(-1)
        mass1 = np.asarray(self.population.mass(jnp.asarray(1.0), jnp.asarray(alpha_mid)), dtype=np.float64).reshape(-1)
        c0 = mass0 @ phi_grid
        c1 = mass1 @ phi_grid

        cached = {
            "eta": np.asarray(eta, dtype=np.float64),
            "phi_grid": phi_grid,
            "phi_nodes": phi_nodes,
            "grad_nodes": None,
            "physical_equations": physical_eq,
            "particle_equations": particle_eq,
            "c0": c0,
            "c1": c1,
        }
        self._exact_geometry_cache[key] = cached
        return cached

    def _exact_grad_nodes(self, eta: Array) -> np.ndarray:
        g = self._exact_geometry(eta)
        if g["grad_nodes"] is None:
            g["grad_nodes"] = np.asarray(
                self.family.feature_gradients(
                    self.reference_nodes, self.family.canonicalize(eta)
                ),
                dtype=np.float64,
            )
        return g["grad_nodes"]

    def _exact_polytope(self, eta: Array) -> ExactBetaPolytope:
        """Return the authoritative common hull polytope for ``eta``.

        Exact infeasibility is a scientific property of a candidate, not a fatal
        program error.  We cache the negative result as well so rejected designs
        are not repeatedly sent through Qhull/HiGHS during exact audits.
        """
        key = self._exact_key(eta)
        cached = self._exact_polytope_cache.get(key)
        if cached is not None:
            return cached
        err = self._exact_polytope_error_cache.get(key)
        if err is not None:
            reason, violation, message = err
            raise ExactFeasibilityError(message, reason=reason, violation=violation)
        try:
            g = self._exact_geometry(eta)
            cached = build_common_quadratic_beta_polytope_2d(
                times=np.asarray(self.times, dtype=np.float64),
                c0=g["c0"],
                c1=g["c1"],
                physical_features=g["phi_grid"],
                particle_features_by_time=g["phi_nodes"],
                particle_mask_by_time=np.asarray(self.reference_in_domain, dtype=bool),
                margin=float(self.feasibility_margin),
                physical_equations=g["physical_equations"],
                particle_equations=g["particle_equations"],
            )
        except ExactFeasibilityError as exc:
            self._exact_polytope_error_cache[key] = (exc.reason, exc.violation, str(exc))
            raise
        self._exact_polytope_cache[key] = cached
        return cached

    def _exact_reconstruct(self, eta: Array, bank: TrialBank, trial: int) -> dict[str, Any]:
        """Endpoint-anchored GLS plus exact common-hull beta projection."""
        cache_key = (self._exact_key(eta), id(bank), int(trial))
        cached = self._exact_reconstruct_cache.get(cache_key)
        if cached is not None:
            return cached
        g = self._exact_geometry(eta)
        poly = self._exact_polytope(eta)
        phi_grid = jnp.asarray(g["phi_grid"], dtype=jnp.float64)
        y, V, c0, c1 = self._measurement_data(phi_grid, bank, trial)
        fit = fit_quadratic_bridge_gls(
            self.times[self.acq_idx], y, V, c0, c1, self.times, self.moment_cfg
        )
        c0_np = np.asarray(c0, dtype=np.float64)
        c1_np = np.asarray(c1, dtype=np.float64)
        if max(np.max(np.abs(c0_np - g["c0"])), np.max(np.abs(c1_np - g["c1"]))) > 1.0e-11:
            poly = build_common_quadratic_beta_polytope_2d(
                times=np.asarray(self.times, dtype=np.float64),
                c0=c0_np,
                c1=c1_np,
                physical_features=g["phi_grid"],
                particle_features_by_time=g["phi_nodes"],
                particle_mask_by_time=np.asarray(self.reference_in_domain, dtype=bool),
                margin=float(self.feasibility_margin),
                physical_equations=g["physical_equations"],
                particle_equations=g["particle_equations"],
            )
        projection = project_metric_polytope_exact_2d(
            np.asarray(fit.beta, dtype=np.float64),
            np.asarray(fit.information, dtype=np.float64),
            poly,
            tol=float(self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("tol", 1.0e-9)))),
        )
        c, c_dot = evaluate_quadratic_bridge(
            jnp.asarray(projection.beta, dtype=jnp.float64), c0, c1, self.times
        )
        result = {
            "c": np.asarray(c, dtype=np.float64),
            "c_dot": np.asarray(c_dot, dtype=np.float64),
            "beta": projection.beta,
            "projection_distance": float(projection.distance),
            "max_unconstrained_violation": float(projection.max_unconstrained_violation),
            "endpoint_feasibility_violation": float(poly.endpoint_max_violation),
        }
        self._exact_reconstruct_cache[cache_key] = result
        return result

    def _exact_tilt(self, phi: np.ndarray, base: np.ndarray, target: np.ndarray, lam0: np.ndarray):
        proj_cfg = self.cfg.get("projection", {})
        return robust_empirical_tilt_exact(
            phi,
            base,
            target,
            lam0=lam0,
            newton_steps=int(proj_cfg.get("max_steps", 300)),
            newton_ridge=float(proj_cfg.get("newton_ridge", 1.0e-7)),
            step_cap=float(proj_cfg.get("step_cap", 20.0)),
            lambda_clip=float(proj_cfg.get("lambda_clip", 1000.0)),
            accept_tol=float(proj_cfg.get("solver_accept_tol", 2.0e-6)),
            lbfgs_maxiter=int(proj_cfg.get("lbfgs_maxiter", 800)),
            retry_multiplier=float(proj_cfg.get("retry_clip_multiplier", 2.0)),
            retries=int(proj_cfg.get("max_retries", 2)),
        )

    def exact_population_result(self, eta: Array) -> dict[str, Any]:
        """Strict authoritative L(eta): invalid alpha/time rows are rejected."""
        eta = self.family.canonicalize(eta)
        key = self._exact_key(eta)
        if key in self._exact_population_result_cache:
            return self._exact_population_result_cache[key]
        g = self._exact_geometry(eta)
        alphas, alpha_w = self.population.alpha_quadrature(
            int(self.cfg["population"].get("alpha_quadrature_n", 5))
        )
        alphas = np.asarray(alphas, dtype=np.float64)
        alpha_w = np.asarray(alpha_w, dtype=np.float64)
        idx = np.asarray(self.population_idx, dtype=np.int32)
        tw = np.asarray(self.time_w, dtype=np.float64)[idx]
        tw = tw / np.sum(tw)
        valid_cfg = self.cfg["validity"]
        max_cal_allowed = float(valid_cfg.get("max_population_calibration_resid", 1.0e-5))
        min_ess_allowed = float(valid_cfg.get("min_ess_fraction", 0.03))
        base_mass_ok = float(np.min(np.asarray(self.reference_base_mass))) >= float(
            valid_cfg.get("min_in_domain_base_mass", 0.995)
        )
        hull_tol = max(1.0e-10, 10.0 * float(self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("tol", 1.0e-9))))

        alpha_scores = []
        max_resid = 0.0
        min_ess = np.inf
        max_hull = -np.inf
        all_valid = bool(base_mass_ok)
        for alpha in alphas:
            truth = np.asarray(
                self.population.masses(self.times, jnp.asarray(alpha)), dtype=np.float64
            ).reshape((len(self.times), -1))
            targets = truth @ g["phi_grid"]
            lam = np.zeros(2, dtype=np.float64)
            vals = []
            for t_idx in idx:
                hull_v = max_hull_violation(g["particle_equations"][int(t_idx)], targets[int(t_idx)])
                max_hull = max(max_hull, hull_v)
                st = self._exact_tilt(
                    g["phi_nodes"][int(t_idx)],
                    np.asarray(self.reference_weights[int(t_idx)], dtype=np.float64),
                    targets[int(t_idx)],
                    lam,
                )
                lam = st.lam
                max_resid = max(max_resid, st.residual_norm)
                min_ess = min(min_ess, st.ess_fraction)
                mass = self._raster_mass_from_weights(
                    int(t_idx), jnp.asarray(st.weights, dtype=jnp.float64)
                )
                vals.append(float(gaussian_mmd2_grid_mass(
                    mass,
                    jnp.asarray(truth[int(t_idx)].reshape((self.grid.n, self.grid.n))),
                    self.mmd_kernel,
                )))
                all_valid = all_valid and (hull_v <= hull_tol)
            alpha_scores.append(float(np.sum(tw * np.asarray(vals))))

        all_valid = bool(
            all_valid
            and max_resid <= max_cal_allowed
            and min_ess >= min_ess_allowed
            and np.all(np.isfinite(alpha_scores))
        )
        value = float(np.sum(alpha_w * np.asarray(alpha_scores))) if all_valid else float("inf")
        result = {
            "valid": all_valid,
            "value": value,
            "max_calibration_residual": float(max_resid),
            "min_ess_fraction": float(min_ess),
            "max_particle_hull_violation": float(max_hull),
        }
        self._exact_population_result_cache[key] = result
        return result

    def _exact_trial_result(
        self,
        eta: Array,
        bank: TrialBank,
        trial: int,
        *,
        compute_law: bool,
        compute_tangent: bool,
        compute_full: bool,
    ) -> dict[str, Any]:
        """One authoritative finite/noisy trial with exact common-hull reconstruction."""
        eta = self.family.canonicalize(eta)
        trial_cache_key = (
            self._exact_key(eta), id(bank), int(trial),
            bool(compute_law), bool(compute_tangent), bool(compute_full),
        )
        cached_trial = self._exact_trial_result_cache.get(trial_cache_key)
        if cached_trial is not None:
            return dict(cached_trial)
        try:
            g = self._exact_geometry(eta)
            poly = self._exact_polytope(eta)
            if compute_tangent or compute_full:
                self._exact_grad_nodes(eta)
            rec = self._exact_reconstruct(eta, bank, trial)
        except ExactFeasibilityError as exc:
            # Hard scientific rejection.  Do not turn expected infeasibility into a
            # process-level exception: exact selection treats this row as invalid.
            failed = {
                "trial": int(trial),
                "alpha": float(bank.alphas[trial]),
                "valid": False,
                "invalid_reason": exc.reason,
                "law_risk": float("nan"),
                "tangent_action": float("nan"),
                "full_action": float("nan"),
                "max_calibration_residual": float("nan"),
                "min_ess_fraction": float("nan"),
                "max_hull_violation": float(exc.violation),
                "feasibility_projection_distance": float("nan"),
                "max_poisson_relative_residual": float("nan"),
                "tangent_full_gap": float("nan"),
                "tangent_lower_bound_violation": float("nan"),
            }
            self._exact_trial_result_cache[trial_cache_key] = failed
            return dict(failed)
        truth = np.asarray(bank.masses[trial], dtype=np.float64)
        valid_cfg = self.cfg["validity"]
        max_cal_allowed = float(valid_cfg.get("max_finite_calibration_resid", 1.0e-3))
        min_ess_allowed = float(valid_cfg.get("min_ess_fraction", 0.03))
        base_mass_ok = float(np.min(np.asarray(self.reference_base_mass))) >= float(
            valid_cfg.get("min_in_domain_base_mass", 0.995)
        )
        hull_tol = max(1.0e-10, 10.0 * float(self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("tol", 1e-9))))
        held_set = set(map(int, np.asarray(self.heldout_idx).tolist()))
        held_w = np.asarray(self.time_w, dtype=np.float64)[np.asarray(self.heldout_idx, dtype=int)]
        held_w = held_w / np.sum(held_w)

        law_vals: list[float] = []
        tangent_vals: list[float] = []
        full_vals: list[float] = []
        full_q_rows: list[Array] = []
        full_h_rows: list[Array] = []
        lam = np.zeros(2, dtype=np.float64)
        max_resid = 0.0
        min_ess = np.inf
        max_hull = -np.inf
        max_poisson = 0.0
        max_tangent_compat = 0.0
        min_cov_eig = np.inf
        all_valid = bool(base_mass_ok and rec["endpoint_feasibility_violation"] <= hull_tol)

        for t_idx in range(len(self.times)):
            target = rec["c"][t_idx]
            physical_v = max_hull_violation(poly.physical_equations, target)
            particle_v = max_hull_violation(poly.particle_equations[t_idx], target)
            hull_v = max(physical_v, particle_v)
            max_hull = max(max_hull, hull_v)
            all_valid = all_valid and (hull_v <= hull_tol)

            st = self._exact_tilt(
                g["phi_nodes"][t_idx],
                np.asarray(self.reference_weights[t_idx], dtype=np.float64),
                target,
                lam,
            )
            lam = st.lam
            max_resid = max(max_resid, st.residual_norm)
            min_ess = min(min_ess, st.ess_fraction)

            m = None
            mean_m = None
            if compute_tangent or compute_full:
                m = np.einsum(
                    "nmd,nd->nm",
                    g["grad_nodes"][t_idx],
                    np.asarray(self.reference_velocity[t_idx]),
                )
                mean_m = np.sum(st.weights[:, None] * m, axis=0)
            if compute_tangent:
                r = mean_m - rec["c_dot"][t_idx]
                G = np.einsum(
                    "n,nmd,nkd->mk", st.weights, g["grad_nodes"][t_idx], g["grad_nodes"][t_idx]
                )
                exact_tangent_ridge = float(self.cfg.get("particle_mfsi", {}).get("exact_tangent_ridge", 0.0))
                if exact_tangent_ridge:
                    G = G + exact_tangent_ridge * np.eye(G.shape[0])
                G_pinv = np.linalg.pinv(
                    G, rcond=float(self.cfg.get("particle_mfsi", {}).get("tangent_pinv_rcond", 1.0e-10))
                )
                coeff = G_pinv @ r
                compat = float(np.linalg.norm(G @ coeff - r))
                max_tangent_compat = max(max_tangent_compat, compat)
                tangent_vals.append(float(r @ coeff))

            need_raster = compute_full or (compute_law and t_idx in held_set)
            ras = None
            if need_raster:
                if compute_full:
                    gg = m @ st.lam
                    mean_g = float(np.sum(st.weights * gg))
                    centered_phi = g["phi_nodes"][t_idx] - st.moments[None, :]
                    cov_phi_g = np.sum(
                        st.weights[:, None] * centered_phi * (gg - mean_g)[:, None], axis=0
                    )
                    exact_cov_ridge = float(self.cfg.get("particle_mfsi", {}).get("exact_covariance_ridge", 0.0))
                    cov = st.covariance + exact_cov_ridge * np.eye(st.covariance.shape[0])
                    eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T))))
                    min_cov_eig = min(min_cov_eig, eig_min)
                    rhs = rec["c_dot"][t_idx] - mean_m - cov_phi_g
                    cov_floor = float(self.cfg.get("particle_mfsi", {}).get("exact_covariance_min_eig", 1.0e-12))
                    if eig_min <= cov_floor:
                        all_valid = False
                        lam_dot = np.linalg.lstsq(cov, rhs, rcond=None)[0]
                    else:
                        lam_dot = np.linalg.solve(cov, rhs)
                    forcing = centered_phi @ lam_dot + gg - mean_g
                    forcing = forcing - float(np.sum(st.weights * forcing))
                else:
                    forcing = np.zeros_like(st.weights)
                ras = rasterize_projected_particles(
                    self.reference_nodes[t_idx],
                    jnp.asarray(st.weights, dtype=jnp.float64),
                    jnp.asarray(forcing, dtype=jnp.float64),
                    self.grid,
                    self.raster_cfg,
                )

            if compute_law and t_idx in held_set:
                law_vals.append(float(gaussian_mmd2_grid_mass(
                    ras.mass,
                    jnp.asarray(truth[t_idx].reshape((self.grid.n, self.grid.n))),
                    self.mmd_kernel,
                )))
            if compute_full:
                if self.full_exact_poisson_backend == "tesseract_cpp":
                    full_q_rows.append(ras.q)
                    full_h_rows.append(ras.h)
                else:
                    pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_cfg)
                    full_vals.append(float(pois.action))
                    max_poisson = max(max_poisson, float(pois.relative_residual))

        if compute_full and self.full_exact_poisson_backend == "tesseract_cpp":
            # The exact tilt remains sequential in time so its multiplier warm
            # starts and all scientific validity checks are unchanged. Once q/h
            # are known, the independent systems belong in one native call.
            from mfsi.poisson_tesseract import solve_weighted_poisson_batch_tesseract

            q_batch = jnp.stack(full_q_rows)
            h_batch = jnp.stack(full_h_rows)
            psi_batch = solve_weighted_poisson_batch_tesseract(
                q_batch, h_batch, self.poisson_cfg
            )
            actions, residuals = _batched_poisson_diagnostics(
                psi_batch,
                q_batch,
                h_batch,
                dx=float(self.poisson_cfg.dx),
                operator_floor_rel=float(self.poisson_cfg.operator_floor_rel),
                gauge_strength=float(self.poisson_cfg.gauge_strength),
            )
            full_vals = np.asarray(actions, dtype=np.float64).tolist()
            max_poisson = float(np.max(np.asarray(residuals, dtype=np.float64)))

        all_valid = bool(
            all_valid
            and max_resid <= max_cal_allowed
            and min_ess >= min_ess_allowed
        )
        tangent_compat_gate = float(self.cfg.get("particle_mfsi", {}).get("max_tangent_compatibility_residual", 1.0e-7))
        if compute_tangent:
            all_valid = all_valid and max_tangent_compat <= tangent_compat_gate

        poisson_gate = valid_cfg.get("max_poisson_relative_residual")
        if compute_full and poisson_gate is not None:
            all_valid = all_valid and max_poisson <= float(poisson_gate)

        law = float(np.sum(held_w * np.asarray(law_vals))) if compute_law and law_vals else float("nan")
        tangent = float(np.sum(np.asarray(self.time_w) * np.asarray(tangent_vals))) if compute_tangent else float("nan")
        full = float(np.sum(np.asarray(self.time_w) * np.asarray(full_vals))) if compute_full else float("nan")
        tangent_full_gap = (full - tangent) if (compute_tangent and compute_full and np.isfinite(tangent) and np.isfinite(full)) else float("nan")
        lower_bound_violation = max(tangent - full, 0.0) if np.isfinite(tangent_full_gap) else float("nan")
        if not all_valid:
            law = tangent = full = float("nan")
            tangent_full_gap = lower_bound_violation = float("nan")
        result = {
            "trial": int(trial),
            "alpha": float(bank.alphas[trial]),
            "valid": all_valid,
            "invalid_reason": None if all_valid else "calibration_ess_hull_or_identifiability_gate",
            "law_risk": law,
            "tangent_action": tangent,
            "full_action": full,
            "max_calibration_residual": float(max_resid),
            "min_ess_fraction": float(min_ess),
            "max_hull_violation": float(max_hull),
            "feasibility_projection_distance": float(rec["projection_distance"]),
            "max_poisson_relative_residual": float(max_poisson) if compute_full else float("nan"),
            "max_tangent_compatibility_residual": float(max_tangent_compat) if compute_tangent else float("nan"),
            "min_covariance_eigenvalue": float(min_cov_eig) if compute_full else float("nan"),
            "tangent_full_gap": float(tangent_full_gap),
            "tangent_lower_bound_violation": float(lower_bound_violation),
        }
        self._exact_trial_result_cache[trial_cache_key] = result
        return dict(result)

    def exact_finite_result(self, eta: Array, bank: TrialBank) -> dict[str, Any]:
        cache_key = (self._exact_key(eta), id(bank))
        if cache_key in self._exact_finite_result_cache:
            return self._exact_finite_result_cache[cache_key]
        rows = [
            self._exact_trial_result(
                eta, bank, r, compute_law=True, compute_tangent=False, compute_full=False
            )
            for r in range(int(bank.masses.shape[0]))
        ]
        valid = bool(all(row["valid"] and np.isfinite(row["law_risk"]) for row in rows))
        value = float(np.mean([row["law_risk"] for row in rows])) if valid else float("inf")
        result = {"valid": valid, "value": value, "rows": rows}
        self._exact_finite_result_cache[cache_key] = result
        return result

    def exact_tangent_result(self, eta: Array, bank: TrialBank) -> dict[str, Any]:
        cache_key = (self._exact_key(eta), id(bank))
        if cache_key in self._exact_tangent_result_cache:
            return self._exact_tangent_result_cache[cache_key]
        rows = [
            self._exact_trial_result(
                eta, bank, r, compute_law=False, compute_tangent=True, compute_full=False
            )
            for r in range(int(bank.masses.shape[0]))
        ]
        valid = bool(all(row["valid"] and np.isfinite(row["tangent_action"]) for row in rows))
        value = float(np.mean([row["tangent_action"] for row in rows])) if valid else float("inf")
        result = {"valid": valid, "value": value, "rows": rows}
        self._exact_tangent_result_cache[cache_key] = result
        return result

    def exact_full_result(
        self, eta: Array, bank: TrialBank, *, trial_count: int | None = None
    ) -> dict[str, Any]:
        count = int(bank.masses.shape[0]) if trial_count is None else min(
            int(trial_count), int(bank.masses.shape[0])
        )
        cache_key = (self._exact_key(eta), id(bank), count)
        if cache_key in self._exact_full_result_cache:
            return self._exact_full_result_cache[cache_key]
        rows = [
            self._exact_trial_result(
                eta, bank, r, compute_law=False, compute_tangent=False, compute_full=True
            )
            for r in range(count)
        ]
        valid = bool(all(row["valid"] and np.isfinite(row["full_action"]) for row in rows))
        value = float(np.mean([row["full_action"] for row in rows])) if valid else float("inf")
        result = {"valid": valid, "value": value, "rows": rows}
        self._exact_full_result_cache[cache_key] = result
        return result

    def evaluate_trials_exact(self, eta: Array, bank: TrialBank) -> list[dict[str, Any]]:
        """Full-fidelity validation rows with exact feasibility and strict validity."""
        eta = self.family.canonicalize(eta)
        eta_deg = _canonical_deg(eta)
        out = []
        for r in range(int(bank.masses.shape[0])):
            row = self._exact_trial_result(
                eta, bank, r, compute_law=True, compute_tangent=True, compute_full=True
            )
            row["theta1_deg"] = float(eta_deg[0])
            row["theta2_deg"] = float(eta_deg[1])
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Hard empirical projection / law state
    # ------------------------------------------------------------------

    def _project_state(self, t_idx: int, phi_nodes_t: Array, target: Array, lam0: Array | None = None):
        """Hard projection only; no raster/KDE work."""
        return self.projector.project(
            phi_nodes_t,
            self.reference_weights[t_idx],
            target,
            lam0=lam0,
        )

    def _raster_mass_from_weights(self, t_idx: int, weights: Array) -> Array:
        """Raster the already-calibrated projected law with the canonical KDE."""
        zero = jnp.zeros_like(weights)
        ras = rasterize_projected_particles(
            self.reference_nodes[t_idx], weights, zero, self.grid, self.raster_cfg
        )
        return ras.mass

    def _project_mass(self, t_idx: int, phi_nodes_t: Array, target: Array, lam0: Array | None = None):
        state = self._project_state(t_idx, phi_nodes_t, target, lam0)
        return state, self._raster_mass_from_weights(t_idx, state.weights)

    def _validity(
        self,
        calibration: Array,
        ess: Array,
        projection_distance: Array,
        poisson_rel: Array | None = None,
    ):
        """Scientific validity gate for a reconstructed finite/noisy trial.

        Matches the learned-reference finite-measurement convention: calibration,
        relative ESS, and frozen-reference in-domain mass determine validity.
        Poisson residual is a numerical diagnostic by default, not a scientific
        inclusion gate.  If a future experiment explicitly declares
        ``validity.max_poisson_relative_residual``, that additional gate is applied.
        """
        validity = self.cfg["validity"]
        ok = (
            (calibration <= float(validity.get("max_finite_calibration_resid", 1.0e-3)))
            & (ess >= float(validity.get("min_ess_fraction", 0.03)))
            & (jnp.min(self.reference_base_mass) >= float(validity.get("min_in_domain_base_mass", 0.995)))
        )

        poisson_gate = validity.get("max_poisson_relative_residual")
        if poisson_rel is not None and poisson_gate is not None:
            ok = ok & (poisson_rel <= float(poisson_gate))

        return ok

    # ------------------------------------------------------------------
    # Authoritative law objectives used for final re-scoring
    # ------------------------------------------------------------------

    def population_loss(self, eta: Array) -> Array:
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, _ = self._geometry(eta)
        alphas, alpha_w = self.population.alpha_quadrature(
            int(self.cfg["population"].get("alpha_quadrature_n", 5))
        )
        tw = self.time_w[self.population_idx]
        tw = tw / jnp.sum(tw)

        def one_alpha(alpha):
            truth = self.population.masses(self.times, alpha).reshape((len(self.times), -1))
            targets = truth @ phi_grid
            lam = jnp.zeros(2, dtype=jnp.float64)
            vals = []
            max_resid = jnp.asarray(0.0)
            min_ess = jnp.asarray(jnp.inf)
            for t_idx in np.asarray(self.population_idx, dtype=int):
                st, mass = self._project_mass(t_idx, phi_nodes[t_idx], targets[t_idx], lam)
                lam = st.lam
                vals.append(
                    gaussian_mmd2_grid_mass(
                        mass,
                        truth[t_idx].reshape((self.grid.n, self.grid.n)),
                        self.mmd_kernel,
                    )
                )
                max_resid = jnp.maximum(max_resid, jnp.linalg.norm(st.residual))
                min_ess = jnp.minimum(min_ess, st.ess_fraction)
            vals = jnp.stack(vals)
            valid = (
                (max_resid <= float(self.cfg.get("validity", {}).get("max_population_calibration_resid", 1.0e-5)))
                & (min_ess >= float(self.cfg.get("validity", {}).get("min_ess_fraction", 0.03)))
                & (jnp.min(self.reference_base_mass) >= float(self.cfg.get("validity", {}).get("min_in_domain_base_mass", 0.995)))
            )
            score = jnp.sum(tw * vals)
            return jnp.where(valid, score, score + 1.0e3)

        vals = jax.vmap(one_alpha)(alphas)
        return jnp.sum(alpha_w * vals)

    def finite_risk(self, eta: Array, bank: TrialBank) -> Array:
        """Authoritative finite law risk without irrelevant raster work.

        Every time node is still calibrated for the residual/ESS validity gate.
        KDE/MMD is performed only on held-out nodes because only those terms enter R.
        Sensor geometry is formed once and reused across all trials.
        """
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, _ = self._geometry(eta)
        held_w = self.time_w[self.heldout_idx]
        held_w = held_w / jnp.sum(held_w)
        held_set = set(map(int, np.asarray(self.heldout_idx).tolist()))
        rows = []
        for trial in range(int(bank.masses.shape[0])):
            rec = self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)
            truth = bank.masses[trial]
            lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
            held_vals = []
            max_resid = jnp.asarray(0.0)
            min_ess = jnp.asarray(jnp.inf)
            for t_idx in range(len(self.times)):
                st = self._project_state(t_idx, phi_nodes[t_idx], rec.c[t_idx], lam)
                lam = st.lam
                max_resid = jnp.maximum(max_resid, jnp.linalg.norm(st.residual))
                min_ess = jnp.minimum(min_ess, st.ess_fraction)
                if t_idx in held_set:
                    mass = self._raster_mass_from_weights(t_idx, st.weights)
                    held_vals.append(
                        gaussian_mmd2_grid_mass(
                            mass,
                            truth[t_idx].reshape((self.grid.n, self.grid.n)),
                            self.mmd_kernel,
                        )
                    )
            risk = jnp.sum(held_w * jnp.stack(held_vals))
            valid = self._validity(max_resid, min_ess, rec.projection_distance)
            rows.append(jnp.where(valid, risk, risk + 1.0e3))
        return jnp.mean(jnp.stack(rows))

    # ------------------------------------------------------------------
    # Action objectives
    # ------------------------------------------------------------------

    def _one_trial_metrics_from_geometry(
        self,
        eta: Array,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
        trial: int | Array,
        *,
        full: bool,
    ) -> TrialMetrics:
        rec = self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)
        truth = bank.masses[trial]

        law_vals = []
        tangent_vals = []
        full_vals = []
        max_resid = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        max_poisson = jnp.asarray(0.0)

        held_set = set(map(int, np.asarray(self.heldout_idx).tolist()))
        held_weights = self.time_w[self.heldout_idx]
        held_weights = held_weights / jnp.sum(held_weights)

        for t_idx in range(len(self.times)):
            st = particle_mfsi_state(
                phi=phi_nodes[t_idx],
                grad_phi=grad_nodes[t_idx],
                velocity=self.reference_velocity[t_idx],
                base_weights=self.reference_weights[t_idx],
                target=rec.c[t_idx],
                target_dot=rec.c_dot[t_idx],
                projector=self.projector,
                cfg=self.particle_cfg,
            )
            max_resid = jnp.maximum(max_resid, jnp.linalg.norm(st.projection.residual))
            min_ess = jnp.minimum(min_ess, st.projection.ess_fraction)
            tangent_vals.append(st.tangent_action)

            ras = rasterize_projected_particles(
                self.reference_nodes[t_idx],
                st.projection.weights,
                st.forcing,
                self.grid,
                self.raster_cfg,
            )
            if t_idx in held_set:
                law_vals.append(
                    gaussian_mmd2_grid_mass(
                        ras.mass,
                        truth[t_idx].reshape((self.grid.n, self.grid.n)),
                        self.mmd_kernel,
                    )
                )

            if full:
                pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_cfg)
                full_vals.append(pois.action)
                max_poisson = jnp.maximum(max_poisson, pois.relative_residual)

        tangent = jnp.sum(self.time_w * jnp.stack(tangent_vals))
        full_action = (
            jnp.sum(self.time_w * jnp.stack(full_vals)) if full else jnp.asarray(jnp.nan)
        )
        law = jnp.sum(held_weights * jnp.stack(law_vals))
        valid = self._validity(
            max_resid, min_ess, rec.projection_distance, max_poisson if full else None
        )
        return TrialMetrics(
            law_risk=law,
            tangent_action=tangent,
            full_action=full_action,
            max_calibration_residual=max_resid,
            min_ess_fraction=min_ess,
            max_projection_distance=rec.projection_distance,
            max_poisson_relative_residual=max_poisson,
            valid=valid,
        )

    def _one_trial_metrics(self, eta: Array, bank: TrialBank, trial: int, *, full: bool) -> TrialMetrics:
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, grad_nodes = self._geometry(eta)
        return self._one_trial_metrics_from_geometry(
            eta, phi_grid, phi_nodes, grad_nodes, bank, trial, full=full
        )

    def _particle_forcing_only(
        self,
        *,
        phi: Array,
        grad_phi: Array,
        velocity: Array,
        base_weights: Array,
        target: Array,
        target_dot: Array,
        lam0: Array,
    ):
        """Particle MFSI forcing without the tangent Gram/action calculation.

        Stage 4 only needs projected weights and h for the Poisson solve.  The
        tangent action is intentionally omitted here; the authoritative evaluator
        still uses ``particle_mfsi_state`` for final reporting.
        """
        projection = self.projector.project(phi, base_weights, target, lam0=lam0)
        w = projection.weights
        lam = projection.lam

        # m_i = J Phi(x_i) u_i
        m = jnp.einsum("nmd,nd->nm", grad_phi, velocity)
        mean_m = jnp.sum(w[:, None] * m, axis=0)
        g = m @ lam
        mean_g = jnp.sum(w * g)
        centered_phi = phi - projection.moments[None, :]
        cov_phi_g = jnp.sum(w[:, None] * centered_phi * (g - mean_g)[:, None], axis=0)

        cov = projection.covariance + float(self.particle_cfg.covariance_ridge) * jnp.eye(
            phi.shape[-1], dtype=jnp.float64
        )
        rhs = target_dot - mean_m - cov_phi_g
        lam_dot = jnp.linalg.solve(cov, rhs)
        forcing = centered_phi @ lam_dot + g - mean_g
        forcing = forcing - jnp.sum(w * forcing)
        return projection, forcing

    def _one_trial_full_action_gradient(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
        trial: int | Array,
    ) -> Array:
        """Reduced-cost differentiable full action used only by stage-4 Adam."""
        rec = self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)
        lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
        actions = []
        max_resid = jnp.asarray(0.0, dtype=jnp.float64)
        min_ess = jnp.asarray(jnp.inf, dtype=jnp.float64)

        # Static Python iteration over a small configured subset (typically 7 of 21).
        for t_idx in np.asarray(self.full_gradient_time_idx, dtype=np.int32).tolist():
            projection, forcing = self._particle_forcing_only(
                phi=phi_nodes[t_idx],
                grad_phi=grad_nodes[t_idx],
                velocity=self.reference_velocity[t_idx],
                base_weights=self.reference_weights[t_idx],
                target=rec.c[t_idx],
                target_dot=rec.c_dot[t_idx],
                lam0=lam,
            )
            lam = projection.lam
            max_resid = jnp.maximum(max_resid, jnp.linalg.norm(projection.residual))
            min_ess = jnp.minimum(min_ess, projection.ess_fraction)
            ras = rasterize_projected_particles(
                self.reference_nodes[t_idx],
                projection.weights,
                forcing,
                self.full_gradient_grid,
                self.raster_cfg,
            )
            pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_gradient_cfg)
            actions.append(pois.action)

        action = jnp.sum(self.full_gradient_time_w * jnp.stack(actions))
        valid = self._validity(max_resid, min_ess, rec.projection_distance)
        return jnp.where(valid, action, action + 1.0e5)

    def _one_trial_full_action_gradient_systems(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
        trial: int | Array,
    ) -> tuple[Array, Array, Array]:
        """Collect one trial's q/h systems while retaining lambda warm starts."""
        rec = self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, trial)
        lam = jnp.zeros(phi_nodes.shape[-1], dtype=jnp.float64)
        q_rows = []
        h_rows = []
        max_resid = jnp.asarray(0.0, dtype=jnp.float64)
        min_ess = jnp.asarray(jnp.inf, dtype=jnp.float64)

        for t_idx in np.asarray(self.full_gradient_time_idx, dtype=np.int32).tolist():
            projection, forcing = self._particle_forcing_only(
                phi=phi_nodes[t_idx],
                grad_phi=grad_nodes[t_idx],
                velocity=self.reference_velocity[t_idx],
                base_weights=self.reference_weights[t_idx],
                target=rec.c[t_idx],
                target_dot=rec.c_dot[t_idx],
                lam0=lam,
            )
            lam = projection.lam
            max_resid = jnp.maximum(max_resid, jnp.linalg.norm(projection.residual))
            min_ess = jnp.minimum(min_ess, projection.ess_fraction)
            ras = rasterize_projected_particles(
                self.reference_nodes[t_idx],
                projection.weights,
                forcing,
                self.full_gradient_grid,
                self.raster_cfg,
            )
            q_rows.append(ras.q)
            h_rows.append(ras.h)

        valid = self._validity(max_resid, min_ess, rec.projection_distance)
        return jnp.stack(q_rows), jnp.stack(h_rows), valid

    def _full_action_gradient_tesseract(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
    ) -> Array:
        """Solve every trial/time system with one explicit batched native call."""
        from mfsi.poisson_tesseract import solve_weighted_poisson_batch_tesseract

        q_batch, h_batch, valid = self._full_action_gradient_system_batch(
            phi_grid, phi_nodes, grad_nodes, bank
        )

        # This is the only native/Tesseract call in one proxy objective evaluation.
        psi_batch = solve_weighted_poisson_batch_tesseract(
            q_batch, h_batch, self.poisson_gradient_cfg
        )
        physical_actions = jax.vmap(
            lambda psi, q: self.poisson_gradient_cfg.cell_area
            * jnp.sum(
                psi
                * weighted_laplacian(psi, q, self.poisson_gradient_cfg.dx)
            )
        )(psi_batch, q_batch)
        actions_by_trial = physical_actions.reshape(
            (int(bank.masses.shape[0]), len(self.full_gradient_time_idx))
        )
        actions_by_trial = jnp.sum(
            actions_by_trial * self.full_gradient_time_w[None, :], axis=1
        )
        return jnp.mean(jnp.where(valid, actions_by_trial, actions_by_trial + 1.0e5))

    def _full_action_gradient_system_batch(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
    ) -> tuple[Array, Array, Array]:
        """Return systems in trial-major, selected-time-minor order."""
        trial_rows = [
            self._one_trial_full_action_gradient_systems(
                phi_grid, phi_nodes, grad_nodes, bank, trial
            )
            for trial in range(int(bank.masses.shape[0]))
        ]
        q_batch = jnp.concatenate([row[0] for row in trial_rows], axis=0)
        h_batch = jnp.concatenate([row[1] for row in trial_rows], axis=0)
        valid = jnp.stack([row[2] for row in trial_rows])
        return q_batch, h_batch, valid

    def full_action_gradient(self, eta: Array, bank: TrialBank) -> Array:
        """Multi-fidelity full-action objective for stage 4 optimization only.

        Geometry is computed once per eta, trials are vmapped, Newton multipliers
        are warm-started through the selected time nodes, and the Poisson solve uses
        the configured gradient-only tolerance.  Final selection/validation never
        reports this proxy; they use ``full_action`` at full fidelity.
        """
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, grad_nodes = self._geometry(eta)
        if self.full_gradient_poisson_backend == "tesseract_cpp":
            return self._full_action_gradient_tesseract(
                phi_grid, phi_nodes, grad_nodes, bank
            )
        # The optimizer already vmaps across starts. Keep the tiny CRN prefix as a
        # static loop to avoid a large nested-vmap CG graph and excessive GPU memory.
        values = [
            self._one_trial_full_action_gradient(phi_grid, phi_nodes, grad_nodes, bank, r)
            for r in range(int(bank.masses.shape[0]))
        ]
        return jnp.mean(jnp.stack(values))

    def _tangent_trials_from_geometry(
        self,
        phi_grid: Array,
        phi_nodes: Array,
        grad_nodes: Array,
        bank: TrialBank,
    ) -> Array:
        """Exact tangent action for all CRN trials, batched across trials.

        T only requires the calibrated projected weights, E_q[J Phi u], and the
        tangent Gram matrix.  It does not require lambda_dot, forcing h, KDE,
        held-out MMD, or a Poisson solve.
        """
        R = int(bank.masses.shape[0])
        trial_idx = jnp.arange(R, dtype=jnp.int32)
        rec = jax.vmap(
            lambda r: self._reconstruct_from_geometry(phi_grid, phi_nodes, bank, r)
        )(trial_idx)

        M = int(phi_nodes.shape[-1])
        lam0 = jnp.zeros((R, M), dtype=jnp.float64)
        action0 = jnp.zeros((R,), dtype=jnp.float64)
        max_resid0 = jnp.zeros((R,), dtype=jnp.float64)
        min_ess0 = jnp.full((R,), jnp.inf, dtype=jnp.float64)
        eye = jnp.eye(M, dtype=jnp.float64)

        def step(carry, xs):
            lam, action, max_resid, min_ess = carry
            phi_t, grad_t, velocity_t, base_t, target_t, target_dot_t, w_t = xs

            def one(lam_i, target_i, target_dot_i):
                proj = self.projector.project(phi_t, base_t, target_i, lam0=lam_i)
                m = jnp.einsum("nmd,nd->nm", grad_t, velocity_t)
                mean_m = jnp.sum(proj.weights[:, None] * m, axis=0)
                r = mean_m - target_dot_i
                G = jnp.einsum(
                    "n,nmd,nkd->mk", proj.weights, grad_t, grad_t
                ) + float(self.particle_cfg.tangent_ridge) * eye
                tangent = r @ jnp.linalg.solve(G, r)
                return (
                    proj.lam,
                    tangent,
                    jnp.linalg.norm(proj.residual),
                    proj.ess_fraction,
                )

            lam, tangent, resid, ess = jax.vmap(one)(lam, target_t, target_dot_t)
            return (
                lam,
                action + w_t * tangent,
                jnp.maximum(max_resid, resid),
                jnp.minimum(min_ess, ess),
            ), None

        xs = (
            phi_nodes,
            grad_nodes,
            self.reference_velocity,
            self.reference_weights,
            jnp.swapaxes(rec.c, 0, 1),
            jnp.swapaxes(rec.c_dot, 0, 1),
            self.time_w,
        )
        (_, action, max_resid, min_ess), _ = jax.lax.scan(
            step, (lam0, action0, max_resid0, min_ess0), xs
        )
        valid = self._validity(max_resid, min_ess, rec.projection_distance)
        valid = valid & (
            rec.endpoint_feasibility_violation
            <= float(self.cfg.get("feasibility", {}).get("feasibility_tol", self.cfg.get("feasibility", {}).get("tol", 1e-9)))
        )
        return jnp.where(valid, action, action + 1.0e5)

    def tangent_action(self, eta: Array, bank: TrialBank) -> Array:
        """Authoritative tangent objective, exact but transport-only."""
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, grad_nodes = self._geometry(eta)
        return jnp.mean(self._tangent_trials_from_geometry(phi_grid, phi_nodes, grad_nodes, bank))

    def tangent_action_gradient(self, eta: Array, bank: TrialBank) -> Array:
        """Same exact T definition on a fixed CRN prefix for stage-3 gradients."""
        return self.tangent_action(eta, bank)

    def full_action(self, eta: Array, bank: TrialBank) -> Array:
        eta = self.family.canonicalize(eta)
        phi_grid, phi_nodes, grad_nodes = self._geometry(eta)
        vals = []
        for trial in range(int(bank.masses.shape[0])):
            row = self._one_trial_metrics_from_geometry(
                eta, phi_grid, phi_nodes, grad_nodes, bank, trial, full=True
            )
            vals.append(jnp.where(row.valid, row.full_action, row.full_action + 1.0e5))
        return jnp.mean(jnp.stack(vals))

    def evaluate_trials(self, eta: Array, bank: TrialBank, *, full: bool = True) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        eta = self.family.canonicalize(eta)
        eta_deg = _canonical_deg(eta)
        phi_grid, phi_nodes, grad_nodes = self._geometry(eta)
        for trial in range(int(bank.masses.shape[0])):
            m = self._one_trial_metrics_from_geometry(
                eta, phi_grid, phi_nodes, grad_nodes, bank, trial, full=full
            )
            rows.append({
                "trial": int(trial),
                "alpha": float(bank.alphas[trial]),
                "theta1_deg": float(eta_deg[0]),
                "theta2_deg": float(eta_deg[1]),
                "law_risk": float(m.law_risk),
                "tangent_action": float(m.tangent_action),
                "full_action": float(m.full_action),
                "max_calibration_residual": float(m.max_calibration_residual),
                "min_ess_fraction": float(m.min_ess_fraction),
                "feasibility_projection_distance": float(m.max_projection_distance),
                "max_poisson_relative_residual": float(m.max_poisson_relative_residual),
                "valid": bool(m.valid),
            })
        return rows


# -----------------------------------------------------------------------------
# Tangent/full selection after the accelerated law stages
# -----------------------------------------------------------------------------


def _exact_law_feasible(exp: ToyExperiment, eta: Array, bank: TrialBank, L_max: float, R_max: float):
    L = float(exp.population_loss(eta))
    if L > L_max + 1e-12:
        return False, L, float("inf")
    R = float(exp.finite_risk(eta, bank))
    return R <= R_max + 1e-12, L, R


def _design_key(family, eta: Array) -> tuple[float, ...]:
    eta = family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
    return tuple(np.round(np.asarray(eta, dtype=np.float64), 12))


def _dedupe_designs(family, designs: Array) -> Array:
    """Canonicalize and remove duplicate designs while preserving first occurrence.

    This matters when Population/Law/Tangent coincide: without de-duplication the
    stage-3/4 optimizer can spend multiple expensive Adam trajectories on the exact
    same incumbent while the log misleadingly reports them as distinct starts.
    """
    arr = jnp.asarray(designs, dtype=jnp.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = []
    seen: set[tuple[float, ...]] = set()
    for eta in arr:
        eta = family.canonicalize(eta)
        key = _design_key(family, eta)
        if key not in seen:
            seen.add(key)
            out.append(eta)
    if not out:
        return jnp.zeros((0, arr.shape[-1]), dtype=jnp.float64)
    return jnp.stack(out)


def _select_action_design(
    *,
    name: str,
    exp: ToyExperiment,
    objective,
    fast_objective,
    exact_result,
    starts: Array,
    audit_starts: Array,
    selection_bank: TrialBank,
    L_max: float,
    R_max: float,
    fast_population,
    fast_risk,
    fast_joint=None,
    population_eta: Array,
    law_eta: Array,
    cfg: OptimizerConfig,
    optimize_start_count: int | None = None,
    mandatory_start_count: int = 1,
    exact_prescreen_result=None,
    exact_rescore: int | None = None,
    mandatory_exact: Array | None = None,
    exact_audit_limit: int | None = None,
    min_exact_law_valid: int = 1,
    min_exact_finalists: int = 1,
    vectorize_starts: bool = True,
) -> tuple[Array, list[dict[str, Any]]]:
    """Search cheaply; decide only with exact law/action evaluations.

    Search objectives are deliberately multi-fidelity.  The candidate actually
    selected must pass the authoritative ConvexHull, hard I-projection, L and R
    checks.  ``exact_audit_limit`` only controls how many proxy-ranked candidates
    receive that expensive audit; mandatory incumbents are always retained.
    """
    # The manuscript uses additive tolerances.  Map the same additive slack onto
    # the fast surrogate around its own incumbent, rather than reintroducing a
    # multiplicative constraint in the search graph.
    exact_L_anchor = float(exp.exact_population_result(population_eta)["value"])
    exact_R_anchor = float(exp.exact_finite_result(law_eta, selection_bank)["value"])
    epsilon_l = max(float(L_max) - exact_L_anchor, 0.0)
    epsilon_r = max(float(R_max) - exact_R_anchor, 0.0)
    fast_population_eval = jax.jit(fast_population)
    fast_risk_eval = jax.jit(fast_risk)
    fast_joint_eval = jax.jit(fast_joint) if fast_joint is not None else None
    fast_L_anchor = float(fast_population_eval(population_eta))
    fast_R_anchor = float(fast_risk_eval(law_eta))
    fast_L_limit = fast_L_anchor + epsilon_l
    fast_R_limit = fast_R_anchor + epsilon_r
    min_sep = math.radians(float(exp.cfg["measurement"]["min_sep_deg"]))

    # Use the same normalized slack both in the differentiable penalty and when
    # deciding which basins deserve expensive gradient trajectories.  Previously
    # start allocation ignored L/R feasibility and could optimize attractive but
    # scientifically doomed low-action starts.
    l_scale = max(epsilon_l, 1.0e-6 * max(abs(fast_L_anchor), 1.0), 1.0e-10)
    r_scale = max(epsilon_r, 1.0e-6 * max(abs(fast_R_anchor), 1.0), 1.0e-10)

    if fast_joint is not None:
        def law_intersection_violation(eta):
            L, R = fast_joint(eta)
            # Normalize violations by the *allowed additive slack*.  Scaling by
            # |L| or |R| made an epsilon-sized violation look numerically tiny
            # (e.g. 2e-4 / 6e-2), so the action gradient could overwhelm the
            # constraint penalty and generate endpoints that exact auditing later
            # rejected.  This scaling makes one epsilon of violation O(1).
            return jnp.maximum((L - fast_L_limit) / l_scale, (R - fast_R_limit) / r_scale)
        constraints = (
            (projective_separation_violation(min_sep), 0.0),
            (law_intersection_violation, 0.0),
        )
    else:
        constraints = (
            (projective_separation_violation(min_sep), 0.0),
            (fast_population, fast_L_limit),
            (fast_risk, fast_R_limit),
        )

    screen_objective = jax.jit(fast_objective)
    raw_starts = jnp.asarray(starts, dtype=jnp.float64)
    raw_all_starts = jnp.asarray(audit_starts, dtype=jnp.float64)
    starts = _dedupe_designs(exp.family, raw_starts)
    all_starts = _dedupe_designs(exp.family, raw_all_starts)
    distinct_screen_start_count = int(all_starts.shape[0])

    # Mandatory starts are defined by the caller's leading incumbents, but count
    # UNIQUE incumbents.  If Law=Population=Tangent, this is one start, not three.
    raw_mandatory = raw_starts[: min(mandatory_start_count, int(raw_starts.shape[0]))]
    mandatory_starts = _dedupe_designs(exp.family, raw_mandatory)
    mandatory_keys_for_optimization = {
        _design_key(exp.family, eta) for eta in mandatory_starts
    }

    # Spend gradient trajectories only when explicitly requested.  Rank nonmandatory
    # basins by normalized fast L/R violation first, action proxy second.  This
    # allocates reverse-CG work to starts that have a realistic chance of surviving
    # the authoritative scientific screens.
    if cfg.steps > 0:
        if optimize_start_count is not None and int(starts.shape[0]) > int(optimize_start_count):
            mandatory = [eta for eta in mandatory_starts]
            scored = []
            for eta_i in starts:
                key = _design_key(exp.family, eta_i)
                if key in mandatory_keys_for_optimization:
                    continue
                if fast_joint_eval is not None:
                    fL, fR = fast_joint_eval(eta_i)
                else:
                    fL, fR = fast_population_eval(eta_i), fast_risk_eval(eta_i)
                law_violation = max(
                    0.0,
                    (float(fL) - fast_L_limit) / l_scale,
                    (float(fR) - fast_R_limit) / r_scale,
                )
                sep_violation = max(0.0, float(projective_separation_violation(min_sep)(eta_i)))
                scored.append((max(law_violation, sep_violation), float(screen_objective(eta_i)), eta_i))
            scored.sort(key=lambda x: (x[0], x[1]))
            chosen = list(mandatory)
            for _, _, eta_i in scored:
                if len(chosen) >= int(optimize_start_count):
                    break
                if _design_key(exp.family, eta_i) not in {_design_key(exp.family, x) for x in chosen}:
                    chosen.append(eta_i)
            starts = jnp.stack(chosen) if chosen else starts[:1]

        gradient_start_count = int(starts.shape[0])
        candidates = optimize_multistart_candidates(
            objective,
            starts,
            cfg,
            constraints=constraints,
            canonicalize=exp.family.canonicalize,
            vectorize_starts=vectorize_starts,
        )
    else:
        gradient_start_count = 0
        candidates = []

    pool: dict[tuple[float, float], Array] = {}
    for eta in all_starts:
        eta = exp.family.canonicalize(eta)
        pool[tuple(np.round(np.asarray(eta), 12))] = eta
    for cand in candidates:
        eta = exp.family.canonicalize(cand.eta)
        pool[tuple(np.round(np.asarray(eta), 12))] = eta

    mandatory_keys: set[tuple[float, float]] = set()
    raw_audit_mandatory = raw_all_starts[: min(mandatory_start_count, int(raw_all_starts.shape[0]))]
    for eta in _dedupe_designs(exp.family, raw_audit_mandatory):
        mandatory_keys.add(_design_key(exp.family, eta))
    if mandatory_exact is not None:
        mandatory_arr = jnp.asarray(mandatory_exact, dtype=jnp.float64)
        if mandatory_arr.ndim == 1:
            mandatory_arr = mandatory_arr[None, :]
        for eta in mandatory_arr:
            eta = exp.family.canonicalize(eta)
            mandatory_keys.add(tuple(np.round(np.asarray(eta), 12)))
            pool[tuple(np.round(np.asarray(eta), 12))] = eta

    # Proxy-rank before robust exact L/R auditing.  We retain proxy-infeasible
    # points at the back of the ranking because the fast and exact evaluators are
    # intentionally not identical near feasibility boundaries.
    screened = []
    for key, eta in pool.items():
        if float(projective_separation_violation(min_sep)(eta)) > 0.0:
            continue
        if fast_joint_eval is not None:
            fL, fR = fast_joint_eval(eta)
        else:
            fL, fR = fast_population_eval(eta), fast_risk_eval(eta)
        proxy = float(screen_objective(eta))
        violation = max(
            0.0,
            float(fL) - fast_L_limit,
            float(fR) - fast_R_limit,
        )
        screened.append((key, eta, violation, proxy))
    screened.sort(key=lambda row: (row[2], row[3]))

    screened_candidate_count = len(screened)
    if exact_audit_limit is None:
        initial_keep = len(screened)
    else:
        initial_keep = max(1, min(int(exact_audit_limit), len(screened)))
    target_valid = max(1, int(min_exact_law_valid))

    # Audit the proxy-ranked tranche first, plus mandatory incumbents.  If too few
    # candidates survive the authoritative L/R gates, progressively expand the
    # audit rather than concluding from an accidentally tiny finalist set.
    initial_keys = {row[0] for row in screened[:initial_keep]} | mandatory_keys
    initial_rows = [row for row in screened if row[0] in initial_keys]
    remaining_rows = [row for row in screened if row[0] not in initial_keys]
    audit_order = initial_rows + remaining_rows

    law_valid_rows: list[dict[str, Any]] = []
    exact_law_audited_count = 0
    mandatory_audited = len(initial_rows)
    for _, eta, _, proxy in audit_order:
        if exact_law_audited_count >= mandatory_audited and len(law_valid_rows) >= target_valid:
            break
        exact_law_audited_count += 1
        pop = exp.exact_population_result(eta)
        if not pop["valid"] or float(pop["value"]) > L_max + 1.0e-12:
            continue
        fin = exp.exact_finite_result(eta, selection_bank)
        if not fin["valid"] or float(fin["value"]) > R_max + 1.0e-12:
            continue
        law_valid_rows.append({
            "eta": eta,
            "population_loss": float(pop["value"]),
            "finite_risk": float(fin["value"]),
            "screen_objective": float(proxy),
        })

    if not law_valid_rows:
        raise RuntimeError(f"No scientifically valid law-feasible {name} candidate after exact audit")

    if exact_prescreen_result is not None:
        pre_rows = []
        for row in law_valid_rows:
            rec = exact_prescreen_result(row["eta"])
            if rec["valid"] and np.isfinite(rec["value"]):
                rr = dict(row)
                rr["prescreen_objective"] = float(rec["value"])
                pre_rows.append(rr)
        if not pre_rows:
            raise RuntimeError(f"No valid {name} candidate survived exact action pre-screen")
        pre_rows.sort(key=lambda r: r["prescreen_objective"])

        keep = max(
            1,
            int(exact_rescore or len(pre_rows)),
            int(min_exact_finalists),
        )
        keep = min(keep, len(pre_rows))
        finalists = pre_rows[:keep]
        by_key = {tuple(np.round(np.asarray(r["eta"]), 12)): r for r in pre_rows}
        final_keys = {tuple(np.round(np.asarray(r["eta"]), 12)) for r in finalists}
        for key in mandatory_keys:
            if key in by_key and key not in final_keys:
                finalists.append(by_key[key])
                final_keys.add(key)
    else:
        finalists = law_valid_rows
        requested_rescore = max(int(exact_rescore or len(finalists)), int(min_exact_finalists))
        if len(finalists) > requested_rescore:
            finalists = sorted(finalists, key=lambda r: r["screen_objective"])
            base = finalists[: max(1, requested_rescore)]
            by_key = {tuple(np.round(np.asarray(r["eta"]), 12)): r for r in finalists}
            final_keys = {tuple(np.round(np.asarray(r["eta"]), 12)) for r in base}
            for key in mandatory_keys:
                if key in by_key and key not in final_keys:
                    base.append(by_key[key])
            finalists = base

    rows: list[dict[str, Any]] = []
    for row in finalists:
        rec = exact_result(row["eta"])
        rr = dict(row)
        rr["objective"] = float(rec["value"])
        rr["action_valid"] = bool(rec["valid"])
        if "prescreen_objective" in row:
            rr["prescreen_objective"] = row["prescreen_objective"]
        rr["distinct_screen_start_count"] = int(distinct_screen_start_count)
        rr["gradient_start_count"] = int(gradient_start_count)
        rr["screened_candidate_count"] = int(screened_candidate_count)
        rr["exact_law_audited_count"] = int(exact_law_audited_count)
        rr["exact_law_valid_count"] = int(len(law_valid_rows))
        rr["exact_finalist_count"] = int(len(finalists))
        rows.append(rr)

    valid_rows = [r for r in rows if r["action_valid"] and np.isfinite(r["objective"])]
    if not valid_rows:
        raise RuntimeError(f"No scientifically valid {name} candidate on the full action bank")
    best = min(valid_rows, key=lambda r: r["objective"])
    return jnp.asarray(best["eta"], dtype=jnp.float64), rows


def _local_design_cloud(family, centers: Array, *, count_per_center: int, radius_deg: float) -> Array:
    """Deterministic multiscale local cloud around UNIQUE incumbent centers.

    Tight additive law screens can leave a very narrow feasible neighborhood.  An
    area-uniform disk undersamples that neighborhood.  Quadratic radial spacing
    deliberately includes sub-degree perturbations while still reaching the full
    configured radius.  Duplicate centers are removed before any candidates are
    generated.
    """
    centers = _dedupe_designs(family, centers)
    count_per_center = max(0, int(count_per_center))
    if count_per_center == 0 or int(centers.shape[0]) == 0:
        return jnp.zeros((0, centers.shape[-1]), dtype=jnp.float64)
    max_radius = math.radians(float(radius_deg))
    out = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for c in np.asarray(centers, dtype=np.float64):
        for k in range(count_per_center):
            u = (k + 1.0) / float(count_per_center)
            radius = max_radius * (u * u)
            phase = golden * k
            delta = radius * np.asarray([math.cos(phase), math.sin(phase)], dtype=np.float64)
            out.append(family.canonicalize(jnp.asarray(c + delta, dtype=jnp.float64)))
    return _dedupe_designs(family, jnp.stack(out)) if out else jnp.zeros((0, centers.shape[-1]), dtype=jnp.float64)

def run_experiment(cfg: dict[str, Any], output_dir: Path, *, smoke: bool = False) -> dict[str, Any]:
    # Normalize the validity schema once, before hashing/caching or constructing
    # JAX objectives. These are the learned-reference reference-run scientific
    # validity gates; Poisson residual remains diagnostic unless explicitly gated.
    cfg = dict(cfg)
    cfg["validity"] = dict(cfg.get("validity", {}))
    cfg["validity"].setdefault("max_population_calibration_resid", 1.0e-5)
    cfg["validity"].setdefault("max_finite_calibration_resid", 1.0e-3)
    cfg["validity"].setdefault("min_ess_fraction", 0.03)
    cfg["validity"].setdefault("min_in_domain_base_mass", 0.995)
    cfg["optimization"] = dict(cfg.get("optimization", {}))
    if smoke and cfg["optimization"].get("full_exact_poisson_backend") == "tesseract_cpp":
        # Smoke deliberately caps the scientific CG solve at 30 iterations. Keep
        # its historical wiring check on JAX rather than weakening the native
        # backend's strict convergence contract for production runs.
        cfg["optimization"]["full_exact_poisson_backend"] = "jax"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Frozen reference: reuse only when checkpoint metadata matches the current
    # reference-training and endpoint configuration.
    # ------------------------------------------------------------------
    endpoints = ToyEndpointSource(
        radius=float(cfg["population"]["radius"]),
        sigma=float(cfg["population"]["sigma"]),
    )
    reference, checkpoint, reference_metadata = ensure_reference(endpoints, cfg, output_dir)

    # Build/load the frozen rollout bank. It is worth saving separately from the
    # NN checkpoint because rolling 10^4--10^5 particles through RK4 is itself an
    # avoidable future cost.
    times = jnp.linspace(0.0, 1.0, int(cfg["poisson"]["time_n"]), dtype=jnp.float64)
    reference_bank_path = output_dir / "reference_bank.npz"
    reference_bank_signature = hashlib.sha256(json.dumps({
        "checkpoint_sha256": _file_sha256(checkpoint),
        "reference": cfg.get("reference", {}),
        "times": np.asarray(times, dtype=np.float64).tolist(),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    reuse_reference_bank = False
    if reference_bank_path.exists():
        with np.load(reference_bank_path, allow_pickle=False) as z:
            saved_times = np.asarray(z["times"] if "times" in z.files else [], dtype=np.float64)
            ref_nodes_np = np.asarray(z["reference_particles"], dtype=np.float64)
            expected_n = None
            if str(cfg["reference"].get("bank_mode", "gauss-hermite")) == "gauss-hermite":
                gh = int(cfg["reference"].get("gh_order", 36))
                expected_n = 2 * gh * gh
            saved_signature = str(np.asarray(z["signature"]).item()) if "signature" in z.files else ""
            shape_ok = (
                ref_nodes_np.shape[0] == len(times)
                and (expected_n is None or ref_nodes_np.shape[1] == expected_n)
                and saved_times.shape == np.asarray(times).shape
                and np.allclose(saved_times, np.asarray(times), rtol=0.0, atol=1.0e-14)
                and saved_signature == reference_bank_signature
            )
            if shape_ok:
                print("[reference] reusing compatible reference_bank.npz", flush=True)
                ref_nodes = jnp.asarray(ref_nodes_np, dtype=jnp.float64)
                ref_velocity = jnp.asarray(z["reference_velocity"], dtype=jnp.float64)
                ref_weights = jnp.asarray(z["base_weights"], dtype=jnp.float64)
                reuse_reference_bank = True
            else:
                print("[reference] cached reference bank is incompatible; rebuilding", flush=True)
    if not reuse_reference_bank:
        print("[reference] rolling frozen reference bank", flush=True)
        ref_nodes, ref_velocity, ref_weights, _ = build_reference_bank(reference, endpoints, times, cfg)

    exp = ToyExperiment(
        cfg,
        reference,
        reference_nodes=ref_nodes,
        reference_velocity=ref_velocity,
        reference_weights=ref_weights,
    )

    if not reuse_reference_bank:
        save_reference_bank(
            output_dir,
            times=exp.times,
            nodes=exp.reference_nodes,
            velocity=exp.reference_velocity,
            base_weights=exp.reference_weights,
            in_domain_mask=exp.reference_in_domain,
            in_domain_base_mass=exp.reference_base_mass,
            signature=reference_bank_signature,
        )

    # ------------------------------------------------------------------
    # CRN banks: selection and validation are disjoint and persisted.
    # ------------------------------------------------------------------
    rnd = cfg["randomness"]
    law_trials = int(rnd["law_trials"])
    action_trials = int(rnd["action_trials"])
    validation_trials = int(rnd["validation_trials"])
    # Law and action selection share one CRN bank, but each stage must receive the
    # number of trials declared in the config.  The previous implementation built
    # only ``law_trials`` rows and then sliced it for ``action_trials``, silently
    # capping the action bank whenever action_trials > law_trials.
    selection_trials = max(law_trials, action_trials)
    finite_n = int(cfg["measurement"]["finite_n"])

    def load_or_make_bank(name: str, trials: int, namespace: int) -> TrialBank:
        path = output_dir / f"{name}_bank.npz"
        bank_signature = hashlib.sha256(json.dumps({
            "name": name,
            "trials": int(trials),
            "namespace": int(namespace),
            "seed": int(cfg["seed"]),
            "finite_n": int(finite_n),
            "acq_idx": np.asarray(exp.acq_idx, dtype=np.int32).tolist(),
            "times": np.asarray(exp.times, dtype=np.float64).tolist(),
            "population": cfg.get("population", {}),
            "measurement_noise": float(cfg["measurement"].get("obs_noise_std", 0.0)),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                masses = np.asarray(z["masses"])
                indices = np.asarray(z["sample_indices"])
                detector = np.asarray(z["detector_z"])
                alphas = np.asarray(z["alphas"])
                saved_acq = np.asarray(z["acquisition_indices"] if "acquisition_indices" in z.files else [], dtype=np.int32)
                saved_signature = str(np.asarray(z["signature"]).item()) if "signature" in z.files else ""
                compatible = (
                    saved_signature == bank_signature
                    and masses.shape[0] == int(trials)
                    and masses.shape[1] == len(exp.times)
                    and indices.shape[:2] == (int(trials), len(exp.acq_idx))
                    and indices.shape[2] == int(finite_n)
                    and detector.shape[:2] == (int(trials), len(exp.acq_idx))
                    and alphas.shape[0] == int(trials)
                    and np.array_equal(saved_acq, np.asarray(exp.acq_idx, dtype=np.int32))
                )
                if compatible:
                    print(f"[bank] reusing compatible {name}_bank.npz", flush=True)
                    return TrialBank(
                        masses=jnp.asarray(masses, dtype=jnp.float64),
                        sample_indices=jnp.asarray(indices, dtype=jnp.int32),
                        detector_z=jnp.asarray(detector, dtype=jnp.float64),
                        alphas=jnp.asarray(alphas, dtype=jnp.float64),
                    )
                print(f"[bank] cached {name}_bank.npz is incompatible; rebuilding", flush=True)
        bank = make_trial_bank(
            exp.population,
            exp.times,
            exp.acq_idx,
            finite_n=finite_n,
            trials=trials,
            seed=int(cfg["seed"]),
            namespace=namespace,
        )
        save_trial_bank(output_dir, name, bank, acq_idx=exp.acq_idx, signature=bank_signature)
        return bank

    selection = load_or_make_bank(
        "selection",
        selection_trials,
        int(rnd["selection_namespace"]),
    )
    validation = load_or_make_bank(
        "validation",
        validation_trials,
        int(rnd["validation_namespace"]),
    )

    law_bank = TrialBank(
        masses=selection.masses[:law_trials],
        sample_indices=selection.sample_indices[:law_trials],
        detector_z=selection.detector_z[:law_trials],
        alphas=selection.alphas[:law_trials],
    )
    action_bank = TrialBank(
        masses=selection.masses[:action_trials],
        sample_indices=selection.sample_indices[:action_trials],
        detector_z=selection.detector_z[:action_trials],
        alphas=selection.alphas[:action_trials],
    )
    if int(law_bank.masses.shape[0]) != law_trials:
        raise RuntimeError(
            f"law bank has {law_bank.masses.shape[0]} trials, expected {law_trials}"
        )
    if int(action_bank.masses.shape[0]) != action_trials:
        raise RuntimeError(
            f"action bank has {action_bank.masses.shape[0]} trials, expected {action_trials}"
        )
    grad_tangent_trials = int(
        cfg["optimization"].get("tangent_gradient_trials", min(action_trials, 8))
    )
    grad_tangent_trials = max(1, min(grad_tangent_trials, action_trials))
    grad_tangent_bank = TrialBank(
        masses=action_bank.masses[:grad_tangent_trials],
        sample_indices=action_bank.sample_indices[:grad_tangent_trials],
        detector_z=action_bank.detector_z[:grad_tangent_trials],
        alphas=action_bank.alphas[:grad_tangent_trials],
    )

    grad_full_trials = int(cfg["optimization"].get("full_gradient_trials", min(action_trials, 4)))
    grad_full_trials = max(1, min(grad_full_trials, action_trials))
    grad_action_bank = TrialBank(
        masses=action_bank.masses[:grad_full_trials],
        sample_indices=action_bank.sample_indices[:grad_full_trials],
        detector_z=action_bank.detector_z[:grad_full_trials],
        alphas=action_bank.alphas[:grad_full_trials],
    )

    full_prescreen_trials = int(
        cfg["optimization"].get("full_prescreen_trials", min(action_trials, 8))
    )
    full_prescreen_trials = max(1, min(full_prescreen_trials, action_trials))

    # ------------------------------------------------------------------
    # Smoke: exercise the complete scientific path once, but do not pretend a
    # two-step optimizer is a meaningful miniature experiment.
    # ------------------------------------------------------------------
    starts = random_projective_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 17),
        int(cfg["optimization"]["start_count"]),
        min_sep_rad=math.radians(float(cfg["measurement"]["min_sep_deg"])),
    )
    if smoke:
        # Smoke mode must return a genuinely valid exact scientific row, not merely
        # "not crash" with NaN sentinels.  The configured optimizer starts are a
        # deliberately small set and may all be infeasible, so perform a broader
        # objective-blind random feasibility probe.  No historical/known optimum
        # angles are used and no law/action value is consulted when choosing eta.
        smoke_bank = TrialBank(
            masses=selection.masses[:1],
            sample_indices=selection.sample_indices[:1],
            detector_z=selection.detector_z[:1],
            alphas=selection.alphas[:1],
        )

        probe_count = int(
            cfg.get("optimization", {}).get(
                "smoke_feasibility_probe_count",
                max(64, 8 * int(cfg["optimization"]["start_count"])),
            )
        )
        probe_count = max(probe_count, int(starts.shape[0]))
        extra = random_projective_starts(
            jax.random.PRNGKey(int(cfg["seed"]) + 17017),
            probe_count,
            min_sep_rad=math.radians(float(cfg["measurement"]["min_sep_deg"])),
        )
        probe_pool = jnp.concatenate([starts, extra], axis=0)

        # Canonical-key de-duplication is done on the host because this is a
        # non-differentiated smoke diagnostic.
        seen: set[tuple[float, float]] = set()
        candidates: list[Array] = []
        for i in range(int(probe_pool.shape[0])):
            candidate = exp.family.canonicalize(probe_pool[i])
            key = exp._exact_key(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

        smoke_rejections: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}
        eta = None
        precheck = None
        print(
            f"[smoke] searching {len(candidates)} objective-blind random designs "
            "for one exact-valid probe",
            flush=True,
        )
        for i, candidate in enumerate(candidates):
            try:
                exp._exact_polytope(candidate)
            except ExactFeasibilityError as exc:
                reason_counts[exc.reason] = reason_counts.get(exc.reason, 0) + 1
                if len(smoke_rejections) < 24:
                    smoke_rejections.append({
                        "index": int(i),
                        "eta_deg": _canonical_deg(candidate),
                        "reason": exc.reason,
                        "violation": float(exc.violation),
                    })
                continue

            # Cheap exact precheck: run the full hard calibration/ESS/hull validity
            # logic on one CRN, but skip MMD, tangent algebra, rasterization and
            # Poisson.  Only one candidate that passes this gate pays for the full
            # smoke evaluation below.
            row = exp._exact_trial_result(
                candidate,
                smoke_bank,
                0,
                compute_law=False,
                compute_tangent=False,
                compute_full=False,
            )
            if not bool(row["valid"]):
                reason = str(row.get("invalid_reason") or "exact_trial_invalid")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if len(smoke_rejections) < 24:
                    smoke_rejections.append({
                        "index": int(i),
                        "eta_deg": _canonical_deg(candidate),
                        "reason": reason,
                        "violation": float(row.get("max_hull_violation", float("nan"))),
                        "max_calibration_residual": float(
                            row.get("max_calibration_residual", float("nan"))
                        ),
                        "min_ess_fraction": float(row.get("min_ess_fraction", float("nan"))),
                    })
                continue

            eta = candidate
            precheck = row
            print(
                f"[smoke] exact-valid probe found at candidate {i}: "
                f"eta_deg={_canonical_deg(candidate)}",
                flush=True,
            )
            break

        if eta is None:
            diagnostic = {
                "schema_version": 1,
                "experiment": "toy_example",
                "smoke": True,
                "status": "no_exact_valid_probe",
                "candidate_count": int(len(candidates)),
                "reason_counts": reason_counts,
                "sample_rejections": smoke_rejections,
                "reference": {
                    "checkpoint": str(checkpoint),
                    "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass)),
                    "bank_shape": list(map(int, exp.reference_nodes.shape)),
                },
            }
            write_json(output_dir / "smoke_feasibility.json", diagnostic)
            raise RuntimeError(
                "Smoke could not find an exact-valid design among "
                f"{len(candidates)} broad objective-blind random probes. "
                "This is not a numerical-NaN failure; every probe was rejected by "
                "the declared feasibility/calibration/ESS gates. See "
                f"{output_dir / 'smoke_feasibility.json'} for reason counts and "
                "sample violations. If endpoint_outside_common_hull dominates, "
                "inspect/rebuild the frozen reference bank before trusting a proper run."
            )

        metrics = exp.evaluate_trials_exact(eta, smoke_bank)[0]
        if not bool(metrics["valid"]):
            raise RuntimeError(
                "Internal smoke inconsistency: the exact-valid precheck passed but the "
                "full exact smoke evaluation failed. This should be investigated rather "
                "than emitted as NaN output."
            )
        result = {
            "schema_version": 2,
            "experiment": "toy_example",
            "smoke": True,
            "config": cfg,
            "config_hash": _config_hash(cfg),
            "reference": {
                "checkpoint": str(checkpoint),
                "metadata": reference_metadata,
                "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass)),
            },
            "smoke_design_deg": _canonical_deg(eta),
            "smoke_feasibility_probe": {
                "found_exact_valid_probe": True,
                "candidate_count": int(len(candidates)),
                "reason_counts_before_success": reason_counts,
                "sample_rejections": smoke_rejections,
                "precheck": {
                    "max_calibration_residual": float(precheck["max_calibration_residual"]),
                    "min_ess_fraction": float(precheck["min_ess_fraction"]),
                    "max_hull_violation": float(precheck["max_hull_violation"]),
                },
            },
            "smoke_metrics": {
                "law_risk": float(metrics["law_risk"]),
                "tangent_action": float(metrics["tangent_action"]),
                "full_action": float(metrics["full_action"]),
                "max_calibration_residual": float(metrics["max_calibration_residual"]),
                "min_ess_fraction": float(metrics["min_ess_fraction"]),
                "max_poisson_relative_residual": float(metrics["max_poisson_relative_residual"]),
                "valid": bool(metrics["valid"]),
            },
        }
        write_json(output_dir / "result.json", result)
        _write_csv(output_dir / "result.candidate_summary.csv", [{
            "design": "smoke_probe",
            "theta1_deg": result["smoke_design_deg"][0],
            "theta2_deg": result["smoke_design_deg"][1],
            **result["smoke_metrics"],
        }])
        _write_csv(output_dir / "result.validation_trials.csv", [])
        return result

    # ------------------------------------------------------------------
    # 1/4 and 2/4: accelerated law stages.
    # ------------------------------------------------------------------
    law_selection = optimize_population_and_law(
        exp=exp,
        selection_bank=law_bank,
        dense_selection_bank=law_bank,
        starts=starts,
        output_dir=output_dir,
    )
    population_eta = law_selection["population_eta"]
    law_eta = law_selection["law_eta"]
    L_star = float(law_selection["L_star"])
    L_max = float(law_selection["L_max"])
    epsilon_l = float(law_selection.get("epsilon_l", L_max - L_star))
    R_star = float(law_selection["R_star"])
    R_max = float(law_selection["R_max"])
    epsilon_r = float(law_selection.get("epsilon_r", R_max - R_star))
    fast = law_selection["fast_evaluator"]
    gradient_fast = law_selection.get("gradient_evaluator", fast)

    # ------------------------------------------------------------------
    # 3/4 tangent action. Full scientific evaluator, but fast law constraints.
    # ------------------------------------------------------------------
    tangent_local = _local_design_cloud(
        exp.family,
        jnp.stack([law_eta, population_eta]),
        count_per_center=int(cfg["optimization"].get("tangent_local_starts", 10)),
        radius_deg=float(cfg["optimization"].get("tangent_start_perturb_deg", 6.0)),
    )
    tangent_starts = jnp.concatenate(
        [law_eta[None, :], population_eta[None, :], starts, tangent_local], axis=0
    )
    tangent_distinct_starts = _dedupe_designs(exp.family, tangent_starts)
    tangent_start_count = int(
        cfg["optimization"].get("tangent_start_count", min(5, int(tangent_starts.shape[0])))
    )
    tangent_start_count = max(2, min(tangent_start_count, int(tangent_starts.shape[0])))
    print(
        "[3/4] optimizing tangent action "
        f"(exact T algebra; {grad_tangent_trials}/{action_trials} CRN trials for gradients; "
        f"{int(tangent_distinct_starts.shape[0])} distinct starts screened, {tangent_start_count} optimized)",
        flush=True,
    )
    tangent_proxy_anchor = max(
        float(exp.tangent_action_gradient(law_eta, grad_tangent_bank)), 1.0e-12
    )
    tangent_eta, tangent_rows = _select_action_design(
        name="tangent",
        exp=exp,
        # Positive constant normalization leaves the minimizer unchanged and keeps
        # Adam/constraint scales O(1).
        objective=lambda eta: exp.tangent_action_gradient(eta, grad_tangent_bank) / tangent_proxy_anchor,
        fast_objective=lambda eta: exp.tangent_action_gradient(eta, grad_tangent_bank),
        exact_result=lambda eta: exp.exact_tangent_result(eta, action_bank),
        starts=tangent_starts,
        audit_starts=tangent_starts,
        selection_bank=law_bank,
        L_max=L_max,
        R_max=R_max,
        fast_population=gradient_fast.population_loss,
        fast_risk=gradient_fast.finite_risk,
        fast_joint=gradient_fast.population_and_finite,
        population_eta=population_eta,
        law_eta=law_eta,
        cfg=_optimizer_cfg(cfg, "tangent"),
        optimize_start_count=tangent_start_count,
        mandatory_start_count=2,
        exact_audit_limit=int(cfg["optimization"].get("tangent_exact_audit_candidates", 8)),
        exact_rescore=int(cfg["optimization"].get("tangent_exact_rescore_candidates", 6)),
        mandatory_exact=jnp.stack([law_eta, population_eta]),
        min_exact_law_valid=int(cfg["optimization"].get("tangent_min_exact_law_valid", 4)),
        min_exact_finalists=int(cfg["optimization"].get("tangent_min_exact_finalists", 4)),
    )

    # ------------------------------------------------------------------
    # 4/4 full weighted-Poisson action.  Keep the differentiable inverse-design
    # step, but make it multi-fidelity: Adam differentiates a lower-resolution
    # discretization of the same full MFSI action.  Exact common-CRN full action
    # on the scientific grid decides the winner.
    # ------------------------------------------------------------------
    opt4 = cfg["optimization"]
    local = _local_design_cloud(
        exp.family,
        jnp.stack([law_eta, tangent_eta, population_eta]),
        count_per_center=int(opt4.get("full_local_starts", 4)),
        radius_deg=float(opt4.get("full_start_perturb_deg", 4.0)),
    )
    random_count = max(0, min(int(opt4.get("full_random_starts", 2)), int(starts.shape[0])))
    random_subset = starts[:random_count]
    full_starts = jnp.concatenate(
        [
            law_eta[None, :],
            tangent_eta[None, :],
            population_eta[None, :],
            random_subset,
            local,
        ],
        axis=0,
    )
    full_distinct_starts = _dedupe_designs(exp.family, full_starts)
    full_cfg = _optimizer_cfg(cfg, "full")
    full_start_count = int(opt4.get("full_start_count", min(3, int(full_starts.shape[0]))))
    full_start_count = max(1, min(full_start_count, int(full_starts.shape[0])))
    print(
        "[4/4] optimizing differentiable full weighted-Poisson proxy "
        f"({len(exp.full_gradient_time_idx)}/{len(exp.times)} times, "
        f"{grad_action_bank.masses.shape[0]} CRN trials, "
        f"grid={exp.full_gradient_grid.n}x{exp.full_gradient_grid.n}, "
        f"{full_distinct_starts.shape[0]} distinct starts screened, {full_start_count} optimized, "
        f"steps={full_cfg.steps}, CG tol={exp.poisson_gradient_cfg.cg_tol:g}, "
        f"maxiter={exp.poisson_gradient_cfg.cg_maxiter}, "
        f"proxy_backend={exp.full_gradient_poisson_backend}, "
        f"exact_backend={exp.full_exact_poisson_backend})",
        flush=True,
    )
    full_proxy_anchor = max(
        float(exp.full_action_gradient(law_eta, grad_action_bank)), 1.0e-12
    )
    full_eta, full_rows = _select_action_design(
        name="full",
        exp=exp,
        # Same A_proxy minimizer, better-conditioned gradients and penalties.
        objective=lambda eta: exp.full_action_gradient(eta, grad_action_bank) / full_proxy_anchor,
        fast_objective=lambda eta: exp.full_action_gradient(eta, grad_action_bank),
        exact_result=lambda eta: exp.exact_full_result(eta, action_bank),
        starts=full_starts,
        audit_starts=full_starts,
        selection_bank=law_bank,
        L_max=L_max,
        R_max=R_max,
        fast_population=gradient_fast.population_loss,
        fast_risk=gradient_fast.finite_risk,
        fast_joint=gradient_fast.population_and_finite,
        population_eta=population_eta,
        law_eta=law_eta,
        cfg=full_cfg,
        optimize_start_count=full_start_count,
        mandatory_start_count=3,
        vectorize_starts=False,
        exact_audit_limit=int(opt4.get("full_exact_audit_candidates", 8)),
        exact_prescreen_result=lambda eta: exp.exact_full_result(
            eta, action_bank, trial_count=full_prescreen_trials
        ),
        exact_rescore=int(opt4.get("full_exact_rescore_candidates", 8)),
        mandatory_exact=jnp.stack([law_eta, tangent_eta, population_eta]),
        min_exact_law_valid=int(opt4.get("full_min_exact_law_valid", 10)),
        min_exact_finalists=int(opt4.get("full_min_exact_finalists", 8)),
    )

    # Audit whether the multi-fidelity stage-4 objective preserves the ranking of
    # the scientific full action on exactly audited finalists.  This is diagnostic
    # only; selection already uses the full-fidelity exact action.
    full_proxy_rows: list[dict[str, Any]] = []
    for row in full_rows:
        full_proxy_rows.append({
            "theta1_deg": _canonical_deg(row["eta"])[0],
            "theta2_deg": _canonical_deg(row["eta"])[1],
            "proxy_action": float(row.get("screen_objective", float("nan"))),
            "prescreen_action": float(row.get("prescreen_objective", float("nan"))),
            "full_action": float(row.get("objective", float("nan"))),
            "action_valid": bool(row.get("action_valid", False)),
            "screened_candidate_count": int(row.get("screened_candidate_count", 0)),
            "exact_law_audited_count": int(row.get("exact_law_audited_count", 0)),
            "exact_law_valid_count": int(row.get("exact_law_valid_count", 0)),
            "exact_finalist_count": int(row.get("exact_finalist_count", 0)),
        })
    agreement = [
        (r["proxy_action"], r["full_action"])
        for r in full_proxy_rows
        if r["action_valid"] and np.isfinite(r["proxy_action"]) and np.isfinite(r["full_action"])
    ]
    if len(agreement) >= 2:
        proxy_vals = np.asarray([x[0] for x in agreement], dtype=np.float64)
        full_vals = np.asarray([x[1] for x in agreement], dtype=np.float64)
        pearson = float(np.corrcoef(proxy_vals, full_vals)[0, 1])
        pr = np.empty(len(proxy_vals), dtype=np.float64)
        fr = np.empty(len(full_vals), dtype=np.float64)
        pr[np.argsort(proxy_vals)] = np.arange(len(proxy_vals), dtype=np.float64)
        fr[np.argsort(full_vals)] = np.arange(len(full_vals), dtype=np.float64)
        spearman = float(np.corrcoef(pr, fr)[0, 1])
        same_best = bool(int(np.argmin(proxy_vals)) == int(np.argmin(full_vals)))
    else:
        pearson = spearman = float("nan")
        same_best = False
    full_proxy_agreement = {
        "candidate_count": int(len(agreement)),
        "pearson": pearson,
        "spearman_rank": spearman,
        "same_best_candidate": same_best,
    }
    if full_rows:
        full_search_funnel = {
            "distinct_screen_starts": int(full_rows[0].get("distinct_screen_start_count", 0)),
            "gradient_starts": int(full_rows[0].get("gradient_start_count", 0)),
            "screened_candidates": int(full_rows[0].get("screened_candidate_count", 0)),
            "exact_law_audited": int(full_rows[0].get("exact_law_audited_count", 0)),
            "exact_law_valid": int(full_rows[0].get("exact_law_valid_count", 0)),
            "exact_full_finalists": int(full_rows[0].get("exact_finalist_count", len(full_rows))),
        }
    else:
        full_search_funnel = {
            "distinct_screen_starts": 0,
            "gradient_starts": 0,
            "screened_candidates": 0,
            "exact_law_audited": 0,
            "exact_law_valid": 0,
            "exact_full_finalists": 0,
        }

    designs = {
        "population": population_eta,
        "law": law_eta,
        "tangent": tangent_eta,
        "full": full_eta,
    }

    # ------------------------------------------------------------------
    # Fresh disjoint validation.
    # ------------------------------------------------------------------
    print("[validation] evaluating disjoint CRN bank", flush=True)
    validation_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    per_design_rows: dict[str, list[dict[str, Any]]] = {}
    for name, eta in designs.items():
        rows = exp.evaluate_trials_exact(eta, validation)
        for row in rows:
            row["design"] = name
        per_design_rows[name] = rows
        validation_rows.extend(rows)
        valid_rows = [r for r in rows if r["valid"]]
        lb_violations = [
            float(r.get("tangent_lower_bound_violation", float("nan")))
            for r in valid_rows
            if np.isfinite(r.get("tangent_lower_bound_violation", float("nan")))
        ]
        lb_tol = float(cfg.get("validity", {}).get("tangent_lower_bound_tol", 1.0e-6))
        summaries[name] = {
            "eta_deg": _canonical_deg(eta),
            "law_risk": _mean_se([r["law_risk"] for r in valid_rows]),
            "tangent_action": _mean_se([r["tangent_action"] for r in valid_rows]),
            "full_action": _mean_se([r["full_action"] for r in valid_rows]),
            "valid_fraction": float(len(valid_rows) / max(len(rows), 1)),
            "tangent_lower_bound_check": {
                "tolerance": lb_tol,
                "max_violation": float(max(lb_violations, default=0.0)),
                "violating_trials": int(sum(v > lb_tol for v in lb_violations)),
                "checked_trials": int(len(lb_violations)),
            },
        }

    law_full = _paired_reduction(
        [r["full_action"] for r in per_design_rows["full"]],
        [r["full_action"] for r in per_design_rows["law"]],
    )
    bootstrap = _bootstrap_ratio_reduction(
        [r["full_action"] for r in per_design_rows["full"]],
        [r["full_action"] for r in per_design_rows["law"]],
        reps=int(rnd.get("bootstrap_reps", 5000)),
        seed=int(cfg["seed"]) + 991,
    )

    candidate_rows = []
    for name, eta in designs.items():
        candidate_rows.append({
            "design": name,
            "theta1_deg": _canonical_deg(eta)[0],
            "theta2_deg": _canonical_deg(eta)[1],
            "population_loss_selection": float(exp.exact_population_result(eta)["value"]),
            "finite_risk_selection": float(exp.exact_finite_result(eta, law_bank)["value"]),
            "tangent_action_selection": float(exp.exact_tangent_result(eta, action_bank)["value"]),
            "full_action_selection": float(exp.exact_full_result(eta, action_bank)["value"]),
            "validation_law_mean": summaries[name]["law_risk"]["mean"],
            "validation_full_action_mean": summaries[name]["full_action"]["mean"],
            "validation_valid_fraction": summaries[name]["valid_fraction"],
        })

    # Formal selection-bank certificates for the reported designs.  These are
    # the authoritative quantities used by the lexicographic constraints; fresh
    # validation below is deliberately disjoint and is not the constraint test.
    selection_certificates: dict[str, Any] = {}
    for row in candidate_rows:
        name = str(row["design"])
        L_sel = float(row["population_loss_selection"])
        R_sel = float(row["finite_risk_selection"])
        required = ["L"] if name == "population" else (["L"] if name == "law" else ["L", "R"])
        pass_L = bool(np.isfinite(L_sel) and L_sel <= L_max + 1.0e-12)
        pass_R = bool(np.isfinite(R_sel) and R_sel <= R_max + 1.0e-12)
        certified = pass_L and (pass_R if "R" in required else True)
        selection_certificates[name] = {
            "eta_deg": [float(row["theta1_deg"]), float(row["theta2_deg"])],
            "required_screens": required,
            "L_selection": L_sel,
            "L_star": float(L_star),
            "L_max": float(L_max),
            "L_excess_from_star": float(L_sel - L_star),
            "L_slack_to_max": float(L_max - L_sel),
            "passes_L": pass_L,
            "R_selection": R_sel,
            "R_star": float(R_star),
            "R_max": float(R_max),
            "R_excess_from_star": float(R_sel - R_star),
            "R_slack_to_max": float(R_max - R_sel),
            "passes_R": pass_R,
            "full_action_selection": float(row["full_action_selection"]),
            "tangent_action_selection": float(row["tangent_action_selection"]),
            "certified": bool(certified),
        }

    result = {
        "schema_version": 3,
        "experiment": "toy_example",
        "smoke": False,
        "config": cfg,
        "config_hash": _config_hash(cfg),
        "reference": {
            "checkpoint": str(checkpoint),
            "metadata": reference_metadata,
            "reference_bank": "reference_bank.npz",
            "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass)),
        },
        "randomness": {
            "selection_bank": "selection_bank.npz",
            "validation_bank": "validation_bank.npz",
            "selection_namespace": int(rnd["selection_namespace"]),
            "validation_namespace": int(rnd["validation_namespace"]),
            "law_trials_effective": int(law_bank.masses.shape[0]),
            "action_trials_effective": int(action_bank.masses.shape[0]),
            "validation_trials_effective": int(validation.masses.shape[0]),
        },
        "law_screens": {
            "L_star": L_star,
            "L_max": L_max,
            "R_star": R_star,
            "R_max": R_max,
            "epsilon_l": epsilon_l,
            "epsilon_r": epsilon_r,
        },
        "selection_certificates": selection_certificates,
        "selection": {
            "population_optimum_deg": _canonical_deg(population_eta),
            "law_optimum_deg": _canonical_deg(law_eta),
            "tangent_optimum_deg": _canonical_deg(tangent_eta),
            "full_optimum_deg": _canonical_deg(full_eta),
            "optimizer_starts_deg": [_canonical_deg(row) for row in starts],
        },
        "selection_protocol": {
            "population_and_law": (
                "Differentiable batched search with proxy-ranked exact auditing; every selectable "
                "candidate is rechecked with exact ConvexHull feasibility, robust hard I-projection, "
                "strict all-trial validity, and the complete selection CRN bank."
            ),
            "tangent": (
                "Exact tangent algebra on a small CRN gradient prefix; proxy-ranked candidates "
                "are rechecked against the exact law screens, then the declared finalists and "
                "mandatory Law/Population incumbents are evaluated on the complete action bank."
            ),
            "full": (
                f"Gradient-based optimization of a lower-fidelity discretization of the same full "
                f"weighted-Poisson MFSI action ({grad_action_bank.masses.shape[0]} CRN trials, "
                f"{len(exp.full_gradient_time_idx)} time nodes, "
                f"{exp.full_gradient_grid.n}x{exp.full_gradient_grid.n} Poisson grid); exact "
                f"{full_prescreen_trials}-trial common-CRN pre-screen; top "
                f"{int(cfg['optimization'].get('full_exact_rescore_candidates', 3))} plus "
                "Law/Tangent/Population incumbents are re-estimated at full time/CG fidelity "
                "on the complete action bank. The proxy is never reported as the scientific action."
            ),
        },
        "full_proxy_agreement": full_proxy_agreement,
        "full_gradient_proxy": {
            "poisson_backend": exp.full_gradient_poisson_backend,
            "cache_signature": exp.full_gradient_cache_signature,
            "grid_n": int(exp.full_gradient_grid.n),
            "trials": int(grad_action_bank.masses.shape[0]),
            "time_indices": np.asarray(
                exp.full_gradient_time_idx, dtype=np.int32
            ).tolist(),
            "cg_tol": float(exp.poisson_gradient_cfg.cg_tol),
            "cg_maxiter": int(exp.poisson_gradient_cfg.cg_maxiter),
        },
        "full_exact_evaluator": {
            "poisson_backend": exp.full_exact_poisson_backend,
            "grid_n": int(exp.grid.n),
            "time_n": int(len(exp.times)),
            "time_batch_per_trial": int(len(exp.times)),
            "cg_tol": float(exp.poisson_cfg.cg_tol),
            "cg_maxiter": int(exp.poisson_cfg.cg_maxiter),
        },
        "full_search_funnel": full_search_funnel,
        "selection_audit": {
            "tangent": [
                {
                    **{k: v for k, v in row.items() if k != "eta"},
                    "eta_deg": _canonical_deg(row["eta"]),
                }
                for row in tangent_rows
            ],
            "full": [
                {
                    **{k: v for k, v in row.items() if k != "eta"},
                    "eta_deg": _canonical_deg(row["eta"]),
                }
                for row in full_rows
            ],
        },
        "validation": summaries,
        "contrasts": {
            "full_vs_law_full_action_reduction": law_full,
            "full_vs_law_ratio_of_means_bootstrap_95": bootstrap,
        },
    }

    write_json(output_dir / "result.json", result)
    _write_csv(output_dir / "result.candidate_summary.csv", candidate_rows)
    _write_csv(output_dir / "result.full_proxy_vs_full.csv", full_proxy_rows)
    _write_csv(output_dir / "result.validation_trials.csv", validation_rows)

    manifest = {
        "schema_version": 3,
        "config_hash": result["config_hash"],
        "artifacts": {
            "reference_checkpoint": "reference.npz",
            "reference_bank": "reference_bank.npz",
            "selection_bank": "selection_bank.npz",
            "validation_bank": "validation_bank.npz",
            "result": "result.json",
            "candidate_summary": "result.candidate_summary.csv",
            "full_proxy_vs_full": "result.full_proxy_vs_full.csv",
            "validation_trials": "result.validation_trials.csv",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return result
