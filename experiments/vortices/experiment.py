from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from functools import partial
from pathlib import Path
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from domain import DoubleGyreConfig, DoubleGyreTruth, EmpiricalEndpointSource, InitialLawConfig
from mfsi.cache import file_sha256, fingerprint, load_npz_cache, save_npz_cache
from mfsi.design import random_point_sensor_starts
from mfsi.exact_feasibility import robust_empirical_tilt_exact
from mfsi.flow_matching import FlowMatchingConfig, train_reference_flow
from mfsi.grid import RectangularGrid2D
from mfsi.measurements import GaussianPointSensors2D
from mfsi.metrics import gaussian_mmd2_grid_mass, multiscale_gaussian_mmd_kernel_rect
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.particles import ParticleMFSIConfig, particle_mfsi_state
from mfsi.poisson import PoissonConfig, solve_weighted_poisson, weighted_laplacian
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.raster import RasterConfig, rasterize_projected_particles_rect
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint

Array = jax.Array


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _progress_iter(items, *, desc: str, total: int | None = None):
    """Log-friendly progress with elapsed time and ETA."""
    if total is None:
        total = len(items)
    total = int(total)
    started = time.perf_counter()
    last_report = started
    completed = 0
    stride = max(1, total // 10) if total else 1
    print(f"[progress] {desc}: 0/{total}", flush=True)
    try:
        for completed, item in enumerate(items, start=1):
            yield item
            now = time.perf_counter()
            if completed == 1 or completed == total or completed % stride == 0 or now - last_report >= 15.0:
                elapsed = now - started
                rate = completed / max(elapsed, 1.0e-12)
                eta = (total - completed) / max(rate, 1.0e-12)
                print(
                    f"[progress] {desc}: {completed}/{total} "
                    f"({100.0 * completed / max(total, 1):5.1f}%) "
                    f"elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(eta)} rate={rate:.2f}/s",
                    flush=True,
                )
                last_report = now
    finally:
        if completed < total:
            print(
                f"[progress] {desc}: stopped after {completed}/{total} "
                f"elapsed={_format_duration(time.perf_counter() - started)}",
                flush=True,
            )


@partial(jax.jit, static_argnames=("dx", "operator_floor_rel", "gauge_strength"))
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
    gauge = flat_q / jnp.maximum(jnp.linalg.norm(flat_q, axis=-1, keepdims=True), 1.0e-300)
    gauge = gauge.reshape(q.shape)
    operator_psi = jax.vmap(lambda p, qo: weighted_laplacian(p, qo, dx))(psi, q_operator)
    gauge_projection = jnp.sum(gauge * psi, axis=(-2, -1), keepdims=True)
    residual = operator_psi + gauge_strength * gauge * gauge_projection - rhs
    residual_norm = jnp.linalg.norm(residual.reshape((residual.shape[0], -1)), axis=-1)
    rhs_norm = jnp.linalg.norm(rhs.reshape((rhs.shape[0], -1)), axis=-1)
    relative_residual = residual_norm / jnp.maximum(rhs_norm, 1.0e-14)
    physical_operator = jax.vmap(lambda p, qp: weighted_laplacian(p, qp, dx))(psi, q)
    actions = (dx * dx) * jnp.sum(psi * physical_operator, axis=(-2, -1))
    return actions, relative_residual



# VORTEX_BOX_REFERENCE_FIX_V1
# Shadow only the vortex experiment's generic reference symbols.  The toy and
# reusable mfsi implementations are untouched.
from bounded_reference import (
    BoxTransformedReferenceFlow as MLPReferenceFlow,
    train_box_reference_flow as train_reference_flow,
)
class ObservationTrialBank(NamedTuple):
    sample_indices: Array  # [trials, acquisition_times, finite_n]
    detector_z: Array      # [trials, acquisition_times, n_observables]


class Reconstruction(NamedTuple):
    c: Array
    c_dot: Array
    coefficients: Array
    residual_sum_squares: Array
    roughness: Array


class TrialMetrics(NamedTuple):
    law_risk: Array
    tangent_action: Array
    full_action: Array
    max_calibration_residual: Array
    min_ess_fraction: Array
    max_poisson_relative_residual: Array
    valid: Array


def _trap_weights(times: Array) -> Array:
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be a strictly increasing 1-D array")
    w = np.zeros_like(t)
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    if len(t) > 2:
        w[1:-1] = 0.5 * (t[2:] - t[:-2])
    w /= w.sum()
    return jnp.asarray(w, dtype=jnp.float64)


def _nested_acquisition_indices(time_n: int, k: int) -> np.ndarray:
    if k < 2 or k > time_n:
        raise ValueError(f"acquisition_k must satisfy 2 <= K <= time_n; got K={k}, T={time_n}")
    raw = np.rint(np.linspace(0, time_n - 1, k)).astype(np.int32)
    raw[0], raw[-1] = 0, time_n - 1
    raw = np.unique(raw)
    if len(raw) != k:
        interior = np.arange(1, time_n - 1, dtype=np.int32)
        want = k - 2
        chosen = (
            interior[np.rint(np.linspace(0, len(interior) - 1, want)).astype(int)]
            if want
            else np.empty((0,), dtype=np.int32)
        )
        raw = np.concatenate([[0], chosen, [time_n - 1]]).astype(np.int32)
    if len(np.unique(raw)) != k:
        raise ValueError("could not construct unique acquisition indices")
    return raw


def _mean_se(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "se": float("nan"), "n": 0}
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return {"mean": float(np.mean(x)), "se": se, "n": int(len(x))}


def _empirical_coordinate_support_gaps(
    features: np.ndarray,
    base_weights: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Return a cheap necessary feasibility certificate for moment targets.

    Every coordinate of a moment produced by non-negative reweighting must lie
    between the smallest and largest active particle feature. The gap is positive
    inside that coordinate box and negative outside it. This is not a complete
    convex-hull test, but a negative value is an exact infeasibility certificate.
    """
    phi = np.asarray(features, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if target.ndim == 2:
        target = target[None, ...]
    if phi.ndim != 3 or base.shape != phi.shape[:2]:
        raise ValueError("expected features [time, particle, moment] and matching weights")
    if target.ndim != 3 or target.shape[1:] != (phi.shape[0], phi.shape[2]):
        raise ValueError("expected targets [batch, time, moment]")

    lower = np.empty((phi.shape[0], phi.shape[2]), dtype=np.float64)
    upper = np.empty_like(lower)
    for t_idx in range(phi.shape[0]):
        active = base[t_idx] > 0.0
        if not np.any(active):
            raise ValueError(f"reference weights have empty support at time index {t_idx}")
        lower[t_idx] = np.min(phi[t_idx, active], axis=0)
        upper[t_idx] = np.max(phi[t_idx, active], axis=0)
    margins = np.minimum(target - lower[None, ...], upper[None, ...] - target)
    return np.min(margins, axis=(1, 2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truth_from_cfg(cfg: dict[str, Any]) -> DoubleGyreTruth:
    t = cfg["truth"]
    initial = t.get("initial", {})
    return DoubleGyreTruth(
        flow=DoubleGyreConfig(
            amplitude=float(t.get("amplitude", 0.1)),
            epsilon=float(t.get("epsilon", 0.25)),
            horizon=float(t.get("horizon", 10.0)),
            period=float(t.get("period", t.get("horizon", 10.0))),
        ),
        initial=InitialLawConfig(
            background_weight=float(initial.get("background_weight", 0.10)),
            mixture_weights=tuple(float(x) for x in initial.get("mixture_weights", [0.30, 0.20, 0.25, 0.25])),
            centers=tuple(tuple(float(v) for v in p) for p in initial.get(
                "centers", [[0.45, 0.25], [0.78, 0.72], [1.28, 0.28], [1.62, 0.68]]
            )),
            std_x=float(initial.get("std_x", 0.07)),
            std_y=float(initial.get("std_y", 0.07)),
        ),
    )


def ensure_truth_bank(
    truth: DoubleGyreTruth,
    cfg: dict[str, Any],
    output_dir: Path,
    times: Array,
) -> tuple[Array, str]:
    tcfg = cfg["truth"]
    signature = fingerprint({
        "schema": 1,
        "truth": tcfg,
        "seed": int(cfg["seed"]),
        "times": np.asarray(times).tolist(),
    })
    path = output_dir / "truth_bank.npz"
    loaded = load_npz_cache(path, signature=signature)
    if loaded is not None:
        arrays, _ = loaded
        print("[truth] reusing compatible truth_bank.npz", flush=True)
        return jnp.asarray(arrays["particles"], dtype=jnp.float64), signature

    print("[truth] integrating hidden double-gyre tracer bank", flush=True)
    bank = truth.make_bank(
        seed=int(cfg["seed"]) + int(tcfg.get("truth_seed_offset", 1001)),
        n=int(tcfg.get("truth_particles", 50000)),
        times=times,
        substeps_per_interval=int(tcfg.get("rk4_substeps_per_time_interval", 32)),
    )
    save_npz_cache(
        path,
        {"times": times, "particles": bank.particles},
        signature=signature,
        metadata={"role": "hidden_oracle_truth"},
    )
    return bank.particles, signature


def ensure_reference_endpoints(
    truth: DoubleGyreTruth,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[EmpiricalEndpointSource, str]:
    ref = cfg["reference"]
    n = int(ref.get("endpoint_particles", 50000))
    signature = fingerprint({
        "schema": 1,
        "truth": cfg["truth"],
        "n": n,
        "seed": int(cfg["seed"]) + int(ref.get("endpoint_seed_offset", 2001)),
        "truth_substeps": int(cfg["truth"].get("endpoint_rk4_substeps", 256)),
    })
    path = output_dir / "reference_endpoints.npz"
    loaded = load_npz_cache(path, signature=signature)
    if loaded is not None:
        arrays, _ = loaded
        print("[reference] reusing compatible endpoint dataset", flush=True)
        return EmpiricalEndpointSource(
            jnp.asarray(arrays["x0"], dtype=jnp.float64),
            jnp.asarray(arrays["x1"], dtype=jnp.float64),
        ), signature

    seed = int(cfg["seed"]) + int(ref.get("endpoint_seed_offset", 2001))
    x0 = jnp.asarray(truth.sample_initial_numpy(seed, n), dtype=jnp.float64)
    endpoints = truth.rollout(
        x0,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        substeps_per_interval=int(cfg["truth"].get("endpoint_rk4_substeps", 256)),
    )
    x1 = endpoints[-1]
    save_npz_cache(
        path,
        {"x0": x0, "x1": x1},
        signature=signature,
        metadata={"role": "reference_training_endpoints"},
    )
    return EmpiricalEndpointSource(x0, x1), signature


def _flow_matching_cfg(cfg: dict[str, Any]) -> FlowMatchingConfig:
    block = cfg.get("reference_training", {})
    return FlowMatchingConfig(
        seed=int(block.get("seed", cfg["seed"])),
        hidden_width=int(block.get("hidden_width", 128)),
        hidden_layers=int(block.get("hidden_layers", 4)),
        train_steps=int(block.get("train_steps", 12000)),
        batch_size=int(block.get("batch_size", 2048)),
        learning_rate=float(block.get("learning_rate", 1.0e-3)),
        min_learning_rate_ratio=float(block.get("min_learning_rate_ratio", 0.05)),
        adam_beta1=float(block.get("adam_beta1", 0.9)),
        adam_beta2=float(block.get("adam_beta2", 0.999)),
        adam_eps=float(block.get("adam_eps", 1.0e-8)),
        grad_clip_norm=float(block.get("grad_clip_norm", 10.0)),
        bridge_schedule=str(block.get("bridge_schedule", "linear")),
        bridge_noise_std=float(block.get("bridge_noise_std", 0.0)),
        log_every=int(block.get("log_every", 500)),
    )


def ensure_reference(
    source: EmpiricalEndpointSource,
    endpoint_signature: str,
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[MLPReferenceFlow, Path, dict[str, Any]]:
    path = output_dir / "reference.npz"
    train_cfg = _flow_matching_cfg(cfg)
    training_signature = fingerprint(asdict(train_cfg))
    substeps = int(cfg["reference"].get("rk4_substeps_per_time_interval", 16))

    if path.exists():
        flow = MLPReferenceFlow.from_npz(path, substeps_per_interval=substeps)
        metadata = dict(flow.metadata or {})
        if (
            metadata.get("experiment") == "vortices_double_gyre"
            and metadata.get("endpoint_signature") == endpoint_signature
            and metadata.get("training_signature") == training_signature
        ):
            print("[reference] reusing compatible reference.npz", flush=True)
            return flow, path, metadata
        print("[reference] cached checkpoint incompatible; retraining", flush=True)

    print("[reference] training endpoint-only flow-matching reference", flush=True)
    flow, history = train_reference_flow(source, train_cfg, substeps_per_interval=substeps)
    metadata = dict(flow.metadata or {})
    metadata.update({
        "experiment": "vortices_double_gyre",
        "endpoint_signature": endpoint_signature,
        "training_signature": training_signature,
        "history": history,
    })
    flow = MLPReferenceFlow(flow.params, substeps_per_interval=substeps, metadata=metadata)
    save_npz_checkpoint(path, flow.params, metadata)
    return flow, path, metadata


def ensure_reference_bank(
    truth: DoubleGyreTruth,
    flow: MLPReferenceFlow,
    checkpoint: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    times: Array,
) -> tuple[Array, Array, Array]:
    ref = cfg["reference"]
    n = int(ref.get("particles", 32768))
    seed = int(cfg["seed"]) + int(ref.get("bank_seed_offset", 3001))
    signature = fingerprint({
        "schema": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "reference": ref,
        "seed": seed,
        "times": np.asarray(times).tolist(),
        "truth_initial": cfg["truth"].get("initial", {}),
    })
    path = output_dir / "reference_bank.npz"
    loaded = load_npz_cache(path, signature=signature)
    if loaded is not None:
        arrays, _ = loaded
        print("[reference] reusing compatible reference_bank.npz", flush=True)
        return (
            jnp.asarray(arrays["nodes"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["weights"], dtype=jnp.float64),
        )

    print("[reference] rolling frozen reference particle bank", flush=True)
    x0 = jnp.asarray(truth.sample_initial_numpy(seed, n), dtype=jnp.float64)
    nodes = flow.rollout(x0, times)
    velocity = jax.vmap(lambda t, x: flow.velocity(x, t))(times, nodes)
    weights = jnp.full((len(times), n), 1.0 / float(n), dtype=jnp.float64)
    save_npz_cache(
        path,
        {"times": times, "nodes": nodes, "velocity": velocity, "weights": weights},
        signature=signature,
        metadata={"role": "frozen_reference_rollout"},
    )
    return nodes, velocity, weights


def make_observation_bank(
    *,
    seed: int,
    namespace: int,
    trials: int,
    acquisition_k: int,
    finite_n: int,
    truth_particle_count: int,
    n_observables: int,
) -> ObservationTrialBank:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(namespace)]))
    idx = rng.integers(
        0,
        int(truth_particle_count),
        size=(int(trials), int(acquisition_k), int(finite_n)),
        dtype=np.int32,
    )
    z = rng.standard_normal((int(trials), int(acquisition_k), int(n_observables)))
    return ObservationTrialBank(
        sample_indices=jnp.asarray(idx, dtype=jnp.int32),
        detector_z=jnp.asarray(z, dtype=jnp.float64),
    )


def ensure_observation_bank(
    *,
    name: str,
    exp: "VortexExperiment",
    trials: int,
    namespace: int,
    output_dir: Path,
) -> ObservationTrialBank:
    signature = fingerprint({
        "schema": 1,
        "name": name,
        "seed": int(exp.cfg["seed"]),
        "namespace": int(namespace),
        "trials": int(trials),
        "finite_n": int(exp.cfg["measurement"]["finite_n"]),
        "acq_idx": np.asarray(exp.acq_idx).tolist(),
        "truth_particle_count": int(exp.truth_particles.shape[1]),
        "n_observables": int(exp.family.n_sensors),
    })
    path = output_dir / f"{name}_bank.npz"
    loaded = load_npz_cache(path, signature=signature)
    if loaded is not None:
        arrays, _ = loaded
        print(f"[bank] reusing compatible {name}_bank.npz", flush=True)
        return ObservationTrialBank(
            jnp.asarray(arrays["sample_indices"], dtype=jnp.int32),
            jnp.asarray(arrays["detector_z"], dtype=jnp.float64),
        )
    bank = make_observation_bank(
        seed=int(exp.cfg["seed"]),
        namespace=int(namespace),
        trials=int(trials),
        acquisition_k=len(exp.acq_idx),
        finite_n=int(exp.cfg["measurement"]["finite_n"]),
        truth_particle_count=int(exp.truth_particles.shape[1]),
        n_observables=int(exp.family.n_sensors),
    )
    save_npz_cache(
        path,
        {"sample_indices": bank.sample_indices, "detector_z": bank.detector_z},
        signature=signature,
        metadata={"role": name},
    )
    return bank


def prefix_bank(bank: ObservationTrialBank, count: int) -> ObservationTrialBank:
    count = min(int(count), int(bank.sample_indices.shape[0]))
    return ObservationTrialBank(bank.sample_indices[:count], bank.detector_z[:count])


class VortexExperiment:
    """Controlled-oracle double-gyre benchmark using the shared MFSI core."""

    def __init__(
        self,
        cfg: dict[str, Any],
        reference: MLPReferenceFlow,
        *,
        truth_particles: Array,
        reference_nodes: Array,
        reference_velocity: Array,
        reference_weights: Array,
    ):
        self.cfg = cfg
        self.reference = reference
        self.truth = _truth_from_cfg(cfg)
        self.truth_particles = jnp.asarray(truth_particles, dtype=jnp.float64)
        self.reference_nodes = jnp.asarray(reference_nodes, dtype=jnp.float64)
        self.reference_velocity = jnp.asarray(reference_velocity, dtype=jnp.float64)
        self.reference_weights = jnp.asarray(reference_weights, dtype=jnp.float64)

        p = cfg["poisson"]
        self.grid = RectangularGrid2D(
            x_min=float(p.get("x_min", 0.0)), x_max=float(p.get("x_max", 2.0)),
            y_min=float(p.get("y_min", 0.0)), y_max=float(p.get("y_max", 1.0)),
            nx=int(p["grid_nx"]), ny=int(p["grid_ny"]),
        )
        dx = self.grid.require_isotropic_spacing()
        self.times = jnp.linspace(0.0, 1.0, int(p["time_n"]), dtype=jnp.float64)
        if self.truth_particles.shape[0] != len(self.times):
            raise ValueError("truth bank time dimension does not match poisson.time_n")
        if self.reference_nodes.shape[:2] != self.reference_weights.shape:
            raise ValueError("reference node/weight shapes are inconsistent")
        if self.reference_nodes.shape != self.reference_velocity.shape:
            raise ValueError("reference node/velocity shapes are inconsistent")
        if self.reference_nodes.shape[0] != len(self.times):
            raise ValueError("reference bank time dimension does not match poisson.time_n")

        self.time_w = _trap_weights(self.times)
        mcfg = cfg["measurement"]
        self.acq_idx = jnp.asarray(
            _nested_acquisition_indices(len(self.times), int(mcfg["acquisition_k"])),
            dtype=jnp.int32,
        )
        self.family = GaussianPointSensors2D(
            width=float(mcfg.get("sensor_width", 0.12)),
            n_sensors=int(mcfg.get("n_sensors", 4)),
        )
        scfg = cfg.get("moment_reconstruction", {})
        self.spline_cfg = AnchoredCubicSplineConfig(
            internal_knots=int(scfg.get("internal_knots", 3)),
            smoothing=float(scfg.get("smoothing", 1.0e-4)),
            ridge_rel=float(scfg.get("ridge_rel", 1.0e-10)),
            roughness_quadrature_order=int(scfg.get("roughness_quadrature_order", 8)),
        )
        self.reconstructor = AnchoredCubicSplineReconstructor(
            self.times[self.acq_idx], self.times, self.spline_cfg
        )

        proj = cfg["projection"]
        self.iprojection_backend = str(proj.get("trajectory_backend", "jax"))
        if self.iprojection_backend not in {"jax", "tesseract_cpp"}:
            raise ValueError("projection.trajectory_backend must be 'jax' or 'tesseract_cpp'")
        if self.iprojection_backend == "tesseract_cpp":
            from mfsi.projection_tesseract import (
                TesseractIProjectionUnavailable,
                is_tesseract_iprojection_available,
            )

            if not is_tesseract_iprojection_available():
                raise TesseractIProjectionUnavailable(
                    "vortices requests the native I-projection backend, but "
                    "Tesseract-JAX or the compiled extension is unavailable"
                )
        self.projector = EmpiricalIProjector(
            IProjectionConfig(
                max_steps=int(proj.get("search_max_steps", proj.get("max_steps", 300))),
                residual_tol=float(proj.get("search_residual_tol", proj.get("residual_tol", 1.0e-10))),
                newton_ridge=float(proj.get("newton_ridge", 1.0e-7)),
                step_cap=float(proj.get("step_cap", 20.0)),
                lambda_clip=float(proj.get("lambda_clip", 1000.0)),
                line_search_steps=int(proj.get("search_line_search_steps", proj.get("line_search_steps", 8))),
                implicit_ridge=float(proj.get("implicit_ridge", 0.0)),
            ),
            trajectory_backend=self.iprojection_backend,
        )
        # Authoritative batching gets the full Newton budget.  Its outputs are
        # accepted only after explicit residual/ESS checks and retain the robust
        # scalar solver as a per-system fallback.
        self.exact_projector = EmpiricalIProjector(
            IProjectionConfig(
                max_steps=int(proj.get("max_steps", 300)),
                residual_tol=float(proj.get("residual_tol", 1.0e-10)),
                newton_ridge=float(proj.get("newton_ridge", 1.0e-7)),
                step_cap=float(proj.get("step_cap", 20.0)),
                lambda_clip=float(proj.get("lambda_clip", 1000.0)),
                line_search_steps=int(proj.get("line_search_steps", 8)),
                implicit_ridge=float(proj.get("implicit_ridge", 0.0)),
            ),
            trajectory_backend=self.iprojection_backend,
        )
        pcfg = cfg.get("particle_mfsi", {})
        self.particle_cfg = ParticleMFSIConfig(
            covariance_ridge=float(pcfg.get("covariance_ridge", 1.0e-7)),
            tangent_ridge=float(pcfg.get("tangent_ridge", 1.0e-7)),
        )
        rcfg = cfg.get("raster", {})
        self.raster_cfg = RasterConfig(
            bandwidth=float(rcfg.get("bandwidth", 0.0)),
            truncate=float(rcfg.get("truncate", 4.0)),
        )
        self.poisson_cfg = PoissonConfig(
            dx=dx,
            operator_floor_rel=float(p.get("operator_floor_rel", 2.0e-5)),
            cg_tol=float(p.get("cg_tol", 1.0e-8)),
            cg_maxiter=int(p.get("cg_maxiter", 520)),
            gauge_strength=float(p.get("gauge_strength", 1.0)),
        )

        opt = cfg.get("optimization", {})
        gx = int(opt.get("full_gradient_grid_nx", max(16, self.grid.nx // 2)))
        gy = int(opt.get("full_gradient_grid_ny", max(8, self.grid.ny // 2)))
        self.full_gradient_grid = RectangularGrid2D(
            self.grid.x_min, self.grid.x_max, self.grid.y_min, self.grid.y_max, gx, gy
        )
        gdx = self.full_gradient_grid.require_isotropic_spacing()
        self.poisson_gradient_cfg = PoissonConfig(
            dx=gdx,
            operator_floor_rel=float(opt.get("full_gradient_operator_floor_rel", p.get("operator_floor_rel", 2.0e-5))),
            cg_tol=float(opt.get("full_gradient_cg_tol", 1.0e-6)),
            cg_maxiter=int(opt.get("full_gradient_cg_maxiter", 120)),
            gauge_strength=float(p.get("gauge_strength", 1.0)),
        )
        grad_time_n = max(3, min(int(opt.get("full_gradient_time_n", 7)), len(self.times)))
        grad_idx = np.unique(np.rint(np.linspace(0, len(self.times) - 1, grad_time_n)).astype(np.int32))
        self.full_gradient_time_idx = jnp.asarray(grad_idx, dtype=jnp.int32)
        self.full_gradient_time_w = _trap_weights(self.times[self.full_gradient_time_idx])
        self.full_gradient_poisson_backend = str(opt.get("full_gradient_poisson_backend", "jax"))
        self.full_exact_poisson_backend = str(
            opt.get("full_exact_poisson_backend", self.full_gradient_poisson_backend)
        )
        for key, backend in (
            ("full_gradient_poisson_backend", self.full_gradient_poisson_backend),
            ("full_exact_poisson_backend", self.full_exact_poisson_backend),
        ):
            if backend not in {"jax", "tesseract_cpp"}:
                raise ValueError(f"optimization.{key} must be 'jax' or 'tesseract_cpp'")
        if "tesseract_cpp" in {
            self.full_gradient_poisson_backend,
            self.full_exact_poisson_backend,
        }:
            from mfsi.poisson_tesseract import (
                TesseractPoissonUnavailable,
                is_tesseract_poisson_available,
            )

            if not is_tesseract_poisson_available():
                raise TesseractPoissonUnavailable(
                    "vortices requests the native Poisson backend, but "
                    "Tesseract-JAX or the compiled extension is unavailable"
                )

        self.reference_in_domain = self.grid.in_domain(self.reference_nodes)
        self.reference_base_mass = jnp.sum(
            jnp.where(self.reference_in_domain, self.reference_weights, 0.0), axis=-1
        )
        masked = jnp.where(self.reference_in_domain, self.reference_weights, 0.0)
        self.reference_weights = masked / jnp.maximum(jnp.sum(masked, axis=-1, keepdims=True), 1.0e-300)

        truth_in = self.grid.in_domain(self.truth_particles)
        self.truth_in_domain_fraction = jnp.mean(truth_in.astype(jnp.float64), axis=-1)
        if float(jnp.min(self.truth_in_domain_fraction)) < 0.999999:
            raise RuntimeError("hidden double-gyre truth left the declared physical domain")

        self.mmd_kernel = multiscale_gaussian_mmd_kernel_rect(
            self.grid.nx,
            self.grid.ny,
            self.grid.dx,
            self.grid.dy,
            cfg["law"].get("mmd_bandwidths", [0.05, 0.10, 0.20, 0.40]),
        )
        self.truth_masses = self._truth_grid_masses()
        self.truth_kernel_potential = jax.vmap(
            lambda mass: jsp.signal.fftconvolve(mass, self.mmd_kernel, mode="same")
        )(self.truth_masses)
        self.truth_kernel_self = jnp.sum(
            self.truth_masses * self.truth_kernel_potential, axis=(-2, -1)
        )
        self._exact_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._exact_geometry_cache: tuple[
            tuple[float, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] | None = None

    def _truth_grid_masses(self) -> Array:
        n = int(self.truth_particles.shape[1])
        w = jnp.full((n,), 1.0 / float(n), dtype=jnp.float64)
        z = jnp.zeros((n,), dtype=jnp.float64)
        masses = [
            rasterize_projected_particles_rect(x, w, z, self.grid, self.raster_cfg).mass
            for x in self.truth_particles
        ]
        return jnp.stack(masses)

    def _geometry(self, eta: Array) -> tuple[Array, Array, Array]:
        eta = self.family.canonicalize(eta)
        phi_truth = self.family.features(self.truth_particles, eta)
        phi_ref = self.family.features(self.reference_nodes, eta)
        grad_ref = self.family.feature_gradients(self.reference_nodes, eta)
        return phi_truth, phi_ref, grad_ref

    def _law_risk_rows(self, masses: Array) -> Array:
        """MMD rows with the truth-side convolution precomputed once per run."""
        kernel = self.mmd_kernel
        truth_potential = self.truth_kernel_potential
        truth_self = self.truth_kernel_self

        def one_trial(trial_masses):
            projected_potential = jax.vmap(
                lambda mass: jsp.signal.fftconvolve(mass, kernel, mode="same")
            )(trial_masses)
            projected_self = jnp.sum(
                trial_masses * projected_potential, axis=(-2, -1)
            )
            cross = jnp.sum(trial_masses * truth_potential, axis=(-2, -1))
            return jnp.maximum(projected_self + truth_self - 2.0 * cross, 0.0)

        return jax.vmap(one_trial)(masses)

    def _law_risk_at_time(self, mass: Array, t_idx: int) -> Array:
        projected = jsp.signal.fftconvolve(mass, self.mmd_kernel, mode="same")
        value = (
            jnp.sum(mass * projected)
            + self.truth_kernel_self[t_idx]
            - 2.0 * jnp.sum(mass * self.truth_kernel_potential[t_idx])
        )
        return jnp.maximum(value, 0.0)

    def _measurement_reconstruction(
        self, phi_truth: Array, bank: ObservationTrialBank, trial: int | Array
    ) -> Reconstruction:
        phi_acq = phi_truth[self.acq_idx]
        idx = bank.sample_indices[trial]
        z = bank.detector_z[trial]
        sampled = jax.vmap(lambda p, ii: jnp.mean(p[ii], axis=0))(phi_acq, idx)
        exact = jnp.mean(phi_acq, axis=1)
        y = sampled + float(self.cfg["measurement"].get("obs_noise_std", 0.0)) * z
        endpoint = (self.acq_idx == 0) | (self.acq_idx == len(self.times) - 1)
        y = jnp.where(endpoint[:, None], exact, y)
        fit = self.reconstructor.reconstruct(y, exact[0], exact[-1])
        return Reconstruction(fit.c, fit.c_dot, fit.coefficients, fit.residual_sum_squares, fit.roughness)

    def _measurement_reconstruction_batch(
        self, phi_truth: Array, bank: ObservationTrialBank
    ) -> Reconstruction:
        trials = jnp.arange(int(bank.sample_indices.shape[0]), dtype=jnp.int32)
        return jax.vmap(lambda trial: self._measurement_reconstruction(phi_truth, bank, trial))(trials)

    def _particle_trajectory(
        self,
        *,
        phi: Array,
        grad_phi: Array,
        velocity: Array,
        base_weights: Array,
        targets: Array,
        targets_dot: Array,
    ):
        """Project and form particle forcing/tangent action for a `[B,T]` bank."""
        projection = self.projector.project_trajectory(phi, base_weights, targets)
        weights = projection.weights
        advective = jnp.einsum("tnmd,tnd->tnm", grad_phi, velocity)
        mean_advective = jnp.einsum("btn,tnm->btm", weights, advective)
        g = jnp.einsum("tnm,btm->btn", advective, projection.lam)
        mean_g = jnp.einsum("btn,btn->bt", weights, g)
        centered_phi = phi[None, :, :, :] - projection.moments[:, :, None, :]
        cov_phi_g = jnp.einsum(
            "btn,btnm,btn->btm",
            weights,
            centered_phi,
            g - mean_g[:, :, None],
        )
        eye = jnp.eye(phi.shape[-1], dtype=phi.dtype)
        lambda_dot = jnp.linalg.solve(
            projection.covariance + float(self.particle_cfg.covariance_ridge) * eye,
            (targets_dot - mean_advective - cov_phi_g)[..., None],
        )[..., 0]
        forcing = jnp.einsum("btnm,btm->btn", centered_phi, lambda_dot)
        forcing = forcing + g - mean_g[:, :, None]
        forcing = forcing - jnp.einsum("btn,btn->bt", weights, forcing)[:, :, None]

        tangent_gram = jnp.einsum("btn,tnmd,tnkd->btmk", weights, grad_phi, grad_phi)
        tangent_residual = mean_advective - targets_dot
        tangent_coeff = jnp.linalg.solve(
            tangent_gram + float(self.particle_cfg.tangent_ridge) * eye,
            tangent_residual[..., None],
        )[..., 0]
        tangent_action = jnp.einsum("btm,btm->bt", tangent_residual, tangent_coeff)
        return projection, forcing, tangent_action

    def _raster_trajectory(self, weights: Array, forcing: Array, *, time_idx: Array, grid):
        nodes = self.reference_nodes[time_idx]

        def raster_trial(w_trial, f_trial):
            return jax.vmap(
                lambda x, w, f: rasterize_projected_particles_rect(
                    x, w, f, grid, self.raster_cfg
                )
            )(nodes, w_trial, f_trial)

        return jax.vmap(raster_trial)(weights, forcing)

    def _poisson_batch(
        self,
        q: Array,
        h: Array,
        *,
        cfg: PoissonConfig,
        backend: str,
    ) -> tuple[Array, Array]:
        """Return `[B,T]` physical actions/residuals from one flattened batch."""
        leading_shape = q.shape[:2]
        q_flat = q.reshape((-1,) + q.shape[-2:])
        h_flat = h.reshape((-1,) + h.shape[-2:])
        if backend == "tesseract_cpp":
            from mfsi.poisson_tesseract import solve_weighted_poisson_batch_tesseract

            psi = solve_weighted_poisson_batch_tesseract(q_flat, h_flat, cfg)
            actions, residuals = _batched_poisson_diagnostics(
                psi,
                q_flat,
                h_flat,
                dx=float(cfg.dx),
                operator_floor_rel=float(cfg.operator_floor_rel),
                gauge_strength=float(cfg.gauge_strength),
            )
        else:
            solved = jax.vmap(lambda q_one, h_one: solve_weighted_poisson(q_one, h_one, cfg))(
                q_flat, h_flat
            )
            actions, residuals = solved.action, solved.relative_residual
        return actions.reshape(leading_shape), residuals.reshape(leading_shape)

    def _validity(self, max_resid: Array, min_ess: Array, poisson_rel: Array | None = None) -> Array:
        v = self.cfg.get("validity", {})
        ok = (
            (max_resid <= float(v.get("max_finite_calibration_resid", 1.0e-3)))
            & (min_ess >= float(v.get("min_ess_fraction", 0.03)))
            & (jnp.min(self.reference_base_mass) >= float(v.get("min_in_domain_base_mass", 0.995)))
        )
        if poisson_rel is not None and v.get("max_poisson_relative_residual") is not None:
            ok = ok & (poisson_rel <= float(v["max_poisson_relative_residual"]))
        return ok

    def _raster_projected_mass(self, t_idx: int, weights: Array, *, grid=None) -> Array:
        grid = self.grid if grid is None else grid
        zero = jnp.zeros_like(weights)
        return rasterize_projected_particles_rect(
            self.reference_nodes[t_idx], weights, zero, grid, self.raster_cfg
        ).mass

    def population_loss(self, eta: Array) -> Array:
        """DG-Exact oracle risk: exact hidden moments, no sparse observation layer."""
        phi_truth, phi_ref, _ = self._geometry(eta)
        targets = jnp.mean(phi_truth, axis=1)[None, :, :]
        projection = self.projector.project_trajectory(phi_ref, self.reference_weights, targets)
        zeros = jnp.zeros_like(projection.weights)
        rasters = self._raster_trajectory(
            projection.weights,
            zeros,
            time_idx=jnp.arange(len(self.times), dtype=jnp.int32),
            grid=self.grid,
        )
        vals = self._law_risk_rows(rasters.mass)[0]
        risk = jnp.sum(self.time_w * vals)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual[0], axis=-1))
        min_ess = jnp.min(projection.ess_fraction[0])
        v = self.cfg.get("validity", {})
        valid = (
            (max_resid <= float(v.get("max_population_calibration_resid", 1.0e-5)))
            & (min_ess >= float(v.get("min_ess_fraction", 0.03)))
            & (jnp.min(self.reference_base_mass) >= float(v.get("min_in_domain_base_mass", 0.995)))
        )
        return jnp.where(valid, risk, risk + float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3)))

    def finite_risk(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, _ = self._geometry(eta)
        rec = self._measurement_reconstruction_batch(phi_truth, bank)
        projection = self.projector.project_trajectory(phi_ref, self.reference_weights, rec.c)
        rasters = self._raster_trajectory(
            projection.weights,
            jnp.zeros_like(projection.weights),
            time_idx=jnp.arange(len(self.times), dtype=jnp.int32),
            grid=self.grid,
        )
        vals = self._law_risk_rows(rasters.mass)
        risks = jnp.sum(vals * self.time_w[None, :], axis=1)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
        min_ess = jnp.min(projection.ess_fraction, axis=1)
        valid = self._validity(max_resid, min_ess)
        penalty = float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3))
        return jnp.mean(jnp.where(valid, risks, risks + penalty))

    def _one_trial_metrics_from_geometry(
        self,
        phi_truth: Array,
        phi_ref: Array,
        grad_ref: Array,
        bank: ObservationTrialBank,
        trial: int | Array,
        *,
        full: bool,
    ) -> TrialMetrics:
        rec = self._measurement_reconstruction(phi_truth, bank, trial)
        law_vals, tangent_vals, full_vals = [], [], []
        max_resid = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        max_poisson = jnp.asarray(0.0)
        for t_idx in range(len(self.times)):
            st = particle_mfsi_state(
                phi=phi_ref[t_idx],
                grad_phi=grad_ref[t_idx],
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
            ras = rasterize_projected_particles_rect(
                self.reference_nodes[t_idx], st.projection.weights, st.forcing, self.grid, self.raster_cfg
            )
            law_vals.append(gaussian_mmd2_grid_mass(ras.mass, self.truth_masses[t_idx], self.mmd_kernel))
            if full:
                pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_cfg)
                full_vals.append(pois.action)
                max_poisson = jnp.maximum(max_poisson, pois.relative_residual)
        law = jnp.sum(self.time_w * jnp.stack(law_vals))
        tangent = jnp.sum(self.time_w * jnp.stack(tangent_vals))
        full_action = jnp.sum(self.time_w * jnp.stack(full_vals)) if full else jnp.asarray(jnp.nan)
        valid = self._validity(max_resid, min_ess, max_poisson if full else None)
        return TrialMetrics(law, tangent, full_action, max_resid, min_ess, max_poisson, valid)

    def tangent_action(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        rec = self._measurement_reconstruction_batch(phi_truth, bank)
        projection, _, tangent = self._particle_trajectory(
            phi=phi_ref,
            grad_phi=grad_ref,
            velocity=self.reference_velocity,
            base_weights=self.reference_weights,
            targets=rec.c,
            targets_dot=rec.c_dot,
        )
        values = jnp.sum(tangent * self.time_w[None, :], axis=1)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
        min_ess = jnp.min(projection.ess_fraction, axis=1)
        return jnp.mean(jnp.where(self._validity(max_resid, min_ess), values, values + 1.0e5))

    def tangent_action_gradient(self, eta: Array, bank: ObservationTrialBank) -> Array:
        return self.tangent_action(eta, bank)

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
        projection = self.projector.project(phi, base_weights, target, lam0=lam0)
        w = projection.weights
        m = jnp.einsum("nmd,nd->nm", grad_phi, velocity)
        mean_m = jnp.sum(w[:, None] * m, axis=0)
        g = m @ projection.lam
        mean_g = jnp.sum(w * g)
        centered_phi = phi - projection.moments[None, :]
        cov_phi_g = jnp.sum(w[:, None] * centered_phi * (g - mean_g)[:, None], axis=0)
        cov = projection.covariance + float(self.particle_cfg.covariance_ridge) * jnp.eye(phi.shape[-1])
        lam_dot = jnp.linalg.solve(cov, target_dot - mean_m - cov_phi_g)
        forcing = centered_phi @ lam_dot + g - mean_g
        forcing = forcing - jnp.sum(w * forcing)
        return projection, forcing

    def _one_trial_full_action_gradient(
        self,
        phi_truth: Array,
        phi_ref: Array,
        grad_ref: Array,
        bank: ObservationTrialBank,
        trial: int | Array,
    ) -> Array:
        rec = self._measurement_reconstruction(phi_truth, bank, trial)
        lam = jnp.zeros((self.family.n_sensors,), dtype=jnp.float64)
        vals = []
        max_resid = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        for t_idx in np.asarray(self.full_gradient_time_idx, dtype=np.int32).tolist():
            projection, forcing = self._particle_forcing_only(
                phi=phi_ref[t_idx], grad_phi=grad_ref[t_idx], velocity=self.reference_velocity[t_idx],
                base_weights=self.reference_weights[t_idx], target=rec.c[t_idx], target_dot=rec.c_dot[t_idx], lam0=lam,
            )
            lam = projection.lam
            max_resid = jnp.maximum(max_resid, jnp.linalg.norm(projection.residual))
            min_ess = jnp.minimum(min_ess, projection.ess_fraction)
            ras = rasterize_projected_particles_rect(
                self.reference_nodes[t_idx], projection.weights, forcing, self.full_gradient_grid, self.raster_cfg
            )
            vals.append(solve_weighted_poisson(ras.q, ras.h, self.poisson_gradient_cfg).action)
        action = jnp.sum(self.full_gradient_time_w * jnp.stack(vals))
        return jnp.where(self._validity(max_resid, min_ess), action, action + 1.0e5)

    def full_action_gradient(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        rec = self._measurement_reconstruction_batch(phi_truth, bank)
        time_idx = self.full_gradient_time_idx
        projection, forcing, _ = self._particle_trajectory(
            phi=phi_ref[time_idx],
            grad_phi=grad_ref[time_idx],
            velocity=self.reference_velocity[time_idx],
            base_weights=self.reference_weights[time_idx],
            targets=rec.c[:, time_idx],
            targets_dot=rec.c_dot[:, time_idx],
        )
        rasters = self._raster_trajectory(
            projection.weights,
            forcing,
            time_idx=time_idx,
            grid=self.full_gradient_grid,
        )
        actions, _ = self._poisson_batch(
            rasters.q,
            rasters.h,
            cfg=self.poisson_gradient_cfg,
            backend=self.full_gradient_poisson_backend,
        )
        values = jnp.sum(actions * self.full_gradient_time_w[None, :], axis=1)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
        min_ess = jnp.min(projection.ess_fraction, axis=1)
        return jnp.mean(jnp.where(self._validity(max_resid, min_ess), values, values + 1.0e5))

    def full_action(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        rec = self._measurement_reconstruction_batch(phi_truth, bank)
        projection, forcing, _ = self._particle_trajectory(
            phi=phi_ref,
            grad_phi=grad_ref,
            velocity=self.reference_velocity,
            base_weights=self.reference_weights,
            targets=rec.c,
            targets_dot=rec.c_dot,
        )
        all_times = jnp.arange(len(self.times), dtype=jnp.int32)
        rasters = self._raster_trajectory(
            projection.weights, forcing, time_idx=all_times, grid=self.grid
        )
        actions, residuals = self._poisson_batch(
            rasters.q,
            rasters.h,
            cfg=self.poisson_cfg,
            backend=self.full_exact_poisson_backend,
        )
        values = jnp.sum(actions * self.time_w[None, :], axis=1)
        max_resid = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
        min_ess = jnp.min(projection.ess_fraction, axis=1)
        max_poisson = jnp.max(residuals, axis=1)
        valid = self._validity(max_resid, min_ess, max_poisson)
        return jnp.mean(jnp.where(valid, values, values + 1.0e5))

    def _exact_tilt(
        self,
        phi: np.ndarray,
        base: np.ndarray,
        target: np.ndarray,
        lam0: np.ndarray,
        *,
        newton_steps: int | None = None,
    ):
        p = self.cfg.get("projection", {})
        return robust_empirical_tilt_exact(
            phi, base, target, lam0=lam0,
            newton_steps=(
                int(p.get("max_steps", 300))
                if newton_steps is None
                else int(newton_steps)
            ),
            newton_ridge=float(p.get("newton_ridge", 1.0e-7)),
            step_cap=float(p.get("step_cap", 20.0)),
            lambda_clip=float(p.get("lambda_clip", 1000.0)),
            accept_tol=float(p.get("solver_accept_tol", 2.0e-6)),
            lbfgs_maxiter=int(p.get("lbfgs_maxiter", 800)),
            retry_multiplier=float(p.get("retry_clip_multiplier", 2.0)),
            retries=int(p.get("max_retries", 2)),
        )

    def _exact_key(self, eta: Array) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(self.family.canonicalize(eta), dtype=np.float64), 12))

    def _exact_geometry(self, eta: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-entry host cache; exact trial banks reuse geometry for every trial."""
        key = self._exact_key(eta)
        cached = self._exact_geometry_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        geometry = tuple(np.asarray(x, dtype=np.float64) for x in self._geometry(eta))
        self._exact_geometry_cache = (key, geometry)
        return geometry

    def exact_population_result(self, eta: Array) -> dict[str, Any]:
        key = ("population", self._exact_key(eta))
        if key in self._exact_cache:
            return dict(self._exact_cache[key])
        phi_truth_np, phi_ref_np, _ = self._exact_geometry(eta)
        targets = np.mean(phi_truth_np, axis=1)
        lam = np.zeros(self.family.n_sensors, dtype=np.float64)
        vals = []
        max_resid, min_ess = 0.0, np.inf
        valid = float(np.min(np.asarray(self.reference_base_mass))) >= float(self.cfg["validity"].get("min_in_domain_base_mass", 0.995))
        for t_idx in range(len(self.times)):
            st = self._exact_tilt(phi_ref_np[t_idx], np.asarray(self.reference_weights[t_idx]), targets[t_idx], lam)
            lam = st.lam
            max_resid = max(max_resid, st.residual_norm)
            min_ess = min(min_ess, st.ess_fraction)
            mass = self._raster_projected_mass(t_idx, jnp.asarray(st.weights))
            vals.append(float(self._law_risk_at_time(mass, t_idx)))
        valid = bool(valid and max_resid <= float(self.cfg["validity"].get("max_population_calibration_resid", 1.0e-5)) and min_ess >= float(self.cfg["validity"].get("min_ess_fraction", 0.03)))
        value = float(np.sum(np.asarray(self.time_w) * np.asarray(vals))) if valid else float("inf")
        out = {"valid": valid, "value": value, "max_calibration_residual": max_resid, "min_ess_fraction": min_ess}
        self._exact_cache[key] = out
        return dict(out)

    def _exact_chunk_results(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        trial_indices: list[int],
        *,
        compute_law: bool,
        compute_tangent: bool,
        compute_full: bool,
    ) -> list[dict[str, Any]]:
        """Authoritative metrics with native batched tilts and robust fallbacks.

        The native solve is accepted only after evaluating the same moment residual
        and ESS diagnostics used by the robust scalar path.  Failed systems either
        receive an exact separating-support certificate or are replaced by
        ``robust_empirical_tilt_exact`` before scientific metrics are formed.  Thus
        batching changes numerical work, not acceptance semantics.
        """
        cache_keys = [
            (
                "trial",
                self._exact_key(eta),
                id(bank),
                int(trial),
                compute_law,
                compute_tangent,
                compute_full,
            )
            for trial in trial_indices
        ]
        if all(key in self._exact_cache for key in cache_keys):
            return [dict(self._exact_cache[key]) for key in cache_keys]

        phi_truth_np, phi_ref_np, grad_np = self._exact_geometry(eta)
        chunk_bank = ObservationTrialBank(
            bank.sample_indices[jnp.asarray(trial_indices, dtype=jnp.int32)],
            bank.detector_z[jnp.asarray(trial_indices, dtype=jnp.int32)],
        )
        rec = self._measurement_reconstruction_batch(
            jnp.asarray(phi_truth_np), chunk_bank
        )
        targets = np.asarray(rec.c, dtype=np.float64)
        projection = self.exact_projector.project_trajectory(
            jnp.asarray(phi_ref_np), self.reference_weights, rec.c
        )
        lam = np.asarray(projection.lam, dtype=np.float64).copy()
        weights = np.asarray(projection.weights, dtype=np.float64).copy()
        moments = np.asarray(projection.moments, dtype=np.float64).copy()
        residual = np.asarray(projection.residual, dtype=np.float64).copy()
        covariance = np.asarray(projection.covariance, dtype=np.float64).copy()
        ess_fraction = np.asarray(projection.ess_fraction, dtype=np.float64).copy()

        projection_cfg = self.cfg.get("projection", {})
        accept_tol = float(projection_cfg.get("solver_accept_tol", 2.0e-6))
        failed = ~np.isfinite(lam).all(axis=-1)
        failed |= ~np.isfinite(weights).all(axis=-1)
        failed |= ~np.isfinite(residual).all(axis=-1)
        failed |= ~np.isfinite(covariance).all(axis=(-2, -1))
        failed |= ~np.isfinite(ess_fraction)
        failed |= np.linalg.norm(residual, axis=-1) > accept_tol
        base_weights = np.asarray(self.reference_weights, dtype=np.float64)
        batch = len(trial_indices)
        support_tol = float(projection_cfg.get("support_certificate_tol", 1.0e-10))
        support_gap = _empirical_coordinate_support_gaps(
            phi_ref_np, base_weights, targets
        )
        projection_valid = support_gap >= -support_tol
        fallback_count = np.zeros(batch, dtype=np.int32)

        # A negative support gap is an exact separating-hyperplane certificate:
        # max_x <d, phi(x)> < <d, target>, so no empirical reweighting can match
        # the target. Coordinate bounds catch simple spline overshoot even if a
        # clipped native dual iterate is a poor certificate direction; the dual
        # check below additionally detects non-coordinate hull violations.
        finite_residual_tol = float(
            self.cfg["validity"].get("max_finite_calibration_resid", 1.0e-3)
        )
        for b, t_idx in np.argwhere(failed):
            direction = lam[b, t_idx]
            direction_norm = float(np.linalg.norm(direction))
            if np.isfinite(direction_norm) and direction_norm > 1.0e-14:
                direction = direction / direction_norm
                active = base_weights[t_idx] > 0.0
                gap = float(
                    np.max(phi_ref_np[t_idx, active] @ direction)
                    - targets[b, t_idx] @ direction
                )
                support_gap[b] = min(support_gap[b], gap)
                if gap < -support_tol:
                    projection_valid[b] = False

        # Only uncertified systems need the robust SciPy fallback.  Work on the
        # largest native residual first so a genuinely invalid trial is rejected
        # early; a failed fallback makes further solves for that trial irrelevant.
        failed_rows = sorted(
            np.argwhere(failed).tolist(),
            key=lambda bt: float(np.linalg.norm(residual[bt[0], bt[1]])),
            reverse=True,
        )
        for b, t_idx in failed_rows:
            if not projection_valid[b]:
                continue
            warm = lam[b, t_idx]
            state = self._exact_tilt(
                phi_ref_np[t_idx],
                base_weights[t_idx],
                targets[b, t_idx],
                warm,
                # The native exact projector has already exhausted this same
                # Newton budget.  Go directly to the independent convex-dual
                # L-BFGS fallback instead of repeating hundreds of host steps.
                newton_steps=int(projection_cfg.get("fallback_newton_steps", 0)),
            )
            fallback_count[b] += 1
            lam[b, t_idx] = state.lam
            weights[b, t_idx] = state.weights
            moments[b, t_idx] = state.moments
            residual[b, t_idx] = state.residual
            covariance[b, t_idx] = state.covariance
            ess_fraction[b, t_idx] = state.ess_fraction
            # Exact action scoring needs an actual projection, not merely a
            # nearby target within the looser experiment-level reporting gate.
            if state.residual_norm > accept_tol:
                projection_valid[b] = False

        max_resid = np.max(np.linalg.norm(residual, axis=-1), axis=1)
        min_ess = np.min(ess_fraction, axis=1)
        time_count = len(self.times)
        tangent_values = np.full((batch, time_count), np.nan, dtype=np.float64)
        max_compat = np.zeros(batch, dtype=np.float64)
        min_cov_eig = np.full(batch, np.inf, dtype=np.float64)
        forcing = np.zeros_like(weights)

        if compute_tangent or compute_full:
            advective = np.einsum(
                "tnmd,tnd->tnm",
                grad_np,
                np.asarray(self.reference_velocity, dtype=np.float64),
            )
            mean_advective = np.einsum("btn,tnm->btm", weights, advective)
        if compute_tangent:
            tangent_residual = mean_advective - np.asarray(rec.c_dot, dtype=np.float64)
            tangent_gram = np.einsum(
                "btn,tnmd,tnkd->btmk", weights, grad_np, grad_np
            )
            ridge = float(
                self.cfg.get("particle_mfsi", {}).get("exact_tangent_ridge", 0.0)
            )
            if ridge:
                tangent_gram = tangent_gram + ridge * np.eye(self.family.n_sensors)
            pinv = np.linalg.pinv(
                tangent_gram,
                rcond=float(
                    self.cfg.get("particle_mfsi", {}).get(
                        "tangent_pinv_rcond", 1.0e-10
                    )
                ),
            )
            tangent_coeff = np.einsum("btmk,btk->btm", pinv, tangent_residual)
            compatibility = np.einsum(
                "btmk,btk->btm", tangent_gram, tangent_coeff
            ) - tangent_residual
            max_compat = np.max(np.linalg.norm(compatibility, axis=-1), axis=1)
            tangent_values = np.einsum(
                "btm,btm->bt", tangent_residual, tangent_coeff
            )

        if compute_full:
            g = np.einsum("tnm,btm->btn", advective, lam)
            mean_g = np.einsum("btn,btn->bt", weights, g)
            centered_phi = phi_ref_np[None, :, :, :] - moments[:, :, None, :]
            cov_phi_g = np.einsum(
                "btn,btnm,btn->btm",
                weights,
                centered_phi,
                g - mean_g[:, :, None],
            )
            cov = covariance.copy()
            exact_ridge = float(
                self.cfg.get("particle_mfsi", {}).get("exact_covariance_ridge", 0.0)
            )
            if exact_ridge:
                cov += exact_ridge * np.eye(self.family.n_sensors)
            eigenvalues = np.linalg.eigvalsh(0.5 * (cov + np.swapaxes(cov, -1, -2)))
            min_cov_by_time = np.min(eigenvalues, axis=-1)
            min_cov_eig = np.min(min_cov_by_time, axis=1)
            rhs = (
                np.asarray(rec.c_dot, dtype=np.float64)
                - mean_advective
                - cov_phi_g
            )
            cov_floor = float(
                self.cfg.get("particle_mfsi", {}).get(
                    "exact_covariance_min_eig", 1.0e-8
                )
            )
            lambda_dot = np.empty_like(rhs)
            for b in range(batch):
                for t_idx in range(time_count):
                    if min_cov_by_time[b, t_idx] <= cov_floor:
                        lambda_dot[b, t_idx] = np.linalg.lstsq(
                            cov[b, t_idx], rhs[b, t_idx], rcond=None
                        )[0]
                    else:
                        lambda_dot[b, t_idx] = np.linalg.solve(
                            cov[b, t_idx], rhs[b, t_idx]
                        )
            forcing = np.einsum("btnm,btm->btn", centered_phi, lambda_dot)
            forcing += g - mean_g[:, :, None]
            forcing -= np.einsum("btn,btn->bt", weights, forcing)[:, :, None]

        law_values = np.full((batch, time_count), np.nan, dtype=np.float64)
        full_values = np.full((batch, time_count), np.nan, dtype=np.float64)
        max_poisson = np.zeros(batch, dtype=np.float64)
        if compute_law or compute_full:
            all_times = jnp.arange(time_count, dtype=jnp.int32)
            rasters = self._raster_trajectory(
                jnp.asarray(weights),
                jnp.asarray(forcing),
                time_idx=all_times,
                grid=self.grid,
            )
            if compute_law:
                law_values = np.asarray(
                    self._law_risk_rows(rasters.mass), dtype=np.float64
                )
            if compute_full:
                actions, poisson_residuals = self._poisson_batch(
                    rasters.q,
                    rasters.h,
                    cfg=self.poisson_cfg,
                    backend=self.full_exact_poisson_backend,
                )
                full_values = np.asarray(actions, dtype=np.float64)
                max_poisson = np.max(
                    np.asarray(poisson_residuals, dtype=np.float64), axis=1
                )

        time_w = np.asarray(self.time_w, dtype=np.float64)
        law = np.sum(law_values * time_w[None, :], axis=1) if compute_law else np.full(batch, np.nan)
        tangent = np.sum(tangent_values * time_w[None, :], axis=1) if compute_tangent else np.full(batch, np.nan)
        full = np.sum(full_values * time_w[None, :], axis=1) if compute_full else np.full(batch, np.nan)
        valid = (
            projection_valid
            & np.isfinite(max_resid)
            & (max_resid <= float(self.cfg["validity"].get("max_finite_calibration_resid", 1.0e-3)))
            & (min_ess >= float(self.cfg["validity"].get("min_ess_fraction", 0.03)))
            & (
                float(np.min(np.asarray(self.reference_base_mass)))
                >= float(self.cfg["validity"].get("min_in_domain_base_mass", 0.995))
            )
        )
        if compute_tangent:
            valid &= max_compat <= float(
                self.cfg.get("particle_mfsi", {}).get(
                    "max_tangent_compatibility_residual", 1.0e-7
                )
            )
        if compute_full:
            valid &= min_cov_eig > float(
                self.cfg.get("particle_mfsi", {}).get(
                    "exact_covariance_min_eig", 1.0e-8
                )
            )
            poisson_gate = self.cfg["validity"].get("max_poisson_relative_residual")
            if poisson_gate is not None:
                valid &= max_poisson <= float(poisson_gate)

        rec_rss = np.asarray(rec.residual_sum_squares, dtype=np.float64)
        rec_roughness = np.asarray(rec.roughness, dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for local, trial in enumerate(trial_indices):
            gap = (
                full[local] - tangent[local]
                if compute_tangent and compute_full
                else float("nan")
            )
            lower_bound_violation = (
                max(tangent[local] - full[local], 0.0)
                if np.isfinite(gap)
                else float("nan")
            )
            row = {
                "trial": int(trial),
                "valid": bool(valid[local]),
                "invalid_reason": (
                    None
                    if valid[local]
                    else (
                        "target_outside_empirical_moment_hull"
                        if support_gap[local] < -support_tol
                        else "calibration_ess_identifiability_or_numerical_gate"
                    )
                ),
                "law_risk": float(law[local]) if valid[local] else float("nan"),
                "tangent_action": float(tangent[local]) if valid[local] else float("nan"),
                "full_action": float(full[local]) if valid[local] else float("nan"),
                "max_calibration_residual": float(max_resid[local]),
                "min_ess_fraction": float(min_ess[local]),
                "max_poisson_relative_residual": float(max_poisson[local]) if compute_full else float("nan"),
                "max_tangent_compatibility_residual": float(max_compat[local]) if compute_tangent else float("nan"),
                "min_covariance_eigenvalue": float(min_cov_eig[local]) if compute_full else float("nan"),
                "spline_residual_sum_squares": float(rec_rss[local]),
                "spline_roughness": float(rec_roughness[local]),
                "tangent_full_gap": float(gap) if valid[local] else float("nan"),
                "tangent_lower_bound_violation": float(lower_bound_violation) if valid[local] else float("nan"),
                "native_projection_failed_systems": int(np.sum(failed[local])),
                "robust_projection_fallback_systems": int(fallback_count[local]),
                "min_empirical_hull_support_gap": (
                    float(support_gap[local])
                    if np.isfinite(support_gap[local])
                    else float("nan")
                ),
            }
            self._exact_cache[cache_keys[local]] = row
            rows.append(dict(row))
        return rows

    def _evaluate_exact_batched(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        *,
        compute_law: bool,
        compute_tangent: bool,
        compute_full: bool,
        trial_count: int | None = None,
        progress_desc: str | None = None,
        stop_on_invalid: bool = False,
    ) -> list[dict[str, Any]]:
        count = int(bank.sample_indices.shape[0])
        if trial_count is not None:
            count = min(count, int(trial_count))
        chunk_size = max(
            1,
            int(self.cfg.get("optimization", {}).get("exact_batch_trials", 4)),
        )
        chunks = [list(range(start, min(start + chunk_size, count))) for start in range(0, count, chunk_size)]
        iterator = chunks
        if progress_desc is not None:
            iterator = _progress_iter(chunks, desc=progress_desc, total=len(chunks))
        rows: list[dict[str, Any]] = []
        for indices in iterator:
            chunk_rows = self._exact_chunk_results(
                eta,
                bank,
                indices,
                compute_law=compute_law,
                compute_tangent=compute_tangent,
                compute_full=compute_full,
            )
            rows.extend(chunk_rows)
            if stop_on_invalid and any(not row["valid"] for row in chunk_rows):
                break
        return rows

    def _exact_trial_result(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        trial: int,
        *,
        compute_law: bool,
        compute_tangent: bool,
        compute_full: bool,
    ) -> dict[str, Any]:
        cache_key = ("trial", self._exact_key(eta), id(bank), int(trial), compute_law, compute_tangent, compute_full)
        if cache_key in self._exact_cache:
            return dict(self._exact_cache[cache_key])
        phi_truth_np, phi_ref_np, grad_ref_np = self._exact_geometry(eta)
        phi_truth = jnp.asarray(phi_truth_np)
        rec = self._measurement_reconstruction(phi_truth, bank, trial)
        grad_np = grad_ref_np if (compute_tangent or compute_full) else None
        rec_c = np.asarray(rec.c)
        rec_cd = np.asarray(rec.c_dot)
        projection_cfg = self.cfg.get("projection", {})
        support_tol = float(projection_cfg.get("support_certificate_tol", 1.0e-10))
        support_gap = float(
            _empirical_coordinate_support_gaps(
                phi_ref_np,
                np.asarray(self.reference_weights, dtype=np.float64),
                rec_c,
            )[0]
        )
        lam = np.zeros(self.family.n_sensors, dtype=np.float64)
        law_vals, tan_vals, full_vals = [], [], []
        full_q_rows, full_h_rows = [], []
        max_resid, min_ess, max_poisson = 0.0, np.inf, 0.0
        max_compat, min_cov_eig = 0.0, np.inf
        valid = (
            float(np.min(np.asarray(self.reference_base_mass)))
            >= float(self.cfg["validity"].get("min_in_domain_base_mass", 0.995))
            and support_gap >= -support_tol
        )
        for t_idx in range(len(self.times)):
            st = self._exact_tilt(phi_ref_np[t_idx], np.asarray(self.reference_weights[t_idx]), rec_c[t_idx], lam)
            lam = st.lam
            max_resid = max(max_resid, st.residual_norm)
            min_ess = min(min_ess, st.ess_fraction)
            m = mean_m = None
            if compute_tangent or compute_full:
                m = np.einsum("nmd,nd->nm", grad_np[t_idx], np.asarray(self.reference_velocity[t_idx]))
                mean_m = np.sum(st.weights[:, None] * m, axis=0)
            if compute_tangent:
                r = mean_m - rec_cd[t_idx]
                G = np.einsum("n,nmd,nkd->mk", st.weights, grad_np[t_idx], grad_np[t_idx])
                ridge = float(self.cfg.get("particle_mfsi", {}).get("exact_tangent_ridge", 0.0))
                if ridge:
                    G = G + ridge * np.eye(G.shape[0])
                pinv = np.linalg.pinv(G, rcond=float(self.cfg.get("particle_mfsi", {}).get("tangent_pinv_rcond", 1.0e-10)))
                coeff = pinv @ r
                max_compat = max(max_compat, float(np.linalg.norm(G @ coeff - r)))
                tan_vals.append(float(r @ coeff))
            forcing = np.zeros_like(st.weights)
            if compute_full:
                gg = m @ st.lam
                mean_g = float(np.sum(st.weights * gg))
                centered = phi_ref_np[t_idx] - st.moments[None, :]
                cov_phi_g = np.sum(st.weights[:, None] * centered * (gg - mean_g)[:, None], axis=0)
                cov = np.asarray(st.covariance)
                exact_ridge = float(self.cfg.get("particle_mfsi", {}).get("exact_covariance_ridge", 0.0))
                if exact_ridge:
                    cov = cov + exact_ridge * np.eye(cov.shape[0])
                eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T))))
                min_cov_eig = min(min_cov_eig, eig_min)
                rhs = rec_cd[t_idx] - mean_m - cov_phi_g
                floor = float(self.cfg.get("particle_mfsi", {}).get("exact_covariance_min_eig", 1.0e-8))
                if eig_min <= floor:
                    valid = False
                    lam_dot = np.linalg.lstsq(cov, rhs, rcond=None)[0]
                else:
                    lam_dot = np.linalg.solve(cov, rhs)
                forcing = centered @ lam_dot + gg - mean_g
                forcing -= float(np.sum(st.weights * forcing))
            if compute_law or compute_full:
                ras = rasterize_projected_particles_rect(
                    self.reference_nodes[t_idx], jnp.asarray(st.weights), jnp.asarray(forcing), self.grid, self.raster_cfg
                )
                if compute_law:
                    law_vals.append(float(self._law_risk_at_time(ras.mass, t_idx)))
                if compute_full:
                    if self.full_exact_poisson_backend == "tesseract_cpp":
                        full_q_rows.append(ras.q)
                        full_h_rows.append(ras.h)
                    else:
                        pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_cfg)
                        full_vals.append(float(pois.action))
                        max_poisson = max(max_poisson, float(pois.relative_residual))

        if compute_full and self.full_exact_poisson_backend == "tesseract_cpp":
            q_batch = jnp.stack(full_q_rows)[None, ...]
            h_batch = jnp.stack(full_h_rows)[None, ...]
            actions, residuals = self._poisson_batch(
                q_batch,
                h_batch,
                cfg=self.poisson_cfg,
                backend=self.full_exact_poisson_backend,
            )
            full_vals = np.asarray(actions[0], dtype=np.float64).tolist()
            max_poisson = float(np.max(np.asarray(residuals[0], dtype=np.float64)))

        valid = bool(
            valid
            and max_resid <= float(projection_cfg.get("solver_accept_tol", 2.0e-6))
            and min_ess >= float(self.cfg["validity"].get("min_ess_fraction", 0.03))
        )
        if compute_tangent:
            valid = valid and max_compat <= float(self.cfg.get("particle_mfsi", {}).get("max_tangent_compatibility_residual", 1.0e-7))
        if compute_full and self.cfg["validity"].get("max_poisson_relative_residual") is not None:
            valid = valid and max_poisson <= float(self.cfg["validity"]["max_poisson_relative_residual"])

        law = float(np.sum(np.asarray(self.time_w) * np.asarray(law_vals))) if compute_law else float("nan")
        tangent = float(np.sum(np.asarray(self.time_w) * np.asarray(tan_vals))) if compute_tangent else float("nan")
        full = float(np.sum(np.asarray(self.time_w) * np.asarray(full_vals))) if compute_full else float("nan")
        gap = full - tangent if compute_tangent and compute_full and np.isfinite(full) and np.isfinite(tangent) else float("nan")
        lbv = max(tangent - full, 0.0) if np.isfinite(gap) else float("nan")
        if not valid:
            law = tangent = full = float("nan")
            gap = lbv = float("nan")
        out = {
            "trial": int(trial), "valid": valid,
            "invalid_reason": (
                None
                if valid
                else (
                    "target_outside_empirical_moment_hull"
                    if support_gap < -support_tol
                    else "calibration_ess_identifiability_or_numerical_gate"
                )
            ),
            "law_risk": law, "tangent_action": tangent, "full_action": full,
            "max_calibration_residual": float(max_resid), "min_ess_fraction": float(min_ess),
            "max_poisson_relative_residual": float(max_poisson) if compute_full else float("nan"),
            "max_tangent_compatibility_residual": float(max_compat) if compute_tangent else float("nan"),
            "min_covariance_eigenvalue": float(min_cov_eig) if compute_full else float("nan"),
            "spline_residual_sum_squares": float(rec.residual_sum_squares),
            "spline_roughness": float(rec.roughness),
            "tangent_full_gap": gap, "tangent_lower_bound_violation": lbv,
        }
        self._exact_cache[cache_key] = out
        return dict(out)

    def exact_finite_result(self, eta: Array, bank: ObservationTrialBank) -> dict[str, Any]:
        rows = self._evaluate_exact_batched(
            eta,
            bank,
            compute_law=True,
            compute_tangent=False,
            compute_full=False,
            stop_on_invalid=True,
        )
        valid = bool(all(r["valid"] and np.isfinite(r["law_risk"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["law_risk"] for r in rows])) if valid else float("inf"), "rows": rows}

    def exact_tangent_result(self, eta: Array, bank: ObservationTrialBank) -> dict[str, Any]:
        rows = self._evaluate_exact_batched(
            eta,
            bank,
            compute_law=False,
            compute_tangent=True,
            compute_full=False,
            stop_on_invalid=True,
        )
        valid = bool(all(r["valid"] and np.isfinite(r["tangent_action"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["tangent_action"] for r in rows])) if valid else float("inf"), "rows": rows}

    def exact_full_result(self, eta: Array, bank: ObservationTrialBank, *, trial_count: int | None = None) -> dict[str, Any]:
        rows = self._evaluate_exact_batched(
            eta,
            bank,
            compute_law=False,
            compute_tangent=False,
            compute_full=True,
            trial_count=trial_count,
            stop_on_invalid=True,
        )
        valid = bool(all(r["valid"] and np.isfinite(r["full_action"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["full_action"] for r in rows])) if valid else float("inf"), "rows": rows}

    def evaluate_trials_exact(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        *,
        progress_desc: str | None = None,
    ) -> list[dict[str, Any]]:
        centers = np.asarray(self.family.centers(self.family.canonicalize(eta)), dtype=np.float64).tolist()
        out = self._evaluate_exact_batched(
            eta,
            bank,
            compute_law=True,
            compute_tangent=True,
            compute_full=True,
            progress_desc=progress_desc,
        )
        for row in out:
            row["centers"] = centers
        return out


def _paired_reduction(full_values: list[float], law_values: list[float]) -> dict[str, float | int]:
    a = np.asarray(full_values, dtype=np.float64)
    b = np.asarray(law_values, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"n": 0, "ratio_of_means_reduction": float("nan"), "mean_paired_difference": float("nan")}
    return {
        "n": int(len(a)),
        "ratio_of_means_reduction": float(1.0 - np.mean(a) / np.mean(b)),
        "mean_paired_difference": float(np.mean(b - a)),
    }


def run_experiment(cfg: dict[str, Any], output_dir: Path, *, smoke: bool = False) -> dict[str, Any]:
    run_started = time.perf_counter()
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("validity", {})
    cfg["validity"].setdefault("max_population_calibration_resid", 1.0e-5)
    cfg["validity"].setdefault("max_finite_calibration_resid", 1.0e-3)
    cfg["validity"].setdefault("min_ess_fraction", 0.03)
    cfg["validity"].setdefault("min_in_domain_base_mass", 0.995)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_hash = fingerprint(cfg)
    timing_path = output_dir / "run_timing.json"
    previous_phases: dict[str, float] = {}
    if timing_path.exists():
        try:
            previous = json.loads(timing_path.read_text(encoding="utf-8"))
            if previous.get("config_hash") == config_hash:
                previous_phases = {
                    str(name): float(seconds)
                    for name, seconds in previous.get("phases_seconds", {}).items()
                }
                print(
                    "[timing] previous compatible run took "
                    f"{_format_duration(float(previous['total_seconds']))}",
                    flush=True,
                )
        except (OSError, ValueError, TypeError, KeyError):
            print("[timing] ignoring unreadable prior run_timing.json", flush=True)

    phase_timings: dict[str, float] = {}

    def begin_phase(name: str) -> float:
        expected = previous_phases.get(name)
        suffix = (
            f"; previous compatible run={_format_duration(expected)}"
            if expected is not None
            else ""
        )
        print(f"[timing] starting {name}{suffix}", flush=True)
        return time.perf_counter()

    def finish_phase(name: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        phase_timings[name] = elapsed
        print(
            f"[timing] finished {name} in {_format_duration(elapsed)}; "
            f"run elapsed={_format_duration(time.perf_counter() - run_started)}",
            flush=True,
        )

    def timing_payload() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config_hash": config_hash,
            "total_seconds": float(time.perf_counter() - run_started),
            "phases_seconds": dict(phase_timings),
        }

    setup_started = begin_phase("setup_and_cached_inputs")

    times = jnp.linspace(0.0, 1.0, int(cfg["poisson"]["time_n"]), dtype=jnp.float64)
    truth = _truth_from_cfg(cfg)
    truth_particles, truth_signature = ensure_truth_bank(truth, cfg, output_dir, times)
    endpoint_source, endpoint_signature = ensure_reference_endpoints(truth, cfg, output_dir)
    reference, checkpoint, reference_metadata = ensure_reference(endpoint_source, endpoint_signature, cfg, output_dir)
    ref_nodes, ref_velocity, ref_weights = ensure_reference_bank(truth, reference, checkpoint, cfg, output_dir, times)
    exp = VortexExperiment(
        cfg, reference,
        truth_particles=truth_particles,
        reference_nodes=ref_nodes,
        reference_velocity=ref_velocity,
        reference_weights=ref_weights,
    )

    rnd = cfg["randomness"]
    law_trials = int(rnd["law_trials"])
    action_trials = int(rnd["action_trials"])
    validation_trials = int(rnd["validation_trials"])
    selection_trials = max(law_trials, action_trials)
    selection = ensure_observation_bank(
        name="selection", exp=exp, trials=selection_trials,
        namespace=int(rnd["selection_namespace"]), output_dir=output_dir,
    )
    validation = ensure_observation_bank(
        name="validation", exp=exp, trials=validation_trials,
        namespace=int(rnd["validation_namespace"]), output_dir=output_dir,
    )
    law_bank = prefix_bank(selection, law_trials)
    action_bank = prefix_bank(selection, action_trials)

    meas = cfg["measurement"]
    margin = float(meas.get("boundary_margin", 2.0 * float(meas.get("sensor_width", 0.12))))
    x_bounds = (float(cfg["poisson"].get("x_min", 0.0)) + margin, float(cfg["poisson"].get("x_max", 2.0)) - margin)
    y_bounds = (float(cfg["poisson"].get("y_min", 0.0)) + margin, float(cfg["poisson"].get("y_max", 1.0)) - margin)
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 17),
        int(cfg["optimization"]["start_count"]),
        n_sensors=int(meas.get("n_sensors", 4)), x_bounds=x_bounds, y_bounds=y_bounds,
        min_sep=float(meas.get("min_sep", 0.24)),
        oversample=int(cfg["optimization"].get("start_oversample", 64)),
    )
    print(
        "[backends] "
        f"I-projection={exp.iprojection_backend}, "
        f"Poisson proxy={exp.full_gradient_poisson_backend}, "
        f"Poisson exact={exp.full_exact_poisson_backend}",
        flush=True,
    )
    finish_phase("setup_and_cached_inputs", setup_started)

    if smoke:
        smoke_started = begin_phase("smoke_exact_path")
        probes = random_point_sensor_starts(
            jax.random.PRNGKey(int(cfg["seed"]) + 17017),
            max(16, int(cfg["optimization"].get("smoke_probe_count", 32))),
            n_sensors=int(meas.get("n_sensors", 4)), x_bounds=x_bounds, y_bounds=y_bounds,
            min_sep=float(meas.get("min_sep", 0.24)), oversample=128,
        )
        smoke_bank = prefix_bank(selection, 1)
        chosen = None
        for eta in _progress_iter(probes, desc="smoke: exact-valid design search"):
            pre = exp._exact_trial_result(eta, smoke_bank, 0, compute_law=False, compute_tangent=False, compute_full=False)
            if pre["valid"]:
                chosen = eta
                break
        if chosen is None:
            raise RuntimeError("smoke could not find an exact-valid point-sensor design")
        metrics = exp.evaluate_trials_exact(chosen, smoke_bank)[0]
        finish_phase("smoke_exact_path", smoke_started)
        timings = timing_payload()
        result = {
            "schema_version": 1, "experiment": "vortices_double_gyre", "smoke": True,
            "config": cfg, "config_hash": config_hash, "timings_seconds": timings,
            "backends": {
                "iprojection": exp.iprojection_backend,
                "full_gradient_poisson": exp.full_gradient_poisson_backend,
                "full_exact_poisson": exp.full_exact_poisson_backend,
            },
            "truth": {"signature": truth_signature, "min_in_domain_fraction": float(jnp.min(exp.truth_in_domain_fraction))},
            "reference": {"checkpoint": str(checkpoint), "metadata": reference_metadata, "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass))},
            "smoke_design": np.asarray(chosen, dtype=np.float64).tolist(),
            "smoke_centers": np.asarray(exp.family.centers(chosen), dtype=np.float64).tolist(),
            "smoke_metrics": metrics,
        }
        _write_json(output_dir / "result.json", result)
        _write_json(timing_path, timings)
        return result

    from selection import optimize_vortex_designs

    selection_started = begin_phase("stages_1_4_selection")
    sel = optimize_vortex_designs(exp, law_bank, action_bank, starts, output_dir)
    finish_phase("stages_1_4_selection", selection_started)
    designs = {
        "population": sel["population_eta"], "law": sel["law_eta"],
        "tangent": sel["tangent_eta"], "full": sel["full_eta"],
    }
    validation_started = begin_phase("validation_and_certification")
    print("[validation] evaluating disjoint observation bank", flush=True)
    validation_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    per_design: dict[str, list[dict[str, Any]]] = {}
    for name, eta in designs.items():
        rows = exp.evaluate_trials_exact(eta, validation, progress_desc=f"validation {name}")
        for row in rows:
            row["design"] = name
        per_design[name] = rows
        validation_rows.extend(rows)
        valid_rows = [r for r in rows if r["valid"]]
        lb = [r["tangent_lower_bound_violation"] for r in valid_rows if np.isfinite(r["tangent_lower_bound_violation"])]
        summaries[name] = {
            "eta": np.asarray(eta, dtype=np.float64).tolist(),
            "centers": np.asarray(exp.family.centers(eta), dtype=np.float64).tolist(),
            "law_risk": _mean_se([r["law_risk"] for r in valid_rows]),
            "tangent_action": _mean_se([r["tangent_action"] for r in valid_rows]),
            "full_action": _mean_se([r["full_action"] for r in valid_rows]),
            "valid_fraction": float(len(valid_rows) / max(len(rows), 1)),
            "tangent_lower_bound_check": {
                "max_violation": float(max(lb, default=0.0)),
                "tolerance": float(cfg["validity"].get("tangent_lower_bound_tol", 1.0e-6)),
            },
        }

    candidate_rows = []
    for name, eta in designs.items():
        candidate_rows.append({
            "design": name,
            "eta": json.dumps(np.asarray(eta, dtype=np.float64).tolist()),
            "centers": json.dumps(np.asarray(exp.family.centers(eta), dtype=np.float64).tolist()),
            "population_loss_selection": float(exp.exact_population_result(eta)["value"]),
            "finite_risk_selection": float(exp.exact_finite_result(eta, law_bank)["value"]),
            "tangent_action_selection": float(exp.exact_tangent_result(eta, action_bank)["value"]),
            "full_action_selection": float(exp.exact_full_result(eta, action_bank)["value"]),
            "validation_law_mean": summaries[name]["law_risk"]["mean"],
            "validation_full_action_mean": summaries[name]["full_action"]["mean"],
            "validation_valid_fraction": summaries[name]["valid_fraction"],
        })

    L_star, L_max = float(sel["L_star"]), float(sel["L_max"])
    R_star, R_max = float(sel["R_star"]), float(sel["R_max"])
    certificates = {}
    for row in candidate_rows:
        name = row["design"]
        L, R = float(row["population_loss_selection"]), float(row["finite_risk_selection"])
        required = ["L"] if name == "population" else (["L"] if name == "law" else ["L", "R"])
        passes_L = np.isfinite(L) and L <= L_max + 1.0e-12
        passes_R = np.isfinite(R) and R <= R_max + 1.0e-12
        certificates[name] = {
            "required_screens": required, "L_selection": L, "L_star": L_star, "L_max": L_max,
            "L_excess_from_star": L - L_star, "L_slack_to_max": L_max - L, "passes_L": bool(passes_L),
            "R_selection": R, "R_star": R_star, "R_max": R_max,
            "R_excess_from_star": R - R_star, "R_slack_to_max": R_max - R, "passes_R": bool(passes_R),
            "certified": bool(passes_L and (passes_R if "R" in required else True)),
        }

    result = {
        "schema_version": 1, "experiment": "vortices_double_gyre", "smoke": False,
        "config": cfg, "config_hash": config_hash,
        "backends": {
            "iprojection": exp.iprojection_backend,
            "full_gradient_poisson": exp.full_gradient_poisson_backend,
            "full_exact_poisson": exp.full_exact_poisson_backend,
        },
        "truth": {"signature": truth_signature, "truth_bank": "truth_bank.npz", "min_in_domain_fraction": float(jnp.min(exp.truth_in_domain_fraction))},
        "reference": {"checkpoint": str(checkpoint), "metadata": reference_metadata, "reference_bank": "reference_bank.npz", "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass))},
        "randomness": {"selection_bank": "selection_bank.npz", "validation_bank": "validation_bank.npz", "law_trials_effective": law_trials, "action_trials_effective": action_trials, "validation_trials_effective": validation_trials},
        "law_screens": {"L_star": L_star, "L_max": L_max, "R_star": R_star, "R_max": R_max, "epsilon_l": L_max - L_star, "epsilon_r": R_max - R_star},
        "selection": {name + "_optimum": np.asarray(eta, dtype=np.float64).tolist() for name, eta in designs.items()},
        "selection_centers": {name: np.asarray(exp.family.centers(eta), dtype=np.float64).tolist() for name, eta in designs.items()},
        "selection_certificates": certificates,
        "selection_audit": sel.get("audit", {}),
        "selection_timings_seconds": sel.get("stage_timings_seconds", {}),
        "validation": summaries,
        "contrasts": {"full_vs_law_full_action_reduction": _paired_reduction(
            [r["full_action"] for r in per_design["full"]], [r["full_action"] for r in per_design["law"]]
        )},
    }
    finish_phase("validation_and_certification", validation_started)
    timings = timing_payload()
    result["timings_seconds"] = timings
    _write_json(output_dir / "result.json", result)
    _write_json(timing_path, timings)
    _write_csv(output_dir / "result.candidate_summary.csv", candidate_rows)
    _write_csv(output_dir / "result.validation_trials.csv", validation_rows)
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1, "config_hash": result["config_hash"],
        "artifacts": {
            "truth_bank": "truth_bank.npz", "reference_checkpoint": "reference.npz",
            "reference_endpoints": "reference_endpoints.npz", "reference_bank": "reference_bank.npz",
            "selection_bank": "selection_bank.npz", "validation_bank": "validation_bank.npz",
            "result": "result.json", "candidate_summary": "result.candidate_summary.csv",
            "validation_trials": "result.validation_trials.csv", "run_timing": "run_timing.json",
        },
    })
    return result
