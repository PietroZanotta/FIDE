from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
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
from mfsi.poisson import PoissonConfig, solve_weighted_poisson
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.raster import RasterConfig, rasterize_projected_particles_rect
from mfsi.reference import MLPReferenceFlow, save_npz_checkpoint

Array = jax.Array



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
        self.projector = EmpiricalIProjector(IProjectionConfig(
            max_steps=int(proj.get("search_max_steps", proj.get("max_steps", 300))),
            residual_tol=float(proj.get("search_residual_tol", proj.get("residual_tol", 1.0e-10))),
            newton_ridge=float(proj.get("newton_ridge", 1.0e-7)),
            step_cap=float(proj.get("step_cap", 20.0)),
            lambda_clip=float(proj.get("lambda_clip", 1000.0)),
            line_search_steps=int(proj.get("search_line_search_steps", proj.get("line_search_steps", 8))),
            implicit_ridge=float(proj.get("implicit_ridge", 0.0)),
        ))
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
        self._exact_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

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
        targets = jnp.mean(phi_truth, axis=1)
        lam = jnp.zeros((self.family.n_sensors,), dtype=jnp.float64)
        vals = []
        max_resid = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        for t_idx in range(len(self.times)):
            st = self.projector.project(phi_ref[t_idx], self.reference_weights[t_idx], targets[t_idx], lam0=lam)
            lam = st.lam
            max_resid = jnp.maximum(max_resid, jnp.linalg.norm(st.residual))
            min_ess = jnp.minimum(min_ess, st.ess_fraction)
            qmass = self._raster_projected_mass(t_idx, st.weights)
            vals.append(gaussian_mmd2_grid_mass(qmass, self.truth_masses[t_idx], self.mmd_kernel))
        risk = jnp.sum(self.time_w * jnp.stack(vals))
        v = self.cfg.get("validity", {})
        valid = (
            (max_resid <= float(v.get("max_population_calibration_resid", 1.0e-5)))
            & (min_ess >= float(v.get("min_ess_fraction", 0.03)))
            & (jnp.min(self.reference_base_mass) >= float(v.get("min_in_domain_base_mass", 0.995)))
        )
        return jnp.where(valid, risk, risk + float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3)))

    def finite_risk(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, _ = self._geometry(eta)
        rows = []
        for trial in range(int(bank.sample_indices.shape[0])):
            rec = self._measurement_reconstruction(phi_truth, bank, trial)
            lam = jnp.zeros((self.family.n_sensors,), dtype=jnp.float64)
            vals = []
            max_resid = jnp.asarray(0.0)
            min_ess = jnp.asarray(jnp.inf)
            for t_idx in range(len(self.times)):
                st = self.projector.project(phi_ref[t_idx], self.reference_weights[t_idx], rec.c[t_idx], lam0=lam)
                lam = st.lam
                max_resid = jnp.maximum(max_resid, jnp.linalg.norm(st.residual))
                min_ess = jnp.minimum(min_ess, st.ess_fraction)
                qmass = self._raster_projected_mass(t_idx, st.weights)
                vals.append(gaussian_mmd2_grid_mass(qmass, self.truth_masses[t_idx], self.mmd_kernel))
            risk = jnp.sum(self.time_w * jnp.stack(vals))
            valid = self._validity(max_resid, min_ess)
            rows.append(jnp.where(valid, risk, risk + float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3))))
        return jnp.mean(jnp.stack(rows))

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
        vals = []
        for trial in range(int(bank.sample_indices.shape[0])):
            row = self._one_trial_metrics_from_geometry(phi_truth, phi_ref, grad_ref, bank, trial, full=False)
            vals.append(jnp.where(row.valid, row.tangent_action, row.tangent_action + 1.0e5))
        return jnp.mean(jnp.stack(vals))

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
        vals = [
            self._one_trial_full_action_gradient(phi_truth, phi_ref, grad_ref, bank, trial)
            for trial in range(int(bank.sample_indices.shape[0]))
        ]
        return jnp.mean(jnp.stack(vals))

    def full_action(self, eta: Array, bank: ObservationTrialBank) -> Array:
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        vals = []
        for trial in range(int(bank.sample_indices.shape[0])):
            row = self._one_trial_metrics_from_geometry(phi_truth, phi_ref, grad_ref, bank, trial, full=True)
            vals.append(jnp.where(row.valid, row.full_action, row.full_action + 1.0e5))
        return jnp.mean(jnp.stack(vals))

    def _exact_tilt(self, phi: np.ndarray, base: np.ndarray, target: np.ndarray, lam0: np.ndarray):
        p = self.cfg.get("projection", {})
        return robust_empirical_tilt_exact(
            phi, base, target, lam0=lam0,
            newton_steps=int(p.get("max_steps", 300)),
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

    def exact_population_result(self, eta: Array) -> dict[str, Any]:
        key = ("population", self._exact_key(eta))
        if key in self._exact_cache:
            return dict(self._exact_cache[key])
        phi_truth, phi_ref, _ = self._geometry(eta)
        phi_truth_np = np.asarray(phi_truth)
        phi_ref_np = np.asarray(phi_ref)
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
            vals.append(float(gaussian_mmd2_grid_mass(mass, self.truth_masses[t_idx], self.mmd_kernel)))
        valid = bool(valid and max_resid <= float(self.cfg["validity"].get("max_population_calibration_resid", 1.0e-5)) and min_ess >= float(self.cfg["validity"].get("min_ess_fraction", 0.03)))
        value = float(np.sum(np.asarray(self.time_w) * np.asarray(vals))) if valid else float("inf")
        out = {"valid": valid, "value": value, "max_calibration_residual": max_resid, "min_ess_fraction": min_ess}
        self._exact_cache[key] = out
        return dict(out)

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
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        rec = self._measurement_reconstruction(phi_truth, bank, trial)
        phi_ref_np = np.asarray(phi_ref)
        grad_np = np.asarray(grad_ref) if (compute_tangent or compute_full) else None
        rec_c = np.asarray(rec.c)
        rec_cd = np.asarray(rec.c_dot)
        lam = np.zeros(self.family.n_sensors, dtype=np.float64)
        law_vals, tan_vals, full_vals = [], [], []
        max_resid, min_ess, max_poisson = 0.0, np.inf, 0.0
        max_compat, min_cov_eig = 0.0, np.inf
        valid = float(np.min(np.asarray(self.reference_base_mass))) >= float(self.cfg["validity"].get("min_in_domain_base_mass", 0.995))
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
                floor = float(self.cfg.get("particle_mfsi", {}).get("exact_covariance_min_eig", 1.0e-12))
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
                    law_vals.append(float(gaussian_mmd2_grid_mass(ras.mass, self.truth_masses[t_idx], self.mmd_kernel)))
                if compute_full:
                    pois = solve_weighted_poisson(ras.q, ras.h, self.poisson_cfg)
                    full_vals.append(float(pois.action))
                    max_poisson = max(max_poisson, float(pois.relative_residual))

        valid = bool(
            valid
            and max_resid <= float(self.cfg["validity"].get("max_finite_calibration_resid", 1.0e-3))
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
            "invalid_reason": None if valid else "calibration_ess_identifiability_or_numerical_gate",
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
        rows = [self._exact_trial_result(eta, bank, r, compute_law=True, compute_tangent=False, compute_full=False) for r in range(int(bank.sample_indices.shape[0]))]
        valid = bool(all(r["valid"] and np.isfinite(r["law_risk"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["law_risk"] for r in rows])) if valid else float("inf"), "rows": rows}

    def exact_tangent_result(self, eta: Array, bank: ObservationTrialBank) -> dict[str, Any]:
        rows = [self._exact_trial_result(eta, bank, r, compute_law=False, compute_tangent=True, compute_full=False) for r in range(int(bank.sample_indices.shape[0]))]
        valid = bool(all(r["valid"] and np.isfinite(r["tangent_action"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["tangent_action"] for r in rows])) if valid else float("inf"), "rows": rows}

    def exact_full_result(self, eta: Array, bank: ObservationTrialBank, *, trial_count: int | None = None) -> dict[str, Any]:
        count = int(bank.sample_indices.shape[0]) if trial_count is None else min(int(trial_count), int(bank.sample_indices.shape[0]))
        rows = [self._exact_trial_result(eta, bank, r, compute_law=False, compute_tangent=False, compute_full=True) for r in range(count)]
        valid = bool(all(r["valid"] and np.isfinite(r["full_action"]) for r in rows))
        return {"valid": valid, "value": float(np.mean([r["full_action"] for r in rows])) if valid else float("inf"), "rows": rows}

    def evaluate_trials_exact(self, eta: Array, bank: ObservationTrialBank) -> list[dict[str, Any]]:
        centers = np.asarray(self.family.centers(self.family.canonicalize(eta)), dtype=np.float64).tolist()
        out = []
        for r in range(int(bank.sample_indices.shape[0])):
            row = self._exact_trial_result(eta, bank, r, compute_law=True, compute_tangent=True, compute_full=True)
            row["centers"] = centers
            out.append(row)
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
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("validity", {})
    cfg["validity"].setdefault("max_population_calibration_resid", 1.0e-5)
    cfg["validity"].setdefault("max_finite_calibration_resid", 1.0e-3)
    cfg["validity"].setdefault("min_ess_fraction", 0.03)
    cfg["validity"].setdefault("min_in_domain_base_mass", 0.995)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if smoke:
        probes = random_point_sensor_starts(
            jax.random.PRNGKey(int(cfg["seed"]) + 17017),
            max(16, int(cfg["optimization"].get("smoke_probe_count", 32))),
            n_sensors=int(meas.get("n_sensors", 4)), x_bounds=x_bounds, y_bounds=y_bounds,
            min_sep=float(meas.get("min_sep", 0.24)), oversample=128,
        )
        smoke_bank = prefix_bank(selection, 1)
        chosen = None
        for eta in probes:
            pre = exp._exact_trial_result(eta, smoke_bank, 0, compute_law=False, compute_tangent=False, compute_full=False)
            if pre["valid"]:
                chosen = eta
                break
        if chosen is None:
            raise RuntimeError("smoke could not find an exact-valid point-sensor design")
        metrics = exp.evaluate_trials_exact(chosen, smoke_bank)[0]
        result = {
            "schema_version": 1, "experiment": "vortices_double_gyre", "smoke": True,
            "config": cfg, "config_hash": fingerprint(cfg),
            "truth": {"signature": truth_signature, "min_in_domain_fraction": float(jnp.min(exp.truth_in_domain_fraction))},
            "reference": {"checkpoint": str(checkpoint), "metadata": reference_metadata, "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass))},
            "smoke_design": np.asarray(chosen, dtype=np.float64).tolist(),
            "smoke_centers": np.asarray(exp.family.centers(chosen), dtype=np.float64).tolist(),
            "smoke_metrics": metrics,
        }
        _write_json(output_dir / "result.json", result)
        return result

    from selection import optimize_vortex_designs

    sel = optimize_vortex_designs(exp, law_bank, action_bank, starts, output_dir)
    designs = {
        "population": sel["population_eta"], "law": sel["law_eta"],
        "tangent": sel["tangent_eta"], "full": sel["full_eta"],
    }
    print("[validation] evaluating disjoint observation bank", flush=True)
    validation_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    per_design: dict[str, list[dict[str, Any]]] = {}
    for name, eta in designs.items():
        rows = exp.evaluate_trials_exact(eta, validation)
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
        "config": cfg, "config_hash": fingerprint(cfg),
        "truth": {"signature": truth_signature, "truth_bank": "truth_bank.npz", "min_in_domain_fraction": float(jnp.min(exp.truth_in_domain_fraction))},
        "reference": {"checkpoint": str(checkpoint), "metadata": reference_metadata, "reference_bank": "reference_bank.npz", "min_in_domain_base_mass": float(jnp.min(exp.reference_base_mass))},
        "randomness": {"selection_bank": "selection_bank.npz", "validation_bank": "validation_bank.npz", "law_trials_effective": law_trials, "action_trials_effective": action_trials, "validation_trials_effective": validation_trials},
        "law_screens": {"L_star": L_star, "L_max": L_max, "R_star": R_star, "R_max": R_max, "epsilon_l": L_max - L_star, "epsilon_r": R_max - R_star},
        "selection": {name + "_optimum": np.asarray(eta, dtype=np.float64).tolist() for name, eta in designs.items()},
        "selection_centers": {name: np.asarray(exp.family.centers(eta), dtype=np.float64).tolist() for name, eta in designs.items()},
        "selection_certificates": certificates,
        "selection_audit": sel.get("audit", {}),
        "validation": summaries,
        "contrasts": {"full_vs_law_full_action_reduction": _paired_reduction(
            [r["full_action"] for r in per_design["full"]], [r["full_action"] for r in per_design["law"]]
        )},
    }
    _write_json(output_dir / "result.json", result)
    _write_csv(output_dir / "result.candidate_summary.csv", candidate_rows)
    _write_csv(output_dir / "result.validation_trials.csv", validation_rows)
    _write_json(output_dir / "manifest.json", {
        "schema_version": 1, "config_hash": result["config_hash"],
        "artifacts": {
            "truth_bank": "truth_bank.npz", "reference_checkpoint": "reference.npz",
            "reference_endpoints": "reference_endpoints.npz", "reference_bank": "reference_bank.npz",
            "selection_bank": "selection_bank.npz", "validation_bank": "validation_bank.npz",
            "result": "result.json", "candidate_summary": "result.candidate_summary.csv",
            "validation_trials": "result.validation_trials.csv",
        },
    })
    return result
