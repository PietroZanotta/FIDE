#!/usr/bin/env python3
"""Execute the prospectively frozen Vortices V2 sensor-selection stage.

This is an execution harness, not a method-definition module.  It consumes the
immutable base freeze plus the additive local-proposal seed amendment, reuses
the already-created selection bank, checkpoints every authoritative candidate,
and never creates or reads a validation bank.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np


V2_DIR = Path(__file__).resolve().parent
REPO = V2_DIR.parents[1]
V1_DIR = V2_DIR
HERE = V2_DIR / "outputs" / "prospective_v2" / "selection"
SRC = REPO / "src"
for path in (SRC, REPO / "experiments", V1_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

jax.config.update("jax_enable_x64", True)

from bounded_reference import BoxTransformedReferenceFlow
from experiment import ObservationTrialBank, VortexExperiment
from mfsi.decomposition import raster_tangent_projection
from mfsi.design import (
    OptimizerConfig,
    optimize_multistart_candidates,
    point_box_violation,
    point_separation_violation,
)
from mfsi.raster import (
    reflected_gaussian_cell_mass_matrix_1d,
    reflected_particle_flux_rect,
)
from core import (
    DevelopmentContext,
    ParticleState,
    hard_fiber_particle_state,
    independent_poisson,
    make_grid,
    rasterize_trajectory_v2,
    solve_v2,
)
from selection_contract import (
    candidate_key,
    deterministic_local_cloud,
    generated_starts,
    geometry_is_feasible,
    load_selection_config,
    normalized_trapezoid_weights,
    sha256_file,
    validate_selection_config,
)


CONFIG = V2_DIR / "VORTICES_V2_SELECTION_CONFIG.json"
SEEDS = V2_DIR / "VORTICES_V2_SELECTION_SEED_SCHEDULE.json"
MANIFEST = V2_DIR / "VORTICES_V2_FREEZE_MANIFEST.json"
PROTOCOL = V2_DIR / "VORTICES_V2_SELECTION_VALIDATION_PROTOCOL_FROZEN.md"
CONTRACT = V2_DIR / "selection_contract.py"
AMENDMENT = V2_DIR / "VORTICES_V2_SELECTION_SEED_AMENDMENT.md"
AMENDMENT_RECEIPT = V2_DIR / "outputs/prospective_v2/freeze/selection_seed_amendment_receipt.json"
REFERENCE_ROOT = V2_DIR / "outputs/prospective_v2/references"
BANDWIDTH_RECEIPT = V2_DIR / "outputs/prospective_v2/freeze/common_bandwidth_receipt.json"
BANK_PATH = HERE / "shared_selection_bank.npz"
BANK_RECEIPT = HERE / "shared_selection_bank_receipt.json"
V1_CONFIG = V1_DIR / "base_experiment_config.json"
TRUTH_PATH = V1_DIR / "inputs" / "truth_bank.npz"

BASE_HASHES = {
    CONFIG: "8c06b8afee434d945d456c8e9524a4dc5f2ef95ead7c5eaf0ea9fa0de56f93d6",
    PROTOCOL: "48176166e9a22cf0f5a50fc78c77d3216a19a588ad04f0da45c37474e4015b0e",
    MANIFEST: "8debba44ce7c26cb09a3d49819d3083f0858e52578396b0b221ce72516aff8bd",
    CONTRACT: "17964b423642d4702ff32366a596533e2732c223b59acc136e82493bf9965441",
    SEEDS: "d045ea16772e4b44867bc3a27b1d5333ff9093f87b51c98c2ba7469964968e91",
    AMENDMENT: "a0a71b368e35a7335a15615b1dd85255b1c47fbe1fd3fada33b4442fd42280fe",
    BANK_PATH: "1096a255beffa781ee5a9bec881a2778b11f3bf5b8674389d7120180f5280d3b",
    BANK_RECEIPT: "92e978a0abbbb2da683ce04422d3b307c966223e19e4f636d1ec80dd21fb0e53",
    BANDWIDTH_RECEIPT: "59b871c9838dbe3f4141f499c9c3f4cf5a23ad1e82a4c93d851447e22f0da44d",
}


def json_safe(value: Any) -> Any:
    """Convert diagnostic-only nonfinite placeholders to strict-JSON nulls."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prefix(bank: ObservationTrialBank, count: int) -> ObservationTrialBank:
    return ObservationTrialBank(bank.sample_indices[:count], bank.detector_z[:count])


def cid(eta: Any) -> str:
    key = json.dumps(candidate_key(eta), separators=(",", ":")).encode()
    return hashlib.sha256(key).hexdigest()[:16]


def canonical(eta: Any) -> np.ndarray:
    return np.asarray(candidate_key(eta), dtype=np.float64)


def finite_or_none(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def add_candidate(pool: dict[tuple[float, ...], dict[str, Any]], eta: Any, label: str) -> None:
    key = candidate_key(eta)
    if key in pool:
        if label not in pool[key]["provenance"]:
            pool[key]["provenance"].append(label)
        return
    pool[key] = {"eta": list(map(float, key)), "provenance": [label], "candidate_id": cid(key)}


def progress(items: list[Any], label: str) -> Iterable[Any]:
    started = time.perf_counter()
    total = len(items)
    print(f"[{label}] 0/{total}", flush=True)
    for index, item in enumerate(items, 1):
        yield item
        if index == 1 or index == total or index % max(1, total // 10) == 0:
            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-12)
            remaining = (total - index) / max(rate, 1e-12)
            print(f"[{label}] {index}/{total} elapsed={elapsed:.1f}s eta={remaining:.1f}s", flush=True)


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any], float]:
    for path, expected in BASE_HASHES.items():
        measured = sha256_file(path)
        if measured != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}: {measured} != {expected}")
    amendment_receipt = load_json(AMENDMENT_RECEIPT)
    if amendment_receipt["status"] != "FROZEN_PROSPECTIVE_SELECTION_SEED_AMENDMENT":
        raise RuntimeError("seed amendment is not frozen")
    config = load_selection_config(CONFIG)
    validate_selection_config(config)
    schedule = load_json(SEEDS)
    root = int(config["optimization"]["optimizer_root_seed"])
    allowances = list(map(float, config["risk_and_geometry"]["risk_allowance_percentages"]))
    if allowances != list(map(float, schedule["allowance_order_percent"])):
        raise RuntimeError("seed-schedule allowance order mismatch")
    for ai, allowance in enumerate(allowances):
        key = str(allowance)
        if int(schedule["tangent_local_cloud_seed_by_allowance"][key]) != root + 1000 + ai:
            raise RuntimeError("Tangent seed derivation mismatch")
        observed = list(map(int, schedule["full_local_cloud_seeds_by_allowance_and_round"][key]))
        expected = [root + 2000 + 10 * ai + ri for ri in range(3)]
        if observed != expected:
            raise RuntimeError("Full seed derivation mismatch")
    bank_receipt = load_json(BANK_RECEIPT)
    if bank_receipt["namespace"] != 410000101 or bank_receipt["trials"] != 128:
        raise RuntimeError("selection bank identity mismatch")
    with np.load(BANK_PATH, allow_pickle=False) as raw:
        if raw["sample_indices"].shape != (128, 9, 2000):
            raise RuntimeError("selection sample-index shape mismatch")
        if raw["detector_z"].shape != (128, 9, 4):
            raise RuntimeError("selection detector shape mismatch")
        np.testing.assert_array_equal(raw["trial_ids"], np.arange(128))
    if (V2_DIR / "outputs/prospective_v2/validation/shared_validation_bank.npz").exists():
        raise RuntimeError("validation bank exists before selection freeze")
    bandwidth = float(load_json(BANDWIDTH_RECEIPT)["common_physical_bandwidth"])
    if bandwidth != 0.058816544123815116:
        raise RuntimeError("common bandwidth mismatch")
    return config, schedule, bandwidth


def load_experiments() -> tuple[list[VortexExperiment], list[DevelopmentContext], ObservationTrialBank, np.ndarray]:
    cfg = load_json(V1_CONFIG)
    with np.load(TRUTH_PATH, allow_pickle=False) as raw:
        times = np.asarray(raw["times"], dtype=np.float64)
        truth = jnp.asarray(raw["particles"], dtype=jnp.float64)
    with np.load(BANK_PATH, allow_pickle=False) as raw:
        bank = ObservationTrialBank(
            jnp.asarray(raw["sample_indices"], dtype=jnp.int32),
            jnp.asarray(raw["detector_z"], dtype=jnp.float64),
        )
    experiments: list[VortexExperiment] = []
    contexts: list[DevelopmentContext] = []
    for seed in (310000101, 310000102, 310000103):
        root = REFERENCE_ROOT / f"reference_seed_{seed}"
        receipt = load_json(root / "qualification_receipt.json")
        if receipt["status"] != "PASS":
            raise RuntimeError(f"reference {seed} is not qualified")
        reference = BoxTransformedReferenceFlow.from_npz(
            root / "reference.npz",
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        with np.load(root / "reference_bank.npz", allow_pickle=False) as raw:
            np.testing.assert_allclose(raw["times"], times, rtol=0, atol=0)
            nodes = jnp.asarray(raw["nodes"], dtype=jnp.float64)
            velocity = jnp.asarray(raw["velocity"], dtype=jnp.float64)
            weights = jnp.asarray(raw["weights"], dtype=jnp.float64)
        exp = VortexExperiment(
            cfg,
            reference,
            truth_particles=truth,
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=weights,
        )
        experiments.append(exp)
        contexts.append(DevelopmentContext(exp, bank, times, cfg, root, BANK_PATH, 410000101))
    return experiments, contexts, bank, times


class Evaluator:
    def __init__(self, config: dict[str, Any], exps: list[VortexExperiment], contexts: list[DevelopmentContext], bank: ObservationTrialBank, bandwidth: float):
        self.config = config
        self.exps = exps
        self.contexts = contexts
        self.bank = bank
        self.bandwidth = bandwidth
        self.cache = HERE / "candidate_cache"
        self.cache.mkdir(exist_ok=True)
        self.gates = config["candidate_numerical_gates"]
        self.weights = normalized_trapezoid_weights(range(21))
        self.kernel_cache = HERE / "reflected_kernel_cache"
        self.kernel_cache.mkdir(exist_ok=True)

    def _kernel_pair(self, reference_index: int, context: DevelopmentContext, grid, time_index: int):
        root = self.kernel_cache / f"{grid.nx}x{grid.ny}" / f"reference_{reference_index}"
        x_path = root / f"time_{time_index:02d}_x.npy"
        y_path = root / f"time_{time_index:02d}_y.npy"
        if not x_path.exists() or not y_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            nodes = jnp.asarray(context.exp.reference_nodes[time_index], dtype=jnp.float64)
            x_edges = jnp.linspace(float(grid.x_min), float(grid.x_max), int(grid.nx) + 1, dtype=jnp.float64)
            y_edges = jnp.linspace(float(grid.y_min), float(grid.y_max), int(grid.ny) + 1, dtype=jnp.float64)
            kernel_x = np.asarray(reflected_gaussian_cell_mass_matrix_1d(
                x_edges, nodes[:, 0], bandwidth=self.bandwidth, image_pairs=4
            ), dtype=np.float64)
            kernel_y = np.asarray(reflected_gaussian_cell_mass_matrix_1d(
                y_edges, nodes[:, 1], bandwidth=self.bandwidth, image_pairs=4
            ), dtype=np.float64)
            x_tmp = x_path.with_suffix(".tmp.npy")
            y_tmp = y_path.with_suffix(".tmp.npy")
            np.save(x_tmp, kernel_x, allow_pickle=False)
            np.save(y_tmp, kernel_y, allow_pickle=False)
            os.replace(x_tmp, x_path)
            os.replace(y_tmp, y_path)
        return np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")

    def prewarm_reflected_kernels(self, grid_shape: tuple[int, int]) -> None:
        """Materialize immutable reference/grid/time kernels before threading."""
        grid = make_grid(*grid_shape)
        for reference_index, context in enumerate(self.contexts):
            for time_index in range(21):
                self._kernel_pair(reference_index, context, grid, time_index)

    @staticmethod
    @jax.jit
    def _apply_reflected_kernels(kernel_x, kernel_y, weights, forcing, cell_area):
        def one(weight, one_forcing):
            mass = (kernel_y * weight[None, :]) @ kernel_x.T
            source_mass = (kernel_y * (weight * one_forcing)[None, :]) @ kernel_x.T
            source_before = jnp.sum(source_mass)
            total_mass = jnp.sum(mass)
            source_mass = source_mass - mass * source_before / jnp.maximum(total_mass, 1.0e-300)
            return mass, source_mass / cell_area
        mass, source = jax.vmap(one)(weights, forcing)
        return mass, mass / cell_area, source

    def _raster_state_chunk(self, states, reference_index: int, context: DevelopmentContext, grid):
        weights = np.stack([state.weights for state in states])
        forcing = np.stack([state.forcing for state in states])
        mass_rows, q_rows, source_rows = [], [], []
        for time_index in range(21):
            kernel_x, kernel_y = self._kernel_pair(reference_index, context, grid, time_index)
            mass, q, source = self._apply_reflected_kernels(
                jnp.asarray(kernel_x), jnp.asarray(kernel_y),
                jnp.asarray(weights[:, time_index]), jnp.asarray(forcing[:, time_index]),
                jnp.asarray(grid.cell_area, dtype=jnp.float64),
            )
            mass_rows.append(np.asarray(mass, dtype=np.float64))
            q_rows.append(np.asarray(q, dtype=np.float64))
            source_rows.append(np.asarray(source, dtype=np.float64))
        return {
            "mass": np.stack(mass_rows, axis=1),
            "q": np.stack(q_rows, axis=1),
            "source": np.stack(source_rows, axis=1),
        }

    @staticmethod
    def _fiber_static(context: DevelopmentContext, eta: Any) -> dict[str, Any]:
        """Compute geometry/reference terms that are invariant across trials."""
        exp = context.exp
        eta_jax = exp.family.canonicalize(jnp.asarray(eta, dtype=jnp.float64))
        nodes = np.asarray(exp.reference_nodes, dtype=np.float64)
        velocity = np.asarray(exp.reference_velocity, dtype=np.float64)
        base = np.asarray(exp.reference_weights, dtype=np.float64).copy()
        base /= np.sum(base, axis=1, keepdims=True)
        return {
            "eta_jax": eta_jax,
            "eta_np": np.asarray(eta_jax, dtype=np.float64),
            "phi_truth": exp.family.features(exp.truth_particles, eta_jax),
            "nodes": nodes,
            "velocity": velocity,
            "base": base,
            "phi": np.asarray(exp.family.features(jnp.asarray(nodes), eta_jax), dtype=np.float64),
            "grad": np.asarray(exp.family.feature_gradients(jnp.asarray(nodes), eta_jax), dtype=np.float64),
        }

    @staticmethod
    def _hard_fiber_state_from_static(
        context: DevelopmentContext,
        eta: Any,
        trial: int,
        static: dict[str, Any],
    ) -> ParticleState:
        """Apply the unchanged hard fiber using trial-invariant cached arrays."""
        exp = context.exp
        rec = exp._measurement_reconstruction(static["phi_truth"], context.bank, int(trial))
        nodes = static["nodes"]
        velocity = static["velocity"]
        base = static["base"]
        phi = static["phi"]
        grad = static["grad"]

        projection = exp.exact_projector.project_trajectory(
            jnp.asarray(phi), jnp.asarray(base), jnp.asarray(rec.c)[None]
        )
        lam = np.asarray(projection.lam[0], dtype=np.float64)
        weights = np.asarray(projection.weights[0], dtype=np.float64)
        moments = np.asarray(projection.moments[0], dtype=np.float64)
        residual = np.asarray(projection.residual[0], dtype=np.float64)
        covariance = np.asarray(projection.covariance[0], dtype=np.float64)
        ess = np.asarray(projection.ess_fraction[0], dtype=np.float64)

        advective = np.einsum("tnmd,tnd->tnm", grad, velocity)
        mean_advective = np.einsum("tn,tnm->tm", weights, advective)
        g = np.einsum("tnm,tm->tn", advective, lam)
        mean_g = np.einsum("tn,tn->t", weights, g)
        centered = phi - moments[:, None, :]
        cov_phi_g = np.einsum(
            "tn,tnm,tn->tm", weights, centered, g - mean_g[:, None]
        )
        target_dot = np.asarray(rec.c_dot, dtype=np.float64)
        rhs = target_dot - mean_advective - cov_phi_g
        eigenvalues = np.linalg.eigvalsh(
            0.5 * (covariance + np.swapaxes(covariance, -1, -2))
        )
        lambda_dot = np.empty_like(rhs)
        for time_index in range(len(context.times)):
            if float(eigenvalues[time_index, 0]) <= 1.0e-6:
                lambda_dot[time_index] = np.linalg.lstsq(
                    covariance[time_index], rhs[time_index], rcond=None
                )[0]
            else:
                lambda_dot[time_index] = np.linalg.solve(
                    covariance[time_index], rhs[time_index]
                )
        forcing = np.einsum("tnm,tm->tn", centered, lambda_dot)
        forcing += g - mean_g[:, None]
        forcing -= np.einsum("tn,tn->t", weights, forcing)[:, None]
        moment_defect = target_dot - mean_advective

        return ParticleState(
            eta=np.asarray(static["eta_np"], dtype=np.float64),
            trial=int(trial),
            particle_count=int(nodes.shape[1]),
            nodes=nodes,
            velocity=velocity,
            base_weights=base,
            weights=weights,
            forcing=forcing,
            lam=lam,
            lambda_dot=lambda_dot,
            moments=moments,
            target=np.asarray(rec.c, dtype=np.float64),
            target_dot=target_dot,
            moment_defect=moment_defect,
            calibration_residual=np.linalg.norm(residual, axis=-1),
            ess_fraction=ess,
            covariance_min_eigenvalue=eigenvalues[:, 0],
        )

    def _cached(self, group: str, eta: Any, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        path = self.cache / group / f"{cid(eta)}.json"
        if path.exists():
            row = load_json(path)
            if tuple(row["candidate_key"]) != candidate_key(eta):
                raise RuntimeError(f"candidate cache collision at {path}")
            return row
        row = fn()
        row["candidate_key"] = list(candidate_key(eta))
        row["candidate_id"] = cid(eta)
        row["input_identity"] = {
            "selection_bank_sha256": BASE_HASHES[BANK_PATH],
            "selection_config_sha256": BASE_HASHES[CONFIG],
            "seed_schedule_sha256": BASE_HASHES[SEEDS],
            "bandwidth": self.bandwidth,
        }
        atomic_json(path, row)
        return row

    def population(self, eta: Any) -> dict[str, Any]:
        def compute() -> dict[str, Any]:
            refs = [exp.exact_population_result(jnp.asarray(eta)) for exp in self.exps]
            geometric = geometry_is_feasible(
                eta, box=self.config["risk_and_geometry"]["center_box"],
                minimum_separation=self.config["risk_and_geometry"]["minimum_pairwise_separation"],
                tolerance=self.config["optimization"]["common_adam"]["feasibility_tolerance"],
            )
            valid = geometric and all(bool(row["valid"]) and np.isfinite(row["value"]) for row in refs)
            compact = [{**row, "value": finite_or_none(row["value"])} for row in refs]
            return {
                "kind": "population_exact_three_reference",
                "valid": valid,
                "value": float(np.mean([row["value"] for row in refs])) if valid else None,
                "geometry_feasible": geometric,
                "per_reference": compact,
            }
        return self._cached("population", eta, compute)

    def risk(self, eta: Any) -> dict[str, Any]:
        bank64 = prefix(self.bank, 64)
        def compute() -> dict[str, Any]:
            refs = [exp.exact_finite_result(jnp.asarray(eta), bank64) for exp in self.exps]
            valid = all(bool(row["valid"]) and np.isfinite(row["value"]) for row in refs)
            compact = []
            for row in refs:
                compact.append({
                    "valid": bool(row["valid"]),
                    "value": finite_or_none(row["value"]),
                    "maximum_calibration_residual": max((x["max_calibration_residual"] for x in row["rows"]), default=float("inf")),
                    "minimum_ess_fraction": min((x["min_ess_fraction"] for x in row["rows"]), default=0.0),
                    "minimum_support_gap": min((x.get("empirical_hull_support_gap", float("inf")) for x in row["rows"]), default=float("inf")),
                })
            return {
                "kind": "finite_law_exact_three_reference_64",
                "valid": valid,
                "value": float(np.mean([row["value"] for row in refs])) if valid else None,
                "per_reference": compact,
            }
        return self._cached("risk_64", eta, compute)

    def tangent(self, eta: Any, trials: int) -> dict[str, Any]:
        bank = prefix(self.bank, trials)
        def compute() -> dict[str, Any]:
            refs = [exp.exact_tangent_result(jnp.asarray(eta), bank) for exp in self.exps]
            valid = all(bool(row["valid"]) and np.isfinite(row["value"]) for row in refs)
            compact = []
            for row in refs:
                compact.append({
                    "valid": bool(row["valid"]),
                    "value": finite_or_none(row["value"]),
                    "maximum_calibration_residual": max((x["max_calibration_residual"] for x in row["rows"]), default=float("inf")),
                    "minimum_ess_fraction": min((x["min_ess_fraction"] for x in row["rows"]), default=0.0),
                    "maximum_tangent_compatibility_residual": max((x["max_tangent_compatibility_residual"] for x in row["rows"]), default=float("inf")),
                })
            return {
                "kind": f"tangent_exact_three_reference_{trials}",
                "valid": valid,
                "value": float(np.mean([row["value"] for row in refs])) if valid else None,
                "per_reference": compact,
            }
        return self._cached(f"tangent_{trials}", eta, compute)

    def full(self, eta: Any, trials: int, grid_shape: tuple[int, int], *, decomposition: bool) -> dict[str, Any]:
        group = f"full_{grid_shape[0]}x{grid_shape[1]}_{trials}_{'decomp' if decomposition else 'basic'}"
        def compute() -> dict[str, Any]:
            grid = make_grid(*grid_shape)
            per_ref = []
            all_actions = []
            overall_valid = True
            for ref_index, context in enumerate(self.contexts):
                fiber_static = self._fiber_static(context, eta)
                actions = []
                maxima = {
                    "maximum_calibration_residual": 0.0,
                    "minimum_ess_fraction": float("inf"),
                    "maximum_mass_error": 0.0,
                    "maximum_source_compatibility_absolute": 0.0,
                    "maximum_poisson_relative_residual": 0.0,
                    "maximum_component_compatibility_residual": 0.0,
                    "maximum_component_count": 0,
                    "maximum_full_moment_rate_residual": 0.0,
                    "maximum_tangent_moment_rate_residual": 0.0,
                    "maximum_hidden_nullspace_residual": 0.0,
                    "maximum_orthogonality_absolute": 0.0,
                    "maximum_pythagorean_absolute": 0.0,
                    "maximum_raw_hierarchy_violation": 0.0,
                }
                trial_rows = []
                features = (
                    context.exp.family.features(grid.points(), jnp.asarray(eta, dtype=jnp.float64))
                    if decomposition else None
                )
                chunk_size = 4
                for begin in range(0, trials, chunk_size):
                    trial_ids = list(range(begin, min(begin + chunk_size, trials)))
                    states = [
                        self._hard_fiber_state_from_static(context, eta, trial, fiber_static)
                        for trial in trial_ids
                    ]
                    raster = self._raster_state_chunk(states, ref_index, context, grid)
                    batch_count = len(states)
                    solved = solve_v2(
                        raster["q"].reshape((batch_count * 21, grid.ny, grid.nx)),
                        raster["source"].reshape((batch_count * 21, grid.ny, grid.nx)),
                        grid,
                    )
                    action_matrix = np.asarray(solved.action, dtype=np.float64).reshape((batch_count, 21))
                    potential = np.asarray(solved.potential, dtype=np.float64).reshape((batch_count, 21, grid.ny, grid.nx))
                    poisson_residual = np.asarray(solved.relative_residual, dtype=np.float64).reshape((batch_count, 21))
                    compatibility = np.asarray(solved.maximum_component_compatibility_residual, dtype=np.float64).reshape((batch_count, 21))
                    component_count = np.asarray(solved.component_count).reshape((batch_count, 21))
                    converged = np.asarray(solved.solver_converged).reshape((batch_count, 21))
                    compatible = np.asarray(solved.compatible).reshape((batch_count, 21))
                    for local_index, (trial, state) in enumerate(zip(trial_ids, states)):
                        action_by_time = action_matrix[local_index]
                        action = float(np.sum(self.weights * action_by_time))
                        diag = {
                            "trial": trial,
                            "action": action,
                            "maximum_calibration_residual": float(np.max(state.calibration_residual)),
                            "minimum_ess_fraction": float(np.min(state.ess_fraction)),
                            "maximum_mass_error": float(np.max(np.abs(np.sum(raster["mass"][local_index], axis=(-2, -1)) - 1.0))),
                            "maximum_source_compatibility_absolute": float(np.max(np.abs(np.sum(raster["source"][local_index], axis=(-2, -1)) * grid.cell_area))),
                            "maximum_poisson_relative_residual": float(np.max(poisson_residual[local_index])),
                            "maximum_component_compatibility_residual": float(np.max(compatibility[local_index])),
                            "maximum_component_count": int(np.max(component_count[local_index])),
                            "solver_converged": bool(np.all(converged[local_index])),
                            "component_compatible": bool(np.all(compatible[local_index])),
                            "strictly_positive_q": bool(np.all(raster["q"][local_index] > 0.0)),
                        }
                        if decomposition:
                            decomp = raster_tangent_projection(
                                jnp.asarray(potential[local_index], dtype=jnp.float64),
                                jnp.asarray(raster["q"][local_index], dtype=jnp.float64),
                                -jnp.asarray(raster["source"][local_index], dtype=jnp.float64),
                                features,
                                dx=float(grid.dx), cell_area=float(grid.cell_area),
                                pinv_rcond=1e-10, operator_floor_rel=0.0,
                                gauge_strength=0.0, source_is_density=True,
                            )
                            diag.update({
                                "maximum_full_moment_rate_residual": float(np.max(np.linalg.norm(np.asarray(decomp.full_moment_residual), axis=-1))),
                                "maximum_tangent_moment_rate_residual": float(np.max(np.linalg.norm(np.asarray(decomp.tangent_moment_residual), axis=-1))),
                                "maximum_hidden_nullspace_residual": float(np.max(np.linalg.norm(np.asarray(decomp.hidden_moment_residual), axis=-1))),
                                "maximum_orthogonality_absolute": float(np.max(np.abs(np.asarray(decomp.tangent_hidden_inner_product)))),
                                "maximum_pythagorean_absolute": float(np.max(np.abs(np.asarray(decomp.pythagorean_residual)))),
                                "maximum_raw_hierarchy_violation": float(np.max(np.asarray(decomp.hierarchy_raw_violation))),
                            })
                        actions.append(action)
                        trial_rows.append(diag)
                        for key in maxima:
                            if key == "minimum_ess_fraction":
                                maxima[key] = min(maxima[key], diag[key])
                            elif key in diag:
                                maxima[key] = max(maxima[key], diag[key])
                        valid = (
                            np.isfinite(action)
                            and diag["maximum_calibration_residual"] <= float(self.gates["maximum_finite_calibration_residual"])
                            and diag["minimum_ess_fraction"] >= float(self.gates["minimum_ess_fraction"])
                            and diag["maximum_mass_error"] <= float(self.gates["maximum_mass_absolute_error"])
                            and diag["maximum_source_compatibility_absolute"] <= float(self.gates["maximum_source_compatibility_absolute"])
                            and diag["maximum_poisson_relative_residual"] <= float(self.gates["maximum_poisson_relative_residual"])
                            and diag["maximum_component_count"] == int(self.gates["required_conductive_component_count"])
                            and diag["solver_converged"] and diag["component_compatible"] and diag["strictly_positive_q"]
                        )
                        if decomposition:
                            valid = valid and (
                                diag["maximum_full_moment_rate_residual"] <= float(self.gates["maximum_full_moment_rate_residual"])
                                and diag["maximum_tangent_moment_rate_residual"] <= float(self.gates["maximum_tangent_moment_rate_residual"])
                                and diag["maximum_hidden_nullspace_residual"] <= float(self.gates["maximum_hidden_nullspace_residual"])
                                and diag["maximum_orthogonality_absolute"] <= float(self.gates["maximum_orthogonality_absolute"])
                                and diag["maximum_pythagorean_absolute"] <= float(self.gates["maximum_pythagorean_absolute"])
                                and diag["maximum_raw_hierarchy_violation"] <= float(self.gates["maximum_raw_hierarchy_violation"])
                            )
                        overall_valid = overall_valid and bool(valid)
                reference_valid = bool(all(
                    np.isfinite(row["action"])
                    and row["maximum_calibration_residual"] <= float(self.gates["maximum_finite_calibration_residual"])
                    and row["minimum_ess_fraction"] >= float(self.gates["minimum_ess_fraction"])
                    and row["maximum_mass_error"] <= float(self.gates["maximum_mass_absolute_error"])
                    and row["maximum_source_compatibility_absolute"] <= float(self.gates["maximum_source_compatibility_absolute"])
                    and row["maximum_poisson_relative_residual"] <= float(self.gates["maximum_poisson_relative_residual"])
                    and row["maximum_component_count"] == int(self.gates["required_conductive_component_count"])
                    and row["solver_converged"] and row["component_compatible"] and row["strictly_positive_q"]
                    for row in trial_rows
                ))
                ref_value = float(np.mean(actions))
                all_actions.append(ref_value)
                per_ref.append({"reference_index": ref_index, "valid": reference_valid, "value": ref_value, "diagnostics": maxima, "trials": trial_rows})
            return {
                "kind": "reflected_v2_full_action",
                "grid": list(grid_shape), "trials": trials, "decomposition": decomposition,
                "valid": bool(overall_valid),
                "value": float(np.mean(all_actions)) if overall_valid else None,
                "per_reference": per_ref,
            }
        return self._cached(group, eta, compute)


def optimizer_cfg(config: dict[str, Any], stage: str) -> OptimizerConfig:
    common = config["optimization"]["common_adam"]
    spec = config["optimization"][stage]
    return OptimizerConfig(
        steps=int(spec["steps"]), learning_rate=float(spec["learning_rate"]),
        beta1=float(common["beta1"]), beta2=float(common["beta2"]),
        eps=float(common["epsilon"]), constraint_penalty=float(common["constraint_penalty"]),
        feasibility_tol=float(common["feasibility_tolerance"]),
    )


def geometry_tools(config: dict[str, Any]):
    risk = config["risk_and_geometry"]
    box = risk["center_box"]
    constraints = (
        (point_box_violation(n_sensors=4, x_bounds=tuple(box[0]), y_bounds=tuple(box[1])), 0.0),
        (point_separation_violation(float(risk["minimum_pairwise_separation"]), n_sensors=4), 0.0),
    )
    lo = jnp.asarray([box[0][0], box[1][0]], dtype=jnp.float64)
    hi = jnp.asarray([box[0][1], box[1][1]], dtype=jnp.float64)
    return constraints, lambda eta: jnp.clip(jnp.asarray(eta).reshape(4, 2), lo, hi).reshape(-1)


def fast_functions(exps: list[VortexExperiment], bank: ObservationTrialBank):
    bank4 = prefix(bank, 4)
    def pop(eta):
        return sum(exp.population_loss(eta) for exp in exps) / 3.0
    def risk4(eta):
        return sum(exp.finite_risk(eta, bank4) for exp in exps) / 3.0
    def tangent4(eta):
        return sum(exp.tangent_action_gradient(eta, bank4) for exp in exps) / 3.0
    return pop, risk4, tangent4


def fast_rank(pool: dict[tuple[float, ...], dict[str, Any]], objective: Callable, constraints: tuple, label: str) -> list[dict[str, Any]]:
    fn = jax.jit(objective)
    rows = []
    for candidate in progress(list(pool.values()), label):
        eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
        violation = max([0.0] + [float(cfn(eta) - upper) for cfn, upper in constraints])
        try:
            value = float(fn(eta))
        except Exception as exc:
            value = 1.0e12
            candidate = dict(candidate, fast_error=repr(exc))
        rows.append(dict(candidate, fast_value=value, fast_violation=violation))
    rows.sort(key=lambda row: (not np.isfinite(row["fast_value"]), row["fast_violation"], row["fast_value"], row["candidate_id"]))
    return rows


def stage_population(config, exps, evaluator, starts, old) -> dict[str, Any]:
    output = HERE / "population/result.json"
    if output.exists():
        return load_json(output)
    pop_fast, _, _ = fast_functions(exps, evaluator.bank)
    geometry, projector = geometry_tools(config)
    spec = config["optimization"]["population"]
    optimized = optimize_multistart_candidates(
        pop_fast, jnp.asarray(starts[: int(spec["optimized_starts"])]), optimizer_cfg(config, "population"),
        constraints=geometry, canonicalize=exps[0].family.canonicalize,
        project_iterate=projector, vectorize_starts=False,
    )
    pool: dict[tuple[float, ...], dict[str, Any]] = {}
    for index, eta in enumerate(starts): add_candidate(pool, eta, f"generated_{index:02d}")
    for index, row in enumerate(optimized): add_candidate(pool, row.eta, f"population_adam_{index:02d}")
    for row in old: add_candidate(pool, row["eta"], row["label"])
    ranked = fast_rank(pool, pop_fast, geometry, "population fast rank")
    audits = []
    for row in progress(ranked[: int(spec["exact_audit_candidates"])], "population exact audit"):
        exact = evaluator.population(row["eta"])
        audits.append(dict(row, exact=exact))
    valid = [row for row in audits if row["exact"]["valid"]]
    if len(valid) < int(spec["minimum_exact_valid_candidates"]):
        raise RuntimeError("Population exact-valid candidate minimum failed")
    winner = min(valid, key=lambda row: row["exact"]["value"])
    result = {
        "status": "PASS", "stage": "population", "winner": winner,
        "L_star": winner["exact"]["value"],
        "L_max": winner["exact"]["value"] + float(config["risk_and_geometry"]["population_slack"]),
        "audits": audits, "candidate_count": len(pool),
    }
    atomic_json(output, result)
    return result


def stage_law(config, exps, evaluator, starts, old, population, anchor_seed=None, pass_index=0) -> dict[str, Any]:
    output = HERE / f"law/pass_{pass_index}/result.json"
    if output.exists():
        return load_json(output)
    pop_fast, risk4, _ = fast_functions(exps, evaluator.bank)
    geometry, projector = geometry_tools(config)
    L_max = float(population["L_max"])
    fast_anchor = float(jax.jit(pop_fast)(jnp.asarray(population["winner"]["eta"])))
    lscale = max(float(config["risk_and_geometry"]["population_slack"]), 1e-10)
    constraints = geometry + ((lambda eta: (pop_fast(eta) - (fast_anchor + lscale)) / lscale, 0.0),)
    spec = config["optimization"]["law"]
    opt_starts = [population["winner"]["eta"]]
    if anchor_seed is not None: opt_starts.append(anchor_seed)
    opt_starts += list(starts)
    optimized = optimize_multistart_candidates(
        risk4, jnp.asarray(opt_starts[: int(spec["optimized_starts"])]), optimizer_cfg(config, "law"),
        constraints=constraints, canonicalize=exps[0].family.canonicalize,
        project_iterate=projector, vectorize_starts=False,
    )
    pool: dict[tuple[float, ...], dict[str, Any]] = {}
    add_candidate(pool, population["winner"]["eta"], "new_population")
    if anchor_seed is not None: add_candidate(pool, anchor_seed, f"anchor_refinement_seed_{pass_index}")
    for index, eta in enumerate(starts): add_candidate(pool, eta, f"generated_{index:02d}")
    for index, row in enumerate(optimized): add_candidate(pool, row.eta, f"law_adam_{index:02d}")
    for row in old: add_candidate(pool, row["eta"], row["label"])
    ranked = fast_rank(pool, risk4, constraints, f"law pass {pass_index} fast rank")
    mandatory = {candidate_key(population["winner"]["eta"])}
    if anchor_seed is not None: mandatory.add(candidate_key(anchor_seed))
    ordered = [r for r in ranked if candidate_key(r["eta"]) in mandatory] + [r for r in ranked if candidate_key(r["eta"]) not in mandatory]
    audits = []
    for row in progress(ordered[: int(spec["exact_audit_candidates"])], f"law pass {pass_index} exact audit"):
        pop = evaluator.population(row["eta"])
        risk = evaluator.risk(row["eta"]) if pop["valid"] and pop["value"] <= L_max + 1e-12 else {"valid": False, "value": None}
        audits.append(dict(row, exact_population=pop, exact_risk=risk, valid=bool(pop["valid"] and pop["value"] <= L_max + 1e-12 and risk["valid"])))
    valid = [row for row in audits if row["valid"]]
    if len(valid) < int(spec["minimum_exact_valid_candidates"]):
        raise RuntimeError("Law exact-valid candidate minimum failed")
    winner = min(valid, key=lambda row: row["exact_risk"]["value"])
    result = {
        "status": "PASS", "stage": "law", "anchor_refinement_pass": pass_index,
        "winner": winner, "R_star": winner["exact_risk"]["value"],
        "audits": audits, "candidate_count": len(pool),
    }
    atomic_json(output, result)
    atomic_json(HERE / "law/current_result.json", result)
    return result


def exact_risk_screen(rows: list[dict[str, Any]], evaluator: Evaluator, L_max: float, R_max: float, count: int, mandatory_eta=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mandatory_key = candidate_key(mandatory_eta) if mandatory_eta is not None else None
    ordered = ([r for r in rows if candidate_key(r["eta"]) == mandatory_key] if mandatory_key else []) + [r for r in rows if candidate_key(r["eta"]) != mandatory_key]
    audits = []
    for row in ordered[:count]:
        pop = evaluator.population(row["eta"])
        risk = evaluator.risk(row["eta"]) if pop["valid"] and pop["value"] <= L_max + 1e-12 else {"valid": False, "value": None}
        valid = bool(pop["valid"] and pop["value"] <= L_max + 1e-12 and risk["valid"] and risk["value"] <= R_max + 1e-12)
        audits.append(dict(row, exact_population=pop, exact_risk=risk, valid=valid))
    return audits, [row for row in audits if row["valid"]]


def tangent_allowance(config, schedule, exps, evaluator, starts, old, population, law, allowance, index, incumbent):
    out = HERE / "allowances" / f"risk_{str(allowance).replace('.', 'p')}pct" / "tangent.json"
    if out.exists(): return load_json(out)
    pop_fast, risk4, tan4 = fast_functions(exps, evaluator.bank)
    geometry, projector = geometry_tools(config)
    L_max = float(population["L_max"]); R_max = float(law["risk_caps"][str(allowance)])
    pop_anchor = float(jax.jit(pop_fast)(jnp.asarray(population["winner"]["eta"])))
    risk_anchor = float(jax.jit(risk4)(jnp.asarray(law["winner"]["eta"])))
    lscale = max(float(config["risk_and_geometry"]["population_slack"]), 1e-10)
    rscale = max(R_max - float(law["R_star"]), 1e-10)
    constraints = geometry + ((lambda eta: jnp.maximum((pop_fast(eta)-(pop_anchor+lscale))/lscale, (risk4(eta)-(risk_anchor+rscale))/rscale), 0.0),)
    centers = [population["winner"]["eta"], law["winner"]["eta"]]
    if incumbent is not None: centers.append(incumbent["winner"]["eta"])
    local = deterministic_local_cloud(centers, count_per_center=12, scale=.08, seed=int(schedule["tangent_local_cloud_seed_by_allowance"][str(allowance)]), box=config["risk_and_geometry"]["center_box"])
    spec = config["optimization"]["tangent"]
    optimizer_starts = centers + local + list(starts)
    optimized = optimize_multistart_candidates(
        tan4, jnp.asarray(optimizer_starts[:int(spec["optimized_starts"])]), optimizer_cfg(config, "tangent"),
        constraints=constraints, canonicalize=exps[0].family.canonicalize, project_iterate=projector, vectorize_starts=False,
    )
    pool = {}
    for label, eta in zip(("new_population", "new_law", "previous_tighter_tangent"), centers): add_candidate(pool, eta, label)
    for i, eta in enumerate(local): add_candidate(pool, eta, f"tangent_local_{i:03d}")
    for i, eta in enumerate(starts): add_candidate(pool, eta, f"generated_{i:02d}")
    for i, row in enumerate(optimized): add_candidate(pool, row.eta, f"tangent_adam_{i:02d}")
    for row in old: add_candidate(pool, row["eta"], row["label"])
    ranked = fast_rank(pool, tan4, constraints, f"tangent {allowance}% fast rank")
    audits, feasible = exact_risk_screen(ranked, evaluator, L_max, R_max, int(spec["exact_risk_audit_candidates"]), incumbent["winner"]["eta"] if incumbent else None)
    if not feasible: raise RuntimeError(f"no risk-feasible Tangent candidate at {allowance}%")
    prescreen = []
    for row in progress(feasible, f"tangent {allowance}% 32-trial prescreen"):
        rec = evaluator.tangent(row["eta"], 32)
        if rec["valid"]: prescreen.append(dict(row, prescreen=rec))
    prescreen.sort(key=lambda row: row["prescreen"]["value"])
    promoted = prescreen[:int(spec["promoted_candidates"])]
    if incumbent is not None and all(candidate_key(r["eta"]) != candidate_key(incumbent["winner"]["eta"]) for r in promoted):
        promoted.append(next(r for r in prescreen if candidate_key(r["eta"]) == candidate_key(incumbent["winner"]["eta"])))
    finals = []
    for row in progress(promoted, f"tangent {allowance}% 128-trial final"):
        rec = evaluator.tangent(row["eta"], 128)
        finals.append(dict(row, final=rec))
    valid = [r for r in finals if r["final"]["valid"]]
    if incumbent is None: winner = min(valid, key=lambda r:r["final"]["value"])
    else:
        incumbent_row = next(r for r in valid if candidate_key(r["eta"]) == candidate_key(incumbent["winner"]["eta"]))
        challenger = min(valid, key=lambda r:r["final"]["value"])
        winner = challenger if challenger["final"]["value"] < incumbent_row["final"]["value"] - 1e-6 else incumbent_row
    result = {"status":"PASS","stage":"tangent","allowance_percent":allowance,"winner":winner,"risk_audits":audits,"prescreen":prescreen,"finalists":finals,"seed":int(schedule["tangent_local_cloud_seed_by_allowance"][str(allowance)])}
    atomic_json(out,result); return result


def full_allowance(config, schedule, evaluator, starts, old, population, law, tangent, allowance, index, incumbent):
    out = HERE / "allowances" / f"risk_{str(allowance).replace('.', 'p')}pct" / "full.json"
    if out.exists(): return load_json(out)
    L_max=float(population["L_max"]); R_max=float(law["risk_caps"][str(allowance)])
    centers=[population["winner"]["eta"],law["winner"]["eta"],tangent["winner"]["eta"]]
    if incumbent is not None: centers.append(incumbent["winner"]["eta"])
    pool={}
    for i,eta in enumerate(starts): add_candidate(pool,eta,f"generated_{i:02d}")
    labels=["new_population","new_law","current_tangent","previous_tighter_full"]
    for label,eta in zip(labels,centers): add_candidate(pool,eta,label)
    for row in old: add_candidate(pool,row["eta"],row["label"])
    round_rows=[]
    seeds=list(map(int,schedule["full_local_cloud_seeds_by_allowance_and_round"][str(allowance)]))
    for ri,(scale,seed) in enumerate(zip((.06,.03,.015),seeds)):
        local=deterministic_local_cloud(centers,count_per_center=10,scale=scale,seed=seed,box=config["risk_and_geometry"]["center_box"])
        for i,eta in enumerate(local): add_candidate(pool,eta,f"full_round_{ri+1}_local_{i:03d}")
        round_rows.append({"round":ri+1,"scale":scale,"seed":seed,"generated":len(local),"pool_after":len(pool)})
    proxy=[]
    proxy_candidates = list(pool.values())
    def evaluate_proxy(row):
        return dict(row, proxy=evaluator.full(row["eta"], 32, (64, 32), decomposition=False))
    with ThreadPoolExecutor(max_workers=4) as workers:
        started = time.perf_counter()
        print(f"[full {allowance}% reflected 32-trial proxy] 0/{len(proxy_candidates)}", flush=True)
        for completed, evaluated in enumerate(workers.map(evaluate_proxy, proxy_candidates), 1):
            proxy.append(evaluated)
            if completed == 1 or completed == len(proxy_candidates) or completed % max(1, len(proxy_candidates)//10) == 0:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-12)
                remaining = (len(proxy_candidates) - completed) / max(rate, 1e-12)
                print(f"[full {allowance}% reflected 32-trial proxy] {completed}/{len(proxy_candidates)} elapsed={elapsed:.1f}s eta={remaining:.1f}s", flush=True)
    proxy.sort(key=lambda r:(not r["proxy"]["valid"], r["proxy"]["value"] if r["proxy"]["valid"] else float("inf"),r["candidate_id"]))
    audits,feasible=exact_risk_screen(proxy,evaluator,L_max,R_max,30,incumbent["winner"]["eta"] if incumbent else None)
    if not feasible: raise RuntimeError(f"no risk-feasible Full candidate at {allowance}%")
    feasible.sort(key=lambda r:r["proxy"]["value"])
    promoted=feasible[:8]
    if incumbent is not None and all(candidate_key(r["eta"])!=candidate_key(incumbent["winner"]["eta"]) for r in promoted):
        promoted.append(next(r for r in feasible if candidate_key(r["eta"])==candidate_key(incumbent["winner"]["eta"])))
    # Exact finalist calls are scientifically independent.  Prewarming avoids
    # concurrent creation of the shared immutable kernel files; the
    # order-preserving map retains the prospectively ranked receipt order.
    evaluator.prewarm_reflected_kernels((256,128))
    def evaluate_final(row):
        return dict(row,final=evaluator.full(row["eta"],128,(256,128),decomposition=True))
    finals=[]
    with ThreadPoolExecutor(max_workers=min(4,len(promoted))) as workers:
        started=time.perf_counter()
        print(f"[full {allowance}% reflected 128-trial exact] 0/{len(promoted)}",flush=True)
        for completed,evaluated in enumerate(workers.map(evaluate_final,promoted),1):
            finals.append(evaluated)
            elapsed=time.perf_counter()-started
            rate=completed/max(elapsed,1e-12)
            remaining=(len(promoted)-completed)/max(rate,1e-12)
            print(f"[full {allowance}% reflected 128-trial exact] {completed}/{len(promoted)} elapsed={elapsed:.1f}s eta={remaining:.1f}s",flush=True)
    valid=[r for r in finals if r["final"]["valid"]]
    if not valid: raise RuntimeError(f"no exact-valid Full finalist at {allowance}%")
    if incumbent is None: winner=min(valid,key=lambda r:r["final"]["value"])
    else:
        incumbent_row=next(r for r in valid if candidate_key(r["eta"])==candidate_key(incumbent["winner"]["eta"]))
        challenger=min(valid,key=lambda r:r["final"]["value"])
        winner=challenger if challenger["final"]["value"]<incumbent_row["final"]["value"]-1e-6 else incumbent_row
    result={"status":"PASS","stage":"full","allowance_percent":allowance,"winner":winner,"rounds":round_rows,"risk_audits":audits,"finalists":finals}
    atomic_json(out,result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--through",choices=("population","law","tangent","full"),default="full"); args=parser.parse_args()
    config,schedule,bandwidth=verify_inputs()
    exps,contexts,bank,times=load_experiments()
    evaluator=Evaluator(config,exps,contexts,bank,bandwidth)
    starts=generated_starts(config)
    old=config["optimization"]["old_v1_proposals"]
    population=stage_population(config,exps,evaluator,starts,old)
    if args.through=="population": return
    law=stage_law(config,exps,evaluator,starts,old,population)
    allowances=list(map(float,config["risk_and_geometry"]["risk_allowance_percentages"]))
    law["risk_caps"]={str(p):float(law["R_star"]+(p/100.0)*abs(law["R_star"])) for p in allowances}
    atomic_json(HERE/"law/current_result.json",law)
    if args.through=="law": return
    tangent_results=[]; full_results=[]; tan_inc=None; full_inc=None
    for index,allowance in enumerate(allowances):
        tangent=tangent_allowance(config,schedule,exps,evaluator,starts,old,population,law,allowance,index,tan_inc)
        tangent_results.append(tangent); tan_inc=tangent
        if args.through=="tangent": continue
        full=full_allowance(config,schedule,evaluator,starts,old,population,law,tangent,allowance,index,full_inc)
        full_results.append(full); full_inc=full
        discovered=[]
        for stage in (tangent,full):
            for row in stage["risk_audits"]:
                if row["exact_risk"].get("valid"):
                    discovered.append((row["exact_risk"]["value"],row["eta"]))
        if discovered:
            best_r,best_eta=min(discovered,key=lambda item:item[0])
            if best_r<float(law["R_star"])-float(config["optimization"]["law"]["anchor_consistency_tolerance"]):
                raise RuntimeError(f"LAW_ANCHOR_REFINEMENT_REQUIRED:{best_r}:{json.dumps(best_eta)}")
    if args.through=="tangent": return
    winners={
        "schema_version":1,"status":"FROZEN_SELECTION_WINNERS","common_bandwidth":bandwidth,
        "selection_bank_sha256":BASE_HASHES[BANK_PATH],"population":population,"law":law,
        "tangent":tangent_results,"full":full_results,"validation_namespace_used":False,
    }
    atomic_json(HERE/"frozen_winners.json",winners)
    print(json.dumps({"status":"PASS","frozen_winners":str(HERE/"frozen_winners.json")},indent=2),flush=True)


if __name__ == "__main__":
    main()
