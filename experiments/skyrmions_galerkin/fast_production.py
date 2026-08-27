"""Accelerated, semantics-preserving production Galerkin design loop.

The analytic hybrid basis and its state derivatives do not depend on sensor
coordinates.  This module evaluates them once, records a fingerprinted float64
cache, and keeps stable-shape JAX contractions alive for repeated eta calls.
The authoritative rank-aware solve and all production acceptance audits remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import GalerkinSystem, aggregate_quadratic_values, rank_aware_quadratic_solve
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import HybridInvariantDictionary, load_dictionary
from .production_galerkin import _normalized_chunk, make_basis_evaluators
from .production_gradient import _forcing_state_payload
from .production_workflow import load_production_data
from .workflow import OUTPUT_ROOT, PreparedExperiment, selection_risk, write_json

Array = jax.Array

FAST_ROOT = OUTPUT_ROOT / "fast_production_3pct"
CACHE_VERSION = "fast_hybrid_basis_v1_float64_time_sharded"


def require_fast_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = FAST_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"fast-production output must be beneath {root}, got {resolved}")
    return resolved


def _sync(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        value = vars(value)
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function()
    _sync(value)
    return value, time.perf_counter() - start


def _median_timing(function: Callable[[], Any], repeats: int = 3) -> float:
    return statistics.median(_timed(function)[1] for _ in range(int(repeats)))


def _cache_signature(
    cfg: dict[str, Any], artifact_dir: Path, dictionary_path: Path,
    dictionary: HybridInvariantDictionary, data: PreparedExperiment,
) -> tuple[str, dict[str, Any]]:
    manifest = artifact_dir / "isolated_artifact_manifest.json"
    metadata = {
        "cache_version": CACHE_VERSION,
        "artifact_manifest_sha256": file_sha256(manifest),
        "dictionary_sha256": file_sha256(dictionary_path),
        "basis_size": dictionary.size,
        "box": list(dictionary.box),
        "train_configurations_shape": list(data.ritz_train_bank.configurations.shape),
        "train_configurations_dtype": str(data.ritz_train_bank.configurations.dtype),
        "normalization_shape": list(dictionary.energy_scales.shape),
        "scientific_config_hash": fingerprint({
            key: cfg[key] for key in (
                "physics", "measurement", "moment_reconstruction", "projection",
                "forcing", "production_galerkin",
            )
        }),
    }
    return fingerprint(metadata), metadata


def basis_cache_memory_estimates(data: PreparedExperiment, basis_size: int) -> dict[str, Any]:
    particles = int(data.ritz_train_bank.configurations.shape[-2])
    coordinates = particles * int(data.ritz_train_bank.configurations.shape[-1])
    result: dict[str, Any] = {}
    for name, bank in (
        ("train", data.ritz_train_bank),
        ("audit", data.ritz_audit_bank),
        ("validation_fit", data.validation_fit_bank),
        ("validation_audit", data.validation_audit_bank),
    ):
        times, samples = map(int, bank.configurations.shape[:2])
        values = times * samples * basis_size * 8
        gradients = times * samples * basis_size * coordinates * 8
        result[name] = {
            "time_nodes": times,
            "samples_per_time": samples,
            "basis_values_bytes": values,
            "basis_gradients_bytes": gradients,
            "total_bytes": values + gradients,
            "total_gib": (values + gradients) / (1024.0 ** 3),
        }
    per_sample_gram = (
        int(data.ritz_train_bank.configurations.shape[0])
        * int(data.ritz_train_bank.configurations.shape[1])
        * basis_size * basis_size * 8
    )
    result["rejected_train_per_sample_gram"] = {
        "bytes": per_sample_gram,
        "gib": per_sample_gram / (1024.0 ** 3),
        "cached": False,
    }
    return result


def build_or_load_train_basis_cache(
    cfg: dict[str, Any], artifact_dir: Path, dictionary_path: Path,
    dictionary: HybridInvariantDictionary, data: PreparedExperiment, cache_dir: Path,
) -> tuple[list[Array], list[Array], dict[str, Any]]:
    cache_dir = require_fast_output_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature, metadata = _cache_signature(
        cfg, artifact_dir, dictionary_path, dictionary, data
    )
    metadata_path = cache_dir / "train_basis_metadata.json"
    expected_times = int(data.ritz_train_bank.configurations.shape[0])
    complete = False
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            complete = bool(
                previous.get("signature") == signature
                and all((cache_dir / f"train_values_t{t:02d}.npy").is_file()
                        and (cache_dir / f"train_gradients_t{t:02d}.npy").is_file()
                        for t in range(expected_times))
            )
        except (OSError, ValueError, TypeError):
            complete = False
    build_seconds = 0.0
    if not complete:
        start_time = time.perf_counter()
        evaluators = make_basis_evaluators(dictionary, expected_times)
        chunk_size = int(cfg["production_galerkin"]["chunk_size"])
        sample_count = int(data.ritz_train_bank.configurations.shape[1])
        for time_index in range(expected_times):
            value_chunks: list[np.ndarray] = []
            gradient_chunks: list[np.ndarray] = []
            for start in range(0, sample_count, chunk_size):
                stop = min(start + chunk_size, sample_count)
                values, gradients = evaluators[time_index](
                    data.ritz_train_bank.configurations[time_index, start:stop]
                )
                _sync((values, gradients))
                value_chunks.append(np.asarray(values, dtype=np.float64))
                gradient_chunks.append(np.asarray(gradients, dtype=np.float64))
            np.save(cache_dir / f"train_values_t{time_index:02d}.npy",
                    np.concatenate(value_chunks, axis=0), allow_pickle=False)
            np.save(cache_dir / f"train_gradients_t{time_index:02d}.npy",
                    np.concatenate(gradient_chunks, axis=0), allow_pickle=False)
        build_seconds = time.perf_counter() - start_time
        write_json(metadata_path, {
            **metadata,
            "signature": signature,
            "build_seconds": build_seconds,
            "memory_estimates": basis_cache_memory_estimates(data, dictionary.size),
        })
    values = [
        jnp.asarray(np.load(cache_dir / f"train_values_t{t:02d}.npy", mmap_mode="r"))
        for t in range(expected_times)
    ]
    gradients = [
        jnp.asarray(np.load(cache_dir / f"train_gradients_t{t:02d}.npy", mmap_mode="r"))
        for t in range(expected_times)
    ]
    _sync((values, gradients))
    return values, gradients, {
        "signature": signature,
        "cache_hit": complete,
        "build_seconds": build_seconds,
        "metadata_path": str(metadata_path),
        "memory_estimates": basis_cache_memory_estimates(data, dictionary.size),
    }


@jax.jit
def _assemble_gram(gradients: Array, weights: Array) -> tuple[Array, Array]:
    raw = jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients)
    symmetry = jnp.linalg.norm(raw - raw.T) / jnp.maximum(jnp.linalg.norm(raw), 1e-30)
    return 0.5 * (raw + raw.T), symmetry


@jax.jit
def _assemble_load(values: Array, weights: Array, forcing: Array) -> tuple[Array, Array, Array]:
    mean = jnp.einsum("n,nk->k", weights, values)
    forcing_mean = jnp.einsum("n,n->", weights, forcing)
    load = jnp.einsum("n,n,nk->k", weights, forcing, values) - forcing_mean * mean
    return load, mean, forcing_mean


@jax.jit
def _potential_kinetic(values: Array, gradients: Array, coefficients: Array) -> tuple[Array, Array]:
    potential = jnp.einsum("k,nk->n", coefficients, values)
    gradient = jnp.einsum("k,nkpd->npd", coefficients, gradients)
    return potential, jnp.sum(gradient * gradient, axis=(-2, -1))


def _system_from_cached(
    values: list[Array], gradients: list[Array], weights: Array, forcing: Array,
    basis_size: int,
) -> GalerkinSystem:
    grams, loads, means, forcing_means, symmetry = [], [], [], [], []
    for time_index, (value_row, gradient_row) in enumerate(zip(values, gradients, strict=True)):
        gradient_prefix = gradient_row[:, :basis_size]
        gram, raw_symmetry = _assemble_gram(gradient_prefix, weights[time_index])
        load, mean, forcing_mean = _assemble_load(
            value_row[:, :basis_size], weights[time_index], forcing[time_index]
        )
        grams.append(gram)
        loads.append(load)
        means.append(mean)
        forcing_means.append(forcing_mean)
        symmetry.append(raw_symmetry)
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return GalerkinSystem(
        gram=jnp.stack(grams), load=jnp.stack(loads), basis_means=jnp.stack(means),
        centered_basis=empty, weights=empty, forcing=empty,
        raw_symmetry_residual=jnp.stack(symmetry), forcing_mean=jnp.stack(forcing_means),
    )


@dataclass
class FastEvaluation:
    eta: Array
    reconstruction: Any
    train_state: Any
    system: GalerkinSystem
    solve: Any
    action: Array
    gradient: Array | None
    risk: Array


class FastProductionContext:
    """Persistent production arrays, feature cache, and compiled hot functions."""

    def __init__(
        self, cfg: dict[str, Any], artifact_dir: Path,
        dictionary_path: Path | None = None, cache_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.artifact_dir = Path(artifact_dir).resolve()
        self.dictionary_path = Path(dictionary_path or (
            PRODUCTION_ROOT / "convergence" / "features" / "hybrid_dictionary.npz"
        )).resolve()
        self.cache_dir = require_fast_output_path(cache_dir or FAST_ROOT / "cache")
        load_start = time.perf_counter()
        self.data = load_production_data(cfg, self.artifact_dir)
        self.dictionary = load_dictionary(
            self.dictionary_path, box=tuple(cfg["physics"]["box"])
        )
        self.artifact_load_seconds = time.perf_counter() - load_start
        self.values, self.gradients, self.cache_info = build_or_load_train_basis_cache(
            cfg, self.artifact_dir, self.dictionary_path, self.dictionary,
            self.data, self.cache_dir,
        )
        settings = cfg["production_galerkin"]
        self.rank_tolerance = float(settings["relative_rank_tolerance"])
        problem = self.data.selection_problem
        bank = self.data.ritz_train_bank

        def envelope(design: Array, potential_rows: Array, kinetic_rows: Array) -> Array:
            reconstruction = reconstruct_moments(design, problem)
            state = forcing_state(design, problem, bank, reconstruction)
            weights, forcing = state.projection.weights, state.forcing
            kinetic = jnp.einsum("tn,tn->t", weights, kinetic_rows)
            potential_mean = jnp.einsum("tn,tn->t", weights, potential_rows)
            forcing_mean = jnp.einsum("tn,tn->t", weights, forcing)
            linear = jnp.einsum("tn,tn,tn->t", weights, forcing, potential_rows)
            linear = linear - forcing_mean * potential_mean
            return -2.0 * jnp.sum(problem.time_weights * (0.5 * kinetic + linear))

        self._envelope_value_grad = jax.jit(jax.value_and_grad(envelope, argnums=0))
        self._risk = jax.jit(lambda design: selection_risk(design, self.data))

    def assemble(self, weights: Array, forcing: Array, basis_size: int = 160) -> GalerkinSystem:
        return _system_from_cached(
            self.values, self.gradients, weights, forcing, int(basis_size)
        )

    def potential_rows(self, coefficients: Array, basis_size: int) -> tuple[Array, Array]:
        potentials, kinetic = [], []
        for t, (values, gradients) in enumerate(zip(self.values, self.gradients, strict=True)):
            p, q = _potential_kinetic(
                values[:, :basis_size], gradients[:, :basis_size], coefficients[t]
            )
            potentials.append(p)
            kinetic.append(q)
        return jnp.stack(potentials), jnp.stack(kinetic)

    def evaluate(
        self, eta: Array, *, basis_size: int = 160, with_gradient: bool = True,
    ) -> FastEvaluation:
        eta = wrap_periodic(
            jnp.asarray(eta, dtype=jnp.float64), self.data.selection_problem.family
        )
        reconstruction = reconstruct_moments(eta, self.data.selection_problem)
        state = forcing_state(
            eta, self.data.selection_problem, self.data.ritz_train_bank, reconstruction
        )
        system = self.assemble(
            state.projection.weights, state.forcing, basis_size=basis_size
        )
        solve = rank_aware_quadratic_solve(
            system.gram, system.load, relative_rank_tolerance=self.rank_tolerance
        )
        aggregate = aggregate_quadratic_values(
            solve, self.data.selection_problem.time_weights
        )
        gradient = None
        action = aggregate["action"]
        if with_gradient:
            potential, kinetic = self.potential_rows(solve.coefficients, basis_size)
            action, gradient = self._envelope_value_grad(eta, potential, kinetic)
        risk = self._risk(eta)
        return FastEvaluation(
            eta, reconstruction, state, system, solve, action, gradient, risk
        )

    def search_payload(self, evaluation: FastEvaluation) -> dict[str, Any]:
        aggregate = aggregate_quadratic_values(
            evaluation.solve, self.data.selection_problem.time_weights
        )
        return {
            "eta": np.asarray(evaluation.eta).tolist(),
            "action": float(evaluation.action),
            "risk": float(evaluation.risk),
            "gradient": None if evaluation.gradient is None else np.asarray(evaluation.gradient).tolist(),
            "gradient_norm": None if evaluation.gradient is None else float(jnp.linalg.norm(evaluation.gradient)),
            "identity_relerr": float(aggregate["identity_relerr"]),
            "rank_by_time": np.asarray(evaluation.solve.numerical_rank).tolist(),
            "worst_range_residual": float(jnp.max(evaluation.solve.range_residual)),
            "worst_stationarity_residual": float(jnp.max(evaluation.solve.stationarity_residual)),
            "worst_retained_condition": float(jnp.max(evaluation.solve.condition_number)),
            "train_forcing_audit": _forcing_state_payload(
                evaluation.train_state, self.data.selection_problem
            ),
            "geometry_valid": bool(self.data.selection_problem.family.geometry_valid(evaluation.eta)),
        }


__all__ = [
    "FAST_ROOT", "FastEvaluation", "FastProductionContext",
    "basis_cache_memory_estimates", "require_fast_output_path", "_median_timing", "_sync", "_timed",
]
