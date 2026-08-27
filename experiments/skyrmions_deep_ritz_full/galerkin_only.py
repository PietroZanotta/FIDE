"""GPU-first fixed-feature Galerkin Full solver for the isolated 3% study."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import GalerkinSystem, aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only_data import SelectionGalerkinData, selection_risk
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import (
    HybridInvariantDictionary, _ordered_wavevectors, load_dictionary,
)
from .production_galerkin import audit_hybrid_solutions

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

Array = jax.Array

GALERKIN_ONLY_ROOT = OUTPUT_ROOT / "galerkin_only_3pct"
OLD_DICTIONARY = PRODUCTION_ROOT / "convergence" / "features" / "hybrid_dictionary.npz"
OLD_TRAIN_CACHE = OUTPUT_ROOT / "fast_production_3pct" / "cache"
CACHE_VERSION = "galerkin_only_fixed_basis_v1_float64_time_sharded"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save_dictionary(path: Path, dictionary: HybridInvariantDictionary) -> None:
    path = require_galerkin_only_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavevectors=np.asarray(dictionary.wavevectors),
        radial_centers=np.asarray(dictionary.radial_centers),
        radial_widths=np.asarray(dictionary.radial_widths),
        feature_kind=np.asarray(dictionary.feature_kind),
        feature_source_index=np.asarray(dictionary.feature_source_index),
        base_means=np.asarray(dictionary.base_means),
        energy_scales=np.asarray(dictionary.energy_scales),
    )


def _forcing_state_payload(state: Any, problem: Any) -> dict[str, Any]:
    maximum_projection = float(jnp.max(jnp.linalg.norm(
        state.projection.residual, axis=-1
    )))
    minimum_ess = float(jnp.min(state.projection.ess_fraction))
    maximum_mean = float(jnp.max(jnp.abs(state.forcing_mean_before_centering)))
    maximum_condition = float(jnp.max(state.covariance_condition))
    post_mean = float(jnp.max(jnp.abs(jnp.einsum(
        "tn,tn->t", state.projection.weights, state.forcing
    ))))
    cfg = problem.forcing_config
    return {
        "valid": bool(
            maximum_projection <= cfg.projection_tolerance
            and minimum_ess >= cfg.minimum_ess_fraction
            and maximum_mean <= cfg.forcing_mean_tolerance
            and maximum_condition <= cfg.max_covariance_condition
        ),
        "maximum_projection_residual": maximum_projection,
        "minimum_ess_fraction": minimum_ess,
        "maximum_forcing_mean": maximum_mean,
        "maximum_post_centering_forcing_mean": post_mean,
        "maximum_covariance_condition": maximum_condition,
    }


@dataclass(frozen=True)
class GalerkinCertificateThresholds:
    maximum_weak_residual: float = 0.12
    maximum_energy_residual: float = 0.08
    maximum_gauge_residual: float = 1.0e-9
    maximum_moment_rate_residual: float = 0.10


@dataclass
class GalerkinOnlyEvaluation:
    eta: Array
    reconstruction: Any
    train_state: Any
    system: GalerkinSystem
    solve: Any
    action: Array
    gradient: Array | None
    risk: Array


def require_galerkin_only_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = GALERKIN_ONLY_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Galerkin-only output must be beneath {root}, got {resolved}")
    return resolved


def execution_device() -> jax.Device:
    """Prefer CUDA when present and otherwise return the default CPU device."""

    try:
        gpu = jax.devices("gpu")
    except RuntimeError:
        gpu = []
    return gpu[0] if gpu else jax.devices("cpu")[0]


def device_payload(device: jax.Device | None = None) -> dict[str, Any]:
    selected = device or execution_device()
    return {
        "platform": selected.platform,
        "device": str(selected),
        "device_kind": getattr(selected, "device_kind", str(selected)),
        "float64": bool(jax.config.x64_enabled),
        "gpu_preferred": True,
        "cpu_fallback": selected.platform != "gpu",
    }


def _sync(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    _sync(value)
    return value, time.perf_counter() - started


def timing_pair(function: Callable[[], Any], repeats: int = 3) -> dict[str, float]:
    _, first = timed(function)
    steady = [timed(function)[1] for _ in range(int(repeats))]
    return {
        "first_seconds": first,
        "steady_median_seconds": statistics.median(steady),
    }


def prefix_dictionary(
    dictionary: HybridInvariantDictionary, basis_size: int,
) -> HybridInvariantDictionary:
    size = int(basis_size)
    if size < 1 or size > dictionary.size:
        raise ValueError(f"invalid dictionary prefix {size} of {dictionary.size}")
    return HybridInvariantDictionary(
        box=dictionary.box,
        wavevectors=dictionary.wavevectors,
        radial_centers=dictionary.radial_centers,
        radial_widths=dictionary.radial_widths,
        feature_kind=dictionary.feature_kind[:size],
        feature_source_index=dictionary.feature_source_index[:size],
        base_means=dictionary.base_means[:, :size],
        energy_scales=dictionary.energy_scales[:, :size],
    )


def _fourier_suffix_values_gradients(
    configurations: Array, wavevectors: Array,
) -> tuple[Array, Array]:
    x = jnp.asarray(configurations, dtype=jnp.float64)
    vectors = jnp.asarray(wavevectors, dtype=jnp.float64)
    phases = jnp.einsum("...pd,md->...pm", x, vectors)
    particles = float(x.shape[-2])
    cosine = jnp.mean(jnp.cos(phases), axis=-2)
    sine = jnp.mean(jnp.sin(phases), axis=-2)
    cosine_gradient = jnp.moveaxis(
        -jnp.sin(phases)[..., :, :, None] * vectors[None, None, :, :] / particles,
        -3, -2,
    )
    sine_gradient = jnp.moveaxis(
        jnp.cos(phases)[..., :, :, None] * vectors[None, None, :, :] / particles,
        -3, -2,
    )
    values = jnp.stack((cosine, sine), axis=-1).reshape(x.shape[:-2] + (-1,))
    gradients = jnp.stack(
        (cosine_gradient, sine_gradient), axis=-3
    ).reshape(x.shape[:-2] + (-1,) + x.shape[-2:])
    return values, gradients


def _next_wavevectors(
    existing: Array, count: int, box: tuple[float, float],
) -> Array:
    old = np.asarray(existing, dtype=np.float64)
    candidates = np.asarray(
        _ordered_wavevectors(max(4 * (len(old) + count), 512), box),
        dtype=np.float64,
    )
    selected: list[np.ndarray] = []
    for row in candidates:
        if any(np.array_equal(row, prior) for prior in old):
            continue
        selected.append(row)
        if len(selected) == count:
            return jnp.asarray(np.stack(selected), dtype=jnp.float64)
    raise RuntimeError("could not construct enough deterministic Fourier extensions")


def _extended_dictionary_signature(
    cfg: dict[str, Any], artifact_dir: Path, maximum_size: int,
) -> dict[str, Any]:
    return {
        "kind": "nested_hybrid_fourier_extension_v1",
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "old_dictionary_sha256": file_sha256(OLD_DICTIONARY),
        "old_size": 160,
        "maximum_size": int(maximum_size),
        "dtype": "float64",
        "box": list(cfg["physics"]["box"]),
        "extension": "append next globally ordered unseen periodic Fourier cos/sin pairs",
        "normalization": "selection train base-weight mean and Dirichlet energy",
        "configuration_hash": fingerprint(cfg["production_galerkin"]),
    }


def build_or_load_extended_dictionary(
    cfg: dict[str, Any], artifact_dir: Path, data: SelectionGalerkinData,
    output_dir: Path, *, maximum_size: int = 280,
) -> tuple[HybridInvariantDictionary, dict[str, Any]]:
    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / f"dictionary_K{maximum_size}.npz"
    metadata_path = output_dir / f"dictionary_K{maximum_size}.json"
    signature_payload = _extended_dictionary_signature(cfg, artifact_dir, maximum_size)
    signature = fingerprint(signature_payload)
    if dictionary_path.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return load_dictionary(
                dictionary_path, box=tuple(cfg["physics"]["box"])
            ), {**previous, "cache_hit": True}

    old = load_dictionary(OLD_DICTIONARY, box=tuple(cfg["physics"]["box"]))
    base_path = output_dir / "dictionary_K240.npz"
    base = (
        load_dictionary(base_path, box=tuple(cfg["physics"]["box"]))
        if maximum_size > 240 and base_path.is_file()
        else old
    )
    if old.size != 160 or maximum_size < base.size or (maximum_size - base.size) % 2:
        raise ValueError("extension must preserve K=160 and append Fourier pairs")
    added_vectors = (maximum_size - base.size) // 2
    suffix_wavevectors = _next_wavevectors(base.wavevectors, added_vectors, base.box)
    wavevectors = jnp.concatenate((base.wavevectors, suffix_wavevectors), axis=0)
    kinds = jnp.concatenate((
        base.feature_kind,
        jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), added_vectors),
    ))
    new_sources = jnp.repeat(
        jnp.arange(base.wavevectors.shape[0], wavevectors.shape[0], dtype=jnp.int32), 2
    )
    sources = jnp.concatenate((base.feature_source_index, new_sources))

    chunk_size = int(cfg["production_galerkin"]["chunk_size"])
    evaluator = jax.jit(
        lambda rows: _fourier_suffix_values_gradients(rows, suffix_wavevectors)
    )
    means, scales = [], []
    started = time.perf_counter()
    for time_index in range(int(data.train_bank.configurations.shape[0])):
        mean = jnp.zeros((maximum_size - base.size,), dtype=jnp.float64)
        energy = jnp.zeros_like(mean)
        sample_count = int(data.train_bank.configurations.shape[1])
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            values, gradients = evaluator(
                data.train_bank.configurations[time_index, start:stop]
            )
            weights = data.train_bank.base_weights[time_index, start:stop]
            mean = mean + jnp.einsum("n,nk->k", weights, values)
            energy = energy + jnp.einsum(
                "n,nkpd,nkpd->k", weights, gradients, gradients
            )
        means.append(mean)
        scales.append(jnp.sqrt(jnp.maximum(energy, 1.0e-12)))
    dictionary = HybridInvariantDictionary(
        box=base.box,
        wavevectors=wavevectors,
        radial_centers=base.radial_centers,
        radial_widths=base.radial_widths,
        feature_kind=kinds,
        feature_source_index=sources,
        base_means=jnp.concatenate((base.base_means, jnp.stack(means)), axis=1),
        energy_scales=jnp.concatenate((base.energy_scales, jnp.stack(scales)), axis=1),
    )
    prefix_exact = bool(
        np.array_equal(np.asarray(dictionary.feature_kind[:base.size]), np.asarray(base.feature_kind))
        and np.array_equal(np.asarray(dictionary.feature_source_index[:base.size]), np.asarray(base.feature_source_index))
        and np.array_equal(np.asarray(dictionary.base_means[:, :base.size]), np.asarray(base.base_means))
        and np.array_equal(np.asarray(dictionary.energy_scales[:, :base.size]), np.asarray(base.energy_scales))
        and np.array_equal(np.asarray(dictionary.wavevectors[:base.wavevectors.shape[0]]), np.asarray(base.wavevectors))
    )
    if not prefix_exact:
        raise RuntimeError(f"extended dictionary changed an existing K={base.size} coordinate")
    _save_dictionary(dictionary_path, dictionary)
    metadata = {
        **signature_payload,
        "signature": signature,
        "dictionary_sha256": file_sha256(dictionary_path),
        "normalization_seconds": time.perf_counter() - started,
        "prefix_160_exact": True,
        "preserved_prefix_size": base.size,
        "new_fourier_wavevector_count": added_vectors,
        "new_coordinate_count": maximum_size - old.size,
        "eta_independent": True,
        "prefix_nested": True,
        "cache_hit": False,
    }
    _write_json(metadata_path, metadata)
    return dictionary, metadata


def _cache_signature(
    cfg: dict[str, Any], artifact_dir: Path, dictionary_path: Path,
    dictionary: HybridInvariantDictionary, data: SelectionGalerkinData,
) -> tuple[str, dict[str, Any]]:
    metadata = {
        "cache_version": CACHE_VERSION,
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "basis_definition_sha256": file_sha256(dictionary_path),
        "basis_size": dictionary.size,
        "normalization_hash": fingerprint({
            "means": np.asarray(dictionary.base_means).tolist(),
            "scales": np.asarray(dictionary.energy_scales).tolist(),
        }),
        "dtype": "float64",
        "configuration_hash": fingerprint({
            key: cfg[key] for key in (
                "physics", "measurement", "moment_reconstruction",
                "projection", "forcing", "production_galerkin",
            )
        }),
        "train_shape": list(data.train_bank.configurations.shape),
    }
    return fingerprint(metadata), metadata


def _cache_files_complete(cache_dir: Path, time_count: int) -> bool:
    return all(
        (cache_dir / f"train_values_t{t:02d}.npy").is_file()
        and (cache_dir / f"train_gradients_t{t:02d}.npy").is_file()
        for t in range(time_count)
    )


def load_train_cache(
    cache_dir: Path, *, time_count: int, basis_size: int,
) -> tuple[list[Array], list[Array]]:
    values, gradients = [], []
    for time_index in range(time_count):
        value = np.load(
            cache_dir / f"train_values_t{time_index:02d}.npy", mmap_mode="r"
        )[:, :basis_size]
        gradient = np.load(
            cache_dir / f"train_gradients_t{time_index:02d}.npy", mmap_mode="r"
        )[:, :basis_size]
        values.append(jnp.asarray(value, dtype=jnp.float64))
        gradients.append(jnp.asarray(gradient, dtype=jnp.float64))
    _sync((values, gradients))
    return values, gradients


def build_or_load_train_cache(
    cfg: dict[str, Any], artifact_dir: Path, dictionary_path: Path,
    dictionary: HybridInvariantDictionary, data: SelectionGalerkinData,
    cache_dir: Path,
) -> tuple[list[Array], list[Array], dict[str, Any]]:
    cache_dir = require_galerkin_only_output_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature, metadata = _cache_signature(
        cfg, artifact_dir, dictionary_path, dictionary, data
    )
    metadata_path = cache_dir / "train_basis_metadata.json"
    time_count = int(data.train_bank.configurations.shape[0])
    complete = False
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        complete = bool(
            previous.get("signature") == signature
            and _cache_files_complete(cache_dir, time_count)
        )
    build_seconds = 0.0
    if not complete:
        if dictionary.size < 160:
            raise ValueError("production Galerkin cache requires at least K=160")
        possible_K240 = GALERKIN_ONLY_ROOT / "cache" / "K240"
        use_K240 = bool(
            dictionary.size > 240
            and (possible_K240 / "train_basis_metadata.json").is_file()
            and _cache_files_complete(possible_K240, time_count)
        )
        base_cache = possible_K240 if use_K240 else OLD_TRAIN_CACHE
        base_size = 240 if use_K240 else 160
        base_wavevector_count = 104 if use_K240 else 64
        base_metadata = json.loads(
            (base_cache / "train_basis_metadata.json").read_text(encoding="utf-8")
        )
        if (
            base_metadata.get("basis_size") != base_size
            or base_metadata.get("artifact_manifest_sha256")
            != metadata["artifact_manifest_sha256"]
        ):
            raise RuntimeError(f"validated K={base_size} cache is incompatible")
        if not use_K240 and base_metadata.get("dictionary_sha256") != file_sha256(OLD_DICTIONARY):
            raise RuntimeError("validated K=160 dictionary signature is incompatible")
        suffix = dictionary.wavevectors[base_wavevector_count:]
        suffix_means = dictionary.base_means[:, base_size:]
        suffix_scales = dictionary.energy_scales[:, base_size:]
        evaluator = jax.jit(
            lambda rows: _fourier_suffix_values_gradients(rows, suffix)
        )
        chunk_size = int(cfg["production_galerkin"]["chunk_size"])
        sample_count = int(data.train_bank.configurations.shape[1])
        particles = int(data.train_bank.configurations.shape[-2])
        started = time.perf_counter()
        for time_index in range(time_count):
            old_values = np.load(
                base_cache / f"train_values_t{time_index:02d}.npy", mmap_mode="r"
            )
            old_gradients = np.load(
                base_cache / f"train_gradients_t{time_index:02d}.npy", mmap_mode="r"
            )
            values_file = np.lib.format.open_memmap(
                cache_dir / f"train_values_t{time_index:02d}.npy", mode="w+",
                dtype=np.float64, shape=(sample_count, dictionary.size),
            )
            gradients_file = np.lib.format.open_memmap(
                cache_dir / f"train_gradients_t{time_index:02d}.npy", mode="w+",
                dtype=np.float64,
                shape=(sample_count, dictionary.size, particles, 2),
            )
            values_file[:, :base_size] = old_values
            gradients_file[:, :base_size] = old_gradients
            for start in range(0, sample_count, chunk_size):
                stop = min(start + chunk_size, sample_count)
                values, gradients = evaluator(
                    data.train_bank.configurations[time_index, start:stop]
                )
                values = (values - suffix_means[time_index]) / suffix_scales[time_index]
                gradients = gradients / suffix_scales[time_index, :, None, None]
                values_file[start:stop, base_size:] = np.asarray(values)
                gradients_file[start:stop, base_size:] = np.asarray(gradients)
            values_file.flush()
            gradients_file.flush()
        build_seconds = time.perf_counter() - started
        _write_json(metadata_path, {
            **metadata,
            "signature": signature,
            "build_seconds": build_seconds,
            "reused_exact_K160_prefix": True,
            "reused_prefix_size": base_size,
        })
    values, gradients = load_train_cache(
        cache_dir, time_count=time_count, basis_size=dictionary.size
    )
    return values, gradients, {
        "signature": signature,
        "metadata_path": str(metadata_path),
        "cache_hit": complete,
        "build_seconds": build_seconds,
        "reused_exact_K160_prefix": True,
    }


@jax.jit
def _assemble_gram(gradients: Array, weights: Array) -> tuple[Array, Array]:
    raw = jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients)
    symmetry = jnp.linalg.norm(raw - raw.T) / jnp.maximum(jnp.linalg.norm(raw), 1e-30)
    return 0.5 * (raw + raw.T), symmetry


@jax.jit
def _assemble_load(values: Array, weights: Array, forcing: Array):
    mean = jnp.einsum("n,nk->k", weights, values)
    forcing_mean = jnp.einsum("n,n->", weights, forcing)
    load = jnp.einsum("n,n,nk->k", weights, forcing, values) - forcing_mean * mean
    return load, mean, forcing_mean


@jax.jit
def _potential_kinetic(values: Array, gradients: Array, coefficients: Array):
    potential = jnp.einsum("k,nk->n", coefficients, values)
    gradient = jnp.einsum("k,nkpd->npd", coefficients, gradients)
    return potential, jnp.sum(gradient * gradient, axis=(-2, -1))


def _system_from_cached(
    values: list[Array], gradients: list[Array], weights: Array,
    forcing: Array, basis_size: int,
) -> GalerkinSystem:
    grams, loads, means, forcing_means, symmetry = [], [], [], [], []
    for time_index, (value_row, gradient_row) in enumerate(
        zip(values, gradients, strict=True)
    ):
        gram, residual = _assemble_gram(
            gradient_row[:, :basis_size], weights[time_index]
        )
        load, mean, forcing_mean = _assemble_load(
            value_row[:, :basis_size], weights[time_index], forcing[time_index]
        )
        grams.append(gram)
        loads.append(load)
        means.append(mean)
        forcing_means.append(forcing_mean)
        symmetry.append(residual)
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return GalerkinSystem(
        gram=jnp.stack(grams), load=jnp.stack(loads),
        basis_means=jnp.stack(means), centered_basis=empty,
        weights=empty, forcing=empty,
        raw_symmetry_residual=jnp.stack(symmetry),
        forcing_mean=jnp.stack(forcing_means),
    )


class GalerkinOnlyContext:
    """Persistent selection-only arrays and fixed-basis cache."""

    def __init__(
        self, cfg: dict[str, Any], artifact_dir: Path,
        data: SelectionGalerkinData, dictionary_path: Path,
        *, cache_dir: Path, reuse_validated_K160: bool = False,
    ) -> None:
        self.cfg = cfg
        self.artifact_dir = Path(artifact_dir).resolve()
        self.data = data
        self.dictionary_path = Path(dictionary_path).resolve()
        started = time.perf_counter()
        self.dictionary = load_dictionary(
            self.dictionary_path, box=tuple(cfg["physics"]["box"])
        )
        self.dictionary_load_seconds = time.perf_counter() - started
        if reuse_validated_K160:
            if self.dictionary.size != 160:
                raise ValueError("validated cache reuse is only defined for K=160")
            old_metadata = json.loads(
                (OLD_TRAIN_CACHE / "train_basis_metadata.json").read_text(encoding="utf-8")
            )
            if (
                old_metadata.get("basis_size") != 160
                or old_metadata.get("dictionary_sha256") != file_sha256(self.dictionary_path)
                or old_metadata.get("artifact_manifest_sha256")
                != file_sha256(self.artifact_dir / "isolated_artifact_manifest.json")
            ):
                raise RuntimeError("validated K=160 cache signature mismatch")
            cache_started = time.perf_counter()
            self.values, self.gradients = load_train_cache(
                OLD_TRAIN_CACHE,
                time_count=int(data.train_bank.configurations.shape[0]),
                basis_size=160,
            )
            self.cache_info = {
                "cache_hit": True,
                "source": str(OLD_TRAIN_CACHE),
                "signature": old_metadata["signature"],
                "load_seconds": time.perf_counter() - cache_started,
                "reused_validated_K160": True,
            }
        else:
            self.values, self.gradients, self.cache_info = build_or_load_train_cache(
                cfg, self.artifact_dir, self.dictionary_path, self.dictionary,
                data, cache_dir,
            )
        settings = cfg["production_galerkin"]
        self.rank_tolerance = float(settings["relative_rank_tolerance"])
        problem, bank = data.selection_problem, data.train_bank

        def envelope(design: Array, potential_rows: Array, kinetic_rows: Array) -> Array:
            reconstruction = reconstruct_moments(design, problem)
            state = forcing_state(design, problem, bank, reconstruction)
            weights, forcing = state.projection.weights, state.forcing
            kinetic = jnp.einsum("tn,tn->t", weights, kinetic_rows)
            potential_mean = jnp.einsum("tn,tn->t", weights, potential_rows)
            forcing_mean = jnp.einsum("tn,tn->t", weights, forcing)
            linear = jnp.einsum("tn,tn,tn->t", weights, forcing, potential_rows)
            linear = linear - forcing_mean * potential_mean
            return -2.0 * jnp.sum(
                problem.time_weights * (0.5 * kinetic + linear)
            )

        self._envelope_value_grad = jax.jit(jax.value_and_grad(envelope, argnums=0))
        self._risk = jax.jit(lambda design: selection_risk(design, data))
        self._risk_value_grad = jax.jit(jax.value_and_grad(
            lambda design: selection_risk(design, data)
        ))

    @property
    def maximum_basis_size(self) -> int:
        return self.dictionary.size

    def assemble(self, weights: Array, forcing: Array, basis_size: int) -> GalerkinSystem:
        return _system_from_cached(
            self.values, self.gradients, weights, forcing, int(basis_size)
        )

    def potential_rows(self, coefficients: Array, basis_size: int):
        potential, kinetic = [], []
        for time_index, (values, gradients) in enumerate(
            zip(self.values, self.gradients, strict=True)
        ):
            p, q = _potential_kinetic(
                values[:, :basis_size], gradients[:, :basis_size],
                coefficients[time_index],
            )
            potential.append(p)
            kinetic.append(q)
        return jnp.stack(potential), jnp.stack(kinetic)

    def evaluate(
        self, eta: Array, *, basis_size: int, with_gradient: bool = True,
    ) -> GalerkinOnlyEvaluation:
        eta = wrap_periodic(
            jnp.asarray(eta, dtype=jnp.float64),
            self.data.selection_problem.family,
        )
        reconstruction = reconstruct_moments(eta, self.data.selection_problem)
        state = forcing_state(
            eta, self.data.selection_problem, self.data.train_bank, reconstruction
        )
        system = self.assemble(state.projection.weights, state.forcing, basis_size)
        solve = rank_aware_quadratic_solve(
            system.gram, system.load,
            relative_rank_tolerance=self.rank_tolerance,
        )
        aggregate = aggregate_quadratic_values(
            solve, self.data.selection_problem.time_weights
        )
        action, gradient = aggregate["action"], None
        if with_gradient:
            potential, kinetic = self.potential_rows(solve.coefficients, basis_size)
            action, gradient = self._envelope_value_grad(eta, potential, kinetic)
        return GalerkinOnlyEvaluation(
            eta=eta, reconstruction=reconstruction, train_state=state,
            system=system, solve=solve, action=action, gradient=gradient,
            risk=self._risk(eta),
        )

    def payload(self, evaluation: GalerkinOnlyEvaluation) -> dict[str, Any]:
        aggregate = aggregate_quadratic_values(
            evaluation.solve, self.data.selection_problem.time_weights
        )
        ranks = evaluation.solve.numerical_rank
        size = int(evaluation.solve.coefficients.shape[-1])
        return {
            "eta": np.asarray(evaluation.eta).tolist(),
            "basis_size": size,
            "action": float(evaluation.action),
            "risk": float(evaluation.risk),
            "gradient": None if evaluation.gradient is None else np.asarray(evaluation.gradient).tolist(),
            "gradient_norm": None if evaluation.gradient is None else float(jnp.linalg.norm(evaluation.gradient)),
            "identity_relerr": float(aggregate["identity_relerr"]),
            "rank_by_time": np.asarray(ranks).tolist(),
            "minimum_rank_fraction": float(jnp.min(ranks / float(size))),
            "worst_range_residual": float(jnp.max(evaluation.solve.range_residual)),
            "worst_stationarity_residual": float(jnp.max(evaluation.solve.stationarity_residual)),
            "worst_retained_condition": float(jnp.max(evaluation.solve.condition_number)),
            "worst_symmetry_residual": float(jnp.max(evaluation.system.raw_symmetry_residual)),
            "train_forcing_audit": _forcing_state_payload(
                evaluation.train_state, self.data.selection_problem
            ),
            "geometry_valid": bool(
                self.data.selection_problem.family.geometry_valid(evaluation.eta)
            ),
        }

    def certify(
        self, evaluation: GalerkinOnlyEvaluation,
    ) -> dict[str, Any]:
        eta = evaluation.eta
        audit_state = forcing_state(
            eta, self.data.selection_problem, self.data.audit_bank,
            evaluation.reconstruction,
        )
        prefix = prefix_dictionary(
            self.dictionary, int(evaluation.solve.coefficients.shape[-1])
        )
        thresholds = GalerkinCertificateThresholds(
            **self.cfg["production_galerkin"]["certificate_thresholds"]
        )
        adapter = SimpleNamespace(
            ritz_audit_bank=self.data.audit_bank,
            selection_problem=self.data.selection_problem,
        )
        certificate = audit_hybrid_solutions(
            prefix, evaluation.solve.coefficients[None], adapter, eta,
            evaluation.reconstruction, audit_state, thresholds,
            chunk_size=int(self.cfg["production_galerkin"]["chunk_size"]),
        )[0]
        payload = self.payload(evaluation)
        settings = self.cfg["production_galerkin"]
        algebra_valid = bool(
            payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
            and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
            and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
            and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
            and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
            and payload["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
        )
        audit_forcing = _forcing_state_payload(
            audit_state, self.data.selection_problem
        )
        return {
            **payload,
            "algebra_valid": algebra_valid,
            "audit_forcing_audit": audit_forcing,
            "heldout_certificate": certificate,
            "certified": bool(
                algebra_valid
                and payload["geometry_valid"]
                and payload["train_forcing_audit"]["valid"]
                and audit_forcing["valid"]
                and certificate["valid"]
            ),
        }


def basis_memory_estimate(data: SelectionGalerkinData, basis_size: int) -> dict[str, Any]:
    times, samples, particles, dimensions = map(
        int, data.train_bank.configurations.shape
    )
    values = times * samples * basis_size * 8
    gradients = times * samples * basis_size * particles * dimensions * 8
    return {
        "basis_size": int(basis_size),
        "train_values_bytes": values,
        "train_gradients_bytes": gradients,
        "train_total_gib": (values + gradients) / 1024.0**3,
        "per_sample_gram_cached": False,
        "per_sample_gram_gib": times * samples * basis_size**2 * 8 / 1024.0**3,
    }


__all__ = [
    "CACHE_VERSION", "GALERKIN_ONLY_ROOT", "OLD_DICTIONARY", "OLD_TRAIN_CACHE",
    "GalerkinCertificateThresholds", "GalerkinOnlyContext",
    "GalerkinOnlyEvaluation", "basis_memory_estimate",
    "build_or_load_extended_dictionary", "build_or_load_train_cache",
    "device_payload", "execution_device", "prefix_dictionary",
    "require_galerkin_only_output_path", "timed", "timing_pair",
]
