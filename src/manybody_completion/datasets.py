"""Synthetic ensemble dataset generation and matched-regime selection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .config import require_keys
from .energies import EnergyParameters, total_energy_per_configuration
from .observables import (
    PairBasis,
    angular_cosine_moments,
    ensemble_pair_moments,
    pair_diagnostics,
)
from .simulators import LangevinConfig, random_uniform_ensemble, simulate_overdamped_langevin


def _energy_params(raw: dict[str, Any]) -> EnergyParameters:
    allowed = set(EnergyParameters.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown energy parameters: {unknown}")
    return EnergyParameters(**raw)


def _langevin_config(raw: dict[str, Any]) -> LangevinConfig:
    allowed = set(LangevinConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown simulator parameters: {unknown}")
    return LangevinConfig(**raw)


def _pair_basis(raw: dict[str, Any], dtype: jnp.dtype) -> PairBasis:
    require_keys(raw, ("num_basis", "r_min", "r_max"), "pair_basis")
    return PairBasis.uniform(
        int(raw["num_basis"]),
        float(raw["r_min"]),
        float(raw["r_max"]),
        None if raw.get("width") is None else float(raw["width"]),
        dtype=dtype,
    )


def generate_dataset(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate reusable ensemble samples for all configured regimes."""
    require_keys(
        config,
        ("seed", "dtype", "box", "num_particles", "num_replicas", "samples_per_regime", "pair_basis", "simulator", "regimes"),
        "dataset config",
    )
    dtype_name = str(config["dtype"])
    if dtype_name not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32

    seed = int(config["seed"])
    box = jnp.asarray(config["box"], dtype=dtype)
    n = int(config["num_particles"])
    m = int(config["num_replicas"])
    samples_per_regime = int(config["samples_per_regime"])
    basis = _pair_basis(config["pair_basis"], dtype)
    simulator = _langevin_config(config["simulator"])
    angular_orders = jnp.asarray(config.get("angular_orders", [1, 2, 4, 6]), dtype=dtype)
    overlap_threshold = float(config.get("overlap_threshold", 0.10))

    all_coordinates: list[np.ndarray] = []
    all_pair_moments: list[np.ndarray] = []
    all_angular_moments: list[np.ndarray] = []
    all_energy: list[np.ndarray] = []
    all_min_distance: list[np.ndarray] = []
    all_overlap_fraction: list[np.ndarray] = []
    all_labels: list[int] = []
    all_regime_names: list[str] = []
    all_parameter_vectors: list[np.ndarray] = []

    key = jax.random.PRNGKey(seed)
    regime_metadata: list[dict[str, Any]] = []
    for label, regime in enumerate(config["regimes"]):
        require_keys(regime, ("name", "family", "energy"), f"regime[{label}]")
        family = str(regime["family"])
        if family not in {"pair", "angular"}:
            raise ValueError(f"unsupported family {family!r}")
        energy_params = _energy_params(regime["energy"])
        regime_metadata.append(
            {
                "label": label,
                "name": str(regime["name"]),
                "family": family,
                "energy": asdict(energy_params),
            }
        )
        parameter_vector = np.asarray(
            [
                energy_params.r0,
                energy_params.kappa,
                energy_params.pair_strength,
                energy_params.angular_strength,
                energy_params.angular_target_cosine,
                energy_params.angular_neighbor_scale,
                simulator.temperature,
            ],
            dtype=dtype_name,
        )
        for _ in range(samples_per_regime):
            key, init_key, simulation_key = jax.random.split(key, 3)
            initial = random_uniform_ensemble(init_key, m, n, box, dtype=dtype)
            final, _ = simulate_overdamped_langevin(
                simulation_key,
                initial,
                box,
                energy_params,
                simulator,
                family,
            )
            pair = ensemble_pair_moments(final, box, basis)
            angular = jnp.mean(
                angular_cosine_moments(
                    final,
                    box,
                    angular_orders,
                    energy_params.angular_neighbor_scale,
                ),
                axis=0,
            )
            energy = total_energy_per_configuration(final, box, energy_params, family)
            diagnostics = pair_diagnostics(final, box, overlap_threshold)

            all_coordinates.append(np.asarray(final))
            all_pair_moments.append(np.asarray(pair))
            all_angular_moments.append(np.asarray(angular))
            all_energy.append(np.asarray(energy))
            all_min_distance.append(np.asarray(diagnostics["minimum_pair_distance"]))
            all_overlap_fraction.append(np.asarray(diagnostics["overlap_fraction"]))
            all_labels.append(label)
            all_regime_names.append(str(regime["name"]))
            all_parameter_vectors.append(parameter_vector)

    arrays = {
        "coordinates": np.stack(all_coordinates),
        "pair_moments": np.stack(all_pair_moments),
        "angular_moments": np.stack(all_angular_moments),
        "energy_per_replica": np.stack(all_energy),
        "minimum_pair_distance": np.stack(all_min_distance),
        "overlap_fraction": np.stack(all_overlap_fraction),
        "regime_label": np.asarray(all_labels, dtype=np.int32),
        "regime_name": np.asarray(all_regime_names),
        "parameter_vector": np.stack(all_parameter_vectors),
        "pair_basis_centers": np.asarray(basis.centers),
        "pair_basis_widths": np.asarray(basis.widths),
        "angular_orders": np.asarray(angular_orders),
        "box": np.asarray(box),
    }
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "dtype": dtype_name,
        "num_particles": n,
        "num_replicas": m,
        "samples_per_regime": samples_per_regime,
        "simulator": asdict(simulator),
        "parameter_vector_fields": [
            "r0",
            "kappa",
            "pair_strength",
            "angular_strength",
            "angular_target_cosine",
            "angular_neighbor_scale",
            "temperature",
        ],
        "regimes": regime_metadata,
        "source_config": config,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
    }
    return arrays, metadata


def save_dataset(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output_path: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    np.savez_compressed(output_path, **arrays)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, metadata_path


def whitened_pair_distance(pair_moments: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Return all pairwise whitened distances between samples."""
    centered = pair_moments - np.mean(pair_moments, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    covariance = covariance + ridge * np.eye(covariance.shape[0])
    whitener = np.linalg.inv(np.linalg.cholesky(covariance))
    whitened = centered @ whitener.T
    delta = whitened[:, None, :] - whitened[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def select_matched_cross_regime_pairs(
    arrays: dict[str, np.ndarray],
    max_pairs: int = 10,
    ridge: float = 1e-6,
) -> list[dict[str, float | int | str]]:
    """Rank cross-regime pairs by close pair statistics and separated angles."""
    pair = np.asarray(arrays["pair_moments"])
    angular = np.asarray(arrays["angular_moments"])
    labels = np.asarray(arrays["regime_label"])
    names = np.asarray(arrays["regime_name"])
    pair_distance = whitened_pair_distance(pair, ridge=ridge)
    angular_scale = np.std(angular, axis=0) + 1e-8
    angular_delta = (angular[:, None, :] - angular[None, :, :]) / angular_scale
    angular_distance = np.linalg.norm(angular_delta, axis=-1)

    candidates: list[dict[str, float | int | str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                continue
            pd = float(pair_distance[i, j])
            ad = float(angular_distance[i, j])
            score = ad / (pd + 1e-6)
            candidates.append(
                {
                    "sample_a": i,
                    "sample_b": j,
                    "regime_a": str(names[i]),
                    "regime_b": str(names[j]),
                    "pair_distance": pd,
                    "angular_distance": ad,
                    "score": score,
                }
            )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:max_pairs]
