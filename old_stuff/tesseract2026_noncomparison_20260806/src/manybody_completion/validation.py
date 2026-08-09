"""Dataset validation and ambiguity-readiness diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .energies import EnergyParameters, total_energy_per_configuration
from .geometry import translate
from .observables import (
    PairBasis,
    angular_cosine_moments,
    ensemble_pair_moments,
    pair_diagnostics,
)


@dataclass(frozen=True)
class ValidationTolerances:
    recomputation_atol: float = 2e-9
    invariance_atol: float = 2e-9
    coordinate_atol: float = 1e-12
    covariance_rtol: float = 1e-10


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _standardized_centroid_distance(values: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) != 2:
        return float("nan")
    scale = np.std(values, axis=0, ddof=1) + 1e-12
    centroids = [np.mean(values[labels == label], axis=0) for label in unique]
    return float(np.linalg.norm((centroids[0] - centroids[1]) / scale))


def validate_dataset(
    dataset_path: str | Path,
    metadata_path: str | Path | None = None,
    tolerances: ValidationTolerances | None = None,
) -> dict[str, Any]:
    """Recompute stored quantities and return a structured validation report."""
    if tolerances is None:
        tolerances = ValidationTolerances()
    dataset_path = Path(dataset_path)
    metadata_path = Path(metadata_path) if metadata_path else dataset_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    required = {
        "coordinates",
        "pair_moments",
        "angular_moments",
        "energy_per_replica",
        "minimum_pair_distance",
        "overlap_fraction",
        "regime_label",
        "pair_basis_centers",
        "pair_basis_widths",
        "angular_orders",
        "box",
        "parameter_vector",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"dataset is missing arrays: {missing}")

    coordinates = arrays["coordinates"]
    pair_moments = arrays["pair_moments"]
    angular_moments = arrays["angular_moments"]
    labels = arrays["regime_label"]
    s, m, n, d = coordinates.shape
    r = pair_moments.shape[-1]
    q = angular_moments.shape[-1]
    shape_checks = {
        "coordinates": coordinates.ndim == 4 and d == 2,
        "pair_moments": pair_moments.shape == (s, r),
        "angular_moments": angular_moments.shape == (s, q),
        "energy_per_replica": arrays["energy_per_replica"].shape == (s, m),
        "minimum_pair_distance": arrays["minimum_pair_distance"].shape == (s, m),
        "overlap_fraction": arrays["overlap_fraction"].shape == (s, m),
        "labels": labels.shape == (s,),
    }

    dtype_name = metadata.get("dtype", str(coordinates.dtype))
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    box = jnp.asarray(arrays["box"], dtype=dtype)
    basis = PairBasis(
        centers=jnp.asarray(arrays["pair_basis_centers"], dtype=dtype),
        widths=jnp.asarray(arrays["pair_basis_widths"], dtype=dtype),
    )
    orders = jnp.asarray(arrays["angular_orders"], dtype=dtype)
    overlap_threshold = float(metadata["source_config"].get("overlap_threshold", 0.10))

    recomputed_pair = []
    recomputed_angular = []
    recomputed_energy = []
    recomputed_minimum = []
    recomputed_overlap = []
    regimes_by_label = {int(item["label"]): item for item in metadata["regimes"]}
    for sample_index in range(s):
        sample = jnp.asarray(coordinates[sample_index], dtype=dtype)
        label = int(labels[sample_index])
        regime = regimes_by_label[label]
        params = EnergyParameters(**regime["energy"])
        recomputed_pair.append(np.asarray(ensemble_pair_moments(sample, box, basis)))
        recomputed_angular.append(
            np.asarray(
                jnp.mean(
                    angular_cosine_moments(
                        sample, box, orders, params.angular_neighbor_scale
                    ),
                    axis=0,
                )
            )
        )
        recomputed_energy.append(
            np.asarray(
                total_energy_per_configuration(sample, box, params, regime["family"])
            )
        )
        pair_diag = pair_diagnostics(sample, box, overlap_threshold)
        recomputed_minimum.append(np.asarray(pair_diag["minimum_pair_distance"]))
        recomputed_overlap.append(np.asarray(pair_diag["overlap_fraction"]))

    recomputed_pair = np.stack(recomputed_pair)
    recomputed_angular = np.stack(recomputed_angular)
    recomputed_energy = np.stack(recomputed_energy)
    recomputed_minimum = np.stack(recomputed_minimum)
    recomputed_overlap = np.stack(recomputed_overlap)
    recomputation_errors = {
        "pair_moments": _max_abs(recomputed_pair, pair_moments),
        "angular_moments": _max_abs(recomputed_angular, angular_moments),
        "energy_per_replica": _max_abs(recomputed_energy, arrays["energy_per_replica"]),
        "minimum_pair_distance": _max_abs(recomputed_minimum, arrays["minimum_pair_distance"]),
        "overlap_fraction": _max_abs(recomputed_overlap, arrays["overlap_fraction"]),
    }

    # Probe U1--U3-style invariances on the first stored ensemble.
    sample = jnp.asarray(coordinates[0], dtype=dtype)
    regime = regimes_by_label[int(labels[0])]
    params = EnergyParameters(**regime["energy"])
    permutation = np.arange(n)[::-1]
    shifted = translate(sample, jnp.asarray([0.317, -0.229], dtype=dtype), box)
    permuted = sample[:, permutation, :]

    def observables(x):
        return (
            ensemble_pair_moments(x, box, basis),
            jnp.mean(
                angular_cosine_moments(x, box, orders, params.angular_neighbor_scale),
                axis=0,
            ),
            total_energy_per_configuration(x, box, params, regime["family"]),
        )

    reference_outputs = observables(sample)
    shifted_outputs = observables(shifted)
    permuted_outputs = observables(permuted)
    invariance_errors = {
        "translation_pair": _max_abs(reference_outputs[0], shifted_outputs[0]),
        "translation_angular": _max_abs(reference_outputs[1], shifted_outputs[1]),
        "translation_energy": _max_abs(reference_outputs[2], shifted_outputs[2]),
        "permutation_pair": _max_abs(reference_outputs[0], permuted_outputs[0]),
        "permutation_angular": _max_abs(reference_outputs[1], permuted_outputs[1]),
        "permutation_energy": _max_abs(reference_outputs[2], permuted_outputs[2]),
    }

    centered_pair = pair_moments - np.mean(pair_moments, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered_pair, compute_uv=False)
    rank_threshold = tolerances.covariance_rtol * max(float(singular_values[0]), 1.0)
    effective_rank = int(np.sum(singular_values > rank_threshold))

    regime_summaries = []
    for label in np.unique(labels):
        mask = labels == label
        regime_summaries.append(
            {
                "label": int(label),
                "name": regimes_by_label[int(label)]["name"],
                "num_samples": int(np.sum(mask)),
                "mean_pair_moments": np.mean(pair_moments[mask], axis=0).tolist(),
                "mean_angular_moments": np.mean(angular_moments[mask], axis=0).tolist(),
                "mean_energy": float(np.mean(arrays["energy_per_replica"][mask])),
                "mean_minimum_pair_distance": float(
                    np.mean(arrays["minimum_pair_distance"][mask])
                ),
                "mean_overlap_fraction": float(np.mean(arrays["overlap_fraction"][mask])),
            }
        )

    numerical_pass = (
        all(shape_checks.values())
        and np.all(np.isfinite(coordinates))
        and np.all(coordinates >= -tolerances.coordinate_atol)
        and np.all(coordinates < arrays["box"] + tolerances.coordinate_atol)
        and max(recomputation_errors.values()) <= tolerances.recomputation_atol
        and max(invariance_errors.values()) <= tolerances.invariance_atol
    )

    # This is a diagnostic, not an acceptance threshold for a smoke dataset.
    ambiguity = {
        "pair_centroid_distance_standardized": _standardized_centroid_distance(
            pair_moments, labels
        ),
        "angular_centroid_distance_standardized": _standardized_centroid_distance(
            angular_moments, labels
        ),
        "effective_pair_rank": effective_rank,
        "maximum_possible_centered_rank": min(s - 1, r),
        "enough_samples_for_full_pair_covariance": bool(s > r),
        "status": "calibration_required" if s <= r else "ready_for_matching_review",
    }

    return {
        "dataset": str(dataset_path),
        "metadata": str(metadata_path),
        "shape": {"S": s, "M": m, "N": n, "D": d, "R": r, "Q": q},
        "shape_checks": shape_checks,
        "finite": bool(all(np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind in "fiu")),
        "coordinate_range": {
            "minimum": float(np.min(coordinates)),
            "maximum": float(np.max(coordinates)),
            "box": arrays["box"].tolist(),
        },
        "recomputation_max_abs_errors": recomputation_errors,
        "invariance_max_abs_errors": invariance_errors,
        "pair_singular_values": singular_values.tolist(),
        "regime_summaries": regime_summaries,
        "ambiguity_diagnostics": ambiguity,
        "numerical_validation_passed": bool(numerical_pass),
    }
