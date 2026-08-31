"""Prospectively frozen Vortices V2 selection/search contract.

This module contains only deterministic geometry/search helpers and the one
allowed reflected Full-action evaluator.  It does not train references, create
observation banks, optimize a scientific design, or inspect validation data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jax
import numpy as np

from core import make_grid, rasterize_trajectory_v2, solve_v2
from mfsi.design import random_point_sensor_starts


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "VORTICES_V2_SELECTION_CONFIG.json"
EVALUATOR_IDENTITY = "vortices_v2_reflection_neumann_hard_projection"
FORBIDDEN_EVALUATOR_IDENTITIES = {
    "v1_hard_bin",
    "v1_shrinking_0p35_cell_bandwidth",
    "v2_source_column_normalized",
    "density_floor_scientific_action",
    "v1_native_regularized_action",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_selection_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f"{context} is incomplete; missing {missing}")


def validate_selection_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless every frozen search and inference field is present."""
    _require(
        config,
        (
            "reference_replicates", "observation_banks", "common_bandwidth",
            "scientific_evaluator", "risk_and_geometry", "optimization",
            "selection_objectives", "candidate_numerical_gates", "validation",
            "artifact_destinations",
        ),
        "selection config",
    )
    references = config["reference_replicates"]
    seeds = [int(value) for value in references["training_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct reference-training seeds are required")
    if set(seeds) & set(map(int, references["disallowed_historical_training_seeds"])):
        raise ValueError("a frozen reference seed is historical")
    rollout_seeds = list(map(int, references["rollout"]["seeds"]))
    if rollout_seeds != [seed + 3001 for seed in seeds]:
        raise ValueError("reference rollout seeds violate the frozen seed+3001 rule")

    banks = config["observation_banks"]
    if int(banks["selection_namespace"]) == int(banks["validation_namespace"]):
        raise ValueError("selection and validation namespaces must be distinct")
    if not banks["shared_across_all_references_and_methods"]:
        raise ValueError("observation banks must be shared across references and methods")
    if int(banks["selection_master_trials"]) != 128:
        raise ValueError("the master selection bank must contain exactly 128 trials")
    expected_prefixes = {
        "tangent_full_prescreen": [0, 32],
        "law_finite_risk": [0, 64],
        "tangent_full_final": [0, 128],
    }
    if banks["selection_prefixes"] != expected_prefixes:
        raise ValueError("selection prefixes do not match the frozen nesting")
    if int(banks["validation_trials"]) != 1024:
        raise ValueError("validation must contain exactly 1,024 shared trials")

    evaluator = config["scientific_evaluator"]
    _require(
        evaluator,
        (
            "identity", "projection", "scalar_raster", "source_raster",
            "reference_flux", "equation", "boundary_condition",
            "reflected_image_pairs", "exact_grid", "scientific_time_indices",
            "density_floor", "precision",
        ),
        "scientific evaluator",
    )
    if evaluator["identity"] != EVALUATOR_IDENTITY:
        raise ValueError("unexpected scientific evaluator identity")
    if any(value in str(evaluator) for value in FORBIDDEN_EVALUATOR_IDENTITIES):
        raise ValueError("a forbidden legacy evaluator identity is configured")
    if evaluator["exact_grid"] != [256, 128]:
        raise ValueError("scientific grid must be 256 x 128")
    if evaluator["scientific_time_indices"] != list(range(21)):
        raise ValueError("scientific evaluator must use all 21 time nodes")
    if float(evaluator["density_floor"]) != 0.0:
        raise ValueError("scientific density floor must be exactly zero")
    if evaluator["precision"] != "float64":
        raise ValueError("V2 scientific precision must be float64")
    if int(evaluator["reflected_image_pairs"]) != 4:
        raise ValueError("exactly four reflected image pairs are frozen")

    optimization = config["optimization"]
    _require(
        optimization,
        (
            "optimizer_root_seed", "generated_start_count",
            "start_oversampling_factor", "raw_start_pool_count",
            "retained_start_count", "common_adam", "population", "law",
            "tangent", "full", "full_search_proxy", "candidate_handling",
            "allowance_nesting", "checkpointing", "old_v1_proposals",
        ),
        "optimization",
    )
    if int(optimization["raw_start_pool_count"]) != (
        int(optimization["generated_start_count"])
        * int(optimization["start_oversampling_factor"])
    ):
        raise ValueError("raw start-pool count is inconsistent")
    for stage in ("population", "law", "tangent"):
        _require(
            optimization[stage],
            ("optimizer", "steps", "learning_rate"),
            f"optimization.{stage}",
        )
    _require(
        optimization["full"],
        (
            "optimizer", "adam_steps", "learning_rate", "global_generated_starts",
            "local_search_rounds", "local_proposals_per_center_per_round",
            "local_scales", "exact_risk_audit_candidates",
            "prescreen_trial_prefix", "promoted_candidates",
            "final_exact_trial_prefix", "final_rescores_per_promoted_candidate",
        ),
        "optimization.full",
    )
    proxy = optimization["full_search_proxy"]
    if proxy["evaluator_identity"] != "same_reflected_v2_action_as_scientific_evaluator":
        raise ValueError("Full search proxy is not the V2 reflected evaluator")
    if proxy["grid"] != [64, 32] or proxy["time_indices"] != list(range(21)):
        raise ValueError("Full proxy grid/time nodes differ from the frozen rule")
    if float(proxy["density_floor"]) != 0.0:
        raise ValueError("Full proxy density floor must be zero")
    if not proxy["fixed_common_physical_bandwidth"]:
        raise ValueError("Full proxy must use the frozen common physical bandwidth")

    validation = config["validation"]
    if validation["bootstrap_pairing"] != (
        "one_common_1024_index_vector_per_resample_for_all_three_references_law_and_all_six_allowances"
    ):
        raise ValueError("cross-reference bootstrap pairing is not frozen")
    if int(validation["bootstrap_resamples"]) != 100000:
        raise ValueError("bootstrap resample count must be 100,000")
    if int(validation["bootstrap_seed"]) != 821775:
        raise ValueError("bootstrap seed must be 821775")


def canonical_centers(eta: Sequence[float]) -> np.ndarray:
    centers = np.asarray(eta, dtype=np.float64).reshape((4, 2))
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def candidate_key(eta: Sequence[float]) -> tuple[float, ...]:
    return tuple(np.round(canonical_centers(eta).ravel(), 12))


def geometry_is_feasible(
    eta: Sequence[float],
    *,
    box: Sequence[Sequence[float]],
    minimum_separation: float,
    tolerance: float,
) -> bool:
    centers = canonical_centers(eta)
    x_bounds, y_bounds = np.asarray(box, dtype=np.float64)
    inside = (
        np.all(centers[:, 0] >= x_bounds[0] - tolerance)
        and np.all(centers[:, 0] <= x_bounds[1] + tolerance)
        and np.all(centers[:, 1] >= y_bounds[0] - tolerance)
        and np.all(centers[:, 1] <= y_bounds[1] + tolerance)
    )
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    distances[np.eye(4, dtype=bool)] = np.inf
    return bool(inside and np.min(distances) >= minimum_separation - tolerance)


def generated_starts(config: Mapping[str, Any]) -> np.ndarray:
    validate_selection_config(config)
    opt = config["optimization"]
    risk = config["risk_and_geometry"]
    box = risk["center_box"]
    return np.asarray(
        random_point_sensor_starts(
            jax.random.PRNGKey(int(opt["optimizer_root_seed"])),
            int(opt["generated_start_count"]),
            n_sensors=4,
            x_bounds=tuple(map(float, box[0])),
            y_bounds=tuple(map(float, box[1])),
            min_sep=float(risk["minimum_pairwise_separation"]),
            oversample=int(opt["start_oversampling_factor"]),
        ),
        dtype=np.float64,
    )


def observation_bank_identity(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    """Return the reference-independent bank identity frozen for a phase."""
    validate_selection_config(config)
    banks = config["observation_banks"]
    if phase == "selection":
        namespace = int(banks["selection_namespace"])
        trials = int(banks["selection_master_trials"])
    elif phase == "validation":
        namespace = int(banks["validation_namespace"])
        trials = int(banks["validation_trials"])
    else:
        raise ValueError("phase must be 'selection' or 'validation'")
    return {
        "phase": phase,
        "generation_seed": int(banks["generation_seed"]),
        "namespace": namespace,
        "trials": trials,
        "finite_particles": int(banks["finite_particles"]),
        "acquisition_indices": list(banks["acquisition_indices_on_21_node_grid"]),
        "observables": int(banks["observables"]),
        "shared_across_reference_seeds": list(
            map(int, config["reference_replicates"]["training_seeds"])
        ),
    }


def deterministic_local_cloud(
    centers: Sequence[Sequence[float]],
    *,
    count_per_center: int,
    scale: float,
    seed: int,
    box: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    """The frozen V1-style quadratic multiscale Gaussian local proposal rule."""
    rng = np.random.default_rng(int(seed))
    lo = np.asarray([box[0][0], box[1][0]], dtype=np.float64)
    hi = np.asarray([box[0][1], box[1][1]], dtype=np.float64)
    proposals: dict[tuple[float, ...], np.ndarray] = {}
    for eta in centers:
        base = canonical_centers(eta)
        for index in range(int(count_per_center)):
            fraction = (index + 1.0) / float(count_per_center)
            step = float(scale) * fraction * fraction
            candidate = np.clip(base + step * rng.standard_normal(base.shape), lo, hi)
            proposals[candidate_key(candidate.ravel())] = candidate.ravel()
    return list(proposals.values())


def normalized_trapezoid_weights(indices: Sequence[int]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 2 or np.any(np.diff(indices) <= 0):
        raise ValueError("time indices must be a strictly increasing vector")
    times = indices.astype(np.float64) / 20.0
    weights = np.zeros(len(times), dtype=np.float64)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    weights /= np.sum(weights)
    return weights


def evaluate_reflected_v2_action(
    state: Any,
    *,
    bandwidth: float,
    grid_shape: Sequence[int],
    time_indices: Sequence[int],
    image_pairs: int = 4,
) -> dict[str, Any]:
    """Evaluate only the frozen reflected V2 action on a prepared particle state."""
    if not np.isfinite(bandwidth) or float(bandwidth) <= 0.0:
        raise ValueError("a positive frozen common bandwidth is required")
    if int(image_pairs) != 4:
        raise ValueError("the frozen evaluator requires four reflected image pairs")
    nx, ny = map(int, grid_shape)
    grid = make_grid(nx, ny)
    raster = rasterize_trajectory_v2(
        state, grid, bandwidth=float(bandwidth), image_pairs=int(image_pairs)
    )
    solved = solve_v2(raster["q"], raster["source"], grid)
    indices = np.asarray(time_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(solved.action)):
        raise ValueError("time index outside the 21-node scientific trajectory")
    weights = normalized_trapezoid_weights(indices)
    action_by_time = np.asarray(solved.action, dtype=np.float64)
    return {
        "evaluator_identity": EVALUATOR_IDENTITY,
        "grid": [nx, ny],
        "time_indices": indices.tolist(),
        "action": float(np.sum(weights * action_by_time[indices])),
        "action_by_time": action_by_time.tolist(),
        "maximum_poisson_relative_residual": float(
            np.max(np.asarray(solved.relative_residual, dtype=np.float64)[indices])
        ),
        "strictly_positive_q": bool(np.all(raster["q"][indices] > 0.0)),
        "density_floor": 0.0,
    }


@dataclass(frozen=True)
class AuditedCandidate:
    eta: tuple[float, ...]
    objective: float
    valid_by_reference: tuple[bool, bool, bool]
    passes_population: bool
    passes_finite_risk: bool
    provenance: str

    @property
    def fully_feasible(self) -> bool:
        return bool(
            all(self.valid_by_reference)
            and self.passes_population
            and self.passes_finite_risk
            and np.isfinite(self.objective)
        )


def choose_nested_winner(
    incumbent: AuditedCandidate,
    candidates: Sequence[AuditedCandidate],
    *,
    tie_tolerance: float = 1e-6,
) -> AuditedCandidate:
    """Apply the frozen mandatory-incumbent and strict replacement rule."""
    if not incumbent.fully_feasible:
        raise ValueError("mandatory incumbent failed its current exact audit")
    winner = incumbent
    for candidate in candidates:
        if candidate.fully_feasible and candidate.objective < winner.objective - tie_tolerance:
            winner = candidate
    return winner


validate_selection_config(load_selection_config())
