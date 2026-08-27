"""Efficient development-only replicate-gate architecture preflight.

This module consumes only sealed Boolean eligibility results.  It contains no
candidate/bank generator, information projection, risk calculation, optimizer,
Galerkin assembly, eigensolve, Deep Ritz solver, validation loader, or official
protocol writer.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .pareto_v3_common import ROOT, file_sha256
from .pareto_v3_diagnostic import _symmetry_aware_distance


VERSION = "skyrmion_galerkin_dev_replicate_gate_preflight_v2"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
SOURCE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_fresh_bank_robustness_v1"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
ARCHITECTURE_GRID_PATH = OUTPUT_ROOT / "architecture_grid.json"
SUBSET_MANIFEST_PATH = OUTPUT_ROOT / "subset_manifest.json"
EXACT_PATH = OUTPUT_ROOT / "exact_hypergeometric_results.json"
EXACT_ARRAY_PATH = OUTPUT_ROOT / "candidate_exact_probabilities.npz"
RESAMPLING_PATH = OUTPUT_ROOT / "resampling_results.json"
DIVERSITY_PATH = OUTPUT_ROOT / "diversity_diagnostics.json"
HARD_BANK_PATH = OUTPUT_ROOT / "hard_bank_diagnostics.json"
RECOMMENDATION_PATH = OUTPUT_ROOT / "recommended_official_gate.json"
NOTES_PATH = OUTPUT_ROOT / "protocol_design_notes.md"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

CANDIDATE_COUNT = 4433
PAIR_COUNT = 32
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
PER_BANK_RESS = 0.05
SCREEN_N = 8192
AUDIT_N = 16384
GLOBAL_SEED = 20260825
SUBSET_NAMESPACE = VERSION + ":subset_schedule_v1"
SCORE_CHUNK = 1024

EXPECTED_SOURCE_HASHES = {
    "candidate_robustness_summary.json": "99121aed7b18d70128cee7cdcc9d4d61dfd64e98d68d25ffb9131164c1a0db77",
    "allowance_summary.json": "e8f21ec7f78a4d1996505c6bed9e85e5b2f7b2ef57752b9af401388d63231a0f",
    "failure_mode_summary.json": "91281495a59de92c7a5cd3bb7a1ba8b540860ddda5fe0537b5ffc88595b7205e",
    "time_node_summary.json": "32ac42f574c054d07ae198c21c2fdd7af7a16364487e6ab900fe8b1708e18d53",
    "summary.json": "998a59b5bdb195e15379be085b60d31c5d5c6edd8edf63f78a64ea351f1e8740",
    "inventory.json": "3d837c2ea4108283749bdfa1d661e0a90ebd5ad39693ea942641c23f94df466e",
    "candidate_freeze.json": "3fae7f1cc7479d0d5413f89838aba9b0ccd8d24374dd27de699d780e5a3e1f4d",
    "bank_manifest.json": "9e2d30fa15ed29c27e415b032ce9bd4a7b4c673bc9dda1891cc8c8f7201845d3",
    "bank_inventory.json": "e5dd4f14e84b1cbc8e74af1a477a415490b98258bc0d7e4332464737ac02338d",
}

ARCHITECTURES = (
    {"architecture_id": "A1", "label": "1/1", "required_passes": 1, "M": 1, "reference_passes": 32},
    {"architecture_id": "A2", "label": "2/2", "required_passes": 2, "M": 2, "reference_passes": 32},
    {"architecture_id": "A3", "label": "3/4", "required_passes": 3, "M": 4, "reference_passes": 24},
    {"architecture_id": "A4", "label": "4/4", "required_passes": 4, "M": 4, "reference_passes": 32},
    {"architecture_id": "A5", "label": "6/8", "required_passes": 6, "M": 8, "reference_passes": 24},
    {"architecture_id": "A6", "label": "7/8", "required_passes": 7, "M": 8, "reference_passes": 28},
    {"architecture_id": "A7", "label": "8/8", "required_passes": 8, "M": 8, "reference_passes": 32},
    {"architecture_id": "A8", "label": "12/16", "required_passes": 12, "M": 16, "reference_passes": 24},
    {"architecture_id": "A9", "label": "14/16", "required_passes": 14, "M": 16, "reference_passes": 28},
    {"architecture_id": "A10", "label": "16/16", "required_passes": 16, "M": 16, "reference_passes": 32},
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False
    ).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_code_hashes() -> dict[str, str]:
    names = (
        "replicate_gate_preflight_v2.py",
        "replicate_gate_preflight_v2_run.py",
        "test_replicate_gate_preflight_v2.py",
    )
    return {name: file_sha256(ROOT / name) for name in names}


def verify_and_freeze_sources() -> dict[str, Any]:
    actual = {name: file_sha256(SOURCE_ROOT / name) for name in EXPECTED_SOURCE_HASHES}
    if actual != EXPECTED_SOURCE_HASHES:
        changed = [name for name in actual if actual[name] != EXPECTED_SOURCE_HASHES[name]]
        raise RuntimeError("fresh-bank source seal mismatch: " + ", ".join(changed))

    summary = json.loads((SOURCE_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary["candidate_count"] != CANDIDATE_COUNT or summary["replicate_count"] != PAIR_COUNT:
        raise RuntimeError("fresh-bank dimensions changed")
    if summary["validation_accessed"] or summary["full_kf_constructed"]:
        raise RuntimeError("fresh-bank firewall flags changed")

    cache_rows = {int(row["replicate_id"]): row for row in summary["cache_resume"]["completed"]}
    verified_stage_artifacts = []
    for replicate in range(PAIR_COUNT):
        root = SOURCE_ROOT / "replicates" / f"replicate_{replicate:02d}"
        inventory_path = root / "replicate_inventory.json"
        if file_sha256(inventory_path) != cache_rows[replicate]["replicate_inventory_sha256"]:
            raise RuntimeError(f"replicate inventory changed: {replicate}")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for row in inventory["artifacts"]:
            path = SOURCE_ROOT / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"sealed replicate result changed: {row['path']}")
            verified_stage_artifacts.append(row["path"])

    payload = {
        "schema_version": 1,
        "development_only": True,
        "source_version": "skyrmion_galerkin_dev_fresh_bank_robustness_v1",
        "source_hashes": actual,
        "analysis_source_hashes": _source_code_hashes(),
        "candidate_count": CANDIDATE_COUNT,
        "fresh_development_pairs": PAIR_COUNT,
        "verified_replicate_stage_artifact_count": len(verified_stage_artifacts),
        "new_scientific_evaluations": 0,
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def freeze_architecture_grid() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "development_only": True,
        "architectures": list(ARCHITECTURES),
        "primary_comparison": ["3/4", "6/8", "12/16"],
        "strictness_comparison": [["6/8", "7/8"], ["12/16", "14/16"]],
        "same_gate_all_allowances": True,
        "per_bank_ress_threshold": PER_BANK_RESS,
        "allowances_percent": list(ALLOWANCES),
        "new_architectures_after_scoring_permitted": False,
    }
    _atomic_json(ARCHITECTURE_GRID_PATH, payload)
    return payload


def _mask(indices: Iterable[int]) -> int:
    value = 0
    for index in indices:
        value |= 1 << int(index)
    return value


def _sample_unique_subsets(M: int, count: int) -> tuple[list[int], dict[str, Any]]:
    digest = hashlib.sha256(f"{GLOBAL_SEED}:{SUBSET_NAMESPACE}:M={M}".encode()).hexdigest()
    seed = int(digest[:16], 16) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    masks: list[int] = []
    seen: set[int] = set()
    while len(masks) < count:
        candidate = _mask(np.sort(rng.choice(PAIR_COUNT, size=M, replace=False)))
        if candidate not in seen:
            seen.add(candidate)
            masks.append(candidate)
    return masks, {"seed": seed, "seed_sha256": digest}


def freeze_subset_manifest() -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    for M in (1, 2, 4):
        masks = [_mask(combo) for combo in itertools.combinations(range(PAIR_COUNT), M)]
        schedules[str(M)] = {
            "M": M,
            "method": "complete_enumeration",
            "count": len(masks),
            "subset_masks_uint32": masks,
        }
    for M, count in ((8, 5000), (16, 2500)):
        masks, seed = _sample_unique_subsets(M, count)
        schedules[str(M)] = {
            "M": M,
            "method": "unique_uniform_rng_draws",
            "count": len(masks),
            "rng_namespace": SUBSET_NAMESPACE,
            **seed,
            "subset_masks_uint32": masks,
        }
    payload = {
        "schema_version": 1,
        "development_only": True,
        "pair_count": PAIR_COUNT,
        "same_schedule_reused_across_allowances_and_shared_M": True,
        "jaccard_schedule": "first min(1000, subset_count) frozen entries",
        "diversity_schedule": "first min(200, subset_count) frozen entries",
        "schedules": schedules,
    }
    _atomic_json(SUBSET_MANIFEST_PATH, payload)
    return payload


def freeze_design() -> dict[str, Any]:
    source = verify_and_freeze_sources()
    grid = freeze_architecture_grid()
    subsets = freeze_subset_manifest()
    return {
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "architecture_grid_sha256": file_sha256(ARCHITECTURE_GRID_PATH),
        "subset_manifest_sha256": file_sha256(SUBSET_MANIFEST_PATH),
        "source": source,
        "grid": grid,
        "subsets": subsets,
    }


def reconstruct_eligibility() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    eligible = np.empty((len(ALLOWANCES), CANDIDATE_COUNT, PAIR_COUNT), dtype=bool)
    for replicate in range(PAIR_COUNT):
        path = SOURCE_ROOT / "replicates" / f"replicate_{replicate:02d}" / "audit_results.npz"
        with np.load(path, allow_pickle=False) as arrays:
            dual = np.asarray(arrays["dual_bank_eligible"], dtype=bool)
        if dual.shape != (len(ALLOWANCES), CANDIDATE_COUNT):
            raise RuntimeError(f"unexpected eligibility shape for replicate {replicate}: {dual.shape}")
        eligible[:, :, replicate] = dual

    pass_counts = np.sum(eligible, axis=2, dtype=np.int16)
    weights = np.left_shift(np.uint32(1), np.arange(PAIR_COUNT, dtype=np.uint32))
    pass_masks = np.sum(eligible.astype(np.uint32) * weights[None, None, :], axis=2, dtype=np.uint32)
    expected_best = np.asarray([26, 29, 31, 31, 32, 32])
    expected_ge24 = np.asarray([55, 310, 957, 1314, 1663, 1886])
    if not np.array_equal(np.max(pass_counts, axis=1), expected_best):
        raise RuntimeError("reconstructed best pass counts do not match sealed summary")
    if not np.array_equal(np.sum(pass_counts >= 24, axis=1), expected_ge24):
        raise RuntimeError("reconstructed >=24/32 counts do not match sealed summary")

    freeze = json.loads((SOURCE_ROOT / "candidate_freeze.json").read_text(encoding="utf-8"))
    rows = freeze["rows"]
    if len(rows) != CANDIDATE_COUNT:
        raise RuntimeError("candidate freeze row count changed")
    return pass_masks, pass_counts, rows


def popcount_uint32(values: np.ndarray) -> np.ndarray:
    """Vectorized SWAR population count for uint32 arrays."""
    x = np.asarray(values, dtype=np.uint32).copy()
    x -= (x >> np.uint32(1)) & np.uint32(0x55555555)
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((x * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.uint8)


def exact_hypergeometric_probability(successes: int, M: int, required: int) -> float:
    denominator = math.comb(PAIR_COUNT, M)
    numerator = sum(
        math.comb(successes, hits) * math.comb(PAIR_COUNT - successes, M - hits)
        for hits in range(required, min(M, successes) + 1)
        if 0 <= M - hits <= PAIR_COUNT - successes
    )
    return float(numerator / denominator)


def _probability_lookup() -> np.ndarray:
    return np.asarray(
        [
            [exact_hypergeometric_probability(s, row["M"], row["required_passes"]) for s in range(33)]
            for row in ARCHITECTURES
        ],
        dtype=np.float64,
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    return {
        "count": int(values.size),
        "minimum": int(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": int(np.max(values)),
    }


def _mask_histogram(candidate_masks: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(candidate_masks, dtype=np.uint32), return_inverse=True)
    weights = np.bincount(inverse, minlength=len(unique)).astype(np.int32)
    reference_weights = np.bincount(
        inverse, weights=np.asarray(reference, dtype=np.int32), minlength=len(unique)
    ).astype(np.int32)
    return unique, weights, reference_weights


def score_subsets(
    candidate_masks: np.ndarray,
    subset_masks: np.ndarray,
    required: int,
    reference: np.ndarray,
    jaccard_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score subsets with mask histograms; no candidate/subset Python loops."""
    unique, weights, reference_weights = _mask_histogram(candidate_masks, reference)
    survivors = np.empty(len(subset_masks), dtype=np.int32)
    jaccard = np.empty(jaccard_count, dtype=np.float64)
    reference_count = int(np.sum(reference))
    for start in range(0, len(subset_masks), SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, len(subset_masks))
        intersections = unique[:, None] & subset_masks[None, start:stop]
        selected = popcount_uint32(intersections) >= int(required)
        survivors[start:stop] = weights @ selected
        overlap_stop = min(stop, jaccard_count)
        if start < overlap_stop:
            local = selected[:, : overlap_stop - start]
            intersection_count = reference_weights @ local
            union_count = survivors[start:overlap_stop] + reference_count - intersection_count
            jaccard[start:overlap_stop] = np.divide(
                intersection_count,
                union_count,
                out=np.ones_like(intersection_count, dtype=np.float64),
                where=union_count != 0,
            )
    return survivors, jaccard


def _exact_results(pass_counts: np.ndarray, pass_masks: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = _probability_lookup()
    probabilities = lookup[:, pass_counts]
    aggregate = []
    thresholds = (0.50, 0.75, 0.90, 0.95, 0.99)
    for arch_index, architecture in enumerate(ARCHITECTURES):
        for allowance_index, allowance in enumerate(ALLOWANCES):
            values = probabilities[arch_index, allowance_index]
            reference = pass_counts[allowance_index] >= int(architecture["reference_passes"])
            true_positive = float(np.sum(values[reference]))
            false_positive = float(np.sum(values[~reference]))
            false_negative = float(np.sum(1.0 - values[reference]))
            true_negative = float(np.sum(1.0 - values[~reference]))
            aggregate.append(
                {
                    "architecture": architecture["label"],
                    "allowance_percent": allowance,
                    "expected_survivors": float(np.sum(values)),
                    "candidate_probability_threshold_counts": {
                        str(level): int(np.sum(values >= level)) for level in thresholds
                    },
                    "reference_passes_out_of_32": int(architecture["reference_passes"]),
                    "expected_classification_counts": {
                        "true_positive": true_positive,
                        "false_positive": false_positive,
                        "false_negative": false_negative,
                        "true_negative": true_negative,
                        "precision_like": true_positive / (true_positive + false_positive)
                        if true_positive + false_positive else 1.0,
                        "recall_like": true_positive / (true_positive + false_negative)
                        if true_positive + false_negative else 1.0,
                    },
                }
            )
    _atomic_npz(
        EXACT_ARRAY_PATH,
        candidate_id=np.asarray([row["candidate_id"] for row in rows]),
        allowance_percent=np.asarray(ALLOWANCES),
        architecture_label=np.asarray([row["label"] for row in ARCHITECTURES]),
        pass_counts=pass_counts,
        pass_masks=pass_masks,
        exact_pass_probability=probabilities,
    )
    result = {
        "schema_version": 1,
        "authoritative_candidate_level_method": "exact hypergeometric tail",
        "candidate_count": CANDIDATE_COUNT,
        "candidate_array_path": EXACT_ARRAY_PATH.name,
        "candidate_array_sha256": file_sha256(EXACT_ARRAY_PATH),
        "aggregate": aggregate,
    }
    _atomic_json(EXACT_PATH, result)
    return result


def _resampling_results(pass_masks: np.ndarray, pass_counts: np.ndarray, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, float], np.ndarray]]:
    rows = []
    survivor_cache: dict[tuple[str, float], np.ndarray] = {}
    for architecture in ARCHITECTURES:
        subset_masks = np.asarray(
            manifest["schedules"][str(architecture["M"])]["subset_masks_uint32"],
            dtype=np.uint32,
        )
        jaccard_count = min(1000, len(subset_masks))
        for allowance_index, allowance in enumerate(ALLOWANCES):
            reference = pass_counts[allowance_index] >= int(architecture["reference_passes"])
            survivors, jaccard = score_subsets(
                pass_masks[allowance_index], subset_masks,
                int(architecture["required_passes"]), reference, jaccard_count
            )
            survivor_cache[(architecture["label"], allowance)] = survivors
            rows.append(
                {
                    "architecture": architecture["label"],
                    "allowance_percent": allowance,
                    "subset_count": len(subset_masks),
                    "feasible_set_probabilities": {
                        "zero": float(np.mean(survivors == 0)),
                        "ge_1": float(np.mean(survivors >= 1)),
                        "ge_5": float(np.mean(survivors >= 5)),
                        "ge_10": float(np.mean(survivors >= 10)),
                        "ge_25": float(np.mean(survivors >= 25)),
                    },
                    "survivor_distribution": _distribution(survivors),
                    "same_fraction_full32_jaccard": {
                        "reference_passes": int(architecture["reference_passes"]),
                        "subset_count": jaccard_count,
                        "p10": float(np.quantile(jaccard, 0.10)),
                        "median": float(np.median(jaccard)),
                        "p90": float(np.quantile(jaccard, 0.90)),
                    },
                }
            )
    result = {"schema_version": 1, "method": "uint32 bitmask histogram popcount", "rows": rows}
    _atomic_json(RESAMPLING_PATH, result)
    return result, survivor_cache


_PERMUTATIONS = np.asarray(list(itertools.permutations(range(4))), dtype=np.int8)


def _vectorized_symmetry_distances(etas: np.ndarray, reference: np.ndarray, box: np.ndarray) -> np.ndarray:
    centers = np.asarray(etas, dtype=np.float64).reshape((-1, 4, 2))
    ref = np.asarray(reference, dtype=np.float64).reshape((4, 2))
    permuted = centers[:, _PERMUTATIONS, :]
    delta = permuted - ref[None, None, :, :]
    delta -= np.round(delta / box[None, None, None, :]) * box[None, None, None, :]
    return np.sqrt(np.min(np.sum(delta * delta, axis=(2, 3)), axis=1))


def _representative_maxmin(
    indices: np.ndarray,
    etas: np.ndarray,
    candidate_ids: np.ndarray,
    full_counts: np.ndarray,
    box: np.ndarray,
    maximum: int = 10,
) -> tuple[list[int], float]:
    if not len(indices):
        return [], 0.0
    order = np.lexsort((candidate_ids[indices], -full_counts[indices]))
    remaining = indices[order]
    selected = [int(remaining[0])]
    active = np.ones(len(remaining), dtype=bool)
    active[0] = False
    minimum_distance = _vectorized_symmetry_distances(etas[remaining], etas[selected[0]], box)
    while np.any(active) and len(selected) < min(maximum, len(indices)):
        scores = np.where(active, minimum_distance, -np.inf)
        best_distance = float(np.max(scores))
        tied = np.flatnonzero(np.isclose(scores, best_distance, rtol=0.0, atol=1e-15))
        if len(tied) > 1:
            tied_indices = remaining[tied]
            tie_order = np.lexsort((candidate_ids[tied_indices], -full_counts[tied_indices]))
            chosen_position = int(tied[tie_order[0]])
        else:
            chosen_position = int(tied[0])
        chosen = int(remaining[chosen_position])
        selected.append(chosen)
        active[chosen_position] = False
        distance = _vectorized_symmetry_distances(etas[remaining], etas[chosen], box)
        minimum_distance = np.minimum(minimum_distance, distance)

    pairwise = [
        _symmetry_aware_distance(etas[left], etas[right], box)
        for position, left in enumerate(selected)
        for right in selected[position + 1 :]
    ]
    return selected, min(pairwise) if pairwise else 0.0


def _diversity_results(
    pass_masks: np.ndarray,
    pass_counts: np.ndarray,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    survivor_cache: dict[tuple[str, float], np.ndarray],
) -> dict[str, Any]:
    selected_architectures = {"3/4", "6/8", "7/8", "12/16", "14/16"}
    etas = np.asarray([row["eta"] for row in rows], dtype=np.float64)
    candidate_ids = np.asarray([row["candidate_id"] for row in rows])
    box = np.asarray([2.0, 1.0], dtype=np.float64)
    output = []
    for architecture in ARCHITECTURES:
        if architecture["label"] not in selected_architectures:
            continue
        subset_masks = np.asarray(
            manifest["schedules"][str(architecture["M"])]["subset_masks_uint32"][:200],
            dtype=np.uint32,
        )
        for allowance_index, allowance in enumerate(ALLOWANCES[:4]):
            survivors = survivor_cache[(architecture["label"], allowance)][: len(subset_masks)]
            representative_position = int(np.argmax(survivors))
            counts = popcount_uint32(
                pass_masks[allowance_index] & subset_masks[representative_position]
            )
            eligible_indices = np.flatnonzero(counts >= int(architecture["required_passes"]))
            shortlist, minimum_distance = _representative_maxmin(
                eligible_indices, etas, candidate_ids, pass_counts[allowance_index], box
            )
            output.append(
                {
                    "architecture": architecture["label"],
                    "allowance_percent": allowance,
                    "diagnostic_subset_count": len(subset_masks),
                    "fraction_supporting_ge_3_diverse_starts": float(np.mean(survivors >= 3)),
                    "fraction_supporting_ge_5_diverse_starts": float(np.mean(survivors >= 5)),
                    "fraction_supporting_ge_10_diverse_starts": float(np.mean(survivors >= 10)),
                    "representative_subset_position": representative_position,
                    "representative_maxmin_candidate_ids": [candidate_ids[index] for index in shortlist],
                    "representative_minimum_pairwise_symmetry_distance": minimum_distance,
                }
            )
    result = {
        "schema_version": 1,
        "metric": "existing periodic/permutation-aware sensor distance",
        "availability_definition": "a deterministic max-min list has size min(10, eligible canonical geometries)",
        "rows": output,
    }
    _atomic_json(DIVERSITY_PATH, result)
    return result


def _hard_bank_results(
    manifest: dict[str, Any], survivor_cache: dict[tuple[str, float], np.ndarray]
) -> dict[str, Any]:
    singleton = survivor_cache[("1/1", 0.5)]
    hard_ids = np.flatnonzero(singleton == 0)
    if len(hard_ids) != 4:
        raise RuntimeError(f"expected four hard 0.5% banks, found {len(hard_ids)}")
    hard_mask = np.uint32(_mask(hard_ids))
    chosen = {"3/4", "6/8", "7/8", "12/16", "14/16"}
    rows = []
    for architecture in ARCHITECTURES:
        if architecture["label"] not in chosen:
            continue
        subset_masks = np.asarray(
            manifest["schedules"][str(architecture["M"])]["subset_masks_uint32"], dtype=np.uint32
        )
        hard_counts = popcount_uint32(subset_masks & hard_mask)
        survivors = survivor_cache[(architecture["label"], 0.5)]
        for label, selected in (
            ("0", hard_counts == 0),
            ("1", hard_counts == 1),
            ("2", hard_counts == 2),
            (">=3", hard_counts >= 3),
        ):
            values = survivors[selected]
            rows.append(
                {
                    "architecture": architecture["label"],
                    "hard_bank_group": label,
                    "subset_count": int(len(values)),
                    "median_survivors": float(np.median(values)) if len(values) else None,
                    "p10_survivors": float(np.quantile(values, 0.10)) if len(values) else None,
                    "empty_set_rate": float(np.mean(values == 0)) if len(values) else None,
                }
            )
    result = {"schema_version": 1, "hard_0p5_bank_ids": hard_ids.tolist(), "rows": rows}
    _atomic_json(HARD_BANK_PATH, result)
    return result


def _rows_by_key(payload: dict[str, Any]) -> dict[tuple[str, float], dict[str, Any]]:
    return {(row["architecture"], float(row["allowance_percent"])): row for row in payload["rows"]}


def _recommend(
    exact: dict[str, Any], resampling: dict[str, Any], diversity: dict[str, Any]
) -> dict[str, Any]:
    exact_rows = {(row["architecture"], float(row["allowance_percent"])): row for row in exact["aggregate"]}
    sample_rows = _rows_by_key(resampling)
    diverse_rows = _rows_by_key(diversity)
    evidence = {}
    for label in ("3/4", "6/8", "12/16"):
        half = sample_rows[(label, 0.5)]
        evidence[label] = {
            "0p5_empty_set_rate": half["feasible_set_probabilities"]["zero"],
            "0p5_p10_survivors": half["survivor_distribution"]["p10"],
            "0p5_median_survivors": half["survivor_distribution"]["median"],
            "0p5_median_jaccard": half["same_fraction_full32_jaccard"]["median"],
            "0p5_expected_survivors": exact_rows[(label, 0.5)]["expected_survivors"],
            "0p5_fraction_ge10_diverse": diverse_rows[(label, 0.5)]["fraction_supporting_ge_10_diverse_starts"],
            "relative_pair_cost": next(row["M"] for row in ARCHITECTURES if row["label"] == label),
        }

    ready = {
        label: values["0p5_empty_set_rate"] <= 0.05
        and values["0p5_fraction_ge10_diverse"] >= 0.90
        and values["0p5_median_jaccard"] >= 0.25
        for label, values in evidence.items()
    }
    if ready["6/8"]:
        gain_6_over_3 = (
            evidence["6/8"]["0p5_median_jaccard"] - evidence["3/4"]["0p5_median_jaccard"]
        )
        gain_12_over_6 = (
            evidence["12/16"]["0p5_median_jaccard"] - evidence["6/8"]["0p5_median_jaccard"]
        )
        empty_gain_12 = evidence["6/8"]["0p5_empty_set_rate"] - evidence["12/16"]["0p5_empty_set_rate"]
        if ready["12/16"] and gain_12_over_6 > max(0.10, gain_6_over_3) and empty_gain_12 > 0.02:
            recommendation = "RECOMMEND_12_OF_16"
            M, required = 16, 12
            reason = "12/16 materially improves both set agreement and empty-set stability beyond 6/8."
        else:
            recommendation = "RECOMMEND_6_OF_8"
            M, required = 8, 6
            reason = "6/8 is ready at low allowances; 12/16 does not add enough stability to justify twice the pair cost."
    elif ready["3/4"]:
        recommendation = "RECOMMEND_3_OF_4"
        M, required = 4, 3
        reason = "3/4 meets the frozen readiness checks and the larger 75% gates do not provide necessary added stability."
    elif ready["12/16"]:
        recommendation = "RECOMMEND_12_OF_16"
        M, required = 16, 12
        reason = "Only 12/16 among the primary 75% architectures meets the frozen readiness checks."
    else:
        recommendation = "NO_REPLICATE_GATE_ARCHITECTURE_READY"
        M, required = 0, 0
        reason = "None of the frozen primary architectures meets the multi-metric readiness checks."

    result = {
        "schema_version": 1,
        "development_only": True,
        "recommendation": recommendation,
        "recommended_M": M,
        "recommended_required_passes": required,
        "recommended_fraction": required / M if M else None,
        "per_bank_ress_threshold": PER_BANK_RESS,
        "screen_N": SCREEN_N,
        "audit_N": AUDIT_N,
        "allowances": list(ALLOWANCES),
        "same_gate_all_allowances": True,
        "allowance_failure_independent": True,
        "candidate_pool_source_count": CANDIDATE_COUNT,
        "development_bank_pairs_source": PAIR_COUNT,
        "principal_evidence": evidence,
        "reason": reason,
        "official_protocol_created": False,
        "official_banks_created": False,
        "selection_frozen": False,
        "validation_accessed": False,
    }
    _atomic_json(RECOMMENDATION_PATH, result)
    return result


def _write_notes(recommendation: dict[str, Any]) -> None:
    text = f"""# Next Official Galerkin Partial-Curve Protocol Design Notes

This is a development-only handoff. No official protocol or bank was created by the preflight.

## Recommended replicate gate

- Recommendation: `{recommendation['recommendation']}`
- Fresh official replicate pairs: `{recommendation['recommended_M']}`
- Required complete pair passes: `{recommendation['recommended_required_passes']}`
- Per-bank rESS threshold: `0.05` unchanged
- Per pair: screen `N=8192`, independent audit `N=16384`
- Same replicate gate for every allowance
- Allowances independently succeed or fail; 0.5% failure does not abort 1--5%

## Required next-stage ordering

1. Create a separately named official Galerkin partial-curve protocol.
2. Prospectively freeze the improved candidate-generation strategy.
3. Freeze all fresh official bank-pair seeds before candidate evaluation.
4. Apply robust-start eligibility using the recommended replicate gate.
5. Run Tangent and Full as independent branches.
6. Run `K=280` Full only after cheap eligibility.
7. Perform authoritative finalist certification.
8. Freeze the complete selection before generating or accessing fresh validation.
"""
    encoded = text.encode()
    if NOTES_PATH.exists():
        if NOTES_PATH.read_bytes() != encoded:
            raise RuntimeError("refusing to overwrite sealed protocol design notes")
    else:
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTES_PATH.write_bytes(encoded)


def _build_summary(
    exact: dict[str, Any],
    resampling: dict[str, Any],
    diversity: dict[str, Any],
    hard: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    exact_rows = {(row["architecture"], float(row["allowance_percent"])): row for row in exact["aggregate"]}
    sample_rows = _rows_by_key(resampling)
    diversity_rows = _rows_by_key(diversity)
    primary = {}
    for label in ("3/4", "6/8", "12/16"):
        primary[label] = {}
        for allowance in ALLOWANCES[:4]:
            row = sample_rows[(label, allowance)]
            primary[label][str(allowance)] = {
                "empty_set_rate": row["feasible_set_probabilities"]["zero"],
                "p10_survivors": row["survivor_distribution"]["p10"],
                "median_survivors": row["survivor_distribution"]["median"],
                "median_jaccard": row["same_fraction_full32_jaccard"]["median"],
                "expected_survivors": exact_rows[(label, allowance)]["expected_survivors"],
                "fraction_ge10_diverse": diversity_rows[(label, allowance)]["fraction_supporting_ge_10_diverse_starts"],
            }
        primary[label]["relative_pair_cost"] = int(label.split("/")[1])
    strict = {
        label: {
            str(allowance): {
                "empty_set_rate": sample_rows[(label, allowance)]["feasible_set_probabilities"]["zero"],
                "p10_survivors": sample_rows[(label, allowance)]["survivor_distribution"]["p10"],
                "median_survivors": sample_rows[(label, allowance)]["survivor_distribution"]["median"],
                "expected_survivors": exact_rows[(label, allowance)]["expected_survivors"],
            }
            for allowance in ALLOWANCES[:4]
        }
        for label in ("6/8", "7/8", "12/16", "14/16")
    }
    return {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_verified": True,
        "candidate_count": CANDIDATE_COUNT,
        "fresh_development_pairs": PAIR_COUNT,
        "new_scientific_evaluations": 0,
        "primary_75_percent_family": primary,
        "strictness_comparison": strict,
        "hard_0p5_bank_ids": hard["hard_0p5_bank_ids"],
        "recommendation": recommendation,
        "firewalls": {
            "candidate_generation": False,
            "bank_generation": False,
            "information_projection": False,
            "scientific_risk_recomputation": False,
            "tangent": False,
            "full": False,
            "eigensolve": False,
            "deep_ritz": False,
            "validation": False,
            "official_protocol_created": False,
        },
        "seals": {
            "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
            "architecture_grid_sha256": file_sha256(ARCHITECTURE_GRID_PATH),
            "subset_manifest_sha256": file_sha256(SUBSET_MANIFEST_PATH),
        },
    }


def _write_inventory() -> dict[str, Any]:
    paths = [
        SOURCE_SEAL_PATH,
        ARCHITECTURE_GRID_PATH,
        SUBSET_MANIFEST_PATH,
        EXACT_ARRAY_PATH,
        EXACT_PATH,
        RESAMPLING_PATH,
        DIVERSITY_PATH,
        HARD_BANK_PATH,
        RECOMMENDATION_PATH,
        NOTES_PATH,
        SUMMARY_PATH,
    ]
    payload = {
        "schema_version": 1,
        "artifact_count": len(paths),
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
        ],
    }
    _atomic_json(INVENTORY_PATH, payload)
    return payload


def _verify_cached_analysis() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists() and not INVENTORY_PATH.exists():
        return None
    if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
        raise RuntimeError("incomplete sealed preflight analysis")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"sealed preflight artifact changed: {row['path']}")
    result = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    result["cache_hit"] = True
    return result


def analyze() -> dict[str, Any]:
    freeze_design()
    cached = _verify_cached_analysis()
    if cached is not None:
        return cached
    pass_masks, pass_counts, rows = reconstruct_eligibility()
    manifest = json.loads(SUBSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    exact = _exact_results(pass_counts, pass_masks, rows)
    resampling, survivor_cache = _resampling_results(pass_masks, pass_counts, manifest)
    diversity = _diversity_results(pass_masks, pass_counts, rows, manifest, survivor_cache)
    hard = _hard_bank_results(manifest, survivor_cache)
    recommendation = _recommend(exact, resampling, diversity)
    _write_notes(recommendation)
    summary = _build_summary(exact, resampling, diversity, hard, recommendation)
    _atomic_json(SUMMARY_PATH, summary)
    _write_inventory()
    summary["cache_hit"] = False
    return summary


__all__ = [
    "ALLOWANCES",
    "ARCHITECTURES",
    "CANDIDATE_COUNT",
    "PAIR_COUNT",
    "PER_BANK_RESS",
    "SUBSET_MANIFEST_PATH",
    "SUMMARY_PATH",
    "analyze",
    "exact_hypergeometric_probability",
    "freeze_design",
    "popcount_uint32",
    "reconstruct_eligibility",
    "score_subsets",
]
