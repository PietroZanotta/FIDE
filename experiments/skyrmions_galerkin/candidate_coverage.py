"""Development-only candidate-coverage and risk/rESS geometry study.

This module may read the frozen v2 selection-side banks and completed v3
diagnostics.  It deliberately has no validation, Tangent optimizer, Full
Galerkin, eigensolve, or official-protocol entry point.
"""

from __future__ import annotations

import csv
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc, spearmanr

from mfsi.projection import EmpiricalIProjector

from .full_gradient import reconstruct_moments
from .galerkin_only_data import selection_risk
from .pareto_v2_common import require_protocol as require_v2_protocol
from .pareto_v2_selection import _screen_rows_batched, load_bank, selection_data
from .pareto_v3_common import (
    ALLOWANCES,
    ALL_ALLOWANCES_DIAGNOSTIC_ROOT,
    MINIMUM_RESS,
    ROOT,
    V2_OUTPUT_ROOT,
    eta_key,
    file_sha256,
    payload_sha256,
    read_json,
    selection_ceiling,
    verify_v2_frozen,
    verify_v3_phase1_frozen,
)
from .pareto_v3_diagnostic import (
    _official_v3_firewall,
    _symmetry_aware_distance,
)
from .risk import integrated_risk


VERSION = "skyrmion_galerkin_dev_candidate_coverage_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
GENERATOR_SPEC_PATH = OUTPUT_ROOT / "generator_spec.json"
CANDIDATE_POOL_PATH = OUTPUT_ROOT / "candidate_pool.json"
SCREEN_RESULTS_PATH = OUTPUT_ROOT / "screen_results.json"
AUDIT_RESULTS_PATH = OUTPUT_ROOT / "audit_results.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"
RISK_RESS_CSV_PATH = OUTPUT_ROOT / "risk_ress_rows.csv"
PATH_DIAGNOSTICS_PATH = OUTPUT_ROOT / "path_diagnostics.json"

REQUESTED_NEW_COUNT = 4096
COMPONENT_TARGETS = {
    "local_cloud": 1640,
    "periodic_path": 1024,
    "risk_tangent": 816,
    "sobol_global": 616,
}
LOCAL_SCALES = (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
TANGENT_RADII = (
    0.0001,
    0.0002,
    0.0004,
    0.0007,
    0.001,
    0.0015,
    0.0022,
    0.0032,
    0.0045,
    0.0064,
    0.009,
    0.0125,
    0.017,
    0.023,
    0.031,
    0.041,
    0.055,
)
CANDIDATE_BATCH_SIZE = 8
EXPECTED_ALL_ALLOWANCES_HASHES = {
    "summary.json": "3b9c43f4486bfc07708182285e4a66fe9a0dc550e9f7184e0e66e92ac4ec4867",
    "inventory.json": "8a9a529c9947260c33eaa2caef871e1528d6dec0c864e0713ca468d957a72bda",
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _full_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"coverage output must be beneath {root}: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Any, *, immutable: bool = True) -> None:
    path = _output_path(path)
    if immutable and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable coverage artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = _output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable coverage artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            fields = list(rows[0]) if rows else ["candidate_id"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def derive_seed(global_seed: int, label: str) -> dict[str, Any]:
    text = f"{int(global_seed)}:{VERSION}:{label}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "label": label,
        "derivation_text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in ("candidate_coverage.py", "candidate_coverage_run.py", "config.json")
    }


def _verify_frozen_sources(cfg: dict[str, Any]) -> dict[str, Any]:
    v2 = verify_v2_frozen()
    phase1 = verify_v3_phase1_frozen()
    v2_protocol = require_v2_protocol(cfg)
    verified = {}
    for name, expected in EXPECTED_ALL_ALLOWANCES_HASHES.items():
        path = ALL_ALLOWANCES_DIAGNOSTIC_ROOT / name
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen all-allowance {name} hash changed")
        verified[name] = actual
    previous = read_json(ALL_ALLOWANCES_DIAGNOSTIC_ROOT / "summary.json")
    if previous.get("candidate_pool_count") != 337:
        raise RuntimeError("frozen original candidate count changed")
    expected_counts = [0, 1, 12, 35, 53, 59]
    if [row["dual_bank_eligible_count"] for row in previous["allowances"]] != expected_counts:
        raise RuntimeError("frozen original dual-bank counts changed")
    firewall = _official_v3_firewall()
    return {
        "v2": v2,
        "phase1": phase1,
        "v2_protocol": v2_protocol,
        "all_allowances": previous,
        "all_allowances_hashes": verified,
        "official_v3_firewall": firewall,
    }


def canonicalize_eta(eta: Any, law_eta: Any, box: Any) -> np.ndarray:
    """Wrap and order unordered sensors by repository Law matching."""
    from experiments.skyrmions_deep_ritz.visualize_authoritative import _match_to_law

    box_array = np.asarray(box, dtype=np.float64)
    centers = np.mod(np.asarray(eta, dtype=np.float64).reshape((-1, 2)), box_array)
    law = np.mod(np.asarray(law_eta, dtype=np.float64).reshape((-1, 2)), box_array)
    ordered, _ = _match_to_law(centers, law, box_array)
    return np.asarray(ordered, dtype=np.float64).reshape(-1)


def periodic_interpolate(
    start: Any, end: Any, alpha: float, law_eta: Any, box: Any
) -> np.ndarray:
    """Permutation-matched minimum-image interpolation on the sensor torus."""
    from experiments.skyrmions_deep_ritz.visualize_authoritative import (
        _match_to_law,
        _periodic_delta,
    )

    box_array = np.asarray(box, dtype=np.float64)
    left = canonicalize_eta(start, law_eta, box_array).reshape((-1, 2))
    right = canonicalize_eta(end, law_eta, box_array).reshape((-1, 2))
    ordered, _ = _match_to_law(right, left, box_array)
    interpolated = np.mod(
        left + float(alpha) * _periodic_delta(left, ordered, box_array), box_array
    )
    return canonicalize_eta(interpolated.reshape(-1), law_eta, box_array)


def minimum_periodic_separation(eta: Any, box: Any) -> float:
    centers = np.asarray(eta, dtype=np.float64).reshape((-1, 2))
    box_array = np.asarray(box, dtype=np.float64)
    delta = centers[:, None, :] - centers[None, :, :]
    delta -= box_array * np.round(delta / box_array)
    distances = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _anchor_rows(previous: dict[str, Any], law_eta: list[float]) -> list[dict[str, Any]]:
    records = {row["candidate_id"]: row for row in previous["per_candidate_records"]}
    ordered_ids: list[str] = []
    for row in previous["allowances"]:
        if row["allowance_percent"] < 1.0:
            continue
        ordered_ids.append(row["best_robust_candidate_id"])
        ordered_ids.extend(
            item["candidate_id"]
            for item in row["distinct_basin_information"]["maxmin_shortlist"]
        )
    result = [{"anchor_id": "Law", "eta": law_eta, "source": "frozen Law"}]
    seen = {"Law"}
    for candidate_id in ordered_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        row = records[candidate_id]
        result.append(
            {
                "anchor_id": candidate_id,
                "eta": row["eta"],
                "source": "frozen all-allowance robust/max-min diagnostic",
                "risk": row["scientific_selection_risk"],
                "robust_ress": row["robust_ress"],
            }
        )
    required = {"candidate_078", "candidate_074", "candidate_093", "candidate_168", "candidate_080"}
    missing = required - seen
    if missing:
        raise RuntimeError(f"required coverage anchors missing: {sorted(missing)}")
    return result


def _path_pairs(anchors: list[dict[str, Any]]) -> list[dict[str, str]]:
    ids = [row["anchor_id"] for row in anchors if row["anchor_id"] != "Law"]
    pairs = [("Law", candidate_id) for candidate_id in ids]
    witness_pairs = (
        ("candidate_078", "candidate_074"),
        ("candidate_074", "candidate_093"),
        ("candidate_093", "candidate_168"),
        ("candidate_168", "candidate_080"),
        ("candidate_078", "candidate_080"),
        ("candidate_061", "candidate_094"),
        ("candidate_159", "candidate_112"),
        ("candidate_058", "candidate_173"),
    )
    available = set(ids)
    pairs.extend(pair for pair in witness_pairs if set(pair) <= available)
    return [
        {"path_id": f"path_{index:02d}_{left}_to_{right}", "start": left, "end": right}
        for index, (left, right) in enumerate(pairs)
    ]


def generator_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    frozen = _verify_frozen_sources(cfg)
    previous = frozen["all_allowances"]
    law_eta = [float(value) for value in previous["per_candidate_records"][0]["eta"]]
    # The first pool row is not contractually Law; use the explicit frozen config anchor.
    law_eta = [float(value) for value in cfg["envelope"]["law_eta"]]
    anchors = _anchor_rows(previous, law_eta)
    paths = _path_pairs(anchors)
    seeds = {
        label: derive_seed(cfg["seed"], label)
        for label in ("local_cloud", "periodic_path", "risk_tangent", "sobol_global")
    }
    return {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "previous_development_diagnostics_informed_generator": True,
        "requested_new_unique_count": REQUESTED_NEW_COUNT,
        "component_targets": COMPONENT_TARGETS,
        "dtype": "float64",
        "seed_namespace": VERSION,
        "seed_records": seeds,
        "canonicalization": "periodic wrap followed by exhaustive sensor permutation matched to frozen Law",
        "candidate_identity": "repository eta_key over canonical float64 eta",
        "minimum_sensor_separation": float(cfg["measurement"]["min_separation"]),
        "box": [float(value) for value in cfg["physics"]["box"]],
        "law_eta": law_eta,
        "anchors": anchors,
        "local_cloud": {
            "scales": list(LOCAL_SCALES),
            "distribution": "deterministic Gaussian periodic perturbations balanced over anchor/scale cells",
        },
        "periodic_paths": {
            "paths": paths,
            "interpolation": "repository permutation matching plus periodic minimum-image displacement",
            "alpha_sequence": "deterministic golden-ratio low-discrepancy interior sequence",
        },
        "risk_tangent": {
            "direction_count_minimum": 24,
            "radii": list(TANGENT_RADII),
            "signs": [-1, 1],
            "construction": "Gaussian directions projected orthogonal to exact frozen selection-risk gradient at Law",
            "optimizer_run": False,
        },
        "global": {
            "method": "scipy.stats.qmc.Sobol",
            "scramble": True,
            "reservoir_power": 14,
        },
        "risk_shells_percent": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        "interpretation_rule": {
            "COVERAGE_LIMITED": "new 0.5% survivors >=5, or new 1% survivors >=10 and candidate_078 local 1% survivor fraction >=0.10",
            "LOW_RISK_OVERLAP_IS_INTRINSICALLY_SPARSE_ON_TESTED_DEVELOPMENT_BANKS": "new 0.5% survivors=0, new 1% survivors<=5, and candidate_078 local 1% survivor fraction <0.02",
            "MIXED": "all other outcomes",
        },
        "frozen_source_hashes": {
            "v2_output_tree": frozen["v2"]["output_tree_sha256"],
            "v2_protocol": frozen["v2_protocol"]["protocol_sha256"],
            "phase1": frozen["phase1"]["verified_hashes"],
            "all_allowances": frozen["all_allowances_hashes"],
        },
        "source_sha256": _source_hashes(),
        "validation_access_permitted": False,
        "tangent_optimizer_permitted": False,
        "full_galerkin_permitted": False,
        "official_protocol_permitted": False,
    }


def _balanced_quotas(total: int, count: int) -> list[int]:
    base, remainder = divmod(int(total), int(count))
    return [base + (1 if index < remainder else 0) for index in range(count)]


def generate_candidate_pool(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate and seal the complete pool before any periodic-audit scoring."""
    if CANDIDATE_POOL_PATH.exists():
        spec = read_json(GENERATOR_SPEC_PATH)
        pool = read_json(CANDIDATE_POOL_PATH)
        if spec != generator_spec(cfg):
            raise RuntimeError("coverage generator specification changed")
        if pool.get("generator_spec_sha256") != file_sha256(GENERATOR_SPEC_PATH):
            raise RuntimeError("coverage pool generator-spec seal changed")
        if pool.get("candidate_rows_sha256") != payload_sha256(pool["rows"]):
            raise RuntimeError("coverage candidate-pool payload changed")
        return {**pool, "cache_hit": True}

    started = time.perf_counter()
    spec = generator_spec(cfg)
    if GENERATOR_SPEC_PATH.exists():
        if read_json(GENERATOR_SPEC_PATH) != spec:
            raise RuntimeError("different immutable coverage generator spec exists")
    else:
        _atomic_json(GENERATOR_SPEC_PATH, spec)

    law_eta = np.asarray(spec["law_eta"], dtype=np.float64)
    box = np.asarray(spec["box"], dtype=np.float64)
    minimum_separation = float(spec["minimum_sensor_separation"])
    original = read_json(V2_OUTPUT_ROOT / "screening" / "candidate_pool.json")["rows"]
    old_keys = {
        eta_key(canonicalize_eta(row["eta"], law_eta, box)) for row in original
    }
    rows: list[dict[str, Any]] = []
    new_keys: set[str] = set()
    counts = {
        "raw_generated_count": 0,
        "geometry_invalid_rejected_count": 0,
        "within_new_duplicates": 0,
        "duplicates_against_v2": 0,
    }

    def consider(raw_eta: Any, metadata: dict[str, Any]) -> bool:
        counts["raw_generated_count"] += 1
        raw = np.asarray(raw_eta, dtype=np.float64).reshape(-1)
        canonical = canonicalize_eta(raw, law_eta, box)
        if minimum_periodic_separation(canonical, box) < minimum_separation:
            counts["geometry_invalid_rejected_count"] += 1
            return False
        key = eta_key(canonical)
        if key in old_keys:
            counts["duplicates_against_v2"] += 1
            return False
        if key in new_keys:
            counts["within_new_duplicates"] += 1
            return False
        new_keys.add(key)
        rows.append(
            {
                "candidate_id": f"coverage_{len(rows):04d}",
                "eta": canonical.tolist(),
                "raw_eta": raw.tolist(),
                "canonical_eta": canonical.tolist(),
                "eta_sha256": key,
                "geometry_valid": True,
                "minimum_periodic_sensor_separation": minimum_periodic_separation(canonical, box),
                **metadata,
            }
        )
        return True

    anchors = spec["anchors"]
    anchor_map = {
        row["anchor_id"]: np.asarray(row["eta"], dtype=np.float64) for row in anchors
    }

    # A. Balanced multi-scale local clouds.
    cells = [(row["anchor_id"], float(scale)) for row in anchors for scale in LOCAL_SCALES]
    local_quotas = _balanced_quotas(COMPONENT_TARGETS["local_cloud"], len(cells))
    for cell_index, ((anchor_id, scale), quota) in enumerate(zip(cells, local_quotas, strict=True)):
        seed = derive_seed(cfg["seed"], f"local_cloud:{cell_index}")["seed"]
        rng = np.random.default_rng(seed)
        accepted = 0
        attempt = 0
        while accepted < quota:
            raw = anchor_map[anchor_id] + scale * rng.normal(size=law_eta.shape)
            accepted += int(
                consider(
                    raw,
                    {
                        "generation_method": "local_cloud",
                        "anchor_id": anchor_id,
                        "perturbation_scale": scale,
                        "generation_attempt": attempt,
                    },
                )
            )
            attempt += 1
            if attempt > 10000:
                raise RuntimeError(f"could not fill local cell {anchor_id}/{scale}")

    # B. Dense periodic Law/witness and witness/witness paths.
    paths = spec["periodic_paths"]["paths"]
    path_quotas = _balanced_quotas(COMPONENT_TARGETS["periodic_path"], len(paths))
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for path, quota in zip(paths, path_quotas, strict=True):
        accepted = 0
        attempt = 0
        while accepted < quota:
            alpha = ((attempt + 1) * golden) % 1.0
            raw = periodic_interpolate(
                anchor_map[path["start"]], anchor_map[path["end"]], alpha, law_eta, box
            )
            accepted += int(
                consider(
                    raw,
                    {
                        "generation_method": "periodic_path",
                        "anchor_id": path["end"],
                        "nearest_anchor_hint": path["end"],
                        "perturbation_scale": None,
                        "path_id": path["path_id"],
                        "path_start": path["start"],
                        "path_end": path["end"],
                        "alpha": float(alpha),
                        "generation_attempt": attempt,
                    },
                )
            )
            attempt += 1
            if attempt > 100000:
                raise RuntimeError(f"could not fill periodic path {path['path_id']}")

    # C. Risk-tangent feasible-manifold exploration; this is generation only,
    # never a Tangent objective optimization.
    data = selection_data(cfg, "search_train", "periodic_audit")
    law = jnp.asarray(law_eta, dtype=jnp.float64)
    _, risk_gradient = jax.value_and_grad(lambda eta: selection_risk(eta, data))(law)
    gradient = np.asarray(risk_gradient, dtype=np.float64)
    gradient_norm2 = max(float(np.dot(gradient, gradient)), 1.0e-30)
    tangent_seed = spec["seed_records"]["risk_tangent"]["seed"]
    tangent_rng = np.random.default_rng(tangent_seed)
    direction_index = 0
    tangent_start = len(rows)
    while len(rows) - tangent_start < COMPONENT_TARGETS["risk_tangent"]:
        direction = tangent_rng.normal(size=law_eta.shape)
        direction -= float(np.dot(direction, gradient)) / gradient_norm2 * gradient
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-14:
            continue
        direction /= norm
        for radius in TANGENT_RADII:
            for sign in (-1, 1):
                if len(rows) - tangent_start >= COMPONENT_TARGETS["risk_tangent"]:
                    break
                consider(
                    law_eta + sign * float(radius) * direction,
                    {
                        "generation_method": "risk_tangent",
                        "anchor_id": "Law",
                        "perturbation_scale": float(radius),
                        "tangent_direction_index": direction_index,
                        "tangent_sign": sign,
                    },
                )
        direction_index += 1
        if direction_index > 10000:
            raise RuntimeError("could not fill risk-tangent component")

    # D. Deterministic scrambled Sobol global coverage.
    global_start = len(rows)
    sobol = qmc.Sobol(
        d=law_eta.size,
        scramble=True,
        seed=spec["seed_records"]["sobol_global"]["seed"],
    )
    unit = sobol.random_base2(m=int(spec["global"]["reservoir_power"]))
    tiled_box = np.tile(box, law_eta.size // 2)
    for sobol_index, point in enumerate(unit):
        if len(rows) - global_start >= COMPONENT_TARGETS["sobol_global"]:
            break
        consider(
            point * tiled_box,
            {
                "generation_method": "sobol_global",
                "anchor_id": None,
                "perturbation_scale": None,
                "sobol_index": sobol_index,
            },
        )
    if len(rows) - global_start != COMPONENT_TARGETS["sobol_global"]:
        raise RuntimeError("Sobol reservoir did not contain enough feasible unique rows")

    if len(rows) != REQUESTED_NEW_COUNT:
        raise RuntimeError(f"coverage pool has {len(rows)} rows, expected {REQUESTED_NEW_COUNT}")
    observed_components = {
        name: sum(row["generation_method"] == name for row in rows)
        for name in COMPONENT_TARGETS
    }
    if observed_components != COMPONENT_TARGETS:
        raise RuntimeError(f"coverage component counts changed: {observed_components}")

    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "generator_spec_sha256": file_sha256(GENERATOR_SPEC_PATH),
        "requested_count": REQUESTED_NEW_COUNT,
        **counts,
        "component_counts": observed_components,
        "final_new_unique_count": len(rows),
        "original_v2_count_excluded": len(original),
        "candidate_rows_sha256": payload_sha256(rows),
        "generated_before_audit_evaluation": True,
        "generation_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    _atomic_json(CANDIDATE_POOL_PATH, result)
    return result


def _verify_pool(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not GENERATOR_SPEC_PATH.is_file() or not CANDIDATE_POOL_PATH.is_file():
        raise RuntimeError("coverage generator and pool must be frozen first")
    spec = read_json(GENERATOR_SPEC_PATH)
    if spec != generator_spec(cfg):
        raise RuntimeError("coverage generator spec no longer matches code/config/sources")
    pool = read_json(CANDIDATE_POOL_PATH)
    if pool.get("generator_spec_sha256") != file_sha256(GENERATOR_SPEC_PATH):
        raise RuntimeError("coverage generator-spec hash mismatch")
    if pool.get("candidate_rows_sha256") != payload_sha256(pool["rows"]):
        raise RuntimeError("coverage candidate rows hash mismatch")
    return spec, pool


def _risk_rows_batched(data: Any, rows: list[dict[str, Any]], batch_size: int) -> list[float]:
    """Exact frozen scientific risk using candidate-batched I-projection."""
    problem = data.selection_problem
    bank = data.projection_bank
    projector = EmpiricalIProjector(
        problem.projection_config, trajectory_backend=problem.projection_backend
    )

    def preprocess_one(eta: jax.Array) -> tuple[jax.Array, jax.Array]:
        reconstruction = reconstruct_moments(eta, problem)
        features = problem.family.features(bank.configurations, eta)
        return reconstruction.values, features

    preprocess = jax.jit(jax.vmap(preprocess_one))
    risk_batch = jax.jit(
        jax.vmap(
            lambda weights: integrated_risk(
                weights,
                data.reference_features,
                data.truth_means,
                data.whitening,
                problem.time_weights,
            )
        )
    )
    result: list[float] = []
    for start in range(0, len(rows), int(batch_size)):
        selected = rows[start : start + int(batch_size)]
        actual = len(selected)
        etas = np.asarray([row["eta"] for row in selected], dtype=np.float64)
        if actual < int(batch_size):
            etas = np.concatenate(
                (etas, np.repeat(etas[-1:], int(batch_size) - actual, axis=0))
            )
        targets, features = preprocess(jnp.asarray(etas, dtype=jnp.float64))
        projected = projector.project_candidate_trajectories(
            features, bank.base_weights, targets
        )
        values = np.asarray(risk_batch(projected.weights), dtype=np.float64)
        result.extend(float(value) for value in values[:actual])
    return result


def _support_payload(diagnostics: dict[str, Any], problem: Any) -> dict[str, Any]:
    projection = bool(
        float(diagnostics["maximum_projection_residual"])
        <= float(problem.forcing_config.projection_tolerance)
    )
    ress = float(diagnostics["minimum_ess_fraction"])
    forcing = bool(
        float(diagnostics["maximum_forcing_mean"])
        <= float(problem.forcing_config.forcing_mean_tolerance)
    )
    covariance = bool(
        float(diagnostics["maximum_covariance_condition"])
        <= float(problem.forcing_config.max_covariance_condition)
    )
    return {
        "projection_valid": projection,
        "minimum_ress": ress,
        "ress_valid": bool(ress >= MINIMUM_RESS),
        "forcing_valid": forcing,
        "covariance_valid": covariance,
        "support_valid": bool(projection and ress >= MINIMUM_RESS and forcing and covariance),
        "maximum_projection_residual": float(diagnostics["maximum_projection_residual"]),
        "maximum_forcing_mean": float(diagnostics["maximum_forcing_mean"]),
        "maximum_covariance_condition": float(diagnostics["maximum_covariance_condition"]),
    }


def evaluate_screen(cfg: dict[str, Any]) -> dict[str, Any]:
    if SCREEN_RESULTS_PATH.exists():
        _, pool = _verify_pool(cfg)
        saved = read_json(SCREEN_RESULTS_PATH)
        if saved.get("candidate_pool_sha256") != file_sha256(CANDIDATE_POOL_PATH):
            raise RuntimeError("coverage screen source pool changed")
        if saved.get("screen_rows_sha256") != payload_sha256(saved["rows"]):
            raise RuntimeError("coverage screen rows changed")
        return {**saved, "cache_hit": True}

    _, pool = _verify_pool(cfg)
    data = selection_data(cfg, "search_train", "periodic_audit")
    started_risk = time.perf_counter()
    risks = _risk_rows_batched(data, pool["rows"], CANDIDATE_BATCH_SIZE)
    risk_seconds = time.perf_counter() - started_risk
    rows_with_risk = [
        {
            **row,
            "scientific_selection_risk": risk,
            "risk_increase_pct": 100.0 * (risk / 5.186549474478042 - 1.0),
        }
        for row, risk in zip(pool["rows"], risks, strict=True)
    ]
    started_screen = time.perf_counter()
    evaluated = _screen_rows_batched(
        cfg, data, load_bank("screen"), rows_with_risk, CANDIDATE_BATCH_SIZE
    )
    screen_seconds = time.perf_counter() - started_screen
    normalized = []
    for row in evaluated:
        support = _support_payload(row["screen"], data.selection_problem)
        normalized.append(
            {
                **{key: value for key, value in row.items() if key not in {"screen", "projection_valid", "minimum_ess_fraction"}},
                "screen": support,
                "screen_eligible": support["support_valid"],
                "validation_accessed": False,
                "tangent_optimization_run": False,
                "full_kf_constructed": False,
            }
        )
    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH),
        "candidate_count": len(normalized),
        "law_selection_risk": 5.186549474478042,
        "risk_seconds": risk_seconds,
        "screen_projection_seconds": screen_seconds,
        "candidate_batched_projection": True,
        "batch_size": CANDIDATE_BATCH_SIZE,
        "screen_rows_sha256": payload_sha256(normalized),
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "rows": normalized,
    }
    _atomic_json(SCREEN_RESULTS_PATH, result)
    return result


def evaluate_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    if AUDIT_RESULTS_PATH.exists():
        _verify_pool(cfg)
        screen = read_json(SCREEN_RESULTS_PATH)
        saved = read_json(AUDIT_RESULTS_PATH)
        if saved.get("screen_results_sha256") != file_sha256(SCREEN_RESULTS_PATH):
            raise RuntimeError("coverage audit source screen result changed")
        if saved.get("audit_rows_sha256") != payload_sha256(saved["rows"]):
            raise RuntimeError("coverage audit rows changed")
        return {**saved, "cache_hit": True}

    _verify_pool(cfg)
    if not SCREEN_RESULTS_PATH.is_file():
        raise RuntimeError("screen results must be frozen before audit evaluation")
    screen = read_json(SCREEN_RESULTS_PATH)
    if screen.get("candidate_pool_sha256") != file_sha256(CANDIDATE_POOL_PATH):
        raise RuntimeError("screen/pool seal mismatch before audit")
    pool_mtime_ns = CANDIDATE_POOL_PATH.stat().st_mtime_ns
    audit_started_ns = time.time_ns()
    if pool_mtime_ns > audit_started_ns:
        raise RuntimeError("candidate pool was not frozen before audit start")
    ceiling = selection_ceiling(screen["law_selection_risk"], 5.0)
    relevant = [
        row
        for row in screen["rows"]
        if row["scientific_selection_risk"] <= ceiling and row["screen_eligible"]
    ]
    data = selection_data(cfg, "search_train", "periodic_audit")
    started = time.perf_counter()
    evaluated = _screen_rows_batched(
        cfg, data, load_bank("periodic_audit"), relevant, CANDIDATE_BATCH_SIZE
    )
    elapsed = time.perf_counter() - started
    normalized = []
    for source, row in zip(relevant, evaluated, strict=True):
        audit = _support_payload(row["screen"], data.selection_problem)
        robust = min(float(source["screen"]["minimum_ress"]), audit["minimum_ress"])
        normalized.append(
            {
                "candidate_id": source["candidate_id"],
                "eta": source["eta"],
                "eta_sha256": source["eta_sha256"],
                "generation_method": source["generation_method"],
                "anchor_id": source["anchor_id"],
                "perturbation_scale": source["perturbation_scale"],
                "path_id": source.get("path_id"),
                "alpha": source.get("alpha"),
                "scientific_selection_risk": source["scientific_selection_risk"],
                "risk_increase_pct": source["risk_increase_pct"],
                "geometry_valid": source["geometry_valid"],
                "screen": source["screen"],
                "audit": audit,
                "robust_ress": robust,
                "complete_dual_bank_eligible": bool(source["screen_eligible"] and audit["support_valid"]),
                "validation_accessed": False,
                "tangent_optimization_run": False,
                "full_kf_constructed": False,
            }
        )
    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH),
        "screen_results_sha256": file_sha256(SCREEN_RESULTS_PATH),
        "candidate_pool_mtime_ns_before_audit": pool_mtime_ns,
        "audit_started_ns": audit_started_ns,
        "candidate_pool_frozen_before_audit": pool_mtime_ns <= audit_started_ns,
        "audit_candidate_rule": "scientific risk <= exact 5% ceiling and complete frozen screen support",
        "audit_candidate_count": len(normalized),
        "audit_seconds": elapsed,
        "candidate_batched_projection": True,
        "batch_size": CANDIDATE_BATCH_SIZE,
        "audit_rows_sha256": payload_sha256(normalized),
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "rows": normalized,
    }
    _atomic_json(AUDIT_RESULTS_PATH, result)
    return result


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {key: None for key in ("count", "minimum", "p10", "p25", "median", "p75", "p90", "maximum", "fraction_ge_0p05")}
    quantiles = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(len(array)),
        **{
            name: float(value)
            for name, value in zip(
                ("minimum", "p10", "p25", "median", "p75", "p90", "maximum"),
                quantiles,
                strict=True,
            )
        },
        "fraction_ge_0p05": float(np.mean(array >= MINIMUM_RESS)),
    }


def _spearman(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(value_key) is not None]
    if len(usable) < 2:
        return {"n": len(usable), "rho": None, "two_sided_pvalue": None}
    result = spearmanr(
        [row["risk_increase_pct"] for row in usable],
        [row[value_key] for row in usable],
    )
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "n": len(usable),
        "rho": None if not math.isfinite(rho) else rho,
        "two_sided_pvalue": None if not math.isfinite(pvalue) else pvalue,
    }


RISK_BINS = (
    ("le_0p5", -math.inf, 0.5),
    ("0p5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("2_to_3", 2.0, 3.0),
    ("3_to_4", 3.0, 4.0),
    ("4_to_5", 4.0, 5.0),
)


def _risk_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for label, lower, upper in RISK_BINS:
        selected = [
            row
            for row in rows
            if row["risk_increase_pct"] <= upper
            and (lower == -math.inf or row["risk_increase_pct"] > lower)
        ]
        output.append(
            {
                "bin": label,
                "lower_exclusive_percent": None if lower == -math.inf else lower,
                "upper_inclusive_percent": upper,
                "candidate_count": len(selected),
                "screen_ress": _distribution(row["screen_minimum_ress"] for row in selected),
                "audit_ress": _distribution(row["audit_minimum_ress"] for row in selected if row.get("audit_minimum_ress") is not None),
                "robust_ress": _distribution(row["robust_ress"] for row in selected if row.get("robust_ress") is not None),
            }
        )
    return output


def _nearest_anchor(row: dict[str, Any], anchors: list[dict[str, Any]], box: Any) -> str:
    return min(
        anchors,
        key=lambda anchor: (
            _symmetry_aware_distance(row["eta"], anchor["eta"], box),
            anchor["anchor_id"],
        ),
    )["anchor_id"]


def _maxmin_combined(
    rows: list[dict[str, Any]], anchors: list[dict[str, Any]], box: Any, maximum: int = 10
) -> list[dict[str, Any]]:
    remaining = sorted(rows, key=lambda row: (-row["robust_ress"], row["candidate_id"]))
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < int(maximum):
        chosen = min(
            remaining,
            key=lambda row: (
                -min(_symmetry_aware_distance(row["eta"], old["eta"], box) for old in selected),
                -row["robust_ress"],
                row["candidate_id"],
            ),
        )
        selected.append(chosen)
        remaining = [row for row in remaining if row["candidate_id"] != chosen["candidate_id"]]
    return [
        {
            "candidate_id": row["candidate_id"],
            "scientific_selection_risk": row["scientific_selection_risk"],
            "screen_ress": row["screen_minimum_ress"],
            "audit_ress": row["audit_minimum_ress"],
            "robust_ress": row["robust_ress"],
            "generation_source": row["generation_method"],
            "nearest_anchor": _nearest_anchor(row, anchors, box),
        }
        for row in selected
    ]


def _local_basin_summary(
    screen_rows: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in screen_rows:
        if row["generation_method"] == "local_cloud":
            groups.setdefault((row["anchor_id"], float(row["perturbation_scale"])), []).append(row)
    output = []
    for (anchor_id, scale), rows in sorted(groups.items()):
        audited = [audit_by_id[row["candidate_id"]] for row in rows if row["candidate_id"] in audit_by_id]
        output.append(
            {
                "anchor_id": anchor_id,
                "perturbation_scale": scale,
                "generated": len(rows),
                "geometry_valid": sum(row["geometry_valid"] for row in rows),
                **{
                    f"inside_{str(allowance).replace('.', 'p')}_percent": sum(
                        row["scientific_selection_risk"] <= selection_ceiling(5.186549474478042, allowance)
                        for row in rows
                    )
                    for allowance in (0.5, 1.0, 2.0, 3.0)
                },
                "screen_ress_pass": sum(row["screen"]["ress_valid"] for row in rows),
                "screen_support_pass": sum(row["screen_eligible"] for row in rows),
                "dual_bank_pass": sum(row["complete_dual_bank_eligible"] for row in audited),
                "audited_count": len(audited),
                "audit_ress": _distribution(row["audit"]["minimum_ress"] for row in audited),
                "robust_ress": _distribution(row["robust_ress"] for row in audited),
            }
        )
    return output


def _path_diagnostics(
    screen_rows: list[dict[str, Any]], audit_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in screen_rows:
        if row["generation_method"] != "periodic_path":
            continue
        audit = audit_by_id.get(row["candidate_id"])
        grouped.setdefault(row["path_id"], []).append(
            {
                "candidate_id": row["candidate_id"],
                "alpha": row["alpha"],
                "risk_increase_pct": row["risk_increase_pct"],
                "screen_minimum_ress": row["screen"]["minimum_ress"],
                "audit_minimum_ress": None if audit is None else audit["audit"]["minimum_ress"],
                "robust_ress": None if audit is None else audit["robust_ress"],
                "complete_dual_bank_eligible": False if audit is None else audit["complete_dual_bank_eligible"],
            }
        )
    paths = []
    for path_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["alpha"], row["candidate_id"]))
        crossings = []
        for left, right in zip(rows[:-1], rows[1:]):
            for threshold in (0.5, 1.0, 2.0, 3.0):
                if (left["risk_increase_pct"] <= threshold) != (right["risk_increase_pct"] <= threshold):
                    crossings.append({"quantity": "risk_increase_pct", "threshold": threshold, "bracket_candidate_ids": [left["candidate_id"], right["candidate_id"]]})
            for key in ("screen_minimum_ress", "audit_minimum_ress", "robust_ress"):
                if left[key] is not None and right[key] is not None and ((left[key] >= MINIMUM_RESS) != (right[key] >= MINIMUM_RESS)):
                    crossings.append({"quantity": key, "threshold": MINIMUM_RESS, "bracket_candidate_ids": [left["candidate_id"], right["candidate_id"]]})
        paths.append({"path_id": path_id, "evaluated_rows": rows, "evaluated_crossing_brackets": crossings})
    return {
        "schema_version": 1,
        "development_only": True,
        "crossings_are_evaluated_brackets_not_interpolated_claims": True,
        "path_count": len(paths),
        "paths": paths,
    }


def _interpretation(
    allowance_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_allowance = {row["allowance_percent"]: row for row in allowance_rows}
    new_half = by_allowance[0.5]["new_dual_bank_count"]
    new_one = by_allowance[1.0]["new_dual_bank_count"]
    candidate_078 = [row for row in local_rows if row["anchor_id"] == "candidate_078"]
    local_generated_inside = sum(row["inside_1p0_percent"] for row in candidate_078)
    local_survivors = 0
    # dual_bank_pass includes any <=5%-audited row; count exact 1% below from
    # the allowance summary's candidate list, computed by caller.
    one_ids = set(by_allowance[1.0]["new_eligible_candidate_ids"])
    local_ids = set(by_allowance[1.0]["new_candidate_078_local_ids"])
    local_survivors = len(one_ids & local_ids)
    fraction = 0.0 if local_generated_inside == 0 else local_survivors / local_generated_inside
    if new_half >= 5 or (new_one >= 10 and fraction >= 0.10):
        label = "COVERAGE_LIMITED"
        reason = "expanded targeted sampling found a substantial new low-risk dual-bank neighborhood"
    elif new_half == 0 and new_one <= 5 and fraction < 0.02:
        label = "LOW_RISK_OVERLAP_IS_INTRINSICALLY_SPARSE_ON_TESTED_DEVELOPMENT_BANKS"
        reason = "dense targeted sampling found no 0.5% witness and very few 1% survivors, including a thin candidate_078 neighborhood"
    else:
        label = "MIXED"
        reason = "expanded coverage improved low-risk survivor counts, but the low-risk neighborhood evidence remains limited or uneven"
    return {
        "label": label,
        "development_only": True,
        "reason": reason,
        "new_0p5_percent_survivors": new_half,
        "new_1_percent_survivors": new_one,
        "candidate_078_local_inside_1_percent": local_generated_inside,
        "candidate_078_local_1_percent_survivors": local_survivors,
        "candidate_078_local_1_percent_survivor_fraction": fraction,
    }


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    if SUMMARY_PATH.exists() or INVENTORY_PATH.exists():
        if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
            raise RuntimeError("incomplete coverage summary/inventory pair")
        _verify_frozen_sources(cfg)
        inventory = read_json(INVENTORY_PATH)
        for row in inventory["artifacts"]:
            path = OUTPUT_ROOT / row["path"]
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"coverage cached artifact changed: {row['path']}")
        if inventory.get("source_sha256") != _source_hashes():
            raise RuntimeError("coverage implementation/config changed after completion")
        _official_v3_firewall()
        return {**read_json(SUMMARY_PATH), "cache_hit": True}

    started = time.perf_counter()
    frozen = _verify_frozen_sources(cfg)
    spec, pool = _verify_pool(cfg)
    screen = evaluate_screen(cfg)
    audit = evaluate_audit(cfg)
    original_summary = frozen["all_allowances"]
    original_rows = []
    for row in original_summary["per_candidate_records"]:
        original_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "eta": row["eta"],
                "eta_sha256": row["eta_sha256"],
                "generation_method": "original_v2",
                "anchor_id": None,
                "scientific_selection_risk": row["scientific_selection_risk"],
                "risk_increase_pct": 100.0 * (row["scientific_selection_risk"] / 5.186549474478042 - 1.0),
                "screen_minimum_ress": row["screen_minimum_ress"],
                "audit_minimum_ress": row["audit_minimum_ress"],
                "robust_ress": row["robust_ress"],
                "screen_support_valid": row["screen_support_valid"],
                "audit_support_valid": row["audit_support_valid"],
                "complete_dual_bank_eligible": bool(row["screen_support_valid"] and row["audit_support_valid"]),
            }
        )
    audit_by_id = {row["candidate_id"]: row for row in audit["rows"]}
    new_rows = []
    for row in screen["rows"]:
        audited = audit_by_id.get(row["candidate_id"])
        new_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "eta": row["eta"],
                "eta_sha256": row["eta_sha256"],
                "generation_method": row["generation_method"],
                "anchor_id": row["anchor_id"],
                "perturbation_scale": row["perturbation_scale"],
                "scientific_selection_risk": row["scientific_selection_risk"],
                "risk_increase_pct": row["risk_increase_pct"],
                "screen_minimum_ress": row["screen"]["minimum_ress"],
                "audit_minimum_ress": None if audited is None else audited["audit"]["minimum_ress"],
                "robust_ress": None if audited is None else audited["robust_ress"],
                "screen_support_valid": row["screen_eligible"],
                "audit_support_valid": False if audited is None else audited["audit"]["support_valid"],
                "complete_dual_bank_eligible": False if audited is None else audited["complete_dual_bank_eligible"],
            }
        )
    combined = original_rows + new_rows
    pool_by_id = {row["candidate_id"]: row for row in pool["rows"]}
    allowance_rows = []
    box = spec["box"]
    for index, allowance in enumerate(ALLOWANCES):
        ceiling = selection_ceiling(5.186549474478042, allowance)
        original_inside = [row for row in original_rows if row["scientific_selection_risk"] <= ceiling]
        new_inside = [row for row in new_rows if row["scientific_selection_risk"] <= ceiling]
        original_screen = [row for row in original_inside if row["screen_support_valid"]]
        new_screen = [row for row in new_inside if row["screen_support_valid"]]
        original_eligible = [row for row in original_inside if row["complete_dual_bank_eligible"]]
        new_eligible = [row for row in new_inside if row["complete_dual_bank_eligible"]]
        combined_eligible = original_eligible + new_eligible
        audited_new = [row for row in new_screen if row["audit_minimum_ress"] is not None]
        audited_combined = original_screen + audited_new
        candidate_078_local_ids = [
            row["candidate_id"]
            for row in new_inside
            if pool_by_id[row["candidate_id"]]["generation_method"] == "local_cloud"
            and pool_by_id[row["candidate_id"]]["anchor_id"] == "candidate_078"
        ]
        best_new_audit = max(audited_new, key=lambda row: (row["audit_minimum_ress"], row["candidate_id"]), default=None)
        best_combined_audit = max(audited_combined, key=lambda row: (row["audit_minimum_ress"], row["candidate_id"]), default=None)
        best_new_robust = max(audited_new, key=lambda row: (row["robust_ress"], row["candidate_id"]), default=None)
        best_combined_robust = max(audited_combined, key=lambda row: (row["robust_ress"], row["candidate_id"]), default=None)
        expected_original = original_summary["allowances"][index]
        if len(original_eligible) != expected_original["dual_bank_eligible_count"]:
            raise RuntimeError(f"original dual-bank reproduction failed at {allowance}%")
        allowance_rows.append(
            {
                "allowance_percent": allowance,
                "risk_ceiling": ceiling,
                "original_v2_screen_feasible_count": len(original_screen),
                "original_v2_dual_bank_count": len(original_eligible),
                "new_candidates_inside_risk_ceiling": len(new_inside),
                "new_screen_feasible_count": len(new_screen),
                "new_dual_bank_count": len(new_eligible),
                "combined_unique_screen_feasible_count": len(original_screen) + len(new_screen),
                "combined_unique_dual_bank_count": len(combined_eligible),
                "best_new_audit_minimum_ress": None if best_new_audit is None else best_new_audit["audit_minimum_ress"],
                "best_new_audit_candidate_id": None if best_new_audit is None else best_new_audit["candidate_id"],
                "best_combined_audit_minimum_ress": None if best_combined_audit is None else best_combined_audit["audit_minimum_ress"],
                "best_combined_audit_candidate_id": None if best_combined_audit is None else best_combined_audit["candidate_id"],
                "best_new_robust_ress": None if best_new_robust is None else best_new_robust["robust_ress"],
                "best_new_robust_candidate_id": None if best_new_robust is None else best_new_robust["candidate_id"],
                "best_combined_robust_ress": None if best_combined_robust is None else best_combined_robust["robust_ress"],
                "best_combined_robust_candidate_id": None if best_combined_robust is None else best_combined_robust["candidate_id"],
                "new_eligible_candidate_ids": sorted(row["candidate_id"] for row in new_eligible),
                "new_candidate_078_local_ids": sorted(candidate_078_local_ids),
                "combined_diversity_maxmin_shortlist": _maxmin_combined(combined_eligible, spec["anchors"], box),
            }
        )

    local = _local_basin_summary(screen["rows"], audit_by_id)
    path_payload = _path_diagnostics(screen["rows"], audit_by_id)
    _atomic_json(PATH_DIAGNOSTICS_PATH, path_payload)
    correlations = {}
    risk_bins = {}
    for label, rows in (("original", original_rows), ("new", new_rows), ("combined", combined)):
        correlations[label] = {
            "risk_vs_screen_ress": _spearman(rows, "screen_minimum_ress"),
            "risk_vs_audit_ress": _spearman(rows, "audit_minimum_ress"),
            "risk_vs_robust_ress": _spearman(rows, "robust_ress"),
        }
        risk_bins[label] = _risk_bin_summary(rows)
    interpretation = _interpretation(allowance_rows, local)

    csv_rows = [
        {
            "candidate_id": row["candidate_id"],
            "source": row["generation_method"],
            "anchor_id": row.get("anchor_id"),
            "scientific_selection_risk": row["scientific_selection_risk"],
            "risk_increase_pct": row["risk_increase_pct"],
            "screen_minimum_ress": row["screen_minimum_ress"],
            "audit_minimum_ress": row.get("audit_minimum_ress"),
            "robust_ress": row.get("robust_ress"),
            "screen_support_valid": row["screen_support_valid"],
            "audit_support_valid": row["audit_support_valid"],
            "complete_dual_bank_eligible": row["complete_dual_bank_eligible"],
        }
        for row in combined
    ]
    _atomic_csv(RISK_RESS_CSV_PATH, csv_rows)
    result = {
        "schema_version": 1,
        "version": VERSION,
        "purpose": "development-only candidate-coverage and empirical risk-rESS geometry study",
        "development_only": True,
        "official_result": False,
        "generator_spec_sha256": file_sha256(GENERATOR_SPEC_PATH),
        "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH),
        "screen_results_sha256": file_sha256(SCREEN_RESULTS_PATH),
        "audit_results_sha256": file_sha256(AUDIT_RESULTS_PATH),
        "path_diagnostics_sha256": file_sha256(PATH_DIAGNOSTICS_PATH),
        "risk_ress_csv_sha256": file_sha256(RISK_RESS_CSV_PATH),
        "frozen_source_hashes": spec["frozen_source_hashes"],
        "candidate_counts": {
            "original": len(original_rows),
            "new": len(new_rows),
            "combined": len(combined),
            "new_audited": len(audit["rows"]),
        },
        "allowances": allowance_rows,
        "local_basin_width": local,
        "spearman_correlations": correlations,
        "risk_bins": risk_bins,
        "development_interpretation": interpretation,
        "timings_seconds": {
            "generation": pool["generation_seconds"],
            "risk_evaluation": screen["risk_seconds"],
            "screen_evaluation": screen["screen_projection_seconds"],
            "audit_evaluation": audit["audit_seconds"],
            "summary_statistics": time.perf_counter() - started,
        },
        "candidate_pool_frozen_before_audit": audit["candidate_pool_frozen_before_audit"],
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "eigensolve_run": False,
        "deep_ritz_run": False,
        "official_protocol_created": False,
        "selection_frozen": False,
        "firewall_after": _official_v3_firewall(),
    }
    _atomic_json(SUMMARY_PATH, result)
    artifacts = []
    for path in (
        GENERATOR_SPEC_PATH,
        CANDIDATE_POOL_PATH,
        SCREEN_RESULTS_PATH,
        AUDIT_RESULTS_PATH,
        PATH_DIAGNOSTICS_PATH,
        RISK_RESS_CSV_PATH,
        SUMMARY_PATH,
    ):
        artifacts.append(
            {
                "path": str(path.relative_to(OUTPUT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    inventory = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "artifacts": artifacts,
        "source_sha256": _source_hashes(),
        "frozen_source_hashes": spec["frozen_source_hashes"],
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    _atomic_json(INVENTORY_PATH, inventory)
    _verify_frozen_sources(cfg)
    _official_v3_firewall()
    return result


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    generate_candidate_pool(cfg)
    evaluate_screen(cfg)
    evaluate_audit(cfg)
    return summarize(cfg)


__all__ = [
    "ALLOWANCES",
    "AUDIT_RESULTS_PATH",
    "CANDIDATE_POOL_PATH",
    "COMPONENT_TARGETS",
    "GENERATOR_SPEC_PATH",
    "INVENTORY_PATH",
    "MINIMUM_RESS",
    "OUTPUT_ROOT",
    "PATH_DIAGNOSTICS_PATH",
    "REQUESTED_NEW_COUNT",
    "RISK_RESS_CSV_PATH",
    "SCREEN_RESULTS_PATH",
    "SUMMARY_PATH",
    "VERSION",
    "canonicalize_eta",
    "derive_seed",
    "evaluate_audit",
    "evaluate_screen",
    "generate_candidate_pool",
    "generator_spec",
    "minimum_periodic_separation",
    "periodic_interpolate",
    "run",
    "summarize",
]
