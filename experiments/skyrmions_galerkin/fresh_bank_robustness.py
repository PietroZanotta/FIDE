"""Fresh-bank robustness study for the frozen 4,433-candidate pool.

Development-only.  This module has no candidate generator, validation loader,
Tangent/Full optimizer, Galerkin assembly, eigensolve, or Deep Ritz entry point.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
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
from scipy.stats import pearsonr, spearmanr

from mfsi.projection import EmpiricalIProjector

from .candidate_coverage import canonicalize_eta, minimum_periodic_separation
from .full_gradient import reconstruct_moments
from .galerkin_only_data import GalerkinReferenceBank, load_selection_galerkin_data
from .pareto_v2_common import ARTIFACT_DIR, read_json as read_v2_json
from .pareto_v2_selection import _generate_bank
from .pareto_v3_common import (
    ALLOWANCES,
    MINIMUM_RESS,
    ROOT,
    V2_OUTPUT_ROOT,
    file_sha256,
    payload_sha256,
    selection_ceiling,
    verify_v2_frozen,
    verify_v3_phase1_frozen,
)
from .pareto_v3_diagnostic import _official_v3_firewall, _symmetry_aware_distance


VERSION = "skyrmion_galerkin_dev_fresh_bank_robustness_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
CANDIDATE_FREEZE_PATH = OUTPUT_ROOT / "candidate_freeze.json"
BANK_MANIFEST_PATH = OUTPUT_ROOT / "bank_manifest.json"
BANK_INVENTORY_PATH = OUTPUT_ROOT / "bank_inventory.json"
SCREEN_INDEX_PATH = OUTPUT_ROOT / "replicate_screen_results.json"
AUDIT_INDEX_PATH = OUTPUT_ROOT / "replicate_audit_results.json"
CANDIDATE_SUMMARY_PATH = OUTPUT_ROOT / "candidate_robustness_summary.json"
ALLOWANCE_SUMMARY_PATH = OUTPUT_ROOT / "allowance_summary.json"
FAILURE_SUMMARY_PATH = OUTPUT_ROOT / "failure_mode_summary.json"
TIME_NODE_SUMMARY_PATH = OUTPUT_ROOT / "time_node_summary.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

COVERAGE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_candidate_coverage_v1"
EXPECTED_COVERAGE_HASHES = {
    "generator_spec.json": "e889aae23c7649f579c8108088441eb68318abb65d8bfe1d49557c2f9aed9600",
    "candidate_pool.json": "da5b07e16c9c44d1e44d7831c6badb3a8a5218e6198d1cb1a7c0963a995db5e9",
    "summary.json": "2278c34d366a71f37c70a0c8ec30376a7788a24526dce042502b5c108fcc744e",
    "inventory.json": "fe5559527652c0b99f20d61cf9de9f29ca0e09256d08e17cf1c3f93f0da3808b",
}
LAW_RISK = 5.186549474478042
REPLICATE_COUNT = 32
SCREEN_N = 8192
AUDIT_N = 16384
BATCH_SIZE = 8
OLD_HALF_PERCENT_WITNESSES = (
    "coverage_2895",
    "coverage_3062",
    "coverage_0638",
    "coverage_2893",
    "coverage_0771",
    "coverage_1958",
)
PASS_LEVELS = (16, 24, 28, 30, 32)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _array_sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"fresh-bank output must be beneath {root}: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Any) -> None:
    path = _output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable fresh-bank artifact: {path}")
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


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = _output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable fresh-bank artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(temporary, **{name: np.asarray(value) for name, value in arrays.items()})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _source_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in ("fresh_bank_robustness.py", "fresh_bank_robustness_run.py", "config.json")
    }


def derive_seed(global_seed: int, replicate: int, role: str) -> dict[str, Any]:
    if role not in {"screen", "audit"}:
        raise ValueError(f"invalid fresh-bank role: {role}")
    text = f"{int(global_seed)}:{VERSION}:replicate_{int(replicate):02d}:{role}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "namespace": VERSION,
        "replicate_id": int(replicate),
        "role": role,
        "derivation_text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def _verify_coverage_sources() -> dict[str, Any]:
    verify_v2_frozen()
    verify_v3_phase1_frozen()
    actual = {}
    for name, expected in EXPECTED_COVERAGE_HASHES.items():
        digest = file_sha256(COVERAGE_ROOT / name)
        if digest != expected:
            raise RuntimeError(f"frozen coverage {name} hash changed")
        actual[name] = digest
    inventory = json.loads((COVERAGE_ROOT / "inventory.json").read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        if file_sha256(COVERAGE_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"frozen coverage artifact changed: {row['path']}")
    _official_v3_firewall()
    return {"hashes": actual, "inventory": inventory}


def freeze_candidates(cfg: dict[str, Any]) -> dict[str, Any]:
    """Seal exactly the prior 337+4096 rows; no generation is permitted."""
    if CANDIDATE_FREEZE_PATH.exists():
        _verify_coverage_sources()
        saved = json.loads(CANDIDATE_FREEZE_PATH.read_text(encoding="utf-8"))
        if saved.get("source_sha256") != _source_hashes():
            raise RuntimeError("fresh-bank candidate-freeze implementation/config changed")
        if saved.get("candidate_rows_sha256") != payload_sha256(saved["rows"]):
            raise RuntimeError("fresh-bank frozen candidate rows changed")
        return {**saved, "cache_hit": True}

    coverage = _verify_coverage_sources()
    law_eta = [float(value) for value in cfg["envelope"]["law_eta"]]
    box = [float(value) for value in cfg["physics"]["box"]]
    old_payload = json.loads(
        (V2_OUTPUT_ROOT / "screening" / "candidate_pool.json").read_text(encoding="utf-8")
    )
    new_pool = json.loads((COVERAGE_ROOT / "candidate_pool.json").read_text(encoding="utf-8"))
    new_screen = json.loads((COVERAGE_ROOT / "screen_results.json").read_text(encoding="utf-8"))
    new_risk = {
        row["candidate_id"]: float(row["scientific_selection_risk"])
        for row in new_screen["rows"]
    }
    rows = []
    keys: set[str] = set()
    for source in old_payload["rows"]:
        eta = canonicalize_eta(source["eta"], law_eta, box)
        key = _payload_sha256([float(value) for value in eta])
        if key in keys:
            raise RuntimeError(f"duplicate canonical original geometry: {source['candidate_id']}")
        keys.add(key)
        rows.append(
            {
                "candidate_id": source["candidate_id"],
                "eta": eta.tolist(),
                "canonical_eta_sha256": key,
                "source_pool": "original_v2",
                "generation_method": "original_v2",
                "anchor_id": None,
                "fixed_scientific_selection_risk": float(source["scientific_selection_risk"]),
            }
        )
    for source in new_pool["rows"]:
        eta = canonicalize_eta(source["eta"], law_eta, box)
        key = _payload_sha256([float(value) for value in eta])
        if key in keys:
            raise RuntimeError(f"duplicate canonical new geometry: {source['candidate_id']}")
        keys.add(key)
        rows.append(
            {
                "candidate_id": source["candidate_id"],
                "eta": eta.tolist(),
                "canonical_eta_sha256": key,
                "source_pool": "coverage_v1",
                "generation_method": source["generation_method"],
                "anchor_id": source["anchor_id"],
                "fixed_scientific_selection_risk": new_risk[source["candidate_id"]],
            }
        )
    if len(rows) != 4433 or len(keys) != 4433:
        raise RuntimeError(f"combined frozen candidate count changed: {len(rows)}/{len(keys)}")
    if len(old_payload["rows"]) != 337 or len(new_pool["rows"]) != 4096:
        raise RuntimeError("candidate source membership changed")
    law_rows = [row for row in rows if row["candidate_id"] == "candidate_000"]
    if len(law_rows) != 1 or law_rows[0]["eta"] != canonicalize_eta(law_eta, law_eta, box).tolist():
        raise RuntimeError("frozen Law candidate identity changed")
    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "candidate_pool_frozen": True,
        "candidate_generation_permitted": False,
        "candidate_count": len(rows),
        "original_v2_membership_count": 337,
        "coverage_v1_membership_count": 4096,
        "unique_canonical_geometry_count": len(keys),
        "law_candidate_id": "candidate_000",
        "law_eta": law_eta,
        "box": box,
        "minimum_sensor_separation": float(cfg["measurement"]["min_separation"]),
        "risk_semantics": {
            "fixed_law_selection_risk": LAW_RISK,
            "candidate_risk_source": "authoritative fixed selection projection bank; independent of fresh rESS screen/audit pairs",
            "fresh_bank_dependent_risk_recomputed": False,
            "reason": "normal frozen selection semantics uses a dedicated selection projection bank, not the rESS support bank",
        },
        "coverage_source_hashes": coverage["hashes"],
        "source_sha256": _source_hashes(),
        "candidate_rows_sha256": payload_sha256(rows),
        "rows": rows,
    }
    _atomic_json(CANDIDATE_FREEZE_PATH, result)
    return result


def freeze_bank_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    if BANK_MANIFEST_PATH.exists():
        candidates = freeze_candidates(cfg)
        saved = json.loads(BANK_MANIFEST_PATH.read_text(encoding="utf-8"))
        if saved.get("candidate_freeze_sha256") != file_sha256(CANDIDATE_FREEZE_PATH):
            raise RuntimeError("fresh-bank manifest candidate freeze changed")
        if saved.get("manifest_body_sha256") != payload_sha256(saved["replicates"]):
            raise RuntimeError("fresh-bank manifest replicate rows changed")
        if candidates["candidate_count"] != 4433:
            raise RuntimeError("candidate freeze changed")
        return {**saved, "cache_hit": True}

    freeze = freeze_candidates(cfg)
    v2_protocol = read_v2_json(V2_OUTPUT_ROOT / "protocol.json")
    old_seeds = {
        int(row["seed"]) for row in v2_protocol["banks"]["seed_records"]
    }
    generation_config = {
        "physics": cfg["physics"],
        "reference_substeps": int(cfg["banks"]["reference_substeps"]),
        "screen_N": SCREEN_N,
        "audit_N": AUDIT_N,
        "dtype": "float64",
        "reference_checkpoint_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"),
        "generator": "pareto_v2_selection._generate_bank exact existing implementation",
    }
    replicates = []
    seen: set[int] = set()
    for replicate in range(REPLICATE_COUNT):
        screen = derive_seed(cfg["seed"], replicate, "screen")
        audit = derive_seed(cfg["seed"], replicate, "audit")
        for record in (screen, audit):
            if record["seed"] in seen or record["seed"] in old_seeds:
                raise RuntimeError("fresh-bank seed collision")
            seen.add(record["seed"])
        replicates.append(
            {
                "replicate_id": replicate,
                "seed_namespace": VERSION,
                "screen_seed": screen,
                "screen_N": SCREEN_N,
                "audit_seed": audit,
                "audit_N": AUDIT_N,
                "dtype": "float64",
                "reference_checkpoint_sha256": generation_config["reference_checkpoint_sha256"],
                "generation_configuration_sha256": payload_sha256(generation_config),
            }
        )
    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "prospectively_frozen_before_candidate_evaluation": True,
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
        "candidate_count": freeze["candidate_count"],
        "replicate_count": REPLICATE_COUNT,
        "total_bank_count": 2 * REPLICATE_COUNT,
        "all_seeds_unique": len(seen) == 2 * REPLICATE_COUNT,
        "fresh_seeds_disjoint_from_v2": seen.isdisjoint(old_seeds),
        "generation_configuration": generation_config,
        "replicates": replicates,
        "manifest_body_sha256": payload_sha256(replicates),
        "validation_seed_namespace_used": False,
        "source_sha256": _source_hashes(),
    }
    _atomic_json(BANK_MANIFEST_PATH, result)
    return result


def _bank_path(replicate: int, role: str) -> Path:
    size = SCREEN_N if role == "screen" else AUDIT_N
    return OUTPUT_ROOT / "banks" / f"replicate_{replicate:02d}_{role}_N{size}.npz"


def _verify_bank_file(path: Path, record: dict[str, Any], role: str) -> dict[str, Any]:
    expected_n = SCREEN_N if role == "screen" else AUDIT_N
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["role"].item()) != role:
            raise RuntimeError(f"fresh bank role changed: {path}")
        if int(arrays["seed"].item()) != int(record[f"{role}_seed"]["seed"]):
            raise RuntimeError(f"fresh bank seed changed: {path}")
        if tuple(arrays["configurations"].shape[:2]) != (
            int(record.get("time_nodes", arrays["configurations"].shape[0])), expected_n
        ):
            raise RuntimeError(f"fresh bank shape changed: {path}")
        initial_hash = _array_sha256(arrays["configurations"][0])
    return {
        "path": str(path.relative_to(OUTPUT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "initial_state_sha256": initial_hash,
        "replicate_id": record["replicate_id"],
        "role": role,
        "N": expected_n,
        "seed": record[f"{role}_seed"],
    }


def generate_banks(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    manifest = freeze_bank_manifest(cfg)
    if BANK_INVENTORY_PATH.exists():
        inventory = json.loads(BANK_INVENTORY_PATH.read_text(encoding="utf-8"))
        if inventory.get("bank_manifest_sha256") != file_sha256(BANK_MANIFEST_PATH):
            raise RuntimeError("fresh bank inventory manifest changed")
        for row in inventory["banks"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"fresh bank changed: {row['path']}")
        return {**inventory, "cache_hit": True}

    banks = []
    timings = []
    for record in manifest["replicates"]:
        for role in ("screen", "audit"):
            path = _bank_path(record["replicate_id"], role)
            started = time.perf_counter()
            if not path.exists():
                samples = SCREEN_N if role == "screen" else AUDIT_N
                bank = _generate_bank(
                    cfg, int(record[f"{role}_seed"]["seed"]), samples
                )
                _atomic_npz(
                    path,
                    configurations=bank.configurations,
                    velocity=bank.velocity,
                    base_weights=bank.base_weights,
                    role=np.asarray(role),
                    replicate_id=np.asarray(record["replicate_id"]),
                    seed=np.asarray(record[f"{role}_seed"]["seed"]),
                    seed_sha256=np.asarray(record[f"{role}_seed"]["sha256"]),
                    generation_configuration_sha256=np.asarray(record["generation_configuration_sha256"]),
                )
                del bank
                cache_hit = False
            else:
                cache_hit = True
            bank_row = _verify_bank_file(path, record, role)
            elapsed = time.perf_counter() - started
            bank_row["generation_seconds"] = elapsed
            bank_row["cache_hit"] = cache_hit
            banks.append(bank_row)
            timings.append(elapsed)
            if progress is not None:
                progress(record["replicate_id"], role, cache_hit, elapsed)
    initial_hashes = [row["initial_state_sha256"] for row in banks]
    if len(initial_hashes) != len(set(initial_hashes)):
        raise RuntimeError("fresh bank initial states are not pairwise independent")
    result = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
        "candidate_evaluation_started": False,
        "bank_count": len(banks),
        "replicate_count": REPLICATE_COUNT,
        "pairwise_distinct_initial_state_hashes": True,
        "total_generation_seconds_this_invocation": sum(timings),
        "banks": banks,
    }
    _atomic_json(BANK_INVENTORY_PATH, result)
    return result


def _load_bank(path: Path) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


class _FreshBankEvaluator:
    """Candidate-batched exact forcing diagnostics with prospective ESS argmin."""

    def __init__(self, problem: Any):
        self.problem = problem
        self.projector = EmpiricalIProjector(
            problem.projection_config, trajectory_backend=problem.projection_backend
        )

        def preprocess_one(eta: jax.Array, configurations: jax.Array, velocity: jax.Array):
            reconstruction = reconstruct_moments(eta, problem)
            return (
                reconstruction.values,
                reconstruction.derivatives,
                problem.family.features(configurations, eta),
                problem.family.jvp(configurations, velocity, eta),
            )

        self.preprocess = jax.jit(
            lambda etas, configurations, velocity: jax.vmap(
                lambda eta: preprocess_one(eta, configurations, velocity)
            )(etas)
        )

    def evaluate(self, etas: np.ndarray, bank: GalerkinReferenceBank) -> dict[str, np.ndarray]:
        count = len(etas)
        result = {
            "maximum_projection_residual": np.empty(count, dtype=np.float64),
            "minimum_ress": np.empty(count, dtype=np.float64),
            "controlling_ress_time_index": np.empty(count, dtype=np.int16),
            "maximum_forcing_mean": np.empty(count, dtype=np.float64),
            "maximum_covariance_condition": np.empty(count, dtype=np.float64),
            "projection_valid": np.empty(count, dtype=bool),
            "ress_valid": np.empty(count, dtype=bool),
            "forcing_valid": np.empty(count, dtype=bool),
            "covariance_valid": np.empty(count, dtype=bool),
            "support_valid": np.empty(count, dtype=bool),
        }
        problem = self.problem
        for start in range(0, count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, count)
            actual = stop - start
            batch = etas[start:stop]
            if actual < BATCH_SIZE:
                batch = np.concatenate(
                    (batch, np.repeat(batch[-1:], BATCH_SIZE - actual, axis=0))
                )
            targets, derivatives, features, advective = self.preprocess(
                jnp.asarray(batch, dtype=jnp.float64),
                bank.configurations,
                bank.velocity,
            )
            projected = self.projector.project_candidate_trajectories(
                features, bank.base_weights, targets
            )
            for local in range(actual):
                weights = projected.weights[local]
                lam = projected.lam[local]
                moment_m = jnp.einsum("tn,tnr->tr", weights, advective[local])
                scalar_m = jnp.einsum("tnr,tr->tn", advective[local], lam)
                centered_phi = features[local] - projected.moments[local, :, None, :]
                centered_g = scalar_m - jnp.einsum("tn,tn->t", weights, scalar_m)[:, None]
                covariance_phi_g = jnp.einsum(
                    "tn,tnr,tn->tr", weights, centered_phi, centered_g
                )
                rhs = derivatives[local] - moment_m - covariance_phi_g
                regularized = projected.covariance[local] + float(
                    problem.forcing_config.covariance_ridge
                ) * jnp.eye(features.shape[-1])
                lambda_dot = jax.vmap(jnp.linalg.solve)(regularized, rhs)
                forcing = (
                    jnp.einsum(
                        "tr,tnr->tn",
                        lambda_dot,
                        features[local] - targets[local, :, None, :],
                    )
                    + jnp.einsum(
                        "tr,tnr->tn", lam, advective[local] - moment_m[:, None, :]
                    )
                )
                mean = jnp.einsum("tn,tn->t", weights, forcing)
                eigenvalues = jnp.linalg.eigvalsh(regularized)
                condition = eigenvalues[:, -1] / jnp.maximum(eigenvalues[:, 0], 1e-300)
                ess = projected.ess_fraction[local]
                index = start + local
                projection_residual = float(
                    jnp.max(jnp.linalg.norm(projected.residual[local], axis=-1))
                )
                minimum_ress = float(jnp.min(ess))
                forcing_mean = float(jnp.max(jnp.abs(mean)))
                maximum_condition = float(jnp.max(condition))
                projection_valid = projection_residual <= float(
                    problem.forcing_config.projection_tolerance
                )
                ress_valid = minimum_ress >= MINIMUM_RESS
                forcing_valid = forcing_mean <= float(
                    problem.forcing_config.forcing_mean_tolerance
                )
                covariance_valid = maximum_condition <= float(
                    problem.forcing_config.max_covariance_condition
                )
                result["maximum_projection_residual"][index] = projection_residual
                result["minimum_ress"][index] = minimum_ress
                result["controlling_ress_time_index"][index] = int(jnp.argmin(ess))
                result["maximum_forcing_mean"][index] = forcing_mean
                result["maximum_covariance_condition"][index] = maximum_condition
                result["projection_valid"][index] = projection_valid
                result["ress_valid"][index] = ress_valid
                result["forcing_valid"][index] = forcing_valid
                result["covariance_valid"][index] = covariance_valid
                result["support_valid"][index] = bool(
                    projection_valid and ress_valid and forcing_valid and covariance_valid
                )
        return result


def _replicate_root(replicate: int) -> Path:
    return OUTPUT_ROOT / "replicates" / f"replicate_{replicate:02d}"


def _stage_paths(replicate: int, role: str) -> tuple[Path, Path]:
    root = _replicate_root(replicate)
    return root / f"{role}_results.npz", root / f"{role}_summary.json"


def _verify_stage(replicate: int, role: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    data_path, summary_path = _stage_paths(replicate, role)
    if not data_path.exists() and not summary_path.exists():
        return None
    if not data_path.exists() or not summary_path.exists():
        raise RuntimeError(f"incomplete sealed {role} stage for replicate {replicate}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("data_sha256") != file_sha256(data_path):
        raise RuntimeError(f"sealed {role} result changed for replicate {replicate}")
    with np.load(data_path, allow_pickle=False) as arrays:
        data = {name: np.asarray(arrays[name]) for name in arrays.files}
    return summary, data


def _stage_summary(
    replicate: int,
    role: str,
    data_path: Path,
    elapsed: float,
    candidate_count: int,
    bank_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "replicate_id": replicate,
        "role": role,
        "candidate_count": candidate_count,
        "data_path": str(data_path.relative_to(OUTPUT_ROOT)),
        "data_sha256": file_sha256(data_path),
        "bank_path": str(bank_path.relative_to(OUTPUT_ROOT)),
        "bank_sha256": file_sha256(bank_path),
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
        "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
        "elapsed_seconds": elapsed,
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
    }


def evaluate_replicates(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    freeze = freeze_candidates(cfg)
    manifest = freeze_bank_manifest(cfg)
    bank_inventory = generate_banks(cfg)
    if bank_inventory.get("candidate_evaluation_started") is not False:
        raise RuntimeError("bank inventory prospective-order contract changed")
    etas = np.asarray([row["eta"] for row in freeze["rows"]], dtype=np.float64)
    risks = np.asarray(
        [row["fixed_scientific_selection_risk"] for row in freeze["rows"]],
        dtype=np.float64,
    )
    base = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    evaluator = _FreshBankEvaluator(base.selection_problem)
    completed = []
    for record in manifest["replicates"]:
        replicate = int(record["replicate_id"])
        screen_cached = _verify_stage(replicate, "screen")
        if screen_cached is None:
            bank_path = _bank_path(replicate, "screen")
            bank = _load_bank(bank_path)
            started = time.perf_counter()
            screen = evaluator.evaluate(etas, bank)
            elapsed = time.perf_counter() - started
            screen_data_path, screen_summary_path = _stage_paths(replicate, "screen")
            _atomic_npz(
                screen_data_path,
                candidate_index=np.arange(len(etas), dtype=np.int32),
                **screen,
            )
            screen_summary = _stage_summary(
                replicate, "screen", screen_data_path, elapsed, len(etas), bank_path
            )
            screen_summary["law_candidate_diagnostics"] = {
                name: value[0].item() for name, value in screen.items()
            }
            _atomic_json(screen_summary_path, screen_summary)
            screen_cached = (screen_summary, screen)
            del bank
            screen_hit = False
        else:
            screen_hit = True
        screen_summary, screen = screen_cached

        audit_cached = _verify_stage(replicate, "audit")
        if audit_cached is None:
            relevant = np.flatnonzero(
                (risks <= selection_ceiling(LAW_RISK, 5.0)) & screen["support_valid"]
            )
            bank_path = _bank_path(replicate, "audit")
            bank = _load_bank(bank_path)
            started = time.perf_counter()
            subset = evaluator.evaluate(etas[relevant], bank)
            elapsed = time.perf_counter() - started
            count = len(etas)
            audit = {
                "audit_performed": np.zeros(count, dtype=bool),
                "maximum_projection_residual": np.full(count, np.nan),
                "minimum_ress": np.full(count, np.nan),
                "controlling_ress_time_index": np.full(count, -1, dtype=np.int16),
                "maximum_forcing_mean": np.full(count, np.nan),
                "maximum_covariance_condition": np.full(count, np.nan),
                "projection_valid": np.zeros(count, dtype=bool),
                "ress_valid": np.zeros(count, dtype=bool),
                "forcing_valid": np.zeros(count, dtype=bool),
                "covariance_valid": np.zeros(count, dtype=bool),
                "support_valid": np.zeros(count, dtype=bool),
                "robust_ress_pair": np.full(count, np.nan),
            }
            audit["audit_performed"][relevant] = True
            for name, values in subset.items():
                audit[name][relevant] = values
            audit["robust_ress_pair"][relevant] = np.minimum(
                screen["minimum_ress"][relevant], audit["minimum_ress"][relevant]
            )
            dual = np.stack(
                [
                    (risks <= selection_ceiling(LAW_RISK, allowance))
                    & screen["support_valid"]
                    & audit["support_valid"]
                    for allowance in ALLOWANCES
                ]
            )
            audit_data_path, audit_summary_path = _stage_paths(replicate, "audit")
            _atomic_npz(
                audit_data_path,
                candidate_index=np.arange(count, dtype=np.int32),
                dual_bank_eligible=dual,
                **audit,
            )
            audit_summary = _stage_summary(
                replicate, "audit", audit_data_path, elapsed, len(relevant), bank_path
            )
            audit_summary["audit_candidate_count"] = len(relevant)
            audit_summary["eligible_counts_by_allowance"] = {
                str(float(allowance)): int(np.sum(dual[index]))
                for index, allowance in enumerate(ALLOWANCES)
            }
            audit_summary["law_candidate_diagnostics"] = {
                name: value[0].item()
                for name, value in audit.items()
                if name != "dual_bank_eligible"
            }
            _atomic_json(audit_summary_path, audit_summary)
            audit_cached = (audit_summary, {**audit, "dual_bank_eligible": dual})
            del bank
            audit_hit = False
        else:
            audit_hit = True
        audit_summary, _ = audit_cached
        inventory_path = _replicate_root(replicate) / "replicate_inventory.json"
        if inventory_path.exists():
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            for row in inventory["artifacts"]:
                if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                    raise RuntimeError(f"replicate {replicate} artifact changed")
        else:
            paths = [
                *_stage_paths(replicate, "screen"),
                *_stage_paths(replicate, "audit"),
            ]
            inventory = {
                "schema_version": 1,
                "replicate_id": replicate,
                "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
                "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
                "artifacts": [
                    {
                        "path": str(path.relative_to(OUTPUT_ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in paths
                ],
            }
            _atomic_json(inventory_path, inventory)
        completed.append(
            {
                "replicate_id": replicate,
                "screen_cache_hit": screen_hit,
                "audit_cache_hit": audit_hit,
                "screen_seconds": screen_summary["elapsed_seconds"],
                "audit_seconds": audit_summary["elapsed_seconds"],
                "audit_candidate_count": audit_summary["audit_candidate_count"],
                "replicate_inventory_sha256": file_sha256(inventory_path),
            }
        )
        if progress is not None:
            progress(completed[-1], audit_summary["eligible_counts_by_allowance"])
    return {
        "replicate_count": len(completed),
        "completed": completed,
        "cache_hit_count": sum(
            row["screen_cache_hit"] and row["audit_cache_hit"] for row in completed
        ),
    }


def _distribution(values: Iterable[float], *, extended: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    names = ("minimum", "p05", "p10", "p25", "median", "p75", "p90", "p95", "maximum")
    if not len(array):
        result = {"count": 0, **{name: None for name in names}}
        if extended:
            result.update({"mean": None, "std": None})
        return result
    quantiles = np.quantile(array, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1])
    result = {
        "count": int(len(array)),
        **{name: float(value) for name, value in zip(names, quantiles, strict=True)},
    }
    if extended:
        result.update({"mean": float(np.mean(array)), "std": float(np.std(array, ddof=0))})
    return result


def _wilson_lower(passes: int, total: int = REPLICATE_COUNT) -> float:
    z = 1.6448536269514722
    p = passes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 2:
        return {"n": int(np.sum(mask)), "pearson": None, "spearman": None}
    pearson = pearsonr(left[mask], right[mask]).statistic
    spearman = spearmanr(left[mask], right[mask]).statistic
    return {"n": int(np.sum(mask)), "pearson": float(pearson), "spearman": float(spearman)}


def _failure_mode(
    index: int,
    ceiling: float,
    risk: np.ndarray,
    screen: dict[str, np.ndarray],
    audit: dict[str, np.ndarray],
) -> str:
    if risk[index] > ceiling:
        return "outside_risk_ceiling"
    if not screen["projection_valid"][index]:
        return "screen_projection_failure"
    if not screen["forcing_valid"][index] or not screen["covariance_valid"][index]:
        return "screen_forcing_or_covariance_failure"
    if not screen["ress_valid"][index]:
        return "screen_ress_below_0p05"
    if not audit["audit_performed"][index]:
        return "audit_not_performed_unexpected"
    if not audit["projection_valid"][index]:
        return "audit_projection_failure"
    if not audit["forcing_valid"][index] or not audit["covariance_valid"][index]:
        return "audit_forcing_or_covariance_failure"
    if not audit["ress_valid"][index]:
        return "audit_ress_below_0p05"
    return "complete_dual_bank_pass"


def _maxmin_shortlist(
    indices: list[int],
    rows: list[dict[str, Any]],
    pass_counts: np.ndarray,
    robust: np.ndarray,
    box: Any,
    maximum: int = 10,
) -> list[int]:
    def median(index: int) -> float:
        values = robust[:, index]
        values = values[np.isfinite(values)]
        return -math.inf if not len(values) else float(np.median(values))

    remaining = sorted(
        indices,
        key=lambda index: (-int(pass_counts[index]), -median(index), rows[index]["candidate_id"]),
    )
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < maximum:
        chosen = min(
            remaining,
            key=lambda index: (
                -min(
                    _symmetry_aware_distance(rows[index]["eta"], rows[old]["eta"], box)
                    for old in selected
                ),
                -int(pass_counts[index]),
                -median(index),
                rows[index]["candidate_id"],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _frequency(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in np.asarray(values).reshape(-1) if int(value) >= 0)
    return {str(key): counts[key] for key in sorted(counts)}


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    if SUMMARY_PATH.exists() or INVENTORY_PATH.exists():
        if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
            raise RuntimeError("incomplete fresh-bank summary/inventory pair")
        _verify_coverage_sources()
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        if inventory.get("source_sha256") != _source_hashes():
            raise RuntimeError("fresh-bank implementation/config changed after completion")
        for row in inventory["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"fresh-bank cached artifact changed: {row['path']}")
        _official_v3_firewall()
        return {**json.loads(SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}

    started = time.perf_counter()
    freeze = freeze_candidates(cfg)
    manifest = freeze_bank_manifest(cfg)
    bank_inventory = generate_banks(cfg)
    evaluation = evaluate_replicates(cfg)
    if evaluation["replicate_count"] != REPLICATE_COUNT:
        raise RuntimeError("not all prospectively frozen replicates are complete")
    rows = freeze["rows"]
    count = len(rows)
    risks = np.asarray([row["fixed_scientific_selection_risk"] for row in rows])
    screens = []
    audits = []
    for replicate in range(REPLICATE_COUNT):
        screen_stage = _verify_stage(replicate, "screen")
        audit_stage = _verify_stage(replicate, "audit")
        if screen_stage is None or audit_stage is None:
            raise RuntimeError(f"replicate {replicate} incomplete during summary")
        screens.append(screen_stage[1])
        audits.append(audit_stage[1])
    screen_arrays = {
        name: np.stack([row[name] for row in screens])
        for name in screens[0]
        if name != "candidate_index"
    }
    audit_arrays = {
        name: np.stack([row[name] for row in audits])
        for name in audits[0]
        if name != "candidate_index"
    }
    dual = audit_arrays["dual_bank_eligible"]  # [replicate, allowance, candidate]
    pass_counts = np.sum(dual, axis=0)  # [allowance, candidate]
    robust = audit_arrays["robust_ress_pair"]

    candidate_rows = []
    for index, source in enumerate(rows):
        audit_values = audit_arrays["minimum_ress"][:, index]
        robust_values = robust[:, index]
        allowance_stats = []
        for allowance_index, allowance in enumerate(ALLOWANCES):
            passes = int(pass_counts[allowance_index, index])
            allowance_stats.append(
                {
                    "allowance_percent": allowance,
                    "screen_pass_count": int(
                        np.sum(
                            (risks[index] <= selection_ceiling(LAW_RISK, allowance))
                            & screen_arrays["support_valid"][:, index]
                        )
                    ),
                    "audit_performed_count": int(np.sum(audit_arrays["audit_performed"][:, index])),
                    "audit_support_pass_count": int(np.sum(audit_arrays["support_valid"][:, index])),
                    "complete_dual_bank_pass_count": passes,
                    "pass_fraction": passes / REPLICATE_COUNT,
                    "wilson_one_sided_95_lower": _wilson_lower(passes),
                }
            )
        candidate_rows.append(
            {
                "candidate_id": source["candidate_id"],
                "eta": source["eta"],
                "source_pool": source["source_pool"],
                "generation_method": source["generation_method"],
                "anchor_id": source["anchor_id"],
                "fixed_scientific_selection_risk": source["fixed_scientific_selection_risk"],
                "risk_increase_pct": 100.0 * (source["fixed_scientific_selection_risk"] / LAW_RISK - 1.0),
                "fresh_replicates_evaluated": REPLICATE_COUNT,
                "allowances": allowance_stats,
                "fresh_audit_ress_performed_only": _distribution(audit_values, extended=True),
                "fresh_robust_ress_performed_only": _distribution(robust_values, extended=True),
            }
        )
    candidate_payload = {
        "schema_version": 1,
        "version": VERSION,
        "audit_quantiles_use_performed_audits_only": True,
        "threshold_pass_fraction_counts_screen_failures_as_replicate_failures": True,
        "candidate_count": count,
        "rows": candidate_rows,
    }
    _atomic_json(CANDIDATE_SUMMARY_PATH, candidate_payload)

    allowance_rows = []
    diversity = {}
    replicate_counts = np.sum(dual, axis=2)  # [replicate, allowance]
    for allowance_index, allowance in enumerate(ALLOWANCES):
        counts = pass_counts[allowance_index]
        maximum = int(np.max(counts))
        tied = np.flatnonzero(counts == maximum)
        def robust_median(index: int) -> float:
            values = robust[:, index]
            values = values[np.isfinite(values)]
            return -math.inf if not len(values) else float(np.median(values))
        best = min(
            tied,
            key=lambda index: (-robust_median(index), rows[index]["candidate_id"]),
        )
        best_distribution = _distribution(robust[:, best], extended=True)
        row = {
            "allowance_percent": allowance,
            "risk_ceiling": selection_ceiling(LAW_RISK, allowance),
            "candidates_ever_passing": int(np.sum(counts >= 1)),
            "candidates_ge_16_of_32": int(np.sum(counts >= 16)),
            "candidates_ge_24_of_32": int(np.sum(counts >= 24)),
            "candidates_ge_28_of_32": int(np.sum(counts >= 28)),
            "candidates_ge_30_of_32": int(np.sum(counts >= 30)),
            "candidates_32_of_32": int(np.sum(counts == 32)),
            "maximum_pass_count": maximum,
            "maximum_pass_fraction": maximum / REPLICATE_COUNT,
            "best_candidate_id": rows[best]["candidate_id"],
            "candidates_tied_at_maximum": len(tied),
            "best_candidate_median_robust_ress": best_distribution["median"],
            "best_candidate_p10_robust_ress": best_distribution["p10"],
            "replicate_survivor_distribution": _distribution(replicate_counts[:, allowance_index]),
            "replicates_with_zero": int(np.sum(replicate_counts[:, allowance_index] == 0)),
            "replicates_with_ge_1": int(np.sum(replicate_counts[:, allowance_index] >= 1)),
            "replicates_with_ge_5": int(np.sum(replicate_counts[:, allowance_index] >= 5)),
            "replicates_with_ge_10": int(np.sum(replicate_counts[:, allowance_index] >= 10)),
            "survivors_by_replicate": [int(value) for value in replicate_counts[:, allowance_index]],
        }
        allowance_rows.append(row)
        diversity[str(float(allowance))] = {}
        for threshold in (24, 28, 30):
            eligible = list(np.flatnonzero(counts >= threshold))
            selected = _maxmin_shortlist(
                eligible, rows, counts, robust, freeze["box"], maximum=10
            )
            diversity[str(float(allowance))][f"ge_{threshold}_of_32"] = [
                {
                    "candidate_id": rows[index]["candidate_id"],
                    "eta": rows[index]["eta"],
                    "pass_count": int(counts[index]),
                    "pass_fraction": int(counts[index]) / REPLICATE_COUNT,
                    "median_robust_ress": _distribution(robust[:, index])["median"],
                    "p10_robust_ress": _distribution(robust[:, index])["p10"],
                    "fixed_scientific_selection_risk": risks[index],
                    "risk_increase_pct": 100.0 * (risks[index] / LAW_RISK - 1.0),
                    "generation_source": rows[index]["generation_method"],
                }
                for index in selected
            ]
    allowance_payload = {
        "schema_version": 1,
        "version": VERSION,
        "replicate_count": REPLICATE_COUNT,
        "allowances": allowance_rows,
        "symmetry_aware_diversity": diversity,
    }
    _atomic_json(ALLOWANCE_SUMMARY_PATH, allowance_payload)

    old_by_id = {row["candidate_id"]: index for index, row in enumerate(rows)}
    witness_rows = []
    failure_partition = {}
    for candidate_id in OLD_HALF_PERCENT_WITNESSES:
        index = old_by_id[candidate_id]
        modes = Counter(
            _failure_mode(
                index,
                selection_ceiling(LAW_RISK, 0.5),
                risks,
                screens[replicate],
                audits[replicate],
            )
            for replicate in range(REPLICATE_COUNT)
        )
        if sum(modes.values()) != REPLICATE_COUNT:
            raise RuntimeError(f"failure modes do not partition {candidate_id}")
        coverage_audit = json.loads((COVERAGE_ROOT / "audit_results.json").read_text(encoding="utf-8"))
        old = next(row for row in coverage_audit["rows"] if row["candidate_id"] == candidate_id)
        passes = int(pass_counts[0, index])
        witness_rows.append(
            {
                "candidate_id": candidate_id,
                "old_bank_risk": old["scientific_selection_risk"],
                "old_screen_minimum_ress": old["screen"]["minimum_ress"],
                "old_audit_minimum_ress": old["audit"]["minimum_ress"],
                "fresh_0p5_pass_count": passes,
                "fresh_0p5_pass_fraction": passes / REPLICATE_COUNT,
                "fresh_robust_ress_performed_only": _distribution(robust[:, index], extended=True),
                "failure_modes": dict(sorted(modes.items())),
            }
        )
        failure_partition[candidate_id] = dict(sorted(modes.items()))

    disagreement_rows = []
    for allowance in ALLOWANCES:
        inside = risks <= selection_ceiling(LAW_RISK, allowance)
        inside_matrix = np.broadcast_to(inside, screen_arrays["minimum_ress"].shape)
        screen_ge = screen_arrays["minimum_ress"] >= MINIMUM_RESS
        audit_done = audit_arrays["audit_performed"]
        audit_ge = audit_arrays["minimum_ress"] >= MINIMUM_RESS
        both_values = inside_matrix & audit_done
        delta = audit_arrays["minimum_ress"][both_values] - screen_arrays["minimum_ress"][both_values]
        disagreement_rows.append(
            {
                "allowance_percent": allowance,
                "candidate_replicate_pairs_inside_risk": int(np.sum(inside_matrix)),
                "screen_ress_ge_and_audit_ress_ge": int(np.sum(inside_matrix & screen_ge & audit_done & audit_ge)),
                "screen_ress_ge_and_audit_ress_lt": int(np.sum(inside_matrix & screen_ge & audit_done & ~audit_ge)),
                "screen_ress_lt": int(np.sum(inside_matrix & ~screen_ge)),
                "screen_ress_ge_but_audit_not_performed_due_other_screen_gate": int(np.sum(inside_matrix & screen_ge & ~audit_done)),
                "audit_minus_screen_ress": {
                    "mean": None if not len(delta) else float(np.mean(delta)),
                    "median": None if not len(delta) else float(np.median(delta)),
                    "p10": None if not len(delta) else float(np.quantile(delta, 0.1)),
                    "p90": None if not len(delta) else float(np.quantile(delta, 0.9)),
                },
                "screen_audit_association": _correlation(
                    screen_arrays["minimum_ress"][both_values],
                    audit_arrays["minimum_ress"][both_values],
                ),
            }
        )
    failure_payload = {
        "schema_version": 1,
        "version": VERSION,
        "old_0p5_percent_witnesses": witness_rows,
        "witness_failure_mode_partitions": failure_partition,
        "screen_audit_instability": disagreement_rows,
    }
    _atomic_json(FAILURE_SUMMARY_PATH, failure_payload)

    half_inside = risks <= selection_ceiling(LAW_RISK, 0.5)
    high_pass = pass_counts[0] >= 24
    time_payload = {
        "schema_version": 1,
        "version": VERSION,
        "physical_time_node_count": int(cfg["physics"]["time_nodes"]),
        "screen_all_evaluated": _frequency(screen_arrays["controlling_ress_time_index"]),
        "audit_all_performed": _frequency(
            audit_arrays["controlling_ress_time_index"][audit_arrays["audit_performed"]]
        ),
        "screen_0p5_percent_candidates": _frequency(
            screen_arrays["controlling_ress_time_index"][:, half_inside]
        ),
        "audit_0p5_percent_performed": _frequency(
            audit_arrays["controlling_ress_time_index"][:, half_inside]
        ),
        "screen_0p5_percent_ge_24_of_32_candidates": _frequency(
            screen_arrays["controlling_ress_time_index"][:, high_pass]
        ),
        "audit_0p5_percent_ge_24_of_32_candidates": _frequency(
            audit_arrays["controlling_ress_time_index"][:, high_pass]
        ),
    }
    _atomic_json(TIME_NODE_SUMMARY_PATH, time_payload)

    screen_index = {
        "schema_version": 1,
        "replicate_count": REPLICATE_COUNT,
        "rows": [
            {
                "replicate_id": replicate,
                "data_sha256": _verify_stage(replicate, "screen")[0]["data_sha256"],
                "summary_sha256": file_sha256(_stage_paths(replicate, "screen")[1]),
            }
            for replicate in range(REPLICATE_COUNT)
        ],
    }
    audit_index = {
        "schema_version": 1,
        "replicate_count": REPLICATE_COUNT,
        "rows": [
            {
                "replicate_id": replicate,
                "data_sha256": _verify_stage(replicate, "audit")[0]["data_sha256"],
                "summary_sha256": file_sha256(_stage_paths(replicate, "audit")[1]),
            }
            for replicate in range(REPLICATE_COUNT)
        ],
    }
    _atomic_json(SCREEN_INDEX_PATH, screen_index)
    _atomic_json(AUDIT_INDEX_PATH, audit_index)

    half = allowance_rows[0]
    half_diverse = diversity["0.5"]["ge_24_of_32"]
    if (
        half["candidates_ge_24_of_32"] >= 3
        and half["replicate_survivor_distribution"]["median"] >= 3
        and half["replicates_with_zero"] <= 3
        and len(half_diverse) >= 2
        and (half["best_candidate_p10_robust_ress"] or 0.0) >= MINIMUM_RESS
    ):
        label = "STRONG_FRESH_BANK_SUPPORT"
        recommendation = "Evidence is sufficient to consider a separately named, prospectively frozen official partial-curve Galerkin protocol; do not launch it automatically."
    elif (
        half["candidates_ge_16_of_32"] >= 1
        and half["replicates_with_ge_1"] >= 16
        and len(diversity["1.0"]["ge_24_of_32"]) >= 2
    ):
        label = "MODERATE_FRESH_BANK_SUPPORT"
        recommendation = "Evidence may justify considering a separately frozen official partial-curve protocol, with fresh official banks and all selection/validation firewalls."
    elif half["candidates_ever_passing"] > 0 or allowance_rows[1]["candidates_ge_16_of_32"] > 0:
        label = "WEAK_FRESH_BANK_SUPPORT"
        recommendation = "Another official Full run is not yet justified; low-risk fresh-bank stability needs stronger evidence."
    else:
        label = "NO_FRESH_BANK_SUPPORT"
        recommendation = "Another official Full run is not justified by this fresh-bank study."
    interpretation = {
        "label": label,
        "development_only": True,
        "evidence": {
            "0p5_candidates_ge_24_of_32": half["candidates_ge_24_of_32"],
            "0p5_median_survivors_per_replicate": half["replicate_survivor_distribution"]["median"],
            "0p5_replicates_with_zero": half["replicates_with_zero"],
            "0p5_diverse_ge_24_shortlist_count": len(half_diverse),
            "0p5_best_candidate_p10_robust_ress": half["best_candidate_p10_robust_ress"],
        },
        "recommended_next_scientific_step": recommendation,
        "future_official_requirements": [
            "fresh official banks",
            "independent allowance failure/success",
            "frozen improved candidate-generation strategy",
            "Tangent and Full as separate branches",
            "K=280 Full only after cheap eligibility gates",
            "selection freeze before validation",
        ],
    }
    summary = {
        "schema_version": 1,
        "version": VERSION,
        "purpose": "development-only fresh-bank robustness of the frozen 4,433-candidate pool",
        "development_only": True,
        "candidate_pool_frozen": True,
        "candidate_count": count,
        "replicate_count": REPLICATE_COUNT,
        "total_bank_count": 2 * REPLICATE_COUNT,
        "candidate_freeze_sha256": file_sha256(CANDIDATE_FREEZE_PATH),
        "bank_manifest_sha256": file_sha256(BANK_MANIFEST_PATH),
        "bank_inventory_sha256": file_sha256(BANK_INVENTORY_PATH),
        "scientific_constants": {
            "fixed_law_selection_risk": LAW_RISK,
            "allowances_percent": list(ALLOWANCES),
            "minimum_ress": MINIMUM_RESS,
            "projection_tolerance": 2e-6,
            "forcing_mean_tolerance": 2e-7,
            "maximum_covariance_condition": 1e10,
            "dtype": "float64",
            "screen_N": SCREEN_N,
            "audit_N": AUDIT_N,
        },
        "risk_semantics": freeze["risk_semantics"],
        "allowances": allowance_rows,
        "old_0p5_percent_witnesses": witness_rows,
        "development_interpretation": interpretation,
        "timings": {
            "bank_generation_seconds_recorded": sum(row["generation_seconds"] for row in bank_inventory["banks"] if not row["cache_hit"]),
            "screen_seconds_by_replicate": [row["screen_seconds"] for row in evaluation["completed"]],
            "audit_seconds_by_replicate": [row["audit_seconds"] for row in evaluation["completed"]],
            "summary_seconds": time.perf_counter() - started,
        },
        "total_candidate_screen_trajectories": count * REPLICATE_COUNT,
        "total_candidate_audit_trajectories": sum(row["audit_candidate_count"] for row in evaluation["completed"]),
        "cache_resume": evaluation,
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "eigensolve_run": False,
        "deep_ritz_run": False,
        "official_protocol_created": False,
        "selection_frozen": False,
        "firewall_after": _official_v3_firewall(),
    }
    _atomic_json(SUMMARY_PATH, summary)
    top_paths = (
        CANDIDATE_FREEZE_PATH,
        BANK_MANIFEST_PATH,
        BANK_INVENTORY_PATH,
        SCREEN_INDEX_PATH,
        AUDIT_INDEX_PATH,
        CANDIDATE_SUMMARY_PATH,
        ALLOWANCE_SUMMARY_PATH,
        FAILURE_SUMMARY_PATH,
        TIME_NODE_SUMMARY_PATH,
        SUMMARY_PATH,
    )
    bank_paths = sorted((OUTPUT_ROOT / "banks").glob("*.npz"))
    replicate_paths = sorted((OUTPUT_ROOT / "replicates").rglob("*"))
    artifacts = [
        {
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (*top_paths, *bank_paths, *replicate_paths)
        if path.is_file()
    ]
    inventory = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_sha256": _source_hashes(),
        "coverage_source_hashes": EXPECTED_COVERAGE_HASHES,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    _atomic_json(INVENTORY_PATH, inventory)
    _verify_coverage_sources()
    _official_v3_firewall()
    return summary


def run(cfg: dict[str, Any], bank_progress: Any | None = None, replicate_progress: Any | None = None) -> dict[str, Any]:
    freeze_candidates(cfg)
    freeze_bank_manifest(cfg)
    generate_banks(cfg, progress=bank_progress)
    evaluate_replicates(cfg, progress=replicate_progress)
    return summarize(cfg)


__all__ = [
    "ALLOWANCES",
    "AUDIT_INDEX_PATH",
    "AUDIT_N",
    "BANK_INVENTORY_PATH",
    "BANK_MANIFEST_PATH",
    "CANDIDATE_FREEZE_PATH",
    "CANDIDATE_SUMMARY_PATH",
    "EXPECTED_COVERAGE_HASHES",
    "FAILURE_SUMMARY_PATH",
    "INVENTORY_PATH",
    "LAW_RISK",
    "OLD_HALF_PERCENT_WITNESSES",
    "OUTPUT_ROOT",
    "REPLICATE_COUNT",
    "SCREEN_INDEX_PATH",
    "SCREEN_N",
    "SUMMARY_PATH",
    "TIME_NODE_SUMMARY_PATH",
    "VERSION",
    "derive_seed",
    "evaluate_replicates",
    "freeze_bank_manifest",
    "freeze_candidates",
    "generate_banks",
    "run",
    "summarize",
]
