"""Reuse-only pooled/cross-fit population-overlap diagnostic.

This module reads the sealed nested-N master banks and candidate panel.  It
contains no reference generator, candidate generator, optimization branch,
Galerkin assembly, eigensolve, Deep Ritz path, validation loader, or official
protocol writer.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as scipy_stats

from mfsi.projection import EmpiricalIProjector

from .full_gradient import reconstruct_moments
from .galerkin_only_data import load_selection_galerkin_data
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v3_common import ROOT, file_sha256


VERSION = "skyrmion_galerkin_dev_population_overlap_v1"
SOURCE_VERSION = "skyrmion_galerkin_dev_ress_n_convergence_v1"
SOURCE_ROOT = ROOT / "outputs" / SOURCE_VERSION
OUTPUT_ROOT = ROOT / "outputs" / VERSION

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
BASE_AUDIT_PATH = OUTPUT_ROOT / "base_measure_audit.json"
NODE_MANIFEST_PATH = OUTPUT_ROOT / "node_analysis_manifest.json"
CROSSFIT_MANIFEST_PATH = OUTPUT_ROOT / "crossfit_manifest.json"
BOOTSTRAP_MANIFEST_PATH = OUTPUT_ROOT / "bootstrap_manifest.json"
BOOTSTRAP_COUNTS_PATH = OUTPUT_ROOT / "bootstrap_block_counts.npz"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

BANK_COUNT = 32
MASTER_N = 65536
POOLED_N = BANK_COUNT * MASTER_N
PANEL_COUNT = 64
HIGH_PASS_COUNT = 55
CONTROL_COUNT = 8
FEATURE_DIMENSION = 4
PRIMARY_NODES = (7, 8)
CANDIDATE_BATCH_SIZE = 8
CROSSFIT_FOLDS = 4
CROSSFIT_SEED = 2026082601
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026082602
MINIMUM_RESS = 0.05

EXPECTED_SOURCE_HASHES = {
    "source_seal.json": "c812abeaa6fdda3242ab175571f9b77ff2cf68372028aa3689571f8cf95b74db",
    "candidate_panel.json": "f2a6437899383072634c4c2c596e35c49275b6fb47ee9b05a3425d35a81a0189",
    "master_bank_manifest.json": "ca6fdeec773408a704c262c244d0b8783522dc0b70435dc205cd0d6cd6ea11fe",
    "master_bank_inventory.json": "79d26a677dd29637766112a54142d01c104714b88466e91853b66e095146cd13",
    "law_convergence.json": "1c7a2c7acd27166012a3889eaa5a656e91342485a0f1e0d3e15da72c1fc45d88",
    "candidate_convergence.json": "84c6b2bcbc8e2b9147b95be2be4640654773c5d3510d5601ec11a7e7cb180ec5",
    "time7_diagnostics.json": "b7d79707408ff48b47d04e6d8424151e01ada78f61f6314d1ca5c36544934d73",
    "threshold_flip_summary.json": "7b33ec79a901cca8e0a4bf14efb30b8745829539ce5a21be6d826721030c8232",
    "summary.json": "295975b93e8b4db11007de3f0adf1afd5c1b06df7efea194d9251cbefd0c3a2a",
    "inventory.json": "2d72c01654873c0ea5bb98500b227851e71d8e657e98fbeb9a20c32765ab0c4f",
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _inside(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must be beneath {root}: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Any) -> None:
    path = _inside(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
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
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **{name: np.asarray(value) for name, value in arrays.items()})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _distribution(values: Any, *, mean_std: bool = True) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result = {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }
    if mean_std:
        result.update(mean=float(np.mean(array)), std=float(np.std(array)))
    return result


def _code_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in (
            "population_overlap.py",
            "population_overlap_run.py",
            "test_population_overlap.py",
            "config.json",
        )
    }


def _source_inventory() -> dict[str, Any]:
    return json.loads((SOURCE_ROOT / "master_bank_inventory.json").read_text(encoding="utf-8"))


def _bank_rows() -> list[dict[str, Any]]:
    rows = _source_inventory()["banks"]
    if len(rows) != BANK_COUNT:
        raise RuntimeError("source master-bank count changed")
    return rows


def _bank_path(row: dict[str, Any]) -> Path:
    return SOURCE_ROOT / row["path"]


def verify_and_freeze_sources() -> dict[str, Any]:
    started = time.perf_counter()
    observed = {name: file_sha256(SOURCE_ROOT / name) for name in EXPECTED_SOURCE_HASHES}
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("sealed nested-N source differs")
    bank_rows = _bank_rows()
    for row in bank_rows:
        path = _bank_path(row)
        if int(row["N"]) != MASTER_N or file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"sealed master bank differs: {path}")
    panel = json.loads((SOURCE_ROOT / "candidate_panel.json").read_text(encoding="utf-8"))
    if panel["candidate_count"] != PANEL_COUNT or len(panel["rows"]) != PANEL_COUNT:
        raise RuntimeError("source candidate panel changed")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_hashes": observed,
        "analysis_source_hashes": _code_hashes(),
        "verified_master_bank_count": BANK_COUNT,
        "verified_master_bank_hashes": {row["path"]: row["sha256"] for row in bank_rows},
        "candidate_panel_sha256": observed["candidate_panel.json"],
        "source_verification_seconds": time.perf_counter() - started,
        "new_reference_samples_generated": 0,
        "candidate_generation_permitted": False,
        "reference_generation_permitted": False,
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def _panel() -> dict[str, Any]:
    return json.loads((SOURCE_ROOT / "candidate_panel.json").read_text(encoding="utf-8"))


def _etas() -> np.ndarray:
    return np.asarray([row["eta"] for row in _panel()["rows"]], dtype=np.float64)


def _load_problem(cfg: dict[str, Any]) -> Any:
    return load_selection_galerkin_data(cfg, ARTIFACT_DIR).selection_problem


def _candidate_targets(problem: Any, etas: np.ndarray) -> np.ndarray:
    function = jax.jit(jax.vmap(lambda eta: reconstruct_moments(eta, problem).values))
    return np.asarray(function(jnp.asarray(etas, dtype=jnp.float64)), dtype=np.float64)


def _stable_logmeanexp(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    count = math.prod(array.shape[item] for item in ((axis,) if isinstance(axis, int) else axis))
    result = maximum + np.log(np.sum(np.exp(array - maximum), axis=axis, keepdims=True)) - math.log(count)
    return np.squeeze(result, axis=axis)


def audit_base_measure(cfg: dict[str, Any]) -> dict[str, Any]:
    verify_and_freeze_sources()
    if BASE_AUDIT_PATH.exists():
        payload = json.loads(BASE_AUDIT_PATH.read_text(encoding="utf-8"))
        if payload["source_seal_sha256"] != file_sha256(SOURCE_SEAL_PATH):
            raise RuntimeError("base-measure audit source seal changed")
        return {**payload, "cache_hit": True}

    expected = np.float64(1.0 / MASTER_N)
    maximum_weight_error = 0.0
    maximum_sum_error = 0.0
    dtype_set: set[str] = set()
    for row in _bank_rows():
        with np.load(_bank_path(row), allow_pickle=False) as arrays:
            weights = np.asarray(arrays["base_weights"])
            dtype_set.add(str(weights.dtype))
            maximum_weight_error = max(maximum_weight_error, float(np.max(np.abs(weights - expected))))
            maximum_sum_error = max(maximum_sum_error, float(np.max(np.abs(np.sum(weights, axis=1) - 1.0))))
    if dtype_set != {"float64"} or maximum_weight_error != 0.0 or maximum_sum_error != 0.0:
        raise RuntimeError("master-bank base measure is not exact uniform float64")

    problem = _load_problem(cfg)
    etas = _etas()[:CANDIDATE_BATCH_SIZE]
    targets = _candidate_targets(problem, etas)
    first_row = _bank_rows()[0]
    with np.load(_bank_path(first_row), allow_pickle=False) as arrays:
        configurations = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        base_weights = jnp.asarray(arrays["base_weights"], dtype=jnp.float64)
    feature_function = jax.jit(jax.vmap(lambda eta: problem.family.features(configurations, eta)))
    features = feature_function(jnp.asarray(etas, dtype=jnp.float64))
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    projected = projector.project_candidate_trajectories(
        features, base_weights, jnp.asarray(targets, dtype=jnp.float64)
    )
    scores = jnp.einsum("ctnm,ctm->ctn", features, projected.lam)
    logz1 = jax.scipy.special.logsumexp(scores, axis=-1) - math.log(MASTER_N)
    logz2 = jax.scipy.special.logsumexp(2.0 * scores, axis=-1) - math.log(MASTER_N)
    identity = jnp.exp(2.0 * logz1 - logz2)
    identity_error = float(np.max(np.abs(np.asarray(identity) - np.asarray(projected.ess_fraction))))
    with np.load(
        SOURCE_ROOT / "results" / "replicate_00" / "bank_A_N65536.npz",
        allow_pickle=False,
    ) as sealed:
        sealed_ress = np.asarray(sealed["ress_trajectory"][:CANDIDATE_BATCH_SIZE])
    sealed_error = float(np.max(np.abs(np.asarray(projected.ess_fraction) - sealed_ress)))
    if identity_error > 2e-12 or sealed_error > 2e-12:
        raise RuntimeError("finite-bank rESS/Renyi identity did not reproduce authoritative results")
    payload = {
        "schema_version": 1,
        "development_only": True,
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "base_measure": "uniform empirical reference measure",
        "base_weight_dtype": "float64",
        "bank_count_checked": BANK_COUNT,
        "time_node_count_checked": 13,
        "expected_weight": float(expected),
        "maximum_uniform_weight_discrepancy": maximum_weight_error,
        "maximum_weight_sum_discrepancy": maximum_sum_error,
        "identity": "rESS = exp(2*log(mean(exp(s))) - log(mean(exp(2*s))))",
        "identity_candidates_checked": CANDIDATE_BATCH_SIZE,
        "identity_nodes_checked": 13,
        "maximum_identity_discrepancy": identity_error,
        "maximum_authoritative_sealed_result_discrepancy": sealed_error,
        "identity_verified": True,
        "weighted_analogue_required": False,
    }
    _atomic_json(BASE_AUDIT_PATH, payload)
    return payload


def freeze_node_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    audit_base_measure(cfg)
    if NODE_MANIFEST_PATH.exists():
        payload = json.loads(NODE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload["base_measure_audit_sha256"] != file_sha256(BASE_AUDIT_PATH):
            raise RuntimeError("node manifest base-measure seal changed")
        return {**payload, "cache_hit": True}
    counts = np.zeros((PANEL_COUNT, 13), dtype=np.int64)
    for replicate in range(16):
        for role in ("A", "B"):
            path = SOURCE_ROOT / "results" / f"replicate_{replicate:02d}" / f"bank_{role}_N65536.npz"
            with np.load(path, allow_pickle=False) as arrays:
                controlling = np.asarray(arrays["controlling_time_index"], dtype=np.int64)
            counts[np.arange(PANEL_COUNT), controlling] += 1
    additional = sorted(
        node for node in range(13)
        if node not in PRIMARY_NODES and bool(np.any(counts[:, node] >= 2))
    )
    nodes = list(PRIMARY_NODES) + additional
    interpretation_rule = {
        "candidate_above": "95% lower block endpoint >0.05 and cross-fit minimum >0.05",
        "candidate_below": "95% upper block endpoint <0.05 and cross-fit maximum <0.05",
        "candidate_intersects": "neither entirely above nor entirely below",
        "substantial_fraction": 0.50,
        "heterogeneous_fraction": 0.20,
        "gate_intersection_fraction": 0.40,
        "labels": {
            "POPULATION_OVERLAP_CLEARLY_ABOVE_GATE": "Law nodes 7/8 and >=50% of high-pass minimum-node results are above",
            "POPULATION_OVERLAP_BELOW_GATE": "Law nodes 7/8 and >=50% of high-pass minimum-node results are below",
            "GATE_INTERSECTS_RELEVANT_POPULATION": "Law intersects or >=40% of high-pass minimum-node intervals intersect, absent strong above/below heterogeneity",
            "MIXED_POPULATION_OVERLAP": "Law nodes span above and below, >=20% of high-pass rows are above and >=20% below, or remaining evidence is materially mixed",
        },
    }
    payload = {
        "schema_version": 1,
        "development_only": True,
        "base_measure_audit_sha256": file_sha256(BASE_AUDIT_PATH),
        "candidate_panel_sha256": file_sha256(SOURCE_ROOT / "candidate_panel.json"),
        "primary_nodes": list(PRIMARY_NODES),
        "additional_nodes": additional,
        "analyzed_nodes": nodes,
        "inclusion_rule": "primary 7/8 plus any other N65536 argmin on >=2/32 banks for any frozen candidate",
        "candidate_node_argmin_counts": counts.tolist(),
        "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "batch_size_memory_basis": "three nodes x eight candidates x 32 banks x 65536 samples x four float64 features ~= 6.0 GiB",
        "interpretation_rule": interpretation_rule,
        "frozen_before_pooled_computation": True,
    }
    _atomic_json(NODE_MANIFEST_PATH, payload)
    return payload


def freeze_crossfit_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    nodes = freeze_node_manifest(cfg)
    if CROSSFIT_MANIFEST_PATH.exists():
        payload = json.loads(CROSSFIT_MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload["node_analysis_manifest_sha256"] != file_sha256(NODE_MANIFEST_PATH):
            raise RuntimeError("cross-fit node manifest seal changed")
        return {**payload, "cache_hit": True}
    bank_rows = _bank_rows()
    permutation = np.random.default_rng(CROSSFIT_SEED).permutation(BANK_COUNT)
    folds = []
    for fold_index, held_indices in enumerate(permutation.reshape(CROSSFIT_FOLDS, -1)):
        held = sorted(int(value) for value in held_indices)
        train = [index for index in range(BANK_COUNT) if index not in held]
        folds.append({
            "fold": fold_index,
            "training_bank_indices": train,
            "held_out_bank_indices": held,
            "training_bank_paths": [bank_rows[index]["path"] for index in train],
            "held_out_bank_paths": [bank_rows[index]["path"] for index in held],
        })
    payload = {
        "schema_version": 1,
        "development_only": True,
        "node_analysis_manifest_sha256": file_sha256(NODE_MANIFEST_PATH),
        "seed": CROSSFIT_SEED,
        "fold_count": CROSSFIT_FOLDS,
        "training_banks_per_fold": 24,
        "held_out_banks_per_fold": 8,
        "folds": folds,
        "folds_sha256": _payload_sha256(folds),
        "frozen_before_crossfit": True,
    }
    _atomic_json(CROSSFIT_MANIFEST_PATH, payload)
    return payload


def freeze_bootstrap_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    freeze_crossfit_manifest(cfg)
    if BOOTSTRAP_MANIFEST_PATH.exists():
        payload = json.loads(BOOTSTRAP_MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload["crossfit_manifest_sha256"] != file_sha256(CROSSFIT_MANIFEST_PATH):
            raise RuntimeError("bootstrap cross-fit seal changed")
        if payload["block_counts_sha256"] != file_sha256(BOOTSTRAP_COUNTS_PATH):
            raise RuntimeError("bootstrap block counts changed")
        return {**payload, "cache_hit": True}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, BANK_COUNT, size=(BOOTSTRAP_REPLICATES, BANK_COUNT), dtype=np.int16)
    counts = np.zeros((BOOTSTRAP_REPLICATES, BANK_COUNT), dtype=np.int16)
    for bank_index in range(BANK_COUNT):
        counts[:, bank_index] = np.sum(draws == bank_index, axis=1)
    if not np.all(np.sum(counts, axis=1) == BANK_COUNT):
        raise RuntimeError("bootstrap block count construction failed")
    _atomic_npz(BOOTSTRAP_COUNTS_PATH, block_counts=counts)
    payload = {
        "schema_version": 1,
        "development_only": True,
        "crossfit_manifest_sha256": file_sha256(CROSSFIT_MANIFEST_PATH),
        "seed": BOOTSTRAP_SEED,
        "replicate_count": BOOTSTRAP_REPLICATES,
        "independent_block_count": BANK_COUNT,
        "draws_per_replicate": BANK_COUNT,
        "resampling_unit": "sealed N65536 master bank block",
        "particle_bootstrap_used": False,
        "log_space_combination": True,
        "block_counts_path": str(BOOTSTRAP_COUNTS_PATH.relative_to(OUTPUT_ROOT)),
        "block_counts_sha256": file_sha256(BOOTSTRAP_COUNTS_PATH),
        "frozen_before_bootstrap": True,
    }
    _atomic_json(BOOTSTRAP_MANIFEST_PATH, payload)
    return payload


class PooledNewtonSolver:
    """Production-semantics Newton solve over bank-block feature tensors."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._solve = jax.jit(self._build_solver())
        self._bank_stats = jax.jit(self._build_bank_stats())
        self._tail = jax.jit(self._build_tail())

    @staticmethod
    def _statistics(phi: jax.Array, lam: jax.Array, mask: jax.Array):
        scores = jnp.einsum("cbnm,cm->cbn", phi, lam)
        included = mask[None, :, None]
        masked = jnp.where(included, scores, -jnp.inf)
        maximum = jnp.max(masked, axis=(1, 2))
        exponential = jnp.where(included, jnp.exp(scores - maximum[:, None, None]), 0.0)
        denominator = jnp.sum(exponential, axis=(1, 2))
        mean = jnp.einsum("cbn,cbnm->cm", exponential, phi) / denominator[:, None]
        second = jnp.einsum("cbn,cbni,cbnj->cij", exponential, phi, phi) / denominator[:, None, None]
        covariance = second - jnp.einsum("ci,cj->cij", mean, mean)
        sample_count = phi.shape[2] * jnp.sum(mask)
        logz = maximum + jnp.log(denominator) - jnp.log(sample_count)
        return logz, mean, covariance

    def _build_solver(self):
        cfg = self.cfg
        scales = 0.5 ** jnp.arange(cfg.line_search_steps, dtype=jnp.float64)

        def solve(phi: jax.Array, target: jax.Array, bank_mask: jax.Array, lam0: jax.Array):
            eye = jnp.eye(phi.shape[-1], dtype=jnp.float64)

            def residual_state(lam):
                logz, mean, covariance = self._statistics(phi, lam, bank_mask)
                residual = mean - target
                return logz, mean, covariance, residual, jnp.linalg.norm(residual, axis=-1)

            def dual_for_lam(lam_value):
                logz, _, _ = self._statistics(phi, lam_value, bank_mask)
                return logz - jnp.einsum("cm,cm->c", lam_value, target)

            lam0_clipped = jnp.clip(lam0, -cfg.lambda_clip, cfg.lambda_clip)
            initial = residual_state(lam0_clipped)
            state = (
                jnp.asarray(0, dtype=jnp.int32),
                lam0_clipped,
                initial[4],
                jnp.zeros((phi.shape[0],), dtype=jnp.int32),
            )

            def condition(value):
                step, _, residual_norm, _ = value
                return (step < cfg.max_steps) & jnp.any(residual_norm > cfg.residual_tol)

            def body(value):
                step, lam, residual_norm, iterations = value
                _, _, covariance, residual, _ = residual_state(lam)
                delta = jnp.linalg.solve(
                    covariance + cfg.newton_ridge * eye[None, :, :], residual[..., None]
                )[..., 0]
                delta_norm = jnp.linalg.norm(delta, axis=-1)
                cap = jnp.minimum(1.0, cfg.step_cap / jnp.maximum(delta_norm, 1e-30))
                delta = delta * cap[:, None]
                candidates = jnp.clip(
                    lam[None, :, :] - scales[:, None, None] * delta[None, :, :],
                    -cfg.lambda_clip,
                    cfg.lambda_clip,
                )
                duals = jax.lax.map(dual_for_lam, candidates)
                chosen = jnp.take_along_axis(
                    candidates,
                    jnp.argmin(duals, axis=0)[None, :, None],
                    axis=0,
                )[0]
                active = residual_norm > cfg.residual_tol
                next_lam = jnp.where(active[:, None], chosen, lam)
                next_norm = residual_state(next_lam)[4]
                return step + 1, next_lam, next_norm, iterations + active.astype(jnp.int32)

            _, lam, _, iterations = jax.lax.while_loop(condition, body, state)
            logz, mean, covariance, residual, residual_norm = residual_state(lam)
            return lam, iterations, logz, mean, covariance, residual, residual_norm

        return solve

    @staticmethod
    def _build_bank_stats():
        def bank_stats(phi: jax.Array, lam: jax.Array, target: jax.Array):
            scores = jnp.einsum("cbnm,cm->cbn", phi, lam)
            logz1 = jax.scipy.special.logsumexp(scores, axis=-1) - math.log(MASTER_N)
            logz2 = jax.scipy.special.logsumexp(2.0 * scores, axis=-1) - math.log(MASTER_N)
            normalized = jax.nn.softmax(scores, axis=-1)
            means = jnp.einsum("cbn,cbnm->cbm", normalized, phi)
            residual = jnp.linalg.norm(means - target[:, None, :], axis=-1)
            return logz1, logz2, means, residual
        return bank_stats

    @staticmethod
    def _build_tail():
        k001 = math.ceil(0.001 * POOLED_N)
        k01 = math.ceil(0.01 * POOLED_N)
        k05 = math.ceil(0.05 * POOLED_N)

        def tail(phi: jax.Array, lam: jax.Array):
            scores = jnp.einsum("cbnm,cm->cbn", phi, lam).reshape((phi.shape[0], -1))
            scores2 = 2.0 * scores
            total2 = jax.scipy.special.logsumexp(scores2, axis=-1)
            top2 = jax.lax.top_k(scores2, k05)[0]
            share001 = jnp.exp(jax.scipy.special.logsumexp(top2[:, :k001], axis=-1) - total2)
            share01 = jnp.exp(jax.scipy.special.logsumexp(top2[:, :k01], axis=-1) - total2)
            share05 = jnp.exp(jax.scipy.special.logsumexp(top2, axis=-1) - total2)
            total1 = jax.scipy.special.logsumexp(scores, axis=-1)
            top1 = jax.lax.top_k(scores, k01)[0]
            q_top01 = jnp.exp(jax.scipy.special.logsumexp(top1, axis=-1) - total1)
            maximum_q = jnp.exp(jnp.max(scores, axis=-1) - total1)
            return share001, share01, share05, q_top01, maximum_q
        return tail

    def solve(self, phi: Any, target: Any, bank_mask: Any, lam0: Any) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(value) for value in self._solve(
            jnp.asarray(phi, dtype=jnp.float64),
            jnp.asarray(target, dtype=jnp.float64),
            jnp.asarray(bank_mask, dtype=bool),
            jnp.asarray(lam0, dtype=jnp.float64),
        ))

    def bank_stats(self, phi: Any, lam: Any, target: Any) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(value) for value in self._bank_stats(
            jnp.asarray(phi, dtype=jnp.float64),
            jnp.asarray(lam, dtype=jnp.float64),
            jnp.asarray(target, dtype=jnp.float64),
        ))

    def tail(self, phi: Any, lam: Any) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(value) for value in self._tail(
            jnp.asarray(phi, dtype=jnp.float64), jnp.asarray(lam, dtype=jnp.float64)
        ))


def _feature_cache(problem: Any, etas: np.ndarray, nodes: list[int]) -> np.ndarray:
    cache = np.empty(
        (len(nodes), len(etas), BANK_COUNT, MASTER_N, FEATURE_DIMENSION),
        dtype=np.float64,
    )
    feature_function = jax.jit(jax.vmap(
        lambda eta, configurations: problem.family.features(configurations, eta),
        in_axes=(0, None),
    ))
    for bank_index, row in enumerate(_bank_rows()):
        with np.load(_bank_path(row), allow_pickle=False) as arrays:
            configurations = np.asarray(arrays["configurations"])[nodes]
        features = np.asarray(feature_function(
            jnp.asarray(etas, dtype=jnp.float64),
            jnp.asarray(configurations, dtype=jnp.float64),
        ))
        cache[:, :, bank_index] = np.swapaxes(features, 0, 1)
        del configurations, features
    return cache


def _variable_ress(candidate_indices: np.ndarray, nodes: list[int]) -> np.ndarray:
    result = np.empty((len(candidate_indices), len(nodes), BANK_COUNT), dtype=np.float64)
    bank_index = 0
    for replicate in range(16):
        for role in ("A", "B"):
            path = SOURCE_ROOT / "results" / f"replicate_{replicate:02d}" / f"bank_{role}_N65536.npz"
            with np.load(path, allow_pickle=False) as arrays:
                trajectory = np.asarray(arrays["ress_trajectory"])[candidate_indices]
            result[:, :, bank_index] = trajectory[:, nodes]
            bank_index += 1
    return result


def _combine_heldout(
    logz1: np.ndarray, logz2: np.ndarray, means: np.ndarray,
    target: np.ndarray, held: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected1 = logz1[:, held]
    selected2 = logz2[:, held]
    pooled1 = _stable_logmeanexp(selected1, axis=1)
    pooled2 = _stable_logmeanexp(selected2, axis=1)
    maximum = np.max(selected1, axis=1, keepdims=True)
    weights = np.exp(selected1 - maximum)
    combined_mean = np.einsum("cb,cbm->cm", weights, means[:, held]) / np.sum(weights, axis=1)[:, None]
    residual = np.linalg.norm(combined_mean - target, axis=1)
    d2 = pooled2 - 2.0 * pooled1
    return pooled1, pooled2, d2, np.exp(-d2), residual


def _bootstrap_from_bank_logs(
    logz1: np.ndarray, logz2: np.ndarray, counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shift1 = np.max(logz1, axis=1, keepdims=True)
    shift2 = np.max(logz2, axis=1, keepdims=True)
    scaled1 = np.exp(logz1 - shift1)
    scaled2 = np.exp(logz2 - shift2)
    mean1 = counts @ scaled1.T / BANK_COUNT
    mean2 = counts @ scaled2.T / BANK_COUNT
    boot_logz1 = np.log(mean1).T + shift1
    boot_logz2 = np.log(mean2).T + shift2
    d2 = boot_logz2 - 2.0 * boot_logz1
    return -d2, d2, np.exp(-d2)


def _batch_path(batch_index: int) -> Path:
    return OUTPUT_ROOT / "batches" / f"batch_{batch_index:02d}.npz"


def _batch_inventory_path(batch_index: int) -> Path:
    return OUTPUT_ROOT / "batches" / f"batch_{batch_index:02d}.json"


def _verify_batch(batch_index: int) -> dict[str, Any] | None:
    path, inventory_path = _batch_path(batch_index), _batch_inventory_path(batch_index)
    if not path.exists() and not inventory_path.exists():
        return None
    if not path.exists() or not inventory_path.exists():
        raise RuntimeError(f"incomplete population-overlap batch {batch_index}")
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if payload["batch_index"] != batch_index or payload["result_sha256"] != file_sha256(path):
        raise RuntimeError(f"sealed population-overlap batch changed: {batch_index}")
    for key, seal_path in (
        ("source_seal_sha256", SOURCE_SEAL_PATH),
        ("node_manifest_sha256", NODE_MANIFEST_PATH),
        ("crossfit_manifest_sha256", CROSSFIT_MANIFEST_PATH),
        ("bootstrap_manifest_sha256", BOOTSTRAP_MANIFEST_PATH),
    ):
        if payload[key] != file_sha256(seal_path):
            raise RuntimeError(f"batch {batch_index} manifest seal changed")
    return payload


def analyze_batches(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    bootstrap_manifest = freeze_bootstrap_manifest(cfg)
    node_manifest = json.loads(NODE_MANIFEST_PATH.read_text(encoding="utf-8"))
    crossfit = json.loads(CROSSFIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    nodes = [int(value) for value in node_manifest["analyzed_nodes"]]
    with np.load(BOOTSTRAP_COUNTS_PATH, allow_pickle=False) as arrays:
        block_counts = np.asarray(arrays["block_counts"], dtype=np.float64)
    panel = _panel()
    all_etas = _etas()
    problem = _load_problem(cfg)
    solver = PooledNewtonSolver(problem.projection_config)
    completed = []
    for batch_index, start in enumerate(range(0, PANEL_COUNT, CANDIDATE_BATCH_SIZE)):
        cached = _verify_batch(batch_index)
        if cached is not None:
            completed.append(cached)
            if progress is not None:
                progress(batch_index, True, float(cached["total_seconds"]))
            continue
        batch_started = time.perf_counter()
        stop = min(start + CANDIDATE_BATCH_SIZE, PANEL_COUNT)
        indices = np.arange(start, stop, dtype=np.int16)
        etas = all_etas[start:stop]
        targets_all = _candidate_targets(problem, etas)
        targets = targets_all[:, nodes]
        feature_started = time.perf_counter()
        features = _feature_cache(problem, etas, nodes)
        feature_seconds = time.perf_counter() - feature_started
        variable_ress = _variable_ress(indices, nodes)

        shape_cn = (len(indices), len(nodes))
        pooled_lambda = np.empty(shape_cn + (FEATURE_DIMENSION,), dtype=np.float64)
        pooled_iterations = np.empty(shape_cn, dtype=np.int32)
        pooled_residual = np.empty(shape_cn, dtype=np.float64)
        pooled_logz1 = np.empty(shape_cn, dtype=np.float64)
        pooled_logz2 = np.empty(shape_cn, dtype=np.float64)
        fixed_logz1 = np.empty(shape_cn + (BANK_COUNT,), dtype=np.float64)
        fixed_logz2 = np.empty_like(fixed_logz1)
        fixed_residual = np.empty_like(fixed_logz1)
        fixed_ress = np.empty_like(fixed_logz1)
        crossfit_lambda = np.empty(shape_cn + (CROSSFIT_FOLDS, FEATURE_DIMENSION), dtype=np.float64)
        crossfit_iterations = np.empty(shape_cn + (CROSSFIT_FOLDS,), dtype=np.int32)
        crossfit_train_residual = np.empty(shape_cn + (CROSSFIT_FOLDS,), dtype=np.float64)
        crossfit_logz1 = np.empty(shape_cn + (CROSSFIT_FOLDS,), dtype=np.float64)
        crossfit_logz2 = np.empty_like(crossfit_logz1)
        crossfit_d2 = np.empty_like(crossfit_logz1)
        crossfit_ress = np.empty_like(crossfit_logz1)
        crossfit_held_residual = np.empty_like(crossfit_logz1)
        bootstrap_log_ress = np.empty(shape_cn + (BOOTSTRAP_REPLICATES,), dtype=np.float64)
        bootstrap_d2 = np.empty_like(bootstrap_log_ress)
        bootstrap_ress = np.empty_like(bootstrap_log_ress)
        tail_z2_top001 = np.empty(shape_cn, dtype=np.float64)
        tail_z2_top01 = np.empty(shape_cn, dtype=np.float64)
        tail_z2_top05 = np.empty(shape_cn, dtype=np.float64)
        top01_q_mass = np.empty(shape_cn, dtype=np.float64)
        maximum_q = np.empty(shape_cn, dtype=np.float64)

        pooled_seconds = fixed_seconds = crossfit_seconds = tail_seconds = bootstrap_seconds = 0.0
        for node_index, node in enumerate(nodes):
            phi = features[node_index]
            target = targets[:, node_index]
            lam0 = np.zeros((len(indices), FEATURE_DIMENSION), dtype=np.float64)
            started = time.perf_counter()
            solved = solver.solve(phi, target, np.ones(BANK_COUNT, dtype=bool), lam0)
            lam, iterations, logz, _, _, _, residual_norm = solved
            pooled_seconds += time.perf_counter() - started
            pooled_lambda[:, node_index] = lam
            pooled_iterations[:, node_index] = iterations
            pooled_residual[:, node_index] = residual_norm

            started = time.perf_counter()
            bank_logz1, bank_logz2, bank_means, bank_residual = solver.bank_stats(phi, lam, target)
            fixed_seconds += time.perf_counter() - started
            fixed_logz1[:, node_index] = bank_logz1
            fixed_logz2[:, node_index] = bank_logz2
            fixed_residual[:, node_index] = bank_residual
            fixed_ress[:, node_index] = np.exp(2.0 * bank_logz1 - bank_logz2)
            pooled_logz1[:, node_index] = _stable_logmeanexp(bank_logz1, axis=1)
            pooled_logz2[:, node_index] = _stable_logmeanexp(bank_logz2, axis=1)
            if np.max(np.abs(pooled_logz1[:, node_index] - logz)) > 2e-11:
                raise RuntimeError("pooled logZ accumulation mismatch")

            started = time.perf_counter()
            for fold in crossfit["folds"]:
                fold_index = int(fold["fold"])
                mask = np.ones(BANK_COUNT, dtype=bool)
                held = [int(value) for value in fold["held_out_bank_indices"]]
                mask[held] = False
                fit = solver.solve(phi, target, mask, lam0)
                fold_lam, fold_iterations, _, _, _, _, train_residual = fit
                fold_bank = solver.bank_stats(phi, fold_lam, target)
                fold_values = _combine_heldout(
                    fold_bank[0], fold_bank[1], fold_bank[2], target, held
                )
                crossfit_lambda[:, node_index, fold_index] = fold_lam
                crossfit_iterations[:, node_index, fold_index] = fold_iterations
                crossfit_train_residual[:, node_index, fold_index] = train_residual
                crossfit_logz1[:, node_index, fold_index] = fold_values[0]
                crossfit_logz2[:, node_index, fold_index] = fold_values[1]
                crossfit_d2[:, node_index, fold_index] = fold_values[2]
                crossfit_ress[:, node_index, fold_index] = fold_values[3]
                crossfit_held_residual[:, node_index, fold_index] = fold_values[4]
            crossfit_seconds += time.perf_counter() - started

            started = time.perf_counter()
            tail = solver.tail(phi, lam)
            tail_seconds += time.perf_counter() - started
            tail_z2_top001[:, node_index] = tail[0]
            tail_z2_top01[:, node_index] = tail[1]
            tail_z2_top05[:, node_index] = tail[2]
            top01_q_mass[:, node_index] = tail[3]
            maximum_q[:, node_index] = tail[4]

            started = time.perf_counter()
            boot = _bootstrap_from_bank_logs(bank_logz1, bank_logz2, block_counts)
            bootstrap_seconds += time.perf_counter() - started
            bootstrap_log_ress[:, node_index] = boot[0]
            bootstrap_d2[:, node_index] = boot[1]
            bootstrap_ress[:, node_index] = boot[2]

        if np.max(pooled_residual) > float(problem.forcing_config.projection_tolerance):
            raise RuntimeError(f"pooled calibration failed projection tolerance in batch {batch_index}")
        result_path = _batch_path(batch_index)
        _atomic_npz(
            result_path,
            candidate_index=indices,
            nodes=np.asarray(nodes, dtype=np.int16),
            eta=etas,
            pooled_lambda=pooled_lambda,
            pooled_lambda_norm=np.linalg.norm(pooled_lambda, axis=-1),
            pooled_iterations=pooled_iterations,
            pooled_projection_residual=pooled_residual,
            pooled_logz1=pooled_logz1,
            pooled_logz2=pooled_logz2,
            pooled_d2=pooled_logz2 - 2.0 * pooled_logz1,
            pooled_ress=np.exp(2.0 * pooled_logz1 - pooled_logz2),
            fixed_lambda_bank_logz1=fixed_logz1,
            fixed_lambda_bank_logz2=fixed_logz2,
            fixed_lambda_bank_ress=fixed_ress,
            fixed_lambda_bank_projection_residual=fixed_residual,
            variable_lambda_bank_ress=variable_ress,
            crossfit_lambda=crossfit_lambda,
            crossfit_lambda_norm=np.linalg.norm(crossfit_lambda, axis=-1),
            crossfit_iterations=crossfit_iterations,
            crossfit_train_projection_residual=crossfit_train_residual,
            crossfit_held_logz1=crossfit_logz1,
            crossfit_held_logz2=crossfit_logz2,
            crossfit_held_d2=crossfit_d2,
            crossfit_held_ress=crossfit_ress,
            crossfit_held_projection_residual=crossfit_held_residual,
            bootstrap_log_ress=bootstrap_log_ress,
            bootstrap_d2=bootstrap_d2,
            bootstrap_ress=bootstrap_ress,
            z2_top_0p1pct_share=tail_z2_top001,
            z2_top_1pct_share=tail_z2_top01,
            z2_top_5pct_share=tail_z2_top05,
            top_1pct_projected_q_mass=top01_q_mass,
            maximum_normalized_projected_weight=maximum_q,
        )
        total_seconds = time.perf_counter() - batch_started
        inventory = {
            "schema_version": 1,
            "batch_index": batch_index,
            "candidate_indices": indices.tolist(),
            "candidate_ids": [panel["rows"][int(index)]["candidate_id"] for index in indices],
            "nodes": nodes,
            "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
            "node_manifest_sha256": file_sha256(NODE_MANIFEST_PATH),
            "crossfit_manifest_sha256": file_sha256(CROSSFIT_MANIFEST_PATH),
            "bootstrap_manifest_sha256": file_sha256(BOOTSTRAP_MANIFEST_PATH),
            "result_path": str(result_path.relative_to(OUTPUT_ROOT)),
            "result_sha256": file_sha256(result_path),
            "timings": {
                "feature_cache_seconds": feature_seconds,
                "pooled_calibration_seconds": pooled_seconds,
                "fixed_lambda_seconds": fixed_seconds,
                "crossfit_seconds": crossfit_seconds,
                "tail_seconds": tail_seconds,
                "bootstrap_seconds": bootstrap_seconds,
            },
            "total_seconds": total_seconds,
            "new_reference_samples_generated": 0,
            "validation_accessed": False,
            "official_protocol_created": False,
        }
        _atomic_json(_batch_inventory_path(batch_index), inventory)
        completed.append(inventory)
        del features
        if progress is not None:
            progress(batch_index, False, total_seconds)
    return {
        "schema_version": 1,
        "completed_batch_count": len(completed),
        "candidate_count": sum(len(row["candidate_indices"]) for row in completed),
        "nodes": nodes,
        "bootstrap_manifest_sha256": file_sha256(BOOTSTRAP_MANIFEST_PATH),
        "completed": completed,
    }


def _load_batches() -> dict[str, np.ndarray]:
    arrays_by_name: dict[str, list[np.ndarray]] = {}
    expected_indices = []
    for batch_index in range(math.ceil(PANEL_COUNT / CANDIDATE_BATCH_SIZE)):
        if _verify_batch(batch_index) is None:
            raise RuntimeError(f"missing population-overlap batch {batch_index}")
        with np.load(_batch_path(batch_index), allow_pickle=False) as arrays:
            for name in arrays.files:
                if name == "nodes":
                    continue
                arrays_by_name.setdefault(name, []).append(np.asarray(arrays[name]))
            expected_indices.extend(np.asarray(arrays["candidate_index"]).tolist())
    if expected_indices != list(range(PANEL_COUNT)):
        raise RuntimeError("population-overlap batch candidate ordering changed")
    return {name: np.concatenate(parts, axis=0) for name, parts in arrays_by_name.items()}


def _classify(lower: float, upper: float, crossfit_min: float, crossfit_max: float) -> str:
    if lower > MINIMUM_RESS and crossfit_min > MINIMUM_RESS:
        return "entirely_above_0.05"
    if upper < MINIMUM_RESS and crossfit_max < MINIMUM_RESS:
        return "entirely_below_0.05"
    return "contains_or_intersects_0.05"


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    return float(scipy_stats.spearmanr(x, y).statistic)


def _candidate_rows(arrays: dict[str, np.ndarray], nodes: list[int]) -> list[dict[str, Any]]:
    panel = _panel()
    rows = []
    for candidate_index, source in enumerate(panel["rows"]):
        node_rows = []
        for node_index, node in enumerate(nodes):
            fixed = arrays["fixed_lambda_bank_ress"][candidate_index, node_index]
            variable = arrays["variable_lambda_bank_ress"][candidate_index, node_index]
            fixed_log = np.log(fixed)
            residual = arrays["fixed_lambda_bank_projection_residual"][candidate_index, node_index]
            crossfit = arrays["crossfit_held_ress"][candidate_index, node_index]
            boot_r = arrays["bootstrap_ress"][candidate_index, node_index]
            boot_log = arrays["bootstrap_log_ress"][candidate_index, node_index]
            boot_d2 = arrays["bootstrap_d2"][candidate_index, node_index]
            lower, upper = np.quantile(boot_r, [0.025, 0.975])
            variable_sd = float(np.std(variable))
            fixed_sd = float(np.std(fixed))
            classification = _classify(float(lower), float(upper), float(np.min(crossfit)), float(np.max(crossfit)))
            node_rows.append({
                "node": node,
                "pooled_lambda": arrays["pooled_lambda"][candidate_index, node_index].tolist(),
                "pooled_lambda_norm": float(arrays["pooled_lambda_norm"][candidate_index, node_index]),
                "pooled_projection_residual": float(arrays["pooled_projection_residual"][candidate_index, node_index]),
                "pooled_iterations": int(arrays["pooled_iterations"][candidate_index, node_index]),
                "pooled_logz1": float(arrays["pooled_logz1"][candidate_index, node_index]),
                "pooled_logz2": float(arrays["pooled_logz2"][candidate_index, node_index]),
                "pooled_d2": float(arrays["pooled_d2"][candidate_index, node_index]),
                "pooled_ress": float(arrays["pooled_ress"][candidate_index, node_index]),
                "bootstrap_95_interval_ress": [float(lower), float(upper)],
                "bootstrap_95_interval_log_ress": [float(value) for value in np.quantile(boot_log, [0.025, 0.975])],
                "bootstrap_95_interval_d2": [float(value) for value in np.quantile(boot_d2, [0.025, 0.975])],
                "crossfit_ress": crossfit.tolist(),
                "crossfit_ress_distribution": _distribution(crossfit),
                "crossfit_held_projection_residual": arrays["crossfit_held_projection_residual"][candidate_index, node_index].tolist(),
                "fixed_lambda_ress_distribution": _distribution(fixed),
                "fixed_lambda_log_ress_distribution": _distribution(fixed_log),
                "fixed_lambda_held_projection_residual_distribution": _distribution(residual),
                "variable_lambda_ress_distribution": _distribution(variable),
                "variable_lambda_ress_sd": variable_sd,
                "fixed_lambda_ress_sd": fixed_sd,
                "fixed_over_variable_variance_ratio": None if variable_sd == 0.0 else (fixed_sd / variable_sd) ** 2,
                "fixed_over_variable_sd_ratio": None if variable_sd == 0.0 else fixed_sd / variable_sd,
                "fixed_variable_ress_correlation": _safe_correlation(fixed, variable, "pearson"),
                "z2_top_0p1pct_share": float(arrays["z2_top_0p1pct_share"][candidate_index, node_index]),
                "z2_top_1pct_share": float(arrays["z2_top_1pct_share"][candidate_index, node_index]),
                "z2_top_5pct_share": float(arrays["z2_top_5pct_share"][candidate_index, node_index]),
                "top_1pct_projected_q_mass": float(arrays["top_1pct_projected_q_mass"][candidate_index, node_index]),
                "maximum_normalized_projected_weight": float(arrays["maximum_normalized_projected_weight"][candidate_index, node_index]),
                "classification": classification,
            })
        boot_min = np.min(arrays["bootstrap_ress"][candidate_index], axis=0)
        crossfit_min = np.min(arrays["crossfit_held_ress"][candidate_index], axis=0)
        lower, upper = np.quantile(boot_min, [0.025, 0.975])
        rows.append({
            **source,
            "nodes": node_rows,
            "minimum_over_analyzed_nodes": {
                "pooled_ress": float(np.min(arrays["pooled_ress"][candidate_index])),
                "bootstrap_95_interval_ress": [float(lower), float(upper)],
                "crossfit_ress": crossfit_min.tolist(),
                "crossfit_mean": float(np.mean(crossfit_min)),
                "crossfit_minimum": float(np.min(crossfit_min)),
                "crossfit_maximum": float(np.max(crossfit_min)),
                "classification": _classify(
                    float(lower), float(upper), float(np.min(crossfit_min)), float(np.max(crossfit_min))
                ),
            },
        })
    return rows


def _group_summary(rows: list[dict[str, Any]], indices: list[int], nodes: list[int]) -> dict[str, Any]:
    selected = [rows[index] for index in indices]
    by_node = []
    for node_index, node in enumerate(nodes):
        node_rows = [row["nodes"][node_index] for row in selected]
        by_node.append({
            "node": node,
            "pooled_ress": _distribution([row["pooled_ress"] for row in node_rows]),
            "bootstrap_lower_endpoint": _distribution([row["bootstrap_95_interval_ress"][0] for row in node_rows]),
            "bootstrap_upper_endpoint": _distribution([row["bootstrap_95_interval_ress"][1] for row in node_rows]),
            "crossfit_mean": _distribution([row["crossfit_ress_distribution"]["mean"] for row in node_rows]),
            "crossfit_minimum": _distribution([row["crossfit_ress_distribution"]["minimum"] for row in node_rows]),
            "pooled_lambda_norm": _distribution([row["pooled_lambda_norm"] for row in node_rows]),
            "z2_top_1pct_share": _distribution([row["z2_top_1pct_share"] for row in node_rows]),
            "fixed_lambda_bank_sd": _distribution([row["fixed_lambda_ress_sd"] for row in node_rows]),
            "variable_lambda_bank_sd": _distribution([row["variable_lambda_ress_sd"] for row in node_rows]),
            "fixed_over_variable_variance_ratio": _distribution([
                row["fixed_over_variable_variance_ratio"] for row in node_rows
                if row["fixed_over_variable_variance_ratio"] is not None
            ]),
            "classification_counts": dict(Counter(row["classification"] for row in node_rows)),
        })
    minimum_rows = [row["minimum_over_analyzed_nodes"] for row in selected]
    return {
        "candidate_count": len(indices),
        "candidate_ids": [row["candidate_id"] for row in selected],
        "by_node": by_node,
        "minimum_over_analyzed_nodes": {
            "pooled_ress": _distribution([row["pooled_ress"] for row in minimum_rows]),
            "bootstrap_lower_endpoint": _distribution([row["bootstrap_95_interval_ress"][0] for row in minimum_rows]),
            "bootstrap_upper_endpoint": _distribution([row["bootstrap_95_interval_ress"][1] for row in minimum_rows]),
            "crossfit_mean": _distribution([row["crossfit_mean"] for row in minimum_rows]),
            "crossfit_minimum": _distribution([row["crossfit_minimum"] for row in minimum_rows]),
            "classification_counts": dict(Counter(row["classification"] for row in minimum_rows)),
        },
    }


def _finite_bank_comparison(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    source = json.loads((SOURCE_ROOT / "candidate_convergence.json").read_text(encoding="utf-8"))["rows"]
    pooled = np.asarray([rows[index]["minimum_over_analyzed_nodes"]["pooled_ress"] for index in indices])
    means = []
    medians = []
    pair_fractions = []
    for index in indices:
        nlevel = next(item for item in source[index]["N_levels"] if item["N"] == MASTER_N)
        distribution = nlevel["minimum_ress_distribution_32_banks"]
        means.append(distribution["mean"])
        medians.append(distribution["median"])
        pair_fractions.append(nlevel["pair_pass_fraction"])
    means_array, medians_array = np.asarray(means), np.asarray(medians)
    return {
        "candidate_count": len(indices),
        "pearson_pooled_vs_mean_finite_bank": _safe_correlation(pooled, means_array, "pearson"),
        "spearman_pooled_vs_mean_finite_bank": _safe_correlation(pooled, means_array, "spearman"),
        "pearson_pooled_vs_median_finite_bank": _safe_correlation(pooled, medians_array, "pearson"),
        "spearman_pooled_vs_median_finite_bank": _safe_correlation(pooled, medians_array, "spearman"),
        "mean_finite_bank_minus_pooled_bias": _distribution(means_array - pooled),
        "median_finite_bank_minus_pooled_bias": _distribution(medians_array - pooled),
        "existing_pair_pass_fraction": _distribution(pair_fractions),
    }


def _interpret(rows: list[dict[str, Any]], high_indices: list[int]) -> dict[str, Any]:
    law_categories = [rows[0]["nodes"][index]["classification"] for index in (0, 1)]
    high_categories = [rows[index]["minimum_over_analyzed_nodes"]["classification"] for index in high_indices]
    counts = Counter(high_categories)
    above = counts["entirely_above_0.05"] / len(high_indices)
    below = counts["entirely_below_0.05"] / len(high_indices)
    intersects = counts["contains_or_intersects_0.05"] / len(high_indices)
    law_above = all(value == "entirely_above_0.05" for value in law_categories)
    law_below = all(value == "entirely_below_0.05" for value in law_categories)
    law_spans = "entirely_above_0.05" in law_categories and "entirely_below_0.05" in law_categories
    strong_heterogeneity = law_spans or (above >= 0.20 and below >= 0.20)
    if strong_heterogeneity:
        label = "MIXED_POPULATION_OVERLAP"
        explanation = "Law nodes and/or candidate subsets occupy materially different sides of the gate."
        next_step = "Separate the node- and subset-specific overlap mechanisms before changing any proposal or gate semantics."
    elif law_above and above >= 0.50:
        label = "POPULATION_OVERLAP_CLEARLY_ABOVE_GATE"
        explanation = "Law and a substantial low-risk region are clearly above the gate under block and cross-fit evidence."
        next_step = "Study a more efficient importance/reference proposal or estimator while preserving the frozen reference law and information projection."
    elif law_below and below >= 0.50:
        label = "POPULATION_OVERLAP_BELOW_GATE"
        explanation = "Law and a substantial low-risk region are consistently below the gate."
        next_step = "Run a separate reference-model/reference-proposal/gate-meaning investigation; do not declare the threshold wrong automatically."
    elif "contains_or_intersects_0.05" in law_categories or intersects >= 0.40:
        label = "GATE_INTERSECTS_RELEVANT_POPULATION"
        explanation = "The 0.05 threshold cuts through Law and/or much of the scientifically relevant low-risk overlap regime."
        next_step = "Qualify what rESS=0.05 is intended to guarantee and whether that criterion predicts the downstream Galerkin quantities; do not lower it here."
    else:
        label = "MIXED_POPULATION_OVERLAP"
        explanation = "The remaining pooled/cross-fit evidence is materially mixed across nodes or candidate subsets."
        next_step = "Resolve the node- and subset-specific overlap differences in a separately frozen development study."
    return {
        "label": label,
        "explanation": explanation,
        "law_node7_node8_classifications": law_categories,
        "high_pass_minimum_node_classification_counts": dict(counts),
        "high_pass_fractions": {"above": above, "intersects": intersects, "below": below},
        "recommended_next_scientific_step": next_step,
    }


def _verify_cached_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists() and not INVENTORY_PATH.exists():
        return None
    if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
        raise RuntimeError("incomplete population-overlap summary")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"sealed population-overlap artifact changed: {row['path']}")
    return {**json.loads(SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    analyze_batches(cfg)
    cached = _verify_cached_summary()
    if cached is not None:
        return cached
    started = time.perf_counter()
    arrays = _load_batches()
    nodes = json.loads(NODE_MANIFEST_PATH.read_text(encoding="utf-8"))["analyzed_nodes"]
    rows = _candidate_rows(arrays, nodes)
    panel = _panel()
    high_indices = [index for index, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"]
    control_indices = [index for index, row in enumerate(panel["rows"]) if "control" in row["panel_role"]]
    if len(high_indices) != HIGH_PASS_COUNT or len(control_indices) != CONTROL_COUNT:
        raise RuntimeError("frozen panel roles changed")
    law_payload = {"schema_version": 1, "candidate": rows[0]}
    high_payload = {"schema_version": 1, "group": _group_summary(rows, high_indices, nodes)}
    control_payload = {"schema_version": 1, "group": _group_summary(rows, control_indices, nodes)}
    candidate_payload = {"schema_version": 1, "rows": rows}
    fixed_payload = {
        "schema_version": 1,
        "rows": [
            {"candidate_id": row["candidate_id"], "nodes": [
                {key: node_row[key] for key in (
                    "node", "fixed_lambda_ress_distribution", "fixed_lambda_log_ress_distribution",
                    "fixed_lambda_held_projection_residual_distribution", "variable_lambda_ress_distribution",
                    "variable_lambda_ress_sd", "fixed_lambda_ress_sd", "fixed_over_variable_variance_ratio",
                    "fixed_over_variable_sd_ratio", "fixed_variable_ress_correlation",
                )}
                for node_row in row["nodes"]
            ]}
            for row in rows
        ],
    }
    pooled_calibration = {
        "schema_version": 1,
        "rows": [{"candidate_id": row["candidate_id"], "eta": row["eta"], "nodes": [
            {key: node_row[key] for key in (
                "node", "pooled_lambda", "pooled_lambda_norm", "pooled_projection_residual", "pooled_iterations",
            )}
            for node_row in row["nodes"]
        ]} for row in rows],
    }
    pooled_overlap = {
        "schema_version": 1,
        "rows": [{"candidate_id": row["candidate_id"], "nodes": [
            {key: node_row[key] for key in (
                "node", "pooled_logz1", "pooled_logz2", "pooled_d2", "pooled_ress",
                "bootstrap_95_interval_ress", "bootstrap_95_interval_log_ress", "bootstrap_95_interval_d2", "classification",
            )}
            for node_row in row["nodes"]
        ], "minimum_over_analyzed_nodes": row["minimum_over_analyzed_nodes"]} for row in rows],
    }
    crossfit_payload = {
        "schema_version": 1,
        "rows": [{"candidate_id": row["candidate_id"], "nodes": [
            {key: node_row[key] for key in (
                "node", "crossfit_ress", "crossfit_ress_distribution", "crossfit_held_projection_residual",
            )}
            for node_row in row["nodes"]
        ]} for row in rows],
    }
    tail_payload = {
        "schema_version": 1,
        "rows": [{"candidate_id": row["candidate_id"], "nodes": [
            {key: node_row[key] for key in (
                "node", "pooled_lambda_norm", "pooled_d2", "z2_top_0p1pct_share", "z2_top_1pct_share",
                "z2_top_5pct_share", "top_1pct_projected_q_mass", "maximum_normalized_projected_weight",
            )}
            for node_row in row["nodes"]
        ]} for row in rows],
    }
    output_payloads = {
        "pooled_calibration.json": pooled_calibration,
        "pooled_overlap.json": pooled_overlap,
        "fixed_lambda_bank_results.json": fixed_payload,
        "crossfit_results.json": crossfit_payload,
        "tail_concentration.json": tail_payload,
        "law_population_overlap.json": law_payload,
        "candidate_population_overlap.json": candidate_payload,
        "high_pass_population_overlap.json": high_payload,
        "control_population_overlap.json": control_payload,
    }
    for name, payload in output_payloads.items():
        _atomic_json(OUTPUT_ROOT / name, payload)
    interpretation = _interpret(rows, high_indices)
    batch_inventories = [
        json.loads(_batch_inventory_path(index).read_text(encoding="utf-8"))
        for index in range(math.ceil(PANEL_COUNT / CANDIDATE_BATCH_SIZE))
    ]
    comparison = {
        "law_and_high_pass": _finite_bank_comparison(rows, [0] + high_indices),
        "controls": _finite_bank_comparison(rows, control_indices),
    }
    timings: dict[str, Any] = {
        "source_verification_seconds": json.loads(SOURCE_SEAL_PATH.read_text())["source_verification_seconds"],
        "feature_cache_seconds": float(sum(row["timings"]["feature_cache_seconds"] for row in batch_inventories)),
        "pooled_calibration_seconds": float(sum(row["timings"]["pooled_calibration_seconds"] for row in batch_inventories)),
        "fixed_lambda_seconds": float(sum(row["timings"]["fixed_lambda_seconds"] for row in batch_inventories)),
        "crossfit_seconds": float(sum(row["timings"]["crossfit_seconds"] for row in batch_inventories)),
        "tail_seconds": float(sum(row["timings"]["tail_seconds"] for row in batch_inventories)),
        "bootstrap_seconds": float(sum(row["timings"]["bootstrap_seconds"] for row in batch_inventories)),
        "batch_total_seconds": float(sum(row["total_seconds"] for row in batch_inventories)),
        "summary_seconds": time.perf_counter() - started,
    }
    summary = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_verified": True,
        "candidate_panel_count": PANEL_COUNT,
        "existing_master_bank_count": BANK_COUNT,
        "samples_per_master_bank": MASTER_N,
        "pooled_samples_per_node": POOLED_N,
        "new_reference_samples_generated": 0,
        "analyzed_nodes": nodes,
        "base_measure": json.loads(BASE_AUDIT_PATH.read_text(encoding="utf-8")),
        "law": law_payload,
        "high_pass": high_payload,
        "controls": control_payload,
        "finite_bank_comparison": comparison,
        "interpretation": interpretation,
        "timings": timings,
        "firewalls": {
            "new_banks_generated": False,
            "candidate_generation": False,
            "tangent_run": False,
            "full_kf_constructed": False,
            "eigensolve_run": False,
            "deep_ritz_run": False,
            "validation_accessed": False,
            "official_protocol_created": False,
        },
        "seals": {
            "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
            "base_measure_audit_sha256": file_sha256(BASE_AUDIT_PATH),
            "node_analysis_manifest_sha256": file_sha256(NODE_MANIFEST_PATH),
            "crossfit_manifest_sha256": file_sha256(CROSSFIT_MANIFEST_PATH),
            "bootstrap_manifest_sha256": file_sha256(BOOTSTRAP_MANIFEST_PATH),
        },
    }
    _atomic_json(SUMMARY_PATH, summary)
    artifact_paths = [
        SOURCE_SEAL_PATH, BASE_AUDIT_PATH, NODE_MANIFEST_PATH, CROSSFIT_MANIFEST_PATH,
        BOOTSTRAP_COUNTS_PATH, BOOTSTRAP_MANIFEST_PATH,
        *[_batch_inventory_path(index) for index in range(math.ceil(PANEL_COUNT / CANDIDATE_BATCH_SIZE))],
        *[OUTPUT_ROOT / name for name in output_payloads], SUMMARY_PATH,
    ]
    inventory = {
        "schema_version": 1,
        "artifact_count": len(artifact_paths),
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in artifact_paths],
    }
    _atomic_json(INVENTORY_PATH, inventory)
    return {**summary, "cache_hit": False}


def run(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    freeze_bootstrap_manifest(cfg)
    analyze_batches(cfg, progress=progress)
    return summarize(cfg)


__all__ = [
    "BASE_AUDIT_PATH", "BANK_COUNT", "BOOTSTRAP_COUNTS_PATH", "BOOTSTRAP_MANIFEST_PATH",
    "BOOTSTRAP_REPLICATES", "CANDIDATE_BATCH_SIZE", "CROSSFIT_MANIFEST_PATH",
    "MASTER_N", "MINIMUM_RESS", "NODE_MANIFEST_PATH", "OUTPUT_ROOT", "PANEL_COUNT",
    "POOLED_N", "PRIMARY_NODES", "SOURCE_ROOT", "SOURCE_SEAL_PATH", "SUMMARY_PATH",
    "PooledNewtonSolver", "analyze_batches", "audit_base_measure", "freeze_bootstrap_manifest",
    "freeze_crossfit_manifest", "freeze_node_manifest", "run", "summarize", "verify_and_freeze_sources",
]
