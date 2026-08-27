"""Development-only nested-N rESS convergence study.

The experiment freezes a 64-geometry diagnostic panel and 16 independent
master-bank pairs, then reuses nested prefixes of each N=65536 master bank.
It does not contain a candidate generator, Tangent/Full optimizer, K/f
assembly, eigensolve, Deep Ritz solver, validation loader, or official writer.
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
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.projection import EmpiricalIProjector

from .full_gradient import reconstruct_moments
from .galerkin_only_data import GalerkinReferenceBank, load_selection_galerkin_data
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v2_selection import _generate_bank
from .pareto_v3_common import ROOT, file_sha256
from .pareto_v3_diagnostic import _symmetry_aware_distance


VERSION = "skyrmion_galerkin_dev_ress_n_convergence_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
FRESH_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_fresh_bank_robustness_v1"
PREFLIGHT_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_replicate_gate_preflight_v2"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
PANEL_PATH = OUTPUT_ROOT / "candidate_panel.json"
MANIFEST_PATH = OUTPUT_ROOT / "master_bank_manifest.json"
BANK_INVENTORY_PATH = OUTPUT_ROOT / "master_bank_inventory.json"
LAW_PATH = OUTPUT_ROOT / "law_convergence.json"
CANDIDATE_PATH = OUTPUT_ROOT / "candidate_convergence.json"
TIME7_PATH = OUTPUT_ROOT / "time7_diagnostics.json"
FLIP_PATH = OUTPUT_ROOT / "threshold_flip_summary.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

PAIR_COUNT = 16
MASTER_BANK_COUNT = 32
MASTER_N = 65536
N_LADDER = (8192, 16384, 32768, 65536)
ROLES = ("A", "B")
PANEL_COUNT = 64
HIGH_PASS_COUNT = 55
CONTROL_COUNT = 8
BATCH_SIZE = 8
LAW_RISK = 5.186549474478042
HALF_PERCENT_CEILING = LAW_RISK * 1.005
MINIMUM_RESS = 0.05
GLOBAL_SEED = 20260825
SEED_NAMESPACE = VERSION

EXPECTED_FRESH_HASHES = {
    "candidate_robustness_summary.json": "99121aed7b18d70128cee7cdcc9d4d61dfd64e98d68d25ffb9131164c1a0db77",
    "allowance_summary.json": "e8f21ec7f78a4d1996505c6bed9e85e5b2f7b2ef57752b9af401388d63231a0f",
    "summary.json": "998a59b5bdb195e15379be085b60d31c5d5c6edd8edf63f78a64ea351f1e8740",
    "inventory.json": "3d837c2ea4108283749bdfa1d661e0a90ebd5ad39693ea942641c23f94df466e",
    "candidate_freeze.json": "3fae7f1cc7479d0d5413f89838aba9b0ccd8d24374dd27de699d780e5a3e1f4d",
    "bank_manifest.json": "9e2d30fa15ed29c27e415b032ce9bd4a7b4c673bc9dda1891cc8c8f7201845d3",
    "bank_inventory.json": "e5dd4f14e84b1cbc8e74af1a477a415490b98258bc0d7e4332464737ac02338d",
}

EXPECTED_PREFLIGHT_HASHES = {
    "source_seal.json": "30ecbde6594d197084d6d32ab888ae8edf5ff964ef7320edb2802614c1aed075",
    "architecture_grid.json": "a20d1849e03ef3197186a2d092f0515fc19df63b9d1337520d4f4acd773b8c4c",
    "subset_manifest.json": "330c2e89a266253ddb89bcfbc098af7598433532ec607335accd5926925fbe1d",
    "recommended_official_gate.json": "b6fbd51c01ef2097e9f5997b1d8ec9ef86220e2ec1116df85ab52d48a02f4cb0",
    "summary.json": "87c56fe469ec90b7c13b2207d02a09b678d02d935c062e5762800e28f8da4cc6",
    "inventory.json": "e9a34ace9a8ff1f67df3322510446185eee95378daf025de37c89e9779d30868",
}


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _array_sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


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


def _atomic_npz(path: Path, *, compressed: bool = False, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **{name: np.asarray(value) for name, value in arrays.items()})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _code_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in (
            "ress_n_convergence.py",
            "ress_n_convergence_run.py",
            "test_ress_n_convergence.py",
            "config.json",
        )
    }


def verify_and_freeze_sources() -> dict[str, Any]:
    fresh = {name: file_sha256(FRESH_ROOT / name) for name in EXPECTED_FRESH_HASHES}
    preflight = {name: file_sha256(PREFLIGHT_ROOT / name) for name in EXPECTED_PREFLIGHT_HASHES}
    if fresh != EXPECTED_FRESH_HASHES:
        raise RuntimeError("sealed fresh-bank source differs")
    if preflight != EXPECTED_PREFLIGHT_HASHES:
        raise RuntimeError("sealed replicate-gate preflight differs")
    fresh_summary = json.loads((FRESH_ROOT / "summary.json").read_text(encoding="utf-8"))
    preflight_summary = json.loads((PREFLIGHT_ROOT / "summary.json").read_text(encoding="utf-8"))
    if fresh_summary["candidate_count"] != 4433 or fresh_summary["replicate_count"] != 32:
        raise RuntimeError("fresh-bank dimensions changed")
    if preflight_summary["recommendation"]["recommendation"] != "NO_REPLICATE_GATE_ARCHITECTURE_READY":
        raise RuntimeError("preflight conclusion changed")
    payload = {
        "schema_version": 1,
        "development_only": True,
        "fresh_bank_hashes": fresh,
        "preflight_hashes": preflight,
        "analysis_source_hashes": _code_hashes(),
        "validation_accessed": False,
        "official_protocol_created": False,
        "selection_frozen": False,
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def _distribution(values: np.ndarray, *, extended: bool = True) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    result = {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }
    if extended:
        result.update(mean=float(np.mean(values)), std=float(np.std(values)))
    return result


def _row_pass_count(row: dict[str, Any]) -> int:
    match = next(item for item in row["allowances"] if float(item["allowance_percent"]) == 0.5)
    return int(match["complete_dual_bank_pass_count"])


def _median_robust(row: dict[str, Any]) -> float:
    value = row["fresh_robust_ress_performed_only"].get("median")
    return -math.inf if value is None else float(value)


def _maxmin_controls(pool: list[dict[str, Any]], count: int, box: list[float]) -> list[dict[str, Any]]:
    remaining = sorted(
        pool,
        key=lambda row: (-_row_pass_count(row), -_median_robust(row), row["candidate_id"]),
    )
    if len(remaining) < count:
        raise RuntimeError("insufficient deterministic control candidates")
    selected = [remaining.pop(0)]
    while remaining and len(selected) < count:
        chosen = min(
            remaining,
            key=lambda row: (
                -min(_symmetry_aware_distance(row["eta"], old["eta"], box) for old in selected),
                -_row_pass_count(row),
                -_median_robust(row),
                row["candidate_id"],
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def freeze_candidate_panel() -> dict[str, Any]:
    verify_and_freeze_sources()
    if PANEL_PATH.exists():
        payload = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
        if payload["source_seal_sha256"] != file_sha256(SOURCE_SEAL_PATH):
            raise RuntimeError("candidate panel source seal changed")
        if payload["panel_rows_sha256"] != _payload_sha256(payload["rows"]):
            raise RuntimeError("candidate panel rows changed")
        return {**payload, "cache_hit": True}

    candidates = json.loads(
        (FRESH_ROOT / "candidate_robustness_summary.json").read_text(encoding="utf-8")
    )["rows"]
    freeze = json.loads((FRESH_ROOT / "candidate_freeze.json").read_text(encoding="utf-8"))
    box = [float(value) for value in freeze["box"]]
    law = [row for row in candidates if row["candidate_id"] == "candidate_000"]
    high = [
        row for row in candidates
        if row["candidate_id"] != "candidate_000"
        and float(row["fixed_scientific_selection_risk"]) <= HALF_PERCENT_CEILING
        and _row_pass_count(row) >= 24
    ]
    if len(law) != 1 or len(high) != HIGH_PASS_COUNT:
        raise RuntimeError(f"panel anchor counts changed: Law={len(law)}, high={len(high)}")
    excluded = {row["candidate_id"] for row in law + high}
    inside = [
        row for row in candidates
        if row["candidate_id"] not in excluded
        and float(row["fixed_scientific_selection_risk"]) <= HALF_PERCENT_CEILING
    ]
    middle = [row for row in inside if 16 <= _row_pass_count(row) <= 23]
    lower = [row for row in inside if _row_pass_count(row) <= 15]
    controls = _maxmin_controls(middle, 4, box) + _maxmin_controls(lower, 4, box)

    roles = {law[0]["candidate_id"]: "law"}
    roles.update({row["candidate_id"]: "high_pass_ge24_of_32" for row in high})
    roles.update({row["candidate_id"]: "middle_control_16_to_23" for row in controls[:4]})
    roles.update({row["candidate_id"]: "lower_control_0_to_15" for row in controls[4:]})
    ordered = law + sorted(high, key=lambda row: row["candidate_id"]) + controls
    rows = [
        {
            "panel_index": index,
            "candidate_id": row["candidate_id"],
            "eta": [float(value) for value in row["eta"]],
            "fixed_scientific_selection_risk": float(row["fixed_scientific_selection_risk"]),
            "law_relative_risk_increase_percent": float(row["risk_increase_pct"]),
            "source_pool": row["source_pool"],
            "generation_source": row["generation_method"],
            "old_32_pair_pass_count": _row_pass_count(row),
            "panel_role": roles[row["candidate_id"]],
        }
        for index, row in enumerate(ordered)
    ]
    if len(rows) != PANEL_COUNT or len({row["candidate_id"] for row in rows}) != PANEL_COUNT:
        raise RuntimeError("diagnostic panel is not exactly 64 unique candidates")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "candidate_count": PANEL_COUNT,
        "law_count": 1,
        "high_pass_count": HIGH_PASS_COUNT,
        "control_count": CONTROL_COUNT,
        "control_rule": {
            "middle": "four symmetry-aware max-min candidates with 16-23 passes",
            "lower": "four symmetry-aware max-min candidates with 0-15 passes",
            "risk_ceiling": HALF_PERCENT_CEILING,
        },
        "candidate_generation_permitted": False,
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "source_candidate_freeze_sha256": file_sha256(FRESH_ROOT / "candidate_freeze.json"),
        "box": box,
        "rows": rows,
        "panel_rows_sha256": _payload_sha256(rows),
    }
    _atomic_json(PANEL_PATH, payload)
    return payload


def derive_seed(replicate: int, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(role)
    text = f"{GLOBAL_SEED}:{SEED_NAMESPACE}:replicate_{replicate:02d}:bank_{role}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "namespace": SEED_NAMESPACE,
        "derivation_text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def freeze_master_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    panel = freeze_candidate_panel()
    if MANIFEST_PATH.exists():
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload["candidate_panel_sha256"] != file_sha256(PANEL_PATH):
            raise RuntimeError("master manifest candidate panel changed")
        if payload["manifest_rows_sha256"] != _payload_sha256(payload["replicates"]):
            raise RuntimeError("master manifest rows changed")
        return {**payload, "cache_hit": True}

    old_fresh = json.loads((FRESH_ROOT / "bank_manifest.json").read_text(encoding="utf-8"))
    old_seeds = {
        int(row[f"{key}_seed"]["seed"])
        for row in old_fresh["replicates"]
        for key in ("screen", "audit")
    }
    records = []
    seen: set[int] = set()
    for replicate in range(PAIR_COUNT):
        banks = {}
        for role in ROLES:
            seed = derive_seed(replicate, role)
            if seed["seed"] in seen or seed["seed"] in old_seeds:
                raise RuntimeError("master-bank seed collision")
            seen.add(seed["seed"])
            banks[role] = seed
        records.append({"replicate_id": replicate, "banks": banks})
    generation = {
        "generator": "pareto_v2_selection._generate_bank exact existing implementation",
        "master_N": MASTER_N,
        "nested_prefix_N": list(N_LADDER),
        "prefix_base_weight_rule": "slice then renormalize independently at each physical time",
        "dtype": "float64",
        "reference_checkpoint_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"),
        "physics": cfg["physics"],
        "reference_substeps": int(cfg["banks"]["reference_substeps"]),
    }
    interpretation_rule = {
        "frozen_before_evaluation": True,
        "material_variance_reduction": "median candidate bank-std ratio N65536/N8192 <= 0.75 and nested absolute changes decline",
        "threshold_stabilization": "final transition pair-decision disagreement <= 0.10 and below first-transition disagreement",
        "threshold_boundary_overlap": "Law median or median high-pass candidate p10 at N65536 lies in [0.045,0.055]",
        "stable_adequate_support": "Law median and median high-pass candidate p10 at N65536 are both >0.055",
        "stable_inadequate_support": "Law median or median high-pass candidate p10 at N65536 is <0.045",
        "persistent_node7_overlap": "node 7 controls >=70% of high-pass trajectories and has median rESS <0.05 and below other-node median",
        "labels": {
            "N_LIMITED_SUPPORT_ESTIMATION": "variance and decisions stabilize with a clear adequate support margin",
            "BORDERLINE_POPULATION_OVERLAP": "variance and decisions stabilize but Law or panel remains near threshold",
            "PERSISTENT_REFERENCE_PROPOSAL_MISMATCH": "larger N does not materially repair low-rESS/weight concentration",
            "MIXED_N_AND_PROPOSAL_EFFECT": "N helps materially while persistent node-7 overlap problems remain",
        },
    }
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "prospectively_frozen_before_bank_generation": True,
        "candidate_panel_sha256": file_sha256(PANEL_PATH),
        "candidate_count": panel["candidate_count"],
        "master_pair_count": PAIR_COUNT,
        "master_bank_count": MASTER_BANK_COUNT,
        "master_N": MASTER_N,
        "N_ladder": list(N_LADDER),
        "nested_prefixes": True,
        "all_seeds_unique": len(seen) == MASTER_BANK_COUNT,
        "fresh_seeds_disjoint_from_previous_fresh_study": seen.isdisjoint(old_seeds),
        "generation_configuration": generation,
        "generation_configuration_sha256": _payload_sha256(generation),
        "interpretation_rule": interpretation_rule,
        "replicates": records,
        "manifest_rows_sha256": _payload_sha256(records),
        "validation_seed_namespace_used": False,
        "official_protocol_created": False,
    }
    _atomic_json(MANIFEST_PATH, payload)
    return payload


def _bank_path(replicate: int, role: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"replicate_{replicate:02d}_{role}_N{MASTER_N}.npz"


def _bank_metadata(path: Path, record: dict[str, Any], role: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["role"].item()) != role:
            raise RuntimeError(f"bank role changed: {path}")
        if int(arrays["seed"].item()) != int(record["banks"][role]["seed"]):
            raise RuntimeError(f"bank seed changed: {path}")
        configurations = arrays["configurations"]
        velocity = arrays["velocity"]
        base_weights = arrays["base_weights"]
        if configurations.shape != (13, MASTER_N, 16, 2):
            raise RuntimeError(f"master bank size changed: {path}")
        if velocity.shape != configurations.shape or base_weights.shape != (13, MASTER_N):
            raise RuntimeError(f"master bank array dimensions changed: {path}")
        if any(array.dtype != np.float64 for array in (configurations, velocity, base_weights)):
            raise RuntimeError(f"master bank dtype changed: {path}")
        if not np.allclose(np.sum(base_weights, axis=1), 1.0, rtol=0.0, atol=1e-14):
            raise RuntimeError(f"master bank base weights are not normalized: {path}")
        initial = _array_sha256(configurations[0])
    return {
        "path": str(path.relative_to(OUTPUT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "initial_state_sha256": initial,
        "replicate_id": int(record["replicate_id"]),
        "role": role,
        "N": MASTER_N,
        "seed": record["banks"][role],
    }


def generate_master_banks(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    manifest = freeze_master_manifest(cfg)
    if BANK_INVENTORY_PATH.exists():
        payload = json.loads(BANK_INVENTORY_PATH.read_text(encoding="utf-8"))
        if payload["master_bank_manifest_sha256"] != file_sha256(MANIFEST_PATH):
            raise RuntimeError("master bank inventory manifest changed")
        for row in payload["banks"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"sealed master bank changed: {row['path']}")
        return {**payload, "cache_hit": True}

    rows = []
    timings = []
    for record in manifest["replicates"]:
        for role in ROLES:
            path = _bank_path(int(record["replicate_id"]), role)
            started = time.perf_counter()
            cache_hit = path.exists()
            if not cache_hit:
                bank = _generate_bank(cfg, int(record["banks"][role]["seed"]), MASTER_N)
                _atomic_npz(
                    path,
                    configurations=bank.configurations,
                    velocity=bank.velocity,
                    base_weights=bank.base_weights,
                    replicate_id=np.asarray(record["replicate_id"]),
                    role=np.asarray(role),
                    seed=np.asarray(record["banks"][role]["seed"]),
                    seed_sha256=np.asarray(record["banks"][role]["sha256"]),
                    generation_configuration_sha256=np.asarray(manifest["generation_configuration_sha256"]),
                )
                del bank
            row = _bank_metadata(path, record, role)
            elapsed = time.perf_counter() - started
            row.update(generation_seconds=elapsed, cache_hit=cache_hit)
            rows.append(row)
            timings.append(elapsed)
            if progress is not None:
                progress(int(record["replicate_id"]), role, cache_hit, elapsed)
    initial = [row["initial_state_sha256"] for row in rows]
    if len(initial) != len(set(initial)):
        raise RuntimeError("master bank initial states are not pairwise distinct")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "master_bank_manifest_sha256": file_sha256(MANIFEST_PATH),
        "candidate_evaluation_started": False,
        "bank_count": len(rows),
        "master_pair_count": PAIR_COUNT,
        "pairwise_distinct_initial_state_hashes": True,
        "total_generation_seconds_this_invocation": float(sum(timings)),
        "banks": rows,
    }
    _atomic_json(BANK_INVENTORY_PATH, payload)
    return payload


class _NestedEvaluator:
    """Candidate-batched projection and support diagnostics at one nested N."""

    def __init__(self, problem: Any):
        self.problem = problem
        self.projector = EmpiricalIProjector(
            problem.projection_config, trajectory_backend=problem.projection_backend
        )
        self.preprocess = jax.jit(
            jax.vmap(
                lambda eta, configurations, velocity: (
                    reconstruct_moments(eta, problem).values,
                    reconstruct_moments(eta, problem).derivatives,
                    problem.family.features(configurations, eta),
                    problem.family.jvp(configurations, velocity, eta),
                ),
                in_axes=(0, None, None),
            )
        )
        self.postprocessors: dict[int, Any] = {}

    def _postprocessor(self, N: int):
        if N in self.postprocessors:
            return self.postprocessors[N]
        top_count = max(1, int(math.ceil(0.01 * N)))
        problem = self.problem

        @jax.jit
        def postprocess(
            weights, lam, moments, covariance, residual, ess,
            features, advective, derivatives,
        ):
            moment_m = jnp.einsum("btn,btnr->btr", weights, advective)
            scalar_m = jnp.einsum("btnr,btr->btn", advective, lam)
            centered_phi = features - moments[:, :, None, :]
            centered_g = scalar_m - jnp.einsum("btn,btn->bt", weights, scalar_m)[:, :, None]
            covariance_phi_g = jnp.einsum("btn,btnr,btn->btr", weights, centered_phi, centered_g)
            rhs = derivatives - moment_m - covariance_phi_g
            regularized = covariance + float(problem.forcing_config.covariance_ridge) * jnp.eye(features.shape[-1])
            lambda_dot = jnp.linalg.solve(regularized, rhs[..., None])[..., 0]
            forcing = (
                jnp.einsum("btr,btnr->btn", lambda_dot, features - moments[:, :, None, :])
                + jnp.einsum("btr,btnr->btn", lam, advective - moment_m[:, :, None, :])
            )
            forcing_mean = jnp.abs(jnp.einsum("btn,btn->bt", weights, forcing))
            covariance_eigenvalues = jnp.linalg.eigvalsh(covariance)
            regularized_eigenvalues = jnp.linalg.eigvalsh(regularized)
            condition = regularized_eigenvalues[..., -1] / jnp.maximum(
                regularized_eigenvalues[..., 0], 1e-300
            )
            projection_residual = jnp.linalg.norm(residual, axis=-1)
            top_mass = jnp.sum(jax.lax.top_k(weights, top_count)[0], axis=-1)
            sum_q2 = jnp.sum(weights * weights, axis=-1)
            return (
                ess,
                jnp.linalg.norm(lam, axis=-1),
                jnp.max(weights, axis=-1),
                top_mass,
                sum_q2,
                -jnp.log(jnp.maximum(ess, 1e-300)),
                covariance_eigenvalues,
                regularized_eigenvalues,
                condition,
                projection_residual,
                forcing_mean,
            )

        self.postprocessors[N] = postprocess
        return postprocess

    def evaluate(
        self,
        etas: np.ndarray,
        configurations: jax.Array,
        velocity: jax.Array,
        base_weights: jax.Array,
        N: int,
    ) -> dict[str, np.ndarray]:
        configurations = configurations[:, :N]
        velocity = velocity[:, :N]
        weights0 = base_weights[:, :N]
        weights0 = weights0 / jnp.sum(weights0, axis=1, keepdims=True)
        trajectory_names = (
            "ress_trajectory", "lambda_norm", "maximum_normalized_weight",
            "top_1pct_weight_mass", "sum_q_squared", "empirical_D2",
            "covariance_condition_trajectory", "projection_residual_trajectory",
            "forcing_mean_trajectory",
        )
        collected: dict[str, list[np.ndarray]] = {name: [] for name in trajectory_names}
        covariance_eigenvalues: list[np.ndarray] = []
        regularized_covariance_eigenvalues: list[np.ndarray] = []
        postprocess = self._postprocessor(N)
        for start in range(0, len(etas), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(etas))
            batch = etas[start:stop]
            actual = len(batch)
            if actual < BATCH_SIZE:
                batch = np.concatenate((batch, np.repeat(batch[-1:], BATCH_SIZE - actual, axis=0)))
            targets, derivatives, features, advective = self.preprocess(
                jnp.asarray(batch, dtype=jnp.float64), configurations, velocity
            )
            projected = self.projector.project_candidate_trajectories(features, weights0, targets)
            values = postprocess(
                projected.weights, projected.lam, projected.moments,
                projected.covariance, projected.residual, projected.ess_fraction,
                features, advective, derivatives,
            )
            numpy_values = [np.asarray(value)[:actual] for value in values]
            for name, value in zip(trajectory_names[:6], numpy_values[:6], strict=True):
                collected[name].append(value)
            covariance_eigenvalues.append(numpy_values[6])
            regularized_covariance_eigenvalues.append(numpy_values[7])
            for name, value in zip(trajectory_names[6:], numpy_values[8:], strict=True):
                collected[name].append(value)
        result = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
        result["covariance_eigenvalues"] = np.concatenate(covariance_eigenvalues, axis=0)
        result["regularized_covariance_eigenvalues"] = np.concatenate(
            regularized_covariance_eigenvalues, axis=0
        )
        result["minimum_ress"] = np.min(result["ress_trajectory"], axis=1)
        result["controlling_time_index"] = np.argmin(result["ress_trajectory"], axis=1).astype(np.int16)
        result["maximum_projection_residual"] = np.max(result["projection_residual_trajectory"], axis=1)
        result["maximum_forcing_mean"] = np.max(result["forcing_mean_trajectory"], axis=1)
        result["maximum_covariance_condition"] = np.max(result["covariance_condition_trajectory"], axis=1)
        result["projection_valid"] = result["maximum_projection_residual"] <= float(self.problem.forcing_config.projection_tolerance)
        result["ress_valid"] = result["minimum_ress"] >= MINIMUM_RESS
        result["forcing_valid"] = result["maximum_forcing_mean"] <= float(self.problem.forcing_config.forcing_mean_tolerance)
        result["covariance_valid"] = result["maximum_covariance_condition"] <= float(self.problem.forcing_config.max_covariance_condition)
        result["support_valid"] = (
            result["projection_valid"] & result["ress_valid"]
            & result["forcing_valid"] & result["covariance_valid"]
        )
        return result


def _result_root(replicate: int) -> Path:
    return OUTPUT_ROOT / "results" / f"replicate_{replicate:02d}"


def _result_path(replicate: int, role: str, N: int) -> Path:
    return _result_root(replicate) / f"bank_{role}_N{N}.npz"


def _result_summary_path(replicate: int, role: str, N: int) -> Path:
    return _result_root(replicate) / f"bank_{role}_N{N}.json"


def _verify_result(replicate: int, role: str, N: int) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    path, summary_path = _result_path(replicate, role, N), _result_summary_path(replicate, role, N)
    if not path.exists() and not summary_path.exists():
        return None
    if not path.exists() or not summary_path.exists():
        raise RuntimeError(f"incomplete result stage {replicate}/{role}/{N}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (int(summary["replicate_id"]), str(summary["role"]), int(summary["N"])) != (
        replicate, role, N
    ):
        raise RuntimeError(f"result identity changed: {summary_path}")
    if summary["result_sha256"] != file_sha256(path):
        raise RuntimeError(f"sealed result changed: {path}")
    if summary["candidate_panel_sha256"] != file_sha256(PANEL_PATH):
        raise RuntimeError("result candidate panel seal changed")
    if summary["master_bank_manifest_sha256"] != file_sha256(MANIFEST_PATH):
        raise RuntimeError("result master-bank manifest seal changed")
    with np.load(path, allow_pickle=False) as arrays:
        result = {name: np.asarray(arrays[name]) for name in arrays.files}
    return summary, result


def _load_master(path: Path) -> tuple[jax.Array, jax.Array, jax.Array]:
    with np.load(path, allow_pickle=False) as arrays:
        return (
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _pair_inventory(replicate: int) -> dict[str, Any] | None:
    path = _result_root(replicate) / "pair_inventory.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["replicate_id"]) != replicate:
        raise RuntimeError(f"pair inventory identity changed: {path}")
    if payload["candidate_panel_sha256"] != file_sha256(PANEL_PATH):
        raise RuntimeError(f"pair inventory panel seal changed: {path}")
    if payload["master_bank_manifest_sha256"] != file_sha256(MANIFEST_PATH):
        raise RuntimeError(f"pair inventory manifest seal changed: {path}")
    for row in payload["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"pair result changed: {row['path']}")
    pair_path = _result_root(replicate) / "pair_results.npz"
    if payload["completion"]["pair_result_sha256"] != file_sha256(pair_path):
        raise RuntimeError(f"pair completion seal changed: {pair_path}")
    return payload


def evaluate_master_pairs(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    panel = freeze_candidate_panel()
    manifest = freeze_master_manifest(cfg)
    banks = generate_master_banks(cfg)
    if banks["bank_count"] != MASTER_BANK_COUNT:
        raise RuntimeError("all master banks must exist before evaluation")
    bank_hashes = {
        (int(row["replicate_id"]), str(row["role"])): str(row["sha256"])
        for row in banks["banks"]
    }
    etas = np.asarray([row["eta"] for row in panel["rows"]], dtype=np.float64)
    problem = load_selection_galerkin_data(cfg, ARTIFACT_DIR).selection_problem
    evaluator = _NestedEvaluator(problem)
    completed = []
    for record in manifest["replicates"]:
        replicate = int(record["replicate_id"])
        cached_inventory = _pair_inventory(replicate)
        if cached_inventory is not None:
            completed.append({**cached_inventory["completion"], "cache_hit": True})
            if progress is not None:
                progress(replicate, "pair", 0, True, 0.0)
            continue
        stage_paths = []
        timings = {}
        for role in ROLES:
            configurations, velocity, base_weights = _load_master(_bank_path(replicate, role))
            for N in N_LADDER:
                cached = _verify_result(replicate, role, N)
                if cached is None:
                    started = time.perf_counter()
                    result = evaluator.evaluate(etas, configurations, velocity, base_weights, N)
                    elapsed = time.perf_counter() - started
                    path = _result_path(replicate, role, N)
                    _atomic_npz(
                        path,
                        compressed=True,
                        candidate_index=np.arange(PANEL_COUNT, dtype=np.int16),
                        **result,
                    )
                    summary = {
                        "schema_version": 1,
                        "replicate_id": replicate,
                        "role": role,
                        "N": N,
                        "candidate_count": PANEL_COUNT,
                        "candidate_panel_sha256": file_sha256(PANEL_PATH),
                        "master_bank_manifest_sha256": file_sha256(MANIFEST_PATH),
                        "master_bank_sha256": bank_hashes[(replicate, role)],
                        "result_path": str(path.relative_to(OUTPUT_ROOT)),
                        "result_sha256": file_sha256(path),
                        "elapsed_seconds": elapsed,
                        "support_pass_count": int(np.sum(result["support_valid"])),
                        "validation_accessed": False,
                        "tangent_run": False,
                        "full_kf_constructed": False,
                    }
                    _atomic_json(_result_summary_path(replicate, role, N), summary)
                    cache_hit = False
                else:
                    summary, _ = cached
                    elapsed = float(summary["elapsed_seconds"])
                    cache_hit = True
                timings[f"{role}_{N}"] = elapsed
                stage_paths.extend((_result_path(replicate, role, N), _result_summary_path(replicate, role, N)))
                if progress is not None:
                    progress(replicate, role, N, cache_hit, elapsed)
            del configurations, velocity, base_weights

        pass_by_role = {}
        min_ress_by_role = {}
        for role in ROLES:
            pass_by_role[role] = []
            min_ress_by_role[role] = []
            for N in N_LADDER:
                _, result = _verify_result(replicate, role, N)  # type: ignore[misc]
                pass_by_role[role].append(result["support_valid"])
                min_ress_by_role[role].append(result["minimum_ress"])
        pass_A = np.asarray(pass_by_role["A"])
        pass_B = np.asarray(pass_by_role["B"])
        pair_path = _result_root(replicate) / "pair_results.npz"
        pair_arrays = {
            "N": np.asarray(N_LADDER),
            "pass_A": pass_A,
            "pass_B": pass_B,
            "pass_pair": pass_A & pass_B,
            "minimum_ress_A": np.asarray(min_ress_by_role["A"]),
            "minimum_ress_B": np.asarray(min_ress_by_role["B"]),
        }
        if pair_path.exists():
            with np.load(pair_path, allow_pickle=False) as cached_pair:
                if set(cached_pair.files) != set(pair_arrays) or any(
                    not np.array_equal(cached_pair[name], value)
                    for name, value in pair_arrays.items()
                ):
                    raise RuntimeError(f"incomplete pair cache differs: {pair_path}")
        else:
            _atomic_npz(pair_path, compressed=True, **pair_arrays)
        stage_paths.append(pair_path)
        completion = {
            "replicate_id": replicate,
            "pair_result_sha256": file_sha256(pair_path),
            "evaluation_seconds": float(sum(timings.values())),
            "projection_and_diagnostics_seconds_by_N": {
                str(N): float(timings[f"A_{N}"] + timings[f"B_{N}"])
                for N in N_LADDER
            },
        }
        inventory = {
            "schema_version": 1,
            "replicate_id": replicate,
            "candidate_panel_sha256": file_sha256(PANEL_PATH),
            "master_bank_manifest_sha256": file_sha256(MANIFEST_PATH),
            "completion": completion,
            "artifacts": [
                {
                    "path": str(path.relative_to(OUTPUT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in stage_paths
            ],
        }
        _atomic_json(_result_root(replicate) / "pair_inventory.json", inventory)
        completed.append({**completion, "cache_hit": False})
    return {
        "schema_version": 1,
        "completed_pair_count": len(completed),
        "candidate_panel_sha256": file_sha256(PANEL_PATH),
        "master_bank_manifest_sha256": file_sha256(MANIFEST_PATH),
        "completed": completed,
        "total_candidate_bank_N_trajectories": PANEL_COUNT * MASTER_BANK_COUNT * len(N_LADDER),
    }


def _load_all_results() -> dict[str, np.ndarray]:
    fields = (
        "ress_trajectory", "lambda_norm", "maximum_normalized_weight",
        "top_1pct_weight_mass", "sum_q_squared", "empirical_D2",
        "covariance_eigenvalues", "regularized_covariance_eigenvalues",
        "covariance_condition_trajectory",
        "projection_residual_trajectory", "forcing_mean_trajectory",
        "minimum_ress", "controlling_time_index", "support_valid",
    )
    storage: dict[str, Any] = {field: [[None for _ in range(MASTER_BANK_COUNT)] for _ in N_LADDER] for field in fields}
    bank_index = 0
    for replicate in range(PAIR_COUNT):
        for role in ROLES:
            for N_index, N in enumerate(N_LADDER):
                _, result = _verify_result(replicate, role, N)  # type: ignore[misc]
                for field in fields:
                    storage[field][N_index][bank_index] = result[field]
            bank_index += 1
    return {
        field: np.asarray(storage[field])
        for field in fields
    }


def _candidate_summary(
    panel: dict[str, Any], results: dict[str, np.ndarray], pair_pass: np.ndarray
) -> dict[str, Any]:
    rows = []
    for candidate_index, source in enumerate(panel["rows"]):
        levels = []
        for N_index, N in enumerate(N_LADDER):
            values = results["minimum_ress"][N_index, :, candidate_index]
            pair_count = int(np.sum(pair_pass[N_index, :, candidate_index]))
            levels.append(
                {
                    "N": N,
                    "minimum_ress_distribution_32_banks": _distribution(values),
                    "individual_bank_pass_count": int(np.sum(results["support_valid"][N_index, :, candidate_index])),
                    "individual_bank_pass_fraction": float(np.mean(results["support_valid"][N_index, :, candidate_index])),
                    "pair_pass_count": pair_count,
                    "pair_pass_fraction": pair_count / PAIR_COUNT,
                }
            )
        rows.append({**source, "N_levels": levels})
    high_indices = [index for index, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"]
    high = []
    for N_index, N in enumerate(N_LADDER):
        minimum_ress = results["minimum_ress"][N_index][:, high_indices]
        support_valid = results["support_valid"][N_index][:, high_indices]
        pair_valid = pair_pass[N_index][:, high_indices]
        candidate_medians = np.median(minimum_ress, axis=0)
        candidate_p10 = np.quantile(minimum_ress, 0.10, axis=0)
        bank_fractions = np.mean(support_valid, axis=0)
        pair_fractions = np.mean(pair_valid, axis=0)
        pair_counts = np.sum(pair_valid, axis=0)
        high.append(
            {
                "N": N,
                "candidate_median_ress_distribution": _distribution(candidate_medians),
                "candidate_p10_ress_distribution": _distribution(candidate_p10),
                "bank_pass_fraction_distribution": _distribution(bank_fractions),
                "pair_pass_fraction_distribution": _distribution(pair_fractions),
                "candidates_ge_8_of_16_pairs": int(np.sum(pair_counts >= 8)),
                "candidates_ge_12_of_16_pairs": int(np.sum(pair_counts >= 12)),
                "candidates_ge_14_of_16_pairs": int(np.sum(pair_counts >= 14)),
                "candidates_16_of_16_pairs": int(np.sum(pair_counts == 16)),
            }
        )
    return {"schema_version": 1, "rows": rows, "high_pass_panel": high}


def _nested_summary(panel: dict[str, Any], results: dict[str, np.ndarray]) -> dict[str, Any]:
    groups = {
        "law": [0],
        "high_pass": [index for index, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"],
        "controls": [index for index, row in enumerate(panel["rows"]) if "control" in row["panel_role"]],
    }
    rows = []
    for transition in range(len(N_LADDER) - 1):
        for label, indices in groups.items():
            delta = (
                results["minimum_ress"][transition + 1, :, indices]
                - results["minimum_ress"][transition, :, indices]
            ).reshape(-1)
            absolute = np.abs(delta)
            rows.append(
                {
                    "from_N": N_LADDER[transition],
                    "to_N": N_LADDER[transition + 1],
                    "group": label,
                    "signed_delta": _distribution(delta),
                    "absolute_delta": _distribution(absolute),
                }
            )
    return {"rows": rows}


def _pair_pass_matrix(results: dict[str, np.ndarray]) -> np.ndarray:
    support = results["support_valid"]
    return support[:, 0::2, :] & support[:, 1::2, :]


def _flip_summary(pair_pass: np.ndarray, panel: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "all": np.arange(PANEL_COUNT),
        "law": np.asarray([0]),
        "high_pass": np.asarray([index for index, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"]),
        "controls": np.asarray([index for index, row in enumerate(panel["rows"]) if "control" in row["panel_role"]]),
    }
    rows = []
    for transition in range(len(N_LADDER) - 1):
        for label, indices in groups.items():
            old = pair_pass[transition][:, indices]
            new = pair_pass[transition + 1][:, indices]
            total = old.size
            rows.append(
                {
                    "from_N": N_LADDER[transition],
                    "to_N": N_LADDER[transition + 1],
                    "group": label,
                    "fail_to_pass": int(np.sum(~old & new)),
                    "pass_to_fail": int(np.sum(old & ~new)),
                    "total_disagreements": int(np.sum(old != new)),
                    "total_decisions": int(total),
                    "disagreement_rate": float(np.mean(old != new)),
                }
            )
    return {"rows": rows}


def _time7_summary(panel: dict[str, Any], results: dict[str, np.ndarray]) -> dict[str, Any]:
    groups = {
        "all_panel": np.arange(PANEL_COUNT),
        "law": np.asarray([0]),
        "high_pass": np.asarray([index for index, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"]),
        "controls": np.asarray([index for index, row in enumerate(panel["rows"]) if "control" in row["panel_role"]]),
    }
    fields = (
        "ress_trajectory", "lambda_norm", "maximum_normalized_weight",
        "top_1pct_weight_mass", "sum_q_squared", "empirical_D2",
        "covariance_condition_trajectory",
    )
    rows = []
    for N_index, N in enumerate(N_LADDER):
        controlling = results["controlling_time_index"][N_index]
        for label, indices in groups.items():
            entry = {
                "N": N,
                "group": label,
                "controlling_node_frequency": {
                    str(key): value for key, value in sorted(Counter(controlling[:, indices].reshape(-1).tolist()).items())
                },
                "node7_controlling_fraction": float(np.mean(controlling[:, indices] == 7)),
            }
            for field in fields:
                node7 = results[field][N_index, :, indices, 7].reshape(-1)
                other = np.delete(results[field][N_index, :, indices, :], 7, axis=-1).reshape(-1)
                entry[field] = {
                    "node7": _distribution(node7),
                    "other_nodes": _distribution(other),
                }
            eigenvalues = results["covariance_eigenvalues"][N_index, :, indices, 7, :].reshape((-1, results["covariance_eigenvalues"].shape[-1]))
            entry["node7_covariance_eigenvalue_min"] = _distribution(eigenvalues[:, 0])
            entry["node7_covariance_eigenvalue_max"] = _distribution(eigenvalues[:, -1])
            regularized = results["regularized_covariance_eigenvalues"][N_index, :, indices, 7, :].reshape(
                (-1, results["regularized_covariance_eigenvalues"].shape[-1])
            )
            entry["node7_regularized_covariance_eigenvalue_min"] = _distribution(regularized[:, 0])
            entry["node7_regularized_covariance_eigenvalue_max"] = _distribution(regularized[:, -1])
            rows.append(entry)
    return {"rows": rows}


def _law_summary(results: dict[str, np.ndarray], pair_pass: np.ndarray) -> dict[str, Any]:
    rows = []
    for N_index, N in enumerate(N_LADDER):
        values = results["minimum_ress"][N_index, :, 0]
        controlling = results["controlling_time_index"][N_index, :, 0]
        rows.append(
            {
                "N": N,
                "individual_bank_pass_count": int(np.sum(results["support_valid"][N_index, :, 0])),
                "pair_pass_count": int(np.sum(pair_pass[N_index, :, 0])),
                "minimum_ress_distribution": _distribution(values),
                "controlling_node_frequency": {
                    str(key): value for key, value in sorted(Counter(controlling.tolist()).items())
                },
            }
        )
    return {"rows": rows}


def _interpret(
    panel: dict[str, Any], results: dict[str, np.ndarray], pair_pass: np.ndarray,
    candidate: dict[str, Any], nested: dict[str, Any], flips: dict[str, Any], time7: dict[str, Any],
) -> dict[str, Any]:
    candidate_std = np.std(results["minimum_ress"], axis=1)
    median_std = np.median(candidate_std, axis=1)
    std_ratio = float(median_std[-1] / median_std[0])
    high_nested = [row for row in nested["rows"] if row["group"] == "high_pass"]
    absolute_medians = [row["absolute_delta"]["median"] for row in high_nested]
    all_flips = [row for row in flips["rows"] if row["group"] == "all"]
    first_flip, final_flip = all_flips[0]["disagreement_rate"], all_flips[-1]["disagreement_rate"]
    law_median = candidate["rows"][0]["N_levels"][-1]["minimum_ress_distribution_32_banks"]["median"]
    high_p10_median = candidate["high_pass_panel"][-1]["candidate_p10_ress_distribution"]["median"]
    time7_last = next(row for row in time7["rows"] if row["N"] == MASTER_N and row["group"] == "high_pass")
    node7_fraction = time7_last["node7_controlling_fraction"]
    node7_ress = time7_last["ress_trajectory"]["node7"]["median"]
    other_ress = time7_last["ress_trajectory"]["other_nodes"]["median"]
    variance_reduced = std_ratio <= 0.75 and absolute_medians[-1] <= absolute_medians[0]
    threshold_stabilized = final_flip <= 0.10 and final_flip < first_flip
    boundary_overlap = 0.045 <= law_median <= 0.055 or 0.045 <= high_p10_median <= 0.055
    stable_adequate_support = law_median > 0.055 and high_p10_median > 0.055
    stable_inadequate_support = law_median < 0.045 or high_p10_median < 0.045
    persistent_node7 = node7_fraction >= 0.70 and node7_ress < 0.05 and node7_ress < other_ress

    if variance_reduced and threshold_stabilized and stable_adequate_support and not persistent_node7:
        label = "N_LIMITED_SUPPORT_ESTIMATION"
        next_step = "Consider a separately frozen official protocol using the smallest N shown to be converged; compare N=32768 and N=65536 before choosing."
    elif variance_reduced and threshold_stabilized and boundary_overlap:
        label = "BORDERLINE_POPULATION_OVERLAP"
        next_step = "Do not merely increase N or add replicate votes; examine the scientific meaning of the 0.05 support gate relative to the frozen proposal."
    elif persistent_node7 and (not variance_reduced or stable_inadequate_support):
        label = "PERSISTENT_REFERENCE_PROPOSAL_MISMATCH"
        next_step = "Run a separately defined development study of the reference proposal near the controlling physical-time region; do not alter it here."
    else:
        label = "MIXED_N_AND_PROPOSAL_EFFECT"
        next_step = "Larger N improves finite-sample behavior, but persistent node-7 overlap limitations must be resolved before an official Full run."
    return {
        "label": label,
        "evidence": {
            "median_candidate_bank_std_by_N": {str(N): float(value) for N, value in zip(N_LADDER, median_std, strict=True)},
            "std_ratio_N65536_over_N8192": std_ratio,
            "high_pass_median_absolute_change_by_transition": absolute_medians,
            "first_pair_decision_flip_rate": first_flip,
            "final_pair_decision_flip_rate": final_flip,
            "Law_N65536_median_ress": law_median,
            "high_pass_N65536_median_candidate_p10_ress": high_p10_median,
            "high_pass_N65536_node7_controlling_fraction": node7_fraction,
            "high_pass_N65536_node7_median_ress": node7_ress,
            "high_pass_N65536_other_node_median_ress": other_ress,
            "variance_materially_reduced": variance_reduced,
            "threshold_decisions_stabilized": threshold_stabilized,
            "threshold_boundary_overlap": boundary_overlap,
            "stable_adequate_support": stable_adequate_support,
            "stable_inadequate_support": stable_inadequate_support,
            "persistent_node7_overlap": persistent_node7,
        },
        "recommended_next_scientific_step": next_step,
    }


def _write_result_indices() -> list[Path]:
    paths = []
    for N in N_LADDER:
        index_path = OUTPUT_ROOT / f"results_N{N}.json"
        rows = []
        for replicate in range(PAIR_COUNT):
            for role in ROLES:
                path = _result_path(replicate, role, N)
                rows.append(
                    {
                        "replicate_id": replicate,
                        "role": role,
                        "path": str(path.relative_to(OUTPUT_ROOT)),
                        "sha256": file_sha256(path),
                    }
                )
        _atomic_json(index_path, {"schema_version": 1, "N": N, "rows": rows})
        paths.append(index_path)
    return paths


def _verify_cached_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists() and not INVENTORY_PATH.exists():
        return None
    if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
        raise RuntimeError("incomplete sealed convergence summary")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"sealed convergence artifact changed: {row['path']}")
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {**payload, "cache_hit": True}


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    panel = freeze_candidate_panel()
    freeze_master_manifest(cfg)
    evaluate_master_pairs(cfg)
    cached = _verify_cached_summary()
    if cached is not None:
        return cached
    started = time.perf_counter()
    results = _load_all_results()
    pair_pass = _pair_pass_matrix(results)
    candidate = _candidate_summary(panel, results, pair_pass)
    nested = _nested_summary(panel, results)
    flips = _flip_summary(pair_pass, panel)
    time7 = _time7_summary(panel, results)
    law = _law_summary(results, pair_pass)
    interpretation = _interpret(panel, results, pair_pass, candidate, nested, flips, time7)
    _atomic_json(LAW_PATH, law)
    _atomic_json(CANDIDATE_PATH, candidate)
    _atomic_json(TIME7_PATH, time7)
    _atomic_json(FLIP_PATH, flips)
    indices = _write_result_indices()
    pair_inventories = [_result_root(replicate) / "pair_inventory.json" for replicate in range(PAIR_COUNT)]
    bank_inventory = json.loads(BANK_INVENTORY_PATH.read_text(encoding="utf-8"))
    completion = [json.loads(path.read_text(encoding="utf-8"))["completion"] for path in pair_inventories]
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_verified": True,
        "diagnostic_candidate_count": PANEL_COUNT,
        "master_pair_count": PAIR_COUNT,
        "master_bank_count": MASTER_BANK_COUNT,
        "N_ladder": list(N_LADDER),
        "nested_prefixes": True,
        "per_bank_ress_threshold": MINIMUM_RESS,
        "total_candidate_bank_N_trajectories": PANEL_COUNT * MASTER_BANK_COUNT * len(N_LADDER),
        "law_convergence": law["rows"],
        "high_pass_panel": candidate["high_pass_panel"],
        "nested_convergence": nested["rows"],
        "threshold_flips": flips["rows"],
        "interpretation": interpretation,
        "timings": {
            "master_bank_generation_seconds": bank_inventory["total_generation_seconds_this_invocation"],
            "evaluation_seconds_total": float(sum(row["evaluation_seconds"] for row in completion)),
            "projection_and_diagnostics_seconds_by_N": {
                str(N): float(sum(
                    row["projection_and_diagnostics_seconds_by_N"][str(N)]
                    for row in completion
                ))
                for N in N_LADDER
            },
            "summary_seconds": time.perf_counter() - started,
        },
        "firewalls": {
            "tangent_run": False,
            "full_kf_constructed": False,
            "eigensolve_run": False,
            "deep_ritz_run": False,
            "validation_accessed": False,
            "official_protocol_created": False,
            "selection_frozen": False,
        },
        "seals": {
            "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
            "candidate_panel_sha256": file_sha256(PANEL_PATH),
            "master_bank_manifest_sha256": file_sha256(MANIFEST_PATH),
            "master_bank_inventory_sha256": file_sha256(BANK_INVENTORY_PATH),
        },
    }
    _atomic_json(SUMMARY_PATH, payload)
    artifact_paths = [
        SOURCE_SEAL_PATH, PANEL_PATH, MANIFEST_PATH, BANK_INVENTORY_PATH,
        *indices, LAW_PATH, CANDIDATE_PATH, TIME7_PATH, FLIP_PATH,
        *pair_inventories, SUMMARY_PATH,
    ]
    inventory = {
        "schema_version": 1,
        "artifact_count": len(artifact_paths),
        "artifacts": [
            {
                "path": str(path.relative_to(OUTPUT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in artifact_paths
        ],
    }
    _atomic_json(INVENTORY_PATH, inventory)
    return {**payload, "cache_hit": False}


def run(
    cfg: dict[str, Any],
    bank_progress: Any | None = None,
    evaluation_progress: Any | None = None,
) -> dict[str, Any]:
    freeze_candidate_panel()
    freeze_master_manifest(cfg)
    generate_master_banks(cfg, progress=bank_progress)
    evaluate_master_pairs(cfg, progress=evaluation_progress)
    return summarize(cfg)


__all__ = [
    "BANK_INVENTORY_PATH",
    "CONTROL_COUNT",
    "HIGH_PASS_COUNT",
    "MANIFEST_PATH",
    "MASTER_N",
    "MINIMUM_RESS",
    "N_LADDER",
    "OUTPUT_ROOT",
    "PAIR_COUNT",
    "PANEL_COUNT",
    "PANEL_PATH",
    "SOURCE_SEAL_PATH",
    "SUMMARY_PATH",
    "_NestedEvaluator",
    "derive_seed",
    "evaluate_master_pairs",
    "freeze_candidate_panel",
    "freeze_master_manifest",
    "generate_master_banks",
    "run",
    "summarize",
    "verify_and_freeze_sources",
]
