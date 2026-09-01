"""JAX-only development diagnosis of the terminal V3.4 energy failure."""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto_v3_chunked_guard_run as v3_runtime
from .galerkin import GalerkinSystem, aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import GalerkinCertificateThresholds, prefix_dictionary
from .galerkin_only_data import GalerkinReferenceBank
from .full_gradient import forcing_state, reconstruct_moments
from .jax_galerkin_v2 import forcing_payload
from .production_basis import HybridInvariantDictionary, load_dictionary
from .production_galerkin import assemble_hybrid_system, audit_hybrid_solutions
from .galerkin_only import _fourier_suffix_values_gradients, _next_wavevectors


base = v3_runtime.study.base
V3_ROOT = v3_runtime.study.OUTPUT_ROOT
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs" / "development_v4_energy_diagnosis"
PROTOCOL_PATH = OUTPUT_ROOT / "development_protocol.json"
OPERATIONAL_AMENDMENT_PATHS = tuple(
    OUTPUT_ROOT / f"development_operational_amendment_v{index}.json"
    for index in (1, 2)
)
BANK_MANIFEST_PATH = OUTPUT_ROOT / "banks" / "manifest.json"
DICTIONARY_PATH = OUTPUT_ROOT / "artifacts" / "dictionary_K440.npz"
DIAGNOSTICS_PATH = OUTPUT_ROOT / "energy_failure_diagnostics.json"
RUNNER_PATH = ROOT / "v4_energy_development_run.py"
TEST_PATH = ROOT / "test_v4_energy_development.py"

DIAGNOSTIC_ROOT = 20261101
TIME_INDEX_MIDPOINT = 6
N_LADDER = (16384, 32768, 65536, 131072)
N_MAX = 131072
SPLIT_COUNT = 4
K_LADDER = (120, 180, 220, 280, 360, 440)
K_N_GRID = (180, 280, 360, 440)
N_K_GRID = (32768, 65536, 131072)
CHUNK_SIZE = 256
RANK_TOLERANCE = 1.0e-12
ENERGY_FLOOR = 1.0e-12
ROLE_IDS = {
    "scaling_fit": 4001,
    "scaling_audit": 4002,
    **{f"split_{index}_fit": 4100 + 2 * index for index in range(SPLIT_COUNT)},
    **{f"split_{index}_audit": 4101 + 2 * index for index in range(SPLIT_COUNT)},
}
ROLE_COUNTS = {
    "scaling_fit": N_MAX,
    "scaling_audit": N_MAX,
    **{f"split_{index}_fit": 65536 for index in range(SPLIT_COUNT)},
    **{f"split_{index}_audit": 65536 for index in range(SPLIT_COUNT)},
}
_PROBLEM_CACHE: Any | None = None
_DICTIONARY_CACHE: dict[int, HybridInvariantDictionary] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _atomic_bytes(path: Path, data: bytes) -> None:
    resolved = path.resolve()
    if resolved != OUTPUT_ROOT.resolve() and OUTPUT_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"development output escaped namespace: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite development artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        generated = Path(temporary + ".npz")
        os.replace(generated, path)
    finally:
        for candidate in (Path(temporary), Path(temporary + ".npz")):
            if candidate.exists():
                candidate.unlink()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _source_hashes() -> dict[str, str]:
    return {
        Path(__file__).name: _sha256(Path(__file__)),
        RUNNER_PATH.name: _sha256(RUNNER_PATH),
        TEST_PATH.name: _sha256(TEST_PATH),
    }


def _call_graph() -> dict[str, Any]:
    forbidden_modules = {"mfsi.galerkin_tesseract", ".pareto_v2_selection"}
    forbidden_calls = {"assemble_galerkin_chunk_tesseract", "evaluate_galerkin_action"}
    violations = []
    sources = (Path(__file__), RUNNER_PATH)
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append({"source": source.name, "import": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                if module in forbidden_modules:
                    violations.append({"source": source.name, "import": module})
            elif isinstance(node, ast.Call):
                called = (
                    node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if called in forbidden_calls:
                    violations.append({"source": source.name, "call": called})
    return {
        "passed": not violations,
        "native_galerkin_reachable": bool(violations),
        "violations": violations,
        "sources": {path.name: _sha256(path) for path in sources},
    }


def _role_seed(role: str) -> int:
    key = jax.random.fold_in(jax.random.PRNGKey(DIAGNOSTIC_ROOT), ROLE_IDS[role])
    adapted = jax.random.fold_in(key, 0x5EED)
    return int(jax.random.bits(adapted, (), dtype=jnp.uint32)) % (2**31 - 1)


def _geometry_protocol() -> list[dict[str, Any]]:
    selection = _read(V3_ROOT / "selection" / "selection_seal.json")
    unique: dict[str, dict[str, Any]] = {}
    for row in selection["rows"]:
        unique.setdefault(row["eta_sha256"], {
            "id": "v3_" + ("law" if row["method"] == "Law" else f"{row['allowance_percent']:g}pct"),
            "group": "v3_selected",
            "eta": row["eta"],
            "eta_sha256": row["eta_sha256"],
            "risk": row["exact_risk"],
        })
    parent = _read(
        ROOT / "outputs" / "official_b1_galerkin_pareto_v2_1_single_seed_amended"
        / "amendment_final_summary.json"
    )
    parent_law = next(row for row in parent["rows"] if row["method"] == "Law")
    parent_full = next(
        row for row in parent["rows"]
        if row["method"] == "Full" and row["allowance_percent"] == 2.0
    )
    extras = [
        {
            "id": "historical_v2_1_law", "group": "historical",
            "eta": parent_law["eta"], "eta_sha256": parent_law["eta_sha256"],
            "risk": parent_law["exact_risk"],
        },
        {
            "id": "historical_v2_1_full_2pct", "group": "historical",
            "eta": parent_full["eta"], "eta_sha256": parent_full["eta_sha256"],
            "risk": parent_full["exact_risk"],
        },
    ]
    exact = _read(V3_ROOT / "feasibility" / "exact_receipts.json")["rows"]
    excluded = set(unique) | {row["eta_sha256"] for row in extras}
    supported = [row for row in exact if row["jointly_supported"] and row["eta_sha256"] not in excluded]
    hash_ordered = sorted(
        supported,
        key=lambda row: hashlib.sha256(("v4-geometry:" + row["eta_sha256"]).encode()).hexdigest(),
    )
    for index, row in enumerate(hash_ordered[:3]):
        extras.append({
            "id": f"hash_supported_{index}", "group": "hash_supported",
            "eta": row["eta"], "eta_sha256": row["eta_sha256"],
            "risk": row["exact_scientific_risk"],
        })
    excluded.update(row["eta_sha256"] for row in extras)
    high = sorted(
        (row for row in exact if row["jointly_supported"] and row["eta_sha256"] not in excluded),
        key=lambda row: (-row["exact_scientific_risk"], row["eta_sha256"]),
    )[:3]
    for index, row in enumerate(high):
        extras.append({
            "id": f"high_risk_supported_{index}", "group": "high_risk_supported",
            "eta": row["eta"], "eta_sha256": row["eta_sha256"],
            "risk": row["exact_scientific_risk"],
        })
    return [*unique.values(), *extras]


def freeze(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    v3_runtime.activate()
    if PROTOCOL_PATH.exists():
        return require_protocol()
    existing = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
    if existing:
        raise RuntimeError(f"development outputs exist before freeze: {existing}")
    graph = _call_graph()
    if not graph["passed"]:
        raise RuntimeError(f"native Galerkin reachable: {graph['violations']}")
    body = {
        "schema_version": 1,
        "status": "FROZEN_DEVELOPMENT_ONLY_BEFORE_NEW_BANKS_OR_DIAGNOSTIC_OUTCOMES",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "create_v4_authority": False,
        "v3_4_terminal_authority_immutable": True,
        "v3_protocol_sha256": _read(V3_ROOT / "protocol_v3_support_robust.json")["v3_protocol_sha256"],
        "v3_selection_seal_sha256": _sha256(V3_ROOT / "selection" / "selection_seal.json"),
        "v3_heldout_results_sha256": _sha256(V3_ROOT / "heldout_validation" / "results.json"),
        "v3_failure_diagnostic_sha256": _sha256(V3_ROOT / "heldout_validation" / "failure_diagnostic.json"),
        "v3_complete_tree_sha256_before_development": _tree_sha256(V3_ROOT),
        "diagnostic_root_seed": DIAGNOSTIC_ROOT,
        "diagnostic_seed_schedule_not_scientific_replicates": True,
        "alternate_authoritative_root_seeds_tested": [],
        "roles": [{
            "role": role,
            "role_id": ROLE_IDS[role],
            "samples": ROLE_COUNTS[role],
            "integer_seed_adapter": _role_seed(role),
            "derivation": f"fold_in(PRNGKey({DIAGNOSTIC_ROOT}), {ROLE_IDS[role]})",
        } for role in ROLE_IDS],
        "geometries": _geometry_protocol(),
        "selected_geometry_ids": ["v3_law", "v3_0.5pct", "v3_1pct", "v3_2pct"],
        "representative_geometry_id": "v3_law",
        "N_ladder": list(N_LADDER),
        "N_maximum": N_MAX,
        "N_262144_omitted_preoutcome_reason": (
            "32 GiB host RAM and 24 GiB GPU make paired 262,144 banks plus K440 diagnostics "
            "impractical; stop at 131,072 unless the frozen ladder is inconclusive"
        ),
        "equal_N_scaling_all_selected_geometries": True,
        "separate_fit_audit_scaling_representative_only": True,
        "repeated_split_count": SPLIT_COUNT,
        "repeated_splits_representative_only": True,
        "K_ladder": list(K_LADDER),
        "K_by_N_grid": {"K": list(K_N_GRID), "N": list(N_K_GRID)},
        "K_extension": (
            "preserve frozen K280 prefix exactly; append next ordered Fourier cos/sin pairs; "
            "normalize suffix on eta-independent scaling_fit master bank"
        ),
        "rank_tolerance": RANK_TOLERANCE,
        "energy_floor": ENERGY_FLOOR,
        "chunk_size": CHUNK_SIZE,
        "dtype": "float64",
        "scientific_backend": "jax",
        "native_galerkin_allowed": False,
        "numpy_scipy_scope": "independent solve of saved 280x280 matrices only",
        "source_hashes": _source_hashes(),
        "jax_only_call_graph": graph,
    }
    protocol = {**body, "protocol_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    _atomic_json(PROTOCOL_PATH, protocol)
    if progress:
        progress(f"development protocol frozen: {protocol['protocol_sha256']}")
    return protocol


def require_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("development protocol is not frozen")
    protocol = _read(PROTOCOL_PATH)
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != protocol["protocol_sha256"]:
        raise RuntimeError("development protocol digest mismatch")
    observed_sources = _source_hashes()
    if protocol["source_hashes"] != observed_sources:
        available_amendments = [path for path in OPERATIONAL_AMENDMENT_PATHS if path.exists()]
        if not available_amendments:
            raise RuntimeError("development scientific source changed after freeze")
        amendment = _read(available_amendments[-1])
        if amendment["original_protocol_sha256"] != protocol["protocol_sha256"]:
            raise RuntimeError("operational amendment targets another protocol")
        if amendment["amended_source_hashes"] != observed_sources:
            raise RuntimeError("development source differs from sealed operational amendment")
    if _sha256(V3_ROOT / "selection" / "selection_seal.json") != protocol["v3_selection_seal_sha256"]:
        raise RuntimeError("V3.4 selection seal changed")
    if _sha256(V3_ROOT / "heldout_validation" / "results.json") != protocol["v3_heldout_results_sha256"]:
        raise RuntimeError("V3.4 heldout result changed")
    return protocol


def verify_v3_immutable() -> dict[str, Any]:
    protocol = require_protocol()
    observed = _tree_sha256(V3_ROOT)
    expected = protocol["v3_complete_tree_sha256_before_development"]
    return {
        "passed": observed == expected,
        "expected_sha256": expected,
        "observed_sha256": observed,
    }


def _bank_path(role: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{role}_N{ROLE_COUNTS[role]}.npz"


def generate_banks(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol()
    if BANK_MANIFEST_PATH.exists():
        return _read(BANK_MANIFEST_PATH)
    cfg = base.effective_config()
    with np.load(V3_ROOT / "heldout_validation" / "truth.npz", allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
    truth_model = base.SkyrmionTruth(base._physics_config(cfg))
    flow = base.load_reference(V3_ROOT / "artifacts" / "reference.npz")
    rows = []
    for role, count in ROLE_COUNTS.items():
        path = _bank_path(role)
        record_path = path.with_suffix(".json")
        if record_path.exists():
            rows.append(_read(record_path))
            continue
        started = time.perf_counter()
        x, velocity, weights, initial_hash = base._rollout_bank(
            cfg, flow, truth_model, times, seed=_role_seed(role), samples=count
        )
        _atomic_npz(
            path,
            configurations=x,
            velocity=velocity,
            base_weights=weights,
            diagnostic_root_seed=np.asarray(DIAGNOSTIC_ROOT),
            derived_role_seed=np.asarray(_role_seed(role)),
        )
        record = {
            "role": role, "samples": count, "sha256": _sha256(path),
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "initial_state_sha256": initial_hash,
            "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(record_path, record)
        rows.append(record)
        if progress:
            progress(f"development bank {role} N={count}")
        del x, velocity, weights
        gc.collect()
    manifest = {
        "schema_version": 1,
        "passed": len({row["initial_state_sha256"] for row in rows}) == len(rows),
        "protocol_sha256": require_protocol()["protocol_sha256"],
        "diagnostic_root_seed": DIAGNOSTIC_ROOT,
        "roles_are_not_scientific_replicates": True,
        "rows": rows,
    }
    _atomic_json(BANK_MANIFEST_PATH, manifest)
    return manifest


def _load_bank(role: str, count: int | None = None) -> GalerkinReferenceBank:
    with np.load(_bank_path(role), allow_pickle=False) as arrays:
        stop = ROLE_COUNTS[role] if count is None else int(count)
        weights = jnp.asarray(arrays["base_weights"][:, :stop], dtype=jnp.float64)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"][:, :stop], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"][:, :stop], dtype=jnp.float64),
            weights,
        )


def _v3_heldout_bank(kind: str) -> GalerkinReferenceBank:
    count = 16384
    path = V3_ROOT / "heldout_validation" / f"reference_{kind}_N{count}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def build_dictionary(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol()
    generate_banks(progress)
    metadata_path = DICTIONARY_PATH.with_suffix(".json")
    if metadata_path.exists():
        return _read(metadata_path)
    old_path = V3_ROOT / "artifacts" / "dictionary_K280.npz"
    old = load_dictionary(old_path, box=base.BOX)
    if old.size != 280:
        raise RuntimeError("frozen dictionary is not K280")
    added_vectors = (440 - 280) // 2
    suffix_vectors = _next_wavevectors(old.wavevectors, added_vectors, old.box)
    bank = _load_bank("scaling_fit", N_MAX)
    evaluator = jax.jit(lambda rows: _fourier_suffix_values_gradients(rows, suffix_vectors))
    means, scales = [], []
    started = time.perf_counter()
    for time_index in range(int(bank.configurations.shape[0])):
        mean = jnp.zeros((160,), dtype=jnp.float64)
        energy = jnp.zeros((160,), dtype=jnp.float64)
        for start in range(0, N_MAX, CHUNK_SIZE):
            values, gradients = evaluator(bank.configurations[time_index, start:start + CHUNK_SIZE])
            weights = bank.base_weights[time_index, start:start + CHUNK_SIZE]
            mean += jnp.einsum("n,nk->k", weights, values)
            energy += jnp.einsum("n,nkpd,nkpd->k", weights, gradients, gradients)
        means.append(mean)
        scales.append(jnp.sqrt(jnp.maximum(energy, 1.0e-12)))
    kinds = jnp.concatenate((old.feature_kind, jnp.tile(jnp.asarray([0, 1], dtype=jnp.int32), added_vectors)))
    sources = jnp.concatenate((
        old.feature_source_index,
        jnp.repeat(jnp.arange(old.wavevectors.shape[0], old.wavevectors.shape[0] + added_vectors, dtype=jnp.int32), 2),
    ))
    dictionary = HybridInvariantDictionary(
        box=old.box,
        wavevectors=jnp.concatenate((old.wavevectors, suffix_vectors)),
        radial_centers=old.radial_centers,
        radial_widths=old.radial_widths,
        feature_kind=kinds,
        feature_source_index=sources,
        base_means=jnp.concatenate((old.base_means, jnp.stack(means)), axis=1),
        energy_scales=jnp.concatenate((old.energy_scales, jnp.stack(scales)), axis=1),
    )
    if not (
        np.array_equal(np.asarray(dictionary.feature_kind[:280]), np.asarray(old.feature_kind))
        and np.array_equal(np.asarray(dictionary.feature_source_index[:280]), np.asarray(old.feature_source_index))
        and np.array_equal(np.asarray(dictionary.base_means[:, :280]), np.asarray(old.base_means))
        and np.array_equal(np.asarray(dictionary.energy_scales[:, :280]), np.asarray(old.energy_scales))
    ):
        raise RuntimeError("K440 dictionary changed frozen K280 prefix")
    _atomic_npz(
        DICTIONARY_PATH,
        wavevectors=dictionary.wavevectors,
        radial_centers=dictionary.radial_centers,
        radial_widths=dictionary.radial_widths,
        feature_kind=dictionary.feature_kind,
        feature_source_index=dictionary.feature_source_index,
        base_means=dictionary.base_means,
        energy_scales=dictionary.energy_scales,
    )
    metadata = {
        "schema_version": 1,
        "sha256": _sha256(DICTIONARY_PATH),
        "size": 440,
        "frozen_K280_sha256": _sha256(old_path),
        "frozen_K280_prefix_exact": True,
        "suffix": "next ordered Fourier cosine/sine pairs",
        "suffix_normalization_bank": "scaling_fit_N131072",
        "normalization_seconds": time.perf_counter() - started,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _problem() -> Any:
    global _PROBLEM_CACHE
    if _PROBLEM_CACHE is None:
        cfg = base.effective_config()
        root = V3_ROOT / "heldout_validation"
        with np.load(root / "truth.npz", allow_pickle=False) as arrays:
            times = jnp.asarray(arrays["times"], dtype=jnp.float64)
            truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        with np.load(root / "observation_noise.npz", allow_pickle=False) as arrays:
            noise = jnp.asarray(arrays["detector_noise"], dtype=jnp.float64)
        problem = base._problem(
            cfg, truth, times, noise_seed=base.role_seed("heldout_observation_noise")
        )
        _PROBLEM_CACHE = problem.__class__(
            problem.truth_configurations,
            problem.times,
            problem.time_weights,
            problem.acquisition_indices,
            problem.finite_configuration_count,
            noise,
            problem.family,
            problem.reconstructor,
            problem.projection_config,
            problem.forcing_config,
            "jax",
            problem.box,
        )
    return _PROBLEM_CACHE


def _geometry(geometry_id: str) -> dict[str, Any]:
    return next(row for row in require_protocol()["geometries"] if row["id"] == geometry_id)


def _dictionary(K: int) -> HybridInvariantDictionary:
    if K not in _DICTIONARY_CACHE:
        if K <= 280:
            full = load_dictionary(V3_ROOT / "artifacts" / "dictionary_K280.npz", box=base.BOX)
        else:
            build_dictionary()
            full = load_dictionary(DICTIONARY_PATH, box=base.BOX)
        _DICTIONARY_CACHE[K] = prefix_dictionary(full, K)
    return _DICTIONARY_CACHE[K]


def _cache_key(geometry: dict[str, Any], role: str, N: int, K: int) -> str:
    if role in ROLE_COUNTS:
        bank_digest = _sha256(_bank_path(role))
    elif role in {"v3_heldout_fit", "v3_heldout_audit"}:
        kind = role.removeprefix("v3_heldout_")
        bank_digest = _sha256(
            V3_ROOT / "heldout_validation" / f"reference_{kind}_N16384.npz"
        )
    else:
        raise ValueError(f"unknown bank role: {role}")
    return hashlib.sha256(_canonical({
        "eta": geometry["eta_sha256"], "role": role, "N": N, "K": K,
        "bank": bank_digest,
        "dictionary": _sha256(
            V3_ROOT / "artifacts" / "dictionary_K280.npz" if K <= 280 else DICTIONARY_PATH
        ),
    })).hexdigest()[:24]


def _system_cache_path(geometry: dict[str, Any], role: str, N: int, K: int) -> Path:
    return OUTPUT_ROOT / "systems" / geometry["id"] / f"{role}_N{N}_K{K}_{_cache_key(geometry, role, N, K)}.npz"


def _assemble_system(
    geometry: dict[str, Any], role: str, bank: GalerkinReferenceBank, K: int,
) -> tuple[GalerkinSystem, Any, Any, dict[str, float]]:
    problem = _problem()
    eta = jnp.asarray(geometry["eta"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, problem)
    started = time.perf_counter()
    state = forcing_state(eta, problem, bank, reconstruction)
    jax.block_until_ready(state.projection.weights)
    forcing_seconds = time.perf_counter() - started
    cache_path = _system_cache_path(geometry, role, int(bank.configurations.shape[1]), K)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as arrays:
            empty = jnp.zeros((0,), dtype=jnp.float64)
            system = GalerkinSystem(
                gram=jnp.asarray(arrays["gram"]), load=jnp.asarray(arrays["load"]),
                basis_means=jnp.asarray(arrays["basis_means"]),
                centered_basis=empty, weights=empty, forcing=empty,
                raw_symmetry_residual=jnp.asarray(arrays["raw_symmetry_residual"]),
                forcing_mean=jnp.asarray(arrays["forcing_mean"]),
            )
        assembly_seconds = 0.0
    else:
        started = time.perf_counter()
        system = assemble_hybrid_system(
            _dictionary(K), bank, state.projection.weights, state.forcing,
            chunk_size=CHUNK_SIZE,
        )
        jax.block_until_ready(system.gram)
        assembly_seconds = time.perf_counter() - started
        _atomic_npz(
            cache_path, gram=system.gram, load=system.load,
            basis_means=system.basis_means,
            raw_symmetry_residual=system.raw_symmetry_residual,
            forcing_mean=system.forcing_mean,
        )
    return system, state, reconstruction, {
        "forcing_seconds": forcing_seconds, "assembly_seconds": assembly_seconds,
    }


def _energy_rows(coefficients: Any, fit: GalerkinSystem, audit: GalerkinSystem) -> list[dict[str, Any]]:
    a = jnp.asarray(coefficients)
    Kf, ff, Ka, fa = fit.gram, fit.load, audit.gram, audit.load
    fit_q = jnp.einsum("ti,tij,tj->t", a, Kf, a)
    fit_l = jnp.einsum("ti,ti->t", a, ff)
    audit_q = jnp.einsum("ti,tij,tj->t", a, Ka, a)
    audit_l = jnp.einsum("ti,ti->t", a, fa)
    numerator = jnp.abs(audit_q + audit_l)
    denominator = jnp.maximum(audit_q + jnp.abs(audit_l), ENERGY_FLOOR)
    ratio = numerator / denominator
    median_denominator = jnp.median(denominator)
    rows = []
    for index in range(int(a.shape[0])):
        rows.append({
            "time_index": index,
            "t": index / 12.0,
            "numerator": float(numerator[index]),
            "denominator": float(denominator[index]),
            "relative_energy_residual": float(ratio[index]),
            "fit_quadratic": float(fit_q[index]),
            "fit_linear": float(fit_l[index]),
            "audit_quadratic": float(audit_q[index]),
            "audit_linear": float(audit_l[index]),
            "coefficient_norm": float(jnp.linalg.norm(a[index])),
            "fit_load_norm": float(jnp.linalg.norm(ff[index])),
            "audit_load_norm": float(jnp.linalg.norm(fa[index])),
            "numerator_over_audit_action_density": float(numerator[index] / jnp.maximum(audit_q[index], ENERGY_FLOOR)),
            "numerator_over_fit_action_density": float(numerator[index] / jnp.maximum(fit_q[index], ENERGY_FLOOR)),
            "numerator_over_audit_load_norm": float(numerator[index] / jnp.maximum(jnp.linalg.norm(fa[index]), ENERGY_FLOOR)),
            "numerator_over_median_denominator": float(numerator[index] / jnp.maximum(median_denominator, ENERGY_FLOOR)),
        })
    return rows


def _solve(system: GalerkinSystem) -> Any:
    return rank_aware_quadratic_solve(
        system.gram, system.load, relative_rank_tolerance=RANK_TOLERANCE
    )


def _pair_result(
    geometry: dict[str, Any], fit_role: str, fit_bank: GalerkinReferenceBank,
    audit_role: str, audit_bank: GalerkinReferenceBank, K: int,
    *, physical_certificate: bool = True,
) -> dict[str, Any]:
    fit, fit_state, reconstruction, fit_timing = _assemble_system(
        geometry, fit_role, fit_bank, K
    )
    audit, audit_state, _, audit_timing = _assemble_system(
        geometry, audit_role, audit_bank, K
    )
    started = time.perf_counter()
    solve = _solve(fit)
    jax.block_until_ready(solve.coefficients)
    solve_seconds = time.perf_counter() - started
    energy = _energy_rows(solve.coefficients, fit, audit)
    certificate = None
    certificate_seconds = 0.0
    if physical_certificate:
        started = time.perf_counter()
        adapter = SimpleNamespace(selection_problem=_problem(), ritz_audit_bank=audit_bank)
        certificate = audit_hybrid_solutions(
            _dictionary(K), solve.coefficients[None], adapter,
            jnp.asarray(geometry["eta"], dtype=jnp.float64), reconstruction,
            audit_state,
            GalerkinCertificateThresholds(
                **base.effective_config()["production_galerkin"]["certificate_thresholds"]
            ),
            chunk_size=CHUNK_SIZE,
        )[0]
        certificate_seconds = time.perf_counter() - started
    fit_forcing = forcing_payload(fit_state, _problem())
    audit_forcing = forcing_payload(audit_state, _problem())
    certificate_energy_difference = None
    if certificate is not None:
        certificate_energy_difference = abs(
            float(certificate["maximum_energy_residual"])
            - max(row["relative_energy_residual"] for row in energy)
        )
    time_weights = _problem().time_weights
    fit_action = float(jnp.sum(time_weights * jnp.asarray([row["fit_quadratic"] for row in energy])))
    audit_action = float(jnp.sum(time_weights * jnp.asarray([row["audit_quadratic"] for row in energy])))
    eigenvalues = jnp.linalg.eigvalsh(fit.gram)
    largest = eigenvalues[:, -1]
    retained = eigenvalues > RANK_TOLERANCE * largest[:, None]
    smallest = jnp.min(jnp.where(retained, eigenvalues, jnp.inf), axis=1)
    return {
        "geometry_id": geometry["id"], "eta_sha256": geometry["eta_sha256"],
        "fit_role": fit_role, "audit_role": audit_role,
        "fit_N": int(fit_bank.configurations.shape[1]),
        "audit_N": int(audit_bank.configurations.shape[1]), "K": K,
        "fit_action": fit_action, "audit_action": audit_action,
        "maximum_energy_residual": max(row["relative_energy_residual"] for row in energy),
        "midpoint_energy_residual": energy[TIME_INDEX_MIDPOINT]["relative_energy_residual"],
        "midpoint_numerator": energy[TIME_INDEX_MIDPOINT]["numerator"],
        "midpoint_denominator": energy[TIME_INDEX_MIDPOINT]["denominator"],
        "energy_by_time": energy,
        "minimum_rank": int(jnp.min(jnp.sum(retained, axis=1))),
        "maximum_condition": float(jnp.max(largest / smallest)),
        "fit_forcing": fit_forcing,
        "audit_forcing": audit_forcing,
        "certificate": certificate,
        "matrix_vs_direct_certificate_max_energy_difference": certificate_energy_difference,
        "timings_seconds": {
            **{f"fit_{key}": value for key, value in fit_timing.items()},
            **{f"audit_{key}": value for key, value in audit_timing.items()},
            "eigensolve": solve_seconds,
            "certificate": certificate_seconds,
        },
    }


def _result_path(section: str, name: str) -> Path:
    return OUTPUT_ROOT / "results" / section / f"{name}.json"


def run_baseline(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol()
    output = _result_path("baseline", "all_geometries")
    if output.exists():
        return _read(output)
    fit, audit = _v3_heldout_bank("fit"), _v3_heldout_bank("audit")
    rows = []
    for geometry in require_protocol()["geometries"]:
        row = _pair_result(
            geometry, "v3_heldout_fit", fit, "v3_heldout_audit", audit, 280
        )
        rows.append(row)
        if progress:
            progress(f"baseline geometry {geometry['id']}")
    selected = [row for row in rows if row["geometry_id"] in require_protocol()["selected_geometry_ids"]]
    curves = np.asarray([[point["relative_energy_residual"] for point in row["energy_by_time"]] for row in selected])
    comparisons = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            comparisons.append({
                "left": selected[i]["geometry_id"], "right": selected[j]["geometry_id"],
                "correlation": float(np.corrcoef(curves[i], curves[j])[0, 1]),
                "maximum_absolute_difference": float(np.max(np.abs(curves[i] - curves[j]))),
            })
    result = {"rows": rows, "selected_pairwise_trajectory_comparisons": comparisons}
    _atomic_json(output, result)
    return result


def run_scaling(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress)
    output = _result_path("scaling", "K280")
    if output.exists():
        return _read(output)
    rows = []
    selected = [_geometry(identifier) for identifier in require_protocol()["selected_geometry_ids"]]
    for geometry in selected:
        for N in N_LADDER:
            rows.append(_pair_result(
                geometry, "scaling_fit", _load_bank("scaling_fit", N),
                "scaling_audit", _load_bank("scaling_audit", N), 280,
            ))
            if progress:
                progress(f"scaling {geometry['id']} N={N}")
    representative = _geometry(require_protocol()["representative_geometry_id"])
    for N in N_LADDER[:-1]:
        rows.append(_pair_result(
            representative, "scaling_fit", _load_bank("scaling_fit", N),
            "scaling_audit", _load_bank("scaling_audit", N_MAX), 280,
            physical_certificate=False,
        ))
        rows[-1]["decomposition_axis"] = "fit_N"
        rows.append(_pair_result(
            representative, "scaling_fit", _load_bank("scaling_fit", N_MAX),
            "scaling_audit", _load_bank("scaling_audit", N), 280,
            physical_certificate=False,
        ))
        rows[-1]["decomposition_axis"] = "audit_N"
    result = {"rows": rows}
    _atomic_json(output, result)
    return result


def run_splits(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress)
    output = _result_path("splits", "K280_N65536")
    if output.exists():
        return _read(output)
    geometry = _geometry(require_protocol()["representative_geometry_id"])
    rows = []
    for index in range(SPLIT_COUNT):
        rows.append(_pair_result(
            geometry, f"split_{index}_fit", _load_bank(f"split_{index}_fit"),
            f"split_{index}_audit", _load_bank(f"split_{index}_audit"), 280,
        ))
        rows[-1]["split_index"] = index
        if progress:
            progress(f"repeated split {index}")
    values = np.asarray([row["midpoint_energy_residual"] for row in rows])
    result = {
        "rows": rows,
        "midpoint_summary": {
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values)),
        },
    }
    _atomic_json(output, result)
    return result


def run_decomposition(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress)
    output = _result_path("decomposition", "law_N65536")
    if output.exists():
        return _read(output)
    geometry = _geometry(require_protocol()["representative_geometry_id"])
    fit_bank, audit_bank = _load_bank("scaling_fit", 65536), _load_bank("scaling_audit", 65536)
    fit, _, _, _ = _assemble_system(geometry, "scaling_fit", fit_bank, 280)
    audit, _, _, _ = _assemble_system(geometry, "scaling_audit", audit_bank, 280)
    fit_solve, audit_solve = _solve(fit), _solve(audit)
    combinations = {
        "a_fit_on_fit": _energy_rows(fit_solve.coefficients, fit, fit),
        "a_fit_on_audit": _energy_rows(fit_solve.coefficients, fit, audit),
        "a_audit_on_audit": _energy_rows(audit_solve.coefficients, audit, audit),
        "a_audit_on_fit": _energy_rows(audit_solve.coefficients, audit, fit),
    }
    delta = fit_solve.coefficients - audit_solve.coefficients
    metric = 0.5 * (fit.gram + audit.gram)
    delta_energy = jnp.einsum("ti,tij,tj->t", delta, metric, delta)
    fit_energy = jnp.einsum("ti,tij,tj->t", fit_solve.coefficients, metric, fit_solve.coefficients)
    result = {
        "geometry_id": geometry["id"], "N": 65536,
        "combinations": combinations,
        "coefficient_difference": {
            "euclidean_by_time": np.asarray(jnp.linalg.norm(delta, axis=1)).tolist(),
            "retained_metric_by_time": np.asarray(jnp.sqrt(jnp.maximum(delta_energy, 0.0))).tolist(),
            "relative_metric_by_time": np.asarray(jnp.sqrt(jnp.maximum(delta_energy, 0.0)) / jnp.maximum(jnp.sqrt(jnp.maximum(fit_energy, 0.0)), 1e-12)).tolist(),
        },
    }
    _atomic_json(output, result)
    return result


def run_spectrum(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress)
    output = _result_path("spectrum", "law_N65536_K280")
    if output.exists():
        return _read(output)
    geometry = _geometry(require_protocol()["representative_geometry_id"])
    fit, _, _, _ = _assemble_system(geometry, "scaling_fit", _load_bank("scaling_fit", 65536), 280)
    audit, _, _, _ = _assemble_system(geometry, "scaling_audit", _load_bank("scaling_audit", 65536), 280)
    rows = []
    summary = []
    for time_index in range(13):
        ef, Uf = jnp.linalg.eigh(fit.gram[time_index])
        ea, Ua = jnp.linalg.eigh(audit.gram[time_index])
        pf = Uf.T @ fit.load[time_index]
        pa = Ua.T @ audit.load[time_index]
        retained_f = ef > RANK_TOLERANCE * ef[-1]
        retained_a = ea > RANK_TOLERANCE * ea[-1]
        mode_energy_f = jnp.where(retained_f, pf * pf / jnp.maximum(ef, 1e-300), 0.0)
        mode_energy_a = jnp.where(retained_a, pa * pa / jnp.maximum(ea, 1e-300), 0.0)
        overlap = Uf[:, retained_f].T @ Ua[:, retained_a]
        singular = jnp.linalg.svd(overlap, compute_uv=False)
        summary.append({
            "time_index": time_index, "t": time_index / 12.0,
            "fit_rank": int(jnp.sum(retained_f)), "audit_rank": int(jnp.sum(retained_a)),
            "fit_smallest_retained": float(jnp.min(jnp.where(retained_f, ef, jnp.inf))),
            "audit_smallest_retained": float(jnp.min(jnp.where(retained_a, ea, jnp.inf))),
            "fit_largest": float(ef[-1]), "audit_largest": float(ea[-1]),
            "fit_condition": float(ef[-1] / jnp.min(jnp.where(retained_f, ef, jnp.inf))),
            "audit_condition": float(ea[-1] / jnp.min(jnp.where(retained_a, ea, jnp.inf))),
            "minimum_retained_subspace_cosine": float(jnp.min(singular)),
            "fit_weakest_decile_energy_fraction": float(jnp.sum(mode_energy_f[:28]) / jnp.maximum(jnp.sum(mode_energy_f), 1e-300)),
            "audit_weakest_decile_energy_fraction": float(jnp.sum(mode_energy_a[:28]) / jnp.maximum(jnp.sum(mode_energy_a), 1e-300)),
        })
        for mode in range(280):
            rows.append({
                "time_index": time_index, "t": time_index / 12.0, "mode": mode,
                "fit_eigenvalue": float(ef[mode]), "audit_eigenvalue": float(ea[mode]),
                "fit_retained": bool(retained_f[mode]), "audit_retained": bool(retained_a[mode]),
                "fit_load_projection": float(pf[mode]), "audit_load_projection": float(pa[mode]),
                "fit_coefficient_energy": float(mode_energy_f[mode]),
                "audit_coefficient_energy": float(mode_energy_a[mode]),
            })
    result = {"summary_by_time": summary, "mode_rows": rows}
    _atomic_json(output, result)
    return result


def _prefix_system(system: GalerkinSystem, K: int) -> GalerkinSystem:
    empty = jnp.zeros((0,), dtype=jnp.float64)
    gram = system.gram[:, :K, :K]
    symmetry = jax.vmap(lambda matrix: jnp.linalg.norm(matrix - matrix.T) / jnp.maximum(jnp.linalg.norm(matrix), 1e-30))(gram)
    return GalerkinSystem(
        gram=gram, load=system.load[:, :K], basis_means=system.basis_means[:, :K],
        centered_basis=empty, weights=empty, forcing=empty,
        raw_symmetry_residual=symmetry, forcing_mean=system.forcing_mean,
    )


def run_k_grid(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress); build_dictionary(progress)
    output = _result_path("k_grid", "law")
    if output.exists():
        return _read(output)
    geometry = _geometry(require_protocol()["representative_geometry_id"])
    rows = []
    for N in N_K_GRID:
        fit_bank, audit_bank = _load_bank("scaling_fit", N), _load_bank("scaling_audit", N)
        fit440, fit_state, reconstruction, fit_timing = _assemble_system(geometry, "scaling_fit", fit_bank, 440)
        audit440, audit_state, _, audit_timing = _assemble_system(geometry, "scaling_audit", audit_bank, 440)
        coefficients = []
        metadata = []
        for K in K_LADDER:
            fit, audit = _prefix_system(fit440, K), _prefix_system(audit440, K)
            solve = _solve(fit)
            energy = _energy_rows(solve.coefficients, fit, audit)
            coefficients.append(jnp.pad(solve.coefficients, ((0, 0), (0, 440 - K))))
            eigenvalues = jnp.linalg.eigvalsh(fit.gram)
            retained = eigenvalues > RANK_TOLERANCE * eigenvalues[:, -1, None]
            metadata.append({
                "N": N, "K": K,
                "fit_action": float(jnp.sum(_problem().time_weights * jnp.asarray([r["fit_quadratic"] for r in energy]))),
                "audit_action": float(jnp.sum(_problem().time_weights * jnp.asarray([r["audit_quadratic"] for r in energy]))),
                "maximum_energy_residual": max(r["relative_energy_residual"] for r in energy),
                "midpoint_energy_residual": energy[6]["relative_energy_residual"],
                "midpoint_numerator": energy[6]["numerator"],
                "midpoint_denominator": energy[6]["denominator"],
                "minimum_rank": int(jnp.min(jnp.sum(retained, axis=1))),
                "maximum_condition": float(jnp.max(eigenvalues[:, -1] / jnp.min(jnp.where(retained, eigenvalues, jnp.inf), axis=1))),
                "energy_by_time": energy,
                "fit_assembly_seconds": fit_timing["assembly_seconds"],
                "audit_assembly_seconds": audit_timing["assembly_seconds"],
            })
        adapter = SimpleNamespace(selection_problem=_problem(), ritz_audit_bank=audit_bank)
        started = time.perf_counter()
        certificates = audit_hybrid_solutions(
            _dictionary(440), jnp.stack(coefficients), adapter,
            jnp.asarray(geometry["eta"], dtype=jnp.float64), reconstruction, audit_state,
            GalerkinCertificateThresholds(
                **base.effective_config()["production_galerkin"]["certificate_thresholds"]
            ), chunk_size=CHUNK_SIZE,
        )
        certificate_seconds = time.perf_counter() - started
        for row, certificate in zip(metadata, certificates, strict=True):
            row["certificate"] = certificate
            row["certificate_batch_seconds"] = certificate_seconds
            rows.append(row)
        if progress:
            progress(f"K ladder/grid N={N}")
    result = {"rows": rows}
    _atomic_json(output, result)
    return result


def run_crosscheck(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    require_protocol(); generate_banks(progress)
    output = _result_path("crosscheck", "numpy_saved_K280")
    if output.exists():
        return _read(output)
    geometry = _geometry(require_protocol()["representative_geometry_id"])
    fit, _, _, _ = _assemble_system(geometry, "scaling_fit", _load_bank("scaling_fit", 65536), 280)
    audit, _, _, _ = _assemble_system(geometry, "scaling_audit", _load_bank("scaling_audit", 65536), 280)
    saved = OUTPUT_ROOT / "crosscheck" / "saved_K280_systems.npz"
    _atomic_npz(saved, K_fit=fit.gram, f_fit=fit.load, K_audit=audit.gram, f_audit=audit.load)
    jax_solve = _solve(fit)
    jax_energy = _energy_rows(jax_solve.coefficients, fit, audit)
    rows = []
    for time_index in range(13):
        Kf, ff = np.asarray(fit.gram[time_index]), np.asarray(fit.load[time_index])
        values, vectors = np.linalg.eigh(Kf)
        retained = values > RANK_TOLERANCE * values[-1]
        projected = vectors.T @ ff
        coefficients = -(vectors[:, retained] @ (projected[retained] / values[retained]))
        Ka, fa = np.asarray(audit.gram[time_index]), np.asarray(audit.load[time_index])
        q, linear = coefficients @ Ka @ coefficients, coefficients @ fa
        ratio = abs(q + linear) / max(q + abs(linear), ENERGY_FLOOR)
        rows.append({
            "time_index": time_index,
            "rank_numpy": int(np.sum(retained)),
            "rank_jax": int(np.asarray(jax_solve.numerical_rank[time_index])),
            "maximum_eigenvalue_difference": float(np.max(np.abs(values - np.asarray(jax_solve.eigenvalues[time_index])))),
            "coefficient_difference_norm": float(np.linalg.norm(coefficients - np.asarray(jax_solve.coefficients[time_index]))),
            "numpy_energy_residual": float(ratio),
            "jax_energy_residual": jax_energy[time_index]["relative_energy_residual"],
        })
    result = {"saved_system_sha256": _sha256(saved), "rows": rows}
    _atomic_json(output, result)
    if progress:
        progress("NumPy saved-matrix cross-check complete")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    stream = tempfile.SpooledTemporaryFile(mode="w+", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows); stream.seek(0)
    _atomic_bytes(path, stream.read().encode())
    stream.close()


def record_stage_performance(stage: str, wall_seconds: float) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "wall_seconds": float(wall_seconds),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "native_galerkin_used": False,
    }
    path = OUTPUT_ROOT / "performance" / f"{stage}.json"
    if not path.exists():
        _atomic_json(path, payload)
    return _read(path)


def finalize(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol()
    baseline = run_baseline(progress)
    scaling = run_scaling(progress)
    splits = run_splits(progress)
    decomposition = run_decomposition(progress)
    spectrum = run_spectrum(progress)
    k_grid = run_k_grid(progress)
    crosscheck = run_crosscheck(progress)
    v3_integrity = verify_v3_immutable()
    if not v3_integrity["passed"]:
        raise RuntimeError("V3.4 terminal authority changed during development diagnosis")
    selected_baseline = [
        row for row in baseline["rows"] if row["geometry_id"] in protocol["selected_geometry_ids"]
    ]
    energy_rows = [
        {"geometry_id": row["geometry_id"], **point}
        for row in selected_baseline for point in row["energy_by_time"]
    ]
    _write_csv(
        OUTPUT_ROOT / "energy_residual_by_time.csv", energy_rows,
        ["geometry_id", "time_index", "t", "numerator", "denominator", "relative_energy_residual", "fit_quadratic", "fit_linear", "audit_quadratic", "audit_linear", "coefficient_norm", "fit_load_norm", "audit_load_norm", "numerator_over_audit_action_density", "numerator_over_fit_action_density", "numerator_over_audit_load_norm", "numerator_over_median_denominator"],
    )
    scaling_csv = [{
        key: row.get(key) for key in (
            "geometry_id", "decomposition_axis", "fit_N", "audit_N", "K",
            "maximum_energy_residual", "midpoint_energy_residual", "midpoint_numerator",
            "midpoint_denominator", "fit_action", "audit_action", "minimum_rank", "maximum_condition",
        )
    } | {
        "maximum_weak_residual": None if row["certificate"] is None else row["certificate"]["maximum_weak_residual"]
    } for row in scaling["rows"]]
    _write_csv(OUTPUT_ROOT / "sample_size_scaling.csv", scaling_csv, list(scaling_csv[0]))
    split_csv = [{
        "split_index": row["split_index"], "fit_N": row["fit_N"], "audit_N": row["audit_N"],
        "maximum_energy_residual": row["maximum_energy_residual"],
        "midpoint_energy_residual": row["midpoint_energy_residual"],
        "weak_residual": row["certificate"]["maximum_weak_residual"],
        "fit_action": row["fit_action"], "audit_action": row["audit_action"],
    } for row in splits["rows"]]
    _write_csv(OUTPUT_ROOT / "split_variability.csv", split_csv, list(split_csv[0]))
    spectrum_rows = spectrum["mode_rows"]
    _write_csv(OUTPUT_ROOT / "galerkin_spectrum_midpoint.csv", spectrum_rows, list(spectrum_rows[0]))
    k_rows = [{
        key: row.get(key) for key in (
            "N", "K", "maximum_energy_residual", "midpoint_energy_residual",
            "midpoint_numerator", "midpoint_denominator", "fit_action", "audit_action",
            "minimum_rank", "maximum_condition",
        )
    } | {"maximum_weak_residual": row["certificate"]["maximum_weak_residual"]} for row in k_grid["rows"]]
    _write_csv(OUTPUT_ROOT / "k_ladder.csv", k_rows, list(k_rows[0]))
    payload = {
        "schema_version": 1,
        "development_only": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "v3_4_authority_changed": False,
        "v3_4_integrity": v3_integrity,
        "baseline": baseline,
        "sample_size_scaling": scaling,
        "split_variability": splits,
        "fit_audit_decomposition": decomposition,
        "spectrum": spectrum,
        "k_grid": k_grid,
        "independent_saved_matrix_crosscheck": crosscheck,
    }
    _atomic_json(DIAGNOSTICS_PATH, payload)
    if progress:
        progress("development diagnostics and required CSV files complete")
    return payload


__all__ = [
    "freeze", "require_protocol", "generate_banks", "build_dictionary",
    "run_baseline", "run_scaling", "run_splits", "run_decomposition",
    "run_spectrum", "run_k_grid", "run_crosscheck", "finalize",
    "record_stage_performance", "verify_v3_immutable",
]
