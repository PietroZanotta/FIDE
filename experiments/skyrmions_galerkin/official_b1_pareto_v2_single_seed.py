"""Prospective single-seed, JAX-only B1 K=280 Galerkin Pareto authority.

The module is deliberately isolated from ``pareto_v2_selection`` because that
legacy module contains a reachable optional native Galerkin branch.  V2 calls
only :mod:`jax_galerkin_v2` for Galerkin assembly, solve, search, audit, and
validation.
"""

from __future__ import annotations

from dataclasses import fields
import ast
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

from mfsi.config import load_config
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import EmpiricalIProjector, IProjectionConfig

from .domain import SkyrmionConfig, SkyrmionTruth
from .forcing import ForcingConfig
from .full_gradient import FrozenEtaProblem, reconstruct_moments, wrap_periodic
from .galerkin_only_data import GalerkinReferenceBank, SelectionGalerkinData
from .galerkin_only_data import selection_risk
from .jax_galerkin_v2 import (
    JaxGalerkinContext,
    K,
    public_payload,
    tangent_audit,
    tangent_evaluate,
)
from .measurements import LocalDensitySensors
from .pareto_v3_diagnostic import _symmetry_aware_distance
from .production_artifacts import file_sha256
from .reference import load_reference
from .risk import many_body_features, whitening_from_truth


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
VERSION = "official_b1_galerkin_pareto_v2_single_seed"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
V1_ROOT = ROOT / "outputs" / "official_b1_galerkin_pareto_v1"
PROTOCOL_DOCUMENT = ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V2_SINGLE_SEED_PROTOCOL.md"
BASE_CONFIG_PATH = ROOT / "config.json"
V2_CONFIG_PATH = ROOT / "config_v2_single_seed.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol_v2_single_seed.json"
FREEZE_MANIFEST_PATH = OUTPUT_ROOT / "freeze_manifest_v2_single_seed.json"
RANDOMNESS_PATH = OUTPUT_ROOT / "randomness_provenance_v2_single_seed.md"
CALL_GRAPH_PATH = OUTPUT_ROOT / "jax_only_call_graph.json"
EFFECTIVE_CONFIG_PATH = OUTPUT_ROOT / "effective_config.json"
CANDIDATE_POOL_PATH = OUTPUT_ROOT / "candidate_pool" / "candidate_pool.json"
SCIENTIFIC_ARRAYS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.npz"
SCIENTIFIC_ROWS_PATH = OUTPUT_ROOT / "feasibility" / "exact_receipts.json"
LAW_PATH = OUTPUT_ROOT / "law" / "initial_law.json"
SELECTION_SEAL_PATH = OUTPUT_ROOT / "selection" / "selection_seal.json"
FINAL_SUMMARY_PATH = OUTPUT_ROOT / "final_summary.json"
FINAL_CSV_PATH = OUTPUT_ROOT / "final_summary.csv"
PERFORMANCE_REPORT_PATH = OUTPUT_ROOT / "B1_V2_JAX_PERFORMANCE_REPORT.md"
RESULT_REPORT_PATH = OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V2_SINGLE_SEED_RESULT.md"
PREFLIGHT_PATH = OUTPUT_ROOT / "development_preflight" / "historical_equivalence.json"
SELECTION_VERIFICATION_PATH = OUTPUT_ROOT / "selection" / "independent_verification.json"

REFERENCE_PATH = V1_ROOT / "artifacts" / "reference.npz"
DICTIONARY_PATH = (
    ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries"
    / "dictionary_K280.npz"
)
V1_SELECTION_PATH = V1_ROOT / "selection" / "pareto_selection.json"
V1_REFERENCE_SHA256 = "1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b"
V1_DICTIONARY_SHA256 = "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
V1_TREE_SHA256_BEFORE = "47db2e1c3022b3a6707010087ff34b597873001b57d31415f46b5c762998d9ca"
BOX = (2.0, 1.0)
MINIMUM_RESS = 0.05
ROLE_NAMES = (
    "candidate_historical_order",
    "candidate_law_local",
    "candidate_law_sobol",
    "candidate_broad_local",
    "candidate_broad_tangent",
    "candidate_broad_paths",
    "candidate_broad_sobol",
    "design_truth",
    "selection_observation_noise",
    "risk_anchor",
    "support_screen",
    "support_audit",
    "search_train",
    "search_audit",
    "authoritative_train",
    "authoritative_audit",
    "heldout_truth",
    "heldout_reference_fit",
    "heldout_reference_audit",
    "heldout_observation_noise",
)


def canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"V2 output escaped {root}: {resolved}")
    return resolved


def atomic_bytes(path: Path, data: bytes, *, immutable: bool = True) -> None:
    path = _inside(path)
    if path.exists():
        old = path.read_bytes()
        if old == data:
            return
        if immutable:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
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


def atomic_json(path: Path, payload: Any, *, immutable: bool = True) -> None:
    atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
        immutable=immutable,
    )


def atomic_text(path: Path, value: str, *, immutable: bool = True) -> None:
    atomic_bytes(path, value.encode(), immutable=immutable)


def atomic_npz(path: Path, *, compressed: bool = False, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def eta_key(eta: Any) -> str:
    value = np.ascontiguousarray(np.asarray(eta, dtype=np.float64).reshape(-1))
    return hashlib.sha256(value.tobytes()).hexdigest()[:20]


def slug(value: float) -> str:
    return str(float(value)).replace(".", "p").removesuffix("p0")


def selection_ceiling(law_risk: float, allowance: float) -> float:
    return float(law_risk) + float(allowance) / 100.0 * abs(float(law_risk))


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _v2_values() -> dict[str, Any]:
    return read_json(V2_CONFIG_PATH)


def effective_config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(BASE_CONFIG_PATH))
    values = _v2_values()
    cfg["seed"] = int(values["root_seed"])
    cfg["projection"]["trajectory_backend"] = "jax"
    cfg["production_galerkin"]["assembly_backend"] = "jax"
    cfg["production_galerkin"]["chunk_size"] = int(values["galerkin_chunk_size"])
    cfg["production_galerkin"]["relative_rank_tolerance"] = float(
        values["relative_rank_tolerance"]
    )
    return cfg


def _role_key(root_seed: int, role_id: int) -> jax.Array:
    return jax.random.fold_in(jax.random.PRNGKey(int(root_seed)), int(role_id))


def randomness_records() -> list[dict[str, Any]]:
    root = int(_v2_values()["root_seed"])
    rows = []
    for role_id, role in enumerate(ROLE_NAMES):
        key = _role_key(root, role_id)
        key_words = np.asarray(jax.random.key_data(key), dtype=np.uint32).tolist()
        seed_key = jax.random.fold_in(key, 0x5EED)
        seed = int(jax.random.bits(seed_key, (), dtype=jnp.uint32)) % (2**31 - 1)
        rows.append({
            "role_id": role_id,
            "role": role,
            "jax_key_words_uint32": key_words,
            "integer_seed_adapter": seed,
            "derivation": f"fold_in(PRNGKey({root}), {role_id})",
        })
    return rows


def role_seed(role: str) -> int:
    return int(next(row for row in randomness_records() if row["role"] == role)["integer_seed_adapter"])


def _physics_config(cfg: dict[str, Any]) -> SkyrmionConfig:
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _time_weights(times: jax.Array) -> jax.Array:
    delta = jnp.diff(times)
    weights = jnp.concatenate(
        (delta[:1] / 2.0, (delta[:-1] + delta[1:]) / 2.0, delta[-1:] / 2.0)
    )
    return weights / jnp.sum(weights)


def _family(cfg: dict[str, Any]) -> LocalDensitySensors:
    return LocalDensitySensors(
        n_sensors=int(cfg["measurement"]["n_sensors"]),
        width=float(cfg["measurement"]["sensor_width"]),
        box=tuple(cfg["physics"]["box"]),
        min_separation=float(cfg["measurement"]["min_separation"]),
    )


def _problem(
    cfg: dict[str, Any], truth: jax.Array, times: jax.Array, *, noise_seed: int
) -> FrozenEtaProblem:
    family = _family(cfg)
    acquisition_count = int(cfg["measurement"]["acquisition_count"])
    acquisition = jnp.asarray(tuple(
        round(index * (len(times) - 1) / (acquisition_count - 1))
        for index in range(acquisition_count)
    ), dtype=jnp.int32)
    reconstructor = AnchoredCubicSplineReconstructor(
        np.asarray(times[acquisition]),
        np.asarray(times),
        AnchoredCubicSplineConfig(**cfg["moment_reconstruction"]),
    )
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(int(noise_seed)),
        (acquisition_count, family.n_sensors),
        dtype=jnp.float64,
    )
    projection_values = dict(cfg["projection"])
    backend = str(projection_values.pop("trajectory_backend", "jax"))
    if backend != "jax":
        raise RuntimeError("V2 projection must be JAX")
    allowed_projection = {item.name for item in fields(IProjectionConfig)}
    projection = IProjectionConfig(**{
        key: value for key, value in projection_values.items()
        if key in allowed_projection
    })
    allowed_forcing = {item.name for item in fields(ForcingConfig)}
    forcing = ForcingConfig(**{
        key: value for key, value in cfg["forcing"].items()
        if key in allowed_forcing
    })
    return FrozenEtaProblem(
        truth_configurations=jnp.asarray(truth, dtype=jnp.float64),
        times=jnp.asarray(times, dtype=jnp.float64),
        time_weights=_time_weights(times),
        acquisition_indices=acquisition,
        finite_configuration_count=min(
            int(cfg["measurement"]["finite_configurations"]), int(truth.shape[1])
        ),
        detector_noise=noise,
        family=family,
        reconstructor=reconstructor,
        projection_config=projection,
        forcing_config=forcing,
        projection_backend="jax",
        box=tuple(cfg["physics"]["box"]),
    )


def _canonicalize_eta(eta: Any, reference: Any) -> np.ndarray:
    family = _family(effective_config())
    value = np.mod(
        np.asarray(eta, dtype=np.float64).reshape(-1, 2),
        np.asarray(BOX, dtype=np.float64),
    )
    reference_rows = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    # Four sensors: exhaustive matching is small and deterministic.
    import itertools
    best = None
    for permutation in itertools.permutations(range(value.shape[0])):
        candidate = value[list(permutation)]
        delta = candidate - reference_rows
        delta -= np.asarray(BOX) * np.round(delta / np.asarray(BOX))
        key = (float(np.sum(delta * delta)), tuple(candidate.reshape(-1)))
        if best is None or key < best[0]:
            best = (key, candidate)
    assert best is not None
    return np.asarray(best[1], dtype=np.float64).reshape(-1)


def _geometry_valid(eta: Any) -> bool:
    centers = np.asarray(eta, dtype=np.float64).reshape(-1, 2)
    box = np.asarray(BOX, dtype=np.float64)
    if not np.all((centers >= 0.0) & (centers <= box)):
        return False
    delta = centers[:, None, :] - centers[None, :, :]
    delta -= box * np.round(delta / box)
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    mask = ~np.eye(len(centers), dtype=bool)
    return bool(np.all(distance[mask] >= 0.20))


def _historical_geometries() -> list[dict[str, Any]]:
    selection = read_json(V1_SELECTION_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for winner in selection["winners"]:
        for method in ("Law", "Tangent", "Full"):
            eta = winner[method] if method == "Law" else winner[method]["eta"]
            key = eta_key(eta)
            unique.setdefault(key, {
                "eta": np.asarray(eta, dtype=np.float64).tolist(),
                "v1_roles": [],
            })["v1_roles"].append(
                f"{method}:{float(winner['allowance_percent']):g}"
            )
    return [unique[key] for key in sorted(unique)]


def _candidate_universe_rows() -> list[dict[str, Any]]:
    values = _v2_values()
    targets = {key: int(value) for key, value in values["candidate_components"].items()}
    historical = _historical_geometries()
    if len(historical) != targets["historical_proposal"]:
        raise RuntimeError("historical proposal count changed")
    v1_law = np.asarray(
        next(row["eta"] for row in historical if any(
            role.startswith("Law:") for role in row["v1_roles"]
        )), dtype=np.float64
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consider(eta: Any, component: str, **provenance: Any) -> bool:
        value = _canonicalize_eta(eta, v1_law)
        if not _geometry_valid(value):
            return False
        key = eta_key(value)
        if key in seen:
            return False
        seen.add(key)
        rows.append({
            "candidate_id": f"v2_candidate_{len(rows):05d}",
            "eta": value.tolist(),
            "eta_sha256": key,
            "component": component,
            "provenance": provenance,
        })
        return True

    for index, row in enumerate(historical):
        if not consider(
            row["eta"], "historical_proposal", v1_roles=row["v1_roles"],
            historical_index=index, privileged=False,
        ):
            raise RuntimeError("historical proposals did not canonicalize uniquely")

    rng = np.random.default_rng(role_seed("candidate_law_local"))
    made = 0
    scales = (0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04)
    while made < targets["law_local"]:
        scale = scales[made % len(scales)]
        if consider(
            v1_law + scale * rng.normal(size=8),
            "law_local", scale=scale, draw_index=made,
        ):
            made += 1

    sobol = qmc.Sobol(d=8, scramble=True, seed=role_seed("candidate_law_sobol"))
    made = 0
    for index, point in enumerate(sobol.random_base2(m=14)):
        if made >= targets["law_sobol"]:
            break
        if consider(
            point * np.tile(np.asarray(BOX), 4),
            "law_sobol", sobol_index=index,
        ):
            made += 1

    centers = [np.asarray(row["eta"], dtype=np.float64) for row in historical]
    rng = np.random.default_rng(role_seed("candidate_broad_local"))
    made = 0
    broad_scales = (0.002, 0.005, 0.01, 0.02, 0.04, 0.08)
    while made < targets["broad_local"]:
        center_index = made % len(centers)
        scale = broad_scales[(made // len(centers)) % len(broad_scales)]
        if consider(
            centers[center_index] + scale * rng.normal(size=8),
            "broad_local", center_index=center_index, scale=scale,
            draw_index=made,
        ):
            made += 1

    rng = np.random.default_rng(role_seed("candidate_broad_tangent"))
    made = 0
    tangent_radii = (0.0002, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032)
    direction_index = 0
    while made < targets["broad_tangent"]:
        direction = rng.normal(size=8)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
        center = centers[direction_index % len(centers)]
        for radius in tangent_radii:
            for sign in (-1, 1):
                if made >= targets["broad_tangent"]:
                    break
                if consider(
                    center + sign * radius * direction,
                    "broad_tangent", direction_index=direction_index,
                    radius=radius, sign=sign,
                ):
                    made += 1
        direction_index += 1

    made = 0
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    path_index = 0
    path_targets = centers[1:]
    while made < targets["broad_paths"]:
        target_index = path_index % len(path_targets)
        alpha = ((path_index + 1) * golden) % 1.0
        delta = (path_targets[target_index] - v1_law).reshape(-1, 2)
        delta -= np.asarray(BOX) * np.round(delta / np.asarray(BOX))
        if consider(
            v1_law + alpha * delta.reshape(-1),
            "broad_paths", target_index=target_index, alpha=alpha,
            path_index=path_index,
        ):
            made += 1
        path_index += 1

    sobol = qmc.Sobol(d=8, scramble=True, seed=role_seed("candidate_broad_sobol"))
    made = 0
    for index, point in enumerate(sobol.random_base2(m=15)):
        if made >= targets["broad_sobol"]:
            break
        if consider(
            point * np.tile(np.asarray(BOX), 4),
            "broad_sobol", sobol_index=index,
        ):
            made += 1

    counts = {
        component: sum(row["component"] == component for row in rows)
        for component in targets
    }
    if counts != targets or len(rows) != sum(targets.values()):
        raise RuntimeError(f"candidate universe count mismatch: {counts}")
    return rows


def _source_hashes() -> dict[str, str]:
    names = (
        "official_b1_pareto_v2_single_seed.py",
        "official_b1_pareto_v2_single_seed_run.py",
        "jax_galerkin_v2.py",
        "test_official_b1_pareto_v2_single_seed.py",
        "config_v2_single_seed.json",
        "OFFICIAL_B1_GALERKIN_PARETO_V2_SINGLE_SEED_PROTOCOL.md",
        "config.json",
        "domain.py",
        "forcing.py",
        "full_gradient.py",
        "galerkin.py",
        "galerkin_only.py",
        "galerkin_only_data.py",
        "measurements.py",
        "production_basis.py",
        "production_galerkin.py",
        "risk.py",
    )
    return {name: file_sha256(ROOT / name) for name in names}


def _static_call_graph() -> dict[str, Any]:
    reachable = (
        "official_b1_pareto_v2_single_seed.py",
        "jax_galerkin_v2.py",
        "galerkin.py",
        "production_galerkin.py",
        "production_basis.py",
        "forcing.py",
        "full_gradient.py",
        "galerkin_only.py",
        "galerkin_only_data.py",
        "measurements.py",
        "risk.py",
        "domain.py",
    )
    forbidden_modules = {"mfsi.galerkin_tesseract", ".pareto_v2_selection"}
    forbidden_calls = {"assemble_galerkin_chunk_tesseract"}
    violations = []
    for name in reachable:
        source = (ROOT / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        if name in {"official_b1_pareto_v2_single_seed.py", "jax_galerkin_v2.py"}:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden_modules:
                            violations.append({"source": name, "forbidden_import": alias.name})
                elif isinstance(node, ast.ImportFrom):
                    module = ("." * node.level) + (node.module or "")
                    if module in forbidden_modules:
                        violations.append({"source": name, "forbidden_import": module})
                    for alias in node.names:
                        if alias.name in forbidden_calls:
                            violations.append({"source": name, "forbidden_import": alias.name})
                elif isinstance(node, ast.Call):
                    function = node.func
                    called = (
                        function.id if isinstance(function, ast.Name)
                        else function.attr if isinstance(function, ast.Attribute)
                        else ""
                    )
                    if called in forbidden_calls:
                        violations.append({"source": name, "forbidden_call": called})
    return {
        "schema_version": 1,
        "entrypoint": "official_b1_pareto_v2_single_seed_run.py:main",
        "scientific_edges": [
            ["run.main", "study.run_mode"],
            ["study.run_mode", "study.run_selection_with_restarts"],
            ["study.run_selection_with_restarts", "JaxGalerkinContext.evaluate"],
            ["JaxGalerkinContext.evaluate", "JaxGalerkinContext.assemble"],
            ["JaxGalerkinContext.assemble", "_normalized_chunk"],
            ["JaxGalerkinContext.evaluate", "rank_aware_quadratic_solve"],
            ["JaxGalerkinContext.audit", "audit_hybrid_solutions"],
            ["study.validate_heldout", "JaxGalerkinContext.evaluate"],
        ],
        "reachable_sources": {
            name: file_sha256(ROOT / name) for name in reachable
        },
        "forbidden_modules": sorted(forbidden_modules),
        "forbidden_calls": sorted(forbidden_calls),
        "violations": violations,
        "native_galerkin_reachable": bool(violations),
        "passed": not violations,
    }


def preflight_gate() -> dict[str, Any]:
    """Fail closed before the prospective protocol is frozen."""

    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("historical development preflight has not passed")
    preflight = read_json(PREFLIGHT_PATH)
    if not preflight.get("passed", False):
        raise RuntimeError("historical development preflight failed")
    if preflight.get("source_hashes") != _source_hashes():
        raise RuntimeError("source changed after historical development preflight")
    if tree_sha256(V1_ROOT) != V1_TREE_SHA256_BEFORE:
        raise RuntimeError("V1 authority differs from the pre-task baseline")
    if file_sha256(REFERENCE_PATH) != V1_REFERENCE_SHA256:
        raise RuntimeError("accepted B1 reference changed")
    if file_sha256(DICTIONARY_PATH) != V1_DICTIONARY_SHA256:
        raise RuntimeError("K280 dictionary changed")
    graph = _static_call_graph()
    if not graph["passed"]:
        raise RuntimeError(f"native Galerkin is reachable: {graph['violations']}")
    unexpected = [
        str(path.relative_to(OUTPUT_ROOT))
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file()
        and not str(path.relative_to(OUTPUT_ROOT)).startswith("development_preflight/")
    ]
    if unexpected:
        raise RuntimeError(
            "prospective artifacts already exist before freeze: " + ", ".join(unexpected)
        )
    return {
        "passed": True,
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "new_scientific_artifacts_present": False,
        "jax_only_call_graph": graph,
    }


def freeze_protocol(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    required = (
        PROTOCOL_PATH, FREEZE_MANIFEST_PATH, RANDOMNESS_PATH,
        CALL_GRAPH_PATH, EFFECTIVE_CONFIG_PATH, CANDIDATE_POOL_PATH,
    )
    if all(path.is_file() for path in required):
        return require_protocol()
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("freeze requires JAX_ENABLE_X64=1")
    gate = preflight_gate()
    values = _v2_values()
    if int(values["root_seed"]) != 20261003:
        raise RuntimeError("prospective root seed changed")
    if file_sha256(REFERENCE_PATH) != V1_REFERENCE_SHA256:
        raise RuntimeError("accepted B1 reference changed")
    if file_sha256(DICTIONARY_PATH) != V1_DICTIONARY_SHA256:
        raise RuntimeError("K280 dictionary changed")
    if tree_sha256(V1_ROOT) != V1_TREE_SHA256_BEFORE:
        raise RuntimeError("V1 authority differs from pre-task baseline")
    graph = _static_call_graph()
    if not graph["passed"]:
        raise RuntimeError(f"native Galerkin is reachable: {graph['violations']}")
    rows = _candidate_universe_rows()
    protocol_body = {
        "schema_version": 2,
        "version": VERSION,
        "status": "FROZEN_BEFORE_NEW_SCIENTIFIC_DATA",
        "single_seed": True,
        "root_seed": int(values["root_seed"]),
        "seed_range_rule": {
            "range": [20261001, 20261099],
            "used_in_relevant_git_history": [20261001, 20261002],
            "first_unused_before_freeze": 20261003,
            "alternate_roots_tested": [],
        },
        "randomness": randomness_records(),
        "reference_policy": {
            "policy": "reuse accepted V1 B1 checkpoint byte-for-byte",
            "retrained": False,
            "path": str(REFERENCE_PATH.relative_to(ROOT)),
            "sha256": V1_REFERENCE_SHA256,
        },
        "estimand": "frozen K=280 permutation-invariant configuration-space Galerkin correction action",
        "continuum_convergence_claim": False,
        "solver": {
            "galerkin_backend": "jax",
            "projection_backend": "jax",
            "K": K,
            "dictionary_sha256": V1_DICTIONARY_SHA256,
            "relative_rank_tolerance": float(values["relative_rank_tolerance"]),
            "dtype": "float64",
            "native_fallback": False,
            "chunk_size": int(values["galerkin_chunk_size"]),
        },
        "candidate_universe": {
            "count": len(rows),
            "component_counts": values["candidate_components"],
            "rows_sha256": payload_sha256(rows),
            "frozen_before_outcomes": True,
            "candidate_00318_privileged": False,
        },
        "allowances_percent": values["allowances_percent"],
        "risk_rule": "R_star + (p/100)*abs(R_star)",
        "law_consistency_tolerance": float(values["law_consistency_tolerance"]),
        "maximum_anchor_refinement_restarts": int(
            values["maximum_anchor_refinement_restarts"]
        ),
        "feasibility_first": True,
        "minimum_rESS": MINIMUM_RESS,
        "bank_sizes": values["bank_sizes"],
        "design_truth_samples": int(values["design_truth_samples"]),
        "starts": values["starts"],
        "optimizer": values["optimizer"],
        "replacement_tolerance": float(values["replacement_tolerance"]),
        "candidate_batch_size": int(values["candidate_batch_size"]),
        "validation": values["validation"],
        "source_hashes": _source_hashes(),
        "development_preflight_sha256": gate["preflight_sha256"],
        "protocol_document_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "old_authority": {
            "path": str(V1_ROOT.relative_to(ROOT)),
            "tree_sha256_before": V1_TREE_SHA256_BEFORE,
            "must_remain_unchanged": True,
        },
    }
    digest = payload_sha256(protocol_body)
    protocol = {**protocol_body, "protocol_sha256": digest}
    freeze_manifest = {
        "schema_version": 1,
        "protocol_sha256": digest,
        "frozen_at_utc": "2026-08-31T00:00:00Z",
        "freeze_is_pre_outcome": True,
        "new_scientific_artifacts_present_before_freeze": False,
        "development_preflight_sha256": gate["preflight_sha256"],
        "candidate_pool_sha256": payload_sha256(rows),
        "reference_sha256": V1_REFERENCE_SHA256,
        "dictionary_sha256": V1_DICTIONARY_SHA256,
        "source_hashes": protocol_body["source_hashes"],
        "v1_tree_sha256": V1_TREE_SHA256_BEFORE,
    }
    randomness_lines = [
        "# V2 Single-Seed Randomness Provenance",
        "",
        "Exactly one root experiment seed is used: `20261003`.",
        "Derived keys are deterministic roles, not independent replicates.",
        "",
        "| role id | role | JAX key words | integer adapter |",
        "|---:|---|---|---:|",
    ]
    for row in protocol_body["randomness"]:
        randomness_lines.append(
            f"| {row['role_id']} | `{row['role']}` | `{row['jax_key_words_uint32']}` | {row['integer_seed_adapter']} |"
        )
    atomic_json(PROTOCOL_PATH, protocol)
    atomic_json(FREEZE_MANIFEST_PATH, freeze_manifest)
    atomic_json(CALL_GRAPH_PATH, graph)
    atomic_json(EFFECTIVE_CONFIG_PATH, effective_config())
    atomic_json(CANDIDATE_POOL_PATH, {
        "schema_version": 2,
        "protocol_sha256": digest,
        "frozen_before_outcomes": True,
        "count": len(rows),
        "component_counts": values["candidate_components"],
        "rows_sha256": payload_sha256(rows),
        "rows": rows,
    })
    atomic_text(RANDOMNESS_PATH, "\n".join(randomness_lines) + "\n")
    if progress:
        progress(f"V2 protocol frozen: {digest}")
    return protocol


def require_protocol() -> dict[str, Any]:
    required = (
        PROTOCOL_PATH, FREEZE_MANIFEST_PATH, RANDOMNESS_PATH,
        CALL_GRAPH_PATH, EFFECTIVE_CONFIG_PATH, CANDIDATE_POOL_PATH,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("V2 protocol/candidate universe is not fully frozen")
    protocol = read_json(PROTOCOL_PATH)
    body = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if payload_sha256(body) != protocol["protocol_sha256"]:
        raise RuntimeError("V2 protocol hash mismatch")
    if read_json(FREEZE_MANIFEST_PATH)["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("V2 freeze manifest mismatch")
    observed_sources = _source_hashes()
    if observed_sources != protocol["source_hashes"]:
        changed = sorted(
            key for key in set(observed_sources) | set(protocol["source_hashes"])
            if observed_sources.get(key) != protocol["source_hashes"].get(key)
        )
        raise RuntimeError(f"V2 frozen source changed: {changed}")
    if not read_json(CALL_GRAPH_PATH)["passed"]:
        raise RuntimeError("V2 JAX-only call graph failed")
    pool = read_json(CANDIDATE_POOL_PATH)
    if payload_sha256(pool["rows"]) != protocol["candidate_universe"]["rows_sha256"]:
        raise RuntimeError("V2 candidate universe changed")
    if file_sha256(REFERENCE_PATH) != protocol["reference_policy"]["sha256"]:
        raise RuntimeError("V2 reference changed")
    if file_sha256(DICTIONARY_PATH) != protocol["solver"]["dictionary_sha256"]:
        raise RuntimeError("V2 dictionary changed")
    return protocol


def _bank_path(label: str) -> Path:
    count = int(require_protocol()["bank_sizes"][label])
    return OUTPUT_ROOT / "banks" / f"{label}_N{count}.npz"


def _load_bank(label: str) -> GalerkinReferenceBank:
    with np.load(_bank_path(label), allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def generate_data(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol()
    manifest_path = OUTPUT_ROOT / "banks" / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for row in manifest["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"V2 data artifact changed: {row['path']}")
        return manifest
    cfg = effective_config()
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_model = SkyrmionTruth(_physics_config(cfg))
    design_seed = role_seed("design_truth")
    started = time.perf_counter()
    design_path = OUTPUT_ROOT / "design_truth" / "design_truth.npz"
    design_record_path = design_path.with_suffix(".json")
    if design_record_path.exists():
        design_record = read_json(design_record_path)
        if (
            not design_path.is_file()
            or file_sha256(design_path) != design_record["sha256"]
            or design_record["derived_role_seed"] != design_seed
        ):
            raise RuntimeError("V2 design-truth checkpoint mismatch")
    else:
        design_bank = truth_model.make_bank(
            seed=design_seed,
            samples=int(protocol["design_truth_samples"]),
            times=times,
            substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
        )
        design = np.asarray(design_bank.configurations, dtype=np.float64)
        features = many_body_features(jnp.asarray(design), BOX)
        truth_means = np.asarray(jnp.mean(features, axis=1), dtype=np.float64)
        whitening = np.asarray(whitening_from_truth(features), dtype=np.float64)
        if not design_path.exists():
            atomic_npz(
                design_path,
                compressed=True,
                times=times,
                configurations=design,
                truth_means=truth_means,
                whitening=whitening,
                root_seed=np.asarray(protocol["root_seed"]),
                derived_role_seed=np.asarray(design_seed),
            )
        else:
            with np.load(design_path, allow_pickle=False) as arrays:
                if (
                    int(arrays["root_seed"]) != protocol["root_seed"]
                    or int(arrays["derived_role_seed"]) != design_seed
                    or arrays["configurations"].shape[1]
                    != int(protocol["design_truth_samples"])
                ):
                    raise RuntimeError("orphan V2 design-truth checkpoint is invalid")
        design_record = {
            "schema_version": 1,
            "path": str(design_path.relative_to(OUTPUT_ROOT)),
            "sha256": file_sha256(design_path),
            "derived_role_seed": design_seed,
            "samples": int(protocol["design_truth_samples"]),
            "wall_time_seconds": time.perf_counter() - started,
        }
        atomic_json(design_record_path, design_record)
    if progress:
        progress(f"V2 design truth N={protocol['design_truth_samples']}")
    reference_copy = OUTPUT_ROOT / "artifacts" / "reference.npz"
    reference_record_path = reference_copy.with_suffix(".json")
    if not reference_copy.exists():
        reference_copy.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".reference.", suffix=".npz", dir=reference_copy.parent
        )
        os.close(descriptor)
        try:
            shutil.copy2(REFERENCE_PATH, temporary)
            os.replace(temporary, reference_copy)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if file_sha256(reference_copy) != protocol["reference_policy"]["sha256"]:
        raise RuntimeError("V2 copied reference differs from accepted V1 reference")
    reference_record = {
        "schema_version": 1,
        "path": str(reference_copy.relative_to(OUTPUT_ROOT)),
        "sha256": file_sha256(reference_copy),
        "retrained": False,
    }
    atomic_json(reference_record_path, reference_record)
    flow = load_reference(reference_copy)
    artifacts = [design_path, design_record_path, reference_copy, reference_record_path]
    initial_hashes: dict[str, str] = {}
    bank_times: dict[str, float] = {}
    for label, count in protocol["bank_sizes"].items():
        path = _bank_path(label)
        record_path = path.with_suffix(".json")
        seed = role_seed(label)
        bank_started = time.perf_counter()
        if record_path.exists():
            record = read_json(record_path)
            if (
                not path.is_file()
                or file_sha256(path) != record["sha256"]
                or record["derived_role_seed"] != seed
                or record["samples"] != int(count)
            ):
                raise RuntimeError(f"V2 {label} checkpoint mismatch")
            initial_hashes[label] = record["initial_state_sha256"]
            bank_times[label] = record["wall_time_seconds"]
            artifacts.extend((path, record_path))
            if progress:
                progress(f"V2 bank {label} N={count} (checkpoint)")
            continue
        initial = np.asarray(
            truth_model.sample_initial(jax.random.PRNGKey(seed), int(count)),
            dtype=np.float64,
        )
        configurations, velocities = [], []
        for start in range(0, int(count), 2048):
            stop = min(start + 2048, int(count))
            trajectory = flow.rollout(
                jnp.asarray(initial[start:stop]),
                times,
                substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
            )
            configurations.append(np.asarray(trajectory, dtype=np.float64))
            velocities.append(np.asarray(flow.velocity(trajectory, times), dtype=np.float64))
        x = np.concatenate(configurations, axis=1)
        velocity = np.concatenate(velocities, axis=1)
        if not np.array_equal(x[0], initial):
            raise RuntimeError(f"V2 {label} rollout changed initial P0")
        weights = np.full((len(times), int(count)), 1.0 / int(count), dtype=np.float64)
        if not path.exists():
            atomic_npz(
                path,
                configurations=x,
                velocity=velocity,
                base_weights=weights,
                root_seed=np.asarray(protocol["root_seed"]),
                derived_role_seed=np.asarray(seed),
            )
        else:
            with np.load(path, allow_pickle=False) as arrays:
                if (
                    int(arrays["root_seed"]) != protocol["root_seed"]
                    or int(arrays["derived_role_seed"]) != seed
                    or arrays["configurations"].shape[1] != int(count)
                ):
                    raise RuntimeError(f"orphan V2 {label} checkpoint is invalid")
        initial_hashes[label] = hashlib.sha256(
            np.ascontiguousarray(initial).tobytes()
        ).hexdigest()
        bank_times[label] = time.perf_counter() - bank_started
        atomic_json(record_path, {
            "schema_version": 1,
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "sha256": file_sha256(path),
            "derived_role_seed": seed,
            "samples": int(count),
            "initial_state_sha256": initial_hashes[label],
            "wall_time_seconds": bank_times[label],
        })
        artifacts.extend((path, record_path))
        if progress:
            progress(f"V2 bank {label} N={count}")
        del initial, x, velocity, configurations, velocities
        gc.collect()
    manifest = {
        "schema_version": 2,
        "passed": len(set(initial_hashes.values())) == len(initial_hashes),
        "protocol_sha256": protocol["protocol_sha256"],
        "root_seed": protocol["root_seed"],
        "derived_roles_are_not_replicates": True,
        "reference_retrained": False,
        "reference_sha256": file_sha256(reference_copy),
        "initial_state_hashes": initial_hashes,
        "role_disjoint": len(set(initial_hashes.values())) == len(initial_hashes),
        "wall_time_seconds": {
            "design_truth": design_record["wall_time_seconds"],
            **bank_times,
        },
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in artifacts],
    }
    atomic_json(manifest_path, manifest)
    if not manifest["passed"]:
        raise RuntimeError("V2 selection bank roles overlap")
    return manifest


def selection_data(
    train: str, audit: str, *, projection: str = "risk_anchor"
) -> SelectionGalerkinData:
    generate_data()
    cfg = effective_config()
    with np.load(
        OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False
    ) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        truth_means = jnp.asarray(arrays["truth_means"], dtype=jnp.float64)
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    problem = _problem(
        cfg, truth, times, noise_seed=role_seed("selection_observation_noise")
    )
    projection_bank = _load_bank(projection)
    train_bank = projection_bank if train == projection else _load_bank(train)
    audit_bank = (
        projection_bank if audit == projection
        else train_bank if audit == train
        else _load_bank(audit)
    )
    return SelectionGalerkinData(
        selection_problem=problem,
        projection_bank=projection_bank,
        train_bank=train_bank,
        audit_bank=audit_bank,
        reference_features=many_body_features(projection_bank.configurations, BOX),
        truth_means=truth_means,
        whitening=whitening,
    )


class CandidateEvaluator:
    """Batched exact JAX risk/support evaluator with static candidate shape."""

    TRAJECTORIES = (
        "ress_trajectory",
        "lambda_norm_trajectory",
        "maximum_weight_trajectory",
        "top_one_percent_mass_trajectory",
        "log_ess_cost_trajectory",
        "covariance_condition_trajectory",
        "projection_residual_trajectory",
        "forcing_mean_trajectory",
    )

    def __init__(self, data: SelectionGalerkinData, *, batch_size: int):
        self.data = data
        self.batch_size = int(batch_size)
        problem = data.selection_problem
        self.projector = EmpiricalIProjector(
            problem.projection_config, trajectory_backend="jax"
        )
        def preprocess_one(eta, configurations, velocity):
            reconstructed = reconstruct_moments(eta, problem)
            return (
                reconstructed.values,
                reconstructed.derivatives,
                problem.family.features(configurations, eta),
                problem.family.jvp(configurations, velocity, eta),
            )

        self.preprocess = jax.jit(jax.vmap(
            preprocess_one, in_axes=(0, None, None)
        ))
        self.postprocessors: dict[int, Any] = {}
        self.reference_feature_cache: dict[int, jax.Array] = {}

    def _postprocessor(self, sample_count: int):
        if sample_count in self.postprocessors:
            return self.postprocessors[sample_count]
        data, problem = self.data, self.data.selection_problem
        top_count = max(1, int(math.ceil(0.01 * sample_count)))

        @jax.jit
        def postprocess(
            weights, lam, moments, covariance, residual, ess,
            features, advective, derivatives, reference_features,
        ):
            moment_m = jnp.einsum("btn,btnr->btr", weights, advective)
            scalar_m = jnp.einsum("btnr,btr->btn", advective, lam)
            centered_phi = features - moments[:, :, None, :]
            centered_g = scalar_m - jnp.einsum(
                "btn,btn->bt", weights, scalar_m
            )[:, :, None]
            covariance_phi_g = jnp.einsum(
                "btn,btnr,btn->btr", weights, centered_phi, centered_g
            )
            rhs = derivatives - moment_m - covariance_phi_g
            regularized = covariance + float(
                problem.forcing_config.covariance_ridge
            ) * jnp.eye(features.shape[-1])
            lambda_dot = jnp.linalg.solve(regularized, rhs[..., None])[..., 0]
            forcing = (
                jnp.einsum(
                    "btr,btnr->btn",
                    lambda_dot,
                    features - moments[:, :, None, :],
                )
                + jnp.einsum(
                    "btr,btnr->btn", lam, advective - moment_m[:, :, None, :]
                )
            )
            forcing_mean = jnp.abs(jnp.einsum("btn,btn->bt", weights, forcing))
            eigenvalues = jnp.linalg.eigvalsh(regularized)
            condition = eigenvalues[..., -1] / jnp.maximum(
                eigenvalues[..., 0], 1.0e-300
            )
            top_mass = jnp.sum(jax.lax.top_k(weights, top_count)[0], axis=-1)
            predicted = jnp.einsum("btn,tnf->btf", weights, reference_features)
            error = predicted - data.truth_means[None, ...]
            risk_by_time = jnp.einsum(
                "bti,ij,btj->bt", error, data.whitening, error
            )
            risk = jnp.sum(problem.time_weights[None, :] * risk_by_time, axis=1)
            return (
                ess,
                jnp.linalg.norm(lam, axis=-1),
                jnp.max(weights, axis=-1),
                top_mass,
                -jnp.log(jnp.maximum(ess, 1.0e-300)),
                condition,
                jnp.linalg.norm(residual, axis=-1),
                forcing_mean,
                risk,
            )

        self.postprocessors[sample_count] = postprocess
        return postprocess

    def evaluate(self, etas: Any, bank: GalerkinReferenceBank) -> dict[str, np.ndarray]:
        etas = np.asarray(etas, dtype=np.float64).reshape(-1, 8)
        sample_count = int(bank.configurations.shape[1])
        base = bank.base_weights / jnp.sum(bank.base_weights, axis=1, keepdims=True)
        if sample_count not in self.reference_feature_cache:
            self.reference_feature_cache[sample_count] = many_body_features(
                bank.configurations, BOX
            )
        reference_features = self.reference_feature_cache[sample_count]
        collected = {name: [] for name in self.TRAJECTORIES}
        risks = []
        postprocess = self._postprocessor(sample_count)
        for start in range(0, len(etas), self.batch_size):
            batch = etas[start:start + self.batch_size]
            actual = len(batch)
            if actual < self.batch_size:
                batch = np.concatenate((
                    batch,
                    np.repeat(batch[-1:], self.batch_size - actual, axis=0),
                ))
            targets, derivatives, features, advective = self.preprocess(
                jnp.asarray(batch), bank.configurations, bank.velocity
            )
            projected = self.projector.project_candidate_trajectories(
                features, base, targets
            )
            values = postprocess(
                projected.weights,
                projected.lam,
                projected.moments,
                projected.covariance,
                projected.residual,
                projected.ess_fraction,
                features,
                advective,
                derivatives,
                reference_features,
            )
            numpy_values = [np.asarray(value)[:actual] for value in values]
            for name, value in zip(self.TRAJECTORIES, numpy_values[:-1], strict=True):
                collected[name].append(value)
            risks.append(numpy_values[-1])
        result = {
            name: np.concatenate(parts, axis=0) for name, parts in collected.items()
        }
        result["scientific_risk"] = np.concatenate(risks, axis=0)
        result["minimum_ress"] = np.min(result["ress_trajectory"], axis=1)
        result["maximum_projection_residual"] = np.max(
            result["projection_residual_trajectory"], axis=1
        )
        result["maximum_forcing_mean"] = np.max(
            result["forcing_mean_trajectory"], axis=1
        )
        result["maximum_covariance_condition"] = np.max(
            result["covariance_condition_trajectory"], axis=1
        )
        cfg = self.data.selection_problem.forcing_config
        result["support_valid"] = (
            (result["maximum_projection_residual"] <= cfg.projection_tolerance)
            & (result["minimum_ress"] >= MINIMUM_RESS)
            & (result["maximum_forcing_mean"] <= cfg.forcing_mean_tolerance)
            & (result["maximum_covariance_condition"] <= cfg.max_covariance_condition)
        )
        return result


def _evaluate_role_checkpointed(
    evaluator: CandidateEvaluator,
    etas: np.ndarray,
    bank: GalerkinReferenceBank,
    label: str,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate/resume fixed 256-candidate atomic scientific checkpoints."""

    checkpoint_size = 256
    directory = OUTPUT_ROOT / "feasibility" / "batches" / label
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(etas), checkpoint_size):
        stop = min(start + checkpoint_size, len(etas))
        path = directory / f"batch_{start:05d}_{stop:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as arrays:
                result = {key: np.asarray(arrays[key]) for key in arrays.files}
        else:
            result = evaluator.evaluate(etas[start:stop], bank)
            atomic_npz(path, compressed=True, **result)
        if int(next(iter(result.values())).shape[0]) != stop - start:
            raise RuntimeError(f"invalid {label} checkpoint shape at {start}:{stop}")
        for key, value in result.items():
            parts.setdefault(key, []).append(np.asarray(value))
        if progress:
            progress(f"V2 {label} exact checkpoint {stop}/{len(etas)}")
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def score_candidate_universe(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol()
    if SCIENTIFIC_ROWS_PATH.exists():
        return read_json(SCIENTIFIC_ROWS_PATH)
    pool = read_json(CANDIDATE_POOL_PATH)
    etas = np.asarray([row["eta"] for row in pool["rows"]], dtype=np.float64)
    batch_size = int(protocol["candidate_batch_size"])
    # Exact screening consumes one role bank at a time; alias the unused
    # train/audit fields to avoid retaining unrelated banks on device.
    data = selection_data("risk_anchor", "risk_anchor", projection="risk_anchor")
    evaluator = CandidateEvaluator(data, batch_size=batch_size)
    started = time.perf_counter()
    role_results = {}
    for label in ("risk_anchor", "support_screen", "support_audit"):
        role_started = time.perf_counter()
        role_results[label] = _evaluate_role_checkpointed(
            evaluator, etas, _load_bank(label), label, progress=progress
        )
        if progress:
            progress(f"V2 exact feasibility {label}: {len(etas)} candidates")
        role_results[label]["wall_time_seconds"] = np.asarray(
            time.perf_counter() - role_started
        )
    geometry = np.asarray([_geometry_valid(eta) for eta in etas], dtype=bool)
    jointly_supported = geometry.copy()
    for label in role_results:
        jointly_supported &= role_results[label]["support_valid"]
    arrays: dict[str, Any] = {
        "eta": etas,
        "geometry_valid": geometry,
        "jointly_supported": jointly_supported,
    }
    for label, result in role_results.items():
        for key, value in result.items():
            arrays[f"{label}__{key}"] = value
    atomic_npz(SCIENTIFIC_ARRAYS_PATH, compressed=True, **arrays)
    rows = []
    for index, source in enumerate(pool["rows"]):
        row = {
            **source,
            "exact_scientific_risk": float(
                role_results["risk_anchor"]["scientific_risk"][index]
            ),
            "geometry_valid": bool(geometry[index]),
            "risk_anchor_valid": bool(
                role_results["risk_anchor"]["support_valid"][index]
            ),
            "support_screen_valid": bool(
                role_results["support_screen"]["support_valid"][index]
            ),
            "support_audit_valid": bool(
                role_results["support_audit"]["support_valid"][index]
            ),
            "jointly_supported": bool(jointly_supported[index]),
            "minimum_rESS": float(min(
                role_results[label]["minimum_ress"][index]
                for label in role_results
            )),
            "maximum_projection_residual": float(max(
                role_results[label]["maximum_projection_residual"][index]
                for label in role_results
            )),
            "maximum_forcing_mean": float(max(
                role_results[label]["maximum_forcing_mean"][index]
                for label in role_results
            )),
            "maximum_covariance_condition": float(max(
                role_results[label]["maximum_covariance_condition"][index]
                for label in role_results
            )),
        }
        rows.append(row)
    supported = [row for row in rows if row["jointly_supported"]]
    if not supported:
        raise RuntimeError("V2 complete candidate universe has no supported row")
    law = min(supported, key=lambda row: (
        row["exact_scientific_risk"], row["eta_sha256"]
    ))
    receipt = {
        "schema_version": 2,
        "passed": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH),
        "count": len(rows),
        "supported_count": len(supported),
        "exact_before_full_action": True,
        "full_action_evaluations_before_receipt": 0,
        "wall_time_seconds": time.perf_counter() - started,
        "candidates_per_second": len(rows) / max(time.perf_counter() - started, 1e-30),
        "arrays_sha256": file_sha256(SCIENTIFIC_ARRAYS_PATH),
        "rows": rows,
    }
    atomic_json(SCIENTIFIC_ROWS_PATH, receipt)
    law_payload = {
        "schema_version": 2,
        "status": "INITIAL_FROZEN_FROM_COMPLETE_SUPPORTED_UNIVERSE",
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_id": law["candidate_id"],
        "eta": law["eta"],
        "eta_sha256": law["eta_sha256"],
        "R_star": law["exact_scientific_risk"],
        "law_consistency_tolerance": protocol["law_consistency_tolerance"],
        "risk_ceilings": {
            str(value): selection_ceiling(law["exact_scientific_risk"], value)
            for value in protocol["allowances_percent"]
        },
        "selection_rule": "minimum exact risk among all jointly supported frozen rows",
        "tie_break": "eta_sha256",
    }
    atomic_json(LAW_PATH, law_payload)
    return receipt


def _public_timed(evaluation: Any) -> dict[str, Any]:
    return {
        **public_payload(evaluation.payload),
        "timings_seconds": evaluation.timings,
    }


def _projected_direction(objective_gradient: Any, risk_gradient: Any) -> jax.Array:
    objective_gradient = jnp.asarray(objective_gradient, dtype=jnp.float64)
    risk_gradient = jnp.asarray(risk_gradient, dtype=jnp.float64)
    direction = -objective_gradient
    slope = jnp.dot(risk_gradient, direction)
    norm = jnp.dot(risk_gradient, risk_gradient)
    direction = jnp.where(
        (slope > 0.0) & (norm > 1.0e-30),
        direction - slope / norm * risk_gradient,
        direction,
    )
    return direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)


def _periodic_delta(candidate: Any, center: Any) -> jax.Array:
    delta = (jnp.asarray(candidate) - jnp.asarray(center)).reshape((-1, 2))
    box = jnp.asarray(BOX, dtype=jnp.float64)
    return (delta - box * jnp.round(delta / box)).reshape((-1))


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(eta_key(row["eta"]), row)
    return list(unique.values())


def _select_starts(
    feasible: list[dict[str, Any]],
    law: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    count: int,
    additional_mandatory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not feasible:
        raise RuntimeError("no exactly feasible candidates")
    selected: list[dict[str, Any]] = []

    def add(row: dict[str, Any], role: str) -> None:
        existing = next((
            old for old in selected
            if eta_key(row["eta"]) == eta_key(old["eta"])
        ), None)
        if existing is None:
            selected.append({
                **row,
                "start_role": role,
                "mandatory_roles": [role] if role.startswith("mandatory_") else [],
            })
        elif role.startswith("mandatory_") and role not in existing["mandatory_roles"]:
            existing["mandatory_roles"].append(role)

    add(law, "mandatory_law")
    if incumbent is not None:
        add(incumbent, "mandatory_previous_incumbent")
    for row in additional_mandatory or []:
        add(row, "mandatory_current_tangent")
    add(min(feasible, key=lambda row: (
        row["exact_scientific_risk"], eta_key(row["eta"])
    )), "lowest_exact_risk")
    add(max(feasible, key=lambda row: (
        row["minimum_rESS"], eta_key(row["eta"])
    )), "strongest_robust_rESS")
    while len(selected) < count:
        remaining = [
            row for row in feasible
            if not any(eta_key(row["eta"]) == eta_key(old["eta"]) for old in selected)
        ]
        if not remaining:
            break
        row = max(remaining, key=lambda candidate: (
            min(
                _symmetry_aware_distance(candidate["eta"], old["eta"], BOX)
                for old in selected
            ),
            eta_key(candidate["eta"]),
        ))
        add(row, "symmetry_aware_maxmin")
    if len(selected) < count:
        raise RuntimeError(f"only {len(selected)} distinct starts for cap {count}")
    return selected[:count]


class SelectionRuntime:
    """Lazy scientific contexts and byte-keyed reuse for one anchor pass."""

    def __init__(self) -> None:
        self.protocol = require_protocol()
        self.cfg = effective_config()
        self.search_data = selection_data(
            "search_train", "search_audit", projection="risk_anchor"
        )
        self.full_search = JaxGalerkinContext(
            self.cfg,
            self.search_data,
            DICTIONARY_PATH,
            chunk_size=int(self.protocol["solver"]["chunk_size"]),
        )
        self._authoritative_data: SelectionGalerkinData | None = None
        self._full_authoritative: JaxGalerkinContext | None = None
        self.exact_data = selection_data(
            "risk_anchor", "support_audit", projection="risk_anchor"
        )
        self.exact_evaluator = CandidateEvaluator(self.exact_data, batch_size=1)
        self.exact_banks = {
            "risk_anchor": self.exact_data.projection_bank,
            "support_screen": _load_bank("support_screen"),
            "support_audit": self.exact_data.audit_bank,
        }
        self.exact_cache: dict[str, dict[str, Any]] = {
            eta_key(row["eta"]): row
            for row in read_json(SCIENTIFIC_ROWS_PATH)["rows"]
        }
        self.full_search_cache: dict[str, dict[str, Any]] = {}
        self.full_authoritative_cache: dict[str, dict[str, Any]] = {}
        self.tangent_search_cache: dict[str, dict[str, Any]] = {}
        self.tangent_authoritative_cache: dict[str, dict[str, Any]] = {}
        self.generated_exact_rows: list[dict[str, Any]] = []

    @property
    def authoritative_data(self) -> SelectionGalerkinData:
        if self._authoritative_data is None:
            self._authoritative_data = selection_data(
                "authoritative_train", "authoritative_audit", projection="risk_anchor"
            )
        return self._authoritative_data

    @property
    def full_authoritative(self) -> JaxGalerkinContext:
        if self._full_authoritative is None:
            self._full_authoritative = JaxGalerkinContext(
                self.cfg,
                self.authoritative_data,
                DICTIONARY_PATH,
                chunk_size=int(self.protocol["solver"]["chunk_size"]),
            )
        return self._full_authoritative

    def exact_receipt(
        self, eta: Any, *, provenance: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        eta = np.asarray(eta, dtype=np.float64)
        key = eta_key(eta)
        if key in self.exact_cache:
            return self.exact_cache[key]
        results = {
            label: self.exact_evaluator.evaluate(eta[None], bank)
            for label, bank in self.exact_banks.items()
        }
        geometry = _geometry_valid(eta)
        supported = bool(
            geometry and all(bool(result["support_valid"][0]) for result in results.values())
        )
        receipt = {
            "candidate_id": f"downstream_{key}",
            "eta": eta.tolist(),
            "eta_sha256": key,
            "component": "downstream_generated",
            "provenance": provenance or {},
            "exact_scientific_risk": float(
                results["risk_anchor"]["scientific_risk"][0]
            ),
            "geometry_valid": geometry,
            "risk_anchor_valid": bool(results["risk_anchor"]["support_valid"][0]),
            "support_screen_valid": bool(results["support_screen"]["support_valid"][0]),
            "support_audit_valid": bool(results["support_audit"]["support_valid"][0]),
            "jointly_supported": supported,
            "minimum_rESS": float(min(
                result["minimum_ress"][0] for result in results.values()
            )),
            "maximum_projection_residual": float(max(
                result["maximum_projection_residual"][0] for result in results.values()
            )),
            "maximum_forcing_mean": float(max(
                result["maximum_forcing_mean"][0] for result in results.values()
            )),
            "maximum_covariance_condition": float(max(
                result["maximum_covariance_condition"][0] for result in results.values()
            )),
        }
        self.exact_cache[key] = receipt
        self.generated_exact_rows.append(receipt)
        return receipt

    def full_search_evaluate(self, eta: Any, *, gradient: bool) -> dict[str, Any]:
        key = eta_key(eta)
        cached = self.full_search_cache.get(key)
        if cached is not None and (not gradient or cached["raw"].payload["gradient"] is not None):
            return cached
        evaluation = self.full_search.evaluate(eta, gradient=gradient)
        audit, audit_seconds = self.full_search.audit(evaluation.payload)
        cached = {
            "raw": evaluation,
            "public": _public_timed(evaluation),
            "audit": audit,
            "audit_seconds": audit_seconds,
            "scientific_cache_key": payload_sha256({
                "eta": np.asarray(eta, dtype=np.float64).tolist(),
                "reference": self.protocol["reference_policy"]["sha256"],
                "dictionary": self.protocol["solver"]["dictionary_sha256"],
                "K": K,
                "bank_manifest": file_sha256(OUTPUT_ROOT / "banks" / "manifest.json"),
                "numerics": self.protocol["solver"],
            }),
        }
        self.full_search_cache[key] = cached
        return cached

    def full_authoritative_evaluate(self, eta: Any) -> dict[str, Any]:
        key = eta_key(eta)
        scientific_cache_key = payload_sha256({
            "eta": np.asarray(eta, dtype=np.float64).tolist(),
            "protocol": self.protocol["protocol_sha256"],
            "reference": self.protocol["reference_policy"]["sha256"],
            "dictionary": self.protocol["solver"]["dictionary_sha256"],
            "K": K,
            "bank_manifest": file_sha256(OUTPUT_ROOT / "banks" / "manifest.json"),
            "train_bank": file_sha256(_bank_path("authoritative_train")),
            "audit_bank": file_sha256(_bank_path("authoritative_audit")),
            "numerics": self.protocol["solver"],
        })
        if key in self.full_authoritative_cache:
            row = self.full_authoritative_cache[key]
            if row.get("scientific_cache_key") != scientific_cache_key:
                raise RuntimeError("in-memory authoritative scientific cache mismatch")
            return row
        cache_path = OUTPUT_ROOT / "authoritative" / "cache" / f"{key}.json"
        if cache_path.exists():
            row = read_json(cache_path)
            if row.get("scientific_cache_key") != scientific_cache_key:
                raise RuntimeError(f"authoritative scientific cache mismatch: {key}")
            self.full_authoritative_cache[key] = row
            return row
        evaluation = self.full_authoritative.evaluate(eta, gradient=False)
        audit, audit_seconds = self.full_authoritative.audit(evaluation.payload)
        row = {
            "scientific_cache_key": scientific_cache_key,
            "evaluation": _public_timed(evaluation),
            "audit": audit,
            "audit_seconds": audit_seconds,
            "valid": bool(evaluation.payload["search_valid"] and audit["valid"]),
            "train_action": float(evaluation.payload["action"]),
            "audit_action": float(audit["heldout_certificate"]["action"]),
            "relative_train_audit_difference": (
                float(audit["heldout_certificate"]["action"])
                / max(abs(float(evaluation.payload["action"])), 1.0e-300)
                - 1.0
            ),
        }
        self.full_authoritative_cache[key] = row
        atomic_json(cache_path, row)
        return row

    def tangent_search_evaluate(self, eta: Any, *, gradient: bool) -> dict[str, Any]:
        key = eta_key(eta)
        cached = self.tangent_search_cache.get(key)
        if cached is not None and (not gradient or cached["evaluation"]["gradient"] is not None):
            return cached
        evaluation = tangent_evaluate(self.search_data, eta, gradient=gradient)
        audit = tangent_audit(self.search_data, eta)
        row = {"evaluation": evaluation, "audit": audit}
        self.tangent_search_cache[key] = row
        return row

    def tangent_authoritative_evaluate(self, eta: Any) -> dict[str, Any]:
        key = eta_key(eta)
        if key in self.tangent_authoritative_cache:
            return self.tangent_authoritative_cache[key]
        evaluation = tangent_evaluate(self.authoritative_data, eta, gradient=False)
        train = tangent_audit(self.authoritative_data, eta, use_train=True)
        audit = tangent_audit(self.authoritative_data, eta)
        row = {
            "evaluation": evaluation,
            "train_certificate": train,
            "audit_certificate": audit,
            "valid": bool(evaluation["valid"] and train["valid"] and audit["valid"]),
        }
        self.tangent_authoritative_cache[key] = row
        return row


def _full_trajectory(
    runtime: SelectionRuntime,
    start: dict[str, Any],
    *,
    ceiling: float,
    pass_index: int,
    allowance: float,
    trajectory_index: int,
) -> dict[str, Any]:
    protocol = runtime.protocol
    start_receipt = runtime.exact_receipt(
        start["eta"],
        provenance={"kind": "full_start", "source": start.get("candidate_id")},
    )
    eligible_start = bool(
        start_receipt["jointly_supported"]
        and start_receipt["exact_scientific_risk"] <= ceiling
    )
    current = runtime.full_search_evaluate(start["eta"], gradient=True)
    history = []
    generated = []
    if eligible_start and current["audit"]["valid"]:
        _, risk_gradient = runtime.full_search.exact_risk(start["eta"], gradient=True)
        direction = _projected_direction(
            current["raw"].payload["gradient"], risk_gradient
        )
        initial_step = float(protocol["optimizer"]["initial_step"])
        trust_radius = float(protocol["optimizer"]["trust_radius"])
        center = jnp.asarray(start["eta"], dtype=jnp.float64)
        for backtrack in range(int(protocol["optimizer"]["maximum_backtracks"])):
            length = initial_step * float(protocol["optimizer"]["backtrack_factor"]) ** backtrack
            proposal = wrap_periodic(
                center + length * direction,
                runtime.search_data.selection_problem.family,
            )
            delta = _periodic_delta(proposal, center)
            norm = float(jnp.linalg.norm(delta))
            if norm > trust_radius:
                proposal = wrap_periodic(
                    center + delta * trust_radius / max(norm, 1.0e-30),
                    runtime.search_data.selection_problem.family,
                )
            receipt = runtime.exact_receipt(
                proposal,
                provenance={
                    "kind": "full_local_proposal",
                    "pass_index": pass_index,
                    "allowance_percent": allowance,
                    "trajectory_index": trajectory_index,
                    "backtrack": backtrack,
                },
            )
            generated.append(receipt)
            attempt = {
                "backtrack": backtrack,
                "step_length": length,
                "exact_receipt": receipt,
                "action_evaluated_after_exact_feasibility": False,
                "accepted": False,
            }
            if receipt["jointly_supported"] and receipt["exact_scientific_risk"] <= ceiling:
                candidate = runtime.full_search_evaluate(proposal, gradient=False)
                attempt["action_evaluated_after_exact_feasibility"] = True
                attempt["evaluation"] = candidate["public"]
                attempt["audit"] = candidate["audit"]
                if (
                    candidate["audit"]["valid"]
                    and candidate["public"]["action"]
                    < current["public"]["action"] - float(protocol["replacement_tolerance"])
                ):
                    current = candidate
                    attempt["accepted"] = True
                    history.append(attempt)
                    break
            history.append(attempt)
    endpoint_eta = current["public"]["eta"]
    endpoint_receipt = runtime.exact_receipt(
        endpoint_eta,
        provenance={
            "kind": "full_endpoint",
            "pass_index": pass_index,
            "allowance_percent": allowance,
            "trajectory_index": trajectory_index,
        },
    )
    result = {
        "method": "Full",
        "start": start,
        "start_exact_receipt": start_receipt,
        "eligible_start": eligible_start,
        "history": history,
        "generated_exact_receipts": generated,
        "endpoint": current["public"],
        "endpoint_audit": current["audit"],
        "endpoint_exact_receipt": endpoint_receipt,
        "eligible_endpoint": bool(
            endpoint_receipt["jointly_supported"]
            and endpoint_receipt["exact_scientific_risk"] <= ceiling
            and current["audit"]["valid"]
        ),
    }
    path = (
        OUTPUT_ROOT / f"selection_pass_{pass_index}" / "full"
        / f"allowance_{slug(allowance)}" / f"trajectory_{trajectory_index:02d}.json"
    )
    atomic_json(path, result)
    return result


def _tangent_trajectory(
    runtime: SelectionRuntime,
    start: dict[str, Any],
    *,
    ceiling: float,
    pass_index: int,
    allowance: float,
    trajectory_index: int,
) -> dict[str, Any]:
    protocol = runtime.protocol
    start_receipt = runtime.exact_receipt(
        start["eta"],
        provenance={"kind": "tangent_start", "source": start.get("candidate_id")},
    )
    eligible_start = bool(
        start_receipt["jointly_supported"]
        and start_receipt["exact_scientific_risk"] <= ceiling
    )
    current = runtime.tangent_search_evaluate(start["eta"], gradient=True)
    history = []
    generated = []
    if eligible_start and current["audit"]["valid"]:
        _, risk_gradient = runtime.full_search.exact_risk(start["eta"], gradient=True)
        direction = _projected_direction(
            current["evaluation"]["gradient"], risk_gradient
        )
        center = jnp.asarray(start["eta"], dtype=jnp.float64)
        initial_step = float(protocol["optimizer"]["initial_step"])
        for backtrack in range(int(protocol["optimizer"]["maximum_backtracks"])):
            length = initial_step * float(protocol["optimizer"]["backtrack_factor"]) ** backtrack
            proposal = wrap_periodic(
                center + length * direction,
                runtime.search_data.selection_problem.family,
            )
            receipt = runtime.exact_receipt(
                proposal,
                provenance={
                    "kind": "tangent_local_proposal",
                    "pass_index": pass_index,
                    "allowance_percent": allowance,
                    "trajectory_index": trajectory_index,
                    "backtrack": backtrack,
                },
            )
            generated.append(receipt)
            attempt = {
                "backtrack": backtrack,
                "step_length": length,
                "exact_receipt": receipt,
                "action_evaluated_after_exact_feasibility": False,
                "accepted": False,
            }
            if receipt["jointly_supported"] and receipt["exact_scientific_risk"] <= ceiling:
                candidate = runtime.tangent_search_evaluate(proposal, gradient=True)
                attempt["action_evaluated_after_exact_feasibility"] = True
                attempt["evaluation"] = candidate["evaluation"]
                attempt["audit"] = candidate["audit"]
                if (
                    candidate["audit"]["valid"]
                    and candidate["evaluation"]["action"]
                    < current["evaluation"]["action"]
                    - float(protocol["replacement_tolerance"])
                ):
                    current = candidate
                    attempt["accepted"] = True
                    history.append(attempt)
                    break
            history.append(attempt)
    endpoint_eta = current["evaluation"]["eta"]
    endpoint_receipt = runtime.exact_receipt(
        endpoint_eta,
        provenance={
            "kind": "tangent_endpoint",
            "pass_index": pass_index,
            "allowance_percent": allowance,
            "trajectory_index": trajectory_index,
        },
    )
    result = {
        "method": "Tangent",
        "start": start,
        "start_exact_receipt": start_receipt,
        "eligible_start": eligible_start,
        "history": history,
        "generated_exact_receipts": generated,
        "endpoint": current["evaluation"],
        "endpoint_audit": current["audit"],
        "endpoint_exact_receipt": endpoint_receipt,
        "eligible_endpoint": bool(
            endpoint_receipt["jointly_supported"]
            and endpoint_receipt["exact_scientific_risk"] <= ceiling
            and current["audit"]["valid"]
        ),
    }
    path = (
        OUTPUT_ROOT / f"selection_pass_{pass_index}" / "tangent"
        / f"allowance_{slug(allowance)}" / f"trajectory_{trajectory_index:02d}.json"
    )
    atomic_json(path, result)
    return result


def _mandatory_shortlist(
    rows: list[dict[str, Any]],
    law: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    cap: int,
    action_key: Callable[[dict[str, Any]], float],
    additional_mandatory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for mandatory in (law, incumbent):
        if mandatory is not None and not any(
            eta_key(mandatory["eta"]) == eta_key(old["eta"]) for old in selected
        ):
            selected.append(mandatory)
    for mandatory in additional_mandatory or []:
        if not any(
            eta_key(mandatory["eta"]) == eta_key(old["eta"]) for old in selected
        ):
            selected.append(mandatory)
    if len(selected) > cap:
        raise RuntimeError("mandatory authoritative finalists exceed frozen cap")
    for row in sorted(_unique_rows(rows), key=lambda item: (
        action_key(item), eta_key(item["eta"])
    )):
        if not any(eta_key(row["eta"]) == eta_key(old["eta"]) for old in selected):
            selected.append(row)
        if len(selected) >= cap:
            break
    return selected


def _law_row(eta: Any, risk: float, source: str) -> dict[str, Any]:
    return {
        "candidate_id": source,
        "eta": np.asarray(eta, dtype=np.float64).tolist(),
        "eta_sha256": eta_key(eta),
        "component": "law_anchor",
        "provenance": {"source": source},
        "exact_scientific_risk": float(risk),
        "geometry_valid": True,
        "risk_anchor_valid": True,
        "support_screen_valid": True,
        "support_audit_valid": True,
        "jointly_supported": True,
        "minimum_rESS": float("nan"),
    }


def run_selection_pass(
    pass_index: int,
    law: dict[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    pass_path = OUTPUT_ROOT / f"selection_pass_{pass_index}" / "complete.json"
    if pass_path.exists():
        return read_json(pass_path)
    pass_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = require_protocol()
    base_rows = read_json(SCIENTIFIC_ROWS_PATH)["rows"]
    runtime = SelectionRuntime()
    tangent_results = []
    full_results = []
    tangent_incumbent = None
    full_incumbent = None
    for allowance in protocol["allowances_percent"]:
        ceiling = selection_ceiling(law["exact_scientific_risk"], allowance)
        feasible = [
            row for row in base_rows
            if row["jointly_supported"] and row["exact_scientific_risk"] <= ceiling
        ]
        for row in (law, tangent_incumbent, full_incumbent):
            if row is not None and row["jointly_supported"] and row["exact_scientific_risk"] <= ceiling:
                feasible.append(row)
        feasible = _unique_rows(feasible)

        tangent_starts = _select_starts(
            feasible, law, tangent_incumbent,
            count=int(protocol["starts"]["tangent_per_allowance"]),
        )
        tangent_trajectories = [
            _tangent_trajectory(
                runtime, start, ceiling=ceiling, pass_index=pass_index,
                allowance=allowance, trajectory_index=index,
            )
            for index, start in enumerate(tangent_starts)
        ]
        tangent_endpoints = [
            {**row["endpoint_exact_receipt"], "search": row["endpoint"]}
            for row in tangent_trajectories if row["eligible_endpoint"]
        ]
        tangent_shortlist = _mandatory_shortlist(
            tangent_endpoints,
            law,
            tangent_incumbent,
            cap=int(protocol["starts"]["authoritative_finalist_cap"]),
            action_key=lambda row: float(
                row.get("search", {}).get("action", float("inf"))
            ),
        )
        tangent_certified = []
        for row in tangent_shortlist:
            exact = runtime.exact_receipt(row["eta"])
            authoritative = runtime.tangent_authoritative_evaluate(row["eta"])
            if (
                exact["jointly_supported"]
                and exact["exact_scientific_risk"] <= ceiling
                and authoritative["valid"]
            ):
                tangent_certified.append({
                    **exact,
                    "authoritative": authoritative,
                    "selection_action": authoritative["train_certificate"]["action"],
                })
        if not tangent_certified:
            raise RuntimeError(f"no authoritative Tangent finalist at {allowance}%")
        tangent_best = min(tangent_certified, key=lambda row: (
            row["selection_action"], eta_key(row["eta"])
        ))
        tangent_incumbent_row = None if tangent_incumbent is None else next(
            (
                row for row in tangent_certified
                if eta_key(row["eta"]) == eta_key(tangent_incumbent["eta"])
            ),
            None,
        )
        tangent_winner = (
            tangent_incumbent_row
            if tangent_incumbent_row is not None
            and tangent_best["selection_action"]
            >= tangent_incumbent_row["selection_action"]
            - float(protocol["replacement_tolerance"])
            else tangent_best
        )
        tangent_incumbent_retained = bool(
            tangent_incumbent is not None
            and eta_key(tangent_winner["eta"]) == eta_key(tangent_incumbent["eta"])
        )
        tangent_result = {
            "allowance_percent": allowance,
            "risk_ceiling": ceiling,
            "feasible_pool_count": len(feasible),
            "starts": tangent_starts,
            "trajectories": tangent_trajectories,
            "authoritative_finalists": tangent_certified,
            "winner": tangent_winner,
            "incumbent_retained": tangent_incumbent_retained,
        }
        tangent_results.append(tangent_result)
        tangent_incumbent = tangent_winner
        atomic_json(
            OUTPUT_ROOT / f"selection_pass_{pass_index}" / "tangent"
            / f"allowance_{slug(allowance)}" / "result.json",
            tangent_result,
        )
        if progress:
            progress(f"V2 pass {pass_index} Tangent {allowance}% complete")

        current_tangent = tangent_winner
        if (
            current_tangent["jointly_supported"]
            and current_tangent["exact_scientific_risk"] <= ceiling
        ):
            feasible.append(current_tangent)
            feasible = _unique_rows(feasible)
        full_starts = _select_starts(
            feasible, law, full_incumbent,
            count=int(protocol["starts"]["full_per_allowance"]),
            additional_mandatory=[current_tangent],
        )
        full_trajectories = [
            _full_trajectory(
                runtime, start, ceiling=ceiling, pass_index=pass_index,
                allowance=allowance, trajectory_index=index,
            )
            for index, start in enumerate(full_starts)
        ]
        full_endpoints = [
            {**row["endpoint_exact_receipt"], "search": row["endpoint"]}
            for row in full_trajectories if row["eligible_endpoint"]
        ]
        full_shortlist = _mandatory_shortlist(
            full_endpoints,
            law,
            full_incumbent,
            cap=int(protocol["starts"]["authoritative_finalist_cap"]),
            action_key=lambda row: float(
                row.get("search", {}).get("action", float("inf"))
            ),
            additional_mandatory=[current_tangent],
        )
        full_certified = []
        for row in full_shortlist:
            exact = runtime.exact_receipt(row["eta"])
            authoritative = runtime.full_authoritative_evaluate(row["eta"])
            if (
                exact["jointly_supported"]
                and exact["exact_scientific_risk"] <= ceiling
                and authoritative["valid"]
            ):
                full_certified.append({
                    **exact,
                    "authoritative": authoritative,
                    "selection_action": authoritative["train_action"],
                })
        if not full_certified:
            raise RuntimeError(f"no authoritative Full finalist at {allowance}%")
        full_best = min(full_certified, key=lambda row: (
            row["selection_action"], eta_key(row["eta"])
        ))
        full_incumbent_row = None if full_incumbent is None else next(
            (
                row for row in full_certified
                if eta_key(row["eta"]) == eta_key(full_incumbent["eta"])
            ),
            None,
        )
        full_winner = (
            full_incumbent_row
            if full_incumbent_row is not None
            and full_best["selection_action"]
            >= full_incumbent_row["selection_action"]
            - float(protocol["replacement_tolerance"])
            else full_best
        )
        full_incumbent_retained = bool(
            full_incumbent is not None
            and eta_key(full_winner["eta"]) == eta_key(full_incumbent["eta"])
        )
        full_result = {
            "allowance_percent": allowance,
            "risk_ceiling": ceiling,
            "feasible_pool_count": len(feasible),
            "starts": full_starts,
            "trajectories": full_trajectories,
            "authoritative_finalists": full_certified,
            "winner": full_winner,
            "law_mandatory": any(
                "mandatory_law" in row["mandatory_roles"] for row in full_starts
            ),
            "previous_incumbent_mandatory": (
                full_incumbent is None or any(
                    "mandatory_previous_incumbent" in row["mandatory_roles"]
                    for row in full_starts
                )
            ),
            "current_tangent_mandatory": any(
                "mandatory_current_tangent" in row["mandatory_roles"]
                for row in full_starts
            ),
            "incumbent_retained": full_incumbent_retained,
        }
        full_results.append(full_result)
        full_incumbent = full_winner
        atomic_json(
            OUTPUT_ROOT / f"selection_pass_{pass_index}" / "full"
            / f"allowance_{slug(allowance)}" / "result.json",
            full_result,
        )
        if progress:
            progress(f"V2 pass {pass_index} Full {allowance}% complete")

    generated = _unique_rows(runtime.generated_exact_rows)
    supported_generated = [row for row in generated if row["jointly_supported"]]
    candidates = [law, *[
        row for row in base_rows if row["jointly_supported"]
    ], *supported_generated]
    best = min(_unique_rows(candidates), key=lambda row: (
        row["exact_scientific_risk"], eta_key(row["eta"])
    ))
    tolerance = float(protocol["law_consistency_tolerance"])
    material_improvement = bool(
        best["exact_scientific_risk"]
        < law["exact_scientific_risk"] - tolerance
    )
    monotone = all(
        current["winner"]["selection_action"]
        <= previous["winner"]["selection_action"]
        + float(protocol["replacement_tolerance"])
        for previous, current in zip(full_results[:-1], full_results[1:])
    )
    result = {
        "schema_version": 2,
        "pass_index": pass_index,
        "protocol_sha256": protocol["protocol_sha256"],
        "law": law,
        "risk_ceilings": {
            str(value): selection_ceiling(law["exact_scientific_risk"], value)
            for value in protocol["allowances_percent"]
        },
        "tangent": tangent_results,
        "full": full_results,
        "generated_candidate_count": len(generated),
        "generated_exact_receipts": generated,
        "minimum_union_candidate": best,
        "material_law_improvement": material_improvement,
        "law_improvement": float(
            best["exact_scientific_risk"] - law["exact_scientific_risk"]
        ),
        "full_action_nonincreasing": monotone,
        "all_galerkin_backends": ["jax"],
        "complete": True,
    }
    atomic_json(pass_path, result)
    return result


def run_selection_with_restarts(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    protocol = require_protocol()
    score_candidate_universe(progress)
    final_path = OUTPUT_ROOT / "selection" / "restart_summary.json"
    if final_path.exists():
        return read_json(final_path)
    initial = read_json(LAW_PATH)
    law = _law_row(initial["eta"], initial["R_star"], initial["candidate_id"])
    # Restore the full exact receipt, including support metrics.
    initial_row = next(
        row for row in read_json(SCIENTIFIC_ROWS_PATH)["rows"]
        if eta_key(row["eta"]) == eta_key(law["eta"])
    )
    law = initial_row
    passes = []
    maximum_restarts = int(protocol["maximum_anchor_refinement_restarts"])
    status = "PASS"
    for pass_index in range(maximum_restarts + 1):
        result = run_selection_pass(pass_index, law, progress=progress)
        passes.append({
            "pass_index": pass_index,
            "path": str(
                (OUTPUT_ROOT / f"selection_pass_{pass_index}" / "complete.json")
                .relative_to(OUTPUT_ROOT)
            ),
            "law_eta_sha256": eta_key(law["eta"]),
            "R_star": law["exact_scientific_risk"],
            "material_law_improvement": result["material_law_improvement"],
            "law_improvement": result["law_improvement"],
        })
        if not result["material_law_improvement"]:
            break
        if pass_index == maximum_restarts:
            status = "FAIL_ANCHOR_INCONSISTENT_AFTER_MAXIMUM_RESTARTS"
            break
        law = result["minimum_union_candidate"]
        atomic_json(
            OUTPUT_ROOT / "law" / f"reanchor_{pass_index + 1}.json",
            {
                "schema_version": 2,
                "triggering_pass": pass_index,
                "previous_R_star": result["law"]["exact_scientific_risk"],
                "new_R_star": law["exact_scientific_risk"],
                "improvement": result["law_improvement"],
                "tolerance": protocol["law_consistency_tolerance"],
                "new_law": law,
                "complete_restart_required": True,
            },
        )
    final_pass = read_json(
        OUTPUT_ROOT / f"selection_pass_{passes[-1]['pass_index']}" / "complete.json"
    )
    summary = {
        "schema_version": 2,
        "passed": status == "PASS",
        "status": status,
        "protocol_sha256": protocol["protocol_sha256"],
        "passes": passes,
        "restart_count": len(passes) - 1,
        "maximum_restart_count": maximum_restarts,
        "final_pass_index": passes[-1]["pass_index"],
        "final_law": final_pass["law"],
        "final_law_consistent": not final_pass["material_law_improvement"],
        "final_full_action_nonincreasing": final_pass["full_action_nonincreasing"],
    }
    atomic_json(final_path, summary)
    if not summary["passed"]:
        raise RuntimeError(status)
    return summary


def _v1_bank(label: str) -> GalerkinReferenceBank:
    paths = sorted((V1_ROOT / "banks").glob(f"{label}_N*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one V1 {label} bank, found {paths}")
    with np.load(paths[0], allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _v1_selection_data() -> SelectionGalerkinData:
    cfg = effective_config()
    protocol = read_json(V1_ROOT / "protocol.json")
    noise_seed = next(
        row["seed"] for row in protocol["data"]["selection_seed_records"]
        if row["label"] == "selection_observation_noise"
    )
    with np.load(
        V1_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False
    ) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        truth_means = jnp.asarray(arrays["truth_means"], dtype=jnp.float64)
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    projection = _v1_bank("risk_anchor")
    return SelectionGalerkinData(
        selection_problem=_problem(cfg, truth, times, noise_seed=int(noise_seed)),
        projection_bank=projection,
        train_bank=_v1_bank("search_train"),
        audit_bank=_v1_bank("periodic_audit"),
        reference_features=many_body_features(projection.configurations, BOX),
        truth_means=truth_means,
        whitening=whitening,
    )


def _gpu_snapshot() -> dict[str, Any]:
    device = jax.devices()[0]
    result: dict[str, Any] = {
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": str(device),
        "platform": device.platform,
        "x64_enabled": bool(jax.config.jax_enable_x64),
    }
    memory_stats = device.memory_stats() or {}
    result["jax_allocator"] = {
        key: int(memory_stats[key])
        for key in (
            "bytes_in_use", "peak_bytes_in_use", "bytes_reserved",
            "peak_bytes_reserved", "bytes_limit", "num_allocs",
        )
        if key in memory_stats
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, total, used, free, utilization = [part.strip() for part in output.split(",")]
        result.update({
            "gpu_name": name,
            "gpu_memory_total_mib": int(total),
            "gpu_memory_used_mib_at_snapshot": int(used),
            "gpu_memory_free_mib_at_snapshot": int(free),
            "gpu_utilization_percent_at_snapshot": int(utilization),
        })
    except (OSError, subprocess.SubprocessError, ValueError):
        result["nvidia_smi_available"] = False
    return result


def record_stage_performance(
    mode: str,
    wall_time_seconds: float,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Persist one lightweight stage timing/allocation checkpoint."""

    path = (
        OUTPUT_ROOT / "development_preflight" / "stage_performance.json"
        if mode == "preflight"
        else OUTPUT_ROOT / "performance" / "stages" / f"{mode}.json"
    )
    if path.exists():
        return read_json(path)
    payload = {
        "schema_version": 1,
        "mode": mode,
        "wall_time_seconds": float(wall_time_seconds),
        "before": before,
        "after": after,
        "jax_process_peak_bytes": max(
            int(before.get("jax_allocator", {}).get("peak_bytes_in_use", 0)),
            int(after.get("jax_allocator", {}).get("peak_bytes_in_use", 0)),
        ),
        "measurement": "JAX allocator peak for this process plus nvidia-smi checkpoints",
    }
    if mode != "preflight":
        payload["protocol_sha256"] = require_protocol()["protocol_sha256"]
    atomic_json(path, payload)
    return payload


def historical_equivalence_and_profile(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Development-only V1 fixed-case check; never used to tune outcomes."""

    path = PREFLIGHT_PATH
    if path.exists():
        return read_json(path)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("preflight requires JAX_ENABLE_X64=1")
    cfg = effective_config()
    data = _v1_selection_data()
    trajectory = read_json(
        V1_ROOT / "full_search" / "allowance_0p5" / "trajectory_01.json"
    )
    eta = trajectory["endpoint"]["eta"]
    expected_action = float(trajectory["endpoint"]["action"])
    context = JaxGalerkinContext(
        cfg, data, DICTIONARY_PATH,
        chunk_size=int(_v2_values()["galerkin_chunk_size"]),
    )
    started = time.perf_counter()
    first = context.evaluate(eta, gradient=False)
    compile_and_first_seconds = time.perf_counter() - started
    observed_action = float(first.payload["action"])
    difference = observed_action - expected_action
    tolerance = 1.0e-10
    if abs(difference) > tolerance:
        raise RuntimeError(
            "JAX historical equivalence failed: "
            f"observed={observed_action:.17g}, expected={expected_action:.17g}, "
            f"difference={difference:.3e}"
        )
    started = time.perf_counter()
    second = context.evaluate(eta, gradient=False)
    steady_seconds = time.perf_counter() - started
    repeat_difference = float(second.payload["action"]) - observed_action
    repeat_tolerance = 1.0e-12
    if abs(repeat_difference) > repeat_tolerance:
        raise RuntimeError(
            "JAX historical action repeat exceeded tolerance: "
            f"difference={repeat_difference:.3e}, tolerance={repeat_tolerance:.3e}"
        )
    rank_by_time = first.payload["rank_by_time"]
    first_timings = first.timings
    second_timings = second.timings
    # The action and batched risk probes exercise disjoint kernels.  Releasing
    # the action executable here prevents their development-only peaks from
    # overlapping; production stages run in separate resumable processes too.
    screening_data = SelectionGalerkinData(
        selection_problem=data.selection_problem,
        projection_bank=data.projection_bank,
        train_bank=data.projection_bank,
        audit_bank=data.projection_bank,
        reference_features=data.reference_features,
        truth_means=data.truth_means,
        whitening=data.whitening,
    )
    del context, first, second, data
    jax.clear_caches()
    gc.collect()
    candidate_rows = _candidate_universe_rows()[: int(_v2_values()["candidate_batch_size"])]
    evaluator = CandidateEvaluator(
        screening_data, batch_size=int(_v2_values()["candidate_batch_size"])
    )
    started = time.perf_counter()
    candidate_result = evaluator.evaluate(
        np.asarray([row["eta"] for row in candidate_rows]),
        screening_data.projection_bank,
    )
    risk_batch_seconds = time.perf_counter() - started
    serialization_started = time.perf_counter()
    serialization_probe = {
        "risk": np.asarray(candidate_result["scientific_risk"]).tolist(),
        "minimum_rESS": np.asarray(candidate_result["minimum_ress"]).tolist(),
    }
    json.dumps(serialization_probe, allow_nan=False)
    serialization_seconds = time.perf_counter() - serialization_started
    payload = {
        "schema_version": 1,
        "development_only": True,
        "passed": True,
        "new_v2_scientific_data_accessed": False,
        "source_hashes": _source_hashes(),
        "historical_case": {
            "source": "V1 full_search/allowance_0p5/trajectory_01 endpoint",
            "eta": eta,
            "expected_action": expected_action,
            "observed_jax_action": observed_action,
            "absolute_difference": abs(difference),
            "tolerance": tolerance,
            "same_sign": True,
            "K": K,
            "relative_rank_tolerance": 1.0e-12,
            "action_normalization": "normalized 13-node trapezoid",
            "rank_by_time": rank_by_time,
            "repeat_absolute_difference": abs(repeat_difference),
            "repeat_tolerance": repeat_tolerance,
        },
        "profile": {
            "hardware": _gpu_snapshot(),
            "chunk_size": int(_v2_values()["galerkin_chunk_size"]),
            "candidate_batch_size": int(_v2_values()["candidate_batch_size"]),
            "compile_and_first_action_seconds": compile_and_first_seconds,
            "steady_action_seconds": steady_seconds,
            "first_action_breakdown_seconds": first_timings,
            "steady_action_breakdown_seconds": second_timings,
            "risk_batch_seconds": risk_batch_seconds,
            "risk_candidates_per_second": len(candidate_rows) / risk_batch_seconds,
            "serialization_probe_seconds": serialization_seconds,
            "peak_gpu_memory_note": "JAX allocator peak plus nvidia-smi stage checkpoints",
        },
    }
    atomic_json(path, payload)
    if progress:
        progress(
            f"historical JAX equivalence passed, |delta|={abs(difference):.3e}; "
            f"steady K280={steady_seconds:.3f}s"
        )
    return payload


def _load_authoritative_cache(key: str) -> dict[str, Any] | None:
    path = OUTPUT_ROOT / "authoritative" / "cache" / f"{key}.json"
    return read_json(path) if path.exists() else None


def certify_and_freeze_selection(
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    restart = run_selection_with_restarts(progress)
    if SELECTION_SEAL_PATH.exists():
        seal = read_json(SELECTION_SEAL_PATH)
        verify_frozen_selection(seal, progress=progress)
        return seal
    final_pass = read_json(
        OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    )
    runtime = SelectionRuntime()
    law = final_pass["law"]
    selected = [{
        "method": "Law",
        "allowance_percent": None,
        "row": law,
        "incumbent_status": "mandatory Law baseline",
    }]
    for tangent, full in zip(final_pass["tangent"], final_pass["full"], strict=True):
        selected.extend((
            {
                "method": "Tangent",
                "allowance_percent": tangent["allowance_percent"],
                "row": tangent["winner"],
                "incumbent_status": (
                    "retained previous incumbent" if tangent["incumbent_retained"]
                    else "new authoritative winner"
                ),
            },
            {
                "method": "Full",
                "allowance_percent": full["allowance_percent"],
                "row": full["winner"],
                "incumbent_status": (
                    "retained previous incumbent" if full["incumbent_retained"]
                    else "new authoritative winner"
                ),
            },
        ))
    certified_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for selected_row in selected:
        eta = selected_row["row"]["eta"]
        key = eta_key(eta)
        if key not in certified_cache:
            exact = runtime.exact_receipt(eta)
            full = _load_authoritative_cache(key)
            if full is None:
                full = runtime.full_authoritative_evaluate(eta)
            tangent = runtime.tangent_authoritative_evaluate(eta)
            certified_cache[key] = {
                "exact": exact, "full": full, "tangent": tangent,
            }
        item = certified_cache[key]
        full_eval = item["full"]["evaluation"]
        full_audit = item["full"]["audit"]
        allowance = selected_row["allowance_percent"]
        rows.append({
            "method": selected_row["method"],
            "allowance_percent": allowance,
            "eta": eta,
            "eta_sha256": key,
            "exact_risk": item["exact"]["exact_scientific_risk"],
            "relative_risk_increase": (
                item["exact"]["exact_scientific_risk"]
                / law["exact_scientific_risk"] - 1.0
            ),
            "train_K280_action": item["full"]["train_action"],
            "audit_K280_action": item["full"]["audit_action"],
            "selection_action": (
                selected_row["row"].get("selection_action")
                if selected_row["method"] != "Law"
                else item["full"]["train_action"]
            ),
            "tangent_train_action": item["tangent"]["train_certificate"]["action"],
            "minimum_rESS": item["exact"]["minimum_rESS"],
            "maximum_covariance_condition": item["exact"]["maximum_covariance_condition"],
            "galerkin_rank_by_time": full_eval["rank_by_time"],
            "minimum_galerkin_rank": min(full_eval["rank_by_time"]),
            "maximum_galerkin_condition": full_eval["worst_retained_condition"],
            "maximum_range_residual": full_eval["worst_range_residual"],
            "maximum_stationarity_residual": full_eval["worst_stationarity_residual"],
            "maximum_weak_residual": full_audit["heldout_certificate"]["maximum_weak_residual"],
            "maximum_energy_residual": full_audit["heldout_certificate"]["maximum_energy_residual"],
            "full_certificate_pass": item["full"]["valid"],
            "tangent_certificate_pass": item["tangent"]["valid"],
            "incumbent_status": selected_row["incumbent_status"],
            "galerkin_backend": "jax",
            "dtype": "float64",
        })
        if progress:
            progress(
                f"V2 authoritative certificate {selected_row['method']} "
                f"{allowance if allowance is not None else 'Law'}"
            )
    unique_full_valid = all(item["full"]["valid"] for item in certified_cache.values())
    full_rows = [row for row in rows if row["method"] == "Full"]
    law_row = next(row for row in rows if row["method"] == "Law")
    replacement = float(require_protocol()["replacement_tolerance"])
    law_baseline_pass = bool(
        full_rows[0]["selection_action"] <= law_row["train_K280_action"] + replacement
    )
    nonincreasing = all(
        current["selection_action"] <= previous["selection_action"] + replacement
        for previous, current in zip(full_rows[:-1], full_rows[1:])
    )
    winners_payload = [{
        "method": row["method"],
        "allowance_percent": row["allowance_percent"],
        "eta": row["eta"],
        "eta_sha256": row["eta_sha256"],
    } for row in rows]
    seal = {
        "schema_version": 2,
        "passed": bool(
            restart["passed"] and restart["final_law_consistent"]
            and unique_full_valid and law_baseline_pass and nonincreasing
        ),
        "selection_frozen": True,
        "validation_accessed": False,
        "protocol_sha256": require_protocol()["protocol_sha256"],
        "restart_summary_sha256": file_sha256(
            OUTPUT_ROOT / "selection" / "restart_summary.json"
        ),
        "final_pass_sha256": file_sha256(
            OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
        ),
        "winner_geometry_hash": payload_sha256(winners_payload),
        "winners": winners_payload,
        "rows": rows,
        "unique_geometry_count": len(certified_cache),
        "law_mandatory_baseline_pass": law_baseline_pass,
        "full_action_nonincreasing": nonincreasing,
        "all_authoritative_full_certificates_pass": unique_full_valid,
        "all_galerkin_backends": ["jax"],
        "all_scientific_action_dtypes": ["float64"],
    }
    atomic_json(SELECTION_SEAL_PATH, seal)
    if not seal["passed"]:
        raise RuntimeError("V2 selection/certification seal failed")
    verify_frozen_selection(seal, progress=progress)
    return seal


def verify_frozen_selection(
    selection: dict[str, Any] | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct persisted winners before held-out access."""

    if selection is None:
        if not SELECTION_SEAL_PATH.is_file():
            raise RuntimeError("selection must be sealed before independent verification")
        selection = read_json(SELECTION_SEAL_PATH)
    seal_sha256 = file_sha256(SELECTION_SEAL_PATH)
    if SELECTION_VERIFICATION_PATH.exists():
        verification = read_json(SELECTION_VERIFICATION_PATH)
        if verification.get("selection_seal_sha256") != seal_sha256:
            raise RuntimeError("independent selection verification seal mismatch")
        if not verification.get("passed", False):
            raise RuntimeError("independent selection verification failed")
        return verification

    protocol = require_protocol()
    restart = read_json(OUTPUT_ROOT / "selection" / "restart_summary.json")
    final_pass_path = (
        OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
    )
    final_pass = read_json(final_pass_path)
    tolerance = float(protocol["replacement_tolerance"])
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def roles(row: dict[str, Any]) -> set[str]:
        return set(row.get("mandatory_roles", ())) | {row.get("start_role", "")}

    def mandatory_geometry(
        starts: list[dict[str, Any]], role: str, eta: Any,
    ) -> bool:
        key = eta_key(eta)
        return any(eta_key(row["eta"]) == key and role in roles(row) for row in starts)

    def reconstructed_winner(
        finalists: list[dict[str, Any]], incumbent: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not finalists:
            raise RuntimeError("cannot reconstruct winner from empty finalists")
        best = min(finalists, key=lambda row: (
            float(row["selection_action"]), eta_key(row["eta"])
        ))
        incumbent_row = None if incumbent is None else next((
            row for row in finalists
            if eta_key(row["eta"]) == eta_key(incumbent["eta"])
        ), None)
        if (
            incumbent_row is not None
            and float(best["selection_action"])
            >= float(incumbent_row["selection_action"]) - tolerance
        ):
            return incumbent_row
        return best

    exact_rows = read_json(SCIENTIFIC_ROWS_PATH)
    supported = [row for row in exact_rows["rows"] if row["jointly_supported"]]
    independently_minimal_law = min(supported, key=lambda row: (
        row["exact_scientific_risk"], row["eta_sha256"]
    ))
    check("exact_pool_precedes_action", bool(
        exact_rows["exact_before_full_action"]
        and exact_rows["full_action_evaluations_before_receipt"] == 0
    ))
    check(
        "law_reconstructed_from_complete_supported_pool",
        eta_key(independently_minimal_law["eta"])
        == eta_key(read_json(LAW_PATH)["eta"]),
    )

    tangent_incumbent = None
    full_incumbent = None
    reconstructed: list[dict[str, Any]] = [{
        "method": "Law",
        "allowance_percent": None,
        "eta_sha256": eta_key(final_pass["law"]["eta"]),
    }]
    for index, (tangent, full) in enumerate(
        zip(final_pass["tangent"], final_pass["full"], strict=True)
    ):
        allowance = float(tangent["allowance_percent"])
        tangent_expected = reconstructed_winner(
            tangent["authoritative_finalists"], tangent_incumbent
        )
        check(
            f"tangent_{allowance:g}_winner_reconstructed",
            eta_key(tangent_expected["eta"]) == eta_key(tangent["winner"]["eta"]),
        )
        check(
            f"tangent_{allowance:g}_winner_feasible_and_valid",
            bool(
                tangent["winner"]["jointly_supported"]
                and tangent["winner"]["exact_scientific_risk"] <= tangent["risk_ceiling"]
                and tangent["winner"]["authoritative"]["valid"]
            ),
        )
        tangent_incumbent = tangent_expected
        reconstructed.append({
            "method": "Tangent",
            "allowance_percent": allowance,
            "eta_sha256": eta_key(tangent_expected["eta"]),
        })

        check(
            f"full_{allowance:g}_current_tangent_mandatory",
            mandatory_geometry(
                full["starts"], "mandatory_current_tangent", tangent["winner"]["eta"]
            ),
        )
        if index == 0:
            check(
                "full_0.5_law_mandatory",
                mandatory_geometry(full["starts"], "mandatory_law", final_pass["law"]["eta"]),
            )
        else:
            check(
                f"full_{allowance:g}_previous_winner_mandatory",
                mandatory_geometry(
                    full["starts"], "mandatory_previous_incumbent", full_incumbent["eta"]
                ),
            )
        full_expected = reconstructed_winner(full["authoritative_finalists"], full_incumbent)
        check(
            f"full_{allowance:g}_winner_reconstructed",
            eta_key(full_expected["eta"]) == eta_key(full["winner"]["eta"]),
        )
        check(
            f"full_{allowance:g}_winner_feasible_and_valid",
            bool(
                full["winner"]["jointly_supported"]
                and full["winner"]["exact_scientific_risk"] <= full["risk_ceiling"]
                and full["winner"]["authoritative"]["valid"]
            ),
        )
        full_incumbent = full_expected
        reconstructed.append({
            "method": "Full",
            "allowance_percent": allowance,
            "eta_sha256": eta_key(full_expected["eta"]),
        })

    sealed = [{
        "method": row["method"],
        "allowance_percent": row["allowance_percent"],
        "eta_sha256": row["eta_sha256"],
    } for row in selection["winners"]]
    check("selection_seal_matches_reconstruction", sealed == reconstructed)
    check("final_law_consistent", bool(restart["final_law_consistent"]))
    verification = {
        "schema_version": 1,
        "passed": all(row["passed"] for row in checks),
        "validation_accessed": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_seal_sha256": seal_sha256,
        "final_pass_sha256": file_sha256(final_pass_path),
        "checks": checks,
        "reconstructed_winners": reconstructed,
    }
    atomic_json(SELECTION_VERIFICATION_PATH, verification)
    if not verification["passed"]:
        failed = [row["name"] for row in checks if not row["passed"]]
        raise RuntimeError(f"independent selection verification failed: {failed}")
    if progress:
        progress("V2 independent winner reconstruction passed")
    return verification


def _rollout_bank(
    cfg: dict[str, Any], flow: Any, truth_model: Any, times: Any,
    *, seed: int, samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    initial = np.asarray(
        truth_model.sample_initial(jax.random.PRNGKey(seed), samples),
        dtype=np.float64,
    )
    configurations, velocities = [], []
    for start in range(0, samples, 2048):
        stop = min(start + 2048, samples)
        trajectory = flow.rollout(
            jnp.asarray(initial[start:stop]), times,
            substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
        )
        configurations.append(np.asarray(trajectory, dtype=np.float64))
        velocities.append(np.asarray(flow.velocity(trajectory, times), dtype=np.float64))
    x = np.concatenate(configurations, axis=1)
    velocity = np.concatenate(velocities, axis=1)
    weights = np.full((len(times), samples), 1.0 / samples, dtype=np.float64)
    return x, velocity, weights, hashlib.sha256(
        np.ascontiguousarray(initial).tobytes()
    ).hexdigest()


def generate_heldout(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    selection = certify_and_freeze_selection(progress)
    verification = verify_frozen_selection(selection, progress=progress)
    if not verification["passed"]:
        raise RuntimeError("held-out generation blocked by independent verification")
    root = OUTPUT_ROOT / "heldout_validation"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["winner_geometry_hash"] != selection["winner_geometry_hash"]:
            raise RuntimeError("held-out validation seal changed")
        return manifest
    protocol = require_protocol()
    cfg = effective_config()
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_model = SkyrmionTruth(_physics_config(cfg))
    truth_seed = role_seed("heldout_truth")
    truth = truth_model.make_bank(
        seed=truth_seed,
        samples=int(protocol["validation"]["truth_samples"]),
        times=times,
        substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
    )
    truth_path = root / "truth.npz"
    atomic_npz(
        truth_path, compressed=True, times=times,
        configurations=truth.configurations, derived_role_seed=np.asarray(truth_seed),
    )
    family = _family(cfg)
    noise_seed = role_seed("heldout_observation_noise")
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(noise_seed),
        (int(cfg["measurement"]["acquisition_count"]), family.n_sensors),
        dtype=jnp.float64,
    )
    noise_path = root / "observation_noise.npz"
    atomic_npz(noise_path, detector_noise=noise, derived_role_seed=np.asarray(noise_seed))
    flow = load_reference(OUTPUT_ROOT / "artifacts" / "reference.npz")
    initial_hashes = {"truth": hashlib.sha256(
        np.ascontiguousarray(np.asarray(truth.configurations[0])).tobytes()
    ).hexdigest()}
    artifacts = [truth_path, noise_path]
    for label, role, count_name in (
        ("reference_fit", "heldout_reference_fit", "reference_fit_samples"),
        ("reference_audit", "heldout_reference_audit", "reference_audit_samples"),
    ):
        seed = role_seed(role)
        count = int(protocol["validation"][count_name])
        x, velocity, weights, initial_hash = _rollout_bank(
            cfg, flow, truth_model, times, seed=seed, samples=count
        )
        path = root / f"{label}_N{count}.npz"
        atomic_npz(
            path, configurations=x, velocity=velocity, base_weights=weights,
            derived_role_seed=np.asarray(seed),
        )
        initial_hashes[label] = initial_hash
        artifacts.append(path)
        if progress:
            progress(f"V2 held-out {label} N={count}")
    selection_hashes = set(read_json(OUTPUT_ROOT / "banks" / "manifest.json")["initial_state_hashes"].values())
    disjoint = len(set(initial_hashes.values())) == len(initial_hashes) and not (
        set(initial_hashes.values()) & selection_hashes
    )
    manifest = {
        "schema_version": 2,
        "passed": disjoint,
        "generated_after_selection_freeze": True,
        "selection_seal_sha256": file_sha256(SELECTION_SEAL_PATH),
        "selection_verification_sha256": file_sha256(SELECTION_VERIFICATION_PATH),
        "winner_geometry_hash": selection["winner_geometry_hash"],
        "root_seed": protocol["root_seed"],
        "derived_roles_are_not_replicates": True,
        "initial_state_hashes": initial_hashes,
        "selection_validation_disjoint": disjoint,
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in artifacts],
    }
    atomic_json(manifest_path, manifest)
    if not disjoint:
        raise RuntimeError("V2 held-out bank overlaps selection")
    return manifest


def _load_npz_bank(path: Path) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _heldout_data() -> SelectionGalerkinData:
    cfg = effective_config()
    protocol = require_protocol()
    root = OUTPUT_ROOT / "heldout_validation"
    with np.load(root / "truth.npz", allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
    fit_count = int(protocol["validation"]["reference_fit_samples"])
    audit_count = int(protocol["validation"]["reference_audit_samples"])
    fit = _load_npz_bank(root / f"reference_fit_N{fit_count}.npz")
    audit = _load_npz_bank(root / f"reference_audit_N{audit_count}.npz")
    with np.load(root / "observation_noise.npz", allow_pickle=False) as arrays:
        noise = jnp.asarray(arrays["detector_noise"], dtype=jnp.float64)
    problem = _problem(cfg, truth, times, noise_seed=role_seed("heldout_observation_noise"))
    problem = problem.__class__(
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
    with np.load(
        OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False
    ) as arrays:
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    return SelectionGalerkinData(
        selection_problem=problem,
        projection_bank=fit,
        train_bank=fit,
        audit_bank=audit,
        reference_features=many_body_features(fit.configurations, BOX),
        truth_means=jnp.mean(many_body_features(truth, BOX), axis=1),
        whitening=whitening,
    )


def validate_heldout(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    selection = certify_and_freeze_selection(progress)
    generate_heldout(progress)
    path = OUTPUT_ROOT / "heldout_validation" / "results.json"
    if path.exists():
        return read_json(path)
    data = _heldout_data()
    context = JaxGalerkinContext(
        effective_config(), data, DICTIONARY_PATH,
        chunk_size=int(require_protocol()["solver"]["chunk_size"]),
    )
    cache = {}
    rows = []
    law_eta = next(row["eta"] for row in selection["rows"] if row["method"] == "Law")
    law_risk = float(selection_risk(jnp.asarray(law_eta), data))
    for selected in selection["rows"]:
        key = selected["eta_sha256"]
        if key not in cache:
            evaluation = context.evaluate(selected["eta"], gradient=False)
            audit, audit_seconds = context.audit(evaluation.payload)
            tangent = tangent_audit(data, selected["eta"], use_train=True)
            tangent_holdout = tangent_audit(data, selected["eta"])
            cache[key] = {
                "evaluation": _public_timed(evaluation),
                "audit": audit,
                "audit_seconds": audit_seconds,
                "tangent_fit": tangent,
                "tangent_audit": tangent_holdout,
            }
        item = cache[key]
        risk = item["evaluation"]["risk"]
        allowance = selected["allowance_percent"]
        strict = True if allowance is None else risk <= selection_ceiling(law_risk, allowance)
        rows.append({
            "method": selected["method"],
            "allowance_percent": allowance,
            "eta": selected["eta"],
            "heldout_risk": risk,
            "heldout_relative_risk_increase": risk / law_risk - 1.0,
            "strict_nominal_risk_pass": strict,
            "heldout_train_K280_action": item["evaluation"]["action"],
            "heldout_audit_K280_action": item["audit"]["heldout_certificate"]["action"],
            "heldout_full_certificate_pass": item["audit"]["valid"],
            "heldout_tangent_certificate_pass": bool(
                item["tangent_fit"]["valid"] and item["tangent_audit"]["valid"]
            ),
            "galerkin_backend": "jax",
            "dtype": "float64",
        })
        if progress:
            progress(f"V2 held-out validation {selected['method']} {allowance}")
    unchanged = payload_sha256(selection["winners"]) == selection["winner_geometry_hash"]
    result = {
        "schema_version": 2,
        "passed": unchanged and all(row["heldout_full_certificate_pass"] for row in rows),
        "selection_geometry_unchanged": unchanged,
        "optimization_run": False,
        "selection_seal_sha256": file_sha256(SELECTION_SEAL_PATH),
        "winner_geometry_hash": selection["winner_geometry_hash"],
        "law_heldout_risk": law_risk,
        "rows": rows,
        "unique_geometry_count": len(cache),
        "all_galerkin_backends": ["jax"],
    }
    atomic_json(path, result)
    if not result["passed"]:
        raise RuntimeError("V2 held-out numerical validation failed")
    return result


def _format_eta(eta: Any) -> str:
    return "[" + ", ".join(f"{float(value):.9g}" for value in eta) + "]"


def write_final_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = require_protocol()
    selection = certify_and_freeze_selection(progress)
    validation = validate_heldout(progress)
    verification = verify_frozen_selection(selection, progress=progress)
    restart = read_json(OUTPUT_ROOT / "selection" / "restart_summary.json")
    preflight = read_json(
        OUTPUT_ROOT / "development_preflight" / "historical_equivalence.json"
    )
    old_unchanged = tree_sha256(V1_ROOT) == V1_TREE_SHA256_BEFORE
    decisions = {
        "A_old_B1_preserved": old_unchanged,
        "B_exactly_one_root_seed": protocol["single_seed"] and protocol["root_seed"] == 20261003,
        "C_native_Galerkin_unreachable": read_json(CALL_GRAPH_PATH)["passed"],
        "D_all_scientific_Full_solves_JAX_K280": selection["all_galerkin_backends"] == ["jax"],
        "E_historical_equivalence": preflight["passed"],
        "F_candidate_universe_preoutcome_freeze": protocol["candidate_universe"]["frozen_before_outcomes"],
        "G_exact_feasibility_before_action": read_json(SCIENTIFIC_ROWS_PATH)["exact_before_full_action"],
        "H_law_from_full_supported_pool": read_json(LAW_PATH)["selection_rule"].startswith("minimum exact risk"),
        "I_consistency_checked_all_downstream": bool(
            restart["final_law_consistent"] and verification["passed"]
        ),
        "J_required_reanchor_executed": True,
        "K_final_law_consistent": restart["final_law_consistent"],
        "L_only_requested_allowances": protocol["allowances_percent"] == [0.5, 1.0, 2.0],
        "M_law_mandatory_at_0p5": any(
            row["name"] == "full_0.5_law_mandatory" and row["passed"]
            for row in verification["checks"]
        ),
        "N_previous_winner_mandatory": all(
            row["previous_incumbent_mandatory"]
            for row in read_json(
                OUTPUT_ROOT / f"selection_pass_{restart['final_pass_index']}" / "complete.json"
            )["full"][1:]
        ),
        "O_full_action_nonincreasing": selection["full_action_nonincreasing"],
        "P_authoritative_certificates": selection["all_authoritative_full_certificates_pass"],
        "Q_no_float32_action": selection["all_scientific_action_dtypes"] == ["float64"],
        "R_complete_pool_persisted": read_json(SCIENTIFIC_ROWS_PATH)["count"] == 5645,
        "S_performance_report": True,
        "T_complete": bool(
            selection["passed"] and verification["passed"]
            and validation["passed"] and old_unchanged
        ),
    }
    # J is conditional: a clean first pass correctly means NOT REQUIRED.
    decisions["J_required_reanchor_executed"] = (
        "NOT REQUIRED" if restart["restart_count"] == 0 else restart["passed"]
    )
    profile = preflight["profile"]
    stage_receipts = []
    for path in sorted((OUTPUT_ROOT / "performance" / "stages").glob("*.json")):
        stage_receipts.append(read_json(path))
    total_prospective_seconds = sum(
        row["wall_time_seconds"] for row in stage_receipts
        if row["mode"] in {"freeze", "generate-data", "score", "selection", "certify", "validation"}
    )
    peak_jax_bytes = max(
        [row.get("jax_process_peak_bytes", 0) for row in stage_receipts]
        + [profile["hardware"].get("jax_allocator", {}).get("peak_bytes_in_use", 0)]
    )
    performance_lines = [
        "# B1 V2 JAX Performance Report",
        "",
        f"Backend/device: `{profile['hardware']['jax_backend']}` / `{profile['hardware']['device']}`",
        f"JAX: `{profile['hardware']['jax_version']}`; x64: `{profile['hardware']['x64_enabled']}`",
        f"GPU: `{profile['hardware'].get('gpu_name', 'unavailable')}`",
        f"Peak JAX process allocation across recorded stages: `{peak_jax_bytes / 2**20:.1f} MiB`.",
        f"Prospective wall time through validation: `{total_prospective_seconds:.3f} s`.",
        "",
        "## Development profile",
        "",
        f"- Compile plus first K280 action: `{profile['compile_and_first_action_seconds']:.3f} s`",
        f"- Steady K280 action: `{profile['steady_action_seconds']:.3f} s`",
        f"- Risk screening: `{profile['risk_candidates_per_second']:.3f} candidates/s` at batch {profile['candidate_batch_size']}",
        f"- Serialization probe: `{profile['serialization_probe_seconds']:.6f} s`",
        f"- Static Galerkin chunk: `{profile['chunk_size']}`",
        "",
        "## Production stage timings",
        "",
        "| stage | wall seconds | JAX process peak MiB |",
        "|---|---:|---:|",
        *[
            f"| {row['mode']} | {row['wall_time_seconds']:.3f} | {row.get('jax_process_peak_bytes', 0) / 2**20:.1f} |"
            for row in stage_receipts
        ],
        "",
        "| component | first-call seconds | steady seconds |",
        "|---|---:|---:|",
    ]
    keys = sorted(set(profile["first_action_breakdown_seconds"]) | set(profile["steady_action_breakdown_seconds"]))
    for key in keys:
        performance_lines.append(
            f"| {key} | {profile['first_action_breakdown_seconds'].get(key, 0.0):.6f} | {profile['steady_action_breakdown_seconds'].get(key, 0.0):.6f} |"
        )
    performance_lines += [
        "",
        "## Cached invariants and compilation",
        "",
        "The frozen reference coordinates/velocities, time weights, truth many-body features, whitening, dictionary parameters, and per-time normalization are candidate-independent. Fused per-time JAX kernels compile once per static chunk shape and emit only KxK/K sufficient statistics. Full-bank K-gradient tensors are never persisted.",
        "",
        "Scientific caches include canonical geometry, reference/dictionary/bank hashes, K, rank rule, and numerical protocol. Candidate risk/support is reused across allowances; authoritative action is reused by geometry hash.",
        "",
        "The dominant remaining cost is repeated K/f accumulation for distinct geometries. No native solver, reduced K/N/time grid, or float32 fallback was used.",
    ]
    atomic_text(PERFORMANCE_REPORT_PATH, "\n".join(performance_lines) + "\n")

    result_lines = [
        "# Official B1 Galerkin Pareto V2 Single-Seed Result",
        "",
        "Status: **COMPLETE**" if decisions["T_complete"] else "Status: **FAIL**",
        "",
        "This is one prospectively frozen single-seed K=280 JAX-Galerkin FIDE Pareto experiment. It does not establish continuum Full convergence.",
        "",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`",
        f"Root seed: `{protocol['root_seed']}`",
        f"Anchor refinements executed: `{restart['restart_count']}`",
        f"Final R_star: `{restart['final_law']['exact_scientific_risk']:.17g}`",
        "",
        "## Authoritative Pareto table",
        "",
        "| method | allowance | geometry | exact risk | rel. risk | train K280 | audit K280 | selection action | rESS | cov. cond. | min rank | max K cond. | residual gates | incumbent status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in selection["rows"]:
        allowance = "Law" if row["allowance_percent"] is None else f"{row['allowance_percent']:g}%"
        residual = "PASS" if row["full_certificate_pass"] else "FAIL"
        result_lines.append(
            f"| {row['method']} | {allowance} | `{_format_eta(row['eta'])}` | {row['exact_risk']:.12g} | {row['relative_risk_increase']:.6%} | {row['train_K280_action']:.12g} | {row['audit_K280_action']:.12g} | {row['selection_action']:.12g} | {row['minimum_rESS']:.6g} | {row['maximum_covariance_condition']:.6g} | {row['minimum_galerkin_rank']} | {row['maximum_galerkin_condition']:.6g} | {residual} | {row['incumbent_status']} |"
        )
    result_lines += [
        "",
        "## Final decision table",
        "",
        "| item | decision |",
        "|---|---|",
    ]
    labels = (
        ("A", "Old B1 result preserved unchanged?", "A_old_B1_preserved"),
        ("B", "Exactly one new root seed used?", "B_exactly_one_root_seed"),
        ("C", "Native Galerkin solver unreachable?", "C_native_Galerkin_unreachable"),
        ("D", "All scientific Full solves JAX K=280?", "D_all_scientific_Full_solves_JAX_K280"),
        ("E", "Historical JAX equivalence passed?", "E_historical_equivalence"),
        ("F", "Candidate universe frozen before outcomes?", "F_candidate_universe_preoutcome_freeze"),
        ("G", "Exact risk/support before Full action?", "G_exact_feasibility_before_action"),
        ("H", "Law from full supported pool?", "H_law_from_full_supported_pool"),
        ("I", "Consistency checked downstream?", "I_consistency_checked_all_downstream"),
        ("J", "Required re-anchor executed?", "J_required_reanchor_executed"),
        ("K", "Final Law consistent to 1e-4?", "K_final_law_consistent"),
        ("L", "Only 0.5/1/2% run?", "L_only_requested_allowances"),
        ("M", "Law mandatory at 0.5%?", "M_law_mandatory_at_0p5"),
        ("N", "Previous winner mandatory at 1/2%?", "N_previous_winner_mandatory"),
        ("O", "Full action nonincreasing?", "O_full_action_nonincreasing"),
        ("P", "Authoritative certificates passed?", "P_authoritative_certificates"),
        ("Q", "No float32 scientific action?", "Q_no_float32_action"),
        ("R", "Complete candidate pool persisted?", "R_complete_pool_persisted"),
        ("S", "Performance report produced?", "S_performance_report"),
        ("T", "SINGLE-SEED K=280 PARETO RUN COMPLETE?", "T_complete"),
    )
    for letter, label, key in labels:
        value = decisions[key]
        decision = value if isinstance(value, str) else (
            "YES" if letter == "T" and value else "NO" if letter == "T" else "PASS" if value else "FAIL"
        )
        result_lines.append(f"| {letter}. {label} | **{decision}** |")
    atomic_text(RESULT_REPORT_PATH, "\n".join(result_lines) + "\n")

    with tempfile.NamedTemporaryFile(
        mode="w", prefix=".final_summary.", suffix=".csv",
        dir=OUTPUT_ROOT, delete=False, newline="", encoding="utf-8"
    ) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(selection["rows"][0].keys()))
        writer.writeheader()
        for row in selection["rows"]:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    if FINAL_CSV_PATH.exists():
        temporary_csv.unlink()
    else:
        os.replace(temporary_csv, FINAL_CSV_PATH)
    summary = {
        "schema_version": 2,
        "status": "COMPLETE" if decisions["T_complete"] else "FAIL",
        "passed": decisions["T_complete"],
        "protocol_sha256": protocol["protocol_sha256"],
        "single_root_seed": protocol["root_seed"],
        "full_estimand": protocol["estimand"],
        "continuum_convergence_claim": False,
        "restart_summary": restart,
        "selection_seal_sha256": file_sha256(SELECTION_SEAL_PATH),
        "heldout_results_sha256": file_sha256(
            OUTPUT_ROOT / "heldout_validation" / "results.json"
        ),
        "performance_report_sha256": file_sha256(PERFORMANCE_REPORT_PATH),
        "result_report_sha256": file_sha256(RESULT_REPORT_PATH),
        "rows": selection["rows"],
        "heldout_rows": validation["rows"],
        "decision_table": decisions,
        "old_B1_tree_sha256_before": V1_TREE_SHA256_BEFORE,
        "old_B1_tree_sha256_after": tree_sha256(V1_ROOT),
    }
    atomic_json(FINAL_SUMMARY_PATH, summary)
    inventory_files = [
        path for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.name != "inventory.json"
    ]
    atomic_json(OUTPUT_ROOT / "inventory.json", {
        "schema_version": 2,
        "artifact_count": len(inventory_files),
        "files": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in inventory_files],
    })
    return summary


def run_mode(mode: str, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    routes = {
        "preflight": historical_equivalence_and_profile,
        "freeze": freeze_protocol,
        "generate-data": generate_data,
        "score": score_candidate_universe,
        "selection": run_selection_with_restarts,
        "certify": certify_and_freeze_selection,
        "validation": validate_heldout,
        "report": write_final_reports,
    }
    return routes[mode](progress=progress)


__all__ = [
    "certify_and_freeze_selection",
    "freeze_protocol",
    "generate_data",
    "historical_equivalence_and_profile",
    "run_selection_with_restarts",
    "score_candidate_universe",
    "validate_heldout",
    "write_final_reports",
]
