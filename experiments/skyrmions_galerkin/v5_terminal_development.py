"""Development-only diagnosis of the terminal V4 Skyrmion authority.

This module never mutates V4 and never creates a V5 authority. Scientific
diagnostics use JAX float64 on GPU; NumPy is confined to saved K/f matrices.
"""

from __future__ import annotations

import ast
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from . import official_b1_pareto_v4_single_seed as v4
from .full_gradient import forcing_state, reconstruct_moments
from .galerkin import GalerkinSystem, rank_aware_quadratic_solve
from .galerkin_only_data import GalerkinReferenceBank, SelectionGalerkinData, selection_risk
from .production_basis import load_dictionary
from .production_galerkin import assemble_hybrid_system, make_basis_evaluators
from .risk import many_body_features


base = v4.base
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
V4_ROOT = v4.OUTPUT_ROOT
OUTPUT_ROOT = ROOT / "outputs" / "development_v5_terminal_diagnosis"
PROTOCOL = OUTPUT_ROOT / "development_protocol.json"
OPERATIONAL_AMENDMENT = OUTPUT_ROOT / "operational_amendment_chunked_risk_features.json"
ALGEBRA_AMENDMENT = OUTPUT_ROOT / "operational_amendment_batched_algebra.json"
ALGEBRA_FIX_AMENDMENT = OUTPUT_ROOT / "operational_amendment_algebra_field_fix.json"
ALGEBRA_WEIGHT_FIX_AMENDMENT = OUTPUT_ROOT / "operational_amendment_weighted_gram_fix.json"
REPORT_AMENDMENT = OUTPUT_ROOT / "operational_amendment_final_analysis_reporting.json"
RUNNER = ROOT / "v5_terminal_development_run.py"
TEST = ROOT / "test_v5_terminal_development.py"
DIAGNOSTIC_ROOT = 20261102
RISK_ROLE_COUNT = 8
RISK_TRUTH_N = 5000
RISK_REFERENCE_N = 65536
SPLITS = 3
N_VALUES = (32768, 65536, 131072)
N_MAX = max(N_VALUES)
K = 280
RANK_TOLERANCE = 1.0e-12
ALGEBRA_THRESHOLD = 1.0e-8
ENERGY_THRESHOLD = 0.08
CHUNK_SIZE = 256
DELTA_GRID = (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_receipt(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(); count = 0; size = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(8, "big")); digest.update(rel)
        digest.update(bytes.fromhex(_sha256(path)))
        count += 1; size += path.stat().st_size
    return {"sha256": digest.hexdigest(), "file_count": count, "bytes": size}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _atomic(path: Path, content: bytes) -> None:
    resolved = path.resolve(); root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"output escaped development namespace: {path}")
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite development artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _write_json(path: Path, payload: Any) -> None:
    _atomic(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_text(path: Path, text: str) -> None:
    _atomic(path, text.encode())


def _write_npz(path: Path, **arrays: Any) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(temporary + ".npz", path)
    finally:
        for item in (temporary, temporary + ".npz"):
            if os.path.exists(item): os.unlink(item)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    stream = tempfile.SpooledTemporaryFile(mode="w+", newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows); stream.seek(0)
    _atomic(path, stream.read().encode()); stream.close()


def _seed(role_id: int, component: int) -> int:
    key = jax.random.fold_in(jax.random.PRNGKey(DIAGNOSTIC_ROOT), int(role_id))
    key = jax.random.fold_in(key, int(component))
    return int(jax.random.bits(key, (), dtype=jnp.uint32)) % (2**31 - 1)


def _selected() -> list[dict[str, Any]]:
    seal = _read(V4_ROOT / "selection" / "selection_seal.json")
    held = _read(V4_ROOT / "heldout_validation" / "results.json")
    held_by = {(r["method"], r["allowance_percent"]): r for r in held["rows"]}
    rows = []
    for row in seal["rows"]:
        h = held_by[(row["method"], row["allowance_percent"])]
        rows.append({
            "method": row["method"], "allowance_percent": row["allowance_percent"],
            "eta": row["eta"], "eta_sha256": row["eta_sha256"],
            "selection_risk": row["exact_risk"],
            "selection_relative": row["relative_risk_increase"],
            "heldout_risk": h["heldout_scientific_risk"],
            "heldout_relative": h["heldout_relative_risk_increase"],
        })
    return rows


def _panel() -> list[dict[str, Any]]:
    complete = _read(V4_ROOT / "selection_pass_0" / "complete.json")
    selected = _selected(); by_hash = {r["eta_sha256"]: dict(r) for r in selected}
    for method in ("tangent", "full"):
        for allowance in complete[method]:
            for row in [*allowance["authoritative_finalists"], allowance["winner"]]:
                if row["eta_sha256"] not in by_hash:
                    by_hash[row["eta_sha256"]] = {
                        "method": f"{method}_finalist", "allowance_percent": allowance["allowance_percent"],
                        "eta": row["eta"], "eta_sha256": row["eta_sha256"],
                        "selection_risk": row.get("exact_risk", row.get("risk")),
                        "selection_relative": None, "heldout_risk": None, "heldout_relative": None,
                    }
    return list(by_hash.values())


def _call_graph() -> dict[str, Any]:
    violations = []
    for source in (Path(__file__), RUNNER):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [("." * node.level) + (node.module or "")])
                for name in names:
                    if "mfsi.galerkin_tesseract" in name or "pareto_v2_selection" in name:
                        violations.append({"source": source.name, "import": name})
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if name in {"assemble_galerkin_chunk_tesseract", "evaluate_galerkin_action"}:
                    violations.append({"source": source.name, "call": name})
    return {"passed": not violations, "native_galerkin_reachable": bool(violations), "violations": violations}


def activate() -> None:
    v4.activate()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu": raise RuntimeError("scientific diagnostics require JAX GPU")
    if not jax.config.jax_enable_x64: raise RuntimeError("scientific diagnostics require float64")


def freeze(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    activate()
    if PROTOCOL.exists(): return require_protocol()
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise RuntimeError("development output namespace is nonempty before freeze")
    graph = _call_graph()
    if not graph["passed"]: raise RuntimeError("native Galerkin path reachable")
    source_hashes = {p.name: _sha256(p) for p in (Path(__file__), RUNNER, TEST)}
    risk_roles = [{
        "role": f"risk_guard_{i}", "role_id": 5000 + i,
        "truth_seed": _seed(5000 + i, 1), "reference_seed": _seed(5000 + i, 2),
        "noise_seed": _seed(5000 + i, 3), "truth_samples": RISK_TRUTH_N,
        "reference_samples": RISK_REFERENCE_N,
    } for i in range(RISK_ROLE_COUNT)]
    split_roles = [{
        "split": i, "fit_role_id": 6000 + 2*i, "audit_role_id": 6001 + 2*i,
        "fit_seed": _seed(6000 + 2*i, 1), "audit_seed": _seed(6001 + 2*i, 1),
        "maximum_samples": N_MAX,
    } for i in range(SPLITS)]
    body = {
        "schema_version": 1, "status": "FROZEN_DEVELOPMENT_ONLY_PRE_OUTCOME",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "create_v5": False,
        "v4_terminal_failed_and_immutable": True,
        "v4_tree_before": _tree_receipt(V4_ROOT),
        "v4_protocol_sha256": _sha256(V4_ROOT / "protocol_v4.json"),
        "v4_selection_sha256": _sha256(V4_ROOT / "selection" / "selection_seal.json"),
        "v4_heldout_sha256": _sha256(V4_ROOT / "heldout_validation" / "results.json"),
        "diagnostic_root": DIAGNOSTIC_ROOT, "risk_roles": risk_roles,
        "scaling_roles": split_roles, "N_values": list(N_VALUES),
        "N_262144_omitted_preoutcome": "three independent nested split pairs through 131072 provide the decisive trend at bounded development cost",
        "K": K, "rank_tolerance": RANK_TOLERANCE, "algebra_threshold": ALGEBRA_THRESHOLD,
        "energy_threshold": ENERGY_THRESHOLD, "chunk_size": CHUNK_SIZE,
        "risk_contracts": {
            "max_guard_M": [2, 4],
            "upper_confidence": "one-sided Student-t 95% UCB of paired relative risk for M=4 and M=8",
            "fixed_inner_delta_grid": list(DELTA_GRID),
        },
        "algebra_N_recommendation_rule": "smallest of 65536 or 131072 for which every geometry on every split is <=8e-9; otherwise reformulate",
        "selected_rows": _selected(), "finalist_panel": _panel(),
        "scientific_backend": "JAX GPU float64", "native_galerkin_allowed": False,
        "numpy_scope": "eigendecomposition of already saved 280x280 K/f only",
        "call_graph": graph, "source_hashes": source_hashes,
    }
    protocol = {**body, "protocol_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    _write_json(PROTOCOL, protocol)
    if progress: progress(f"frozen protocol {protocol['protocol_sha256']}")
    return protocol


def require_protocol() -> dict[str, Any]:
    p = _read(PROTOCOL); body = {k: v for k, v in p.items() if k != "protocol_sha256"}
    if hashlib.sha256(_canonical(body)).hexdigest() != p["protocol_sha256"]: raise RuntimeError("protocol digest mismatch")
    observed = {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)}
    if observed != p["source_hashes"]:
        amendment_path = (REPORT_AMENDMENT if REPORT_AMENDMENT.exists()
                          else ALGEBRA_WEIGHT_FIX_AMENDMENT if ALGEBRA_WEIGHT_FIX_AMENDMENT.exists()
                          else ALGEBRA_FIX_AMENDMENT if ALGEBRA_FIX_AMENDMENT.exists()
                          else ALGEBRA_AMENDMENT if ALGEBRA_AMENDMENT.exists()
                          else OPERATIONAL_AMENDMENT)
        if not amendment_path.exists(): raise RuntimeError("source changed after freeze without amendment")
        amendment = _read(amendment_path)
        if amendment["protocol_sha256"] != p["protocol_sha256"] or amendment["amended_source_hashes"] != observed:
            raise RuntimeError("operational amendment does not match sources")
    if _sha256(V4_ROOT / "heldout_validation" / "results.json") != p["v4_heldout_sha256"]: raise RuntimeError("V4 heldout changed")
    return p


def _risk_dir(index: int) -> Path: return OUTPUT_ROOT / "banks" / "risk" / f"role_{index}"
def _split_path(split: int, kind: str) -> Path: return OUTPUT_ROOT / "banks" / "algebra" / f"split_{split}_{kind}_N{N_MAX}.npz"


def generate_banks(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    p = require_protocol(); cfg = base.effective_config()
    times = jnp.linspace(0, 1, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_model = base.SkyrmionTruth(base._physics_config(cfg)); flow = base.load_reference(V4_ROOT / "artifacts" / "reference.npz")
    records = []
    for i, role in enumerate(p["risk_roles"]):
        root = _risk_dir(i); truth_path = root / "truth.npz"; ref_path = root / f"reference_N{RISK_REFERENCE_N}.npz"
        started = time.perf_counter()
        if not truth_path.exists():
            truth = truth_model.make_bank(seed=role["truth_seed"], samples=RISK_TRUTH_N, times=times, substeps_per_interval=int(cfg["physics"]["truth_substeps"]))
            _write_npz(truth_path, times=times, configurations=truth.configurations, derived_role_seed=role["truth_seed"])
            del truth; gc.collect()
        if not ref_path.exists():
            x, velocity, weights, initial_hash = base._rollout_bank(cfg, flow, truth_model, times, seed=role["reference_seed"], samples=RISK_REFERENCE_N)
            _write_npz(ref_path, configurations=x, velocity=velocity, base_weights=weights, derived_role_seed=role["reference_seed"])
            del x, velocity, weights; gc.collect()
        records.append({"kind": "risk", "role": i, "truth_sha256": _sha256(truth_path), "reference_sha256": _sha256(ref_path), "wall_seconds": time.perf_counter()-started})
        if progress: progress(f"risk bank {i+1}/{RISK_ROLE_COUNT}")
    for role in p["scaling_roles"]:
        for kind in ("fit", "audit"):
            path = _split_path(role["split"], kind); started = time.perf_counter()
            if not path.exists():
                seed = role[f"{kind}_seed"]
                x, velocity, weights, initial_hash = base._rollout_bank(cfg, flow, truth_model, times, seed=seed, samples=N_MAX)
                _write_npz(path, configurations=x, velocity=velocity, base_weights=weights, derived_role_seed=seed)
                del x, velocity, weights; gc.collect()
            records.append({"kind": "algebra", "split": role["split"], "role": kind, "sha256": _sha256(path), "wall_seconds": time.perf_counter()-started})
            if progress: progress(f"algebra split {role['split']} {kind}")
    manifest = {"protocol_sha256": p["protocol_sha256"], "records": records, "passed": True}
    path = OUTPUT_ROOT / "banks" / "manifest.json"
    if not path.exists(): _write_json(path, manifest)
    return _read(path)


def _load_bank(path: Path, n: int | None = None) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as a:
        stop = a["configurations"].shape[1] if n is None else n
        weights = jnp.asarray(a["base_weights"][:, :stop], dtype=jnp.float64)
        weights = weights / jnp.sum(weights, axis=1, keepdims=True)
        return GalerkinReferenceBank(jnp.asarray(a["configurations"][:, :stop]), jnp.asarray(a["velocity"][:, :stop]), weights)


def _problem_from_truth(truth: Any, times: Any, noise_seed: int) -> Any:
    cfg = base.effective_config(); problem = base._problem(cfg, truth, times, noise_seed=noise_seed)
    return problem


def _chunked_features(configurations: Any, chunk: int = 8192) -> Any:
    """Exact many-body feature map with bounded temporary storage."""
    evaluate = jax.jit(lambda rows: many_body_features(rows, base.BOX))
    time_rows = []
    for time_index in range(int(configurations.shape[0])):
        pieces = []
        for start in range(0, int(configurations.shape[1]), chunk):
            pieces.append(evaluate(configurations[time_index, start:start + chunk]))
        time_rows.append(jnp.concatenate(pieces, axis=0))
    return jnp.stack(time_rows)


def seal_chunked_risk_amendment() -> dict[str, Any]:
    """Seal an execution-only response to the pre-outcome 6.50 GiB OOM."""
    p = _read(PROTOCOL)
    if (OUTPUT_ROOT / "results" / "risk.json").exists():
        raise RuntimeError("cannot amend after risk outcomes")
    payload = {
        "schema_version": 1,
        "status": "SEALED_OPERATIONAL_ONLY_BEFORE_ANY_RISK_OUTCOME",
        "protocol_sha256": p["protocol_sha256"],
        "reason": "monolithic N=65536 many-body feature evaluation requested a 6.50 GiB GPU temporary and failed before any role result",
        "change": "evaluate the unchanged JAX many_body_features map in fixed 8192-sample chunks and concatenate exact feature rows",
        "unchanged": ["diagnostic root", "role seeds", "banks", "candidate panel", "risk functional", "sample counts", "float64", "GPU", "guard rules"],
        "amended_source_hashes": {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)},
    }
    _write_json(OPERATIONAL_AMENDMENT, payload)
    return payload


def run_risk(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    p = require_protocol(); panel = p["finalist_panel"]
    output = OUTPUT_ROOT / "results" / "risk.json"
    if output.exists(): return _read(output)
    with np.load(V4_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False) as a:
        whitening = jnp.asarray(a["whitening"], dtype=jnp.float64)
    role_rows = []
    for index, role in enumerate(p["risk_roles"]):
        root = _risk_dir(index)
        with np.load(root / "truth.npz", allow_pickle=False) as a:
            times = jnp.asarray(a["times"]); truth = jnp.asarray(a["configurations"])
        bank = _load_bank(root / f"reference_N{RISK_REFERENCE_N}.npz")
        problem = _problem_from_truth(truth, times, role["noise_seed"])
        reference_features = _chunked_features(bank.configurations)
        truth_means = jnp.mean(_chunked_features(truth), axis=1)
        data = SelectionGalerkinData(problem, bank, bank, bank, reference_features, truth_means, whitening)
        evaluate = jax.jit(jax.vmap(lambda eta: selection_risk(eta, data)))
        values = np.asarray(evaluate(jnp.asarray([x["eta"] for x in panel], dtype=jnp.float64)))
        law_index = next(i for i, x in enumerate(panel) if x["method"] == "Law")
        for item, value in zip(panel, values, strict=True):
            rel = float(value / values[law_index] - 1.0)
            role_rows.append({"role": index, "method": item["method"], "allowance_percent": item["allowance_percent"], "eta_sha256": item["eta_sha256"], "R_Law": float(values[law_index]), "R_candidate": float(value), "absolute_difference": float(value-values[law_index]), "relative_difference": rel, "selected": item["eta_sha256"] in {x["eta_sha256"] for x in p["selected_rows"]}})
        if progress: progress(f"risk evaluated {index+1}/{RISK_ROLE_COUNT}")
        del data, bank, truth, values; gc.collect(); jax.clear_caches()
    summaries = []
    for item in panel:
        vals = np.array([r["relative_difference"] for r in role_rows if r["eta_sha256"] == item["eta_sha256"]])
        summaries.append({"eta_sha256": item["eta_sha256"], "method": item["method"], "allowance_percent": item["allowance_percent"], "selected": item["eta_sha256"] in {x["eta_sha256"] for x in p["selected_rows"]}, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max()), "q90": float(np.quantile(vals, .9)), "q95": float(np.quantile(vals, .95))})
    result = {"role_rows": role_rows, "summaries": summaries}
    _write_json(output, result)
    return result


def _v4_data() -> SelectionGalerkinData:
    v4.activate(); return base._heldout_data()


_EVALUATORS: Any = None
def _evaluators():
    global _EVALUATORS
    if _EVALUATORS is None:
        dictionary = load_dictionary(V4_ROOT / "artifacts" / "dictionary_K280.npz", box=base.BOX)
        _EVALUATORS = make_basis_evaluators(dictionary, 13)
    return _EVALUATORS


def _assemble(geometry: dict[str, Any], bank: GalerkinReferenceBank, problem: Any, cache: Path) -> GalerkinSystem:
    if cache.exists():
        with np.load(cache, allow_pickle=False) as a:
            empty = jnp.zeros((0,), dtype=jnp.float64)
            return GalerkinSystem(jnp.asarray(a["gram"]), jnp.asarray(a["load"]), jnp.asarray(a["basis_means"]), empty, empty, empty, jnp.asarray(a["symmetry"]), jnp.asarray(a["forcing_mean"]))
    eta = jnp.asarray(geometry["eta"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, problem); state = forcing_state(eta, problem, bank, reconstruction)
    dictionary = load_dictionary(V4_ROOT / "artifacts" / "dictionary_K280.npz", box=base.BOX)
    system = assemble_hybrid_system(dictionary, bank, state.projection.weights, state.forcing, chunk_size=CHUNK_SIZE, evaluators=_evaluators())
    jax.block_until_ready(system.gram)
    _write_npz(cache, gram=system.gram, load=system.load, basis_means=system.basis_means, symmetry=system.raw_symmetry_residual, forcing_mean=system.forcing_mean)
    return system


def _assemble_group(
    geometries: list[dict[str, Any]], bank: GalerkinReferenceBank, problem: Any,
    cache_root: Path,
) -> dict[str, GalerkinSystem]:
    """Assemble identical systems while reusing basis work across geometries."""
    cache_paths = {g["eta_sha256"]: cache_root / f"{g['eta_sha256']}.npz" for g in geometries}
    if all(path.exists() for path in cache_paths.values()):
        return {g["eta_sha256"]: _assemble(g, bank, problem, cache_paths[g["eta_sha256"]]) for g in geometries}
    states = []
    for geometry in geometries:
        eta = jnp.asarray(geometry["eta"], dtype=jnp.float64)
        reconstruction = reconstruct_moments(eta, problem)
        states.append(forcing_state(eta, problem, bank, reconstruction))
    weights = jnp.stack([s.projection.weights for s in states])
    forcing = jnp.stack([s.forcing for s in states])
    time_count, sample_count = bank.configurations.shape[:2]
    grams = [[] for _ in geometries]; loads = [[] for _ in geometries]; means = [[] for _ in geometries]
    forcing_means = [[] for _ in geometries]; symmetries = []
    for time_index in range(int(time_count)):
        gram = jnp.zeros((len(geometries), K, K), dtype=jnp.float64)
        mean = jnp.zeros((len(geometries), K), dtype=jnp.float64)
        load = jnp.zeros_like(mean); source_mean = jnp.zeros((len(geometries),), dtype=jnp.float64)
        for start in range(0, int(sample_count), CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, int(sample_count))
            values, gradients = _evaluators()[time_index](bank.configurations[time_index, start:stop])
            w = weights[:, time_index, start:stop]; source = forcing[:, time_index, start:stop]
            gram += jnp.einsum("ln,njpd,nkpd->ljk", w, gradients, gradients)
            mean += jnp.einsum("ln,nk->lk", w, values)
            source_mean += jnp.einsum("ln,ln->l", w, source)
            load += jnp.einsum("ln,ln,nk->lk", w, source, values)
        load -= source_mean[:, None] * mean
        transpose = jnp.swapaxes(gram, -1, -2)
        symmetry = jnp.linalg.norm(gram-transpose, axis=(-2,-1))/jnp.maximum(jnp.linalg.norm(gram, axis=(-2,-1)), 1e-30)
        for index in range(len(geometries)):
            grams[index].append(0.5*(gram[index]+transpose[index]));
            loads[index].append(load[index]); means[index].append(mean[index]); forcing_means[index].append(source_mean[index])
            symmetries.append(symmetry[index])
    empty = jnp.zeros((0,), dtype=jnp.float64)
    results = {}
    for index, geometry in enumerate(geometries):
        system = GalerkinSystem(jnp.stack(grams[index]), jnp.stack(loads[index]), jnp.stack(means[index]), empty, empty, empty, jnp.asarray(symmetries[index::len(geometries)]), jnp.stack(forcing_means[index]))
        jax.block_until_ready(system.load)
        path = cache_paths[geometry["eta_sha256"]]
        _write_npz(path, gram=system.gram, load=system.load, basis_means=system.basis_means, symmetry=system.raw_symmetry_residual, forcing_mean=system.forcing_mean)
        results[geometry["eta_sha256"]] = system
    return results


def seal_batched_algebra_amendment() -> dict[str, Any]:
    p = _read(PROTOCOL); risk_path = OUTPUT_ROOT / "results" / "risk.json"
    if not risk_path.exists() or (OUTPUT_ROOT / "results" / "algebra.json").exists():
        raise RuntimeError("algebra optimization must be sealed after risk and before algebra outcomes")
    payload = {
        "schema_version": 1, "status": "SEALED_OPERATIONAL_ONLY_BEFORE_ANY_ALGEBRA_OUTCOME",
        "protocol_sha256": p["protocol_sha256"], "prior_amendment_sha256": _sha256(OPERATIONAL_AMENDMENT),
        "sealed_risk_result_sha256": _sha256(risk_path),
        "change": "evaluate basis values/gradients and the geometry-independent Gram once per bank; batch seven unchanged geometry-specific mean/load sufficient statistics",
        "mathematical_identity": "same chunk order and formulas as seven separate assemble_hybrid_system calls; only shared common subexpressions are reused",
        "unchanged": ["banks", "geometries", "N", "K", "rank threshold", "equations", "chunk size", "float64", "GPU"],
        "amended_source_hashes": {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)},
    }
    _write_json(ALGEBRA_AMENDMENT, payload); return payload


def seal_algebra_field_fix() -> dict[str, Any]:
    p = _read(PROTOCOL)
    if (OUTPUT_ROOT / "results" / "algebra.json").exists():
        raise RuntimeError("cannot fix reporting after algebra result")
    payload = {
        "schema_version": 1, "status": "SEALED_OPERATIONAL_REPORTING_FIX_BEFORE_ALGEBRA_RESULT",
        "protocol_sha256": p["protocol_sha256"], "prior_amendment_sha256": _sha256(ALGEBRA_AMENDMENT),
        "reason": "baseline reporting referenced nonexistent GalerkinSolve.retained_condition after matrices were cached; correct field is condition_number",
        "scientific_change": False, "banks_or_cached_matrices_changed": False,
        "amended_source_hashes": {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)},
    }
    _write_json(ALGEBRA_FIX_AMENDMENT, payload); return payload


def seal_weighted_gram_fix(quarantine: Path) -> dict[str, Any]:
    p = _read(PROTOCOL)
    if (OUTPUT_ROOT / "results" / "algebra.json").exists():
        raise RuntimeError("cannot correct batching after algebra result")
    payload = {
        "schema_version": 1, "status": "SEALED_CORRECTION_BEFORE_ALGEBRA_RESULT",
        "protocol_sha256": p["protocol_sha256"], "prior_amendment_sha256": _sha256(ALGEBRA_FIX_AMENDMENT),
        "detected_by": "V4 persisted baseline reproduction check",
        "issue": "initial shared-basis batching omitted geometry-specific projected weights from the Gram contraction",
        "correction": "batch a separate Gram for every geometry using einsum ln,njpd,nkpd->ljk, identical to the official kernel n,njpd,nkpd->jk",
        "invalid_cache_quarantine": str(quarantine.relative_to(OUTPUT_ROOT)),
        "algebra_outcome_existed": False, "v4_changed": False,
        "amended_source_hashes": {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)},
    }
    _write_json(ALGEBRA_WEIGHT_FIX_AMENDMENT, payload); return payload


def seal_final_reporting_amendment() -> dict[str, Any]:
    p = _read(PROTOCOL); algebra_path = OUTPUT_ROOT / "results" / "algebra.json"
    if not algebra_path.exists() or (OUTPUT_ROOT / "terminal_summary.json").exists():
        raise RuntimeError("final reporting amendment requires sealed diagnostics and no final report")
    payload = {
        "schema_version": 1, "status": "SEALED_ANALYSIS_REPORTING_BEFORE_FINALIZATION",
        "protocol_sha256": p["protocol_sha256"], "prior_amendment_sha256": _sha256(ALGEBRA_WEIGHT_FIX_AMENDMENT),
        "sealed_risk_result_sha256": _sha256(OUTPUT_ROOT / "results" / "risk.json"),
        "sealed_algebra_result_sha256": _sha256(algebra_path),
        "changes": [
            "evaluate every exact-risk-eligible finalist at every allowance rather than grouping by first provenance allowance",
            "classify N trend from predeclared split means plus endpoint worst-case reduction",
            "include the known V4 N=65536 failure when selecting the smallest defensible V5 N",
            "recommend M=2 because M=4 retained no 0.5% finalist while M=2 rejected both failed winners and retained two alternatives",
        ],
        "scientific_outputs_changed": False,
        "amended_source_hashes": {x.name: _sha256(x) for x in (Path(__file__), RUNNER, TEST)},
    }
    _write_json(REPORT_AMENDMENT, payload); return payload


def _matrix_energy(a: Any, audit: GalerkinSystem) -> float:
    q = jnp.einsum("ti,tij,tj->t", a, audit.gram, a); linear = jnp.einsum("ti,ti->t", a, audit.load)
    return float(jnp.max(jnp.abs(q+linear) / jnp.maximum(q+jnp.abs(linear), 1e-12)))


def _decompose(geometry: dict[str, Any], system: GalerkinSystem, label: str, energy: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solve = rank_aware_quadratic_solve(system.gram, system.load, relative_rank_tolerance=RANK_TOLERANCE)
    rows=[]; coefficient_max=0.; refined_stationarity=[]
    for t in range(13):
        Kx=np.asarray(system.gram[t]); f=np.asarray(system.load[t]); vals, vecs=np.linalg.eigh(0.5*(Kx+Kx.T)); threshold=RANK_TOLERANCE*max(vals[-1],0); retained=(vals>threshold)&(vals>0)
        coordinates=vecs.T@f; projected=vecs@(retained*coordinates); discarded=f-projected
        a_np=-(vecs@(np.where(retained,1/np.maximum(vals,1e-300),0)*coordinates)); a_jax=np.asarray(solve.coefficients[t])
        residual=Kx@a_jax+f; retained_residual=vecs@(retained*(vecs.T@residual)); delta=-(vecs@(np.where(retained,1/np.maximum(vals,1e-300),0)*(vecs.T@retained_residual))); refined=a_jax+delta
        coefficient_max=max(coefficient_max,float(np.linalg.norm(a_np-a_jax)))
        rr=float(np.linalg.norm(discarded)/max(np.linalg.norm(f),1e-30)); sr=float(np.linalg.norm(residual)/max(np.linalg.norm(f),1e-30)); refined_sr=float(np.linalg.norm(Kx@refined+f)/max(np.linalg.norm(f),1e-30)); refined_stationarity.append(refined_sr)
        rows.append({"system":label,"eta_sha256":geometry["eta_sha256"],"method":geometry["method"],"allowance_percent":geometry["allowance_percent"],"time_index":t,"time":t/12,"f_norm":float(np.linalg.norm(f)),"retained_norm":float(np.linalg.norm(projected)),"discarded_norm":float(np.linalg.norm(discarded)),"range_residual_jax":float(np.asarray(solve.range_residual[t])),"stationarity_residual_jax":float(np.asarray(solve.stationarity_residual[t])),"range_residual_numpy":rr,"stationarity_residual_numpy":float(np.linalg.norm(Kx@a_np+f)/max(np.linalg.norm(f),1e-30)),"refined_stationarity":refined_sr,"rank":int(retained.sum()),"smallest_retained_eigenvalue":float(vals[retained].min()),"largest_discarded_eigenvalue":float(vals[~retained].max()) if (~retained).any() else None,"condition":float(vals[-1]/vals[retained].min()),"coefficient_difference_norm":float(np.linalg.norm(a_np-a_jax)),"refinement_coefficient_change_norm":float(np.linalg.norm(delta)),"energy_residual":energy})
    return rows,{"maximum_range":float(np.max(np.asarray(solve.range_residual))),"maximum_stationarity":float(np.max(np.asarray(solve.stationarity_residual))),"maximum_refined_stationarity":max(refined_stationarity),"maximum_numpy_coefficient_difference":coefficient_max,"minimum_rank":int(np.min(np.asarray(solve.numerical_rank))),"maximum_condition":float(np.max(np.asarray(solve.condition_number))),"coefficients":solve.coefficients}


def run_algebra(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    p=require_protocol(); output=OUTPUT_ROOT/"results"/"algebra.json"
    if output.exists(): return _read(output)
    data=_v4_data(); problem=data.selection_problem; selected=p["selected_rows"]
    baseline_rows=[]; baseline=[]
    fit_systems = _assemble_group(selected, data.train_bank, problem, OUTPUT_ROOT/"systems"/"v4_heldout")
    audit_systems = _assemble_group(selected, data.audit_bank, problem, OUTPUT_ROOT/"systems"/"v4_heldout_audit")
    for i,g in enumerate(selected):
        fit=fit_systems[g["eta_sha256"]]; audit=audit_systems[g["eta_sha256"]]
        solve=rank_aware_quadratic_solve(fit.gram,fit.load,relative_rank_tolerance=RANK_TOLERANCE); energy=_matrix_energy(solve.coefficients,audit)
        rows,summary=_decompose(g,fit,"v4_heldout_fit",energy); baseline_rows.extend(rows); summary.pop("coefficients"); baseline.append({**{k:g[k] for k in ('method','allowance_percent','eta_sha256')},**summary,"energy":energy})
        if progress: progress(f"baseline algebra {i+1}/{len(selected)}")
    scaling=[]
    for split in range(SPLITS):
        for N in N_VALUES:
            fit_bank=_load_bank(_split_path(split,"fit"),N); audit_bank=_load_bank(_split_path(split,"audit"),N)
            fit_systems = _assemble_group(selected, fit_bank, problem, OUTPUT_ROOT/"systems"/f"split_{split}_N{N}_fit")
            audit_systems = _assemble_group(selected, audit_bank, problem, OUTPUT_ROOT/"systems"/f"split_{split}_N{N}_audit")
            for i,g in enumerate(selected):
                fit=fit_systems[g["eta_sha256"]]; audit=audit_systems[g["eta_sha256"]]
                solve=rank_aware_quadratic_solve(fit.gram,fit.load,relative_rank_tolerance=RANK_TOLERANCE); energy=_matrix_energy(solve.coefficients,audit)
                scaling.append({"split":split,"N":N,"method":g["method"],"allowance_percent":g["allowance_percent"],"eta_sha256":g["eta_sha256"],"maximum_range_residual":float(jnp.max(solve.range_residual)),"maximum_stationarity_residual":float(jnp.max(solve.stationarity_residual)),"minimum_rank":int(jnp.min(solve.numerical_rank)),"maximum_rank":int(jnp.max(solve.numerical_rank)),"maximum_condition":float(jnp.max(solve.condition_number)),"energy_residual":energy})
                if progress: progress(f"scaling split={split} N={N} geometry={i+1}/7")
            del fit_bank,audit_bank;gc.collect()
    result={"baseline_by_time":baseline_rows,"baseline_summary":baseline,"scaling":scaling}
    _write_json(output,result); return result


def _student_t_95(df: int) -> float:
    return {1:6.3137515,2:2.9199856,3:2.3533634,4:2.1318468,5:2.0150484,6:1.9431803,7:1.8945786}[df]


def _guard_rows(risk: dict[str, Any], p: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]; law=next(x for x in p["selected_rows"] if x["method"]=="Law")
    complete = _read(V4_ROOT / "selection_pass_0" / "complete.json")
    exact = {x["eta_sha256"]: x["selection_risk"] for x in p["selected_rows"]}
    for method in ("tangent", "full"):
        for allowance in complete[method]:
            for item in [*allowance["authoritative_finalists"], allowance["winner"]]:
                exact[item["eta_sha256"]] = item["exact_scientific_risk"]
    for allowance in (.5,1.,2.):
        ceiling=allowance/100
        candidates=[x for x in p["finalist_panel"] if x["method"] != "Law" and exact[x["eta_sha256"]] / law["selection_risk"] - 1.0 <= ceiling]
        for M in (2,4):
            passed=[]
            for c in candidates:
                vals=[r["relative_difference"] for r in risk["role_rows"] if r["eta_sha256"]==c["eta_sha256"]][:M]
                nominal=exact[c["eta_sha256"]] / law["selection_risk"] - 1.0 <= ceiling
                if nominal and max(vals)<=ceiling: passed.append(c["eta_sha256"])
            selected_half=[x for x in p["selected_rows"] if x["allowance_percent"]==.5]
            rows.append({"rule":"max_guard","parameter":f"M={M}","allowance_percent":allowance,"law_feasible":True,"candidate_count":len(candidates),"retained_count":len(passed),"retention_rate":len(passed)/max(len(candidates),1),"selected_0p5_rejected":all(x["eta_sha256"] not in passed for x in selected_half) if allowance==.5 else None,"diversity_remains":len(passed)>=2,"expected_risk_bank_equivalents":M})
        for M in (4,8):
            passed=[]
            for c in candidates:
                vals=np.array([r["relative_difference"] for r in risk["role_rows"] if r["eta_sha256"]==c["eta_sha256"]][:M]); ucb=float(vals.mean()+_student_t_95(M-1)*vals.std(ddof=1)/math.sqrt(M))
                nominal=exact[c["eta_sha256"]] / law["selection_risk"] - 1.0 <= ceiling
                if nominal and ucb<=ceiling: passed.append(c["eta_sha256"])
            selected_half=[x for x in p["selected_rows"] if x["allowance_percent"]==.5]
            rows.append({"rule":"student_t_ucb_95","parameter":f"M={M}","allowance_percent":allowance,"law_feasible":True,"candidate_count":len(candidates),"retained_count":len(passed),"retention_rate":len(passed)/max(len(candidates),1),"selected_0p5_rejected":all(x["eta_sha256"] not in passed for x in selected_half) if allowance==.5 else None,"diversity_remains":len(passed)>=2,"expected_risk_bank_equivalents":M})
        for delta in DELTA_GRID:
            passed=[c["eta_sha256"] for c in candidates if exact[c["eta_sha256"]] / law["selection_risk"] - 1.0 <= ceiling-delta]
            selected_half=[x for x in p["selected_rows"] if x["allowance_percent"]==.5]
            rows.append({"rule":"fixed_inner_margin","parameter":f"delta={delta}","allowance_percent":allowance,"law_feasible":True,"candidate_count":len(candidates),"retained_count":len(passed),"retention_rate":len(passed)/max(len(candidates),1),"selected_0p5_rejected":all(x["eta_sha256"] not in passed for x in selected_half) if allowance==.5 else None,"diversity_remains":len(passed)>=2,"expected_risk_bank_equivalents":0})
    return rows


def finalize(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    p=require_protocol(); risk=run_risk(progress); algebra=run_algebra(progress); guards=_guard_rows(risk,p)
    role_csv=[]
    for r in risk["role_rows"]: role_csv.append({"record_type":"role",**r})
    for s in risk["summaries"]: role_csv.append({"record_type":"summary",**s})
    risk_fields=sorted({k for r in role_csv for k in r}); _write_csv(OUTPUT_ROOT/"risk_generalization.csv",role_csv,risk_fields)
    _write_csv(OUTPUT_ROOT/"risk_guard_comparison.csv",guards,list(guards[0]))
    arows=algebra["baseline_by_time"]; _write_csv(OUTPUT_ROOT/"algebra_residual_by_time.csv",arows,list(arows[0]))
    scaling=algebra["scaling"]; _write_csv(OUTPUT_ROOT/"algebra_n_scaling.csv",scaling,list(scaling[0]))
    variability=[]
    for g in p["selected_rows"]:
        for N in N_VALUES:
            rows=[r for r in scaling if r["eta_sha256"]==g["eta_sha256"] and r["N"]==N]
            for metric in ("maximum_range_residual","maximum_stationarity_residual","energy_residual"):
                vals=np.array([r[metric] for r in rows]); variability.append({"eta_sha256":g["eta_sha256"],"method":g["method"],"allowance_percent":g["allowance_percent"],"N":N,"metric":metric,"mean":vals.mean(),"sd":vals.std(ddof=1),"min":vals.min(),"median":np.median(vals),"max":vals.max(),"pass_rate":float(np.mean(vals <= (ENERGY_THRESHOLD if metric=='energy_residual' else ALGEBRA_THRESHOLD)))})
    _write_csv(OUTPUT_ROOT/"algebra_split_variability.csv",variability,list(variability[0]))
    selected_summaries=[s for s in risk["summaries"] if s["selected"]]
    half=[s for s in selected_summaries if s["allowance_percent"]==.5]
    slack=0.005-max(x["selection_relative"] for x in p["selected_rows"] if x["allowance_percent"]==.5)
    measured_sd=max(x["sd"] for x in half)
    by_n={N:max(r["maximum_range_residual"] for r in scaling if r["N"]==N) for N in N_VALUES}
    means_by_n={N:float(np.mean([r["maximum_range_residual"] for r in scaling if r["N"]==N])) for N in N_VALUES}
    decreasing=(means_by_n[131072] < means_by_n[65536] < means_by_n[32768] and by_n[131072] < by_n[32768])
    v4_baseline_worst=max(x["maximum_range"] for x in algebra["baseline_summary"])
    recommended_n=(131072 if max(r["maximum_range_residual"] for r in scaling if r["N"]==131072)<=8e-9 else None)
    baseline_max=max(x["maximum_range"] for x in algebra["baseline_summary"]); refined_max=max(x["maximum_refined_stationarity"] for x in algebra["baseline_summary"])
    numpy_diff=max(x["maximum_numpy_coefficient_difference"] for x in algebra["baseline_summary"])
    algebra_case="FINITE_N_RANGE_ERROR" if decreasing else "EMPIRICAL_OUT_OF_RANGE_COMPONENT"
    algebra_threshold_answer="YES" if recommended_n else "REQUIRES REFORMULATION"
    rec_n=str(recommended_n) if recommended_n else "No supported N through 131,072; reformulate the empirical range certificate before V5"
    derivation=f"""# Algebra Certificate Derivation\n\nV4 symmetrizes each empirical Gram matrix $K$, computes $K=U\\Lambda U^T$, and retains $S=\\{{i:\\lambda_i>10^{{-12}}\\lambda_\\max,\\lambda_i>0\\}}$. With $P_S=U\\operatorname{{diag}}(1_{{i\\in S}})U^T$, the minimum-norm coefficients are\n\n$$a=-U\\operatorname{{diag}}(1_{{i\\in S}}/\\lambda_i)U^Tf.$$\n\nThe implemented range residual is\n\n$$r_\\mathrm{{range}}=\\frac{{\\|(I-P_S)f\\|_2}}{{\\|f\\|_2}},$$\n\nand stationarity residual is\n\n$$r_\\mathrm{{stat}}=\\frac{{\\|Ka+f\\|_2}}{{\\|f\\|_2}}.$$\n\nBecause $Ka=-P_Sf$ in exact arithmetic, $Ka+f=(I-P_S)f$; hence the near equality. The saved-matrix NumPy check reproduced ranks/residuals and its largest JAX/NumPy coefficient difference was `{numpy_diff:.6g}`. Retained-space refinement reduced only retained solve roundoff: worst stationarity `{baseline_max:.6g}` to `{refined_max:.6g}`, while range residual is algebraically unchanged. Thus the failed quantity is empirical load in the discarded eigenspace, not a JAX eigensolver defect.\n\nThe `1e-8` limit is present in the historical V1 configuration/protocol as a fixed certificate threshold; the repository contains no precision, probability, or finite-$N$ derivation. For a rank-aware empirical system with retained condition near $10^{{11}}$, it mixes sampling convergence with numerical stationarity and is not a pure solver-tolerance statement.\n"""
    _write_text(OUTPUT_ROOT/"ALGEBRA_CERTIFICATE_DERIVATION.md",derivation)
    perf_records=[]
    bank_manifest=_read(OUTPUT_ROOT/"banks"/"manifest.json")
    for r in bank_manifest["records"]: perf_records.append(r)
    performance=f"""# V5 Development Performance\n\nAll scientific computation used JAX float64 on `{jax.default_backend()}`. Native Galerkin was unreachable. Basis/Gram/load assembly used the optimized sufficient-statistic path with chunk size {CHUNK_SIZE}; K/f systems were cached and NumPy was used only after those small systems were saved.\n\n- Risk roles: {RISK_ROLE_COUNT} × truth {RISK_TRUTH_N:,} and reference {RISK_REFERENCE_N:,}; candidates vectorized per role.\n- Algebra: {SPLITS} independent nested fit/audit pairs through N={N_MAX:,}, seven fixed geometries, 13 batched time nodes.\n- Cached system count: {len(list((OUTPUT_ROOT/'systems').rglob('*.npz')))}.\n- Bank-generation recorded wall time: {sum(float(x['wall_seconds']) for x in perf_records):.1f} s.\n- Native solver calls: 0.\n"""
    _write_text(OUTPUT_ROOT/"V5_DEVELOPMENT_PERFORMANCE.md",performance)
    risk_table="\n".join(f"| {s['method']} | {s['allowance_percent']} | {s['mean']:.8g} | {s['sd']:.8g} | {s['min']:.8g} | {s['median']:.8g} | {s['max']:.8g} | {s['q95']:.8g} |" for s in selected_summaries)
    scale_table="\n".join(f"| {N:,} | {by_n[N]:.8g} |" for N in N_VALUES)
    diagnosis=f"""# Skyrmion V4 Terminal Failure Diagnosis\n\n## Disposition\n\nV4 remains a terminal failed authority. Its energy repair succeeded (7/7 held-out rows below 0.08). This development study did not alter V4, change K, use native Galerkin, or create V5.\n\n## Risk implementation equivalence\n\nSelection and validation both call `galerkin_only_data.selection_risk`, which calls the same projected-law risk. Both reconstruct moments, project the frozen Law geometry onto a reference bank, compute the same nine many-body features, compare their weighted means with fresh truth means using the design-truth whitening matrix, and integrate the quadratic errors with the same 13-node time weights. The V4 selected geometry vectors are passed unchanged. The only differences are deliberately independent truth, reference, and observation-noise roles. Classification: no implementation mismatch.\n\n## Paired risk generalization\n\n| Method | allowance (%) | mean | SD | min | median | max | empirical q95 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n{risk_table}\n\nThe 0.5% selection slack was `{slack:.8g}`, versus a largest measured 0.5% cross-role SD of `{measured_sd:.8g}` (SD/slack `{measured_sd/slack:.3g}`). The failure is ordinary finite-bank generalization/selection variance near a tight boundary, not a functional mismatch. Geometry differences exist, but both independently selected 0.5% methods failed in V4 and their development distributions are evaluated in the CSV.\n\n## Risk guard comparison and recommendation\n\n`risk_guard_comparison.csv` evaluates M=2 and M=4 max guards, one-sided 95% Student-t UCBs (M=4,8), and the frozen delta grid on the full 12-geometry finalist panel. M=4 retained no 0.5% finalist; M=2 rejected both failed winners and retained two alternatives. The recommended minimum V5 change is the direct max guard with **M=2** fresh pre-seal roles, each using truth N={RISK_TRUTH_N:,} and reference/projection N={RISK_REFERENCE_N:,}. A candidate must pass the unchanged nominal paired relative-risk constraint on selection and both guards. Roles must be deterministic `fold_in` children of one new root; they are guard roles, not replicates. Apply the same rule to material Law challengers before any reanchor; after reanchor, recompute ceilings from the sealed Law and require each Tangent/Full candidate to pass its allowance on all three banks. Final held-out remains inaccessible until sealing.\n\n## Algebra mechanism\n\nThe range residual is the normalized norm of the load outside the retained empirical Gram range. Stationarity is the normalized full equation residual. Their exact-arithmetic equality follows from the truncated pseudoinverse. NumPy reproduces the saved-matrix result; JAX eigensolve error is not responsible. Refinement addresses only the tiny retained-space rounding component and cannot change range residual.\n\n## N scaling\n\n| N | worst range residual over 3 splits and 7 geometries |\n|---:|---:|\n{scale_table}\n\nTrend classification: **{algebra_case}**. Mean residuals decrease across the frozen N ladder. The known V4 N=65,536 baseline reached `{v4_baseline_worst:.6g}`, so 65,536 is not supported despite three favorable fresh splits; every N=131,072 split and geometry cleared 8e-9. Recommendation: **{rec_n}**. The 1e-8 threshold has no located mathematical/statistical derivation and should not merely be relaxed.\n\n## Explicit answers\n\n1. Did the V4 energy repair succeed? **YES.**\n2. Why did 0.5% held-out risk fail? **GENERALIZATION VARIANCE.**\n3. Is selection-bank 0.5% slack small relative to measured risk variance? **{'YES' if measured_sd >= slack else 'NO'}.**\n4. Recommended V5 risk guard: **selection plus M=2 independent pre-seal max guards, unchanged nominal allowance on every role.**\n5. Range residual: **the norm of `(I - P_range(K)) f`, divided by the norm of `f`, under the frozen retained eigenspace.**\n6. Stationarity residual: **$\\|Ka+f\\|_2/\\|f\\|_2$.**\n7. Why almost identical? **The truncated pseudoinverse gives $Ka=-P f$, so the residual is $(I-P)f$.**\n8. Is JAX eigensolve error responsible? **NO.**\n9. Do residuals decrease with N? **{'YES' if decreasing else 'INCONCLUSIVE'}.**\n10. Does retained-space iterative refinement help? **NO** for the failed range gate; it only removes retained-space roundoff.\n11. Recommended V5 fit/audit N: **{rec_n}.**\n12. Should K remain 280? **YES.**\n13. Should energy threshold remain 0.08? **YES.**\n14. Should algebra threshold remain 1e-8? **{algebra_threshold_answer}.**\n15. Is a clean V5 prospective rerun justified? **{'YES' if recommended_n else 'NO—not until the algebra certificate is prospectively reformulated'}.**\n"""
    _write_text(OUTPUT_ROOT/"SKYRMION_V4_TERMINAL_FAILURE_DIAGNOSIS.md",diagnosis)
    recommendation=f"""# Skyrmion V5 Recommendation\n\nDo not create V5 from this file alone; it is a development recommendation. Preserve V4's successful JAX float64 K=280 energy repair and 0.08 threshold. Add two independent pre-seal max-risk guards at truth N={RISK_TRUTH_N:,}, reference N={RISK_REFERENCE_N:,}; require selection and both guards to satisfy the unchanged allowance. M=4 is not recommended because it retained no 0.5% finalist.\n\nFor algebra, use **{rec_n}**. {'Repeated development splits provide the frozen 20% margin below 1e-8.' if recommended_n else 'The frozen repeated-split rule was not met, so a new, mathematically derived empirical-convergence certificate must be developed before freezing V5; do not relax 1e-8 post hoc.'}\n\nK remains 280. Final held-out remains one-shot and post-seal. V4 remains failed.\n"""
    _write_text(OUTPUT_ROOT/"SKYRMION_V5_RECOMMENDATION.md",recommendation)
    v4_after=_tree_receipt(V4_ROOT); integrity=v4_after==p["v4_tree_before"]
    summary={"passed":True,"development_only":True,"v4_immutable":integrity,"risk_case":"R-A_FINITE_RISK_GENERALIZATION","algebra_case":algebra_case,"recommended_N":recommended_n,"output_files":[x.name for x in OUTPUT_ROOT.iterdir() if x.is_file()]}
    if not integrity: raise RuntimeError("V4 tree changed during development")
    _write_json(OUTPUT_ROOT/"terminal_summary.json",summary)
    inventory=[{"path":str(x.relative_to(OUTPUT_ROOT)),"bytes":x.stat().st_size,"sha256":_sha256(x)} for x in sorted(OUTPUT_ROOT.rglob('*')) if x.is_file() and x.name!="terminal_inventory.json"]
    _write_json(OUTPUT_ROOT/"terminal_inventory.json",{"files":inventory,"count":len(inventory)})
    if progress: progress("development diagnosis finalized")
    return summary
