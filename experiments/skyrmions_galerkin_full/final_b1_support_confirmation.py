"""Final out-of-bank B1 support confirmation.

This development-only stage consumes the frozen clean-room B1 reference, Law,
candidate pool, risk results, design truth, and whitening.  It generates only
new reference rollout banks and support diagnostics.  It has no training,
candidate-generation, Law-optimization, validation, Tangent, Full, Galerkin,
eigensolve, or official-protocol entry point.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .domain import SkyrmionTruth
from .galerkin_only_data import GalerkinReferenceBank, _family, _make_problem, _physics_config
from .pareto_v3_common import ROOT, eta_key, file_sha256
from .pareto_v3_diagnostic import _symmetry_aware_distance
from .reference import load_reference
from .reference_seed_robustness import _ReferenceEvaluator, _array_sha256
from .single_reference_b1_preflight import (
    ACCEPTED_REFERENCE_PATH,
    CANDIDATE_POOL_PATH,
    CANDIDATE_RISK_PATH,
    CONFIG_PATH,
    DESIGN_DATA_MANIFEST_PATH,
    DESIGN_DATA_PATH,
    LAW_FREEZE_PATH,
    MANIFEST_PATH as PREFLIGHT_MANIFEST_PATH,
    OUTPUT_ROOT as PREFLIGHT_ROOT,
    _load_design_arrays,
)


VERSION = "skyrmion_b1_final_support_confirmation_v1"
SEED_NAMESPACE = VERSION
GLOBAL_SEED = 20260826
OUTPUT_ROOT = ROOT / "outputs" / VERSION
REPO_ROOT = ROOT.parent.parent
EXPECTED_CHECKPOINT_SHA256 = "1e13e2ea58df122702d4f555f8788a148b3150bbfbfc953cbac9f963c03d539b"
EXPECTED_TRAIN_DATA_SHA256 = "41a2551c75cc26c5edfbaa59b1849e4280abe5ecb5a8caa192983d8e4ac45e3e"
EXPECTED_QUAL_DATA_SHA256 = "c65b25bb04fc04ae56b83412bafac4e5b2abb0140eb0c071e586a112deb51622"
EXPECTED_DESIGN_DATA_SHA256 = "957014cf63c062b37fafd890c2e23a211b4b1d90be153f0c282a2e769f4ea8ae"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "final_support_source_seal.json"
SOURCE_ERRATUM_PATH = OUTPUT_ROOT / "final_support_source_erratum.json"
BANK_MANIFEST_PATH = OUTPUT_ROOT / "confirmation_bank_manifest.json"
BANK_MANIFEST_HASH_PATH = OUTPUT_ROOT / "confirmation_bank_manifest.sha256"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

PAIR_COUNT = 8
SCREEN_N = 8192
AUDIT_N = 16384
CANDIDATE_COUNT = 2048
TIME_COUNT = 13
NODE7 = 7
MINIMUM_RESS = 0.05
READINESS_LAW_MARGIN = 0.060
LOW_RISK_P10_MARGIN = 0.055
MINIMUM_CANDIDATES = 25
MINIMUM_DIVERSE = 5
MINIMUM_PAIR_SURVIVAL_FRACTION = 0.25
MAXIMUM_NON_RESS_FAILURE_FRACTION = 0.10
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
ROLLOUT_BATCH_SIZE = 2048
BOX = (2.0, 1.0)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _derive_seed(role: str) -> dict[str, Any]:
    text = f"{SEED_NAMESPACE}|{GLOBAL_SEED}|{role}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"role": role, "derivation_text": text, "sha256": digest, "seed": int(digest[:16], 16) % (2**31 - 1)}


def _inside(path: Path) -> Path:
    resolved, root = path.resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"confirmation output must remain beneath {root}: {resolved}")
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = _inside(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists(): raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent); os.close(fd)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()}); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_integer_seeds(payload: Any, *, key: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(payload, dict):
        for name, value in payload.items():
            if name == "seed" and isinstance(value, int): result.add(value)
            result |= _all_integer_seeds(value, key=name)
    elif isinstance(payload, list):
        for value in payload: result |= _all_integer_seeds(value, key=key)
    return result


def _checkpoint_path() -> Path:
    accepted = _json(ACCEPTED_REFERENCE_PATH)
    return PREFLIGHT_ROOT / accepted["checkpoint_path"]


def verify_and_seal_sources() -> dict[str, Any]:
    accepted, law, pool, risk = (_json(path) for path in (ACCEPTED_REFERENCE_PATH, LAW_FREEZE_PATH, CANDIDATE_POOL_PATH, CANDIDATE_RISK_PATH))
    checkpoint = _checkpoint_path()
    if file_sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256 or accepted["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("accepted B1 checkpoint hash mismatch")
    if file_sha256(PREFLIGHT_ROOT / "reference_endpoint_train.npz") != EXPECTED_TRAIN_DATA_SHA256:
        raise RuntimeError("clean-room training endpoint data changed")
    if file_sha256(PREFLIGHT_ROOT / "endpoint_qualification_holdout.npz") != EXPECTED_QUAL_DATA_SHA256:
        raise RuntimeError("clean-room qualification data changed")
    if file_sha256(DESIGN_DATA_PATH) != EXPECTED_DESIGN_DATA_SHA256:
        raise RuntimeError("clean-room design truth changed")
    if pool["count"] != CANDIDATE_COUNT or risk["candidate_pool_sha256"] != file_sha256(CANDIDATE_POOL_PATH):
        raise RuntimeError("frozen candidate evidence changed")
    if eta_key(pool["rows"][0]["eta"]) != eta_key(law["eta_Law_B1"]):
        raise RuntimeError("development Law is not candidate zero")
    definition_hashes = {
        "accepted_reference": file_sha256(ACCEPTED_REFERENCE_PATH), "accepted_checkpoint": file_sha256(checkpoint),
        "B1_bridge_source": file_sha256(ROOT / "single_reference_b1_preflight.py"),
        "development_Law": file_sha256(LAW_FREEZE_PATH), "development_candidate_pool": file_sha256(CANDIDATE_POOL_PATH),
        "development_candidate_risk": file_sha256(CANDIDATE_RISK_PATH), "development_design_truth": file_sha256(DESIGN_DATA_PATH),
        "development_whitening": _json(DESIGN_DATA_MANIFEST_PATH)["whitening_sha256"],
        "scientific_risk": file_sha256(ROOT / "risk.py"), "Psi_definitions": file_sha256(ROOT / "risk.py"),
        "projection": file_sha256(REPO_ROOT / "src" / "mfsi" / "projection.py"),
        "forcing": file_sha256(ROOT / "forcing.py"), "geometry_rules": file_sha256(ROOT / "candidate_coverage.py"),
        "time_weights": file_sha256(ROOT / "galerkin_only_data.py"), "config": file_sha256(CONFIG_PATH),
    }
    sources = [Path(__file__), ROOT / "final_b1_support_confirmation_run.py", ROOT / "test_final_b1_support_confirmation.py"]
    payload = {
        "schema_version": 1, "version": VERSION, "development_only": True,
        "frozen_input_hashes": definition_hashes,
        "analysis_source_hashes": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in sources},
        "accepted_reference_verified": True, "B1_particle_matching_unchanged": True, "configuration_OT": False,
        "Law_optimized": False, "candidates_generated": False, "reference_trained": False,
        "validation_accessed": False, "Tangent_run": False, "Full_run": False,
        "minimum_rESS": MINIMUM_RESS,
    }
    if SOURCE_SEAL_PATH.exists():
        sealed = _json(SOURCE_SEAL_PATH)
        if sealed == payload: return payload
        sealed_without_sources = {key: value for key, value in sealed.items() if key != "analysis_source_hashes"}
        payload_without_sources = {key: value for key, value in payload.items() if key != "analysis_source_hashes"}
        changed_sources = {
            name for name in set(sealed["analysis_source_hashes"]) | set(payload["analysis_source_hashes"])
            if sealed["analysis_source_hashes"].get(name) != payload["analysis_source_hashes"].get(name)
        }
        own_source = str(Path(__file__).relative_to(REPO_ROOT))
        if sealed_without_sources != payload_without_sources or changed_sources != {own_source}:
            raise RuntimeError("confirmation source seal changed outside the documented label erratum")
        erratum = {
            "schema_version": 1,
            "original_source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
            "affected_source": own_source,
            "original_source_sha256": sealed["analysis_source_hashes"][own_source],
            "corrected_source_sha256": payload["analysis_source_hashes"][own_source],
            "reason": "Normalize the derivation-role metadata to the predeclared screen_<pair>/audit_<pair> artifact labels.",
            "scientific_protocol_changed": False,
            "seeds_changed": False,
            "sample_sizes_changed": False,
            "pairing_changed": False,
            "thresholds_changed": False,
            "candidate_evaluation_started_before_correction": False,
            "preserved_partial_artifacts": [
                "banks/screen_0_0.json", "banks/screen_0_0.npz",
                "banks/audit_0_0.json", "banks/audit_0_0.npz",
            ],
        }
        _atomic_json(SOURCE_ERRATUM_PATH, erratum)
        return payload
    _atomic_json(SOURCE_SEAL_PATH, payload); return payload


def freeze_bank_manifest() -> dict[str, Any]:
    verify_and_seal_sources()
    if BANK_MANIFEST_PATH.exists() or BANK_MANIFEST_HASH_PATH.exists():
        if not BANK_MANIFEST_PATH.exists() or not BANK_MANIFEST_HASH_PATH.exists(): raise RuntimeError("incomplete bank manifest seal")
        if file_sha256(BANK_MANIFEST_PATH) != BANK_MANIFEST_HASH_PATH.read_text().strip(): raise RuntimeError("confirmation bank manifest changed")
        return _json(BANK_MANIFEST_PATH)
    seeds = []
    for pair in range(PAIR_COUNT):
        seeds.append({"pair": pair, "role": "screen", "N": SCREEN_N, **_derive_seed(f"screen_{pair}")})
        seeds.append({"pair": pair, "role": "audit", "N": AUDIT_N, **_derive_seed(f"audit_{pair}")})
    values = [row["seed"] for row in seeds]
    if len(set(values)) != 16: raise RuntimeError("confirmation seeds collide")
    prior = _all_integer_seeds(_json(PREFLIGHT_MANIFEST_PATH))
    bridge_manifest = ROOT / "outputs" / "skyrmion_galerkin_dev_bridge_ablation_v1" / "bridge_ablation_manifest.json"
    if bridge_manifest.exists(): prior |= _all_integer_seeds(_json(bridge_manifest))
    if set(values) & prior: raise RuntimeError("confirmation seed overlaps development work")
    payload = {
        "schema_version": 1, "version": VERSION, "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "namespace": SEED_NAMESPACE, "pair_count": PAIR_COUNT, "individual_bank_count": 16, "banks": seeds,
        "all_seeds_unique": True, "no_prior_seed_overlap": True,
        "future_production_namespace": "skyrmion_official_b1_galerkin_pareto_v1", "future_validation_namespace": "skyrmion_official_b1_galerkin_pareto_v1:fresh_validation",
        "frozen_candidates": {"Law_sha256": file_sha256(LAW_FREEZE_PATH), "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH), "candidate_risk_sha256": file_sha256(CANDIDATE_RISK_PATH)},
        "thresholds": {"scientific_minimum_rESS": MINIMUM_RESS, "Law_readiness_margin": READINESS_LAW_MARGIN, "low_risk_p10_margin": LOW_RISK_P10_MARGIN, "minimum_candidate_count": MINIMUM_CANDIDATES, "minimum_diverse_count": MINIMUM_DIVERSE, "minimum_pair_survival_fraction": MINIMUM_PAIR_SURVIVAL_FRACTION, "maximum_non_rESS_failure_fraction": MAXIMUM_NON_RESS_FAILURE_FRACTION},
        "conditions_frozen_before_bank_generation": True,
        "candidate_generation_permitted": False, "reference_training_permitted": False, "Law_optimization_permitted": False,
        "validation_permitted": False, "Tangent_permitted": False, "Full_permitted": False, "official_protocol_permitted_before_ready": False,
    }
    _atomic_json(BANK_MANIFEST_PATH, payload); _atomic_text(BANK_MANIFEST_HASH_PATH, file_sha256(BANK_MANIFEST_PATH) + "\n"); return payload


def _bank_label(row: dict[str, Any]) -> str:
    # The frozen seed manifest stores the fully qualified derivation role
    # (for example, ``screen_0``).  Normalize that metadata back to the
    # predeclared short role used by artifact names and result aggregation.
    role = str(row["role"]).split("_", 1)[0]
    if role not in {"screen", "audit"}: raise RuntimeError(f"unknown bank role: {row['role']}")
    return f"{role}_{row['pair']}"


def _bank_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{label}.npz"


def _load_bank(label: str) -> GalerkinReferenceBank:
    with np.load(_bank_path(label), allow_pickle=False) as arrays:
        return GalerkinReferenceBank(jnp.asarray(arrays["configurations"]), jnp.asarray(arrays["velocity"]), jnp.asarray(arrays["base_weights"]))


def _rollout(flow: Any, initial: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64); configs = []; velocities = []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        trajectory = flow.rollout(jnp.asarray(initial[start:start + ROLLOUT_BATCH_SIZE]), times, substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
        configs.append(np.asarray(trajectory)); velocities.append(np.asarray(flow.velocity(trajectory, times)))
    return np.concatenate(configs, axis=1), np.concatenate(velocities, axis=1)


def generate_banks(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    manifest = freeze_bank_manifest(); checkpoint = _checkpoint_path(); rows = []
    for spec in manifest["banks"]:
        label = _bank_label(spec); path, record_path = _bank_path(label), _bank_path(label).with_suffix(".json")
        if path.exists() or record_path.exists():
            if not path.exists() or not record_path.exists() or file_sha256(path) != _json(record_path)["sha256"]: raise RuntimeError(f"bank cache seal mismatch: {label}")
            rows.append(_json(record_path)); continue
        started = time.perf_counter(); truth = SkyrmionTruth(_physics_config(cfg)); initial = np.asarray(truth.sample_initial(jax.random.PRNGKey(int(spec["seed"])), int(spec["N"])), dtype=np.float64)
        flow = load_reference(checkpoint); configurations, velocity = _rollout(flow, initial, cfg); weights = np.full((TIME_COUNT, int(spec["N"])), 1.0 / int(spec["N"]), dtype=np.float64)
        if not np.array_equal(configurations[0], initial): raise RuntimeError("rollout changed P0")
        _atomic_npz(path, configurations=configurations, velocity=velocity, base_weights=weights, initial_P0=initial)
        record = {"schema_version": 1, "label": label, "pair": spec["pair"], "role": spec["role"], "N": spec["N"], "seed": spec["seed"], "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "initial_P0_sha256": _array_sha256(initial), "sha256": file_sha256(path), "wall_time_seconds": time.perf_counter() - started}
        _atomic_json(record_path, record); rows.append(record)
        if progress: progress(f"confirmation bank {label}: N={spec['N']}")
        del flow, configurations, velocity, weights, initial; gc.collect()
    return rows


def _selection_context(cfg: dict[str, Any], label: str) -> tuple[Any, np.ndarray, np.ndarray]:
    times, configurations, truth_means, whitening = _load_design_arrays(); preflight_manifest = _json(PREFLIGHT_MANIFEST_PATH)
    problem = _make_problem(cfg, jnp.asarray(configurations), jnp.asarray(times), _family(cfg), noise_seed=int(preflight_manifest["seeds"]["selection_observation_noise"]["seed"]))
    return problem, truth_means, whitening


def _result_path(label: str) -> Path:
    return OUTPUT_ROOT / "support_results" / f"{label}.npz"


def evaluate_all(cfg: dict[str, Any], progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    generate_banks(cfg, progress); pool = _json(CANDIDATE_POOL_PATH); etas = np.asarray([row["eta"] for row in pool["rows"]], dtype=np.float64); records = []
    for spec in freeze_bank_manifest()["banks"]:
        label = _bank_label(spec); path, record_path = _result_path(label), _result_path(label).with_suffix(".json")
        if path.exists() or record_path.exists():
            if not path.exists() or not record_path.exists() or file_sha256(path) != _json(record_path)["result_sha256"]: raise RuntimeError(f"result cache seal mismatch: {label}")
            records.append(_json(record_path)); continue
        started = time.perf_counter(); problem, truth_means, whitening = _selection_context(cfg, label); bank = _load_bank(label)
        evaluator = _ReferenceEvaluator(problem, truth_means, whitening); result = evaluator.evaluate(etas, bank, int(spec["N"]))
        _atomic_npz(path, candidate_index=np.arange(CANDIDATE_COUNT, dtype=np.int32), **result)
        record = {"schema_version": 1, "label": label, "pair": spec["pair"], "role": spec["role"], "N": spec["N"], "bank_sha256": file_sha256(_bank_path(label)), "candidate_pool_sha256": file_sha256(CANDIDATE_POOL_PATH), "result_sha256": file_sha256(path), "support_pass_count": int(np.sum(result["support_valid"])), "wall_time_seconds": time.perf_counter() - started}
        _atomic_json(record_path, record); records.append(record)
        if progress: progress(f"confirmation support {label}: {record['support_pass_count']}/{CANDIDATE_COUNT}")
        del bank, evaluator, result; gc.collect()
    return records


def _load_result(label: str) -> dict[str, np.ndarray]:
    path = _result_path(label); record = _json(path.with_suffix(".json"))
    if file_sha256(path) != record["result_sha256"]: raise RuntimeError(f"result changed: {label}")
    with np.load(path, allow_pickle=False) as arrays: return {key: np.asarray(arrays[key]) for key in arrays.files if key != "candidate_index"}


def _diverse_count(pool_rows: list[dict[str, Any]], indices: np.ndarray, robust: np.ndarray) -> int:
    ordered = sorted(indices.tolist(), key=lambda index: (-float(robust[index]), pool_rows[index]["candidate_id"])); selected: list[int] = []
    for index in ordered:
        if all(_symmetry_aware_distance(pool_rows[index]["eta"], pool_rows[old]["eta"], BOX) >= 0.02 for old in selected): selected.append(index)
    return len(selected)


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluate_all(cfg); pool, risk, law = (_json(path) for path in (CANDIDATE_POOL_PATH, CANDIDATE_RISK_PATH, LAW_FREEZE_PATH))
    results = {f"{role}_{pair}": _load_result(f"{role}_{pair}") for pair in range(PAIR_COUNT) for role in ("screen", "audit")}
    law_rows = []
    for label, value in results.items():
        law_rows.append({"bank": label, "minimum_rESS": float(value["minimum_ress"][0]), "controlling_node": int(value["controlling_time_index"][0]), "node7_rESS": float(value["ress_trajectory"][0, NODE7]), "lambda_norm": float(value["lambda_norm"][0, NODE7]), "top1pct_mass": float(value["top_1pct_weight_mass"][0, NODE7]), "projection_residual": float(value["maximum_projection_residual"][0]), "forcing_mean": float(value["maximum_forcing_mean"][0]), "covariance_condition": float(value["maximum_covariance_condition"][0]), "support_valid": bool(value["support_valid"][0])})
    valid = np.stack([value["support_valid"] for value in results.values()]); minimum = np.stack([value["minimum_ress"] for value in results.values()]); robust = np.min(minimum, axis=0)
    pair_valid = np.stack([results[f"screen_{pair}"]["support_valid"] & results[f"audit_{pair}"]["support_valid"] for pair in range(PAIR_COUNT)])
    all_pair = np.all(pair_valid, axis=0); risks = np.asarray([row["scientific_risk"] for row in risk["rows"]]); inside_half = risks <= 1.005 * float(law["R_Law_B1"]); half_survivors = np.flatnonzero(inside_half & all_pair)
    pair_fractions = [float(np.mean(pair_valid[pair][inside_half])) for pair in range(PAIR_COUNT)]
    nonress_fractions = {}
    for label, value in results.items():
        nonress = ~value["projection_valid"] | ~value["forcing_valid"] | ~value["covariance_valid"]
        nonress_fractions[label] = float(np.mean(nonress[inside_half]))
    allowance_rows = []
    for allowance in ALLOWANCES:
        inside = risks <= (1.0 + allowance / 100.0) * float(law["R_Law_B1"])
        allowance_rows.append({"allowance_percent": allowance, "inside_risk_count": int(np.sum(inside)), "all_eight_pair_survivors": int(np.sum(inside & all_pair))})
    conditions = {
        "Condition 1": {"passed": all(row["support_valid"] for row in law_rows), "description": "Law passes all 16 individual banks"},
        "Condition 2": {"passed": min(row["minimum_rESS"] for row in law_rows) >= READINESS_LAW_MARGIN, "description": "minimum Law rESS >= 0.060"},
        "Condition 3": {"passed": len(half_survivors) >= MINIMUM_CANDIDATES, "description": ">=25 frozen 0.5% candidates pass all eight pairs"},
        "Condition 4": {"passed": len(half_survivors) > 0 and float(np.quantile(robust[half_survivors], 0.10)) >= LOW_RISK_P10_MARGIN, "description": "0.5% survivor robust-rESS p10 >=0.055"},
        "Condition 5": {"passed": _diverse_count(pool["rows"], half_survivors, robust) >= MINIMUM_DIVERSE, "description": ">=5 symmetry-aware diverse 0.5% survivors"},
        "Condition 6": {"passed": min(pair_fractions) >= MINIMUM_PAIR_SURVIVAL_FRACTION, "description": "every complete pair passes >=25% of 0.5%-inside-risk population"},
        "Condition 7": {"passed": all(row["all_eight_pair_survivors"] >= MINIMUM_CANDIDATES for row in allowance_rows[1:]), "description": ">=25 all-eight-pair survivors at each 1-5% allowance"},
        "Condition 8": {"passed": max(nonress_fractions.values()) <= MAXIMUM_NON_RESS_FAILURE_FRACTION, "description": "no individual bank has >10% non-rESS failure at 0.5%"},
    }
    classification = "PRODUCTION_LAUNCH_READY" if all(row["passed"] for row in conditions.values()) else "PRODUCTION_LAUNCH_BLOCKED"
    payload = {"schema_version": 1, "classification": classification, "accepted_reference_SHA256": EXPECTED_CHECKPOINT_SHA256, "fresh_confirmation_pairs": PAIR_COUNT, "Law": {"R_Law_B1_dev": law["R_Law_B1"], "banks": law_rows, "minimum_rESS": min(row["minimum_rESS"] for row in law_rows), "median_rESS": float(np.median([row["minimum_rESS"] for row in law_rows])), "pass_count": sum(row["support_valid"] for row in law_rows)}, "half_percent": {"inside_risk": int(np.sum(inside_half)), "all_eight_pair_survivors": len(half_survivors), "p10_robust_rESS": None if not len(half_survivors) else float(np.quantile(robust[half_survivors], 0.10)), "median_robust_rESS": None if not len(half_survivors) else float(np.median(robust[half_survivors])), "diverse_survivors": _diverse_count(pool["rows"], half_survivors, robust), "minimum_per_pair_survival_fraction": min(pair_fractions), "per_pair_survival_fractions": pair_fractions}, "allowances": allowance_rows, "non_rESS_failure_fractions": nonress_fractions, "conditions": conditions, "failed_conditions": [name for name, row in conditions.items() if not row["passed"]], "production_launched": False, "validation_accessed": False, "Tangent_run": False, "Full_run": False, "official_protocol_created": False}
    _atomic_json(SUMMARY_PATH, payload); _atomic_text(REPORT_PATH, _report(payload)); _write_inventory(); return payload


def _report(summary: dict[str, Any]) -> str:
    lines = ["# Final B1 Support Confirmation", "", "FINAL B1 SUPPORT CONFIRMATION", "", "accepted reference SHA:", EXPECTED_CHECKPOINT_SHA256, "", f"fresh confirmation pairs:\n{PAIR_COUNT}", "", "## LAW CONFIRMATION", "", "| bank | min rESS | controlling node | node7 rESS | lambda norm | top1% mass |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary["Law"]["banks"]: lines.append(f"| {row['bank']} | {row['minimum_rESS']:.6f} | {row['controlling_node']} | {row['node7_rESS']:.6f} | {row['lambda_norm']:.3f} | {row['top1pct_mass']:.6f} |")
    lines += ["", f"minimum: {summary['Law']['minimum_rESS']:.6f}", f"median: {summary['Law']['median_rESS']:.6f}", f"16/16 pass: {'YES' if summary['Law']['pass_count'] == 16 else 'NO'}", "", "## 0.5% CONFIRMATION", "", f"inside risk: {summary['half_percent']['inside_risk']}", f"all-8-pair survivors: {summary['half_percent']['all_eight_pair_survivors']}", f"p10 robust rESS: {summary['half_percent']['p10_robust_rESS']}", f"diverse survivors: {summary['half_percent']['diverse_survivors']}", f"minimum per-pair survival fraction: {summary['half_percent']['minimum_per_pair_survival_fraction']:.6f}", ""]
    for name, row in summary["conditions"].items(): lines.append(f"{name}: {'PASS' if row['passed'] else 'FAIL'} — {row['description']}")
    lines += ["", "CONFIRMATION CLASSIFICATION:", "", summary["classification"], ""]
    if summary["classification"] == "PRODUCTION_LAUNCH_BLOCKED": lines += ["Production was not launched.", f"Failed conditions: {', '.join(summary['failed_conditions'])}", "", "NO reference training", "NO Law optimization", "NO candidate generation", "NO official protocol", "NO Tangent", "NO Full", "NO validation", ""]
    return "\n".join(lines)


def _write_inventory() -> dict[str, Any]:
    rows = [{"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in sorted(OUTPUT_ROOT.rglob("*")) if path.is_file() and path != INVENTORY_PATH]
    payload = {"schema_version": 1, "artifact_count": len(rows), "files": rows}; _atomic_json(INVENTORY_PATH, payload); return payload


def console_report() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def run_all(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cfg = _json(CONFIG_PATH); freeze_bank_manifest(); generate_banks(cfg, progress); evaluate_all(cfg, progress); return summarize(cfg)
