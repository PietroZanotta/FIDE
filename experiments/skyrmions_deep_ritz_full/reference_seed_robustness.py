"""Endpoint-only reference-checkpoint seed robustness study.

This development-only module crosses the immutable production reference and
six same-protocol endpoint-only retrainings with prospectively frozen common
reference-bank seeds.  It deliberately has no candidate generator, geometry
optimizer, validation loader, Tangent/Full construction, eigensolve, or Ritz
solver.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.projection import EmpiricalIProjector

from .domain import SkyrmionTruth, minimum_image
from .full_gradient import reconstruct_moments
from .galerkin_only_data import (
    GalerkinReferenceBank,
    _family,
    _make_problem,
    _physics_config,
)
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v3_common import ROOT, file_sha256
from .reference import (
    ReferenceTrainingConfig,
    equivariant_velocity,
    load_reference,
    save_reference,
    train_endpoint_reference,
)
from .risk import many_body_features, whitening_from_truth


VERSION = "skyrmion_galerkin_dev_reference_seed_robustness_v1"
SEED_NAMESPACE = VERSION
GLOBAL_SEED = 20260826
OUTPUT_ROOT = ROOT / "outputs" / VERSION
N_CONVERGENCE_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_ress_n_convergence_v1"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
MANIFEST_HASH_PATH = OUTPUT_ROOT / "experiment_manifest.sha256"
PANEL_SOURCE_PATH = N_CONVERGENCE_ROOT / "candidate_panel.json"
PANEL_PATH = OUTPUT_ROOT / "candidate_panel_reference.json"
BRIDGE_EVAL_PATH = OUTPUT_ROOT / "bridge_eval.json"
PHASE_A_BANK_MANIFEST_PATH = OUTPUT_ROOT / "phase_a_bank_manifest.json"
PHASE_A_SUMMARY_PATH = OUTPUT_ROOT / "phase_a_summary.json"
RANKING_PATH = OUTPUT_ROOT / "phase_a_reference_ranking.json"
PHASE_B_BANK_MANIFEST_PATH = OUTPUT_ROOT / "phase_b_bank_manifest.json"
PHASE_B_SUMMARY_PATH = OUTPUT_ROOT / "phase_b_summary.json"
VARIANCE_PATH = OUTPUT_ROOT / "variance_decomposition.json"
RISK_SHIFT_PATH = OUTPUT_ROOT / "risk_shift_summary.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

BASELINE_CHECKPOINT_PATH = ARTIFACT_DIR / "reference.npz"
BASELINE_MANIFEST_PATH = ARTIFACT_DIR / "reference_manifest.json"
TRUTH_BANKS_PATH = ARTIFACT_DIR / "truth_banks.npz"
ISOLATED_MANIFEST_PATH = ARTIFACT_DIR / "isolated_artifact_manifest.json"
CONFIG_PATH = ROOT / "config.json"

EXPECTED_N_CONVERGENCE_HASHES = {
    "candidate_panel.json": "f2a6437899383072634c4c2c596e35c49275b6fb47ee9b05a3425d35a81a0189",
    "summary.json": "295975b93e8b4db11007de3f0adf1afd5c1b06df7efea194d9251cbefd0c3a2a",
    "inventory.json": "2d72c01654873c0ea5bb98500b227851e71d8e657e98fbeb9a20c32765ab0c4f",
}
EXPECTED_BASELINE_CHECKPOINT_SHA256 = "f0aa333a38cbd7f99748c83e4a13335e40b81e85385f333dd81b597dfcfad3a9"
EXPECTED_BASELINE_MANIFEST_SHA256 = "0063e7ed9f752d32a8b3d1d8fbb63e6828b5a6f2a902e40d6ef51ef7bf3fbee7"
EXPECTED_CONFIG_SHA256 = "0497ec5203f8010c0d530f2d7b196b900d1cfdf55569f5c59e436f44f59e2369"
EXPECTED_PANEL_SHA256 = EXPECTED_N_CONVERGENCE_HASHES["candidate_panel.json"]

MODEL_LABELS = tuple(f"model_{index:02d}" for index in range(7))
NEW_MODEL_LABELS = MODEL_LABELS[1:]
BASELINE_LABEL = MODEL_LABELS[0]
PHASE_A_BANK_COUNT = 8
PHASE_B_BANK_COUNT = 4
PHASE_A_N = 32768
PHASE_B_N = 65536
TIME_COUNT = 13
NODE7 = 7
PANEL_COUNT = 64
HIGH_PASS_COUNT = 55
CONTROL_COUNT = 8
CANDIDATE_BATCH_SIZE = 8
ROLLOUT_BATCH_SIZE = 2048
MINIMUM_RESS = 0.05
BRIDGE_EVAL_EXAMPLES = 131072
BRIDGE_EVAL_BATCH_SIZE = 512

RESULT_TRAJECTORIES = (
    "ress_trajectory",
    "lambda_norm",
    "maximum_normalized_weight",
    "top_1pct_weight_mass",
    "empirical_D2",
    "covariance_condition_trajectory",
    "projection_residual_trajectory",
    "forcing_mean_trajectory",
)


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
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
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


def _atomic_text(path: Path, value: str) -> None:
    path = _inside(path)
    encoded = value.encode()
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
        raise RuntimeError(f"refusing to overwrite sealed result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_reference(path: Path, flow: Any) -> None:
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        save_reference(temporary, flow)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _distribution(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        raise ValueError("cannot summarize an empty array")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _code_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in (
            "reference_seed_robustness.py",
            "reference_seed_robustness_run.py",
            "test_reference_seed_robustness.py",
        )
    }


def _derive_seed(role: str) -> dict[str, Any]:
    text = f"{GLOBAL_SEED}:{SEED_NAMESPACE}:{role}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "role": role,
        "namespace": SEED_NAMESPACE,
        "derivation_text": text,
        "sha256": digest,
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def _load_endpoint_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Read only the endpoint members of the sealed production archive."""

    with np.load(TRUTH_BANKS_PATH, allow_pickle=False) as arrays:
        endpoint0 = np.asarray(arrays["endpoint0"], dtype=np.float64)
        endpoint1 = np.asarray(arrays["endpoint1"], dtype=np.float64)
    return endpoint0, endpoint1


def _load_design_context(cfg: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray]:
    """Read selection/design truth only; no held-out member is requested."""

    with np.load(TRUTH_BANKS_PATH, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=np.float64)
        design = np.asarray(arrays["design"], dtype=np.float64)
    expected_times = np.linspace(0.0, 1.0, TIME_COUNT, dtype=np.float64)
    if not np.array_equal(times, expected_times):
        raise RuntimeError("sealed design time grid changed")
    family = _family(cfg)
    noise_seed = int(cfg["seed"]) + int(cfg["banks"]["seed_offsets"]["observation"])
    problem = _make_problem(
        cfg,
        jnp.asarray(design, dtype=jnp.float64),
        jnp.asarray(times, dtype=jnp.float64),
        family,
        noise_seed=noise_seed,
    )
    truth_features = many_body_features(jnp.asarray(design), tuple(cfg["physics"]["box"]))
    truth_means = np.asarray(jnp.mean(truth_features, axis=1), dtype=np.float64)
    whitening = np.asarray(whitening_from_truth(truth_features), dtype=np.float64)
    return problem, truth_means, whitening


def verify_and_freeze_sources() -> dict[str, Any]:
    observed_n = {
        name: file_sha256(N_CONVERGENCE_ROOT / name)
        for name in EXPECTED_N_CONVERGENCE_HASHES
    }
    if observed_n != EXPECTED_N_CONVERGENCE_HASHES:
        raise RuntimeError("immutable N-convergence artifacts differ from expected hashes")
    fixed = {
        "baseline_checkpoint": file_sha256(BASELINE_CHECKPOINT_PATH),
        "baseline_reference_manifest": file_sha256(BASELINE_MANIFEST_PATH),
        "architecture_and_training_config": file_sha256(CONFIG_PATH),
        "endpoint_and_design_archive": file_sha256(TRUTH_BANKS_PATH),
        "isolated_artifact_manifest": file_sha256(ISOLATED_MANIFEST_PATH),
        "reference_training_bridge_and_rollout_source": file_sha256(ROOT / "reference.py"),
        "reference_construction_source": file_sha256(ROOT / "experiment.py"),
        "production_workflow_source": file_sha256(ROOT / "workflow.py"),
        "selection_problem_loader_source": file_sha256(ROOT / "galerkin_only_data.py"),
        "projection_diagnostics_source": file_sha256(ROOT / "ress_n_convergence.py"),
        "scientific_risk_source": file_sha256(ROOT / "risk.py"),
        "measurement_reconstruction_and_projected_risk_source": file_sha256(ROOT / "full_gradient.py"),
    }
    if fixed["baseline_checkpoint"] != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("production reference checkpoint hash changed")
    if fixed["baseline_reference_manifest"] != EXPECTED_BASELINE_MANIFEST_SHA256:
        raise RuntimeError("production reference manifest hash changed")
    if fixed["architecture_and_training_config"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("authoritative architecture/training config hash changed")
    panel = json.loads(PANEL_SOURCE_PATH.read_text(encoding="utf-8"))
    roles = Counter(row["panel_role"] for row in panel["rows"])
    if (
        panel.get("candidate_count") != PANEL_COUNT
        or roles["law"] != 1
        or roles["high_pass_ge24_of_32"] != HIGH_PASS_COUNT
        or sum(value for key, value in roles.items() if "control" in key) != CONTROL_COUNT
    ):
        raise RuntimeError("frozen diagnostic panel membership changed")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "n_convergence_expected_and_observed_hashes": observed_n,
        "immutable_source_hashes": fixed,
        "analysis_source_hashes": _code_hashes(),
        "candidate_panel_roles": dict(roles),
        "baseline_checkpoint_immutable": True,
        "intermediate_truth_used_for_training": False,
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    if SOURCE_SEAL_PATH.exists():
        cached = json.loads(SOURCE_SEAL_PATH.read_text(encoding="utf-8"))
        if cached != payload:
            raise RuntimeError("sealed source inputs or analysis code changed")
        return {**cached, "cache_hit": True}
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return {**payload, "cache_hit": False}


def freeze_experiment_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    verify_and_freeze_sources()
    if PANEL_PATH.exists():
        if file_sha256(PANEL_PATH) != EXPECTED_PANEL_SHA256:
            raise RuntimeError("candidate-panel reference copy changed")
    else:
        PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PANEL_SOURCE_PATH, PANEL_PATH)
        if file_sha256(PANEL_PATH) != EXPECTED_PANEL_SHA256:
            raise RuntimeError("candidate-panel copy failed hash verification")

    if MANIFEST_PATH.exists() or MANIFEST_HASH_PATH.exists():
        if not MANIFEST_PATH.exists() or not MANIFEST_HASH_PATH.exists():
            raise RuntimeError("incomplete experiment-manifest seal")
        observed = file_sha256(MANIFEST_PATH)
        expected = MANIFEST_HASH_PATH.read_text(encoding="utf-8").strip()
        if observed != expected:
            raise RuntimeError("experiment manifest changed after freezing")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest["source_seal_sha256"] != file_sha256(SOURCE_SEAL_PATH):
            raise RuntimeError("experiment manifest source seal changed")
        return {**manifest, "cache_hit": True}

    original = ReferenceTrainingConfig(**cfg["reference_training"])
    training_seeds = [_derive_seed(f"reference_training_{index:02d}") for index in range(1, 7)]
    phase_a_seeds = [_derive_seed(f"phase_a_bank_{index:02d}") for index in range(PHASE_A_BANK_COUNT)]
    phase_b_seeds = [_derive_seed(f"phase_b_bank_{index:02d}") for index in range(PHASE_B_BANK_COUNT)]
    bridge_seed = _derive_seed("deterministic_bridge_evaluation")
    all_values = [row["seed"] for row in training_seeds + phase_a_seeds + phase_b_seeds + [bridge_seed]]
    if len(all_values) != len(set(all_values)):
        raise RuntimeError("prospective seed collision")
    fresh_configs = {}
    baseline_config = asdict(original)
    for label, record in zip(NEW_MODEL_LABELS, training_seeds, strict=True):
        current = asdict(replace(original, seed=record["seed"]))
        differences = {key for key in current if current[key] != baseline_config[key]}
        if differences != {"seed"}:
            raise RuntimeError("new reference config differs by more than training seed")
        fresh_configs[label] = current
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "seed_namespace": SEED_NAMESPACE,
        "development_only": True,
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "candidate_panel_sha256": file_sha256(PANEL_PATH),
        "baseline": {
            "label": BASELINE_LABEL,
            "checkpoint_path": str(BASELINE_CHECKPOINT_PATH),
            "checkpoint_sha256": file_sha256(BASELINE_CHECKPOINT_PATH),
            "training_config": baseline_config,
        },
        "new_training_seeds": training_seeds,
        "new_training_configs": fresh_configs,
        "phase_a": {"N": PHASE_A_N, "common_bank_seeds": phase_a_seeds},
        "phase_b": {"N": PHASE_B_N, "common_bank_seeds": phase_b_seeds},
        "bridge_evaluation": {
            "seed": bridge_seed,
            "examples": BRIDGE_EVAL_EXAMPLES,
            "batch_size": BRIDGE_EVAL_BATCH_SIZE,
            "endpoint_only": True,
        },
        "fixed_constants": {
            "models": list(MODEL_LABELS),
            "new_model_count": 6,
            "candidate_count": PANEL_COUNT,
            "high_pass_count": HIGH_PASS_COUNT,
            "control_count": CONTROL_COUNT,
            "time_nodes": TIME_COUNT,
            "reference_substeps": int(cfg["banks"]["reference_substeps"]),
            "minimum_rESS": MINIMUM_RESS,
            "projection_tolerance": float(cfg["forcing"]["projection_tolerance"]),
            "forcing_mean_tolerance": float(cfg["forcing"]["forcing_mean_tolerance"]),
            "maximum_covariance_condition": float(cfg["forcing"]["max_covariance_condition"]),
        },
        "ranking_rule": [
            "highest pooled median node-7 rESS over the 55 high-pass candidates and 8 Phase-A banks",
            "higher Law median node-7 rESS",
            "lower median node-7 top-1% projected-weight mass",
            "lower deterministic bridge-evaluation CFM MSE",
            "lexicographic model label only if all scientific tie-break values are exactly equal",
        ],
        "phase_b_advance_count": 2,
        "interpretation_thresholds": {
            "clear_phase_a_median_node7_delta": 0.003,
            "clear_phase_a_positive_fraction": 0.65,
            "persistent_phase_b_median_node7_delta": 0.003,
            "persistent_phase_b_positive_fraction": 0.65,
            "maximum_nondegraded_cfm_mse_ratio": 1.05,
            "maximum_median_risk_increase_shift_percentage_points": 0.5,
            "material_between_model_sd": 0.0025,
            "training_quality_instability_mse_ratio": 1.10,
            "training_quality_tracking_spearman": -0.60,
        },
        "measurement_targets_fixed_across_models": True,
        "risk_inputs_fixed": {
            "design_truth": True,
            "nine_collective_features": True,
            "whitening": True,
            "time_weights": True,
        },
        "intermediate_truth_training_permitted": False,
        "validation_access_permitted": False,
        "candidate_generation_permitted": False,
        "geometry_optimization_permitted": False,
        "tangent_full_galerkin_eigensolve_deep_ritz_permitted": False,
        "official_outputs_permitted": False,
        "frozen_before_training": True,
    }
    _atomic_json(MANIFEST_PATH, payload)
    _atomic_text(MANIFEST_HASH_PATH, file_sha256(MANIFEST_PATH) + "\n")
    return {**payload, "cache_hit": False}


def _checkpoint_path(label: str) -> Path:
    if label == BASELINE_LABEL:
        return BASELINE_CHECKPOINT_PATH
    return OUTPUT_ROOT / "reference_models" / label / "checkpoint.npz"


def _model_record_path(label: str) -> Path:
    return OUTPUT_ROOT / "reference_models" / ("baseline.json" if label == BASELINE_LABEL else f"{label}/training.json")


def _parameter_count(flow: Any) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(flow.params)))


def train_models(cfg: dict[str, Any], progress: Callable[..., None] | None = None) -> dict[str, Any]:
    manifest = freeze_experiment_manifest(cfg)
    endpoint0, endpoint1 = _load_endpoint_arrays()
    baseline_flow = load_reference(BASELINE_CHECKPOINT_PATH)
    baseline_training = dict((baseline_flow.metadata or {}).get("training", {}))
    if baseline_training != manifest["baseline"]["training_config"]:
        raise RuntimeError("baseline checkpoint metadata differs from frozen training config")
    baseline_record = {
        "schema_version": 1,
        "label": BASELINE_LABEL,
        "status": "existing_immutable_baseline",
        "checkpoint_path": str(BASELINE_CHECKPOINT_PATH),
        "checkpoint_sha256": file_sha256(BASELINE_CHECKPOINT_PATH),
        "training_config": baseline_training,
        "parameter_count": _parameter_count(baseline_flow),
        "installed": True,
        "overwritten": False,
    }
    _atomic_json(_model_record_path(BASELINE_LABEL), baseline_record)
    rows = [baseline_record]
    for label in NEW_MODEL_LABELS:
        checkpoint, record_path = _checkpoint_path(label), _model_record_path(label)
        if checkpoint.exists() or record_path.exists():
            if not checkpoint.exists() or not record_path.exists():
                raise RuntimeError(f"incomplete sealed model cache: {label}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["checkpoint_sha256"] != file_sha256(checkpoint):
                raise RuntimeError(f"sealed checkpoint changed: {label}")
            if record["experiment_manifest_sha256"] != file_sha256(MANIFEST_PATH):
                raise RuntimeError(f"training manifest changed: {label}")
            rows.append({**record, "cache_hit": True})
            if progress:
                progress("train", label, True, float(record["wall_time_seconds"]))
            continue
        training = ReferenceTrainingConfig(**manifest["new_training_configs"][label])
        started = time.perf_counter()
        try:
            flow, history = train_endpoint_reference(
                endpoint0, endpoint1, training, box=tuple(cfg["physics"]["box"])
            )
            elapsed = time.perf_counter() - started
            _atomic_reference(checkpoint, flow)
            status = "complete"
            failure = None
        except Exception as exc:
            elapsed = time.perf_counter() - started
            status = "failed"
            failure = {"type": type(exc).__name__, "message": str(exc)}
            history = []
            flow = None
        record = {
            "schema_version": 1,
            "label": label,
            "status": status,
            "failure": failure,
            "training_seed": int(training.seed),
            "training_config": asdict(training),
            "config_sha256": _payload_sha256(asdict(training)),
            "experiment_manifest_sha256": file_sha256(MANIFEST_PATH),
            "endpoint_archive_sha256": file_sha256(TRUTH_BANKS_PATH),
            "checkpoint_path": str(checkpoint.relative_to(OUTPUT_ROOT)) if status == "complete" else None,
            "checkpoint_sha256": file_sha256(checkpoint) if status == "complete" else None,
            "final_training_loss": float(history[-1]["loss"]) if history else None,
            "training_loss_trajectory_summary": {
                "logged_points": len(history),
                "first_loss": float(history[0]["loss"]) if history else None,
                "minimum_logged_loss": float(min(row["loss"] for row in history)) if history else None,
                "maximum_logged_loss": float(max(row["loss"] for row in history)) if history else None,
                "final_loss": float(history[-1]["loss"]) if history else None,
            },
            "history": history,
            "wall_time_seconds": elapsed,
            "parameter_count": _parameter_count(flow) if flow is not None else None,
            "training_steps_completed": int(history[-1]["step"]) if history else 0,
            "endpoint_only": True,
            "intermediate_truth_used": False,
            "installed": False,
        }
        _atomic_json(record_path, record)
        if progress:
            progress("train", label, False, elapsed)
        if status != "complete":
            raise RuntimeError(f"reference training failed and was sealed: {label}: {failure}")
        rows.append(record)
    if file_sha256(BASELINE_CHECKPOINT_PATH) != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("baseline checkpoint changed during retraining")
    return {"models": rows, "cache_hit": all(row.get("cache_hit", False) for row in rows[1:])}


def evaluate_bridge_quality(cfg: dict[str, Any], progress: Callable[..., None] | None = None) -> dict[str, Any]:
    train_models(cfg, progress=progress)
    if BRIDGE_EVAL_PATH.exists():
        payload = json.loads(BRIDGE_EVAL_PATH.read_text(encoding="utf-8"))
        if payload["experiment_manifest_sha256"] != file_sha256(MANIFEST_PATH):
            raise RuntimeError("bridge evaluation manifest changed")
        return {**payload, "cache_hit": True}
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    endpoint0, endpoint1 = _load_endpoint_arrays()
    seed = int(manifest["bridge_evaluation"]["seed"]["seed"])
    count = int(manifest["bridge_evaluation"]["examples"])
    batch_size = int(manifest["bridge_evaluation"]["batch_size"])
    keys = jax.random.split(jax.random.PRNGKey(seed), count // batch_size)
    box = tuple(cfg["physics"]["box"])
    x0_all, x1_all = jnp.asarray(endpoint0), jnp.asarray(endpoint1)
    box_array = jnp.asarray(box, dtype=jnp.float64)

    @jax.jit
    def make_batch(key: jax.Array):
        kt, k0, k1, kz = jax.random.split(key, 4)
        idx0 = jax.random.randint(k0, (batch_size,), 0, x0_all.shape[0])
        idx1 = jax.random.randint(k1, (batch_size,), 0, x1_all.shape[0])
        x0, x1 = x0_all[idx0], x1_all[idx1]
        t = jax.random.uniform(kt, (batch_size,), dtype=jnp.float64)
        displacement = minimum_image(x1 - x0, box_array)
        noise = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        gamma = 0.01 * jnp.sin(jnp.pi * t)[:, None, None]
        gamma_dot = 0.01 * jnp.pi * jnp.cos(jnp.pi * t)[:, None, None]
        xt = jnp.mod(x0 + t[:, None, None] * displacement + gamma * noise, box_array)
        target = displacement + gamma_dot * noise
        return t, xt, target

    batches = [tuple(np.asarray(item) for item in make_batch(key)) for key in keys]
    input_hash = hashlib.sha256()
    for batch in batches:
        for value in batch:
            input_hash.update(np.ascontiguousarray(value).tobytes())
    model_rows = []
    for label in MODEL_LABELS:
        flow = load_reference(_checkpoint_path(label))

        @jax.jit
        def errors(t: jax.Array, x: jax.Array, target: jax.Array) -> jax.Array:
            predicted = equivariant_velocity(flow.params, t, x, box=box)
            return jnp.sum((predicted - target) ** 2, axis=(-2, -1))

        started = time.perf_counter()
        squared = np.concatenate([
            np.asarray(errors(jnp.asarray(t), jnp.asarray(x), jnp.asarray(target)))
            for t, x, target in batches
        ])
        row = {
            "label": label,
            "checkpoint_sha256": file_sha256(_checkpoint_path(label)),
            "CFM_velocity_MSE": float(np.mean(squared)),
            "median_per_example_squared_error": float(np.median(squared)),
            "p90_squared_error": float(np.quantile(squared, 0.90)),
            "p99_squared_error": float(np.quantile(squared, 0.99)),
            "example_count": int(squared.size),
            "wall_time_seconds": time.perf_counter() - started,
        }
        model_rows.append(row)
        if progress:
            progress("bridge-eval", label, False, row["wall_time_seconds"])
    payload = {
        "schema_version": 1,
        "experiment_manifest_sha256": file_sha256(MANIFEST_PATH),
        "seed": seed,
        "evaluation_input_sha256": input_hash.hexdigest(),
        "endpoint_only": True,
        "models": model_rows,
    }
    _atomic_json(BRIDGE_EVAL_PATH, payload)
    return payload


def _initial_states(cfg: dict[str, Any], seed: int, N: int) -> np.ndarray:
    truth = SkyrmionTruth(_physics_config(cfg))
    return np.asarray(
        truth.sample_initial(jax.random.PRNGKey(int(seed)), int(N)), dtype=np.float64
    )


def _generate_model_bank(cfg: dict[str, Any], label: str, initial: np.ndarray) -> GalerkinReferenceBank:
    flow = load_reference(_checkpoint_path(label))
    times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
    configurations, velocities = [], []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        stop = min(start + ROLLOUT_BATCH_SIZE, len(initial))
        trajectory = flow.rollout(
            jnp.asarray(initial[start:stop]),
            times,
            substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
        )
        configurations.append(np.asarray(trajectory))
        velocities.append(np.asarray(flow.velocity(trajectory, times)))
    x = np.concatenate(configurations, axis=1)
    v = np.concatenate(velocities, axis=1)
    weights = np.full(x.shape[:2], 1.0 / len(initial), dtype=np.float64)
    if not np.array_equal(x[0], initial):
        raise RuntimeError("rollout did not preserve common initial states")
    return GalerkinReferenceBank(jnp.asarray(x), jnp.asarray(v), jnp.asarray(weights))


class _ReferenceEvaluator:
    """Authoritative projection/forcing diagnostics plus fixed-input risk."""

    def __init__(self, problem: Any, truth_means: np.ndarray, whitening: np.ndarray):
        self.problem = problem
        self.truth_means = jnp.asarray(truth_means, dtype=jnp.float64)
        self.whitening = jnp.asarray(whitening, dtype=jnp.float64)
        self.projector = EmpiricalIProjector(
            problem.projection_config, trajectory_backend=problem.projection_backend
        )
        self.preprocess = jax.jit(jax.vmap(
            lambda eta, configurations, velocity: (
                reconstruct_moments(eta, problem).values,
                reconstruct_moments(eta, problem).derivatives,
                problem.family.features(configurations, eta),
                problem.family.jvp(configurations, velocity, eta),
            ),
            in_axes=(0, None, None),
        ))
        self.postprocessors: dict[int, Any] = {}

    def _postprocessor(self, N: int):
        if N in self.postprocessors:
            return self.postprocessors[N]
        top_count = max(1, int(math.ceil(0.01 * N)))
        problem, truth_means, whitening = self.problem, self.truth_means, self.whitening

        @jax.jit
        def postprocess(weights, lam, moments, covariance, residual, ess, features, advective, derivatives, reference_features):
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
            eigenvalues = jnp.linalg.eigvalsh(regularized)
            condition = eigenvalues[..., -1] / jnp.maximum(eigenvalues[..., 0], 1e-300)
            top_mass = jnp.sum(jax.lax.top_k(weights, top_count)[0], axis=-1)
            predicted = jnp.einsum("btn,tnf->btf", weights, reference_features)
            error = predicted - truth_means[None, ...]
            risk_by_time = jnp.einsum("bti,ij,btj->bt", error, whitening, error)
            risk = jnp.sum(problem.time_weights[None, :] * risk_by_time, axis=1)
            return (
                ess,
                jnp.linalg.norm(lam, axis=-1),
                jnp.max(weights, axis=-1),
                top_mass,
                -jnp.log(jnp.maximum(ess, 1e-300)),
                condition,
                jnp.linalg.norm(residual, axis=-1),
                forcing_mean,
                risk,
            )

        self.postprocessors[N] = postprocess
        return postprocess

    def evaluate(self, etas: np.ndarray, bank: GalerkinReferenceBank, N: int) -> dict[str, np.ndarray]:
        configurations = bank.configurations[:, :N]
        velocity = bank.velocity[:, :N]
        base = bank.base_weights[:, :N]
        base = base / jnp.sum(base, axis=1, keepdims=True)
        reference_features = many_body_features(configurations, self.problem.box)
        collected: dict[str, list[np.ndarray]] = {name: [] for name in RESULT_TRAJECTORIES}
        risks: list[np.ndarray] = []
        postprocess = self._postprocessor(N)
        for start in range(0, len(etas), CANDIDATE_BATCH_SIZE):
            stop = min(start + CANDIDATE_BATCH_SIZE, len(etas))
            batch = etas[start:stop]
            actual = len(batch)
            if actual < CANDIDATE_BATCH_SIZE:
                batch = np.concatenate((batch, np.repeat(batch[-1:], CANDIDATE_BATCH_SIZE - actual, axis=0)))
            targets, derivatives, features, advective = self.preprocess(
                jnp.asarray(batch), configurations, velocity
            )
            projected = self.projector.project_candidate_trajectories(features, base, targets)
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
            for name, value in zip(RESULT_TRAJECTORIES, numpy_values[:-1], strict=True):
                collected[name].append(value)
            risks.append(numpy_values[-1])
        result = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
        result["scientific_risk"] = np.concatenate(risks, axis=0)
        result["minimum_ress"] = np.min(result["ress_trajectory"], axis=1)
        result["controlling_time_index"] = np.argmin(result["ress_trajectory"], axis=1).astype(np.int16)
        result["maximum_projection_residual"] = np.max(result["projection_residual_trajectory"], axis=1)
        result["maximum_forcing_mean"] = np.max(result["forcing_mean_trajectory"], axis=1)
        result["maximum_covariance_condition"] = np.max(result["covariance_condition_trajectory"], axis=1)
        result["projection_valid"] = result["maximum_projection_residual"] <= float(self.problem.forcing_config.projection_tolerance)
        result["ress_valid"] = result["minimum_ress"] >= MINIMUM_RESS
        result["forcing_valid"] = result["maximum_forcing_mean"] <= float(self.problem.forcing_config.forcing_mean_tolerance)
        result["covariance_valid"] = result["maximum_covariance_condition"] <= float(self.problem.forcing_config.max_covariance_condition)
        result["support_valid"] = result["projection_valid"] & result["ress_valid"] & result["forcing_valid"] & result["covariance_valid"]
        return result


def _phase_result_path(phase: str, label: str, bank_index: int) -> Path:
    return OUTPUT_ROOT / f"phase_{phase}_results" / label / f"bank_{bank_index:02d}.npz"


def _phase_result_record_path(phase: str, label: str, bank_index: int) -> Path:
    return OUTPUT_ROOT / f"phase_{phase}_results" / label / f"bank_{bank_index:02d}.json"


def _load_result(phase: str, label: str, bank_index: int) -> dict[str, np.ndarray]:
    path = _phase_result_path(phase, label, bank_index)
    record_path = _phase_result_record_path(phase, label, bank_index)
    if not path.exists() or not record_path.exists():
        raise RuntimeError(f"missing phase-{phase} result {label}/bank {bank_index}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["result_sha256"] != file_sha256(path):
        raise RuntimeError(f"sealed phase-{phase} result changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def _run_phase(
    cfg: dict[str, Any],
    *,
    phase: str,
    labels: tuple[str, ...],
    N: int,
    seed_records: list[dict[str, Any]],
    manifest_path: Path,
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    evaluate_bridge_quality(cfg, progress=progress)
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    etas = np.asarray([row["eta"] for row in panel["rows"]], dtype=np.float64)
    problem, truth_means, whitening = _load_design_context(cfg)
    evaluator = _ReferenceEvaluator(problem, truth_means, whitening)
    bank_rows = []
    for bank_index, seed_record in enumerate(seed_records):
        seed = int(seed_record["seed"])
        initial = _initial_states(cfg, seed, N)
        initial_hash = _array_sha256(initial)
        model_rows = []
        for label in labels:
            result_path = _phase_result_path(phase, label, bank_index)
            record_path = _phase_result_record_path(phase, label, bank_index)
            if result_path.exists() or record_path.exists():
                if not result_path.exists() or not record_path.exists():
                    raise RuntimeError(f"incomplete phase-{phase} result cache: {label}/{bank_index}")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                checks = (
                    record["result_sha256"] == file_sha256(result_path),
                    record["checkpoint_sha256"] == file_sha256(_checkpoint_path(label)),
                    record["candidate_panel_sha256"] == file_sha256(PANEL_PATH),
                    record["initial_state_sha256"] == initial_hash,
                    record["seed"] == seed,
                    record["N"] == N,
                )
                if not all(checks):
                    raise RuntimeError(f"phase-{phase} result seal mismatch: {label}/{bank_index}")
                model_rows.append({**record, "cache_hit": True})
                if progress:
                    progress(f"phase-{phase}", f"{label}/bank_{bank_index:02d}", True, float(record["wall_time_seconds"]))
                continue
            started = time.perf_counter()
            bank = _generate_model_bank(cfg, label, initial)
            if _array_sha256(np.asarray(bank.configurations[0])) != initial_hash:
                raise RuntimeError("common initial-state hash differs across checkpoints")
            result = evaluator.evaluate(etas, bank, N)
            elapsed = time.perf_counter() - started
            _atomic_npz(
                result_path,
                candidate_index=np.arange(PANEL_COUNT, dtype=np.int16),
                measurement_target_sha256=np.asarray(_measurement_target_hash(problem, etas)),
                **result,
            )
            record = {
                "schema_version": 1,
                "phase": phase,
                "label": label,
                "bank_index": bank_index,
                "seed": seed,
                "seed_sha256": seed_record["sha256"],
                "N": N,
                "checkpoint_sha256": file_sha256(_checkpoint_path(label)),
                "candidate_panel_sha256": file_sha256(PANEL_PATH),
                "experiment_manifest_sha256": file_sha256(MANIFEST_PATH),
                "initial_state_sha256": initial_hash,
                "measurement_target_sha256": _measurement_target_hash(problem, etas),
                "result_path": str(result_path.relative_to(OUTPUT_ROOT)),
                "result_sha256": file_sha256(result_path),
                "wall_time_seconds": elapsed,
                "minimum_ress": _distribution(result["minimum_ress"]),
                "node7_ress": _distribution(result["ress_trajectory"][:, NODE7]),
                "support_pass_count": int(np.sum(result["support_valid"])),
            }
            _atomic_json(record_path, record)
            model_rows.append(record)
            if progress:
                progress(f"phase-{phase}", f"{label}/bank_{bank_index:02d}", False, elapsed)
            del bank, result
        if len({row["initial_state_sha256"] for row in model_rows}) != 1:
            raise RuntimeError("model results do not share a common initial-state hash")
        if len({row["measurement_target_sha256"] for row in model_rows}) != 1:
            raise RuntimeError("measurement targets differ across reference models")
        bank_rows.append({
            "bank_index": bank_index,
            "seed_record": seed_record,
            "N": N,
            "initial_state_sha256": initial_hash,
            "model_results": model_rows,
        })
        del initial
    manifest = {
        "schema_version": 1,
        "phase": phase,
        "experiment_manifest_sha256": file_sha256(MANIFEST_PATH),
        "candidate_panel_sha256": file_sha256(PANEL_PATH),
        "N": N,
        "model_labels": list(labels),
        "bank_count": len(seed_records),
        "common_random_numbers": True,
        "banks": bank_rows,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _measurement_target_hash(problem: Any, etas: np.ndarray) -> str:
    reconstruct = jax.jit(jax.vmap(lambda eta: reconstruct_moments(eta, problem)))
    values = reconstruct(jnp.asarray(etas, dtype=jnp.float64))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(values.values)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(values.derivatives)).tobytes())
    return digest.hexdigest()


def run_phase_a(cfg: dict[str, Any], progress: Callable[..., None] | None = None) -> dict[str, Any]:
    manifest = freeze_experiment_manifest(cfg)
    return _run_phase(
        cfg,
        phase="a",
        labels=MODEL_LABELS,
        N=PHASE_A_N,
        seed_records=manifest["phase_a"]["common_bank_seeds"],
        manifest_path=PHASE_A_BANK_MANIFEST_PATH,
        progress=progress,
    )


def _panel_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    law = np.asarray([i for i, row in enumerate(panel["rows"]) if row["panel_role"] == "law"], dtype=np.int64)
    high = np.asarray([i for i, row in enumerate(panel["rows"]) if row["panel_role"] == "high_pass_ge24_of_32"], dtype=np.int64)
    controls = np.asarray([i for i, row in enumerate(panel["rows"]) if "control" in row["panel_role"]], dtype=np.int64)
    if (len(law), len(high), len(controls)) != (1, HIGH_PASS_COUNT, CONTROL_COUNT):
        raise RuntimeError("candidate role counts changed")
    return law, high, controls


def _stack_phase(phase: str, label: str, bank_count: int) -> dict[str, np.ndarray]:
    rows = [_load_result(phase, label, index) for index in range(bank_count)]
    return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0] if key not in {"measurement_target_sha256"}}


def _paired_summary(delta: np.ndarray) -> dict[str, float | int]:
    distribution = _distribution(delta)
    distribution["fraction_positive"] = float(np.mean(np.asarray(delta) > 0))
    return distribution


def _model_phase_a_summary(label: str, values: dict[str, np.ndarray], law: int, high: np.ndarray) -> dict[str, Any]:
    law_min = values["minimum_ress"][:, law]
    high_min = values["minimum_ress"][:, high]
    high_node7 = values["ress_trajectory"][:, high, NODE7]
    pass_counts = np.sum(values["support_valid"][:, high], axis=0)
    law_risk = values["scientific_risk"][:, law]
    relative_risk = 100.0 * (values["scientific_risk"][:, high] / law_risk[:, None] - 1.0)
    controlling = Counter(values["controlling_time_index"][:, high].reshape(-1).tolist())
    return {
        "label": label,
        "law": {
            "bank_passes": int(np.sum(values["support_valid"][:, law])),
            "bank_total": PHASE_A_BANK_COUNT,
            "minimum_ress": _distribution(law_min),
            "node7_ress": _distribution(values["ress_trajectory"][:, law, NODE7]),
            "node7_lambda_norm": _distribution(values["lambda_norm"][:, law, NODE7]),
            "node7_top1pct_mass": _distribution(values["top_1pct_weight_mass"][:, law, NODE7]),
            "controlling_node_counts": {str(key): int(value) for key, value in sorted(Counter(values["controlling_time_index"][:, law].tolist()).items())},
            "fixed_law_scientific_risk": _distribution(law_risk),
        },
        "high_pass_panel": {
            "candidate_median_minimum_ress": _distribution(np.median(high_min, axis=0)),
            "candidate_p10_minimum_ress": _distribution(np.quantile(high_min, 0.10, axis=0)),
            "candidate_bank_pass_counts": _distribution(pass_counts),
            "node7_ress": _distribution(high_node7),
            "candidate_node7_medians": _distribution(np.median(high_node7, axis=0)),
            "node7_lambda_norm": _distribution(values["lambda_norm"][:, high, NODE7]),
            "node7_top1pct_mass": _distribution(values["top_1pct_weight_mass"][:, high, NODE7]),
            "relative_scientific_risk_increase_percent": _distribution(relative_risk),
            "within_relative_risk_percent_counts": {
                "0.5": int(np.sum(np.median(relative_risk, axis=0) <= 0.5)),
                "1.0": int(np.sum(np.median(relative_risk, axis=0) <= 1.0)),
                "2.0": int(np.sum(np.median(relative_risk, axis=0) <= 2.0)),
            },
            "candidate_pass_count_thresholds": {
                "at_least_4_of_8": int(np.sum(pass_counts >= 4)),
                "at_least_6_of_8": int(np.sum(pass_counts >= 6)),
                "at_least_7_of_8": int(np.sum(pass_counts >= 7)),
                "8_of_8": int(np.sum(pass_counts == 8)),
            },
            "controlling_node_counts": {str(key): int(value) for key, value in sorted(controlling.items())},
        },
        "gate_diagnostics": {
            "projection_failure_count": int(np.sum(~values["projection_valid"])),
            "rESS_failure_count": int(np.sum(~values["ress_valid"])),
            "forcing_failure_count": int(np.sum(~values["forcing_valid"])),
            "covariance_failure_count": int(np.sum(~values["covariance_valid"])),
        },
    }


def _paired_model_comparison(current: dict[str, np.ndarray], baseline: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    return {
        "minimum_rESS": _paired_summary(current["minimum_ress"][:, indices] - baseline["minimum_ress"][:, indices]),
        "node7_rESS": _paired_summary(current["ress_trajectory"][:, indices, NODE7] - baseline["ress_trajectory"][:, indices, NODE7]),
        "node7_lambda_norm": _paired_summary(current["lambda_norm"][:, indices, NODE7] - baseline["lambda_norm"][:, indices, NODE7]),
        "node7_top1pct_mass": _paired_summary(current["top_1pct_weight_mass"][:, indices, NODE7] - baseline["top_1pct_weight_mass"][:, indices, NODE7]),
        "scientific_risk": _paired_summary(current["scientific_risk"][:, indices] - baseline["scientific_risk"][:, indices]),
    }


def _variance_components(values: np.ndarray) -> tuple[float, float, float]:
    """Two-way descriptive SDs for a [model, bank] table."""

    grand = np.mean(values)
    model_effect = np.mean(values, axis=1) - grand
    bank_effect = np.mean(values, axis=0) - grand
    residual = values - grand - model_effect[:, None] - bank_effect[None, :]
    return float(np.std(model_effect)), float(np.std(bank_effect)), float(np.std(residual))


def summarize_phase_a(cfg: dict[str, Any]) -> dict[str, Any]:
    run_phase_a(cfg)
    if PHASE_A_SUMMARY_PATH.exists() and VARIANCE_PATH.exists() and RISK_SHIFT_PATH.exists():
        return {**json.loads(PHASE_A_SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}
    law_array, high, controls = _panel_indices()
    law = int(law_array[0])
    stacked = {label: _stack_phase("a", label, PHASE_A_BANK_COUNT) for label in MODEL_LABELS}
    models = [_model_phase_a_summary(label, stacked[label], law, high) for label in MODEL_LABELS]
    baseline = stacked[BASELINE_LABEL]
    paired = {}
    for label in NEW_MODEL_LABELS:
        paired[label] = {
            "law": _paired_model_comparison(stacked[label], baseline, law_array),
            "high_pass_panel": _paired_model_comparison(stacked[label], baseline, high),
            "controls": _paired_model_comparison(stacked[label], baseline, controls),
        }
    node7 = np.stack([stacked[label]["ress_trajectory"][:, :, NODE7] for label in MODEL_LABELS], axis=0)
    by_candidate = np.asarray([_variance_components(node7[:, :, index]) for index in range(PANEL_COUNT)])
    variance = {
        "schema_version": 1,
        "method": "descriptive balanced two-way mean-effect decomposition; no population random-effects inference",
        "component_order": ["between_model_SD", "between_bank_SD", "interaction_residual_SD"],
        "law": dict(zip(("between_model_SD", "between_bank_SD", "interaction_residual_SD"), by_candidate[law].tolist(), strict=True)),
        "high_pass_panel": {name: _distribution(by_candidate[high, index]) for index, name in enumerate(("between_model_SD", "between_bank_SD", "interaction_residual_SD"))},
        "controls": {name: _distribution(by_candidate[controls, index]) for index, name in enumerate(("between_model_SD", "between_bank_SD", "interaction_residual_SD"))},
        "per_candidate": [
            {"panel_index": index, "between_model_SD": float(row[0]), "between_bank_SD": float(row[1]), "interaction_residual_SD": float(row[2])}
            for index, row in enumerate(by_candidate)
        ],
    }
    _atomic_json(VARIANCE_PATH, variance)
    risk_shift = {
        "schema_version": 1,
        "fixed_design_truth_and_whitening": True,
        "models": {
            label: {
                "law_risk": _distribution(stacked[label]["scientific_risk"][:, law]),
                "high_panel_relative_risk_increase_percent": _distribution(
                    100.0 * (stacked[label]["scientific_risk"][:, high] / stacked[label]["scientific_risk"][:, law, None] - 1.0)
                ),
            }
            for label in MODEL_LABELS
        },
        "paired_vs_baseline": {label: paired[label]["high_pass_panel"]["scientific_risk"] for label in NEW_MODEL_LABELS},
    }
    _atomic_json(RISK_SHIFT_PATH, risk_shift)
    payload = {
        "schema_version": 1,
        "phase": "A",
        "N": PHASE_A_N,
        "bank_count": PHASE_A_BANK_COUNT,
        "models": models,
        "paired_vs_baseline": paired,
        "variance_decomposition_sha256": file_sha256(VARIANCE_PATH),
        "risk_shift_summary_sha256": file_sha256(RISK_SHIFT_PATH),
    }
    _atomic_json(PHASE_A_SUMMARY_PATH, payload)
    return payload


def rank_phase_a(cfg: dict[str, Any]) -> dict[str, Any]:
    summarize_phase_a(cfg)
    evaluate_bridge_quality(cfg)
    if RANKING_PATH.exists():
        payload = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
        if len(payload["selected_models"]) != 2 or BASELINE_LABEL in payload["selected_models"]:
            raise RuntimeError("sealed Phase-A ranking changed")
        return {**payload, "cache_hit": True}
    _, high, _ = _panel_indices()
    bridge = {row["label"]: row for row in json.loads(BRIDGE_EVAL_PATH.read_text(encoding="utf-8"))["models"]}
    rows = []
    for label in NEW_MODEL_LABELS:
        values = _stack_phase("a", label, PHASE_A_BANK_COUNT)
        rows.append({
            "label": label,
            "primary_pooled_high_panel_node7_rESS_median": float(np.median(values["ress_trajectory"][:, high, NODE7])),
            "tie1_Law_node7_rESS_median": float(np.median(values["ress_trajectory"][:, 0, NODE7])),
            "tie2_high_panel_node7_top1pct_mass_median": float(np.median(values["top_1pct_weight_mass"][:, high, NODE7])),
            "tie3_CFM_velocity_MSE": float(bridge[label]["CFM_velocity_MSE"]),
        })
    ordered = sorted(rows, key=lambda row: (
        -row["primary_pooled_high_panel_node7_rESS_median"],
        -row["tie1_Law_node7_rESS_median"],
        row["tie2_high_panel_node7_top1pct_mass_median"],
        row["tie3_CFM_velocity_MSE"],
        row["label"],
    ))
    payload = {
        "schema_version": 1,
        "phase_a_summary_sha256": file_sha256(PHASE_A_SUMMARY_PATH),
        "bridge_eval_sha256": file_sha256(BRIDGE_EVAL_PATH),
        "ranking_rule": json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["ranking_rule"],
        "ranking": [{**row, "rank": index + 1} for index, row in enumerate(ordered)],
        "selected_models": [ordered[0]["label"], ordered[1]["label"]],
        "baseline_always_advances": BASELINE_LABEL,
        "phase_b_model_labels": [BASELINE_LABEL, ordered[0]["label"], ordered[1]["label"]],
        "selection_frozen_before_phase_b": True,
    }
    _atomic_json(RANKING_PATH, payload)
    return payload


def run_phase_b(cfg: dict[str, Any], progress: Callable[..., None] | None = None) -> dict[str, Any]:
    ranking = rank_phase_a(cfg)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return _run_phase(
        cfg,
        phase="b",
        labels=tuple(ranking["phase_b_model_labels"]),
        N=PHASE_B_N,
        seed_records=manifest["phase_b"]["common_bank_seeds"],
        manifest_path=PHASE_B_BANK_MANIFEST_PATH,
        progress=progress,
    )


def summarize_phase_b(cfg: dict[str, Any]) -> dict[str, Any]:
    run_phase_b(cfg)
    if PHASE_B_SUMMARY_PATH.exists():
        return {**json.loads(PHASE_B_SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
    labels = ranking["phase_b_model_labels"]
    law_array, high, _ = _panel_indices()
    law = int(law_array[0])
    stacked = {label: _stack_phase("b", label, PHASE_B_BANK_COUNT) for label in labels}
    baseline = stacked[BASELINE_LABEL]
    models = []
    paired = {}
    for label in labels:
        values = stacked[label]
        high_min = values["minimum_ress"][:, high]
        pass_counts = np.sum(values["support_valid"][:, high], axis=0)
        law_risk = values["scientific_risk"][:, law]
        relative_risk = 100.0 * (values["scientific_risk"][:, high] / law_risk[:, None] - 1.0)
        models.append({
            "label": label,
            "law": {
                "rESS_each_bank": values["minimum_ress"][:, law].tolist(),
                "minimum_ress": _distribution(values["minimum_ress"][:, law]),
                "node7_rESS": _distribution(values["ress_trajectory"][:, law, NODE7]),
                "node7_lambda_norm": _distribution(values["lambda_norm"][:, law, NODE7]),
                "node7_top1pct_mass": _distribution(values["top_1pct_weight_mass"][:, law, NODE7]),
                "scientific_risk": _distribution(law_risk),
            },
            "high_pass_panel": {
                "candidate_median_minimum_rESS": _distribution(np.median(high_min, axis=0)),
                "candidate_p10_minimum_rESS": _distribution(np.quantile(high_min, 0.10, axis=0)),
                "node7_rESS": _distribution(values["ress_trajectory"][:, high, NODE7]),
                "candidate_bank_pass_fraction": float(np.mean(values["support_valid"][:, high])),
                "candidates_passing_4_of_4": int(np.sum(pass_counts == 4)),
                "candidates_passing_at_least_3_of_4": int(np.sum(pass_counts >= 3)),
                "candidates_passing_at_least_2_of_4": int(np.sum(pass_counts >= 2)),
                "relative_scientific_risk_increase_percent": _distribution(relative_risk),
            },
        })
        if label != BASELINE_LABEL:
            paired[label] = {
                "high_pass_panel": _paired_model_comparison(values, baseline, high),
                "law": _paired_model_comparison(values, baseline, law_array),
            }
    payload = {
        "schema_version": 1,
        "phase": "B",
        "N": PHASE_B_N,
        "bank_count": PHASE_B_BANK_COUNT,
        "models": models,
        "paired_vs_baseline": paired,
        "phase_a_ranking_sha256": file_sha256(RANKING_PATH),
        "phase_b_cannot_change_selection": True,
    }
    _atomic_json(PHASE_B_SUMMARY_PATH, payload)
    return payload


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


def _interpret(cfg: dict[str, Any], phase_a: dict[str, Any], phase_b: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    thresholds = manifest["interpretation_thresholds"]
    bridge = {row["label"]: row for row in json.loads(BRIDGE_EVAL_PATH.read_text(encoding="utf-8"))["models"]}
    phase_a_models = {row["label"]: row for row in phase_a["models"]}
    phase_b_models = {row["label"]: row for row in phase_b["models"]}
    selected = json.loads(RANKING_PATH.read_text(encoding="utf-8"))["selected_models"]
    baseline_risk = phase_a_models[BASELINE_LABEL]["high_pass_panel"]["relative_scientific_risk_increase_percent"]["median"]
    checks = {}
    confirmed = []
    for label in selected:
        a = phase_a["paired_vs_baseline"][label]["high_pass_panel"]["node7_rESS"]
        b = phase_b["paired_vs_baseline"][label]["high_pass_panel"]["node7_rESS"]
        mse_ratio = bridge[label]["CFM_velocity_MSE"] / bridge[BASELINE_LABEL]["CFM_velocity_MSE"]
        risk_shift = phase_a_models[label]["high_pass_panel"]["relative_scientific_risk_increase_percent"]["median"] - baseline_risk
        good = (
            a["median"] >= thresholds["clear_phase_a_median_node7_delta"]
            and a["fraction_positive"] >= thresholds["clear_phase_a_positive_fraction"]
            and b["median"] >= thresholds["persistent_phase_b_median_node7_delta"]
            and b["fraction_positive"] >= thresholds["persistent_phase_b_positive_fraction"]
            and mse_ratio <= thresholds["maximum_nondegraded_cfm_mse_ratio"]
            and risk_shift <= thresholds["maximum_median_risk_increase_shift_percentage_points"]
        )
        checks[label] = {"phase_a": a, "phase_b": b, "CFM_MSE_ratio": mse_ratio, "median_risk_shift_percentage_points": risk_shift, "confirmed": bool(good)}
        if good:
            confirmed.append(label)
    losses = np.asarray([bridge[label]["CFM_velocity_MSE"] for label in MODEL_LABELS])
    supports = np.asarray([phase_a_models[label]["high_pass_panel"]["node7_rESS"]["median"] for label in MODEL_LABELS])
    correlation = _spearman(losses, supports)
    worst_ratio = float(np.max(losses) / np.min(losses))
    variance = json.loads(VARIANCE_PATH.read_text(encoding="utf-8"))["high_pass_panel"]
    between = float(variance["between_model_SD"]["median"])
    bank = float(variance["between_bank_SD"]["median"])
    material_variation = between >= thresholds["material_between_model_sd"]
    quality_instability = (
        worst_ratio >= thresholds["training_quality_instability_mse_ratio"]
        and correlation <= thresholds["training_quality_tracking_spearman"]
    )
    if confirmed:
        label = "CHECKPOINT_SEED_EFFECT_CONFIRMED"
        next_step = "Run a separate prospectively frozen study of seed-selection criteria versus robust multi-reference treatment; do not install a checkpoint from this development result."
    elif quality_instability:
        label = "TRAINING_QUALITY_INSTABILITY"
        next_step = "Repair and qualify endpoint-only reference optimization stability before changing the bridge or selecting a checkpoint."
    elif material_variation:
        label = "REFERENCE_TRAINING_SEED_SENSITIVITY"
        next_step = "Investigate a prospectively specified robust multi-reference treatment rather than choosing the luckiest seed."
    elif between < thresholds["material_between_model_sd"] and not confirmed:
        label = "CHECKPOINT_ROBUST_STRUCTURAL_MISMATCH"
        next_step = "Study prospectively specified endpoint-only bridge/reference constructions while continuing to withhold intermediate truth."
    else:
        label = "MIXED_REFERENCE_EFFECT"
        next_step = "Separate training-quality, model-seed, and endpoint-bridge effects in a new prospectively frozen development study."
    diagnostics = {
        "per_selected_model_confirmation_checks": checks,
        "bridge_CFM_MSE_to_phase_a_support_spearman": correlation,
        "bridge_CFM_MSE_worst_to_best_ratio": worst_ratio,
        "high_panel_between_model_SD_median": between,
        "high_panel_between_bank_SD_median": bank,
        "material_model_variation": bool(material_variation),
        "training_quality_instability": bool(quality_instability),
        "confirmed_models": confirmed,
    }
    return label, next_step, diagnostics


def _report_text(summary: dict[str, Any]) -> str:
    bridge = {row["label"]: row for row in json.loads(BRIDGE_EVAL_PATH.read_text(encoding="utf-8"))["models"]}
    phase_a = {row["label"]: row for row in summary["phase_a"]["models"]}
    phase_b = {row["label"]: row for row in summary["phase_b"]["models"]}
    selected = summary["phase_a_selected_models"]
    lines = [
        "# Endpoint-Only Reference-Checkpoint Robustness Study",
        "",
        "SOURCE VERIFIED",
        "",
        f"baseline reference: `{BASELINE_CHECKPOINT_PATH}`",
        f"baseline checkpoint SHA-256: `{file_sha256(BASELINE_CHECKPOINT_PATH)}`",
        "new endpoint-only references trained: 6",
        "intermediate truth used: NO",
        "validation accessed: NO",
        "",
        "## Reference training quality",
        "",
        "| model | CFM eval loss | training status |",
        "|---|---:|---|",
    ]
    for label in MODEL_LABELS:
        record = json.loads(_model_record_path(label).read_text(encoding="utf-8"))
        lines.append(f"| {label} | {bridge[label]['CFM_velocity_MSE']:.8f} | {record['status']} |")
    lines.extend(["", "## Phase A — Law", "", "| model | rESS passes/8 | median min-rESS | node7 median | lambda norm | top1% mass | fixed-Law risk |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label in MODEL_LABELS:
        row = phase_a[label]["law"]
        lines.append(f"| {label} | {row['bank_passes']}/8 | {row['minimum_ress']['median']:.6f} | {row['node7_ress']['median']:.6f} | {row['node7_lambda_norm']['median']:.3f} | {row['node7_top1pct_mass']['median']:.6f} | {row['fixed_law_scientific_risk']['median']:.6f} |")
    lines.extend(["", "## Phase A — high-pass panel", "", "| model | median node7 rESS | candidates >=6/8 | candidates 8/8 | median top1% mass | median relative risk increase |", "|---|---:|---:|---:|---:|---:|"])
    for label in MODEL_LABELS:
        row = phase_a[label]["high_pass_panel"]
        lines.append(f"| {label} | {row['node7_ress']['median']:.6f} | {row['candidate_pass_count_thresholds']['at_least_6_of_8']} | {row['candidate_pass_count_thresholds']['8_of_8']} | {row['node7_top1pct_mass']['median']:.6f} | {row['relative_scientific_risk_increase_percent']['median']:.4f}% |")
    variance = summary["variance_decomposition"]["high_pass_panel"]
    lines.extend([
        "", "## Reference/bank variance", "",
        f"- Between-model SD (panel median): {variance['between_model_SD']['median']:.6f}",
        f"- Between-bank SD (panel median): {variance['between_bank_SD']['median']:.6f}",
        f"- Interaction SD (panel median): {variance['interaction_residual_SD']['median']:.6f}",
        "", "## Phase-A selected models for confirmation", "",
        f"1. {selected[0]}", f"2. {selected[1]}", "",
        "Selection rule: predeclared node-7 rESS ranking.",
        "", "## Phase B — N=65536 independent confirmation", "",
        "| model | Law median rESS | panel median node7 rESS | panel candidate-bank pass fraction | median paired node7 delta vs baseline |",
        "|---|---:|---:|---:|---:|",
    ])
    for label in [BASELINE_LABEL, *selected]:
        row = phase_b[label]
        paired = 0.0 if label == BASELINE_LABEL else summary["phase_b"]["paired_vs_baseline"][label]["high_pass_panel"]["node7_rESS"]["median"]
        lines.append(f"| {label} | {row['law']['minimum_ress']['median']:.6f} | {row['high_pass_panel']['node7_rESS']['median']:.6f} | {row['high_pass_panel']['candidate_bank_pass_fraction']:.4f} | {paired:+.6f} |")
    lines.extend([
        "", "## Development interpretation", "",
        summary["development_interpretation"], "",
        "Recommended next scientific step:", "",
        summary["recommended_next_scientific_step"], "",
        "NO intermediate-truth training", "",
        "NO Tangent", "",
        "NO Full", "",
        "NO validation", "",
        "NO official reference replacement", "",
        "NO official protocol created", "",
    ])
    return "\n".join(lines)


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    phase_a = summarize_phase_a(cfg)
    ranking = rank_phase_a(cfg)
    phase_b = summarize_phase_b(cfg)
    if SUMMARY_PATH.exists() and REPORT_PATH.exists() and INVENTORY_PATH.exists():
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        for row in inventory["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"sealed final artifact changed: {row['path']}")
        return {**json.loads(SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}
    interpretation, next_step, diagnostics = _interpret(cfg, phase_a, phase_b)
    bridge = json.loads(BRIDGE_EVAL_PATH.read_text(encoding="utf-8"))
    law_node7 = np.asarray([next(row for row in phase_a["models"] if row["label"] == label)["law"]["node7_ress"]["median"] for label in MODEL_LABELS])
    panel_node7 = np.asarray([next(row for row in phase_a["models"] if row["label"] == label)["high_pass_panel"]["node7_ress"]["median"] for label in MODEL_LABELS])
    losses = np.asarray([next(row for row in bridge["models"] if row["label"] == label)["CFM_velocity_MSE"] for label in MODEL_LABELS])
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_verified": True,
        "baseline_reference": {"path": str(BASELINE_CHECKPOINT_PATH), "sha256": file_sha256(BASELINE_CHECKPOINT_PATH)},
        "new_endpoint_only_references_trained": 6,
        "intermediate_truth_used": False,
        "validation_accessed": False,
        "phase_a": phase_a,
        "phase_a_selected_models": ranking["selected_models"],
        "phase_b": phase_b,
        "variance_decomposition": json.loads(VARIANCE_PATH.read_text(encoding="utf-8")),
        "reference_quality_associations": {
            "CFM_MSE_vs_Law_node7_rESS_spearman": _spearman(losses, law_node7),
            "CFM_MSE_vs_high_panel_node7_rESS_spearman": _spearman(losses, panel_node7),
            "exploratory_only": True,
            "statistical_significance_claimed": False,
        },
        "interpretation_diagnostics": diagnostics,
        "development_interpretation": interpretation,
        "recommended_next_scientific_step": next_step,
        "tangent_run": False,
        "full_run": False,
        "galerkin_constructed": False,
        "eigensolve_run": False,
        "deep_ritz_run": False,
        "official_reference_replaced": False,
        "official_protocol_created": False,
    }
    _atomic_json(SUMMARY_PATH, payload)
    _atomic_text(REPORT_PATH, _report_text(payload))
    artifacts = sorted(
        [path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path != INVENTORY_PATH],
        key=lambda path: str(path.relative_to(OUTPUT_ROOT)),
    )
    inventory = {
        "schema_version": 1,
        "version": VERSION,
        "artifact_count": len(artifacts),
        "artifacts": [
            {"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in artifacts
        ],
    }
    _atomic_json(INVENTORY_PATH, inventory)
    return payload


def run(cfg: dict[str, Any], progress: Callable[..., None] | None = None) -> dict[str, Any]:
    freeze_experiment_manifest(cfg)
    train_models(cfg, progress=progress)
    evaluate_bridge_quality(cfg, progress=progress)
    run_phase_a(cfg, progress=progress)
    summarize_phase_a(cfg)
    rank_phase_a(cfg)
    run_phase_b(cfg, progress=progress)
    return summarize(cfg)


__all__ = [
    "BASELINE_CHECKPOINT_PATH", "BASELINE_LABEL", "BRIDGE_EVAL_PATH", "EXPECTED_N_CONVERGENCE_HASHES",
    "EXPECTED_PANEL_SHA256", "INVENTORY_PATH", "MANIFEST_HASH_PATH", "MANIFEST_PATH", "MINIMUM_RESS",
    "MODEL_LABELS", "NEW_MODEL_LABELS", "OUTPUT_ROOT", "PANEL_PATH", "PHASE_A_BANK_MANIFEST_PATH",
    "PHASE_A_N", "PHASE_A_SUMMARY_PATH", "PHASE_B_BANK_MANIFEST_PATH", "PHASE_B_N",
    "PHASE_B_SUMMARY_PATH", "RANKING_PATH", "REPORT_PATH", "SOURCE_SEAL_PATH", "SUMMARY_PATH",
    "TRUTH_BANKS_PATH", "VARIANCE_PATH", "evaluate_bridge_quality", "freeze_experiment_manifest",
    "rank_phase_a", "run", "run_phase_a", "run_phase_b", "summarize", "summarize_phase_a",
    "summarize_phase_b", "train_models", "verify_and_freeze_sources",
]
