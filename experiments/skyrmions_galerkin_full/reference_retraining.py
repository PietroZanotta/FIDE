"""Three-model reference retraining diagnostic on regenerated physical data.

This development-only experiment leaves the production checkpoint untouched.
It regenerates one shared endpoint-training ensemble and one disjoint physical
evaluation trajectory, trains three independent reference flows, and compares
them with the production flow on fixed CRNs and matched development banks.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .domain import SkyrmionTruth, minimum_image
from .galerkin_only_data import _physics_config, load_selection_galerkin_data
from .pareto_v2_common import ARTIFACT_DIR
from .pareto_v3_common import ROOT, file_sha256
from .reference import (
    ReferenceTrainingConfig,
    equivariant_velocity,
    load_reference,
    save_reference,
    train_endpoint_reference,
)
from .ress_n_convergence import _NestedEvaluator


VERSION = "skyrmion_galerkin_dev_reference_retraining_ensemble_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
TRAIN_DATA_PATH = OUTPUT_ROOT / "regenerated_training_endpoints.npz"
EVAL_DATA_PATH = OUTPUT_ROOT / "regenerated_evaluation_truth.npz"
DATA_INVENTORY_PATH = OUTPUT_ROOT / "data_inventory.json"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

ORIGINAL_CHECKPOINT_PATH = ARTIFACT_DIR / "reference.npz"
ORIGINAL_MANIFEST_PATH = ARTIFACT_DIR / "reference_manifest.json"
PANEL_PATH = ROOT / "outputs" / "skyrmion_galerkin_dev_ress_n_convergence_v1" / "candidate_panel.json"

EXPECTED_SOURCE_HASHES = {
    "reference.npz": "f0aa333a38cbd7f99748c83e4a13335e40b81e85385f333dd81b597dfcfad3a9",
    "reference_manifest.json": "0063e7ed9f752d32a8b3d1d8fbb63e6828b5a6f2a902e40d6ef51ef7bf3fbee7",
    "candidate_panel.json": "f2a6437899383072634c4c2c596e35c49275b6fb47ee9b05a3425d35a81a0189",
    "config.json": "0497ec5203f8010c0d530f2d7b196b900d1cfdf55569f5c59e436f44f59e2369",
}

MODEL_LABELS = ("retrained_0", "retrained_1", "retrained_2")
ALL_FLOW_LABELS = ("original",) + MODEL_LABELS
MODEL_SEEDS = (2026082611, 2026082612, 2026082613)
TRAIN_DATA_SEED = 2026082601
EVAL_DATA_SEED = 2026082602
BANK_INITIAL_SEED = 2026082603
FIXED_CRN_SEED = 2026082604
TRAIN_ENDPOINT_SAMPLES = 12_000
EVAL_TRUTH_SAMPLES = 6_000
BANK_SAMPLES = 65_536
EVALUATION_BATCH_COUNT = 256
EVALUATION_BATCH_SIZE = 512
ANALYZED_NODES = (6, 7, 8)
MINIMUM_RESS = 0.05


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


def _atomic_npz(path: Path, *, compressed: bool, **arrays: Any) -> None:
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


def _atomic_reference(path: Path, flow: Any) -> None:
    path = _inside(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        save_reference(temporary, flow)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _distribution(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _code_hashes() -> dict[str, str]:
    return {
        name: file_sha256(ROOT / name)
        for name in (
            "reference_retraining.py",
            "reference_retraining_run.py",
            "test_reference_retraining.py",
        )
    }


def verify_and_freeze_sources() -> dict[str, Any]:
    observed = {
        "reference.npz": file_sha256(ORIGINAL_CHECKPOINT_PATH),
        "reference_manifest.json": file_sha256(ORIGINAL_MANIFEST_PATH),
        "candidate_panel.json": file_sha256(PANEL_PATH),
        "config.json": file_sha256(ROOT / "config.json"),
    }
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("reference-retraining source differs")
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    if panel["candidate_count"] != 64 or len(panel["rows"]) != 64:
        raise RuntimeError("frozen 64-candidate panel changed")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_hashes": observed,
        "analysis_source_hashes": _code_hashes(),
        "candidate_count": 64,
        "production_checkpoint_mutated": False,
        "official_protocol_created": False,
        "downstream_solver_run": False,
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    verify_and_freeze_sources()
    if PROTOCOL_PATH.exists():
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if payload["source_seal_sha256"] != file_sha256(SOURCE_SEAL_PATH):
            raise RuntimeError("retraining protocol source seal changed")
        return {**payload, "cache_hit": True}
    original = ReferenceTrainingConfig(**cfg["reference_training"])
    fresh_configs = [asdict(replace(original, seed=seed)) for seed in MODEL_SEEDS]
    for config, seed in zip(fresh_configs, MODEL_SEEDS, strict=True):
        differences = {
            key for key in config if config[key] != asdict(original)[key]
        }
        if differences != {"seed"} or config["seed"] != seed:
            raise RuntimeError("fresh model config differs by more than seed")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "source_seal_sha256": file_sha256(SOURCE_SEAL_PATH),
        "physics": cfg["physics"],
        "original_training_config": asdict(original),
        "fresh_training_configs": dict(zip(MODEL_LABELS, fresh_configs, strict=True)),
        "regenerated_data": {
            "training_endpoint_seed": TRAIN_DATA_SEED,
            "training_endpoint_samples": TRAIN_ENDPOINT_SAMPLES,
            "evaluation_truth_seed": EVAL_DATA_SEED,
            "evaluation_truth_samples": EVAL_TRUTH_SAMPLES,
            "training_and_evaluation_disjoint": True,
            "truth_substeps_per_interval": int(cfg["physics"]["truth_substeps"]),
        },
        "matched_reference_bank": {
            "initial_seed": BANK_INITIAL_SEED,
            "samples": BANK_SAMPLES,
            "time_nodes": int(cfg["physics"]["time_nodes"]),
            "reference_substeps": int(cfg["banks"]["reference_substeps"]),
            "same_initial_configurations_for_all_flows": True,
        },
        "fixed_crn_evaluation": {
            "seed": FIXED_CRN_SEED,
            "batch_count": EVALUATION_BATCH_COUNT,
            "batch_size": EVALUATION_BATCH_SIZE,
            "same_crn_for_all_flows": True,
        },
        "analyzed_rESS_nodes": list(ANALYZED_NODES),
        "rESS_threshold": MINIMUM_RESS,
        "predeclared_interpretation": {
            "model_consistency_spread": 0.01,
            "nondegraded_loss_ratio": 1.05,
            "RETRAINS_CONSISTENTLY_REPAIR_SUPPORT": "all three Law minimum rESS and high-panel median minimum rESS >=0.05, model spreads <=0.01, and physical-fit metrics are nondegraded",
            "RETRAINS_IMPROVE_FIT_BUT_SUPPORT_REMAINS": "all three improve fixed-CRN loss and truth-moment error, but at least one common support statistic remains below 0.05",
            "RETRAINS_REPRODUCE_SUPPORT_PROBLEM": "fresh-model support spreads <=0.01 and all three retain a below-gate common support statistic",
            "REFERENCE_TRAINING_SEED_SENSITIVE": "fresh-model support spreads exceed 0.01 or conclusions differ materially",
        },
        "production_checkpoint_installation_permitted": False,
        "official_work_permitted": False,
        "frozen_before_data_generation": True,
    }
    _atomic_json(PROTOCOL_PATH, payload)
    return payload


def generate_regenerated_data(cfg: dict[str, Any]) -> dict[str, Any]:
    freeze_protocol(cfg)
    if DATA_INVENTORY_PATH.exists():
        payload = json.loads(DATA_INVENTORY_PATH.read_text(encoding="utf-8"))
        if payload["protocol_sha256"] != file_sha256(PROTOCOL_PATH):
            raise RuntimeError("regenerated-data protocol changed")
        for row in payload["artifacts"]:
            if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError(f"regenerated data changed: {row['path']}")
        return {**payload, "cache_hit": True}
    physics = _physics_config(cfg)
    truth = SkyrmionTruth(physics)
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_substeps = int(cfg["physics"]["truth_substeps"])
    started = time.perf_counter()
    training = truth.make_bank(
        seed=TRAIN_DATA_SEED,
        samples=TRAIN_ENDPOINT_SAMPLES,
        times=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        substeps_per_interval=truth_substeps * (len(times) - 1),
    )
    training_seconds = time.perf_counter() - started
    _atomic_npz(
        TRAIN_DATA_PATH,
        compressed=True,
        endpoint0=np.asarray(training.configurations[0]),
        endpoint1=np.asarray(training.configurations[-1]),
        seed=np.asarray(TRAIN_DATA_SEED),
    )
    del training
    started = time.perf_counter()
    evaluation = truth.make_bank(
        seed=EVAL_DATA_SEED,
        samples=EVAL_TRUTH_SAMPLES,
        times=times,
        substeps_per_interval=truth_substeps,
    )
    evaluation_seconds = time.perf_counter() - started
    _atomic_npz(
        EVAL_DATA_PATH,
        compressed=True,
        times=np.asarray(times),
        configurations=np.asarray(evaluation.configurations),
        seed=np.asarray(EVAL_DATA_SEED),
    )
    artifacts = []
    for path in (TRAIN_DATA_PATH, EVAL_DATA_PATH):
        artifacts.append({
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    payload = {
        "schema_version": 1,
        "development_only": True,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "training_endpoint_seed": TRAIN_DATA_SEED,
        "evaluation_truth_seed": EVAL_DATA_SEED,
        "seeds_disjoint": TRAIN_DATA_SEED != EVAL_DATA_SEED,
        "training_endpoint_shape": [TRAIN_ENDPOINT_SAMPLES, 16, 2],
        "evaluation_truth_shape": [13, EVAL_TRUTH_SAMPLES, 16, 2],
        "training_generation_seconds": training_seconds,
        "evaluation_generation_seconds": evaluation_seconds,
        "artifacts": artifacts,
        "official_data_generated": False,
    }
    _atomic_json(DATA_INVENTORY_PATH, payload)
    return payload


def _checkpoint_path(label: str) -> Path:
    return OUTPUT_ROOT / "models" / f"{label}.npz"


def _training_result_path(label: str) -> Path:
    return OUTPUT_ROOT / "models" / f"{label}_training.json"


def train_models(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    generate_regenerated_data(cfg)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    with np.load(TRAIN_DATA_PATH, allow_pickle=False) as arrays:
        endpoint0 = np.asarray(arrays["endpoint0"], dtype=np.float64)
        endpoint1 = np.asarray(arrays["endpoint1"], dtype=np.float64)
    completed = []
    for label in MODEL_LABELS:
        checkpoint, result_path = _checkpoint_path(label), _training_result_path(label)
        if checkpoint.exists() or result_path.exists():
            if not checkpoint.exists() or not result_path.exists():
                raise RuntimeError(f"incomplete retraining cache for {label}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload["checkpoint_sha256"] != file_sha256(checkpoint):
                raise RuntimeError(f"retrained checkpoint changed: {label}")
            completed.append({**payload, "cache_hit": True})
            if progress is not None:
                progress("train", label, True, float(payload["training_seconds"]))
            continue
        training = ReferenceTrainingConfig(**protocol["fresh_training_configs"][label])
        started = time.perf_counter()
        flow, history = train_endpoint_reference(
            endpoint0,
            endpoint1,
            training,
            box=tuple(cfg["physics"]["box"]),
        )
        elapsed = time.perf_counter() - started
        _atomic_reference(checkpoint, flow)
        payload = {
            "schema_version": 1,
            "label": label,
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "data_inventory_sha256": file_sha256(DATA_INVENTORY_PATH),
            "training_config": asdict(training),
            "checkpoint_path": str(checkpoint.relative_to(OUTPUT_ROOT)),
            "checkpoint_sha256": file_sha256(checkpoint),
            "training_seconds": elapsed,
            "training_steps_completed": history[-1]["step"],
            "first_logged_loss": history[0]["loss"],
            "final_logged_loss": history[-1]["loss"],
            "history": history,
            "installed": False,
        }
        _atomic_json(result_path, payload)
        completed.append({**payload, "cache_hit": False})
        if progress is not None:
            progress("train", label, False, elapsed)
    if file_sha256(ORIGINAL_CHECKPOINT_PATH) != EXPECTED_SOURCE_HASHES["reference.npz"]:
        raise RuntimeError("production checkpoint changed during retraining")
    return {"models": completed, "data_inventory_sha256": file_sha256(DATA_INVENTORY_PATH)}


def _flow_path(label: str) -> Path:
    return ORIGINAL_CHECKPOINT_PATH if label == "original" else _checkpoint_path(label)


def _bank_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{label}_N{BANK_SAMPLES}.npz"


def _bank_result_path(label: str) -> Path:
    return OUTPUT_ROOT / "banks" / f"{label}_result.json"


def generate_matched_banks(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    train_models(cfg, progress=progress)
    physics = _physics_config(cfg)
    truth = SkyrmionTruth(physics)
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    initial = truth.sample_initial(jax.random.PRNGKey(BANK_INITIAL_SEED), BANK_SAMPLES)
    rows = []
    for label in ALL_FLOW_LABELS:
        path, result_path = _bank_path(label), _bank_result_path(label)
        if path.exists() or result_path.exists():
            if not path.exists() or not result_path.exists():
                raise RuntimeError(f"incomplete matched-bank cache for {label}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload["bank_sha256"] != file_sha256(path):
                raise RuntimeError(f"matched bank changed: {label}")
            rows.append({**payload, "cache_hit": True})
            if progress is not None:
                progress("bank", label, True, float(payload["generation_seconds"]))
            continue
        flow = load_reference(_flow_path(label))
        configurations, velocities = [], []
        started = time.perf_counter()
        for start in range(0, BANK_SAMPLES, 2048):
            stop = min(start + 2048, BANK_SAMPLES)
            trajectory = flow.rollout(
                initial[start:stop],
                times,
                substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
            )
            configurations.append(np.asarray(trajectory))
            velocities.append(np.asarray(flow.velocity(trajectory, times)))
        x = np.concatenate(configurations, axis=1)
        v = np.concatenate(velocities, axis=1)
        w = np.full((len(times), BANK_SAMPLES), 1.0 / BANK_SAMPLES, dtype=np.float64)
        elapsed = time.perf_counter() - started
        _atomic_npz(
            path,
            compressed=False,
            configurations=x,
            velocity=v,
            base_weights=w,
            label=np.asarray(label),
            flow_checkpoint_sha256=np.asarray(file_sha256(_flow_path(label))),
            initial_seed=np.asarray(BANK_INITIAL_SEED),
        )
        payload = {
            "schema_version": 1,
            "label": label,
            "bank_path": str(path.relative_to(OUTPUT_ROOT)),
            "bank_sha256": file_sha256(path),
            "flow_checkpoint_sha256": file_sha256(_flow_path(label)),
            "initial_seed": BANK_INITIAL_SEED,
            "sample_count": BANK_SAMPLES,
            "generation_seconds": elapsed,
            "matched_initial_configurations": True,
        }
        _atomic_json(result_path, payload)
        rows.append({**payload, "cache_hit": False})
        if progress is not None:
            progress("bank", label, False, elapsed)
    return {"banks": rows, "all_matched_initial_seed": BANK_INITIAL_SEED}


def _fixed_crn_losses(params: Any, endpoint0: np.ndarray, endpoint1: np.ndarray, box: tuple[float, float]) -> np.ndarray:
    x0_all = jnp.asarray(endpoint0, dtype=jnp.float64)
    x1_all = jnp.asarray(endpoint1, dtype=jnp.float64)
    box_array = jnp.asarray(box, dtype=jnp.float64)

    @jax.jit
    def one(key: jax.Array) -> jax.Array:
        kt, k0, k1, kz = jax.random.split(key, 4)
        idx0 = jax.random.randint(k0, (EVALUATION_BATCH_SIZE,), 0, x0_all.shape[0])
        idx1 = jax.random.randint(k1, (EVALUATION_BATCH_SIZE,), 0, x1_all.shape[0])
        x0, x1 = x0_all[idx0], x1_all[idx1]
        t = jax.random.uniform(kt, (EVALUATION_BATCH_SIZE,), dtype=jnp.float64)
        displacement = minimum_image(x1 - x0, box_array)
        noise = jax.random.normal(kz, x0.shape, dtype=jnp.float64)
        gamma = 0.01 * jnp.sin(jnp.pi * t)[:, None, None]
        gamma_dot = 0.01 * jnp.pi * jnp.cos(jnp.pi * t)[:, None, None]
        xt = jnp.mod(x0 + t[:, None, None] * displacement + gamma * noise, box_array)
        target = displacement + gamma_dot * noise
        predicted = equivariant_velocity(params, t, xt, box=box)
        return jnp.mean(jnp.sum((predicted - target) ** 2, axis=(-2, -1)))

    keys = jax.random.split(jax.random.PRNGKey(FIXED_CRN_SEED), EVALUATION_BATCH_COUNT)
    return np.asarray(jax.lax.map(one, keys), dtype=np.float64)


def _truth_moment_errors(problem: Any, etas: np.ndarray, reference: np.ndarray, truth: np.ndarray) -> np.ndarray:
    feature = jax.jit(jax.vmap(
        lambda eta, configurations: problem.family.features(configurations, eta),
        in_axes=(0, None),
    ))
    errors = []
    for start in range(0, len(etas), 8):
        batch = jnp.asarray(etas[start:start + 8], dtype=jnp.float64)
        ref_mean = jnp.mean(feature(batch, jnp.asarray(reference, dtype=jnp.float64)), axis=2)
        truth_mean = jnp.mean(feature(batch, jnp.asarray(truth, dtype=jnp.float64)), axis=2)
        errors.append(np.linalg.norm(np.asarray(ref_mean - truth_mean), axis=-1))
    return np.concatenate(errors, axis=0)


def _evaluation_path(label: str) -> Path:
    return OUTPUT_ROOT / "evaluations" / f"{label}.npz"


def _evaluation_result_path(label: str) -> Path:
    return OUTPUT_ROOT / "evaluations" / f"{label}.json"


def evaluate_models(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    banks = generate_matched_banks(cfg, progress=progress)
    problem = load_selection_galerkin_data(cfg, ARTIFACT_DIR).selection_problem
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    etas = np.asarray([row["eta"] for row in panel["rows"]], dtype=np.float64)
    with np.load(EVAL_DATA_PATH, allow_pickle=False) as arrays:
        evaluation_truth = np.asarray(arrays["configurations"])[list(ANALYZED_NODES)]
        endpoint0 = np.asarray(arrays["configurations"])[0]
        endpoint1 = np.asarray(arrays["configurations"])[-1]
    evaluator = _NestedEvaluator(problem)
    rows = []
    for label in ALL_FLOW_LABELS:
        path, result_path = _evaluation_path(label), _evaluation_result_path(label)
        if path.exists() or result_path.exists():
            if not path.exists() or not result_path.exists():
                raise RuntimeError(f"incomplete model evaluation cache for {label}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload["result_sha256"] != file_sha256(path):
                raise RuntimeError(f"model evaluation changed: {label}")
            rows.append({**payload, "cache_hit": True})
            if progress is not None:
                progress("evaluate", label, True, float(payload["evaluation_seconds"]))
            continue
        started = time.perf_counter()
        with np.load(_bank_path(label), allow_pickle=False) as arrays:
            configurations = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
            velocity = jnp.asarray(arrays["velocity"], dtype=jnp.float64)
            base_weights = jnp.asarray(arrays["base_weights"], dtype=jnp.float64)
        projected = evaluator.evaluate(etas, configurations, velocity, base_weights, BANK_SAMPLES)
        reference_nodes = np.asarray(configurations)[list(ANALYZED_NODES)]
        moment_errors = _truth_moment_errors(problem, etas, reference_nodes, evaluation_truth)
        flow = load_reference(_flow_path(label))
        losses = _fixed_crn_losses(flow.params, endpoint0, endpoint1, tuple(cfg["physics"]["box"]))
        elapsed = time.perf_counter() - started
        _atomic_npz(
            path,
            compressed=True,
            candidate_index=np.arange(64, dtype=np.int16),
            analyzed_nodes=np.asarray(ANALYZED_NODES, dtype=np.int16),
            ress_trajectory=projected["ress_trajectory"],
            minimum_ress=projected["minimum_ress"],
            controlling_time_index=projected["controlling_time_index"],
            support_valid=projected["support_valid"],
            truth_moment_error=moment_errors,
            fixed_crn_loss=losses,
        )
        payload = {
            "schema_version": 1,
            "label": label,
            "flow_checkpoint_sha256": file_sha256(_flow_path(label)),
            "bank_sha256": file_sha256(_bank_path(label)),
            "result_path": str(path.relative_to(OUTPUT_ROOT)),
            "result_sha256": file_sha256(path),
            "evaluation_seconds": elapsed,
            "fixed_crn_loss": _distribution(losses),
            "truth_moment_error_all_candidates_nodes": _distribution(moment_errors),
            "minimum_ress_all_candidates": _distribution(projected["minimum_ress"]),
            "new_reference_bank_count": 1,
        }
        _atomic_json(result_path, payload)
        rows.append({**payload, "cache_hit": False})
        if progress is not None:
            progress("evaluate", label, False, elapsed)
    return {"evaluations": rows, "matched_banks": banks["banks"]}


def _verify_cached_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists() and not INVENTORY_PATH.exists():
        return None
    if not SUMMARY_PATH.exists() or not INVENTORY_PATH.exists():
        raise RuntimeError("incomplete retraining-ensemble summary")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    for row in inventory["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"sealed retraining artifact changed: {row['path']}")
    return {**json.loads(SUMMARY_PATH.read_text(encoding="utf-8")), "cache_hit": True}


def summarize(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluate_models(cfg)
    cached = _verify_cached_summary()
    if cached is not None:
        return cached
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    high = np.asarray([
        index for index, row in enumerate(panel["rows"])
        if row["panel_role"] == "high_pass_ge24_of_32"
    ], dtype=np.int64)
    model_rows = []
    for label in ALL_FLOW_LABELS:
        with np.load(_evaluation_path(label), allow_pickle=False) as arrays:
            values = {name: np.asarray(arrays[name]) for name in arrays.files}
        node_values = values["ress_trajectory"][:, list(ANALYZED_NODES)]
        model_rows.append({
            "label": label,
            "flow_checkpoint_sha256": file_sha256(_flow_path(label)),
            "fixed_crn_loss": _distribution(values["fixed_crn_loss"]),
            "truth_moment_error": {
                str(node): _distribution(values["truth_moment_error"][:, node_index])
                for node_index, node in enumerate(ANALYZED_NODES)
            },
            "law_ress": {
                str(node): float(node_values[0, node_index])
                for node_index, node in enumerate(ANALYZED_NODES)
            },
            "law_minimum_ress_all_13_nodes": float(values["minimum_ress"][0]),
            "high_pass_ress": {
                str(node): _distribution(node_values[high, node_index])
                for node_index, node in enumerate(ANALYZED_NODES)
            },
            "high_pass_minimum_ress_all_13_nodes": _distribution(values["minimum_ress"][high]),
            "high_pass_pass_count": int(np.sum(values["support_valid"][high])),
            "controlling_node_frequency_high_pass": {
                str(node): int(np.sum(values["controlling_time_index"][high] == node))
                for node in range(13)
                if np.any(values["controlling_time_index"][high] == node)
            },
        })
    original = model_rows[0]
    fresh = model_rows[1:]
    law_mins = np.asarray([row["law_minimum_ress_all_13_nodes"] for row in fresh])
    high_medians = np.asarray([row["high_pass_minimum_ress_all_13_nodes"]["median"] for row in fresh])
    loss_means = np.asarray([row["fixed_crn_loss"]["mean"] for row in fresh])
    error_means = np.asarray([
        np.mean([row["truth_moment_error"][str(node)]["mean"] for node in ANALYZED_NODES])
        for row in fresh
    ])
    original_error = np.mean([
        original["truth_moment_error"][str(node)]["mean"] for node in ANALYZED_NODES
    ])
    consistent = np.ptp(law_mins) <= 0.01 and np.ptp(high_medians) <= 0.01
    nondegraded = (
        np.all(loss_means <= 1.05 * original["fixed_crn_loss"]["mean"])
        and np.all(error_means <= 1.05 * original_error)
    )
    all_improve = (
        np.all(loss_means < original["fixed_crn_loss"]["mean"])
        and np.all(error_means < original_error)
    )
    support_repaired = np.all(law_mins >= MINIMUM_RESS) and np.all(high_medians >= MINIMUM_RESS)
    common_support_failure = np.all(law_mins < MINIMUM_RESS) or np.all(high_medians < MINIMUM_RESS)
    if consistent and nondegraded and support_repaired:
        label = "RETRAINS_CONSISTENTLY_REPAIR_SUPPORT"
        next_step = "Prospectively replicate the matched-bank comparison before considering any checkpoint change."
    elif consistent and all_improve and common_support_failure:
        label = "RETRAINS_IMPROVE_FIT_BUT_SUPPORT_REMAINS"
        next_step = "Reference fit improves, but the support geometry remains; continue with the population-overlap/proposal diagnostic."
    elif consistent and common_support_failure:
        label = "RETRAINS_REPRODUCE_SUPPORT_PROBLEM"
        next_step = "Retraining does not repair the common support failure; investigate the reference objective/proposal rather than optimizer seed."
    else:
        label = "REFERENCE_TRAINING_SEED_SENSITIVE"
        next_step = "Quantify training/data variability with more replicated datasets or improve training stability before selecting a checkpoint."
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "regenerated_training_endpoint_samples": TRAIN_ENDPOINT_SAMPLES,
        "regenerated_evaluation_truth_samples": EVAL_TRUTH_SAMPLES,
        "fresh_model_count": 3,
        "matched_flow_count": 4,
        "matched_bank_samples_per_flow": BANK_SAMPLES,
        "models": model_rows,
        "fresh_model_diagnostics": {
            "law_minimum_ress_values": law_mins.tolist(),
            "law_minimum_ress_spread": float(np.ptp(law_mins)),
            "high_pass_median_minimum_ress_values": high_medians.tolist(),
            "high_pass_median_minimum_ress_spread": float(np.ptp(high_medians)),
            "fixed_crn_loss_means": loss_means.tolist(),
            "truth_moment_error_means": error_means.tolist(),
            "consistent_under_predeclared_0p01_spread": bool(consistent),
            "physical_fit_nondegraded": bool(nondegraded),
            "all_physical_fit_metrics_improve": bool(all_improve),
            "support_repaired_on_matched_bank": bool(support_repaired),
        },
        "development_interpretation": label,
        "recommended_next_scientific_step": next_step,
        "production_checkpoint_installed": False,
        "official_protocol_created": False,
        "validation_accessed": False,
        "downstream_tangent_or_full_run": False,
    }
    _atomic_json(SUMMARY_PATH, payload)
    artifacts = [SOURCE_SEAL_PATH, PROTOCOL_PATH, TRAIN_DATA_PATH, EVAL_DATA_PATH, DATA_INVENTORY_PATH]
    for label_value in MODEL_LABELS:
        artifacts.extend((_checkpoint_path(label_value), _training_result_path(label_value)))
    for label_value in ALL_FLOW_LABELS:
        artifacts.extend((
            _bank_path(label_value), _bank_result_path(label_value),
            _evaluation_path(label_value), _evaluation_result_path(label_value),
        ))
    artifacts.append(SUMMARY_PATH)
    inventory = {
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": [{
            "path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        } for path in artifacts],
    }
    _atomic_json(INVENTORY_PATH, inventory)
    return {**payload, "cache_hit": False}


def run(cfg: dict[str, Any], progress: Any | None = None) -> dict[str, Any]:
    freeze_protocol(cfg)
    generate_regenerated_data(cfg)
    train_models(cfg, progress=progress)
    generate_matched_banks(cfg, progress=progress)
    evaluate_models(cfg, progress=progress)
    return summarize(cfg)


__all__ = [
    "ALL_FLOW_LABELS", "BANK_SAMPLES", "MODEL_LABELS",
    "DATA_INVENTORY_PATH", "EVAL_DATA_PATH", "EXPECTED_SOURCE_HASHES", "INVENTORY_PATH",
    "MINIMUM_RESS", "ORIGINAL_CHECKPOINT_PATH", "OUTPUT_ROOT", "PANEL_PATH", "PROTOCOL_PATH",
    "SOURCE_SEAL_PATH", "SUMMARY_PATH", "TRAIN_DATA_PATH", "evaluate_models", "freeze_protocol",
    "generate_matched_banks", "generate_regenerated_data", "run", "summarize", "train_models",
    "verify_and_freeze_sources",
]
