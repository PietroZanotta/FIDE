"""Runtime V3 amendment: sample-chunked many-body guard features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from . import official_b1_pareto_v3_support_robust as study
from . import official_b1_pareto_v3_isolated_guard_run as isolated
from . import official_b1_pareto_v3_memory_safe_run as memory_v3_1


AMENDMENT_PATH = study.OUTPUT_ROOT / "chunked_guard_amendment_v3_4.json"
RESULT_PATH = study.OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_4_CHUNKED_GUARD_RESULT.md"
SOURCE_PATH = Path(__file__)
MODULE = "experiments.skyrmions_galerkin.official_b1_pareto_v3_chunked_guard_run"
FEATURE_SAMPLE_CHUNK = 8192
ORIGINAL_MANY_BODY_FEATURES = study.base.many_body_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _atomic_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite V3.4 artifact: {path}")
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


def chunked_many_body_features(
    configurations: Any,
    box: tuple[float, float] = (2.0, 1.0),
) -> jax.Array:
    configurations = jnp.asarray(configurations, dtype=jnp.float64)
    samples = int(configurations.shape[-3])
    if samples <= FEATURE_SAMPLE_CHUNK:
        return ORIGINAL_MANY_BODY_FEATURES(configurations, box)
    pieces = []
    for start in range(0, samples, FEATURE_SAMPLE_CHUNK):
        piece = ORIGINAL_MANY_BODY_FEATURES(
            configurations[..., start:start + FEATURE_SAMPLE_CHUNK, :, :], box
        )
        piece.block_until_ready()
        pieces.append(piece)
    return jnp.concatenate(pieces, axis=-2)


def prepare_amendment(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = study.require_v3()
    parent = isolated.prepare_amendment()
    if AMENDMENT_PATH.exists():
        amendment = json.loads(AMENDMENT_PATH.read_text())
        body = {key: value for key, value in amendment.items() if key != "amendment_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != amendment["amendment_sha256"]:
            raise RuntimeError("V3.4 chunked-guard amendment digest mismatch")
        if amendment["source_sha256"] != _sha256(SOURCE_PATH):
            raise RuntimeError("V3.4 chunked-guard source changed after freeze")
        return amendment
    checkpoint_files = sorted(
        (study.OUTPUT_ROOT / "law" / "guard_role_checkpoints").rglob("*.npz")
    )
    expected_roles = {
        "law_guard_screen.npz",
        "law_guard_search_train.npz",
        "law_guard_periodic_audit.npz",
    }
    if {path.name for path in checkpoint_files} != expected_roles:
        raise RuntimeError(f"unexpected pre-amendment guard checkpoints: {checkpoint_files}")
    forbidden = [
        path for root in (
            study.OUTPUT_ROOT / "law" / "guard_cache",
            study.OUTPUT_ROOT / "selection",
            study.OUTPUT_ROOT / "authoritative",
            study.OUTPUT_ROOT / "heldout_validation",
        ) if root.exists() for path in root.rglob("*") if path.is_file()
    ]
    forbidden.extend(
        path for root in study.OUTPUT_ROOT.glob("selection_pass_*")
        for path in root.rglob("*") if path.is_file()
    )
    if study.base.LAW_PATH.exists() or forbidden:
        raise RuntimeError("candidate receipts, Law, action, or validation exist before V3.4")
    body = {
        "schema_version": 1,
        "status": "FROZEN_AFTER_THREE_UNINSPECTED_ROLE_CHECKPOINTS_BEFORE_ANY_CANDIDATE_RECEIPT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "runtime-only algebraically equivalent feature-memory amendment",
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "parent_isolated_guard_amendment_sha256": parent["amendment_sha256"],
        "source_sha256": _sha256(SOURCE_PATH),
        "trigger": "fresh-process 65,536 role OOM in monolithic many_body_features temporary",
        "sole_change": "evaluate independent sample-axis many-body features in fixed chunks and concatenate",
        "feature_sample_chunk": FEATURE_SAMPLE_CHUNK,
        "scientific_semantics_changed": False,
        "development_equivalence": {
            "input": "first 8,193 samples of frozen 65,536 guard bank",
            "maximum_absolute_feature_difference": 5.551115123125783e-17,
            "maximum_relative_feature_difference": 1.1652837586044806e-15,
            "allclose_rtol_atol_1e_12": True,
            "note": "42 of 958,581 entries differed by shape-dependent floating-point roundoff",
        },
        "checkpoint_values_inspected_before_freeze": False,
        "preexisting_role_checkpoint_hashes": {
            str(path.relative_to(study.OUTPUT_ROOT)): _sha256(path)
            for path in checkpoint_files
        },
        "unchanged": {
            "root_seed": study.ROOT_SEED,
            "candidate_stopping_block_size": study.GUARD_BLOCK_SIZE,
            "evaluator_minibatch_size": 1,
            "guard_roles_and_sample_counts": study.GUARD_COUNTS,
            "minimum_rESS": 0.05,
            "candidate_order": "exact risk then eta_sha256",
            "dtype": "float64",
            "backend": "jax",
        },
        "candidate_guard_receipts_available_before_freeze": False,
        "law_available_before_freeze": False,
        "action_outcomes_available_before_freeze": False,
        "validation_accessed": False,
    }
    amendment = {
        **body,
        "amendment_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    _atomic_json(AMENDMENT_PATH, amendment)
    if progress:
        progress(f"V3.4 chunked-guard amendment frozen: {amendment['amendment_sha256']}")
    return amendment


def activate() -> dict[str, Any]:
    amendment = prepare_amendment()
    study.base.many_body_features = chunked_many_body_features
    isolated.MODULE = MODULE
    study.guard_qualify_rows = isolated.isolated_guard_qualify_rows
    return amendment


def worker(role: str, input_path: Path, output_path: Path) -> None:
    activate()
    isolated.worker(role, input_path, output_path)


def write_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    amendment = activate()
    summary = study.write_reports(progress)
    memory_v3_1._atomic_text(RESULT_PATH, "\n".join((
        "# Official B1 Galerkin Pareto V3.4 Chunked-Guard Result",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"V3 protocol: `{amendment['v3_protocol_sha256']}`",
        f"V3.4 chunked-guard amendment: `{amendment['amendment_sha256']}`",
        "Scientific change: `none`; independent sample features were chunked with <=5.56e-17 development discrepancy.",
        "",
    )))
    return summary


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "refreeze-law", "selection", "certify", "validation", "report", "all"))
    parser.add_argument("--worker-role", choices=tuple(study.GUARD_COUNTS))
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_role:
        if args.worker_input is None or args.worker_output is None:
            parser.error("worker mode requires input and output")
        worker(args.worker_role, args.worker_input, args.worker_output)
        return
    if args.mode is None:
        parser.error("--mode is required outside worker mode")
    routes = {
        "freeze": prepare_amendment,
        "refreeze-law": study.refreeze_law,
        "selection": study.run_selection_with_restarts,
        "certify": study.certify,
        "validation": study.validate,
        "report": write_reports,
    }
    order = ("freeze", "refreeze-law", "selection", "certify", "validation", "report")
    activate()
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in order if args.mode == "all" else (args.mode,):
            print(f"starting={mode}", flush=True)
            before = study.base._gpu_snapshot()
            started = time.perf_counter()
            try:
                result = routes[mode](progress=progress)
            except BaseException as error:
                if mode not in {"freeze", "refreeze-law"}:
                    study.write_failure_report(mode, error)
                raise
            elapsed = time.perf_counter() - started
            after = study.base._gpu_snapshot()
            if mode not in {"freeze", "report"}:
                study.base.record_stage_performance(f"v3_4_{mode}", elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}", flush=True,
            )


if __name__ == "__main__":
    main()
