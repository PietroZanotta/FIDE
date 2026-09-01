"""Final pre-receipt V3 amendment: isolate each guard role in a JAX process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from . import official_b1_pareto_v3_support_robust as study
from . import official_b1_pareto_v3_low_memory_run as low_memory
from . import official_b1_pareto_v3_memory_safe_run as memory_v3_1


AMENDMENT_PATH = study.OUTPUT_ROOT / "isolated_guard_amendment_v3_3.json"
RESULT_PATH = study.OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_3_ISOLATED_GUARD_RESULT.md"
SOURCE_PATH = Path(__file__)
MODULE = "experiments.skyrmions_galerkin.official_b1_pareto_v3_isolated_guard_run"


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
            raise RuntimeError(f"refusing to overwrite V3.3 artifact: {path}")
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


def prepare_amendment(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    protocol = study.require_v3()
    parent = low_memory.prepare_amendment()
    if AMENDMENT_PATH.exists():
        amendment = json.loads(AMENDMENT_PATH.read_text())
        body = {key: value for key, value in amendment.items() if key != "amendment_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != amendment["amendment_sha256"]:
            raise RuntimeError("V3.3 isolated-guard amendment digest mismatch")
        if amendment["source_sha256"] != _sha256(SOURCE_PATH):
            raise RuntimeError("V3.3 isolated-guard source changed after freeze")
        return amendment
    forbidden = [
        path for root in (
            study.OUTPUT_ROOT / "law",
            study.OUTPUT_ROOT / "selection",
            study.OUTPUT_ROOT / "authoritative",
            study.OUTPUT_ROOT / "heldout_validation",
        ) if root.exists() for path in root.rglob("*") if path.is_file()
    ]
    forbidden.extend(
        path for root in study.OUTPUT_ROOT.glob("selection_pass_*")
        for path in root.rglob("*") if path.is_file()
    )
    if forbidden:
        raise RuntimeError(f"V3 outcomes exist before isolated-guard amendment: {forbidden}")
    body = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_GUARD_RECEIPT_OR_ACTION_OUTCOME",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "final pre-outcome operational memory-scheduling amendment",
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "parent_low_memory_amendment_sha256": parent["amendment_sha256"],
        "source_sha256": _sha256(SOURCE_PATH),
        "trigger": "same-process BFC fragmentation before the 65,536 guard allocation",
        "sole_change": "evaluate each frozen guard role in a fresh JAX process and seal a role checkpoint",
        "scientific_semantics_changed": False,
        "unchanged": {
            "root_seed": study.ROOT_SEED,
            "candidate_stopping_block_size": study.GUARD_BLOCK_SIZE,
            "evaluator_minibatch_size": 1,
            "guard_roles_and_sample_counts": study.GUARD_COUNTS,
            "minimum_rESS": 0.05,
            "candidate_order": "exact risk then eta_sha256",
            "guard_banks": {
                role: _sha256(study._guard_path(role)) for role in study.GUARD_COUNTS
            },
            "dtype": "float64",
            "backend": "jax",
        },
        "guard_results_available_before_freeze": False,
        "action_outcomes_available_before_freeze": False,
        "validation_accessed": False,
    }
    amendment = {
        **body,
        "amendment_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    _atomic_json(AMENDMENT_PATH, amendment)
    if progress:
        progress(f"V3.3 isolated-guard amendment frozen: {amendment['amendment_sha256']}")
    return amendment


def _checkpoint_root(rows: list[dict[str, Any]]) -> Path:
    key = hashlib.sha256(_canonical([
        study.base.eta_key(row["eta"]) for row in rows
    ])).hexdigest()
    return study.OUTPUT_ROOT / "law" / "guard_role_checkpoints" / key


def worker(role: str, input_path: Path, output_path: Path) -> None:
    prepare_amendment()
    rows = json.loads(input_path.read_text())["rows"]
    bank = study._load_guard(role)
    data = low_memory._light_guard_data(bank)
    evaluator = study.base.CandidateEvaluator(data, batch_size=1)
    result = evaluator.evaluate(
        np.asarray([row["eta"] for row in rows], dtype=np.float64), bank
    )
    study.base.atomic_npz(
        output_path,
        support_valid=result["support_valid"],
        minimum_ress=result["minimum_ress"],
        maximum_projection_residual=result["maximum_projection_residual"],
        maximum_forcing_mean=result["maximum_forcing_mean"],
        maximum_covariance_condition=result["maximum_covariance_condition"],
    )


def isolated_guard_qualify_rows(
    rows: list[dict[str, Any]],
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    amendment = prepare_amendment(progress)
    study.generate_data(progress)
    cached: dict[str, dict[str, Any]] = {}
    missing = []
    for row in rows:
        path = study._guard_cache_path(row["eta"])
        if path.exists():
            cached[study.base.eta_key(row["eta"])] = json.loads(path.read_text())
        else:
            missing.append(row)
    if missing:
        root = _checkpoint_root(missing)
        input_path = root / "input.json"
        _atomic_json(input_path, {"rows": missing})
        by_role: dict[str, dict[str, np.ndarray]] = {}
        for role in study.GUARD_COUNTS:
            output_path = root / f"{role}.npz"
            if not output_path.exists():
                environment = dict(os.environ)
                environment["JAX_ENABLE_X64"] = "1"
                environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                subprocess.run(
                    [
                        sys.executable, "-m", MODULE,
                        "--worker-role", role,
                        "--worker-input", str(input_path),
                        "--worker-output", str(output_path),
                    ],
                    check=True,
                    env=environment,
                )
            with np.load(output_path, allow_pickle=False) as arrays:
                by_role[role] = {name: np.asarray(arrays[name]) for name in arrays.files}
            if progress:
                progress(f"V3.3 isolated guard role {role} sealed")
        for index, row in enumerate(missing):
            support = {
                role: {
                    "support_valid": bool(result["support_valid"][index]),
                    "minimum_rESS": float(result["minimum_ress"][index]),
                    "maximum_projection_residual": float(result["maximum_projection_residual"][index]),
                    "maximum_forcing_mean": float(result["maximum_forcing_mean"][index]),
                    "maximum_covariance_condition": float(result["maximum_covariance_condition"][index]),
                }
                for role, result in by_role.items()
            }
            receipt = {
                "schema_version": 1,
                "candidate_id": row.get(
                    "candidate_id", f"downstream_{study.base.eta_key(row['eta'])}"
                ),
                "eta": row["eta"],
                "eta_sha256": study.base.eta_key(row["eta"]),
                "exact_scientific_risk": row["exact_scientific_risk"],
                "support_by_fresh_guard_role": support,
                "support_robust": all(item["support_valid"] for item in support.values()),
                "minimum_guard_rESS": min(item["minimum_rESS"] for item in support.values()),
                "threshold": 0.05,
                "authoritative_audit_used": False,
                "isolated_guard_amendment_sha256": amendment["amendment_sha256"],
            }
            study._atomic_json(study._guard_cache_path(row["eta"]), receipt)
            cached[receipt["eta_sha256"]] = receipt
        if progress:
            progress(f"V3.3 isolated guard-qualified {len(missing)} candidates")
    return [cached[study.base.eta_key(row["eta"])] for row in rows]


def activate() -> dict[str, Any]:
    amendment = prepare_amendment()
    study.guard_qualify_rows = isolated_guard_qualify_rows
    return amendment


def write_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    amendment = activate()
    summary = study.write_reports(progress)
    memory_v3_1._atomic_text(RESULT_PATH, "\n".join((
        "# Official B1 Galerkin Pareto V3.3 Isolated-Guard Result",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"V3 protocol: `{amendment['v3_protocol_sha256']}`",
        f"V3.3 isolated-guard amendment: `{amendment['amendment_sha256']}`",
        "Scientific change: `none`; each guard role used a fresh JAX process.",
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
                study.base.record_stage_performance(f"v3_3_{mode}", elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}", flush=True,
            )


if __name__ == "__main__":
    main()
