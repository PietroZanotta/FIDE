"""Second pre-receipt V3 amendment: omit unused risk features in guard screening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
import numpy as np

from .galerkin_only_data import SelectionGalerkinData
from . import official_b1_pareto_v3_support_robust as study
from . import official_b1_pareto_v3_memory_safe_run as memory_v3_1


AMENDMENT_PATH = study.OUTPUT_ROOT / "low_memory_amendment_v3_2.json"
RESULT_PATH = study.OUTPUT_ROOT / "OFFICIAL_B1_GALERKIN_PARETO_V3_2_LOW_MEMORY_RESULT.md"
SOURCE_PATH = Path(__file__)


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
            raise RuntimeError(f"refusing to overwrite V3.2 artifact: {path}")
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
    prior = memory_v3_1.prepare_amendment()
    if AMENDMENT_PATH.exists():
        amendment = json.loads(AMENDMENT_PATH.read_text())
        body = {key: value for key, value in amendment.items() if key != "amendment_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != amendment["amendment_sha256"]:
            raise RuntimeError("V3.2 low-memory amendment digest mismatch")
        if amendment["source_sha256"] != _sha256(SOURCE_PATH):
            raise RuntimeError("V3.2 low-memory source changed after freeze")
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
        raise RuntimeError(f"V3 outcomes exist before low-memory amendment: {forbidden}")
    body = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_GUARD_RECEIPT_OR_ACTION_OUTCOME",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "second pre-outcome operational memory-scheduling amendment",
        "v3_protocol_sha256": protocol["v3_protocol_sha256"],
        "parent_memory_amendment_sha256": prior["amendment_sha256"],
        "source_sha256": _sha256(SOURCE_PATH),
        "trigger": "65,536 guard OOM before receipt because generic selection_data eagerly computed unused risk-anchor reference features",
        "sole_changes": [
            "construct the identical frozen selection problem without the unused SelectionGalerkinData.reference_features tensor",
            "use evaluator minibatch size one while preserving the frozen eight-candidate stopping block",
        ],
        "scientific_semantics_changed": False,
        "unchanged": {
            "root_seed": study.ROOT_SEED,
            "candidate_stopping_block_size": study.GUARD_BLOCK_SIZE,
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
        progress(f"V3.2 low-memory amendment frozen: {amendment['amendment_sha256']}")
    return amendment


def _light_guard_data(bank: Any) -> SelectionGalerkinData:
    cfg = study.base.effective_config()
    with np.load(
        study.OUTPUT_ROOT / "design_truth" / "design_truth.npz", allow_pickle=False
    ) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
        truth_means = jnp.asarray(arrays["truth_means"], dtype=jnp.float64)
        whitening = jnp.asarray(arrays["whitening"], dtype=jnp.float64)
    problem = study.base._problem(
        cfg, truth, times,
        noise_seed=study.role_seed("selection_observation_noise"),
    )
    return SelectionGalerkinData(
        selection_problem=problem,
        projection_bank=bank,
        train_bank=bank,
        audit_bank=bank,
        reference_features=jnp.empty((0,), dtype=jnp.float64),
        truth_means=truth_means,
        whitening=whitening,
    )


def low_memory_guard_qualify_rows(
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
        etas = np.asarray([row["eta"] for row in missing], dtype=np.float64)
        by_role: dict[str, dict[str, np.ndarray]] = {}
        for role in study.GUARD_COUNTS:
            bank = study._load_guard(role)
            data = _light_guard_data(bank)
            evaluator = study.base.CandidateEvaluator(data, batch_size=1)
            by_role[role] = evaluator.evaluate(etas, bank)
            del evaluator, data, bank
            jax.clear_caches()
            gc.collect()
            if progress:
                progress(f"V3.2 low-memory guard role {role} released")
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
                "low_memory_amendment_sha256": amendment["amendment_sha256"],
            }
            study._atomic_json(study._guard_cache_path(row["eta"]), receipt)
            cached[receipt["eta_sha256"]] = receipt
        if progress:
            progress(f"V3.2 low-memory guard-qualified {len(missing)} candidates")
    return [cached[study.base.eta_key(row["eta"])] for row in rows]


def activate() -> dict[str, Any]:
    amendment = prepare_amendment()
    study.guard_qualify_rows = low_memory_guard_qualify_rows
    return amendment


def write_reports(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    amendment = activate()
    summary = study.write_reports(progress)
    memory_v3_1._atomic_text(RESULT_PATH, "\n".join((
        "# Official B1 Galerkin Pareto V3.2 Low-Memory Result",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"V3 protocol: `{amendment['v3_protocol_sha256']}`",
        f"V3.1 memory amendment: `{amendment['parent_memory_amendment_sha256']}`",
        f"V3.2 low-memory amendment: `{amendment['amendment_sha256']}`",
        "Scientific change: `none`; only unused-tensor omission and evaluator minibatching changed.",
        "",
    )))
    return summary


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("freeze", "refreeze-law", "selection", "certify", "validation", "report", "all"),
    )
    args = parser.parse_args()
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
                study.base.record_stage_performance(f"v3_2_{mode}", elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
