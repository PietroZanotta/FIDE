#!/usr/bin/env python3
"""Run the frozen V2.1-C3-64 independent-holdout confirmation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import run_v2_1_reduced_confirmatory as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = HERE / "outputs" / "prospective_v2_1_c3_64"
OLD_RUN = HERE / "outputs" / "prospective_v2_1_confirmatory_0p5_to_2pct"
PROTOCOL = HERE / "VORTICES_V2_1_C3_64_CONFIRMATORY_PROTOCOL_FROZEN.md"
RETIREMENT = OLD_RUN / "RETIRED_OUTCOME_BLIND_RUNTIME_AMENDMENT.json"
EXECUTION = RUN / "execution_receipt.json"
BANK = RUN / "shared_confirmatory_bank.npz"
BANK_RECEIPT = RUN / "shared_confirmatory_bank_receipt.json"
ACTION_RECEIPT = RUN / "exact_action_evaluation_receipt.json"
INFERENCE = RUN / "simultaneous_inference.json"
RISK_RECEIPT = RUN / "finite_risk_evaluation_receipt.json"
BENCHMARK = (
    HERE
    / "outputs"
    / "development_runtime_benchmarks"
    / "fused_confirmatory_cells.json"
)
TRIALS = 64
GENERATION_SEED = 22
NAMESPACE = 23
BOOTSTRAP_SEED = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("freeze", "bank", "evaluate", "analyze", "risk", "all")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def load_json(path: Path) -> dict[str, Any]:
    return base.load_json(path)


def atomic_json(path: Path, value: Any) -> None:
    base.atomic_json(path, value)


def configure_base() -> None:
    base.RUN = RUN
    base.PROTOCOL = PROTOCOL
    base.EXECUTION = EXECUTION
    base.BANK = BANK
    base.BANK_RECEIPT = BANK_RECEIPT
    base.ACTION_RECEIPT = ACTION_RECEIPT
    base.INFERENCE = INFERENCE
    base.RISK_RECEIPT = RISK_RECEIPT
    base.TRIALS = TRIALS
    base.GENERATION_SEED = GENERATION_SEED
    base.NAMESPACE = NAMESPACE
    base.BOOTSTRAP_SEED = BOOTSTRAP_SEED


def frozen_paths() -> dict[str, Path]:
    paths = {
        "protocol": PROTOCOL,
        "runner": Path(__file__),
        "base_exact_runner": Path(base.__file__),
        "selection_config": base.CONFIG,
        "pause_receipt": HERE / "outputs" / "prospective_v2_1" / "PAUSED_AFTER_2PCT.json",
        "selection_bank": base.SELECTION / "shared_selection_bank.npz",
        "law": base.BRANCH / "law" / "current_result.json",
        "full_0p5": base.BRANCH / "allowances" / "risk_0p5pct" / "full.json",
        "full_1p0": base.BRANCH / "allowances" / "risk_1p0pct" / "full.json",
        "full_2p0": base.BRANCH / "allowances" / "risk_2p0pct" / "full.json",
        "freeze_manifest": HERE / "VORTICES_V2_1_FREEZE_MANIFEST.json",
        "exact_solver": HERE / "v2_1_parallel_exact_solver.py",
        "selection_harness": HERE / "execute_v2_1_selection.py",
        "truth_bank": HERE / "inputs" / "truth_bank.npz",
        "retired_c3_receipt": RETIREMENT,
        "rejected_fused_benchmark": BENCHMARK,
    }
    reference_root = HERE / "outputs" / "prospective_v2" / "references"
    for index, seed in enumerate((310000101, 310000102, 310000103)):
        root = reference_root / f"reference_seed_{seed}"
        paths[f"reference_{index}"] = root / "reference.npz"
        paths[f"reference_bank_{index}"] = root / "reference_bank.npz"
        paths[f"reference_qualification_{index}"] = root / "qualification_receipt.json"
    return paths


def write_retirement() -> None:
    action_cells = sorted((OLD_RUN / "action_evaluations").glob("*/reference_*.json"))
    forbidden = [
        *action_cells,
        OLD_RUN / "exact_action_evaluation_receipt.json",
        OLD_RUN / "simultaneous_inference.json",
        OLD_RUN / "bootstrap_effects.npz",
        OLD_RUN / "finite_risk_evaluation_receipt.json",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"old C3 run contains reportable outcomes: {present}")
    required = (
        OLD_RUN / "execution_receipt.json",
        OLD_RUN / "shared_confirmatory_bank.npz",
        OLD_RUN / "shared_confirmatory_bank_receipt.json",
        OLD_RUN / "action_evaluation.log",
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError("old C3 provenance is incomplete")
    benchmark = load_json(BENCHMARK)
    if benchmark.get("status") != "FAIL" or benchmark.get("exact_structural_and_float_equality") is not False:
        raise RuntimeError("fused-cell acceleration must remain rejected")
    payload = {
        "schema_version": 1,
        "status": "PERMANENTLY_RETIRED_BEFORE_ANY_REPORTED_OR_INSPECTED_OUTCOME",
        "retired_experiment_id": "V2.1-C3-1024",
        "reason": "outcome-blind time-only amendment to prospectively justified 64 trials",
        "bank_was_generated": True,
        "evaluator_entered_computation": True,
        "action_cells_written": 0,
        "action_master_receipt_written": False,
        "inference_written": False,
        "outcome_inspected": False,
        "resumption_forbidden": True,
        "bank_analysis_forbidden": True,
        "replacement_seed_namespace_bootstrap": [22, 23, 24],
        "old_execution_receipt_sha256": sha256_file(required[0]),
        "old_bank_sha256": sha256_file(required[1]),
        "old_bank_receipt_sha256": sha256_file(required[2]),
        "old_log_sha256": sha256_file(required[3]),
        "rejected_fused_benchmark_sha256": sha256_file(BENCHMARK),
    }
    if RETIREMENT.exists():
        if load_json(RETIREMENT) != payload:
            raise RuntimeError("old C3 retirement receipt changed")
    else:
        atomic_json(RETIREMENT, payload)


def verify_paused_and_selected(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen C3-64 input(s): {missing}")
    pause = load_json(paths["pause_receipt"])
    if (
        pause.get("status") != "PAUSED_BY_USER_AFTER_2PCT_PARTIAL_PARETO"
        or pause.get("resume_authorization_present") is not False
        or pause.get("selection_process_running") is not False
    ):
        raise RuntimeError("the original 3--5% selection state is not paused")
    for tag in ("0p5", "1p0", "2p0"):
        receipt = load_json(paths[f"full_{tag}"])
        if receipt.get("status") != "PASS" or len(receipt["winner"]["eta"]) != 8:
            raise RuntimeError(f"Full {tag} selection receipt is not a frozen PASS")
    for index in range(3):
        if load_json(paths[f"reference_qualification_{index}"]).get("status") != "PASS":
            raise RuntimeError(f"reference {index} is not qualified")


def freeze_execution() -> None:
    write_retirement()
    paths = frozen_paths()
    verify_paused_and_selected(paths)
    if any(path.exists() for path in (BANK, BANK_RECEIPT, ACTION_RECEIPT, INFERENCE)):
        raise RuntimeError("C3-64 outcomes already exist; refusing retrospective freeze")
    development_report = (
        HERE / "outputs" / "prospective_v2_1" / "interim_validation_after_2pct" / "interim_directional_report.json"
    )
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FRESH_64_TRIAL_BANK_GENERATION",
        "experiment_id": "V2.1-C3-64",
        "design_classification": "development_adaptive_prospectively_confirmed_independent_holdout",
        "claim_scope": "confirmatory_truncated_pareto_0p5_to_2pct_only",
        "method_version": "V2.1_UNCHANGED",
        "allowance_percentages": [0.5, 1.0, 2.0],
        "designs": list(base.DESIGNS),
        "reference_count": 3,
        "trials": TRIALS,
        "generation_seed": GENERATION_SEED,
        "namespace": NAMESPACE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 100000,
        "simultaneous_family": "9_reference_by_allowance_effects",
        "no_confirmatory_top_up": True,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "development_report_used_for_scope_and_sample_size_sha256": sha256_file(development_report),
        "outcomes_generated_before_freeze": False,
        "original_3_to_5pct_selection_remains_paused": True,
    }
    if EXECUTION.exists():
        if load_json(EXECUTION) != payload:
            raise RuntimeError("existing C3-64 execution freeze differs")
    else:
        atomic_json(EXECUTION, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def verify_execution() -> dict[str, Any]:
    if not EXECUTION.is_file():
        raise RuntimeError("C3-64 execution must be frozen first")
    receipt = load_json(EXECUTION)
    if receipt.get("status") != "FROZEN_BEFORE_FRESH_64_TRIAL_BANK_GENERATION":
        raise RuntimeError("invalid C3-64 execution receipt")
    paths = frozen_paths()
    verify_paused_and_selected(paths)
    if {name: sha256_file(path) for name, path in paths.items()} != receipt.get("input_sha256"):
        raise RuntimeError("a frozen C3-64 input changed")
    return receipt


def generate_bank() -> None:
    execution = verify_execution()
    if BANK.exists() or BANK_RECEIPT.exists():
        raise RuntimeError("C3-64 bank already exists; refusing regeneration")
    config, _overlay = base.load_resolved_config()
    banks = config["observation_banks"]
    acquisition = list(map(int, banks["acquisition_indices_on_21_node_grid"]))
    bank = base.make_observation_bank(
        seed=GENERATION_SEED,
        namespace=NAMESPACE,
        trials=TRIALS,
        acquisition_k=len(acquisition),
        finite_n=int(banks["finite_particles"]),
        truth_particle_count=50000,
        n_observables=int(banks["observables"]),
    )
    indices = np.asarray(bank.sample_indices, dtype=np.int32)
    detector = np.asarray(bank.detector_z, dtype=np.float64)
    if indices.shape != (64, 9, 2000) or detector.shape != (64, 9, 4):
        raise RuntimeError("unexpected C3-64 bank shape")
    identity = {
        "schema_version": 1,
        "status": "FROZEN_SHARED_C3_64_CONFIRMATORY_BANK",
        "experiment_id": "V2.1-C3-64",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_64_TRIAL",
        "generation_seed": GENERATION_SEED,
        "namespace": NAMESPACE,
        "trials": TRIALS,
        "finite_particles": 2000,
        "truth_particle_count": 50000,
        "acquisition_indices": acquisition,
        "observables": 4,
        "shared_across_all_references_and_methods": True,
        "rng": "numpy.default_rng(SeedSequence([22,23]))",
        "execution_receipt_sha256": sha256_file(EXECUTION),
        "execution_input_sha256": execution["input_sha256"],
    }
    signature = base.fingerprint(identity)
    RUN.mkdir(parents=True, exist_ok=True)
    temporary = BANK.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_indices=indices,
        detector_z=detector,
        trial_ids=np.arange(TRIALS, dtype=np.int32),
        acquisition_indices=np.asarray(acquisition, dtype=np.int32),
        identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
        signature=np.asarray(signature),
    )
    os.replace(temporary, BANK)
    receipt = {
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank_path": str(BANK.resolve()),
        "bank_sha256": sha256_file(BANK),
        "signature": signature,
        "sample_indices_shape": list(indices.shape),
        "sample_indices_dtype": str(indices.dtype),
        "detector_z_shape": list(detector.shape),
        "detector_z_dtype": str(detector.dtype),
    }
    atomic_json(BANK_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def verify_bank() -> dict[str, Any]:
    verify_execution()
    receipt = load_json(BANK_RECEIPT)
    identity = (
        receipt.get("status"),
        receipt.get("generation_seed"),
        receipt.get("namespace"),
        receipt.get("trials"),
        receipt.get("data_role"),
    )
    expected = (
        "FROZEN_SHARED_C3_64_CONFIRMATORY_BANK",
        22,
        23,
        64,
        "FINAL_CONFIRMATORY_0P5_TO_2PCT_64_TRIAL",
    )
    if identity != expected or sha256_file(BANK) != receipt.get("bank_sha256"):
        raise RuntimeError("C3-64 bank identity/hash mismatch")
    return receipt


def analyze() -> None:
    verify_bank()
    master = load_json(ACTION_RECEIPT)
    if master.get("status") != "COMPLETE_REDUCED_CONFIRMATORY_EXACT_ACTION_EVALUATION":
        raise RuntimeError("C3-64 exact action evaluation is incomplete")
    actions = np.empty((4, 3, TRIALS), dtype=np.float64)
    all_rows: list[dict[str, Any]] = []
    for design_index, design in enumerate(base.DESIGNS):
        for reference in range(3):
            path = RUN / "action_evaluations" / design / f"reference_{reference}.json"
            relative = str(path.relative_to(RUN))
            if sha256_file(path) != master["evaluation_receipts"].get(relative):
                raise RuntimeError(f"C3-64 action-cell hash mismatch: {relative}")
            payload = load_json(path)
            rows = payload["rows"]
            if payload.get("status") != "COMPLETE" or [row["trial"] for row in rows] != list(range(TRIALS)):
                raise RuntimeError(f"invalid C3-64 cell: {relative}")
            if not all(row["valid"] for row in rows):
                raise RuntimeError(f"numerical-invalid trial in {relative}")
            actions[design_index, reference] = [row["action"] for row in rows]
            all_rows.extend(rows)
    law = actions[0]
    full = actions[1:].transpose(1, 0, 2)
    observed = 1.0 - np.mean(full, axis=2) / np.mean(law, axis=1)[:, None]
    bootstrap = np.empty((100000, 3, 3), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    digest = hashlib.sha256()
    for begin in range(0, 100000, 512):
        count = min(512, 100000 - begin)
        indices = rng.integers(0, TRIALS, size=(count, TRIALS), dtype=np.int32)
        digest.update(indices.tobytes(order="C"))
        law_means = np.mean(law[:, indices], axis=-1).T
        full_means = np.mean(full[:, :, indices], axis=-1).transpose(2, 0, 1)
        bootstrap[begin : begin + count] = 1.0 - full_means / law_means[:, :, None]
    critical = float(np.quantile(np.max(np.abs(bootstrap - observed[None]), axis=(1, 2)), 0.95))
    pointwise = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    bootstrap_path = RUN / "bootstrap_effects.npz"
    temporary = RUN / "bootstrap_effects.tmp.npz"
    np.savez_compressed(temporary, effects=bootstrap)
    os.replace(temporary, bootstrap_path)
    summaries = {}
    for index, design in enumerate(base.DESIGNS):
        summaries[design] = []
        for reference in range(3):
            array = actions[index, reference]
            mean = float(np.mean(array))
            se = float(np.std(array, ddof=1) / math.sqrt(TRIALS))
            summaries[design].append({"mean": mean, "standard_error": se, "relative_standard_error": se / abs(mean)})
    max_rse = max(row["relative_standard_error"] for group in summaries.values() for row in group)
    lower, upper = observed - critical, observed + critical
    gates = {
        "exactly_three_qualified_references": True,
        "exactly_64_shared_trials": True,
        "all_768_exact_action_evaluations_numerically_valid": len(all_rows) == 768 and all(row["valid"] for row in all_rows),
        "all_nine_simultaneous_lower_bounds_strictly_positive": bool(np.all(lower > 0.0)),
        "maximum_simultaneous_half_width_at_most_0p05": critical <= 0.05,
        "all_relative_standard_errors_at_most_0p10": max_rse <= 0.10,
        "no_outcome_dependent_confirmatory_amendment": True,
        "no_trial_top_up": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "experiment_id": "V2.1-C3-64",
        "design_classification": "development_adaptive_prospectively_confirmed_independent_holdout",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_64_TRIAL",
        "claim_scope": "truncated_pareto_0p5_to_2pct_only",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowance_percentages": [0.5, 1.0, 2.0],
        "reference_count": 3,
        "trials": TRIALS,
        "effects": observed.tolist(),
        "pointwise_lower": pointwise[0].tolist(),
        "pointwise_upper": pointwise[1].tolist(),
        "simultaneous_critical_half_width": critical,
        "simultaneous_lower": lower.tolist(),
        "simultaneous_upper": upper.tolist(),
        "equal_reference_effects": np.mean(observed, axis=0).tolist(),
        "between_reference_effect_ranges": (np.max(observed, axis=0) - np.min(observed, axis=0)).tolist(),
        "summaries": summaries,
        "maximum_relative_standard_error": max_rse,
        "shared_index_stream_sha256": digest.hexdigest(),
        "bootstrap_resamples": 100000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_effects_sha256": sha256_file(bootstrap_path),
        "gates": gates,
        "numerical_extrema": {
            "minimum_ess_fraction": float(min(row["minimum_ess_fraction"] for row in all_rows)),
            "maximum_calibration_residual": float(max(row["maximum_calibration_residual"] for row in all_rows)),
            "maximum_poisson_relative_residual": float(max(row["maximum_poisson_relative_residual"] for row in all_rows)),
            "maximum_mass_error": float(max(row["maximum_mass_error"] for row in all_rows)),
        },
        "protocol_sha256": sha256_file(PROTOCOL),
        "execution_receipt_sha256": sha256_file(EXECUTION),
        "bank_receipt_sha256": sha256_file(BANK_RECEIPT),
        "exact_action_receipt_sha256": sha256_file(ACTION_RECEIPT),
        "retired_c3_receipt_sha256": sha256_file(RETIREMENT),
        "original_3_to_5pct_selection_remains_paused": True,
    }
    atomic_json(INFERENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def main() -> int:
    configure_base()
    base.frozen_paths = frozen_paths
    base.verify_paused_and_selected = verify_paused_and_selected
    base.verify_bank = verify_bank
    stage = parse_args().stage
    if stage in ("freeze", "all"):
        freeze_execution()
    if stage in ("bank", "all"):
        generate_bank()
    if stage in ("evaluate", "all"):
        base.evaluate_actions()
    if stage in ("analyze", "all"):
        analyze()
    if stage in ("risk", "all"):
        base.evaluate_risk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
