#!/usr/bin/env python3
"""Execute the frozen V2.1-C3 reduced 0.5--2% confirmatory experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = HERE / "outputs" / "prospective_v2_1_confirmatory_0p5_to_2pct"
SELECTION = HERE / "outputs" / "prospective_v2_1" / "selection"
BRANCH = SELECTION / "law_refinement_1_feasible_anchor"
PROTOCOL = HERE / "VORTICES_V2_1_REDUCED_CONFIRMATORY_PROTOCOL_FROZEN.md"
EXECUTION = RUN / "execution_receipt.json"
BANK = RUN / "shared_confirmatory_bank.npz"
BANK_RECEIPT = RUN / "shared_confirmatory_bank_receipt.json"
ACTION_RECEIPT = RUN / "exact_action_evaluation_receipt.json"
INFERENCE = RUN / "simultaneous_inference.json"
RISK_RECEIPT = RUN / "finite_risk_evaluation_receipt.json"
DESIGNS = ("law", "full_0p5", "full_1p0", "full_2p0")
ALLOWANCES = (0.5, 1.0, 2.0)
GENERATION_SEED = 19
NAMESPACE = 20
BOOTSTRAP_SEED = 21
TRIALS = 1024
WORKERS = 4

for path in (
    REPO / "src",
    REPO / "experiments",
    REPO / "experiments" / "vortices_percentage",
    HERE,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bounded_reference import BoxTransformedReferenceFlow  # noqa: E402
from core import DevelopmentContext  # noqa: E402
from experiment import ObservationTrialBank, VortexExperiment, make_observation_bank  # noqa: E402
import execute_v2_1_selection as selection_runner  # noqa: E402
from mfsi.cache import fingerprint  # noqa: E402
from v2_1_contract import CONFIG, load_resolved_config  # noqa: E402
from v2_1_fast_orchestration import install_parallel_exact_solver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("freeze", "bank", "evaluate", "analyze", "risk", "all")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def frozen_paths() -> dict[str, Path]:
    paths = {
        "protocol": PROTOCOL,
        "runner": Path(__file__),
        "selection_config": CONFIG,
        "pause_receipt": HERE / "outputs" / "prospective_v2_1" / "PAUSED_AFTER_2PCT.json",
        "selection_bank": SELECTION / "shared_selection_bank.npz",
        "law": BRANCH / "law" / "current_result.json",
        "full_0p5": BRANCH / "allowances" / "risk_0p5pct" / "full.json",
        "full_1p0": BRANCH / "allowances" / "risk_1p0pct" / "full.json",
        "full_2p0": BRANCH / "allowances" / "risk_2p0pct" / "full.json",
        "freeze_manifest": HERE / "VORTICES_V2_1_FREEZE_MANIFEST.json",
        "exact_solver": HERE / "v2_1_parallel_exact_solver.py",
        "selection_harness": HERE / "execute_v2_1_selection.py",
        "truth_bank": HERE / "inputs" / "truth_bank.npz",
    }
    reference_root = HERE / "outputs" / "prospective_v2" / "references"
    for index, seed in enumerate((310000101, 310000102, 310000103)):
        root = reference_root / f"reference_seed_{seed}"
        paths[f"reference_{index}"] = root / "reference.npz"
        paths[f"reference_bank_{index}"] = root / "reference_bank.npz"
        paths[f"reference_qualification_{index}"] = root / "qualification_receipt.json"
    return paths


def verify_paused_and_selected(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen confirmatory input(s): {missing}")
    pause = load_json(paths["pause_receipt"])
    if (
        pause.get("status") != "PAUSED_BY_USER_AFTER_2PCT_PARTIAL_PARETO"
        or pause.get("resume_authorization_present") is not False
        or pause.get("selection_process_running") is not False
    ):
        raise RuntimeError("the original 3--5% selection state is not safely paused")
    for tag in ("0p5", "1p0", "2p0"):
        receipt = load_json(paths[f"full_{tag}"])
        if receipt.get("status") != "PASS" or len(receipt["winner"]["eta"]) != 8:
            raise RuntimeError(f"Full {tag} selection receipt is not a frozen PASS")
    for index in range(3):
        if load_json(paths[f"reference_qualification_{index}"]).get("status") != "PASS":
            raise RuntimeError(f"reference {index} is not qualified")
    original = HERE / "outputs" / "prospective_v2_1"
    forbidden = (
        original / "stress" / "shared_stress_bank.npz",
        original / "stress_test" / "shared_stress_bank.npz",
        original / "validation" / "shared_validation_bank.npz",
        original / "RESUME_AFTER_2PCT.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("original V2.1 stress/final/resume state unexpectedly changed")


def freeze_execution() -> None:
    paths = frozen_paths()
    verify_paused_and_selected(paths)
    if any(path.exists() for path in (BANK, BANK_RECEIPT, ACTION_RECEIPT, INFERENCE)):
        raise RuntimeError("confirmatory outcomes already exist; refusing retrospective freeze")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CONFIRMATORY_BANK_GENERATION",
        "experiment_id": "V2.1-C3",
        "claim_scope": "confirmatory_truncated_pareto_0p5_to_2pct_only",
        "method_version": "V2.1_UNCHANGED",
        "allowance_percentages": list(ALLOWANCES),
        "designs": list(DESIGNS),
        "reference_count": 3,
        "trials": TRIALS,
        "generation_seed": GENERATION_SEED,
        "namespace": NAMESPACE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 100000,
        "simultaneous_family": "9_reference_by_allowance_effects",
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "development_report_used_only_to_choose_reduced_scope_sha256": sha256_file(
            HERE
            / "outputs"
            / "prospective_v2_1"
            / "interim_validation_after_2pct"
            / "interim_directional_report.json"
        ),
        "outcomes_generated_before_freeze": False,
        "original_3_to_5pct_selection_remains_paused": True,
        "original_namespace_12_used": False,
        "original_namespace_13_used": False,
    }
    if EXECUTION.exists():
        prior = load_json(EXECUTION)
        if prior != payload:
            raise RuntimeError("existing execution freeze differs from current inputs")
    else:
        atomic_json(EXECUTION, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def verify_execution() -> dict[str, Any]:
    if not EXECUTION.is_file():
        raise RuntimeError("confirmatory execution must be frozen first")
    receipt = load_json(EXECUTION)
    if receipt.get("status") != "FROZEN_BEFORE_CONFIRMATORY_BANK_GENERATION":
        raise RuntimeError("invalid confirmatory execution receipt")
    paths = frozen_paths()
    verify_paused_and_selected(paths)
    current = {name: sha256_file(path) for name, path in paths.items()}
    if current != receipt.get("input_sha256"):
        raise RuntimeError("a frozen confirmatory input changed after execution freeze")
    return receipt


def generate_bank() -> None:
    execution = verify_execution()
    if BANK.exists() or BANK_RECEIPT.exists():
        raise RuntimeError("confirmatory bank already exists; refusing regeneration")
    config, _overlay = load_resolved_config()
    banks = config["observation_banks"]
    acquisition = list(map(int, banks["acquisition_indices_on_21_node_grid"]))
    bank = make_observation_bank(
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
    if indices.shape != (1024, 9, 2000) or detector.shape != (1024, 9, 4):
        raise RuntimeError("unexpected confirmatory bank shape")
    identity = {
        "schema_version": 1,
        "status": "FROZEN_SHARED_CONFIRMATORY_BANK",
        "experiment_id": "V2.1-C3",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_ONLY",
        "generation_seed": GENERATION_SEED,
        "namespace": NAMESPACE,
        "trials": TRIALS,
        "finite_particles": 2000,
        "truth_particle_count": 50000,
        "acquisition_indices": acquisition,
        "observables": 4,
        "shared_across_all_references_and_methods": True,
        "rng": "numpy.default_rng(SeedSequence([19,20]))",
        "execution_receipt_sha256": sha256_file(EXECUTION),
        "execution_input_sha256": execution["input_sha256"],
    }
    signature = fingerprint(identity)
    temporary = BANK.with_suffix(".tmp.npz")
    RUN.mkdir(parents=True, exist_ok=True)
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
        "original_namespace_12_used": False,
        "original_namespace_13_used": False,
    }
    atomic_json(BANK_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def verify_bank() -> dict[str, Any]:
    verify_execution()
    if not BANK.is_file() or not BANK_RECEIPT.is_file():
        raise RuntimeError("confirmatory bank and receipt are required")
    receipt = load_json(BANK_RECEIPT)
    identity = (
        receipt.get("status"),
        receipt.get("generation_seed"),
        receipt.get("namespace"),
        receipt.get("trials"),
        receipt.get("data_role"),
    )
    expected = (
        "FROZEN_SHARED_CONFIRMATORY_BANK",
        19,
        20,
        1024,
        "FINAL_CONFIRMATORY_0P5_TO_2PCT_ONLY",
    )
    if identity != expected or sha256_file(BANK) != receipt.get("bank_sha256"):
        raise RuntimeError("confirmatory bank identity/hash mismatch")
    return receipt


def load_experiments(config: dict[str, Any]):
    v1 = HERE
    reference_root = HERE / "outputs" / "prospective_v2" / "references"
    cfg = load_json(v1 / "base_experiment_config.json")
    with np.load(v1 / "inputs" / "truth_bank.npz", allow_pickle=False) as raw:
        times = np.asarray(raw["times"], dtype=np.float64)
        truth = jnp.asarray(raw["particles"], dtype=jnp.float64)
    with np.load(BANK, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["trial_ids"], np.arange(TRIALS))
        bank = ObservationTrialBank(
            jnp.asarray(raw["sample_indices"], dtype=jnp.int32),
            jnp.asarray(raw["detector_z"], dtype=jnp.float64),
        )
    experiments, contexts = [], []
    for seed in config["reference_replicates"]["training_seeds"]:
        root = reference_root / f"reference_seed_{seed}"
        if load_json(root / "qualification_receipt.json").get("status") != "PASS":
            raise RuntimeError(f"reference {seed} is not qualified")
        reference = BoxTransformedReferenceFlow.from_npz(
            root / "reference.npz",
            substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
        )
        with np.load(root / "reference_bank.npz", allow_pickle=False) as raw:
            np.testing.assert_allclose(raw["times"], times, rtol=0, atol=0)
            nodes = jnp.asarray(raw["nodes"], dtype=jnp.float64)
            velocity = jnp.asarray(raw["velocity"], dtype=jnp.float64)
            weights = jnp.asarray(raw["weights"], dtype=jnp.float64)
        experiment = VortexExperiment(
            cfg,
            reference,
            truth_particles=truth,
            reference_nodes=nodes,
            reference_velocity=velocity,
            reference_weights=weights,
        )
        experiments.append(experiment)
        contexts.append(DevelopmentContext(experiment, bank, times, cfg, root, BANK, NAMESPACE))
    return experiments, contexts, bank


def design_geometries() -> list[tuple[str, list[float]]]:
    designs = [("law", load_json(BRANCH / "law" / "current_result.json")["winner"]["eta"])]
    for tag in ("0p5", "1p0", "2p0"):
        receipt = load_json(BRANCH / "allowances" / f"risk_{tag}pct" / "full.json")
        designs.append((f"full_{tag}", receipt["winner"]["eta"]))
    return [(name, list(map(float, eta))) for name, eta in designs]


def trial_valid(row: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        np.isfinite(row["action"])
        and row["maximum_calibration_residual"] <= float(gates["maximum_finite_calibration_residual"])
        and row["minimum_ess_fraction"] >= float(gates["minimum_ess_fraction"])
        and row["maximum_mass_error"] <= float(gates["maximum_mass_absolute_error"])
        and row["maximum_source_compatibility_absolute"] <= float(gates["maximum_source_compatibility_absolute"])
        and row["maximum_poisson_relative_residual"] <= float(gates["maximum_poisson_relative_residual"])
        and row["maximum_component_count"] == int(gates["required_conductive_component_count"])
        and row["solver_converged"]
        and row["component_compatible"]
        and row["strictly_positive_q"]
        and row["maximum_full_moment_rate_residual"] <= float(gates["maximum_full_moment_rate_residual"])
        and row["maximum_tangent_moment_rate_residual"] <= float(gates["maximum_tangent_moment_rate_residual"])
        and row["maximum_hidden_nullspace_residual"] <= float(gates["maximum_hidden_nullspace_residual"])
        and row["maximum_orthogonality_absolute"] <= float(gates["maximum_orthogonality_absolute"])
        and row["maximum_pythagorean_absolute"] <= float(gates["maximum_pythagorean_absolute"])
        and row["maximum_raw_hierarchy_violation"] <= float(gates["maximum_raw_hierarchy_violation"])
    )


def evaluation_hashes() -> dict[str, str]:
    values = {"execution_receipt": sha256_file(EXECUTION), "bank": sha256_file(BANK), "bank_receipt": sha256_file(BANK_RECEIPT)}
    for name, path in frozen_paths().items():
        values[name] = sha256_file(path)
    return values


def evaluate_reference(
    harness: Any,
    evaluator: Any,
    eta: list[float],
    design: str,
    reference_index: int,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    destination = RUN / "action_evaluations" / design / f"reference_{reference_index}.json"
    if destination.exists():
        saved = load_json(destination)
        if saved.get("input_sha256") != input_hashes or saved.get("eta") != eta:
            raise RuntimeError(f"stale confirmatory cell: {destination}")
        return saved
    started = time.perf_counter()
    context = evaluator.contexts[reference_index]
    grid = harness.make_grid(256, 128)
    static = evaluator._fiber_static(context, eta)
    features = context.exp.family.features(grid.points(), jnp.asarray(eta, dtype=jnp.float64))
    rows: list[dict[str, Any]] = []
    for begin in range(0, TRIALS, 4):
        trial_ids = list(range(begin, min(begin + 4, TRIALS)))
        states = [evaluator._hard_fiber_state_from_static(context, eta, trial, static) for trial in trial_ids]
        raster = evaluator._raster_state_chunk(states, reference_index, context, grid)
        batch_count = len(states)
        solved = harness.solve_v2(
            raster["q"].reshape((batch_count * 21, grid.ny, grid.nx)),
            raster["source"].reshape((batch_count * 21, grid.ny, grid.nx)),
            grid,
        )
        action_matrix = np.asarray(solved.action, dtype=np.float64).reshape((batch_count, 21))
        potential = np.asarray(solved.potential, dtype=np.float64).reshape((batch_count, 21, grid.ny, grid.nx))
        poisson = np.asarray(solved.relative_residual, dtype=np.float64).reshape((batch_count, 21))
        compatibility = np.asarray(solved.maximum_component_compatibility_residual, dtype=np.float64).reshape((batch_count, 21))
        component_count = np.asarray(solved.component_count).reshape((batch_count, 21))
        converged = np.asarray(solved.solver_converged).reshape((batch_count, 21))
        compatible = np.asarray(solved.compatible).reshape((batch_count, 21))
        for local_index, (trial, state) in enumerate(zip(trial_ids, states, strict=True)):
            decomp = harness.raster_tangent_projection(
                jnp.asarray(potential[local_index], dtype=jnp.float64),
                jnp.asarray(raster["q"][local_index], dtype=jnp.float64),
                -jnp.asarray(raster["source"][local_index], dtype=jnp.float64),
                features,
                dx=float(grid.dx),
                cell_area=float(grid.cell_area),
                pinv_rcond=1e-10,
                operator_floor_rel=0.0,
                gauge_strength=0.0,
                source_is_density=True,
            )
            action_by_time = action_matrix[local_index]
            row = {
                "trial": int(trial),
                "action": float(np.sum(evaluator.weights * action_by_time)),
                "action_by_time": action_by_time.tolist(),
                "maximum_calibration_residual": float(np.max(state.calibration_residual)),
                "minimum_ess_fraction": float(np.min(state.ess_fraction)),
                "maximum_mass_error": float(np.max(np.abs(np.sum(raster["mass"][local_index], axis=(-2, -1)) - 1.0))),
                "maximum_source_compatibility_absolute": float(np.max(np.abs(np.sum(raster["source"][local_index], axis=(-2, -1)) * grid.cell_area))),
                "maximum_poisson_relative_residual": float(np.max(poisson[local_index])),
                "maximum_component_compatibility_residual": float(np.max(compatibility[local_index])),
                "maximum_component_count": int(np.max(component_count[local_index])),
                "solver_converged": bool(np.all(converged[local_index])),
                "component_compatible": bool(np.all(compatible[local_index])),
                "strictly_positive_q": bool(np.all(raster["q"][local_index] > 0.0)),
                "maximum_full_moment_rate_residual": float(np.max(np.linalg.norm(np.asarray(decomp.full_moment_residual), axis=-1))),
                "maximum_tangent_moment_rate_residual": float(np.max(np.linalg.norm(np.asarray(decomp.tangent_moment_residual), axis=-1))),
                "maximum_hidden_nullspace_residual": float(np.max(np.linalg.norm(np.asarray(decomp.hidden_moment_residual), axis=-1))),
                "maximum_orthogonality_absolute": float(np.max(np.abs(np.asarray(decomp.tangent_hidden_inner_product)))),
                "maximum_pythagorean_absolute": float(np.max(np.abs(np.asarray(decomp.pythagorean_residual)))),
                "maximum_raw_hierarchy_violation": float(np.max(np.asarray(decomp.hierarchy_raw_violation))),
            }
            row["valid"] = trial_valid(row, evaluator.gates)
            rows.append(row)
        if begin == 0 or begin + 4 == TRIALS or (begin + 4) % 32 == 0:
            print(
                f"[confirmatory {design} reference {reference_index}] {begin + 4}/{TRIALS} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    result = {
        "schema_version": 1,
        "status": "COMPLETE" if all(row["valid"] for row in rows) else "NUMERICAL_INVALID",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_ONLY",
        "design": design,
        "reference_index": reference_index,
        "eta": eta,
        "grid": [256, 128],
        "trials": TRIALS,
        "input_sha256": input_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    atomic_json(destination, result)
    return result


def evaluate_actions() -> None:
    verify_bank()
    input_hashes = evaluation_hashes()
    harness = selection_runner.load_harness()
    config, _overlay = load_resolved_config()
    _manifest, bandwidth = selection_runner.configure_harness(harness, config)
    install_parallel_exact_solver(harness)
    experiments, contexts, bank = load_experiments(config)
    evaluator = harness.Evaluator(config, experiments, contexts, bank, bandwidth)
    evaluator.cache = RUN / "candidate_cache"
    evaluator.kernel_cache = SELECTION / "reflected_kernel_cache"
    jobs = [(design, eta, reference) for design, eta in design_geometries() for reference in range(3)]
    print(f"[confirmatory exact evaluation] 0/{len(jobs)} reference-design cells", flush=True)

    def run(job):
        design, eta, reference = job
        return evaluate_reference(harness, evaluator, eta, design, reference, input_hashes)

    with ThreadPoolExecutor(max_workers=WORKERS) as workers:
        results = list(workers.map(run, jobs))
    if any(result["status"] != "COMPLETE" for result in results):
        raise RuntimeError("confirmatory action evaluation encountered a numerical-invalid trial")
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE_REDUCED_CONFIRMATORY_EXACT_ACTION_EVALUATION",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_ONLY",
        "input_sha256": input_hashes,
        "designs": list(DESIGNS),
        "reference_count": 3,
        "trials": TRIALS,
        "total_trial_evaluations": 12 * TRIALS,
        "evaluation_receipts": {
            str(path.relative_to(RUN)): sha256_file(path)
            for path in sorted((RUN / "action_evaluations").glob("*/reference_*.json"))
        },
    }
    atomic_json(ACTION_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def distribution(array: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return {"mean": mean, "standard_error": se, "relative_standard_error": se / abs(mean)}


def analyze() -> None:
    verify_bank()
    if not ACTION_RECEIPT.is_file():
        raise RuntimeError("exact action evaluation is incomplete")
    master = load_json(ACTION_RECEIPT)
    if master.get("status") != "COMPLETE_REDUCED_CONFIRMATORY_EXACT_ACTION_EVALUATION":
        raise RuntimeError("invalid exact action master receipt")
    actions = np.empty((4, 3, TRIALS), dtype=np.float64)
    all_rows: list[dict[str, Any]] = []
    for design_index, design in enumerate(DESIGNS):
        for reference in range(3):
            path = RUN / "action_evaluations" / design / f"reference_{reference}.json"
            relative = str(path.relative_to(RUN))
            if sha256_file(path) != master["evaluation_receipts"].get(relative):
                raise RuntimeError(f"confirmatory action-cell hash mismatch: {relative}")
            payload = load_json(path)
            rows = payload["rows"]
            if payload.get("status") != "COMPLETE" or [row["trial"] for row in rows] != list(range(TRIALS)):
                raise RuntimeError(f"invalid confirmatory cell: {relative}")
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
    for begin in range(0, 100000, 128):
        count = min(128, 100000 - begin)
        indices = rng.integers(0, TRIALS, size=(count, TRIALS), dtype=np.int32)
        digest.update(indices.tobytes(order="C"))
        law_means = np.mean(law[:, indices], axis=-1).T
        full_means = np.mean(full[:, :, indices], axis=-1).transpose(2, 0, 1)
        bootstrap[begin : begin + count] = 1.0 - full_means / law_means[:, :, None]
    deviation = np.max(np.abs(bootstrap - observed[None]), axis=(1, 2))
    critical = float(np.quantile(deviation, 0.95))
    pointwise = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    bootstrap_path = RUN / "bootstrap_effects.npz"
    temporary = RUN / "bootstrap_effects.tmp.npz"
    np.savez_compressed(temporary, effects=bootstrap)
    os.replace(temporary, bootstrap_path)
    summaries = {
        design: [distribution(actions[index, reference]) for reference in range(3)]
        for index, design in enumerate(DESIGNS)
    }
    max_rse = max(row["relative_standard_error"] for group in summaries.values() for row in group)
    lower, upper = observed - critical, observed + critical
    gates = {
        "exactly_three_qualified_references": True,
        "exactly_1024_shared_trials": True,
        "all_12288_exact_action_evaluations_numerically_valid": len(all_rows) == 12288 and all(row["valid"] for row in all_rows),
        "all_nine_simultaneous_lower_bounds_strictly_positive": bool(np.all(lower > 0.0)),
        "maximum_simultaneous_half_width_at_most_0p05": critical <= 0.05,
        "all_relative_standard_errors_at_most_0p10": max_rse <= 0.10,
        "no_outcome_dependent_confirmatory_amendment": True,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "experiment_id": "V2.1-C3",
        "data_role": "FINAL_CONFIRMATORY_0P5_TO_2PCT_ONLY",
        "claim_scope": "truncated_pareto_0p5_to_2pct_only",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowance_percentages": list(ALLOWANCES),
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
        "original_3_to_5pct_selection_remains_paused": True,
    }
    atomic_json(INFERENCE, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def compact_risk_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial": int(row["trial"]),
        "valid": bool(row["valid"]),
        "invalid_reason": row.get("invalid_reason"),
        "law_risk": float(row["law_risk"]),
        "maximum_calibration_residual": float(row["max_calibration_residual"]),
        "minimum_ess_fraction": float(row["min_ess_fraction"]),
        "spline_residual_sum_squares": float(row["spline_residual_sum_squares"]),
        "spline_roughness": float(row["spline_roughness"]),
    }


def evaluate_risk() -> None:
    verify_bank()
    if not INFERENCE.is_file():
        raise RuntimeError("primary confirmatory inference must complete before secondary risk")
    input_hashes = {**evaluation_hashes(), "inference": sha256_file(INFERENCE)}
    config, _overlay = load_resolved_config()
    experiments, _contexts, bank = load_experiments(config)
    jobs = [(design, eta, reference) for design, eta in design_geometries() for reference in range(3)]

    def evaluate(job):
        design, eta, reference = job
        destination = RUN / "risk_evaluations" / design / f"reference_{reference}.json"
        if destination.exists():
            saved = load_json(destination)
            if saved.get("input_sha256") != input_hashes or saved.get("eta") != eta:
                raise RuntimeError(f"stale confirmatory risk cell: {destination}")
            return saved
        started = time.perf_counter()
        result = experiments[reference].exact_finite_result(jnp.asarray(eta, dtype=jnp.float64), bank)
        rows = [compact_risk_row(row) for row in result["rows"]]
        valid = bool(result["valid"] and len(rows) == TRIALS and [row["trial"] for row in rows] == list(range(TRIALS)) and all(row["valid"] for row in rows))
        payload = {
            "schema_version": 1,
            "status": "COMPLETE" if valid else "NUMERICAL_INVALID",
            "data_role": "FINAL_CONFIRMATORY_SECONDARY_RISK",
            "design": design,
            "reference_index": reference,
            "eta": eta,
            "trials": len(rows),
            "mean_finite_law_risk": float(result["value"]),
            "elapsed_seconds": time.perf_counter() - started,
            "input_sha256": input_hashes,
            "rows": rows,
        }
        atomic_json(destination, payload)
        print(f"[confirmatory risk] {design}/reference_{reference} status={payload['status']} elapsed={payload['elapsed_seconds']:.1f}s", flush=True)
        return payload

    print(f"[confirmatory risk] 0/{len(jobs)} cells", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as workers:
        results = list(workers.map(evaluate, jobs))
    if any(result["status"] != "COMPLETE" for result in results):
        raise RuntimeError("confirmatory finite-risk cross-evaluation failed")
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE_REDUCED_CONFIRMATORY_FINITE_RISK_EVALUATION",
        "data_role": "FINAL_CONFIRMATORY_SECONDARY_RISK",
        "input_sha256": input_hashes,
        "cells": {
            str(path.relative_to(RUN)): sha256_file(path)
            for path in sorted((RUN / "risk_evaluations").glob("*/reference_*.json"))
        },
        "cell_count": len(results),
        "trial_count": sum(len(result["rows"]) for result in results),
    }
    atomic_json(RISK_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def main() -> int:
    stage = parse_args().stage
    if stage in ("freeze", "all"):
        freeze_execution()
    if stage in ("bank", "all"):
        generate_bank()
    if stage in ("evaluate", "all"):
        evaluate_actions()
    if stage in ("analyze", "all"):
        analyze()
    if stage in ("risk", "all"):
        evaluate_risk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
