"""Prospective common-task qualification for the three B1 Law geometries.

The three reference checkpoints receive independent Law fits, but every fit and
every Galerkin certificate uses the same frozen candidate generator, matched
initial conditions, dictionary, discretization ladder, and certificate gates.
The resulting handoff is intended to be consumed by a later per-seed Pareto
protocol; it never edits or relabels an existing Pareto result.
"""

from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from . import official_b1_pareto as single
from . import pareto_v2_selection as selection_engine
from . import resolution_study
from . import three_reference_pareto as ensemble
from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import GalerkinCertificateThresholds, prefix_dictionary
from .galerkin_only_data import selection_risk
from .pareto_v3_common import file_sha256, payload_sha256
from .production_galerkin import (
    assemble_hybrid_system,
    audit_hybrid_solutions,
    make_basis_evaluators,
)
from .production_gradient import production_hybrid_envelope_value_and_grad


ROOT = Path(__file__).resolve().parent
VERSION = "skyrmion_b1_three_law_common_task_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
HANDOFF_PATH = OUTPUT_ROOT / "pareto_handoff.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"

SOURCE_ROOT = ROOT / "outputs" / ensemble.VERSION
FLOW_IDS = tuple(ensemble.FLOW_IDS)
FLOW_PATHS = dict(ensemble.FLOW_PATHS)
FLOW_SHA256 = dict(ensemble.FLOW_SHA256)
K_LADDER = tuple(resolution_study.K_LADDER)
RANK_TOLERANCES = tuple(resolution_study.RANK_TOLERANCES)
DEFAULT_RANK_TOLERANCE = 1.0e-12
DEVELOPMENT_TRAIN = "search_train"
DEVELOPMENT_AUDIT = "periodic_audit"
CONFIRMATION_TRAIN = "authoritative_train"
CONFIRMATION_AUDIT = "authoritative_audit"

QUALIFICATION_THRESHOLDS = {
    "rank_tolerance_action_spread": 0.02,
    "rank_tolerance_energy_spread": 0.01,
    "rank_tolerance_gradient_cosine_minimum": 0.995,
    "neighbor_action_relative_tolerance": 0.05,
    "neighbor_gradient_cosine_minimum": 0.995,
    "neighbor_gradient_relative_tolerance": 0.10,
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _atomic_bytes(path: Path, value: bytes) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"qualification output escaped {root}: {resolved}")
    if resolved.exists() and resolved.read_bytes() == value:
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _law_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "laws" / flow_id / "official_law.json"


def _case_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "development" / f"{flow_id}.json"


def _confirmation_path(flow_id: str) -> Path:
    return OUTPUT_ROOT / "confirmation" / f"{flow_id}.json"


def relative_change(high: float, low: float) -> float:
    return abs(float(high) - float(low)) / max(abs(float(high)), 1.0e-12)


def gradient_comparison(high: Iterable[float], low: Iterable[float]) -> dict[str, float]:
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    high_norm = float(np.linalg.norm(high_array))
    low_norm = float(np.linalg.norm(low_array))
    return {
        "cosine": float(
            np.dot(high_array, low_array) / max(high_norm * low_norm, 1.0e-30)
        ),
        "relative_difference": float(
            np.linalg.norm(high_array - low_array) / max(high_norm, 1.0e-12)
        ),
    }


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    source_protocol = _read(SOURCE_ROOT / "protocol.json")
    source_manifest = _read(SOURCE_ROOT / "banks" / "manifest.json")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "status": "FROZEN_BEFORE_LAW_FITTING",
        "purpose": (
            "choose one Galerkin discretization that certifies the independently "
            "fitted Law geometry for every B1 reference-flow seed"
        ),
        "flow_ids": list(FLOW_IDS),
        "reference_checkpoint_sha256": {
            flow_id: FLOW_SHA256[flow_id] for flow_id in FLOW_IDS
        },
        "source_three_reference_protocol_sha256": source_protocol["protocol_sha256"],
        "source_bank_manifest_sha256": file_sha256(SOURCE_ROOT / "banks" / "manifest.json"),
        "matched_initial_configurations_across_flows": bool(
            source_manifest["matched_initial_configurations_across_flows"]
        ),
        "law_fit": {
            "mode": "independent fit for each reference flow",
            "common_algorithm": "official B1 frozen pool plus three refinement rounds",
            "common_design_truth_sha256": file_sha256(
                SOURCE_ROOT / "design_truth" / "design_truth.npz"
            ),
            "law_search_role": "law_search",
            "risk_anchor_role": "risk_anchor",
            "support_required": True,
        },
        "development_qualification": {
            "train_role": DEVELOPMENT_TRAIN,
            "audit_role": DEVELOPMENT_AUDIT,
            "K_ladder": list(K_LADDER),
            "rank_tolerances": list(RANK_TOLERANCES),
            "default_rank_tolerance": DEFAULT_RANK_TOLERANCE,
            "selection_rule": (
                "smallest K for which all three diagonal Law/flow cases have a "
                "complete certificate at every rank tolerance, satisfy tolerance "
                "robustness, and satisfy neighboring-K action/gradient stability"
            ),
            **QUALIFICATION_THRESHOLDS,
        },
        "authoritative_confirmation": {
            "train_role": CONFIRMATION_TRAIN,
            "audit_role": CONFIRMATION_AUDIT,
            "required_for_every_flow": True,
            "rank_tolerance": DEFAULT_RANK_TOLERANCE,
        },
        "pareto_contract": {
            "use_recommended_K_and_rank_tolerance_unchanged_for_every_seed_and_allowance": True,
            "include_corresponding_Law_as_mandatory_Full_candidate": True,
            "fail_closed_if_Law_confirmation_fails": True,
            "risk_rule": "R_seed(eta) <= (1+p/100) R_seed(Law_seed)",
            "guarantee": (
                "each nonnegative allowance has a certified feasible Full fallback; "
                "the selected Full action cannot exceed the same-metric Law action "
                "beyond replacement tolerance"
            ),
        },
        "certificate_thresholds": copy.deepcopy(
            cfg["production_galerkin"]["certificate_thresholds"]
        ),
        "algebra_thresholds": {
            key: cfg["production_galerkin"][key]
            for key in (
                "maximum_range_residual",
                "maximum_stationarity_residual",
                "maximum_identity_relerr",
                "maximum_symmetry_residual",
                "minimum_rank_fraction",
                "maximum_retained_condition",
            )
        },
        "dictionary_sha256": file_sha256(resolution_study.DICTIONARY_PATH),
        "source_sha256": file_sha256(Path(__file__)),
        "validation_accessed": False,
    }
    payload["protocol_sha256"] = payload_sha256(payload)
    return payload


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(cfg)
    if PROTOCOL_PATH.exists():
        old = _read(PROTOCOL_PATH)
        if old != payload:
            raise RuntimeError("three-Law qualification protocol changed after freezing")
        return old
    _atomic_json(PROTOCOL_PATH, payload)
    _atomic_text(OUTPUT_ROOT / "protocol_hash.txt", payload["protocol_sha256"] + "\n")
    return payload


def _activate_source(flow_id: str) -> None:
    """Point the shared engine at one flow while retaining the matched banks."""
    if flow_id not in FLOW_IDS:
        raise ValueError(f"unknown flow {flow_id}")
    ensemble.VERSION = "skyrmion_b1_galerkin_pareto_3references_v1"
    ensemble.OUTPUT_ROOT = SOURCE_ROOT
    ensemble.PROTOCOL_PATH = SOURCE_ROOT / "protocol.json"
    ensemble.PROTOCOL_HASH_PATH = SOURCE_ROOT / "protocol_hash.txt"
    ensemble.DESIGN_PATH = SOURCE_ROOT / "design_truth" / "design_truth.npz"
    ensemble.DESIGN_RECORD = SOURCE_ROOT / "design_truth" / "manifest.json"
    ensemble.ARTIFACT_DIR = SOURCE_ROOT / "artifacts"
    ensemble.FLOW_IDS = (flow_id,)
    ensemble.FLOW_PATHS = {flow_id: FLOW_PATHS[flow_id]}
    ensemble.FLOW_SHA256 = {flow_id: FLOW_SHA256[flow_id]}
    ensemble.require_protocol = lambda _cfg: _read(SOURCE_ROOT / "protocol.json")
    # Qualification consumes the already sealed matched banks.  Prevent the
    # single-flow view from rewriting their original three-flow manifest.
    ensemble.generate_banks = lambda _cfg, _progress=None: _read(
        SOURCE_ROOT / "banks" / "manifest.json"
    )
    ensemble._SHARED_SELECTION.clear()
    ensemble._FLOW_SELECTION_SHARED.clear()


def fit_law(cfg: dict[str, Any], flow_id: str,
            progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Fit one Law without regenerating or mutating the shared source banks."""
    destination = _law_path(flow_id)
    if destination.exists():
        return _read(destination)
    _activate_source(flow_id)
    ensemble._activate()
    law_root = destination.parent
    single.OUTPUT_ROOT = law_root
    single.LAW_PATH = destination
    single.LAW_POOL_PATH = law_root / "search_pool.json"
    single.LAW_RESULTS_PATH = law_root / "search_results.json"
    single.DESIGN_PATH = SOURCE_ROOT / "design_truth" / "design_truth.npz"
    single.CHECKPOINT = ensemble.FLOW_PATHS[flow_id]
    single.CHECKPOINT_SHA256 = ensemble.FLOW_SHA256[flow_id]
    single.generate_banks = lambda _cfg, _progress=None: _read(
        SOURCE_ROOT / "banks" / "manifest.json"
    )
    single.require_protocol = lambda _cfg: _read(SOURCE_ROOT / "protocol.json")
    single._bank_path = lambda label: ensemble._bank_path(label, flow_id)
    single._evaluate = lambda etas, local_cfg, label: ensemble.evaluate_references(
        etas, local_cfg, label
    )
    result = single.reconstruct_law(cfg, progress)
    if result["checkpoint_sha256"] != FLOW_SHA256[flow_id]:
        raise RuntimeError(f"Law checkpoint mismatch for {flow_id}")
    return result


def fit_all_laws(cfg: dict[str, Any],
                 progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    protocol = freeze_protocol(cfg)
    rows = {}
    for flow_id in FLOW_IDS:
        if progress:
            progress(f"Law fit {flow_id}")
        law = fit_law(cfg, flow_id, progress)
        rows[flow_id] = {
            "eta": law["eta_Law_official"],
            "risk": law["R_Law_official"],
            "sha256": file_sha256(_law_path(flow_id)),
        }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "protocol_sha256": protocol["protocol_sha256"],
        "laws": rows,
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "laws" / "summary.json", result)
    return result


def _evaluate_development_ladder(cfg: dict[str, Any], flow_id: str,
                                 progress: Callable[[str], None] | None) -> dict[str, Any]:
    output = _case_path(flow_id)
    if output.exists():
        return _read(output)
    _activate_source(flow_id)
    law = _read(_law_path(flow_id))
    data = ensemble.selection_data_for_flow(
        cfg, DEVELOPMENT_TRAIN, DEVELOPMENT_AUDIT, flow_id
    )
    dictionary = selection_engine.FullContext(cfg, data).dictionary
    eta = wrap_periodic(
        jnp.asarray(law["eta_Law_official"], dtype=jnp.float64),
        data.selection_problem.family,
    )
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    train_state = forcing_state(eta, data.selection_problem, data.train_bank, reconstruction)
    audit_state = forcing_state(eta, data.selection_problem, data.audit_bank, reconstruction)
    evaluators = make_basis_evaluators(
        dictionary, int(data.train_bank.configurations.shape[0])
    )
    full_system = assemble_hybrid_system(
        dictionary,
        data.train_bank,
        train_state.projection.weights,
        train_state.forcing,
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
        evaluators=evaluators,
    )
    solves: list[Any] = []
    metadata: list[tuple[int, float, Any]] = []
    aggregates: list[dict[str, Any]] = []
    algebra_rows: list[dict[str, Any]] = []
    padded: list[jax.Array] = []
    for K in K_LADDER:
        system = resolution_study._prefix_system(full_system, K)
        for tolerance in RANK_TOLERANCES:
            solve = rank_aware_quadratic_solve(
                system.gram, system.load, relative_rank_tolerance=tolerance
            )
            aggregate = aggregate_quadratic_values(
                solve, data.selection_problem.time_weights
            )
            solves.append(solve)
            metadata.append((K, tolerance, system))
            aggregates.append(aggregate)
            algebra_rows.append(
                resolution_study._algebra(cfg, system, solve, aggregate, K)
            )
            padded.append(
                jnp.pad(
                    solve.coefficients,
                    ((0, 0), (0, int(dictionary.size) - K)),
                )
            )
    coefficients = jnp.stack(padded)
    certificate_adapter = SimpleNamespace(
        selection_problem=data.selection_problem,
        ritz_audit_bank=data.audit_bank,
    )
    certificates = audit_hybrid_solutions(
        dictionary,
        coefficients,
        certificate_adapter,
        eta,
        reconstruction,
        audit_state,
        GalerkinCertificateThresholds(
            **cfg["production_galerkin"]["certificate_thresholds"]
        ),
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )
    potentials, kinetics = resolution_study._batch_potential_rows(
        dictionary,
        coefficients,
        data.train_bank,
        evaluators,
        int(cfg["production_galerkin"]["chunk_size"]),
    )
    train_forcing = resolution_study._forcing_state_payload(
        train_state, data.selection_problem
    )
    audit_forcing = resolution_study._forcing_state_payload(
        audit_state, data.selection_problem
    )
    adapter = SimpleNamespace(
        selection_problem=data.selection_problem,
        ritz_train_bank=data.train_bank,
    )
    rows = []
    for index, ((K, tolerance, _), solve, aggregate, algebra, certificate) in enumerate(
        zip(metadata, solves, aggregates, algebra_rows, certificates, strict=True)
    ):
        value, gradient = production_hybrid_envelope_value_and_grad(
            eta, solve.coefficients, adapter, potentials[index], kinetics[index]
        )
        complete = bool(
            data.selection_problem.family.geometry_valid(eta)
            and train_forcing["valid"]
            and audit_forcing["valid"]
            and algebra["valid"]
            and certificate["valid"]
        )
        rows.append(
            {
                "K": K,
                "rank_tolerance": tolerance,
                "scientific_risk": float(selection_risk(eta, data)),
                "train_action": float(value),
                "quadratic_train_action": float(aggregate["action"]),
                "audit_action": float(certificate["action"]),
                "gradient": np.asarray(gradient).tolist(),
                "gradient_norm": float(jnp.linalg.norm(gradient)),
                "algebra": algebra,
                "heldout_certificate": certificate,
                "train_forcing": train_forcing,
                "audit_forcing": audit_forcing,
                "complete_certificate": complete,
            }
        )
    result = {
        "schema_version": 1,
        "flow_id": flow_id,
        "law_sha256": file_sha256(_law_path(flow_id)),
        "eta": law["eta_Law_official"],
        "rows": rows,
        "validation_accessed": False,
    }
    _atomic_json(output, result)
    if progress:
        passed = sum(row["complete_certificate"] for row in rows)
        progress(f"development ladder {flow_id}: {passed}/{len(rows)} complete")
    del coefficients, potentials, kinetics, full_system, dictionary, data
    gc.collect()
    jax.clear_caches()
    return result


def qualify_rows(results: list[dict[str, Any]]) -> dict[str, Any]:
    settings = QUALIFICATION_THRESHOLDS
    candidates = []
    for K in K_LADDER:
        flow_checks = []
        for result in results:
            tolerance_rows = [row for row in result["rows"] if row["K"] == K]
            default = next(
                row
                for row in tolerance_rows
                if row["rank_tolerance"] == DEFAULT_RANK_TOLERANCE
            )
            actions = [row["train_action"] for row in tolerance_rows]
            energies = [
                row["heldout_certificate"]["maximum_energy_residual"]
                for row in tolerance_rows
            ]
            cosines = [
                gradient_comparison(right["gradient"], left["gradient"])["cosine"]
                for left, right in zip(tolerance_rows[:-1], tolerance_rows[1:])
            ]
            robust = bool(
                all(row["complete_certificate"] for row in tolerance_rows)
                and (max(actions) - min(actions))
                / max(abs(default["train_action"]), 1.0e-12)
                <= settings["rank_tolerance_action_spread"]
                and max(energies) - min(energies)
                <= settings["rank_tolerance_energy_spread"]
                and min(cosines, default=1.0)
                >= settings["rank_tolerance_gradient_cosine_minimum"]
            )
            neighbor = None
            if K != K_LADDER[-1]:
                next_K = K_LADDER[K_LADDER.index(K) + 1]
                next_row = next(
                    row
                    for row in result["rows"]
                    if row["K"] == next_K
                    and row["rank_tolerance"] == DEFAULT_RANK_TOLERANCE
                )
                neighbor = {
                    "action_relative_change": relative_change(
                        next_row["train_action"], default["train_action"]
                    ),
                    **gradient_comparison(next_row["gradient"], default["gradient"]),
                }
            stable = bool(
                neighbor is None
                or (
                    neighbor["action_relative_change"]
                    <= settings["neighbor_action_relative_tolerance"]
                    and neighbor["cosine"]
                    >= settings["neighbor_gradient_cosine_minimum"]
                    and neighbor["relative_difference"]
                    <= settings["neighbor_gradient_relative_tolerance"]
                )
            )
            flow_checks.append(
                {
                    "flow_id": result["flow_id"],
                    "default_complete_certificate": default["complete_certificate"],
                    "robust_to_rank_tolerance": robust,
                    "stable_neighbor": stable,
                    "neighbor": neighbor,
                    "default_energy_residual": default["heldout_certificate"][
                        "maximum_energy_residual"
                    ],
                    "default_action": default["train_action"],
                    "default_algebra_valid": default["algebra"]["valid"],
                }
            )
        qualified = all(
            row["default_complete_certificate"]
            and row["robust_to_rank_tolerance"]
            and row["stable_neighbor"]
            for row in flow_checks
        )
        candidates.append(
            {"K": K, "qualified": qualified, "flow_checks": flow_checks}
        )
    chosen = next((row["K"] for row in candidates if row["qualified"]), None)
    return {
        "qualification_candidates": candidates,
        "recommended_K": chosen,
        "recommended_rank_tolerance": (
            DEFAULT_RANK_TOLERANCE if chosen is not None else None
        ),
        "development_qualified": chosen is not None,
    }


def run_development(cfg: dict[str, Any],
                    progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    protocol = freeze_protocol(cfg)
    fit_all_laws(cfg, progress)
    results = [
        _evaluate_development_ladder(cfg, flow_id, progress) for flow_id in FLOW_IDS
    ]
    qualification = qualify_rows(results)
    payload = {
        "schema_version": 1,
        "status": "QUALIFIED" if qualification["development_qualified"] else "NOT_QUALIFIED",
        "protocol_sha256": protocol["protocol_sha256"],
        "flows": [
            {
                "flow_id": row["flow_id"],
                "law_sha256": row["law_sha256"],
                "development_sha256": file_sha256(_case_path(row["flow_id"])),
            }
            for row in results
        ],
        **qualification,
        "validation_accessed": False,
    }
    _atomic_json(OUTPUT_ROOT / "development" / "summary.json", payload)
    return payload


def run_confirmation(cfg: dict[str, Any], development: dict[str, Any],
                     progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    K = development["recommended_K"]
    tolerance = development["recommended_rank_tolerance"]
    if K is None or tolerance is None:
        return {
            "status": "NOT_RUN",
            "reason": "no common development-qualified discretization",
            "passed": False,
            "rows": [],
        }
    rows = []
    for flow_id in FLOW_IDS:
        output = _confirmation_path(flow_id)
        if output.exists():
            row = _read(output)
        else:
            _activate_source(flow_id)
            law = _read(_law_path(flow_id))
            data = ensemble.selection_data_for_flow(
                cfg, CONFIRMATION_TRAIN, CONFIRMATION_AUDIT, flow_id
            )
            context = selection_engine.FullContext(cfg, data)
            prefix = prefix_dictionary(context.dictionary, K)
            evaluators = make_basis_evaluators(
                prefix, int(data.train_bank.configurations.shape[0])
            )
            case = resolution_study.evaluate_case(
                cfg,
                data,
                context.dictionary,
                data.train_bank,
                data.audit_bank,
                law["eta_Law_official"],
                K=K,
                rank_tolerance=tolerance,
                evaluators=evaluators,
            )
            row = {
                "schema_version": 1,
                "flow_id": flow_id,
                "law_sha256": file_sha256(_law_path(flow_id)),
                **case,
                "validation_accessed": False,
            }
            _atomic_json(output, row)
            del context, data, prefix, evaluators
            gc.collect()
            jax.clear_caches()
        rows.append(row)
        if progress:
            progress(
                f"authoritative Law confirmation {flow_id}: "
                f"{'PASS' if row['complete_certificate'] else 'FAIL'}"
            )
    return {
        "status": "PASS" if all(row["complete_certificate"] for row in rows) else "FAIL",
        "passed": all(row["complete_certificate"] for row in rows),
        "K": K,
        "rank_tolerance": tolerance,
        "rows": rows,
        "validation_accessed": False,
    }


def _write_report(result: dict[str, Any]) -> None:
    development = result["development"]
    confirmation = result["confirmation"]
    lines = [
        "# Three-Law common-task Galerkin qualification",
        "",
        f"Status: **{result['status']}**",
        "",
        "Each B1 reference flow has its own independently fitted Law geometry. "
        "The three diagonal Law/flow cases use a common dictionary, K ladder, "
        "rank-tolerance ladder, matched initial conditions, and unchanged hard gates.",
        "",
        "## Development qualification",
        "",
        "| K | all three qualified | flow | complete | algebra | energy | neighbor stable | tolerance robust |",
        "|---:|:---:|:--|:---:|:---:|---:|:---:|:---:|",
    ]
    for candidate in development["qualification_candidates"]:
        for index, check in enumerate(candidate["flow_checks"]):
            lines.append(
                f"| {candidate['K'] if index == 0 else ''} | "
                f"{'YES' if candidate['qualified'] else 'NO' if index == 0 else ''} | "
                f"{check['flow_id']} | {'PASS' if check['default_complete_certificate'] else 'FAIL'} | "
                f"{'PASS' if check['default_algebra_valid'] else 'FAIL'} | "
                f"{check['default_energy_residual']:.6g} | "
                f"{'PASS' if check['stable_neighbor'] else 'FAIL'} | "
                f"{'PASS' if check['robust_to_rank_tolerance'] else 'FAIL'} |"
            )
    lines.extend(["", "## Authoritative confirmation", ""])
    if not confirmation["rows"]:
        lines.append(f"Not run: {confirmation['reason']}.")
    else:
        lines.extend(
            [
                f"Frozen setting: K={confirmation['K']}, rank tolerance={confirmation['rank_tolerance']:.1e}.",
                "",
                "| flow | Law risk | Full action | algebra | energy | complete |",
                "|:--|---:|---:|:---:|---:|:---:|",
            ]
        )
        for row in confirmation["rows"]:
            lines.append(
                f"| {row['flow_id']} | {row['scientific_risk']:.9g} | "
                f"{row['train_action']:.9g} | "
                f"{'PASS' if row['algebra']['valid'] else 'FAIL'} | "
                f"{row['heldout_certificate']['maximum_energy_residual']:.6g} | "
                f"{'PASS' if row['complete_certificate'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Pareto handoff",
            "",
            "A handoff is released only when all three authoritative Law certificates pass. "
            "Every later allowance must rescore and retain its seed's Law as a mandatory "
            "same-Full-metric candidate.",
        ]
    )
    _atomic_text(REPORT_PATH, "\n".join(lines) + "\n")


def run(cfg: dict[str, Any], progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    if RESULT_PATH.exists():
        return _read(RESULT_PATH)
    protocol = freeze_protocol(cfg)
    development = run_development(cfg, progress)
    confirmation = run_confirmation(cfg, development, progress)
    passed = bool(development["development_qualified"] and confirmation["passed"])
    result = {
        "schema_version": 1,
        "version": VERSION,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "protocol_sha256": protocol["protocol_sha256"],
        "development": development,
        "confirmation": confirmation,
        "validation_accessed": False,
    }
    _atomic_json(RESULT_PATH, result)
    if passed:
        handoff = {
            "schema_version": 1,
            "status": "FROZEN_FOR_PARETO",
            "qualification_result_sha256": file_sha256(RESULT_PATH),
            "protocol_sha256": protocol["protocol_sha256"],
            "K": confirmation["K"],
            "rank_tolerance": confirmation["rank_tolerance"],
            "dictionary_sha256": protocol["dictionary_sha256"],
            "laws": {
                row["flow_id"]: {
                    "eta": row["eta"],
                    "risk": row["scientific_risk"],
                    "law_sha256": row["law_sha256"],
                    "confirmation_sha256": file_sha256(
                        _confirmation_path(row["flow_id"])
                    ),
                }
                for row in confirmation["rows"]
            },
            "mandatory_candidate_rule": (
                "include Law_seed in every allowance and select by the same Full metric"
            ),
            "validation_accessed": False,
        }
        _atomic_json(HANDOFF_PATH, handoff)
    _write_report(result)
    return result


def load_default_config() -> dict[str, Any]:
    return single.official_config(load_config(ensemble.CONFIG_PATH))


__all__ = [
    "DEFAULT_RANK_TOLERANCE",
    "FLOW_IDS",
    "HANDOFF_PATH",
    "K_LADDER",
    "OUTPUT_ROOT",
    "PROTOCOL_PATH",
    "QUALIFICATION_THRESHOLDS",
    "RANK_TOLERANCES",
    "fit_all_laws",
    "freeze_protocol",
    "load_default_config",
    "qualify_rows",
    "run",
    "run_confirmation",
    "run_development",
]
