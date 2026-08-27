"""Bounded production workflows using only the fixed Galerkin Full solver."""

from __future__ import annotations

import json
import gc
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .full_gradient import forcing_state, reconstruct_moments
from .full_gradient import wrap_periodic
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import (
    GALERKIN_ONLY_ROOT, OLD_DICTIONARY, GalerkinOnlyContext,
    basis_memory_estimate, build_or_load_extended_dictionary, device_payload,
    require_galerkin_only_output_path, timed, timing_pair,
)
from .galerkin_only import _forcing_state_payload
from mfsi.cache import fingerprint
from .galerkin_only_data import (
    load_selection_galerkin_data, load_validation_galerkin_data,
    validation_risk,
)
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import load_dictionary
from .production_galerkin import (
    _normalized_chunk, assemble_hybrid_system, audit_hybrid_solutions,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _relative(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(
        np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1.0e-30)
    )


def galerkin_only_static_audit() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = [
        root / "galerkin_only.py",
        root / "galerkin_only_data.py",
        root / "galerkin_only_workflow.py",
        root / "galerkin_only_run.py",
    ]
    forbidden_calls = (
        "solve_" + "deep_ritz(", "audit_" + "deep_ritz(",
        "authoritative_" + "evaluate(",
        "run_production_" + "authoritative", "_cached_" + "authoritative(",
    )
    findings = []
    for path in files:
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        for token in forbidden_calls:
            if token in source:
                findings.append({"file": path.name, "token": token})
    return {
        "passed": not findings,
        "files": [str(path) for path in files if path.is_file()],
        "forbidden_call_findings": findings,
        "scientific_candidate_solver": "fixed-feature rank-aware Galerkin",
    }


def run_galerkin_only_benchmark(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, data_seconds = timed(lambda: load_selection_galerkin_data(cfg, artifact_dir))
    context, context_seconds = timed(lambda: GalerkinOnlyContext(
        cfg, artifact_dir, data, OLD_DICTIONARY,
        cache_dir=GALERKIN_ONLY_ROOT / "cache" / "K160",
        reuse_validated_K160=True,
    ))
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    value_timing = timing_pair(
        lambda: context.evaluate(eta, basis_size=160, with_gradient=False), 3
    )
    gradient_timing = timing_pair(
        lambda: context.evaluate(eta, basis_size=160, with_gradient=True), 5
    )
    evaluation = context.evaluate(eta, basis_size=160, with_gradient=True)
    problem, bank = data.selection_problem, data.train_bank
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    system = context.assemble(state.projection.weights, state.forcing, 160)
    solve = rank_aware_quadratic_solve(
        system.gram, system.load,
        relative_rank_tolerance=float(
            cfg["production_galerkin"]["relative_rank_tolerance"]
        ),
    )
    potential, kinetic = context.potential_rows(solve.coefficients, 160)
    stage_timings = {
        "selection_data_load": {"seconds": data_seconds},
        "context_initialization": {"seconds": context_seconds},
        "cached_basis_load": {
            "seconds": context.cache_info["load_seconds"],
            "cache_hit": context.cache_info["cache_hit"],
        },
        "projection_and_forcing": timing_pair(
            lambda: forcing_state(eta, problem, bank, reconstruction), 3
        ),
        "K_f_assembly": timing_pair(
            lambda: context.assemble(state.projection.weights, state.forcing, 160), 3
        ),
        "rank_aware_solve": timing_pair(
            lambda: rank_aware_quadratic_solve(
                system.gram, system.load,
                relative_rank_tolerance=float(
                    cfg["production_galerkin"]["relative_rank_tolerance"]
                ),
            ), 3,
        ),
        "action": timing_pair(
            lambda: aggregate_quadratic_values(solve, problem.time_weights)["action"], 3
        ),
        "fixed_coefficient_value_gradient": timing_pair(
            lambda: context._envelope_value_grad(eta, potential, kinetic), 3
        ),
        "complete_value_only": value_timing,
        "complete_value_gradient": gradient_timing,
    }
    certificate, certificate_seconds = timed(lambda: context.certify(evaluation))
    stage_timings["heldout_selection_certification"] = {
        "first_seconds": certificate_seconds
    }

    cpu_reference_path = (
        GALERKIN_ONLY_ROOT.parent / "fast_production_3pct" / "profiling" / "result.json"
    )
    cpu_reference = json.loads(cpu_reference_path.read_text(encoding="utf-8"))
    artifact_hash = file_sha256(artifact_dir / "isolated_artifact_manifest.json")
    if (
        cpu_reference["artifact_manifest_sha256"] != artifact_hash
        or cpu_reference["dictionary_sha256"] != file_sha256(OLD_DICTIONARY)
        or not cpu_reference["equivalence"]["passed"]
    ):
        raise RuntimeError("validated CPU K=160 reference is incompatible")
    cpu_action = float(cpu_reference["equivalence"]["fast_action"])
    cpu_gradient = cpu_reference["equivalence"]["fast_gradient"]
    action_relative = abs(float(evaluation.action) - cpu_action) / max(
        abs(cpu_action), 1.0e-30
    )
    gradient_relative = _relative(evaluation.gradient, cpu_gradient)
    platform_equivalence = {
        "passed": bool(action_relative <= 1.0e-10 and gradient_relative <= 1.0e-8),
        "cpu_reference": str(cpu_reference_path),
        "cpu_action": cpu_action,
        "current_action": float(evaluation.action),
        "action_relative_difference": action_relative,
        "cpu_gradient": cpu_gradient,
        "current_gradient": np.asarray(evaluation.gradient).tolist(),
        "gradient_relative_difference": gradient_relative,
        "tolerances": {"action_relative": 1.0e-10, "gradient_relative": 1.0e-8},
    }
    static_audit = galerkin_only_static_audit()
    result = {
        "ran": True,
        "passed": bool(
            platform_equivalence["passed"]
            and certificate["certified"]
            and static_audit["passed"]
        ),
        "device": device_payload(),
        "basis_size": 160,
        "artifact_manifest_sha256": artifact_hash,
        "dictionary_sha256": file_sha256(OLD_DICTIONARY),
        "selection_only_loader": True,
        "validation_arrays_loaded": False,
        "cache": context.cache_info,
        "memory": basis_memory_estimate(data, 160),
        "timings": stage_timings,
        "evaluation": context.payload(evaluation),
        "heldout_certification": certificate,
        "cpu_current_platform_equivalence": platform_equivalence,
        "static_call_graph_audit": static_audit,
    }
    _write_json(output_dir / "result.json", result)
    return result


def _gradient_comparison(lower: dict[str, Any], upper: dict[str, Any]) -> dict[str, Any]:
    left = np.asarray(lower["gradient"], dtype=np.float64)
    right = np.asarray(upper["gradient"], dtype=np.float64)
    return {
        "lower_K": int(lower["basis_size"]),
        "upper_K": int(upper["basis_size"]),
        "cosine_similarity": float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        ),
        "relative_gradient_difference": float(
            np.linalg.norm(right - left) / max(np.linalg.norm(right), 1.0e-30)
        ),
        "relative_action_increment": float(
            (float(upper["action"]) - float(lower["action"]))
            / max(abs(float(upper["action"])), 1.0e-30)
        ),
    }


def _convergence_row(
    cfg: dict[str, Any], artifact_dir: Path, context: GalerkinOnlyContext,
    eta: jax.Array, basis_size: int, output_dir: Path,
) -> dict[str, Any]:
    result_path = output_dir / f"K{basis_size}.json"
    signature = fingerprint({
        "kind": "galerkin_only_convergence_row_v1",
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "dictionary_sha256": file_sha256(context.dictionary_path),
        "basis_size": int(basis_size),
        "dtype": "float64",
        "configuration_hash": fingerprint(cfg["production_galerkin"]),
        "eta": np.asarray(eta, dtype=np.float64).tolist(),
    })
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return {**previous, "cache_hit": True}
    evaluation, evaluation_seconds = timed(lambda: context.evaluate(
        eta, basis_size=basis_size, with_gradient=True
    ))
    certification, certification_seconds = timed(
        lambda: context.certify(evaluation)
    )
    row = {
        **certification,
        "signature": signature,
        "cache_hit": False,
        "evaluation_seconds": evaluation_seconds,
        "certification_seconds": certification_seconds,
        "train_action": float(evaluation.action),
        "selection_audit_action": float(
            certification["heldout_certificate"]["action"]
        ),
    }
    _write_json(result_path, row)
    return row


def _practical_K_selection(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    comparison_by_upper = {row["upper_K"]: row for row in comparisons}
    candidates = []
    for row in rows[1:]:
        comparison = comparison_by_upper[row["basis_size"]]
        criteria = {
            "certificates": bool(row["certified"]),
            "action_increment_le_2pct": bool(
                comparison["relative_action_increment"] <= 0.02
            ),
            "gradient_cosine": bool(comparison["cosine_similarity"] >= 0.995),
            "gradient_relative_difference": bool(
                comparison["relative_gradient_difference"] <= 0.05
            ),
            "rank_range_stationarity_condition": bool(row["algebra_valid"]),
        }
        candidates.append({
            "K": row["basis_size"], "criteria": criteria,
            "all_primary_criteria": all(criteria.values()),
        })
    primary = next((row for row in candidates if row["all_primary_criteria"]), None)
    if primary is not None:
        return {
            "passed": True,
            "selected_K": primary["K"],
            "action_convergence_complete": True,
            "selection_reason": "smallest tested K satisfying every declared gate",
            "candidate_gates": candidates,
        }
    largest = rows[-1]
    final_comparison = comparisons[-1]
    fallback_valid = bool(
        largest["certified"]
        and largest["algebra_valid"]
        and final_comparison["cosine_similarity"] >= 0.995
        and final_comparison["relative_gradient_difference"] <= 0.05
    )
    return {
        "passed": fallback_valid,
        "selected_K": largest["basis_size"] if fallback_valid else None,
        "action_convergence_complete": False,
        "selection_reason": (
            "largest scientifically valid tested K; action convergence incomplete"
            if fallback_valid else "bounded ladder did not yield a defensible discretization"
        ),
        "candidate_gates": candidates,
    }


def run_galerkin_only_convergence(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = GALERKIN_ONLY_ROOT / "benchmark" / "result.json"
    if not benchmark_path.is_file() or not json.loads(
        benchmark_path.read_text(encoding="utf-8")
    ).get("passed", False):
        raise RuntimeError("GPU-first K=160 benchmark prerequisite is missing")
    data = load_selection_galerkin_data(cfg, artifact_dir)
    dictionary_dir = GALERKIN_ONLY_ROOT / "cache" / "dictionaries"
    dictionary, dictionary_metadata = build_or_load_extended_dictionary(
        cfg, artifact_dir, data, dictionary_dir, maximum_size=240
    )
    dictionary_path = dictionary_dir / "dictionary_K240.npz"
    context = GalerkinOnlyContext(
        cfg, artifact_dir, data, dictionary_path,
        cache_dir=GALERKIN_ONLY_ROOT / "cache" / "K240",
    )
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    rows = [
        _convergence_row(cfg, artifact_dir, context, eta, size, output_dir)
        for size in (160, 200, 240)
    ]
    comparisons = [
        _gradient_comparison(lower, upper)
        for lower, upper in zip(rows[:-1], rows[1:], strict=True)
    ]
    final_increment = comparisons[-1]["relative_action_increment"]
    K280_memory = basis_memory_estimate(data, 280)
    K280_feasible = bool(
        K280_memory["train_total_gib"] <= 12.0
        and device_payload()["platform"] == "gpu"
    )
    ran_K280 = False
    if final_increment > 0.015 and K280_feasible:
        del context
        gc.collect()
        jax.clear_caches()
        dictionary, dictionary_metadata_280 = build_or_load_extended_dictionary(
            cfg, artifact_dir, data, dictionary_dir, maximum_size=280
        )
        dictionary_path = dictionary_dir / "dictionary_K280.npz"
        context = GalerkinOnlyContext(
            cfg, artifact_dir, data, dictionary_path,
            cache_dir=GALERKIN_ONLY_ROOT / "cache" / "K280",
        )
        row_280 = _convergence_row(
            cfg, artifact_dir, context, eta, 280, output_dir
        )
        rows.append(row_280)
        comparisons.append(_gradient_comparison(rows[-2], rows[-1]))
        ran_K280 = True
        dictionary_metadata = {
            "K240": dictionary_metadata,
            "K280": dictionary_metadata_280,
        }
    selection = _practical_K_selection(rows, comparisons)
    result = {
        "ran": True,
        "passed": bool(selection["passed"]),
        "device": device_payload(),
        "eta0": np.asarray(eta).tolist(),
        "dictionary": dictionary_metadata,
        "prefix_160_exact": True,
        "ladder": rows,
        "neighbor_comparisons": comparisons,
        "K280_rule": {
            "K200_to_K240_increment": final_increment,
            "increment_above_1_5pct": final_increment > 0.015,
            "memory_feasible": K280_feasible,
            "ran_K280": ran_K280,
            "maximum_allowed_K": 280,
        },
        "memory": {
            "K240": basis_memory_estimate(data, 240),
            "K280": K280_memory,
        },
        "practical_discretization": selection,
        "validation_arrays_loaded": False,
    }
    _write_json(output_dir / "result.json", result)
    return result


def run_selected_K_profile(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence_path = GALERKIN_ONLY_ROOT / "convergence" / "result.json"
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    if not convergence.get("passed", False):
        raise RuntimeError("practical Galerkin discretization prerequisite failed")
    selected_K = int(convergence["practical_discretization"]["selected_K"])
    dictionary_path = (
        GALERKIN_ONLY_ROOT / "cache" / "dictionaries"
        / f"dictionary_K{selected_K}.npz"
    )
    data, data_seconds = timed(lambda: load_selection_galerkin_data(cfg, artifact_dir))
    context, context_seconds = timed(lambda: GalerkinOnlyContext(
        cfg, artifact_dir, data, dictionary_path,
        cache_dir=GALERKIN_ONLY_ROOT / "cache" / f"K{selected_K}",
    ))
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    value_timing = timing_pair(
        lambda: context.evaluate(eta, basis_size=selected_K, with_gradient=False), 5
    )
    gradient_timing = timing_pair(
        lambda: context.evaluate(eta, basis_size=selected_K, with_gradient=True), 5
    )
    evaluation = context.evaluate(eta, basis_size=selected_K, with_gradient=True)
    certificate, certificate_seconds = timed(lambda: context.certify(evaluation))
    result = {
        "ran": True,
        "passed": bool(certificate["certified"]),
        "device": device_payload(),
        "selected_K": selected_K,
        "action_convergence_complete": convergence[
            "practical_discretization"
        ]["action_convergence_complete"],
        "selection_data_load_seconds": data_seconds,
        "context_initialization_seconds": context_seconds,
        "cache": context.cache_info,
        "value_only": value_timing,
        "value_gradient": gradient_timing,
        "selection_certification_seconds": certificate_seconds,
        "memory": basis_memory_estimate(data, selected_K),
        "eta0": certificate,
        "validation_arrays_loaded": False,
    }
    _write_json(output_dir / "result.json", result)
    return result


def _periodic_delta(candidate: jax.Array, center: jax.Array, box) -> jax.Array:
    shaped = (candidate - center).reshape((-1, 2))
    box_array = jnp.asarray(box, dtype=jnp.float64)
    return (shaped - box_array * jnp.round(shaped / box_array)).reshape((-1,))


def _search_algebra_valid(cfg: dict[str, Any], payload: dict[str, Any]) -> bool:
    settings = cfg["production_galerkin"]
    return bool(
        payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        and payload["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
    )


def _optimization_starts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    starts = [{"id": "eta0", "eta": cfg["envelope"]["eta0"]}]
    historical_refinement = (
        PRODUCTION_ROOT / "refinement" / "result.json"
    )
    if historical_refinement.is_file():
        payload = json.loads(historical_refinement.read_text(encoding="utf-8"))
        starts.append({"id": "previous_galerkin_tiny", "eta": payload["end_eta"]})
    prior_multistart = (
        GALERKIN_ONLY_ROOT.parent / "fast_production_3pct" / "multistart" / "result.json"
    )
    if prior_multistart.is_file():
        payload = json.loads(prior_multistart.read_text(encoding="utf-8"))
        for index, row in enumerate(payload.get("trajectories", [])[:4]):
            starts.append({
                "id": f"prior_galerkin_endpoint_{index + 1}",
                "eta": row["end_eta"],
            })
    unique = []
    for row in starts:
        eta = np.asarray(row["eta"], dtype=np.float64)
        if any(np.linalg.norm(eta - np.asarray(old["eta"])) <= 1.0e-12 for old in unique):
            continue
        unique.append({"id": row["id"], "eta": eta.tolist()})
        if len(unique) == 6:
            break
    return unique


def _trajectory_signature(
    cfg: dict[str, Any], artifact_dir: Path, dictionary_path: Path,
    selected_K: int, start: dict[str, Any],
) -> str:
    return fingerprint({
        "kind": "galerkin_only_trust_trajectory_v1",
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "dictionary_sha256": file_sha256(dictionary_path),
        "selected_K": selected_K,
        "start": start,
        "radius": 2.0e-4,
        "initial_step": 5.0e-5,
        "maximum_steps": 8,
        "backtracking_steps": 10,
        "replacement_tolerance": 1.0e-10,
        "configuration_hash": fingerprint({
            "projection": cfg["projection"], "forcing": cfg["forcing"],
            "production_galerkin": cfg["production_galerkin"],
        }),
    })


def _run_trust_trajectory(
    cfg: dict[str, Any], artifact_dir: Path, context: GalerkinOnlyContext,
    selected_K: int, start: dict[str, Any], output_path: Path,
    *, risk_ceiling: float,
) -> dict[str, Any]:
    signature = _trajectory_signature(
        cfg, artifact_dir, context.dictionary_path, selected_K, start
    )
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature and not previous.get("in_progress", True):
            return {**previous, "cache_hit": True}
    family = context.data.selection_problem.family
    center = wrap_periodic(jnp.asarray(start["eta"], dtype=jnp.float64), family)
    eta = center
    current = context.evaluate(eta, basis_size=selected_K, with_gradient=True)
    start_certificate = context.certify(current)
    exact_start_valid = bool(
        start_certificate["certified"]
        and float(current.risk) <= risk_ceiling
    )
    if not exact_start_valid:
        result = {
            "signature": signature, "in_progress": False, "cache_hit": False,
            "start_id": start["id"], "start_eta": np.asarray(center).tolist(),
            "start_certificate": start_certificate,
            "eligible": False, "reason": "start failed exact selection gates",
            "history": [], "evaluation_count": 1, "certification_count": 1,
        }
        _write_json(output_path, result)
        return result
    last_certified_eta = eta
    last_certified = start_certificate
    history = []
    evaluation_count = 1
    certification_count = 1
    trust_radius = 2.0e-4
    step_length = 5.0e-5
    replacement_tolerance = 1.0e-10
    for step in range(8):
        _, risk_gradient = context._risk_value_grad(eta)
        direction = -current.gradient
        risk_slope = float(jnp.dot(risk_gradient, direction))
        risk_norm_sq = float(jnp.dot(risk_gradient, risk_gradient))
        if risk_slope > 0.0 and risk_norm_sq > 1.0e-30:
            direction = direction - (risk_slope / risk_norm_sq) * risk_gradient
            direction = direction - 0.02 * jnp.linalg.norm(direction) * (
                risk_gradient / jnp.sqrt(risk_norm_sq)
            )
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
        accepted = False
        attempts = []
        accepted_candidate = None
        for backtrack in range(10):
            length = step_length * (0.5 ** backtrack)
            proposal = wrap_periodic(eta + length * direction, family)
            total_delta = _periodic_delta(proposal, center, family.box)
            if float(jnp.linalg.norm(total_delta)) > trust_radius * (1.0 + 1.0e-12):
                attempts.append({"length": length, "accepted": False, "reason": "trust_radius"})
                continue
            if not bool(family.geometry_valid(proposal)):
                attempts.append({"length": length, "accepted": False, "reason": "geometry"})
                continue
            risk = float(context._risk(proposal))
            if risk > risk_ceiling:
                attempts.append({
                    "length": length, "accepted": False, "reason": "risk",
                    "risk": risk,
                })
                continue
            candidate = context.evaluate(
                proposal, basis_size=selected_K, with_gradient=True
            )
            evaluation_count += 1
            payload = context.payload(candidate)
            delta = _periodic_delta(proposal, eta, family.box)
            predicted = max(-float(jnp.dot(current.gradient, delta)), 1.0e-30)
            actual = float(current.action) - float(candidate.action)
            rho = actual / predicted
            rank_stable = bool(np.array_equal(
                np.asarray(candidate.solve.numerical_rank),
                np.asarray(current.solve.numerical_rank),
            ))
            accepted = bool(
                actual > replacement_tolerance
                and payload["train_forcing_audit"]["valid"]
                and payload["geometry_valid"]
                and _search_algebra_valid(cfg, payload)
                and rank_stable
            )
            attempts.append({
                "length": length, "accepted": accepted, "risk": risk,
                "action": float(candidate.action), "actual_reduction": actual,
                "predicted_reduction": predicted, "rho": rho,
                "rank_stable": rank_stable,
            })
            if accepted:
                accepted_candidate = candidate
                step_length = min(
                    7.5e-5 if rho >= 0.75 else length,
                    trust_radius,
                )
                break
        if accepted and accepted_candidate is not None:
            eta, current = accepted_candidate.eta, accepted_candidate
            if (step + 1) % 4 == 0:
                checkpoint = context.certify(current)
                certification_count += 1
                if checkpoint["certified"]:
                    last_certified_eta, last_certified = eta, checkpoint
                else:
                    eta = last_certified_eta
                    current = context.evaluate(
                        eta, basis_size=selected_K, with_gradient=True
                    )
                    evaluation_count += 1
                    accepted = False
                    attempts.append({
                        "accepted": False,
                        "reason": "periodic_heldout_certificate_failed_reverted",
                        "heldout_certificate": checkpoint["heldout_certificate"],
                    })
        history.append({
            "step": step + 1, "accepted": accepted,
            "eta": np.asarray(eta).tolist(), "action": float(current.action),
            "risk": float(current.risk), "step_length_next": step_length,
            "attempts": attempts,
        })
        partial = {
            "signature": signature, "in_progress": True,
            "start_id": start["id"], "start_eta": np.asarray(center).tolist(),
            "history": history, "evaluation_count": evaluation_count,
            "certification_count": certification_count,
        }
        _write_json(output_path, partial)
        if not accepted:
            break
        if float(jnp.linalg.norm(_periodic_delta(eta, center, family.box))) >= 0.999 * trust_radius:
            break
    final_certificate = context.certify(current)
    certification_count += 1
    if final_certificate["certified"]:
        last_certified_eta, last_certified = eta, final_certificate
    elif not np.array_equal(np.asarray(eta), np.asarray(last_certified_eta)):
        eta = last_certified_eta
        current = context.evaluate(eta, basis_size=selected_K, with_gradient=True)
        evaluation_count += 1
    result = {
        "signature": signature, "in_progress": False, "cache_hit": False,
        "start_id": start["id"], "start_eta": np.asarray(center).tolist(),
        "end_eta": np.asarray(eta).tolist(),
        "start_action": float(start_certificate["action"]),
        "end_action": float(last_certified["action"]),
        "action_reduction": float(start_certificate["action"] - last_certified["action"]),
        "end_risk": float(last_certified["risk"]),
        "steps_accepted": sum(row["accepted"] for row in history),
        "history": history, "evaluation_count": evaluation_count,
        "certification_count": certification_count,
        "start_certificate": start_certificate,
        "final_certificate": last_certified,
        "eligible": bool(
            last_certified["certified"]
            and float(last_certified["risk"]) <= risk_ceiling
        ),
    }
    _write_json(output_path, result)
    return result


def run_galerkin_only_optimization(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = GALERKIN_ONLY_ROOT / "profile_selected_K" / "result.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profile.get("passed", False):
        raise RuntimeError("selected-K profile prerequisite failed")
    selected_K = int(profile["selected_K"])
    data = load_selection_galerkin_data(cfg, artifact_dir)
    dictionary_path = (
        GALERKIN_ONLY_ROOT / "cache" / "dictionaries"
        / f"dictionary_K{selected_K}.npz"
    )
    context = GalerkinOnlyContext(
        cfg, artifact_dir, data, dictionary_path,
        cache_dir=GALERKIN_ONLY_ROOT / "cache" / f"K{selected_K}",
    )
    law_eta = jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64)
    law_risk = float(context._risk(law_eta))
    risk_ceiling = 1.03 * law_risk
    starts = _optimization_starts(cfg)
    trajectory_dir = require_galerkin_only_output_path(
        GALERKIN_ONLY_ROOT / "trajectories"
    )
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for index, start in enumerate(starts):
        trajectories.append(_run_trust_trajectory(
            cfg, artifact_dir, context, selected_K, start,
            trajectory_dir / f"start_{index:02d}.json",
            risk_ceiling=risk_ceiling,
        ))
    eligible = [row for row in trajectories if row.get("eligible", False)]
    if not eligible:
        raise RuntimeError("no exactly feasible Galerkin-certified start survived")
    finalists = sorted(eligible, key=lambda row: float(row["end_action"]))
    winner = finalists[0]
    eta0_row = next(row for row in trajectories if row["start_id"] == "eta0")
    eta0_certificate = eta0_row["start_certificate"]
    replacement_tolerance = 1.0e-10
    improved = bool(
        float(winner["end_action"])
        < float(eta0_certificate["action"]) - replacement_tolerance
    )
    frozen = winner["final_certificate"]
    result = {
        "ran": True,
        "passed": True,
        "device": device_payload(),
        "scientific_solver": "fixed-feature rank-aware Galerkin",
        "selected_K": selected_K,
        "dictionary_sha256": file_sha256(dictionary_path),
        "law_risk": law_risk,
        "risk_ceiling": risk_ceiling,
        "replacement_tolerance": replacement_tolerance,
        "trust_region": {
            "radius": 2.0e-4, "initial_step": 5.0e-5,
            "maximum_steps_per_start": 8, "backtracking_steps": 10,
            "periodic_certificate_every": 4,
        },
        "start_count": len(starts),
        "starts": starts,
        "trajectories": trajectories,
        "finalists": finalists,
        "eta0": eta0_certificate,
        "winner": frozen,
        "winner_source": winner["start_id"],
        "winner_selection_risk_increase_percent": (
            (float(frozen["risk"]) / law_risk - 1.0) * 100.0
        ),
        "selection_action_difference_winner_minus_eta0": (
            float(frozen["action"]) - float(eta0_certificate["action"])
        ),
        "selection_improved": improved,
        "selection_frozen": True,
        "validation_accessed": False,
        "validation_arrays_loaded": False,
        "original_production_incumbent_modified": False,
        "total_galerkin_evaluations": sum(
            int(row.get("evaluation_count", 0)) for row in trajectories
        ),
        "total_selection_certifications": sum(
            int(row.get("certification_count", 0)) for row in trajectories
        ),
    }
    _write_json(output_dir / "result.json", result)
    return result


def _validation_action_uncertainty(
    dictionary, coefficients: jax.Array, bank, weights: jax.Array,
    time_weights: jax.Array, *, chunk_size: int,
) -> dict[str, Any]:
    """Use the production weighted-sample SE convention on the audit bank."""

    kinetic, kinetic_second = [], []
    sample_count = int(bank.configurations.shape[1])
    evaluators = [
        jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
        for t in range(int(bank.configurations.shape[0]))
    ]
    for time_index in range(int(bank.configurations.shape[0])):
        first = jnp.asarray(0.0, dtype=jnp.float64)
        second = jnp.asarray(0.0, dtype=jnp.float64)
        for start in range(0, sample_count, int(chunk_size)):
            stop = min(start + int(chunk_size), sample_count)
            _, gradients = evaluators[time_index](
                bank.configurations[time_index, start:stop]
            )
            potential_gradient = jnp.einsum(
                "k,nkpd->npd", coefficients[time_index], gradients
            )
            rows = jnp.sum(potential_gradient * potential_gradient, axis=(-2, -1))
            chunk_weights = weights[time_index, start:stop]
            first = first + jnp.einsum("n,n->", chunk_weights, rows)
            second = second + jnp.einsum(
                "n,n->", chunk_weights, rows * rows
            )
        kinetic.append(first)
        kinetic_second.append(second)
    kinetic = jnp.stack(kinetic)
    kinetic_second = jnp.stack(kinetic_second)
    variance = jnp.maximum(kinetic_second - kinetic * kinetic, 0.0)
    effective_samples = 1.0 / jnp.maximum(
        jnp.sum(weights * weights, axis=-1), 1.0e-300
    )
    standard_error = jnp.sqrt(jnp.sum(
        (time_weights * jnp.sqrt(
            variance / jnp.maximum(effective_samples, 1.0)
        )) ** 2
    ))
    return {
        "action": float(jnp.sum(time_weights * kinetic)),
        "action_standard_error": float(standard_error),
        "kinetic_by_time": np.asarray(kinetic).tolist(),
        "uncertainty_convention": (
            "production weighted empirical audit-sample standard error; "
            "no pseudo-blocks introduced"
        ),
    }


def _evaluate_validation_candidate(
    cfg: dict[str, Any], data, dictionary, eta: jax.Array,
) -> dict[str, Any]:
    problem = data.validation_problem
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    reconstruction = reconstruct_moments(eta, problem)
    fit_state = forcing_state(eta, problem, data.fit_bank, reconstruction)
    chunk_size = int(cfg["production_galerkin"]["chunk_size"])
    started = time.perf_counter()
    system = assemble_hybrid_system(
        dictionary, data.fit_bank, fit_state.projection.weights,
        fit_state.forcing, chunk_size=chunk_size,
    )
    solve = rank_aware_quadratic_solve(
        system.gram, system.load,
        relative_rank_tolerance=float(
            cfg["production_galerkin"]["relative_rank_tolerance"]
        ),
    )
    aggregate = aggregate_quadratic_values(solve, problem.time_weights)
    fit_seconds = time.perf_counter() - started
    audit_state = forcing_state(
        eta, problem, data.audit_bank, reconstruction
    )
    adapter = SimpleNamespace(
        ritz_audit_bank=data.audit_bank, selection_problem=problem,
    )
    from .galerkin_only import GalerkinCertificateThresholds
    thresholds = GalerkinCertificateThresholds(
        **cfg["production_galerkin"]["certificate_thresholds"]
    )
    started = time.perf_counter()
    certificate = audit_hybrid_solutions(
        dictionary, solve.coefficients[None], adapter, eta, reconstruction,
        audit_state, thresholds, chunk_size=chunk_size,
    )[0]
    audit_seconds = time.perf_counter() - started
    uncertainty = _validation_action_uncertainty(
        dictionary, solve.coefficients, data.audit_bank,
        audit_state.projection.weights, problem.time_weights,
        chunk_size=chunk_size,
    )
    if not np.isclose(
        uncertainty["action"], certificate["action"], rtol=1.0e-10, atol=1.0e-12
    ):
        raise RuntimeError("validation audit action/uncertainty pass mismatch")
    settings = cfg["production_galerkin"]
    rank_fraction = solve.numerical_rank / float(dictionary.size)
    algebra_valid = bool(
        float(aggregate["identity_relerr"]) <= float(settings["maximum_identity_relerr"])
        and float(jnp.max(solve.range_residual)) <= float(settings["maximum_range_residual"])
        and float(jnp.max(solve.stationarity_residual)) <= float(settings["maximum_stationarity_residual"])
        and float(jnp.max(system.raw_symmetry_residual)) <= float(settings["maximum_symmetry_residual"])
        and float(jnp.max(solve.condition_number)) <= float(settings["maximum_retained_condition"])
        and float(jnp.min(rank_fraction)) >= float(settings["minimum_rank_fraction"])
    )
    return {
        "eta": np.asarray(eta).tolist(),
        "validation_fit_action": float(aggregate["action"]),
        "validation_audit_action": float(certificate["action"]),
        "action_standard_error": uncertainty["action_standard_error"],
        "kinetic_by_time": uncertainty["kinetic_by_time"],
        "uncertainty_convention": uncertainty["uncertainty_convention"],
        "risk": float(validation_risk(eta, data)),
        "geometry_valid": bool(problem.family.geometry_valid(eta)),
        "fit_forcing_audit": _forcing_state_payload(fit_state, problem),
        "audit_forcing_audit": _forcing_state_payload(audit_state, problem),
        "identity_relerr": float(aggregate["identity_relerr"]),
        "rank_by_time": np.asarray(solve.numerical_rank).tolist(),
        "minimum_rank_fraction": float(jnp.min(rank_fraction)),
        "worst_retained_condition": float(jnp.max(solve.condition_number)),
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
        "algebra_valid": algebra_valid,
        "heldout_certificate": certificate,
        "timings": {
            "validation_fit_assembly_and_solve_seconds": fit_seconds,
            "validation_audit_certificate_seconds": audit_seconds,
        },
    }


def run_galerkin_only_validation(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
    *, selection_result: Path,
) -> dict[str, Any]:
    """Perform the single sealed Galerkin validation after frozen selection."""

    output_dir = require_galerkin_only_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_result = require_galerkin_only_output_path(selection_result)
    selection = json.loads(selection_result.read_text(encoding="utf-8"))
    if not (
        selection.get("passed", False)
        and selection.get("selection_frozen", False)
        and not selection.get("validation_accessed", True)
    ):
        raise RuntimeError("validation requires an untouched frozen selection result")
    selected_K = int(selection["selected_K"])
    dictionary_path = (
        GALERKIN_ONLY_ROOT / "cache" / "dictionaries"
        / f"dictionary_K{selected_K}.npz"
    )
    if file_sha256(dictionary_path) != selection["dictionary_sha256"]:
        raise RuntimeError("frozen selection dictionary signature changed")
    signature = fingerprint({
        "kind": "galerkin_only_sealed_validation_v1",
        "selection_result_sha256": file_sha256(selection_result),
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "dictionary_sha256": file_sha256(dictionary_path),
        "selected_K": selected_K,
        "dtype": "float64",
        "configuration_hash": fingerprint({
            "physics": cfg["physics"], "measurement": cfg["measurement"],
            "projection": cfg["projection"], "forcing": cfg["forcing"],
            "production_galerkin": cfg["production_galerkin"],
        }),
    })
    result_path = output_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return {**previous, "cache_hit": True}
        raise RuntimeError("sealed validation result exists with a different signature")
    _write_json(output_dir / "seal.json", {
        "signature": signature,
        "selection_result": str(selection_result),
        "selection_result_sha256": file_sha256(selection_result),
        "winner_frozen_before_validation": True,
        "validation_used_for_selection": False,
    })

    # This is the first validation-artifact access in the new workflow.
    data, load_seconds = timed(lambda: load_validation_galerkin_data(cfg, artifact_dir))
    dictionary = load_dictionary(
        dictionary_path, box=tuple(cfg["physics"]["box"])
    )
    if dictionary.size != selected_K:
        raise RuntimeError("sealed validation basis size changed")
    law_eta = jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64)
    law_risk = float(validation_risk(law_eta, data))
    risk_ceiling = 1.03 * law_risk
    eta0 = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    winner_eta = jnp.asarray(selection["winner"]["eta"], dtype=jnp.float64)
    eta0_result = _evaluate_validation_candidate(
        cfg, data, dictionary, eta0
    )
    winner_result = _evaluate_validation_candidate(
        cfg, data, dictionary, winner_eta
    )
    for row in (eta0_result, winner_result):
        row["law_risk"] = law_risk
        row["risk_ceiling"] = risk_ceiling
        row["risk_valid"] = bool(row["risk"] <= risk_ceiling)
        row["valid"] = bool(
            row["risk_valid"]
            and row["geometry_valid"]
            and row["fit_forcing_audit"]["valid"]
            and row["audit_forcing_audit"]["valid"]
            and row["algebra_valid"]
            and row["heldout_certificate"]["valid"]
        )
    action_difference = (
        winner_result["validation_audit_action"]
        - eta0_result["validation_audit_action"]
    )
    validation_success = bool(
        selection["selection_improved"]
        and winner_result["valid"]
        and action_difference < -float(selection["replacement_tolerance"])
    )
    result = {
        "signature": signature,
        "cache_hit": False,
        "ran": True,
        "passed": True,
        "validation_success": validation_success,
        "device": device_payload(),
        "scientific_solver": "fixed-feature rank-aware Galerkin",
        "selected_K": selected_K,
        "dictionary_sha256": file_sha256(dictionary_path),
        "selection_result_sha256": file_sha256(selection_result),
        "selection_was_frozen_before_validation": True,
        "validation_used_for_selection": False,
        "validation_arrays_loaded_only_after_freeze": True,
        "validation_data_load_seconds": load_seconds,
        "law_risk": law_risk,
        "risk_ceiling": risk_ceiling,
        "eta0": eta0_result,
        "winner": winner_result,
        "winner_minus_eta0_validation_action": action_difference,
        "validation_reversal": not validation_success,
        "winner_geometry_remains_frozen": True,
        "original_production_incumbent_modified": False,
        "pareto_sweep_run": False,
    }
    _write_json(result_path, result)
    return result


__all__ = [
    "galerkin_only_static_audit", "run_galerkin_only_benchmark",
    "run_galerkin_only_convergence", "run_selected_K_profile",
    "run_galerkin_only_optimization", "run_galerkin_only_validation",
]
