"""Read-only retrospective checks for the frozen 3% Galerkin pair.

This module never optimizes eta and never writes to the sealed Galerkin-only
selection or validation trees.  New artifacts are confined to
``outputs/final_3pct_crosscheck``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import (
    GALERKIN_ONLY_ROOT, GalerkinCertificateThresholds, GalerkinOnlyContext,
    _forcing_state_payload, device_payload,
)
from .galerkin_only_data import (
    load_selection_galerkin_data, load_validation_galerkin_data,
    selection_risk, validation_risk,
)
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .production_basis import load_dictionary
from .production_galerkin import assemble_hybrid_system, audit_hybrid_solutions
from .production_gradient import _direction_gate

Array = jax.Array

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "final_3pct_crosscheck"
DICTIONARY_PATH = (
    GALERKIN_ONLY_ROOT / "cache" / "dictionaries" / "dictionary_K280.npz"
)
CACHE_PATH = GALERKIN_ONLY_ROOT / "cache" / "K280"
SEALED_SELECTION = GALERKIN_ONLY_ROOT / "selection" / "result.json"
SEALED_VALIDATION = GALERKIN_ONLY_ROOT / "validation" / "result.json"
PRODUCTION_DIRECTIONS = PRODUCTION_ROOT / "gradient_checks" / "result.json"
SIZES = (160, 200, 240, 280)
ETA_GRAD = (
    0.895371148114089, 0.205982940238786,
    1.334525121515147, 0.865464965382237,
    0.750749623351011, 0.518133188490931,
    1.642405611981796, 0.588309862016330,
)


def require_crosscheck_output_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"final cross-check output must be beneath {root}, got {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_crosscheck_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def risk_ratio(risk: float, law_risk: float) -> float:
    return float(risk) / float(law_risk) - 1.0


def validation_protocol(cfg: dict[str, Any], validation_law_risk: float) -> dict[str, Any]:
    allowance = float(cfg["search"]["risk_allowance_percent"]) / 100.0
    slack = float(cfg["search"]["validation_relative_slack"])
    strict_multiplier = 1.0 + allowance
    declared_multiplier = strict_multiplier + slack
    return {
        "resolved": True,
        "selection_multiplier": strict_multiplier,
        "strict_3pct_multiplier": strict_multiplier,
        "declared_validation_relative_slack": slack,
        "declared_validation_multiplier": declared_multiplier,
        "validation_law_risk": float(validation_law_risk),
        "strict_3pct_ceiling": strict_multiplier * float(validation_law_risk),
        "declared_plus5pp_ceiling": declared_multiplier * float(validation_law_risk),
        "actual_predeclared_ceiling": declared_multiplier * float(validation_law_risk),
        "interpretation": (
            "validation_relative_slack=0.05 is an additive five percentage "
            "points in the risk multiplier, on top of the p=3 selection allowance"
        ),
        "evidence": [
            {
                "path": "experiments/skyrmions_deep_ritz/config.json",
                "lines": "126-130",
                "fact": "risk_allowance_percent=3.0 and validation_relative_slack=0.05",
                "predeclared_original_experiment": True,
            },
            {
                "path": "experiments/skyrmions_deep_ritz/experiment.py",
                "lines": "1383-1385",
                "fact": "validation gate is (1 + allowance/100 + validation_relative_slack) * validation Law risk",
                "predeclared_original_experiment": True,
            },
            {
                "path": "experiments/skyrmions_deep_ritz/README.md",
                "lines": "823-830",
                "fact": "published local report says validation passes the declared 3% plus 5% neighborhood",
                "preexisting_before_current_validation": True,
            },
            {
                "path": "experiments/skyrmions_deep_ritz_full/outputs/production_galerkin/artifacts/result.json",
                "fact": "frozen config records slack=0.05 and scientific_risk_neighborhood_pass=true",
                "frozen_protocol_metadata": True,
            },
        ],
    }


def _base_signature(cfg: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    return {
        "artifact_manifest_sha256": file_sha256(
            artifact_dir / "isolated_artifact_manifest.json"
        ),
        "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "sealed_selection_sha256": file_sha256(SEALED_SELECTION),
        "sealed_validation_sha256": file_sha256(SEALED_VALIDATION),
        "sizes": list(SIZES),
        "dtype": "float64",
        "eta0": cfg["envelope"]["eta0"],
        "eta_grad": list(ETA_GRAD),
        "eta_law": cfg["envelope"]["law_eta"],
        "configuration_hash": fingerprint({
            key: cfg[key] for key in (
                "physics", "measurement", "moment_reconstruction",
                "projection", "forcing", "production_galerkin", "search",
            )
        }),
    }


def _algebra_payload(cfg: dict[str, Any], system, solve, time_weights: Array) -> dict[str, Any]:
    settings = cfg["production_galerkin"]
    aggregate = aggregate_quadratic_values(solve, time_weights)
    size = int(solve.coefficients.shape[-1])
    minimum_rank_fraction = float(jnp.min(solve.numerical_rank / float(size)))
    payload = {
        "action": float(aggregate["action"]),
        "objective": float(aggregate["objective"]),
        "identity_relerr": float(aggregate["identity_relerr"]),
        "rank_by_time": np.asarray(solve.numerical_rank).tolist(),
        "minimum_rank_fraction": minimum_rank_fraction,
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "worst_retained_condition": float(jnp.max(solve.condition_number)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
    }
    payload["valid"] = bool(
        payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        and minimum_rank_fraction >= float(settings["minimum_rank_fraction"])
    )
    return payload


def _solve_ladder(cfg: dict[str, Any], system, time_weights: Array):
    solves, algebra = [], []
    for size in SIZES:
        solve = rank_aware_quadratic_solve(
            system.gram[:, :size, :size], system.load[:, :size],
            relative_rank_tolerance=float(
                cfg["production_galerkin"]["relative_rank_tolerance"]
            ),
        )
        solves.append(solve)
        algebra.append(_algebra_payload(cfg, system, solve, time_weights))
    return solves, algebra


def _padded_coefficients(solves: list[Any], maximum: int) -> Array:
    return jnp.stack([
        jnp.pad(solve.coefficients, ((0, 0), (0, maximum - size)))
        for solve, size in zip(solves, SIZES, strict=True)
    ])


def _certify_view(
    cfg: dict[str, Any], dictionary, solves: list[Any], problem, bank,
    eta: Array, reconstruction, state,
) -> list[dict[str, Any]]:
    adapter = SimpleNamespace(ritz_audit_bank=bank, selection_problem=problem)
    return audit_hybrid_solutions(
        dictionary, _padded_coefficients(solves, dictionary.size), adapter,
        eta, reconstruction, state,
        GalerkinCertificateThresholds(
            **cfg["production_galerkin"]["certificate_thresholds"]
        ),
        chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
    )


def _designs(cfg: dict[str, Any]) -> dict[str, Array]:
    return {
        "law": jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64),
        "eta0": jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64),
        "eta_grad": jnp.asarray(ETA_GRAD, dtype=jnp.float64),
    }


def run_protocol(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_crosscheck_output_path(output_dir)
    sealed = json.loads(SEALED_VALIDATION.read_text(encoding="utf-8"))
    law_risk = float(sealed["law_risk"])
    protocol = validation_protocol(cfg, law_risk)
    rows = {}
    for name in ("eta0", "winner"):
        risk = float(sealed[name]["risk"])
        rows["eta_grad" if name == "winner" else name] = {
            "risk": risk,
            "risk_ratio": risk_ratio(risk, law_risk),
            "passes_strict_3pct": risk <= protocol["strict_3pct_ceiling"],
            "passes_declared_plus5pp": risk <= protocol["actual_predeclared_ceiling"],
        }
    result = {
        "ran": True,
        "passed": True,
        "signature": fingerprint({"kind": "final_protocol_v1", **_base_signature(cfg, artifact_dir)}),
        "protocol": protocol,
        "frozen_pair": rows,
        "sealed_validation_read_only": True,
        "selection_mutated": False,
    }
    write_json(output_dir / "result.json", result)
    return result


def _gradient_point_payload(context: GalerkinOnlyContext, eta: Array, center_rank) -> dict[str, Any]:
    evaluation = context.evaluate(eta, basis_size=280, with_gradient=False)
    payload = context.payload(evaluation)
    settings = context.cfg["production_galerkin"]
    payload["algebra_valid"] = bool(
        payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"] <= float(settings["maximum_stationarity_residual"])
        and payload["worst_symmetry_residual"] <= float(settings["maximum_symmetry_residual"])
        and payload["worst_retained_condition"] <= float(settings["maximum_retained_condition"])
        and payload["minimum_rank_fraction"] >= float(settings["minimum_rank_fraction"])
    )
    audit_state = forcing_state(
        evaluation.eta, context.data.selection_problem, context.data.audit_bank,
        evaluation.reconstruction,
    )
    payload["audit_forcing_audit"] = _forcing_state_payload(
        audit_state, context.data.selection_problem
    )
    payload["rank_stable"] = payload["rank_by_time"] == center_rank
    payload["hard_gates_passed"] = bool(
        payload["algebra_valid"]
        and payload["geometry_valid"]
        and payload["train_forcing_audit"]["valid"]
        and payload["audit_forcing_audit"]["valid"]
    )
    return payload


def directional_fd_row(
    context: GalerkinOnlyContext, eta: Array, direction: Array,
    epsilon: float, ad: float, center_rank,
) -> dict[str, Any]:
    family = context.data.selection_problem.family
    plus_eta = wrap_periodic(eta + float(epsilon) * direction, family)
    minus_eta = wrap_periodic(eta - float(epsilon) * direction, family)
    plus = _gradient_point_payload(context, plus_eta, center_rank)
    minus = _gradient_point_payload(context, minus_eta, center_rank)
    fd = (float(plus["action"]) - float(minus["action"])) / (2.0 * float(epsilon))
    absolute = abs(fd - float(ad))
    relative = absolute / max(abs(fd), abs(float(ad)), 1.0e-12)
    accepted = bool(
        plus["rank_stable"] and minus["rank_stable"]
        and plus["hard_gates_passed"] and minus["hard_gates_passed"]
    )
    return {
        "epsilon": float(epsilon), "fd": fd,
        "absolute_discrepancy": absolute,
        "relative_discrepancy": relative,
        "rank_center": center_rank,
        "rank_plus": plus["rank_by_time"],
        "rank_minus": minus["rank_by_time"],
        "rank_stable": bool(plus["rank_stable"] and minus["rank_stable"]),
        "accepted": accepted, "plus": plus, "minus": minus,
    }


def run_gradient_check(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_crosscheck_output_path(output_dir)
    signature = fingerprint({"kind": "direct_K280_gradient_v1", **_base_signature(cfg, artifact_dir)})
    result_path = output_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature and not previous.get("in_progress", True):
            return {**previous, "cache_hit": True}
        raise RuntimeError("K=280 gradient output exists with incompatible signature")
    data = load_selection_galerkin_data(cfg, artifact_dir)
    context = GalerkinOnlyContext(
        cfg, artifact_dir, data, DICTIONARY_PATH, cache_dir=CACHE_PATH
    )
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    center = context.evaluate(eta, basis_size=280, with_gradient=True)
    repeated = context.evaluate(eta, basis_size=280, with_gradient=True)
    center_certificate = context.certify(center)
    tolerance = float(
        cfg["production_galerkin"]["gradient"]["determinism_absolute_tolerance"]
    )
    deterministic = bool(
        abs(float(center.action - repeated.action)) <= tolerance
        and float(jnp.max(jnp.abs(center.gradient - repeated.gradient))) <= tolerance
    )
    old = json.loads(PRODUCTION_DIRECTIONS.read_text(encoding="utf-8"))
    directions = [
        jnp.asarray(row["direction"], dtype=jnp.float64)
        for row in old["directions"]
    ]
    settings = cfg["production_galerkin"]["gradient"]
    center_rank = np.asarray(center.solve.numerical_rank).tolist()
    direction_results = []
    write_json(result_path, {
        "signature": signature, "in_progress": True,
        "eta_gradient": np.asarray(center.gradient).tolist(), "directions": [],
    })
    for index, direction in enumerate(directions):
        ad = float(jnp.vdot(center.gradient, direction))
        rows = [
            directional_fd_row(
                context, eta, direction, float(epsilon), ad, center_rank
            )
            for epsilon in settings["epsilon_ladder"]
        ]
        result = {
            "index": index, "direction": np.asarray(direction).tolist(),
            "ad_directional_derivative": ad, "rows": rows,
            **_direction_gate(rows, ad, settings),
        }
        direction_results.append(result)
        write_json(result_path, {
            "signature": signature, "in_progress": True,
            "eta_gradient": np.asarray(center.gradient).tolist(),
            "directions": direction_results,
        })
    passed_count = sum(row["passed"] for row in direction_results)
    passed = bool(
        deterministic and center_certificate["certified"]
        and bool(jnp.all(jnp.isfinite(center.gradient)))
        and passed_count >= int(settings["required_passed_directions"])
    )
    result = {
        "signature": signature, "in_progress": False, "cache_hit": False,
        "ran": True, "passed": passed, "device": device_payload(),
        "basis_size": 280, "eta0": np.asarray(eta).tolist(),
        "center_action": float(center.action),
        "eta_gradient": np.asarray(center.gradient).tolist(),
        "gradient_finite": bool(jnp.all(jnp.isfinite(center.gradient))),
        "deterministic": deterministic, "center": center_certificate,
        "direction_source": str(PRODUCTION_DIRECTIONS),
        "directions_reused_exactly": True,
        "passed_direction_count": passed_count,
        "required_passed_direction_count": int(settings["required_passed_directions"]),
        "directions": direction_results,
        "no_optimization": True, "selection_mutated": False,
    }
    write_json(result_path, result)
    return result


def _row(
    size: int, algebra: dict[str, Any], fit_certificate: dict[str, Any],
    audit_certificate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "basis_size": int(size),
        "fit_action": float(algebra["action"]),
        "audit_action": float(audit_certificate["action"]),
        "algebra": algebra,
        "fit_certificate": fit_certificate,
        "audit_certificate": audit_certificate,
        "valid": bool(
            algebra["valid"] and fit_certificate["valid"]
            and audit_certificate["valid"]
        ),
    }


def run_selection_ladder(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_crosscheck_output_path(output_dir)
    signature = fingerprint({"kind": "frozen_selection_ladder_v1", **_base_signature(cfg, artifact_dir)})
    result_path = output_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return {**previous, "cache_hit": True}
        raise RuntimeError("selection ladder output exists with incompatible signature")
    data = load_selection_galerkin_data(cfg, artifact_dir)
    context = GalerkinOnlyContext(
        cfg, artifact_dir, data, DICTIONARY_PATH, cache_dir=CACHE_PATH
    )
    dictionary = context.dictionary
    problem = data.selection_problem
    designs = {}
    for name, eta in _designs(cfg).items():
        eta = wrap_periodic(eta, problem.family)
        reconstruction = reconstruct_moments(eta, problem)
        fit_state = forcing_state(eta, problem, data.train_bank, reconstruction)
        audit_state = forcing_state(eta, problem, data.audit_bank, reconstruction)
        system = context.assemble(
            fit_state.projection.weights, fit_state.forcing, 280
        )
        solves, algebra = _solve_ladder(cfg, system, problem.time_weights)
        fit_certificates = _certify_view(
            cfg, dictionary, solves, problem, data.train_bank,
            eta, reconstruction, fit_state,
        )
        audit_certificates = _certify_view(
            cfg, dictionary, solves, problem, data.audit_bank,
            eta, reconstruction, audit_state,
        )
        designs[name] = {
            "eta": np.asarray(eta).tolist(),
            "risk": float(selection_risk(eta, data)),
            "fit_forcing_audit": _forcing_state_payload(fit_state, problem),
            "audit_forcing_audit": _forcing_state_payload(audit_state, problem),
            "geometry_valid": bool(problem.family.geometry_valid(eta)),
            "ladder": [
                _row(size, row, fit, audit)
                for size, row, fit, audit in zip(
                    SIZES, algebra, fit_certificates, audit_certificates, strict=True
                )
            ],
        }
    result = {
        "signature": signature, "cache_hit": False, "ran": True,
        "passed": all(
            row["valid"] for design in designs.values() for row in design["ladder"]
        ),
        "device": device_payload(), "sizes": list(SIZES), "designs": designs,
        "views": ["selection_train", "selection_audit"],
        "validation_accessed": False, "no_optimization": True,
        "selection_mutated": False,
    }
    write_json(result_path, result)
    return result


def run_validation_ladder(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    output_dir = require_crosscheck_output_path(output_dir)
    signature = fingerprint({"kind": "frozen_validation_ladder_v1", **_base_signature(cfg, artifact_dir)})
    result_path = output_dir / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            return {**previous, "cache_hit": True}
        raise RuntimeError("validation ladder output exists with incompatible signature")
    if not SEALED_VALIDATION.is_file():
        raise RuntimeError("retrospective validation requires the existing sealed result")
    data = load_validation_galerkin_data(cfg, artifact_dir)
    dictionary = load_dictionary(
        DICTIONARY_PATH, box=tuple(cfg["physics"]["box"])
    )
    problem = data.validation_problem
    designs = {}
    for name, eta in _designs(cfg).items():
        eta = wrap_periodic(eta, problem.family)
        reconstruction = reconstruct_moments(eta, problem)
        fit_state = forcing_state(eta, problem, data.fit_bank, reconstruction)
        audit_state = forcing_state(eta, problem, data.audit_bank, reconstruction)
        system = assemble_hybrid_system(
            dictionary, data.fit_bank, fit_state.projection.weights,
            fit_state.forcing,
            chunk_size=int(cfg["production_galerkin"]["chunk_size"]),
        )
        solves, algebra = _solve_ladder(cfg, system, problem.time_weights)
        fit_certificates = _certify_view(
            cfg, dictionary, solves, problem, data.fit_bank,
            eta, reconstruction, fit_state,
        )
        audit_certificates = _certify_view(
            cfg, dictionary, solves, problem, data.audit_bank,
            eta, reconstruction, audit_state,
        )
        designs[name] = {
            "eta": np.asarray(eta).tolist(),
            "risk": float(validation_risk(eta, data)),
            "fit_forcing_audit": _forcing_state_payload(fit_state, problem),
            "audit_forcing_audit": _forcing_state_payload(audit_state, problem),
            "geometry_valid": bool(problem.family.geometry_valid(eta)),
            "ladder": [
                _row(size, row, fit, audit)
                for size, row, fit, audit in zip(
                    SIZES, algebra, fit_certificates, audit_certificates, strict=True
                )
            ],
        }
    result = {
        "signature": signature, "cache_hit": False, "ran": True,
        "passed": all(
            row["valid"] for design in designs.values() for row in design["ladder"]
        ),
        "device": device_payload(), "sizes": list(SIZES), "designs": designs,
        "views": ["validation_fit", "validation_audit"],
        "validation_was_already_open": True,
        "retrospective_frozen_geometries_only": True,
        "no_optimization": True, "selection_mutated": False,
    }
    write_json(result_path, result)
    return result


def frozen_pair_rows(result: dict[str, Any], action_key: str) -> list[dict[str, Any]]:
    eta0 = {row["basis_size"]: row for row in result["designs"]["eta0"]["ladder"]}
    eta_grad = {
        row["basis_size"]: row for row in result["designs"]["eta_grad"]["ladder"]
    }
    rows = []
    for size in SIZES:
        old = float(eta0[size][action_key])
        new = float(eta_grad[size][action_key])
        rows.append({
            "basis_size": size, "eta0_action": old, "eta_grad_action": new,
            "delta": new - old,
            "relative_improvement": (old - new) / old,
            "eta_grad_better": new < old,
            "eta0_diagnostics": eta0[size],
            "eta_grad_diagnostics": eta_grad[size],
        })
    return rows


def common_solver_comparison(
    result: dict[str, Any], action_key: str,
) -> dict[str, Any]:
    actions = {}
    risks = {}
    rows = {}
    for name in ("law", "eta0", "eta_grad"):
        row = next(
            item for item in result["designs"][name]["ladder"]
            if item["basis_size"] == 280
        )
        actions[name] = float(row[action_key])
        risks[name] = float(result["designs"][name]["risk"])
        rows[name] = row
    return {
        "actions": actions, "risks": risks, "diagnostics": rows,
        "fide_improvement_eta_grad_over_law": (
            actions["law"] - actions["eta_grad"]
        ) / actions["law"],
        "eta0_improvement_over_law": (
            actions["law"] - actions["eta0"]
        ) / actions["law"],
        "continuous_improvement_eta_grad_over_eta0": (
            actions["eta0"] - actions["eta_grad"]
        ) / actions["eta0"],
    }


def run_summary(cfg: dict[str, Any], artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_crosscheck_output_path(output_dir)
    protocol_result = json.loads((OUTPUT_ROOT / "protocol" / "result.json").read_text())
    gradient = json.loads((OUTPUT_ROOT / "gradient" / "result.json").read_text())
    selection = json.loads((OUTPUT_ROOT / "selection_ladder" / "result.json").read_text())
    validation = json.loads((OUTPUT_ROOT / "validation_ladder" / "result.json").read_text())
    pair = {
        "selection_train": frozen_pair_rows(selection, "fit_action"),
        "selection_audit": frozen_pair_rows(selection, "audit_action"),
        "validation_fit": frozen_pair_rows(validation, "fit_action"),
        "validation_audit": frozen_pair_rows(validation, "audit_action"),
    }
    common = {
        "selection_train": common_solver_comparison(selection, "fit_action"),
        "selection_audit": common_solver_comparison(selection, "audit_action"),
        "validation_fit": common_solver_comparison(validation, "fit_action"),
        "validation_audit": common_solver_comparison(validation, "audit_action"),
    }
    selection_law = float(selection["designs"]["law"]["risk"])
    validation_law = float(validation["designs"]["law"]["risk"])
    protocol = validation_protocol(cfg, validation_law)
    risk_rows = {}
    for name in ("law", "eta0", "eta_grad"):
        selection_value = float(selection["designs"][name]["risk"])
        validation_value = float(validation["designs"][name]["risk"])
        risk_rows[name] = {
            "selection_risk": selection_value,
            "selection_ratio": risk_ratio(selection_value, selection_law),
            "validation_risk": validation_value,
            "validation_ratio": risk_ratio(validation_value, validation_law),
        }
    risk_rows["eta_grad"]["distance_to_selection_3pct_boundary"] = (
        0.03 - risk_rows["eta_grad"]["selection_ratio"]
    )
    risk_rows["eta_grad"]["distance_to_declared_validation_boundary"] = (
        protocol["declared_validation_multiplier"] - 1.0
        - risk_rows["eta_grad"]["validation_ratio"]
    )
    all_orderings = all(
        row["eta_grad_better"] for rows in pair.values() for row in rows
    )
    eta_grad_certified = all(
        row["valid"]
        for source in (selection, validation)
        for row in source["designs"]["eta_grad"]["ladder"]
    )
    validation_risk_passed = bool(
        validation["designs"]["eta_grad"]["risk"]
        <= protocol["actual_predeclared_ceiling"]
    )
    fide_confirmed = bool(
        common["selection_train"]["fide_improvement_eta_grad_over_law"] > 0.0
        and common["selection_audit"]["fide_improvement_eta_grad_over_law"] > 0.0
        and common["validation_fit"]["fide_improvement_eta_grad_over_law"] > 0.0
        and common["validation_audit"]["fide_improvement_eta_grad_over_law"] > 0.0
    )
    if not protocol["resolved"]:
        classification = "E. PROTOCOL AMBIGUITY PREVENTS FINAL CLASSIFICATION"
    elif not gradient["passed"]:
        classification = "D. K=280 GRADIENT NOT VALIDATED"
    elif not all_orderings:
        classification = "C. CONTINUOUS IMPROVEMENT NOT ROBUST TO GALERKIN SPACE"
    elif not validation_risk_passed:
        classification = "B. CONTINUOUS GALERKIN ACTION IMPROVEMENT ROBUST, VALIDATION RISK FAILS"
    elif eta_grad_certified and fide_confirmed:
        classification = "A. 3% CONTINUOUS GALERKIN REFINEMENT ROBUSTLY VALIDATED"
    else:
        classification = "C. CONTINUOUS IMPROVEMENT NOT ROBUST TO GALERKIN SPACE"
    result = {
        "signature": fingerprint({"kind": "final_crosscheck_summary_v1", **_base_signature(cfg, artifact_dir)}),
        "ran": True, "passed": classification.startswith("A."),
        "protocol": {
            "initial_predeclared_resolution": protocol_result,
            "recomputed_crosscheck_values": protocol,
        },
        "gradient": gradient,
        "pairwise_ordering": pair, "common_K280": common,
        "risk": risk_rows, "all_pairwise_orderings_pass": all_orderings,
        "eta_grad_all_certificates_pass": eta_grad_certified,
        "predeclared_validation_risk_passed": validation_risk_passed,
        "fide_improvement_confirmed": fide_confirmed,
        "classification": classification,
        "no_optimization": True, "selection_mutated": False,
        "pareto_sweep_run": False,
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CACHE_PATH", "DICTIONARY_PATH", "ETA_GRAD", "OUTPUT_ROOT", "SIZES",
    "common_solver_comparison", "directional_fd_row", "frozen_pair_rows",
    "require_crosscheck_output_path", "risk_ratio", "run_gradient_check",
    "run_protocol", "run_selection_ladder", "run_summary",
    "run_validation_ladder", "validation_protocol",
]
