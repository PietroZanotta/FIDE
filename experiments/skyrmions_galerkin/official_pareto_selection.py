"""Selection-only phases for the official K=280 Galerkin Pareto sweep.

This module intentionally contains no validation loader or validation-bank path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import fingerprint

from .full_gradient import forcing_state, wrap_periodic
from .galerkin_only import (
    GalerkinOnlyContext, _forcing_state_payload, device_payload,
)
from .galerkin_only_data import load_selection_galerkin_data
from .measurements import local_sensor_designs, random_sensor_designs
from .official_pareto_common import (
    ALLOWANCES, ARTIFACT_DIR, DICTIONARY_PATH, HISTORICAL_PARETO, OFFICIAL_K,
    OUTPUT_ROOT, TRAIN_CACHE, allowance_slug, derived_seed, payload_sha256,
    actions_nonincreasing, read_json, require_frozen_protocol, retain_incumbent,
    selection_ceiling, write_json,
)
from .production_artifacts import file_sha256


# Geometry values are copied without loading the old Pareto JSON during selection.
# Their source file is hash-frozen by the protocol, but its old validation fields
# never enter this module's process.
HISTORICAL_GEOMETRIES = (
    (0.5, [0.8882240021144415, 0.2265900282857875, 1.3089283029966885, 0.8628255147902797,
           0.7866652061176428, 0.5418032213434409, 1.6161758592555022, 0.584353406982718]),
    (1.0, [0.8916497660872147, 0.21592104181723273, 1.3254990498968335, 0.861978425574543,
           0.7740333752184337, 0.5278590825172568, 1.6268094810638665, 0.5775087460426114]),
    (2.0, [0.894577442995983, 0.20411161892557242, 1.3400864770591099, 0.8635508182176649,
           0.76001873964639, 0.5143515626749267, 1.6376150652013575, 0.5666851609212433]),
    (3.0, [0.8954153767761239, 0.20592631632470587, 1.3343788098383822, 0.8654288352917223,
           0.7508355365766083, 0.5179100329264751, 1.6423735249784726, 0.5883599695898114]),
    (4.0, [0.8954153767761239, 0.20592631632470587, 1.3343788098383822, 0.8654288352917223,
           0.7508355365766083, 0.5179100329264751, 1.6423735249784726, 0.5883599695898114]),
    (5.0, [0.8954153767761239, 0.20592631632470587, 1.3343788098383822, 0.8654288352917223,
           0.7508355365766083, 0.5179100329264751, 1.6423735249784726, 0.5883599695898114]),
)

EXPECTED_ETA0_ACTION = 0.2935000591956778
ETA_GRAD = np.asarray([
    0.895371148114089, 0.205982940238786,
    1.334525121515147, 0.865464965382237,
    0.750749623351011, 0.518133188490931,
    1.642405611981796, 0.588309862016330,
], dtype=np.float64)
EXPECTED_ETA0_GRADIENT = np.asarray([
    0.6208703416229037, -0.4782185775165369, -2.6498354290746398,
    -0.8797681284028329, 0.516433358902474, -1.2408541681590477,
    -0.3738357554626508, -0.9392258457096148,
])
EXPECTED_ACTIONS = {
    "law": 0.374832445634,
    "eta0": 0.2935000591956778,
    "eta_grad": 0.292740724350,
}


def _require_hot_cache() -> None:
    metadata = TRAIN_CACHE / "metadata.json"
    if not metadata.is_file():
        # The existing implementation uses this exact filename.
        candidates = list(TRAIN_CACHE.glob("*metadata*.json"))
        if len(candidates) != 1:
            raise RuntimeError("validated K=280 train cache is incomplete")
        metadata = candidates[0]
    for index in range(13):
        for stem in ("train_values", "train_gradients"):
            if not (TRAIN_CACHE / f"{stem}_t{index:02d}.npy").is_file():
                raise RuntimeError("validated K=280 train cache is incomplete")


def selection_context(cfg: dict[str, Any]) -> GalerkinOnlyContext:
    _require_hot_cache()
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    return GalerkinOnlyContext(
        cfg, ARTIFACT_DIR, data, DICTIONARY_PATH, cache_dir=TRAIN_CACHE,
    )


def _signature(protocol: dict[str, Any], kind: str, extra: Any = None) -> str:
    return fingerprint({
        "kind": kind,
        "protocol_sha256": protocol["protocol_sha256"],
        "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "artifact_manifest_sha256": file_sha256(
            ARTIFACT_DIR / "isolated_artifact_manifest.json"
        ),
        "extra": extra,
    })


def run_reproduction(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    output_path = OUTPUT_ROOT / "reproduction" / "result.json"
    signature = _signature(protocol, "official_selection_reproduction_v1")
    if output_path.is_file():
        previous = read_json(output_path)
        if previous.get("signature") == signature and previous.get("passed"):
            return {**previous, "cache_hit": True}
        raise RuntimeError("incompatible official reproduction output exists")
    context = selection_context(cfg)
    designs = {
        "law": jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64),
        "eta0": jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64),
        "eta_grad": jnp.asarray(ETA_GRAD, dtype=jnp.float64),
    }
    rows: dict[str, Any] = {}
    for name, eta in designs.items():
        first = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=True)
        second = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=True)
        certificate = context.certify(first)
        rows[name] = {
            "certificate": certificate,
            "repeated_action_absolute_difference": abs(float(first.action) - float(second.action)),
            "repeated_gradient_max_absolute_difference": float(jnp.max(jnp.abs(first.gradient - second.gradient))),
            "gradient_finite": bool(jnp.all(jnp.isfinite(first.gradient))),
            "expected_action": EXPECTED_ACTIONS[name],
            "action_relative_discrepancy": abs(float(first.action) - EXPECTED_ACTIONS[name]) / max(abs(EXPECTED_ACTIONS[name]), 1e-30),
        }
    eta0_gradient = np.asarray(rows["eta0"]["certificate"]["gradient"])
    gradient_relerr = float(np.linalg.norm(eta0_gradient - EXPECTED_ETA0_GRADIENT) / np.linalg.norm(EXPECTED_ETA0_GRADIENT))
    law = rows["law"]["certificate"]
    law_expected_status = bool(
        not law["certified"]
        and law["algebra_valid"]
        and law["train_forcing_audit"]["valid"]
        and law["audit_forcing_audit"]["valid"]
        and not law["heldout_certificate"]["valid"]
    )
    passed = bool(
        all(row["gradient_finite"] for row in rows.values())
        and all(row["repeated_action_absolute_difference"] <= 1e-12 for row in rows.values())
        and all(row["repeated_gradient_max_absolute_difference"] <= 1e-12 for row in rows.values())
        and all(row["action_relative_discrepancy"] <= 2e-9 for row in rows.values())
        and gradient_relerr <= 1e-8
        and rows["eta0"]["certificate"]["certified"]
        and rows["eta_grad"]["certificate"]["certified"]
        and law_expected_status
    )
    result = {
        "ran": True, "passed": passed, "cache_hit": False,
        "signature": signature, "device": device_payload(), "basis_size": OFFICIAL_K,
        "designs": rows, "eta0_gradient_relative_discrepancy": gradient_relerr,
        "law_known_selection_audit_energy_failure_reproduced": law_expected_status,
        "validation_accessed": False, "old_validation_arrays_loaded": False,
    }
    write_json(output_path, result, overwrite=False)
    if not passed:
        raise RuntimeError("official K=280 selection reproduction failed")
    return result


def _eta_key(eta: Any) -> str:
    return payload_sha256(np.asarray(eta, dtype=np.float64).tolist())[:16]


def _deduplicate(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        eta = np.asarray(row["eta"], dtype=np.float64)
        if any(np.linalg.norm(eta - np.asarray(old["eta"])) <= 1e-12 for old in kept):
            continue
        kept.append(row)
        if len(kept) >= int(limit):
            break
    return kept


def prepare_start_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    reproduction = read_json(OUTPUT_ROOT / "reproduction" / "result.json")
    if not reproduction.get("passed"):
        raise RuntimeError("successful reproduction is required before start generation")
    output_path = OUTPUT_ROOT / "selection" / "starts.json"
    signature = _signature(protocol, "official_start_manifest_v1")
    if output_path.is_file():
        previous = read_json(output_path)
        if previous.get("signature") == signature:
            return {**previous, "cache_hit": True}
        raise RuntimeError("incompatible official start manifest exists")
    context = selection_context(cfg)
    family = context.data.selection_problem.family
    law_eta = np.asarray(cfg["envelope"]["law_eta"], dtype=np.float64)
    law_risk = float(context._risk(jnp.asarray(law_eta)))
    historical = [
        {"id": f"historical_{allowance_slug(p)}pct", "eta": eta,
         "provenance": "frozen historical skyrmion Pareto geometry",
         "historical_allowance_percent": p}
        for p, eta in HISTORICAL_GEOMETRIES
    ]
    historical_unique = _deduplicate(historical, len(historical))
    local_seed = derived_seed(cfg["seed"], "local_starts")["seed"]
    local_values = local_sensor_designs(
        jax.random.PRNGKey(local_seed),
        jnp.asarray([row["eta"] for row in historical_unique]),
        count_per_center=2, scale=2.0e-4, family=family,
    )
    local = [
        {"id": f"local_{index:02d}", "eta": np.asarray(eta).tolist(),
         "provenance": "predeclared deterministic local perturbation"}
        for index, eta in enumerate(np.asarray(local_values))
    ]
    global_seed = derived_seed(cfg["seed"], "global_starts")["seed"]
    global_values = random_sensor_designs(
        jax.random.PRNGKey(global_seed), count=48, family=family, oversample=16,
    )
    global_rows = [
        {"id": f"global_{index:02d}", "eta": np.asarray(eta).tolist(),
         "provenance": "predeclared deterministic global candidate"}
        for index, eta in enumerate(np.asarray(global_values))
    ]
    derived_rows = local + global_rows
    for row in derived_rows:
        eta = jnp.asarray(row["eta"], dtype=jnp.float64)
        row["geometry_valid"] = bool(family.geometry_valid(eta))
        row["risk"] = float(context._risk(eta)) if row["geometry_valid"] else None
        row["action"] = None
        if row["geometry_valid"] and row["risk"] <= selection_ceiling(law_risk, max(ALLOWANCES)):
            evaluation = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=False)
            payload = context.payload(evaluation)
            row["action"] = payload["action"]
            row["train_numerically_valid"] = bool(
                payload["train_forcing_audit"]["valid"] and payload["geometry_valid"]
            )
        else:
            row["train_numerically_valid"] = False
    static: dict[str, Any] = {}
    for allowance in ALLOWANCES:
        ceiling = selection_ceiling(law_risk, allowance)
        rows: list[dict[str, Any]] = [{
            "id": "law", "eta": law_eta.tolist(), "provenance": "frozen Law geometry",
            "risk": law_risk,
        }]
        feasible_history = []
        for row in historical_unique:
            risk = float(context._risk(jnp.asarray(row["eta"], dtype=jnp.float64)))
            enriched = {**row, "risk": risk}
            if risk <= ceiling:
                feasible_history.append(enriched)
        feasible_history.sort(key=lambda row: (
            abs(float(row["historical_allowance_percent"]) - allowance), row["id"]
        ))
        rows.extend(feasible_history)
        eta_grad_risk = float(context._risk(jnp.asarray(ETA_GRAD, dtype=jnp.float64)))
        if eta_grad_risk <= ceiling:
            rows.append({
                "id": "previous_continuous_eta_grad", "eta": np.asarray(ETA_GRAD).tolist(),
                "provenance": "selection-frozen Galerkin 3pct continuous geometry",
                "risk": eta_grad_risk,
            })
        feasible_derived = [
            row for row in derived_rows
            if row.get("train_numerically_valid") and float(row["risk"]) <= ceiling
        ]
        feasible_derived.sort(key=lambda row: (float(row["action"]), row["id"]))
        rows.extend(feasible_derived[:2])
        # Leave one slot for the dynamically resolved mandatory incumbent.
        static_limit = 8 if allowance == ALLOWANCES[0] else 7
        static[allowance_slug(allowance)] = {
            "allowance_percent": allowance, "selection_ceiling": ceiling,
            "mandatory_incumbent_placeholder": allowance != ALLOWANCES[0],
            "starts": _deduplicate(rows, static_limit),
        }
    result = {
        "ran": True, "passed": True, "cache_hit": False, "signature": signature,
        "protocol_sha256": protocol["protocol_sha256"],
        "historical_source_sha256": file_sha256(HISTORICAL_PARETO),
        "historical_file_not_parsed_in_selection_process": True,
        "old_validation_accessed": False, "law_risk": law_risk,
        "derived_candidate_pool": derived_rows, "allowances": static,
        "incumbent_resolution_rule": "prepend exact preceding certified winner then deduplicate",
    }
    write_json(output_path, result, overwrite=False)
    return result


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


def _periodic_delta(candidate: jax.Array, center: jax.Array, box: Any) -> jax.Array:
    shaped = (candidate - center).reshape((-1, 2))
    box_array = jnp.asarray(box, dtype=jnp.float64)
    return (shaped - box_array * jnp.round(shaped / box_array)).reshape((-1,))


def _certify_cached(context: GalerkinOnlyContext, evaluation: Any) -> dict[str, Any]:
    cache_path = OUTPUT_ROOT / "selection" / "certification_cache" / f"{_eta_key(evaluation.eta)}.json"
    signature = fingerprint({
        "kind": "official_selection_certificate_v1",
        "protocol_sha256": read_json(OUTPUT_ROOT / "protocol.json")["protocol_sha256"],
        "eta": np.asarray(evaluation.eta).tolist(), "basis_size": OFFICIAL_K,
    })
    if cache_path.is_file():
        old = read_json(cache_path)
        if old.get("signature") == signature:
            return old["certificate"]
        raise RuntimeError("incompatible selection certificate cache entry")
    certificate = context.certify(evaluation)
    write_json(cache_path, {"signature": signature, "certificate": certificate}, overwrite=False)
    return certificate


def _trajectory_signature(protocol: dict[str, Any], start: dict[str, Any], allowance: float,
                          risk_ceiling: float) -> str:
    return _signature(protocol, "official_trust_trajectory_v1", {
        "start": start, "allowance_percent": allowance, "risk_ceiling": risk_ceiling,
        "optimizer": protocol["optimizer"],
    })


def _run_trajectory(cfg: dict[str, Any], protocol: dict[str, Any],
                    context: GalerkinOnlyContext, start: dict[str, Any],
                    allowance: float, output_path: Path, risk_ceiling: float) -> dict[str, Any]:
    signature = _trajectory_signature(protocol, start, allowance, risk_ceiling)
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("signature") == signature and not old.get("in_progress", True):
            return {**old, "cache_hit": True}
        raise RuntimeError(f"incompatible trajectory output exists: {output_path}")
    settings = protocol["optimizer"]
    family = context.data.selection_problem.family
    center = wrap_periodic(jnp.asarray(start["eta"], dtype=jnp.float64), family)
    eta = center
    current = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=True)
    start_certificate = _certify_cached(context, current)
    if not (start_certificate["certified"] and float(current.risk) <= risk_ceiling):
        result = {
            "signature": signature, "in_progress": False, "cache_hit": False,
            "allowance_percent": allowance, "start": start,
            "start_certificate": start_certificate, "eligible": False,
            "reason": "start failed exact selection gates", "history": [],
            "evaluation_count": 1, "certification_count": 1,
        }
        write_json(output_path, result, overwrite=False)
        return result
    last_certified_eta, last_certified = eta, start_certificate
    history: list[dict[str, Any]] = []
    evaluation_count, certification_count = 1, 1
    step_length = float(settings["initial_step"])
    for step in range(int(settings["maximum_accepted_step_attempts"])):
        _, risk_gradient = context._risk_value_grad(eta)
        direction = -current.gradient
        risk_slope = float(jnp.dot(risk_gradient, direction))
        risk_norm_sq = float(jnp.dot(risk_gradient, risk_gradient))
        if risk_slope > 0.0 and risk_norm_sq > 1e-30:
            direction = direction - (risk_slope / risk_norm_sq) * risk_gradient
            direction = direction - 0.02 * jnp.linalg.norm(direction) * (
                risk_gradient / jnp.sqrt(risk_norm_sq)
            )
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
        accepted = False
        accepted_candidate = None
        attempts = []
        for backtrack in range(int(settings["maximum_backtracks_per_step"])):
            length = step_length * float(settings["backtrack_factor"]) ** backtrack
            proposal = wrap_periodic(eta + length * direction, family)
            if float(jnp.linalg.norm(_periodic_delta(proposal, center, family.box))) > float(settings["trust_radius"]) * (1 + 1e-12):
                attempts.append({"length": length, "accepted": False, "reason": "trust_radius"})
                continue
            if not bool(family.geometry_valid(proposal)):
                attempts.append({"length": length, "accepted": False, "reason": "geometry"})
                continue
            risk = float(context._risk(proposal))
            if risk > risk_ceiling:
                attempts.append({"length": length, "accepted": False, "reason": "risk", "risk": risk})
                continue
            candidate = context.evaluate(proposal, basis_size=OFFICIAL_K, with_gradient=True)
            evaluation_count += 1
            payload = context.payload(candidate)
            delta = _periodic_delta(proposal, eta, family.box)
            predicted = max(-float(jnp.dot(current.gradient, delta)), 1e-30)
            actual = float(current.action) - float(candidate.action)
            rho = actual / predicted
            rank_stable = bool(np.array_equal(
                np.asarray(candidate.solve.numerical_rank), np.asarray(current.solve.numerical_rank)
            ))
            accepted = bool(
                actual > float(settings["replacement_tolerance"])
                and payload["train_forcing_audit"]["valid"]
                and payload["geometry_valid"] and _search_algebra_valid(cfg, payload)
                and rank_stable
            )
            attempts.append({
                "length": length, "accepted": accepted, "risk": risk,
                "action": float(candidate.action), "actual_reduction": actual,
                "predicted_reduction": predicted, "rho": rho, "rank_stable": rank_stable,
            })
            if accepted:
                accepted_candidate = candidate
                step_length = min(
                    float(settings["successful_step_cap"]) if rho >= 0.75 else length,
                    float(settings["trust_radius"]),
                )
                break
        if accepted and accepted_candidate is not None:
            eta, current = accepted_candidate.eta, accepted_candidate
            if (step + 1) % int(settings["periodic_full_certificate_every"]) == 0:
                checkpoint = _certify_cached(context, current)
                certification_count += 1
                if checkpoint["certified"]:
                    last_certified_eta, last_certified = eta, checkpoint
                else:
                    eta = last_certified_eta
                    current = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=True)
                    evaluation_count += 1
                    accepted = False
                    attempts.append({"accepted": False, "reason": "heldout_certificate_failed_reverted"})
        history.append({
            "step": step + 1, "accepted": accepted, "eta": np.asarray(eta).tolist(),
            "action": float(current.action), "risk": float(current.risk),
            "step_length_next": step_length, "attempts": attempts,
        })
        if not accepted:
            break
        if float(jnp.linalg.norm(_periodic_delta(eta, center, family.box))) >= 0.999 * float(settings["trust_radius"]):
            break
    final_certificate = _certify_cached(context, current)
    certification_count += 1
    if final_certificate["certified"] and float(final_certificate["risk"]) <= risk_ceiling:
        last_certified_eta, last_certified = eta, final_certificate
    elif not np.array_equal(np.asarray(eta), np.asarray(last_certified_eta)):
        eta = last_certified_eta
    result = {
        "signature": signature, "in_progress": False, "cache_hit": False,
        "allowance_percent": allowance, "start": start,
        "end_eta": np.asarray(last_certified_eta).tolist(),
        "start_action": float(start_certificate["action"]),
        "end_action": float(last_certified["action"]),
        "action_reduction": float(start_certificate["action"] - last_certified["action"]),
        "end_risk": float(last_certified["risk"]),
        "steps_accepted": sum(bool(row["accepted"]) for row in history),
        "history": history, "evaluation_count": evaluation_count,
        "certification_count": certification_count,
        "start_certificate": start_certificate, "final_certificate": last_certified,
        "eligible": bool(last_certified["certified"] and float(last_certified["risk"]) <= risk_ceiling),
    }
    write_json(output_path, result, overwrite=False)
    return result


def run_selection_sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    reproduction = read_json(OUTPUT_ROOT / "reproduction" / "result.json")
    starts_manifest = read_json(OUTPUT_ROOT / "selection" / "starts.json")
    if not reproduction.get("passed") or not starts_manifest.get("passed"):
        raise RuntimeError("reproduction and start manifest must pass before selection")
    draft_path = OUTPUT_ROOT / "selection" / "selection_draft.json"
    signature = _signature(protocol, "official_sequential_selection_v1", starts_manifest["signature"])
    if draft_path.is_file():
        old = read_json(draft_path)
        if old.get("signature") == signature and old.get("passed"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible official selection draft exists")
    context = selection_context(cfg)
    law_risk = float(starts_manifest["law_risk"])
    law_action = float(reproduction["designs"]["law"]["certificate"]["action"])
    results = []
    incumbent: dict[str, Any] | None = None
    for allowance in ALLOWANCES:
        slug = allowance_slug(allowance)
        ceiling = selection_ceiling(law_risk, allowance)
        static = list(starts_manifest["allowances"][slug]["starts"])
        actual_starts = []
        if incumbent is not None:
            actual_starts.append({
                "id": "mandatory_preceding_incumbent", "eta": incumbent["eta"],
                "provenance": f"certified winner at {results[-1]['allowance_percent']} percent",
                "risk": incumbent["risk"],
            })
        actual_starts.extend(static)
        actual_starts = _deduplicate(actual_starts, 8)
        allowance_dir = OUTPUT_ROOT / "selection" / f"allowance_{slug}"
        trajectories = []
        for index, start in enumerate(actual_starts):
            trajectories.append(_run_trajectory(
                cfg, protocol, context, start, allowance,
                allowance_dir / f"trajectory_{index:02d}_{start['id']}.json", ceiling,
            ))
        eligible = [row for row in trajectories if row.get("eligible")]
        candidates = [{
            "kind": "trajectory_endpoint", "source": row["start"]["id"],
            "eta": row["end_eta"], "action": row["end_action"],
            "certificate": row["final_certificate"],
        } for row in eligible]
        if incumbent is not None:
            candidates.append({
                "kind": "retained_incumbent", "source": "mandatory_preceding_incumbent",
                "eta": incumbent["eta"], "action": incumbent["action"],
                "certificate": incumbent,
            })
        if not candidates:
            raise RuntimeError(f"no certified feasible candidate at allowance {allowance}")
        candidates.sort(key=lambda row: (float(row["action"]), row["source"]))
        best = candidates[0]
        retained = False
        if incumbent is not None and retain_incumbent(
            best["action"], incumbent["action"], protocol["optimizer"]["replacement_tolerance"]
        ):
            winner = incumbent
            retained = True
            winner_source = "mandatory_preceding_incumbent_retained"
        else:
            winner = best["certificate"]
            winner_source = best["source"]
        if not (winner["certified"] and float(winner["risk"]) <= ceiling):
            raise RuntimeError("winner failed exact frozen selection gates")
        previous_action = None if incumbent is None else float(incumbent["action"])
        row = {
            "allowance_percent": allowance, "risk_ceiling": ceiling,
            "actual_starts": actual_starts, "trajectories": trajectories,
            "eligible_candidate_count": len(candidates), "winner": winner,
            "winner_source": winner_source, "incumbent_retained": retained,
            "previous_incumbent_action": previous_action,
            "action_difference_from_previous_incumbent": None if previous_action is None else float(winner["action"]) - previous_action,
            "selection_risk_increase_percent": 100.0 * (float(winner["risk"]) / law_risk - 1.0),
            "budget_used_fraction": (float(winner["risk"]) / law_risk - 1.0) / (allowance / 100.0),
            "selection_reduction_vs_law": (law_action - float(winner["action"])) / law_action,
        }
        write_json(allowance_dir / "result.json", row, overwrite=False)
        results.append(row)
        incumbent = winner
    actions = [float(row["winner"]["action"]) for row in results]
    monotone = actions_nonincreasing(actions, protocol["optimizer"]["replacement_tolerance"])
    result = {
        "ran": True, "passed": monotone, "cache_hit": False,
        "signature": signature, "protocol_sha256": protocol["protocol_sha256"],
        "basis_size": OFFICIAL_K, "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "law_eta": cfg["envelope"]["law_eta"], "law_risk": law_risk,
        "law_selection_action": law_action, "allowances": results,
        "selection_action_nonincreasing": monotone,
        "selection_frozen": False, "validation_accessed": False,
        "old_validation_accessed": False, "deep_ritz_used": False,
    }
    write_json(draft_path, result, overwrite=False)
    if not monotone:
        raise RuntimeError("official selection action is not nonincreasing")
    return result


def _admissible_direction(context: GalerkinOnlyContext, eta: jax.Array,
                          ceiling: float, seed: int) -> jax.Array:
    _, risk_gradient = context._risk_value_grad(eta)
    norm_sq = jnp.dot(risk_gradient, risk_gradient)
    family = context.data.selection_problem.family
    for attempt in range(256):
        direction = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(seed), attempt), eta.shape, dtype=jnp.float64)
        if float(norm_sq) > 1e-30:
            direction = direction - jnp.dot(direction, risk_gradient) / norm_sq * risk_gradient
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
        valid = True
        for epsilon in (3e-4, 1e-4):
            for sign in (-1.0, 1.0):
                point = wrap_periodic(eta + sign * epsilon * direction, family)
                valid = valid and bool(family.geometry_valid(point)) and float(context._risk(point)) <= ceiling
        if valid:
            return direction
    raise RuntimeError("could not construct a deterministic admissible finalist direction")


def run_finalist_audits(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    draft = read_json(OUTPUT_ROOT / "selection" / "selection_draft.json")
    if not draft.get("passed") or draft.get("selection_frozen"):
        raise RuntimeError("unfrozen successful selection draft is required")
    output_path = OUTPUT_ROOT / "finalist_gradient_audits" / "result.json"
    signature = _signature(protocol, "official_finalist_audits_v1", draft["signature"])
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("signature") == signature and old.get("passed"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible finalist audit output exists")
    context = selection_context(cfg)
    unique: list[dict[str, Any]] = []
    for row in draft["allowances"]:
        key = _eta_key(row["winner"]["eta"])
        existing = next((item for item in unique if item["eta_key"] == key), None)
        if existing is None:
            unique.append({
                "eta_key": key, "eta": row["winner"]["eta"],
                "allowances_percent": [row["allowance_percent"]],
                "ceiling": row["risk_ceiling"],
            })
        else:
            existing["allowances_percent"].append(row["allowance_percent"])
            existing["ceiling"] = min(existing["ceiling"], row["risk_ceiling"])
    base_seed = derived_seed(cfg["seed"], "finalist_audit_directions")["seed"]
    audits = []
    for index, item in enumerate(unique):
        eta = jnp.asarray(item["eta"], dtype=jnp.float64)
        center = context.evaluate(eta, basis_size=OFFICIAL_K, with_gradient=True)
        direction = _admissible_direction(context, eta, item["ceiling"], base_seed + index)
        ad = float(jnp.dot(center.gradient, direction))
        rows = []
        center_rank = np.asarray(center.solve.numerical_rank)
        for epsilon in (3e-4, 1e-4):
            points = []
            for sign in (1.0, -1.0):
                point_eta = wrap_periodic(eta + sign * epsilon * direction, context.data.selection_problem.family)
                evaluation = context.evaluate(point_eta, basis_size=OFFICIAL_K, with_gradient=False)
                payload = context.payload(evaluation)
                audit_state = forcing_state(
                    point_eta, context.data.selection_problem, context.data.audit_bank,
                    evaluation.reconstruction,
                )
                audit_forcing = _forcing_state_payload(audit_state, context.data.selection_problem)
                points.append({
                    "action": float(evaluation.action), "risk": float(evaluation.risk),
                    "rank_stable": bool(np.array_equal(np.asarray(evaluation.solve.numerical_rank), center_rank)),
                    "train_forcing_valid": payload["train_forcing_audit"]["valid"],
                    "audit_forcing_valid": audit_forcing["valid"],
                    "algebra_valid": _search_algebra_valid(cfg, payload),
                    "geometry_valid": payload["geometry_valid"],
                })
            fd = (points[0]["action"] - points[1]["action"]) / (2.0 * epsilon)
            rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
            valid_points = all(
                point["rank_stable"] and point["train_forcing_valid"]
                and point["audit_forcing_valid"] and point["algebra_valid"]
                and point["geometry_valid"]
                for point in points
            )
            rows.append({
                "epsilon": epsilon, "finite_difference": fd,
                "relative_discrepancy": rel, "sign_agrees": bool(fd * ad > 0.0),
                "points_valid": valid_points, "plus": points[0], "minus": points[1],
            })
        passed = bool(all(row["points_valid"] and row["sign_agrees"] for row in rows)
                      and min(row["relative_discrepancy"] for row in rows) <= 0.02)
        audits.append({
            **item, "direction": np.asarray(direction).tolist(),
            "ad_directional_derivative": ad, "rows": rows, "passed": passed,
        })
        write_json(OUTPUT_ROOT / "finalist_gradient_audits" / f"winner_{index:02d}.json", audits[-1], overwrite=False)
    result = {
        "ran": True, "passed": all(row["passed"] for row in audits),
        "cache_hit": False, "signature": signature,
        "unique_winner_count": len(unique), "audits": audits,
        "selection_modified": False, "validation_accessed": False,
    }
    write_json(output_path, result, overwrite=False)
    if not result["passed"]:
        raise RuntimeError("a frozen finalist failed the local derivative audit")
    return result


def freeze_selection(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    draft_path = OUTPUT_ROOT / "selection" / "selection_draft.json"
    audit_path = OUTPUT_ROOT / "finalist_gradient_audits" / "result.json"
    draft, audits = read_json(draft_path), read_json(audit_path)
    if not draft.get("passed") or not audits.get("passed"):
        raise RuntimeError("successful sweep and finalist audits are required")
    selection_path = OUTPUT_ROOT / "selection" / "pareto_selection.json"
    manifest_path = OUTPUT_ROOT / "selection" / "manifest.json"
    if selection_path.exists() or manifest_path.exists():
        raise RuntimeError("official selection is already frozen; refusing overwrite")
    frozen = {
        **draft,
        "selection_frozen": True, "validation_accessed": False,
        "winner_geometry_immutable": True,
        "finalist_gradient_audits_sha256": file_sha256(audit_path),
        "start_manifest_sha256": file_sha256(OUTPUT_ROOT / "selection" / "starts.json"),
        "selection_draft_sha256": file_sha256(draft_path),
    }
    write_json(selection_path, frozen, overwrite=False)
    selection_hash = file_sha256(selection_path)
    manifest = {
        "schema_version": 1, "selection_frozen": True,
        "validation_accessed": False, "deep_ritz_used": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "pareto_selection_sha256": selection_hash,
        "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "artifact_manifest_sha256": file_sha256(ARTIFACT_DIR / "isolated_artifact_manifest.json"),
        "allowances_percent": list(ALLOWANCES),
        "winning_etas": [row["winner"]["eta"] for row in draft["allowances"]],
        "winner_count": len(draft["allowances"]),
        "validation_arrays_generated": False,
        "old_validation_accessed_during_selection": False,
    }
    write_json(manifest_path, manifest, overwrite=False)
    return {**manifest, "manifest_sha256": file_sha256(manifest_path), "passed": True}


__all__ = [
    "HISTORICAL_GEOMETRIES", "freeze_selection", "prepare_start_manifest",
    "run_finalist_audits", "run_reproduction", "run_selection_sweep",
    "selection_context",
]
