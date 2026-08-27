"""Development-only v2-bank dual-support start diagnostic for Pareto v3."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time
from typing import Any

import numpy as np

from .pareto_v2_selection import (
    _distance,
    _screen_rows_batched,
    load_bank,
    selection_data,
)
from .pareto_v2_common import require_protocol as require_v2_protocol
from .pareto_v3_common import (
    ALLOWANCES,
    ALL_ALLOWANCES_DIAGNOSTIC_ROOT,
    OUTPUT_ROOT,
    PROTOCOL_HASH_PATH,
    PROTOCOL_PATH,
    ROOT,
    DIAGNOSTIC_ROOT,
    MINIMUM_RESS,
    V2_OUTPUT_ROOT,
    atomic_json,
    eta_key,
    file_sha256,
    read_json,
    selection_ceiling,
    verify_v2_frozen,
    verify_v3_phase1_frozen,
)


def _quantiles(values: list[float]) -> dict[str, float]:
    levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    rows = np.quantile(np.asarray(values, dtype=np.float64), levels)
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "p05", "p25", "median", "p75", "p95", "maximum"),
            rows,
            strict=True,
        )
    }


def diagnose_v2_audit_starts(cfg: dict[str, Any]) -> dict[str, Any]:
    path = DIAGNOSTIC_ROOT / "summary.json"
    if path.exists():
        saved = read_json(path)
        current = verify_v2_frozen()
        if saved["v2_output_tree_sha256"] != current["output_tree_sha256"]:
            raise RuntimeError("v2 changed after the diagnostic")
        return {**saved, "cache_hit": True}

    inventory = verify_v2_frozen()
    screening_path = V2_OUTPUT_ROOT / "screening" / "candidate_pool.json"
    screening = read_json(screening_path)
    law_risk = float(screening["law_risk"])
    ceiling = selection_ceiling(law_risk, 0.5)
    candidates = [
        row
        for row in screening["rows"]
        if float(row["scientific_selection_risk"]) <= ceiling
        and bool(row["geometry_valid"])
        and bool(row["projection_valid"])
        and float(row["minimum_ess_fraction"]) >= MINIMUM_RESS
    ]
    if len(candidates) != 193:
        raise RuntimeError(
            f"v2 0.5% screening-feasible count changed: {len(candidates)}"
        )

    data = selection_data(cfg, "search_train", "periodic_audit")
    audit_bank = load_bank("periodic_audit")
    started = time.perf_counter()
    audited = _screen_rows_batched(cfg, data, audit_bank, candidates, 8)
    elapsed = time.perf_counter() - started
    projection_tolerance = float(data.selection_problem.forcing_config.projection_tolerance)
    forcing_tolerance = float(data.selection_problem.forcing_config.forcing_mean_tolerance)
    condition_limit = float(data.selection_problem.forcing_config.max_covariance_condition)

    rows = []
    for screen, audit in zip(candidates, audited, strict=True):
        diagnostics = audit["screen"]
        audit_projection = (
            float(diagnostics["maximum_projection_residual"])
            <= projection_tolerance
        )
        audit_ress = float(diagnostics["minimum_ess_fraction"])
        audit_support = bool(
            audit_projection
            and audit_ress >= MINIMUM_RESS
            and float(diagnostics["maximum_forcing_mean"]) <= forcing_tolerance
            and float(diagnostics["maximum_covariance_condition"]) <= condition_limit
        )
        rows.append(
            {
                "candidate_id": screen["candidate_id"],
                "eta": screen["eta"],
                "scientific_selection_risk": screen["scientific_selection_risk"],
                "screen_ress": screen["minimum_ess_fraction"],
                "audit_ress": audit_ress,
                "robust_ress": min(
                    float(screen["minimum_ess_fraction"]), audit_ress
                ),
                "audit_projection_valid": audit_projection,
                "audit_support_valid": audit_support,
                "audit": diagnostics,
            }
        )

    projection_rows = [row for row in rows if row["audit_projection_valid"]]
    robust = [
        row
        for row in projection_rows
        if row["audit_ress"] >= MINIMUM_RESS
    ]
    robust_support = [row for row in rows if row["audit_support_valid"]]
    robust.sort(key=lambda row: (-row["robust_ress"], row["candidate_id"]))
    box = data.selection_problem.family.box
    pairwise = [
        _distance(left["eta"], right["eta"], box)
        for left, right in combinations(robust, 2)
    ]
    if len(robust) >= 10:
        classification = "A. ROBUST 0.5% STARTS CLEARLY EXIST"
    elif robust:
        classification = "B. ROBUST 0.5% STARTS EXIST BUT ARE RARE"
    else:
        classification = "C. NO ROBUST 0.5% START FOUND ON v2 BANKS"

    result = {
        "schema_version": 3,
        "development_diagnostic_only": True,
        "official_v3_result": False,
        "v2_frozen": True,
        "v2_output_tree_sha256": inventory["output_tree_sha256"],
        "v2_protocol_sha256": inventory["expected_hashes"]["inner_protocol"],
        "v2_screening_sha256": file_sha256(screening_path),
        "allowance_percent": 0.5,
        "law_risk": law_risk,
        "risk_ceiling": ceiling,
        "minimum_ress_threshold": MINIMUM_RESS,
        "classification_rule": {
            "A_minimum_count": 10,
            "B_count_range": [1, 9],
            "C_count": 0,
        },
        "classification": classification,
        "proceed_to_v3": bool(robust),
        "total_screening_feasible": len(candidates),
        "audit_projection_valid_count": len(projection_rows),
        "audit_ress_valid_count": len(robust),
        "audit_full_support_valid_count": len(robust_support),
        "audit_ress_distribution": _quantiles(
            [row["audit_ress"] for row in rows]
        ),
        "best_audit_ress": max(row["audit_ress"] for row in rows),
        "top_20_robust_candidates": robust[:20],
        "robust_geometry_diversity": {
            "candidate_count": len(robust),
            "pair_count": len(pairwise),
            "minimum_periodic_distance": None if not pairwise else min(pairwise),
            "median_periodic_distance": None
            if not pairwise
            else float(np.median(pairwise)),
            "maximum_periodic_distance": None if not pairwise else max(pairwise),
        },
        "all_rows": rows,
        "elapsed_seconds": elapsed,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "validation_accessed": False,
    }
    atomic_json(DIAGNOSTIC_ROOT / "v2_inventory.json", inventory, immutable=True)
    atomic_json(path, result, immutable=True)
    return result


def _official_v3_firewall() -> dict[str, bool]:
    """Prove that the stopped v3 attempt has not entered any official stage."""
    forbidden = {
        "official_protocol_created": PROTOCOL_PATH.exists()
        or PROTOCOL_HASH_PATH.exists(),
        "official_banks_created": (OUTPUT_ROOT / "banks").exists(),
        "official_selection_created": (OUTPUT_ROOT / "selection").exists(),
        "validation_created_or_accessed": (OUTPUT_ROOT / "fresh_validation").exists(),
    }
    entered = [name for name, value in forbidden.items() if value]
    if entered:
        raise RuntimeError(
            "development diagnostic firewall violated: " + ", ".join(entered)
        )
    return {name: False for name in forbidden}


def _symmetry_aware_distance(left: Any, right: Any, box: Any) -> float:
    """Minimum periodic sensor distance over label permutations.

    This reuses the repository's publication-visualization alignment routine,
    which exhaustively matches the four unordered sensors under periodic
    squared displacement.
    """
    from experiments.skyrmions_deep_ritz.visualize_authoritative import (
        _match_to_law,
    )

    centers = np.asarray(left, dtype=np.float64).reshape((-1, 2))
    reference = np.asarray(right, dtype=np.float64).reshape((-1, 2))
    _, delta = _match_to_law(
        centers, reference, np.asarray(box, dtype=np.float64)
    )
    return float(np.linalg.norm(delta))


def _maxmin_shortlist(
    candidates: list[dict[str, Any]], box: Any, maximum: int = 10
) -> list[dict[str, Any]]:
    """Deterministic robust-first, symmetry-aware max-min diversity list."""
    remaining = sorted(
        candidates, key=lambda row: (-row["robust_ress"], row["candidate_id"])
    )
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < int(maximum):
        scored = [
            (
                min(
                    _symmetry_aware_distance(row["eta"], old["eta"], box)
                    for old in selected
                ),
                row,
            )
            for row in remaining
        ]
        _, chosen = min(
            scored,
            key=lambda item: (
                -item[0],
                -item[1]["robust_ress"],
                item[1]["candidate_id"],
            ),
        )
        selected.append(chosen)
        remaining = [
            row for row in remaining if row["candidate_id"] != chosen["candidate_id"]
        ]
    return [
        {
            "candidate_id": row["candidate_id"],
            "eta": row["eta"],
            "eta_sha256": row["eta_sha256"],
            "scientific_selection_risk": row["scientific_selection_risk"],
            "screen_ress": row["screen_minimum_ress"],
            "audit_ress": row["audit_minimum_ress"],
            "robust_ress": row["robust_ress"],
        }
        for row in selected
    ]


def _diversity_summary(
    candidates: list[dict[str, Any]], box: Any
) -> dict[str, Any]:
    pairwise = [
        _symmetry_aware_distance(left["eta"], right["eta"], box)
        for left, right in combinations(candidates, 2)
    ]
    return {
        "eligible_candidate_count": len(candidates),
        "distance": (
            "minimum periodic Euclidean sensor displacement over all sensor-label "
            "permutations; repository _match_to_law alignment"
        ),
        "pair_count": len(pairwise),
        "minimum_distance": None if not pairwise else min(pairwise),
        "median_distance": None if not pairwise else float(np.median(pairwise)),
        "maximum_distance": None if not pairwise else max(pairwise),
        "maxmin_shortlist": _maxmin_shortlist(candidates, box, maximum=10),
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "config.json": ROOT / "config.json",
        "pareto_v3_common.py": ROOT / "pareto_v3_common.py",
        "pareto_v3_diagnostic.py": ROOT / "pareto_v3_diagnostic.py",
        "pareto_v3_run.py": ROOT / "pareto_v3_run.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _normalize_full_pool_row(
    screen: dict[str, Any],
    audit_diagnostics: dict[str, Any],
    *,
    reused: bool,
    projection_tolerance: float,
    forcing_tolerance: float,
    condition_limit: float,
) -> dict[str, Any]:
    screen_diagnostics = screen["screen"]
    screen_projection = bool(
        float(screen_diagnostics["maximum_projection_residual"])
        <= projection_tolerance
    )
    screen_forcing = bool(
        float(screen_diagnostics["maximum_forcing_mean"]) <= forcing_tolerance
    )
    screen_covariance = bool(
        float(screen_diagnostics["maximum_covariance_condition"])
        <= condition_limit
    )
    screen_ress = float(screen["minimum_ess_fraction"])
    audit_projection = bool(
        float(audit_diagnostics["maximum_projection_residual"])
        <= projection_tolerance
    )
    audit_forcing = bool(
        float(audit_diagnostics["maximum_forcing_mean"]) <= forcing_tolerance
    )
    audit_covariance = bool(
        float(audit_diagnostics["maximum_covariance_condition"])
        <= condition_limit
    )
    audit_ress = float(audit_diagnostics["minimum_ess_fraction"])
    return {
        "candidate_id": screen["candidate_id"],
        "eta": screen["eta"],
        "eta_sha256": eta_key(screen["eta"]),
        "scientific_selection_risk": float(screen["scientific_selection_risk"]),
        "geometry_valid": bool(screen["geometry_valid"]),
        "screen_projection_valid": screen_projection,
        "screen_minimum_ress": screen_ress,
        "screen_ress_valid": bool(screen_ress >= MINIMUM_RESS),
        "screen_forcing_valid": screen_forcing,
        "screen_covariance_valid": screen_covariance,
        "screen_support_valid": bool(
            screen_projection
            and screen_ress >= MINIMUM_RESS
            and screen_forcing
            and screen_covariance
        ),
        "screen_existing_v2_gate_valid": bool(screen["projection_valid"]),
        "screen_maximum_projection_residual": float(
            screen_diagnostics["maximum_projection_residual"]
        ),
        "screen_maximum_covariance_condition": float(
            screen_diagnostics["maximum_covariance_condition"]
        ),
        "screen_maximum_forcing_mean": float(
            screen_diagnostics["maximum_forcing_mean"]
        ),
        "screen_controlling_ress_time_index": None,
        "audit_projection_valid": audit_projection,
        "audit_minimum_ress": audit_ress,
        "audit_ress_valid": bool(audit_ress >= MINIMUM_RESS),
        "audit_forcing_valid": audit_forcing,
        "audit_covariance_valid": audit_covariance,
        "audit_support_valid": bool(
            audit_projection
            and audit_ress >= MINIMUM_RESS
            and audit_forcing
            and audit_covariance
        ),
        "audit_maximum_projection_residual": float(
            audit_diagnostics["maximum_projection_residual"]
        ),
        "audit_maximum_covariance_condition": float(
            audit_diagnostics["maximum_covariance_condition"]
        ),
        "audit_maximum_forcing_mean": float(
            audit_diagnostics["maximum_forcing_mean"]
        ),
        "audit_controlling_ress_time_index": None,
        "robust_ress": min(screen_ress, audit_ress),
        "audit_result_reused_from_phase1": reused,
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
    }


def _verify_cached_all_allowances(
    summary_path: Path, inventory_path: Path
) -> dict[str, Any] | None:
    if not summary_path.exists() and not inventory_path.exists():
        return None
    if not summary_path.exists() or not inventory_path.exists():
        raise RuntimeError("incomplete all-allowance diagnostic artifact pair")
    verify_v3_phase1_frozen()
    inventory = read_json(inventory_path)
    if file_sha256(summary_path) != inventory.get("summary_sha256"):
        raise RuntimeError("all-allowance diagnostic summary hash changed")
    current_sources = _source_hashes()
    if current_sources != inventory.get("diagnostic_source_sha256"):
        raise RuntimeError("all-allowance diagnostic source/config changed")
    _official_v3_firewall()
    return {**read_json(summary_path), "cache_hit": True}


def diagnose_v2_audit_all_allowances(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map unchanged dual-bank support over the complete frozen v2 pool."""
    summary_path = ALL_ALLOWANCES_DIAGNOSTIC_ROOT / "summary.json"
    inventory_path = ALL_ALLOWANCES_DIAGNOSTIC_ROOT / "inventory.json"
    cached = _verify_cached_all_allowances(summary_path, inventory_path)
    if cached is not None:
        return cached

    firewall_before = _official_v3_firewall()
    v2_inventory = verify_v2_frozen()
    phase1 = verify_v3_phase1_frozen()
    v2_protocol = require_v2_protocol(cfg)
    screening_path = V2_OUTPUT_ROOT / "screening" / "candidate_pool.json"
    screening = read_json(screening_path)
    source_rows = list(screening["rows"])
    if len(source_rows) != 337:
        raise RuntimeError(f"frozen v2 candidate count changed: {len(source_rows)}")
    if len({row["candidate_id"] for row in source_rows}) != len(source_rows):
        raise RuntimeError("frozen v2 candidate identifiers are not unique")
    if len({tuple(row["eta"]) for row in source_rows}) != len(source_rows):
        raise RuntimeError("frozen v2 candidate geometries are not unique")

    old_rows = phase1["summary"]["all_rows"]
    old_by_id = {row["candidate_id"]: row for row in old_rows}
    if len(old_by_id) != 193:
        raise RuntimeError("frozen Phase-1 reusable row count changed")
    source_by_id = {row["candidate_id"]: row for row in source_rows}
    for candidate_id, old in old_by_id.items():
        source = source_by_id.get(candidate_id)
        if source is None:
            raise RuntimeError(f"Phase-1 candidate disappeared: {candidate_id}")
        if source["eta"] != old["eta"]:
            raise RuntimeError(f"Phase-1 candidate eta changed: {candidate_id}")
        if float(source["scientific_selection_risk"]) != float(
            old["scientific_selection_risk"]
        ):
            raise RuntimeError(f"Phase-1 candidate risk changed: {candidate_id}")
        if float(source["minimum_ess_fraction"]) != float(old["screen_ress"]):
            raise RuntimeError(f"Phase-1 candidate screen rESS changed: {candidate_id}")

    data = selection_data(cfg, "search_train", "periodic_audit")
    audit_bank = load_bank("periodic_audit")
    missing = [
        row for row in source_rows if row["candidate_id"] not in old_by_id
    ]
    started = time.perf_counter()
    computed = _screen_rows_batched(cfg, data, audit_bank, missing, 8)
    elapsed = time.perf_counter() - started
    computed_by_id = {
        row["candidate_id"]: row["screen"] for row in computed
    }

    projection_tolerance = float(
        data.selection_problem.forcing_config.projection_tolerance
    )
    forcing_tolerance = float(
        data.selection_problem.forcing_config.forcing_mean_tolerance
    )
    condition_limit = float(
        data.selection_problem.forcing_config.max_covariance_condition
    )
    records = []
    for screen in source_rows:
        candidate_id = screen["candidate_id"]
        if candidate_id in old_by_id:
            audit_diagnostics = old_by_id[candidate_id]["audit"]
            reused = True
        else:
            audit_diagnostics = computed_by_id[candidate_id]
            reused = False
        records.append(
            _normalize_full_pool_row(
                screen,
                audit_diagnostics,
                reused=reused,
                projection_tolerance=projection_tolerance,
                forcing_tolerance=forcing_tolerance,
                condition_limit=condition_limit,
            )
        )

    law_risk = float(screening["law_risk"])
    box = data.selection_problem.family.box
    allowance_rows = []
    for allowance in ALLOWANCES:
        ceiling = selection_ceiling(law_risk, allowance)
        risk_rows = [
            row
            for row in records
            if row["scientific_selection_risk"] <= ceiling
        ]
        screen_feasible = [
            row
            for row in risk_rows
            if row["geometry_valid"]
            and row["screen_existing_v2_gate_valid"]
            and row["screen_minimum_ress"] >= MINIMUM_RESS
        ]
        audit_projection = [
            row for row in screen_feasible if row["audit_projection_valid"]
        ]
        audit_ress = [
            row for row in screen_feasible if row["audit_ress_valid"]
        ]
        eligible = [
            row
            for row in screen_feasible
            if row["screen_support_valid"] and row["audit_support_valid"]
        ]
        ranked = sorted(
            screen_feasible,
            key=lambda row: (-row["robust_ress"], row["candidate_id"]),
        )
        best_audit = min(
            screen_feasible,
            key=lambda row: (-row["audit_minimum_ress"], row["candidate_id"]),
        )
        best_robust = ranked[0]
        allowance_rows.append(
            {
                "allowance_percent": allowance,
                "risk_ceiling": ceiling,
                "total_inside_risk_ceiling": len(risk_rows),
                "screen_feasible_count": len(screen_feasible),
                "audit_projection_valid_count": len(audit_projection),
                "audit_ress_valid_count": len(audit_ress),
                "dual_bank_eligible_count": len(eligible),
                "status": (
                    "DUAL_BANK_ELIGIBLE_CANDIDATES_PRESENT"
                    if eligible
                    else "NO_DUAL_BANK_ELIGIBLE_CANDIDATE"
                ),
                "best_audit_minimum_ress": best_audit["audit_minimum_ress"],
                "best_audit_candidate_id": best_audit["candidate_id"],
                "best_robust_ress": best_robust["robust_ress"],
                "best_robust_candidate_id": best_robust["candidate_id"],
                "audit_ress_distribution_over_screen_feasible": _quantiles(
                    [row["audit_minimum_ress"] for row in screen_feasible]
                ),
                "robust_ress_distribution_over_screen_feasible": _quantiles(
                    [row["robust_ress"] for row in screen_feasible]
                ),
                "top_candidates_by_robust_ress": [
                    {
                        "candidate_id": row["candidate_id"],
                        "eta": row["eta"],
                        "scientific_selection_risk": row[
                            "scientific_selection_risk"
                        ],
                        "screen_ress": row["screen_minimum_ress"],
                        "audit_ress": row["audit_minimum_ress"],
                        "robust_ress": row["robust_ress"],
                        "dual_bank_eligible": row in eligible,
                    }
                    for row in ranked[:20]
                ],
                "eligible_candidate_ids": [
                    row["candidate_id"]
                    for row in sorted(eligible, key=lambda row: row["candidate_id"])
                ],
                "distinct_basin_information": _diversity_summary(eligible, box),
                "controlling_audit_time_node_frequency": None,
            }
        )

    first_viable = next(
        (
            row["allowance_percent"]
            for row in allowance_rows
            if row["dual_bank_eligible_count"] > 0
        ),
        None,
    )
    global_ranking = sorted(
        records, key=lambda row: (-row["robust_ress"], row["candidate_id"])
    )
    firewall_after = _official_v3_firewall()
    result = {
        "schema_version": 3,
        "purpose": (
            "development-only map of unchanged screen/audit support over all "
            "allowances in the frozen v2 candidate pool"
        ),
        "development_diagnostic_only": True,
        "official_v3_continuation": False,
        "official_protocol_created": False,
        "source_v2_version": v2_protocol["version"],
        "source_v2_protocol_sha256": v2_protocol["protocol_sha256"],
        "source_v2_output_tree_sha256": v2_inventory["output_tree_sha256"],
        "source_v2_screening_sha256": file_sha256(screening_path),
        "source_phase1_summary_sha256": phase1["verified_hashes"]["summary_json"],
        "source_phase1_inventory_sha256": phase1["verified_hashes"][
            "inventory_json"
        ],
        "frozen_constants": {
            "allowances_percent": list(ALLOWANCES),
            "law_selection_risk": law_risk,
            "minimum_relative_ess": MINIMUM_RESS,
            "risk_rule": "risk <= (1 + p/100) * R_Law_sel; no slack",
            "projection_tolerance": projection_tolerance,
            "forcing_mean_tolerance": forcing_tolerance,
            "maximum_covariance_condition": condition_limit,
            "dtype": "float64",
            "screen_bank_samples": 8192,
            "periodic_audit_bank_samples": 16384,
        },
        "candidate_pool_count": len(records),
        "candidate_pool_unique_id_count": len(
            {row["candidate_id"] for row in records}
        ),
        "candidate_pool_unique_eta_count": len(
            {tuple(row["eta"]) for row in records}
        ),
        "audit_reuse": {
            "reused_exact_phase1_count": len(old_by_id),
            "newly_evaluated_count": len(missing),
            "bank_semantics_config_dtype_and_code_verified_by_v2_protocol": True,
            "candidate_identity_and_eta_verified_exactly": True,
        },
        "allowances": allowance_rows,
        "first_allowance_with_dual_bank_eligible_candidate": first_viable,
        "global_ranking_by_robust_ress": [
            row["candidate_id"] for row in global_ranking
        ],
        "per_candidate_records": records,
        "controlling_ress_time_node": {
            "available": False,
            "reason": (
                "the immutable v2 screen and Phase-1 audit artifacts retain only "
                "the timewise minimum, not its index; exact reuse was preferred "
                "over recomputing the 193 frozen rows solely for this optional field"
            ),
        },
        "elapsed_seconds_for_144_new_audit_evaluations": elapsed,
        "firewall_before": firewall_before,
        "firewall_after": firewall_after,
        "validation_accessed": False,
        "tangent_optimization_run": False,
        "full_kf_constructed": False,
        "selection_or_validation_data_created": False,
    }
    atomic_json(summary_path, result, immutable=True)
    inventory = {
        "schema_version": 3,
        "development_diagnostic_only": True,
        "summary_sha256": file_sha256(summary_path),
        "summary_bytes": summary_path.stat().st_size,
        "diagnostic_source_sha256": _source_hashes(),
        "source_v2_output_tree_sha256": v2_inventory["output_tree_sha256"],
        "source_v2_protocol_sha256": v2_protocol["protocol_sha256"],
        "source_phase1_verified_hashes": phase1["verified_hashes"],
        "validation_accessed": False,
        "official_protocol_created": False,
    }
    atomic_json(inventory_path, inventory, immutable=True)
    verify_v3_phase1_frozen()
    _official_v3_firewall()
    return result
