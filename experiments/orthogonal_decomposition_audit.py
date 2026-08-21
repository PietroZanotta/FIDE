"""Reporting for direct correction-field orthogonal-decomposition audits."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from action_decomposition_audit import geometry_key


def audit_field_decompositions(
    candidates: list[dict[str, Any]],
    *,
    evaluate: Callable[[Any, str], list[dict[str, Any]]],
    time_grid: np.ndarray,
    time_weights: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    time_grid = np.asarray(time_grid, dtype=np.float64)
    time_weights = np.asarray(time_weights, dtype=np.float64)
    if time_grid.shape != time_weights.shape:
        raise ValueError("time grid and quadrature weights must have identical shapes")
    evaluations: dict[str, list[dict[str, Any]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    detail: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates:
        key = geometry_key(candidate["geometry"])
        if key not in evaluations:
            evaluations[key] = evaluate(candidate["geometry"], key)
        rows = evaluations[key]
        details: list[dict[str, Any]] = []
        trial_metrics: list[dict[str, float]] = []
        invalid = 0
        for row in rows:
            if not row.get("valid") or "decomposition_by_time" not in row:
                invalid += 1
                continue
            decomposition = row["decomposition_by_time"]
            arrays = {
                "A_tan": np.asarray(row["tangent_action_by_time"], dtype=np.float64),
                "A_full": np.asarray(row["full_action_by_time"], dtype=np.float64),
                "direct_A_full": np.asarray(
                    decomposition["direct_full_field_energy"], dtype=np.float64
                ),
                "direct_A_tan": np.asarray(
                    decomposition["direct_tangent_field_energy"], dtype=np.float64
                ),
                "direct_A_hid": np.asarray(
                    decomposition["direct_hidden_field_energy"], dtype=np.float64
                ),
                "orthogonality": np.asarray(
                    decomposition["direct_tangent_hidden_inner_product"],
                    dtype=np.float64,
                ),
                "decomposition_residual": np.asarray(
                    decomposition["reported_identity_residual"], dtype=np.float64
                ),
                "polarization_residual": np.asarray(
                    decomposition["discrete_polarization_residual"], dtype=np.float64
                ),
            }
            if any(values.shape != time_grid.shape for values in arrays.values()):
                raise RuntimeError(f"incomplete per-time decomposition for {key}")
            if any(not np.all(np.isfinite(values)) for values in arrays.values()):
                raise RuntimeError(f"non-finite per-time decomposition for {key}")
            for time_index, time in enumerate(time_grid):
                scale = max(abs(arrays["A_full"][time_index]), 1.0e-14)
                details.append(
                    {
                        "trial": int(row["trial"]),
                        "time_index": int(time_index),
                        "time": float(time),
                        **{
                            name: float(values[time_index])
                            for name, values in arrays.items()
                        },
                        "absolute_decomposition_residual": float(
                            abs(arrays["decomposition_residual"][time_index])
                        ),
                        "relative_decomposition_residual": float(
                            abs(arrays["decomposition_residual"][time_index]) / scale
                        ),
                        "absolute_orthogonality_residual": float(
                            abs(arrays["orthogonality"][time_index])
                        ),
                        "relative_orthogonality_residual": float(
                            abs(arrays["orthogonality"][time_index]) / scale
                        ),
                    }
                )
            trial_metrics.append(
                {
                    name: float(np.sum(time_weights * values))
                    for name, values in arrays.items()
                }
            )

        if not trial_metrics:
            raise RuntimeError(f"no valid field decomposition for {key}")
        aggregate = {
            name: float(np.mean([row[name] for row in trial_metrics]))
            for name in trial_metrics[0]
        }
        aggregate_residual = float(
            aggregate["A_full"] - aggregate["A_tan"] - aggregate["direct_A_hid"]
        )
        aggregate_cross = float(aggregate["orthogonality"])
        time_max_decomposition = max(
            row["absolute_decomposition_residual"] for row in details
        )
        time_max_orthogonality = max(
            row["absolute_orthogonality_residual"] for row in details
        )
        trial_max_decomposition = max(
            abs(row["decomposition_residual"]) for row in trial_metrics
        )
        trial_max_orthogonality = max(abs(row["orthogonality"]) for row in trial_metrics)
        maximum_decomposition = max(
            abs(aggregate_residual), time_max_decomposition, trial_max_decomposition
        )
        maximum_orthogonality = max(
            abs(aggregate_cross), time_max_orthogonality, trial_max_orthogonality
        )
        passes = bool(
            invalid == 0
            and maximum_decomposition <= tolerance
            and maximum_orthogonality <= tolerance
        )
        aggregate_rows.append(
            {
                "allowance_percent": candidate["allowance_percent"],
                "method": candidate["method"],
                "A_tan": aggregate["A_tan"],
                "A_full": aggregate["A_full"],
                "direct_A_hid": aggregate["direct_A_hid"],
                "direct_A_tan_field": aggregate["direct_A_tan"],
                "direct_A_full_field": aggregate["direct_A_full"],
                "aggregate_decomposition_residual": aggregate_residual,
                "aggregate_orthogonality_residual": aggregate_cross,
                "maximum_absolute_decomposition_residual": maximum_decomposition,
                "maximum_relative_decomposition_residual": max(
                    row["relative_decomposition_residual"] for row in details
                ),
                "maximum_absolute_orthogonality_residual": maximum_orthogonality,
                "maximum_relative_orthogonality_residual": max(
                    row["relative_orthogonality_residual"] for row in details
                ),
                "maximum_absolute_discrete_polarization_residual": max(
                    abs(row["polarization_residual"]) for row in details
                ),
                "trial_count": len(rows),
                "invalid_trial_count": invalid,
                "time_trial_count": len(details),
                "tolerance": float(tolerance),
                "passes": passes,
                "geometry": json.dumps(candidate["geometry"], separators=(",", ":")),
                "result_path": candidate["result_path"],
                "evaluation_key": key,
            }
        )
        detail.setdefault(key, details)

    finite_decomposition = [
        row["maximum_absolute_decomposition_residual"] for row in aggregate_rows
    ]
    finite_orthogonality = [
        row["maximum_absolute_orthogonality_residual"] for row in aggregate_rows
    ]
    summary = {
        "schema_version": 1,
        "candidate_count": len(aggregate_rows),
        "tolerance": float(tolerance),
        "tolerance_source": "validity.tangent_lower_bound_tol",
        "maximum_absolute_decomposition_residual": max(finite_decomposition),
        "maximum_absolute_orthogonality_residual": max(finite_orthogonality),
        "maximum_absolute_discrete_polarization_residual": max(
            row["maximum_absolute_discrete_polarization_residual"]
            for row in aggregate_rows
        ),
        "invalid_trial_count": sum(row["invalid_trial_count"] for row in aggregate_rows),
        "passing_candidate_count": sum(bool(row["passes"]) for row in aggregate_rows),
        "every_final_candidate_passes": all(bool(row["passes"]) for row in aggregate_rows),
        "hidden_action_definition": (
            "direct weighted Dirichlet energy of delta_hid = delta_* - delta_tan; "
            "not A_full - A_tan"
        ),
        "orthogonality_definition": (
            "direct weighted Dirichlet inner product E[delta_tan . delta_hid]"
        ),
    }
    return aggregate_rows, detail, summary


def save_field_audit(
    rows: list[dict[str, Any]],
    detail: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "orthogonal_decomposition_audit.csv"
    json_path = output_dir / "orthogonal_decomposition_audit.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": summary,
                "candidates": rows,
                "per_time_trial": detail,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
