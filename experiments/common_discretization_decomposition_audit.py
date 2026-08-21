"""Reporting for common-raster orthogonal-decomposition audits."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from action_decomposition_audit import geometry_key


CONDITIONS = (
    "full_moment",
    "tangent_moment",
    "hidden_nullspace",
    "orthogonality",
    "pythagorean",
    "hierarchy",
)


def _level_metrics(
    arrays: dict[str, np.ndarray],
    *,
    tolerance: float,
) -> dict[str, float | int | bool]:
    full_norm = float(np.linalg.norm(arrays["full_moment_residual"]))
    tangent_norm = float(np.linalg.norm(arrays["tangent_moment_residual"]))
    hidden_norm = float(np.linalg.norm(arrays["hidden_moment_residual"]))
    orthogonality = float(arrays["tangent_hidden_inner_product"])
    pythagorean = float(arrays["pythagorean_residual"])
    hierarchy = float(arrays["hierarchy_raw_violation"])
    failures = {
        "full_moment": full_norm > tolerance,
        "tangent_moment": tangent_norm > tolerance,
        "hidden_nullspace": hidden_norm > tolerance,
        "orthogonality": abs(orthogonality) > tolerance,
        "pythagorean": abs(pythagorean) > tolerance,
        "hierarchy": hierarchy > tolerance,
    }
    return {
        "full_moment_residual_norm": full_norm,
        "tangent_moment_residual_norm": tangent_norm,
        "hidden_nullspace_residual_norm": hidden_norm,
        "orthogonality_residual": orthogonality,
        "absolute_orthogonality_residual": abs(orthogonality),
        "pythagorean_residual": pythagorean,
        "absolute_pythagorean_residual": abs(pythagorean),
        "hierarchy_raw_violation": hierarchy,
        **{f"{name}_violates": bool(value) for name, value in failures.items()},
        "any_violation": any(failures.values()),
    }


def audit_common_discretization(
    candidates: list[dict[str, Any]],
    *,
    evaluate: Callable[[Any, str], list[dict[str, Any]]],
    time_grid: np.ndarray,
    time_weights: np.ndarray,
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Audit all candidates without using the particle Tangent action."""
    time_grid = np.asarray(time_grid, dtype=np.float64)
    time_weights = np.asarray(time_weights, dtype=np.float64)
    tolerance = float(tolerance)
    if time_grid.shape != time_weights.shape:
        raise ValueError("time grid and weights must have identical shapes")
    if not np.isclose(np.sum(time_weights), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("time weights must sum to one")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    evaluations: dict[str, list[dict[str, Any]]] = {}
    output_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}

    scalar_names = (
        "full_energy",
        "tangent_energy",
        "hidden_energy",
        "tangent_hidden_inner_product",
        "pythagorean_residual",
        "hierarchy_raw_violation",
    )
    vector_names = (
        "moment_rate_residual",
        "full_moment_residual",
        "tangent_moment_residual",
        "hidden_moment_residual",
    )
    diagnostic_vector_names = (
        "solver_stabilization_moment_shift",
        "full_moment_residual_after_stabilization",
    )

    for candidate_index, candidate in enumerate(candidates):
        key = geometry_key(candidate["geometry"])
        if key not in evaluations:
            evaluations[key] = evaluate(candidate["geometry"], key)
        evaluator_rows = evaluations[key]
        trial_details: list[dict[str, Any]] = []
        trial_array_records: list[dict[str, np.ndarray]] = []
        time_details: list[dict[str, Any]] = []
        invalid_trials = 0

        for evaluator_row in evaluator_rows:
            payload = evaluator_row.get(
                "common_discretization_decomposition_by_time"
            )
            if not evaluator_row.get("valid") or payload is None:
                invalid_trials += 1
                continue
            arrays = {
                name: np.asarray(payload[name], dtype=np.float64)
                for name in scalar_names + vector_names + diagnostic_vector_names
            }
            ranks = np.asarray(payload["gram_rank"], dtype=np.int64)
            coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
            moment_count = coefficients.shape[-1]
            if any(arrays[name].shape != time_grid.shape for name in scalar_names):
                raise RuntimeError(f"incomplete raster scalar data for {key}")
            if any(
                arrays[name].shape != time_grid.shape + (moment_count,)
                for name in vector_names + diagnostic_vector_names
            ):
                raise RuntimeError(f"incomplete raster moment data for {key}")
            if ranks.shape != time_grid.shape:
                raise RuntimeError(f"incomplete raster rank data for {key}")
            if any(not np.all(np.isfinite(values)) for values in arrays.values()):
                raise RuntimeError(f"non-finite raster decomposition data for {key}")

            reported_full = np.asarray(
                evaluator_row["full_action_by_time"], dtype=np.float64
            )
            if reported_full.shape != time_grid.shape:
                raise RuntimeError(f"incomplete authoritative Full action for {key}")

            for time_index, time in enumerate(time_grid):
                level_arrays = {
                    name: arrays[name][time_index]
                    for name in scalar_names + vector_names
                }
                checks = _level_metrics(level_arrays, tolerance=tolerance)
                time_details.append(
                    {
                        "trial": int(evaluator_row["trial"]),
                        "time_index": int(time_index),
                        "time": float(time),
                        "A_full_h": float(arrays["full_energy"][time_index]),
                        "A_tan_h": float(arrays["tangent_energy"][time_index]),
                        "A_hid_h": float(arrays["hidden_energy"][time_index]),
                        "hidden_fraction": float(
                            arrays["hidden_energy"][time_index]
                            / arrays["full_energy"][time_index]
                        ) if arrays["full_energy"][time_index] != 0.0 else float("nan"),
                        "authoritative_full_action": float(reported_full[time_index]),
                        "full_action_reproduction_error": float(
                            arrays["full_energy"][time_index]
                            - reported_full[time_index]
                        ),
                        "r_h": arrays["moment_rate_residual"][time_index].tolist(),
                        "full_moment_residual": arrays[
                            "full_moment_residual"
                        ][time_index].tolist(),
                        "tangent_moment_residual": arrays[
                            "tangent_moment_residual"
                        ][time_index].tolist(),
                        "hidden_moment_residual": arrays[
                            "hidden_moment_residual"
                        ][time_index].tolist(),
                        "solver_stabilization_moment_shift": arrays[
                            "solver_stabilization_moment_shift"
                        ][time_index].tolist(),
                        "full_moment_residual_after_stabilization": arrays[
                            "full_moment_residual_after_stabilization"
                        ][time_index].tolist(),
                        "full_moment_residual_after_stabilization_norm": float(
                            np.linalg.norm(
                                arrays[
                                    "full_moment_residual_after_stabilization"
                                ][time_index]
                            )
                        ),
                        "gram_rank": int(ranks[time_index]),
                        **checks,
                    }
                )

            trial_arrays = {
                name: np.tensordot(time_weights, arrays[name], axes=(0, 0))
                for name in scalar_names + vector_names
            }
            trial_checks = _level_metrics(trial_arrays, tolerance=tolerance)
            trial_array_records.append(trial_arrays)
            trial_details.append(
                {
                    "trial": int(evaluator_row["trial"]),
                    "A_full_h": float(trial_arrays["full_energy"]),
                    "A_tan_h": float(trial_arrays["tangent_energy"]),
                    "A_hid_h": float(trial_arrays["hidden_energy"]),
                    "hidden_fraction": float(
                        trial_arrays["hidden_energy"] / trial_arrays["full_energy"]
                    ) if trial_arrays["full_energy"] != 0.0 else float("nan"),
                    **trial_checks,
                }
            )

        if not trial_details:
            raise RuntimeError(f"no valid common-raster evaluation for {key}")

        aggregate_arrays: dict[str, np.ndarray] = {}
        for name in scalar_names:
            aggregate_arrays[name] = np.asarray(
                np.mean([record[name] for record in trial_array_records], axis=0)
            )
        # Aggregate vector residuals before taking norms, matching the scalar
        # time/trial quadrature and retaining signed cancellation information.
        for name in vector_names:
            aggregate_arrays[name] = np.mean(
                [record[name] for record in trial_array_records],
                axis=0,
            )
        aggregate_checks = _level_metrics(
            aggregate_arrays, tolerance=tolerance
        )

        all_level_rows = time_details + trial_details + [aggregate_checks]
        maximum = {
            "full_moment": max(row["full_moment_residual_norm"] for row in all_level_rows),
            "tangent_moment": max(row["tangent_moment_residual_norm"] for row in all_level_rows),
            "hidden_nullspace": max(row["hidden_nullspace_residual_norm"] for row in all_level_rows),
            "orthogonality": max(row["absolute_orthogonality_residual"] for row in all_level_rows),
            "pythagorean": max(row["absolute_pythagorean_residual"] for row in all_level_rows),
            "hierarchy": max(row["hierarchy_raw_violation"] for row in all_level_rows),
        }
        full_reproduction_max = max(
            abs(row["full_action_reproduction_error"]) for row in time_details
        )
        stabilized_full_residual_max = max(
            row["full_moment_residual_after_stabilization_norm"]
            for row in time_details
        )
        aggregate_violation_count = int(aggregate_checks["any_violation"])
        trial_violation_count = sum(int(row["any_violation"]) for row in trial_details)
        time_violation_count = sum(int(row["any_violation"]) for row in time_details)
        candidate_passes = bool(
            invalid_trials == 0
            and aggregate_violation_count == 0
            and trial_violation_count == 0
            and time_violation_count == 0
        )
        first_failure = next(
            (condition for condition in CONDITIONS if maximum[condition] > tolerance),
            None,
        )
        output_rows.append(
            {
                "allowance_percent": float(candidate["allowance_percent"]),
                "method": candidate["method"],
                "A_full_h": float(aggregate_arrays["full_energy"]),
                "A_tan_h": float(aggregate_arrays["tangent_energy"]),
                "A_hid_h": float(aggregate_arrays["hidden_energy"]),
                "hidden_fraction_A_hid_over_A_full": float(
                    aggregate_arrays["hidden_energy"]
                    / aggregate_arrays["full_energy"]
                ) if aggregate_arrays["full_energy"] != 0.0 else float("nan"),
                "aggregate_full_moment_residual_norm": aggregate_checks[
                    "full_moment_residual_norm"
                ],
                "aggregate_tangent_moment_residual_norm": aggregate_checks[
                    "tangent_moment_residual_norm"
                ],
                "aggregate_hidden_nullspace_residual_norm": aggregate_checks[
                    "hidden_nullspace_residual_norm"
                ],
                "aggregate_orthogonality_residual": aggregate_checks[
                    "orthogonality_residual"
                ],
                "aggregate_pythagorean_residual": aggregate_checks[
                    "pythagorean_residual"
                ],
                "aggregate_hierarchy_raw_violation": aggregate_checks[
                    "hierarchy_raw_violation"
                ],
                "maximum_full_moment_residual": maximum["full_moment"],
                "maximum_tangent_moment_residual": maximum["tangent_moment"],
                "maximum_hidden_nullspace_residual": maximum["hidden_nullspace"],
                "maximum_absolute_orthogonality_residual": maximum["orthogonality"],
                "maximum_absolute_pythagorean_residual": maximum["pythagorean"],
                "maximum_raw_hierarchy_violation": maximum["hierarchy"],
                "maximum_full_action_reproduction_error": full_reproduction_max,
                "maximum_full_moment_residual_after_stabilization": (
                    stabilized_full_residual_max
                ),
                "aggregate_violation_count": aggregate_violation_count,
                "trial_violation_count": trial_violation_count,
                "time_trial_violation_count": time_violation_count,
                "trial_count": len(evaluator_rows),
                "invalid_trial_count": invalid_trials,
                "time_trial_count": len(time_details),
                "minimum_gram_rank": min(row["gram_rank"] for row in time_details),
                "first_failing_condition": first_failure,
                "tolerance": tolerance,
                "passes": candidate_passes,
                "geometry": json.dumps(candidate["geometry"], separators=(",", ":")),
                "result_path": candidate["result_path"],
                "evaluation_key": key,
            }
        )
        detail[f"candidate_{candidate_index}"] = {
            "allowance_percent": float(candidate["allowance_percent"]),
            "method": candidate["method"],
            "geometry": candidate["geometry"],
            "evaluation_key": key,
            "aggregate": aggregate_checks,
            "trials": trial_details,
            "time_trials": time_details,
        }

    summary_maximum = {
        condition: max(
            row[
                {
                    "full_moment": "maximum_full_moment_residual",
                    "tangent_moment": "maximum_tangent_moment_residual",
                    "hidden_nullspace": "maximum_hidden_nullspace_residual",
                    "orthogonality": "maximum_absolute_orthogonality_residual",
                    "pythagorean": "maximum_absolute_pythagorean_residual",
                    "hierarchy": "maximum_raw_hierarchy_violation",
                }[condition]
            ]
            for row in output_rows
        )
        for condition in CONDITIONS
    }
    first_failure = next(
        (condition for condition in CONDITIONS if summary_maximum[condition] > tolerance),
        None,
    )
    summary = {
        "schema_version": 1,
        "candidate_count": len(output_rows),
        "allowance_count": len({row["allowance_percent"] for row in output_rows}),
        "designs": ["law", "tangent", "full"],
        "absolute_tolerance": tolerance,
        "tolerance_source": "saved validity.tangent_lower_bound_tol",
        "maximum_full_moment_rate_residual": summary_maximum["full_moment"],
        "maximum_tangent_moment_rate_residual": summary_maximum["tangent_moment"],
        "maximum_hidden_nullspace_residual": summary_maximum["hidden_nullspace"],
        "maximum_absolute_orthogonality_residual": summary_maximum["orthogonality"],
        "maximum_absolute_pythagorean_residual": summary_maximum["pythagorean"],
        "maximum_raw_hierarchy_violation": summary_maximum["hierarchy"],
        "maximum_absolute_full_action_reproduction_error": max(
            row["maximum_full_action_reproduction_error"] for row in output_rows
        ),
        "maximum_full_moment_rate_residual_after_stabilization": max(
            row["maximum_full_moment_residual_after_stabilization"]
            for row in output_rows
        ),
        "aggregate_violation_count": sum(
            row["aggregate_violation_count"] for row in output_rows
        ),
        "trial_violation_count": sum(row["trial_violation_count"] for row in output_rows),
        "time_trial_violation_count": sum(
            row["time_trial_violation_count"] for row in output_rows
        ),
        "invalid_trial_count": sum(row["invalid_trial_count"] for row in output_rows),
        "passing_candidate_count": sum(bool(row["passes"]) for row in output_rows),
        "every_final_candidate_passes": all(bool(row["passes"]) for row in output_rows),
        "first_failing_condition": first_failure,
        "hidden_fraction_supported": all(bool(row["passes"]) for row in output_rows),
        "hidden_fraction_definition": "direct A_hid,h / A_full,h in the common raster space",
        "particle_tangent_metric_used": False,
        "violation_counts_by_condition": {
            "aggregate": {
                condition: sum(
                    int(candidate["aggregate"][f"{condition}_violates"])
                    for candidate in detail.values()
                )
                for condition in CONDITIONS
            },
            "trial": {
                condition: sum(
                    int(row[f"{condition}_violates"])
                    for candidate in detail.values()
                    for row in candidate["trials"]
                )
                for condition in CONDITIONS
            },
            "time_trial": {
                condition: sum(
                    int(row[f"{condition}_violates"])
                    for candidate in detail.values()
                    for row in candidate["time_trials"]
                )
                for condition in CONDITIONS
            },
        },
    }
    return output_rows, detail, summary


def save_common_discretization_outputs(
    rows: list[dict[str, Any]],
    detail: dict[str, Any],
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "common_discretization_decomposition_audit"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {"schema_version": 1, "summary": summary, "candidates": rows, "detail": detail},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if summary["every_final_candidate_passes"] else "FAIL"
    lines = [
        "# Common-discretization decomposition audit",
        "",
        f"**Overall: {status}.** Absolute tolerance: `{summary['absolute_tolerance']:.3e}`.",
        "",
        "The Tangent correction is the minimum-norm solution in the same physical-"
        "density raster Hilbert space used by the authoritative Full action. The "
        "particle Tangent metric is not used.",
        "",
        "## Summary",
        "",
        f"- Maximum Full moment-rate residual: `{summary['maximum_full_moment_rate_residual']:.12g}`",
        f"- Maximum Tangent moment-rate residual: `{summary['maximum_tangent_moment_rate_residual']:.12g}`",
        f"- Maximum hidden-nullspace residual: `{summary['maximum_hidden_nullspace_residual']:.12g}`",
        f"- Maximum absolute orthogonality residual: `{summary['maximum_absolute_orthogonality_residual']:.12g}`",
        f"- Maximum absolute Pythagorean residual: `{summary['maximum_absolute_pythagorean_residual']:.12g}`",
        f"- Maximum raw hierarchy violation (`A_tan,h - A_full,h`): `{summary['maximum_raw_hierarchy_violation']:.12g}`",
        f"- Maximum Full residual after subtracting floor/gauge stabilization: `{summary['maximum_full_moment_rate_residual_after_stabilization']:.12g}`",
        f"- Violations (aggregate / trial / time-trial): `{summary['aggregate_violation_count']} / {summary['trial_violation_count']} / {summary['time_trial_violation_count']}`",
        f"- First failing condition: `{summary['first_failing_condition']}`",
        f"- Genuine hidden-fraction interpretation supported: `{summary['hidden_fraction_supported']}`",
        f"- Saved geometries unchanged: `{summary.get('saved_candidate_geometries_unchanged')}`",
        f"- Frozen banks unchanged: `{summary.get('frozen_banks_unchanged')}`",
        "",
    ]
    if summary["first_failing_condition"] == "full_moment":
        lines.extend(
            [
                "The first failed implication is Full moment-rate feasibility. "
                "The authoritative linear solve uses `q_h + q_floor` for stability "
                f"(`operator_floor_rel={summary.get('poisson_operator_floor_rel')}`), "
                "while its scientific action—and this requested Hilbert-space audit—"
                "uses physical `q_h`. Full action reproduction is at roundoff and the "
                "raster Tangent constraint solve is at machine precision, so the "
                "downstream nullspace, orthogonality, and Pythagorean failures originate "
                "before the projection, at physical-`q_h` Full feasibility.",
                "",
            ]
        )
    lines.extend(
        [
        "## Candidate table",
        "",
        "| Allowance | Design | A_tan,h | A_full,h | A_hid,h | A_hid,h/A_full,h | max Full feas. | max Tan feas. | max null | max orth. | max Pyth. | max raw hierarchy | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['allowance_percent']:g}% | {row['method'].title()} | "
            f"{row['A_tan_h']:.8g} | {row['A_full_h']:.8g} | "
            f"{row['A_hid_h']:.8g} | {row['hidden_fraction_A_hid_over_A_full']:.8g} | "
            f"{row['maximum_full_moment_residual']:.3e} | "
            f"{row['maximum_tangent_moment_residual']:.3e} | "
            f"{row['maximum_hidden_nullspace_residual']:.3e} | "
            f"{row['maximum_absolute_orthogonality_residual']:.3e} | "
            f"{row['maximum_absolute_pythagorean_residual']:.3e} | "
            f"{row['maximum_raw_hierarchy_violation']:.3e} | "
            f"{'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "`r_h = -sum(phi * q_h * h_h) dx^2`; "
            "`L_h(-grad z)_j = -<grad phi_j, grad z>_{q_h}`. "
            "The Full action, Gram matrix, projection, hidden action, and all inner "
            "products use the physical `q_h` edge weights and cell quadrature from "
            "the authoritative Full evaluator. Residuals and hierarchy gaps are raw "
            "and are never clipped.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, md_path
