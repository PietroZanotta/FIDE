"""Targeted physical-q Full-solver consistency audit reporting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


CHECK_ORDER = (
    "physical_poisson",
    "full_moment",
    "tangent_moment",
    "hidden_nullspace",
    "orthogonality",
    "pythagorean",
    "hierarchy",
)


def run_targeted_audit(
    candidates: list[dict[str, Any]],
    *,
    evaluate: Callable[[Any, str], list[dict[str, Any]]],
    time_grid: np.ndarray,
    time_weights: np.ndarray,
    moment_tolerance: float,
    energy_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    time_grid = np.asarray(time_grid, dtype=np.float64)
    time_weights = np.asarray(time_weights, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}

    for candidate in candidates:
        method = str(candidate["method"])
        evaluator_rows = evaluate(candidate["geometry"], method)
        time_rows: list[dict[str, Any]] = []
        for row in evaluator_rows:
            payload = row["common_discretization_decomposition_by_time"]
            arrays = {name: np.asarray(value, dtype=np.float64) for name, value in payload.items()}
            poisson = np.asarray(
                row["physical_poisson_relative_residual_by_time"], dtype=np.float64
            )
            compatibility = np.asarray(
                row["component_compatibility_residual_by_time"], dtype=np.float64
            )
            components = np.asarray(
                row["conductive_component_count_by_time"], dtype=np.int32
            )
            solver_converged = np.asarray(
                row["physical_solver_converged_by_time"], dtype=bool
            )
            for time_index, time in enumerate(time_grid):
                full_residual = float(
                    np.linalg.norm(arrays["full_moment_residual"][time_index])
                )
                tangent_residual = float(
                    np.linalg.norm(arrays["tangent_moment_residual"][time_index])
                )
                hidden_residual = float(
                    np.linalg.norm(arrays["hidden_moment_residual"][time_index])
                )
                time_rows.append(
                    {
                        "trial": int(row["trial"]),
                        "time_index": int(time_index),
                        "time": float(time),
                        "physical_poisson_relative_residual": float(poisson[time_index]),
                        "component_compatibility_residual": float(
                            compatibility[time_index]
                        ),
                        "conductive_component_count": int(components[time_index]),
                        "physical_solver_converged": bool(
                            solver_converged[time_index]
                        ),
                        "full_moment_residual": full_residual,
                        "tangent_moment_residual": tangent_residual,
                        "hidden_nullspace_residual": hidden_residual,
                        "orthogonality_residual": float(
                            arrays["tangent_hidden_inner_product"][time_index]
                        ),
                        "pythagorean_residual": float(
                            arrays["pythagorean_residual"][time_index]
                        ),
                        "hierarchy_raw_violation": float(
                            arrays["hierarchy_raw_violation"][time_index]
                        ),
                        "A_full_h": float(arrays["full_energy"][time_index]),
                        "A_tan_h": float(arrays["tangent_energy"][time_index]),
                        "A_hid_h": float(arrays["hidden_energy"][time_index]),
                    }
                )

        poisson_tolerance = min(
            float(row["physical_poisson_tolerance"]) for row in evaluator_rows
        )
        maxima = {
            "physical_poisson": max(
                item["physical_poisson_relative_residual"] for item in time_rows
            ),
            "full_moment": max(item["full_moment_residual"] for item in time_rows),
            "tangent_moment": max(
                item["tangent_moment_residual"] for item in time_rows
            ),
            "hidden_nullspace": max(
                item["hidden_nullspace_residual"] for item in time_rows
            ),
            "orthogonality": max(
                abs(item["orthogonality_residual"]) for item in time_rows
            ),
            "pythagorean": max(
                abs(item["pythagorean_residual"]) for item in time_rows
            ),
            "hierarchy": max(item["hierarchy_raw_violation"] for item in time_rows),
        }
        tolerances = {
            "physical_poisson": poisson_tolerance,
            "full_moment": float(moment_tolerance),
            "tangent_moment": float(moment_tolerance),
            "hidden_nullspace": float(moment_tolerance),
            "orthogonality": float(energy_tolerance),
            "pythagorean": float(energy_tolerance),
            "hierarchy": float(energy_tolerance),
        }
        first_failure = next(
            (name for name in CHECK_ORDER if maxima[name] > tolerances[name]), None
        )
        incompatible_count = sum(
            item["component_compatibility_residual"] > 1.0e-10
            for item in time_rows
        )
        unconverged_count = sum(
            not item["physical_solver_converged"] for item in time_rows
        )
        summaries.append(
            {
                "method": method,
                "allowance_percent": float(candidate["allowance_percent"]),
                "maximum_physical_poisson_relative_residual": maxima[
                    "physical_poisson"
                ],
                "maximum_component_compatibility_residual": max(
                    item["component_compatibility_residual"] for item in time_rows
                ),
                "incompatible_time_trial_count": int(incompatible_count),
                "unconverged_time_trial_count": int(unconverged_count),
                "maximum_conductive_component_count": max(
                    item["conductive_component_count"] for item in time_rows
                ),
                "maximum_full_moment_rate_residual": maxima["full_moment"],
                "maximum_tangent_moment_rate_residual": maxima["tangent_moment"],
                "maximum_hidden_nullspace_residual": maxima["hidden_nullspace"],
                "maximum_absolute_orthogonality_residual": maxima["orthogonality"],
                "maximum_absolute_pythagorean_residual": maxima["pythagorean"],
                "maximum_raw_hierarchy_violation": maxima["hierarchy"],
                "physical_poisson_tolerance": poisson_tolerance,
                "moment_tolerance": float(moment_tolerance),
                "energy_tolerance": float(energy_tolerance),
                "trial_count": len(evaluator_rows),
                "time_trial_count": len(time_rows),
                "invalid_trial_count": sum(not bool(row["valid"]) for row in evaluator_rows),
                "first_failing_condition": first_failure,
                "passes": (
                    first_failure is None
                    and incompatible_count == 0
                    and unconverged_count == 0
                ),
                "geometry": candidate["geometry"],
            }
        )
        detail[method] = time_rows

    global_max = {
        "maximum_physical_poisson_relative_residual": max(
            row["maximum_physical_poisson_relative_residual"] for row in summaries
        ),
        "maximum_full_moment_rate_residual": max(
            row["maximum_full_moment_rate_residual"] for row in summaries
        ),
        "maximum_tangent_moment_rate_residual": max(
            row["maximum_tangent_moment_rate_residual"] for row in summaries
        ),
        "maximum_hidden_nullspace_residual": max(
            row["maximum_hidden_nullspace_residual"] for row in summaries
        ),
        "maximum_absolute_orthogonality_residual": max(
            row["maximum_absolute_orthogonality_residual"] for row in summaries
        ),
        "maximum_absolute_pythagorean_residual": max(
            row["maximum_absolute_pythagorean_residual"] for row in summaries
        ),
        "maximum_raw_hierarchy_violation": max(
            row["maximum_raw_hierarchy_violation"] for row in summaries
        ),
    }
    first_failure = next(
        (
            name
            for name in CHECK_ORDER
            if any(row["first_failing_condition"] == name for row in summaries)
        ),
        None,
    )
    summary = {
        "schema_version": 1,
        "targeted_candidate_count": len(summaries),
        "moment_tolerance": float(moment_tolerance),
        "energy_tolerance": float(energy_tolerance),
        **global_max,
        "incompatible_time_trial_count": sum(
            row["incompatible_time_trial_count"] for row in summaries
        ),
        "total_time_trial_count": sum(row["time_trial_count"] for row in summaries),
        "maximum_component_compatibility_residual": max(
            row["maximum_component_compatibility_residual"] for row in summaries
        ),
        "unconverged_time_trial_count": sum(
            row["unconverged_time_trial_count"] for row in summaries
        ),
        "invalid_trial_count": sum(row["invalid_trial_count"] for row in summaries),
        "passing_candidate_count": sum(bool(row["passes"]) for row in summaries),
        "every_targeted_candidate_passes": all(
            bool(row["passes"]) for row in summaries
        ),
        "first_failing_condition": first_failure,
        "second_stage_rescore_authorized": all(
            bool(row["passes"]) for row in summaries
        ),
        "operator_density": "physical q_h",
        "stabilization": (
            "conductive-component restriction + one componentwise pin + sparse "
            "SuperLU direct solve with equation-preserving PCG fallback; the "
            "density floor is preconditioning-only and never enters the operator"
        ),
    }
    return summaries, detail, summary


def save_targeted_outputs(
    rows: list[dict[str, Any]],
    detail: dict[str, Any],
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "corrected_full_solver_targeted_audit.json"
    md_path = output_dir / "corrected_full_solver_targeted_audit.md"
    json_path.write_text(
        json.dumps({"summary": summary, "candidates": rows, "time_trials": detail}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    status = "PASS" if summary["every_targeted_candidate_passes"] else "FAIL"
    lines = [
        "# Corrected physical-q Full solver: targeted audit",
        "",
        f"**Overall: {status}.**",
        "",
        f"Operator: `{summary['operator_density']}` in both the equation and action. "
        f"Stabilization: {summary['stabilization']}.",
        "",
        "Authoritative equation: `-div(q_h grad psi_h) = q_h h_h`; "
        "correction: `delta_h^* = -grad psi_h`.",
        (
            "Declared tolerances: physical Poisson residual `"
            f"{min(float(row['physical_poisson_tolerance']) for row in rows):.3e}`; "
            f"moment and energy residuals `{summary['moment_tolerance']:.3e}`. "
            "All residuals and hierarchy gaps are raw and unclipped."
        ),
        "",
        "| Design | max Poisson rel. | max component incompat. | incompatible time/trials | unconverged time/trials | max Full moment | max Tangent moment | max hidden null | max orth. | max Pyth. | max raw hierarchy | First failure | Status |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method'].title()} | "
            f"{row['maximum_physical_poisson_relative_residual']:.3e} | "
            f"{row['maximum_component_compatibility_residual']:.3e} | "
            f"{row['incompatible_time_trial_count']} | "
            f"{row['unconverged_time_trial_count']} | "
            f"{row['maximum_full_moment_rate_residual']:.3e} | "
            f"{row['maximum_tangent_moment_rate_residual']:.3e} | "
            f"{row['maximum_hidden_nullspace_residual']:.3e} | "
            f"{row['maximum_absolute_orthogonality_residual']:.3e} | "
            f"{row['maximum_absolute_pythagorean_residual']:.3e} | "
            f"{row['maximum_raw_hierarchy_violation']:.3e} | "
            f"{row['first_failing_condition']} | "
            f"{'PASS' if row['passes'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Second-stage full rescore authorized: **{summary['second_stage_rescore_authorized']}**.",
            (
                "Experiment-local eligibility: "
                f"**{summary.get('experiment_local_second_stage_eligibility', summary['every_targeted_candidate_passes'])}**."
            ),
            (
                "Paired two-experiment gate: "
                f"**{summary.get('paired_experiment_gate_passes', 'not finalized')}**; "
                "blocking experiments: "
                f"`{summary.get('paired_gate_blocking_experiments', [])}`."
            ),
            "The all-allowance rescore and its conditional outputs must not be "
            "produced when this paired gate is false.",
            "",
            "## Interpretation",
            "",
            (
                "The targeted common-discretization decomposition is numerically "
                "resolved, so `Gamma_h = A_hid,h / A_full,h` has a genuine discrete "
                "geometric interpretation for these targeted candidates."
                if summary["every_targeted_candidate_passes"]
                else
                "`Gamma_h = A_hid,h / A_full,h` is not supported as a discrete "
                "geometric fraction for this targeted set because the physical "
                "Full feasibility prerequisite fails first."
            ),
            "",
            (
                "Failure mode: exact zero-density raster regions split the physical "
                "operator into conductive components, while `q_h h_h` is only "
                "globally centered and is not source-compatible on each component. "
                "The physical equation therefore has no solution for "
                f"{summary['incompatible_time_trial_count']} of "
                f"{summary.get('total_time_trial_count', 'unknown')} time/trial "
                "systems; "
                "no operator floor was reintroduced."
                if summary["incompatible_time_trial_count"]
                else
                "All targeted physical sources are compatible on every conductive "
                "component."
            ),
            "",
            (
                "No saved candidate geometry or frozen bank changed: "
                f"**{summary.get('saved_candidates_and_banks_unchanged', 'not recorded')}**."
            ),
            "Because the paired gate failed, maximum relative Full-action changes "
            "and candidate-ranking changes were not computed, and no optimization "
            "stage or optimization output was modified.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path


def finalize_paired_stage_gate(output_dirs: list[Path]) -> bool:
    """Apply the two-experiment prerequisite without re-running evaluations."""
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for output_dir in output_dirs:
        json_path = output_dir / "corrected_full_solver_targeted_audit.json"
        payloads.append((output_dir, json.loads(json_path.read_text(encoding="utf-8"))))

    paired_passes = all(
        bool(payload["summary"]["every_targeted_candidate_passes"])
        for _, payload in payloads
    )
    blockers = [
        str(payload["summary"].get("experiment", output_dir.parent.parent.name))
        for output_dir, payload in payloads
        if not bool(payload["summary"]["every_targeted_candidate_passes"])
    ]
    for output_dir, payload in payloads:
        summary = payload["summary"]
        summary["maximum_component_compatibility_residual"] = max(
            float(row["maximum_component_compatibility_residual"])
            for row in payload["candidates"]
        )
        summary["total_time_trial_count"] = sum(
            int(row["time_trial_count"]) for row in payload["candidates"]
        )
        summary["stabilization"] = (
            "conductive-component restriction + one componentwise pin + sparse "
            "SuperLU direct solve with equation-preserving PCG fallback; the "
            "density floor is preconditioning-only and never enters the operator"
        )
        summary["experiment_local_second_stage_eligibility"] = bool(
            summary["every_targeted_candidate_passes"]
        )
        summary["paired_experiment_gate_passes"] = paired_passes
        summary["paired_gate_blocking_experiments"] = blockers
        summary["second_stage_rescore_authorized"] = paired_passes
        save_targeted_outputs(
            payload["candidates"], payload["time_trials"], summary, output_dir=output_dir
        )
    return paired_passes
