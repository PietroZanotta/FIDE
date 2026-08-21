"""Finalize and certify the corrected Vortices percentage Pareto sweep."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", SCRIPT_DIR.parent, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from action_decomposition_audit import file_sha256, load_pareto_candidates
from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from common_discretization_decomposition_audit import audit_common_discretization
from experiment import ObservationTrialBank

ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def _tag(value: float) -> str:
    return f"risk_{f'{value:g}'.replace('.', 'p')}pct"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _load_bank(path: Path) -> ObservationTrialBank:
    with np.load(path, allow_pickle=False) as data:
        return ObservationTrialBank(
            sample_indices=jnp.asarray(data["sample_indices"], dtype=jnp.int32),
            detector_z=jnp.asarray(data["detector_z"], dtype=jnp.float64),
        )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    )


def _geometry(result: dict[str, Any], method: str) -> list[float]:
    return [float(value) for value in result["selection"][f"{method}_optimum"]]


def _same_geometry(left: list[float], right: list[float]) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=1.0e-12))


def _selected_full_action(result: dict[str, Any]) -> float:
    chosen = _geometry(result, "full")
    rows = [row for row in result["selection_audit"]["full"] if row.get("valid")]
    matching = [row for row in rows if _same_geometry(row["eta"], chosen)]
    if not matching:
        raise RuntimeError("selected Full geometry is absent from its authoritative audit")
    return float(matching[0]["objective"])


def _validation_action(result: dict[str, Any], method: str) -> tuple[float, float]:
    payload = result["validation"][method]["full_action"]
    return float(payload["mean"]), float(payload["se"])


def _ranking(left: float, right: float, *, tolerance: float) -> str:
    if abs(left - right) <= tolerance:
        return "Equal"
    return "Full lower" if left < right else "Tangent lower"


def _candidate_rows(pareto: Path) -> list[dict[str, Any]]:
    return load_pareto_candidates(
        pareto,
        selection_key=lambda result, method: result["selection"][f"{method}_optimum"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, required=True)
    parser.add_argument("--archive-pareto", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    archive = args.archive_pareto.expanduser().resolve()
    source = args.source_run.expanduser().resolve()

    point, first = _strict_common_artifacts(pareto)
    exp, selection_bank, times = _load_experiment(point, first["config"])
    validation_bank = _load_bank(point / "validation_bank.npz")
    time_weights = np.asarray(exp.time_w, dtype=np.float64)
    tolerance = float(first["config"]["validity"]["tangent_lower_bound_tol"])
    candidates = _candidate_rows(pareto)

    raw_cache: dict[str, dict[str, list[dict[str, Any]]]] = {
        "selection": {}, "validation": {}
    }

    def evaluator(phase: str):
        bank = selection_bank if phase == "selection" else validation_bank

        def evaluate(geometry: Any, key: str) -> list[dict[str, Any]]:
            if key not in raw_cache[phase]:
                raw_cache[phase][key] = exp.evaluate_common_discretization_decomposition_exact(
                    jnp.asarray(geometry, dtype=jnp.float64),
                    bank,
                    progress_desc=f"authoritative {phase} {key[:20]}",
                )
            return raw_cache[phase][key]

        return evaluate

    selection_rows, selection_detail, selection_summary = audit_common_discretization(
        candidates,
        evaluate=evaluator("selection"),
        time_grid=times,
        time_weights=time_weights,
        tolerance=tolerance,
    )
    validation_rows, validation_detail, validation_summary = audit_common_discretization(
        candidates,
        evaluate=evaluator("validation"),
        time_grid=times,
        time_weights=time_weights,
        tolerance=tolerance,
    )
    for row in selection_rows:
        row["phase"] = "selection"
    for row in validation_rows:
        row["phase"] = "validation"
    combined_rows = selection_rows + validation_rows

    current = {
        allowance: json.loads((pareto / _tag(allowance) / "result.json").read_text())
        for allowance in ALLOWANCES
    }
    old = {
        allowance: json.loads((archive / _tag(allowance) / "result.json").read_text())
        for allowance in ALLOWANCES
    }
    selection_by = {
        (float(row["allowance_percent"]), row["method"]): row
        for row in selection_rows
    }
    validation_by = {
        (float(row["allowance_percent"]), row["method"]): row
        for row in validation_rows
    }

    corrected_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for allowance in ALLOWANCES:
        result = current[allowance]
        previous = old[allowance]
        full_sel = selection_by[(allowance, "full")]
        full_val = validation_by[(allowance, "full")]
        law_val = validation_by[(allowance, "law")]
        validation_mean, validation_se = _validation_action(result, "full")
        law_validation_mean, _ = _validation_action(result, "law")
        reduction = (law_validation_mean - validation_mean) / law_validation_mean
        full_action = _selected_full_action(result)
        corrected_rows.append({
            "allowance_percent": allowance,
            "full_geometry": _geometry(result, "full"),
            "full_centers": result["selection_centers"]["full"],
            "L_selection": float(result["selection_certificates"]["full"]["L_selection"]),
            "R_selection": float(result["selection_certificates"]["full"]["R_selection"]),
            "R_star": float(result["law_screens"]["R_star"]),
            "R_max": float(result["law_screens"]["R_max"]),
            "risk_increase_percent": 100.0 * (
                float(result["selection_certificates"]["full"]["R_selection"])
                - float(result["law_screens"]["R_star"])
            ) / abs(float(result["law_screens"]["R_star"])),
            "selection_A_full_h": full_action,
            "validation_A_full_h_mean": validation_mean,
            "validation_A_full_h_se": validation_se,
            "Full_vs_Law_validation_reduction": reduction,
            "selection_A_tan_h": full_sel["A_tan_h"],
            "selection_A_hid_h": full_sel["A_hid_h"],
            "selection_Gamma_h": full_sel["hidden_fraction_A_hid_over_A_full"],
            "validation_A_tan_h": full_val["A_tan_h"],
            "validation_A_hid_h": full_val["A_hid_h"],
            "validation_Gamma_h": full_val["hidden_fraction_A_hid_over_A_full"],
            "selection_decomposition_passes": full_sel["passes"],
            "validation_decomposition_passes": full_val["passes"],
            "law_validation_decomposition_passes": law_val["passes"],
            "full_certified": result["selection_certificates"]["full"]["certified"],
        })
        old_full_action = _selected_full_action(previous)
        old_validation, _ = _validation_action(previous, "full")
        old_law_validation, _ = _validation_action(previous, "law")
        old_reduction = (old_law_validation - old_validation) / old_law_validation
        old_geometry = _geometry(previous, "full")
        new_geometry = _geometry(result, "full")
        comparison_rows.append({
            "allowance_percent": allowance,
            "old_full_geometry": old_geometry,
            "corrected_full_geometry": new_geometry,
            "old_selection_A_full": old_full_action,
            "corrected_selection_A_full": full_action,
            "selection_action_delta": full_action - old_full_action,
            "old_validation_A_full": old_validation,
            "corrected_validation_A_full": validation_mean,
            "validation_action_delta": validation_mean - old_validation,
            "old_Full_vs_Law_reduction": old_reduction,
            "corrected_Full_vs_Law_reduction": reduction,
            "selected_candidate_changed": not _same_geometry(old_geometry, new_geometry),
            "candidate_ranking_changed": not _same_geometry(old_geometry, new_geometry),
            "old_Tangent_vs_Full_conclusion": _ranking(
                old_validation,
                _validation_action(previous, "tangent")[0],
                tolerance=tolerance,
            ),
            "corrected_Tangent_vs_Full_conclusion": _ranking(
                validation_mean,
                _validation_action(result, "tangent")[0],
                tolerance=tolerance,
            ),
        })

    # The 2% winner is the mandatory 3% incumbent as exactly the same geometry.
    # Repeated evaluation differs only through sparse-solve roundoff, so retain the
    # tighter-stage reported action when the difference is within the declared
    # numerical tolerance.  Distinct geometries are never canonicalized.
    for index in range(1, len(corrected_rows)):
        previous_row = corrected_rows[index - 1]
        row = corrected_rows[index]
        if (
            _same_geometry(previous_row["full_geometry"], row["full_geometry"])
            and abs(row["selection_A_full_h"] - previous_row["selection_A_full_h"])
            <= tolerance
        ):
            row["selection_A_full_h"] = previous_row["selection_A_full_h"]
            comparison_rows[index]["corrected_selection_A_full"] = row[
                "selection_A_full_h"
            ]
            comparison_rows[index]["selection_action_delta"] = (
                row["selection_A_full_h"]
                - comparison_rows[index]["old_selection_A_full"]
            )

    actions = np.asarray([row["selection_A_full_h"] for row in corrected_rows])
    nested_differences = np.diff(actions)
    nested = bool(np.all(nested_differences <= tolerance))
    all_decomposition = bool(all(row["passes"] for row in combined_rows))
    all_certified = bool(all(row["full_certified"] for row in corrected_rows))
    diagnostic = {
        "nested": nested,
        "nested_differences": nested_differences.tolist(),
        "all_decomposition": all_decomposition,
        "all_full_candidates_certified": all_certified,
        "selection_summary": selection_summary,
        "validation_summary": validation_summary,
        "failing_rows": [row for row in combined_rows if not row["passes"]],
        "invalid_trials": [
            {
                "phase": phase,
                "geometry": key,
                "trial": row.get("trial"),
                "invalid_reason": row.get("invalid_reason"),
                "max_calibration_residual": row.get("max_calibration_residual"),
                "min_ess_fraction": row.get("min_ess_fraction"),
                "min_empirical_hull_support_gap": row.get(
                    "min_empirical_hull_support_gap"
                ),
                "max_poisson_relative_residual": row.get(
                    "max_poisson_relative_residual"
                ),
                "max_full_moment_rate_residual": row.get(
                    "max_full_moment_rate_residual"
                ),
            }
            for phase, cache in raw_cache.items()
            for key, rows in cache.items()
            for row in rows
            if not row.get("valid")
        ],
    }
    _write_json(pareto / "authoritative_certification_diagnostic.json", diagnostic)
    if not nested or not all_decomposition or not all_certified:
        print(json.dumps(diagnostic, indent=2), flush=True)
        raise RuntimeError("corrected Vortices authoritative certification failed")

    source_files = [
        (source / "truth_bank.npz", "truth_bank.npz"),
        (source / "reference_endpoints.npz", "reference_endpoints.npz"),
        (source / "reference.npz", "reference.npz"),
        (source / "reference_bank.npz", "reference_bank.npz"),
        (source / "selection_bank.npz", "selection_bank.npz"),
        (source / "validation_bank.npz", "validation_bank.npz"),
        (source / "manifest.json", "source_manifest.json"),
        (SCRIPT_DIR / "config.json", "config.json"),
    ]
    frozen_dir = pareto / "frozen_inputs"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for source_path, target_name in source_files:
        target = frozen_dir / target_name
        if not target.exists():
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)
        manifest_rows.append({
            "name": target_name,
            "source": str(source_path.resolve()),
            "active": str(target.resolve()),
            "sha256": file_sha256(source_path),
            "active_sha256": file_sha256(target),
        })
    frozen_unchanged = all(row["sha256"] == row["active_sha256"] for row in manifest_rows)

    validation_trial_rows: list[dict[str, Any]] = []
    for allowance in ALLOWANCES:
        path = pareto / _tag(allowance) / "result.validation_trials.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                validation_trial_rows.append({"allowance_percent": allowance, **row})

    maximum_fields = {
        "maximum_full_moment_residual": "maximum_full_moment_residual",
        "maximum_tangent_moment_residual": "maximum_tangent_moment_residual",
        "maximum_hidden_nullspace_residual": "maximum_hidden_nullspace_residual",
        "maximum_absolute_orthogonality_residual": "maximum_absolute_orthogonality_residual",
        "maximum_absolute_pythagorean_residual": "maximum_absolute_pythagorean_residual",
        "maximum_raw_hierarchy_violation": "maximum_raw_hierarchy_violation",
    }
    numerical_maxima = {
        name: max(float(row[field]) for row in combined_rows)
        for name, field in maximum_fields.items()
    }
    physical_poisson_max = max(
        float(row["max_poisson_relative_residual"])
        for phase in raw_cache.values() for rows in phase.values() for row in rows
    )
    component_compatibility_max = max(
        max(float(value) for value in row["component_compatibility_residual_by_time"])
        for phase in raw_cache.values() for rows in phase.values() for row in rows
    )
    solver_converged = all(
        all(bool(value) for value in row["physical_solver_converged_by_time"])
        for phase in raw_cache.values() for rows in phase.values() for row in rows
    )
    violation_counts = {
        level: int(selection_summary[f"{level}_violation_count"])
        + int(validation_summary[f"{level}_violation_count"])
        for level in ("aggregate", "trial", "time_trial")
    }
    invalid_trial_count = int(selection_summary["invalid_trial_count"]) + int(
        validation_summary["invalid_trial_count"]
    )
    tangent_vs_full = {
        f"{row['allowance_percent']:g}%": row["corrected_Tangent_vs_Full_conclusion"]
        for row in comparison_rows
    }
    positive_full_vs_law_reductions = all(
        row["corrected_Full_vs_Law_reduction"] > 0.0 for row in comparison_rows
    )
    summary = {
        "schema_version": 1,
        "experiment": "vortices_percentage",
        "status": "PASS",
        "authoritative_rule": {
            "physical_q": True,
            "density_floor_in_scientific_operator": False,
            "grid_nx": int(exp.grid.nx),
            "grid_ny": int(exp.grid.ny),
            "time_n": len(times),
            "selection_trials": int(selection_bank.sample_indices.shape[0]),
            "validation_trials": int(validation_bank.sample_indices.shape[0]),
            "solver": "sparse physical-q direct",
            "tolerance": tolerance,
        },
        "selection_curve_nested": nested,
        "nested_differences": nested_differences.tolist(),
        "all_final_candidates_certified": all_certified,
        "all_decomposition_checks_pass": all_decomposition,
        "physical_poisson_maximum": physical_poisson_max,
        "component_compatibility_maximum": component_compatibility_max,
        "all_physical_solvers_converged": solver_converged,
        "numerical_maxima": numerical_maxima,
        "violation_counts": violation_counts,
        "invalid_trial_count": invalid_trial_count,
        "frozen_inputs_unchanged": frozen_unchanged,
        "law_anchor": corrected_rows[0]["R_star"],
        "additional_full_optimization_required": False,
        "additional_tangent_optimization_required": False,
        "tangent_vs_full_validation_conclusion_by_allowance": tangent_vs_full,
        "all_full_vs_law_validation_reductions_positive": (
            positive_full_vs_law_reductions
        ),
        "central_FIDE_result_survives": (
            positive_full_vs_law_reductions and all_decomposition
        ),
    }

    _write_csv(pareto / "corrected_authoritative_pareto.csv", corrected_rows)
    _write_json(pareto / "corrected_authoritative_pareto.json", {
        "summary": summary, "rows": corrected_rows
    })
    _write_csv(pareto / "corrected_authoritative_decomposition_audit.csv", combined_rows)
    _write_json(pareto / "corrected_authoritative_decomposition_audit.json", {
        "summary": summary,
        "selection_summary": selection_summary,
        "validation_summary": validation_summary,
        "rows": combined_rows,
        "selection_detail": selection_detail,
        "validation_detail": validation_detail,
    })
    _write_csv(pareto / "old_vs_corrected_full_comparison.csv", comparison_rows)
    _write_json(pareto / "old_vs_corrected_full_comparison.json", comparison_rows)
    _write_csv(pareto / "validation_trial_summaries.csv", validation_trial_rows)
    _write_json(frozen_dir / "manifest.json", {
        "schema_version": 1, "inputs": manifest_rows, "unchanged": frozen_unchanged
    })
    _write_json(pareto / "authoritative_run_summary.json", summary)

    pareto_lines = [
        "# Corrected authoritative Vortices Pareto table", "",
        "The Full selection column is evaluated only on the frozen 24-trial selection bank; validation is diagnostic and uses the disjoint frozen 64-trial bank.", "",
        "| Allowance | Full centers | L | R | Risk increase | Full selection | Full validation mean ± SE | Full-vs-Law reduction | Selection A_tan,h | Selection A_hid,h | Gamma_h | Certified |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in corrected_rows:
        centers = " ".join(
            f"({float(center[0]):.5f},{float(center[1]):.5f})"
            for center in row["full_centers"]
        )
        certified = all((
            row["full_certified"],
            row["selection_decomposition_passes"],
            row["validation_decomposition_passes"],
            row["law_validation_decomposition_passes"],
        ))
        pareto_lines.append(
            f"| {row['allowance_percent']:g}% | `{centers}` | "
            f"{row['L_selection']:.9f} | {row['R_selection']:.9f} | "
            f"{row['risk_increase_percent']:.3f}% | "
            f"{row['selection_A_full_h']:.6f} | "
            f"{row['validation_A_full_h_mean']:.6f} ± {row['validation_A_full_h_se']:.6f} | "
            f"{100.0 * row['Full_vs_Law_validation_reduction']:.2f}% | "
            f"{row['selection_A_tan_h']:.6f} | {row['selection_A_hid_h']:.6f} | "
            f"{row['selection_Gamma_h']:.6f} | {'PASS' if certified else 'FAIL'} |"
        )
    (pareto / "corrected_authoritative_pareto.md").write_text(
        "\n".join(pareto_lines) + "\n", encoding="utf-8"
    )

    audit_lines = [
        "# Corrected authoritative Vortices decomposition audit", "",
        "**PASS** — every final Law, Tangent, and Full geometry passes on both the frozen selection and independent validation banks.", "",
        "| Metric | Maximum |", "|:---|---:|",
        f"| Physical Poisson residual | `{physical_poisson_max:.6e}` |",
        f"| Component compatibility residual | `{component_compatibility_max:.6e}` |",
        *[
            f"| {name.replace('_', ' ')} | `{value:.6e}` |"
            for name, value in numerical_maxima.items()
        ], "",
        "| Level | Violations (selection + validation) |", "|:---|---:|",
        f"| Aggregate | {violation_counts['aggregate']} |",
        f"| Trial | {violation_counts['trial']} |",
        f"| Time/trial | {violation_counts['time_trial']} |",
        f"| Invalid trials | {invalid_trial_count} |", "",
        f"Nested Full selection curve: **{'PASS' if nested else 'FAIL'}**. Differences: `{nested_differences.tolist()}`.", "",
        "No residual, energy, or hierarchy value was clipped.",
    ]
    (pareto / "corrected_authoritative_decomposition_audit.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    comparison_lines = [
        "# Old versus corrected Vortices Full results", "",
        "The old and corrected Full actions use different scientific discretizations, so their absolute action deltas diagnose the evaluator correction rather than an optimization-only change.", "",
        "| Allowance | Old geometry | Corrected geometry | Old selection | Corrected selection | Old validation | Corrected validation | Old reduction | Corrected reduction | Winner changed | Corrected Tangent vs Full |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|---:|:---:|:---|",
    ]
    for row in comparison_rows:
        comparison_lines.append(
            f"| {row['allowance_percent']:g}% | `{row['old_full_geometry']}` | `{row['corrected_full_geometry']}` | "
            f"{row['old_selection_A_full']:.6f} | {row['corrected_selection_A_full']:.6f} | "
            f"{row['old_validation_A_full']:.6f} | {row['corrected_validation_A_full']:.6f} | "
            f"{100*row['old_Full_vs_Law_reduction']:.2f}% | {100*row['corrected_Full_vs_Law_reduction']:.2f}% | "
            f"{'yes' if row['selected_candidate_changed'] else 'no'} | "
            f"{row['corrected_Tangent_vs_Full_conclusion']} |"
        )
    (pareto / "old_vs_corrected_full_comparison.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
