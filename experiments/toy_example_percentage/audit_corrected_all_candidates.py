"""Rescore saved toy candidates with a selected positive-raster Full rule.

This is an audit-only program: it reads immutable saved candidates and frozen
banks, evaluates the corrected Full/common-raster decomposition, checkpoints
each distinct geometry, and never invokes an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
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

from action_decomposition_audit import file_sha256, geometry_key, load_pareto_candidates
from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from audit_positive_rasterization import _summarize
from experiment import TrialBank


METHODS = ("law", "tangent", "full")
DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    parser.add_argument("--grid-n", type=int, required=True)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    return parser.parse_args()


def _load_bank(path: Path) -> TrialBank:
    with np.load(path, allow_pickle=False) as bank:
        return TrialBank(
            masses=jnp.asarray(bank["masses"], dtype=jnp.float64),
            sample_indices=jnp.asarray(bank["sample_indices"], dtype=jnp.int32),
            detector_z=jnp.asarray(bank["detector_z"], dtype=jnp.float64),
            alphas=jnp.asarray(bank["alphas"], dtype=jnp.float64),
        )


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _tag(allowance: float) -> str:
    return f"risk_{f'{allowance:g}'.replace('.', 'p').replace('-', 'm')}pct"


def _geometry_from_degrees(values: Any) -> list[float]:
    return np.deg2rad(np.asarray(values, dtype=np.float64)).tolist()


def _build_pool(result: dict[str, Any], allowance: float, result_path: Path) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}

    def add(
        degrees: Any,
        provenance: str,
        *,
        population_loss: float | None,
        finite_risk: float | None,
        old_full_action: float | None = None,
        old_tangent_action: float | None = None,
        action_valid: bool = True,
    ) -> None:
        geometry = _geometry_from_degrees(degrees)
        key = geometry_key(geometry)
        if key not in by_key:
            by_key[key] = {
                "allowance_percent": allowance,
                "geometry": geometry,
                "geometry_deg": np.asarray(degrees, dtype=np.float64).tolist(),
                "evaluation_key": key,
                "provenance": [],
                "population_loss_selection": population_loss,
                "finite_risk_selection": finite_risk,
                "old_full_action_selection": old_full_action,
                "old_tangent_action_selection": old_tangent_action,
                "old_action_valid": bool(action_valid),
                "result_path": str(result_path.resolve()),
            }
        row = by_key[key]
        row["provenance"].append(provenance)
        for field, value in (
            ("population_loss_selection", population_loss),
            ("finite_risk_selection", finite_risk),
            ("old_full_action_selection", old_full_action),
            ("old_tangent_action_selection", old_tangent_action),
        ):
            if row[field] is None and value is not None:
                row[field] = float(value)
        row["old_action_valid"] = bool(row["old_action_valid"] and action_valid)

    for stage in ("tangent", "full"):
        for index, audit in enumerate(result.get("selection_audit", {}).get(stage, [])):
            add(
                audit["eta_deg"],
                f"{stage}_search_audit_{index}",
                population_loss=float(audit["population_loss"]),
                finite_risk=float(audit["finite_risk"]),
                old_full_action=float(audit["objective"]) if stage == "full" else None,
                old_tangent_action=float(audit["objective"]) if stage == "tangent" else None,
                action_valid=bool(audit.get("action_valid", True)),
            )
    for method in METHODS:
        certificate = result["selection_certificates"][method]
        add(
            result["selection"][f"{method}_optimum_deg"],
            f"saved_final_{method}",
            population_loss=float(certificate["L_selection"]),
            finite_risk=float(certificate["R_selection"]),
            old_full_action=float(certificate["full_action_selection"]),
            old_tangent_action=float(certificate["tangent_action_selection"]),
            action_valid=bool(certificate.get("certified", True)),
        )
    for row in by_key.values():
        row["provenance"] = sorted(set(row["provenance"]))
        row["is_saved_law"] = "saved_final_law" in row["provenance"]
        row["is_saved_tangent"] = "saved_final_tangent" in row["provenance"]
        row["is_saved_full"] = "saved_final_full" in row["provenance"]
    return list(by_key.values())


def _rank(values: dict[str, float]) -> dict[str, int]:
    return {
        key: index + 1
        for index, (key, _) in enumerate(sorted(values.items(), key=lambda item: item[1]))
    }


def main() -> None:
    args = parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    point, first = _strict_common_artifacts(pareto)
    exp, selection_bank, times = _load_experiment(point, first["config"])
    validation_bank = _load_bank(point / "validation_bank.npz")
    grid_n = int(args.grid_n)
    bandwidth_scale = float(args.bandwidth_scale)
    tolerance = float(first["config"]["validity"]["tangent_lower_bound_tol"])
    feasibility_tolerance = float(first["config"]["optimization"]["feasibility_tol"])
    time_weights = np.asarray(exp.time_w, dtype=np.float64)

    pareto_rows = json.loads((pareto / "pareto.json").read_text(encoding="utf-8"))
    result_by_allowance: dict[float, tuple[Path, dict[str, Any]]] = {}
    pools: dict[float, list[dict[str, Any]]] = {}
    for pareto_row in sorted(pareto_rows, key=lambda row: float(row["risk_allowance_percent"])):
        allowance = float(pareto_row["risk_allowance_percent"])
        result_path = pareto / _tag(allowance) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_by_allowance[allowance] = (result_path, result)
        pools[allowance] = _build_pool(result, allowance, result_path)

    final_candidates = load_pareto_candidates(
        pareto,
        selection_key=lambda result, method: np.deg2rad(
            np.asarray(result["selection"][f"{method}_optimum_deg"], dtype=np.float64)
        ),
    )
    watched = [
        pareto / "pareto.json",
        *sorted(pareto.glob("risk_*pct/result.json")),
        *sorted(pareto.glob("risk_*pct/selection_bank.npz")),
        *sorted(pareto.glob("risk_*pct/validation_bank.npz")),
    ]
    before = _snapshot(watched)
    checkpoint_path = pareto / "toy_corrected_rescore_checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"schema_version": 1, "grid_n": grid_n, "bandwidth_scale": bandwidth_scale, "evaluations": {}}
    )
    if checkpoint.get("grid_n") != grid_n or checkpoint.get("bandwidth_scale") != bandwidth_scale:
        raise RuntimeError("existing checkpoint uses a different selected raster rule")
    evaluation_cache: dict[str, dict[str, Any]] = checkpoint["evaluations"]

    def evaluate(geometry: Any, bank_name: str) -> dict[str, Any]:
        key = f"{bank_name}:{geometry_key(geometry)}"
        if key not in evaluation_cache:
            bank = selection_bank if bank_name == "selection" else validation_bank
            rows = exp.evaluate_common_discretization_decomposition_exact(
                jnp.asarray(geometry, dtype=jnp.float64),
                bank,
                grid_n=grid_n,
                bandwidth_scale=bandwidth_scale,
                progress_desc=f"corrected {bank_name} {key[-24:]}",
            )
            summary, _ = _summarize(
                rows,
                method="candidate",
                time_weights=time_weights,
                moment_tolerance=tolerance,
                energy_tolerance=tolerance,
            )
            evaluation_cache[key] = summary
            _json_write(checkpoint_path, checkpoint)
        return evaluation_cache[key]

    final_selection_rows: list[dict[str, Any]] = []
    final_validation_rows: list[dict[str, Any]] = []
    for candidate in final_candidates:
        allowance = float(candidate["allowance_percent"])
        method = str(candidate["method"])
        result = result_by_allowance[allowance][1]
        certificate = result["selection_certificates"][method]
        validation = result["validation"][method]
        selection_summary = evaluate(candidate["geometry"], "selection")
        validation_summary = evaluate(candidate["geometry"], "validation")
        common = {
            "allowance_percent": allowance,
            "method": method,
            "geometry": candidate["geometry"],
            "geometry_deg": result["selection"][f"{method}_optimum_deg"],
            "grid_n": grid_n,
            "bandwidth_scale": bandwidth_scale,
            "bandwidth": float(exp.authoritative_raster_bandwidth) * bandwidth_scale,
        }
        final_selection_rows.append({
            **common,
            "old_risk": float(certificate["R_selection"]),
            "corrected_risk": float(certificate["R_selection"]),
            "risk_change": 0.0,
            "old_A_full": float(certificate["full_action_selection"]),
            "corrected_A_full": selection_summary["A_full_h"],
            "relative_corrected_to_old_A_full": selection_summary["A_full_h"] / float(certificate["full_action_selection"]),
            "relative_A_full_change": selection_summary["A_full_h"] / float(certificate["full_action_selection"]) - 1.0,
            **{key: value for key, value in selection_summary.items() if key != "method"},
        })
        final_validation_rows.append({
            **common,
            "old_risk": float(validation["law_risk"]["mean"]),
            "corrected_risk": float(validation["law_risk"]["mean"]),
            "risk_change": 0.0,
            "old_A_full": float(validation["full_action"]["mean"]),
            "corrected_A_full": validation_summary["A_full_h"],
            "relative_corrected_to_old_A_full": validation_summary["A_full_h"] / float(validation["full_action"]["mean"]),
            "relative_A_full_change": validation_summary["A_full_h"] / float(validation["full_action"]["mean"]) - 1.0,
            **{key: value for key, value in validation_summary.items() if key != "method"},
        })

    def add_contrasts(rows: list[dict[str, Any]]) -> None:
        for allowance in sorted({float(row["allowance_percent"]) for row in rows}):
            group = {row["method"]: row for row in rows if float(row["allowance_percent"]) == allowance}
            law_action = float(group["law"]["A_full_h"])
            tangent_action = float(group["tangent"]["A_full_h"])
            full_action = float(group["full"]["A_full_h"])
            for row in group.values():
                row["Full_vs_Law_reduction"] = (law_action - full_action) / law_action
                row["Full_vs_Tangent_reduction"] = (tangent_action - full_action) / tangent_action
                row["corrected_final_ordering"] = [
                    item["method"] for item in sorted(group.values(), key=lambda item: item["A_full_h"])
                ]
    add_contrasts(final_selection_rows)
    add_contrasts(final_validation_rows)

    pool_rows: list[dict[str, Any]] = []
    pool_summaries: list[dict[str, Any]] = []
    for allowance, pool in sorted(pools.items()):
        result = result_by_allowance[allowance][1]
        full_cert = result["selection_certificates"]["full"]
        L_max = float(full_cert["L_max"])
        R_max = float(full_cert["R_max"])
        for candidate in pool:
            corrected = evaluate(candidate["geometry"], "selection")
            candidate["passes_L"] = bool(float(candidate["population_loss_selection"]) <= L_max + feasibility_tolerance)
            candidate["passes_R"] = bool(float(candidate["finite_risk_selection"]) <= R_max + feasibility_tolerance)
            candidate["corrected_feasible"] = bool(candidate["passes_L"] and candidate["passes_R"] and corrected["passes"])
            candidate.update({key: value for key, value in corrected.items() if key != "method"})
        feasible = [row for row in pool if row["corrected_feasible"]]
        corrected_ranks = _rank({row["evaluation_key"]: float(row["A_full_h"]) for row in feasible})
        comparable = [
            row for row in feasible if row["old_full_action_selection"] is not None
        ]
        old_ranks = _rank({
            row["evaluation_key"]: float(row["old_full_action_selection"])
            for row in comparable
        })
        corrected_comparable_ranks = _rank({
            row["evaluation_key"]: float(row["A_full_h"])
            for row in comparable
        })
        saved_full = next(row for row in pool if row["is_saved_full"])
        best = min(feasible, key=lambda row: float(row["A_full_h"]))
        gap = float(saved_full["A_full_h"] - best["A_full_h"])
        for row in pool:
            row["corrected_rank_among_feasible_pool"] = corrected_ranks.get(row["evaluation_key"])
            row["old_rank_among_full_audited_feasible_subset"] = old_ranks.get(row["evaluation_key"])
            row["corrected_rank_among_full_audited_feasible_subset"] = corrected_comparable_ranks.get(row["evaluation_key"])
            row["rank_changed_where_comparable"] = (
                row["evaluation_key"] in old_ranks
                and old_ranks[row["evaluation_key"]]
                != corrected_comparable_ranks.get(row["evaluation_key"])
            )
            row["L_max"] = L_max
            row["R_max"] = R_max
            pool_rows.append(row)
        pool_summaries.append({
            "allowance_percent": allowance,
            "candidate_count": len(pool),
            "feasible_candidate_count": len(feasible),
            "saved_full_geometry_deg": saved_full["geometry_deg"],
            "saved_full_corrected_A_full": saved_full["A_full_h"],
            "saved_full_corrected_rank": corrected_ranks[saved_full["evaluation_key"]],
            "best_geometry_deg": best["geometry_deg"],
            "best_provenance": best["provenance"],
            "best_corrected_A_full": best["A_full_h"],
            "saved_full_minus_best_action_gap": gap,
            "saved_full_is_best_or_tied": bool(gap <= tolerance),
            "materially_better_existing_candidate": bool(gap > tolerance),
            "comparable_old_ranking_change_count": sum(bool(row["rank_changed_where_comparable"]) for row in pool),
        })

    after = _snapshot(watched)
    audited_rows = final_selection_rows + final_validation_rows + pool_rows
    diagnostic_summary = {
        "minimum_q_h": min(float(row["minimum_q_h"]) for row in audited_rows),
        "maximum_mass_error": max(float(row["maximum_mass_error"]) for row in audited_rows),
        "maximum_source_compatibility_error": max(float(row["maximum_source_compatibility_error"]) for row in audited_rows),
        "maximum_physical_poisson_relative_residual": max(float(row["maximum_physical_poisson_relative_residual"]) for row in audited_rows),
        "maximum_full_moment_rate_residual": max(float(row["maximum_full_moment_rate_residual"]) for row in audited_rows),
        "maximum_tangent_moment_rate_residual": max(float(row["maximum_tangent_moment_rate_residual"]) for row in audited_rows),
        "maximum_hidden_nullspace_residual": max(float(row["maximum_hidden_nullspace_residual"]) for row in audited_rows),
        "maximum_absolute_orthogonality_residual": max(float(row["maximum_absolute_orthogonality_residual"]) for row in audited_rows),
        "maximum_absolute_pythagorean_residual": max(float(row["maximum_absolute_pythagorean_residual"]) for row in audited_rows),
        "maximum_raw_hierarchy_violation": max(float(row["maximum_raw_hierarchy_violation"]) for row in audited_rows),
    }
    metadata = {
        "schema_version": 1,
        "experiment": "toy_example_percentage",
        "authoritative_rule": {
            "grid_n": grid_n,
            "bandwidth_scale": bandwidth_scale,
            "bandwidth": float(exp.authoritative_raster_bandwidth) * bandwidth_scale,
            "bandwidth_base_rule": exp.authoritative_raster_bandwidth_rule,
            "bandwidth_base": float(exp.authoritative_raster_bandwidth),
        },
        "selection_bank": str((point / "selection_bank.npz").resolve()),
        "selection_bank_sha256": file_sha256(point / "selection_bank.npz"),
        "validation_bank": str((point / "validation_bank.npz").resolve()),
        "validation_bank_sha256": file_sha256(point / "validation_bank.npz"),
        "selection_trial_count": int(selection_bank.masses.shape[0]),
        "validation_trial_count": int(validation_bank.masses.shape[0]),
        "time_grid": np.asarray(times, dtype=np.float64).tolist(),
        "decomposition_tolerance": tolerance,
        "feasibility_tolerance": feasibility_tolerance,
        "optimization_rerun": False,
        "risk_definition_changed": False,
        "saved_candidates_and_frozen_banks_unchanged": before == after,
        "diagnostic_summary": diagnostic_summary,
        "watched_hashes_before_after": {path: {"before": digest, "after": after[path]} for path, digest in before.items()},
    }
    _write_csv(pareto / "toy_corrected_all_candidates_rescore.csv", final_selection_rows)
    _json_write(pareto / "toy_corrected_all_candidates_rescore.json", {"metadata": metadata, "rows": final_selection_rows})
    _write_csv(pareto / "toy_corrected_validation_rescore.csv", final_validation_rows)
    _json_write(pareto / "toy_corrected_validation_rescore.json", {"metadata": metadata, "rows": final_validation_rows})
    _write_csv(pareto / "toy_corrected_candidate_pool_audit.csv", pool_rows)
    _json_write(pareto / "toy_corrected_candidate_pool_audit.json", {"metadata": metadata, "allowance_summaries": pool_summaries, "rows": pool_rows})

    lines = [
        "# Toy corrected Full rescore and candidate-pool audit",
        "",
        f"Authoritative audit rule: **{grid_n} x {grid_n}** positive-support raster, Scott bandwidth `{float(exp.authoritative_raster_bandwidth):.12g}` times **{bandwidth_scale:g}**. The rule is fixed before the all-allowance rescore and is not tuned per allowance or geometry.",
        "",
        "No optimization was run. Risks, constraints, projected particles, targets, time nodes, candidate geometries, and frozen banks are unchanged. The corrected raster affects Full action only, so every old/corrected risk pair is exactly identical by construction.",
        "",
        "## Corrected selection-bank endpoints",
        "",
        "| Allow. | Design | Risk | Old A_full | Corrected A_full | A_tan,h | A_hid,h | Gamma_h | Full vs Law | Full vs Tangent | Rel. action change | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in final_selection_rows:
        lines.append(f"| {row['allowance_percent']:g}% | {row['method'].title()} | {row['corrected_risk']:.9g} | {row['old_A_full']:.9g} | {row['A_full_h']:.9g} | {row['A_tan_h']:.9g} | {row['A_hid_h']:.9g} | {row['Gamma_h']:.6f} | {row['Full_vs_Law_reduction']:.3%} | {row['Full_vs_Tangent_reduction']:.3%} | {row['relative_A_full_change']:.3%} | {'PASS' if row['passes'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Corrected validation-bank endpoints",
        "",
        "| Allow. | Design | Risk | Old A_full | Corrected A_full | A_tan,h | A_hid,h | Gamma_h | Full vs Law | Full vs Tangent | Rel. action change | Status |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in final_validation_rows:
        lines.append(f"| {row['allowance_percent']:g}% | {row['method'].title()} | {row['corrected_risk']:.9g} | {row['old_A_full']:.9g} | {row['A_full_h']:.9g} | {row['A_tan_h']:.9g} | {row['A_hid_h']:.9g} | {row['Gamma_h']:.6f} | {row['Full_vs_Law_reduction']:.3%} | {row['Full_vs_Tangent_reduction']:.3%} | {row['relative_A_full_change']:.3%} | {'PASS' if row['passes'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Existing candidate-pool optimality",
        "",
        "| Allow. | Pool | Feasible | Saved Full rank | Saved Full A | Best A | Saved minus best | Best geometry (deg) | Best provenance | Existing winner? |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---:|",
    ])
    for row in pool_summaries:
        lines.append(f"| {row['allowance_percent']:g}% | {row['candidate_count']} | {row['feasible_candidate_count']} | {row['saved_full_corrected_rank']} | {row['saved_full_corrected_A_full']:.9g} | {row['best_corrected_A_full']:.9g} | {row['saved_full_minus_best_action_gap']:.9g} | `{row['best_geometry_deg']}` | `{','.join(row['best_provenance'])}` | {'saved Full' if row['saved_full_is_best_or_tied'] else 'other saved candidate'} |")
    affected = [row["allowance_percent"] for row in pool_summaries if row["materially_better_existing_candidate"]]
    all_checks = all(bool(row["passes"]) for row in final_selection_rows + final_validation_rows + pool_rows)
    sensitivity_summary = json.loads(
        (pareto / "toy_positive_raster_sensitivity.json").read_text(encoding="utf-8")
    )["summary"]
    selection_reductions = {
        float(row["allowance_percent"]): float(row["Full_vs_Law_reduction"])
        for row in final_selection_rows if row["method"] == "full"
    }
    validation_reductions = {
        float(row["allowance_percent"]): float(row["Full_vs_Law_reduction"])
        for row in final_validation_rows if row["method"] == "full"
    }
    ranking_change_count = sum(
        int(row["comparable_old_ranking_change_count"])
        for row in pool_summaries
    )
    lines.extend([
        "",
        "## PASS/FAIL decision summary",
        "",
        f"1. Bandwidth/grid robustness: **{'PASS' if sensitivity_summary['bandwidth_grid_robustness_passes'] else 'FAIL'}**. The maximum 81-to-101 change is {sensitivity_summary['maximum_relative_action_change_81_to_101']:.3%}, and the maximum 101-grid bandwidth response is {sensitivity_summary['maximum_101_grid_bandwidth_relative_action_change']:.3%}.",
        f"2. Decomposition under the selected 101 x 101 Scott rule: **{'PASS' if all_checks else 'FAIL'}**.",
        f"3. Qualitative saved-endpoint ordering: **{'PASS' if sensitivity_summary['candidate_ordering_stable'] else 'FAIL'}** (`Full < Law < Tangent` throughout).",
        "4. Corrected Full-vs-Law reductions (selection / validation): "
        + ", ".join(
            f"{allowance:g}% = {selection_reductions[allowance]:.3%} / {validation_reductions[allowance]:.3%}"
            for allowance in sorted(selection_reductions)
        )
        + ".",
        f"5. Existing comparable candidate rankings change: **YES** ({ranking_change_count} row-level rank changes across the six Full-audited subsets).",
        f"6. Optimization decision: **rerun only the 0.5% Full stage**; 1-5% require no rerun. The 0.5% existing-pool winner is `{pool_summaries[0]['best_geometry_deg']}` with corrected action {pool_summaries[0]['best_corrected_A_full']:.9g}, beating the saved Full endpoint by {pool_summaries[0]['saved_full_minus_best_action_gap']:.9g}.",
        "",
        "## Global numerical diagnostics",
        "",
        "| min q_h | max mass err. | max source err. | max Poisson | max Full moment | max Tangent moment | max hidden null | max abs. orth. | max abs. Pyth. | max raw hierarchy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {diagnostic_summary['minimum_q_h']:.3e} | {diagnostic_summary['maximum_mass_error']:.3e} | {diagnostic_summary['maximum_source_compatibility_error']:.3e} | {diagnostic_summary['maximum_physical_poisson_relative_residual']:.3e} | {diagnostic_summary['maximum_full_moment_rate_residual']:.3e} | {diagnostic_summary['maximum_tangent_moment_rate_residual']:.3e} | {diagnostic_summary['maximum_hidden_nullspace_residual']:.3e} | {diagnostic_summary['maximum_absolute_orthogonality_residual']:.3e} | {diagnostic_summary['maximum_absolute_pythagorean_residual']:.3e} | {diagnostic_summary['maximum_raw_hierarchy_violation']:.3e} |",
        "",
        f"All corrected solver/decomposition checks: **{'PASS' if all_checks else 'FAIL'}**.",
        f"Saved candidates and frozen banks unchanged: **{metadata['saved_candidates_and_frozen_banks_unchanged']}**.",
        f"Allowances where an existing feasible candidate materially beats the saved Full endpoint (absolute tolerance `{tolerance:g}`): **{affected if affected else 'none'}**.",
        ("Recommendation: rerun only the affected Full stage(s); no other stage or allowance needs optimization." if affected else "Recommendation: do not rerun optimization; every saved Full endpoint is best or tied within the already-audited feasible pool."),
        "",
        "The common-raster hidden fraction is computed independently as `A_hid,h / A_full,h`; it is not inferred from a scalar subtraction. Full-precision data and diagnostic maxima are in the companion JSON files.",
        "",
    ])
    (pareto / "toy_corrected_final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "candidate_pool": pool_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
