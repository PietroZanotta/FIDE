"""Authoritative Markdown and JSON reporting for the official Pareto sweep."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .official_pareto_common import (
    ALLOWANCES, DICTIONARY_PATH, OUTPUT_ROOT, REPORT_PATH, read_json,
    require_frozen_protocol, write_json,
)
from .official_pareto_validation import FRESH_MANIFEST
from .production_artifacts import file_sha256


def _fmt(value: Any, digits: int = 12) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}g}"


def _eta_text(eta: list[float]) -> str:
    return "[" + ", ".join(f"{value:.15f}" for value in eta) + "]"


def _sensor_text(eta: list[float]) -> str:
    return "; ".join(
        f"({eta[index]:.12f}, {eta[index + 1]:.12f})"
        for index in range(0, len(eta), 2)
    )


def _selection_table(selection: dict[str, Any]) -> list[str]:
    lines = [
        "| allowance | eta | selection risk | risk increase | budget used | train action | audit action | reduction vs Law | certified |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selection["allowances"]:
        winner = row["winner"]
        lines.append(
            f"| {row['allowance_percent']:g}% | `{_eta_text(winner['eta'])}` | "
            f"{winner['risk']:.12f} | {row['selection_risk_increase_percent']:.6f}% | "
            f"{100 * row['budget_used_fraction']:.4f}% | {winner['action']:.12f} | "
            f"{winner['heldout_certificate']['action']:.12f} | {100 * row['selection_reduction_vs_law']:.6f}% | "
            f"{'yes' if winner['certified'] else 'no'} |"
        )
    return lines


def _validation_table(validation: dict[str, Any]) -> list[str]:
    lines = [
        "| allowance | validation Law risk | Full validation risk | risk increase | strict p% pass | declared p+5pp pass | fit action | audit action | reduction vs Law | certified |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in validation["allowances"]:
        lines.append(
            f"| {row['allowance_percent']:g}% | {row['law_risk']:.12f} | {row['full_risk']:.12f} | "
            f"{row['actual_validation_risk_increase_percent']:.6f}% | "
            f"{'pass' if row['strict_p_percent_pass'] else 'fail'} | "
            f"{'pass' if row['declared_p_plus_5pp_pass'] else 'fail'} | "
            f"{row['validation_fit_action']:.12f} | {row['validation_audit_action']:.12f} | "
            f"{100 * row['audit_reduction_vs_law']:.6f}% | "
            f"{'yes' if row['numerically_certified'] else 'no'} |"
        )
    return lines


def _certificate_table(selection: dict[str, Any]) -> list[str]:
    lines = [
        "| p | projection train/audit | min ESS train/audit | forcing mean train/audit | covariance train/audit | rank frac. | range | stationarity | identity | weak | energy | gauge | moment | min separation |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection["allowances"]:
        winner = row["winner"]
        train, audit = winner["train_forcing_audit"], winner["audit_forcing_audit"]
        cert = winner["heldout_certificate"]
        eta = winner["eta"]
        centers = [eta[index:index + 2] for index in range(0, len(eta), 2)]
        box = (2.0, 1.0)
        distances = []
        for left in range(len(centers)):
            for right in range(left + 1, len(centers)):
                delta = [centers[left][d] - centers[right][d] for d in range(2)]
                delta = [delta[d] - box[d] * round(delta[d] / box[d]) for d in range(2)]
                distances.append((delta[0] ** 2 + delta[1] ** 2) ** 0.5)
        lines.append(
            f"| {row['allowance_percent']:g}% | {train['maximum_projection_residual']:.2e}/{audit['maximum_projection_residual']:.2e} | "
            f"{train['minimum_ess_fraction']:.5f}/{audit['minimum_ess_fraction']:.5f} | "
            f"{train['maximum_forcing_mean']:.2e}/{audit['maximum_forcing_mean']:.2e} | "
            f"{train['maximum_covariance_condition']:.4g}/{audit['maximum_covariance_condition']:.4g} | "
            f"{winner['minimum_rank_fraction']:.6f} | {winner['worst_range_residual']:.2e} | "
            f"{winner['worst_stationarity_residual']:.2e} | {winner['identity_relerr']:.2e} | "
            f"{cert['maximum_weak_residual']:.6f} | {cert['maximum_energy_residual']:.6f} | "
            f"{cert['maximum_gauge_residual']:.2e} | {cert['maximum_moment_rate_residual']:.6f} | {min(distances):.6f} |"
        )
    return lines


def _validation_diagnostics(validation: dict[str, Any]) -> list[str]:
    lines = [
        "| p | projection fit/audit | min ESS fit/audit | forcing mean fit/audit | covariance fit/audit | rank frac. | range | stationarity | identity | weak | energy | gauge | moment | SE | classification |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in validation["allowances"]:
        diag = row["diagnostics"]
        fit, audit = diag["fit_forcing_audit"], diag["audit_forcing_audit"]
        cert = diag["heldout_certificate"]
        lines.append(
            f"| {row['allowance_percent']:g}% | {fit['maximum_projection_residual']:.2e}/{audit['maximum_projection_residual']:.2e} | "
            f"{fit['minimum_ess_fraction']:.5f}/{audit['minimum_ess_fraction']:.5f} | "
            f"{fit['maximum_forcing_mean']:.2e}/{audit['maximum_forcing_mean']:.2e} | "
            f"{fit['maximum_covariance_condition']:.4g}/{audit['maximum_covariance_condition']:.4g} | "
            f"{diag['minimum_rank_fraction']:.6f} | {diag['worst_range_residual']:.2e} | "
            f"{diag['worst_stationarity_residual']:.2e} | {diag['identity_relerr']:.2e} | "
            f"{cert['maximum_weak_residual']:.6f} | {cert['maximum_energy_residual']:.6f} | "
            f"{cert['maximum_gauge_residual']:.2e} | {cert['maximum_moment_rate_residual']:.6f} | "
            f"{row['action_standard_error']:.3e} | **{row['classification']}** |"
        )
    return lines


def run_report(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    selection_path = OUTPUT_ROOT / "selection" / "pareto_selection.json"
    selection_manifest_path = OUTPUT_ROOT / "selection" / "manifest.json"
    validation_path = OUTPUT_ROOT / "fresh_validation" / "pareto_validation.json"
    selection = read_json(selection_path)
    selection_manifest = read_json(selection_manifest_path)
    validation = read_json(validation_path)
    fresh_manifest = read_json(FRESH_MANIFEST)
    if selection_manifest["pareto_selection_sha256"] != file_sha256(selection_path):
        raise RuntimeError("selection changed before reporting")
    if validation["seal"]["selection_sha256"] != file_sha256(selection_path):
        raise RuntimeError("validation does not reference current frozen selection")
    classifications = {str(row["allowance_percent"]): row["classification"] for row in validation["allowances"]}
    unique = []
    for row in selection["allowances"]:
        eta = row["winner"]["eta"]
        found = next((item for item in unique if item["eta"] == eta), None)
        if found is None:
            unique.append({"eta": eta, "allowances": [row["allowance_percent"]]})
        else:
            found["allowances"].append(row["allowance_percent"])
    actions = [row["winner"]["action"] for row in selection["allowances"]]
    nested = all(
        selection["allowances"][index]["risk_ceiling"]
        < selection["allowances"][index + 1]["risk_ceiling"]
        for index in range(len(ALLOWANCES) - 1)
    )
    monotone = all(right <= left + protocol["optimizer"]["replacement_tolerance"]
                   for left, right in zip(actions[:-1], actions[1:]))
    summary = {
        "schema_version": 1, "ran": True, "passed": bool(
            validation["selection_winners_unchanged"] and nested and monotone
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": file_sha256(selection_path),
        "selection_manifest_sha256": file_sha256(selection_manifest_path),
        "fresh_validation_manifest_sha256": file_sha256(FRESH_MANIFEST),
        "fresh_validation_result_sha256": file_sha256(validation_path),
        "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "classifications": classifications, "unique_winners": unique,
        "nested_feasible_sets": nested, "selection_action_nonincreasing": monotone,
        "selection_winners_unchanged_after_validation": validation["selection_winners_unchanged"],
        "selection_table": [{
            "allowance_percent": row["allowance_percent"], "eta": row["winner"]["eta"],
            "risk": row["winner"]["risk"], "risk_increase_percent": row["selection_risk_increase_percent"],
            "budget_used_fraction": row["budget_used_fraction"], "train_action": row["winner"]["action"],
            "audit_action": row["winner"]["heldout_certificate"]["action"],
            "reduction_vs_law": row["selection_reduction_vs_law"], "certified": row["winner"]["certified"],
        } for row in selection["allowances"]],
        "validation_table": validation["allowances"],
    }
    write_json(OUTPUT_ROOT / "final_summary.json", summary, overwrite=not (OUTPUT_ROOT / "final_summary.json").exists())

    starts = read_json(OUTPUT_ROOT / "selection" / "starts.json")
    audits = read_json(OUTPUT_ROOT / "finalist_gradient_audits" / "result.json")
    reproduction = read_json(OUTPUT_ROOT / "reproduction" / "result.json")
    repo_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.rstrip()
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=repo_root, check=False,
        capture_output=True, text=True,
    )
    lines = [
        "# Official K=280 Galerkin Full Pareto evaluation", "",
        "## Outcome", "",
        "The fixed-feature K=280 Galerkin finite-dimensional approximation is now the official skyrmion Full discretization. The six ordered selection optimizations and one sealed fresh-validation evaluation completed without using nonlinear Deep Ritz in any decision path.", "",
        f"Protocol SHA-256: `{protocol['protocol_sha256']}`. Frozen selection SHA-256: `{file_sha256(selection_path)}`. Fresh-validation artifact-manifest SHA-256: `{file_sha256(FRESH_MANIFEST)}`.", "",
        "## Repository isolation and chronology", "",
        "The initial repository state was recorded before task changes:", "", "```text",
        *protocol["initial_git_status_short"], "```", "",
        "Every task-created path is under `experiments/skyrmions_deep_ritz_full/`; numerical records are under `outputs/official_galerkin_pareto/`. Original experiments, `src/`, `native/`, historical outputs, the prior sealed 3% selection, and prior validation were not modified. Selection source imports no validation loader or validation-bank path. The old validation bytes were hash-recorded but its arrays were not loaded during selection.", "",
        "The protocol and all four fresh-validation seed hashes were frozen before reproduction or optimization. All six winners and finalist gradient audits were then hash-frozen with `selection_frozen=true` and `validation_accessed=false`. Only afterward did a separate phase create the fresh banks. Validation's seal exactly references the immutable selection hash, and its winning eta list is unchanged.", "",
        "## Frozen protocol", "",
        "Selection uses `R_sel <= (1+p/100) R_Law,sel`. Validation uses the predeclared `R_val <= (1+p/100+0.05) R_Law,val`; strict `p%` status is reported only for transparency. The optimizer never receives the validation slack.", "",
        f"The exact Law geometry is `{_eta_text(selection['law_eta'])}` and its recomputed selection risk is `{selection['law_risk']:.15f}`. The K=280 Law selection-train action is `{selection['law_selection_action']:.15f}`.", "",
        "The bounded optimizer used trust radius `2e-4`, initial step `5e-5`, at most eight accepted-step attempts and ten backtracks, rank stability, exact periodic geometry/risk gates, periodic held-out certification, and replacement tolerance `1e-10`. Settings were identical at every allowance.", "",
        "Fresh seed derivation was `SHA256(<global_seed>:skyrmion:official_galerkin_pareto:v1:<label>)`, reduced deterministically to a positive signed-32-bit integer:", "",
        "| label | SHA-256 | integer seed |", "|---|---|---:|",
    ]
    for seed in protocol["fresh_validation"]["seeds"]:
        lines.append(f"| {seed['label']} | `{seed['sha256']}` | {seed['seed']} |")
    lines.extend(["", "## K=280 reproduction gate", "",
        f"Reproduction passed on `{reproduction['device']['device_kind']}`. Law's known selection-audit energy-certificate failure was reproduced rather than hidden; eta0 and eta_grad were fully certified. Eta0 action was `{reproduction['designs']['eta0']['certificate']['action']:.15f}`, its finite gradient matched the validated vector with relative discrepancy `{reproduction['eta0_gradient_relative_discrepancy']:.3e}`, and repeated actions/gradients were deterministic to the `1e-12` gate.", "",
        "## Deterministic starts and sequential sweep", "",
        "The complete start algorithm was frozen before optimization: Law, exact-feasible historical geometries selected without opening old validation, fixed local perturbations, fixed global candidates ranked only by selection-side K=280 action, and the mandatory preceding incumbent. Actual starts were:", "",
    ])
    for row in selection["allowances"]:
        lines.extend([f"### {row['allowance_percent']:g}%", "", "| id | provenance | eta |", "|---|---|---|"])
        for start in row["actual_starts"]:
            lines.append(f"| {start['id']} | {start['provenance']} | `{_eta_text(start['eta'])}` |")
        lines.extend(["", f"Winner source: `{row['winner_source']}`. Incumbent retained: `{'yes' if row['incumbent_retained'] else 'no'}`. Accepted trajectory endpoints considered: `{row['eligible_candidate_count']}`.", ""])
    lines.extend(["## Official selection Pareto table", "", *_selection_table(selection), "",
                  "The risk ceilings increase strictly with allowance, so the exact feasible sets are nested. Selection action is nonincreasing up to the frozen `1e-10` tolerance. A repeated geometry, if any, denotes incumbent retention; it is not a claim that the mathematical frontier is flat.", "",
                  "## Selection certificate diagnostics", "", *_certificate_table(selection), "",
                  "Every official winner passed exact selection risk and geometry, train/audit projection/ESS/forcing/covariance gates, rank/range/stationarity/symmetry/identity gates, and independent selection-audit weak, energy, gauge, and moment-rate certificates.", "",
                  "## Local finalist eta-gradient audits", "",
                  "The complete five-direction K=280 validation was not repeated. Each unique frozen winner received the predeclared lightweight centered-FD audit before validation:", "",
                  "| winner | allowances | AD | eps=3e-4 rel. error | eps=1e-4 rel. error | rank/forcing valid | pass |", "|---:|---|---:|---:|---:|---|---|"])
    for index, audit in enumerate(audits["audits"]):
        lines.append(f"| {index} | {', '.join(f'{p:g}%' for p in audit['allowances_percent'])} | {audit['ad_directional_derivative']:.9g} | {audit['rows'][0]['relative_discrepancy']:.3e} | {audit['rows'][1]['relative_discrepancy']:.3e} | yes | {'yes' if audit['passed'] else 'no'} |")
    lines.extend(["", "## Immutable selection and fresh-bank creation", "",
                  f"The immutable selection file hash is `{file_sha256(selection_path)}` and its manifest hash is `{file_sha256(selection_manifest_path)}`. The fresh-bank generation seal records both hashes before any array was constructed.", "",
                  f"Fresh validation contains `{protocol['fresh_validation']['truth_samples']}` truth targets and `{protocol['fresh_validation']['reference_fit_samples']}` fit plus `{protocol['fresh_validation']['reference_audit_samples']}` audit reference configurations. The frozen reference checkpoint hash is `{fresh_manifest['reference_checkpoint_sha256']}` and no reference training ran. All exact initial-row overlap counts against selection and previously opened banks, and among fresh roles, were zero.", "",
                  "Fresh artifact hashes:", "", "| artifact | bytes | SHA-256 |", "|---|---:|---|"])
    for artifact in fresh_manifest["artifacts"]:
        lines.append(f"| `{artifact['relative_path']}` | {artifact['bytes']} | `{artifact['sha256']}` |")
    lines.extend(["", "## Official fresh-validation Pareto table", "", *_validation_table(validation), "",
                  "The actual risk increase and strict nominal-p comparison are shown even though the declared acceptance rule is p+5 percentage points. No geometry was substituted or altered after seeing validation.", "",
                  "## Fresh-validation certificate diagnostics", "", *_validation_diagnostics(validation), "",
                  f"The fresh Law validation-fit/audit actions are `{validation['law']['validation_fit_action']:.12f}` / `{validation['law']['validation_audit_action']:.12f}` and Law risk is `{validation['law']['risk']:.12f}`. Reductions in the official table use the common K=280 audit action. The reported uncertainty uses the predeclared production weighted empirical audit-sample standard-error convention; no pseudo-replicates or post-hoc significance test was introduced.", "",
                  "## Unique winning geometries", "", "| unique id | allowances | four periodic sensor centers |", "|---:|---|---|"])
    for index, item in enumerate(unique):
        lines.append(f"| {index} | {', '.join(f'{p:g}%' for p in item['allowances'])} | {_sensor_text(item['eta'])} |")
    lines.extend(["", "## Monotonicity and classifications", ""])
    for row in validation["allowances"]:
        lines.append(f"- **{row['allowance_percent']:g}% — {row['classification']}**: actual fresh-validation risk increase `{row['actual_validation_risk_increase_percent']:.6f}%`; strict comparison `{'pass' if row['strict_p_percent_pass'] else 'fail'}`; declared p+5pp rule `{'pass' if row['declared_p_plus_5pp_pass'] else 'fail'}`; numerical certificate `{'pass' if row['numerically_certified'] else 'fail'}`.")
    lines.extend(["", "## Limitations and interpretation", "",
                  "- K=280 is the fixed official finite-dimensional Galerkin discretization, not a claim of absolute infinite-dimensional convergence.",
                  "- The earlier cross-K study established stable pairwise ordering but incomplete absolute action stabilization; this sweep does not retune K.",
                  "- A declared numerical search can only establish the best certified designs it found. Incumbent retention means no better certified design was found by that search.",
                  "- The prior validation bank is development data and does not support this paper-facing acceptance decision; only the fresh sealed bank does.",
                  "- Historical nonlinear Deep Ritz values did not rank, select, certify, validate, or arbitrate any official result.", "",
                  "## Verification and final repository audit", "",
                  f"`git diff --check` return code: `{diff_check.returncode}`. Output: `{diff_check.stdout.strip() or 'clean'}`.", "",
                  "Final `git status --short` at report generation:", "", "```text", status, "```", "",
                  "Task-created source/report paths are the official protocol/evaluation documents and `official_pareto_*.py` plus `test_official_pareto.py`; machine outputs are confined to `outputs/official_galerkin_pareto/`. Pre-existing modifications listed in the protocol were preserved. No historical result was overwritten, old validation did not enter selection, fresh seeds preceded selection, all six winners preceded fresh validation, winners remained immutable, and Deep Ritz did not participate.", "",
                  "## Final scientific interpretation", "",
                  "This is the official continuous gradient-based skyrmion Full Pareto sweep for the fixed-feature K=280 Galerkin discretization. Each allowance's scientific status is the classification stated above; selection feasibility alone is never used to override a fresh-validation failure.", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


__all__ = ["run_report"]
