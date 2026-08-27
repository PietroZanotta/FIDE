"""Authoritative Markdown report for the official Galerkin Pareto-v2 sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pareto_v2_common import (
    ALLOWANCES, OUTPUT_ROOT, REPORT_PATH, atomic_json, read_json, require_protocol,
)
from .production_artifacts import file_sha256


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _f(value: Any) -> str:
    return f"{float(value):.9g}"


def run_report(cfg: dict[str, Any], *, blocker: str | None = None) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    banks = read_json(OUTPUT_ROOT / "banks" / "manifest.json") if (OUTPUT_ROOT / "banks" / "manifest.json").exists() else {}
    screening = read_json(OUTPUT_ROOT / "screening" / "candidate_pool.json") if (OUTPUT_ROOT / "screening" / "candidate_pool.json").exists() else {}
    tangent = read_json(OUTPUT_ROOT / "tangent" / "selection.json") if (OUTPUT_ROOT / "tangent" / "selection.json").exists() else {}
    full = read_json(OUTPUT_ROOT / "full_search" / "selection.json") if (OUTPUT_ROOT / "full_search" / "selection.json").exists() else {}
    cross = read_json(OUTPUT_ROOT / "selection" / "cross_evaluation.json") if (OUTPUT_ROOT / "selection" / "cross_evaluation.json").exists() else {}
    selection_manifest = read_json(OUTPUT_ROOT / "selection" / "selection_manifest.json") if (OUTPUT_ROOT / "selection" / "selection_manifest.json").exists() else {}
    fresh_manifest = read_json(OUTPUT_ROOT / "fresh_validation" / "artifact_manifest.json") if (OUTPUT_ROOT / "fresh_validation" / "artifact_manifest.json").exists() else {}
    validation = read_json(OUTPUT_ROOT / "fresh_validation" / "results.json") if (OUTPUT_ROOT / "fresh_validation" / "results.json").exists() else {}
    performance = read_json(OUTPUT_ROOT / "performance" / "profile.json") if (OUTPUT_ROOT / "performance" / "profile.json").exists() else {}
    selection_rows = []
    for row in cross.get("rows", []):
        selection_rows.append([_f(row["allowance_percent"]), row["selected_by"], _f(row["risk"]),
            _f(100 * row["risk_increase"]), _f(row["budget_used"]), _f(row["tangent_action"]),
            _f(row["full_action"]), "PASS" if (row["tangent_certificate"]["valid"] and
                (row["full_certificate"]["valid"] or row["selected_by"] != "Full")) else "FAIL/diagnostic"])
    validation_rows = []
    for row in validation.get("rows", []):
        selection_match = next((item for item in cross.get("rows", [])
            if item["allowance_percent"] == row["allowance_percent"] and item["selected_by"] == row["selected_by"]), None)
        validation_rows.append([_f(row["allowance_percent"]), row["selected_by"],
            "n/a" if selection_match is None else _f(100 * selection_match["risk_increase"]),
            _f(100 * row["validation_risk_increase"]), row["strict_p_validation_pass"],
            row["p_plus_5pp_validation_pass"], "n/a" if selection_match is None else _f(selection_match["full_action"]),
            _f(row["full_fit_action"]), _f(100 * row["full_reduction_vs_law"]), row["classification"]])
    starts_rows = []
    for allowance in ALLOWANCES:
        for start in screening.get("starts", {}).get(str(float(allowance)).replace(".", "p").removesuffix("p0"), []):
            starts_rows.append([allowance, start["candidate_id"], start["start_role"], _f(start["scientific_selection_risk"]),
                               _f(start["minimum_ess_fraction"]), start["eta"]])
    tangent_rows = [[row["allowance_percent"], _f(row["winner"]["risk"]), _f(row["winner"]["action"]),
                     row["incumbent_retained"], row["winner"]["eta"]]
                    for row in tangent.get("allowances", [])]
    full_rows = [[row["allowance_percent"], _f(row["winner"]["risk"]), _f(row["winner"]["action"]),
                  row["incumbent_retained"], len(row["shortlist"]), row["winner"]["eta"]]
                 for row in full.get("allowances", [])]
    bank_rows = [[label, size, next((seed["seed"] for seed in banks.get("seed_records", []) if seed["label"] == label), "n/a"),
                  next((item["sha256"] for item in banks.get("artifacts", []) if item["path"].startswith(f"banks/{label}_")), "n/a")]
                 for label, size in protocol["banks"]["sizes"].items()]
    optimization_rows = [[item["classification"], item["name"], item.get("measured_speedup", item.get("expected_speedup")),
                          item["memory"], item["numerical_risk"], item["semantics_change"]]
                         for item in performance.get("optimizations", [])]
    status = "BLOCKED" if blocker else ("COMPLETE" if validation.get("passed") else "INCOMPLETE")
    content = f"""# Official Galerkin Pareto v2 Evaluation

Status: **{status}**

## Repository isolation and methodological context

All v2 code, caches, banks, traces, seals, and reports are confined to `experiments/skyrmions_deep_ritz_full/`. Historical reports and output trees were treated as immutable. The original experiment, `src/`, and `native/` were read only. Deep Ritz did not enter any official decision.

The official Full objective is the fixed-feature K=280 finite-dimensional Galerkin approximation of the weighted-Poisson weak problem. It is not claimed to be an infinite-dimensional converged solution. The rank-aware coefficient solve uses relative tolerance 1e-12; the eta gradient is the fixed-coefficient envelope derivative and never differentiates through an eigensolve.

## Frozen v2 protocol

Protocol SHA-256: `{protocol['protocol_sha256']}`. Dictionary SHA-256: `{protocol['constants']['dictionary_sha256']}`. The protocol froze allowances 0.5–5%, K=280, rESS 0.05, held-out energy residual 0.08, every inherited scientific threshold, all bank sizes/seeds, four starts, four accepted-step attempts, 2e-4 trust radius, 5e-5 initial step, fourth-step audits, three finalists, 1e-10 replacement tolerance, and validation sizes/rules before any v2 bank existed.

Selection uses `R <= (1+p/100) R_Law`. Validation uses the predeclared `R <= (1+p/100+0.05) R_Law`, with strict nominal-p status also reported.

Validation seed records were present in the protocol before selection: `{protocol['validation']['seed_records']}`. No validation loader is imported by the selection module, and the fresh generation command requires the immutable selection hash.

## Selection banks

{_table(bank_rows, ['role', 'N', 'seed', 'SHA-256']) if bank_rows else 'Not generated.'}

Pairwise role disjointness: `{banks.get('pairwise_role_disjoint', 'not run')}`. Reference retrained: `{banks.get('reference_retrained', 'not run')}`.

## Law and start screening

Law is the pre-existing frozen `config.envelope.law_eta`, selected solely by the unchanged scientific risk. Selection Law risk: `{screening.get('law_risk', 'not run')}`. The deterministic pool has `{screening.get('pool_count', 'not run')}` candidates. Stage A used N=8192 for risk, geometry, projection, rESS, forcing, and covariance only; Full K/f solve count was `{screening.get('full_Kf_solve_count', 'not run')}`.

{_table(starts_rows, ['allowance %', 'candidate', 'role', 'risk', 'min rESS', 'eta']) if starts_rows else 'Start screening not complete.'}

Starts and intermediate points needed search numerical validity, not held-out physical validity. Every reported finalist/incumbent/winner was required to pass the applicable independent complete certificate.

## Tangent selection

Tangent used the exact four-observable Gram formula and its own continuous objective under the same exact selection risk ceiling. It was independently audited and used mandatory nested incumbents.

{_table(tangent_rows, ['allowance %', 'risk', 'authoritative Tangent action', 'incumbent retained', 'eta']) if tangent_rows else 'Tangent sweep not complete.'}

Repeated incumbents mean only: no better certified design was found by the declared numerical search.

## Full K280 selection

Search used N=32768, with N=16384 audits at every start, fourth accepted step, and endpoint. No more than three endpoints reached N=65536/N=65536 authoritative recomputation per allowance. Projection, forcing, ESS, covariance, rank, range, stationarity, A=-2J, weak, energy, gauge, and moment-rate gates retained their frozen values.

{_table(full_rows, ['allowance %', 'risk', 'authoritative Full action', 'incumbent retained', 'shortlist', 'eta']) if full_rows else 'Full sweep not complete.'}

## Selection Law/Tangent/Full cross-evaluation

Raw Tangent and Full actions are not compared as like quantities. The primary common metric is authoritative K=280 Full action at each selected geometry.

{_table(selection_rows, ['allowance %', 'selected by', 'risk', 'risk increase %', 'budget used', 'Tangent action', 'Full action', 'certificate']) if selection_rows else 'Cross-evaluation not complete.'}

Frozen selection SHA-256: `{selection_manifest.get('pareto_selection_sha256', 'not frozen')}`. Winner geometry hash: `{selection_manifest.get('winner_geometry_hash', 'not frozen')}`. `validation_accessed=false` at selection freeze: `{selection_manifest.get('validation_accessed', 'not frozen')}`.

## Fresh validation

Fresh arrays were generated only after the selection hash existed: `{fresh_manifest.get('generated_after_selection_freeze', 'not generated')}`. Selection/validation disjointness: `{fresh_manifest.get('selection_validation_disjoint', 'not generated')}`. Fresh artifact hashes: `{fresh_manifest.get('artifacts', [])}`. No eta optimization was run during validation and the geometry seal remained unchanged: `{validation.get('selection_geometry_unchanged', 'not run')}`.

{_table(validation_rows, ['allowance %', 'selected by', 'sel risk inc %', 'val risk inc %', 'strict-p', 'p+5pp', 'sel Full', 'val Full', 'Full reduction vs Law %', 'classification']) if validation_rows else 'Fresh validation not complete.'}

Full and Tangent diagnostic payloads in `fresh_validation/results.json` contain action standard errors, exact empirical uncertainty convention, rank/algebra data, weak/energy/gauge/moment-rate tables, projection/rESS/forcing/covariance gates, and all method classifications. No pseudo-replicates were invented.

## Performance and further optimization

Dominant component: `{performance.get('dominant_component', 'not profiled')}`.

{_table(optimization_rows, ['rank', 'optimization', 'speedup', 'memory', 'numerical risk', 'semantics change']) if optimization_rows else 'Performance audit not complete.'}

Answer to “Is there further computational optimization possible?”: {performance.get('answer', 'not yet audited')}

## Limitations and interpretation

The optimizer is a fixed-budget multistart numerical search, so absence of replacement is not proof of a flat continuous frontier or a global optimum. K=280 is an intentionally finite-dimensional Full discretization. Finite validation samples and empirical certificates do not establish asymptotic convergence. Scientific evidence rests on a frozen method, independent selection audits, fresh validation, common-K280 comparisons, and explicit risk control.

## Blocker diagnostics

{blocker or 'None. The complete official workflow finished.'}
"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    summary = {"schema_version": 2, "status": status, "passed": bool(status == "COMPLETE"),
        "protocol_sha256": protocol["protocol_sha256"], "selection_sha256": selection_manifest.get("pareto_selection_sha256"),
        "report_sha256": file_sha256(REPORT_PATH), "blocker": blocker}
    atomic_json(OUTPUT_ROOT / "final_summary.json", summary)
    return summary

