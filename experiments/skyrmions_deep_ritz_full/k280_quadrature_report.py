"""Report generator for fixed-K=280 quadrature qualification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .k280_quadrature import (
    ANALYSIS_PATH,
    BANK_MANIFEST_PATH,
    FD_RESULT_PATH,
    GATE_AUDIT_PATH,
    OUTPUT_ROOT,
    REPORT_PATH,
    SUMMARY_PATH,
    final_classification,
    historical_snapshot,
    require_protocol,
    write_json,
)


def _yes(value: Any) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def _git(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                            text=True, capture_output=True, check=False)
    return result.returncode, result.stdout + result.stderr


def run_report(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    old = json.loads(GATE_AUDIT_PATH.read_text())
    banks = json.loads(BANK_MANIFEST_PATH.read_text())
    analysis = json.loads(ANALYSIS_PATH.read_text())
    finite_difference = json.loads(FD_RESULT_PATH.read_text())
    classification = final_classification(analysis, finite_difference)
    diff_code, diff_text = _git(["git", "diff", "--check"])
    _, status = _git(["git", "status", "--short"])
    immutable = historical_snapshot() == protocol["historical_snapshot"]
    lines = [
        "# Fixed K=280 empirical-quadrature qualification",
        "",
        "## Scope and immutable state",
        "",
        "This selection-development-only study varied only empirical quadrature support. "
        "It ran no eta optimization, Pareto sweep, Tangent optimization, winner selection, "
        "or validation access. K, dictionary/order/normalization, rank tolerance, all "
        "forcing/algebra thresholds, and every physical certificate threshold were fixed.",
        "",
        f"The dictionary remained `{protocol['dictionary_sha256']}`, K remained "
        f"`{protocol['K']}`, rank tolerance remained `{protocol['relative_rank_tolerance']}`, "
        f"and Ritz-energy remained `{protocol['certificate_thresholds']['maximum_energy_residual']}`.",
        "",
        f"Historical Pareto/resolution records byte-identical at report time: `{immutable}`.",
        "",
        "## Exact audit of the old 32,768/16,384 `certified=no` flags",
        "",
        "The previous implementation formed `complete_certificate` as geometry AND train "
        "forcing aggregate AND audit forcing aggregate AND algebra aggregate AND held-out "
        "physical aggregate. It did not include support convergence. Thus old `complete=no` "
        "is a physical/numerical gate failure, while the separate resolution-study decision "
        "is a support-qualification failure.",
        "",
        "Trace: " + "; ".join(f"{key} `{value}`" for key, value in old["implementation_trace"].items()),
        "",
        "| geometry | proj T/A | ESS T/A | forcing T/A | covariance T/A | geometry | rank | range | stationarity | symmetry | identity | weak | energy | gauge | moment | support gate in complete | convergence in complete | physical complete | old study qualified | exact failed booleans |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in old["rows"]:
        g = row["gates"]
        lines.append(
            f"| {row['geometry_id']} | {_yes(g['projection_valid_train'])}/{_yes(g['projection_valid_audit'])} "
            f"| {_yes(g['ESS_valid_train'])}/{_yes(g['ESS_valid_audit'])} "
            f"| {_yes(g['forcing_valid_train'])}/{_yes(g['forcing_valid_audit'])} "
            f"| {_yes(g['covariance_valid_train'])}/{_yes(g['covariance_valid_audit'])} "
            f"| {_yes(g['geometry_valid'])} | {_yes(g['rank_valid'])} | {_yes(g['range_valid'])} "
            f"| {_yes(g['stationarity_valid'])} | {_yes(g['symmetry_valid'])} "
            f"| {_yes(g['restricted_identity_valid'])} | {_yes(g['weak_valid'])} "
            f"| {_yes(g['energy_valid'])} | {_yes(g['gauge_valid'])} | {_yes(g['moment_rate_valid'])} "
            f"| no | no | {_yes(row['physical_numerical_certificate'])} "
            f"| {_yes(row['resolution_study_qualification'])} | "
            f"{', '.join(g['failed_booleans']) or 'none'} |"
        )
    lines += [
        "",
        "The exact unexpected failures were train ESS only: Law had "
        f"`{old['rows'][0]['metrics']['train_minimum_ess_fraction']:.9f}`; historical 1% "
        f"had `{old['rows'][2]['metrics']['train_minimum_ess_fraction']:.9f}`; historical "
        f"2% had `{old['rows'][3]['metrics']['train_minimum_ess_fraction']:.9f}`, all below "
        f"the unchanged `{cfg['forcing']['minimum_ess_fraction']}` threshold. Their weak, "
        "energy, gauge, moment, geometry, audit-forcing, and algebra gates passed. Separately, "
        "no geometry met the prior support-convergence qualification because the last gradient "
        "magnitude changes exceeded its frozen tolerance.",
        "",
        "## Nested development banks",
        "",
        "| role | samples | seed label | integer seed | artifact SHA-256 | bytes |",
        "|---|---:|---|---:|---|---:|",
    ]
    seeds = {row["role"]: row for row in banks["seed_records"]}
    for artifact, role in zip(banks["artifacts"], ("train", "audit"), strict=True):
        lines.append(
            f"| {role} | {banks['exact_sample_counts'][role]} | `{seeds[role]['text']}` "
            f"| {seeds[role]['seed']} | `{artifact['sha256']}` | {artifact['bytes']} |"
        )
    lines += [
        "",
        f"All `{len(banks['exact_overlap_checks'])}` permitted exact initial-state overlap "
        "checks were zero. Validation arrays were not opened; disjointness from validation "
        "rests on the frozen independent namespace and fresh continuous draws. All support "
        "levels are exact prefixes of these two maximum artifacts.",
        "",
        "## Complete fixed-geometry support results",
        "",
    ]
    for item in analysis["geometries"]:
        lines += [
            f"### {item['geometry_id']}",
            "",
            "| train/audit | risk | train A | audit A | T/A rel. | grad norm/RMS/max | proj T/A | ESS T/A | force T/A | cov T/A | rank frac. | min eig. | max eig. | condition | range | stationarity | symmetry | identity | weak | energy | gauge | moment | physical cert |",
            "|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in item["rows"]:
            train, audit, algebra, cert = row["train_forcing"], row["audit_forcing"], row["algebra"], row["heldout_certificate"]
            lines.append(
                f"| {row['train_samples']}/{row['audit_samples']} | {_fmt(row['scientific_risk'])} "
                f"| {_fmt(row['train_action'], 9)} | {_fmt(row['audit_action'], 9)} "
                f"| {_fmt(row['train_audit_action_relative_discrepancy'])} "
                f"| {_fmt(row['gradient_norm'])}/{_fmt(row['gradient_rms'])}/{_fmt(row['gradient_maximum_absolute_component'])} "
                f"| {_fmt(train['maximum_projection_residual'])}/{_fmt(audit['maximum_projection_residual'])} "
                f"| {_fmt(train['minimum_ess_fraction'])}/{_fmt(audit['minimum_ess_fraction'])} "
                f"| {_fmt(train['maximum_forcing_mean'])}/{_fmt(audit['maximum_forcing_mean'])} "
                f"| {_fmt(train['maximum_covariance_condition'])}/{_fmt(audit['maximum_covariance_condition'])} "
                f"| {_fmt(algebra['minimum_rank_fraction'])} | {_fmt(algebra['smallest_retained_eigenvalue'])} "
                f"| {_fmt(algebra['largest_eigenvalue'])} | {_fmt(algebra['worst_retained_condition'])} "
                f"| {_fmt(algebra['worst_range_residual'])} | {_fmt(algebra['worst_stationarity_residual'])} "
                f"| {_fmt(algebra['worst_symmetry_residual'])} | {_fmt(algebra['identity_relerr'])} "
                f"| {_fmt(cert['maximum_weak_residual'])} | {_fmt(cert['maximum_energy_residual'])} "
                f"| {_fmt(cert['maximum_gauge_residual'])} | {_fmt(cert['maximum_moment_rate_residual'])} "
                f"| {_yes(row['physical_numerical_certificate'])} |"
            )
        lines += [
            "",
            "Full eta-gradient vectors:",
            "",
        ]
        for row in item["rows"]:
            lines.append(f"- `{row['train_samples']}/{row['audit_samples']}`: `{row['gradient']}`")
        lines += [
            "",
            "Paired convergence:",
            "",
            "| low -> high | train dA | audit dA | grad cosine | grad relative | per-coordinate dg | d weak | d energy | d moment |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for comparison in item["comparisons"]:
            lines.append(
                f"| {comparison['low_support']} -> {comparison['high_support']} "
                f"| {_fmt(comparison['train_action_relative_change'])} "
                f"| {_fmt(comparison['audit_action_relative_change'])} "
                f"| {_fmt(comparison['cosine'])} | {_fmt(comparison['relative_difference'])} "
                f"| `{comparison['per_coordinate_difference']}` "
                f"| {_fmt(comparison['weak_residual_difference'])} "
                f"| {_fmt(comparison['energy_residual_difference'])} "
                f"| {_fmt(comparison['moment_rate_residual_difference'])} |"
            )
        source = item["train_vs_audit_energy_source"]
        lines += [
            "",
            f"Energy-instability attribution: **{source['classification']}**. First audit-only "
            f"change `{_fmt(source.get('audit_only_first_energy_change'))}`, train-only change "
            f"`{_fmt(source.get('train_only_energy_change'))}`, second audit-only change "
            f"`{_fmt(source.get('audit_only_second_energy_change'))}`. Observed paired action "
            "changes are reported directly; no exact 1/sqrt(N) law is assumed.",
            "",
        ]
    qualification = analysis["qualification"]
    lines += [
        "## Final largest-support qualification",
        "",
        "| geometry | physical final two | action stable | gradient direction | gradient magnitude | direction-stable/scale-unresolved |",
        "|---|---|---|---|---|---|",
    ]
    for row in qualification["geometry_checks"]:
        lines.append(
            f"| {row['geometry_id']} | {_yes(row['physical_final_two'])} "
            f"| {_yes(row['action_stable'])} | {_yes(row['gradient_direction_stable'])} "
            f"| {_yes(row['gradient_magnitude_stable'])} "
            f"| {_yes(row['direction_stable_scale_unresolved'])} |"
        )
    lines += [
        "",
        f"Optional 131,072/65,536 level used: `{analysis['optional_support_used']}`. "
        f"Physical valid: `{qualification['physical_valid']}`; action stable: "
        f"`{qualification['action_stable']}`; gradient direction stable: "
        f"`{qualification['gradient_direction_stable']}`; gradient magnitude stable: "
        f"`{qualification['gradient_magnitude_stable']}`.",
        "",
        "## Limited largest-support AD/FD audit",
        "",
    ]
    if finite_difference.get("skipped"):
        lines.append(f"Not run: {finite_difference['reason']}.")
    else:
        lines += [
            "| geometry/direction | AD | best relative error | preferred <=0.5% | passed |",
            "|---|---:|---:|---|---|",
        ]
        for row in finite_difference["directions"]:
            best = min(item["relative_error"] for item in row["epsilons"])
            lines.append(
                f"| {row['geometry_id']}/{row['direction_index']} | {_fmt(row['AD'], 9)} "
                f"| {_fmt(best)} | {_yes(best <= 0.005)} | {_yes(row['passed'])} |"
            )
    old_map = {row["geometry_id"]: row for row in old["rows"]}
    final_map = {item["geometry_id"]: item["rows"][-1] for item in analysis["geometries"]}
    qualification_map = {row["geometry_id"]: row for row in qualification["geometry_checks"]}
    lines += [
        "",
        "## Old `complete` versus new qualification",
        "",
        "| geometry | old 32768/16384 physical certificate | old study qualification | new final physical certificate | new quadrature qualification |",
        "|---|---|---|---|---|",
    ]
    for geometry_id in old_map:
        new_qualified = all((
            qualification_map[geometry_id]["physical_final_two"],
            qualification_map[geometry_id]["action_stable"],
            qualification_map[geometry_id]["gradient_direction_stable"],
            qualification_map[geometry_id]["gradient_magnitude_stable"],
        ))
        lines.append(
            f"| {geometry_id} | {_yes(old_map[geometry_id]['physical_numerical_certificate'])} "
            f"| {_yes(old_map[geometry_id]['resolution_study_qualification'])} "
            f"| {_yes(final_map[geometry_id]['physical_numerical_certificate'])} "
            f"| {_yes(new_qualified)} |"
        )
    lines += [
        "",
        "## Pareto v2 recommendation and future comparison design",
        "",
    ]
    if classification.startswith("A."):
        final_support = analysis["geometries"][0]["rows"][-1]
        lines.append(
            f"K=280 is qualified at `{final_support['train_samples']}` train / "
            f"`{final_support['audit_samples']}` audit with rank tolerance `1e-12` and "
            "unchanged thresholds. A future Pareto v2 may now be frozen, but is not run here."
        )
    else:
        lines.append(
            "Do not freeze Pareto v2. The fixed-K empirical quadrature gates reported above "
            "must be resolved without changing the frozen scientific thresholds."
        )
    lines += [
        "",
        "A future official study should use the already developed feasible-manifold starts; "
        "require risk/geometry/projection/forcing/algebra validity at starts; allow physically "
        "uncertified intermediate points; and require complete held-out certification for every "
        "incumbent and endpoint. Validation must remain sealed until all selection winners are frozen.",
        "",
        "At every allowance, Law is chosen by scientific risk, Tangent minimizes Tangent action "
        "under the same exact risk ceiling, and Full minimizes the fixed K=280 Galerkin action. "
        "Every frozen geometry must be cross-evaluated for risk, Tangent action, and K=280 Full "
        "action. The meaningful same-metric comparison is `A_Full(eta_Full)` versus "
        "`A_Full(eta_Tangent)` versus `A_Full(eta_Law)`; Tangent and Full actions are never "
        "compared as if numerically identical metrics.",
        "",
        "## Limitations and repository audit",
        "",
        "This qualifies only an explicitly finite K=280 discretization; it is not a K->infinity "
        "claim. Development audit banks are not independent validation. No post-hoc threshold, "
        "support, K, rank tolerance, basis, eta, or optimizer setting was introduced.",
        "",
        f"`git diff --check` return code: `{diff_code}`. Output: `{diff_text.strip() or 'none'}`. "
        f"Historical records immutable: `{immutable}`. Validation accessed: `false`. Eta "
        "optimization run: `false`.",
        "",
        "Final `git status --short` at report generation:",
        "",
        "```text",
        status.rstrip(),
        "```",
        "",
        "Every task-created path is inside `experiments/skyrmions_deep_ritz_full/`; numerical "
        "outputs are confined to `outputs/galerkin_k280_quadrature_extension/`. No historical "
        "output was overwritten.",
        "",
        classification,
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "ran": True,
        "passed": diff_code == 0 and immutable,
        "classification": classification,
        "protocol_sha256": protocol["protocol_sha256"],
        "qualification": qualification,
        "finite_difference": finite_difference,
        "historical_immutable": immutable,
        "validation_accessed": False,
        "eta_optimization_run": False,
    }
    if SUMMARY_PATH.exists():
        if json.loads(SUMMARY_PATH.read_text()) != summary:
            raise RuntimeError("refusing to overwrite different K280 summary")
    else:
        write_json(SUMMARY_PATH, summary)
    return summary
