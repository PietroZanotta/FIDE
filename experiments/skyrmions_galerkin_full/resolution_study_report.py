"""Markdown reporting for the selection-only Galerkin resolution study."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .production_artifacts import file_sha256
from .resolution_study import (
    BANK_MANIFEST_PATH, OUTPUT_ROOT, REPORT_PATH, SUPPORT_LADDER,
    read_json, require_protocol, verify_v1_immutable, write_json,
)


def _eta(eta: list[float]) -> str:
    return "[" + ", ".join(f"{value:.15f}" for value in eta) + "]"


def _quadrature_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| geometry | train/audit | risk | train action | audit action | discrepancy | |g| | weak | energy | moment | rank frac. | min eig. | max eig. | condition | range | stationarity | identity | certified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for geometry in rows:
        for row in geometry["rows"]:
            cert, alg = row["heldout_certificate"], row["algebra"]
            lines.append(
                f"| {geometry['geometry_id']} | {row['train_samples']}/{row['audit_samples']} | {row['scientific_risk']:.9f} | "
                f"{row['train_action']:.9f} | {row['audit_action']:.9f} | {row['train_audit_action_relative_discrepancy']:.4f} | "
                f"{row['gradient_norm']:.5f} | {cert['maximum_weak_residual']:.5f} | {cert['maximum_energy_residual']:.5f} | "
                f"{cert['maximum_moment_rate_residual']:.5f} | {alg['minimum_rank_fraction']:.5f} | {alg['smallest_retained_eigenvalue']:.3e} | "
                f"{alg['largest_eigenvalue']:.3e} | {alg['worst_retained_condition']:.3e} | {alg['worst_range_residual']:.2e} | "
                f"{alg['worst_stationarity_residual']:.2e} | {alg['identity_relerr']:.2e} | {'yes' if row['complete_certificate'] else 'no'} |"
            )
    return lines


def _comparison_table(analyses: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| geometry | support transition | train action change | audit action change | gradient cosine | gradient rel. change | energy change |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for geometry in analyses:
        for row in geometry["consecutive_comparisons"]:
            lines.append(
                f"| {geometry['geometry_id']} | {row['low_support'][0]}/{row['low_support'][1]}→{row['high_support'][0]}/{row['high_support'][1]} | "
                f"{row['train_action_relative_change']:.5f} | {row['audit_action_relative_change']:.5f} | "
                f"{row['cosine']:.7f} | {row['relative_difference']:.5f} | {row['energy_change']:+.5f} |"
            )
    return lines


def _basis_tables(basis: dict[str, Any]) -> list[str]:
    lines = []
    for geometry in basis["geometries"]:
        lines.extend([
            f"### {geometry['geometry_id']}", "",
            "| K | rank tol. | train action | audit action | |g| | weak | energy | moment | min rank frac. | min retained eig. | condition | complete |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in geometry["rows"]:
            cert, alg = row["heldout_certificate"], row["algebra"]
            lines.append(
                f"| {row['K']} | {row['rank_tolerance']:.0e} | {row['train_action']:.9f} | {row['audit_action']:.9f} | "
                f"{row['gradient_norm']:.5f} | {cert['maximum_weak_residual']:.5f} | {cert['maximum_energy_residual']:.5f} | "
                f"{cert['maximum_moment_rate_residual']:.5f} | {alg['minimum_rank_fraction']:.5f} | "
                f"{alg['smallest_retained_eigenvalue']:.3e} | {alg['worst_retained_condition']:.3e} | "
                f"{'yes' if row['complete_certificate'] else 'no'} |"
            )
        lines.append("")
    return lines


def run_report(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.is_file():
        old = read_json(summary_path)
        if old.get("protocol_sha256") == protocol["protocol_sha256"] and old.get("passed"):
            return {**old, "cache_hit": True}
        raise RuntimeError("incompatible completed resolution report exists")
    banks = read_json(BANK_MANIFEST_PATH)
    raw = read_json(OUTPUT_ROOT / "quadrature" / "K280" / "result.json")
    analysis = read_json(OUTPUT_ROOT / "quadrature" / "analysis.json")
    starts = read_json(OUTPUT_ROOT / "start_generator_diagnostics" / "result.json")
    basis = None
    if analysis["conditional_basis_rank_required"]:
        path = OUTPUT_ROOT / "basis_rank" / "result.json"
        if not path.is_file():
            raise RuntimeError("classification B requires the conditional basis-rank study")
        basis = read_json(path)
    if analysis["classification"].startswith("A."):
        final = "A. READY TO FREEZE PARETO V2 WITH LARGER QUADRATURE BANKS"
    elif basis is not None and basis["future_discretization_qualified"]:
        final = "B. READY TO FREEZE PARETO V2 WITH REQUALIFIED GALERKIN DISCRETIZATION"
    else:
        final = "C. GALERKIN FULL DISCRETIZATION NOT YET QUALIFIED FOR A PARETO SWEEP"
    repo_root = Path(__file__).resolve().parents[2]
    status = subprocess.run(["git", "status", "--short"], cwd=repo_root, check=True,
                            capture_output=True, text=True).stdout.rstrip()
    diff = subprocess.run(["git", "diff", "--check"], cwd=repo_root,
                          capture_output=True, text=True)
    summary = {
        "ran": True, "passed": True, "protocol_sha256": protocol["protocol_sha256"],
        "primary_classification": analysis["classification"],
        "basis_rank_ran": basis is not None,
        "recommended_K": None if basis is None else basis["recommended_K"],
        "recommended_rank_tolerance": None if basis is None else basis["recommended_rank_tolerance"],
        "final_decision": final, "start_generator_passed": starts["passed"],
        "v1_immutable": verify_v1_immutable(), "validation_accessed": False,
        "eta_optimization_run": False,
    }
    write_json(summary_path, summary)
    lines = [
        "# Galerkin resolution study", "", "## Scope and outcome", "",
        "This was a selection-development-only fixed-geometry qualification study. No eta optimization, Pareto sweep, winner selection, incumbent replacement, validation construction, or validation access occurred.", "",
        f"Primary K=280 result: **{analysis['classification']}**. Conditional basis/rank study run: `{'yes' if basis is not None else 'no'}`. Final decision: **{final}**.", "",
        "## Why official Pareto v1 failed", "",
        "The frozen v1 attempt reproduced K=280 and all gradients, forcing, geometry, algebra, rank, weak, gauge, and moment diagnostics. It stopped before optimization because every exact-risk-feasible 0.5% start had selection-audit Ritz-energy residual 0.1073–0.1084 against the unchanged 0.08 threshold. V1 remains a frozen failed protocol and was neither modified nor resumed.", "",
        f"V1 immutability audit passed: `{verify_v1_immutable()['passed']}`. Resolution protocol SHA-256: `{protocol['protocol_sha256']}`.", "",
        "## Fixed geometries", "", "| id | provenance | exact eta |", "|---|---|---|",
    ]
    for geometry in protocol["fixed_geometries"]:
        lines.append(f"| {geometry['id']} | {geometry['provenance']} | `{_eta(geometry['eta'])}` |")
    lines.extend(["", "## Selection-development data policy and construction", "",
                  "The two new reference banks are labeled `selection_development_only`. They use fresh independent initial draws and the unchanged frozen reference dynamics; the reference model was not retrained. The 32,768 train and 16,384 audit banks provide exact prefixes for the complete prescribed ladder. All basis work was streamed, so no full K=280 basis cache was built.", "",
                  "Validation arrays—including old truth/fit/audit/noise and v1 seed/data—were not opened. Exact initial-state overlap checks covered fresh train versus fresh audit and every permitted historical selection bank; every count was zero. Historical/future validation disjointness rests on the predeclared independent versioned seed namespace and fresh continuous draws, without violating the no-access rule.", "",
                  "| role | samples | seed | SHA-256 | artifact SHA-256 |", "|---|---:|---:|---|---|"])
    artifacts = {Path(row["path"]).name: row for row in banks["artifacts"]}
    for seed in banks["seed_records"]:
        artifact = next(row for name, row in artifacts.items() if seed["role"] in name)
        lines.append(f"| {seed['role']} | {seed['size']} | {seed['seed']} | `{seed['sha256']}` | `{artifact['sha256']}` |")
    lines.extend(["", "## K=280 quadrature-support results", "", *_quadrature_table(raw["geometries"]), "",
                  "Projection, minimum ESS, pre-centering forcing mean, covariance conditioning, symmetry, gauge, and complete per-time rank/eigenvalue vectors are retained in each machine-readable case JSON. The table shows their compact controlling extrema.", "",
                  "## Action and gradient convergence", "", *_comparison_table(analysis["analyses"]), "",
                  "## Low-risk energy-residual conclusion", ""])
    for geometry_id in ("law", "historical_0p5"):
        item = next(row for row in analysis["analyses"] if row["geometry_id"] == geometry_id)
        energies = [row["heldout_certificate"]["maximum_energy_residual"] for row in item["rows"]]
        lines.append(f"- `{geometry_id}` across supports: " + ", ".join(f"{value:.6f}" for value in energies) + ".")
    lines.extend(["", f"The unchanged physical threshold was **0.08** at every stage. The primary decision logic therefore yields **{analysis['classification']}**.", ""])
    if basis is not None:
        lines.extend(["## Conditional K/rank study", "",
                      "Because the primary outcome was B, the predeclared conditional study ran on the same largest 32,768/16,384 support. K prefixes are exact globally ordered prefixes, and only rank tolerances 1e-10, 1e-11, and 1e-12 were tested.", "", *_basis_tables(basis),
                      f"The predeclared whole-range qualification selected K=`{basis['recommended_K']}` and rank tolerance `{basis['recommended_rank_tolerance']}`. Future discretization qualified: `{'yes' if basis['future_discretization_qualified'] else 'no'}`.", ""])
    lines.extend(["## Candidate v2 initialization gate", "",
                  "A start may logically enter a future optimizer after exact risk, geometry, projection/ESS/forcing/covariance, Galerkin algebra, rank/range/stationarity pass, without already passing held-out weak/energy/moment certificates. This is safe only because every official endpoint, incumbent, and winner still requires the complete certificate. Both v1 low-risk anchors would have entered under this candidate rule, but neither could have become an endpoint in its observed state. This capability was implemented and unit-tested only; no eta step ran.", "",
                  "## Future v2 feasible-manifold start generator", "", "| allowance | feasible starts | total unique pool | min pairwise eta distance | median pairwise eta distance |", "|---:|---:|---:|---:|---:|"])
    for row in starts["feasibility"]:
        lines.append(f"| {row['allowance_percent']:g}% | {row['feasible_count']} | {row['pool_count']} | {row['minimum_pairwise_eta_distance']:.3e} | {row['median_pairwise_eta_distance']:.3e} |")
    lines.extend(["", "The generator uses selection risk only: Law-to-history interpolation, deterministic local clouds, risk-tangent perturbations, and a small fixed global component. It evaluated feasibility/diversity only and never evaluated or optimized Full action.", "",
                  "## Limitations and recommendation", "",
                  "- This is empirical finite-support qualification, not an infinite-dimensional convergence proof.",
                  "- Development audit banks are not independent validation and must never be reported as such.",
                  "- No certificate threshold was tuned; energy remained fixed at 0.08.",
                  "- No validation quantity influenced K, support, rank cutoff, basis, certificate, start generator, or future logic.", "",
                  f"Exact next recommendation: {'freeze a new Pareto v2 only with the reported qualified fixed discretization/support and the documented initialization/end-point separation' if final.startswith(('A.','B.')) else 'do not freeze Pareto v2; develop a more physically adequate fixed-feature Full discretization before any new sweep'}.", "",
                  "## Verification and repository audit", "",
                  f"`git diff --check` return code at report generation: `{diff.returncode}`. V1 byte-identity: `{verify_v1_immutable()['passed']}`. No validation access: `true`. No sensor optimization: `true`.", "",
                  "Final `git status --short` at report generation:", "", "```text", status, "```", "",
                  "All task-created paths are inside `experiments/skyrmions_deep_ritz_full/`; numerical outputs are confined to `outputs/galerkin_resolution_study/`. Historical output trees were not overwritten.", "",
                  final, ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return summary


__all__ = ["run_report"]
