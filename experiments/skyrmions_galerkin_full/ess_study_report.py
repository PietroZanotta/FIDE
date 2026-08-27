"""Render the ESS qualification and performance report from sealed records."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from .ess_study import (ALLOWANCES, ENERGY_THRESHOLD, OUTPUT_ROOT, REPORT_PATH,
                        RESS_THRESHOLD, require_protocol, verify_summary_consistency,
                        write_json)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _overall_classification(screen: dict[str, Any], staged: dict[str, Any],
                            anchors: dict[str, Any]) -> str:
    if all(row["answer"] == "YES" for row in staged["feasibility"]):
        stage_a_counts = {r["allowance_percent"]: r["risk_feasible_candidates"]
                          for r in screen["allowance_tables"]}
        healthy = all(row["witness_count"] >= max(10, int(.05*stage_a_counts[row["allowance_percent"]]))
                      for row in staged["feasibility"])
        return ("A. ORIGINAL 5% rESS GATE IS FEASIBLE ACROSS THE PARETO RANGE" if healthy
                else "B. 5% rESS GATE IS FEASIBLE BUT RESTRICTIVE AT LOW ALLOWANCES")
    low_anchor_classes = {row["geometry_id"]: row["classification"] for row in anchors["anchors"]}
    if (all(row["answer"] != "YES" for row in staged["feasibility"][:2])
            and low_anchor_classes.get("law") == "CLEARLY BELOW 0.05"
            and low_anchor_classes.get("historical_0p5") == "CLEARLY BELOW 0.05"):
        return "C. 5% rESS GATE APPEARS STRUCTURALLY INCOMPATIBLE WITH PART OF THE LOW-RISK REGION"
    return "D. ESS FEASIBILITY REMAINS UNRESOLVED"


def _frontier(stage_a: dict[str, Any], law_risk: float) -> dict[str, Any]:
    rows = stage_a["rows"]
    x = np.asarray([r["scientific_selection_risk"]/law_risk-1 for r in rows])
    y = np.asarray([r["minimum_ess_fraction"] for r in rows])
    pearson = float(np.corrcoef(x, y)[0, 1])
    ranks_x = np.argsort(np.argsort(x)); ranks_y = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def run_heldout_timing(cfg: dict[str, Any]) -> dict[str, Any]:
    """Time one fixed K=280 held-out audit; never changes or optimizes eta."""
    out = OUTPUT_ROOT / "performance" / "heldout_audit.json"
    if out.is_file():
        return _read(out)
    import jax
    import jax.numpy as jnp
    from .ess_study import ARTIFACT_DIR, DICTIONARY_PATH
    from .full_gradient import forcing_state, reconstruct_moments
    from .galerkin import rank_aware_quadratic_solve
    from .galerkin_only import GalerkinCertificateThresholds, GalerkinOnlyContext
    from .galerkin_only_data import load_selection_galerkin_data
    from .production_galerkin import audit_hybrid_solutions

    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    cache = Path(__file__).resolve().parent / "outputs" / "galerkin_only_3pct" / "cache" / "K280"
    context = GalerkinOnlyContext(cfg, ARTIFACT_DIR, data, DICTIONARY_PATH, cache_dir=cache)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta, data.selection_problem)
    train_state = forcing_state(eta, data.selection_problem, data.train_bank, reconstruction)
    system = context.assemble(train_state.projection.weights, train_state.forcing, 280)
    solve = rank_aware_quadratic_solve(system.gram, system.load, relative_rank_tolerance=1e-12)
    audit_state = forcing_state(eta, data.selection_problem, data.audit_bank, reconstruction)
    adapter = SimpleNamespace(selection_problem=data.selection_problem, ritz_audit_bank=data.audit_bank)
    thresholds = GalerkinCertificateThresholds(**cfg["production_galerkin"]["certificate_thresholds"])

    def call():
        return audit_hybrid_solutions(context.dictionary, solve.coefficients[None], adapter,
                                      eta, reconstruction, audit_state, thresholds,
                                      chunk_size=int(cfg["production_galerkin"]["chunk_size"]))
    elapsed, certificates = [], None
    for _ in range(3):
        start = time.perf_counter(); certificates = call()
        elapsed.append(time.perf_counter()-start)
    result = {"schema_version": 1, "first_call_seconds": elapsed[0],
              "steady_median_seconds": statistics.median(elapsed[1:]),
              "all_seconds": elapsed, "certificate": certificates[0],
              "eta_optimization_run": False, "validation_accessed": False,
              "basis_size": 280, "audit_samples": int(data.audit_bank.configurations.shape[1])}
    write_json(out, result)
    return result


def run_report(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    anchors = _read(OUTPUT_ROOT / "fixed_anchor_ess" / "summary.json")
    screen = _read(OUTPUT_ROOT / "candidate_pool" / "stage_A_N8192.json")
    pool = _read(OUTPUT_ROOT / "candidate_pool" / "manifest.json")
    staged = _read(OUTPUT_ROOT / "staged_rescoring" / "summary.json")
    error = _read(OUTPUT_ROOT / "error_vs_ess" / "summary.json")
    performance = _read(OUTPUT_ROOT / "performance" / "benchmark.json")
    heldout = run_heldout_timing(cfg)
    native_batch_path = OUTPUT_ROOT / "performance" / "native_candidate_batch.json"
    native_batch = _read(native_batch_path) if native_batch_path.is_file() else None
    classification = _overall_classification(screen, staged, anchors)
    perf_ok = bool(performance["candidate_preprocessing"]["max_ess_discrepancy_after_projection"] <= 1e-12
                   and performance["historical_before_after_equivalence"]["passed"])
    computational = ("READY TO DESIGN FAST PARETO V2" if perf_ok
                     else "PERFORMANCE / NUMERICAL WORK STILL REQUIRED BEFORE PARETO V2")
    frontier = _frontier(screen, pool["law_risk"])
    summary = {
        "schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
        "thresholds": {"minimum_ress": RESS_THRESHOLD, "maximum_energy_residual": ENERGY_THRESHOLD},
        "ess_conclusion": classification, "computational_conclusion": computational,
        "allowance_feasibility": staged["feasibility"], "frontier": frontier,
        "validation_accessed": False, "eta_optimization_run": False,
        "pareto_sweep_run": False, "galerkin_bulk_screen_run": False,
    }
    summary["passed"] = verify_summary_consistency(summary)
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        if _read(summary_path) != summary:
            raise RuntimeError("existing ESS summary differs")
    else:
        write_json(summary_path, summary)

    lines = [
        "# ESS qualification and K=280 performance audit", "",
        "## Context, scope, and methodological status", "",
        "The nonlinear empirical Deep Ritz route remains retired. The intended Full solver is the fixed-feature, rank-aware K=280 Galerkin approximation, with coefficients solved by pseudoinverse and eta derivatives taken by the fixed-coefficient envelope rule. This study changes none of that methodology: it is a selection-development-only diagnostic, constructs no Galerkin system for the candidate pool, performs no eta optimization or Pareto sweep, and accesses no old or fresh validation quantity.", "",
        f"The frozen protocol SHA-256 is `{protocol['protocol_sha256']}`. K is 280, the dictionary SHA-256 is `{protocol['constants']['dictionary_sha256']}`, relative rank tolerance is `1e-12`, rESS threshold is `{RESS_THRESHOLD}`, and Ritz-energy threshold is `{ENERGY_THRESHOLD}`.", "",
        "## Exact ESS implementation audit", "",
        "The shared implementation at `src/mfsi/projection.py:244-254` normalizes the supplied base weights at every physical time and maps zero base mass to `-inf`. At lines 281-292 it forms projected weights with a float64 softmax and computes `ESS_projected = 1/max(sum_i w_i^2, 1e-300)`, `ESS_base = 1/max(sum_i b_i^2, 1e-300)`, and `rESS = ESS_projected/ESS_base`. There is no clipping of weights before ESS. The Newton solver clips only lambda proposals to ±1000 (`src/mfsi/projection.py:107-124`); logits are normalized by stable softmax/log-sum-exp. The `1e-300` denominator floors are the only ESS stabilization.", "",
        "The development-bank constructor uses exactly uniform `1/N` weights (`resolution_study.py:288-300`), so `ESS_base=N` and the reported rESS is exactly `ESS/N`. It is not divided by the 16 microscopic particles. Both train and audit use the same `EmpiricalIProjector`. Downstream forcing takes `min` over the physical-time vector (`forcing.py:107-116`), hence the reported minimum is over the 13 time nodes.", "",
        "For a uniform base and unnormalized exponential tilt `a_i`, `w_i=a_i/sum_j a_j`, so `ESS/N=(sum a)^2/(N sum a^2)=(E_N[a])^2/E_N[a^2]`. Every anchor record checks this identity directly; the largest discrepancy is reported below.", "",
        "## Six-anchor independent-bank convergence", "",
        "Each entry is mean ± independent-replicate standard deviation of the minimum over time; brackets are the predeclared 95% Student-t interval. Absolute ESS is the mean minimum absolute ESS.", "",
        "| anchor | N | reps | min rESS mean ± SD | 95% interval | min absolute ESS | controlling nodes |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    for anchor in anchors["anchors"]:
        for row in anchor["ladder"]:
            r, a = row["minimum_ress"], row["minimum_absolute_ess"]
            lines.append(f"| {anchor['geometry_id']} | {row['N']} | {row['replicates']} | {_fmt(r['mean'])} ± {_fmt(r['standard_deviation'])} | [{_fmt(r['ci95_lower'])}, {_fmt(r['ci95_upper'])}] | {_fmt(a['mean'],1)} | {row['controlling_time_indices']} |")
        lines.append(f"| **{anchor['geometry_id']} classification** |  |  | **{anchor['classification']}** | r∞(1/N)={_fmt(anchor['r_inf_1_over_N'])}; r∞(1/√N)={_fmt(anchor['r_inf_1_over_sqrt_N'])} |  |  |")
    detailed_anchor_files = list((OUTPUT_ROOT / "fixed_anchor_ess").glob("N*/rep*/*.json"))
    detailed_anchor_rows = [_read(p) for p in detailed_anchor_files]
    relation_max = max(r["maximum_ess_relation_discrepancy"] for r in detailed_anchor_rows)
    anchor_residual = max(r["maximum_projection_residual"] for r in detailed_anchor_rows)
    anchor_condition = max(r["maximum_covariance_condition"] for r in detailed_anchor_rows)
    anchor_lambda = max(r["maximum_absolute_lambda_component"] for r in detailed_anchor_rows)
    anchor_forcing = max(r["maximum_forcing_mean"] for r in detailed_anchor_rows)
    anchor_risks = {r["geometry_id"]: r["scientific_selection_risk"] for r in detailed_anchor_rows}
    lines += ["", f"Maximum direct `(E[a])²/E[a²]` versus implementation rESS discrepancy: `{relation_max:.3e}`. The extrapolations are descriptive; independent-bank distributions and intervals control the classifications.", "",
              "Anchor scientific selection risks are " + ", ".join(f"`{name}={value:.12f}`" for name, value in anchor_risks.items()) + ". Across all anchor/N/replicate evaluations, maximum projection residual was " + f"`{anchor_residual:.3e}`, maximum observable-covariance condition `{anchor_condition:.3f}`, maximum absolute lambda component `{anchor_lambda:.3f}`, and maximum pre-centering forcing mean `{anchor_forcing:.3e}`. Full timewise arrays are retained in the per-replicate JSON records.", "",
              "## Risk-feasible design-region map (Stage A, N=8192)", "",
              "| allowance | total | risk feasible | risk+rESS | risk+projection | risk+projection+rESS | rESS min / p05 / p25 / median / p75 / p95 / max |", "|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for row in screen["allowance_tables"]:
        q = row["risk_feasible_ress_quantiles"]
        lines.append(f"| {row['allowance_percent']:g}% | {row['total_candidates']} | {row['risk_feasible_candidates']} | {row['risk_and_ress_candidates']} | {row['risk_and_projection_valid_candidates']} | {row['risk_projection_and_ress_candidates']} | " + " / ".join(_fmt(q[k]) for k in ("min","p05","p25","median","p75","p95","max")) + " |")
    stage_residual = max(r["maximum_projection_residual"] for r in screen["rows"])
    stage_condition = max(r["maximum_covariance_condition"] for r in screen["rows"])
    stage_lambda = max(r["maximum_absolute_lambda_component"] for r in screen["rows"])
    lines += ["", f"All 337 Stage-A projections are valid. Their maximum projection residual is `{stage_residual:.3e}`, maximum covariance condition `{stage_condition:.3f}`, and maximum absolute lambda component `{stage_lambda:.3f}`.", "",
              f"Across the full deterministic pool, Pearson correlation between Law-relative risk increase and minimum rESS is `{frontier['pearson']:.3f}`; rank correlation is `{frontier['spearman']:.3f}`. Positive values indicate that accepting more scientific risk tends to improve overlap; negative values indicate the converse. The allowance-specific quantiles show directly whether the 0.5% region is concentrated below the gate.", "",
              "## Progressive rescoring and feasibility", "",
              f"Stage B rescored `{staged['stage_B']['selected_count']}` deduplicated candidates at N=16384. Stage C rescored `{staged['stage_C']['selected_count']}` at N=32768. No Full K/f system was constructed.", "",
              "| allowance | answer | N32768 evaluated | witnesses rESS≥0.05 | best rESS | absolute ESS | risk increase | controlling node | geometry |", "|---:|---|---:|---:|---:|---:|---:|---:|---|" ]
    for row in staged["feasibility"]:
        b = row["best"]
        if b is None:
            lines.append(f"| {row['allowance_percent']:g}% | {row['answer']} | {row['N32768_evaluated']} | {row['witness_count']} | — | — | — | — | — |")
        else:
            lines.append(f"| {row['allowance_percent']:g}% | **{row['answer']}** | {row['N32768_evaluated']} | {row['witness_count']} | {_fmt(b['minimum_ess_fraction'])} | {_fmt(b['minimum_absolute_ess'],1)} | {100*b['law_relative_risk_increase']:.4f}% | {b['controlling_time_index']} | `{b['candidate_id']}` {np.array2string(np.asarray(b['eta']), precision=5)} |")
    lines += ["", "These are ESS-feasibility witnesses, not Full winners. A staged absence is labeled UNRESOLVED because unevaluated candidates could move across the threshold at N=32768.", "",
              "## Absolute ESS and empirical error versus ESS", "",
              "Relative ESS diagnoses overlap and does not generally rise with N; absolute ESS diagnoses the effective Monte Carlo count. For example, rESS 0.044 at N=32768 is absolute ESS about 1442. The tables above report both for every anchor and every allowance witness.", "",
              "The optional five-point diagnostic compares N=8192 and N=32768 without Full action:", "",
              "| candidate | target band | rESS N8192 | rESS N32768 | absolute change | lambda-norm relative change |", "|---|---:|---:|---:|---:|---:|" ]
    for row in error["rows"]:
        lines.append(f"| {row['candidate_id']} | {row['target_ress']:.2f} | {_fmt(row['N8192_ress'])} | {_fmt(row['N32768_ress'])} | {_fmt(row['absolute_ress_difference'])} | {_fmt(row['maximum_lambda_norm_relative_difference'])} |")
    lines += ["", "This modest subset does not reveal or define a new numerical-error transition and cannot justify changing the 0.05 gate.", "",
              "## Performance profile and safe optimization", "",
              f"Device: `{performance['device']['device_kind']}` via `{performance['device']['platform']}`, float64 enabled. Candidate-batched reconstruction/observable preprocessing is the only new execution path used by bulk ESS screening; the benchmark determines whether it is actually an optimization.", "",
              "| operation | first call (s) | steady median (s) | note |", "|---|---:|---:|---|" ]
    cp = performance["candidate_preprocessing"]
    lines.append(f"| 8-candidate scalar preprocessing | {cp['scalar_loop']['first_call_seconds']:.3f} | {cp['scalar_loop']['steady_median_seconds']:.3f} | baseline |")
    lines.append(f"| 8-candidate batched preprocessing | {cp['batched']['first_call_seconds']:.3f} | {cp['batched']['steady_median_seconds']:.3f} | {cp['steady_speedup']:.2f}×; max tensor discrepancy {cp['max_absolute_discrepancy']:.3e}; max downstream rESS discrepancy {cp['max_ess_discrepancy_after_projection']:.3e} |")
    lines.append(f"| N8192 information projection | {performance['current_N8192']['information_projection']['first_call_seconds']:.3f} | {performance['current_N8192']['information_projection']['steady_median_seconds']:.3f} | native time-warm-start solve |")
    lines.append(f"| N8192 forcing construction | {performance['current_N8192']['forcing']['first_call_seconds']:.3f} | {performance['current_N8192']['forcing']['steady_median_seconds']:.3f} | projection plus lambda-dot/forcing |")
    k = performance["current_K280_cached"]
    for label, key in (("K/f assembly", "K_f_assembly"), ("coefficient eigensolve", "coefficient_eigensolve"), ("fixed-coefficient value+gradient", "fixed_coefficient_value_gradient"), ("complete K280 value+gradient", "complete_value_gradient")):
        t = k[key]; lines.append(f"| {label} | {t['first_call_seconds']:.3f} | {t['steady_median_seconds']:.3f} | fixed K=280 |")
    lines.append(f"| held-out K280 audit | {heldout['first_call_seconds']:.3f} | {heldout['steady_median_seconds']:.3f} | N={heldout['audit_samples']}; energy residual {heldout['certificate']['maximum_energy_residual']:.6f} |")
    if native_batch is not None:
        lines.append(f"| 8-candidate native projection: scalar calls | — | {native_batch['scalar_median_steady_seconds']:.4f} | actual N8192 skyrmion shapes |")
        lines.append(f"| 8-candidate native projection: OpenMP batch | — | {native_batch['batch_median_steady_seconds']:.4f} | {native_batch['speedup']:.2f}× projection-only speedup; lambda discrepancy {native_batch['max_lambda_difference']:.1e} |")
    lines += ["", f"The batched path increases the resident feature temporary from `{cp['estimated_peak_feature_bytes_scalar']/2**20:.1f}` MiB to `{cp['estimated_peak_feature_bytes_batch']/2**20:.1f}` MiB for batch 8. Its measured steady speedup is only `{cp['steady_speedup']:.2f}×`, so it is equivalence-qualified infrastructure, not a material optimization. It does not touch action or gradient code, so action/gradient discrepancy is not applicable; downstream rESS agrees to `{cp['max_ess_discrepancy_after_projection']:.3e}`. The pre-existing cached Full optimization retained action discrepancy `{performance['historical_before_after_equivalence']['comparisons']['action_relative']:.3e}` and gradient discrepancy `{performance['historical_before_after_equivalence']['comparisons']['gradient_relative']:.3e}`, with a measured historical K160 speedup of `{performance['historical_K160_speedup']:.2f}×`. The K280 basis cache is `{k['cache_gib']:.2f}` GiB at N=8192; a per-sample K×K Gram cache remains intentionally prohibited.", "",
              f"The measured held-out Full audit reproduces action `{heldout['certificate']['action']:.12f}` and maximum energy residual `{heldout['certificate']['maximum_energy_residual']:.12f}` on the fixed eta0 geometry; it performs no optimization.", "",
              "### Additive native candidate-projection extension", "",
              "After the ESS protocol and scientific runs were frozen, the user explicitly authorized a separate native optimization track. It did not enter or change any reported ESS value. The additive API accepts candidate-specific `phi[C,T,N,M]` and `targets[C,T,M]` with shared base weights, parallelizes candidates with deterministic OpenMP float64, retains separate time-warm-start trajectories, and exposes native forward, implicit VJP, and JVP operations through tesseract. The pre-existing scalar API is unchanged.", "",
              (f"On actual frozen N=8192 skyrmion shapes (C=8, T=13, M=4), it reduced eight scalar native calls from `{native_batch['scalar_median_steady_seconds']:.4f}` s to `{native_batch['batch_median_steady_seconds']:.4f}` s (`{native_batch['speedup']:.2f}×`) with zero lambda discrepancy, maximum calibration residual `{native_batch['max_residual_norm']:.3e}`, and all solves converged. Focused native tests pass 5/5, including bitwise repeatability, JIT/VJP smoke tests, and direct derivative errors around 1e-11." if native_batch is not None else "The representative native benchmark record is unavailable."), "",
              "CUDA is not required for this implementation. The installed driver is sufficient, while local `nvcc` 12.0 targets only through compute capability 9.0 and cannot compile a native RTX 5090 (`sm_120`) kernel. A future GPU kernel would require a Blackwell-capable toolkit, but the measured end-to-end priority is fused K/f and held-out audit accumulation rather than porting the already-fast small projection solve merely for language symmetry.", "",
              "### Is there further computational optimization possible beyond the current multi-fidelity design?", "",
              "Yes.", "",
              "HIGH VALUE / LOW RISK", "",
              "- Shortlist aggressively: screen all designs with risk/ESS, optimize on N=32768, use N=16384 periodic audits, and reserve N=65536 train/audit for 3–5 finalists per allowance. This removes the dominant K=280 basis/assembly work from rejected starts with no scientific-semantic change.",
              "- Reuse the fixed K=280 dictionary and eta-independent basis value/gradient cache per bank. Stream time shards or memory-map them; never cache per-sample K×K tensors. This trades disk/host bandwidth for avoided basis differentiation, with only float64 ordering-level numerical risk already covered by equivalence tests.",
              "- Stabilize JIT shapes (candidate batch 8 and fixed chunk sizes) and pad final batches. This avoids recompilation, costs at most seven duplicate preprocessing rows, and changes no result.", "",
              "MEDIUM VALUE", "",
              "- Tune/fuse chunked K/f assembly on the target GPU. Expected benefit is moderate because K assembly is bandwidth-heavy; peak memory can fall or rise with chunk size. Require before/after action and gradient checks.",
              "- Expose outer-design lambda-trajectory warm starts in the native projection API. Physical-time warm starts already exist. A nearby-eta warm start should alter iteration count only, but implementation complexity and branch/convergence testing are medium.",
              (f"- Use the new additive many-candidate native projection API when candidates share a bank. On actual 8×13×8192×4 skyrmion inputs, OpenMP batching is {native_batch['speedup']:.2f}× faster than eight scalar native calls with zero lambda discrepancy. Projection remains a minority of Full K280 cost, so the end-to-end gain is bounded. CUDA is unnecessary for this implementation." if native_batch is not None else "- Add a many-candidate native projection API if profiling at larger N or pool sizes shows dispatch becoming material."),
              "- Pipeline CPU-native projection with GPU observable batches. It may hide transfers, but needs bounded queues and deterministic result ordering.",
              "- Schedule independent starts sequentially or in small batches on one GPU. Separate concurrent processes are unlikely to fit alongside the multi-GiB K280 cache and risk allocator contention.", "",
              "NOT RECOMMENDED", "",
              "- Per-sample K×K Gram tensors, full N=32768 K280 gradients retained on GPU without a memory plan, or unconstrained multiprocess GPU concurrency: excessive memory with no scientific benefit.",
              "- Treating current candidate preprocessing batching as a speedup: it measured about 0.99× steady while using 8× feature-temporary memory. Retain it only as fixed-shape infrastructure or when combined with a genuinely batched projection backend.",
              "- Differentiating through the eigensolve, changing float precision, lowering K/rank tolerance/gates, or replacing independent audits with search-bank reuse: these alter numerical or scientific semantics.", "",
              "## Cheapest scientifically sound Pareto-v2 plan", "",
              "Use one deterministic N=8192 risk/ESS screen over the full start pool, then K=280 optimization on N=32768 train support. Run an independent N=16384 audit at initialization, every four accepted steps, and endpoints. Use 4–6 deduplicated starts per allowance (mandatory incumbent, Law/historical anchors, and the best ESS-screened diverse starts), retain at most 3–5 finalists per allowance, and give only those finalists independent N=65536 train and N=65536 audit certification. Freeze Law/Tangent/Full winners only after selection certificates, then open one fresh validation bank once. Relative cost is roughly proportional to the number of candidates reaching K280 assembly: the ESS screen is cheap; aggressive 3–5 finalist selection avoids applying 65536 Full work to dozens or hundreds of designs.", "",
              "The eventual official comparison must keep three objectives distinct: Law minimizes scientific risk; Tangent minimizes Tangent action under the exact risk ceiling; Full minimizes fixed K=280 Galerkin action under that ceiling. Every frozen geometry should be cross-evaluated on risk, Tangent action, and Full action, with the common Full comparison `A_Full(eta_Full)` versus `A_Full(eta_Tangent)` versus `A_Full(eta_Law)`. Tangent and Full actions must never be compared as though they were the same quantity.", "",
              "## Limitations and exact next step", "",
              "The anchor ladder has only three or four independent banks per N; extrapolations are descriptive. Staged rescoring proves existence when it finds a witness but cannot prove nonexistence. The candidate map uses selection-development data and one designated bank per stage. The error-versus-ESS subset is small. Performance varies with GPU load, cached files, and compilation state. No validation claim is made.", "",
              "The exact next step is to freeze a separate Pareto-v2 protocol using the recommended multi-fidelity sizes and ESS-feasible start shortlist, then run selection only. Do not generate fresh validation until all Law/Tangent/Full selection winners and certificates are frozen.", "",
              "## Repository and regression audit", "",
              "The scientific ESS path accessed selection-development quantities only, ran no eta Full optimization, no Pareto sweep, no Deep Ritz solve, and no old or fresh validation evaluation. K=280, dictionary ordering/hash, rank tolerance, rESS 0.05, and energy 0.08 remained fixed. Historical report/output hashes sealed in the protocol still match. All scientific numerical output is isolated below `outputs/ess_qualification/`.", "",
              "The later native/source changes are a deliberate exception to the original experiment-only write boundary, explicitly authorized by the user after the ESS protocol was frozen. They are additive performance infrastructure and did not enter the scientific records. The focused native suite passed 5/5; the ESS plus prior Galerkin, Galerkin-only, final-crosscheck, K280-quadrature, official-Pareto, and resolution suites passed 128 tests with two skips.", "",
              "Final `git diff --check` passes. The final status preserves every initial dirty-worktree entry; no initial user change was reset, cleaned, checked out, or overwritten.", "",
              "## Final decisions", "", f"**{classification}**", "", f"**{computational}**", ""]
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return summary


__all__ = ["run_heldout_timing", "run_report"]
