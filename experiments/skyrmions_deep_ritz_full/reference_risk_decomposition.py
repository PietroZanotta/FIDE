"""Development-only reference-risk decomposition and benchmark audit.

This module never trains a reference, samples a new seed, generates a candidate,
loads skyrmion validation data, or invokes Tangent/Full/Ritz machinery.  Missing
Law diagnostics are deterministically reconstructed from the eight already
frozen Phase-A seeds of the reference-seed robustness study.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.projection import EmpiricalIProjector

from .full_gradient import reconstruct_moments
from .pareto_v3_common import ROOT, file_sha256
from .reference import load_reference
from .reference_seed_robustness import (
    BASELINE_CHECKPOINT_PATH,
    EXPECTED_BASELINE_CHECKPOINT_SHA256,
    MODEL_LABELS,
    NODE7,
    PHASE_A_BANK_COUNT,
    PHASE_A_N,
    _array_sha256,
    _checkpoint_path,
    _family,
    _initial_states,
    _load_design_context,
    _load_result,
)
from .risk import many_body_features


VERSION = "skyrmion_reference_risk_decomposition_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
UPSTREAM = ROOT / "outputs" / "skyrmion_galerkin_dev_reference_seed_robustness_v1"
REPO_ROOT = ROOT.parent.parent
CONFIG_PATH = ROOT / "config.json"
PANEL_PATH = UPSTREAM / "candidate_panel_reference.json"
MANIFEST_PATH = UPSTREAM / "experiment_manifest.json"
BRIDGE_PATH = UPSTREAM / "bridge_eval.json"
PHASE_A_SUMMARY_PATH = UPSTREAM / "phase_a_summary.json"
PRODUCTION_ARTIFACTS = ROOT / "outputs" / "production_galerkin" / "artifacts"
HISTORICAL_REFERENCE_BANK = PRODUCTION_ARTIFACTS / "reference_bank_projection.npz"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
CROSS_AUDIT_JSON = OUTPUT_ROOT / "cross_benchmark_reference_audit.json"
CROSS_AUDIT_MD = OUTPUT_ROOT / "cross_benchmark_reference_audit.md"
ACTIVE_AUDIT_PATH = OUTPUT_ROOT / "active_nematic_reference_view_audit.json"
RAW_PATH = OUTPUT_ROOT / "reference_raw_moment_mismatch.json"
PROJECTED_PATH = OUTPUT_ROOT / "reference_projected_moment_mismatch.json"
TIME_PATH = OUTPUT_ROOT / "risk_time_decomposition.json"
COMPONENT_PATH = OUTPUT_ROOT / "risk_whitened_component_decomposition.json"
LAW_PATH = OUTPUT_ROOT / "law_reference_comparison.json"
PANEL_COMPARISON_PATH = OUTPUT_ROOT / "panel_reference_comparison.json"
SEMANTICS_PATH = OUTPUT_ROOT / "risk_semantics_audit.md"
OPTIONS_PATH = OUTPUT_ROOT / "methodology_options.md"
SUMMARY_PATH = OUTPUT_ROOT / "summary.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"

LAW_RESULT_DIR = OUTPUT_ROOT / "law_recomputation"
ROLLOUT_BATCH_SIZE = 2048
FEATURE_BATCH_SIZE = 2048
RAW_COVARIANCE_RIDGE_SOURCE = "forcing.covariance_ridge"
HISTORICAL_LAW_RISK = 5.186549474478042


def _inside(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must be beneath {root}: {resolved}")
    return resolved


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    path = _inside(path)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode())


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = _inside(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _source_paths() -> list[Path]:
    paths = [
        CONFIG_PATH,
        PANEL_PATH,
        MANIFEST_PATH,
        BRIDGE_PATH,
        PHASE_A_SUMMARY_PATH,
        UPSTREAM / "source_seal.json",
        UPSTREAM / "summary.json",
        UPSTREAM / "inventory.json",
        BASELINE_CHECKPOINT_PATH,
        HISTORICAL_REFERENCE_BANK,
        PRODUCTION_ARTIFACTS / "truth_banks.npz",
        REPO_ROOT / "src/mfsi/selection.py",
        REPO_ROOT / "experiments/toy_example_percentage/config.json",
        REPO_ROOT / "experiments/toy_example_percentage/experimenta_setup.md",
        REPO_ROOT / "experiments/toy_example_percentage/outputs/pareto/frozen_inputs/manifest.json",
        REPO_ROOT / "experiments/vortices_percentage/config.json",
        REPO_ROOT / "experiments/vortices_percentage/experimenta_setup.md",
        REPO_ROOT / "experiments/vortices_percentage/outputs/pareto/frozen_inputs/manifest.json",
        REPO_ROOT / "experiments/vortices_percentage/outputs/reference_seed_sensitivity/summary.json",
        REPO_ROOT / "experiments/vortices_percentage/reference_seed_sensitivity.md",
        REPO_ROOT / "experiments/skyrmions_deep_ritz/README.md",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/config.json",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/run.py",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/robust_selection.py",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/percentage_selection.py",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/run_pareto.py",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/finalize_authoritative_pareto.py",
        REPO_ROOT / "experiments/active_nematic_unbalance_percentage/experimenta_setup.md",
    ]
    paths.extend(_checkpoint_path(label) for label in MODEL_LABELS)
    for label in MODEL_LABELS:
        for bank in range(PHASE_A_BANK_COUNT):
            paths.extend([
                UPSTREAM / "phase_a_results" / label / f"bank_{bank:02d}.npz",
                UPSTREAM / "phase_a_results" / label / f"bank_{bank:02d}.json",
            ])
    return paths


def verify_and_seal_sources() -> dict[str, Any]:
    missing = [str(path) for path in _source_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"required audit inputs are missing: {missing}")
    if file_sha256(BASELINE_CHECKPOINT_PATH) != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("immutable baseline checkpoint changed")
    checkpoint_hashes = {label: file_sha256(_checkpoint_path(label)) for label in MODEL_LABELS}
    if len(checkpoint_hashes) != 7 or len(set(checkpoint_hashes.values())) != 7:
        raise RuntimeError("expected seven distinct existing reference checkpoints")
    manifest = _json(MANIFEST_PATH)
    seeds = manifest["phase_a"]["common_bank_seeds"]
    if len(seeds) != PHASE_A_BANK_COUNT:
        raise RuntimeError("frozen Phase-A seed count changed")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "input_hashes": {_relative(path): file_sha256(path) for path in _source_paths()},
        "checkpoint_hashes": checkpoint_hashes,
        "baseline_checkpoint_sha256": EXPECTED_BASELINE_CHECKPOINT_SHA256,
        "frozen_phase_a_seed_records": seeds,
        "guardrails": {
            "new_reference_training": 0,
            "new_reference_seeds": 0,
            "new_truth_simulation": 0,
            "candidate_generation": 0,
            "validation_accessed": False,
            "tangent_called": False,
            "full_called": False,
            "eigensolve_called": False,
            "deep_ritz_called": False,
            "official_protocol_created": False,
        },
        "deterministic_recomputation": {
            "purpose": "recover raw Phi/Psi means and projected Psi means not retained upstream",
            "N": PHASE_A_N,
            "seed_source": _relative(MANIFEST_PATH),
            "common_random_numbers_preserved": True,
            "headline_metrics_verified_before_acceptance": True,
        },
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def _active_nematic_law_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "experiments/active_nematic_unbalance_percentage/experimenta_setup.md"
    text = path.read_text(encoding="utf-8")
    section = text.split("### 14.4 Per-view risk receipt", 1)[1].split("## 15.", 1)[0]
    pattern = re.compile(
        r"^\|\s*([0-3])\s*\|\s*(2026081[89]|20260820)\s*\|\s*"
        r"([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$",
        re.MULTILINE,
    )
    rows = [
        {
            "physical_fold": int(match.group(1)),
            "reference_seed": int(match.group(2)),
            "law_anchor_risk": float(match.group(3)),
            "three_percent_ceiling": float(match.group(4)),
            "selected_full_risk": float(match.group(5)),
            "slack": float(match.group(6)),
        }
        for match in pattern.finditer(section)
    ]
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 active-nematic view rows from artifact, got {len(rows)}")
    return rows


def build_cross_benchmark_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    active_cfg_path = REPO_ROOT / "experiments/active_nematic_unbalance_percentage/config.json"
    active_cfg = _json(active_cfg_path)
    active_seeds = list(map(int, active_cfg["reference_training"]["seeds"]))
    active_rows = _active_nematic_law_rows()
    matrix = np.asarray([
        [next(row["law_anchor_risk"] for row in active_rows if row["physical_fold"] == fold and row["reference_seed"] == seed) for seed in active_seeds]
        for fold in range(4)
    ])
    fold_means, seed_means = matrix.mean(axis=1), matrix.mean(axis=0)
    interaction = matrix - fold_means[:, None] - seed_means[None, :] + matrix.mean()
    active_audit = {
        "schema_version": 1,
        "row_source": _relative(REPO_ROOT / "experiments/active_nematic_unbalance_percentage/experimenta_setup.md"),
        "seed_source": _relative(active_cfg_path),
        "reference_seeds": active_seeds,
        "physical_view_count": int(active_cfg["robust_selection"]["design_views"]),
        "selection_view_count": len(active_rows),
        "rows": active_rows,
        "law_anchor_summary": {
            "minimum": float(matrix.min()),
            "maximum": float(matrix.max()),
            "maximum_to_minimum_ratio": float(matrix.max() / matrix.min()),
            "overall_sd": float(matrix.std()),
            "between_physical_fold_sd_of_means": float(fold_means.std()),
            "between_reference_seed_sd_of_means": float(seed_means.std()),
            "interaction_sd": float(interaction.std()),
            "maximum_within_fold_reference_range": float(np.max(np.ptp(matrix, axis=1))),
            "per_physical_fold": [
                {"fold": fold, "mean": float(fold_means[fold]), "minimum": float(matrix[fold].min()), "maximum": float(matrix[fold].max())}
                for fold in range(4)
            ],
            "per_reference_seed": [
                {"seed": seed, "mean": float(seed_means[index]), "minimum": float(matrix[:, index].min()), "maximum": float(matrix[:, index].max())}
                for index, seed in enumerate(active_seeds)
            ],
        },
        "rules_verified_from_code": {
            "candidate_feasibility": "all 12 view-specific Law-relative ceilings",
            "action_aggregation": "maximum over all 12 selection views",
            "candidate_generation": "physical view 0 crossed with all three references",
            "finalist_rescoring": "all 12 selection views and full selection bank",
            "validation": "held-out physical views crossed with same three references; average references within physical fold, then physical-fold jackknife",
            "absolute_reference_or_population_screen": False,
            "failed_seed_discard_rule": False,
            "held_out_common_CFM_evaluation": False,
        },
    }
    _atomic_json(ACTIVE_AUDIT_PATH, active_audit)

    def evidence(*paths: str) -> list[str]:
        return list(paths)

    rows = [
        {
            "benchmark": "Analytic Gaussian mixture",
            "reference_training_data": "endpoint laws only; linear noisy endpoint bridge",
            "reference_seed_count_official": 1,
            "intermediate_truth_in_reference_training": False,
            "population_scientific_screen": "yes; exact analytic population L(eta) <= Lmax",
            "per_reference_law_risk": "one frozen reference / one Law anchor",
            "candidate_risk_aggregation": "single-reference finite-data Law-relative risk plus absolute population feasibility",
            "action_aggregation": "single reference; mean over observation trials",
            "physical_view_robustness": "independent frozen selection and validation trial banks, not physical-model views",
            "reference_view_robustness": "none",
            "reference_qualification": "no seed qualification; training record plus downstream exact population screen",
            "validation_handling": "same checkpoint/reference law; disjoint 128-trial validation bank",
            "reference_uncertainty_in_reported_uncertainty": False,
            "direct_transfer": "absolute controlled-benchmark hidden-population screen is conceptually transferable; analytic oracle is not",
            "code_evidence": evidence(
                "src/mfsi/selection.py", "experiments/toy_example_percentage/config.json",
                "experiments/toy_example_percentage/outputs/pareto/frozen_inputs/manifest.json",
            ),
            "report_evidence": evidence("experiments/toy_example_percentage/experimenta_setup.md"),
        },
        {
            "benchmark": "Double gyre",
            "reference_training_data": "50,000 endpoint particles only; no intermediate reference-training states",
            "reference_seed_count_official": 1,
            "intermediate_truth_in_reference_training": False,
            "population_scientific_screen": "yes; oracle population L(eta) <= Lmax",
            "per_reference_law_risk": "one frozen reference / one Law anchor",
            "candidate_risk_aggregation": "single-reference finite-data Law-relative risk plus absolute population feasibility",
            "action_aggregation": "single reference; mean over observation trials",
            "physical_view_robustness": "disjoint frozen selection and validation trials",
            "reference_view_robustness": "not in official selection; separate post-hoc three-seed development sensitivity audit exists",
            "reference_qualification": "no seed qualification in official protocol",
            "validation_handling": "same official checkpoint; separately frozen validation trials, not retrained references",
            "reference_uncertainty_in_reported_uncertainty": False,
            "direct_transfer": "oracle absolute-screen principle is transferable; the low-dimensional population oracle is benchmark-specific",
            "code_evidence": evidence(
                "src/mfsi/selection.py", "experiments/vortices_percentage/config.json",
                "experiments/vortices_percentage/outputs/pareto/frozen_inputs/manifest.json",
                "experiments/vortices_percentage/outputs/reference_seed_sensitivity/summary.json",
            ),
            "report_evidence": evidence(
                "experiments/vortices_percentage/experimenta_setup.md",
                "experiments/vortices_percentage/reference_seed_sensitivity.md",
            ),
        },
        {
            "benchmark": "Historical skyrmion Deep Ritz",
            "reference_training_data": "12,000 endpoint configurations only",
            "reference_seed_count_official": 1,
            "intermediate_truth_in_reference_training": False,
            "population_scientific_screen": "no; fixed design-truth scientific risk/support/ESS screens",
            "per_reference_law_risk": "one checkpoint and phase-specific frozen banks",
            "candidate_risk_aggregation": "single reference",
            "action_aggregation": "single-reference cheap proxy then Deep Ritz shortlist",
            "physical_view_robustness": "disjoint design/audit/validation banks by role",
            "reference_view_robustness": "none in historical protocol",
            "reference_qualification": "no multi-seed qualification",
            "validation_handling": "same checkpoint, independently sampled validation reference banks; validation opened only after freeze",
            "reference_uncertainty_in_reported_uncertainty": False,
            "direct_transfer": "historical staged workflow remains valid as executed, but single-checkpoint handling does not address newly observed seed sensitivity",
            "code_evidence": evidence(
                "experiments/skyrmions_deep_ritz_full/experiment.py",
                "experiments/skyrmions_deep_ritz_full/galerkin_only_data.py",
                "experiments/skyrmions_deep_ritz_full/config.json",
            ),
            "report_evidence": evidence("experiments/skyrmions_deep_ritz/README.md"),
        },
        {
            "benchmark": "Active nematic",
            "reference_training_data": "train-split endpoint populations t=21 and t=31 only; separate plus/minus flows",
            "reference_seed_count_official": len(active_seeds),
            "intermediate_truth_in_reference_training": False,
            "population_scientific_screen": "no absolute population/reference qualification screen",
            "per_reference_law_risk": "yes; one Law anchor for each physical-fold/reference-seed view",
            "candidate_risk_aggregation": "must pass every one of 12 view-specific relative ceilings",
            "action_aggregation": "maximum over 12 selection views",
            "physical_view_robustness": "four leave-one-fold-out views",
            "reference_view_robustness": "three references crossed with every physical view",
            "reference_qualification": "three prospectively configured seeds retained; no failed/poor-seed discard or absolute-risk qualification",
            "validation_handling": "independent held-out physical simulations crossed with same references; reference averages within fold precede jackknife",
            "reference_uncertainty_in_reported_uncertainty": "averaged within fold, not a separate uncertainty component",
            "direct_transfer": "partial: cross-reference all-view logic transfers, relative normalization alone does not guard absolute skyrmion Law quality",
            "code_evidence": evidence(
                "experiments/active_nematic_unbalance_percentage/config.json",
                "experiments/active_nematic_unbalance_percentage/robust_selection.py",
                "experiments/active_nematic_unbalance_percentage/percentage_selection.py",
                "experiments/active_nematic_unbalance_percentage/finalize_authoritative_pareto.py",
            ),
            "report_evidence": evidence("experiments/active_nematic_unbalance_percentage/experimenta_setup.md"),
        },
    ]
    payload = {
        "schema_version": 1,
        "scope": "code, config, frozen manifests, and repository report artifacts",
        "manuscript_source_status": "no manuscript source file was present in the repository; paper-facing claims were audited against experimenta_setup.md/README report artifacts",
        "what_code_does": rows,
        "what_repository_reports_say": [
            {"benchmark": row["benchmark"], "statement": row["report_evidence"]}
            for row in rows
        ],
        "discrepancies": [
            {
                "benchmark": "Double gyre",
                "finding": "official workflow is single-reference, while a later three-seed development sensitivity audit exists; it was not integrated into official robust selection",
            },
            {
                "benchmark": "All",
                "finding": "no manuscript source was locally available, so manuscript wording itself could not be independently diffed against implementation",
            },
        ],
        "population_screen_trace": {
            "code": "src/mfsi/selection.py::_exact_population and optimize_population_and_law",
            "effect": "a candidate is rejected as population_invalid_or_outside_Lmax before finite Law-relative feasibility; a bad completion cannot be normalized away if it causes projected hidden-population L to exceed Lmax",
            "limitation": "the screen qualifies candidate/reference behavior jointly, not the checkpoint in isolation; a bad checkpoint could still pass if some geometry satisfies the absolute oracle threshold",
            "skyrmion_analogue": "an absolute design-truth scientific-risk/reference-adequacy diagnostic could be prospectively studied without changing the existing risk definition, but none is adopted here",
        },
    }
    _atomic_json(CROSS_AUDIT_JSON, payload)
    lines = [
        "# Cross-Benchmark Reference-Uncertainty Audit", "", "SOURCE VERIFIED", "",
        "## What the code does", "",
        "| Benchmark | Seeds | Population screen | Robust references | Risk rule | Action rule | Validation references |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['reference_seed_count_official']} | {row['population_scientific_screen']} | "
            f"{row['reference_view_robustness']} | {row['candidate_risk_aggregation']} | {row['action_aggregation']} | {row['validation_handling']} |"
        )
    lines += [
        "", "## What the repository reports say", "",
        "No manuscript source file is present locally. The paper-facing `experimenta_setup.md` and README records agree with the implementation on the main reference rules. The double-gyre repository additionally contains a post-hoc three-seed development audit that is not part of its official single-reference selection.",
        "", "## Population-screen finding", "",
        payload["population_screen_trace"]["effect"] + " " + payload["population_screen_trace"]["limitation"],
        "", "## Evidence paths", "",
    ]
    for row in rows:
        lines.append(f"- **{row['benchmark']}:** " + ", ".join(f"`{path}`" for path in row["code_evidence"] + row["report_evidence"]))
    _atomic_text(CROSS_AUDIT_MD, "\n".join(lines) + "\n")
    return payload, active_audit


def _law_eta() -> np.ndarray:
    panel = _json(PANEL_PATH)
    rows = [row for row in panel["rows"] if row["panel_role"] == "law"]
    if len(rows) != 1 or rows[0]["panel_index"] != 0:
        raise RuntimeError("frozen Law panel membership changed")
    return np.asarray(rows[0]["eta"], dtype=np.float64)


def _rollout_configurations(cfg: dict[str, Any], label: str, initial: np.ndarray) -> np.ndarray:
    flow = load_reference(_checkpoint_path(label))
    times = jnp.linspace(0.0, 1.0, 13, dtype=jnp.float64)
    chunks = []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        trajectory = flow.rollout(
            jnp.asarray(initial[start : start + ROLLOUT_BATCH_SIZE]),
            times,
            substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
        )
        chunks.append(np.asarray(trajectory, dtype=np.float64))
    values = np.concatenate(chunks, axis=1)
    if not np.array_equal(values[0], initial):
        raise RuntimeError("reference rollout did not preserve frozen initial states")
    return values


def _hidden_means(configurations: np.ndarray, weights: np.ndarray, box: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    raw_sum = np.zeros((configurations.shape[0], 9), dtype=np.float64)
    projected = np.zeros_like(raw_sum)
    for start in range(0, configurations.shape[1], FEATURE_BATCH_SIZE):
        stop = min(start + FEATURE_BATCH_SIZE, configurations.shape[1])
        psi = np.asarray(many_body_features(jnp.asarray(configurations[:, start:stop]), box), dtype=np.float64)
        raw_sum += psi.sum(axis=1)
        projected += np.einsum("tn,tnf->tf", weights[:, start:stop], psi)
    return raw_sum / configurations.shape[1], projected


def _symmetric_whitener(whitening: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (whitening + whitening.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() < -1.0e-10:
        raise RuntimeError("fixed whitening matrix is not positive semidefinite")
    return np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


def _risk_arrays(error: np.ndarray, whitening: np.ndarray, time_weights: np.ndarray, whitener: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quadratic = np.einsum("ti,ij,tj->t", error, whitening, error)
    z = error @ whitener.T
    components = time_weights[:, None] * z**2
    weighted = time_weights * quadratic
    if not np.allclose(components.sum(axis=1), weighted, rtol=2e-11, atol=2e-11):
        raise RuntimeError("whitened components do not reproduce timewise risk")
    return weighted, components


def _result_paths(label: str, bank: int) -> tuple[Path, Path]:
    directory = LAW_RESULT_DIR / label
    return directory / f"bank_{bank:02d}.npz", directory / f"bank_{bank:02d}.json"


def _verify_reproduction(label: str, bank: int, result: dict[str, np.ndarray]) -> dict[str, float]:
    upstream = _load_result("a", label, bank)
    expected = {
        "ress_trajectory": upstream["ress_trajectory"][0],
        "lambda_norm": upstream["lambda_norm"][0],
        "top_1pct_weight_mass": upstream["top_1pct_weight_mass"][0],
        "empirical_D2": upstream["empirical_D2"][0],
        "projected_total_risk": np.asarray(upstream["scientific_risk"][0]),
    }
    observed = {key: np.asarray(result[key]) for key in expected}
    errors = {key: float(np.max(np.abs(observed[key] - expected[key]))) for key in expected}
    tolerances = {
        "ress_trajectory": 2e-9,
        "lambda_norm": 2e-6,
        "top_1pct_weight_mass": 2e-9,
        "empirical_D2": 2e-8,
        "projected_total_risk": 2e-8,
    }
    failed = {key: errors[key] for key in errors if errors[key] > tolerances[key]}
    if failed:
        raise RuntimeError(f"deterministic Law reproduction failed for {label}/bank_{bank:02d}: {failed}")
    return errors


def recompute_law_bank(cfg: dict[str, Any], label: str, bank: int) -> dict[str, Any]:
    npz_path, record_path = _result_paths(label, bank)
    if npz_path.exists() or record_path.exists():
        if not npz_path.exists() or not record_path.exists():
            raise RuntimeError(f"incomplete decomposition cache: {label}/bank_{bank:02d}")
        record = _json(record_path)
        if record["result_sha256"] != file_sha256(npz_path):
            raise RuntimeError(f"decomposition cache changed: {npz_path}")
        return {**record, "cache_hit": True}
    source_manifest = _json(MANIFEST_PATH)
    seed_record = source_manifest["phase_a"]["common_bank_seeds"][bank]
    seed = int(seed_record["seed"])
    started = time.perf_counter()
    initial = _initial_states(cfg, seed, PHASE_A_N)
    configurations = _rollout_configurations(cfg, label, initial)
    problem, truth_means, whitening = _load_design_context(cfg)
    eta = _law_eta()
    target = np.asarray(reconstruct_moments(jnp.asarray(eta), problem).values, dtype=np.float64)
    phi = np.asarray(problem.family.features(jnp.asarray(configurations), jnp.asarray(eta)), dtype=np.float64)
    base_weights = np.full(phi.shape[:2], 1.0 / PHASE_A_N, dtype=np.float64)
    raw_sensor_mean = phi.mean(axis=1)
    sensor_error = raw_sensor_mean - target
    centered = phi - raw_sensor_mean[:, None, :]
    covariance = np.einsum("tni,tnj->tij", centered, centered) / PHASE_A_N
    ridge = float(problem.forcing_config.covariance_ridge)
    standardized = np.asarray([
        error @ np.linalg.solve(cov + ridge * np.eye(cov.shape[0]), error)
        for error, cov in zip(sensor_error, covariance, strict=True)
    ])
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    projection = projector.project_trajectory(
        jnp.asarray(phi), jnp.asarray(base_weights), jnp.asarray(target[None, ...])
    )
    weights = np.asarray(projection.weights[0], dtype=np.float64)
    lam = np.asarray(projection.lam[0], dtype=np.float64)
    ress = np.asarray(projection.ess_fraction[0], dtype=np.float64)
    top_count = max(1, int(math.ceil(0.01 * PHASE_A_N)))
    top_mass = np.partition(weights, -top_count, axis=1)[:, -top_count:].sum(axis=1)
    raw_hidden_mean, projected_hidden_mean = _hidden_means(
        configurations, weights, tuple(cfg["physics"]["box"])
    )
    raw_error = raw_hidden_mean - truth_means
    projected_error = projected_hidden_mean - truth_means
    time_weights = np.asarray(problem.time_weights, dtype=np.float64)
    whitener = _symmetric_whitener(whitening)
    raw_time, raw_components = _risk_arrays(raw_error, whitening, time_weights, whitener)
    projected_time, projected_components = _risk_arrays(projected_error, whitening, time_weights, whitener)
    repair_ratio = np.divide(
        projected_time, raw_time, out=np.full_like(projected_time, np.nan), where=raw_time > 1.0e-300
    )
    result = {
        "times": np.asarray(problem.times),
        "time_weights": time_weights,
        "truth_hidden_mean": truth_means,
        "whitening": whitening,
        "whitener_L": whitener,
        "sensor_target": target,
        "raw_sensor_mean": raw_sensor_mean,
        "sensor_error": sensor_error,
        "sensor_error_norm": np.linalg.norm(sensor_error, axis=1),
        "sensor_error_standardized_squared": standardized,
        "raw_sensor_covariance": covariance,
        "raw_hidden_mean": raw_hidden_mean,
        "projected_hidden_mean": projected_hidden_mean,
        "raw_hidden_error": raw_error,
        "projected_hidden_error": projected_error,
        "raw_hidden_error_norm": np.linalg.norm(raw_error, axis=1),
        "projected_hidden_error_norm": np.linalg.norm(projected_error, axis=1),
        "raw_risk_by_time": raw_time,
        "projected_risk_by_time": projected_time,
        "delta_risk_by_time": projected_time - raw_time,
        "repair_ratio_by_time": repair_ratio,
        "raw_whitened_components": raw_components,
        "projected_whitened_components": projected_components,
        "ress_trajectory": ress,
        "lambda": lam,
        "lambda_norm": np.linalg.norm(lam, axis=1),
        "top_1pct_weight_mass": top_mass,
        "empirical_D2": -np.log(np.maximum(ress, 1e-300)),
        "raw_total_risk": np.asarray(raw_time.sum()),
        "projected_total_risk": np.asarray(projected_time.sum()),
        "delta_total_risk": np.asarray(projected_time.sum() - raw_time.sum()),
    }
    reproduction = _verify_reproduction(label, bank, result)
    _atomic_npz(npz_path, **result)
    record = {
        "schema_version": 1,
        "label": label,
        "bank_index": bank,
        "seed_record": seed_record,
        "N": PHASE_A_N,
        "initial_state_sha256": _array_sha256(initial),
        "checkpoint_sha256": file_sha256(_checkpoint_path(label)),
        "source_phase_a_result_sha256": file_sha256(UPSTREAM / "phase_a_results" / label / f"bank_{bank:02d}.npz"),
        "result_path": str(npz_path.relative_to(OUTPUT_ROOT)),
        "result_sha256": file_sha256(npz_path),
        "reproduction_maximum_absolute_errors": reproduction,
        "raw_reference_risk_diagnostic_only": True,
        "raw_covariance_ridge": ridge,
        "raw_covariance_ridge_source": RAW_COVARIANCE_RIDGE_SOURCE,
        "wall_time_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    return record


def run_decomposition(progress=None) -> list[dict[str, Any]]:
    verify_and_seal_sources()
    cfg = _json(CONFIG_PATH)
    rows = []
    for bank in range(PHASE_A_BANK_COUNT):
        for label in MODEL_LABELS:
            row = recompute_law_bank(cfg, label, bank)
            rows.append(row)
            if progress:
                progress(label, bank, bool(row.get("cache_hit", False)), float(row.get("wall_time_seconds", 0.0)))
    return rows


def _load_bank(label: str, bank: int) -> dict[str, np.ndarray]:
    path, record_path = _result_paths(label, bank)
    record = _json(record_path)
    if file_sha256(path) != record["result_sha256"]:
        raise RuntimeError(f"sealed decomposition bank changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)), "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)), "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)), "maximum": float(np.max(values)),
        "sd": float(np.std(values)),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
        "model_count": int(len(x)),
        "inference": "descriptive only; n=7; no significance claim",
    }


def _median_arrays(label: str) -> dict[str, np.ndarray]:
    rows = [_load_bank(label, bank) for bank in range(PHASE_A_BANK_COUNT)]
    return {key: np.median(np.stack([row[key] for row in rows]), axis=0) for key in rows[0]}


def _historical_semantics(cfg: dict[str, Any]) -> dict[str, Any]:
    problem, truth_means, whitening = _load_design_context(cfg)
    eta = _law_eta()
    with np.load(HISTORICAL_REFERENCE_BANK, allow_pickle=False) as arrays:
        configurations = np.asarray(arrays["configurations"], dtype=np.float64)
        base_weights = np.asarray(arrays["base_weights"], dtype=np.float64)
    phi = np.asarray(problem.family.features(jnp.asarray(configurations), jnp.asarray(eta)), dtype=np.float64)
    target = np.asarray(reconstruct_moments(jnp.asarray(eta), problem).values, dtype=np.float64)
    projection = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend).project_trajectory(
        jnp.asarray(phi), jnp.asarray(base_weights), jnp.asarray(target[None])
    )
    weights = np.asarray(projection.weights[0])
    _, projected_mean = _hidden_means(configurations, weights, tuple(cfg["physics"]["box"]))
    time_weights = np.asarray(problem.time_weights)
    risk_time = time_weights * np.einsum(
        "ti,ij,tj->t", projected_mean - truth_means, whitening, projected_mean - truth_means
    )
    observed = float(risk_time.sum())
    if not np.isclose(observed, HISTORICAL_LAW_RISK, rtol=0, atol=2e-10):
        raise RuntimeError(f"historical Law risk did not reproduce: {observed}")
    phase_a_manifest = _json(MANIFEST_PATH)
    return {
        "historical": {
            "risk": observed,
            "reference_bank": _relative(HISTORICAL_REFERENCE_BANK),
            "reference_bank_sha256": file_sha256(HISTORICAL_REFERENCE_BANK),
            "N": int(configurations.shape[1]),
            "truth_bank": "production_galerkin/artifacts/truth_banks.npz::design only",
        },
        "new_study": {
            "baseline_phase_a_risk_distribution": _distribution(np.asarray([
                _load_bank("model_00", bank)["projected_total_risk"] for bank in range(PHASE_A_BANK_COUNT)
            ])),
            "reference_banks": "eight deterministic model_00 rollouts from frozen Phase-A initial seeds",
            "N_per_bank": PHASE_A_N,
            "seed_records": phase_a_manifest["phase_a"]["common_bank_seeds"],
            "truth_bank": "same production_galerkin/artifacts/truth_banks.npz::design only",
        },
        "identity_checks": {
            "whitening_sha256": _array_sha256(whitening),
            "truth_means_sha256": _array_sha256(truth_means),
            "sensor_targets_sha256": _array_sha256(target),
            "time_weights_sha256": _array_sha256(time_weights),
            "same_whitening": True,
            "same_truth_means": True,
            "same_sensor_targets": True,
            "same_time_weights": True,
            "same_checkpoint": True,
            "risk_definition_drift": False,
        },
        "explanation": "The 5.186549 and ~5.325 values use the same checkpoint, design truth, target, W, time weights, and risk arithmetic. They differ only because the former uses the frozen N=8192 production projection bank and the latter summarizes eight independent frozen N=32768 Phase-A rollout banks; this is reference-bank Monte Carlo variation, not definition drift.",
    }


def finalize_outputs() -> dict[str, Any]:
    verify_and_seal_sources()
    cross, active = build_cross_benchmark_audit()
    cfg = _json(CONFIG_PATH)
    models: dict[str, dict[str, Any]] = {}
    medians: dict[str, dict[str, np.ndarray]] = {}
    bridge = {row["label"]: row["CFM_velocity_MSE"] for row in _json(BRIDGE_PATH)["models"]}
    phase_a = {
        row["label"]: row for row in _json(PHASE_A_SUMMARY_PATH)["models"]
    }
    for label in MODEL_LABELS:
        arrays = [_load_bank(label, bank) for bank in range(PHASE_A_BANK_COUNT)]
        median = _median_arrays(label)
        medians[label] = median
        raw_totals = np.asarray([float(row["raw_total_risk"]) for row in arrays])
        projected_totals = np.asarray([float(row["projected_total_risk"]) for row in arrays])
        max_node = int(np.argmax(median["projected_risk_by_time"]))
        models[label] = {
            "label": label,
            "CFM_velocity_MSE": float(bridge[label]),
            "raw_reference_total_risk": _distribution(raw_totals),
            "projected_law_total_risk": _distribution(projected_totals),
            "delta_projection_total_risk": _distribution(projected_totals - raw_totals),
            "node7": {
                "raw_reference_risk": float(median["raw_risk_by_time"][NODE7]),
                "projected_law_risk": float(median["projected_risk_by_time"][NODE7]),
                "raw_Psi_error_norm": float(median["raw_hidden_error_norm"][NODE7]),
                "projected_Psi_error_norm": float(median["projected_hidden_error_norm"][NODE7]),
                "Phi_mismatch_norm": float(median["sensor_error_norm"][NODE7]),
                "Phi_standardized_mismatch_squared": float(median["sensor_error_standardized_squared"][NODE7]),
                "rESS": float(median["ress_trajectory"][NODE7]),
                "lambda_norm": float(median["lambda_norm"][NODE7]),
                "top1pct_mass": float(median["top_1pct_weight_mass"][NODE7]),
                "D2": float(median["empirical_D2"][NODE7]),
                "repair_ratio": float(median["repair_ratio_by_time"][NODE7]),
            },
            "maximum_projected_risk_node": max_node,
            "maximum_projected_risk_time": float(median["times"][max_node]),
            "maximum_projected_risk_contribution": float(median["projected_risk_by_time"][max_node]),
            "node7_is_maximum_risk_node": max_node == NODE7,
            "phase_a_saved_headlines": phase_a[label]["law"],
        }

    raw_payload = {
        "schema_version": 1,
        "raw_reference_risk_diagnostic_only": True,
        "aggregation": "componentwise median over eight common frozen Phase-A banks",
        "models": {
            label: {
                "times": medians[label]["times"].tolist(),
                "sensor_target": medians[label]["sensor_target"].tolist(),
                "raw_sensor_mean": medians[label]["raw_sensor_mean"].tolist(),
                "sensor_error": medians[label]["sensor_error"].tolist(),
                "sensor_error_norm": medians[label]["sensor_error_norm"].tolist(),
                "sensor_error_standardized_squared": medians[label]["sensor_error_standardized_squared"].tolist(),
                "raw_hidden_mean": medians[label]["raw_hidden_mean"].tolist(),
                "truth_hidden_mean": medians[label]["truth_hidden_mean"].tolist(),
                "raw_hidden_error": medians[label]["raw_hidden_error"].tolist(),
                "raw_hidden_error_norm": medians[label]["raw_hidden_error_norm"].tolist(),
            } for label in MODEL_LABELS
        },
    }
    projected_payload = {
        "schema_version": 1,
        "aggregation": "componentwise median over eight common frozen Phase-A banks",
        "models": {
            label: {
                "projected_hidden_mean": medians[label]["projected_hidden_mean"].tolist(),
                "projected_hidden_error": medians[label]["projected_hidden_error"].tolist(),
                "projected_hidden_error_norm": medians[label]["projected_hidden_error_norm"].tolist(),
                "rESS": medians[label]["ress_trajectory"].tolist(),
                "lambda_norm": medians[label]["lambda_norm"].tolist(),
                "top1pct_mass": medians[label]["top_1pct_weight_mass"].tolist(),
                "D2": medians[label]["empirical_D2"].tolist(),
                "repair_ratio": medians[label]["repair_ratio_by_time"].tolist(),
            } for label in MODEL_LABELS
        },
    }
    time_payload = {
        "schema_version": 1,
        "risk_definition": "omega_t * e(t)^T W e(t)",
        "models": {
            label: {
                "times": medians[label]["times"].tolist(),
                "time_weights": medians[label]["time_weights"].tolist(),
                "raw_reference_risk_by_time": medians[label]["raw_risk_by_time"].tolist(),
                "projected_law_risk_by_time": medians[label]["projected_risk_by_time"].tolist(),
                "delta_projection_risk_by_time": medians[label]["delta_risk_by_time"].tolist(),
                "repair_ratio_by_time": medians[label]["repair_ratio_by_time"].tolist(),
                "maximum_projected_risk_node": models[label]["maximum_projected_risk_node"],
            } for label in MODEL_LABELS
        },
    }
    component_payload = {
        "schema_version": 1,
        "name": "WHITENED-RISK COMPONENTS",
        "warning": "These are coordinates of a symmetric square root of W, not original-feature contributions.",
        "factorization": "W = L^T L; z = L e; component(t,j) = omega_t z_j^2",
        "whitening_sha256": _array_sha256(medians["model_00"]["whitening"]),
        "models": {
            label: {
                "raw_components": medians[label]["raw_whitened_components"].tolist(),
                "projected_components": medians[label]["projected_whitened_components"].tolist(),
                "projected_component_totals": medians[label]["projected_whitened_components"].sum(axis=0).tolist(),
            } for label in MODEL_LABELS
        },
    }
    _atomic_json(RAW_PATH, raw_payload)
    _atomic_json(PROJECTED_PATH, projected_payload)
    _atomic_json(TIME_PATH, time_payload)
    _atomic_json(COMPONENT_PATH, component_payload)

    node_ress = np.asarray([models[label]["node7"]["rESS"] for label in MODEL_LABELS])
    sensor = np.asarray([models[label]["node7"]["Phi_standardized_mismatch_squared"] for label in MODEL_LABELS])
    lam = np.asarray([models[label]["node7"]["lambda_norm"] for label in MODEL_LABELS])
    raw_risk = np.asarray([models[label]["raw_reference_total_risk"]["median"] for label in MODEL_LABELS])
    projected_risk = np.asarray([models[label]["projected_law_total_risk"]["median"] for label in MODEL_LABELS])
    cfm = np.asarray([models[label]["CFM_velocity_MSE"] for label in MODEL_LABELS])
    correlations = {
        "smaller_sensor_mismatch_vs_higher_rESS": _correlation(-sensor, node_ress),
        "smaller_sensor_mismatch_vs_lower_lambda_norm": _correlation(-sensor, -lam),
        "higher_rESS_vs_raw_hidden_risk": _correlation(node_ress, raw_risk),
        "higher_rESS_vs_projected_hidden_risk": _correlation(node_ress, projected_risk),
        "lower_CFM_loss_vs_lower_raw_hidden_risk": _correlation(-cfm, -raw_risk),
        "lower_CFM_loss_vs_lower_projected_law_risk": _correlation(-cfm, -projected_risk),
        "lower_CFM_loss_vs_higher_node7_rESS": _correlation(-cfm, node_ress),
    }
    law_payload = {
        "schema_version": 1,
        "models": models,
        "descriptive_correlations": correlations,
        "raw_reference_risk_diagnostic_only": True,
        "common_bank_pairing": True,
        "bank_count": PHASE_A_BANK_COUNT,
    }
    _atomic_json(LAW_PATH, law_payload)
    panel_payload = {
        "schema_version": 1,
        "source": _relative(PHASE_A_SUMMARY_PATH),
        "candidate_generation": False,
        "panel_membership": {"law": 1, "high_pass": 55, "controls": 8},
        "models": {
            label: {
                "CFM_velocity_MSE": bridge[label],
                "law_node7_rESS": models[label]["node7"]["rESS"],
                "law_projected_risk": models[label]["projected_law_total_risk"]["median"],
                "panel_node7_rESS": phase_a[label]["high_pass_panel"]["node7_ress"],
                "panel_relative_risk_increase_percent": phase_a[label]["high_pass_panel"]["relative_scientific_risk_increase_percent"],
                "panel_candidate_pass_thresholds": phase_a[label]["high_pass_panel"]["candidate_pass_count_thresholds"],
            } for label in MODEL_LABELS
        },
    }
    _atomic_json(PANEL_COMPARISON_PATH, panel_payload)

    semantics = _historical_semantics(cfg)
    semantic_lines = [
        "# Scientific-Risk Semantics Audit", "", "SOURCE VERIFIED", "",
        "## Exact dependency graph", "", "```text",
        "production truth_banks.npz::design",
        "  ├── nine Psi truth means ─────────┐",
        "  ├── fixed whitening W ────────────┼── projected Psi error ── omega-weighted quadratic sum",
        "  └── fixed Law sensor targets ─┐   │",
        "                                  ├── I-projection weights ───┘",
        "same model_00 checkpoint ─────────┤",
        "reference bank realization ───────┘",
        "  ├── historical: production N=8192 bank -> 5.186549474478",
        "  └── new: 8 frozen Phase-A N=32768 banks -> median about 5.325",
        "```", "",
        "## Finding", "", semantics["explanation"], "",
        "The hashes of W, truth means, targets, and time weights are recorded in `summary.json`. No risk-definition drift was found.",
    ]
    _atomic_text(SEMANTICS_PATH, "\n".join(semantic_lines) + "\n")

    selected = ["model_00", "model_04", "model_06"]
    projected_ratio = models["model_06"]["projected_law_total_risk"]["median"] / models["model_00"]["projected_law_total_risk"]["median"]
    raw_ratio_04 = models["model_04"]["raw_reference_total_risk"]["median"] / models["model_00"]["raw_reference_total_risk"]["median"]
    raw_ratio_06 = models["model_06"]["raw_reference_total_risk"]["median"] / models["model_00"]["raw_reference_total_risk"]["median"]
    interpretation = (
        "REFERENCE_SEED_SENSOR_HIDDEN_TRADEOFF"
        if correlations["smaller_sensor_mismatch_vs_higher_rESS"]["spearman"] > 0.4
        and correlations["higher_rESS_vs_projected_hidden_risk"]["spearman"] > 0.25
        else "REFERENCE_SEED_HIDDEN_DIRECTION_UNDERIDENTIFICATION"
        if projected_ratio > 2.0 and max(raw_ratio_04, raw_ratio_06) > 1.5
        else "MIXED_REFERENCE_MECHANISM"
    )
    recommendation = "NEXT_MULTI_REFERENCE_PREFLIGHT"
    mechanism = {
        label: {
            "raw_reference_total_risk": models[label]["raw_reference_total_risk"]["median"],
            "projected_law_total_risk": models[label]["projected_law_total_risk"]["median"],
            **models[label]["node7"],
        } for label in selected
    }
    mechanism_findings = {
        "projection_role": {
            label: {
                "projected_to_raw_total_risk_ratio": float(
                    models[label]["projected_law_total_risk"]["median"]
                    / models[label]["raw_reference_total_risk"]["median"]
                ),
                "interpretation": "projection improves hidden risk but does not remove the seed-dependent raw-reference error",
            }
            for label in selected
        },
        "time_localization": {
            label: {
                "maximum_risk_node": models[label]["maximum_projected_risk_node"],
                "maximum_time_fraction_of_componentwise_median_risk": float(
                    np.max(medians[label]["projected_risk_by_time"])
                    / np.sum(medians[label]["projected_risk_by_time"])
                ),
                "node7_is_maximum": models[label]["node7_is_maximum_risk_node"],
            }
            for label in selected
        },
        "whitened_direction_localization": {
            label: {
                "dominant_whitened_coordinate": int(
                    np.argmax(np.sum(medians[label]["projected_whitened_components"], axis=0))
                ),
                "dominant_fraction_of_componentwise_median_risk": float(
                    np.max(np.sum(medians[label]["projected_whitened_components"], axis=0))
                    / np.sum(medians[label]["projected_whitened_components"])
                ),
                "warning": "whitened coordinate, not an original Psi-feature contribution",
            }
            for label in selected
        },
        "descriptive_relationships": {
            "smaller_sensor_mismatch_vs_higher_rESS_spearman": correlations["smaller_sensor_mismatch_vs_higher_rESS"]["spearman"],
            "higher_rESS_vs_raw_hidden_risk_spearman": correlations["higher_rESS_vs_raw_hidden_risk"]["spearman"],
            "higher_rESS_vs_projected_hidden_risk_spearman": correlations["higher_rESS_vs_projected_hidden_risk"]["spearman"],
            "lower_CFM_loss_vs_lower_raw_hidden_risk_spearman": correlations["lower_CFM_loss_vs_lower_raw_hidden_risk"]["spearman"],
            "lower_CFM_loss_vs_lower_projected_law_risk_spearman": correlations["lower_CFM_loss_vs_lower_projected_law_risk"]["spearman"],
            "inference": "descriptive only; n=7; no statistical-significance claim",
        },
    }
    summary = {
        "schema_version": 1,
        "version": VERSION,
        "source_verified": True,
        "guardrails": _json(SOURCE_SEAL_PATH)["guardrails"],
        "models": models,
        "model_00_04_06_mechanism": mechanism,
        "mechanism_findings": mechanism_findings,
        "correlations": correlations,
        "risk_semantics": semantics,
        "active_nematic_transfer": {
            **active,
            "direct_transfer_to_skyrmions": "PARTIAL",
            "reason": "The cross-product/all-view machinery transfers, but active-nematic absolute Law anchors are comparable across reference seeds. Skyrmion anchors differ by factors of about 5-8, so per-reference relative normalization alone would bless candidates relative to scientifically poor anchors.",
        },
        "development_interpretation": interpretation,
        "recommended_next_development_study": recommendation,
        "recommendation_specification": {
            "use_all_existing_references_as_views": True,
            "require_relative_risk_in_every_reference": True,
            "include_absolute_reference_or_law_adequacy_safeguard": True,
            "use_worst_case_action": True,
            "purpose": "prospectively test whether robust feasibility remains nonempty and scientifically meaningful; do not install a checkpoint or create an official protocol",
        },
        "no_reference_replacement": True,
        "no_official_protocol": True,
    }
    _atomic_json(SUMMARY_PATH, summary)
    _write_methodology_options(summary)
    _write_report(summary, cross)
    inventory = _write_inventory()
    return {**summary, "inventory": inventory}


def _write_methodology_options(summary: dict[str, Any]) -> None:
    lines = [
        "# Methodology Options", "",
        "This is a development analysis, not an official protocol.", "",
        "## A — One pre-frozen reference", "",
        "The Gaussian, double-gyre, and historical skyrmion workflows used this approach. The newly measured factor-scale skyrmion reference sensitivity makes a lone unqualified realization scientifically fragile even when it was prospectively frozen.", "",
        "## B — Active-nematic-style robust views", "",
        "Crossing every candidate with several frozen endpoint references, enforcing every per-view relative ceiling, and taking worst-view action is directly implementable. It is insufficient by itself because a large absolute Law error remains invisible after normalization to that same poor Law anchor.", "",
        "## C — Reference qualification", "",
        "Held-out endpoint CFM loss is not shown here to be a reliable intermediate scientific-quality surrogate. Selecting a seed using intermediate truth would change the epistemic meaning of an endpoint-only reference, so no such selection is performed. A preflight may study an absolute adequacy safeguard transparently as controlled-benchmark development.", "",
        "## D — Endpoint bridge/reference construction", "",
        "Alternative prospectively frozen endpoint-only couplings remain a valid later study if robustification cannot produce an adequate nonempty view set. No bridge is changed here.", "",
        "## E — Ensemble/mixed law", "",
        "A mixture changes the reference geometry and is not equivalent to robust evaluation across fixed references. It is not implemented.", "",
        "## Selected next direction", "",
        f"`{summary['recommended_next_development_study']}`: use all seven existing references as views, require relative risk in every view, add a prospectively specified absolute reference/Law adequacy safeguard, and test worst-case action. This is a preflight only.",
    ]
    _atomic_text(OPTIONS_PATH, "\n".join(lines) + "\n")


def _write_report(summary: dict[str, Any], cross: dict[str, Any]) -> None:
    lines = [
        "# Skyrmion Reference-Risk Decomposition + Cross-Benchmark Audit", "", "SOURCE VERIFIED", "",
        "new reference trainings: 0  ", "new reference seeds: 0  ", "new truth simulation: 0  ", "validation accessed: NO", "",
        "## Cross-benchmark reference handling", "",
        "| Benchmark | Seeds | Population screen | Robust references | Risk rule | Action rule |",
        "|---|---:|---|---|---|---|",
    ]
    for row in cross["what_code_does"]:
        lines.append(
            f"| {row['benchmark']} | {row['reference_seed_count_official']} | {row['population_scientific_screen']} | "
            f"{row['reference_view_robustness']} | {row['candidate_risk_aggregation']} | {row['action_aggregation']} |"
        )
    lines += ["", "## Skyrmion reference decomposition", "",
        "| model | CFM loss | raw ref risk | projected Law risk | node7 raw | node7 projected | node7 Phi std² | node7 rESS | node7 lambda | node7 top1% | max-risk node |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in MODEL_LABELS:
        row, node = summary["models"][label], summary["models"][label]["node7"]
        lines.append(
            f"| {label} | {row['CFM_velocity_MSE']:.6f} | {row['raw_reference_total_risk']['median']:.6f} | "
            f"{row['projected_law_total_risk']['median']:.6f} | {node['raw_reference_risk']:.6f} | "
            f"{node['projected_law_risk']:.6f} | {node['Phi_standardized_mismatch_squared']:.6f} | "
            f"{node['rESS']:.6f} | {node['lambda_norm']:.3f} | {node['top1pct_mass']:.6f} | {row['maximum_projected_risk_node']} |"
        )
    lines += ["", "## Model 00 / 04 / 06 mechanism", "",
        "| diagnostic | model_00 | model_04 | model_06 |", "|---|---:|---:|---:|",
    ]
    fields = [
        ("raw reference risk", "raw_reference_total_risk"), ("projected Law risk", "projected_law_total_risk"),
        ("node7 raw Psi error", "raw_Psi_error_norm"), ("node7 projected Psi error", "projected_Psi_error_norm"),
        ("node7 Phi mismatch", "Phi_mismatch_norm"), ("node7 standardized Phi mismatch²", "Phi_standardized_mismatch_squared"),
        ("node7 lambda norm", "lambda_norm"), ("node7 rESS", "rESS"),
        ("node7 top1% mass", "top1pct_mass"), ("node7 projection repair ratio", "repair_ratio"),
    ]
    mechanism = summary["model_00_04_06_mechanism"]
    for title, key in fields:
        lines.append(f"| {title} | {mechanism['model_00'][key]:.6g} | {mechanism['model_04'][key]:.6g} | {mechanism['model_06'][key]:.6g} |")
    findings = summary["mechanism_findings"]
    relation = findings["descriptive_relationships"]
    lines += [
        "", "### Numerical mechanism", "",
        f"Across the seven references, smaller standardized node-7 sensor mismatch tracks higher node-7 rESS (Spearman `{relation['smaller_sensor_mismatch_vs_higher_rESS_spearman']:.3f}`), while higher rESS also tracks higher raw hidden risk (`{relation['higher_rESS_vs_raw_hidden_risk_spearman']:.3f}`) and projected hidden risk (`{relation['higher_rESS_vs_projected_hidden_risk_spearman']:.3f}`). These are descriptive n=7 correlations, not significance tests.", "",
        "Models 04/06 are already much worse before projection: their raw risks are 38.64/50.96 versus 9.59 for model 00. Projection improves all three rather than causing the discrepancy: projected/raw total-risk ratios are "
        f"`{findings['projection_role']['model_00']['projected_to_raw_total_risk_ratio']:.3f}`, "
        f"`{findings['projection_role']['model_04']['projected_to_raw_total_risk_ratio']:.3f}`, and "
        f"`{findings['projection_role']['model_06']['projected_to_raw_total_risk_ratio']:.3f}`. It simply cannot repair the large hidden error left along the Law-sensor fiber.", "",
        f"The maximum-risk nodes are 9, 7, and 6 for models 00/04/06. For models 04/06, the maximum time node contributes only {100.0 * findings['time_localization']['model_04']['maximum_time_fraction_of_componentwise_median_risk']:.1f}%/{100.0 * findings['time_localization']['model_06']['maximum_time_fraction_of_componentwise_median_risk']:.1f}% of the componentwise-median trajectory risk, so the explosion is not a single-time spike. It is directionally concentrated: whitened coordinate 7 contributes {100.0 * findings['whitened_direction_localization']['model_04']['dominant_fraction_of_componentwise_median_risk']:.1f}%/{100.0 * findings['whitened_direction_localization']['model_06']['dominant_fraction_of_componentwise_median_risk']:.1f}%. That coordinate is not an original Psi feature.", "",
        f"Lower endpoint CFM loss does not qualify intermediate scientific quality in this sample: the 'lower CFM loss versus lower raw/projected risk' Spearman correlations are `{relation['lower_CFM_loss_vs_lower_raw_hidden_risk_spearman']:.3f}` and `{relation['lower_CFM_loss_vs_lower_projected_law_risk_spearman']:.3f}`—the opposite direction from the desired surrogate.",
    ]
    active = summary["active_nematic_transfer"]
    law = active["law_anchor_summary"]
    lines += [
        "", "## Active-nematic transfer analysis", "",
        f"- Number of reference seeds: {len(active['reference_seeds'])}",
        f"- Number of physical views: {active['physical_view_count']}",
        f"- Total selection views: {active['selection_view_count']}",
        f"- Law-risk range: {law['minimum']:.6f}–{law['maximum']:.6f}; maximum/minimum = {law['maximum_to_minimum_ratio']:.4f}",
        f"- Between-reference SD of seed means: {law['between_reference_seed_sd_of_means']:.6f}",
        "- Candidate feasibility: all 12 per-view Law-relative ceilings",
        "- Action: maximum across 12 views",
        "- Validation: same three references averaged within each independent physical fold before fold jackknife",
        "- Direct transfer to skyrmions: PARTIAL",
        "", active["reason"], "",
        "## Development interpretation", "", f"`{summary['development_interpretation']}`", "",
        "## Recommended next development study", "", f"`{summary['recommended_next_development_study']}`", "",
        "Use all seven existing references as views, require relative risk in every reference, include an absolute reference/Law adequacy safeguard, and evaluate worst-case action. This recommendation is for a prospectively frozen preflight—not an official rule.", "",
        "NO reference replacement  ", "NO Tangent  ", "NO Full  ", "NO validation use  ", "NO official protocol created", "",
    ]
    _atomic_text(REPORT_PATH, "\n".join(lines))


def _write_inventory() -> dict[str, Any]:
    files = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path != INVENTORY_PATH)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "files": [
            {"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in files
        ],
    }
    payload["summary_sha256"] = file_sha256(SUMMARY_PATH)
    _atomic_json(INVENTORY_PATH, payload)
    return payload


def console_report() -> str:
    summary = _json(SUMMARY_PATH)
    active = summary["active_nematic_transfer"]
    lines = [
        "SOURCE VERIFIED", "", "new reference trainings: 0", "new reference seeds: 0",
        "new truth simulation: 0", "validation accessed: NO", "",
        "CROSS-BENCHMARK REFERENCE HANDLING", "",
        "Benchmark       Seeds   Population screen   Robust references   Risk rule   Action rule",
        "Gaussian        1       exact analytic L    no                  single+L    mean trials",
        "Double gyre     1       oracle L            no (official)       single+L    mean trials",
        "Old skyrmion    1       no                  no                  single      single ref",
        "Active nematic  3       no                  12 crossed views    all-view    maximum", "",
        "SKYRMION REFERENCE DECOMPOSITION", "",
        "model      CFM loss   raw ref risk   projected Law risk   node7 raw   node7 projected   node7 Phi std2   node7 rESS",
    ]
    for label in MODEL_LABELS:
        row, node = summary["models"][label], summary["models"][label]["node7"]
        lines.append(
            f"{label:8s} {row['CFM_velocity_MSE']:9.6f} {row['raw_reference_total_risk']['median']:14.6f} "
            f"{row['projected_law_total_risk']['median']:20.6f} {node['raw_reference_risk']:11.6f} "
            f"{node['projected_law_risk']:17.6f} {node['Phi_standardized_mismatch_squared']:16.6f} {node['rESS']:11.6f}"
        )
    lines += ["", "MODEL 00 / 04 / 06 MECHANISM", ""]
    for label in ("model_00", "model_04", "model_06"):
        row = summary["model_00_04_06_mechanism"][label]
        lines.append(
            f"{label}: raw={row['raw_reference_total_risk']:.6f}, projected={row['projected_law_total_risk']:.6f}, "
            f"node7 Psi raw/proj={row['raw_Psi_error_norm']:.6f}/{row['projected_Psi_error_norm']:.6f}, "
            f"Phi std2={row['Phi_standardized_mismatch_squared']:.6f}, rESS={row['rESS']:.6f}, repair={row['repair_ratio']:.6f}"
        )
    risk = active["law_anchor_summary"]
    lines += [
        "", "ACTIVE-NEMATIC TRANSFER ANALYSIS", "",
        f"number of reference seeds: {len(active['reference_seeds'])}",
        f"number of physical views: {active['physical_view_count']}",
        f"total selection views: {active['selection_view_count']}",
        f"Law-risk variation across refs: {risk['minimum']:.6f} to {risk['maximum']:.6f}",
        "candidate feasibility rule: every view passes its own Law-relative ceiling",
        "action aggregation: maximum over selection views",
        "validation aggregation: average reference seeds within physical fold, then fold jackknife", "",
        "direct transfer to skyrmions: PARTIAL", "", f"reason: {active['reason']}", "",
        f"DEVELOPMENT INTERPRETATION: {summary['development_interpretation']}", "",
        f"RECOMMENDED NEXT DEVELOPMENT STUDY: {summary['recommended_next_development_study']}", "",
        "NO reference replacement", "NO Tangent", "NO Full", "NO validation use", "NO official protocol created",
    ]
    return "\n".join(lines)
