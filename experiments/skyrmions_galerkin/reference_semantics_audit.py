"""Whitening-semantics and endpoint-rollout audit for frozen references.

Development only.  This module consumes sealed derived intermediate-risk tensors
and endpoint arrays only.  It does not train, optimize sensors, access validation
members, compare endpoint rollouts to intermediate truth, or invoke Galerkin,
Tangent, Full, eigensolvers for Full, or Deep Ritz.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .galerkin_only_data import _family
from .pareto_v3_common import ROOT, file_sha256
from .reference import load_reference
from .reference_seed_robustness import (
    BASELINE_CHECKPOINT_PATH,
    EXPECTED_BASELINE_CHECKPOINT_SHA256,
    MODEL_LABELS,
    _array_sha256,
    _checkpoint_path,
)
from .risk import many_body_features


VERSION = "skyrmion_galerkin_dev_reference_semantics_audit_v1"
OUTPUT_ROOT = ROOT / "outputs" / VERSION
REPO_ROOT = ROOT.parent.parent
ROBUST_ROOT = ROOT / "outputs" / "skyrmion_galerkin_dev_reference_seed_robustness_v1"
DECOMP_ROOT = ROOT / "outputs" / "skyrmion_reference_risk_decomposition_v1"
PRODUCTION_ROOT = ROOT / "outputs" / "production_galerkin" / "artifacts"
TRUTH_ARCHIVE = PRODUCTION_ROOT / "truth_banks.npz"
CONFIG_PATH = ROOT / "config.json"

SOURCE_SEAL_PATH = OUTPUT_ROOT / "source_seal.json"
WHITENING_DEFINITION_PATH = OUTPUT_ROOT / "whitening_definition.json"
WHITENING_SPECTRUM_PATH = OUTPUT_ROOT / "whitening_spectrum.json"
COORDINATE7_PATH = OUTPUT_ROOT / "whitened_coordinate7_audit.json"
MODAL_PATH = OUTPUT_ROOT / "metric_eigenmode_decomposition.json"
UNWHITENED_PATH = OUTPUT_ROOT / "unwhitened_psi_errors.json"
ENDPOINT_MANIFEST_PATH = OUTPUT_ROOT / "endpoint_eval_manifest.json"
ENDPOINT_RESULTS_PATH = OUTPUT_ROOT / "endpoint_rollout_results.json"
ENDPOINT_FEATURE_PATH = OUTPUT_ROOT / "endpoint_feature_summary.json"
JOINT_SUMMARY_PATH = OUTPUT_ROOT / "joint_summary.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
INVENTORY_PATH = OUTPUT_ROOT / "inventory.json"
ENDPOINT_CACHE_ROOT = OUTPUT_ROOT / "endpoint_model_cache"

TIME_COUNT = 13
NODE7 = 7
COORDINATE7_INDEX = 7
ENDPOINT_EVAL_N = 8192
ROLLOUT_BATCH_SIZE = 2048
FEATURE_BATCH_SIZE = 2048
RIDGE_RELATIVE = 1.0e-5

PSI_FEATURES = (
    {"index": 0, "name": "pair_distance_gaussian_r0.10", "group": "pair-distance", "definition": "mean ordered-pair Gaussian at distance 0.10, width 0.055"},
    {"index": 1, "name": "pair_distance_gaussian_r0.20", "group": "pair-distance", "definition": "mean ordered-pair Gaussian at distance 0.20, width 0.055"},
    {"index": 2, "name": "pair_distance_gaussian_r0.32", "group": "pair-distance", "definition": "mean ordered-pair Gaussian at distance 0.32, width 0.055"},
    {"index": 3, "name": "pair_distance_gaussian_r0.48", "group": "pair-distance", "definition": "mean ordered-pair Gaussian at distance 0.48, width 0.055"},
    {"index": 4, "name": "structure_factor_k(pi,0)", "group": "structure-factor", "definition": "squared Fourier amplitude / n^2 at k=(pi,0)"},
    {"index": 5, "name": "structure_factor_k(0,2pi)", "group": "structure-factor", "definition": "squared Fourier amplitude / n^2 at k=(0,2pi)"},
    {"index": 6, "name": "structure_factor_k(pi,2pi)", "group": "structure-factor", "definition": "squared Fourier amplitude / n^2 at k=(pi,2pi)"},
    {"index": 7, "name": "structure_factor_k(2pi,0)", "group": "structure-factor", "definition": "squared Fourier amplitude / n^2 at k=(2pi,0)"},
    {"index": 8, "name": "mean_local_hexatic_order_magnitude", "group": "hexatic-order", "definition": "mean magnitude of Gaussian-neighbor-weighted local psi6"},
)


def _inside(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output must be beneath {root}: {resolved}")
    return resolved


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = _inside(path)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite sealed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
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


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _distribution(values: Any) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(x)), "p10": float(np.quantile(x, 0.1)),
        "median": float(np.median(x)), "mean": float(np.mean(x)),
        "p90": float(np.quantile(x, 0.9)), "maximum": float(np.max(x)),
        "sd": float(np.std(x)),
    }


def _rank(values: Any) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x: Any, y: Any) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _inventory_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = inventory.get("files", inventory.get("artifacts"))
    if not isinstance(rows, list):
        raise RuntimeError("unrecognized upstream inventory schema")
    return rows


def _verify_inventory(root: Path) -> dict[str, Any]:
    path = root / "inventory.json"
    inventory = _json(path)
    failures = []
    for row in _inventory_rows(inventory):
        target = root / row["path"]
        if not target.exists() or file_sha256(target) != row["sha256"]:
            failures.append(row["path"])
    if failures:
        raise RuntimeError(f"upstream inventory mismatch under {root}: {failures}")
    if "summary_sha256" in inventory:
        if file_sha256(root / "summary.json") != inventory["summary_sha256"]:
            raise RuntimeError(f"upstream summary digest mismatch: {root}")
    return {
        "root": _relative(root),
        "inventory_sha256": file_sha256(path),
        "verified_artifact_count": len(_inventory_rows(inventory)),
    }


def _analysis_sources() -> list[Path]:
    return [
        Path(__file__), ROOT / "reference_semantics_audit_run.py",
        ROOT / "test_reference_semantics_audit.py", ROOT / "risk.py",
        ROOT / "reference.py", ROOT / "measurements.py",
    ]


def verify_and_seal_sources() -> dict[str, Any]:
    robust = _verify_inventory(ROBUST_ROOT)
    decomposition = _verify_inventory(DECOMP_ROOT)
    if file_sha256(BASELINE_CHECKPOINT_PATH) != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise RuntimeError("immutable baseline checkpoint changed")
    checkpoints = {label: file_sha256(_checkpoint_path(label)) for label in MODEL_LABELS}
    robust_seal = _json(ROBUST_ROOT / "source_seal.json")
    expected = robust_seal["immutable_source_hashes"]["baseline_checkpoint"]
    if checkpoints["model_00"] != expected:
        raise RuntimeError("baseline differs from robustness-study source seal")
    for label, digest in checkpoints.items():
        record_path = ROBUST_ROOT / "reference_models" / ("baseline.json" if label == "model_00" else f"{label}/training.json")
        record = _json(record_path)
        if record["checkpoint_sha256"] != digest:
            raise RuntimeError(f"checkpoint manifest mismatch: {label}")
    for path in _analysis_sources():
        if not path.exists():
            raise RuntimeError(f"analysis source missing: {path}")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "development_only": True,
        "upstream_studies": {"reference_seed_robustness": robust, "reference_risk_decomposition": decomposition},
        "upstream_direct_hashes": {
            _relative(path): file_sha256(path)
            for path in (
                ROBUST_ROOT / "summary.json", ROBUST_ROOT / "inventory.json",
                DECOMP_ROOT / "summary.json", DECOMP_ROOT / "inventory.json",
                DECOMP_ROOT / "source_seal.json", TRUTH_ARCHIVE, CONFIG_PATH,
            )
        },
        "checkpoint_hashes": checkpoints,
        "baseline_checkpoint_sha256": EXPECTED_BASELINE_CHECKPOINT_SHA256,
        "analysis_source_hashes": {_relative(path): file_sha256(path) for path in _analysis_sources()},
        "guardrails": {
            "reference_models": 7, "new_training": 0, "new_random_seeds": 0,
            "intermediate_truth_arrays_accessed": False, "validation_accessed": False,
            "sensor_generation": False, "sensor_optimization": False,
            "tangent": False, "full_K_f": False, "full_eigensolve": False,
            "deep_ritz": False, "reference_selection": False,
            "official_protocol_created": False,
        },
    }
    _atomic_json(SOURCE_SEAL_PATH, payload)
    return payload


def _law_bank_path(label: str, bank: int) -> Path:
    return DECOMP_ROOT / "law_recomputation" / label / f"bank_{bank:02d}.npz"


def _load_law_bank(label: str, bank: int) -> dict[str, np.ndarray]:
    path = _law_bank_path(label, bank)
    record = _json(path.with_suffix(".json"))
    if file_sha256(path) != record["result_sha256"]:
        raise RuntimeError(f"sealed decomposition bank changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in arrays.files}


def _metric_definition() -> dict[str, Any]:
    bank = _load_law_bank("model_00", 0)
    raw_metric = np.asarray(bank["whitening"], dtype=np.float64)
    metric = 0.5 * (raw_metric + raw_metric.T)
    factor = np.asarray(bank["whitener_L"], dtype=np.float64)
    if not np.allclose(factor.T @ factor, metric, rtol=2e-12, atol=2e-12):
        raise RuntimeError("stored whitening factor does not reconstruct metric")
    for label in MODEL_LABELS:
        for index in range(8):
            row = _load_law_bank(label, index)
            if not np.array_equal(row["whitening"], raw_metric):
                raise RuntimeError("risk metric differs across sealed banks")
            if not np.array_equal(row["whitener_L"], factor):
                raise RuntimeError("whitening factor differs across sealed banks")
            if not np.array_equal(row["time_weights"], bank["time_weights"]):
                raise RuntimeError("time weights differ across sealed banks")
    regularized_covariance = 0.5 * (np.linalg.inv(raw_metric) + np.linalg.inv(raw_metric).T)
    scale = float(np.trace(regularized_covariance) / 9.0 / (1.0 + RIDGE_RELATIVE))
    ridge_absolute = RIDGE_RELATIVE * scale
    covariance = regularized_covariance - ridge_absolute * np.eye(9)
    if not np.all(np.linalg.eigvalsh(covariance) > 0.0):
        raise RuntimeError("reconstructed pre-regularization covariance is not positive definite")
    return {
        "raw_metric": raw_metric, "metric": metric, "factor": factor, "covariance": covariance,
        "regularized_covariance": regularized_covariance, "scale": scale,
        "ridge_absolute": ridge_absolute, "time_weights": bank["time_weights"],
    }


def _mode_count(fractions: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(np.cumsum(fractions), threshold, side="left") + 1)


def run_whitening_audit() -> dict[str, Any]:
    verify_and_seal_sources()
    definition = _metric_definition()
    metric, factor = definition["metric"], definition["factor"]
    covariance, regularized = definition["covariance"], definition["regularized_covariance"]
    alpha_asc, vectors_asc = np.linalg.eigh(metric)
    order = np.argsort(alpha_asc)[::-1]
    alpha, vectors = alpha_asc[order], vectors_asc[:, order]
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    regularized_eigenvalues = np.linalg.eigvalsh(regularized)
    if not np.allclose(vectors @ np.diag(alpha) @ vectors.T, metric, rtol=2e-12, atol=2e-12):
        raise RuntimeError("invariant metric eigendecomposition failed")
    definition_payload = {
        "schema_version": 1,
        "authoritative_semantics": {
            "repository_variable_named_whitening": "M = inverse(C + ridge*scale*I)",
            "repository_risk_arithmetic": "delta^T M delta",
            "factor_used_for_whitened_coordinates": "W with M = W^T W; z = W delta",
            "important_nomenclature_correction": "The repository object named whitening is the quadratic metric M, not the factor W used in the prompt notation.",
        },
        "construction": {
            "source_data": "sealed design-truth Psi features, 13 times x 6000 configurations, reused only through sealed derived M; no truth array opened here",
            "flattening": "all 78,000 time/configuration rows pooled",
            "centering": "one global nine-vector mean over pooled rows",
            "covariance_denominator": 77999,
            "covariance_definition": "centered.T @ centered / (n_rows - 1)",
            "relative_ridge": RIDGE_RELATIVE,
            "scale_definition": "max(trace(C)/9, 1e-8)",
            "scale": definition["scale"],
            "absolute_diagonal_ridge": definition["ridge_absolute"],
            "eigenvalue_floor": None,
            "normalization": "none beyond sample-covariance denominator and scale-relative ridge",
            "dtype": "float64",
            "stored_inverse_exactly_symmetric": bool(np.array_equal(definition["raw_metric"], definition["raw_metric"].T)),
            "stored_inverse_maximum_antisymmetric_residue": float(np.max(np.abs(definition["raw_metric"] - definition["raw_metric"].T))),
            "quadratic_metric_is_symmetric_part": True,
            "quadratic_metric_exactly_symmetric": bool(np.array_equal(metric, metric.T)),
            "factorization": "np.linalg.eigh(M) ascending; W = diag(sqrt(alpha_ascending)) @ V_ascending.T",
            "serialized_source": "reference-risk decomposition law_recomputation NPZ files",
        },
        "hashes": {
            "stored_inverse_covariance_raw": _array_sha256(definition["raw_metric"]),
            "risk_metric_M_symmetric_part": _array_sha256(metric),
            "whitening_factor_W": _array_sha256(factor),
            "reconstructed_raw_covariance_C": _array_sha256(covariance),
            "regularized_covariance": _array_sha256(regularized),
            "time_weights": _array_sha256(definition["time_weights"]),
            "Psi_source": file_sha256(ROOT / "risk.py"),
            "risk_decomposition_source": file_sha256(DECOMP_ROOT / "source_seal.json"),
        },
        "feature_definitions": list(PSI_FEATURES),
        "time_weights": definition["time_weights"].tolist(),
    }
    spectrum_payload = {
        "schema_version": 1,
        "ordering": "risk modes descending by alpha; covariance arrays ascending",
        "risk_metric_eigenvalues_alpha_descending": alpha.tolist(),
        "inverse_sqrt_alpha_length_scales": (1.0 / np.sqrt(alpha)).tolist(),
        "risk_metric_condition_number": float(alpha[0] / alpha[-1]),
        "raw_covariance_eigenvalues_ascending": covariance_eigenvalues.tolist(),
        "post_regularization_covariance_eigenvalues_ascending": regularized_eigenvalues.tolist(),
        "covariance_condition_number": float(covariance_eigenvalues[-1] / covariance_eigenvalues[0]),
        "regularization_floor_absolute": definition["ridge_absolute"],
        "smallest_raw_covariance_eigenvalue_to_ridge_ratio": float(covariance_eigenvalues[0] / definition["ridge_absolute"]),
        "floor_dominated": bool(covariance_eigenvalues[0] <= 10.0 * definition["ridge_absolute"]),
        "mode_loadings": [
            {
                "mode_rank_zero_based": int(k),
                "metric_eigenvalue": float(alpha[k]),
                "implied_covariance_eigenvalue": float(1.0 / alpha[k] - definition["ridge_absolute"]),
                "original_Psi_loadings": {PSI_FEATURES[j]["name"]: float(vectors[j, k]) for j in range(9)},
                "largest_absolute_loadings": [
                    {"feature": PSI_FEATURES[j]["name"], "loading": float(vectors[j, k])}
                    for j in np.argsort(np.abs(vectors[:, k]))[::-1][:4]
                ],
            }
            for k in range(9)
        ],
        "warning": "covariance eigenvalues and risk-metric eigenvalues are distinct and inverse-related only after the diagonal ridge",
    }
    _atomic_json(WHITENING_DEFINITION_PATH, definition_payload)
    _atomic_json(WHITENING_SPECTRUM_PATH, spectrum_payload)

    coordinate_rows: dict[str, Any] = {}
    modal_models: dict[str, Any] = {}
    unwhitened_models: dict[str, Any] = {}
    coordinate_metric_eigenvalue = float(factor[COORDINATE7_INDEX] @ metric @ factor[COORDINATE7_INDEX] / (factor[COORDINATE7_INDEX] @ factor[COORDINATE7_INDEX]))
    matched_mode = int(np.argmin(np.abs(alpha - np.sum(factor[COORDINATE7_INDEX] ** 2))))
    for label in MODEL_LABELS:
        banks = [_load_law_bank(label, index) for index in range(8)]
        raw_mode_totals, projected_mode_totals = [], []
        raw_time_rows, projected_time_rows = [], []
        raw_coord_total_fractions, projected_coord_total_fractions = [], []
        raw_metrics, projected_metrics = [], []
        raw_feature_rows, projected_feature_rows = [], []
        exact_coordinate_details = []
        for bank_index, row in enumerate(banks):
            omega = row["time_weights"]
            bank_details = {"bank_index": bank_index, "raw": {}, "projected": {}}
            for kind, error_key, stored_time_key, stored_total_key in (
                ("raw", "raw_hidden_error", "raw_risk_by_time", "raw_total_risk"),
                ("projected", "projected_hidden_error", "projected_risk_by_time", "projected_total_risk"),
            ):
                error = row[error_key]
                z = error @ factor.T
                risk_time_factor = omega * np.sum(z**2, axis=1)
                risk_time_metric = omega * np.einsum("ti,ij,tj->t", error, metric, error)
                modal = omega[:, None] * alpha[None, :] * (error @ vectors) ** 2
                if not np.allclose(risk_time_factor, row[stored_time_key], rtol=2e-10, atol=2e-10):
                    raise RuntimeError(f"factor risk reconstruction failed: {label}/{bank_index}/{kind}")
                if not np.allclose(risk_time_metric, row[stored_time_key], rtol=2e-10, atol=2e-10):
                    raise RuntimeError(f"metric risk reconstruction failed: {label}/{bank_index}/{kind}")
                if not np.allclose(modal.sum(axis=1), row[stored_time_key], rtol=2e-10, atol=2e-10):
                    raise RuntimeError(f"modal time reconstruction failed: {label}/{bank_index}/{kind}")
                if not np.isclose(modal.sum(), float(row[stored_total_key]), rtol=2e-10, atol=2e-10):
                    raise RuntimeError(f"modal total reconstruction failed: {label}/{bank_index}/{kind}")
                coordinate_contribution = omega * z[:, COORDINATE7_INDEX] ** 2
                total_fraction = float(coordinate_contribution.sum() / row[stored_total_key])
                feature = {
                    "time_integrated_absolute_error": np.sum(omega[:, None] * np.abs(error), axis=0),
                    "time_integrated_squared_error": np.sum(omega[:, None] * error**2, axis=0),
                    "maximum_absolute_error": np.max(np.abs(error), axis=0),
                    "maximum_absolute_error_node": np.argmax(np.abs(error), axis=0),
                    "time_integrated_absolute_error_normalized_by_covariance_sd": np.sum(omega[:, None] * np.abs(error), axis=0) / np.sqrt(np.diag(covariance)),
                    "time_integrated_squared_error_normalized_by_covariance_variance": np.sum(omega[:, None] * error**2, axis=0) / np.diag(covariance),
                }
                if kind == "raw":
                    raw_mode_totals.append(modal.sum(axis=0)); raw_time_rows.append(risk_time_metric)
                    raw_coord_total_fractions.append(total_fraction); raw_metrics.append(float(row[stored_total_key])); raw_feature_rows.append(feature)
                else:
                    projected_mode_totals.append(modal.sum(axis=0)); projected_time_rows.append(risk_time_metric)
                    projected_coord_total_fractions.append(total_fraction); projected_metrics.append(float(row[stored_total_key])); projected_feature_rows.append(feature)
                maximum_node = int(np.argmax(risk_time_metric))
                important_nodes = sorted(set((NODE7, maximum_node)))
                for node in important_nodes:
                    terms = factor[COORDINATE7_INDEX] * error[node]
                    bank_details[kind][str(node)] = {
                        "time": float(row["times"][node]),
                        "signed_w7_times_delta_terms": {PSI_FEATURES[j]["name"]: float(terms[j]) for j in range(9)},
                        "z7_from_term_sum": float(np.sum(terms)),
                        "z7_direct": float(z[node, COORDINATE7_INDEX]),
                        "z7_squared": float(z[node, COORDINATE7_INDEX] ** 2),
                        "fraction_of_unweighted_time_node_quadratic_risk": float(z[node, COORDINATE7_INDEX] ** 2 / np.sum(z[node] ** 2)),
                        "fraction_of_total_trajectory_risk": float(coordinate_contribution[node] / row[stored_total_key]),
                    }
            exact_coordinate_details.append(bank_details)
        raw_mode_totals = np.asarray(raw_mode_totals); projected_mode_totals = np.asarray(projected_mode_totals)
        raw_metrics = np.asarray(raw_metrics); projected_metrics = np.asarray(projected_metrics)
        def modal_summary(mode_totals: np.ndarray, totals: np.ndarray) -> dict[str, Any]:
            fractions = mode_totals / totals[:, None]
            sorted_fractions = np.sort(fractions, axis=1)[:, ::-1]
            dominant_indices = np.argmax(fractions, axis=1)
            return {
                "trajectory_total_risk": _distribution(totals),
                "mode_total_risk": [_distribution(mode_totals[:, k]) for k in range(9)],
                "mode_fraction_of_total": [_distribution(fractions[:, k]) for k in range(9)],
                "largest_alpha_mode_fraction": _distribution(fractions[:, 0]),
                "largest_two_alpha_modes_fraction": _distribution(fractions[:, :2].sum(axis=1)),
                "largest_three_alpha_modes_fraction": _distribution(fractions[:, :3].sum(axis=1)),
                "largest_contribution_mode_fraction": _distribution(sorted_fractions[:, 0]),
                "largest_two_contribution_modes_fraction": _distribution(sorted_fractions[:, :2].sum(axis=1)),
                "largest_three_contribution_modes_fraction": _distribution(sorted_fractions[:, :3].sum(axis=1)),
                "dominant_contribution_mode_counts": {str(k): int(np.sum(dominant_indices == k)) for k in range(9)},
                "modes_needed": {
                    str(int(100 * threshold)): _distribution([_mode_count(row, threshold) for row in sorted_fractions])
                    for threshold in (0.50, 0.75, 0.90, 0.95)
                },
                "counterfactual_not_alternative_risk": {
                    "label": "COUNTERFACTUAL DIAGNOSTIC — NOT A VALID ALTERNATIVE RISK",
                    "excluding_largest_metric_eigenmode": _distribution(totals - mode_totals[:, 0]),
                    "excluding_top_two_metric_eigenmodes": _distribution(totals - mode_totals[:, :2].sum(axis=1)),
                },
            }
        modal_models[label] = {
            "raw": modal_summary(raw_mode_totals, raw_metrics),
            "projected": modal_summary(projected_mode_totals, projected_metrics),
            "raw_risk_by_time": [_distribution(np.asarray(raw_time_rows)[:, node]) for node in range(TIME_COUNT)],
            "projected_risk_by_time": [_distribution(np.asarray(projected_time_rows)[:, node]) for node in range(TIME_COUNT)],
            "mode_by_time": {
                "raw_median": np.median(np.asarray([
                    row["time_weights"][:, None] * alpha[None, :] * (row["raw_hidden_error"] @ vectors) ** 2 for row in banks
                ]), axis=0).tolist(),
                "projected_median": np.median(np.asarray([
                    row["time_weights"][:, None] * alpha[None, :] * (row["projected_hidden_error"] @ vectors) ** 2 for row in banks
                ]), axis=0).tolist(),
            },
        }
        def feature_summary(feature_rows: list[dict[str, np.ndarray]]) -> list[dict[str, Any]]:
            return [
                {
                    **PSI_FEATURES[j],
                    **{
                        key: _distribution([row[key][j] for row in feature_rows])
                        for key in feature_rows[0]
                    },
                }
                for j in range(9)
            ]
        unwhitened_models[label] = {
            "raw": feature_summary(raw_feature_rows),
            "projected": feature_summary(projected_feature_rows),
        }
        coordinate_rows[label] = {
            "raw_coordinate7_total_fraction": _distribution(raw_coord_total_fractions),
            "projected_coordinate7_total_fraction": _distribution(projected_coord_total_fractions),
            "exact_per_bank_important_nodes": exact_coordinate_details,
        }
    coordinate_payload = {
        "schema_version": 1,
        "indexing_convention": {
            "reported_coordinate": 7,
            "array_indexing": "zero-based",
            "human_ordinal": "eighth whitening-factor row",
            "meaning": "z[7] = W[7,:] @ delta; it is not Psi_7 and not an original-feature contribution",
            "invariant_metric_mode_rank_zero_based_descending": matched_mode,
            "metric_eigenvalue": coordinate_metric_eigenvalue,
        },
        "w7_row": {PSI_FEATURES[j]["name"]: float(factor[COORDINATE7_INDEX, j]) for j in range(9)},
        "cross_term_warning": "signed w7[j]*delta[j] terms explain the linear combination z7; their squares are not additive feature contributions because z7^2 contains cross terms",
        "models": coordinate_rows,
    }
    modal_payload = {
        "schema_version": 1,
        "authoritative_semantic_decomposition": "M eigenmodes in original Psi-space; c_k(t)=omega_t*alpha_k*(v_k^T delta)^2",
        "risk_mode_eigenvalues": alpha.tolist(),
        "risk_mode_vectors_columns": vectors.tolist(),
        "models": modal_models,
    }
    unwhitened_payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "covariance_standard_deviations": np.sqrt(np.diag(covariance)).tolist(),
        "features": list(PSI_FEATURES),
        "models": unwhitened_models,
    }
    _atomic_json(COORDINATE7_PATH, coordinate_payload)
    _atomic_json(MODAL_PATH, modal_payload)
    _atomic_json(UNWHITENED_PATH, unwhitened_payload)
    return {
        "definition": definition_payload, "spectrum": spectrum_payload,
        "coordinate7": coordinate_payload, "modal": modal_payload,
        "unwhitened": unwhitened_payload,
    }


def _load_endpoints_only() -> tuple[np.ndarray, np.ndarray]:
    with np.load(TRUTH_ARCHIVE, allow_pickle=False) as arrays:
        endpoint0 = np.asarray(arrays["endpoint0"], dtype=np.float64)
        endpoint1 = np.asarray(arrays["endpoint1"], dtype=np.float64)
    return endpoint0, endpoint1


def _mean_psi(configurations: np.ndarray, box: tuple[float, float]) -> np.ndarray:
    total = np.zeros(9, dtype=np.float64)
    for start in range(0, len(configurations), FEATURE_BATCH_SIZE):
        psi = np.asarray(many_body_features(jnp.asarray(configurations[start:start + FEATURE_BATCH_SIZE]), box), dtype=np.float64)
        total += psi.sum(axis=0)
    return total / len(configurations)


def _mean_phi(configurations: np.ndarray, family: Any, eta: np.ndarray) -> np.ndarray:
    total = np.zeros(4, dtype=np.float64)
    for start in range(0, len(configurations), FEATURE_BATCH_SIZE):
        phi = np.asarray(family.features(jnp.asarray(configurations[start:start + FEATURE_BATCH_SIZE]), jnp.asarray(eta)), dtype=np.float64)
        total += phi.sum(axis=0)
    return total / len(configurations)


def _law_eta() -> np.ndarray:
    panel = _json(ROBUST_ROOT / "candidate_panel_reference.json")
    rows = [row for row in panel["rows"] if row["panel_role"] == "law"]
    if len(rows) != 1:
        raise RuntimeError("frozen Law geometry missing or duplicated")
    return np.asarray(rows[0]["eta"], dtype=np.float64)


def _endpoint_cache_path(label: str) -> Path:
    return ENDPOINT_CACHE_ROOT / f"{label}.json"


def _rollout_endpoint(cfg: dict[str, Any], label: str, initial: np.ndarray) -> np.ndarray:
    flow = load_reference(_checkpoint_path(label))
    times = jnp.linspace(0.0, 1.0, TIME_COUNT, dtype=jnp.float64)
    final = []
    for start in range(0, len(initial), ROLLOUT_BATCH_SIZE):
        trajectory = flow.rollout(
            jnp.asarray(initial[start:start + ROLLOUT_BATCH_SIZE]), times,
            substeps_per_interval=int(cfg["banks"]["reference_substeps"]),
        )
        final.append(np.asarray(trajectory[-1], dtype=np.float64))
    return np.concatenate(final, axis=0)


def run_endpoint_audit(progress=None) -> dict[str, Any]:
    verify_and_seal_sources()
    cfg = _json(CONFIG_PATH)
    definition = _metric_definition()
    metric = definition["metric"]
    covariance_sd = np.sqrt(np.diag(definition["covariance"]))
    endpoint0, endpoint1 = _load_endpoints_only()
    if len(endpoint0) < ENDPOINT_EVAL_N:
        raise RuntimeError("endpoint0 ensemble smaller than frozen diagnostic count")
    initial = np.ascontiguousarray(endpoint0[:ENDPOINT_EVAL_N], dtype=np.float64)
    family, eta = _family(cfg), _law_eta()
    target_psi = _mean_psi(endpoint1, tuple(cfg["physics"]["box"]))
    target_phi = _mean_phi(endpoint1, family, eta)
    common = {
        "endpoint_archive_sha256": file_sha256(TRUTH_ARCHIVE),
        "endpoint0_full_array_sha256": _array_sha256(endpoint0),
        "endpoint1_full_array_sha256": _array_sha256(endpoint1),
        "initial_selection": "first 8192 endpoint0 rows; no random sampling",
        "initial_state_sha256": _array_sha256(initial),
        "initial_shape": list(initial.shape), "initial_dtype": str(initial.dtype),
        "target_P1_count": int(len(endpoint1)),
        "target_P1_Psi_mean": target_psi.tolist(), "target_P1_Law_Phi_mean": target_phi.tolist(),
    }
    bridge = {row["label"]: row["CFM_velocity_MSE"] for row in _json(ROBUST_ROOT / "bridge_eval.json")["models"]}
    model_rows = []
    for label in MODEL_LABELS:
        path = _endpoint_cache_path(label)
        if path.exists():
            row = _json(path)
            checks = (
                row["checkpoint_sha256"] == file_sha256(_checkpoint_path(label)),
                row["initial_state_sha256"] == common["initial_state_sha256"],
                row["endpoint_archive_sha256"] == common["endpoint_archive_sha256"],
                row["rollout"]["substeps_per_scientific_interval"] == int(cfg["banks"]["reference_substeps"]),
            )
            if not all(checks):
                raise RuntimeError(f"endpoint cache seal mismatch: {label}")
            model_rows.append(row)
            if progress:
                progress(label, True, float(row["wall_time_seconds"]))
            continue
        started = time.perf_counter()
        final = _rollout_endpoint(cfg, label, initial)
        psi = _mean_psi(final, tuple(cfg["physics"]["box"]))
        phi = _mean_phi(final, family, eta)
        delta_psi, delta_phi = psi - target_psi, phi - target_phi
        row = {
            "schema_version": 1, "label": label,
            "checkpoint_sha256": file_sha256(_checkpoint_path(label)),
            "endpoint_archive_sha256": common["endpoint_archive_sha256"],
            "initial_state_sha256": common["initial_state_sha256"],
            "final_state_sha256": _array_sha256(final),
            "CFM_velocity_MSE": float(bridge[label]),
            "diagnostic_independence": "IN-SAMPLE OR NON-INDEPENDENT ENDPOINT DIAGNOSTIC",
            "rollout": {
                "integrator": "deterministic periodic RK4",
                "time_grid": np.linspace(0.0, 1.0, TIME_COUNT).tolist(),
                "scientific_intervals": TIME_COUNT - 1,
                "substeps_per_scientific_interval": int(cfg["banks"]["reference_substeps"]),
                "total_RK4_steps": (TIME_COUNT - 1) * int(cfg["banks"]["reference_substeps"]),
                "dtype": "float64", "model_specific_tuning": False,
                "intermediate_states_compared_to_truth": False,
            },
            "endpoint_Psi_mean": psi.tolist(),
            "target_P1_Psi_mean": target_psi.tolist(),
            "endpoint_Psi_delta": delta_psi.tolist(),
            "endpoint_Psi_absolute_difference": np.abs(delta_psi).tolist(),
            "endpoint_Psi_standardized_difference": (delta_psi / covariance_sd).tolist(),
            "endpoint_Psi_euclidean_error": float(np.linalg.norm(delta_psi)),
            "endpoint_Psi_standardized_euclidean_error": float(np.linalg.norm(delta_psi / covariance_sd)),
            "endpoint_scientific_quadratic": float(delta_psi @ metric @ delta_psi),
            "endpoint_scientific_whitened_norm": float(np.sqrt(delta_psi @ metric @ delta_psi)),
            "endpoint_Law_Phi_mean": phi.tolist(),
            "target_P1_Law_Phi_mean": target_phi.tolist(),
            "endpoint_Law_Phi_delta": delta_phi.tolist(),
            "endpoint_Law_Phi_absolute_difference": np.abs(delta_phi).tolist(),
            "endpoint_Law_Phi_euclidean_error": float(np.linalg.norm(delta_phi)),
            "preexisting_endpoint_distribution_metric": None,
            "preexisting_metric_audit": "No authoritative skyrmion endpoint-rollout distribution metric was found; only established Psi and Law-Phi observables are reported.",
            "wall_time_seconds": time.perf_counter() - started,
        }
        _atomic_json(path, row)
        model_rows.append(row)
        if progress:
            progress(label, False, float(row["wall_time_seconds"]))
    manifest = {
        "schema_version": 1, "development_only": True,
        "data_separation": {
            "independent_endpoint_holdout_found": False,
            "label": "IN-SAMPLE OR NON-INDEPENDENT ENDPOINT DIAGNOSTIC",
            "reason": "the sealed endpoint0/endpoint1 ensembles were used by reference training and by the existing CFM evaluation sampler; no separate endpoint-only holdout artifact exists",
            "intermediate_truth_used": False, "validation_accessed": False,
            "new_truth_simulation": False,
        },
        "common_inputs": common,
        "rollout_convention_source": _relative(ROOT / "reference.py"),
        "Law_geometry_source": _relative(ROBUST_ROOT / "candidate_panel_reference.json"),
        "scientific_metric_source": _relative(DECOMP_ROOT / "law_recomputation/model_00/bank_00.npz"),
        "model_cache_paths": [str(_endpoint_cache_path(label).relative_to(OUTPUT_ROOT)) for label in MODEL_LABELS],
    }
    results = {
        "schema_version": 1, "common_initial_state_sha256": common["initial_state_sha256"],
        "all_models_identical_initial_state_hash": len({row["initial_state_sha256"] for row in model_rows}) == 1,
        "models": model_rows,
    }
    raw_risks = {label: _json(DECOMP_ROOT / "summary.json")["models"][label]["raw_reference_total_risk"]["median"] for label in MODEL_LABELS}
    losses = np.asarray([row["CFM_velocity_MSE"] for row in model_rows])
    endpoint_quadratic = np.asarray([row["endpoint_scientific_quadratic"] for row in model_rows])
    endpoint_phi = np.asarray([row["endpoint_Law_Phi_euclidean_error"] for row in model_rows])
    intermediate_raw = np.asarray([raw_risks[row["label"]] for row in model_rows])
    feature_summary = {
        "schema_version": 1,
        "models": {
            row["label"]: {
                "CFM_velocity_MSE": row["CFM_velocity_MSE"],
                "endpoint_Psi_euclidean_error": row["endpoint_Psi_euclidean_error"],
                "endpoint_Psi_standardized_euclidean_error": row["endpoint_Psi_standardized_euclidean_error"],
                "endpoint_scientific_quadratic": row["endpoint_scientific_quadratic"],
                "endpoint_Law_Phi_euclidean_error": row["endpoint_Law_Phi_euclidean_error"],
                "intermediate_raw_reference_risk_from_sealed_prior_study": raw_risks[row["label"]],
            }
            for row in model_rows
        },
        "descriptive_spearman_n7": {
            "CFM_loss_vs_endpoint_scientific_quadratic": _spearman(losses, endpoint_quadratic),
            "CFM_loss_vs_endpoint_Law_Phi_error": _spearman(losses, endpoint_phi),
            "CFM_loss_vs_intermediate_raw_risk": _spearman(losses, intermediate_raw),
            "endpoint_scientific_quadratic_vs_intermediate_raw_risk": _spearman(endpoint_quadratic, intermediate_raw),
            "endpoint_Law_Phi_error_vs_intermediate_raw_risk": _spearman(endpoint_phi, intermediate_raw),
            "inference": "descriptive only; n=7; no significance claim",
        },
    }
    _atomic_json(ENDPOINT_MANIFEST_PATH, manifest)
    _atomic_json(ENDPOINT_RESULTS_PATH, results)
    _atomic_json(ENDPOINT_FEATURE_PATH, feature_summary)
    return {"manifest": manifest, "results": results, "features": feature_summary}


def _largest_original_errors(unwhitened: dict[str, Any], label: str, kind: str, count: int = 3) -> list[dict[str, Any]]:
    rows = unwhitened["models"][label][kind]
    ordered = sorted(rows, key=lambda row: row["time_integrated_absolute_error"]["median"], reverse=True)
    return [
        {"feature": row["name"], "integrated_absolute_error": row["time_integrated_absolute_error"]["median"]}
        for row in ordered[:count]
    ]


def finalize() -> dict[str, Any]:
    whitening = run_whitening_audit()
    endpoint = run_endpoint_audit()
    modal, unwhitened = whitening["modal"], whitening["unwhitened"]
    spectrum = whitening["spectrum"]
    baseline_raw_unweighted = sum(row["time_integrated_squared_error"]["median"] for row in unwhitened["models"]["model_00"]["raw"])
    raw_unweighted_ratios = {
        label: sum(row["time_integrated_squared_error"]["median"] for row in unwhitened["models"][label]["raw"]) / baseline_raw_unweighted
        for label in MODEL_LABELS
    }
    selected_top2 = [modal["models"][label]["projected"]["largest_two_contribution_modes_fraction"]["median"] for label in ("model_04", "model_06")]
    if spectrum["floor_dominated"] and max(raw_unweighted_ratios["model_04"], raw_unweighted_ratios["model_06"]) < 1.5:
        part_a = "WHITENING_METRIC_PATHOLOGY_SUSPECTED"
    elif min(selected_top2) >= 0.65 and min(raw_unweighted_ratios["model_04"], raw_unweighted_ratios["model_06"]) >= 1.5:
        part_a = "WHITENING_AMPLIFICATION_IS_MATERIAL"
    elif max(raw_unweighted_ratios["model_04"], raw_unweighted_ratios["model_06"]) >= 1.5:
        part_a = "WHITENING_METRIC_IS_WELL_CONDITIONED_AND_DISCREPANCY_IS_PHYSICAL"
    else:
        part_a = "MIXED_WHITENING_AND_PHYSICAL_DISCREPANCY"
    endpoint_models = endpoint["features"]["models"]
    base_endpoint = endpoint_models["model_00"]["endpoint_scientific_quadratic"]
    selected_endpoint_ratios = [endpoint_models[label]["endpoint_scientific_quadratic"] / base_endpoint for label in ("model_04", "model_06")]
    selected_phi_ratios = [endpoint_models[label]["endpoint_Law_Phi_euclidean_error"] / endpoint_models["model_00"]["endpoint_Law_Phi_euclidean_error"] for label in ("model_04", "model_06")]
    if max(selected_endpoint_ratios + selected_phi_ratios) <= 2.0:
        part_b = "ENDPOINT_QUALITY_COMPARABLE_INTERMEDIATE_PATHS_DIVERGE"
    elif min(selected_endpoint_ratios) > 2.0 and min(selected_phi_ratios) > 2.0:
        part_b = "ENDPOINT_ROLLOUT_FAILURE_EXPLAINS_BAD_REFERENCES"
    else:
        part_b = "ENDPOINT_AND_INTERMEDIATE_QUALITY_BOTH_VARY"
    if part_b == "ENDPOINT_QUALITY_COMPARABLE_INTERMEDIATE_PATHS_DIVERGE" and part_a != "WHITENING_METRIC_PATHOLOGY_SUSPECTED" and min(raw_unweighted_ratios["model_04"], raw_unweighted_ratios["model_06"]) >= 1.5:
        joint = "STRUCTURAL_INTERMEDIATE_REFERENCE_AMBIGUITY"
        recommendation = "Future separately specified robust multi-reference FIDE preflight, including absolute-anchor adequacy; compare alternative prospectively frozen endpoint-only bridges only if robustification remains inadequate."
    elif part_b == "ENDPOINT_ROLLOUT_FAILURE_EXPLAINS_BAD_REFERENCES":
        joint = "REFERENCE_TRAINING_OR_ROLLOUT_PROBLEM"
        recommendation = "Future endpoint-reference training/rollout qualification study using prospectively frozen endpoint-only diagnostics."
    elif part_a == "WHITENING_METRIC_PATHOLOGY_SUSPECTED":
        joint = "SCIENTIFIC_RISK_METRIC_REQUIRES_QUALIFICATION"
        recommendation = "Future prospective scientific-risk metric qualification study; do not modify the frozen metric here."
    elif part_b == "ENDPOINT_AND_INTERMEDIATE_QUALITY_BOTH_VARY":
        joint = "MIXED_REFERENCE_AND_METRIC_EFFECT"
        recommendation = "Future prospectively frozen endpoint-qualification and multi-reference methodology preflight."
    else:
        joint = "INCONCLUSIVE"
        recommendation = "No method change ready."
    model_table = {}
    for label in MODEL_LABELS:
        raw = modal["models"][label]["raw"]
        projected = modal["models"][label]["projected"]
        model_table[label] = {
            "raw_risk": raw["trajectory_total_risk"]["median"],
            "projected_risk": projected["trajectory_total_risk"]["median"],
            "raw_largest_alpha_mode_fraction": raw["largest_alpha_mode_fraction"]["median"],
            "raw_largest_contribution_mode_fraction": raw["largest_contribution_mode_fraction"]["median"],
            "raw_largest_three_contribution_modes_fraction": raw["largest_three_contribution_modes_fraction"]["median"],
            "projected_largest_alpha_mode_fraction": projected["largest_alpha_mode_fraction"]["median"],
            "projected_largest_contribution_mode_fraction": projected["largest_contribution_mode_fraction"]["median"],
            "projected_largest_three_contribution_modes_fraction": projected["largest_three_contribution_modes_fraction"]["median"],
            "largest_raw_original_Psi_errors": _largest_original_errors(unwhitened, label, "raw"),
            "largest_projected_original_Psi_errors": _largest_original_errors(unwhitened, label, "projected"),
            "unwhitened_raw_integrated_squared_error_ratio_vs_model_00": raw_unweighted_ratios[label],
            **endpoint_models[label],
        }
    summary = {
        "schema_version": 1, "version": VERSION, "source_verified": True,
        "guardrails": _json(SOURCE_SEAL_PATH)["guardrails"],
        "whitening_interpretation": part_a,
        "endpoint_interpretation": part_b,
        "joint_development_interpretation": joint,
        "recommended_next_scientific_step": recommendation,
        "decision_inputs": {
            "raw_unwhitened_squared_error_ratios_vs_model_00": raw_unweighted_ratios,
            "projected_top2_mode_fraction_model_04_model_06": selected_top2,
            "selected_endpoint_scientific_quadratic_ratios_vs_model_00": selected_endpoint_ratios,
            "selected_endpoint_Law_Phi_error_ratios_vs_model_00": selected_phi_ratios,
            "metric_floor_dominated": spectrum["floor_dominated"],
            "smallest_covariance_eigenvalue_to_ridge_ratio": spectrum["smallest_raw_covariance_eigenvalue_to_ridge_ratio"],
        },
        "model_table": model_table,
        "coordinate7_convention": whitening["coordinate7"]["indexing_convention"],
        "endpoint_correlations": endpoint["features"]["descriptive_spearman_n7"],
        "no_reference_replacement": True,
        "no_intermediate_truth_model_selection": True,
        "no_official_protocol": True,
    }
    _atomic_json(JOINT_SUMMARY_PATH, summary)
    _write_report(summary, whitening, endpoint)
    _write_inventory()
    return summary


def _write_report(summary: dict[str, Any], whitening: dict[str, Any], endpoint: dict[str, Any]) -> None:
    spectrum, modal = whitening["spectrum"], whitening["modal"]
    lines = [
        "# Whitening-Semantics + Endpoint-Rollout Audit", "", "SOURCE VERIFIED", "",
        "reference models: 7  ", "new training: 0  ", "intermediate truth used: NO  ", "validation accessed: NO", "",
        "## Whitening metric", "",
        "The authoritative repository variable `whitening` is the quadratic metric `M = inv(C + ridge*scale*I)`. The nonnegative-coordinate factor is `W`, with `M = W^T W` and `z = W delta`. This nomenclature distinction is essential.", "",
        f"- Relative regularization: `{RIDGE_RELATIVE:g}`; absolute ridge `{spectrum['regularization_floor_absolute']:.9g}`",
        f"- Raw covariance spectrum: `{spectrum['raw_covariance_eigenvalues_ascending']}`",
        f"- Risk-metric spectrum: `{spectrum['risk_metric_eigenvalues_alpha_descending']}`",
        f"- Risk-metric condition number: `{spectrum['risk_metric_condition_number']:.3f}`",
        f"- Smallest covariance eigenvalue / ridge: `{spectrum['smallest_raw_covariance_eigenvalue_to_ridge_ratio']:.1f}`",
        "- Coordinate-7 convention: zero-based `z[7]`, the eighth factor row; invariant descending mode rank 1 (second-largest alpha)", "",
        "## Model risk decomposition", "",
        "| model | raw risk | projected risk | raw dominant contribution mode | raw top 3 contributions | projected dominant contribution mode | projected top 3 contributions | largest-alpha mode fraction | largest raw original-Psi errors |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in MODEL_LABELS:
        row = summary["model_table"][label]
        errors = ", ".join(item["feature"] for item in row["largest_raw_original_Psi_errors"])
        lines.append(
            f"| {label} | {row['raw_risk']:.6f} | {row['projected_risk']:.6f} | {row['raw_largest_contribution_mode_fraction']:.3f} | "
            f"{row['raw_largest_three_contribution_modes_fraction']:.3f} | {row['projected_largest_contribution_mode_fraction']:.3f} | "
            f"{row['projected_largest_three_contribution_modes_fraction']:.3f} | {row['projected_largest_alpha_mode_fraction']:.3f} | {errors} |"
        )
    lines += ["", "## Model 00 / 04 / 06 dominant invariant modes", ""]
    for label in ("model_00", "model_04", "model_06"):
        fractions = modal["models"][label]["projected"]["mode_fraction_of_total"]
        mode_order = sorted(range(9), key=lambda mode: fractions[mode]["median"], reverse=True)[:3]
        lines += [f"### {label}", ""]
        for mode in mode_order:
            loading = spectrum["mode_loadings"][mode]
            largest = ", ".join(f"{item['feature']}={item['loading']:+.3f}" for item in loading["largest_absolute_loadings"])
            lines.append(f"- Mode {mode}: alpha `{loading['metric_eigenvalue']:.6g}`, median projected contribution `{fractions[mode]['median']:.3%}`; {largest}.")
        lines.append("")
    lines += [
        "## Coordinate-7 semantics", "",
        "The earlier coordinate 7 is exactly zero-based `z[7]`, not `Psi_7`. Signed `W[7,j]*delta[j]` receipts for node 7 and each model's maximum-risk node are in `whitened_coordinate7_audit.json`. They explain the linear combination only; squaring creates cross terms and is not an additive original-feature decomposition.", "",
        "## Counterfactual diagnostic", "",
        "`metric_eigenmode_decomposition.json` reports totals excluding the largest and top two metric modes under the explicit label **COUNTERFACTUAL DIAGNOSTIC — NOT A VALID ALTERNATIVE RISK**. No metric is changed or proposed here.", "",
        "## Endpoint rollout quality", "",
        "This is an **IN-SAMPLE OR NON-INDEPENDENT ENDPOINT DIAGNOSTIC**: no separate endpoint holdout exists. All models use the same first 8192 P0 configurations, deterministic periodic float64 RK4, 13 nodes, and 14 substeps per interval. Only final t=1 is compared with P1.", "",
        "| model | CFM loss | endpoint Psi L2 | endpoint whitened Psi norm | endpoint Law-Phi L2 | intermediate raw risk |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in MODEL_LABELS:
        row = summary["model_table"][label]
        lines.append(
            f"| {label} | {row['CFM_velocity_MSE']:.6f} | {row['endpoint_Psi_euclidean_error']:.6g} | "
            f"{math.sqrt(row['endpoint_scientific_quadratic']):.6g} | {row['endpoint_Law_Phi_euclidean_error']:.6g} | "
            f"{row['intermediate_raw_reference_risk_from_sealed_prior_study']:.6f} |"
        )
    lines += [
        "", "## Interpretations", "",
        f"**Whitening interpretation:** `{summary['whitening_interpretation']}`", "",
        f"**Endpoint interpretation:** `{summary['endpoint_interpretation']}`", "",
        f"**Joint development interpretation:** `{summary['joint_development_interpretation']}`", "",
        f"**Recommended next scientific step:** {summary['recommended_next_scientific_step']}", "",
        "NO reference retraining  ", "NO reference replacement  ", "NO intermediate-truth model selection  ",
        "NO Tangent  ", "NO Full  ", "NO validation  ", "NO official protocol created", "",
    ]
    _atomic_text(REPORT_PATH, "\n".join(lines))


def _write_inventory() -> dict[str, Any]:
    paths = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file() and path != INVENTORY_PATH)
    payload = {
        "schema_version": 1, "version": VERSION,
        "files": [{"path": str(path.relative_to(OUTPUT_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in paths],
        "joint_summary_sha256": file_sha256(JOINT_SUMMARY_PATH),
    }
    _atomic_json(INVENTORY_PATH, payload)
    return payload


def console_report() -> str:
    summary = _json(JOINT_SUMMARY_PATH)
    spectrum = _json(WHITENING_SPECTRUM_PATH)
    lines = [
        "SOURCE VERIFIED", "", "reference models: 7", "new training: 0",
        "intermediate truth used: NO", "validation accessed: NO", "",
        "WHITENING METRIC", "",
        f"regularization: relative={RIDGE_RELATIVE:g}, absolute={spectrum['regularization_floor_absolute']:.9g}",
        f"covariance spectrum: {spectrum['raw_covariance_eigenvalues_ascending']}",
        f"risk-metric spectrum: {spectrum['risk_metric_eigenvalues_alpha_descending']}",
        f"condition number: {spectrum['risk_metric_condition_number']:.6f}",
        "coordinate-7 convention: zero-based z[7], eighth W row, invariant mode rank 1", "",
        "MODEL RISK DECOMPOSITION", "",
        "model      raw risk   projected risk   dominant mode   top 3 contribution modes   largest-alpha mode   largest original-Psi errors",
    ]
    for label in MODEL_LABELS:
        row = summary["model_table"][label]
        errors = ",".join(item["feature"] for item in row["largest_raw_original_Psi_errors"])
        lines.append(f"{label:8s} {row['raw_risk']:10.6f} {row['projected_risk']:16.6f} {row['projected_largest_contribution_mode_fraction']:13.3%} {row['projected_largest_three_contribution_modes_fraction']:26.3%} {row['projected_largest_alpha_mode_fraction']:20.3%} {errors}")
    lines += ["", "MODEL 00 / 04 / 06", ""]
    spectrum_modes = spectrum["mode_loadings"]
    for label in ("model_00", "model_04", "model_06"):
        row = summary["model_table"][label]
        fractions = _json(MODAL_PATH)["models"][label]["projected"]["mode_fraction_of_total"]
        dominant_mode = max(range(9), key=lambda mode: fractions[mode]["median"])
        loading = spectrum_modes[dominant_mode]
        largest = ", ".join(item["feature"] for item in loading["largest_absolute_loadings"][:3])
        lines.append(f"{label}: dominant contribution mode={dominant_mode}, eigenvalue={loading['metric_eigenvalue']:.6g}, projected contribution={fractions[dominant_mode]['median']:.3%}, loadings={largest}")
    lines += ["", "ENDPOINT ROLLOUT QUALITY", "",
        "model      CFM loss   endpoint Psi L2   endpoint whitened norm   endpoint Law-Phi L2",
    ]
    for label in MODEL_LABELS:
        row = summary["model_table"][label]
        lines.append(f"{label:8s} {row['CFM_velocity_MSE']:9.6f} {row['endpoint_Psi_euclidean_error']:17.7g} {math.sqrt(row['endpoint_scientific_quadratic']):24.7g} {row['endpoint_Law_Phi_euclidean_error']:20.7g}")
    lines += ["", f"WHITENING INTERPRETATION: {summary['whitening_interpretation']}", "",
        f"ENDPOINT INTERPRETATION: {summary['endpoint_interpretation']}", "",
        f"JOINT DEVELOPMENT INTERPRETATION: {summary['joint_development_interpretation']}", "",
        f"RECOMMENDED NEXT SCIENTIFIC STEP: {summary['recommended_next_scientific_step']}", "",
        "NO reference retraining", "NO reference replacement", "NO intermediate-truth model selection",
        "NO Tangent", "NO Full", "NO validation", "NO official protocol created",
    ]
    return "\n".join(lines)
