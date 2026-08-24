"""Additive Tangent Pareto analysis reusing the certified Full-run artifacts.

This script never edits or reruns the Law/Full Pareto sweep.  It rescales the
saved feasible geometry pool with a closed-form Tangent action, performs a
small Tangent-only local refinement on selection banks, and validates only the
distinct winners on the frozen disjoint validation banks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from mfsi.cache import file_sha256, fingerprint
from mfsi.io import write_csv, write_json
from experiments.skyrmions_deep_ritz.domain import ConfigurationBank
from experiments.skyrmions_deep_ritz.experiment import (
    _moment_reconstruction,
    _projection_backend,
    _projection_config,
    _selection_risk,
    _time_weights,
)
from experiments.skyrmions_deep_ritz.forcing import strict_project_trajectory
from experiments.skyrmions_deep_ritz.measurements import (
    LocalDensitySensors,
    local_sensor_designs,
)
from experiments.skyrmions_deep_ritz.risk import many_body_features, whitening_from_truth
from experiments.skyrmions_deep_ritz.selection import nested_certified_selection
from experiments.skyrmions_deep_ritz.tangent import (
    TangentCertificateConfig,
    audit_tangent_action,
)
from experiments.skyrmions_deep_ritz.visualize_authoritative import _resolve_result_path

DEFAULT_REFINEMENT_SCALES = (0.035, 0.015)


def _eta_key(eta: Any) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(eta, dtype=np.float64), 12))


def _candidate_id(eta: Any) -> str:
    return "tangent-" + fingerprint({"eta": list(_eta_key(eta))})[:12]


def _load_bank(path: Path) -> dict[str, jax.Array]:
    with np.load(path) as source:
        return {
            name: jnp.asarray(source[name], dtype=jnp.float64)
            for name in ("configurations", "velocity", "base_weights")
        }


def _tangent_config(cfg: dict[str, Any]) -> TangentCertificateConfig:
    return TangentCertificateConfig(
        maximum_gram_condition=float(cfg["forcing"]["max_covariance_condition"]),
        maximum_moment_rate_residual=min(
            1.0e-10,
            float(cfg["certificates"]["maximum_moment_rate_residual"]),
        ),
        minimum_ess_fraction=float(cfg["forcing"]["minimum_ess_fraction"]),
        maximum_projection_residual=float(cfg["forcing"]["projection_tolerance"]),
    )


def _evaluate_tangent(
    eta: jax.Array,
    targets: jax.Array,
    derivatives: jax.Array,
    bank: dict[str, jax.Array],
    family: LocalDensitySensors,
    times: jax.Array,
    time_weights: jax.Array,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    features = family.features(bank["configurations"], eta)
    projection = strict_project_trajectory(
        features,
        bank["base_weights"],
        targets,
        projection_cfg=_projection_config(cfg),
        tolerance=float(cfg["forcing"]["projection_tolerance"]),
        trajectory_backend=_projection_backend(cfg),
    )
    return audit_tangent_action(
        bank["configurations"],
        bank["velocity"],
        projection.weights,
        derivatives,
        eta,
        family,
        time_weights,
        projection_residual=projection.residual,
        ess_fraction=projection.ess_fraction,
        cfg=_tangent_config(cfg),
    )


def _reconstruct(
    eta: jax.Array,
    truth: ConfigurationBank,
    family: LocalDensitySensors,
    cfg: dict[str, Any],
    *,
    validation: bool,
) -> tuple[jax.Array, jax.Array]:
    offsets = cfg["banks"]["seed_offsets"]
    noise_seed = int(cfg["seed"]) + int(offsets["observation"])
    if validation:
        noise_seed += 10000
    targets, derivatives, _ = _moment_reconstruction(
        eta,
        truth,
        family,
        cfg,
        noise_seed=noise_seed,
    )
    return targets, derivatives


def _saved_candidates(
    pareto: dict[str, Any],
    pareto_path: Path,
    *,
    maximum_risk: float,
) -> tuple[list[dict[str, Any]], list[Path]]:
    candidates: dict[tuple[float, ...], dict[str, Any]] = {}
    result_paths: list[Path] = []
    for pareto_row in pareto["rows"]:
        result_path = _resolve_result_path(pareto_row["result"], pareto_path)
        result_paths.append(result_path)
        diagnostics = json.loads(
            (result_path.parent / "search_diagnostics.json").read_text(encoding="utf-8")
        )
        for row in diagnostics["rows"]:
            if (
                not row.get("valid")
                or not row.get("support_valid")
                or float(row.get("risk", float("inf"))) > maximum_risk
            ):
                continue
            key = _eta_key(row["eta"])
            existing = candidates.get(key)
            record = {
                "id": _candidate_id(row["eta"]),
                "eta": [float(value) for value in row["eta"]],
                "risk": float(row["risk"]),
                "source": "saved_full_sweep_pool",
            }
            if existing is None or record["risk"] < existing["risk"]:
                candidates[key] = record
    return list(candidates.values()), result_paths


def _verify_frozen_artifacts(result_paths: list[Path]) -> dict[str, str]:
    names = (
        "truth_banks.npz",
        "reference.npz",
        "reference_bank_projection.npz",
        "reference_bank_ritz_train.npz",
        "reference_bank_ritz_audit.npz",
        "reference_bank_validation_fit.npz",
        "reference_bank_validation_audit.npz",
    )
    hashes: dict[str, str] = {}
    for name in names:
        values = {file_sha256(path.parent / name) for path in result_paths}
        if len(values) != 1:
            raise RuntimeError(f"Pareto rows do not share one frozen {name}")
        hashes[name] = values.pop()
    return hashes


def _score_training_candidate(
    candidate: dict[str, Any],
    *,
    truth: ConfigurationBank,
    bank: dict[str, jax.Array],
    family: LocalDensitySensors,
    cfg: dict[str, Any],
    time_weights: jax.Array,
) -> dict[str, Any]:
    eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
    targets, derivatives = _reconstruct(
        eta, truth, family, cfg, validation=False
    )
    tangent = _evaluate_tangent(
        eta, targets, derivatives, bank, family, truth.times, time_weights, cfg
    )
    return {
        **candidate,
        "action": float(tangent["action"]),
        "valid": bool(tangent["valid"]),
        "training_certificate": tangent,
    }


def _nested(
    candidates: list[dict[str, Any]],
    *,
    anchor_risk: float,
    allowances: list[float],
) -> list[dict[str, Any]]:
    return nested_certified_selection(
        candidates,
        anchor_risk=anchor_risk,
        allowances_percent=allowances,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache-only Tangent extension of the certified skyrmion Pareto sweep"
    )
    parser.add_argument(
        "pareto",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "pareto_authoritative" / "pareto.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--refinement-scales",
        nargs="*",
        type=float,
        default=list(DEFAULT_REFINEMENT_SCALES),
    )
    parser.add_argument("--local-count", type=int, default=12)
    parser.add_argument("--audit-shortlist", type=int, default=10)
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore a matching completed Tangent result and rescore",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    pareto = json.loads(args.pareto.read_text(encoding="utf-8"))
    if not pareto.get("certified") or pareto.get("exploratory_override"):
        raise RuntimeError("Tangent extension requires the certified, non-exploratory Pareto sweep")
    output = args.output or args.pareto.parent / "tangent_analysis"
    output.mkdir(parents=True, exist_ok=True)
    allowances = sorted({float(row["allowance_percent"]) for row in pareto["rows"]})
    anchor_risk = float(pareto["frozen_law_risk"])
    maximum_risk = anchor_risk * (1.0 + max(allowances) / 100.0)

    saved, result_paths = _saved_candidates(
        pareto, args.pareto, maximum_risk=maximum_risk
    )
    artifact_hashes = _verify_frozen_artifacts(result_paths)
    cache_signature = fingerprint({
        "schema": 1,
        "pareto": file_sha256(args.pareto),
        "artifacts": artifact_hashes,
        "refinement_scales": args.refinement_scales,
        "local_count": args.local_count,
        "audit_shortlist": args.audit_shortlist,
    })
    completed_path = output / "tangent_pareto.json"
    if completed_path.is_file() and not args.force:
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        if completed.get("cache_signature") == cache_signature:
            print(f"[tangent] reusing matching completed result: {completed_path}")
            return
    source_index = min(
        range(len(pareto["rows"])),
        key=lambda index: abs(float(pareto["rows"][index]["allowance_percent"]) - 3.0),
    )
    source_path = result_paths[source_index]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cfg = source["config"]
    physics = cfg["physics"]
    measurement = cfg["measurement"]
    family = LocalDensitySensors(
        int(measurement["n_sensors"]),
        float(measurement["sensor_width"]),
        tuple(float(value) for value in physics["box"]),
        float(measurement["min_separation"]),
    )

    with np.load(source_path.parent / "truth_banks.npz") as frozen_truth:
        times = jnp.asarray(frozen_truth["times"], dtype=jnp.float64)
        design_truth = ConfigurationBank(
            times,
            jnp.asarray(frozen_truth["design"], dtype=jnp.float64),
        )
        validation_truth = ConfigurationBank(
            times,
            jnp.asarray(frozen_truth["validation"], dtype=jnp.float64),
        )
    time_weights = _time_weights(times)
    banks = {
        name: _load_bank(source_path.parent / f"reference_bank_{name}.npz")
        for name in (
            "projection",
            "ritz_train",
            "ritz_audit",
            "validation_fit",
            "validation_audit",
        )
    }

    print(
        f"[tangent] scoring {len(saved)} saved feasible geometries on the frozen training bank",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(saved, start=1):
        scored = _score_training_candidate(
            candidate,
            truth=design_truth,
            bank=banks["ritz_train"],
            family=family,
            cfg=cfg,
            time_weights=time_weights,
        )
        if scored["valid"]:
            candidates.append(scored)
        if index % 100 == 0 or index == len(saved):
            print(f"[tangent] scored {index}/{len(saved)}", flush=True)

    existing_keys = {_eta_key(row["eta"]) for row in candidates}
    truth_features = many_body_features(design_truth.configurations, tuple(physics["box"]))
    truth_means = jnp.mean(truth_features, axis=1)
    whitening = whitening_from_truth(truth_features)
    selection_reference_features = many_body_features(
        banks["projection"]["configurations"], tuple(physics["box"])
    )
    refined_count = 0
    for round_index, scale in enumerate(args.refinement_scales):
        training_winners = _nested(
            candidates, anchor_risk=anchor_risk, allowances=allowances
        )
        centers = []
        center_keys: set[tuple[float, ...]] = set()
        for row in training_winners:
            eta = row["candidate"]["eta"]
            key = _eta_key(eta)
            if key not in center_keys:
                center_keys.add(key)
                centers.append(eta)
        local = local_sensor_designs(
            jax.random.PRNGKey(int(cfg["seed"]) + 7301 + round_index),
            jnp.asarray(centers, dtype=jnp.float64),
            count_per_center=int(args.local_count),
            scale=float(scale),
            family=family,
        )
        print(
            f"[tangent] refinement {round_index + 1}: {len(local)} proposals at scale {scale:g}",
            flush=True,
        )
        for eta in local:
            key = _eta_key(eta)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            risk_row = _selection_risk(
                eta,
                design_truth,
                banks["projection"],
                family,
                selection_reference_features,
                truth_means,
                whitening,
                time_weights,
                cfg,
                noise_seed=int(cfg["seed"])
                + int(cfg["banks"]["seed_offsets"]["observation"]),
            )
            if (
                not risk_row.get("valid")
                or float(risk_row.get("risk", float("inf"))) > maximum_risk
            ):
                continue
            tangent = _evaluate_tangent(
                eta,
                risk_row["targets"],
                risk_row["derivatives"],
                banks["ritz_train"],
                family,
                times,
                time_weights,
                cfg,
            )
            if tangent["valid"]:
                candidate = {
                    "id": _candidate_id(eta),
                    "eta": np.asarray(eta).tolist(),
                    "risk": float(risk_row["risk"]),
                    "action": float(tangent["action"]),
                    "valid": True,
                    "source": f"tangent_local_refinement_{round_index + 1}",
                    "training_certificate": tangent,
                }
                candidates.append(candidate)
                refined_count += 1

    shortlist: dict[tuple[float, ...], dict[str, Any]] = {}
    for allowance in allowances:
        limit = anchor_risk * (1.0 + allowance / 100.0)
        eligible = sorted(
            (
                row for row in candidates
                if row["valid"] and float(row["risk"]) <= limit
            ),
            key=lambda row: (float(row["action"]), float(row["risk"]), row["id"]),
        )
        if not eligible:
            raise RuntimeError(f"no Tangent candidate inside the {allowance:g}% risk band")
        for row in eligible[: max(1, int(args.audit_shortlist))]:
            shortlist[_eta_key(row["eta"])] = row

    print(
        f"[tangent] authoritative audit of {len(shortlist)} shortlisted geometries",
        flush=True,
    )
    audited: list[dict[str, Any]] = []
    for candidate in shortlist.values():
        eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
        targets, derivatives = _reconstruct(
            eta, design_truth, family, cfg, validation=False
        )
        certificate = _evaluate_tangent(
            eta,
            targets,
            derivatives,
            banks["ritz_audit"],
            family,
            times,
            time_weights,
            cfg,
        )
        audited.append({
            **candidate,
            "training_action": float(candidate["action"]),
            "action": float(certificate["action"]),
            "valid": bool(certificate["valid"]),
            "selection_certificate": certificate,
        })
    selected = _nested(audited, anchor_risk=anchor_risk, allowances=allowances)

    law_eta = jnp.asarray(pareto["frozen_law_eta"], dtype=jnp.float64)
    law_targets, law_derivatives = _reconstruct(
        law_eta, design_truth, family, cfg, validation=False
    )
    law_selection = _evaluate_tangent(
        law_eta,
        law_targets,
        law_derivatives,
        banks["ritz_audit"],
        family,
        times,
        time_weights,
        cfg,
    )

    validation_truth_features = many_body_features(
        validation_truth.configurations, tuple(physics["box"])
    )
    validation_truth_means = jnp.mean(validation_truth_features, axis=1)
    validation_reference_features = many_body_features(
        banks["validation_fit"]["configurations"], tuple(physics["box"])
    )
    validation_cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def validate_eta(eta_values: Any) -> dict[str, Any]:
        key = _eta_key(eta_values)
        if key in validation_cache:
            return validation_cache[key]
        eta = jnp.asarray(eta_values, dtype=jnp.float64)
        risk_row = _selection_risk(
            eta,
            validation_truth,
            banks["validation_fit"],
            family,
            validation_reference_features,
            validation_truth_means,
            whitening,
            time_weights,
            cfg,
            noise_seed=int(cfg["seed"])
            + int(cfg["banks"]["seed_offsets"]["observation"])
            + 10000,
        )
        if not risk_row.get("valid"):
            raise RuntimeError(f"independent Tangent validation risk failed for {key}")
        tangent = _evaluate_tangent(
            eta,
            risk_row["targets"],
            risk_row["derivatives"],
            banks["validation_audit"],
            family,
            times,
            time_weights,
            cfg,
        )
        if not tangent["valid"]:
            raise RuntimeError(f"independent Tangent certificate failed for {key}")
        validation_cache[key] = {
            "risk": float(risk_row["risk"]),
            "certificate": tangent,
        }
        return validation_cache[key]

    law_validation = validate_eta(law_eta)
    rows: list[dict[str, Any]] = []
    for nested_row in selected:
        allowance = float(nested_row["allowance_percent"])
        candidate = nested_row["candidate"]
        validation = validate_eta(candidate["eta"])
        full_row = next(
            row for row in pareto["rows"]
            if abs(float(row["allowance_percent"]) - allowance) <= 1.0e-12
        )
        selection_action = float(candidate["action"])
        validation_action = float(validation["certificate"]["action"])
        rows.append({
            "allowance_percent": allowance,
            "id": candidate["id"],
            "eta": candidate["eta"],
            "source": candidate["source"],
            "selection_risk": float(candidate["risk"]),
            "extra_risk_percent": 100.0 * (float(candidate["risk"]) / anchor_risk - 1.0),
            "budget_used_fraction": (
                100.0 * (float(candidate["risk"]) / anchor_risk - 1.0) / allowance
                if allowance else 0.0
            ),
            "selection_tangent_action": selection_action,
            "selection_tangent_action_reduction_vs_law": (
                1.0 - selection_action / float(law_selection["action"])
            ),
            "selection_certificate": candidate["selection_certificate"],
            "validation_risk": float(validation["risk"]),
            "validation_tangent_action": validation_action,
            "validation_tangent_action_standard_error": float(
                validation["certificate"]["action_standard_error"]
            ),
            "validation_tangent_action_reduction_vs_law": (
                1.0
                - validation_action
                / float(law_validation["certificate"]["action"])
            ),
            "validation_certificate": validation["certificate"],
            "full_selection_action_at_full_winner": float(full_row["selection_action"]),
            "full_validation_action_at_full_winner": float(full_row["validation_action"]),
            "full_winner_eta": full_row["eta"],
            "valid": True,
        })

    result = {
        "schema_version": 1,
        "experiment": "skyrmions_many_body_deep_ritz_tangent_extension",
        "method": "closed-form many-body Tangent action",
        "scope_override": (
            "User explicitly requested Tangent analysis after the original "
            "Law/Full-only experiment was completed."
        ),
        "certified": True,
        "exploratory": False,
        "existing_full_pareto": str(args.pareto.resolve()),
        "existing_full_pareto_sha256": file_sha256(args.pareto),
        "full_deep_ritz_rerun": False,
        "truth_or_reference_regenerated": False,
        "artifact_hashes": artifact_hashes,
        "cache_signature": cache_signature,
        "selection_protocol": {
            "saved_feasible_geometries_scored": len(saved),
            "valid_training_scores": len(candidates),
            "tangent_local_refined_geometries": refined_count,
            "refinement_scales": [float(value) for value in args.refinement_scales],
            "local_count_per_center": int(args.local_count),
            "authoritative_audit_shortlist": len(audited),
            "distinct_validation_winners": len(validation_cache) - 1,
            "training_bank": "reference_bank_ritz_train.npz",
            "selection_audit_bank": "reference_bank_ritz_audit.npz",
            "validation_fit_bank": "reference_bank_validation_fit.npz",
            "validation_audit_bank": "reference_bank_validation_audit.npz",
        },
        "law": {
            "eta": np.asarray(law_eta).tolist(),
            "selection_risk": anchor_risk,
            "selection_certificate": law_selection,
            "validation_risk": float(law_validation["risk"]),
            "validation_certificate": law_validation["certificate"],
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "tangent_pareto.json", result)
    write_json(output / "candidate_scores.json", {
        "cache_signature": cache_signature,
        "candidates": candidates,
        "audited_candidates": audited,
    })
    write_csv(output / "tangent_pareto.csv", rows)
    print(f"tangent_pareto={output / 'tangent_pareto.json'}", flush=True)
    print(f"elapsed_seconds={result['elapsed_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
