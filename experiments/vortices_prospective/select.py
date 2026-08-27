from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from common import (
    artifact_dirs,
    config_hash,
    experiment_source_hash,
    geometry_valid,
    load_config,
    software_metadata,
    write_json_atomic,
)
from evaluator import ProspectiveEvaluator, ensure_observation_bank
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData


def _generate_global(cfg: dict[str, Any]) -> list[np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 6101]))
    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    count = int(cfg["search"]["global_candidates"])
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 1000:
        attempts += 1
        centers = np.column_stack([
            rng.uniform(margin, 2.0 - margin, int(m["n_sensors"])),
            rng.uniform(margin, 1.0 - margin, int(m["n_sensors"])),
        ])
        eta = centers.reshape(-1)
        if geometry_valid(eta, cfg):
            out.append(eta)
    if len(out) != count:
        raise RuntimeError("could not generate enough separated sensor geometries")
    return out


def _generate_local(
    cfg: dict[str, Any], law_eta: np.ndarray, refinement_pass: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg["seed"]), 6102, int(refinement_pass)])
    )
    m = cfg["measurement"]
    margin = float(m["boundary_margin"])
    count_per_scale = int(cfg["search"]["law_local_candidates"])
    scales = [
        float(value)
        for value in cfg["search"].get(
            "local_scales", [cfg["search"].get("local_scale", 0.025)]
        )
    ]
    count = count_per_scale * len(scales)
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 1000:
        attempts += 1
        scale = scales[len(out) // count_per_scale]
        eta = law_eta + rng.normal(scale=scale, size=law_eta.shape)
        centers = eta.reshape((-1, 2))
        centers[:, 0] = np.clip(centers[:, 0], margin, 2.0 - margin)
        centers[:, 1] = np.clip(centers[:, 1], margin, 1.0 - margin)
        eta = centers.reshape(-1)
        if geometry_valid(eta, cfg):
            out.append(eta)
    return out


def _metric_mean(result: dict[str, Any], key: str) -> float:
    value = result.get(key, {}).get("mean")
    return float(value) if value is not None and np.isfinite(value) else float("inf")


def _candidate_key(eta: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(eta, dtype=np.float64), 12))


def risk_feasible(risk: float, law_risk: float, allowance: float, tolerance: float) -> bool:
    """The one shared prospective feasibility rule used by Law and Full."""
    return bool(float(risk) <= (1.0 + float(allowance)) * float(law_risk) + float(tolerance))


def select(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dirs = artifact_dirs(output_dir)
    dirs["results"].mkdir(parents=True, exist_ok=True)
    endpoint_path = dirs["endpoint"] / "endpoint_data.npz"
    aggregate_path = dirs["prospective"] / "aggregate_predictions.npz"
    rollout_path = dirs["endpoint"] / "reference_rollout.npz"
    reference_receipt_path = dirs["endpoint"] / "reference_receipt.json"
    for path in (endpoint_path, aggregate_path, rollout_path, reference_receipt_path):
        if not path.exists():
            raise FileNotFoundError(f"required prospective input is missing: {path}")
    data = TargetProspectiveData.load(endpoint_path, aggregate_path)
    evaluator = ProspectiveEvaluator(cfg, data, rollout_path)
    bank = ensure_observation_bank(
        dirs["prospective"] / "selection_randomness.npz",
        cfg,
        int(cfg["search"]["selection_trials"]),
        namespace=6201,
    )
    signature = {
        "config_hash": config_hash(cfg),
        "experiment_source_hash": experiment_source_hash(),
        "endpoint_sha256": file_sha256(endpoint_path),
        "aggregate_sha256": file_sha256(aggregate_path),
        "reference_rollout_sha256": file_sha256(rollout_path),
        "selection_randomness_sha256": file_sha256(dirs["prospective"] / "selection_randomness.npz"),
    }
    manifest_path = dirs["results"] / "frozen_manifest.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
        if frozen.get("selection_input_hashes") == signature:
            print("[selection] reusing compatible frozen manifest", flush=True)
            return frozen
        raise RuntimeError(
            "a frozen manifest already exists with different inputs; use a new output directory"
        )

    started = time.perf_counter()
    evaluated: dict[tuple[float, ...], dict[str, Any]] = {}

    def risk_evaluate(eta: np.ndarray, provenance: str) -> dict[str, Any]:
        key = _candidate_key(eta)
        if key not in evaluated:
            result = evaluator.evaluate_prospective(eta, bank, compute_full=False)
            evaluated[key] = {"eta": np.asarray(eta).tolist(), "risk_result": result, "provenance": [provenance]}
        elif provenance not in evaluated[key]["provenance"]:
            evaluated[key]["provenance"].append(provenance)
        return evaluated[key]

    print("[selection] evaluating global aggregate-only risk candidates", flush=True)
    for eta in _generate_global(cfg):
        risk_evaluate(eta, "global")
    valid_global = [row for row in evaluated.values() if row["risk_result"]["valid"]]
    if not valid_global:
        raise RuntimeError("no numerically valid global Law candidate")
    law_row = min(valid_global, key=lambda row: _metric_mean(row["risk_result"], "risk"))
    law_eta = np.asarray(law_row["eta"], dtype=np.float64)

    refinement_passes = int(cfg["search"].get("law_refinement_passes", 1))
    for refinement_pass in range(refinement_passes):
        print(
            f"[selection] refining near the aggregate-risk Law anchor "
            f"pass {refinement_pass + 1}/{refinement_passes}",
            flush=True,
        )
        for eta in _generate_local(cfg, law_eta, refinement_pass):
            risk_evaluate(eta, f"law_local_pass_{refinement_pass + 1}")
        valid_rows = [row for row in evaluated.values() if row["risk_result"]["valid"]]
        law_row = min(valid_rows, key=lambda row: _metric_mean(row["risk_result"], "risk"))
        law_eta = np.asarray(law_row["eta"], dtype=np.float64)
    law_risk = _metric_mean(law_row["risk_result"], "risk")
    allowance = float(cfg["risk_allowance"])
    risk_limit = (1.0 + allowance) * law_risk
    risk_tolerance = float(cfg["validity"]["risk_constraint_tolerance"])
    feasible = [
        row for row in valid_rows
        if risk_feasible(
            _metric_mean(row["risk_result"], "risk"), law_risk, allowance, risk_tolerance
        )
    ]
    tangent_row = min(
        feasible, key=lambda row: _metric_mean(row["risk_result"], "tangent_proxy")
    )
    print(f"[selection] reduced Full proxy rescoring {len(feasible)} risk-feasible candidates", flush=True)
    proxy_rows = []
    for index, row in enumerate(feasible, start=1):
        if index == 1 or index == len(feasible) or index % max(1, len(feasible) // 10) == 0:
            print(f"[selection] Full proxy {index}/{len(feasible)}", flush=True)
        proxy = evaluator.evaluate_full_proxy(np.asarray(row["eta"], dtype=np.float64), bank)
        row["full_proxy_result"] = proxy
        if proxy["valid"]:
            proxy_rows.append(row)
    if not proxy_rows:
        raise RuntimeError("no numerically valid reduced Full proxy candidate")
    proxy_rows.sort(key=lambda row: _metric_mean(row["full_proxy_result"], "full_proxy"))
    shortlist = proxy_rows[: int(cfg["search"]["full_shortlist"])]
    if law_row not in shortlist:
        shortlist.append(law_row)
    if tangent_row not in shortlist:
        shortlist.append(tangent_row)
    authoritative = shortlist[: int(cfg["search"]["authoritative_full_candidates"])]
    if law_row not in authoritative:
        authoritative.append(law_row)
    if tangent_row not in authoritative:
        authoritative.append(tangent_row)

    print(f"[selection] authoritative Full rescoring {len(authoritative)} frozen-bank candidates", flush=True)
    full_rows = []
    for index, row in enumerate(authoritative, start=1):
        print(f"[selection] Full {index}/{len(authoritative)}", flush=True)
        eta = np.asarray(row["eta"], dtype=np.float64)
        result = evaluator.evaluate_prospective(eta, bank, compute_full=True)
        risk_value = _metric_mean(result, "risk")
        if result["valid"] and risk_feasible(risk_value, law_risk, allowance, risk_tolerance):
            full_rows.append({**row, "authoritative_result": result})
    if not full_rows:
        raise RuntimeError("no certified Full candidate satisfies the prospective risk constraint")
    full_row = min(full_rows, key=lambda row: _metric_mean(row["authoritative_result"], "full_action"))
    law_key = _candidate_key(law_eta)
    tangent_eta = np.asarray(tangent_row["eta"], dtype=np.float64)
    tangent_key = _candidate_key(tangent_eta)
    law_full_row = next(
        row for row in full_rows if _candidate_key(np.asarray(row["eta"])) == law_key
    )
    law_result = law_full_row["authoritative_result"]
    tangent_full_row = next(
        row for row in full_rows if _candidate_key(np.asarray(row["eta"])) == tangent_key
    )
    tangent_result = tangent_full_row["authoritative_result"]
    full_result = full_row["authoritative_result"]
    full_eta = np.asarray(full_row["eta"], dtype=np.float64)
    full_risk = _metric_mean(full_result, "risk")

    rows = []
    for row in evaluated.values():
        candidate_risk = _metric_mean(row["risk_result"], "risk")
        rows.append({
            "eta": row["eta"],
            "provenance": row["provenance"],
            "risk": candidate_risk,
            "tangent_proxy": _metric_mean(row["risk_result"], "tangent_proxy"),
            "full_proxy": (
                _metric_mean(row["full_proxy_result"], "full_proxy")
                if "full_proxy_result" in row else None
            ),
            "valid": row["risk_result"]["valid"],
            "risk_feasible": bool(candidate_risk <= risk_limit + risk_tolerance),
        })
    write_json_atomic(dirs["results"] / "selection_candidates.json", {"candidates": rows})
    reference_receipt = json.loads(reference_receipt_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_hidden_validation",
        "experiment": cfg["name"],
        "config": cfg,
        "config_hash": config_hash(cfg),
        "random_seeds": {"root": int(cfg["seed"]), "selection_namespace": 6201},
        "physical_target_parameters": cfg["truth"],
        "endpoint_reference": {
            "checkpoint_sha256": reference_receipt["checkpoint_sha256"],
            "rollout_sha256": reference_receipt["rollout_sha256"],
            "endpoint_sha256": reference_receipt["endpoint_sha256"],
            "training_inputs": ["x0", "x1"],
        },
        "prospective_aggregate_predictor": {
            "artifact_id": data.artifact_id,
            "artifact_sha256": file_sha256(aggregate_path),
            "raw_intermediate_states_exposed": False,
        },
        "risk_definition": {
            "kind": "time_integrated_whitened_aggregate_qoi_squared_error",
            "qoi_names": cfg["qoi"]["names"],
            "qoi_scales": data.qoi_scales.tolist(),
            "optimized_sensor_channels_in_risk": False,
        },
        "risk_allowance": allowance,
        "risk_limit": risk_limit,
        "selection_input_hashes": signature,
        "selected": {
            "Law": {
                "eta": law_eta.tolist(),
                "centers": law_eta.reshape((-1, 2)).tolist(),
                "predicted": law_result,
            },
            "Tangent": {
                "eta": tangent_eta.tolist(),
                "centers": tangent_eta.reshape((-1, 2)).tolist(),
                "predicted": tangent_result,
                "selection_objective": "particle_tangent_action",
                "risk_increase_relative_to_law": (
                    _metric_mean(tangent_result, "risk") / law_risk - 1.0
                ),
            },
            "Full": {
                "eta": full_eta.tolist(),
                "centers": full_eta.reshape((-1, 2)).tolist(),
                "predicted": full_result,
                "risk_increase_relative_to_law": full_risk / law_risk - 1.0,
                "risk_budget_fraction_used": (full_risk / law_risk - 1.0) / allowance if allowance > 0 else 0.0,
            },
        },
        "selection_metrics": {
            "law_risk": law_risk,
            "full_risk": full_risk,
            "tangent_risk": _metric_mean(tangent_result, "risk"),
            "law_full_action": _metric_mean(law_result, "full_action"),
            "tangent_tangent_action": _metric_mean(tangent_result, "tangent_proxy"),
            "tangent_full_action": _metric_mean(tangent_result, "full_action"),
            "full_full_action": _metric_mean(full_result, "full_action"),
            "generated_candidates": len(evaluated),
            "risk_feasible_candidates": len(feasible),
            "valid_full_proxy_candidates": len(proxy_rows),
            "authoritative_full_candidates": len(full_rows),
        },
        "hidden_validation_loaded": False,
        "selection_elapsed_seconds": time.perf_counter() - started,
        "software": software_metadata(),
    }
    write_json_atomic(manifest_path, manifest)
    print(f"[selection] frozen manifest written: {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select(load_config(args.config), args.output_dir)


if __name__ == "__main__":
    main()
