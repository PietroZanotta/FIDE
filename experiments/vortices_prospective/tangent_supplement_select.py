from __future__ import annotations

"""Gradient-optimize and freeze a risk-constrained Tangent supplement.

This module has no validation imports and never opens a hidden-data path.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common import SCRIPT_DIR, artifact_dirs, config_hash, fingerprint, geometry_valid, load_config, write_json_atomic
from evaluator import ProspectiveEvaluator
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from v4_objective import V4CRNBank, V4DifferentiableObjective, canonical_geometry_key, distribution
from v4_select import _adam_multistart, _exact_risk_rows, generate_full_starts

jax.config.update("jax_enable_x64", True)


def _load_bank(path: Path) -> V4CRNBank:
    with np.load(path, allow_pickle=False) as data:
        return V4CRNBank(
            np.asarray(data["sampling_z"], dtype=np.float64),
            np.asarray(data["detector_z"], dtype=np.float64),
        )


def _tangent_trials(objective, eta, sampling_z, detector_z):
    projection, _, _, tangent, _, risks = objective._project(
        eta, sampling_z, detector_z
    )
    action = jnp.sum(objective.evaluator.time_weights[None, :] * tangent, axis=1)
    residual = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
    ess = jnp.min(projection.ess_fraction, axis=1)
    return action, risks, residual, ess


def _constrained_tangent_loss(objective, cfg, eta, sampling_z, detector_z, risk_limit):
    action, risks, residual, ess = _tangent_trials(
        objective, eta, sampling_z, detector_z
    )
    block = cfg["v4"]
    scale = float(block["constraint_softplus_scale"])
    positive = lambda x: scale * jax.nn.softplus(x / scale)
    risk_violation = positive(jnp.mean(risks) - float(risk_limit))
    numerical = (
        positive(jnp.max(residual) - float(cfg["validity"]["max_projection_residual"])) ** 2
        + positive(float(cfg["validity"]["min_ess_fraction"]) - jnp.min(ess)) ** 2
    )
    return (
        jnp.mean(action)
        + float(block["risk_penalty"]) * risk_violation * risk_violation
        + float(block["numerical_penalty"]) * numerical
    )


def select_tangent(protocol_path: str | Path) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    primary_config_path = (SCRIPT_DIR / protocol["primary_config"]).resolve()
    output_dir = (SCRIPT_DIR / protocol["primary_output"]).resolve()
    cfg = load_config(primary_config_path)
    dirs = artifact_dirs(output_dir)
    primary_manifest_path = dirs["results"] / "frozen_manifest.json"
    if not primary_manifest_path.exists():
        raise RuntimeError("Tangent selection requires the frozen primary Law/Full manifest")
    primary = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if primary.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("primary manifest is not sealed")
    mode = str(protocol["mode"])
    hidden_exists = dirs["hidden"].exists() and any(dirs["hidden"].iterdir())
    if mode == "prospective" and hidden_exists:
        raise RuntimeError("prospective Tangent selection refuses to run after hidden data exists")

    supplement_dir = dirs["results"] / "tangent_supplement"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = supplement_dir / "frozen_manifest.json"
    signature = {
        "primary_manifest_sha256": file_sha256(primary_manifest_path),
        "primary_config_hash": config_hash(cfg),
        "protocol_sha256": file_sha256(protocol_path),
        "selection_crn_sha256": file_sha256(dirs["prospective"] / "v4_selection_crn.npz"),
        "source_sha256": file_sha256(Path(__file__)),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_input_hashes") == signature:
            print("[tangent-supplement] reusing compatible frozen supplement", flush=True)
            return existing
        raise RuntimeError("incompatible Tangent supplement already exists")

    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    objective = V4DifferentiableObjective(
        cfg, data, dirs["endpoint"] / "reference_rollout.npz"
    )
    authoritative = ProspectiveEvaluator(
        cfg, data, dirs["endpoint"] / "reference_rollout.npz"
    )
    bank = _load_bank(dirs["prospective"] / "v4_selection_crn.npz")
    law_eta = np.asarray(primary["selected"]["Law"]["eta"], dtype=np.float64)
    law_risk = float(primary["selection_metrics"]["law_risk"])
    risk_limit = float(primary["selection_metrics"]["risk_ceiling"])

    adapted = copy.deepcopy(cfg)
    adapted["v4"]["full_optimizer"].update(
        {
            "starts": int(protocol["starts"]),
            "law_perturbation_starts": int(protocol["law_perturbation_starts"]),
            "law_perturbation_scale": float(protocol["law_perturbation_scale"]),
            "start_oversample": int(protocol["start_oversample"]),
        }
    )
    adapted["seeds"]["full_global_starts"] = int(protocol["seeds"]["global_starts"])
    adapted["seeds"]["full_law_perturbations"] = int(protocol["seeds"]["law_perturbations"])
    starts, provenance = generate_full_starts(adapted, law_eta, tangent_eta=None)
    provenance = [source.replace("Full", "Tangent") for source in provenance]
    settings = {
        key: protocol[key]
        for key in ("steps", "batch_size", "learning_rate", "beta1", "beta2", "eps")
    }
    started = time.perf_counter()
    optimize_bank = bank.prefix(min(int(protocol["crn_trials"]), bank.trials))
    runs = _adam_multistart(
        starts,
        provenance,
        optimize_bank,
        settings,
        cfg,
        lambda eta, s, d: _constrained_tangent_loss(
            objective, cfg, eta, s, d, risk_limit
        ),
        schedule_seed=int(protocol["seeds"]["batch_schedule"]),
        stage="tangent-adam",
    )
    candidates = [
        {
            "candidate_id": f"tangent-grad-{index:03d}",
            "source": run["provenance"],
            "eta": run["final_eta"],
            "gradient_run": run,
        }
        for index, run in enumerate(runs)
    ]
    unique: dict[tuple[float, ...], dict[str, Any]] = {}
    for row in candidates:
        unique.setdefault(canonical_geometry_key(row["eta"]), row)
    candidates = list(unique.values())
    _exact_risk_rows(authoritative, candidates, bank, cfg)
    feasible = []
    for row in candidates:
        result = row["authoritative_risk_result"]
        row["tangent_distribution"] = distribution(
            [trial["tangent_proxy"] for trial in result["trials"] if trial["valid"]]
        )
        row["risk_feasible"] = bool(
            row["geometry_valid"]
            and result["valid"]
            and float(row["risk"]) <= risk_limit + float(cfg["validity"]["risk_constraint_tolerance"])
        )
        if row["risk_feasible"]:
            feasible.append(row)
    if not feasible:
        raise RuntimeError("Tangent supplement produced no exact-risk-feasible candidate")
    selected = min(feasible, key=lambda row: row["tangent_distribution"]["mean"])
    selected_result = authoritative.evaluate_prospective(
        np.asarray(selected["eta"]), bank.as_observation_bank(), compute_full=True
    )
    selected_summary = {
        **{key: value for key, value in selected.items() if key != "authoritative_risk_result"},
        "centers": np.asarray(selected["eta"]).reshape((-1, 2)).tolist(),
        "authoritative_result": selected_result,
    }
    archive = {
        "schema_version": 1,
        "gradient_runs": runs,
        "candidates": candidates,
    }
    archive_path = supplement_dir / "candidate_archive.json"
    write_json_atomic(archive_path, archive)
    manifest = {
        "schema_version": 1,
        "experiment": str(protocol["name"]),
        "mode": mode,
        "status": (
            "frozen_before_hidden_validation"
            if mode == "prospective"
            else "posthoc_selection_after_primary_hidden_but_hidden_not_loaded"
        ),
        "interpretation": protocol["interpretation"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_input_hashes": signature,
        "hidden_data_imported_or_loaded_by_selection": False,
        "primary_hidden_existed_at_selection": hidden_exists,
        "protocol": protocol,
        "risk_anchor": law_risk,
        "risk_ceiling": risk_limit,
        "gradient_starts": len(runs),
        "distinct_candidates": len(candidates),
        "risk_feasible_candidates": len(feasible),
        "selected": selected_summary,
        "candidate_archive_sha256": file_sha256(archive_path),
        "selection_elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"[tangent-supplement] frozen {manifest_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    select_tangent(args.protocol)


if __name__ == "__main__":
    main()
