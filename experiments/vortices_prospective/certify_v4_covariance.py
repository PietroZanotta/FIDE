from __future__ import annotations

"""Post-freeze covariance-conditioning certificate for prospective v4.

This utility is deliberately outside the selection pipeline.  It reads a completed
freeze and the already-generated validation bank, recomputes only the information-
projection covariance matrices, and cannot change or rank sensor geometries.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common import artifact_dirs, load_config, write_json_atomic
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from v4_validate import _realized_bank_and_moments

jax.config.update("jax_enable_x64", True)


def _condition_rows(covariance: np.ndarray, ridge: float) -> list[dict[str, Any]]:
    eigs = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for trial in range(eigs.shape[0]):
        trial_eigs = eigs[trial]
        minimum = np.min(trial_eigs, axis=-1)
        maximum = np.max(trial_eigs, axis=-1)
        raw = np.divide(
            maximum,
            minimum,
            out=np.full_like(maximum, np.inf),
            where=minimum > 0.0,
        )
        regularized = (maximum + ridge) / (minimum + ridge)
        rows.append(
            {
                "trial": trial,
                "min_covariance_eigenvalue": float(np.min(minimum)),
                "max_covariance_eigenvalue": float(np.max(maximum)),
                "max_raw_covariance_condition_number": float(np.max(raw)),
                "max_ridge_regularized_condition_number": float(np.max(regularized)),
            }
        )
    return rows


def _project_conditions(
    evaluator: ProspectiveEvaluator,
    eta: np.ndarray,
    response_mean: np.ndarray,
    response_second: np.ndarray,
    bank: AggregateObservationBank,
    response_cross_second=None,
) -> list[dict[str, Any]]:
    projection, *_ = evaluator._project(
        eta,
        response_mean,
        response_second,
        bank,
        response_cross_second=response_cross_second,
    )
    return _condition_rows(
        np.asarray(projection.covariance),
        float(evaluator.cfg["particle_mfsi"]["covariance_ridge"]),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(rows),
        "minimum_covariance_eigenvalue": min(
            row["min_covariance_eigenvalue"] for row in rows
        ),
        "maximum_covariance_eigenvalue": max(
            row["max_covariance_eigenvalue"] for row in rows
        ),
        "maximum_raw_covariance_condition_number": max(
            row["max_raw_covariance_condition_number"] for row in rows
        ),
        "maximum_ridge_regularized_condition_number": max(
            row["max_ridge_regularized_condition_number"] for row in rows
        ),
    }


def certify(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dirs = artifact_dirs(output_dir)
    manifest_path = dirs["results"] / "frozen_manifest.json"
    validation_path = dirs["results"] / "validation_result.json"
    finalists_path = dirs["results"] / "v4_finalists.json"
    if not manifest_path.exists():
        raise RuntimeError("covariance certification requires the frozen v4 manifest")
    if not validation_path.exists():
        raise RuntimeError("covariance certification requires completed validation")
    if not finalists_path.exists():
        raise RuntimeError("covariance certification requires frozen finalists")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("unexpected v4 freeze status")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation["frozen_manifest_sha256"] != file_sha256(manifest_path):
        raise RuntimeError("validation result is not tied to the current frozen manifest")

    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(
        cfg, data, dirs["endpoint"] / "reference_rollout.npz"
    )

    with np.load(dirs["prospective"] / "v4_selection_crn.npz", allow_pickle=False) as crn:
        selection_bank = AggregateObservationBank(
            np.asarray(crn["sampling_z"], dtype=np.float64),
            np.asarray(crn["detector_z"], dtype=np.float64),
        )
    finalists = json.loads(finalists_path.read_text(encoding="utf-8"))
    selection_rows = []
    selection_candidates = [("Law", finalists["Law"])] + [
        ("Full", row) for row in finalists["Full_finalists"]
    ]
    for method, candidate in selection_candidates:
        eta = np.asarray(candidate["eta"], dtype=np.float64)
        mean, second = evaluator.prospective_population(jnp.asarray(eta))
        rows = _project_conditions(
            evaluator,
            eta,
            np.asarray(mean),
            np.asarray(second),
            selection_bank,
            response_cross_second=evaluator.prospective_cross_second(eta),
        )
        selection_rows.append(
            {
                "method": method,
                "candidate_id": candidate["candidate_id"],
                "source": candidate["source"],
                "eta": candidate["eta"],
                "summary": _summary(rows),
                "trial_certificates": rows,
            }
        )

    with np.load(dirs["hidden"] / "v4_hidden_state_bank.npz", allow_pickle=False) as hidden:
        states = np.asarray(hidden["states"], dtype=np.float64)
    with np.load(
        dirs["hidden"] / "v4_hidden_observation_randomness.npz", allow_pickle=False
    ) as randomness:
        sample_indices = np.asarray(randomness["sample_indices"], dtype=np.int32)
        detector_z = np.asarray(randomness["detector_z"], dtype=np.float64)

    validation_rows = []
    for method in ("Law", "Full"):
        eta = np.asarray(manifest["selected"][method]["eta"], dtype=np.float64)
        bank, mean, second, _ = _realized_bank_and_moments(
            evaluator, eta, states, sample_indices, detector_z
        )
        rows = _project_conditions(evaluator, eta, mean, second, bank)
        validation_rows.append(
            {
                "method": method,
                "eta": eta.tolist(),
                "summary": _summary(rows),
                "trial_certificates": rows,
            }
        )

    result = {
        "schema_version": 1,
        "role": "post_freeze_read_only_covariance_conditioning_certificate",
        "selection_or_geometry_changed": False,
        "frozen_manifest_sha256": file_sha256(manifest_path),
        "validation_result_sha256": file_sha256(validation_path),
        "covariance_ridge": float(cfg["particle_mfsi"]["covariance_ridge"]),
        "selection_authoritative_finalists": selection_rows,
        "fresh_validation": validation_rows,
    }
    out = dirs["results"] / "covariance_conditioning.json"
    write_json_atomic(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = certify(load_config(args.config), args.output_dir)
    print(json.dumps({
        "output": str(Path(args.output_dir) / "results" / "covariance_conditioning.json"),
        "selection_candidates": len(result["selection_authoritative_finalists"]),
        "validation_methods": len(result["fresh_validation"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
