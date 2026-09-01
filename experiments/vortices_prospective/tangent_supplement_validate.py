from __future__ import annotations

"""Evaluate a frozen Tangent supplement on the primary aligned hidden bank."""

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from common import SCRIPT_DIR, artifact_dirs, load_config, write_json_atomic
from evaluator import ProspectiveEvaluator
from frozen_diagnostic_core import paired_statistics
from mfsi.cache import file_sha256
from prospective_data import TargetProspectiveData
from v4_validate import _realized_bank_and_moments


def _values(result: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        [row[key] for row in result["trials"] if row["valid"] and row[key] is not None],
        dtype=np.float64,
    )


def _certification(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["trials"]
    return {
        "valid_fraction": float(result["valid_fraction"]),
        "invalid_trial_count": int(sum(not row["valid"] for row in rows)),
        "max_projection_residual": max(row["max_projection_residual"] for row in rows),
        "min_ess_fraction": min(row["min_ess_fraction"] for row in rows),
        "min_covariance_eigenvalue": min(row["min_covariance_eigenvalue"] for row in rows),
        "max_poisson_relative_residual": max(row["max_poisson_relative_residual"] for row in rows),
        "max_component_compatibility_residual": max(row["max_component_compatibility_residual"] for row in rows),
        "max_full_moment_rate_residual": max(row["max_full_moment_rate_residual"] for row in rows),
        "all_full_solvers_converged": bool(all(row["full_solver_converged"] for row in rows)),
        "nan_or_inf_count": int(
            sum(
                not np.isfinite(row[key])
                for row in rows
                for key in ("scientific_risk", "tangent_proxy", "full_action")
                if row[key] is not None
            )
        ),
    }


def validate_tangent(protocol_path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    cfg = load_config(SCRIPT_DIR / protocol["primary_config"])
    output_dir = (SCRIPT_DIR / protocol["primary_output"]).resolve()
    dirs = artifact_dirs(output_dir)
    primary_manifest_path = dirs["results"] / "frozen_manifest.json"
    primary_validation_path = dirs["results"] / "validation_result.json"
    supplement_dir = dirs["results"] / "tangent_supplement"
    supplement_manifest_path = supplement_dir / "frozen_manifest.json"
    for path in (primary_manifest_path, primary_validation_path, supplement_manifest_path):
        if not path.exists():
            raise RuntimeError(f"Tangent validation prerequisite missing: {path}")
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    primary_validation = json.loads(primary_validation_path.read_text(encoding="utf-8"))
    supplement = json.loads(supplement_manifest_path.read_text(encoding="utf-8"))
    primary_sha = file_sha256(primary_manifest_path)
    if primary_validation["frozen_manifest_sha256"] != primary_sha:
        raise RuntimeError("primary validation is not tied to the current manifest")
    if supplement["selection_input_hashes"]["primary_manifest_sha256"] != primary_sha:
        raise RuntimeError("Tangent supplement is not tied to the current primary manifest")
    if protocol["mode"] == "prospective":
        if supplement["status"] != "frozen_before_hidden_validation":
            raise RuntimeError("prospective Tangent supplement was not frozen pre-hidden")
        if supplement["primary_hidden_existed_at_selection"]:
            raise RuntimeError("prospective Tangent supplement saw a pre-existing hidden bank")

    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(
        cfg, data, dirs["endpoint"] / "reference_rollout.npz"
    )
    with np.load(dirs["hidden"] / "v4_hidden_state_bank.npz", allow_pickle=False) as hidden:
        states = np.asarray(hidden["states"], dtype=np.float64)
    with np.load(
        dirs["hidden"] / "v4_hidden_observation_randomness.npz", allow_pickle=False
    ) as randomness:
        sample_indices = np.asarray(randomness["sample_indices"], dtype=np.int32)
        detector_z = np.asarray(randomness["detector_z"], dtype=np.float64)
    eta = np.asarray(supplement["selected"]["eta"], dtype=np.float64)
    bank, mean, second, qoi_targets = _realized_bank_and_moments(
        evaluator, eta, states, sample_indices, detector_z
    )
    tangent_result = evaluator.evaluate_population(
        jnp.asarray(eta), mean, second, qoi_targets, bank, compute_full=True
    )
    law_result = primary_validation["methods"]["Law"]["realized"]
    full_result = primary_validation["methods"]["Full"]["realized"]
    law_action = _values(law_result, "full_action")
    full_action = _values(full_result, "full_action")
    tangent_action = _values(tangent_result, "full_action")
    law_risk = float(law_result["risk"]["mean"])
    tangent_risk = float(tangent_result["risk"]["mean"])
    result = {
        "schema_version": 1,
        "experiment": protocol["name"],
        "mode": protocol["mode"],
        "interpretation": supplement["interpretation"],
        "primary_manifest_sha256": primary_sha,
        "supplement_manifest_sha256": file_sha256(supplement_manifest_path),
        "aligned_common_randomness_across_law_tangent_full": True,
        "selection_geometry_changed_by_validation": False,
        "Tangent": {
            "eta": eta.tolist(),
            "centers": eta.reshape((-1, 2)).tolist(),
            "predicted": supplement["selected"]["authoritative_result"],
            "realized": tangent_result,
            "certification": _certification(tangent_result),
            "realized_risk_within_law_relative_allowance": bool(
                tangent_risk <= (1.0 + float(cfg["risk_allowance"])) * law_risk
            ),
        },
        "paired_tangent_minus_law": paired_statistics(
            law_action,
            tangent_action,
            bootstrap_seed=int(protocol["seeds"]["validation_bootstrap"]),
        ),
        "paired_full_minus_tangent": paired_statistics(
            tangent_action,
            full_action,
            bootstrap_seed=int(protocol["seeds"]["validation_bootstrap"]) + 1,
        ),
    }
    result_path = supplement_dir / "validation_result.json"
    write_json_atomic(result_path, result)
    tangent = result["Tangent"]
    tl = result["paired_tangent_minus_law"]
    ft = result["paired_full_minus_tangent"]
    lines = [
        f"# {protocol['name'].replace('_', ' ')}",
        "",
        str(result["interpretation"]),
        "",
        "## Tangent geometry",
        "",
        f"Centers: `{tangent['centers']}`.",
        "",
        "## Aligned held-out Full-action comparison",
        "",
        f"Tangent realized mean Full action: `{tangent_result['full_action']['mean']:.6g}`; "
        f"mean scientific risk: `{tangent_result['risk']['mean']:.6g}`; "
        f"risk allowance: `{'PASS' if tangent['realized_risk_within_law_relative_allowance'] else 'FAIL'}`.",
        "",
        f"Tangent - Law paired mean: `{tl['difference_full_minus_law']['mean']:.6g}`, "
        f"95% t CI `{tl['paired_t_95_ci']}`, Tangent lower in "
        f"`{100.0 * tl['fraction_full_lower']:.1f}%` of trials.",
        "",
        f"Full - Tangent paired mean: `{ft['difference_full_minus_law']['mean']:.6g}`, "
        f"95% t CI `{ft['paired_t_95_ci']}`, Full lower in "
        f"`{100.0 * ft['fraction_full_lower']:.1f}%` of trials.",
        "",
        "The original primary manifest and selected Law/Full geometries were not modified.",
        "",
    ]
    (supplement_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    result = validate_tangent(args.protocol)
    print(json.dumps({
        "mode": result["mode"],
        "tangent_mean_full": result["Tangent"]["realized"]["full_action"]["mean"],
        "risk_pass": result["Tangent"]["realized_risk_within_law_relative_allowance"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
