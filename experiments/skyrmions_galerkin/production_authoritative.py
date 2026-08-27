"""Fresh fixed-design Deep Ritz cross-check after validated Galerkin refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from .deep_ritz import load_ritz_checkpoint
from .production_artifacts import require_production_output_path
from .production_workflow import load_production_data
from .workflow import authoritative_evaluate, save_candidate_checkpoint, write_json


def run_production_authoritative_crosscheck(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path,
    *, allowance_percent: float,
) -> dict[str, Any]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refinement_path = output_dir.parent / "refinement" / "result.json"
    if not refinement_path.is_file():
        raise RuntimeError("production refinement result is missing")
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    if not refinement.get("eligible_for_authoritative_crosscheck"):
        result = {
            "ran": False,
            "reason": "refinement did not produce an eligible Galerkin decrease",
            "outcome_classification": "A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED",
        }
        write_json(output_dir / "result.json", result)
        return result
    data = load_production_data(cfg, artifact_dir)
    initial_params, initial_metadata = load_ritz_checkpoint(
        artifact_dir / "ritz_full.npz"
    )
    eta0 = jnp.asarray(refinement["start_eta"], dtype=jnp.float64)
    eta1 = jnp.asarray(refinement["end_eta"], dtype=jnp.float64)
    start = authoritative_evaluate(
        eta0, cfg, data, allowance_percent=allowance_percent,
        initial_params=initial_params, validation=False,
    )
    write_json(output_dir / "eta0.json", start.payload)
    if start.params is not None:
        save_candidate_checkpoint(
            output_dir / "eta0_fresh.npz", start, role="production_crosscheck_eta0"
        )
    partial = {
        "ran": True,
        "in_progress": True,
        "initial_checkpoint_metadata": initial_metadata,
        "eta0": start.payload,
    }
    write_json(output_dir / "result.json", partial)
    end = authoritative_evaluate(
        eta1, cfg, data, allowance_percent=allowance_percent,
        initial_params=initial_params, validation=False,
    )
    write_json(output_dir / "eta1.json", end.payload)
    if end.params is not None:
        save_candidate_checkpoint(
            output_dir / "eta1_fresh.npz", end, role="production_crosscheck_eta1"
        )
    tolerance = float(cfg["envelope"].get("minimum_improvement", 1.0e-6))
    authoritative_improvement = bool(
        start.payload.get("valid", False)
        and end.payload.get("valid", False)
        and float(end.payload["risk"]) <= float(end.payload["risk_limit"])
        and float(end.payload["action"]) < float(start.payload["action"]) - tolerance
    )
    result = {
        "ran": True,
        "in_progress": False,
        "initial_checkpoint_metadata": initial_metadata,
        "eta0": start.payload,
        "eta1": end.payload,
        "replacement_tolerance": tolerance,
        "authoritative_action_difference_eta1_minus_eta0": (
            float(end.payload["action"]) - float(start.payload["action"])
        ),
        "authoritative_improvement": authoritative_improvement,
        "incumbent_replaced": False,
        "outcome_classification": "A. PRODUCTION GALERKIN SOLVER AND ETA GRADIENT VALIDATED",
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = ["run_production_authoritative_crosscheck"]
