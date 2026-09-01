from __future__ import annotations

"""Small CUDA end-to-end qualification of the V6a Pareto freeze sequence."""

import copy
import os
from pathlib import Path

import jax

from run_v6_positive_raster import load_repair_config
from run_v6_smoke import smoke_config
from run_v6a_risk_study import run


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "prospective_v6a_risk_study_smoke_v2"
SOURCE = HERE / "outputs" / "prospective_v6_beta_ablation_smoke_v2"


def config():
    cfg = smoke_config()
    repaired, _ = load_repair_config(
        HERE / "configs" / "production_v6_positive_raster_repair.json"
    )
    cfg["raster"] = copy.deepcopy(repaired["raster"])
    cfg["name"] = "prospective_v6a_risk_study_smoke_v2"
    cfg["v6"]["output_name"] = cfg["name"]
    cfg["v6_fast_execution"] = {
        "tangent_start_batch_size": 3,
        "full_start_batch_size": 3,
        "polish_start_batch_size": 1,
        "prescreen_optimize_starts": 3,
        "prescreen_start_batch_size": 3,
    }
    cfg["v6a_risk_study"] = {
        "schema_version": 1,
        "mode": "smoke",
        "beta": 0.0,
        "allowances": [0.02, 0.03],
        "reuse_existing_design_references": True,
        "v6b_excluded": True,
        "selection_completed_at_every_allowance_before_evaluation_references": True,
        "previous_allowance_incumbents_are_mandatory": True,
        "source_run": str(SOURCE.resolve()),
    }
    return cfg


def main() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"V6a risk-study smoke requires CUDA; got {jax.default_backend()}")
    os.environ["V6A_RISK_STUDY_DISABLE_NOTIFY"] = "1"
    result = run(config(), OUTPUT, "all")
    print({
        "points": len(result["points"]),
        "reference_ids": result["reference_ids"],
        "cache": result["exact_geometry_validation_cache"],
    })


if __name__ == "__main__":
    main()
