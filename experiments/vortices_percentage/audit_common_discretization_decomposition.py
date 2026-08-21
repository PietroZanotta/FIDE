"""Common-raster decomposition audit for saved vortex Pareto candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", SCRIPT_DIR.parent, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from action_decomposition_audit import file_sha256, load_pareto_candidates
from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from common_discretization_decomposition_audit import (
    audit_common_discretization,
    save_common_discretization_outputs,
)


DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto"


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    args = parser.parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    point, first = _strict_common_artifacts(pareto)
    exp, bank, times = _load_experiment(point, first["config"])
    candidates = load_pareto_candidates(
        pareto,
        selection_key=lambda result, method: result["selection"][f"{method}_optimum"],
    )
    candidate_files = [
        pareto / "pareto.json",
        *sorted(pareto.glob("risk_*pct/result.json")),
        *sorted(pareto.glob("risk_*pct/result.candidate_summary.csv")),
    ]
    bank_files = sorted(pareto.glob("risk_*pct/*.npz"))
    before_candidates = _snapshot(candidate_files)
    before_banks = _snapshot(bank_files)

    def evaluate(geometry: Any, key: str) -> list[dict[str, Any]]:
        print(f"[audit] common raster eta={geometry}", flush=True)
        return exp.evaluate_common_discretization_decomposition_exact(
            jnp.asarray(geometry, dtype=jnp.float64),
            bank,
            progress_desc=f"common raster {key[:24]}",
        )

    tolerance = float(first["config"]["validity"]["tangent_lower_bound_tol"])
    rows, detail, summary = audit_common_discretization(
        candidates,
        evaluate=evaluate,
        time_grid=times,
        time_weights=np.asarray(exp.time_w, dtype=np.float64),
        tolerance=tolerance,
    )
    after_candidates = _snapshot(candidate_files)
    after_banks = _snapshot(bank_files)
    summary.update(
        {
            "experiment": "vortices_percentage",
            "authoritative_evaluator": "VortexExperiment authoritative exact raster weighted-Poisson evaluator",
            "raster_grid": {
                "shape": [int(exp.grid.ny), int(exp.grid.nx)],
                "dx": float(exp.grid.dx),
                "dy": float(exp.grid.dy),
            },
            "time_grid": np.asarray(times, dtype=np.float64).tolist(),
            "time_weights": np.asarray(exp.time_w, dtype=np.float64).tolist(),
            "selection_bank": str((point / "selection_bank.npz").resolve()),
            "selection_bank_sha256": file_sha256(point / "selection_bank.npz"),
            "poisson_operator_floor_rel": float(
                first["config"]["poisson"]["operator_floor_rel"]
            ),
            "saved_candidate_geometries_unchanged": before_candidates == after_candidates,
            "frozen_banks_unchanged": before_banks == after_banks,
            "candidate_file_hashes_before_after": {
                path: {"before": digest, "after": after_candidates[path]}
                for path, digest in before_candidates.items()
            },
            "frozen_bank_hashes_before_after": {
                path: {"before": digest, "after": after_banks[path]}
                for path, digest in before_banks.items()
            },
        }
    )
    outputs = save_common_discretization_outputs(
        rows, detail, summary, output_dir=pareto
    )
    print(json.dumps(summary, indent=2), flush=True)
    for output in outputs:
        print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
