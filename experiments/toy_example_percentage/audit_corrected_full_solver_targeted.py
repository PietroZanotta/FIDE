"""Targeted physical-q Full-solver audit for toy 1% candidates."""
from __future__ import annotations

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
from corrected_full_solver_targeted_audit import run_targeted_audit, save_targeted_outputs


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths}


def main() -> None:
    pareto = SCRIPT_DIR / "outputs" / "pareto"
    point, first = _strict_common_artifacts(pareto)
    exp, bank, times = _load_experiment(point, first["config"])
    candidates = [
        row
        for row in load_pareto_candidates(
            pareto,
            selection_key=lambda result, method: np.deg2rad(
                np.asarray(
                    result["selection"][f"{method}_optimum_deg"], dtype=np.float64
                )
            ),
        )
        if float(row["allowance_percent"]) == 1.0
    ]
    watched = [
        pareto / "pareto.json",
        *sorted(pareto.glob("risk_*pct/result.json")),
        *sorted(pareto.glob("risk_*pct/*.npz")),
    ]
    before = _snapshot(watched)

    def evaluate(geometry: Any, method: str):
        return exp.evaluate_common_discretization_decomposition_exact(
            jnp.asarray(geometry, dtype=jnp.float64),
            bank,
            progress_desc=f"corrected toy 1% {method}",
        )

    tolerance = float(first["config"]["validity"]["tangent_lower_bound_tol"])
    rows, detail, summary = run_targeted_audit(
        candidates,
        evaluate=evaluate,
        time_grid=times,
        time_weights=np.asarray(exp.time_w, dtype=np.float64),
        moment_tolerance=tolerance,
        energy_tolerance=tolerance,
    )
    after = _snapshot(watched)
    summary.update(
        {
            "experiment": "toy_example_percentage",
            "allowance_percent": 1.0,
            "saved_candidates_and_banks_unchanged": before == after,
            "watched_hashes_before_after": {
                path: {"before": digest, "after": after[path]}
                for path, digest in before.items()
            },
        }
    )
    outputs = save_targeted_outputs(rows, detail, summary, output_dir=pareto)
    print(json.dumps(summary, indent=2), flush=True)
    for output in outputs:
        print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
