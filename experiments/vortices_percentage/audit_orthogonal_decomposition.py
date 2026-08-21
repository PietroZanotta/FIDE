"""Direct correction-field decomposition audit for saved vortex candidates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", SCRIPT_DIR.parent, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_decomposition_audit import load_pareto_candidates
from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from orthogonal_decomposition_audit import audit_field_decompositions, save_field_audit


def main() -> None:
    pareto = SCRIPT_DIR / "outputs" / "pareto"
    point, first = _strict_common_artifacts(pareto)
    exp, bank, times = _load_experiment(point, first["config"])
    candidates = load_pareto_candidates(
        pareto,
        selection_key=lambda result, method: result["selection"][f"{method}_optimum"],
    )

    def evaluate(geometry: Any, key: str):
        return exp.evaluate_decomposition_exact(
            jnp.asarray(geometry, dtype=jnp.float64),
            bank,
            progress_desc=f"field decomposition {key[:24]}",
        )

    rows, detail, summary = audit_field_decompositions(
        candidates,
        evaluate=evaluate,
        time_grid=times,
        time_weights=np.asarray(exp.time_w, dtype=np.float64),
        tolerance=float(first["config"]["validity"]["tangent_lower_bound_tol"]),
    )
    summary["experiment"] = "vortices_percentage"
    outputs = save_field_audit(rows, detail, summary, output_dir=pareto)
    print(json.dumps(summary, indent=2))
    print("saved", *(str(path) for path in outputs), sep="\n")


if __name__ == "__main__":
    main()
