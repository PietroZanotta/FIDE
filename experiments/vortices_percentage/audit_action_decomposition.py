"""Audit exact Tangent/Full decomposition for saved vortex Pareto candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
EXPERIMENTS_DIR = SCRIPT_DIR.parent
for path in (SRC_DIR, EXPERIMENTS_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from action_decomposition_audit import (
    audit_candidates,
    build_summary,
    file_sha256,
    load_pareto_candidates,
    save_outputs,
)
from bounded_reference import BoxTransformedReferenceFlow
from experiment import ObservationTrialBank, VortexExperiment


DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    return parser.parse_args()


def _evaluator_signature(cfg: dict[str, Any]) -> str:
    payload = {
        key: cfg.get(key)
        for key in (
            "seed", "truth", "measurement", "moment_reconstruction", "projection",
            "particle_mfsi", "raster", "poisson", "validity", "reference",
            "randomness",
        )
    }
    opt = cfg.get("optimization", {})
    payload["exact_optimization"] = {
        key: value
        for key, value in opt.items()
        if key.startswith("full_exact_") or key == "exact_batch_trials"
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _strict_common_artifacts(pareto_dir: Path) -> tuple[Path, dict[str, Any]]:
    point_dirs = sorted(path.parent for path in pareto_dir.glob("risk_*pct/result.json"))
    if not point_dirs:
        raise FileNotFoundError(f"no saved risk_*pct results under {pareto_dir}")
    artifact_names = (
        "truth_bank.npz", "reference_endpoints.npz", "reference.npz",
        "reference_bank.npz", "selection_bank.npz",
    )
    expected_hashes = {
        name: file_sha256(point_dirs[0] / name) for name in artifact_names
    }
    first_result = json.loads((point_dirs[0] / "result.json").read_text(encoding="utf-8"))
    expected_signature = _evaluator_signature(first_result["config"])
    for point in point_dirs[1:]:
        result = json.loads((point / "result.json").read_text(encoding="utf-8"))
        if _evaluator_signature(result["config"]) != expected_signature:
            raise RuntimeError(f"authoritative evaluator settings differ at {point}")
        for name, expected in expected_hashes.items():
            if file_sha256(point / name) != expected:
                raise RuntimeError(f"frozen artifact {name} differs at {point}")
    return point_dirs[0], first_result


def _load_experiment(
    point: Path, cfg: dict[str, Any]
) -> tuple[VortexExperiment, ObservationTrialBank, np.ndarray]:
    reference = BoxTransformedReferenceFlow.from_npz(
        point / "reference.npz",
        substeps_per_interval=int(cfg["reference"]["rk4_substeps_per_time_interval"]),
    )
    with np.load(point / "truth_bank.npz", allow_pickle=False) as truth:
        times = np.asarray(truth["times"], dtype=np.float64)
        particles = jnp.asarray(truth["particles"], dtype=jnp.float64)
    with np.load(point / "reference_bank.npz", allow_pickle=False) as bank:
        ref_times = np.asarray(bank["times"], dtype=np.float64)
        nodes = jnp.asarray(bank["nodes"], dtype=jnp.float64)
        velocity = jnp.asarray(bank["velocity"], dtype=jnp.float64)
        weights = jnp.asarray(bank["weights"], dtype=jnp.float64)
    np.testing.assert_allclose(ref_times, times, rtol=0.0, atol=0.0)
    exp = VortexExperiment(
        cfg,
        reference,
        truth_particles=particles,
        reference_nodes=nodes,
        reference_velocity=velocity,
        reference_weights=weights,
    )
    with np.load(point / "selection_bank.npz", allow_pickle=False) as bank:
        selection = ObservationTrialBank(
            sample_indices=jnp.asarray(bank["sample_indices"], dtype=jnp.int32),
            detector_z=jnp.asarray(bank["detector_z"], dtype=jnp.float64),
        )
    action_trials = int(cfg["randomness"]["action_trials"])
    if int(selection.sample_indices.shape[0]) < action_trials:
        raise RuntimeError("saved selection bank is shorter than action_trials")
    action_bank = ObservationTrialBank(
        sample_indices=selection.sample_indices[:action_trials],
        detector_z=selection.detector_z[:action_trials],
    )
    np.testing.assert_allclose(times, np.asarray(exp.times), rtol=0.0, atol=0.0)
    return exp, action_bank, times


def main() -> None:
    args = parse_args()
    pareto_dir = args.pareto_dir.expanduser().resolve()
    point, first_result = _strict_common_artifacts(pareto_dir)
    cfg = first_result["config"]
    exp, action_bank, times = _load_experiment(point, cfg)
    candidates = load_pareto_candidates(
        pareto_dir,
        selection_key=lambda result, method: result["selection"][f"{method}_optimum"],
    )
    tolerance = float(cfg.get("validity", {}).get("tangent_lower_bound_tol", 1.0e-6))

    def evaluate(geometry: Any, key: str) -> list[dict[str, Any]]:
        print(f"[audit] exact shared-weight evaluation eta={geometry}", flush=True)
        return exp.evaluate_trials_exact(
            jnp.asarray(geometry, dtype=jnp.float64),
            action_bank,
            progress_desc=f"action decomposition {key[:24]}",
        )

    rows, evaluations = audit_candidates(
        candidates,
        evaluate=evaluate,
        tolerance=tolerance,
        time_grid=times,
    )
    selection_path = point / "selection_bank.npz"
    summary = build_summary(
        rows,
        experiment="vortices_percentage",
        tolerance=tolerance,
        selection_bank_path=selection_path,
        time_grid=times,
        evaluator_description=(
            "VortexExperiment.evaluate_trials_exact on the frozen action-selection bank; "
            "exact shared reconstruction/I-projection, full 21-node time grid, and "
            "reported 128x64 native full-Poisson settings"
        ),
    )
    outputs = save_outputs(rows, evaluations, summary, output_dir=pareto_dir)
    print(json.dumps(summary, indent=2), flush=True)
    for path in outputs:
        print(f"saved={path}", flush=True)
    if not summary["every_final_candidate_passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
