"""Audit saved Full designs across every frozen reference model.

The candidate set contains only the per-reference Full designs already selected
on the frozen selection bank.  A common design is chosen by its worst normalized
selection action subject to every reference-specific exact law/validity screen.
Only after that choice is frozen is the common design scored on validation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.config import load_config

from domain import DefectPopulationBank, SplitConfig, make_run_split
from eval import paired_bootstrap_reduction
from experiment import ActiveNematicExperiment, make_observation_bank


DEFAULT_MANIFEST = SCRIPT_DIR / "outputs" / "run_guarded" / "manifest_position_polarity.json"
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"


def _split(cfg: dict[str, Any]) -> SplitConfig:
    block = cfg["splits"]
    return SplitConfig(
        train_runs=int(block["train_runs"]),
        design_runs=int(block["design_runs"]),
        validation_runs=int(block["validation_runs"]),
        seed=int(cfg["seed"]) + int(block.get("seed_offset", 1101)),
    )


def _result_path(row: dict[str, Any], manifest: Path) -> Path:
    path = Path(row["result"])
    if path.is_file():
        return path.resolve()
    local = manifest.parent / path.parent.name / path.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError(path)


def _selected_audit(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    rows = [
        row
        for row in result["selection_candidates"]["full_exact"]
        if np.allclose(np.asarray(row["eta"], dtype=np.float64), eta)
    ]
    if not rows:
        raise ValueError(f"saved {design} design has no exact selection audit")
    return rows[0]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    return {
        "trials": len(rows),
        "valid_trials": len(valid),
        "valid": len(valid) == len(rows),
        "law_risk": float(np.mean([row["law_risk"] for row in valid])) if valid else float("nan"),
        "full_action": float(np.mean([row["full_action"] for row in valid])) if valid else float("nan"),
        "max_calibration_residual": float(
            max(row["max_calibration_residual"] for row in rows)
        ),
        "min_ess_fraction": float(min(row["min_ess_fraction"] for row in rows)),
        "max_poisson_relative_residual": float(
            max(row["max_poisson_relative_residual"] for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--time-guard-points", type=int, default=1)
    parser.add_argument("--selection-trials", type=int)
    parser.add_argument("--validation-trials", type=int)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_paths = [_result_path(row, manifest_path) for row in manifest["runs"]]
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    saved_selection_trials = {
        int(result["observation_banks"]["selection"]["trials"])
        for result in results
    }
    saved_validation_trials = {
        int(result["observation_banks"]["validation"]["trials"])
        for result in results
    }
    if len(saved_selection_trials) != 1 or len(saved_validation_trials) != 1:
        raise ValueError("all saved results must use common selection/validation trial counts")
    selection_trials = (
        int(args.selection_trials)
        if args.selection_trials is not None
        else saved_selection_trials.pop()
    )
    validation_trials = (
        int(args.validation_trials)
        if args.validation_trials is not None
        else saved_validation_trials.pop()
    )
    if selection_trials <= 0 or validation_trials <= 0:
        raise ValueError("selection and validation trial counts must be positive")
    cfg = load_config(args.config.expanduser().resolve(), smoke=False)
    cfg.setdefault("evaluation", {})["time_guard_points"] = int(args.time_guard_points)

    mode = str(results[0]["state"]["mode"])
    population = DefectPopulationBank.load(
        manifest_path.parent / f"positive_defect_bank_{mode}.npz"
    )
    split = make_run_split(_split(cfg))
    times = (population.times - population.times[0]) / (
        population.times[-1] - population.times[0]
    )
    truth_n = int(cfg["randomness"].get("truth_particles", 2048))
    design_truth = population.resample_trajectory(
        run_indices=split.design,
        n=truth_n,
        seed=int(cfg["seed"]) + 4001,
    )
    validation_truth = population.resample_trajectory(
        run_indices=split.validation,
        n=truth_n,
        seed=int(cfg["seed"]) + 4002,
    )
    n_observables = int(cfg["measurement"]["n_sensors"]) * len(
        cfg["measurement"]["channels"]
    )
    selection_bank = make_observation_bank(
        seed=int(cfg["seed"]),
        namespace=int(cfg["randomness"].get("selection_namespace", 9890)),
        trials=selection_trials,
        acquisition_count=int(cfg["measurement"]["acquisition_k"]),
        finite_n=int(cfg["measurement"]["finite_n"]),
        truth_particle_count=truth_n,
        n_observables=n_observables,
    )
    validation_bank = make_observation_bank(
        seed=int(cfg["seed"]),
        namespace=int(cfg["randomness"].get("validation_namespace", 9891)),
        trials=validation_trials,
        acquisition_count=int(cfg["measurement"]["acquisition_k"]),
        finite_n=int(cfg["measurement"]["finite_n"]),
        truth_particle_count=truth_n,
        n_observables=n_observables,
    )

    reference_rows = []
    selection_experiments = []
    validation_experiments = []
    for result, result_path in zip(results, result_paths, strict=True):
        with np.load(result_path.parent / "reference_bank.npz", allow_pickle=False) as bank:
            nodes = jnp.asarray(bank["nodes"])
            velocity = jnp.asarray(bank["velocity"])
            weights = jnp.asarray(bank["weights"])
        selection_experiments.append(
            ActiveNematicExperiment(
                cfg,
                times=jnp.asarray(times),
                truth_particles=jnp.asarray(design_truth),
                reference_nodes=nodes,
                reference_velocity=velocity,
                reference_weights=weights,
            )
        )
        validation_experiments.append(
            ActiveNematicExperiment(
                cfg,
                times=jnp.asarray(times),
                truth_particles=jnp.asarray(validation_truth),
                reference_nodes=nodes,
                reference_velocity=velocity,
                reference_weights=weights,
            )
        )
        law_selection = _selected_audit(result, "law")
        reference_rows.append(
            {
                "reference_seed": int(result["reference_seed"]),
                "risk_max": float(result["risk_max"]),
                "law_selection_action": float(law_selection["action"]["value"]),
                "law_validation_action": float(
                    result["validation"]["law"]["summary"]["full_action"]["mean"]
                ),
                "law_validation_trials": result["validation"]["law"]["trials"],
            }
        )

    candidates = []
    for result in results:
        eta = np.asarray(result["designs"]["full"], dtype=np.float64)
        if not any(np.allclose(eta, np.asarray(row["eta"])) for row in candidates):
            candidates.append(
                {"source_reference_seed": int(result["reference_seed"]), "eta": eta.tolist()}
            )

    for candidate_index, candidate in enumerate(candidates):
        eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
        candidate["selection"] = []
        ratios = []
        feasible = True
        for exp, reference in zip(selection_experiments, reference_rows, strict=True):
            print(
                f"cross_reference candidate={candidate_index + 1}/{len(candidates)} "
                f"reference_seed={reference['reference_seed']} phase=selection",
                flush=True,
            )
            summary = _summarize(exp.exact_trial_rows(eta, selection_bank, full=True))
            summary["reference_seed"] = reference["reference_seed"]
            summary["risk_max"] = reference["risk_max"]
            summary["passes_law"] = bool(summary["law_risk"] <= reference["risk_max"])
            summary["action_ratio_to_own_law"] = float(
                summary["full_action"] / reference["law_selection_action"]
            )
            feasible &= bool(summary["valid"] and summary["passes_law"])
            ratios.append(summary["action_ratio_to_own_law"])
            candidate["selection"].append(summary)
        candidate["feasible_all_references"] = feasible
        candidate["selection_action_ratio_mean"] = float(np.mean(ratios))
        candidate["selection_action_ratio_worst"] = float(np.max(ratios))

    feasible_candidates = [row for row in candidates if row["feasible_all_references"]]
    if not feasible_candidates:
        raise RuntimeError("no saved Full design passes every reference-specific selection screen")
    selected = min(
        feasible_candidates,
        key=lambda row: (
            row["selection_action_ratio_worst"],
            row["selection_action_ratio_mean"],
        ),
    )
    selected["validation"] = []
    eta = jnp.asarray(selected["eta"], dtype=jnp.float64)
    for index, (exp, reference) in enumerate(
        zip(validation_experiments, reference_rows, strict=True)
    ):
        print(
            f"cross_reference selected_source={selected['source_reference_seed']} "
            f"reference_seed={reference['reference_seed']} phase=validation",
            flush=True,
        )
        rows = exp.exact_trial_rows(eta, validation_bank, full=True)
        summary = _summarize(rows)
        summary["reference_seed"] = reference["reference_seed"]
        summary["action_ratio_to_own_law"] = float(
            summary["full_action"] / reference["law_validation_action"]
        )
        summary["action_reduction_to_own_law"] = float(
            1.0 - summary["action_ratio_to_own_law"]
        )
        law_by_trial = {
            int(row["trial"]): float(row["full_action"])
            for row in reference["law_validation_trials"]
            if row["valid"] and np.isfinite(row["full_action"])
        }
        full_by_trial = {
            int(row["trial"]): float(row["full_action"])
            for row in rows
            if row["valid"] and np.isfinite(row["full_action"])
        }
        paired = sorted(set(law_by_trial).intersection(full_by_trial))
        summary["paired_bootstrap_reduction_95"] = paired_bootstrap_reduction(
            np.asarray([law_by_trial[key] for key in paired]),
            np.asarray([full_by_trial[key] for key in paired]),
            reps=int(args.bootstrap_reps),
            seed=20260818 + index,
        )
        selected["validation"].append(summary)

    payload = {
        "schema_version": 1,
        "selection_policy": (
            "minimize worst reference-specific Full/own-Law selection action ratio "
            "subject to every exact reference-specific law and validity screen"
        ),
        "manifest": str(manifest_path),
        "time_guard_points": int(args.time_guard_points),
        "selection_trials": selection_trials,
        "validation_trials": validation_trials,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected_common_design": selected,
    }
    output = args.output or (
        manifest_path.parent / "audits" / "cross_reference_full_candidates.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["selected_common_design"], indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
