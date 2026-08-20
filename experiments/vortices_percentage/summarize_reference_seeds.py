"""Summarize held-out vortices results across learned-reference seeds.

The audit is deliberately read-only with respect to experiment runs.  It checks
that the physical/observation banks are byte-identical, computes paired
bootstrap intervals from the saved validation rows, and reports how much the
selected sensor geometries and frozen reference paths change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_DIR = SCRIPT_DIR / "outputs" / "reference_seed_sensitivity"
SHARED_BANKS = (
    "truth_bank.npz",
    "reference_endpoints.npz",
    "selection_bank.npz",
    "validation_bank.npz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_actions(path: Path) -> dict[str, dict[int, float]]:
    actions: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["valid"].lower() != "true":
                continue
            actions.setdefault(row["design"], {})[int(row["trial"])] = float(
                row["full_action"]
            )
    return actions


def _paired_reduction_arrays(
    actions: dict[str, dict[int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial_ids = np.asarray(
        sorted(set(actions.get("law", {})) & set(actions.get("full", {}))),
        dtype=np.int64,
    )
    if len(trial_ids) == 0:
        raise ValueError("no common valid Law/Full validation trials")
    law = np.asarray([actions["law"][int(i)] for i in trial_ids])
    full = np.asarray([actions["full"][int(i)] for i in trial_ids])
    return trial_ids, law, full


def _ratio_reduction(law: np.ndarray, full: np.ndarray) -> float:
    return float(1.0 - np.mean(full) / np.mean(law))


def _bootstrap_reduction(
    law: np.ndarray,
    full: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> np.ndarray:
    indices = rng.integers(0, len(law), size=(resamples, len(law)))
    return 1.0 - np.mean(full[indices], axis=1) / np.mean(law[indices], axis=1)


def _assignment_rms(a: Any, b: Any) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"incompatible center arrays: {x.shape} and {y.shape}")
    best = min(
        np.mean(np.sum((x - y[list(perm)]) ** 2, axis=1))
        for perm in itertools.permutations(range(len(y)))
    )
    return float(np.sqrt(best))


def _reference_pair_metrics(a: Path, b: Path) -> dict[str, float]:
    with np.load(a) as za, np.load(b) as zb:
        nodes_a = np.asarray(za["nodes"], dtype=np.float64)
        nodes_b = np.asarray(zb["nodes"], dtype=np.float64)
        velocity_a = np.asarray(za["velocity"], dtype=np.float64)
        velocity_b = np.asarray(zb["velocity"], dtype=np.float64)
    if nodes_a.shape != nodes_b.shape or velocity_a.shape != velocity_b.shape:
        raise ValueError(f"incompatible reference banks: {a} and {b}")
    node_delta = nodes_a - nodes_b
    velocity_delta = velocity_a - velocity_b
    velocity_scale = np.sqrt(
        0.5 * (np.mean(velocity_a**2) + np.mean(velocity_b**2))
    )
    return {
        "trajectory_rms": float(np.sqrt(np.mean(node_delta**2))),
        "final_position_rms": float(
            np.sqrt(np.mean(np.sum(node_delta[-1] ** 2, axis=-1)))
        ),
        "velocity_normalized_rmse": float(
            np.sqrt(np.mean(velocity_delta**2)) / velocity_scale
        ),
    }


def _default_results() -> list[Path]:
    paths = [SCRIPT_DIR / "outputs" / "run" / "result.json"]
    paths.extend(sorted(DEFAULT_AUDIT_DIR.glob("reference_seed_*/result.json")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        type=Path,
        help="result.json to include; repeat for multiple runs",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=284_166)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_AUDIT_DIR / "summary.json"
    )
    args = parser.parse_args()

    result_paths = args.result or _default_results()
    if len(result_paths) < 2:
        raise ValueError("at least two completed reference-seed results are required")

    runs = []
    bootstrap: dict[int, np.ndarray] = {}
    trial_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for result_path in result_paths:
        result_path = result_path.resolve()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(result["config"]["reference_training"]["seed"])
        csv_path = result_path.with_suffix(".validation_trials.csv")
        trial_ids, law, full = _paired_reduction_arrays(
            _validation_actions(csv_path)
        )
        samples = _bootstrap_reduction(
            law,
            full,
            rng=rng,
            resamples=int(args.bootstrap_resamples),
        )
        bootstrap[seed] = samples
        trial_data[seed] = (trial_ids, law, full)
        reduction = _ratio_reduction(law, full)
        reported = float(
            result["contrasts"]["full_vs_law_full_action_reduction"][
                "ratio_of_means_reduction"
            ]
        )
        if not np.isclose(reduction, reported, rtol=1e-12, atol=1e-12):
            raise ValueError(f"saved contrast does not match validation CSV: {result_path}")
        history = result["reference"]["metadata"].get("history", [])
        runs.append(
            {
                "reference_seed": seed,
                "result": str(result_path),
                "run_dir": str(result_path.parent),
                "paired_valid_trials": int(len(trial_ids)),
                "law_full_action_mean": float(np.mean(law)),
                "full_full_action_mean": float(np.mean(full)),
                "full_vs_law_reduction": reduction,
                "bootstrap_95_interval": [
                    float(x) for x in np.quantile(samples, [0.025, 0.975])
                ],
                "bootstrap_se": float(np.std(samples, ddof=1)),
                "selection_certified": bool(
                    result["selection_certificates"]["full"]["certified"]
                ),
                "full_valid_fraction": float(result["validation"]["full"]["valid_fraction"]),
                "law_centers": result["selection_centers"]["law"],
                "tangent_centers": result["selection_centers"]["tangent"],
                "full_centers": result["selection_centers"]["full"],
                "last_training_loss": (
                    float(history[-1]["conditional_fm_loss"]) if history else None
                ),
            }
        )

    runs.sort(key=lambda row: row["reference_seed"])
    reductions = np.asarray([row["full_vs_law_reduction"] for row in runs])
    pairwise = []
    for left, right in itertools.combinations(runs, 2):
        seed_a = int(left["reference_seed"])
        seed_b = int(right["reference_seed"])
        ids_a, law_a, full_a = trial_data[seed_a]
        ids_b, law_b, full_b = trial_data[seed_b]
        if not np.array_equal(ids_a, ids_b):
            raise ValueError(f"validation trial IDs differ for seeds {seed_a} and {seed_b}")
        # Bootstrap draws were generated separately above.  Re-use common
        # resampling indices here so differences exploit the shared trial bank.
        pair_rng = np.random.default_rng(args.bootstrap_seed + seed_a + seed_b)
        indices = pair_rng.integers(
            0, len(ids_a), size=(int(args.bootstrap_resamples), len(ids_a))
        )
        red_a = 1.0 - np.mean(full_a[indices], axis=1) / np.mean(law_a[indices], axis=1)
        red_b = 1.0 - np.mean(full_b[indices], axis=1) / np.mean(law_b[indices], axis=1)
        delta = red_b - red_a
        pairwise.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "reduction_b_minus_a": float(
                    right["full_vs_law_reduction"] - left["full_vs_law_reduction"]
                ),
                "paired_bootstrap_95_interval": [
                    float(x) for x in np.quantile(delta, [0.025, 0.975])
                ],
                "full_center_assignment_rms": _assignment_rms(
                    left["full_centers"], right["full_centers"]
                ),
                "law_center_assignment_rms": _assignment_rms(
                    left["law_centers"], right["law_centers"]
                ),
                "tangent_center_assignment_rms": _assignment_rms(
                    left["tangent_centers"], right["tangent_centers"]
                ),
                "reference_bank": _reference_pair_metrics(
                    Path(left["run_dir"]) / "reference_bank.npz",
                    Path(right["run_dir"]) / "reference_bank.npz",
                ),
            }
        )

    shared_bank_hashes: dict[str, dict[str, Any]] = {}
    for name in SHARED_BANKS:
        hashes = {
            str(Path(row["run_dir"]) / name): _sha256(Path(row["run_dir"]) / name)
            for row in runs
        }
        shared_bank_hashes[name] = {
            "identical": len(set(hashes.values())) == 1,
            "sha256": hashes,
        }

    payload = {
        "schema_version": 1,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "reference_seed_count": len(runs),
        "all_shared_banks_byte_identical": all(
            item["identical"] for item in shared_bank_hashes.values()
        ),
        "all_full_selections_certified": all(row["selection_certified"] for row in runs),
        "all_full_validation_trials_valid": all(row["full_valid_fraction"] == 1.0 for row in runs),
        "reduction_mean": float(np.mean(reductions)),
        "reduction_sample_sd": float(np.std(reductions, ddof=1)),
        "reduction_min": float(np.min(reductions)),
        "reduction_max": float(np.max(reductions)),
        "reduction_range": float(np.ptp(reductions)),
        "runs": runs,
        "pairwise": pairwise,
        "shared_bank_hashes": shared_bank_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
