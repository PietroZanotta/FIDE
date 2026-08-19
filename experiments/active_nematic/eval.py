"""Evaluate and visualize completed active-nematic MFSI runs.

This is deliberately pure post-processing.  It reads the saved manifest,
per-reference result JSON files, and the extracted defect bank; it never trains
a reference, repeats design selection, performs an I-projection, or solves a
Poisson problem.

From the repository root::

    .venv/bin/python experiments/active_nematic/eval.py
    .venv/bin/python experiments/active_nematic/eval.py --show
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "run_guarded" / "manifest_position_polarity.json"
DEFAULT_CONFIG = SCRIPT_DIR / "config.json"
DESIGNS = ("law", "tangent", "full")
COLORS = {"law": "#2878B5", "tangent": "#E29D26", "full": "#D1495B"}
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "active_nematic_density", ("#FBFAF5", "#D5E3DF", "#64A69A", "#293241")
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any) -> str:
    if not _finite(value):
        return "n/a"
    value = float(value)
    if value != 0.0 and (abs(value) < 1.0e-4 or abs(value) >= 1.0e5):
        return f"{value:.4e}"
    return f"{value:.7g}"


def _pct(value: Any, digits: int = 1) -> str:
    return "n/a" if not _finite(value) else f"{100.0 * float(value):.{digits}f}%"


def _metric(summary: dict[str, Any], key: str) -> str:
    metric = summary.get(key, {})
    if not _finite(metric.get("mean")):
        return "n/a"
    return (
        f"{_num(metric['mean'])} ± {_num(metric.get('se'))} "
        f"(SE, n={metric.get('n', 0)})"
    )


def _eta(value: Any) -> str:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) % 2:
        return "n/a"
    pairs = array.reshape(-1, 2)
    return "[" + ", ".join(f"({x:.3f}, {y:.3f})" for x, y in pairs) + "]"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def _resolve_result(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file():
        return path.resolve()
    # Absolute paths in saved manifests should not make copied output trees
    # unusable.  Recover the conventional seed-directory location if needed.
    local = manifest_path.parent / path.parent.name / path.name
    if local.is_file():
        return local.resolve()
    raise SystemExit(f"Result listed by manifest does not exist: {path}")


def _load_results(input_path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    data = _read_json(input_path)
    if isinstance(data.get("runs"), list):
        paths = [
            _resolve_result(str(row["result"]), input_path)
            for row in data["runs"]
        ]
    elif "reference_seed" in data and "validation" in data:
        paths = [input_path]
    else:
        raise SystemExit(
            f"Expected an active-nematic manifest or result JSON, got {input_path}."
        )
    results = [_read_json(path) for path in paths]
    for path, result in zip(paths, results, strict=True):
        if result.get("experiment") != "active_nematic_positive_defects":
            raise SystemExit(
                f"Expected active_nematic_positive_defects in {path}, "
                f"got {result.get('experiment')!r}."
            )
    order = np.argsort([int(result["reference_seed"]) for result in results])
    return [results[int(i)] for i in order], [paths[int(i)] for i in order]


def _selected_exact_audit(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row
        for row in result.get("selection_candidates", {}).get("full_exact", [])
        if np.allclose(np.asarray(row.get("eta", []), dtype=np.float64), eta)
    ]
    if not matches:
        raise ValueError(f"no exact selection audit matches the saved {design} design")
    return matches[0]


def _valid_trial_map(block: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["trial"]): row
        for row in block.get("trials", [])
        if bool(row.get("valid")) and _finite(row.get("full_action"))
    }


def paired_action_arrays(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired trial IDs and Law/Full validation action arrays."""
    law = _valid_trial_map(result["validation"]["law"])
    full = _valid_trial_map(result["validation"]["full"])
    trial_ids = np.asarray(sorted(set(law).intersection(full)), dtype=np.int64)
    law_values = np.asarray(
        [float(law[int(i)]["full_action"]) for i in trial_ids], dtype=np.float64
    )
    full_values = np.asarray(
        [float(full[int(i)]["full_action"]) for i in trial_ids], dtype=np.float64
    )
    return trial_ids, law_values, full_values


def paired_action_by_time(
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired trial IDs and instantaneous Law/Full action matrices."""
    law = _valid_trial_map(result["validation"]["law"])
    full = _valid_trial_map(result["validation"]["full"])
    trial_ids = []
    law_rows = []
    full_rows = []
    for trial_id in sorted(set(law).intersection(full)):
        law_values = law[trial_id].get("full_action_by_time")
        full_values = full[trial_id].get("full_action_by_time")
        if law_values is None or full_values is None:
            continue
        law_array = np.asarray(law_values, dtype=np.float64)
        full_array = np.asarray(full_values, dtype=np.float64)
        if (
            law_array.ndim != 1
            or law_array.shape != full_array.shape
            or not np.all(np.isfinite(law_array))
            or not np.all(np.isfinite(full_array))
        ):
            continue
        trial_ids.append(int(trial_id))
        law_rows.append(law_array)
        full_rows.append(full_array)
    if not law_rows:
        empty = np.empty((0, 0), dtype=np.float64)
        return np.asarray([], dtype=np.int64), empty, empty.copy()
    return (
        np.asarray(trial_ids, dtype=np.int64),
        np.stack(law_rows),
        np.stack(full_rows),
    )


def paired_bootstrap_reduction(
    law: np.ndarray,
    full: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap 1-mean(Full)/mean(Law), preserving paired trials."""
    law = np.asarray(law, dtype=np.float64)
    full = np.asarray(full, dtype=np.float64)
    if law.shape != full.shape or law.ndim != 1:
        raise ValueError("law and full action arrays must be paired one-dimensional arrays")
    if len(law) == 0 or not np.all(np.isfinite(law)) or not np.all(np.isfinite(full)):
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan"), "reps": 0}
    estimate = 1.0 - float(np.mean(full) / np.mean(law))
    if reps < 1:
        return {"estimate": estimate, "lower": float("nan"), "upper": float("nan"), "reps": 0}
    rng = np.random.default_rng(int(seed))
    # Chunking bounds temporary memory if a very large replicate count is used.
    draws = np.empty(int(reps), dtype=np.float64)
    cursor = 0
    while cursor < int(reps):
        stop = min(cursor + 20_000, int(reps))
        indices = rng.integers(0, len(law), size=(stop - cursor, len(law)))
        law_means = np.mean(law[indices], axis=1)
        full_means = np.mean(full[indices], axis=1)
        draws[cursor:stop] = 1.0 - full_means / law_means
        cursor = stop
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "reps": int(reps),
    }


def _tail_ratio(values: np.ndarray) -> float:
    if len(values) == 0 or float(np.median(values)) <= 0.0:
        return float("nan")
    return float(np.max(values) / np.median(values))


def _trapezoid_weights_numpy(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    widths = np.diff(times)
    if times.ndim != 1 or len(times) < 2 or np.any(widths <= 0.0):
        raise ValueError("evaluation times must be strictly increasing")
    weights = np.empty_like(times)
    weights[0] = 0.5 * widths[0]
    weights[-1] = 0.5 * widths[-1]
    if len(times) > 2:
        weights[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return weights / np.sum(weights)


def build_statistics(
    results: list[dict[str, Any]],
    paths: list[Path],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (result, path) in enumerate(zip(results, paths, strict=True)):
        law_selection = _selected_exact_audit(result, "law")
        tangent_selection = _selected_exact_audit(result, "tangent")
        full_selection = _selected_exact_audit(result, "full")
        _, law_trials, full_trials = paired_action_arrays(result)
        _, law_by_time, full_by_time = paired_action_by_time(result)
        bootstrap = paired_bootstrap_reduction(
            law_trials,
            full_trials,
            reps=bootstrap_reps,
            seed=int(bootstrap_seed) + index,
        )
        law_summary = result["validation"]["law"]["summary"]
        full_summary = result["validation"]["full"]["summary"]
        epsilon = float(result["risk_max"]) - float(result["risk_star"])
        law_risk = float(law_summary["law_risk"]["mean"])
        full_risk = float(full_summary["law_risk"]["mean"])
        law_action = float(law_summary["full_action"]["mean"])
        full_action = float(full_summary["full_action"]["mean"])
        time_diagnostics = None
        if len(law_by_time):
            times = np.asarray(result["evaluation_times"]["physical"], dtype=np.float64)
            saved_weights = result["evaluation_times"].get("action_weights")
            weights = (
                np.asarray(saved_weights, dtype=np.float64)
                if saved_weights is not None
                else _trapezoid_weights_numpy(times)
            )
            if weights.shape != times.shape or np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
                raise ValueError("saved action weights are invalid")
            action_indices = np.flatnonzero(weights > 0.0)
            if len(action_indices) < 2 or not np.array_equal(
                action_indices,
                np.arange(action_indices[0], action_indices[-1] + 1),
            ):
                raise ValueError("positive action weights must form one contiguous time window")
            law_time_mean = np.mean(law_by_time, axis=0)
            full_time_mean = np.mean(full_by_time, axis=0)
            action_times = times[action_indices]
            law_action_time_mean = law_time_mean[action_indices]
            full_action_time_mean = full_time_mean[action_indices]
            shortened_weights = _trapezoid_weights_numpy(action_times[:-1])
            law_without_final = float(shortened_weights @ law_action_time_mean[:-1])
            full_without_final = float(shortened_weights @ full_action_time_mean[:-1])
            final_index = int(action_indices[-1])
            time_diagnostics = {
                "physical_times": times.tolist(),
                "action_physical_times": action_times.tolist(),
                "law_action_mean_by_time": law_time_mean.tolist(),
                "full_action_mean_by_time": full_time_mean.tolist(),
                "full_to_law_ratio_by_time": (full_time_mean / law_time_mean).tolist(),
                "full_to_law_ratio_by_action_time": (
                    full_action_time_mean / law_action_time_mean
                ).tolist(),
                "law_final_time_contribution_fraction": float(
                    weights[final_index] * law_time_mean[final_index] / law_action
                ),
                "full_final_time_contribution_fraction": float(
                    weights[final_index] * full_time_mean[final_index] / full_action
                ),
                "final_action_time": float(times[final_index]),
                "without_final_time_law_action": law_without_final,
                "without_final_time_full_action": full_without_final,
                "without_final_time_action_ratio": float(
                    full_without_final / law_without_final
                ),
                "without_final_time_action_reduction": float(
                    1.0 - full_without_final / law_without_final
                ),
                "note": (
                    "Post-hoc endpoint-leverage diagnostic only; this is not the "
                    "predeclared full-horizon estimand."
                ),
            }
        rows.append(
            {
                "reference_seed": int(result["reference_seed"]),
                "result": str(path),
                "result_schema_version": result.get("schema_version"),
                "state_mode": result.get("state", {}).get("mode"),
                "designs": result["designs"],
                "risk_star": float(result["risk_star"]),
                "risk_max": float(result["risk_max"]),
                "selection_certified": bool(result.get("selection_certified")),
                "epsilon_r": epsilon,
                "selection": {
                    design: {
                        "law_risk": float(audit["law"]["value"]),
                        "full_action": float(audit["action"]["value"]),
                        "law_slack_to_max": float(
                            result["risk_max"] - audit["law"]["value"]
                        ),
                        "valid": bool(
                            audit["law"]["valid"]
                            and audit["action"]["valid"]
                        ),
                        "certified": bool(
                            audit["law"]["valid"]
                            and audit["action"]["valid"]
                            and audit["law"]["value"] <= result["risk_max"]
                        ),
                    }
                    for design, audit in (
                        ("law", law_selection),
                        ("tangent", tangent_selection),
                        ("full", full_selection),
                    )
                },
                "validation": {
                    design: {
                        "summary": result["validation"][design]["summary"],
                        "valid_fraction": float(
                            result["validation"][design]["summary"]["valid_trials"]
                            / max(result["validation"][design]["summary"]["trials"], 1)
                        ),
                        "full_action_max_to_median": _tail_ratio(
                            np.asarray(
                                [
                                    trial["full_action"]
                                    for trial in result["validation"][design]["trials"]
                                    if trial["valid"] and _finite(trial["full_action"])
                                ],
                                dtype=np.float64,
                            )
                        ),
                        "tangent_lower_bound_max_violation": float(
                            max(
                                [
                                    trial["tangent_action"] - trial["full_action"]
                                    for trial in result["validation"][design]["trials"]
                                    if trial["valid"]
                                    and _finite(trial["tangent_action"])
                                    and _finite(trial["full_action"])
                                ]
                                or [float("nan")]
                            )
                        ),
                    }
                    for design in DESIGNS
                },
                "selection_solver": result.get("full_action_solver", {}),
                "validation_solver": result.get(
                    "validation_full_action_solver",
                    result.get("full_action_solver", {}),
                ),
                "selection_law_risk": float(law_selection["law"]["value"]),
                "selection_full_risk": float(full_selection["law"]["value"]),
                "selection_law_action": float(law_selection["action"]["value"]),
                "selection_full_action": float(full_selection["action"]["value"]),
                "selection_action_ratio": float(
                    full_selection["action"]["value"] / law_selection["action"]["value"]
                ),
                "validation_law_risk": law_risk,
                "validation_full_risk": full_risk,
                "validation_risk_difference": full_risk - law_risk,
                "validation_information_equivalent": bool(full_risk <= law_risk + epsilon),
                "validation_law_action": law_action,
                "validation_full_action": full_action,
                "validation_action_ratio": float(full_action / law_action),
                "validation_action_reduction": float(1.0 - full_action / law_action),
                "validation_action_advantage": bool(full_action < law_action),
                "validation_action_ratio_by_time": (
                    (np.mean(full_by_time, axis=0) / np.mean(law_by_time, axis=0)).tolist()
                    if len(law_by_time)
                    else None
                ),
                "time_diagnostics": time_diagnostics,
                "paired_valid_trials": int(len(law_trials)),
                "law_trials": int(law_summary["trials"]),
                "full_trials": int(full_summary["trials"]),
                "law_valid_trials": int(law_summary["valid_trials"]),
                "full_valid_trials": int(full_summary["valid_trials"]),
                "law_action_max_to_median": _tail_ratio(law_trials),
                "full_action_max_to_median": _tail_ratio(full_trials),
                "paired_bootstrap_reduction_95": bootstrap,
            }
        )
    ratios = np.asarray([row["validation_action_ratio"] for row in rows], dtype=np.float64)
    payload = {
        "schema_version": 1,
        "reference_seed_count": len(rows),
        "all_selection_certified": all(row["selection_certified"] for row in rows),
        "validation_information_equivalent_seeds": sum(
            row["validation_information_equivalent"] for row in rows
        ),
        "validation_action_advantage_seeds": sum(
            row["validation_action_advantage"] for row in rows
        ),
        "validation_joint_success_seeds": sum(
            row["validation_information_equivalent"] and row["validation_action_advantage"]
            for row in rows
        ),
        "validation_action_ratio_mean": float(np.mean(ratios)),
        "validation_action_ratio_median": float(np.median(ratios)),
        "validation_action_ratio_min": float(np.min(ratios)),
        "validation_action_ratio_max": float(np.max(ratios)),
        "bootstrap": {
            "estimand": "1 - mean_paired_full_action / mean_paired_law_action",
            "confidence_level": 0.95,
            "reps_per_reference_seed": int(bootstrap_reps),
            "seed": int(bootstrap_seed),
        },
        "rows": rows,
    }
    return payload


def _print_statistics(stats: dict[str, Any]) -> int:
    print("=" * 96)
    print("ACTIVE NEMATIC — SAVED RESULT EVALUATION")
    print("=" * 96)
    print(f"reference seeds:     {stats['reference_seed_count']}")
    print(f"all selections certified: {stats['all_selection_certified']}")
    failures: list[str] = []
    for row in stats["rows"]:
        print("\n" + "-" * 96)
        print(f"REFERENCE SEED {row['reference_seed']}")
        print("-" * 96)
        print(f"file:       {row['result']}")
        print(f"schema:     {row['result_schema_version']}")
        print(f"state:      {row['state_mode']}")
        solver = row["validation_solver"]
        print(
            f"solver:     {solver.get('backend', 'n/a')}  "
            f"grid={solver.get('grid_shape', 'n/a')}  "
            f"revision={solver.get('native_solver_revision', 'n/a')}"
        )

        print("\nInformation screen")
        print(
            f"  R*={_num(row['risk_star'])}   Rmax={_num(row['risk_max'])}   "
            f"epsilon_R={_num(row['epsilon_r'])}"
        )

        print("\nSelected centers")
        for design in DESIGNS:
            print(f"  {design:<12} {_eta(row['designs'][design])}")

        print("\nSelection-bank certification")
        print(
            f"  {'design':<12} {'R':>12} {'R slack':>12} "
            f"{'full action':>14} {'valid':>8} {'status':>9}"
        )
        for design in DESIGNS:
            selection = row["selection"][design]
            print(
                f"  {design:<12} {_num(selection['law_risk']):>12} "
                f"{_num(selection['law_slack_to_max']):>12} "
                f"{_num(selection['full_action']):>14} "
                f"{str(selection['valid']):>8} "
                f"{'PASS' if selection['certified'] else 'FAIL':>9}"
            )
            if not selection["certified"]:
                failures.append(
                    f"seed {row['reference_seed']}: {design} failed exact selection certificate"
                )

        print("\nIndependent validation")
        for design in DESIGNS:
            validation = row["validation"][design]
            summary = validation["summary"]
            print(
                f"  {design:<12} R={_metric(summary, 'law_risk'):<31} "
                f"Atan={_metric(summary, 'tangent_action'):<31} "
                f"A={_metric(summary, 'full_action'):<31} "
                f"valid={_pct(validation['valid_fraction'])}"
            )
            print(
                f"  {'':<12} action max/median={_num(validation['full_action_max_to_median'])}  "
                f"max(Atan-A)={_num(validation['tangent_lower_bound_max_violation'])}"
            )
            if validation["valid_fraction"] < 0.95:
                failures.append(
                    f"seed {row['reference_seed']}: {design} validation valid fraction "
                    f"{validation['valid_fraction']:.3f} < 0.95"
                )
            violation = validation["tangent_lower_bound_max_violation"]
            if _finite(violation) and violation > 1.0e-6:
                failures.append(
                    f"seed {row['reference_seed']}: {design} violated Atan <= A"
                )

        boot = row["paired_bootstrap_reduction_95"]
        print("\nFull vs Law")
        print(f"  paired valid trials:        {row['paired_valid_trials']}")
        print(f"  selection action reduction: {_pct(1.0 - row['selection_action_ratio'])}")
        print(f"  validation action reduction:{_pct(row['validation_action_reduction']):>9}")
        print(
            f"  paired bootstrap 95% CI:    [{_pct(boot['lower'])}, {_pct(boot['upper'])}] "
            f"(reps={boot['reps']})"
        )
        print(f"  validation Delta R:         {_num(row['validation_risk_difference'])}")
        print(
            f"  law-equivalent within epsilon_R: "
            f"{'YES' if row['validation_information_equivalent'] else 'NO'}"
        )
        print(
            "  Note: validation is out-of-sample; formal R feasibility is "
            "certified on the selection bank above."
        )
        if not row["selection_certified"]:
            failures.append(f"seed {row['reference_seed']}: selection is not certified")
        if row["paired_valid_trials"] == 0:
            failures.append(f"seed {row['reference_seed']}: no paired valid validation trials")

    print("\n" + "=" * 96)
    print("MULTI-REFERENCE SUMMARY")
    print("=" * 96)
    print(
        f"{'seed':<11} {'sel A(F/L)':>11} {'val A(F/L)':>11} "
        f"{'val reduction':>14} {'bootstrap 95% CI':>24} "
        f"{'Delta R val':>12} {'law-equiv.':>12}"
    )
    for row in stats["rows"]:
        boot = row["paired_bootstrap_reduction_95"]
        interval = f"[{_pct(boot['lower'])}, {_pct(boot['upper'])}]"
        print(
            f"{row['reference_seed']:<11d} "
            f"{_num(row['selection_action_ratio']):>11} "
            f"{_num(row['validation_action_ratio']):>11} "
            f"{_pct(row['validation_action_reduction']):>14} "
            f"{interval:>24} "
            f"{_num(row['validation_risk_difference']):>12} "
            f"{str(row['validation_information_equivalent']):>12}"
        )
    print()
    print(
        "Held-out summary: "
        f"law-equivalent {stats['validation_information_equivalent_seeds']}/{stats['reference_seed_count']}, "
        f"action advantage {stats['validation_action_advantage_seeds']}/{stats['reference_seed_count']}, "
        f"joint success {stats['validation_joint_success_seeds']}/{stats['reference_seed_count']}."
    )
    print(
        "Validation Full/Law action ratio across reference seeds: "
        f"median={_num(stats['validation_action_ratio_median'])}, "
        f"range=[{_num(stats['validation_action_ratio_min'])}, "
        f"{_num(stats['validation_action_ratio_max'])}]."
    )
    if all(row.get("time_diagnostics") for row in stats["rows"]):
        print("\nEndpoint-leverage diagnostic (post hoc; not a replacement estimand)")
        print(
            f"{'seed':<11} {'ratio without final t':>23} "
            f"{'Law final-t share':>19} {'Full final-t share':>20}"
        )
        for row in stats["rows"]:
            diagnostic = row["time_diagnostics"]
            print(
                f"{row['reference_seed']:<11d} "
                f"{_num(diagnostic['without_final_time_action_ratio']):>23} "
                f"{_pct(diagnostic['law_final_time_contribution_fraction']):>19} "
                f"{_pct(diagnostic['full_final_time_contribution_fraction']):>20}"
            )
    if 0 < stats["validation_action_advantage_seeds"] < stats["reference_seed_count"]:
        print("Interpretation: the action gain is reference-sensitive; do not headline a pooled action scale.")
    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("Saved results pass the evaluator's structural checks.")
    return 0


def _load_defect_bank(input_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = input_path.parent if "manifest" in input_path.name else input_path.parent.parent
    mode = str(results[0].get("state", {}).get("mode", "position_polarity"))
    bank_path = run_dir / f"positive_defect_bank_{mode}.npz"
    if not bank_path.is_file():
        raise SystemExit(f"Defect bank required for the figure was not found: {bank_path}")
    try:
        with np.load(bank_path, allow_pickle=False) as bank:
            return {
                "path": bank_path,
                "times": np.asarray(bank["times"], dtype=np.float64),
                "states": np.asarray(bank["states"], dtype=np.float64),
                "offsets": np.asarray(bank["offsets"], dtype=np.int64),
                "counts": np.asarray(bank["counts"], dtype=np.float64),
                "box_size": float(bank["box_size"]),
            }
    except (OSError, KeyError) as exc:
        raise SystemExit(f"Could not load defect bank {bank_path}: {exc}") from exc


def _samples_at(bank: dict[str, Any], time_index: int) -> np.ndarray:
    offsets = bank["offsets"]
    chunks = [
        bank["states"][offsets[run, time_index] : offsets[run, time_index + 1]]
        for run in range(offsets.shape[0])
    ]
    return np.concatenate([chunk for chunk in chunks if len(chunk)], axis=0)


def _smooth_periodic(values: np.ndarray, passes: int = 3) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    for _ in range(passes):
        out = (
            4.0 * out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 8.0
    return out


def _density(samples: np.ndarray, box_size: float, bins: int = 72) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=bins, range=((0.0, box_size), (0.0, box_size))
    )
    return _smooth_periodic(hist.T)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#F3F0E9",
            "axes.facecolor": "#FBFAF6",
            "savefig.facecolor": "#F3F0E9",
        }
    )


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.08, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold")


def _draw_population_strip(fig: plt.Figure, spec, bank: dict[str, Any]) -> None:
    indices = np.linspace(0, len(bank["times"]) - 1, min(5, len(bank["times"]))).round().astype(int)
    densities = [_density(_samples_at(bank, int(i)), bank["box_size"]) for i in indices]
    positive = np.concatenate([density[density > 0.0] for density in densities])
    vmax = max(float(np.quantile(positive, 0.995)), 1.0)
    inner = spec.subgridspec(1, len(indices), wspace=0.055)
    axes = []
    for column, (index, density) in enumerate(zip(indices, densities, strict=True)):
        ax = fig.add_subplot(inner[0, column])
        axes.append(ax)
        ax.imshow(
            density,
            origin="lower",
            extent=(0.0, bank["box_size"], 0.0, bank["box_size"]),
            cmap=DENSITY_CMAP,
            norm=PowerNorm(gamma=0.52, vmin=0.0, vmax=vmax),
            interpolation="bilinear",
        )
        ax.set_title(
            rf"$t={bank['times'][index]:g}$",
            loc="right" if column == 0 else "center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
    _panel(axes[0], "A   Defect population")


def _draw_sensor_geometry(
    fig: plt.Figure,
    spec,
    results: list[dict[str, Any]],
    bank: dict[str, Any],
    sensor_width: float,
) -> None:
    seeds = [int(result["reference_seed"]) for result in results]
    background = np.mean(
        np.stack([_density(_samples_at(bank, i), bank["box_size"]) for i in range(len(bank["times"]))]),
        axis=0,
    )
    vmax = max(float(np.quantile(background[background > 0.0], 0.995)), 1.0)
    inner = spec.subgridspec(2, len(results), hspace=0.14, wspace=0.08)
    axes: list[plt.Axes] = []
    for row_index, design in enumerate(("law", "full")):
        for column, (seed, result) in enumerate(zip(seeds, results, strict=True)):
            ax = fig.add_subplot(inner[row_index, column])
            axes.append(ax)
            ax.imshow(
                background,
                origin="lower",
                extent=(0.0, bank["box_size"], 0.0, bank["box_size"]),
                cmap=DENSITY_CMAP,
                norm=PowerNorm(gamma=0.48, vmin=0.0, vmax=vmax),
                interpolation="bilinear",
            )
            eta = np.asarray(result["designs"][design], dtype=np.float64).reshape(-1, 2)
            for sensor_index, (x, y) in enumerate(eta, start=1):
                # Repeat circles across the nearest periodic images so edge sensors
                # are represented with the correct torus geometry.
                for dx in (-bank["box_size"], 0.0, bank["box_size"]):
                    for dy in (-bank["box_size"], 0.0, bank["box_size"]):
                        primary = dx == 0.0 and dy == 0.0
                        ax.add_patch(
                            Circle(
                                (x + dx, y + dy),
                                sensor_width,
                                facecolor=mpl.colors.to_rgba(
                                    COLORS[design], 0.13 if primary else 0.08
                                ),
                                fill=True,
                                edgecolor=mpl.colors.to_rgba(
                                    COLORS[design], 0.95 if primary else 0.72
                                ),
                                linewidth=1.5 if primary else 1.1,
                                linestyle="-" if primary else "--",
                            )
                        )
                ax.scatter([x], [y], s=34, color=COLORS[design], edgecolor="white", lw=0.8, zorder=4)
                ax.text(x, y, str(sensor_index), color="white", fontsize=6, ha="center", va="center", zorder=5)
            if row_index == 0:
                ax.set_title(
                    f"reference {seed}",
                    pad=3,
                    loc="right" if column == 0 else "center",
                )
            if column == 0:
                ax.set_ylabel(design.capitalize(), color=COLORS[design], fontweight="bold")
            ax.set_xlim(0.0, bank["box_size"])
            ax.set_ylim(0.0, bank["box_size"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
    _panel(axes[0], "B   Periodic imaging windows")
    axes[len(results) - 1].text(
        1.0,
        1.08,
        "dashed fragments are wrapped copies of the same sensor",
        transform=axes[len(results) - 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#666C74",
    )


def _draw_counts(ax: plt.Axes, bank: dict[str, Any], results: list[dict[str, Any]]) -> None:
    times = bank["times"]
    counts = bank["counts"]
    mean = np.mean(counts, axis=0)
    se = np.std(counts, axis=0, ddof=1) / np.sqrt(counts.shape[0])
    ax.fill_between(times, mean - 1.96 * se, mean + 1.96 * se, color="#657786", alpha=0.18)
    ax.plot(times, mean, color="#3E5667", marker="o", ms=3.5, lw=1.5)
    design = np.asarray(results[0].get("positive_count", {}).get("design_mean", []), dtype=float)
    validation = np.asarray(results[0].get("positive_count", {}).get("validation_mean", []), dtype=float)
    if design.shape == times.shape:
        ax.plot(times, design, color="#55A868", lw=1.0, ls="--", label="design split")
    if validation.shape == times.shape:
        ax.plot(times, validation, color="#C44E52", lw=1.0, ls=":", label="validation split")
    ax.set_xlabel("physical time")
    ax.set_ylabel("mean extant +1/2 count")
    ax.set_title("C   Defect-count evolution", loc="left")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=7.5)


def _draw_ratio_shift(ax: plt.Axes, stats: dict[str, Any]) -> None:
    for index, row in enumerate(stats["rows"]):
        color = mpl.colormaps["Dark2"](index / max(len(stats["rows"]), 2))
        values = [row["selection_action_ratio"], row["validation_action_ratio"]]
        ax.plot([0, 1], values, marker="o", lw=1.8, color=color, label=str(row["reference_seed"]))
    ax.axhline(1.0, color="#555A60", lw=1.0, ls="--")
    ax.set_xticks([0, 1], ["selection", "held-out validation"])
    ax.set_ylabel("Full / Law mean action")
    ax.set_yscale("log")
    ax.set_title("D   Gain stability", loc="left")
    ax.grid(axis="y", which="both", alpha=0.22)
    ax.legend(title="reference", frameon=False, fontsize=7.2, title_fontsize=7.3)


def _draw_time_actions(ax: plt.Axes, results: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    available = all(row.get("validation_action_ratio_by_time") for row in stats["rows"])
    if not available:
        ax.text(
            0.5,
            0.5,
            "time-resolved action unavailable\nrerun the validation stage",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#656B73",
        )
        ax.set_title("E   Gain through time", loc="left")
        ax.set_axis_off()
        return
    for index, row in enumerate(stats["rows"]):
        diagnostic = row["time_diagnostics"]
        times = np.asarray(diagnostic["action_physical_times"], dtype=np.float64)
        ratios = np.asarray(
            diagnostic["full_to_law_ratio_by_action_time"], dtype=np.float64
        )
        ax.plot(
            times,
            ratios,
            marker="o",
            ms=3.2,
            lw=1.5,
            color=mpl.colormaps["Dark2"](index / max(len(stats["rows"]), 2)),
            label=str(row["reference_seed"]),
        )
    ax.axhline(1.0, color="#555A60", lw=1.0, ls="--")
    ax.set_xlabel("physical time")
    ax.set_ylabel("Full / Law mean action")
    ax.set_yscale("log")
    ax.set_title("E   Gain through time", loc="left")
    ax.grid(axis="y", which="both", alpha=0.22)


def _draw_bootstrap(ax: plt.Axes, stats: dict[str, Any]) -> None:
    rows = stats["rows"]
    y = np.arange(len(rows))
    for yi, row in zip(y, rows, strict=True):
        boot = row["paired_bootstrap_reduction_95"]
        estimate = 100.0 * boot["estimate"]
        lower = 100.0 * boot["lower"]
        upper = 100.0 * boot["upper"]
        color = "#3A8D6D" if lower > 0.0 else "#B65C62" if upper < 0.0 else "#7A7F87"
        ax.errorbar(
            estimate,
            yi,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=color,
            capsize=3,
            lw=1.5,
        )
        ax.text(upper + 2.0, yi, f"{estimate:.1f}%", va="center", fontsize=7.5)
    ax.axvline(0.0, color="#555A60", lw=1.0, ls="--")
    ax.set_yticks(y, [str(row["reference_seed"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("paired mean-action reduction (%)")
    ax.set_title("F   Paired bootstrap 95% intervals", loc="left")
    ax.grid(axis="x", alpha=0.22)


def make_figure(
    results: list[dict[str, Any]],
    stats: dict[str, Any],
    bank: dict[str, Any],
    *,
    sensor_width: float,
) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(16.2, 12.2), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.78, 1.4, 0.94),
        left=0.055,
        right=0.975,
        bottom=0.065,
        top=0.89,
        hspace=0.31,
    )
    _draw_population_strip(fig, outer[0, 0], bank)
    _draw_sensor_geometry(fig, outer[1, 0], results, bank, sensor_width)
    bottom = outer[2, 0].subgridspec(1, 4, wspace=0.34)
    _draw_counts(fig.add_subplot(bottom[0, 0]), bank, results)
    _draw_ratio_shift(fig.add_subplot(bottom[0, 1]), stats)
    _draw_time_actions(fig.add_subplot(bottom[0, 2]), results, stats)
    _draw_bootstrap(fig.add_subplot(bottom[0, 3]), stats)
    n = stats["reference_seed_count"]
    joint = stats["validation_joint_success_seeds"]
    endpoint_dominated = sum(
        row.get("time_diagnostics") is not None
        and row["time_diagnostics"]["law_final_time_contribution_fraction"] > 0.8
        for row in stats["rows"]
    )
    fig.suptitle(
        "Active nematic · where to observe a fluctuating defect population",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#22262C",
    )
    fig.text(
        0.055,
        0.932,
        f"Full-action placement preserves the held-out law tolerance and lowers action for {joint}/{n} "
        f"frozen references; {endpoint_dominated}/{n} Law comparisons are >80% final-time dominated.",
        ha="left",
        fontsize=10.5,
        color="#50565F",
    )
    fig.text(
        0.055,
        0.023,
        "Actions are compared only within a reference seed. Bootstrap intervals resample paired held-out trials; "
        "they do not quantify uncertainty across reference-model training.",
        ha="left",
        fontsize=7.8,
        color="#6B7078",
    )
    return fig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print statistics and make a figure from saved active-nematic results."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument("--dpi", type=int, default=210)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    if args.bootstrap_reps < 0:
        raise SystemExit("--bootstrap-reps must be nonnegative")
    results, paths = _load_results(input_path)
    stats = build_statistics(
        results,
        paths,
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    exit_code = _print_statistics(stats)
    output_dir = input_path.parent if "manifest" in input_path.name else input_path.parent
    stats_path = (args.stats or output_dir / "active_nematic_evaluation_stats.json").expanduser().resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"stats:  {stats_path}")
    if not args.no_figure:
        config = _read_json(args.config.expanduser().resolve())
        sensor_width = float(config.get("measurement", {}).get("sensor_width", 4.0))
        bank = _load_defect_bank(input_path, results)
        figure = make_figure(results, stats, bank, sensor_width=sensor_width)
        figure_path = (args.output or output_dir / "active_nematic_evaluation.png").expanduser().resolve()
        if not figure_path.suffix:
            figure_path = figure_path.with_suffix(".png")
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=int(args.dpi), bbox_inches="tight")
        print(f"figure: {figure_path}")
        if args.show:
            plt.show()
        else:
            plt.close(figure)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
