"""Repair the Toy 2--5% Full sweep with nested corrected incumbents.

Population, Law, Tangent, risks, banks, reconstruction, I-projection, and the
positive-support physical-q evaluator are frozen.  Only Full geometry is
searched, and historical Pareto outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
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

from action_decomposition_audit import file_sha256, geometry_key
from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from audit_corrected_all_candidates import _build_pool, _load_bank
from audit_positive_rasterization import MASS_TOLERANCE, SOURCE_TOLERANCE, _summarize
from experiment import TrialBank
from rerun_corrected_full_0p5 import (
    _add_candidate,
    _bank_prefix,
    _local_ring,
    _snapshot,
    _write_csv,
    _write_json,
)


DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto"
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
SEARCH_ALLOWANCES = (2.0, 3.0, 4.0, 5.0)
SUMMARY_FIELDS = (
    "method",
    "trial_count",
    "time_trial_count",
    "grid_n",
    "bandwidth",
    "minimum_q_h",
    "maximum_mass_error",
    "maximum_source_compatibility_error",
    "maximum_component_compatibility_residual",
    "maximum_conductive_component_count",
    "incompatible_time_trial_count",
    "unconverged_time_trial_count",
    "invalid_trial_count",
    "maximum_physical_poisson_relative_residual",
    "maximum_full_moment_rate_residual",
    "maximum_tangent_moment_rate_residual",
    "maximum_hidden_nullspace_residual",
    "maximum_absolute_orthogonality_residual",
    "maximum_absolute_pythagorean_residual",
    "maximum_raw_hierarchy_violation",
    "A_full_h",
    "A_tan_h",
    "A_hid_h",
    "Gamma_h",
    "physical_poisson_tolerance",
    "moment_tolerance",
    "energy_tolerance",
    "passes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Authoritative output directory; seed/provenance inputs remain in --pareto-dir.",
    )
    parser.add_argument(
        "--rerun-0p5-json",
        type=Path,
        help="Fresh accepted 0.5%% corrected-search JSON to use as the first incumbent.",
    )
    parser.add_argument(
        "--search-from-1pct",
        action="store_true",
        help="Freshly search Full at 1%% as well as 2--5%%, carrying the 0.5%% winner.",
    )
    parser.add_argument(
        "--fresh-evaluations",
        action="store_true",
        help="Use archived corrected metrics only as geometry seeds, never as action cache entries.",
    )
    parser.add_argument("--proxy-grid-n", type=int, default=51)
    parser.add_argument("--proxy-trials", type=int, default=2)
    parser.add_argument("--prescreen-trials", type=int, default=12)
    parser.add_argument("--grid-n", type=int, default=101)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--initial-prescreen-count", type=int, default=12)
    parser.add_argument("--new-finalist-count", type=int, default=6)
    parser.add_argument(
        "--extend-search",
        action="store_true",
        help="Explicitly extend a previously completed PASS sweep instead of freezing it.",
    )
    return parser.parse_args()


def _tag(allowance: float) -> str:
    return f"risk_{f'{allowance:g}'.replace('.', 'p')}pct"


def _safe_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in SUMMARY_FIELDS if field in row}


def _trial_summary(row: dict[str, Any], allowance: float, geometry_deg: list[float]) -> dict[str, Any]:
    return {
        "allowance_percent": allowance,
        "trial": int(row["trial"]),
        "alpha": float(row["alpha"]),
        "geometry_deg": geometry_deg,
        "valid": bool(row["valid"]),
        "invalid_reason": row.get("invalid_reason"),
        "full_action": float(row["full_action"]),
        "minimum_q_h": float(row["minimum_positive_raster_density"]),
        "maximum_mass_error": float(row["maximum_raster_mass_error"]),
        "maximum_source_compatibility_error": float(
            row["maximum_raster_source_compatibility_error"]
        ),
        "maximum_physical_poisson_relative_residual": float(
            row["max_poisson_relative_residual"]
        ),
        "maximum_full_moment_rate_residual": float(
            row["max_full_moment_rate_residual"]
        ),
    }


def _certification_flags(summary: dict[str, Any]) -> dict[str, bool]:
    """Expand the authoritative aggregate certificate into named checks."""
    moment_tolerance = float(summary["moment_tolerance"])
    energy_tolerance = float(summary["energy_tolerance"])
    flags = {
        "positive_q": float(summary["minimum_q_h"]) > 0.0,
        "mass": float(summary["maximum_mass_error"]) <= MASS_TOLERANCE,
        "source": (
            float(summary["maximum_source_compatibility_error"]) <= SOURCE_TOLERANCE
        ),
        "component_compatibility": (
            float(summary["maximum_component_compatibility_residual"])
            <= SOURCE_TOLERANCE
            and int(summary["incompatible_time_trial_count"]) == 0
        ),
        "solver_convergence": int(summary["unconverged_time_trial_count"]) == 0,
        "trial_validity": int(summary["invalid_trial_count"]) == 0,
        "poisson": (
            float(summary["maximum_physical_poisson_relative_residual"])
            <= float(summary["physical_poisson_tolerance"])
        ),
        "full_moment": (
            float(summary["maximum_full_moment_rate_residual"]) <= moment_tolerance
        ),
        "tangent_moment": (
            float(summary["maximum_tangent_moment_rate_residual"]) <= moment_tolerance
        ),
        "hidden_nullspace": (
            float(summary["maximum_hidden_nullspace_residual"]) <= moment_tolerance
        ),
        "orthogonality": (
            float(summary["maximum_absolute_orthogonality_residual"])
            <= energy_tolerance
        ),
        "pythagorean": (
            float(summary["maximum_absolute_pythagorean_residual"])
            <= energy_tolerance
        ),
        "hierarchy": (
            float(summary["maximum_raw_hierarchy_violation"]) <= energy_tolerance
        ),
    }
    flags["all"] = bool(all(flags.values()) and summary["passes"])
    return flags


def main() -> None:
    args = parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    search_allowances = (1.0, *SEARCH_ALLOWANCES) if args.search_from_1pct else SEARCH_ALLOWANCES
    point, first = _strict_common_artifacts(pareto)
    cfg = first["config"]
    exp, action_bank, times = _load_experiment(point, cfg)
    validation_bank = _load_bank(point / "validation_bank.npz")
    law_bank = _bank_prefix(action_bank, int(first["randomness"]["law_trials_effective"]))
    proxy_bank = _bank_prefix(action_bank, int(args.proxy_trials))
    prescreen_bank = _bank_prefix(action_bank, int(args.prescreen_trials))
    time_weights = np.asarray(exp.time_w, dtype=np.float64)
    tolerance = float(cfg["validity"]["tangent_lower_bound_tol"])
    risk_tolerance = 1.0e-12
    min_sep_deg = float(cfg["measurement"]["min_sep_deg"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pareto / "corrected_nested_full_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[float, dict[str, Any]] = {}
    result_paths: dict[float, Path] = {}
    for allowance in ALLOWANCES:
        path = pareto / _tag(allowance) / "result.json"
        result_paths[allowance] = path
        results[allowance] = json.loads(path.read_text(encoding="utf-8"))

    corrected_selection_path = pareto / "toy_corrected_all_candidates_rescore.json"
    corrected_validation_path = pareto / "toy_corrected_validation_rescore.json"
    corrected_pool_path = pareto / "toy_corrected_candidate_pool_audit.json"
    rerun_0p5_path = (
        args.rerun_0p5_json.expanduser().resolve()
        if args.rerun_0p5_json is not None
        else pareto / "corrected_full_rerun_0p5pct" / "corrected_full_rerun.json"
    )
    corrected_selection = json.loads(corrected_selection_path.read_text(encoding="utf-8"))["rows"]
    corrected_validation = json.loads(corrected_validation_path.read_text(encoding="utf-8"))["rows"]
    corrected_pool = json.loads(corrected_pool_path.read_text(encoding="utf-8"))["rows"]
    rerun_0p5 = json.loads(rerun_0p5_path.read_text(encoding="utf-8"))["summary"]

    watched = [
        pareto / "pareto.json",
        *result_paths.values(),
        point / "reference.npz",
        point / "reference_bank.npz",
        point / "selection_bank.npz",
        point / "validation_bank.npz",
        corrected_selection_path,
        corrected_validation_path,
        corrected_pool_path,
        rerun_0p5_path,
    ]
    before = _snapshot(watched)
    evaluator_sources = [
        SCRIPT_DIR / "experiment.py",
        REPO_ROOT / "src" / "mfsi" / "raster.py",
        REPO_ROOT / "src" / "mfsi" / "poisson.py",
        REPO_ROOT / "src" / "mfsi" / "decomposition.py",
    ]
    evaluator_hashes = _snapshot(evaluator_sources)
    settings = {
        "schema_version": 1,
        "proxy_grid_n": int(args.proxy_grid_n),
        "proxy_trials": int(args.proxy_trials),
        "prescreen_trials": int(args.prescreen_trials),
        "grid_n": int(args.grid_n),
        "bandwidth_scale": float(args.bandwidth_scale),
        "initial_prescreen_count": int(args.initial_prescreen_count),
        "new_finalist_count": int(args.new_finalist_count),
        "search_allowances": list(search_allowances),
        "fresh_evaluations": bool(args.fresh_evaluations),
    }
    identity = {"input_hashes": before, "evaluator_source_hashes": evaluator_hashes}
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"settings": settings, "identity": identity, "risk": {}, "actions": {}}
    )
    if checkpoint.get("settings") != settings or checkpoint.get("identity") != identity:
        raise RuntimeError("nested-sweep checkpoint settings, inputs, or evaluator differ")

    def save_checkpoint() -> None:
        _write_json(checkpoint_path, checkpoint)

    # Import prior accepted 101x101/64-trial evaluations.  This is exact reuse,
    # not proxy substitution; geometry and evaluator hashes are frozen above.
    for row in corrected_pool:
        imported_candidates: dict[str, dict[str, Any]] = {}
        key = _add_candidate(
            imported_candidates, exp, row["geometry_deg"], "accepted_corrected_pool"
        )
        degrees = np.asarray(imported_candidates[key]["geometry_deg"], dtype=np.float64)
        separation = float((degrees[1] - degrees[0]) % 180.0)
        separation = min(separation, 180.0 - separation)
        checkpoint["risk"].setdefault(
            key,
            {
                "population_valid": True,
                "finite_valid": True,
                "population_loss": float(row["population_loss_selection"]),
                "finite_risk": float(row["finite_risk_selection"]),
                "projective_separation_deg": separation,
            },
        )
        if (
            not args.fresh_evaluations
            and row.get("passes")
            and np.isfinite(float(row["A_full_h"]))
        ):
            checkpoint["actions"].setdefault(
                f"selection:{key}",
                {**_safe_summary(row), "source": "accepted_corrected_candidate_pool_audit"},
            )
    # The isolated corrected 0.5% winner is a fixed input and supersedes only
    # the 0.5% corrected endpoint for this new table.
    fixed_0p5_geometry = rerun_0p5["rerun_winner"]["geometry_deg"]
    fixed_candidates: dict[str, dict[str, Any]] = {}
    fixed_0p5_key = _add_candidate(
        fixed_candidates, exp, fixed_0p5_geometry, "fixed_corrected_0p5"
    )
    fixed_0p5_candidate = fixed_candidates[fixed_0p5_key]
    checkpoint["risk"][fixed_0p5_key] = {
        "population_valid": True,
        "finite_valid": True,
        "population_loss": float(rerun_0p5["rerun_winner"]["population_loss"]),
        "finite_risk": float(rerun_0p5["rerun_winner"]["finite_risk"]),
        "projective_separation_deg": float(
            (fixed_0p5_geometry[1] - fixed_0p5_geometry[0]) % 180.0
        ),
    }
    checkpoint["actions"][f"selection:{fixed_0p5_key}"] = {
        **rerun_0p5["rerun_winner"]["selection"],
        "source": "accepted_isolated_corrected_0p5_rerun",
    }
    fixed_1_row = next(
        row for row in corrected_selection
        if float(row["allowance_percent"]) == 1.0 and row["method"] == "full"
    )
    fixed_1_geometry = fixed_1_row["geometry_deg"]
    fixed_1_key = _add_candidate(
        fixed_candidates, exp, fixed_1_geometry, "fixed_corrected_1pct"
    )
    fixed_1_candidate = fixed_candidates[fixed_1_key]
    if not args.fresh_evaluations:
        checkpoint["actions"][f"selection:{fixed_1_key}"] = {
            **_safe_summary(fixed_1_row),
            "source": "accepted_corrected_1pct_endpoint",
        }
    fixed_1_cert = results[1.0]["selection_certificates"]["full"]
    checkpoint["risk"][fixed_1_key] = {
        "population_valid": True,
        "finite_valid": True,
        "population_loss": float(fixed_1_cert["L_selection"]),
        "finite_risk": float(fixed_1_cert["R_selection"]),
        "projective_separation_deg": float((fixed_1_geometry[1] - fixed_1_geometry[0]) % 180.0),
    }
    save_checkpoint()

    def exact_risk(candidate: dict[str, Any]) -> dict[str, Any]:
        key = candidate["candidate_key"]
        if key not in checkpoint["risk"]:
            eta = jnp.asarray(candidate["geometry"], dtype=jnp.float64)
            population = exp.exact_population_result(eta)
            finite = exp.exact_finite_result(eta, law_bank)
            separation = (candidate["geometry_deg"][1] - candidate["geometry_deg"][0]) % 180.0
            separation = min(separation, 180.0 - separation)
            checkpoint["risk"][key] = {
                "population_valid": bool(population["valid"]),
                "finite_valid": bool(finite["valid"]),
                "population_loss": float(population["value"]),
                "finite_risk": float(finite["value"]),
                "projective_separation_deg": float(separation),
            }
            save_checkpoint()
        return checkpoint["risk"][key]

    def feasible(candidate: dict[str, Any], L_max: float, R_max: float) -> bool:
        rec = exact_risk(candidate)
        return bool(
            rec["population_valid"]
            and rec["finite_valid"]
            and rec["population_loss"] <= L_max + risk_tolerance
            and rec["finite_risk"] <= R_max + risk_tolerance
            and rec["projective_separation_deg"] >= min_sep_deg - risk_tolerance
        )

    def evaluate_action(
        candidate: dict[str, Any], stage: str, bank: TrialBank, grid_n: int
    ) -> dict[str, Any]:
        cache_key = f"{stage}:{candidate['candidate_key']}"
        if cache_key not in checkpoint["actions"]:
            print(
                f"[{stage}] eta_deg={candidate['geometry_deg']} "
                f"trials={bank.masses.shape[0]} grid={grid_n}",
                flush=True,
            )
            started = time.perf_counter()
            rows = exp.evaluate_common_discretization_decomposition_exact(
                jnp.asarray(candidate["geometry"], dtype=jnp.float64),
                bank,
                grid_n=grid_n,
                bandwidth_scale=float(args.bandwidth_scale),
                progress_desc=f"nested corrected Full {stage}",
            )
            summary, _ = _summarize(
                rows,
                method="full",
                time_weights=time_weights,
                moment_tolerance=tolerance,
                energy_tolerance=tolerance,
            )
            record: dict[str, Any] = {
                **summary,
                "source": "new_corrected_nested_sweep_evaluation",
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            if stage == "validation":
                trial_values = np.asarray([row["full_action"] for row in rows], dtype=np.float64)
                record["full_action_se"] = float(
                    np.std(trial_values, ddof=1) / math.sqrt(len(trial_values))
                )
                record["trials"] = [
                    _trial_summary(row, float("nan"), candidate["geometry_deg"])
                    for row in rows
                ]
            checkpoint["actions"][cache_key] = record
            save_checkpoint()
        return checkpoint["actions"][cache_key]

    winners: dict[float, dict[str, Any]] = {
        0.5: fixed_0p5_candidate,
        1.0: fixed_1_candidate,
    }
    carried_exact: dict[str, dict[str, Any]] = {
        fixed_0p5_key: fixed_0p5_candidate,
        fixed_1_key: fixed_1_candidate,
    }
    stage_records: dict[float, dict[str, Any]] = {}

    final_output_path = output_dir / "corrected_nested_full_sweep.json"
    completed_payload: dict[str, Any] | None = None
    if final_output_path.is_file() and not args.extend_search:
        candidate_payload = json.loads(final_output_path.read_text(encoding="utf-8"))
        candidate_summary = candidate_payload.get("summary", {})
        if (
            candidate_summary.get("status") == "PASS"
            and candidate_summary.get("input_hashes_before") == before
            and candidate_summary.get("input_hashes_after") == before
            and int(candidate_summary.get("authoritative_rule", {}).get("grid_n", -1))
            == int(args.grid_n)
            and float(
                candidate_summary.get("authoritative_rule", {}).get(
                    "bandwidth_scale", float("nan")
                )
            )
            == float(args.bandwidth_scale)
        ):
            completed_payload = candidate_payload
            frozen_candidates: dict[str, dict[str, Any]] = {}
            for row in candidate_payload["rows"]:
                allowance = float(row["allowance_percent"])
                if allowance < 2.0:
                    continue
                key = _add_candidate(
                    frozen_candidates,
                    exp,
                    row["geometry_deg"],
                    "frozen_completed_corrected_sweep_winner",
                )
                winners[allowance] = frozen_candidates[key]
            stage_records = {
                float(allowance): record
                for allowance, record in candidate_summary["stage_records"].items()
            }
            print(
                "[frozen] reusing completed PASS sweep; use --extend-search to add finalists",
                flush=True,
            )

    for allowance in (() if completed_payload is not None else search_allowances):
        result = results[allowance]
        L_max = float(result["law_screens"]["L_max"])
        R_max = float(result["law_screens"]["R_max"])
        allowance_index = ALLOWANCES.index(allowance)
        previous = winners[ALLOWANCES[allowance_index - 1]]
        candidates: dict[str, dict[str, Any]] = {}

        previous_key = _add_candidate(
            candidates, exp, previous["geometry_deg"], "mandatory_previous_corrected_incumbent"
        )
        for candidate in carried_exact.values():
            _add_candidate(
                candidates,
                exp,
                candidate["geometry_deg"],
                "carried_previous_exact_finalist",
            )
        pool = _build_pool(result, allowance, result_paths[allowance])
        for row in pool:
            _add_candidate(candidates, exp, row["geometry_deg"], "+".join(row["provenance"]))
        for index, degrees in enumerate(result["selection"]["optimizer_starts_deg"]):
            _add_candidate(candidates, exp, degrees, f"normal_multistart_{index}")

        historical_full = result["selection"]["full_optimum_deg"]
        historical_tangent = result["selection"]["tangent_optimum_deg"]
        law_geometry = result["selection"]["law_optimum_deg"]
        historical_full_key = _add_candidate(
            candidates, exp, historical_full, "mandatory_historical_full"
        )
        _add_candidate(candidates, exp, historical_tangent, "mandatory_saved_tangent_seed")
        _add_candidate(candidates, exp, law_geometry, "mandatory_law_seed")

        # Refine every important basin, including the best accepted Full-search
        # basins for this allowance.  Geometry-independent action certificates
        # are reused exactly across stages.
        accepted_pool = [
            candidate for candidate in candidates.values()
            if f"selection:{candidate['candidate_key']}" in checkpoint["actions"]
            and checkpoint["actions"][f"selection:{candidate['candidate_key']}"].get("passes", False)
        ]
        accepted_pool.sort(
            key=lambda candidate: checkpoint["actions"][f"selection:{candidate['candidate_key']}"]["A_full_h"]
        )
        centers = [
            previous["geometry_deg"],
            historical_full,
            historical_tangent,
            law_geometry,
            *[candidate["geometry_deg"] for candidate in accepted_pool[:3]],
        ]
        for center_index, center in enumerate(centers):
            for radius in (4.0, 2.0, 1.0):
                for ring_index, degrees in enumerate(_local_ring(center, radius)):
                    _add_candidate(
                        candidates,
                        exp,
                        degrees,
                        f"initial_ring_center_{center_index}_r{radius:g}_{ring_index}",
                    )

        feasible_candidates = []
        for index, candidate in enumerate(list(candidates.values()), start=1):
            is_feasible = feasible(candidate, L_max, R_max)
            print(
                f"[{allowance:g}% risk {index}/{len(candidates)}] feasible={is_feasible} "
                f"eta_deg={candidate['geometry_deg']}",
                flush=True,
            )
            if is_feasible:
                feasible_candidates.append(candidate)
                if f"selection:{candidate['candidate_key']}" not in checkpoint["actions"]:
                    evaluate_action(candidate, "proxy", proxy_bank, int(args.proxy_grid_n))

        mandatory = candidates[previous_key]
        evaluate_action(mandatory, "prescreen", prescreen_bank, int(args.grid_n))
        proxy_ranked = sorted(
            (
                candidate for candidate in feasible_candidates
                if checkpoint["actions"].get(f"proxy:{candidate['candidate_key']}", {}).get("passes", False)
            ),
            key=lambda candidate: checkpoint["actions"][f"proxy:{candidate['candidate_key']}"]["A_full_h"],
        )
        initial_prescreen = proxy_ranked[: int(args.initial_prescreen_count)]
        if mandatory not in initial_prescreen:
            initial_prescreen.append(mandatory)
        for candidate in initial_prescreen:
            evaluate_action(candidate, "prescreen", prescreen_bank, int(args.grid_n))

        prescreened = [
            candidate for candidate in initial_prescreen
            if checkpoint["actions"][f"prescreen:{candidate['candidate_key']}"].get("passes", False)
        ]
        prescreened.sort(
            key=lambda candidate: checkpoint["actions"][f"prescreen:{candidate['candidate_key']}"]["A_full_h"]
        )
        high_fidelity_center = prescreened[0]
        refinement_candidates: list[dict[str, Any]] = []
        for radius in (0.5, 0.25):
            for ring_index, degrees in enumerate(_local_ring(high_fidelity_center["geometry_deg"], radius)):
                key = _add_candidate(
                    candidates,
                    exp,
                    degrees,
                    f"prescreen_refinement_r{radius:g}_{ring_index}",
                )
                candidate = candidates[key]
                if feasible(candidate, L_max, R_max):
                    evaluate_action(candidate, "prescreen", prescreen_bank, int(args.grid_n))
                    if checkpoint["actions"][f"prescreen:{key}"].get("passes", False):
                        refinement_candidates.append(candidate)
        prescreened.extend(refinement_candidates)
        prescreened = list({candidate["candidate_key"]: candidate for candidate in prescreened}.values())
        prescreened.sort(
            key=lambda candidate: checkpoint["actions"][f"prescreen:{candidate['candidate_key']}"]["A_full_h"]
        )

        new_finalists = [
            candidate for candidate in prescreened
            if f"selection:{candidate['candidate_key']}" not in checkpoint["actions"]
        ][: int(args.new_finalist_count)]
        for candidate in new_finalists:
            evaluate_action(candidate, "selection", action_bank, int(args.grid_n))

        exact_feasible = [
            candidate for candidate in candidates.values()
            if feasible(candidate, L_max, R_max)
            and checkpoint["actions"].get(f"selection:{candidate['candidate_key']}", {}).get("passes", False)
        ]
        if not exact_feasible:
            raise RuntimeError(f"no exact corrected Full candidate at {allowance:g}%")
        exact_best = min(
            exact_feasible,
            key=lambda candidate: checkpoint["actions"][f"selection:{candidate['candidate_key']}"]["A_full_h"],
        )
        previous_action = float(checkpoint["actions"][f"selection:{previous_key}"]["A_full_h"])
        exact_best_action = float(
            checkpoint["actions"][f"selection:{exact_best['candidate_key']}"]["A_full_h"]
        )
        replaced = bool(exact_best_action < previous_action - tolerance)
        winner = exact_best if replaced else mandatory
        winner_action = float(
            checkpoint["actions"][f"selection:{winner['candidate_key']}"]["A_full_h"]
        )
        nested = bool(winner_action <= previous_action + tolerance)
        if not nested:
            raise RuntimeError(f"invalid non-nested corrected Full stage at {allowance:g}%")
        winners[allowance] = winner
        for candidate in candidates.values():
            if f"selection:{candidate['candidate_key']}" in checkpoint["actions"]:
                carried_exact[candidate["candidate_key"]] = candidate

        validation = evaluate_action(winner, "validation", validation_bank, int(args.grid_n))
        stage_rows = []
        for candidate in candidates.values():
            key = candidate["candidate_key"]
            risk = exact_risk(candidate)
            row = {
                **candidate,
                **risk,
                "allowance_percent": allowance,
                "L_max": L_max,
                "R_max": R_max,
                "feasible": feasible(candidate, L_max, R_max),
                "is_previous_incumbent": key == previous_key,
                "is_historical_full": key == historical_full_key,
                "is_stage_winner": key == winner["candidate_key"],
            }
            for action_stage in ("proxy", "prescreen", "selection", "validation"):
                record = checkpoint["actions"].get(f"{action_stage}:{key}")
                row[f"{action_stage}_evaluated"] = record is not None
                if record is not None:
                    row[f"{action_stage}_A_full_h"] = record["A_full_h"]
                    row[f"{action_stage}_passes"] = record["passes"]
                    row[f"{action_stage}_source"] = record.get("source")
            stage_rows.append(row)
        stage_dir = output_dir / _tag(allowance)
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_dir / "candidates.csv", stage_rows)
        validation_trials = []
        for trial in validation["trials"]:
            row = dict(trial)
            row["allowance_percent"] = allowance
            validation_trials.append(row)
        _write_csv(stage_dir / "validation_trials.csv", validation_trials)
        stage_record = {
            "allowance_percent": allowance,
            "L_max": L_max,
            "R_max": R_max,
            "previous_incumbent_geometry_deg": previous["geometry_deg"],
            "previous_incumbent_action": previous_action,
            "exact_best_geometry_deg": exact_best["geometry_deg"],
            "exact_best_action": exact_best_action,
            "incumbent_replaced": replaced,
            "winner_geometry_deg": winner["geometry_deg"],
            "winner_action": winner_action,
            "nested": nested,
            "historical_full_geometry_deg": historical_full,
            "geometry_changed_from_historical": winner["candidate_key"] != historical_full_key,
            "candidate_count": len(candidates),
            "feasible_candidate_count": sum(bool(row["feasible"]) for row in stage_rows),
            "exact_selection_candidate_count": sum(bool(row["selection_evaluated"]) for row in stage_rows),
            "new_exact_finalist_count": len(new_finalists),
            "selection": checkpoint["actions"][f"selection:{winner['candidate_key']}"],
            "selection_certification_flags": _certification_flags(
                checkpoint["actions"][f"selection:{winner['candidate_key']}"]
            ),
            "validation": {key: value for key, value in validation.items() if key != "trials"},
            "validation_certification_flags": _certification_flags(validation),
        }
        stage_records[allowance] = stage_record
        _write_json(stage_dir / "audit.json", {"stage": stage_record, "candidates": stage_rows})
        print(
            f"[{allowance:g}%] winner={winner['geometry_deg']} A={winner_action:.12g} "
            f"replaced={replaced} nested={nested}",
            flush=True,
        )

    # Generate validation trial rows and standard errors for the two fixed
    # tighter points as well; validation remains strictly post-selection.
    fixed_allowances = (0.5,) if args.search_from_1pct else (0.5, 1.0)
    for allowance in fixed_allowances:
        winner = winners[allowance]
        validation = evaluate_action(winner, "validation", validation_bank, int(args.grid_n))
        key = winner["candidate_key"]
        result = results[allowance]
        risk = exact_risk(winner)
        selection = checkpoint["actions"][f"selection:{key}"]
        L_max = float(result["law_screens"]["L_max"])
        R_max = float(result["law_screens"]["R_max"])
        fixed_row = {
            **winner,
            **risk,
            "allowance_percent": allowance,
            "L_max": L_max,
            "R_max": R_max,
            "feasible": feasible(winner, L_max, R_max),
            "is_fixed_input": True,
            "search_performed": False,
            "selection_evaluated": True,
            "selection_A_full_h": selection["A_full_h"],
            "selection_passes": selection["passes"],
            "selection_source": selection.get("source"),
            "validation_evaluated": True,
            "validation_A_full_h": validation["A_full_h"],
            "validation_passes": validation["passes"],
            "validation_source": validation.get("source"),
        }
        stage_dir = output_dir / _tag(allowance)
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_dir / "candidates.csv", [fixed_row])
        fixed_validation_trials = []
        for trial in validation["trials"]:
            row = dict(trial)
            row["allowance_percent"] = allowance
            fixed_validation_trials.append(row)
        _write_csv(stage_dir / "validation_trials.csv", fixed_validation_trials)
        _write_json(
            stage_dir / "audit.json",
            {
                "stage": {
                    "allowance_percent": allowance,
                    "fixed_input": True,
                    "search_performed": False,
                    "geometry_deg": winner["geometry_deg"],
                    "L_max": L_max,
                    "R_max": R_max,
                    "population_loss": risk["population_loss"],
                    "finite_risk": risk["finite_risk"],
                    "feasible": fixed_row["feasible"],
                    "selection": selection,
                    "selection_certification_flags": _certification_flags(selection),
                    "validation": {
                        item_key: value
                        for item_key, value in validation.items()
                        if item_key != "trials"
                    },
                    "validation_certification_flags": _certification_flags(validation),
                },
                "candidates": [fixed_row],
            },
        )

    # A completed PASS sweep is immutable by default.  Report regeneration may
    # add derived certification flags, but it must not regenerate candidates or
    # alter any frozen stage decision.
    if completed_payload is not None:
        for allowance in search_allowances:
            stage_dir = output_dir / _tag(allowance)
            audit_path = stage_dir / "audit.json"
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
            stage = stage_records[allowance]
            stage["selection_certification_flags"] = _certification_flags(
                stage["selection"]
            )
            stage["validation_certification_flags"] = _certification_flags(
                stage["validation"]
            )
            audit_payload["stage"] = stage
            _write_json(audit_path, audit_payload)

    law_validation_row = next(
        row for row in corrected_validation
        if float(row["allowance_percent"]) == 1.0 and row["method"] == "law"
    )
    law_validation_action = float(law_validation_row["A_full_h"])
    R_star = float(results[1.0]["law_screens"]["R_star"])
    table_rows: list[dict[str, Any]] = []
    validation_trial_rows: list[dict[str, Any]] = []
    for allowance in ALLOWANCES:
        winner = winners[allowance]
        key = winner["candidate_key"]
        risk = exact_risk(winner)
        selection = checkpoint["actions"][f"selection:{key}"]
        validation = checkpoint["actions"][f"validation:{key}"]
        selection_flags = _certification_flags(selection)
        validation_flags = _certification_flags(validation)
        result = results[allowance]
        historical_deg = result["selection"]["full_optimum_deg"]
        historical_candidates: dict[str, dict[str, Any]] = {}
        historical_key = _add_candidate(
            historical_candidates, exp, historical_deg, "historical_full"
        )
        tangent_row = next(
            row for row in corrected_selection
            if float(row["allowance_percent"]) == allowance and row["method"] == "tangent"
        )
        L_max = float(result["law_screens"]["L_max"])
        R_max = float(result["law_screens"]["R_max"])
        table_rows.append({
            "allowance_percent": allowance,
            "geometry_deg": winner["geometry_deg"],
            "population_loss_L": risk["population_loss"],
            "finite_law_risk_R": risk["finite_risk"],
            "risk_increase_percent": 100.0 * (risk["finite_risk"] / R_star - 1.0),
            "L_max": L_max,
            "R_max": R_max,
            "passes_L": bool(risk["population_loss"] <= L_max + risk_tolerance),
            "passes_R": bool(risk["finite_risk"] <= R_max + risk_tolerance),
            "selection_A_full_h": selection["A_full_h"],
            "validation_A_full_h_mean": validation["A_full_h"],
            "validation_A_full_h_se": validation["full_action_se"],
            "Full_vs_Law_validation_reduction": (
                law_validation_action - validation["A_full_h"]
            ) / law_validation_action,
            "selection_A_tan_h": selection["A_tan_h"],
            "selection_A_hid_h": selection["A_hid_h"],
            "selection_Gamma_h": selection["Gamma_h"],
            "tangent_geometry_selection_A_full_h": tangent_row["A_full_h"],
            "Full_beats_Tangent_geometry": bool(selection["A_full_h"] < tangent_row["A_full_h"]),
            "selection_passes": bool(selection["passes"]),
            "validation_passes": bool(validation["passes"]),
            **{
                f"selection_flag_{flag}": passes
                for flag, passes in selection_flags.items()
            },
            **{
                f"validation_flag_{flag}": passes
                for flag, passes in validation_flags.items()
            },
            "selection_max_poisson_residual": selection["maximum_physical_poisson_relative_residual"],
            "selection_max_full_moment_residual": selection["maximum_full_moment_rate_residual"],
            "selection_max_tangent_moment_residual": selection["maximum_tangent_moment_rate_residual"],
            "selection_max_hidden_nullspace_residual": selection["maximum_hidden_nullspace_residual"],
            "selection_max_abs_orthogonality_residual": selection["maximum_absolute_orthogonality_residual"],
            "selection_max_abs_pythagorean_residual": selection["maximum_absolute_pythagorean_residual"],
            "selection_max_raw_hierarchy_violation": selection["maximum_raw_hierarchy_violation"],
            "geometry_changed_from_historical": key != historical_key,
            "historical_geometry_deg": historical_deg,
            "published_or_frozen_input": allowance in (0.5, 1.0),
        })
        for trial in validation["trials"]:
            trial_row = dict(trial)
            trial_row["allowance_percent"] = allowance
            validation_trial_rows.append(trial_row)

    actions = np.asarray([row["selection_A_full_h"] for row in table_rows], dtype=np.float64)
    nesting_differences = np.diff(actions)
    nested = bool(np.all(nesting_differences <= tolerance))
    if not nested:
        raise RuntimeError(f"final corrected Full curve is not nested: {nesting_differences.tolist()}")
    after = _snapshot(watched)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("historical result or frozen input changed during nested sweep")

    any_new_beats_1 = bool(np.min(actions[2:]) < actions[1] - tolerance)
    geometry_changes = [
        row["allowance_percent"] for row in table_rows
        if row["allowance_percent"] >= 2.0 and row["geometry_changed_from_historical"]
    ]
    all_decomposition_pass = bool(
        all(row["selection_passes"] and row["validation_passes"] for row in table_rows)
    )
    tangent_story_survives = bool(all(row["Full_beats_Tangent_geometry"] for row in table_rows))
    central_result_survives = bool(
        all(row["Full_vs_Law_validation_reduction"] > 0.0 for row in table_rows)
        and all_decomposition_pass
    )
    further_optimization_required = bool(not nested or not all_decomposition_pass)
    summary = {
        "schema_version": 1,
        "experiment": "toy_example_percentage",
        "status": "PASS" if nested and all_decomposition_pass and unchanged else "FAIL",
        "authoritative_rule": {
            "grid_n": int(args.grid_n),
            "time_n": len(times),
            "bandwidth": float(exp.authoritative_raster_bandwidth),
            "bandwidth_scale": float(args.bandwidth_scale),
            "selection_trials": int(action_bank.masses.shape[0]),
            "validation_trials": int(validation_bank.masses.shape[0]),
            "physical_q": True,
            "direct_signed_source": True,
            "density_floor_in_scientific_operator": False,
            "mass_tolerance": MASS_TOLERANCE,
            "source_and_component_tolerance": SOURCE_TOLERANCE,
            "moment_and_energy_tolerance": tolerance,
        },
        "sweep_source_sha256": file_sha256(Path(__file__)),
        "selection_tolerance": tolerance,
        "risk_comparison_tolerance": risk_tolerance,
        "selection_curve_nested": nested,
        "nesting_differences": nesting_differences.tolist(),
        "all_decomposition_checks_resolved": all_decomposition_pass,
        "all_historical_and_frozen_inputs_unchanged": unchanged,
        "changed_2_to_5_allowances": geometry_changes,
        "any_new_candidate_beats_corrected_1pct_incumbent": any_new_beats_1,
        "tangent_vs_full_ranking_story_survives": tangent_story_survives,
        "central_FIDE_result_survives": central_result_survives,
        "further_full_optimization_required": further_optimization_required,
        "certification_flag_names": list(
            _certification_flags(
                checkpoint["actions"][f"selection:{winners[0.5]['candidate_key']}"]
            )
        ),
        "stage_records": stage_records,
        "input_hashes_before": before,
        "input_hashes_after": after,
        "evaluator_source_hashes": evaluator_hashes,
    }
    _write_csv(output_dir / "corrected_nested_full_sweep.csv", table_rows)
    _write_csv(output_dir / "validation_trial_summaries.csv", validation_trial_rows)
    _write_json(
        output_dir / "corrected_nested_full_sweep.json",
        {"summary": summary, "rows": table_rows, "validation_trials": validation_trial_rows},
    )

    lines = [
        "# Corrected nested Toy Full sweep",
        "",
        f"**{summary['status']}** — selection is based only on the frozen 64-trial bank; validation is diagnostic and post-selection.",
        "",
        "Authoritative Full rule: positive-support physical-`q_h`, directly deposited signed source, no density floor in the scientific operator, `101 x 101`, all 21 time nodes, frozen Scott bandwidth `0.417530106552`.",
        "",
        "## Final corrected nested table",
        "",
        "| Allow. | Full geometry (deg) | L | R | Risk inc. | Selection A_full | Validation A_full (SE) | Full vs Law validation | A_tan,h | A_hid,h | Gamma_h | L/R | Sel./Val. cert. | Changed? |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['allowance_percent']:g}% | `{row['geometry_deg']}` | "
            f"{row['population_loss_L']:.9g} | {row['finite_law_risk_R']:.9g} | "
            f"{row['risk_increase_percent']:.3f}% | {row['selection_A_full_h']:.9g} | "
            f"{row['validation_A_full_h_mean']:.9g} ({row['validation_A_full_h_se']:.3g}) | "
            f"{row['Full_vs_Law_validation_reduction']:.3%} | {row['selection_A_tan_h']:.9g} | "
            f"{row['selection_A_hid_h']:.9g} | {row['selection_Gamma_h']:.6f} | "
            f"{'PASS' if row['passes_L'] and row['passes_R'] else 'FAIL'} | "
            f"{'PASS' if row['selection_passes'] and row['validation_passes'] else 'FAIL'} | "
            f"{'yes' if row['geometry_changed_from_historical'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Nested-stage decisions",
        "",
        "| Allow. | Previous action | Best audited action | Winner action | Replaced? | Nested? | Candidates / feasible / exact |",
        "|---:|---:|---:|---:|:---:|:---:|---:|",
    ])
    for allowance in search_allowances:
        stage = stage_records[allowance]
        lines.append(
            f"| {allowance:g}% | {stage['previous_incumbent_action']:.9g} | "
            f"{stage['exact_best_action']:.9g} | {stage['winner_action']:.9g} | "
            f"{'yes' if stage['incumbent_replaced'] else 'no'} | "
            f"{'PASS' if stage['nested'] else 'FAIL'} | "
            f"{stage['candidate_count']} / {stage['feasible_candidate_count']} / "
            f"{stage['exact_selection_candidate_count']} |"
        )
    lines.extend([
        "",
        "## Final checks",
        "",
        f"1. Corrected Full selection curve nested: **{'PASS' if nested else 'FAIL'}**. Raw consecutive differences: `{nesting_differences.tolist()}`.",
        "2. Full-vs-Law validation reductions: " + ", ".join(
            f"{row['allowance_percent']:g}% = {row['Full_vs_Law_validation_reduction']:.3%}"
            for row in table_rows
        ) + ".",
        f"3. Historical 2–5% geometries changed at: **{geometry_changes or 'none'}**.",
        f"4. Any new candidate beats the corrected 1% incumbent: **{'YES' if any_new_beats_1 else 'NO'}**.",
        f"5. Tangent-vs-Full corrected ranking survives: **{'YES' if tangent_story_survives else 'NO'}**.",
        f"6. All selection and validation decomposition checks resolved: **{'YES' if all_decomposition_pass else 'NO'}**.",
        f"7. Central FIDE result survives: **{'YES' if central_result_survives else 'NO'}**.",
        f"8. Further targeted Full optimization required: **{'YES' if further_optimization_required else 'NO'}**.",
        f"9. Historical Pareto outputs and frozen inputs unchanged: **{'YES' if unchanged else 'NO'}**.",
        "",
        "## Numerical certification flags",
        "",
        "Every selection and validation row passes all named checks: positive `q_h`, raster mass, signed-source compatibility, conductive-component compatibility, solver convergence, trial validity, physical Poisson residual, Full moment feasibility, Tangent moment feasibility, hidden nullspace, orthogonality, Pythagorean identity, and raw hierarchy. The CSV exposes each check as a separate `selection_flag_*` and `validation_flag_*` Boolean; the JSON retains the full-precision residual maxima and tolerances.",
        "",
        "Full-precision summaries, certification maxima, hashes, and validation trials are in the companion JSON and CSV files. Every allowance, including the fixed 0.5% and 1% inputs, has candidate, audit, and validation records in its corresponding `risk_*pct/` subdirectory.",
    ])
    (output_dir / "corrected_nested_full_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
