"""Refine only the Toy 0.5% Full design with the corrected physical-q solver.

The published Pareto result is immutable.  This standalone rerun uses the
original frozen reference/selection/validation banks and original L/R limits,
but searches with the positive-support Full rasterization selected by the
post-hoc sensitivity audit.  Search is derivative-free because the scientific
physical-q Poisson solve is a host-side sparse direct solve.
"""
from __future__ import annotations

import argparse
import csv
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
from audit_positive_rasterization import _summarize
from experiment import TrialBank


DEFAULT_PARETO = SCRIPT_DIR / "outputs" / "pareto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-dir", type=Path, default=DEFAULT_PARETO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write the fresh corrected search outside the seed/provenance Pareto tree.",
    )
    parser.add_argument("--proxy-grid-n", type=int, default=51)
    parser.add_argument("--proxy-trials", type=int, default=2)
    parser.add_argument("--prescreen-trials", type=int, default=12)
    parser.add_argument("--grid-n", type=int, default=101)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--prescreen-count", type=int, default=10)
    parser.add_argument("--finalist-count", type=int, default=4)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    normalized = []
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
        normalized.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in normalized:
            writer.writerow({
                key: json.dumps(row.get(key), separators=(",", ":"))
                if isinstance(row.get(key), (list, dict))
                else row.get(key)
                for key in keys
            })


def _bank_prefix(bank: TrialBank, count: int) -> TrialBank:
    count = min(int(count), int(bank.masses.shape[0]))
    return TrialBank(
        masses=bank.masses[:count],
        sample_indices=bank.sample_indices[:count],
        detector_z=bank.detector_z[:count],
        alphas=bank.alphas[:count],
    )


def _canonical_deg(exp: Any, degrees: Any) -> list[float]:
    eta = exp.family.canonicalize(jnp.deg2rad(jnp.asarray(degrees, dtype=jnp.float64)))
    return np.rad2deg(np.asarray(eta, dtype=np.float64)).tolist()


def _add_candidate(
    candidates: dict[str, dict[str, Any]], exp: Any, degrees: Any, provenance: str
) -> str:
    deg = _canonical_deg(exp, degrees)
    geometry = np.deg2rad(np.asarray(deg, dtype=np.float64)).tolist()
    key = geometry_key(geometry)
    if key not in candidates:
        candidates[key] = {
            "candidate_key": key,
            "geometry": geometry,
            "geometry_deg": deg,
            "provenance": [],
        }
    if provenance not in candidates[key]["provenance"]:
        candidates[key]["provenance"].append(provenance)
    return key


def _local_ring(center: Any, radius: float) -> list[list[float]]:
    center = np.asarray(center, dtype=np.float64)
    out = []
    for k in range(8):
        angle = 2.0 * math.pi * k / 8.0
        out.append((center + radius * np.asarray([math.cos(angle), math.sin(angle)])).tolist())
    return out


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in paths}


def main() -> None:
    args = parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    point, first = _strict_common_artifacts(pareto)
    cfg = first["config"]
    exp, action_bank, times = _load_experiment(point, cfg)
    validation_bank = _load_bank(point / "validation_bank.npz")
    result_path = pareto / "risk_0p5pct" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    corrected_validation_path = pareto / "toy_corrected_validation_rescore.json"
    corrected_validation = json.loads(corrected_validation_path.read_text(encoding="utf-8"))
    saved_full_validation_row = next(
        row for row in corrected_validation["rows"]
        if float(row["allowance_percent"]) == 0.5 and row["method"] == "full"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pareto / "corrected_full_rerun_0p5pct"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    law_trials = int(result["randomness"]["law_trials_effective"])
    law_bank = _bank_prefix(action_bank, law_trials)
    proxy_bank = _bank_prefix(action_bank, args.proxy_trials)
    prescreen_bank = _bank_prefix(action_bank, args.prescreen_trials)
    L_max = float(result["law_screens"]["L_max"])
    R_max = float(result["law_screens"]["R_max"])
    feasibility_tol = 1.0e-12
    decomposition_tol = float(cfg["validity"]["tangent_lower_bound_tol"])
    time_weights = np.asarray(exp.time_w, dtype=np.float64)
    min_sep_deg = float(cfg["measurement"]["min_sep_deg"])

    watched = [
        pareto / "pareto.json",
        result_path,
        point / "reference.npz",
        point / "reference_bank.npz",
        point / "selection_bank.npz",
        point / "validation_bank.npz",
        corrected_validation_path,
    ]
    before = _snapshot(watched)
    evaluator_sources = [
        SCRIPT_DIR / "experiment.py",
        REPO_ROOT / "src" / "mfsi" / "raster.py",
        REPO_ROOT / "src" / "mfsi" / "poisson.py",
        REPO_ROOT / "src" / "mfsi" / "decomposition.py",
    ]
    evaluator_source_hashes = _snapshot(evaluator_sources)

    checkpoint_path = output_dir / "checkpoint.json"
    settings = {
        "schema_version": 1,
        "proxy_grid_n": int(args.proxy_grid_n),
        "proxy_trials": int(args.proxy_trials),
        "prescreen_trials": int(args.prescreen_trials),
        "grid_n": int(args.grid_n),
        "bandwidth_scale": float(args.bandwidth_scale),
        "prescreen_count": int(args.prescreen_count),
        "finalist_count": int(args.finalist_count),
    }
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"settings": settings, "risk": {}, "actions": {}}
    )
    if checkpoint["settings"] != settings:
        raise RuntimeError("existing rerun checkpoint has different settings")
    checkpoint_identity = {
        "input_hashes": before,
        "evaluator_source_hashes": evaluator_source_hashes,
    }
    if checkpoint.get("identity") not in (None, checkpoint_identity):
        raise RuntimeError("existing rerun checkpoint has different inputs or evaluator sources")
    checkpoint["identity"] = checkpoint_identity
    _write_json(checkpoint_path, checkpoint)

    candidates: dict[str, dict[str, Any]] = {}
    pool = _build_pool(result, 0.5, result_path)
    for row in pool:
        _add_candidate(candidates, exp, row["geometry_deg"], "+".join(row["provenance"]))
    for index, degrees in enumerate(result["selection"]["optimizer_starts_deg"]):
        _add_candidate(candidates, exp, degrees, f"normal_multistart_{index}")

    saved_full_deg = result["selection"]["full_optimum_deg"]
    saved_tangent_deg = result["selection"]["tangent_optimum_deg"]
    saved_law_deg = result["selection"]["law_optimum_deg"]
    old_pool_best = min(
        (
            row for row in pool
            if "full_search_audit_7" in row["provenance"]
        ),
        key=lambda _: 0,
    )["geometry_deg"]
    # Derive control keys through the same canonicalization path used for every
    # candidate.  Recomputing them directly from JSON degrees can differ by one
    # binary ulp and would defeat exact string-key matching.
    saved_full_key = _add_candidate(candidates, exp, saved_full_deg, "mandatory_published_full")
    old_pool_best_key = _add_candidate(candidates, exp, old_pool_best, "mandatory_existing_pool_winner")
    centers = [old_pool_best, saved_full_deg, saved_tangent_deg, saved_law_deg]
    for center_index, center in enumerate(centers):
        for radius in (2.0, 1.0):
            for ring_index, degrees in enumerate(_local_ring(center, radius)):
                _add_candidate(
                    candidates,
                    exp,
                    degrees,
                    f"initial_ring_center_{center_index}_r{radius:g}_{ring_index}",
                )

    def save_checkpoint() -> None:
        _write_json(checkpoint_path, checkpoint)

    def risk(candidate: dict[str, Any]) -> dict[str, Any]:
        key = candidate["candidate_key"]
        if key not in checkpoint["risk"]:
            eta = jnp.asarray(candidate["geometry"], dtype=jnp.float64)
            population = exp.exact_population_result(eta)
            finite = exp.exact_finite_result(eta, law_bank)
            separation = (
                (candidate["geometry_deg"][1] - candidate["geometry_deg"][0]) % 180.0
            )
            separation = min(separation, 180.0 - separation)
            feasible = bool(
                population["valid"]
                and finite["valid"]
                and float(population["value"]) <= L_max + feasibility_tol
                and float(finite["value"]) <= R_max + feasibility_tol
                and separation >= min_sep_deg - feasibility_tol
            )
            checkpoint["risk"][key] = {
                "population_valid": bool(population["valid"]),
                "finite_valid": bool(finite["valid"]),
                "population_loss": float(population["value"]),
                "finite_risk": float(finite["value"]),
                "projective_separation_deg": float(separation),
                "feasible": feasible,
            }
            save_checkpoint()
        return checkpoint["risk"][key]

    def action(candidate: dict[str, Any], stage: str, bank: TrialBank, grid_n: int) -> dict[str, Any]:
        cache_key = f"{stage}:{candidate['candidate_key']}"
        if cache_key not in checkpoint["actions"]:
            print(
                f"[{stage}] eta_deg={candidate['geometry_deg']} trials={bank.masses.shape[0]} "
                f"grid={grid_n}",
                flush=True,
            )
            started = time.perf_counter()
            rows = exp.evaluate_common_discretization_decomposition_exact(
                jnp.asarray(candidate["geometry"], dtype=jnp.float64),
                bank,
                grid_n=grid_n,
                bandwidth_scale=float(args.bandwidth_scale),
                progress_desc=f"corrected Full {stage}",
            )
            summary, _ = _summarize(
                rows,
                method="full",
                time_weights=time_weights,
                moment_tolerance=decomposition_tol,
                energy_tolerance=decomposition_tol,
            )
            checkpoint["actions"][cache_key] = {
                **summary,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            save_checkpoint()
        return checkpoint["actions"][cache_key]

    # Screen all saved/audited solutions, the normal multistarts, and fixed local
    # rings.  Only exact L/R-feasible points receive a corrected Full evaluation.
    initial_keys = list(candidates)
    for index, key in enumerate(initial_keys, start=1):
        candidate = candidates[key]
        rec = risk(candidate)
        print(
            f"[risk {index}/{len(initial_keys)}] feasible={rec['feasible']} "
            f"eta_deg={candidate['geometry_deg']}",
            flush=True,
        )
        if rec["feasible"]:
            action(candidate, "proxy", proxy_bank, int(args.proxy_grid_n))

    # Deterministic coordinate refinement around the current proxy incumbent.
    # Each round shrinks the angular step and retains all evaluated points.
    for step in (0.5, 0.25, 0.125):
        feasible = [
            candidate for candidate in candidates.values()
            if risk(candidate)["feasible"]
            and f"proxy:{candidate['candidate_key']}" in checkpoint["actions"]
            and checkpoint["actions"][f"proxy:{candidate['candidate_key']}"]["passes"]
        ]
        incumbent = min(
            feasible,
            key=lambda item: checkpoint["actions"][f"proxy:{item['candidate_key']}"]["A_full_h"],
        )
        new_keys = []
        for ring_index, degrees in enumerate(_local_ring(incumbent["geometry_deg"], step)):
            key = _add_candidate(
                candidates,
                exp,
                degrees,
                f"adaptive_ring_r{step:g}_{ring_index}",
            )
            new_keys.append(key)
        for key in new_keys:
            candidate = candidates[key]
            if risk(candidate)["feasible"]:
                action(candidate, "proxy", proxy_bank, int(args.proxy_grid_n))

    proxy_ranked = sorted(
        (
            candidate for candidate in candidates.values()
            if risk(candidate)["feasible"]
            and checkpoint["actions"].get(f"proxy:{candidate['candidate_key']}", {}).get("passes", False)
        ),
        key=lambda item: checkpoint["actions"][f"proxy:{item['candidate_key']}"]["A_full_h"],
    )
    prescreen = proxy_ranked[: max(1, int(args.prescreen_count))]
    mandatory_keys = {saved_full_key, old_pool_best_key}
    for candidate in candidates.values():
        if candidate["candidate_key"] in mandatory_keys and candidate not in prescreen:
            prescreen.append(candidate)
    for candidate in prescreen:
        action(candidate, "prescreen", prescreen_bank, int(args.grid_n))

    prescreen_ranked = sorted(
        (candidate for candidate in prescreen if checkpoint["actions"][f"prescreen:{candidate['candidate_key']}"]["passes"]),
        key=lambda item: checkpoint["actions"][f"prescreen:{item['candidate_key']}"]["A_full_h"],
    )
    finalists = prescreen_ranked[: max(1, int(args.finalist_count))]
    for candidate in candidates.values():
        if candidate["candidate_key"] in mandatory_keys and candidate not in finalists:
            finalists.append(candidate)
    for candidate in finalists:
        action(candidate, "selection", action_bank, int(args.grid_n))

    valid_finalists = [
        candidate for candidate in finalists
        if risk(candidate)["feasible"]
        and checkpoint["actions"][f"selection:{candidate['candidate_key']}"]["passes"]
    ]
    if not valid_finalists:
        raise RuntimeError("no corrected Full finalist passed exact risk/action checks")
    winner = min(
        valid_finalists,
        key=lambda item: checkpoint["actions"][f"selection:{item['candidate_key']}"]["A_full_h"],
    )
    action(winner, "validation", validation_bank, int(args.grid_n))

    rows = []
    for candidate in candidates.values():
        key = candidate["candidate_key"]
        row = {**candidate, **risk(candidate)}
        for stage in ("proxy", "prescreen", "selection", "validation"):
            rec = checkpoint["actions"].get(f"{stage}:{key}")
            row[f"{stage}_evaluated"] = rec is not None
            if rec is not None:
                row[f"{stage}_A_full_h"] = rec["A_full_h"]
                row[f"{stage}_passes"] = rec["passes"]
                row[f"{stage}_trial_count"] = rec["trial_count"]
                row[f"{stage}_grid_n"] = rec["grid_n"]
        row["is_saved_full"] = key == saved_full_key
        row["is_existing_pool_winner"] = key == old_pool_best_key
        row["is_rerun_winner"] = key == winner["candidate_key"]
        rows.append(row)

    saved = next(row for row in rows if row["is_saved_full"])
    existing = next(row for row in rows if row["is_existing_pool_winner"])
    selected = checkpoint["actions"][f"selection:{winner['candidate_key']}"]
    saved_selection = checkpoint["actions"][f"selection:{saved['candidate_key']}"]
    existing_selection = checkpoint["actions"][f"selection:{existing['candidate_key']}"]
    validation = checkpoint["actions"][f"validation:{winner['candidate_key']}"]
    after = _snapshot(watched)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("a published result or frozen input changed during the rerun")

    summary = {
        "schema_version": 1,
        "experiment": "toy_example_percentage",
        "allowance_percent": 0.5,
        "status": "PASS" if selected["passes"] and validation["passes"] and unchanged else "FAIL",
        "search_method": "deterministic multistart derivative-free pattern refinement",
        "search_objective": "corrected positive-support physical-q Full action",
        "proxy_protocol": {
            "grid_n": int(args.proxy_grid_n),
            "trial_count": int(proxy_bank.masses.shape[0]),
            "time_grid": np.asarray(times, dtype=np.float64).tolist(),
            "bank_prefix_sha256_source": file_sha256(point / "selection_bank.npz"),
        },
        "prescreen_protocol": {
            "grid_n": int(args.grid_n),
            "trial_count": int(prescreen_bank.masses.shape[0]),
        },
        "authoritative_protocol": {
            "grid_n": int(args.grid_n),
            "bandwidth_scale": float(args.bandwidth_scale),
            "bandwidth": float(exp.authoritative_raster_bandwidth) * float(args.bandwidth_scale),
            "selection_trial_count": int(action_bank.masses.shape[0]),
            "validation_trial_count": int(validation_bank.masses.shape[0]),
            "time_grid": np.asarray(times, dtype=np.float64).tolist(),
        },
        "risk_constraints": {
            "L_max": L_max,
            "R_max": R_max,
            "comparison_tolerance": feasibility_tol,
            "risk_definitions_unchanged": True,
        },
        "candidate_counts": {
            "generated": len(candidates),
            "risk_feasible": sum(bool(risk(candidate)["feasible"]) for candidate in candidates.values()),
            "proxy_evaluated": sum(f"proxy:{key}" in checkpoint["actions"] for key in candidates),
            "prescreen_evaluated": len(prescreen),
            "current_selection_finalists": len(finalists),
            "selection_evaluated": sum(
                f"selection:{key}" in checkpoint["actions"] for key in candidates
            ),
        },
        "saved_full": {
            "geometry_deg": saved_full_deg,
            "selection_A_full_h": saved_selection["A_full_h"],
            "validation_A_full_h": float(saved_full_validation_row["A_full_h"]),
        },
        "existing_pool_winner": {
            "geometry_deg": old_pool_best,
            "selection_A_full_h": existing_selection["A_full_h"],
        },
        "rerun_winner": {
            "geometry_deg": winner["geometry_deg"],
            "population_loss": risk(winner)["population_loss"],
            "finite_risk": risk(winner)["finite_risk"],
            "selection": selected,
            "validation": validation,
        },
        "lower_than_saved_full": bool(selected["A_full_h"] < saved_selection["A_full_h"] - decomposition_tol),
        "lower_than_existing_pool_winner": bool(selected["A_full_h"] < existing_selection["A_full_h"] - decomposition_tol),
        "saved_full_minus_rerun_action": float(saved_selection["A_full_h"] - selected["A_full_h"]),
        "saved_full_minus_rerun_validation_action": float(
            saved_full_validation_row["A_full_h"] - validation["A_full_h"]
        ),
        "existing_pool_winner_minus_rerun_action": float(existing_selection["A_full_h"] - selected["A_full_h"]),
        "published_results_unchanged": unchanged,
        "frozen_inputs_unchanged": unchanged,
        "input_hashes_before": before,
        "input_hashes_after": after,
        "evaluator_source_hashes": evaluator_source_hashes,
        "published_result_update_performed": False,
    }
    _write_csv(output_dir / "corrected_full_rerun_candidates.csv", rows)
    _write_json(output_dir / "corrected_full_rerun.json", {"summary": summary, "candidates": rows})
    lines = [
        "# Toy 0.5% corrected Full rerun",
        "",
        f"**{summary['status']}** — the published Pareto result and frozen inputs remain unchanged.",
        "",
        r"This isolated rerun uses deterministic multistart pattern refinement because the corrected physical-\(q_h\) sparse direct solve is not differentiable. The proxy and authoritative stages use the same positive-support deposition and physical-\(q_h\) equation; only raster resolution and frozen-bank trial count differ during navigation/prescreening.",
        "",
        "## Result",
        "",
        "| Candidate | Geometry (degrees) | Selection corrected Full action | Gap to rerun winner |",
        "|---|---:|---:|---:|",
        f"| Published Full | `{saved_full_deg}` | {saved_selection['A_full_h']:.12g} | {saved_selection['A_full_h'] - selected['A_full_h']:.12g} |",
        f"| Prior audited-pool winner | `{old_pool_best}` | {existing_selection['A_full_h']:.12g} | {existing_selection['A_full_h'] - selected['A_full_h']:.12g} |",
        f"| Corrected rerun winner | `{winner['geometry_deg']}` | {selected['A_full_h']:.12g} | 0 |",
        "",
        f"The rerun winner has `L={risk(winner)['population_loss']:.12g}` (limit `{L_max:.12g}`) and `R={risk(winner)['finite_risk']:.12g}` (limit `{R_max:.12g}`).",
        f"Its independent 128-trial validation corrected Full action is `{validation['A_full_h']:.12g}`, versus `{saved_full_validation_row['A_full_h']:.12g}` for the published endpoint (improvement `{saved_full_validation_row['A_full_h'] - validation['A_full_h']:.12g}`).",
        "",
        "## Numerical certificate",
        "",
        "| Quantity | Selection maximum | Validation maximum |",
        "|---|---:|---:|",
        f"| Physical Poisson relative residual | {selected['maximum_physical_poisson_relative_residual']:.3e} | {validation['maximum_physical_poisson_relative_residual']:.3e} |",
        f"| Full moment-rate residual | {selected['maximum_full_moment_rate_residual']:.3e} | {validation['maximum_full_moment_rate_residual']:.3e} |",
        f"| Tangent moment-rate residual | {selected['maximum_tangent_moment_rate_residual']:.3e} | {validation['maximum_tangent_moment_rate_residual']:.3e} |",
        f"| Hidden-nullspace residual | {selected['maximum_hidden_nullspace_residual']:.3e} | {validation['maximum_hidden_nullspace_residual']:.3e} |",
        f"| Absolute orthogonality residual | {selected['maximum_absolute_orthogonality_residual']:.3e} | {validation['maximum_absolute_orthogonality_residual']:.3e} |",
        f"| Absolute Pythagorean residual | {selected['maximum_absolute_pythagorean_residual']:.3e} | {validation['maximum_absolute_pythagorean_residual']:.3e} |",
        "",
        "## Protocol and immutability",
        "",
        f"- Search proxy: `{args.proxy_grid_n}x{args.proxy_grid_n}`, `{proxy_bank.masses.shape[0]}` frozen selection trials, all `{len(times)}` time nodes.",
        f"- Prescreen: `{args.grid_n}x{args.grid_n}`, `{prescreen_bank.masses.shape[0]}` frozen selection trials.",
        f"- Final selection: `{args.grid_n}x{args.grid_n}`, all `{action_bank.masses.shape[0]}` frozen selection trials.",
        f"- Independent validation: `{args.grid_n}x{args.grid_n}`, all `{validation_bank.masses.shape[0]}` frozen validation trials.",
        f"- Published results/frozen banks unchanged: **{'YES' if unchanged else 'NO'}**.",
        "- No published candidate was replaced and no risk definition was changed.",
    ]
    (output_dir / "corrected_full_rerun.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
