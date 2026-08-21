"""Targeted Tangent repair for the saved 4% and 5% vortex Pareto points.

Only Tangent is searched and, when improved, replaced.  Full geometries,
certificates, validation summaries, and the risk definition are treated as
immutable inputs and verified after the update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from audit_action_decomposition import _load_experiment, _strict_common_artifacts
from experiment import ObservationTrialBank
from mfsi.design import optimize_multistart_candidates, random_point_sensor_starts
from run_pareto import _row, _save
from selection import (
    _audit_action,
    _box_projector,
    _configured_stage_seeds,
    _dedupe,
    _geometry_constraints,
    _local_cloud,
    _opt_cfg,
    _optimizer_progress,
    _optimizer_starts,
    _prefix_bank,
    _rank_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pareto-dir", type=Path, default=SCRIPT_DIR / "outputs" / "pareto"
    )
    parser.add_argument("--allowance", type=float, nargs="+", default=[4.0, 5.0])
    parser.add_argument(
        "--no-apply", action="store_true", help="report candidates without updating Tangent"
    )
    parser.add_argument(
        "--finalize-from-saved",
        action="store_true",
        help="finalize a previously applied repair without rerunning optimization",
    )
    return parser.parse_args()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tag(percent: float) -> str:
    return f"risk_{f'{percent:g}'.replace('.', 'p')}pct"


def _validation_bank(point: Path) -> ObservationTrialBank:
    with np.load(point / "validation_bank.npz", allow_pickle=False) as bank:
        return ObservationTrialBank(
            sample_indices=jnp.asarray(bank["sample_indices"], dtype=jnp.int32),
            detector_z=jnp.asarray(bank["detector_z"], dtype=jnp.float64),
        )


def _mean_se(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "se": float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _candidate_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _search(point: Path) -> tuple[dict[str, Any], Any, ObservationTrialBank, list[dict[str, Any]], float]:
    result = json.loads((point / "result.json").read_text(encoding="utf-8"))
    cfg = result["config"]
    exp, action_bank, _ = _load_experiment(point, cfg)
    law_bank = _prefix_bank(action_bank, int(cfg["randomness"]["law_trials"]))
    opt = cfg["optimization"]
    measurement = cfg["measurement"]

    margin = float(
        measurement.get(
            "boundary_margin", 2.0 * float(measurement.get("sensor_width", 0.12))
        )
    )
    starts = random_point_sensor_starts(
        jax.random.PRNGKey(int(cfg["seed"]) + 17),
        int(opt["start_count"]),
        n_sensors=int(measurement.get("n_sensors", 4)),
        x_bounds=(exp.grid.x_min + margin, exp.grid.x_max - margin),
        y_bounds=(exp.grid.y_min + margin, exp.grid.y_max - margin),
        min_sep=float(measurement.get("min_sep", 0.24)),
        oversample=int(opt.get("start_oversample", 128)),
    )
    law_eta = jnp.asarray(result["selection"]["law_optimum"], dtype=jnp.float64)
    population_eta = jnp.asarray(result["selection"]["population_optimum"], dtype=jnp.float64)
    old_tangent = jnp.asarray(result["selection"]["tangent_optimum"], dtype=jnp.float64)
    full_eta = jnp.asarray(result["selection"]["full_optimum"], dtype=jnp.float64)
    parameter_count = 2 * int(exp.family.n_sensors)
    configured = _configured_stage_seeds(
        opt, "tangent_seed_etas", parameter_count=parameter_count
    )

    normal_local = _local_cloud(
        exp,
        [law_eta, population_eta] + configured,
        count_per_center=int(opt.get("tangent_local_starts", 12)),
        scale=float(opt.get("tangent_local_scale", 0.08)),
        seed=int(cfg["seed"]) + 401,
    )
    normal_starts = jnp.stack(
        _dedupe(exp, [law_eta, population_eta] + configured + normal_local + list(starts))
    )
    normal_optimizer_starts = _optimizer_starts(normal_starts, cfg, "tangent")
    repair_local = _local_cloud(
        exp,
        [full_eta, old_tangent],
        count_per_center=int(opt.get("tangent_local_starts", 12)),
        scale=float(opt.get("tangent_local_scale", 0.08)),
        seed=int(cfg["seed"]) + 1401,
    )
    optimizer_starts = jnp.stack(
        _dedupe(exp, [full_eta, old_tangent] + list(normal_optimizer_starts))
    )

    law_grad_bank = _prefix_bank(
        law_bank, int(opt.get("law_gradient_trials", min(4, len(law_bank.sample_indices))))
    )
    grad_tangent_bank = _prefix_bank(
        action_bank,
        int(opt.get("tangent_gradient_trials", min(4, len(action_bank.sample_indices)))),
    )
    epsilon_l = float(cfg["law"]["epsilon_l"])
    fast_l_anchor = float(jax.jit(exp.population_loss)(population_eta))
    fast_l_max = fast_l_anchor + epsilon_l
    l_scale = max(epsilon_l, 1.0e-6 * max(abs(fast_l_anchor), 1.0), 1.0e-10)
    fast_r_anchor = float(jax.jit(lambda eta: exp.finite_risk(eta, law_grad_bank))(law_eta))
    relative_limit = float(cfg["law"]["max_relative_risk_violation"])
    fast_epsilon_r = relative_limit * abs(fast_r_anchor)
    fast_r_max = fast_r_anchor + fast_epsilon_r
    r_scale = max(fast_epsilon_r, 1.0e-6 * max(abs(fast_r_anchor), 1.0), 1.0e-10)

    def law_slack(eta):
        return jnp.maximum(
            (exp.population_loss(eta) - fast_l_max) / l_scale,
            (exp.finite_risk(eta, law_grad_bank) - fast_r_max) / r_scale,
        )

    constraints = _geometry_constraints(exp) + ((law_slack, 0.0),)
    raw_objective = lambda eta: exp.tangent_action_gradient(eta, grad_tangent_bank)
    anchor = max(float(raw_objective(law_eta)), 1.0e-12)
    objective = lambda eta: raw_objective(eta) / anchor
    optimizer_cfg = _opt_cfg(cfg, "tangent")
    started = time.perf_counter()
    optimized = optimize_multistart_candidates(
        objective,
        optimizer_starts,
        optimizer_cfg,
        constraints=constraints,
        canonicalize=exp.family.canonicalize,
        project_iterate=_box_projector(exp),
        vectorize_starts=False,
        progress_callback=_optimizer_progress("targeted tangent repair"),
    )
    pool = _dedupe(
        exp,
        [full_eta, old_tangent]
        + list(normal_starts)
        + repair_local
        + [candidate.eta for candidate in optimized],
    )
    ranked = _rank_pool(exp, pool, raw_objective, constraints)
    selected, audit_rows, _ = _audit_action(
        "targeted tangent repair",
        exp,
        ranked,
        law_bank,
        action_bank,
        L_max=float(result["law_screens"]["L_max"]),
        R_max=float(result["law_screens"]["R_max"]),
        exact_action=lambda eta: exp.exact_tangent_result(eta, action_bank),
        audit_limit=int(opt.get("tangent_exact_audit_candidates", 30)),
        finalist_count=int(opt.get("tangent_exact_rescore_candidates", 10)),
        mandatory=[full_eta, old_tangent, law_eta, population_eta] + configured,
    )
    return result, exp, action_bank, audit_rows, time.perf_counter() - started


def _apply_tangent(
    point: Path,
    result: dict[str, Any],
    exp: Any,
    action_bank: ObservationTrialBank,
    selected: list[float],
    audit_rows: list[dict[str, Any]],
) -> None:
    eta = jnp.asarray(selected, dtype=jnp.float64)
    screens = result["law_screens"]
    exact_l = exp.exact_population_result(eta)
    exact_r = exp.exact_finite_result(
        eta, _prefix_bank(action_bank, int(result["config"]["randomness"]["law_trials"]))
    )
    tangent_exact = exp.exact_tangent_result(eta, action_bank)
    full_exact = exp.exact_full_result(eta, action_bank)
    validation = exp.evaluate_trials_exact(
        eta, _validation_bank(point), progress_desc="validation repaired tangent"
    )
    for row in validation:
        row["design"] = "tangent"
    valid = [row for row in validation if row["valid"]]
    lower_bound = [
        row["tangent_lower_bound_violation"]
        for row in valid
        if np.isfinite(row["tangent_lower_bound_violation"])
    ]
    summary = {
        "eta": list(map(float, selected)),
        "centers": np.asarray(exp.family.centers(eta), dtype=np.float64).tolist(),
        "law_risk": _mean_se([row["law_risk"] for row in valid]),
        "tangent_action": _mean_se([row["tangent_action"] for row in valid]),
        "full_action": _mean_se([row["full_action"] for row in valid]),
        "valid_fraction": float(len(valid) / len(validation)),
        "tangent_lower_bound_check": {
            "max_violation": float(max(lower_bound, default=0.0)),
            "tolerance": float(result["config"]["validity"].get("tangent_lower_bound_tol", 1.0e-6)),
        },
    }
    certificate = {
        "required_screens": ["L", "R"],
        "L_selection": float(exact_l["value"]),
        "L_star": float(screens["L_star"]),
        "L_max": float(screens["L_max"]),
        "L_excess_from_star": float(exact_l["value"] - screens["L_star"]),
        "L_slack_to_max": float(screens["L_max"] - exact_l["value"]),
        "passes_L": bool(exact_l["valid"] and exact_l["value"] <= screens["L_max"] + 1.0e-12),
        "R_selection": float(exact_r["value"]),
        "R_star": float(screens["R_star"]),
        "R_max": float(screens["R_max"]),
        "R_excess_from_star": float(exact_r["value"] - screens["R_star"]),
        "R_slack_to_max": float(screens["R_max"] - exact_r["value"]),
        "passes_R": bool(exact_r["valid"] and exact_r["value"] <= screens["R_max"] + 1.0e-12),
    }
    certificate["certified"] = bool(certificate["passes_L"] and certificate["passes_R"])
    if not certificate["certified"]:
        raise RuntimeError("refined Tangent candidate failed the frozen exact risk screens")

    result["selection"]["tangent_optimum"] = list(map(float, selected))
    result["selection_centers"]["tangent"] = summary["centers"]
    result["selection_certificates"]["tangent"] = certificate
    result["selection_audit"]["tangent"] = audit_rows
    result["validation"]["tangent"] = summary
    (point / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    candidate_path = point / "result.candidate_summary.csv"
    candidates = _candidate_summary_rows(candidate_path)
    for row in candidates:
        if row["design"] == "tangent":
            row.update(
                {
                    "eta": json.dumps(list(map(float, selected))),
                    "centers": json.dumps(summary["centers"]),
                    "population_loss_selection": float(exact_l["value"]),
                    "finite_risk_selection": float(exact_r["value"]),
                    "tangent_action_selection": float(tangent_exact["value"]),
                    "full_action_selection": float(full_exact["value"]),
                    "validation_law_mean": summary["law_risk"]["mean"],
                    "validation_full_action_mean": summary["full_action"]["mean"],
                    "validation_valid_fraction": summary["valid_fraction"],
                }
            )
    _write_csv(candidate_path, candidates)

    validation_path = point / "result.validation_trials.csv"
    with validation_path.open(newline="", encoding="utf-8") as handle:
        old_validation = list(csv.DictReader(handle))
    retained = [row for row in old_validation if row["design"] != "tangent"]
    _write_csv(validation_path, retained + validation)


def main() -> None:
    args = parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    _strict_common_artifacts(pareto)
    if args.finalize_from_saved:
        previous_audit_path = pareto / "action_decomposition_audit.csv"
        with previous_audit_path.open(newline="", encoding="utf-8") as handle:
            previous_audit = list(csv.DictReader(handle))
        reports = []
        for allowance in args.allowance:
            point = pareto / _tag(allowance)
            result = json.loads((point / "result.json").read_text(encoding="utf-8"))
            candidates = _candidate_summary_rows(point / "result.candidate_summary.csv")
            tangent = next(row for row in candidates if row["design"] == "tangent")
            previous = next(
                row
                for row in previous_audit
                if float(row["allowance_percent"]) == float(allowance)
                and row["method"] == "tangent"
            )
            full_previous = next(
                row
                for row in previous_audit
                if float(row["allowance_percent"]) == float(allowance)
                and row["method"] == "full"
            )
            old_action = float(previous["A_tan"])
            selected_action = float(tangent["tangent_action_selection"])
            selected = result["selection"]["tangent_optimum"]
            full_eta = result["selection"]["full_optimum"]
            reports.append(
                {
                    "allowance_percent": float(allowance),
                    "old_tangent_eta": json.loads(previous["geometry"]),
                    "old_tangent_action": old_action,
                    "full_seed_eta": full_eta,
                    "full_seed_tangent_action": float(full_previous["A_tan"]),
                    "selected_eta": selected,
                    "selected_tangent_action": selected_action,
                    "absolute_improvement": old_action - selected_action,
                    "relative_improvement": (old_action - selected_action) / old_action,
                    "lower_feasible_tangent_found": selected_action < old_action,
                    "selected_is_full_geometry": bool(
                        np.allclose(selected, full_eta, rtol=0.0, atol=1.0e-12)
                    ),
                    "applied": True,
                    "full_payload_unchanged": True,
                    "risk_definition_unchanged": True,
                    "elapsed_seconds": None,
                    "exact_finalist_rows": result["selection_audit"]["tangent"],
                }
            )
        pareto_rows = []
        for item in json.loads((pareto / "pareto.json").read_text(encoding="utf-8")):
            allowance = float(item["risk_allowance_percent"])
            point = pareto / _tag(allowance)
            result = json.loads((point / "result.json").read_text(encoding="utf-8"))
            pareto_rows.append(_row(result, allowance, point / "result.json"))
        _save(pareto_rows, pareto)
        output = pareto / "tangent_refinement_audit.json"
        output.write_text(
            json.dumps({"schema_version": 1, "reports": reports}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(output), "reports": reports}, indent=2))
        return
    reports = []
    for allowance in args.allowance:
        point = pareto / _tag(allowance)
        result_before = json.loads((point / "result.json").read_text(encoding="utf-8"))
        immutable_full = {
            "selection": result_before["selection"]["full_optimum"],
            "centers": result_before["selection_centers"]["full"],
            "certificate": result_before["selection_certificates"]["full"],
            "validation": result_before["validation"]["full"],
            "selection_audit": result_before["selection_audit"]["full"],
            "law_screens": result_before["law_screens"],
            "law_definition": result_before["config"]["law"],
        }
        immutable_hash = _stable_hash(immutable_full)
        old_eta = result_before["selection"]["tangent_optimum"]
        old_action = float(
            next(
                row["tangent_action_selection"]
                for row in _candidate_summary_rows(point / "result.candidate_summary.csv")
                if row["design"] == "tangent"
            )
        )
        result, exp, action_bank, audit_rows, elapsed = _search(point)
        valid_rows = [row for row in audit_rows if row["valid"] and np.isfinite(row["objective"])]
        best = min(valid_rows, key=lambda row: row["objective"])
        selected = list(map(float, best["eta"]))
        best_action = float(best["objective"])
        improvement = old_action - best_action
        changed = bool(improvement > 1.0e-12 * max(1.0, abs(old_action)))
        if changed and not args.no_apply:
            _apply_tangent(point, result, exp, action_bank, selected, audit_rows)
        result_after = json.loads((point / "result.json").read_text(encoding="utf-8"))
        if _stable_hash(
            {
                "selection": result_after["selection"]["full_optimum"],
                "centers": result_after["selection_centers"]["full"],
                "certificate": result_after["selection_certificates"]["full"],
                "validation": result_after["validation"]["full"],
                "selection_audit": result_after["selection_audit"]["full"],
                "law_screens": result_after["law_screens"],
                "law_definition": result_after["config"]["law"],
            }
        ) != immutable_hash:
            raise RuntimeError("Full result changed during Tangent-only repair")
        full_eta = result_before["selection"]["full_optimum"]
        reports.append(
            {
                "allowance_percent": float(allowance),
                "old_tangent_eta": old_eta,
                "old_tangent_action": old_action,
                "full_seed_eta": full_eta,
                "full_seed_tangent_action": float(
                    exp.exact_tangent_result(jnp.asarray(full_eta), action_bank)["value"]
                ),
                "selected_eta": selected,
                "selected_tangent_action": best_action,
                "absolute_improvement": improvement,
                "relative_improvement": improvement / old_action,
                "lower_feasible_tangent_found": changed,
                "selected_is_full_geometry": bool(
                    np.allclose(selected, full_eta, rtol=0.0, atol=1.0e-12)
                ),
                "applied": bool(changed and not args.no_apply),
                "full_payload_unchanged": True,
                "risk_definition_unchanged": True,
                "elapsed_seconds": elapsed,
                "exact_finalist_rows": audit_rows,
            }
        )
        (pareto / "tangent_refinement_audit.json").write_text(
            json.dumps({"schema_version": 1, "reports": reports}, indent=2) + "\n",
            encoding="utf-8",
        )

    if not args.no_apply:
        pareto_rows = []
        for item in json.loads((pareto / "pareto.json").read_text(encoding="utf-8")):
            allowance = float(item["risk_allowance_percent"])
            point = pareto / _tag(allowance)
            result = json.loads((point / "result.json").read_text(encoding="utf-8"))
            pareto_rows.append(_row(result, allowance, point / "result.json"))
        _save(pareto_rows, pareto)
    output = pareto / "tangent_refinement_audit.json"
    output.write_text(json.dumps({"schema_version": 1, "reports": reports}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "reports": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
