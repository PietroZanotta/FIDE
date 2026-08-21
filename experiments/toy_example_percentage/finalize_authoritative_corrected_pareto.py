"""Build the clean authoritative Toy percentage result tree from fresh audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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

from action_decomposition_audit import file_sha256, geometry_key
from audit_action_decomposition import _load_experiment
from audit_corrected_all_candidates import _load_bank
from audit_positive_rasterization import _summarize
from percentage_pareto_visualization import method_records, save_method_tables
from run_corrected_nested_full_sweep import _certification_flags
from visualize_pareto import save_pareto_suite


ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
METHODS = ("law", "tangent", "full")


def _tag(allowance: float) -> str:
    return f"risk_{f'{allowance:g}'.replace('.', 'p')}pct"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field), separators=(",", ":"))
                if isinstance(row.get(field), (list, dict))
                else row.get(field)
                for field in fields
            })


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_se(values: Any) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(len(array)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pareto", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-corrected", type=Path)
    parser.add_argument("--grid-n", type=int, default=101)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed_pareto.expanduser().resolve()
    source = args.source_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    nested_path = output / "corrected_nested_full_sweep.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    nested_rows = {float(row["allowance_percent"]): row for row in nested["rows"]}
    cfg = json.loads((seed / _tag(0.5) / "result.json").read_text(encoding="utf-8"))["config"]
    exp, selection_bank, times = _load_experiment(source, cfg)
    validation_bank = _load_bank(source / "validation_bank.npz")
    tolerance = float(cfg["validity"]["tangent_lower_bound_tol"])
    time_weights = np.asarray(exp.time_w, dtype=np.float64)
    grid_n = int(args.grid_n)
    bandwidth_scale = float(args.bandwidth_scale)

    source_files = [
        source / "reference.npz",
        source / "reference_bank.npz",
        source / "selection_bank.npz",
        source / "validation_bank.npz",
    ]
    source_hashes_before = {str(path): file_sha256(path) for path in source_files}
    frozen_dir = output / "frozen_inputs"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest = []
    for source_path in source_files:
        target = frozen_dir / source_path.name
        if not target.exists():
            try:
                os.link(source_path, target)
            except OSError:
                shutil.copy2(source_path, target)
        frozen_manifest.append({
            "name": source_path.name,
            "source": str(source_path),
            "active": str(target),
            "sha256": file_sha256(source_path),
            "active_sha256": file_sha256(target),
        })

    legacy: dict[float, dict[str, Any]] = {
        allowance: json.loads((seed / _tag(allowance) / "result.json").read_text(encoding="utf-8"))
        for allowance in ALLOWANCES
    }
    checkpoint_path = output / "corrected_method_evaluation_checkpoint.json"
    identity = {
        "grid_n": grid_n,
        "bandwidth_scale": bandwidth_scale,
        "source_hashes": source_hashes_before,
        "nested_sha256": _sha256(nested_path),
        "evaluator_hashes": {
            str(path): file_sha256(path)
            for path in (
                SCRIPT_DIR / "experiment.py",
                REPO_ROOT / "src/mfsi/raster.py",
                REPO_ROOT / "src/mfsi/poisson.py",
                REPO_ROOT / "src/mfsi/decomposition.py",
            )
        },
    }
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.is_file()
        else {"identity": identity, "evaluations": {}}
    )
    if checkpoint.get("identity") != identity:
        raise RuntimeError("method-evaluation checkpoint identity mismatch")

    def evaluate(degrees: list[float], bank_name: str) -> dict[str, Any]:
        geometry = np.deg2rad(np.asarray(degrees, dtype=np.float64)).tolist()
        key = f"{bank_name}:{geometry_key(geometry)}"
        if key not in checkpoint["evaluations"]:
            bank = selection_bank if bank_name == "selection" else validation_bank
            rows = exp.evaluate_common_discretization_decomposition_exact(
                jnp.asarray(geometry, dtype=jnp.float64),
                bank,
                grid_n=grid_n,
                bandwidth_scale=bandwidth_scale,
                progress_desc=f"authoritative {bank_name} {key[-20:]}",
            )
            summary, _ = _summarize(
                rows,
                method="candidate",
                time_weights=time_weights,
                moment_tolerance=tolerance,
                energy_tolerance=tolerance,
            )
            trial_actions = [float(row["full_action"]) for row in rows]
            _, action_se = _mean_se(trial_actions)
            checkpoint["evaluations"][key] = {
                "summary": summary,
                "full_action_se": action_se,
                "trial_full_actions": trial_actions,
            }
            _write_json(checkpoint_path, checkpoint)
        return checkpoint["evaluations"][key]

    # Freshly evaluate each distinct frozen Law/Tangent geometry. Full summaries
    # come from the fresh nested search and are not copied from pre-correction data.
    for allowance in ALLOWANCES:
        for method in ("law", "tangent"):
            degrees = legacy[allowance]["selection"][f"{method}_optimum_deg"]
            evaluate(degrees, "selection")
            evaluate(degrees, "validation")

    selection_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    # The nested sweep already contains the independently evaluated Full trials.
    # Preserve those rows when adding the freshly rescored Law/Tangent trials so
    # the authoritative validation table covers every published design.
    validation_trial_rows: list[dict[str, Any]] = [
        {**row, "method": "full"}
        for row in nested.get("validation_trials", [])
    ]
    diagnostics: list[dict[str, Any]] = []
    previous_corrected = None
    if args.previous_corrected is not None and args.previous_corrected.is_file():
        previous_corrected = json.loads(args.previous_corrected.read_text(encoding="utf-8"))
    previous_by_allowance = {
        float(row["allowance_percent"]): row
        for row in (previous_corrected or {}).get("rows", [])
    }

    for allowance in ALLOWANCES:
        old = legacy[allowance]
        full_row = nested_rows[allowance]
        stage_audit = json.loads((output / _tag(allowance) / "audit.json").read_text(encoding="utf-8"))["stage"]
        full_selection = stage_audit["selection"]
        full_validation = stage_audit["validation"]
        methods: dict[str, dict[str, Any]] = {}
        for method in ("law", "tangent"):
            degrees = old["selection"][f"{method}_optimum_deg"]
            sel_eval = evaluate(degrees, "selection")
            val_eval = evaluate(degrees, "validation")
            cert = old["selection_certificates"][method]
            old_val = old["validation"][method]
            methods[method] = {
                "geometry_deg": degrees,
                "L": float(cert["L_selection"]),
                "R": float(cert["R_selection"]),
                "selection": sel_eval["summary"],
                "validation": val_eval["summary"],
                "validation_se": float(val_eval["full_action_se"]),
                "validation_risk": old_val.get("law_risk", {}),
                "validation_trials": val_eval["trial_full_actions"],
            }
        methods["full"] = {
            "geometry_deg": full_row["geometry_deg"],
            "L": float(full_row["population_loss_L"]),
            "R": float(full_row["finite_law_risk_R"]),
            "selection": full_selection,
            "validation": full_validation,
            "validation_se": float(full_row["validation_A_full_h_se"]),
            "validation_risk": {},
            "validation_trials": [],
        }
        law_validation_action = float(methods["law"]["validation"]["A_full_h"])
        result = {
            "schema_version": 1,
            "experiment": "toy_example",
            "authoritative_corrected": True,
            "risk_allowance_percent": allowance,
            "config": cfg,
            "law_screens": old["law_screens"],
            "randomness": old["randomness"],
            "selection": {
                f"{method}_optimum_deg": methods[method]["geometry_deg"]
                for method in METHODS
            },
            "selection_certificates": {},
            "validation": {},
            "contrasts": {},
            "authoritative_full_rule": nested["summary"]["authoritative_rule"],
            "frozen_input_manifest": str((frozen_dir / "manifest.json").resolve()),
        }
        for method in METHODS:
            method_data = methods[method]
            sel = method_data["selection"]
            val = method_data["validation"]
            cert_flags = _certification_flags(sel)
            val_flags = _certification_flags(val)
            result["selection_certificates"][method] = {
                "L_selection": method_data["L"],
                "R_selection": method_data["R"],
                "L_max": float(old["law_screens"]["L_max"]),
                "R_max": float(old["law_screens"]["R_max"]),
                "full_action_selection": float(sel["A_full_h"]),
                "tangent_action_selection": float(sel["A_tan_h"]),
                "hidden_action_selection": float(sel["A_hid_h"]),
                "Gamma_h_selection": float(sel["Gamma_h"]),
                "certified": bool(
                    method_data["L"] <= float(old["law_screens"]["L_max"]) + 1e-12
                    and method_data["R"] <= float(old["law_screens"]["R_max"]) + 1e-12
                    and cert_flags["all"]
                ),
                "numerical_flags": cert_flags,
            }
            result["validation"][method] = {
                "law_risk": method_data["validation_risk"],
                "full_action": {
                    "mean": float(val["A_full_h"]),
                    "se": float(method_data["validation_se"]),
                },
                "A_tan_h": float(val["A_tan_h"]),
                "A_hid_h": float(val["A_hid_h"]),
                "Gamma_h": float(val["Gamma_h"]),
                "valid_fraction": 1.0,
                "numerical_flags": val_flags,
            }
            common = {
                "allowance_percent": allowance,
                "method": method,
                "geometry_deg": method_data["geometry_deg"],
                "L": method_data["L"],
                "R": method_data["R"],
            }
            selection_rows.append({
                **common,
                "A_full_h": sel["A_full_h"],
                "A_tan_h": sel["A_tan_h"],
                "A_hid_h": sel["A_hid_h"],
                "Gamma_h": sel["Gamma_h"],
                "certified": result["selection_certificates"][method]["certified"],
                **{f"flag_{name}": passed for name, passed in cert_flags.items()},
            })
            validation_rows.append({
                **common,
                "A_full_h_mean": val["A_full_h"],
                "A_full_h_se": method_data["validation_se"],
                "A_tan_h": val["A_tan_h"],
                "A_hid_h": val["A_hid_h"],
                "Gamma_h": val["Gamma_h"],
                "Full_vs_Law_reduction": (
                    law_validation_action - float(val["A_full_h"])
                ) / law_validation_action,
                **{f"flag_{name}": passed for name, passed in val_flags.items()},
            })
            for phase, summary in (("selection", sel), ("validation", val)):
                diagnostics.append({
                    "allowance_percent": allowance,
                    "method": method,
                    "phase": phase,
                    **summary,
                    **{f"flag_{name}": passed for name, passed in _certification_flags(summary).items()},
                })
            if method != "full":
                for trial, action in enumerate(method_data["validation_trials"]):
                    validation_trial_rows.append({
                        "allowance_percent": allowance,
                        "method": method,
                        "trial": trial,
                        "full_action": action,
                    })
        reduction = (
            law_validation_action - float(full_validation["A_full_h"])
        ) / law_validation_action
        result["contrasts"]["full_vs_law_full_action_reduction"] = {
            "ratio_of_means_reduction": reduction
        }
        point_dir = output / _tag(allowance)
        _write_json(point_dir / "result.json", result)
        previous = previous_by_allowance.get(allowance)
        previous_delta = (
            float(full_row["selection_A_full_h"]) - float(previous["selection_A_full_h"])
            if previous is not None else None
        )
        pareto_rows.append({
            "risk_allowance_percent": allowance,
            "risk_allowance_fraction": allowance / 100.0,
            "R_star": float(old["law_screens"]["R_star"]),
            "R_max": float(old["law_screens"]["R_max"]),
            "full_R_selection": float(full_row["finite_law_risk_R"]),
            "full_R_excess_selection": float(full_row["finite_law_risk_R"])
            - float(old["law_screens"]["R_star"]),
            "full_A_selection": float(full_row["selection_A_full_h"]),
            "full_A_validation_mean": float(full_row["validation_A_full_h_mean"]),
            "full_A_validation_se": float(full_row["validation_A_full_h_se"]),
            "Full_vs_Law_validation_reduction": reduction,
            "full_certified": bool(result["selection_certificates"]["full"]["certified"]),
            "result": str((point_dir / "result.json").resolve()),
            "previous_corrected_selection_action_delta": previous_delta,
        })

    actions = np.asarray([row["full_A_selection"] for row in pareto_rows], dtype=np.float64)
    nested_differences = np.diff(actions)
    nested_passes = bool(np.all(nested_differences <= tolerance))
    all_certified = bool(
        all(bool(row["flag_all"]) for row in selection_rows + validation_rows)
        and all(bool(row["certified"]) for row in selection_rows)
        and all(bool(row["full_certified"]) for row in pareto_rows)
    )
    if not nested_passes or not all_certified:
        raise RuntimeError("authoritative corrected Pareto certification failed")

    source_hashes_after = {str(path): file_sha256(path) for path in source_files}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("frozen source input changed during finalization")
    _write_json(frozen_dir / "manifest.json", {
        "schema_version": 1,
        "inputs": frozen_manifest,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "unchanged": True,
    })
    _write_csv(output / "pareto.csv", pareto_rows)
    _write_json(output / "pareto.json", pareto_rows)
    _write_csv(output / "pareto_methods_selection.csv", selection_rows)
    _write_csv(output / "pareto_methods_validation.csv", validation_rows)
    _write_csv(output / "validation_trial_summaries.csv", validation_trial_rows)
    _write_csv(output / "positive_raster_decomposition_diagnostics.csv", diagnostics)
    maxima_fields = (
        "maximum_mass_error",
        "maximum_source_compatibility_error",
        "maximum_physical_poisson_relative_residual",
        "maximum_full_moment_rate_residual",
        "maximum_tangent_moment_rate_residual",
        "maximum_hidden_nullspace_residual",
        "maximum_absolute_orthogonality_residual",
        "maximum_absolute_pythagorean_residual",
        "maximum_raw_hierarchy_violation",
    )
    global_maxima = {
        field: max(float(row[field]) for row in diagnostics) for field in maxima_fields
    }
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "authoritative_rule": nested["summary"]["authoritative_rule"],
        "selection_curve_nested": nested_passes,
        "nested_differences": nested_differences.tolist(),
        "all_certificates_pass": all_certified,
        "global_numerical_maxima": global_maxima,
        "frozen_inputs_unchanged": source_hashes_before == source_hashes_after,
        "previous_corrected_comparison": [
            {
                "allowance_percent": row["risk_allowance_percent"],
                "selection_action_delta": row["previous_corrected_selection_action_delta"],
            }
            for row in pareto_rows
        ],
    }
    _write_json(output / "positive_raster_decomposition_diagnostics.json", {
        "summary": summary,
        "rows": diagnostics,
    })
    lines = [
        "# Authoritative corrected Toy positive-raster/decomposition audit",
        "",
        "**PASS** — every final Law, Tangent, and Full geometry passes on both frozen selection and independent validation banks.",
        "",
        "| Quantity | Global maximum |",
        "|:---|---:|",
        *[f"| `{field}` | {value:.6e} |" for field, value in global_maxima.items()],
        "",
        f"Corrected Full selection curve nested: **{'PASS' if nested_passes else 'FAIL'}**. Differences: `{nested_differences.tolist()}`.",
        "",
        "No residual or hierarchy value was clipped.",
    ]
    (output / "positive_raster_decomposition_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    records = method_records(pareto_rows, output)
    save_method_tables(records, output)
    save_pareto_suite(pareto_rows, output, output, dpi=int(args.dpi))
    _write_json(output / "authoritative_run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
