"""Fail-closed finalizer for the robust active-nematic Pareto sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from mfsi.cache import file_sha256
from mfsi.io import write_json

from robust_selection import RobustPhysicalViewExperiment
from run import make_experiment, reference_seeds
from run_pareto import _build_views, load_observation_bank
from unbalanced_reference import endpoint_pair_mass_schedule
from unbalanced_state import TwoSpeciesDefectBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pareto-dir", type=Path,
        default=SCRIPT_DIR / "outputs" / "pareto_robust",
    )
    parser.add_argument(
        "--skip-recompute", action="store_true",
        help="perform structural receipt checks only (intended for quick smoke tests)",
    )
    return parser.parse_args()


def _same(left, right, tolerance: float = 1.0e-10) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=tolerance))


def _selected(result: dict[str, Any], design: str) -> dict[str, Any]:
    stage = "law" if design == "law" else ("tangent" if design == "tangent" else "full")
    eta = result["designs"][design]
    rows = [
        row for row in result["selection_candidates"][stage]
        if _same(row["eta"], eta)
    ]
    if not rows:
        raise RuntimeError(f"selected {design} geometry is absent from its exact audit")
    return min(rows, key=lambda row: float(row["audit"]["value"]))


def _full_action_audit(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = result["designs"][design]
    rows = [
        row for row in result["selection_candidates"]["full"]
        if _same(row["eta"], eta)
    ]
    if not rows:
        raise RuntimeError(f"selected {design} geometry is absent from the Full audit")
    return min(rows, key=lambda row: float(row["audit"]["value"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _trial_failures(result: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = result["config"]
    validity = cfg["validity"]
    weights = cfg["unbalanced"]
    wp = float(weights.get("species_weight_plus", 1.0))
    wm = float(weights.get("species_weight_minus", 1.0))
    failures = []
    for design, payload in result["validation_designs"].items():
        for view in payload["views"]:
            for row in view["rows"]:
                reasons = []
                decomposition = (
                    wp * (float(row["move_action_plus"]) + float(row["reaction_action_plus"]))
                    + wm * (float(row["move_action_minus"]) + float(row["reaction_action_minus"]))
                )
                total = float(row["full_unbalanced_action_total"])
                if not bool(row["valid"]):
                    reasons.append("invalid")
                if not np.isclose(decomposition, total, rtol=2.0e-8, atol=1.0e-10):
                    reasons.append("move/reaction decomposition")
                if float(row["max_calibration_residual"]) > float(validity["max_calibration_residual"]):
                    reasons.append("calibration residual")
                if float(row["min_ess_fraction"]) < float(validity["min_ess_fraction"]):
                    reasons.append("ESS")
                if float(row["max_screened_pde_relative_residual"]) > float(
                    validity["max_screened_pde_relative_residual"]
                ):
                    reasons.append("physical screened-PDE residual")
                if not all(math.isfinite(float(row[key])) for key in (
                    "law_risk_total", "full_unbalanced_action_total",
                    "move_action_plus", "reaction_action_plus",
                    "move_action_minus", "reaction_action_minus",
                )):
                    reasons.append("non-finite metric")
                if reasons:
                    failures.append({
                        "allowance_percent": result["allowance_percent"],
                        "design": design,
                        "view": view["label"],
                        "trial": row["trial"],
                        "reasons": reasons,
                    })
    return failures


def _hash_frozen_inputs(frozen: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(candidate for candidate in frozen.rglob("*") if candidate.is_file()):
        if path.name == "manifest.json":
            continue
        rows.append({
            "path": str(path.relative_to(frozen)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    payload = {"schema_version": 1, "files": rows}
    write_json(frozen / "manifest.json", payload)
    return payload


def _verify_existing_frozen_manifest(frozen: Path) -> list[str]:
    path = frozen / "manifest.json"
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest.get("files", []):
        candidate = frozen / row["path"]
        if not candidate.is_file():
            failures.append(f"frozen input disappeared: {row['path']}")
            continue
        if candidate.stat().st_size != int(row["bytes"]):
            failures.append(f"frozen input size changed: {row['path']}")
        elif file_sha256(candidate) != row["sha256"]:
            failures.append(f"frozen input hash changed: {row['path']}")
    return failures


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only fields that are expected to vary along the nested sweep."""
    cleaned = json.loads(json.dumps(config))
    cleaned.get("law", {}).pop("max_relative_risk_violation", None)
    optimization = cleaned.get("optimization", {})
    for key in (
        "fixed_law_anchor",
        "pareto_incumbent_full_eta",
        "pareto_methodology_version",
    ):
        optimization.pop(key, None)
    return cleaned


def _load_frozen_references(
    cfg: dict[str, Any], bank: TwoSpeciesDefectBank, frozen: Path, train_runs
) -> dict[int, dict[str, Any]]:
    schedule = endpoint_pair_mass_schedule(
        bank,
        run_indices=np.asarray(train_runs, dtype=np.int64),
        minimum_mass=float(cfg["unbalanced"]["minimum_mass"]),
    )
    references = {}
    for seed in reference_seeds(cfg):
        seed_dir = frozen / f"reference_seed_{seed}"
        reference: dict[str, Any] = {"schedule": schedule}
        for species in ("plus", "minus"):
            path = seed_dir / f"{species}_reference_bank.npz"
            with np.load(path, allow_pickle=False) as saved:
                reference[species] = {
                    "nodes": jnp.asarray(saved["nodes"]),
                    "velocity": jnp.asarray(saved["velocity"]),
                    "weights": jnp.asarray(saved["weights"]),
                }
        references[int(seed)] = reference
    return references


def _audit_close(saved: dict[str, Any], fresh: dict[str, Any]) -> bool:
    if bool(saved["valid"]) != bool(fresh["valid"]):
        return False
    if not np.isclose(
        float(saved["value"]), float(fresh["value"]), rtol=2.0e-6, atol=2.0e-7
    ):
        return False
    return bool(np.allclose(
        np.asarray(saved.get("view_values", []), dtype=np.float64),
        np.asarray(fresh.get("view_values", []), dtype=np.float64),
        rtol=2.0e-6,
        atol=2.0e-7,
    ))


def _independent_recompute(
    pareto: Path,
    results: list[dict[str, Any]],
    view_manifest: dict[str, Any],
) -> list[str]:
    """Rebuild experiments from frozen inputs and re-evaluate every winner."""
    frozen = pareto / "frozen_inputs"
    cfg = results[0]["config"]
    bank = TwoSpeciesDefectBank.load(frozen / "two_species_defect_bank.npz")
    references = _load_frozen_references(
        cfg, bank, frozen, view_manifest["train_runs"]
    )
    selection_views = _build_views(
        cfg,
        bank,
        references,
        [tuple(row) for row in view_manifest["selection_run_views"]],
        phase="selection",
    )
    validation_views = _build_views(
        cfg,
        bank,
        references,
        [tuple(row) for row in view_manifest["validation_run_views"]],
        phase="validation",
    )
    robust = RobustPhysicalViewExperiment(cfg, selection_views)
    selection_bank = load_observation_bank(frozen / "selection_bank.npz")
    validation_bank = load_observation_bank(frozen / "validation_bank.npz")
    selection_cache: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    validation_cache: dict[tuple[float, ...], dict[str, dict[str, float]]] = {}
    failures = []

    def selection_audit(eta, metric):
        key = (metric, tuple(float(value) for value in eta))
        if key not in selection_cache:
            selection_cache[key] = robust.audit_metric(
                jnp.asarray(eta, dtype=jnp.float64), selection_bank, metric
            )
        return selection_cache[key]

    def validation_summaries(eta):
        key = tuple(float(value) for value in eta)
        if key not in validation_cache:
            summaries = {}
            for view in validation_views:
                rows = view.experiment.certified_trial_rows(
                    jnp.asarray(eta, dtype=jnp.float64), validation_bank
                )
                summaries[view.label] = {
                    "law_risk_total": float(np.mean([row["law_risk_total"] for row in rows])),
                    "full_unbalanced_action_total": float(np.mean([
                        row["full_unbalanced_action_total"] for row in rows
                    ])),
                    "valid_trials": int(sum(bool(row["valid"]) for row in rows)),
                    "trials": int(len(rows)),
                }
            validation_cache[key] = summaries
        return validation_cache[key]

    for result in results:
        percent = float(result["allowance_percent"])
        stage_metric = {
            "law": "law_risk",
            "tangent": "tangent_action",
            "unbalanced_full": "full_action",
        }
        for design, metric in stage_metric.items():
            saved = _selected(result, design)["audit"]
            fresh = selection_audit(result["designs"][design], metric)
            if not _audit_close(saved, fresh):
                failures.append(
                    f"{percent:g}% {design} independent {metric} audit mismatch"
                )
            saved_full = _full_action_audit(result, design)["audit"]
            fresh_full = selection_audit(result["designs"][design], "full_action")
            if not _audit_close(saved_full, fresh_full):
                failures.append(
                    f"{percent:g}% {design} independent Full audit mismatch"
                )

            fresh_validation = validation_summaries(result["designs"][design])
            saved_views = {
                row["label"]: row for row in result["validation_designs"][design]["views"]
            }
            for label, fresh_summary in fresh_validation.items():
                saved_summary = saved_views[label]["summary"]
                for metric_name in (
                    "law_risk_total", "full_unbalanced_action_total"
                ):
                    saved_value = float(saved_summary["metrics"][metric_name]["mean"])
                    if not np.isclose(
                        saved_value,
                        fresh_summary[metric_name],
                        rtol=2.0e-6,
                        atol=2.0e-7,
                    ):
                        failures.append(
                            f"{percent:g}% {design} {label} validation {metric_name} mismatch"
                        )
                if (
                    int(saved_summary["valid_trials"]) != fresh_summary["valid_trials"]
                    or int(saved_summary["trials"]) != fresh_summary["trials"]
                ):
                    failures.append(
                        f"{percent:g}% {design} {label} validation validity mismatch"
                    )
    return failures


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    mpl_config = path.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    x = np.asarray([row["allowance_percent"] for row in rows])
    law = np.asarray([row["validation_law_action"] for row in rows])
    full = np.asarray([row["validation_full_action"] for row in rows])
    full_se = np.asarray([row["validation_full_action_se_across_views"] for row in rows])
    reduction = 100.0 * (1.0 - full / law)
    move = np.asarray([row["validation_full_move_action"] for row in rows])
    reaction = np.asarray([row["validation_full_reaction_action"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    axes[0].plot(x, [row["selection_law_action"] for row in rows], "o--", label="Law")
    axes[0].plot(x, [row["selection_full_action"] for row in rows], "o-", label="Full")
    axes[0].set(title="Robust selection", xlabel="Allowed extra risk (%)", ylabel="Worst-view Full action")
    axes[0].legend()
    axes[1].plot(x, law, "o--", label="Law")
    axes[1].errorbar(x, full, yerr=full_se, marker="o", label="Full", capsize=3)
    axes[1].set(title="Held-out physical views", xlabel="Allowed extra risk (%)", ylabel="Full action")
    axes[1].legend()
    axes[2].stackplot(x, move, reaction, labels=("Move", "Reaction"), alpha=0.8)
    axes[2].set(
        title="Full decomposition",
        xlabel="Allowed extra risk (%)",
        ylabel="Validation action",
    )
    reduction_axis = axes[2].twinx()
    reduction_axis.plot(x, reduction, "ko-", label="Reduction")
    reduction_axis.set_ylabel("Full vs Law reduction (%)")
    handles, labels = axes[2].get_legend_handles_labels()
    extra_handles, extra_labels = reduction_axis.get_legend_handles_labels()
    axes[2].legend(handles + extra_handles, labels + extra_labels, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pareto = args.pareto_dir.expanduser().resolve()
    manifest = json.loads((pareto / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("selection_completed_before_validation"):
        raise RuntimeError("manifest does not certify selection-before-validation ordering")
    points = sorted(manifest["points"], key=lambda row: float(row["allowance_percent"]))
    results = [json.loads(Path(row["result"]).read_text(encoding="utf-8")) for row in points]
    if not results:
        raise RuntimeError("Pareto manifest contains no points")

    law_eta = results[0]["designs"]["law"]
    law_view_star = np.asarray(results[0]["risk_view_star"], dtype=np.float64)
    selection_actions = []
    rows = []
    trial_failures = []
    selection_failures = []
    selection_failures.extend(
        _verify_existing_frozen_manifest(pareto / "frozen_inputs")
    )
    view_manifest = json.loads(
        (pareto / "frozen_inputs" / "view_manifest.json").read_text(encoding="utf-8")
    )
    expected_selection_views = view_manifest["selection_views"]
    expected_view_labels = {row["label"] for row in view_manifest["validation_views"]}
    baseline_scientific_config = _scientific_config(results[0]["config"])
    frozen_config = json.loads(
        (pareto / "frozen_inputs" / "effective_config.json").read_text(encoding="utf-8")
    )
    if _scientific_config(frozen_config) != baseline_scientific_config:
        selection_failures.append("point configuration differs from frozen effective configuration")
    for result in results:
        percent = float(result["allowance_percent"])
        if _scientific_config(result["config"]) != baseline_scientific_config:
            selection_failures.append(f"{percent:g}% changed scientific configuration")
        if result.get("selection_views") != expected_selection_views:
            selection_failures.append(f"{percent:g}% changed frozen selection views")
        if not _same(result["designs"]["law"], law_eta):
            selection_failures.append("Law geometry changed across allowances")
        if not _same(result["risk_view_star"], law_view_star):
            selection_failures.append("Law view anchors changed across allowances")
        expected_maxima = law_view_star + (percent / 100.0) * np.abs(law_view_star)
        if not _same(result["risk_view_maxima"], expected_maxima, tolerance=1.0e-12):
            selection_failures.append(f"{percent:g}% has incorrect view-specific ceilings")
        selected = {name: _selected(result, name) for name in (
            "law", "tangent", "unbalanced_full"
        )}
        full_action_audits = {
            name: _full_action_audit(result, name)
            for name in ("law", "tangent", "unbalanced_full")
        }
        for name, row in selected.items():
            if not row["audit"]["valid"] or not row["law_screen"]["valid"]:
                selection_failures.append(
                    f"{result['allowance_percent']:g}% {name} has an invalid exact audit"
                )
            risk = np.asarray(row["law_screen"].get("view_values", []), dtype=np.float64)
            maximum = np.asarray(result["risk_view_maxima"], dtype=np.float64)
            if risk.shape != maximum.shape or np.any(risk > maximum + 1.0e-12):
                selection_failures.append(
                    f"{result['allowance_percent']:g}% {name} fails a view-specific risk ceiling"
                )
        for name, row in full_action_audits.items():
            if not row["audit"]["valid"] or not row["law_screen"]["valid"]:
                selection_failures.append(
                    f"{result['allowance_percent']:g}% {name} lacks a valid complete-bank Full audit"
                )
        if not result["selection_certified"]:
            selection_failures.append(f"{result['allowance_percent']:g}% is not selection-certified")
        selection_actions.append(
            float(full_action_audits["unbalanced_full"]["audit"]["value"])
        )
        for name, payload in result["validation_designs"].items():
            labels = {row["label"] for row in payload["views"]}
            if labels != expected_view_labels:
                selection_failures.append(
                    f"{result['allowance_percent']:g}% {name} has the wrong validation views"
                )
        trial_failures.extend(_trial_failures(result))

        validation_law = result["validation_designs"]["law"]["physical_view_action"]
        validation_full = result["validation_designs"]["unbalanced_full"]["physical_view_action"]
        metrics = result["validation_designs"]["unbalanced_full"]["summary"]["metrics"]
        unbalanced = result["config"]["unbalanced"]
        wp = float(unbalanced.get("species_weight_plus", 1.0))
        wm = float(unbalanced.get("species_weight_minus", 1.0))
        move = float(
            wp * metrics["move_action_plus"]["mean"]
            + wm * metrics["move_action_minus"]["mean"]
        )
        reaction = float(
            wp * metrics["reaction_action_plus"]["mean"]
            + wm * metrics["reaction_action_minus"]["mean"]
        )
        rows.append({
            "allowance_percent": float(result["allowance_percent"]),
            "full_eta": result["designs"]["unbalanced_full"],
            "risk_star_worst_view": float(result["risk_star"]),
            "risk_max_worst_view": float(result["risk_max"]),
            "selection_law_action": float(full_action_audits["law"]["audit"]["value"]),
            "selection_full_action": float(full_action_audits["unbalanced_full"]["audit"]["value"]),
            "selection_full_risk_worst_view": float(selected["unbalanced_full"]["law_screen"]["value"]),
            "validation_law_action": float(validation_law["mean"]),
            "validation_full_action": float(validation_full["mean"]),
            "validation_full_action_se_across_views": float(validation_full["se_across_views"]),
            "validation_full_vs_law_reduction": (
                1.0 - float(validation_full["mean"]) / float(validation_law["mean"])
            ),
            "validation_full_move_action": move,
            "validation_full_reaction_action": reaction,
            "validation_reaction_fraction": reaction / (move + reaction),
            "validation_views": int(validation_full["views"]),
            "selection_certified": bool(result["selection_certified"]),
        })

    tolerance = float(results[0]["config"].get("pareto", {}).get("nesting_tolerance", 1.0e-8))
    differences = np.diff(np.asarray(selection_actions, dtype=np.float64))
    nested = bool(np.all(differences <= tolerance))
    if not nested:
        selection_failures.append(f"selection action is not nested: {differences.tolist()}")
    recompute_failures = []
    if not args.skip_recompute:
        print("independently recomputing frozen selection and validation winners", flush=True)
        recompute_failures = _independent_recompute(
            pareto, results, view_manifest
        )
        selection_failures.extend(recompute_failures)
    frozen_manifest = _hash_frozen_inputs(pareto / "frozen_inputs")
    diagnostic = {
        "schema_version": 1,
        "passes": not selection_failures and not trial_failures,
        "nested": nested,
        "nested_differences": differences.tolist(),
        "selection_failures": selection_failures,
        "validation_trial_failures": trial_failures,
        "independent_recompute_performed": not args.skip_recompute,
        "independent_recompute_failures": recompute_failures,
        "frozen_input_files": len(frozen_manifest["files"]),
    }
    write_json(pareto / "authoritative_certification_diagnostic.json", diagnostic)
    if not diagnostic["passes"]:
        raise RuntimeError(json.dumps(diagnostic, indent=2))

    summary = {
        "schema_version": 1,
        "experiment": "active_nematic_unbalance_robust_authoritative_pareto",
        "selection_rule": "worst action across physical folds and learned-reference seeds",
        "risk_rule": "each view must satisfy its own Law-relative percentage ceiling",
        "selection_completed_before_validation": True,
        "common_law_geometry": law_eta,
        "selection_curve_nested": nested,
        "all_selection_and_validation_checks_pass": True,
        "rows": rows,
    }
    write_json(pareto / "authoritative_pareto.json", summary)
    _write_csv(pareto / "authoritative_pareto.csv", rows)
    markdown = [
        "# Authoritative robust active-nematic Pareto table",
        "",
        "Selection minimizes worst-view action across frozen physical folds and learned-reference seeds. "
        "Every view has its own percentage ceiling relative to the common Law geometry. Validation was "
        "run only after the complete selection sweep was frozen.",
        "",
        "| Allowance | Selection Law A | Selection Full A | Validation Law A | Validation Full A ± view SE | Full vs Law | Move | Reaction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['allowance_percent']:g}% | {row['selection_law_action']:.6f} | "
            f"{row['selection_full_action']:.6f} | {row['validation_law_action']:.6f} | "
            f"{row['validation_full_action']:.6f} ± {row['validation_full_action_se_across_views']:.6f} | "
            f"{100.0 * row['validation_full_vs_law_reduction']:.2f}% | "
            f"{row['validation_full_move_action']:.6f} | {row['validation_full_reaction_action']:.6f} |"
        )
    markdown.extend([
        "",
        f"Nested robust Full selection curve: **PASS**. Differences: `{differences.tolist()}`.",
        "",
        "All exact risk screens, calibration/ESS gates, physical screened-PDE residuals, "
        "and move/reaction decompositions pass.",
    ])
    (pareto / "authoritative_pareto.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    _plot(rows, pareto / "authoritative_pareto.png")
    print(f"authoritative active-nematic Pareto: {pareto / 'authoritative_pareto.md'}")


if __name__ == "__main__":
    main()
