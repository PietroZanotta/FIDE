"""Run a robust, nested percentage-risk Pareto sweep for active nematics."""

from __future__ import annotations

import argparse
import copy
import csv
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
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
jax.config.update("jax_enable_x64", True)

from mfsi.cache import file_sha256, fingerprint
from mfsi.config import load_config
from mfsi.io import jsonable, write_json

from domain import make_run_split
from percentage_selection import (
    ExactAuditCache,
    ObservationBankPrefixCache,
    optimize_percentage_designs,
)
from robust_selection import (
    PhysicalView,
    RobustPhysicalViewExperiment,
    leave_one_fold_out_views,
)
from run import (
    CONFIG_PATH,
    _materialize_reference_dir,
    charge_diagnostics,
    ensure_reference,
    make_experiment,
    observation_bank,
    reference_seeds,
    split_config,
    summarize_rows,
)
from unbalanced_experiment import UnbalancedObservationBank
from unbalanced_state import TwoSpeciesDefectBank


METHODOLOGY_VERSION = 1
DEFAULT_PERCENTAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=SCRIPT_DIR / "outputs" / "pareto_robust",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--percent", nargs="+", type=float)
    parser.add_argument("--reference-seeds", nargs="+", type=int)
    parser.add_argument(
        "--multistart-backend",
        choices=("serial", "threaded"),
        default="serial",
        help="execution-only optimizer backend; does not alter the scientific config",
    )
    parser.add_argument(
        "--multistart-workers",
        type=int,
        default=2,
        help="concurrent original single-start executables for the threaded backend",
    )
    parser.add_argument(
        "--reuse-exact-evaluations",
        action="store_true",
        help=(
            "reuse byte-identical exact audits/validation designs and remember "
            "authoritative scalar fallback decisions"
        ),
    )
    parser.add_argument(
        "--reuse-prefix-banks",
        action="store_true",
        help="retain exact observation-bank prefix objects across allowances",
    )
    parser.add_argument(
        "--reuse-scalar-geometry",
        action="store_true",
        help="reuse immutable feature tensors across authoritative scalar trials",
    )
    parser.add_argument(
        "--skip-unused-tangent-for-full",
        action="store_true",
        help="omit Tangent-action work when the requested metric is Full only",
    )
    return parser.parse_args()


def _tag(percent: float) -> str:
    return f"risk_{f'{percent:g}'.replace('.', 'p')}pct"


def _copy_or_link(source: Path, target: Path) -> None:
    if target.exists():
        # Initial freezing normally creates a hard link. Inode identity is both
        # stronger and cheaper than hashing the same large file twice on resume.
        # Copied artifacts retain the SHA-256 safeguard below.
        try:
            if os.path.samefile(source, target):
                return
        except OSError:
            pass
        if file_sha256(source) != file_sha256(target):
            raise RuntimeError(
                f"refusing to replace frozen input {target} with different source {source}; "
                "use a new output directory"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def save_observation_bank(path: Path, bank: UnbalancedObservationBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        plus_sample_indices=np.asarray(bank.plus_sample_indices),
        minus_sample_indices=np.asarray(bank.minus_sample_indices),
        plus_detector_z=np.asarray(bank.plus_detector_z),
        minus_detector_z=np.asarray(bank.minus_detector_z),
    )


def load_observation_bank(path: Path) -> UnbalancedObservationBank:
    with np.load(path, allow_pickle=False) as data:
        return UnbalancedObservationBank(
            *(jnp.asarray(data[name]) for name in UnbalancedObservationBank._fields)
        )


def _view_indices(cfg: dict[str, Any], split, phase: str) -> list[tuple[int, ...]]:
    robust = cfg["robust_selection"]
    runs = split.design if phase == "selection" else split.validation
    count_key = "design_views" if phase == "selection" else "validation_views"
    if str(robust.get("view_method", "leave_one_fold_out")) != "leave_one_fold_out":
        raise ValueError("only leave_one_fold_out robust views are implemented")
    seed = (
        int(cfg["seed"])
        + int(robust.get("seed_offset", 1201))
        + (0 if phase == "selection" else 1)
    )
    return leave_one_fold_out_views(runs, count=int(robust[count_key]), seed=seed)


def _build_views(
    cfg: dict[str, Any],
    bank: TwoSpeciesDefectBank,
    references: dict[int, dict[str, Any]],
    run_views: list[tuple[int, ...]],
    *,
    phase: str,
) -> list[PhysicalView]:
    truth_base = int(cfg["seed"]) + (4100 if phase == "selection" else 4200)
    views = []
    for view_index, runs in enumerate(run_views):
        # The physical resample is common across reference seeds for a view.
        truth_seed = truth_base + view_index
        for seed, reference in sorted(references.items()):
            experiment, _ = make_experiment(
                cfg, bank, np.asarray(runs, dtype=np.int64), truth_seed, reference
            )
            views.append(
                PhysicalView(
                    label=f"{phase}_fold_{view_index}_reference_{seed}",
                    reference_seed=int(seed),
                    run_indices=tuple(int(value) for value in runs),
                    experiment=experiment,
                )
            )
    return views


def _selected_row(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    stage = "law" if design == "law" else ("tangent" if design == "tangent" else "full")
    rows = result["selection_candidates"][stage]
    matches = [
        row for row in rows
        if np.allclose(np.asarray(row["eta"]), eta, rtol=0.0, atol=1.0e-10)
    ]
    if not matches:
        raise RuntimeError(f"selected {design} geometry has no exact audit")
    return min(matches, key=lambda row: float(row["audit"]["value"]))


def _full_action_row(result: dict[str, Any], design: str) -> dict[str, Any]:
    eta = np.asarray(result["designs"][design], dtype=np.float64)
    matches = [
        row for row in result["selection_candidates"]["full"]
        if np.allclose(np.asarray(row["eta"]), eta, rtol=0.0, atol=1.0e-10)
    ]
    if not matches:
        raise RuntimeError(f"selected {design} geometry has no complete-bank Full audit")
    return min(matches, key=lambda row: float(row["audit"]["value"]))


def _hydrate_selection_audit_cache(
    cache: ExactAuditCache,
    exp: RobustPhysicalViewExperiment,
    bank: UnbalancedObservationBank,
    result: dict[str, Any],
    *,
    prefix_cache: ObservationBankPrefixCache | None = None,
    full_prescreen_trials: int | None = None,
) -> None:
    """Reuse complete-bank audits already preserved in a certified point."""
    for stage, rows in result["selection_candidates"].items():
        audit_name = {
            "law": "law_risk",
            "tangent": "tangent_action",
            "full": "full_action",
        }[stage]
        for row in rows:
            cache.store(exp, row["eta"], bank, audit_name, row["audit"])
            cache.store(exp, row["eta"], bank, "law_risk", row["law_screen"])
            if (
                stage == "full"
                and prefix_cache is not None
                and full_prescreen_trials is not None
                and row["audit"].get("prescreen_value") is not None
            ):
                prefix = prefix_cache.prefix(bank, full_prescreen_trials)
                cache.store(
                    exp,
                    row["eta"],
                    prefix,
                    "full_action",
                    {
                        "value": float(row["audit"]["prescreen_value"]),
                        "valid": True,
                    },
                )


def _physical_view_summary(view_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([
        row["summary"]["metrics"]["full_unbalanced_action_total"]["mean"]
        for row in view_rows
    ], dtype=np.float64)
    by_physical_fold: dict[tuple[int, ...], list[float]] = {}
    for row, value in zip(view_rows, values, strict=True):
        by_physical_fold.setdefault(tuple(row["run_indices"]), []).append(float(value))
    fold_values = np.asarray(
        [np.mean(group) for group in by_physical_fold.values()], dtype=np.float64
    )
    fold_mean = float(np.mean(fold_values))
    jackknife_se = (
        float(np.sqrt((len(fold_values) - 1) / len(fold_values) * np.sum((fold_values - fold_mean) ** 2)))
        if len(fold_values) > 1 else 0.0
    )
    return {
        "mean": float(np.mean(values)),
        "se_across_views": jackknife_se,
        "se_method": "leave-one-physical-fold-out jackknife after averaging reference seeds",
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "views": int(len(values)),
        "physical_folds": int(len(fold_values)),
    }


def _evaluate_validation(
    views: list[PhysicalView],
    bank: UnbalancedObservationBank,
    designs: dict[str, Any],
    *,
    cache: dict[tuple[tuple[int, ...], bytes], dict[str, Any]] | None = None,
    cache_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    output = {}
    for design, eta in designs.items():
        canonical = np.ascontiguousarray(
            np.asarray(
                views[0].experiment.sensors.canonicalize(
                    jnp.asarray(eta, dtype=jnp.float64)
                ),
                dtype=np.float64,
            )
        )
        cache_key = (canonical.shape, canonical.tobytes())
        if cache is not None and cache_key in cache:
            if cache_stats is not None:
                cache_stats["hits"] = cache_stats.get("hits", 0) + 1
            output[design] = copy.deepcopy(cache[cache_key])
            continue
        if cache_stats is not None:
            cache_stats["misses"] = cache_stats.get("misses", 0) + 1
        view_rows = []
        all_rows = []
        for view in views:
            rows = view.experiment.certified_trial_rows(jnp.asarray(eta), bank)
            annotated = [
                {
                    **row,
                    "physical_view": view.label,
                    "reference_seed": int(view.reference_seed),
                    "run_indices": list(view.run_indices),
                }
                for row in rows
            ]
            all_rows.extend(rows)
            view_rows.append({
                "label": view.label,
                "reference_seed": int(view.reference_seed),
                "run_indices": list(view.run_indices),
                "summary": summarize_rows(rows),
                "rows": annotated,
            })
        evaluated = {
            "summary": summarize_rows(all_rows),
            "physical_view_action": _physical_view_summary(view_rows),
            "views": view_rows,
        }
        output[design] = evaluated
        if cache is not None:
            cache[cache_key] = copy.deepcopy(evaluated)
    return output


def _row(result: dict[str, Any], percent: float, path: Path) -> dict[str, Any]:
    law = _full_action_row(result, "law")
    full = _full_action_row(result, "unbalanced_full")
    validation_law = result["validation_designs"]["law"]["physical_view_action"]
    validation_full = result["validation_designs"]["unbalanced_full"]["physical_view_action"]
    return {
        "allowance_percent": float(percent),
        "risk_star": float(result["risk_star"]),
        "risk_max": float(result["risk_max"]),
        "selection_law_action": float(law["audit"]["value"]),
        "selection_full_action": float(full["audit"]["value"]),
        "selection_full_risk": float(full["law_screen"]["value"]),
        "validation_law_action": float(validation_law["mean"]),
        "validation_law_action_se_across_views": float(validation_law["se_across_views"]),
        "validation_full_action": float(validation_full["mean"]),
        "validation_full_action_se_across_views": float(validation_full["se_across_views"]),
        "validation_full_vs_law_reduction": (
            1.0 - float(validation_full["mean"]) / float(validation_law["mean"])
        ),
        "full_eta": result["designs"]["unbalanced_full"],
        "certified": bool(result["selection_certified"]),
        "result": str(path.resolve()),
    }


def _save_table(output: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output / "pareto.json", {"schema_version": 1, "rows": rows})
    if rows:
        with (output / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                })


def main() -> None:
    args = parse_args()
    if args.multistart_workers < 1:
        raise ValueError("--multistart-workers must be >= 1")
    cfg = load_config(args.config.expanduser().resolve(), smoke=args.smoke)
    if args.reference_seeds is not None:
        cfg["reference_training"]["seeds"] = [int(value) for value in args.reference_seeds]
    percentages = sorted(set(
        args.percent
        if args.percent is not None
        else cfg.get("pareto", {}).get("allowances_percent", DEFAULT_PERCENTAGES)
    ))
    if not percentages or any(not math.isfinite(value) or value < 0.0 for value in percentages):
        raise ValueError("allowances must be finite and nonnegative")

    source = args.input_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_allowances = None
    previous_manifest_path = output / "manifest.json"
    if previous_manifest_path.is_file():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        previous_allowances = sorted(
            float(row["allowance_percent"]) for row in previous_manifest.get("points", [])
        )
    validated_existing = []
    for path in output.glob("risk_*pct/result.json"):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if saved.get("validation_designs") is not None:
            validated_existing.append(path)
    if (
        validated_existing
        and not args.force
        and previous_allowances != [float(value) for value in percentages]
    ):
        raise RuntimeError(
            "this directory already contains post-selection validation for a different "
            "allowance set; use a new output directory or --force so validation cannot "
            "precede selection of newly added points"
        )
    frozen = output / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)
    effective_config_path = frozen / "effective_config.json"
    if effective_config_path.is_file():
        frozen_config = json.loads(effective_config_path.read_text(encoding="utf-8"))
        if frozen_config != cfg:
            raise RuntimeError(
                "effective configuration differs from the frozen Pareto configuration; "
                "use a new output directory"
            )
    defect_source = source / "two_species_defect_bank.npz"
    if not defect_source.is_file():
        raise FileNotFoundError(
            f"missing {defect_source}; run physical-bank and defects stages first"
        )
    _copy_or_link(defect_source, frozen / defect_source.name)
    for optional in ("physical_bank.npz", "defect_bank_audit.json", "charge_balance.json"):
        if (source / optional).is_file():
            _copy_or_link(source / optional, frozen / optional)
    _copy_or_link(
        args.config.expanduser().resolve(), frozen / "source_config.json"
    )
    if not effective_config_path.is_file():
        write_json(effective_config_path, cfg)

    bank = TwoSpeciesDefectBank.load(frozen / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    charge_diagnostics(cfg, bank, split.train)
    charge_diagnostics(cfg, bank, split.design)
    charge_diagnostics(cfg, bank, split.validation)
    references = {}
    for seed in reference_seeds(cfg):
        source_seed_dir = source / f"reference_seed_{seed}"
        frozen_seed_dir = frozen / f"reference_seed_{seed}"
        for species in ("plus", "minus"):
            for suffix in ("reference.npz", "reference_bank.npz"):
                source_reference = source_seed_dir / f"{species}_{suffix}"
                frozen_reference = frozen_seed_dir / f"{species}_{suffix}"
                if source_reference.is_file() and frozen_reference.is_file():
                    _copy_or_link(source_reference, frozen_reference)
        reference_dir = _materialize_reference_dir(frozen, source, seed)
        references[seed] = ensure_reference(cfg, bank, split.train, seed, reference_dir)

    selection_indices = _view_indices(cfg, split, "selection")
    validation_indices = _view_indices(cfg, split, "validation")
    selection_views = _build_views(
        cfg, bank, references, selection_indices, phase="selection"
    )
    validation_views = _build_views(
        cfg, bank, references, validation_indices, phase="validation"
    )
    base_robust = RobustPhysicalViewExperiment(cfg, selection_views)
    if args.skip_unused_tangent_for_full:
        for view in selection_views:
            view.experiment.skip_unused_tangent_for_full_metric = True
    if args.reuse_exact_evaluations:
        for view in validation_views:
            view.experiment.remember_certification_scalar_fallback = True
    if args.reuse_scalar_geometry:
        for view in validation_views:
            view.experiment.reuse_exact_trial_geometry = True
    selection_path = frozen / "selection_bank.npz"
    validation_path = frozen / "validation_bank.npz"
    if selection_path.is_file():
        selection_bank = load_observation_bank(selection_path)
    else:
        selection_bank = observation_bank(
            cfg, selection_views[0].experiment,
            cfg["randomness"]["selection_namespace"],
            cfg["randomness"]["selection_trials"],
        )
        save_observation_bank(selection_path, selection_bank)
    if validation_path.is_file():
        validation_bank = load_observation_bank(validation_path)
    else:
        validation_bank = observation_bank(
            cfg, validation_views[0].experiment,
            cfg["randomness"]["validation_namespace"],
            cfg["randomness"]["validation_trials"],
        )
        save_observation_bank(validation_path, validation_bank)
    write_json(frozen / "view_manifest.json", {
        "schema_version": 1,
        "methodology_version": METHODOLOGY_VERSION,
        "selection_run_views": [list(row) for row in selection_indices],
        "validation_run_views": [list(row) for row in validation_indices],
        "selection_views": base_robust.view_manifest(),
        "validation_views": [
            {
                "label": view.label,
                "reference_seed": view.reference_seed,
                "run_indices": list(view.run_indices),
            }
            for view in validation_views
        ],
        "train_runs": split.train.tolist(),
        "design_runs": split.design.tolist(),
        "validation_runs": split.validation.tolist(),
    })

    point_results: list[tuple[float, Path, dict[str, Any]]] = []
    law_anchor = None
    incumbent_eta = None
    incumbent_action = math.inf
    selection_audit_cache = ExactAuditCache(
        enabled=args.reuse_exact_evaluations
    )
    prefix_cache = ObservationBankPrefixCache(
        enabled=args.reuse_prefix_banks
    )
    tolerance = float(cfg.get("pareto", {}).get("nesting_tolerance", 1.0e-8))
    for percent in percentages:
        point_dir = output / _tag(percent)
        point_dir.mkdir(parents=True, exist_ok=True)
        result_path = point_dir / "result.json"
        point_cfg = json.loads(json.dumps(cfg))
        point_cfg["law"]["max_relative_risk_violation"] = float(percent) / 100.0
        point_cfg["optimization"]["pareto_methodology_version"] = METHODOLOGY_VERSION
        if law_anchor is not None:
            point_cfg["optimization"]["fixed_law_anchor"] = law_anchor
        if incumbent_eta is not None:
            point_cfg["optimization"]["pareto_incumbent_full_eta"] = incumbent_eta
        point_hash = fingerprint(point_cfg)
        cached = None
        if result_path.is_file() and not args.force:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if candidate.get("config_hash") == point_hash:
                cached = candidate
        if cached is None:
            print(f"active robust Pareto allowance={percent:g}%", flush=True)
            robust = base_robust.with_config(point_cfg)
            cache_before = selection_audit_cache.stats()
            prefix_before = prefix_cache.stats()
            selected = optimize_percentage_designs(
                robust,
                selection_bank,
                multistart_backend=args.multistart_backend,
                multistart_workers=args.multistart_workers,
                audit_cache=selection_audit_cache,
                prefix_cache=prefix_cache,
            )
            cache_after = selection_audit_cache.stats()
            prefix_after = prefix_cache.stats()
            result = {
                "schema_version": 2,
                "experiment": "active_nematic_unbalance_robust_pareto",
                "methodology_version": METHODOLOGY_VERSION,
                "config_hash": point_hash,
                "config": point_cfg,
                "allowance_percent": float(percent),
                "risk_star": float(selected["risk_star"]),
                "risk_max": float(selected["risk_max"]),
                "risk_view_star": selected["risk_view_star"],
                "risk_view_maxima": selected["risk_view_maxima"],
                "selection_certified": bool(selected["certified"]),
                "selection_execution": {
                    "multistart_backend": args.multistart_backend,
                    "multistart_workers": (
                        int(args.multistart_workers)
                        if args.multistart_backend == "threaded"
                        else None
                    ),
                    "reuse_exact_evaluations": bool(
                        args.reuse_exact_evaluations
                    ),
                    "exact_audit_cache_delta": {
                        key: int(cache_after[key] - cache_before[key])
                        for key in ("hits", "misses", "entries")
                    },
                    "reuse_prefix_banks": bool(args.reuse_prefix_banks),
                    "skip_unused_tangent_for_full": bool(
                        args.skip_unused_tangent_for_full
                    ),
                    "prefix_cache_delta": {
                        key: int(prefix_after[key] - prefix_before[key])
                        for key in ("hits", "misses", "entries")
                    },
                },
                "designs": {
                    "law": np.asarray(selected["law_eta"]).tolist(),
                    "tangent": np.asarray(selected["tangent_eta"]).tolist(),
                    "unbalanced_full": np.asarray(selected["full_eta"]).tolist(),
                },
                "selection_candidates": selected["candidates"],
                "selection_views": robust.view_manifest(),
                "validation_designs": None,
            }
        else:
            result = cached
            if args.reuse_exact_evaluations:
                _hydrate_selection_audit_cache(
                    selection_audit_cache,
                    base_robust,
                    selection_bank,
                    result,
                    prefix_cache=(
                        prefix_cache if args.reuse_prefix_banks else None
                    ),
                    full_prescreen_trials=int(
                        cfg["optimization"].get(
                            "full_prescreen_trials",
                            min(
                                4,
                                int(selection_bank.plus_sample_indices.shape[0]),
                            ),
                        )
                    ),
                )
        if not result["selection_certified"]:
            raise RuntimeError(f"uncertified selection at {percent:g}%")
        law_row = _selected_row(result, "law")
        full_row = _selected_row(result, "unbalanced_full")
        current_anchor = {
            "eta": result["designs"]["law"],
            "risk_star": float(law_row["audit"]["value"]),
            "risk_view_star": law_row["audit"].get("view_values"),
        }
        if law_anchor is None:
            law_anchor = current_anchor
        elif not np.allclose(law_anchor["eta"], current_anchor["eta"], rtol=0.0, atol=1.0e-10):
            raise RuntimeError("Pareto points do not share one Law geometry")
        action = float(full_row["audit"]["value"])
        if action > incumbent_action + tolerance:
            raise RuntimeError(
                f"non-nested robust action at {percent:g}%: {action} > {incumbent_action}"
            )
        if action < incumbent_action:
            incumbent_action = action
            incumbent_eta = result["designs"]["unbalanced_full"]
        write_json(result_path, jsonable(result))
        point_results.append((percent, result_path, result))

    # Validation begins only after every selection winner has been frozen.
    table_rows = []
    validation_cache = {} if args.reuse_exact_evaluations else None
    validation_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
    for percent, result_path, result in point_results:
        if result.get("validation_designs") is None or args.force:
            print(f"active robust validation allowance={percent:g}%", flush=True)
            result["validation_designs"] = _evaluate_validation(
                validation_views,
                validation_bank,
                result["designs"],
                cache=validation_cache,
                cache_stats=validation_cache_stats,
            )
            result["validation_execution"] = {
                "reuse_exact_evaluations": bool(
                    args.reuse_exact_evaluations
                ),
                "cache_stats_cumulative": dict(validation_cache_stats),
                "reuse_scalar_geometry": bool(args.reuse_scalar_geometry),
            }
            write_json(result_path, jsonable(result))
        table_rows.append(_row(result, percent, result_path))
    _save_table(output, table_rows)
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "methodology_version": METHODOLOGY_VERSION,
        "selection_completed_before_validation": True,
        "points": [
            {"allowance_percent": percent, "result": str(path.resolve())}
            for percent, path, _ in point_results
        ],
    })
    print(f"active robust Pareto complete: {output / 'pareto.json'}", flush=True)


if __name__ == "__main__":
    main()
