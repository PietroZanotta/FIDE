"""Run the vortices percentage-risk Pareto sweep with common random numbers."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import jax

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
jax.config.update("jax_enable_x64", True)

from mfsi.cache import fingerprint
from mfsi.config import load_config
from experiment import run_experiment

DEFAULT_PERCENTAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
PARETO_METHODOLOGY_VERSION = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the maximum percentage of extra finite-law risk."
    )
    parser.add_argument(
        "--percent", nargs="+", type=float, default=list(DEFAULT_PERCENTAGES),
        metavar="PCT", help="allowed extra risk in percent (default: 0.5 1 2 3 4 5)",
    )
    parser.add_argument(
        "--source-run", type=Path, default=SCRIPT_DIR / "outputs" / "run",
        help="compatible run used to seed frozen reference/CRN artifacts",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "pareto")
    parser.add_argument("--force", action="store_true", help="rerun compatible points")
    return parser.parse_args()


def _tag(percent: float) -> str:
    value = f"{percent:g}".replace(".", "p").replace("-", "m")
    return f"risk_{value}pct"


def _link(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _seed(source: Path, target: Path) -> None:
    for name in (
        "truth_bank.npz", "reference_endpoints.npz", "reference.npz",
        "reference_bank.npz", "selection_bank.npz", "validation_bank.npz",
    ):
        _link(source / name, target / name)


def _best_archived_law_seed(source: Path, output: Path) -> list[float] | None:
    """Use archived runs only as candidate generators; every seed is re-audited."""
    paths = [source / "result.json", *sorted(output.glob("*/result.json"))]
    candidates: list[tuple[float, list[float], Path]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            l_max = float(result["law_screens"]["L_max"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        for stage in ("law", "tangent", "full"):
            for row in result.get("selection_audit", {}).get(stage, []):
                try:
                    exact_l = float(row["exact_L"])
                    exact_r = float(row["exact_R"])
                    eta = [float(value) for value in row["eta"]]
                except (KeyError, TypeError, ValueError):
                    continue
                if row.get("valid") and exact_l <= l_max + 1.0e-12 and math.isfinite(exact_r):
                    candidates.append((exact_r, eta, path))
    if not candidates:
        print("[pareto] no archived exact-valid Law seed found; using configured starts", flush=True)
        return None
    risk, eta, path = min(candidates, key=lambda item: item[0])
    print(f"[pareto] archived Law seed: R={risk:.8g} from {path}", flush=True)
    return eta


def _audited_full_action(result: dict[str, Any], selected: list[float]) -> float:
    for row in result.get("selection_audit", {}).get("full", []):
        if row.get("eta") == selected and row.get("valid"):
            value = float(row["objective"])
            if math.isfinite(value):
                return value
    raise RuntimeError("design is missing its exact Full-action audit row")


def _candidate_summary_action(result_path: Path, design: str, column: str) -> float:
    """Read an authoritative post-selection score without stage-crossing assumptions."""
    path = result_path.with_name("result.candidate_summary.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("design") == design:
                value = float(row[column])
                if math.isfinite(value):
                    return value
    raise RuntimeError(f"{design} is missing {column} in {path}")


def _row(result: dict[str, Any], percent: float, result_path: Path) -> dict[str, Any]:
    screens = result["law_screens"]
    full = result["selection_certificates"]["full"]
    tangent = result["selection_certificates"]["tangent"]
    validation = result["validation"]
    law_validation = validation["law"]
    tangent_validation = validation["tangent"]
    full_validation = validation["full"]
    contrast = result.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    return {
        "risk_allowance_percent": percent,
        "risk_allowance_fraction": percent / 100.0,
        "epsilon_r": screens["epsilon_r"],
        "R_star": screens["R_star"],
        "R_max": screens["R_max"],
        "full_R_selection": full["R_selection"],
        "full_R_excess_selection": full["R_excess_from_star"],
        "full_R_slack_selection": full.get("R_slack_to_max"),
        "full_A_selection": _audited_full_action(result, result["selection"]["full_optimum"]),
        "law_A_selection": _audited_full_action(result, result["selection"]["law_optimum"]),
        "tangent_R_selection": tangent["R_selection"],
        "tangent_R_excess_selection": tangent["R_excess_from_star"],
        "tangent_L_selection": tangent["L_selection"],
        "tangent_A_selection": _candidate_summary_action(
            result_path, "tangent", "full_action_selection"
        ),
        "tangent_certified": tangent["certified"],
        "full_certified": full["certified"],
        "anchor_refinement_passes": result.get("law_anchor", {}).get("anchor_refinement_passes", 0),
        "law_R_validation": law_validation["law_risk"]["mean"],
        "law_R_validation_se": law_validation["law_risk"]["se"],
        "law_A_validation": law_validation["full_action"]["mean"],
        "law_A_validation_se": law_validation["full_action"]["se"],
        "law_valid_fraction": law_validation["valid_fraction"],
        "tangent_R_validation": tangent_validation["law_risk"]["mean"],
        "tangent_R_validation_se": tangent_validation["law_risk"]["se"],
        "tangent_A_validation": tangent_validation["full_action"]["mean"],
        "tangent_A_validation_se": tangent_validation["full_action"]["se"],
        "tangent_valid_fraction": tangent_validation["valid_fraction"],
        "full_R_validation": full_validation["law_risk"]["mean"],
        "full_R_validation_se": full_validation["law_risk"]["se"],
        "full_A_validation": full_validation["full_action"]["mean"],
        "full_A_validation_se": full_validation["full_action"]["se"],
        "full_valid_fraction": full_validation["valid_fraction"],
        "validation_action_reduction": contrast.get("ratio_of_means_reduction"),
        "law_centers": json.dumps(result["selection_centers"]["law"]),
        "tangent_centers": json.dumps(result["selection_centers"]["tangent"]),
        "full_centers": json.dumps(result["selection_centers"]["full"]),
        "result": str(result_path),
    }


def _save(rows: list[dict[str, Any]], output: Path) -> None:
    with (output / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "pareto.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    try:
        from visualize_pareto import save_pareto_figure

        save_pareto_figure(rows, output / "pareto.png")
    except Exception as exc:
        print(f"[pareto] plot skipped: {exc}", flush=True)


def _fmt(value: Any, spec: str = ".8g") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return format(number, spec) if math.isfinite(number) else "n/a"


def _print_point_summary(index: int, total: int, row: dict[str, Any]) -> None:
    reduction = row.get("validation_action_reduction")
    reduction_text = _fmt(100.0 * float(reduction), ".3f") + "%" if reduction is not None else "n/a"
    print("-" * 88, flush=True)
    print(f"[pareto {index}/{total}] completed allowance={row['risk_allowance_percent']:g}%", flush=True)
    print(
        "  risk screen: "
        f"R*={_fmt(row['R_star'])}  epsilon_R={_fmt(row['epsilon_r'])}  R_max={_fmt(row['R_max'])}",
        flush=True,
    )
    print(
        "  selected Full: "
        f"delta_R={_fmt(row['full_R_excess_selection'])}  slack={_fmt(row['full_R_slack_selection'])}  "
        f"A={_fmt(row['full_A_selection'])}  certified={bool(row['full_certified'])}",
        flush=True,
    )
    print(
        "  validation: "
        f"R_law={_fmt(row['law_R_validation'])}±{_fmt(row['law_R_validation_se'], '.2g')}  "
        f"R_full={_fmt(row['full_R_validation'])}±{_fmt(row['full_R_validation_se'], '.2g')}  "
        f"A_law={_fmt(row['law_A_validation'])}±{_fmt(row['law_A_validation_se'], '.2g')}  "
        f"A_full={_fmt(row['full_A_validation'])}±{_fmt(row['full_A_validation_se'], '.2g')}  "
        f"A_reduction={reduction_text}",
        flush=True,
    )
    print(
        f"  validation validity: Law={100.0 * float(row['law_valid_fraction']):.1f}%  "
        f"Full={100.0 * float(row['full_valid_fraction']):.1f}%",
        flush=True,
    )
    print(f"  Full centers: {row['full_centers']}", flush=True)
    print(f"  result: {row['result']}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(SCRIPT_DIR / "config.json", smoke=False)
    percentages = sorted(set(float(value) for value in args.percent))
    if not percentages or any(not math.isfinite(value) or value < 0 for value in percentages):
        raise ValueError("risk percentages must be finite and nonnegative")

    print("=" * 88, flush=True)
    print("VORTICES / DOUBLE-GYRE — PERCENTAGE-RISK PARETO SWEEP", flush=True)
    print("=" * 88, flush=True)
    print(f"allowances_percent={percentages}", flush=True)
    print(f"source_run={args.source_run.resolve()}", flush=True)
    print(f"output={args.output.resolve()}", flush=True)
    print(f"force={args.force}", flush=True)
    print("policy: R_max = R* + (allowance_percent / 100) * abs(R*)", flush=True)

    archive_seed = _best_archived_law_seed(args.source_run, args.output)
    shared_anchor: dict[str, Any] | None = None
    incumbent_full_eta: list[float] | None = None
    incumbent_full_action = math.inf
    rows: list[dict[str, Any]] = []
    total = len(percentages)

    for index, percent in enumerate(percentages, start=1):
        point = args.output / _tag(percent)
        point.mkdir(parents=True, exist_ok=True)
        result_path = point / "result.json"
        cfg = json.loads(json.dumps(base_cfg))
        cfg["law"].pop("epsilon_r", None)
        cfg["law"]["max_relative_risk_violation"] = percent / 100.0
        cfg["optimization"]["pareto_methodology_version"] = PARETO_METHODOLOGY_VERSION
        if shared_anchor is not None:
            cfg["optimization"]["fixed_law_anchor"] = shared_anchor
        elif archive_seed is not None:
            cfg["optimization"]["law_anchor_seed_eta"] = archive_seed
        if incumbent_full_eta is not None:
            cfg["optimization"]["pareto_incumbent_full_eta"] = incumbent_full_eta

        print("=" * 88, flush=True)
        print(f"[pareto {index}/{total}] allowance={percent:g}% (fraction={percent / 100.0:g})", flush=True)
        print(f"[pareto {index}/{total}] point_dir={point.resolve()}", flush=True)
        print(
            f"[pareto {index}/{total}] frozen_anchor={'yes' if shared_anchor else 'pending'}  "
            f"incumbent_full={'yes' if incumbent_full_eta else 'none'}",
            flush=True,
        )

        cached = None
        if result_path.exists() and not args.force:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if candidate.get("config_hash") == fingerprint(cfg):
                cached = candidate
                print(f"[pareto {index}/{total}] compatible result found; reusing it", flush=True)
            else:
                print(f"[pareto {index}/{total}] stale result found; rerunning it", flush=True)
        if cached is None:
            _seed(args.source_run, point)
            print(f"[pareto {index}/{total}] seeded frozen truth/reference/CRN banks", flush=True)
            print(f"[pareto {index}/{total}] launching full scientific pipeline", flush=True)
            result = run_experiment(cfg, point, smoke=False)
        else:
            result = cached

        current_anchor = {
            "eta": result["selection"]["law_optimum"],
            "R_star": float(result["law_screens"]["R_star"]),
        }
        if shared_anchor is None:
            shared_anchor = current_anchor
            print(f"[pareto] froze common exact Law anchor R*={shared_anchor['R_star']:.12g}", flush=True)
        elif abs(current_anchor["R_star"] - shared_anchor["R_star"]) > float(
            cfg["optimization"].get("law_anchor_consistency_tol", 1.0e-10)
        ):
            raise RuntimeError(
                "Pareto points do not share one Law anchor: "
                f"expected {shared_anchor['R_star']:.12g}, got {current_anchor['R_star']:.12g}"
            )

        full = result["selection_certificates"]["full"]
        if float(full["R_excess_from_star"]) < -float(
            cfg["optimization"].get("law_anchor_consistency_tol", 1.0e-10)
        ):
            raise RuntimeError(
                f"invalid point at {percent:g}%: Full strictly beats the frozen Law anchor"
            )
        row = _row(result, percent, result_path)
        action = float(row["full_A_selection"])
        action_tol = 1.0e-10 * max(1.0, abs(incumbent_full_action) if math.isfinite(incumbent_full_action) else 1.0)
        if math.isfinite(incumbent_full_action) and action > incumbent_full_action + action_tol:
            raise RuntimeError(
                f"non-nested Pareto action at {percent:g}%: exact action increased "
                f"from {incumbent_full_action:.12g} to {action:.12g}"
            )
        if action < incumbent_full_action:
            incumbent_full_action = action
            incumbent_full_eta = result["selection"]["full_optimum"]

        rows.append(row)
        _save(rows, args.output)
        _print_point_summary(index, total, row)
        print(f"[pareto {index}/{total}] checkpointed table and refreshed figure", flush=True)

    try:
        from visualize_pareto import save_pareto_suite

        save_pareto_suite(rows, args.output, args.output)
    except Exception as exc:
        print(f"[pareto] extended post-processing skipped: {exc}", flush=True)

    print("=" * 88, flush=True)
    print(f"[pareto] complete: {total}/{total} percentage allowances", flush=True)
    print(f"[pareto] table: {args.output / 'pareto.csv'}", flush=True)
    print(f"[pareto] data:  {args.output / 'pareto.json'}", flush=True)
    print(f"[pareto] plot:  {args.output / 'pareto.png'}", flush=True)
    print(f"[pareto] methods: {args.output / 'pareto_methods.png'}", flush=True)
    print(f"[pareto] sensors: {args.output / 'pareto_sensor_layouts.png'}", flush=True)


if __name__ == "__main__":
    main()
