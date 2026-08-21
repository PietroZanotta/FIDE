"""Run the toy percentage-risk Pareto sweep with common random numbers."""
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mfsi.cache import fingerprint
from mfsi.config import load_config
from experiment import run_experiment

DEFAULT_PERCENTAGES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


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


def _run_config_hash(cfg: dict[str, Any]) -> str:
    """Mirror run_experiment's pre-hash default normalization."""
    normalized = json.loads(json.dumps(cfg))
    validity = normalized.setdefault("validity", {})
    validity.setdefault("max_population_calibration_resid", 1.0e-5)
    validity.setdefault("max_finite_calibration_resid", 1.0e-3)
    validity.setdefault("min_ess_fraction", 0.03)
    validity.setdefault("min_in_domain_base_mass", 0.995)
    normalized.setdefault("optimization", {})
    return fingerprint(normalized)


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _seed_artifacts(source: Path, target: Path, *, include_stage12: bool) -> None:
    for name in ("reference.npz", "reference_bank.npz", "selection_bank.npz", "validation_bank.npz"):
        _link_or_copy(source / name, target / name)
    if include_stage12:
        for name in ("population_selection.json", "finite_law_selection.json"):
            _link_or_copy(source / "cache" / name, target / "cache" / name)


def _row(result: dict[str, Any], percent: float, point_dir: Path) -> dict[str, Any]:
    screens = result["law_screens"]
    certificates = result.get("selection_certificates", {})
    full = certificates.get("full", {})
    law = certificates.get("law", {})
    tangent = certificates.get("tangent", {})
    validation = result.get("validation", {})
    contrast = result.get("contrasts", {}).get("full_vs_law_full_action_reduction", {})
    bootstrap = result.get("contrasts", {}).get("full_vs_law_ratio_of_means_bootstrap_95", {})
    agreement = result.get("full_proxy_agreement", {})
    funnel = result.get("full_search_funnel", {})
    return {
        "risk_allowance_percent": percent,
        "risk_allowance_fraction": percent / 100.0,
        "epsilon_r": screens["epsilon_r"],
        "R_star": screens["R_star"],
        "R_max": screens["R_max"],
        "full_theta1_deg": result["selection"]["full_optimum_deg"][0],
        "full_theta2_deg": result["selection"]["full_optimum_deg"][1],
        "law_theta1_deg": result["selection"]["law_optimum_deg"][0],
        "law_theta2_deg": result["selection"]["law_optimum_deg"][1],
        "tangent_theta1_deg": result["selection"]["tangent_optimum_deg"][0],
        "tangent_theta2_deg": result["selection"]["tangent_optimum_deg"][1],
        "full_R_selection": full.get("R_selection"),
        "full_R_excess_selection": full.get("R_excess_from_star"),
        "full_R_slack_selection": full.get("R_slack_to_max"),
        "full_L_selection": full.get("L_selection"),
        "full_certified": full.get("certified"),
        "law_L_selection": law.get("L_selection"),
        "law_R_selection": law.get("R_selection"),
        "law_A_selection": law.get("full_action_selection"),
        "tangent_R_selection": tangent.get("R_selection"),
        "tangent_R_excess_selection": tangent.get("R_excess_from_star"),
        "tangent_L_selection": tangent.get("L_selection"),
        "tangent_A_selection": tangent.get("full_action_selection"),
        "tangent_T_selection": tangent.get("tangent_action_selection"),
        "tangent_certified": tangent.get("certified"),
        "full_A_selection": full.get("full_action_selection"),
        "law_R_validation": validation.get("law", {}).get("law_risk", {}).get("mean"),
        "full_R_validation": validation.get("full", {}).get("law_risk", {}).get("mean"),
        "law_A_validation": validation.get("law", {}).get("full_action", {}).get("mean"),
        "tangent_R_validation": validation.get("tangent", {}).get("law_risk", {}).get("mean"),
        "tangent_A_validation": validation.get("tangent", {}).get("full_action", {}).get("mean"),
        "tangent_A_validation_se": validation.get("tangent", {}).get("full_action", {}).get("se"),
        "tangent_valid_fraction": validation.get("tangent", {}).get("valid_fraction"),
        "full_A_validation": validation.get("full", {}).get("full_action", {}).get("mean"),
        "validation_action_reduction": contrast.get("ratio_of_means_reduction"),
        "validation_ci_lower": bootstrap.get("lower"),
        "validation_ci_upper": bootstrap.get("upper"),
        "proxy_spearman": agreement.get("spearman_rank"),
        "proxy_same_best": agreement.get("same_best_candidate"),
        "exact_full_finalists": funnel.get("exact_full_finalists"),
        "result": str(point_dir / "result.json"),
    }


def _save(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "pareto.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (output / "pareto.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
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
        f"R_law={_fmt(row['law_R_validation'])}  R_full={_fmt(row['full_R_validation'])}  "
        f"A_law={_fmt(row['law_A_validation'])}  A_full={_fmt(row['full_A_validation'])}  "
        f"A_reduction={reduction_text}",
        flush=True,
    )
    print(
        "  stage-4 audit: "
        f"exact_finalists={row.get('exact_full_finalists', 'n/a')}  "
        f"proxy_spearman={_fmt(row.get('proxy_spearman'), '.4f')}  "
        f"same_best={row.get('proxy_same_best', 'n/a')}",
        flush=True,
    )
    print(f"  result: {row['result']}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(SCRIPT_DIR / "config.json", smoke=False)
    percentages = sorted(set(float(value) for value in args.percent))
    if not percentages or any(not math.isfinite(value) or value < 0 for value in percentages):
        raise ValueError("risk percentages must be finite and nonnegative")

    print("=" * 88, flush=True)
    print("TOY EXAMPLE — PERCENTAGE-RISK PARETO SWEEP", flush=True)
    print("=" * 88, flush=True)
    print(f"allowances_percent={percentages}", flush=True)
    print(f"source_run={args.source_run.resolve()}", flush=True)
    print(f"output={args.output.resolve()}", flush=True)
    print(f"force={args.force}", flush=True)
    print("policy: R_max = R* + (allowance_percent / 100) * abs(R*)", flush=True)

    rows: list[dict[str, Any]] = []
    stage12_source: Path | None = None
    common_r_star: float | None = None
    total = len(percentages)
    for index, percent in enumerate(percentages, start=1):
        point = args.output / _tag(percent)
        point.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base_cfg))
        cfg["law"].pop("epsilon_r", None)
        cfg["law"]["max_relative_risk_violation"] = percent / 100.0
        result_path = point / "result.json"

        print("=" * 88, flush=True)
        print(f"[pareto {index}/{total}] allowance={percent:g}% (fraction={percent / 100.0:g})", flush=True)
        print(f"[pareto {index}/{total}] point_dir={point.resolve()}", flush=True)

        cached = None
        if result_path.exists() and not args.force:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
            if candidate.get("config_hash") == _run_config_hash(cfg):
                cached = candidate
                print(f"[pareto {index}/{total}] compatible result found; reusing it", flush=True)
            else:
                print(f"[pareto {index}/{total}] stale result found; rerunning it", flush=True)

        if cached is None:
            _seed_artifacts(args.source_run, point, include_stage12=False)
            if stage12_source is not None:
                _seed_artifacts(stage12_source, point, include_stage12=True)
                print(
                    f"[pareto {index}/{total}] seeded percentage-independent Law-stage cache from {stage12_source}",
                    flush=True,
                )
            print(f"[pareto {index}/{total}] launching full scientific pipeline", flush=True)
            result = run_experiment(cfg, point, smoke=False)
            if stage12_source is None:
                stage12_source = point
        else:
            result = cached
            if stage12_source is None:
                stage12_source = point

        r_star = float(result["law_screens"]["R_star"])
        if common_r_star is None:
            common_r_star = r_star
            print(f"[pareto] common Law anchor R*={r_star:.12g}", flush=True)
        elif not math.isclose(r_star, common_r_star, rel_tol=0.0, abs_tol=1.0e-10):
            raise RuntimeError(
                "Pareto points do not share one Law anchor: "
                f"expected {common_r_star:.12g}, got {r_star:.12g}"
            )

        row = _row(result, percent, point)
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
