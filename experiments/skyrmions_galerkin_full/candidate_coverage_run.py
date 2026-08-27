"""CLI for the development-only skyrmion candidate-coverage study."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .candidate_coverage import (
    OUTPUT_ROOT,
    evaluate_audit,
    evaluate_screen,
    generate_candidate_pool,
    run,
    summarize,
)
from .galerkin_only import execution_device


def _print_summary(result: dict) -> None:
    print("                     ORIGINAL        NEW       COMBINED")
    print("allowance          dual-bank     dual-bank    dual-bank   best combined audit rESS")
    for row in result["allowances"]:
        best = row["best_combined_audit_minimum_ress"]
        rendered = "n/a" if best is None else f"{best:.9f}"
        print(
            f"{row['allowance_percent']:>4g}%"
            f"{row['original_v2_dual_bank_count']:>20d}"
            f"{row['new_dual_bank_count']:>14d}"
            f"{row['combined_unique_dual_bank_count']:>13d}"
            f"{rendered:>27}"
        )

    interpretation = result["development_interpretation"]
    print("candidate_078 neighborhood:")
    print(
        "  1% local candidates generated: "
        f"{interpretation['candidate_078_local_inside_1_percent']}"
    )
    print(
        "  1% dual-bank survivors: "
        f"{interpretation['candidate_078_local_1_percent_survivors']}"
    )
    local_rows = [
        row
        for row in result["local_basin_width"]
        if row["anchor_id"] == "candidate_078"
    ]
    robust_values = [
        row["robust_ress"]["maximum"]
        for row in local_rows
        if row["robust_ress"]["maximum"] is not None
    ]
    medians = [
        row["robust_ress"]["median"]
        for row in local_rows
        if row["robust_ress"]["median"] is not None
    ]
    print(f"  best robust-rESS by scale: {max(robust_values) if robust_values else None}")
    print(f"  median of scale medians: {None if not medians else sorted(medians)[len(medians)//2]}")

    for source in ("original", "new", "combined"):
        rows = result["spearman_correlations"][source]
        print(f"Spearman correlations ({source}):")
        print(f"  risk increase vs screen rESS: {rows['risk_vs_screen_ress']['rho']}")
        print(f"  risk increase vs audit rESS: {rows['risk_vs_audit_ress']['rho']}")
        print(f"  risk increase vs robust rESS: {rows['risk_vs_robust_ress']['rho']}")
    print(f"development interpretation: {interpretation['label']}")
    print(f"  {interpretation['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("generate", "screen", "audit", "summarize", "run"),
        default="run",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        if args.mode == "generate":
            result = generate_candidate_pool(cfg)
        elif args.mode == "screen":
            result = evaluate_screen(cfg)
        elif args.mode == "audit":
            result = evaluate_audit(cfg)
        elif args.mode == "summarize":
            result = summarize(cfg)
        else:
            result = run(cfg)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    if "allowances" in result:
        _print_summary(result)
    else:
        print(f"candidate_count={result.get('candidate_count', result.get('final_new_unique_count', result.get('audit_candidate_count')))}")
        print(f"cache_hit={result.get('cache_hit', False)}")


if __name__ == "__main__":
    main()
