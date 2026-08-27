"""CLI for the independent, audit-aware skyrmion Pareto-v3 workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .pareto_v3_common import OUTPUT_ROOT
from .pareto_v3_diagnostic import (
    diagnose_v2_audit_all_allowances,
    diagnose_v2_audit_starts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "diagnose-v2-audit-starts",
            "diagnose-v2-audit-all-allowances",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        if args.mode == "diagnose-v2-audit-all-allowances":
            result = diagnose_v2_audit_all_allowances(cfg)
        else:
            result = diagnose_v2_audit_starts(cfg)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    if args.mode == "diagnose-v2-audit-all-allowances":
        print(
            "allowance | screen feasible | dual-bank eligible | "
            "best audit rESS | best robust rESS"
        )
        for row in result["allowances"]:
            print(
                f"{row['allowance_percent']:>9g} | "
                f"{row['screen_feasible_count']:>15d} | "
                f"{row['dual_bank_eligible_count']:>18d} | "
                f"{row['best_audit_minimum_ress']:.7f} | "
                f"{row['best_robust_ress']:.7f}"
            )
        first = result["first_allowance_with_dual_bank_eligible_candidate"]
        print(f"first_dual_bank_allowance={first}")
        if first is None:
            print(
                "No candidate in the frozen v2 candidate pool satisfies the "
                "unchanged dual-bank rESS gate on the frozen v2 screen and "
                "periodic-audit banks at any tested allowance."
            )
        else:
            first_row = next(
                row
                for row in result["allowances"]
                if row["allowance_percent"] == first
            )
            print("dual_bank_counts_at_or_above_first=")
            for row in result["allowances"]:
                if row["allowance_percent"] >= first:
                    print(
                        f"  {row['allowance_percent']:g}%: "
                        f"{row['dual_bank_eligible_count']}"
                    )
            print("top_robust_at_first=")
            for row in first_row["top_candidates_by_robust_ress"][:10]:
                print(
                    f"  {row['candidate_id']} robust={row['robust_ress']:.9f} "
                    f"audit={row['audit_ress']:.9f} risk="
                    f"{row['scientific_selection_risk']:.9f}"
                )
    else:
        print(f"classification={result['classification']}")
        print(f"robust_count={result['audit_ress_valid_count']}")


if __name__ == "__main__":
    main()
