"""CLI for the development-only nested-N rESS convergence study."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .ress_n_convergence import (
    OUTPUT_ROOT,
    evaluate_master_pairs,
    freeze_candidate_panel,
    freeze_master_manifest,
    generate_master_banks,
    run,
    summarize,
)


def _bank_progress(replicate: int, role: str, cache_hit: bool, seconds: float) -> None:
    print(
        f"master_bank replicate={replicate:02d} role={role} "
        f"cache_hit={cache_hit} seconds={seconds:.3f}",
        flush=True,
    )


def _evaluation_progress(replicate: int, role: str, N: int, cache_hit: bool, seconds: float) -> None:
    if role == "pair":
        print(f"master_pair replicate={replicate:02d} cache_hit={cache_hit}", flush=True)
    else:
        print(
            f"evaluation replicate={replicate:02d} role={role} N={N} "
            f"cache_hit={cache_hit} seconds={seconds:.3f}",
            flush=True,
        )


def _print_summary(result: dict) -> None:
    print("SOURCE VERIFIED")
    print(f"diagnostic candidates: {result['diagnostic_candidate_count']}")
    print(f"master pairs: {result['master_pair_count']}")
    print("N ladder: 8192 -> 16384 -> 32768 -> 65536")
    print("LAW CONVERGENCE")
    print("N       bank pass/32   pair pass/16   p10 rESS   median rESS   p90 rESS")
    for row in result["law_convergence"]:
        distribution = row["minimum_ress_distribution"]
        print(
            f"{row['N']:<7d} {row['individual_bank_pass_count']:>2d}/32          "
            f"{row['pair_pass_count']:>2d}/16          {distribution['p10']:.6f}   "
            f"{distribution['median']:.6f}      {distribution['p90']:.6f}"
        )
    print("0.5% HIGH-PASS PANEL")
    print("N       median bank-pass  median pair-pass  >=12/16  >=14/16  16/16")
    for row in result["high_pass_panel"]:
        print(
            f"{row['N']:<7d} {row['bank_pass_fraction_distribution']['median']:.5f}           "
            f"{row['pair_pass_fraction_distribution']['median']:.5f}          "
            f"{row['candidates_ge_12_of_16_pairs']:>3d}      "
            f"{row['candidates_ge_14_of_16_pairs']:>3d}     "
            f"{row['candidates_16_of_16_pairs']:>3d}"
        )
    print("NESTED CONVERGENCE")
    print("transition          median |delta rESS|     p90 |delta|")
    for row in result["nested_convergence"]:
        if row["group"] == "high_pass":
            absolute = row["absolute_delta"]
            print(
                f"{row['from_N']} -> {row['to_N']:<7d} "
                f"{absolute['median']:.7f}               {absolute['p90']:.7f}"
            )
    time7 = __import__("json").loads((OUTPUT_ROOT / "time7_diagnostics.json").read_text())
    print("TIME NODE 7")
    print("N       median rESS   median lambda norm   median max weight   median top-1% mass")
    for row in time7["rows"]:
        if row["group"] == "high_pass":
            print(
                f"{row['N']:<7d} {row['ress_trajectory']['node7']['median']:.6f}      "
                f"{row['lambda_norm']['node7']['median']:.6f}             "
                f"{row['maximum_normalized_weight']['node7']['median']:.7f}          "
                f"{row['top_1pct_weight_mass']['node7']['median']:.6f}"
            )
    interpretation = result["interpretation"]
    print(f"DEVELOPMENT INTERPRETATION: {interpretation['label']}")
    print(f"RECOMMENDED NEXT SCIENTIFIC STEP: {interpretation['recommended_next_scientific_step']}")
    print("NO Tangent run")
    print("NO Full run")
    print("NO validation")
    print("NO official protocol created")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("freeze-panel", "freeze-manifest", "generate-banks", "evaluate", "summarize", "run"),
        default="run",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        if args.mode == "freeze-panel":
            result = freeze_candidate_panel()
        elif args.mode == "freeze-manifest":
            result = freeze_master_manifest(cfg)
        elif args.mode == "generate-banks":
            result = generate_master_banks(cfg, progress=_bank_progress)
        elif args.mode == "evaluate":
            result = evaluate_master_pairs(cfg, progress=_evaluation_progress)
        elif args.mode == "summarize":
            result = summarize(cfg)
        else:
            result = run(cfg, bank_progress=_bank_progress, evaluation_progress=_evaluation_progress)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    if "interpretation" in result:
        _print_summary(result)
    else:
        print(f"cache_hit={result.get('cache_hit', False)}")


if __name__ == "__main__":
    main()
