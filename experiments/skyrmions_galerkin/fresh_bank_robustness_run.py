"""CLI for the development-only fresh-bank robustness study."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .fresh_bank_robustness import (
    BANK_MANIFEST_PATH,
    CANDIDATE_FREEZE_PATH,
    OUTPUT_ROOT,
    evaluate_replicates,
    freeze_bank_manifest,
    freeze_candidates,
    generate_banks,
    run,
    summarize,
)
from .galerkin_only import execution_device
from .production_artifacts import file_sha256


def _bank_progress(replicate: int, role: str, cache_hit: bool, seconds: float) -> None:
    print(
        f"bank replicate={replicate:02d} role={role} "
        f"cache_hit={cache_hit} seconds={seconds:.3f}",
        flush=True,
    )


def _replicate_progress(row: dict, counts: dict) -> None:
    print(
        f"replicate={row['replicate_id']:02d} "
        f"screen_cache={row['screen_cache_hit']} audit_cache={row['audit_cache_hit']} "
        f"screen_seconds={row['screen_seconds']:.3f} "
        f"audit_seconds={row['audit_seconds']:.3f} "
        f"audit_candidates={row['audit_candidate_count']} "
        f"eligible_0p5={counts['0.5']} eligible_1={counts['1.0']}",
        flush=True,
    )


def _print_summary(result: dict) -> None:
    print(f"frozen candidate count: {result['candidate_count']}")
    print(f"candidate freeze SHA-256: {result['candidate_freeze_sha256']}")
    print(f"fresh replicate pairs: {result['replicate_count']}")
    print(f"bank manifest SHA-256: {result['bank_manifest_sha256']}")
    print(
        "Allowance | Ever pass | >=16/32 | >=24/32 | >=28/32 | "
        ">=30/32 | 32/32 | Best pass fraction"
    )
    for row in result["allowances"]:
        print(
            f"{row['allowance_percent']:>8g}% | "
            f"{row['candidates_ever_passing']:>9d} | "
            f"{row['candidates_ge_16_of_32']:>7d} | "
            f"{row['candidates_ge_24_of_32']:>7d} | "
            f"{row['candidates_ge_28_of_32']:>7d} | "
            f"{row['candidates_ge_30_of_32']:>7d} | "
            f"{row['candidates_32_of_32']:>5d} | "
            f"{row['maximum_pass_fraction']:.5f}"
        )
    print("old-development 0.5% witnesses:")
    print("candidate       fresh passes fraction   p10 robust-rESS   median")
    for row in result["old_0p5_percent_witnesses"]:
        distribution = row["fresh_robust_ress_performed_only"]
        print(
            f"{row['candidate_id']:<15} "
            f"{row['fresh_0p5_pass_count']:>2d}/32 "
            f"{row['fresh_0p5_pass_fraction']:.5f} "
            f"{str(distribution['p10']):>17} "
            f"{str(distribution['median']):>17}"
        )
    half = result["allowances"][0]
    print("replicate-level 0.5% coverage:")
    print(f"  replicates with zero: {half['replicates_with_zero']}")
    print(f"  replicates with >=1: {half['replicates_with_ge_1']}")
    print(f"  replicates with >=5: {half['replicates_with_ge_5']}")
    print(f"  median survivors: {half['replicate_survivor_distribution']['median']}")
    print(f"  maximum survivors: {half['replicate_survivor_distribution']['maximum']}")
    interpretation = result["development_interpretation"]
    print(f"development interpretation: {interpretation['label']}")
    print(
        "recommended next scientific step: "
        f"{interpretation['recommended_next_scientific_step']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("freeze-candidates", "freeze-manifest", "generate-banks", "evaluate", "summarize", "run"),
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
        if args.mode == "freeze-candidates":
            result = freeze_candidates(cfg)
        elif args.mode == "freeze-manifest":
            result = freeze_bank_manifest(cfg)
        elif args.mode == "generate-banks":
            result = generate_banks(cfg, progress=_bank_progress)
        elif args.mode == "evaluate":
            result = evaluate_replicates(cfg, progress=_replicate_progress)
        elif args.mode == "summarize":
            result = summarize(cfg)
        else:
            result = run(
                cfg,
                bank_progress=_bank_progress,
                replicate_progress=_replicate_progress,
            )
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    if "allowances" in result:
        _print_summary(result)
    else:
        if CANDIDATE_FREEZE_PATH.exists():
            print(f"candidate_freeze_sha256={file_sha256(CANDIDATE_FREEZE_PATH)}")
        if BANK_MANIFEST_PATH.exists():
            print(f"bank_manifest_sha256={file_sha256(BANK_MANIFEST_PATH)}")
        print(f"cache_hit={result.get('cache_hit', False)}")


if __name__ == "__main__":
    main()
