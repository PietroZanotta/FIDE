"""CLI for the development-only three-model reference retraining ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .reference_retraining import (
    OUTPUT_ROOT,
    evaluate_models,
    freeze_protocol,
    generate_matched_banks,
    generate_regenerated_data,
    run,
    summarize,
    train_models,
    verify_and_freeze_sources,
)


def _progress(stage: str, label: str, cache_hit: bool, seconds: float) -> None:
    print(
        f"stage={stage} label={label} cache_hit={cache_hit} seconds={seconds:.3f}",
        flush=True,
    )


def _print_summary(result: dict) -> None:
    print("REFERENCE RETRAINING ENSEMBLE COMPLETE")
    print(
        "model       fixed-CRN loss   mean truth-moment error   "
        "Law min rESS   high-panel median min rESS   high pass"
    )
    for row in result["models"]:
        mean_error = sum(
            entry["mean"] for entry in row["truth_moment_error"].values()
        ) / len(row["truth_moment_error"])
        print(
            f"{row['label']:<11s} {row['fixed_crn_loss']['mean']:<16.8f} "
            f"{mean_error:<25.8f} {row['law_minimum_ress_all_13_nodes']:<14.6f} "
            f"{row['high_pass_minimum_ress_all_13_nodes']['median']:<29.6f} "
            f"{row['high_pass_pass_count']:>2d}/55"
        )
    diagnostics = result["fresh_model_diagnostics"]
    print(f"fresh Law-minimum rESS spread: {diagnostics['law_minimum_ress_spread']:.6f}")
    print(
        "fresh high-panel median-minimum rESS spread: "
        f"{diagnostics['high_pass_median_minimum_ress_spread']:.6f}"
    )
    print(f"DEVELOPMENT INTERPRETATION: {result['development_interpretation']}")
    print(f"NEXT STEP: {result['recommended_next_scientific_step']}")
    print("Production checkpoint unchanged; no model installed")
    print("No official protocol, downstream solve, or validation access")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "verify", "freeze", "generate-data", "train", "generate-banks",
            "evaluate", "summarize", "run",
        ),
        default="run",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    print(f"device={device}", flush=True)
    with jax.default_device(device):
        if args.mode == "verify":
            result = verify_and_freeze_sources()
        elif args.mode == "freeze":
            result = freeze_protocol(cfg)
        elif args.mode == "generate-data":
            result = generate_regenerated_data(cfg)
        elif args.mode == "train":
            result = train_models(cfg, progress=_progress)
        elif args.mode == "generate-banks":
            result = generate_matched_banks(cfg, progress=_progress)
        elif args.mode == "evaluate":
            result = evaluate_models(cfg, progress=_progress)
        elif args.mode == "summarize":
            result = summarize(cfg)
        else:
            result = run(cfg, progress=_progress)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    if "development_interpretation" in result:
        _print_summary(result)
    else:
        print(f"cache_hit={result.get('cache_hit', False)}")


if __name__ == "__main__":
    main()
