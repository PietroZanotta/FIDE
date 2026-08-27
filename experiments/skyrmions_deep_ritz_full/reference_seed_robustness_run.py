"""CLI for the endpoint-only reference-checkpoint robustness study."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .reference_seed_robustness import (
    OUTPUT_ROOT,
    evaluate_bridge_quality,
    freeze_experiment_manifest,
    rank_phase_a,
    run,
    run_phase_a,
    run_phase_b,
    summarize,
    summarize_phase_a,
    summarize_phase_b,
    train_models,
    verify_and_freeze_sources,
)


def _progress(stage: str, label: str, cache_hit: bool, seconds: float) -> None:
    print(
        f"stage={stage} label={label} cache_hit={cache_hit} seconds={seconds:.3f}",
        flush=True,
    )


def _print_summary(result: dict) -> None:
    print("SOURCE VERIFIED")
    print()
    print("baseline reference:")
    print(result["baseline_reference"]["path"])
    print(result["baseline_reference"]["sha256"])
    print()
    print("new endpoint-only references trained:")
    print(result["new_endpoint_only_references_trained"])
    print()
    print("intermediate truth used:")
    print("NO")
    print()
    print("validation accessed:")
    print("NO")
    print()
    print("DEVELOPMENT INTERPRETATION:")
    print(result["development_interpretation"])
    print()
    print("RECOMMENDED NEXT SCIENTIFIC STEP:")
    print(result["recommended_next_scientific_step"])
    print()
    print("NO intermediate-truth training")
    print("NO Tangent")
    print("NO Full")
    print("NO validation")
    print("NO official reference replacement")
    print("NO official protocol created")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "verify",
            "freeze",
            "train",
            "bridge-eval",
            "phase-a",
            "phase-a-summary",
            "rank",
            "phase-b",
            "phase-b-summary",
            "summarize",
            "run",
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
            result = freeze_experiment_manifest(cfg)
        elif args.mode == "train":
            result = train_models(cfg, progress=_progress)
        elif args.mode == "bridge-eval":
            result = evaluate_bridge_quality(cfg, progress=_progress)
        elif args.mode == "phase-a":
            result = run_phase_a(cfg, progress=_progress)
        elif args.mode == "phase-a-summary":
            result = summarize_phase_a(cfg)
        elif args.mode == "rank":
            result = rank_phase_a(cfg)
        elif args.mode == "phase-b":
            result = run_phase_b(cfg, progress=_progress)
        elif args.mode == "phase-b-summary":
            result = summarize_phase_b(cfg)
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
