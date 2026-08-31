"""CLI for independent single-seed, multi-risk B1 Galerkin Pareto runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from . import per_seed_pareto as study
from .three_reference_pareto import CONFIG_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=study.SUPPORTED_SEEDS, default="B1_seed0")
    parser.add_argument(
        "--stage",
        choices=("protocol", "data", "law", "candidates", "screen", "tangent", "full", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_root = study.configure_seed(args.seed)
    if args.dry_run:
        print(f"mode=single-reference multi-risk Pareto")
        print(f"seed={args.seed}")
        print(f"allowances={list(study.engine.ALLOWANCES)}")
        print(f"output_root={output_root}")
        return
    cfg = load_config(args.config)
    print(f"device={execution_device()}", flush=True)
    print(f"seed={args.seed}", flush=True)
    with jax.default_device(execution_device()):
        result = study.run_seed_stage(
            cfg, args.seed, args.stage,
            progress=lambda value: print(value, flush=True),
        )
    print(f"stage={args.stage} status={result.get('status', 'complete')}", flush=True)
    print(f"output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
