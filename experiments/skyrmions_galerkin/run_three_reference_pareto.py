"""CLI for the three-reference B1 Galerkin Pareto study."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from . import three_reference_pareto as study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("protocol", "data", "law", "candidates", "screen",
                                            "tangent", "full", "finalize", "all"), default="all")
    parser.add_argument("--config", type=Path, default=study.CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        payload = study.protocol_payload(cfg)
        print(f"version={payload['version']}")
        print(f"flows={','.join(payload['reference']['flow_ids'])}")
        print(f"allowances={payload['allowances_percent']}")
        print(f"output_root={study.OUTPUT_ROOT}")
        return
    print(f"device={execution_device()}", flush=True)
    with jax.default_device(execution_device()):
        result = study.run_stage(cfg, args.stage, progress=lambda value: print(value, flush=True))
    print(f"stage={args.stage} status={result.get('status', 'complete')}", flush=True)
    print(f"output_root={study.OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()

