"""CLI for fixed-K=280 quadrature qualification; no optimizer mode exists."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from mfsi.config import load_config

from .galerkin_only import execution_device
from .k280_quadrature import (
    OUTPUT_ROOT,
    audit_old_certificates,
    freeze_protocol,
    generate_banks,
    run_evaluate,
    run_finite_difference,
)
from .k280_quadrature_report import run_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "audit-old-certificates",
        "freeze-protocol",
        "generate-banks",
        "evaluate",
        "finite-difference",
        "report",
    ))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    routes = {
        "audit-old-certificates": audit_old_certificates,
        "freeze-protocol": freeze_protocol,
        "generate-banks": generate_banks,
        "evaluate": run_evaluate,
        "finite-difference": run_finite_difference,
        "report": run_report,
    }
    device = jax.devices("cpu")[0] if args.force_cpu else execution_device()
    with jax.default_device(device):
        result = routes[args.mode](cfg)
    print(f"mode={args.mode}")
    print(f"output_root={OUTPUT_ROOT}")
    print(f"passed={result.get('passed', True)}")
    if result.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
