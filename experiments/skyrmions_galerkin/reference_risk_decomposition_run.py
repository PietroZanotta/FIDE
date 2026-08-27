"""CLI for the development-only reference-risk decomposition."""

from __future__ import annotations

import argparse

import jax

from .reference_risk_decomposition import (
    build_cross_benchmark_audit,
    console_report,
    finalize_outputs,
    run_decomposition,
    verify_and_seal_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "audit", "decompose", "finalize", "all", "print"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    def progress(label: str, bank: int, cached: bool, seconds: float) -> None:
        state = "cached" if cached else f"{seconds:.1f}s"
        print(f"Law decomposition {label}/bank_{bank:02d}: {state}", flush=True)

    if args.stage in {"verify", "all"}:
        verify_and_seal_sources()
        print("SOURCE VERIFIED", flush=True)
    if args.stage in {"audit", "all"}:
        build_cross_benchmark_audit()
        print("cross-benchmark audit complete", flush=True)
    if args.stage in {"decompose", "all"}:
        run_decomposition(progress=progress)
    if args.stage in {"finalize", "all"}:
        finalize_outputs()
    if args.stage in {"print", "all"}:
        print(console_report(), flush=True)


if __name__ == "__main__":
    main()
