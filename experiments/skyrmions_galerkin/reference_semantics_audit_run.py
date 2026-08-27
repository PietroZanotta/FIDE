"""CLI for the development-only reference semantics audit."""

from __future__ import annotations

import argparse
import jax

from .reference_semantics_audit import (
    console_report,
    finalize,
    run_endpoint_audit,
    run_whitening_audit,
    verify_and_seal_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "whitening", "endpoint", "finalize", "all", "print"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    def progress(label: str, cached: bool, seconds: float) -> None:
        print(f"endpoint rollout {label}: {'cached' if cached else f'{seconds:.1f}s'}", flush=True)

    if args.stage in {"verify", "all"}:
        verify_and_seal_sources(); print("SOURCE VERIFIED", flush=True)
    if args.stage in {"whitening", "all"}:
        run_whitening_audit(); print("whitening audit complete", flush=True)
    if args.stage in {"endpoint", "all"}:
        run_endpoint_audit(progress=progress)
    if args.stage in {"finalize", "all"}:
        finalize()
    if args.stage in {"print", "all"}:
        print(console_report(), flush=True)


if __name__ == "__main__":
    main()
