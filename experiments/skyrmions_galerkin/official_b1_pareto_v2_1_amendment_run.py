"""Resumable runner for the pre-action B1 V2.1 start-availability amendment."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from . import official_b1_pareto_v2_1_amendment as amendment


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("amend", "selection", "certify", "validation", "report", "all"),
    )
    args = parser.parse_args()
    routes = {
        "amend": amendment.prepare_amendment,
        "selection": amendment.run_selection,
        "certify": amendment.certify,
        "validation": amendment.validate,
        "report": amendment.write_reports,
    }
    order = ("amend", "selection", "certify", "validation", "report")
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in order if args.mode == "all" else (args.mode,):
            print(f"starting={mode}", flush=True)
            before = amendment.base._gpu_snapshot()
            started = time.perf_counter()
            result = routes[mode](progress=progress)
            elapsed = time.perf_counter() - started
            after = amendment.base._gpu_snapshot()
            if mode not in {"amend", "report"}:
                amendment.base.record_stage_performance(mode, elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
