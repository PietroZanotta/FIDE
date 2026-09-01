"""Resumable runner for the support-robust single-seed B1 V3 repair."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from . import official_b1_pareto_v3_support_robust as study


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "freeze", "generate-data", "refreeze-law", "selection",
            "certify", "validation", "report", "all",
        ),
    )
    args = parser.parse_args()
    routes = {
        "freeze": study.prepare_v3,
        "generate-data": study.generate_data,
        "refreeze-law": study.refreeze_law,
        "selection": study.run_selection_with_restarts,
        "certify": study.certify,
        "validation": study.validate,
        "report": study.write_reports,
    }
    order = (
        "freeze", "generate-data", "refreeze-law", "selection",
        "certify", "validation", "report",
    )
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in order if args.mode == "all" else (args.mode,):
            print(f"starting={mode}", flush=True)
            before = study.base._gpu_snapshot()
            started = time.perf_counter()
            try:
                result = routes[mode](progress=progress)
            except BaseException as error:
                if mode not in {"freeze", "generate-data"}:
                    study.write_failure_report(mode, error)
                raise
            elapsed = time.perf_counter() - started
            after = study.base._gpu_snapshot()
            if mode not in {"freeze", "report"}:
                study.base.record_stage_performance(mode, elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
