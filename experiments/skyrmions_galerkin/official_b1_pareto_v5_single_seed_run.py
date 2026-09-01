"""Resumable runner for the prospective Skyrmion Galerkin V5 authority."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from . import official_b1_pareto_v5_single_seed as study


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "freeze", "generate-data", "score", "refreeze-law", "selection",
            "certify", "heldout-generate", "validation", "finalize", "all",
        ),
    )
    parser.add_argument("--worker-role", choices=tuple(study.GUARD_COUNTS))
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_role:
        if args.worker_input is None or args.worker_output is None:
            parser.error("guard worker requires --worker-input and --worker-output")
        study.guard_worker(args.worker_role, args.worker_input, args.worker_output)
        return
    if args.mode is None:
        parser.error("--mode is required outside guard-worker mode")

    routes = {
        "freeze": study.freeze_v5,
        "generate-data": study.generate_data,
        "score": study.score_candidate_universe,
        "refreeze-law": study.refreeze_law,
        "selection": study.run_selection,
        "certify": study.certify_selection,
        "heldout-generate": study.generate_heldout,
        "validation": study.validate_heldout,
        "finalize": study.finalize,
    }
    order = (
        "freeze", "generate-data", "score", "refreeze-law", "selection",
        "certify", "heldout-generate", "validation", "finalize",
    )
    modes = order if args.mode == "all" else (args.mode,)
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in modes:
            if mode != "freeze":
                study.activate()
            print(f"starting={mode}", flush=True)
            before = study.base._gpu_snapshot()
            started = time.perf_counter()
            try:
                result = routes[mode](progress=progress)
            except BaseException as error:
                if study.PROTOCOL_PATH.exists() and mode not in {"freeze", "finalize"}:
                    study.write_terminal_failure(mode, error)
                raise
            elapsed = time.perf_counter() - started
            after = study.base._gpu_snapshot()
            if study.PROTOCOL_PATH.exists() and mode not in {"finalize"}:
                study.activate()
                study.base.record_stage_performance(mode, elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
