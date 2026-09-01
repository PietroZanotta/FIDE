"""Resumable CLI for the prospective single-seed JAX-only B1 V2 study."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from . import official_b1_pareto_v2_single_seed as study


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "preflight",
            "freeze",
            "generate-data",
            "score",
            "selection",
            "certify",
            "validation",
            "report",
            "all",
        ),
    )
    args = parser.parse_args()
    routes = {
        "preflight": study.historical_equivalence_and_profile,
        "freeze": study.freeze_protocol,
        "generate-data": study.generate_data,
        "score": study.score_candidate_universe,
        "selection": study.run_selection_with_restarts,
        "certify": study.certify_and_freeze_selection,
        "validation": study.validate_heldout,
        "report": study.write_final_reports,
    }
    order = (
        "preflight",
        "freeze",
        "generate-data",
        "score",
        "selection",
        "certify",
        "validation",
        "report",
    )
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in order if args.mode == "all" else (args.mode,):
            print(f"starting={mode}", flush=True)
            before = study._gpu_snapshot()
            started = time.perf_counter()
            result = routes[mode](progress=progress)
            elapsed = time.perf_counter() - started
            after = study._gpu_snapshot()
            if mode != "report":
                study.record_stage_performance(mode, elapsed, before, after)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
