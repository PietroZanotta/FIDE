"""Resumable runner for the V3.4 energy-failure development diagnosis."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

from . import v4_energy_development as study


def progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=(
            "freeze", "banks", "dictionary", "baseline", "scaling", "splits",
            "decomposition", "spectrum", "k-grid", "crosscheck", "finalize", "all",
        ),
    )
    args = parser.parse_args()
    routes = {
        "freeze": study.freeze,
        "banks": study.generate_banks,
        "dictionary": study.build_dictionary,
        "baseline": study.run_baseline,
        "scaling": study.run_scaling,
        "splits": study.run_splits,
        "decomposition": study.run_decomposition,
        "spectrum": study.run_spectrum,
        "k-grid": study.run_k_grid,
        "crosscheck": study.run_crosscheck,
        "finalize": study.finalize,
    }
    order = (
        "freeze", "banks", "dictionary", "baseline", "scaling", "splits",
        "decomposition", "spectrum", "k-grid", "crosscheck", "finalize",
    )
    devices = jax.devices("gpu") or jax.devices()
    with jax.default_device(devices[0]):
        for mode in order if args.mode == "all" else (args.mode,):
            started = time.perf_counter()
            print(f"starting={mode}", flush=True)
            result = routes[mode](progress=progress)
            elapsed = time.perf_counter() - started
            study.record_stage_performance(mode, elapsed)
            print(
                f"completed={mode} passed={result.get('passed', True)} "
                f"wall_seconds={elapsed:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
