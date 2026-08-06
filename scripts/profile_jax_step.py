#!/usr/bin/env python3
"""Reusable profiler harness for a compiled training step.

Edit ``build_workload`` to return ``(step_fn, state, inputs)`` from the local
project, then run this script.  The warm-up is excluded from the trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import jax


def build_workload() -> tuple[Callable[..., Any], Any, tuple[Any, ...]]:
    raise NotImplementedError(
        "Connect build_workload() to your project's compiled train chunk. "
        "Return (step_fn, state, inputs)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/jax_profile"))
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    step_fn, state, inputs = build_workload()
    state = step_fn(state, *inputs)
    jax.block_until_ready(state)
    args.output.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(args.output)):
        for _ in range(args.iterations):
            state = step_fn(state, *inputs)
        jax.block_until_ready(state)


if __name__ == "__main__":
    main()
