"""Small opt-in benchmark for candidate-specific native I-projection batching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

BUILD = Path(__file__).resolve().parent / "build"
if str(BUILD) not in sys.path:
    sys.path.insert(0, str(BUILD))

import _candidate_iprojection_native as native  # noqa: E402


def _median_seconds(call, repetitions: int) -> float:
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        call()
        values.append(time.perf_counter() - started)
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--times", type=int, default=9)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--moments", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = (args.candidates, args.times, args.particles, args.moments)
    phi = rng.normal(size=shape)
    base = rng.uniform(0.5, 1.5, size=(args.times, args.particles))
    base /= base.sum(axis=-1, keepdims=True)
    log_base = np.log(base)
    target_logits = rng.normal(
        scale=0.25, size=(args.candidates, args.times, args.particles)
    )
    target_logits -= target_logits.max(axis=-1, keepdims=True)
    target_weights = np.exp(target_logits)
    target_weights /= target_weights.sum(axis=-1, keepdims=True)
    targets = np.einsum("ctn,ctnm->ctm", target_weights, phi)
    solver_args = (100, 1.0e-9, 1.0e-9, 20.0, 1000.0, 6, 0.0)

    def batched():
        return native.solve_candidate_batch(phi, log_base, targets, *solver_args)

    def scalar_loop():
        return [
            native.solve_batch(
                phi[c], log_base, targets[c : c + 1], *solver_args
            )
            for c in range(args.candidates)
        ]

    check = batched()
    reference = np.stack(
        [row["lambda_values"][0] for row in scalar_loop()], axis=0
    )
    batched_seconds = _median_seconds(batched, args.repetitions)
    scalar_seconds = _median_seconds(scalar_loop, args.repetitions)
    print(
        json.dumps(
            {
                "shape": list(shape),
                "repetitions": args.repetitions,
                "batched_seconds": batched_seconds,
                "scalar_loop_seconds": scalar_seconds,
                "speedup": scalar_seconds / batched_seconds,
                "maximum_lambda_difference": float(
                    np.max(np.abs(check["lambda_values"] - reference))
                ),
                "maximum_residual_norm": float(
                    np.max(check["residual_norm"])
                ),
                "all_converged": bool(np.all(check["converged"])),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
