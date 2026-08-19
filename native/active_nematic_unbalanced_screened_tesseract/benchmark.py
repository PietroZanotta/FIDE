"""Small reproducible throughput benchmark for the native trajectory solve."""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time

import numpy as np

BUILD = Path(__file__).resolve().parent / "build"
sys.path.insert(0, str(BUILD))
import _active_nematic_unbalanced_screened_native as native


def main() -> None:
    batch, nx, ny, ntheta = 11, 48, 48, 24
    x = np.arange(nx)[None, :, None, None]
    y = np.arange(ny)[None, None, :, None]
    theta = np.arange(ntheta)[None, None, None, :]
    phase = np.arange(batch)[:, None, None, None]
    q = 1.0 + 0.15 * np.cos(2.0 * np.pi * (x + phase) / nx)
    q = q + 0.10 * np.sin(2.0 * np.pi * y / ny)
    q = q + 0.08 * np.cos(2.0 * np.pi * theta / ntheta)
    q = np.ascontiguousarray(np.broadcast_to(q, (batch, nx, ny, ntheta)))
    h = np.sin(2.0 * np.pi * x / nx) * np.cos(2.0 * np.pi * y / ny)
    h = np.ascontiguousarray(np.broadcast_to(h, q.shape))
    rhs = np.ascontiguousarray(q * h)
    args = (32.0 / nx, 32.0 / ny, 2.0 * np.pi / ntheta, 1.0, 1.0e-7, 1200)
    native.solve_batch(q, rhs, *args, None)
    timings = []
    iterations = None
    for _ in range(7):
        start = time.perf_counter()
        result = native.solve_batch(q, rhs, *args, None)
        timings.append(time.perf_counter() - start)
        iterations = np.asarray(result["iterations"])
    if not np.all(np.asarray(result["converged"], dtype=bool)):
        raise RuntimeError("benchmark system did not converge")
    seconds = statistics.median(timings)
    points = q.size
    print(f"batch={batch} grid={nx}x{ny}x{ntheta}")
    print(f"median_seconds={seconds:.6f}")
    print(f"million_grid_points_per_second={points / seconds / 1.0e6:.2f}")
    print(f"iterations_mean={iterations.mean():.1f} iterations_max={iterations.max()}")
    print(
        "maximum_relative_residual="
        f"{np.max(np.asarray(result['relative_residual'])):.3e}"
    )


if __name__ == "__main__":
    main()
