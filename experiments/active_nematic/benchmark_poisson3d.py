"""Small reproducible forward benchmark for the two 3-D Poisson backends."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

jax.config.update("jax_enable_x64", True)

from periodic_numerics import (  # noqa: E402
    PeriodicGrid3D,
    PeriodicPoissonConfig,
    solve_periodic_weighted_poisson3d_batch_jax,
)
from poisson3d_tesseract import (  # noqa: E402
    solve_periodic_weighted_poisson3d_batch_tesseract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--shape", type=int, nargs=3, default=(24, 24, 12))
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    grid = PeriodicGrid3D(32.0, tuple(args.shape), polarity_metric_radius=1.0)
    cfg = PeriodicPoissonConfig(cg_tol=1.0e-7, cg_maxiter=520)
    shape = (args.batch, *grid.shape)
    q0 = 0.3 + jax.random.uniform(jax.random.PRNGKey(81), shape)
    perturbation = 0.02 * jax.random.normal(jax.random.PRNGKey(82), shape)
    raw_h = jax.random.normal(jax.random.PRNGKey(83), shape)

    def inputs(scale):
        q = q0 * jnp.exp(scale * perturbation)
        h = raw_h - jnp.sum(q * raw_h, axis=(-3, -2, -1), keepdims=True) / jnp.sum(
            q, axis=(-3, -2, -1), keepdims=True
        )
        return q, h

    def objective(solver, scale):
        q, h = inputs(scale)
        return solver(q, h, grid, cfg).action

    forward_functions = {
        "jax": jax.jit(
            lambda scale: objective(
                solve_periodic_weighted_poisson3d_batch_jax, scale
            )
        ),
        "tesseract_cpp": jax.jit(
            lambda scale: objective(
                solve_periodic_weighted_poisson3d_batch_tesseract, scale
            )
        ),
    }
    gradient_functions = {
        name: jax.jit(jax.value_and_grad(lambda scale, function=function: jnp.sum(function(scale))))
        for name, function in forward_functions.items()
    }

    def measure(label, functions):
        values = {}
        for name, function in functions.items():
            jax.block_until_ready(function(0.0))
            durations = []
            for repetition in range(args.repetitions):
                scale = 0.1 * (repetition + 1)
                start = time.perf_counter()
                value = function(scale)
                jax.block_until_ready(value)
                durations.append(time.perf_counter() - start)
                values[name] = value
            print(
                f"{name} {label}: median={np.median(durations):.6f}s "
                f"samples={durations}"
            )
        return values

    values = measure("forward", forward_functions)
    measure("value_and_grad", gradient_functions)

    relative = jnp.max(
        jnp.abs(values["jax"] - values["tesseract_cpp"])
        / jnp.maximum(jnp.abs(values["jax"]), 1.0e-12)
    )
    print(f"maximum relative action difference: {float(relative):.3e}")


if __name__ == "__main__":
    main()
