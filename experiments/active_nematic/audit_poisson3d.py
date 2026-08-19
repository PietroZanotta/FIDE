"""Resolution, metric, and backend audit for the polarity full-action solver."""

from __future__ import annotations

import argparse
import json
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
    rasterize_periodic_particles3d,
    solve_periodic_weighted_poisson3d_batch_jax,
)
from poisson3d_tesseract import native_diagnostics  # noqa: E402


def _particles(seed: int, batch: int, count: int, box_size: float):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    position = box_size * jax.random.uniform(keys[0], (count, 2))
    angle = 2.0 * jnp.pi * jax.random.uniform(keys[1], (count, 1))
    nodes = jnp.concatenate((position, angle), axis=-1)
    raw_weights = jnp.exp(0.5 * jax.random.normal(keys[2], (batch, count)))
    weights = raw_weights / jnp.sum(raw_weights, axis=-1, keepdims=True)
    forcing = jax.random.normal(keys[3], (batch, count))
    forcing = forcing - jnp.sum(weights * forcing, axis=-1, keepdims=True)
    return nodes, weights, forcing


def _audit_case(
    nodes,
    weights,
    forcing,
    *,
    box_size: float,
    shape: tuple[int, int, int],
    radius: float,
    bandwidth: float,
    cfg: PeriodicPoissonConfig,
) -> dict:
    grid = PeriodicGrid3D(box_size, shape, polarity_metric_radius=radius)
    rasters = [
        rasterize_periodic_particles3d(
            nodes, weights[index], forcing[index], grid, bandwidth=bandwidth
        )
        for index in range(weights.shape[0])
    ]
    q = jnp.stack([row.q for row in rasters])
    h = jnp.stack([row.h for row in rasters])
    jax.block_until_ready(h)

    native_start = time.perf_counter()
    native = native_diagnostics(q, h, grid, cfg)
    native_seconds = time.perf_counter() - native_start
    jax_start = time.perf_counter()
    reference = solve_periodic_weighted_poisson3d_batch_jax(q, h, grid, cfg)
    jax.block_until_ready(reference.action)
    jax_seconds = time.perf_counter() - jax_start

    native_action = np.asarray(native["action"])
    reference_action = np.asarray(reference.action)
    relative_action_error = np.abs(native_action - reference_action) / np.maximum(
        np.abs(reference_action), 1.0e-14
    )
    mass = grid.cell_volume * np.sum(np.asarray(q), axis=(-3, -2, -1))
    compatibility = grid.cell_volume * np.sum(
        np.asarray(q * h), axis=(-3, -2, -1)
    )
    return {
        "shape": list(shape),
        "polarity_metric_radius": radius,
        "spacings": list(grid.spacings),
        "cell_volume": grid.cell_volume,
        "bandwidth": bandwidth,
        "native_action": native_action.tolist(),
        "jax_action": reference_action.tolist(),
        "maximum_backend_relative_action_error": float(np.max(relative_action_error)),
        "native_iterations": np.asarray(native["iterations"]).tolist(),
        "native_relative_residual": np.asarray(native["relative_residual"]).tolist(),
        "jax_relative_residual": np.asarray(reference.relative_residual).tolist(),
        "all_native_converged": bool(np.all(native["converged"])),
        "maximum_mass_error": float(np.max(np.abs(mass - 1.0))),
        "maximum_compatibility_error": float(np.max(np.abs(compatibility))),
        "native_seconds": native_seconds,
        "jax_seconds_including_compilation": jax_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--box-size", type=float, default=32.0)
    parser.add_argument("--bandwidth", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "audits" / "poisson3d_audit.json",
    )
    args = parser.parse_args()
    nodes, weights, forcing = _particles(
        args.seed, args.batch, args.particles, args.box_size
    )
    cfg = PeriodicPoissonConfig(
        operator_floor_rel=2.0e-5,
        cg_tol=1.0e-7,
        cg_maxiter=520,
        gauge_strength=1.0,
    )
    cases = [
        ((12, 12, 6), 1.0),
        ((18, 18, 9), 1.0),
        ((24, 24, 12), 1.0),
        ((32, 32, 16), 1.0),
        ((48, 48, 24), 1.0),
        ((18, 18, 9), 0.5),
        ((18, 18, 9), 2.0),
    ]
    rows = [
        _audit_case(
            nodes,
            weights,
            forcing,
            box_size=args.box_size,
            shape=shape,
            radius=radius,
            bandwidth=args.bandwidth,
            cfg=cfg,
        )
        for shape, radius in cases
    ]
    refinement = [row for row in rows if row["polarity_metric_radius"] == 1.0]
    finest = np.asarray(refinement[-1]["native_action"])
    for row in refinement:
        action = np.asarray(row["native_action"])
        row["relative_action_change_from_finest"] = (
            np.abs(action - finest) / np.maximum(np.abs(finest), 1.0e-14)
        ).tolist()
    payload = {
        "schema_version": 1,
        "description": "Synthetic particle-like 3-D periodic Poisson audit",
        "seed": args.seed,
        "batch": args.batch,
        "particles": args.particles,
        "solver_config": {
            "operator_floor_rel": cfg.operator_floor_rel,
            "cg_tol": cfg.cg_tol,
            "cg_maxiter": cfg.cg_maxiter,
            "gauge_strength": cfg.gauge_strength,
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
