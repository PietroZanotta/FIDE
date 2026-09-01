from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.grid import RectangularGrid2D
from mfsi.raster import rasterize_projected_particles_reflected_rect
from reflected_raster import (
    build_reflected_raster_plan,
    common_reference_scott_bandwidth,
    rasterize_reflected_with_plan,
    reference_scott_bandwidth,
)

jax.config.update("jax_enable_x64", True)


def _case():
    rng = np.random.default_rng(91)
    nodes = rng.uniform([0.01, 0.01], [1.99, 0.99], size=(3, 48, 2))
    raw = rng.uniform(0.2, 1.0, size=(2, 3, 48))
    weights = raw / raw.sum(axis=-1, keepdims=True)
    forcing = rng.normal(size=weights.shape)
    forcing -= np.sum(weights * forcing, axis=-1, keepdims=True)
    return nodes, weights, forcing


def test_cached_reflected_plan_matches_authoritative_primitive():
    nodes, weights, forcing = _case()
    grid = RectangularGrid2D(0.0, 2.0, 0.0, 1.0, 16, 8)
    bandwidth = 0.08
    plan = build_reflected_raster_plan(
        nodes, grid, bandwidth=bandwidth, image_pairs=4
    )
    actual = rasterize_reflected_with_plan(weights, forcing, plan)
    expected = []
    for trial in range(weights.shape[0]):
        rows = [
            rasterize_projected_particles_reflected_rect(
                nodes[time],
                weights[trial, time],
                forcing[trial, time],
                grid,
                bandwidth=bandwidth,
                image_pairs=4,
            )
            for time in range(nodes.shape[0])
        ]
        expected.append(np.stack([np.asarray(row.q) for row in rows]))
    assert np.allclose(np.asarray(actual.q), np.stack(expected), rtol=2e-13, atol=2e-14)
    assert np.max(np.abs(np.asarray(actual.source_mass_after_center))) < 2e-15
    assert np.all(np.asarray(actual.q) > 0.0)


def test_cached_reflected_plan_preserves_weight_gradients():
    nodes, weights, forcing = _case()
    grid = RectangularGrid2D(0.0, 2.0, 0.0, 1.0, 12, 6)
    plan = build_reflected_raster_plan(nodes, grid, bandwidth=0.09)
    weights_jax = jnp.asarray(weights)
    forcing_jax = jnp.asarray(forcing)

    def value(w):
        raster = rasterize_reflected_with_plan(w, forcing_jax, plan)
        return jnp.sum(raster.source * raster.source / raster.q) * grid.cell_area

    gradient = np.asarray(jax.grad(value)(weights_jax))
    assert gradient.shape == weights.shape
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_scott_bandwidth_is_physical_and_common_across_references(tmp_path: Path):
    nodes, _, _ = _case()
    weights = np.full(nodes.shape[:2], 1.0 / nodes.shape[1])
    first, _ = reference_scott_bandwidth(nodes, weights)
    shifted = np.clip(nodes * np.asarray([0.8, 0.9]) + 0.03, [0.0, 0.0], [2.0, 1.0])
    second, _ = reference_scott_bandwidth(shifted, weights)
    paths = []
    for index, values in enumerate((nodes, shifted)):
        path = tmp_path / f"reference_{index}.npz"
        np.savez(path, nodes=values, weights=weights)
        paths.append(path)
    common, per_reference = common_reference_scott_bandwidth(paths)
    assert np.allclose(per_reference, [first, second])
    assert common == np.median([first, second])
    # No PDE-grid argument enters this rule.
    assert first > 0.0 and second > 0.0
