from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (REPO / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core import make_grid, solve_v2
from mfsi.design import OptimizerConfig, optimize_multistart_candidates
from mfsi.decomposition import RasterTangentProjection, raster_tangent_projection
from v2_1_parallel_exact_solver import RESULT_FIELDS, solve_v2_parallel
from v2_1_fast_orchestration import parallel_fast_rank
from v2_1_parallel_tangent import ordered_parallel_map


def deterministic_case(batch: int, grid_shape: tuple[int, int]):
    rng = np.random.default_rng(16)
    nx, ny = grid_shape
    q = 0.2 + rng.random((batch, ny, nx))
    source = rng.normal(size=(batch, ny, nx))
    source -= source.mean(axis=(1, 2), keepdims=True)
    return q, source, make_grid(nx, ny)


def assert_bitwise_equal(left, right) -> None:
    for field in RESULT_FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(left, field)),
            np.asarray(getattr(right, field)),
            err_msg=field,
        )


def test_parallel_exact_solver_is_bitwise_identical_including_uneven_slice():
    q, source, grid = deterministic_case(7, (32, 16))
    assert_bitwise_equal(solve_v2(q, source, grid), solve_v2_parallel(q, source, grid, workers=4))


def test_parallel_exact_solver_preserves_single_field_semantics():
    q, source, grid = deterministic_case(1, (32, 16))
    assert_bitwise_equal(solve_v2(q[0], source[0], grid), solve_v2_parallel(q[0], source[0], grid))


def test_threaded_optimizer_path_is_bitwise_identical_to_serial():
    starts = jnp.asarray(
        [[-1.0, 0.2], [0.4, 1.2], [1.5, -0.3], [0.8, 0.9]],
        dtype=jnp.float64,
    )
    cfg = OptimizerConfig(steps=20, learning_rate=0.03)
    objective = lambda eta: jnp.sum((eta - jnp.asarray([0.3, -0.4])) ** 2)
    serial = optimize_multistart_candidates(objective, starts, cfg, vectorize_starts=False)
    threaded = optimize_multistart_candidates(
        objective, starts, cfg, vectorize_starts=False, start_workers=4
    )
    assert len(serial) == len(threaded)
    for expected, actual in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(expected.eta, actual.eta)
        assert expected.value == actual.value
        assert expected.feasible == actual.feasible
        assert expected.violations == actual.violations


def test_parallel_fast_rank_preserves_single_candidate_values_and_order():
    candidates = [
        {"candidate_id": f"c{index}", "eta": [float(x), float(y)]}
        for index, (x, y) in enumerate(((0.4, 0.1), (-0.2, 0.7), (0.1, -0.3), (0.2, 0.2)))
    ]
    pool = {tuple(row["eta"]): row for row in candidates}
    objective = lambda eta: jnp.sum((eta - jnp.asarray([0.15, -0.25])) ** 2)
    constraint = lambda eta: eta[0] + eta[1]
    expected = []
    for candidate in candidates:
        eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
        expected.append(
            dict(
                candidate,
                fast_value=float(jax.jit(objective)(eta)),
                fast_violation=max(0.0, float(constraint(eta) - 0.5)),
            )
        )
    expected.sort(
        key=lambda row: (
            not np.isfinite(row["fast_value"]), row["fast_violation"],
            row["fast_value"], row["candidate_id"],
        )
    )
    actual = parallel_fast_rank(None, pool, objective, ((constraint, 0.5),), "test", workers=4)
    assert actual == expected


def test_ordered_parallel_tangent_map_preserves_input_order():
    rows = list(range(11))
    actual = ordered_parallel_map(
        lambda value: {"input": value, "output": value * value},
        rows,
        workers=4,
        label="test ordered Tangent map",
    )
    assert actual == [
        {"input": value, "output": value * value} for value in rows
    ]


def test_batched_decomposition_is_bitwise_identical_to_per_trial_calls():
    rng = np.random.default_rng(18)
    batch, times, ny, nx, moments = 2, 2, 8, 16, 4
    potential = rng.normal(size=(batch, times, ny, nx)) * 0.01
    q = 0.2 + rng.random((batch, times, ny, nx))
    source = rng.normal(size=(batch, times, ny, nx))
    source -= source.mean(axis=(-2, -1), keepdims=True)
    features = rng.normal(size=(ny, nx, moments)) * 0.1
    kwargs = dict(
        dx=1 / nx, cell_area=(1 / nx) ** 2, pinv_rcond=1e-10,
        operator_floor_rel=0.0, gauge_strength=0.0, source_is_density=True,
    )
    sequential = [
        raster_tangent_projection(
            jnp.asarray(potential[index]), jnp.asarray(q[index]),
            jnp.asarray(source[index]), jnp.asarray(features), **kwargs,
        ) for index in range(batch)
    ]
    batched = raster_tangent_projection(
        jnp.asarray(potential), jnp.asarray(q), jnp.asarray(source),
        jnp.asarray(features), **kwargs,
    )
    for field in RasterTangentProjection._fields:
        expected = np.stack([np.asarray(getattr(row, field)) for row in sequential])
        np.testing.assert_allclose(
            expected, np.asarray(getattr(batched, field)), rtol=5e-15, atol=5e-15,
            err_msg=field,
        )
