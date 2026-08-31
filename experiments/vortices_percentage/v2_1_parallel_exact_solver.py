"""V2.1-only parallel orchestration for the unchanged physical Poisson solver."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from mfsi.poisson import (
    PhysicalPoissonBatchResult,
    PoissonConfig,
    solve_weighted_poisson_source_physical_direct_batch,
)


RESULT_FIELDS = (
    "action",
    "potential",
    "relative_residual",
    "absolute_residual",
    "maximum_component_compatibility_residual",
    "component_count",
    "compatible",
    "solver_converged",
)


def _concatenate(parts: list[PhysicalPoissonBatchResult]) -> PhysicalPoissonBatchResult:
    values = {
        field: np.concatenate([np.asarray(getattr(part, field)) for part in parts], axis=0)
        for field in RESULT_FIELDS
    }
    return PhysicalPoissonBatchResult(**values)


def solve_v2_parallel(q: Any, source: Any, grid: Any, *, workers: int = 4):
    """Solve independent batch rows concurrently with the frozen exact routine.

    Each slice invokes ``solve_weighted_poisson_source_physical_direct_batch``
    unchanged. Contiguous outputs are concatenated in original batch order, so
    every individual sparse factorization, refinement, gate, and reduction is
    numerically identical to sequential execution.
    """
    cfg = PoissonConfig(
        dx=grid.require_isotropic_spacing(),
        operator_floor_rel=0.0,
        cg_tol=1.0e-10,
        cg_maxiter=4000,
        gauge_strength=0.0,
    )
    q_array = np.asarray(q, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    if q_array.ndim == 2:
        q_array = q_array[None]
        source_array = source_array[None]
    if q_array.ndim != 3 or source_array.shape != q_array.shape:
        raise ValueError("q and source must have identical [B,H,W] shapes")
    worker_count = max(1, min(int(workers), int(q_array.shape[0])))
    if worker_count == 1:
        return solve_weighted_poisson_source_physical_direct_batch(
            q_array,
            -source_array,
            cfg,
            compatibility_tolerance=1.0e-10,
            reject_incompatible=False,
        )
    boundaries = np.linspace(0, q_array.shape[0], worker_count + 1, dtype=np.int64)
    spans = [
        (int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(worker_count)
        if boundaries[index] < boundaries[index + 1]
    ]

    def solve_span(span: tuple[int, int]):
        begin, end = span
        return solve_weighted_poisson_source_physical_direct_batch(
            q_array[begin:end],
            -source_array[begin:end],
            cfg,
            compatibility_tolerance=1.0e-10,
            reject_incompatible=False,
        )

    with ThreadPoolExecutor(max_workers=len(spans)) as pool:
        parts = list(pool.map(solve_span, spans))
    return _concatenate(parts)
