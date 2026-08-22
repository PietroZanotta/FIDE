"""Frozen temporal-refinement helpers for the ocean post-dispersion action."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .post_dispersion_regularization import normalized_trapezoid_weights


def nested_source_grids(
    start: int, end: int, source_steps: Sequence[int]
) -> tuple[np.ndarray, ...]:
    """Return endpoint-inclusive source grids and verify exact nesting."""
    start = int(start)
    end = int(end)
    steps = tuple(int(value) for value in source_steps)
    if end <= start or not steps or any(step <= 0 for step in steps):
        raise ValueError("invalid temporal-refinement source contract")
    if any((end - start) % step for step in steps):
        raise ValueError("every temporal step must divide the window exactly")
    grids = tuple(np.arange(start, end + 1, step, dtype=int) for step in steps)
    if any(
        not set(coarse.tolist()).issubset(set(fine.tolist()))
        for coarse, fine in zip(grids, grids[1:])
    ):
        raise ValueError("temporal-refinement grids are not nested")
    return grids


def scaled_window_integral(
    rows_by_source: Mapping[int, Mapping[str, Any]],
    sources: np.ndarray,
    field: str,
    *,
    start_source: int,
    end_source: int,
    source_horizon_days: float,
) -> float:
    """Integrate a source-time action density in the window parameterization."""
    sources = np.asarray(sources, dtype=int)
    normalized = (sources - int(start_source)) / float(
        int(end_source) - int(start_source)
    )
    weights = normalized_trapezoid_weights(normalized)
    window_days = (
        int(end_source) - int(start_source)
    ) * float(source_horizon_days) / 180.0
    time_scale = window_days / float(source_horizon_days)
    density = np.asarray(
        [float(rows_by_source[int(source)][field]) for source in sources],
        dtype=np.float64,
    )
    return float(weights @ (time_scale * time_scale * density))


def relative_change(left: float, right: float) -> float:
    """Symmetric relative change for consecutive temporal integrals."""
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(np.float64).tiny
    )


def summarize_temporal_levels(
    rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the predeclared convergence ladder from full-grid local rows."""
    grids = nested_source_grids(
        int(cfg["window_start_source_index"]),
        int(cfg["window_end_source_index"]),
        cfg["source_steps"],
    )
    expected_counts = tuple(int(value) for value in cfg["node_counts"])
    if tuple(len(grid) for grid in grids) != expected_counts:
        raise ValueError("temporal node counts changed")
    by_source = {int(row["source_time_index"]): row for row in rows}
    expected_sources = set(grids[-1].tolist())
    if set(by_source) != expected_sources or len(by_source) != len(rows):
        raise ValueError("full temporal-refinement panel is incomplete")

    levels: list[dict[str, Any]] = []
    for step, sources in zip(cfg["source_steps"], grids, strict=True):
        tangent = scaled_window_integral(
            by_source,
            sources,
            "tangent_action",
            start_source=int(cfg["window_start_source_index"]),
            end_source=int(cfg["window_end_source_index"]),
            source_horizon_days=float(cfg["source_horizon_days"]),
        )
        full = scaled_window_integral(
            by_source,
            sources,
            "direct_action_qr",
            start_source=int(cfg["window_start_source_index"]),
            end_source=int(cfg["window_end_source_index"]),
            source_horizon_days=float(cfg["source_horizon_days"]),
        )
        previous = levels[-1] if levels else None
        levels.append({
            "source_step": int(step),
            "node_count": len(sources),
            "tangent_action": tangent,
            "full_action": full,
            "tangent_relative_change_from_previous": (
                relative_change(previous["tangent_action"], tangent)
                if previous else None
            ),
            "full_relative_change_from_previous": (
                relative_change(previous["full_action"], full)
                if previous else None
            ),
        })

    tolerance = float(cfg["maximum_consecutive_relative_action_change"])
    changes = [
        value
        for level in levels[1:]
        for value in (
            level["tangent_relative_change_from_previous"],
            level["full_relative_change_from_previous"],
        )
    ]
    all_local_valid = all(bool(row["local_valid"]) for row in rows)
    lower_bound_scale = max(
        abs(levels[-1]["tangent_action"]),
        abs(levels[-1]["full_action"]),
        1.0,
    )
    integrated_lower_bound_valid = bool(
        levels[-1]["tangent_action"]
        <= levels[-1]["full_action"]
        + float(cfg["tangent_full_relative_tolerance"]) * lower_bound_scale
    )
    convergence_valid = bool(all(float(value) <= tolerance for value in changes))
    certified = bool(
        all_local_valid and integrated_lower_bound_valid and convergence_valid
    )
    return {
        "levels": levels,
        "all_local_valid": all_local_valid,
        "local_valid_count": sum(bool(row["local_valid"]) for row in rows),
        "local_case_count": len(rows),
        "integrated_tangent_lower_bound_valid": integrated_lower_bound_valid,
        "maximum_consecutive_relative_action_change_observed": max(changes),
        "convergence_valid": convergence_valid,
        "temporal_quadrature_refinement_certified": certified,
        "production_authorized": bool(
            certified and cfg["production_authorized_if_certified"]
        ),
    }


__all__ = [
    "nested_source_grids",
    "relative_change",
    "scaled_window_integral",
    "summarize_temporal_levels",
]
