"""Aggregation of independently trained scientific-comparison seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .statistics import bootstrap_mean_interval
from .uq import aggregate_seed_higher_order_uq


def _to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy report values to JSON-native containers."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def aggregate_comparison_seed_reports(
    report_paths: list[str | Path],
    *,
    seed: int,
    num_resamples: int,
) -> dict[str, Any]:
    """Aggregate seed-level effects and higher-order uncertainty components."""
    if len(report_paths) < 2:
        raise ValueError("at least two independently trained seed reports are required")
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths
    ]
    methods = tuple(reports[0]["methods"])
    if any(tuple(report["methods"]) != methods for report in reports[1:]):
        raise ValueError("seed reports contain different method sets")

    effect_names = tuple(reports[0]["primary_learned_method_comparison"])
    effects = {
        name: np.asarray(
            [
                report["primary_learned_method_comparison"][name]["estimate"]
                for report in reports
            ],
            dtype=np.float64,
        )
        for name in effect_names
    }
    effect_intervals = {
        name: bootstrap_mean_interval(
            values,
            seed=seed + offset,
            num_resamples=num_resamples,
        ).as_dict()
        for offset, (name, values) in enumerate(effects.items())
    }

    higher_order: dict[str, Any] = {}
    for method in methods:
        higher_order[method] = {}
        for stage in ("raw", "repaired"):
            summaries = [
                report["methods"][method]["results"][stage][
                    "higher_order_conditional_uq"
                ]
                for report in reports
            ]
            higher_order[method][stage] = aggregate_seed_higher_order_uq(summaries)

    return _to_jsonable(
        {
            "schema_version": 1,
            "status": "multi-seed aggregate",
            "num_training_seeds": len(reports),
            "seed_report_paths": [str(path) for path in report_paths],
            "primary_effects_across_training_seeds": effect_intervals,
            "higher_order_uncertainty_decomposition": higher_order,
            "inference_note": (
                "Training seeds, not generated replicas, are the inferential units "
                "for final method-level claims."
            ),
        }
    )
