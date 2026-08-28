"""Robust physical-view aggregation for active-nematic design selection.

Observation trials quantify finite-particle and detector noise conditional on a
physical population.  This module adds the missing outer layer: deterministic
views of the stochastic active-nematic realizations.  Validation views use the
same construction on a disjoint run split and are never exposed to selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PhysicalView:
    label: str
    reference_seed: int
    run_indices: tuple[int, ...]
    experiment: Any


def leave_one_fold_out_views(
    run_indices: Sequence[int], *, count: int, seed: int
) -> list[tuple[int, ...]]:
    """Create deterministic overlapping views without touching held-out runs."""
    runs = np.asarray(run_indices, dtype=np.int64)
    count = int(count)
    if runs.ndim != 1 or len(runs) < 2:
        raise ValueError("robust views require at least two physical runs")
    if count < 2 or count > len(runs):
        raise ValueError("view count must lie between two and the run count")
    shuffled = np.random.default_rng(int(seed)).permutation(runs)
    folds = [row for row in np.array_split(shuffled, count) if len(row)]
    views = []
    for held_out in folds:
        selected = tuple(
            int(value) for value in shuffled if value not in set(held_out.tolist())
        )
        if not selected:
            raise ValueError("a robust physical view cannot be empty")
        views.append(selected)
    return views


def _aggregate(values, *, objective: str, quantile: float):
    values = jnp.asarray(values, dtype=jnp.float64)
    if objective == "max":
        return jnp.max(values)
    if objective == "mean":
        return jnp.mean(values)
    if objective == "upper_quantile":
        return jnp.quantile(values, float(quantile))
    raise ValueError(f"unknown robust objective {objective!r}")


class RobustPhysicalViewExperiment:
    """Expose several physical/reference views through the experiment API."""

    def __init__(self, cfg: dict[str, Any], views: Sequence[PhysicalView]):
        if not views:
            raise ValueError("at least one physical view is required")
        self.cfg = cfg
        self.views = tuple(views)
        first = self.views[0].experiment
        self.sensors = first.sensors
        self.times = first.times
        robust = cfg.get("robust_selection", {})
        self.objective = str(robust.get("objective", "max"))
        self.quantile = float(robust.get("upper_quantile", 0.75))
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("robust_selection.upper_quantile must lie in [0,1]")
        gradient_count = robust.get("gradient_physical_views")
        if gradient_count is None:
            self.gradient_views = self.views
        else:
            gradient_count = int(gradient_count)
            physical_keys = []
            for view in self.views:
                if view.run_indices not in physical_keys:
                    physical_keys.append(view.run_indices)
            if gradient_count < 1 or gradient_count > len(physical_keys):
                raise ValueError(
                    "gradient_physical_views must lie between one and the physical view count"
                )
            retained = set(physical_keys[:gradient_count])
            self.gradient_views = tuple(
                view for view in self.views if view.run_indices in retained
            )

    def with_config(self, cfg: dict[str, Any]) -> "RobustPhysicalViewExperiment":
        rebuilt = []
        for view in self.views:
            exp = view.experiment
            next_exp = type(exp)(
                cfg,
                times=exp.times,
                plus=exp.data["plus"],
                minus=exp.data["minus"],
            )
            for attribute in (
                "skip_unused_tangent_for_full_metric",
            ):
                if hasattr(exp, attribute):
                    setattr(next_exp, attribute, getattr(exp, attribute))
            rebuilt.append(
                PhysicalView(
                    label=view.label,
                    reference_seed=view.reference_seed,
                    run_indices=view.run_indices,
                    experiment=next_exp,
                )
            )
        return RobustPhysicalViewExperiment(cfg, rebuilt)

    def mean_metric_by_view(self, eta, bank, name: str):
        return jnp.stack(
            [
                view.experiment.mean_metric(eta, bank, name)
                for view in self.gradient_views
            ]
        )

    def mean_metric(self, eta, bank, name: str):
        return _aggregate(
            self.mean_metric_by_view(eta, bank, name),
            objective=self.objective,
            quantile=self.quantile,
        )

    def audit_metric(self, eta, bank, name: str) -> dict[str, Any]:
        audits = [view.experiment.audit_metric(eta, bank, name) for view in self.views]
        values = [float(row["value"]) for row in audits]
        return {
            "value": float(
                _aggregate(values, objective=self.objective, quantile=self.quantile)
            ),
            "valid": all(bool(row["valid"]) for row in audits),
            "trials": sum(int(row["trials"]) for row in audits),
            "view_values": values,
            "views": [
                {
                    "label": view.label,
                    "reference_seed": int(view.reference_seed),
                    "run_indices": list(view.run_indices),
                    "audit": audit,
                }
                for view, audit in zip(self.views, audits, strict=True)
            ],
        }

    def view_manifest(self) -> list[dict[str, Any]]:
        gradient_ids = {id(view) for view in self.gradient_views}
        return [
            {
                "label": view.label,
                "reference_seed": int(view.reference_seed),
                "run_indices": list(view.run_indices),
                "used_for_gradient_proxy": id(view) in gradient_ids,
            }
            for view in self.views
        ]
