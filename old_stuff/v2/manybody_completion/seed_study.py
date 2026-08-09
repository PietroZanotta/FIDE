"""Multi-seed aggregation for paired DiffPOP comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from .statistics import paired_effect, percentile_interval
from .uq import aggregate_seed_higher_order_uq


SCALAR_ENDPOINTS = (
    "moment_error",
    "ess_fraction",
    "mode_probability_error",
    "hidden_energy_score",
    "hidden_energy_distance",
    "joint_total_variation",
)


def aggregate_reports(reports: list[dict], report_paths: list[str] | None = None) -> dict:
    if not reports:
        raise ValueError("no reports supplied")
    methods = list(reports[0]["methods"])
    aggregate: dict = {
        "seed_count": len(reports),
        "seeds": [int(report["metadata"]["seed"]) for report in reports],
        "authoritative_reports": report_paths or [],
        "methods": {},
    }
    for method in methods:
        aggregate["methods"][method] = {}
        for endpoint in SCALAR_ENDPOINTS:
            values = [float(report["methods"][method][endpoint]) for report in reports]
            aggregate["methods"][method][endpoint] = percentile_interval(values)
        uq = [report["methods"][method]["higher_order_conditional_uq"] for report in reports]
        aggregate["methods"][method]["higher_order_variance_decomposition"] = (
            aggregate_seed_higher_order_uq(uq)
        )

    full_ess = [r["methods"]["Full-E2E"]["ess_fraction"] for r in reports]
    stop_ess = [r["methods"]["Calibrated-StopGrad"]["ess_fraction"] for r in reports]
    full_score = [r["methods"]["Full-E2E"]["hidden_energy_score"] for r in reports]
    stop_score = [r["methods"]["Calibrated-StopGrad"]["hidden_energy_score"] for r in reports]
    aggregate["paired_full_minus_stopgrad"] = {
        "ess_fraction": paired_effect(full_ess, stop_ess),
        "hidden_energy_score": paired_effect(full_score, stop_score),
    }
    return aggregate


def load_reports(paths: list[str | Path]) -> list[dict]:
    reports = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            reports.append(json.load(handle))
    return reports
