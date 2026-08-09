"""Multi-seed aggregation for paired flow-matching and DiffPOP comparisons."""

from __future__ import annotations

import json
import numpy as np
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
    "particle_mode_probability_error",
    "particle_hidden_energy_score",
    "particle_joint_total_variation",
    "model_moment_error",
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
        endpoint_fallbacks = {
            "particle_mode_probability_error": "mode_probability_error",
            "particle_hidden_energy_score": "hidden_energy_score",
            "particle_joint_total_variation": "joint_total_variation",
            "model_moment_error": "moment_error",
        }
        for endpoint in SCALAR_ENDPOINTS:
            fallback = endpoint_fallbacks.get(endpoint, endpoint)
            values = [
                float(report["methods"][method].get(endpoint, report["methods"][method][fallback]))
                for report in reports
            ]
            aggregate["methods"][method][endpoint] = percentile_interval(values)
        diagnostic_fallbacks = {
            "sampler_calls": 0.0,
            "warm_start_absolute_error": float("nan"),
            "proposal_expected_ess_fraction": float("nan"),
        }
        aggregate["methods"][method]["finite_budget_diagnostics"] = {}
        for diagnostic, fallback_value in diagnostic_fallbacks.items():
            values = [
                float(report["methods"][method].get("diagnostics", {}).get(diagnostic, fallback_value))
                for report in reports
            ]
            finite_values = [value for value in values if np.isfinite(value)]
            aggregate["methods"][method]["finite_budget_diagnostics"][diagnostic] = (
                percentile_interval(finite_values) if finite_values else None
            )
        uq = [report["methods"][method]["higher_order_conditional_uq"] for report in reports]
        aggregate["methods"][method]["higher_order_variance_decomposition"] = (
            aggregate_seed_higher_order_uq(uq)
        )

    full_name = "Flow-DiffPOP-FullE2E"
    stop_name = "Flow-DiffPOP-StopGrad"
    posthoc_name = "Flow-DiffPOP-PostHoc"
    direct_name = "Direct-Conditional-Flow"
    synergy_name = "Flow-DiffPOP-SynergyE2E"

    def paired(left: str, right: str) -> dict:
        return {
            endpoint: paired_effect(
                [r["methods"][left][endpoint] for r in reports],
                [r["methods"][right][endpoint] for r in reports],
            )
            for endpoint in (
                "moment_error",
                "ess_fraction",
                "mode_probability_error",
                "hidden_energy_score",
                "joint_total_variation",
            )
        }

    aggregate["paired_full_minus_direct_flow"] = paired(full_name, direct_name)
    aggregate["paired_posthoc_minus_direct_flow"] = paired(posthoc_name, direct_name)
    aggregate["paired_full_minus_stopgrad"] = paired(full_name, stop_name)
    aggregate["paired_full_minus_posthoc"] = paired(full_name, posthoc_name)
    if synergy_name in methods:
        aggregate["paired_synergy_minus_full"] = paired(synergy_name, full_name)
        aggregate["paired_synergy_minus_posthoc"] = paired(synergy_name, posthoc_name)
        aggregate["paired_synergy_minus_direct_flow"] = paired(synergy_name, direct_name)
    aggregate["seed_decisions"] = [report["decision_summary"] for report in reports]
    aggregate["diffpop_posthoc_supported_seed_fraction"] = sum(
        bool(report["decision_summary"]["diffpop_posthoc_supported_in_this_run"])
        for report in reports
    ) / len(reports)
    aggregate["diffpop_full_supported_seed_fraction"] = sum(
        bool(report["decision_summary"]["diffpop_full_supported_in_this_run"])
        for report in reports
    ) / len(reports)
    if synergy_name in methods:
        aggregate["diffpop_synergy_supported_seed_fraction"] = sum(
            bool(report["decision_summary"]["diffpop_synergy_supported_in_this_run"])
            for report in reports
        ) / len(reports)
    return aggregate


def load_reports(paths: list[str | Path]) -> list[dict]:
    reports = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            reports.append(json.load(handle))
    return reports
