"""Shared reporting helpers for exact Tangent/Full hierarchy audits.

Experiment-local wrappers are responsible only for loading their frozen saved
artifacts and constructing the existing authoritative experiment evaluator.  All
action mathematics remains in each experiment's ``evaluate_trials_exact`` path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


METHODS = ("law", "tangent", "full")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_key(geometry: Any) -> str:
    values = np.asarray(geometry, dtype=np.float64).reshape(-1)
    return json.dumps(values.tolist(), separators=(",", ":"))


def load_pareto_candidates(
    pareto_dir: Path,
    *,
    selection_key: Callable[[dict[str, Any], str], Any],
) -> list[dict[str, Any]]:
    pareto_dir = Path(pareto_dir).expanduser().resolve()
    rows = json.loads((pareto_dir / "pareto.json").read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for pareto_row in sorted(rows, key=lambda row: float(row["risk_allowance_percent"])):
        allowance = float(pareto_row["risk_allowance_percent"])
        tag = f"risk_{f'{allowance:g}'.replace('.', 'p').replace('-', 'm')}pct"
        result_path = pareto_dir / tag / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        reported = _reported_candidate_metrics(result_path)
        for method in METHODS:
            geometry = selection_key(result, method)
            candidate_report = reported.get(method, {})
            candidates.append(
                {
                    "allowance_percent": allowance,
                    "method": method,
                    "geometry": np.asarray(geometry, dtype=np.float64).tolist(),
                    "result_path": str(result_path),
                    "reported_A_tan": _finite_or_none(
                        candidate_report.get("tangent_action_selection")
                    ),
                    "reported_A_full": _finite_or_none(
                        candidate_report.get("full_action_selection")
                    ),
                }
            )
    return candidates


def _reported_candidate_metrics(result_path: Path) -> dict[str, dict[str, str]]:
    path = result_path.with_name("result.candidate_summary.csv")
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("design")): row for row in csv.DictReader(handle)}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def audit_candidates(
    candidates: list[dict[str, Any]],
    *,
    evaluate: Callable[[Any, str], list[dict[str, Any]]],
    tolerance: float,
    time_grid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Evaluate each unique geometry and aggregate raw hierarchy diagnostics."""
    tolerance = float(tolerance)
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("hierarchy tolerance must be finite and nonnegative")
    time_grid = np.asarray(time_grid, dtype=np.float64)
    evaluations: dict[str, list[dict[str, Any]]] = {}
    aggregate: list[dict[str, Any]] = []

    for candidate in candidates:
        key = geometry_key(candidate["geometry"])
        if key not in evaluations:
            evaluations[key] = evaluate(candidate["geometry"], key)
        trial_rows = evaluations[key]
        valid_rows = [
            row
            for row in trial_rows
            if bool(row.get("valid"))
            and _finite_or_none(row.get("tangent_action")) is not None
            and _finite_or_none(row.get("full_action")) is not None
        ]
        invalid_count = len(trial_rows) - len(valid_rows)
        tangent = np.asarray([row["tangent_action"] for row in valid_rows], dtype=np.float64)
        full = np.asarray([row["full_action"] for row in valid_rows], dtype=np.float64)
        trial_raw = tangent - full

        time_raw_values: list[float] = []
        for row in valid_rows:
            tangent_time = np.asarray(row.get("tangent_action_by_time"), dtype=np.float64)
            full_time = np.asarray(row.get("full_action_by_time"), dtype=np.float64)
            if tangent_time.shape != time_grid.shape or full_time.shape != time_grid.shape:
                raise RuntimeError(
                    "authoritative evaluator did not return one Tangent/Full value "
                    f"per time node for geometry {key}"
                )
            raw = tangent_time - full_time
            if not np.all(np.isfinite(raw)):
                raise RuntimeError(f"non-finite time-level hierarchy values for geometry {key}")
            time_raw_values.extend(raw.tolist())
        time_raw = np.asarray(time_raw_values, dtype=np.float64)

        if not len(valid_rows):
            a_tangent = a_full = a_hidden = gamma = aggregate_raw = float("nan")
        else:
            a_tangent = float(np.mean(tangent))
            a_full = float(np.mean(full))
            a_hidden = float(a_full - a_tangent)
            gamma = float(1.0 - a_tangent / a_full) if a_full != 0.0 else float("nan")
            aggregate_raw = float(a_tangent - a_full)

        trial_max = float(np.max(trial_raw)) if trial_raw.size else float("nan")
        time_max = float(np.max(time_raw)) if time_raw.size else float("nan")
        finite_maxima = [
            value for value in (aggregate_raw, trial_max, time_max) if math.isfinite(value)
        ]
        max_raw = max(finite_maxima) if finite_maxima else float("nan")
        aggregate_violations = int(math.isfinite(aggregate_raw) and aggregate_raw > tolerance)
        trial_violations = int(np.count_nonzero(trial_raw > tolerance))
        time_violations = int(np.count_nonzero(time_raw > tolerance))
        passes = bool(
            invalid_count == 0
            and aggregate_violations == 0
            and trial_violations == 0
            and time_violations == 0
        )

        reported_tangent = candidate.get("reported_A_tan")
        reported_full = candidate.get("reported_A_full")
        tangent_delta = (
            float(a_tangent - reported_tangent)
            if reported_tangent is not None and math.isfinite(a_tangent)
            else None
        )
        full_delta = (
            float(a_full - reported_full)
            if reported_full is not None and math.isfinite(a_full)
            else None
        )
        aggregate.append(
            {
                "allowance_percent": candidate["allowance_percent"],
                "method": candidate["method"],
                "A_tan": a_tangent,
                "A_full": a_full,
                "A_hid": a_hidden,
                "Gamma": gamma,
                "hierarchy_tolerance": tolerance,
                "aggregate_raw_violation_A_tan_minus_A_full": aggregate_raw,
                "aggregate_violation_count": aggregate_violations,
                "trial_count": len(trial_rows),
                "invalid_trial_count": invalid_count,
                "trial_max_raw_violation": trial_max,
                "trial_violation_count": trial_violations,
                "time_trial_comparison_count": int(time_raw.size),
                "time_trial_max_raw_violation": time_max,
                "time_trial_violation_count": time_violations,
                "maximum_raw_violation_all_levels": max_raw,
                "reported_A_tan": reported_tangent,
                "reported_A_full": reported_full,
                "audit_minus_reported_A_tan": tangent_delta,
                "audit_minus_reported_A_full": full_delta,
                "all_hierarchy_checks_pass": passes,
                "geometry": json.dumps(candidate["geometry"], separators=(",", ":")),
                "result_path": candidate["result_path"],
                "evaluation_key": key,
            }
        )
    return aggregate, evaluations


def build_summary(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    tolerance: float,
    selection_bank_path: Path,
    time_grid: np.ndarray,
    evaluator_description: str,
) -> dict[str, Any]:
    maxima = [
        float(row["maximum_raw_violation_all_levels"])
        for row in rows
        if _finite_or_none(row.get("maximum_raw_violation_all_levels")) is not None
    ]
    tangent_report_deltas = [
        abs(float(row["audit_minus_reported_A_tan"]))
        for row in rows
        if _finite_or_none(row.get("audit_minus_reported_A_tan")) is not None
    ]
    full_report_deltas = [
        abs(float(row["audit_minus_reported_A_full"]))
        for row in rows
        if _finite_or_none(row.get("audit_minus_reported_A_full")) is not None
    ]
    return {
        "schema_version": 1,
        "experiment": experiment,
        "candidate_count": len(rows),
        "allowance_count": len({float(row["allowance_percent"]) for row in rows}),
        "designs": list(METHODS),
        "authoritative_evaluator": evaluator_description,
        "selection_bank": str(Path(selection_bank_path).resolve()),
        "selection_bank_sha256": file_sha256(selection_bank_path),
        "time_grid": np.asarray(time_grid, dtype=np.float64).tolist(),
        "hierarchy_tolerance": float(tolerance),
        "maximum_raw_violation_all_levels": max(maxima) if maxima else None,
        "aggregate_violation_count": sum(int(row["aggregate_violation_count"]) for row in rows),
        "trial_violation_count": sum(int(row["trial_violation_count"]) for row in rows),
        "time_trial_violation_count": sum(
            int(row["time_trial_violation_count"]) for row in rows
        ),
        "invalid_trial_count": sum(int(row["invalid_trial_count"]) for row in rows),
        "maximum_absolute_audit_minus_reported_A_tan": (
            max(tangent_report_deltas) if tangent_report_deltas else None
        ),
        "maximum_absolute_audit_minus_reported_A_full": (
            max(full_report_deltas) if full_report_deltas else None
        ),
        "every_final_candidate_passes": all(
            bool(row["all_hierarchy_checks_pass"]) for row in rows
        ),
        "raw_violation_definition": "A_tan - A_full; values are not clipped",
    }


def save_outputs(
    rows: list[dict[str, Any]],
    evaluations: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    *,
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "action_decomposition_audit.csv"
    summary_path = output_dir / "action_decomposition_audit_summary.json"
    evaluations_path = output_dir / "action_decomposition_evaluations.json"
    markdown_path = output_dir / "action_decomposition_audit.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    evaluations_path.write_text(
        json.dumps({"schema_version": 1, "evaluations": evaluations}, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(rows, summary), encoding="utf-8")
    return [csv_path, markdown_path, summary_path, evaluations_path]


def _fmt(value: Any, spec: str = ".10g") -> str:
    number = _finite_or_none(value)
    return format(number, spec) if number is not None else "—"


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    status = "PASS" if summary["every_final_candidate_passes"] else "FAIL"
    lines = [
        "# Exact Tangent/Full action-decomposition audit",
        "",
        f"**Overall status: {status}.**",
        "",
        "This is a post-selection audit of the saved final Law, Tangent, and Full candidates. "
        "Each unique geometry was evaluated by the experiment's existing authoritative exact "
        "evaluator on the frozen action-selection bank. Tangent and Full were computed in the "
        "same evaluator call from the same reconstructed targets and projected weights.",
        "",
        f"The hierarchy tolerance is `{summary['hierarchy_tolerance']:.8g}`. Raw violations are "
        "defined as `A_tan - A_full` and are never clipped; negative values indicate hierarchy "
        "slack.",
        "",
        "## Concise summary",
        "",
        f"- Maximum raw violation over aggregate, trial, and time/trial levels: "
        f"`{_fmt(summary['maximum_raw_violation_all_levels'], '.12g')}`.",
        f"- Aggregate violations: `{summary['aggregate_violation_count']}`.",
        f"- Trial-level violations: `{summary['trial_violation_count']}`.",
        f"- Time/trial-level violations: `{summary['time_trial_violation_count']}`.",
        f"- Invalid exact-evaluator trials: `{summary['invalid_trial_count']}`.",
        f"- Maximum absolute audited-minus-reported Tangent action: "
        f"`{_fmt(summary['maximum_absolute_audit_minus_reported_A_tan'], '.12g')}`.",
        f"- Maximum absolute audited-minus-reported Full action: "
        f"`{_fmt(summary['maximum_absolute_audit_minus_reported_A_full'], '.12g')}`.",
        f"- Every saved final candidate passes: "
        f"`{str(summary['every_final_candidate_passes']).lower()}`.",
        f"- Frozen selection bank SHA-256: `{summary['selection_bank_sha256']}`.",
        "",
        "## Candidate table",
        "",
        "| Allow. | Design | A_tan | A_full | A_hid | Gamma | max raw violation | trial viol. | time/trial viol. | Pass |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['allowance_percent']):g}% | {str(row['method']).title()} | "
            f"{_fmt(row['A_tan'])} | {_fmt(row['A_full'])} | {_fmt(row['A_hid'])} | "
            f"{_fmt(row['Gamma'])} | {_fmt(row['maximum_raw_violation_all_levels'], '.4e')} | "
            f"{row['trial_violation_count']} | {row['time_trial_violation_count']} | "
            f"{'yes' if row['all_hierarchy_checks_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Definitions and checks",
            "",
            "```text",
            "A_hid = A_full - A_tan",
            "Gamma = 1 - A_tan / A_full",
            "raw hierarchy violation = A_tan - A_full",
            "violation iff raw hierarchy violation > configured tolerance",
            "```",
            "",
            "The CSV retains full-precision aggregates, raw signed violations at every reported "
            "level, reported-vs-audited action deltas, geometry, and source-result paths. The "
            "evaluation JSON retains the per-trial and per-time values returned by the exact "
            "evaluator for reproducibility.",
            "",
        ]
    )
    return "\n".join(lines)
