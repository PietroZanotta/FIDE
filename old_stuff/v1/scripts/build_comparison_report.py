#!/usr/bin/env python3
"""Build a concise human-readable report from the comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _number(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    lines = [
        "# Homometric scientific comparison",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Information-budget tracks",
        "",
        "Observation-only methods and population-informed learned methods are "
        "reported together but not treated as having equal information.",
        "",
        "## Method summary",
        "",
        "| Method | Stage | Pair error | Repair RMS | B fraction | Far fraction | "
        "Angular MMD² | H energy score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_result in report["methods"].items():
        repair = method_result["results"]["repair_correction"]["estimate"]
        for stage in ("raw", "repaired"):
            result = method_result["results"][stage]
            metrics = result["metrics"]
            uq = result["higher_order_conditional_uq"]
            lines.append(
                "| {method} | {stage} | {pair} | {repair} | {mode_b} | {far} | "
                "{mmd} | {energy} |".format(
                    method=method,
                    stage=stage,
                    pair=_number(metrics["pair_error"]["estimate"]),
                    repair=_number(repair),
                    mode_b=_number(metrics["mode_b_fraction"]["estimate"]),
                    far=_number(metrics["far_fraction"]["estimate"]),
                    mmd=_number(metrics["angular_mmd2"]),
                    energy=_number(uq["multivariate_energy_score"]),
                )
            )
    lines.extend(
        [
            "",
            "## Primary learned-method comparison",
            "",
        ]
    )
    for name, interval in report["primary_learned_method_comparison"].items():
        lines.append(
            f"- `{name}`: {_number(interval['estimate'])} "
            f"(95% CI [{_number(interval['lower'])}, "
            f"{_number(interval['upper'])}])"
        )
    lines.extend(
        [
            "",
            "## Decision gates",
            "",
        ]
    )
    for name, value in report["decision_summary"].items():
        lines.append(f"- **{name}:** {value}")
    lines.extend(
        [
            "",
            "## Higher-order conditional UQ",
            "",
            "Each method includes predictive intervals for every held-out angular "
            "coefficient, reference coverage, interval scores, a multivariate "
            "energy score, A/B/Far probability intervals, and mode-calibration "
            "error. Across independent training seeds, the registered runner "
            "adds aleatoric/epistemic variance decomposition.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
