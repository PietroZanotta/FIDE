"""Artifact writing for one scientific comparison run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from .scientific_comparison import run_scientific_comparison


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def build_markdown_summary(report: dict) -> str:
    target = report["target"]
    rows = []
    for name, method in report["methods"].items():
        rows.append(
            "| {name} | {moment:.5f} | {ess:.3f} | {mode:.4f} | {score:.5f} | {status} |".format(
                name=name,
                moment=method["moment_error"],
                ess=method["ess_fraction"],
                mode=method["mode_probability_error"],
                score=method["hidden_energy_score"],
                status=method["diagnostics"].get("status", "n/a"),
            )
        )
    ambiguity = target["ambiguity_certificate"]
    decision = report["decision_summary"]
    return "\n".join(
        [
            "# DiffPOP scientific comparison summary",
            "",
            f"- Seed: `{report['metadata']['seed']}`",
            f"- Spins: `{report['metadata']['n_spins']}`",
            f"- Joint support atoms: `{report['metadata']['support_size']}`",
            f"- Target pair moment: `{target['target_moment']:.8f}`",
            f"- Pair gap between latent regimes: `{ambiguity['pair_mean_gap']:.3e}`",
            f"- Triplet gap between latent regimes: `{ambiguity['triplet_mean_gap']:.6f}`",
            "",
            "| Method | Moment error | ESS fraction | Mode error | Hidden energy score | Status |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## Exploratory Full-E2E comparison",
            "",
            f"- Full minus StopGrad ESS fraction: `{decision['full_minus_stopgrad_ess_fraction']:.6f}`",
            f"- Full minus StopGrad moment error: `{decision['full_minus_stopgrad_moment_error']:.6f}`",
            f"- Full minus StopGrad hidden energy score: `{decision['full_minus_stopgrad_hidden_energy_score']:.6f}`",
            f"- Calibration gate: `{decision['calibration_gate']}`",
            f"- Mode gate: `{decision['mode_gate']}`",
            "",
            "This bounded run is a mechanism and reproducibility check, not a superiority claim.",
            "",
        ]
    )


def write_run_artifacts(config: dict, output_directory: str | Path) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report, arrays = run_scientific_comparison(config)

    report_path = output / "scientific_comparison_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(output / "scientific_comparison_arrays.npz", **arrays)

    csv_path = output / "scientific_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "moment_error",
                "ess_fraction",
                "mode_probability_error",
                "hidden_energy_score",
                "hidden_energy_distance",
                "joint_total_variation",
                "status",
            ],
        )
        writer.writeheader()
        for name, method in report["methods"].items():
            writer.writerow(
                {
                    "method": name,
                    "moment_error": method["moment_error"],
                    "ess_fraction": method["ess_fraction"],
                    "mode_probability_error": method["mode_probability_error"],
                    "hidden_energy_score": method["hidden_energy_score"],
                    "hidden_energy_distance": method["hidden_energy_distance"],
                    "joint_total_variation": method["joint_total_variation"],
                    "status": method["diagnostics"].get("status", "n/a"),
                }
            )

    (output / "SCIENTIFIC_COMPARISON_SUMMARY.md").write_text(
        build_markdown_summary(report), encoding="utf-8"
    )
    return report
