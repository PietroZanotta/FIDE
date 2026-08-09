"""Artifact writing for one flow-matching and DiffPOP comparison run."""

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
    preferred_order = [
        "MaxEnt-Uniform",
        "Population-Flow",
        "Direct-Conditional-Flow",
        "Flow-One-Shot-Reweight",
        "Flow-DiffPOP-PostHoc",
        "Flow-DiffPOP-StopGrad",
        "Flow-DiffPOP-FullE2E",
        "Flow-DiffPOP-SynergyE2E",
        "Exact-Reference",
    ]
    method_names = [name for name in preferred_order if name in report["methods"]]
    method_names.extend(sorted(set(report["methods"]) - set(method_names)))
    for name in method_names:
        method = report["methods"][name]
        rows.append(
            "| {name} | {moment:.5f} | {ess:.3f} | {calls} | {warm:.3f} | {proposal:.3f} | {mode:.4f} | {particle_mode:.4f} | {score:.5f} | {tv:.4f} | {particle_tv:.4f} | {status} |".format(
                name=name,
                moment=method["moment_error"],
                ess=method["ess_fraction"],
                calls=method["diagnostics"].get("sampler_calls", "—"),
                warm=method["diagnostics"].get("warm_start_absolute_error", float("nan")),
                proposal=method["diagnostics"].get("proposal_expected_ess_fraction", float("nan")),
                mode=method["mode_probability_error"],
                particle_mode=method["particle_mode_probability_error"],
                score=method["hidden_energy_score"],
                tv=method["joint_total_variation"],
                particle_tv=method["particle_joint_total_variation"],
                status=method["diagnostics"].get("status", "n/a"),
            )
        )
    ambiguity = target["ambiguity_certificate"]
    decision = report["decision_summary"]
    return "\n".join(
        [
            "# Flow matching + DiffPOP scientific comparison",
            "",
            f"- Seed: `{report['metadata']['seed']}`",
            f"- Spins: `{report['metadata']['n_spins']}`",
            f"- Joint support atoms: `{report['metadata']['support_size']}`",
            f"- Target pair moment: `{target['target_moment']:.8f}`",
            f"- Pair gap between latent regimes: `{ambiguity['pair_mean_gap']:.3e}`",
            f"- Triplet gap between latent regimes: `{ambiguity['triplet_mean_gap']:.6f}`",
            "",
            "| Method | Fresh moment error | ESS fraction | Sampler calls | Warm error | Proposal ESS | Model mode error | Particle mode error | Hidden energy score | Model TV | Particle TV | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
            "## Exploratory DiffPOP versus flow-matching result",
            "",
            f"- Full DiffPOP minus direct-flow moment error: `{decision['full_minus_direct_flow_moment_error']:.6f}`",
            f"- Full DiffPOP minus direct-flow hidden score: `{decision['full_minus_direct_flow_hidden_energy_score']:.6f}`",
            f"- Full DiffPOP minus direct-flow mode error: `{decision['full_minus_direct_flow_mode_error']:.6f}`",
            f"- Full DiffPOP minus StopGrad ESS fraction: `{decision['full_minus_stopgrad_ess_fraction']:.6f}`",
            f"- Synergy minus Full-E2E ESS fraction: `{decision['synergy_minus_full_ess_fraction']:.6f}`",
            f"- Synergy minus Full-E2E sampler calls: `{decision['synergy_minus_full_sampler_calls']}`",
            f"- Synergy minus Full-E2E hidden score: `{decision['synergy_minus_full_hidden_energy_score']:.6f}`",
            f"- Synergy finite-budget endpoint improved: `{decision['synergy_improves_a_finite_budget_endpoint']}`",
            f"- Post-hoc DiffPOP supported in this run: `{decision['diffpop_posthoc_supported_in_this_run']}`",
            f"- Full DiffPOP supported in this run: `{decision['diffpop_full_supported_in_this_run']}`",
            f"- Synergy DiffPOP supported in this run: `{decision['diffpop_synergy_supported_in_this_run']}`",
            "",
            "The direct conditional flow and the DiffPOP variants have matched network widths. "
            "This bounded run is a mechanism check; seed-level replication is required for a claim.",
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
                "particle_mode_probability_error",
                "particle_hidden_energy_score",
                "particle_joint_total_variation",
                "model_moment_error",
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
                    "particle_mode_probability_error": method["particle_mode_probability_error"],
                    "particle_hidden_energy_score": method["particle_hidden_energy_score"],
                    "particle_joint_total_variation": method["particle_joint_total_variation"],
                    "model_moment_error": method["model_moment_error"],
                    "status": method["diagnostics"].get("status", "n/a"),
                }
            )

    (output / "SCIENTIFIC_COMPARISON_SUMMARY.md").write_text(
        build_markdown_summary(report), encoding="utf-8"
    )
    return report
