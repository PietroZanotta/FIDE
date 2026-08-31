"""Compare frozen empirical-source and matched-KDE reference ensembles.

This is deterministic post-processing only.  It does not train a flow, rerun
the active-nematic solver, optimize a design, or access validation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import jax.numpy as jnp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = SCRIPT_DIR / "outputs" / "reference_t5_t15_v1"
DEFAULT_CANDIDATE = SCRIPT_DIR / "outputs" / "reference_t5_t15_matched_kde_v1"

from mfsi.cache import file_sha256
from mfsi.io import write_json
from domain import make_run_split
from run import reference_training_source, split_config
from risk import (
    PeriodicHistogramGrid,
    histogram_mass,
    multiscale_periodic_kernel_fft,
    periodic_grid_mmd2,
)
from unbalanced_state import TwoSpeciesDefectBank
from visualize_reference_endpoints import _prepare_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _audit_rows(directory: Path) -> dict[tuple[int, str], dict[str, Any]]:
    audit_path = directory / "reference_endpoint_audit.json"
    if audit_path.is_file():
        payload = _load_json(audit_path)
        return {
            (int(row["reference_seed"]), str(row["species"])): row
            for row in payload["rows"]
        }

    cfg = _load_json(directory / "effective_config.json")
    bank = TwoSpeciesDefectBank.load(directory / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    periods = (float(bank.box_size), float(bank.box_size), 2.0 * np.pi)
    grid = PeriodicHistogramGrid(
        periods, tuple(int(value) for value in cfg["law"]["grid_shape"])
    )
    kernel = multiscale_periodic_kernel_fft(
        grid, jnp.asarray(cfg["law"]["mmd_bandwidths"], dtype=jnp.float64)
    )
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
    truth = {}
    for species in ("plus", "minus"):
        for endpoint, time_index in enumerate((0, len(bank.times) - 1)):
            measure = bank.measure(species, time_index, split.train)
            truth[(species, endpoint)] = histogram_mass(
                jnp.asarray(measure.states),
                jnp.asarray(
                    measure.normalized_probabilities(minimum_mass=minimum_mass)
                ),
                grid,
            )

    rows = []
    for seed_dir in sorted(directory.glob("reference_seed_*")):
        seed = int(seed_dir.name.removeprefix("reference_seed_"))
        for species in ("plus", "minus"):
            with np.load(
                seed_dir / f"{species}_reference_bank.npz", allow_pickle=False
            ) as saved:
                nodes = np.asarray(saved["nodes"], dtype=np.float64)
                weights = np.asarray(saved["weights"], dtype=np.float64)
            histograms = [
                histogram_mass(
                    jnp.asarray(nodes[index]), jnp.asarray(weights[index]), grid
                )
                for index in (0, len(nodes) - 1)
            ]
            rows.append(
                {
                    "reference_seed": seed,
                    "species": species,
                    "initial_shape_mmd2": float(
                        periodic_grid_mmd2(
                            histograms[0], truth[(species, 0)], kernel
                        )
                    ),
                    "target_shape_mmd2": float(
                        periodic_grid_mmd2(
                            histograms[1], truth[(species, 1)], kernel
                        )
                    ),
                }
            )
    return {
        (int(row["reference_seed"]), str(row["species"])): row
        for row in rows
    }


def _overlap_rows(directory: Path) -> dict[tuple[int, str], dict[str, Any]]:
    data = _prepare_data(
        SimpleNamespace(
            frozen_inputs=directory,
            spatial_bins=96,
            orientation_bins=96,
        )
    )
    return {
        (int(reference["seed"]), species): {
            "initial": reference["species"][species][0],
            "target": reference["species"][species][1],
        }
        for reference in data["references"]
        for species in ("plus", "minus")
    }


def _final_losses(directory: Path) -> dict[tuple[int, str], float]:
    result = {}
    for seed_dir in sorted(directory.glob("reference_seed_*")):
        seed = int(seed_dir.name.removeprefix("reference_seed_"))
        for species in ("plus", "minus"):
            payload = _load_json(
                seed_dir / f"{species}_reference_training_history.json"
            )
            result[(seed, species)] = float(
                payload["history"][-1]["conditional_fm_loss"]
            )
    return result


def _initial_nodes_identical(
    baseline: Path, candidate: Path, keys: list[tuple[int, str]]
) -> dict[str, bool]:
    result = {}
    for seed, species in keys:
        relative = Path(f"reference_seed_{seed}") / f"{species}_reference_bank.npz"
        with np.load(baseline / relative, allow_pickle=False) as old:
            old_initial = np.asarray(old["nodes"][0])
        with np.load(candidate / relative, allow_pickle=False) as new:
            new_initial = np.asarray(new["nodes"][0])
        result[f"{seed}:{species}"] = bool(np.array_equal(old_initial, new_initial))
    return result


def _training_endpoint_sample_identity(
    baseline: Path, candidate: Path
) -> dict[str, dict[str, bool]]:
    old_cfg = _load_json(baseline / "effective_config.json")
    new_cfg = _load_json(candidate / "effective_config.json")
    old_bank = TwoSpeciesDefectBank.load(baseline / "two_species_defect_bank.npz")
    new_bank = TwoSpeciesDefectBank.load(candidate / "two_species_defect_bank.npz")
    old_runs = make_run_split(split_config(old_cfg)).train
    new_runs = make_run_split(split_config(new_cfg)).train
    old_periods = np.asarray(
        [old_bank.box_size, old_bank.box_size, 2.0 * np.pi], dtype=np.float64
    )
    new_periods = np.asarray(
        [new_bank.box_size, new_bank.box_size, 2.0 * np.pi], dtype=np.float64
    )
    result = {}
    for species in ("plus", "minus"):
        old = reference_training_source(
            old_cfg, old_bank, old_runs, species, old_periods
        )
        new = reference_training_source(
            new_cfg, new_bank, new_runs, species, new_periods
        )
        result[species] = {
            "initial_samples_bitwise_equal": bool(
                np.array_equal(np.asarray(old.x0), np.asarray(new.x0))
            ),
            "target_samples_bitwise_equal": bool(
                np.array_equal(np.asarray(old.x1), np.asarray(new.x1))
            ),
        }
    return result


def compare(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline = baseline.expanduser().resolve()
    candidate = candidate.expanduser().resolve()
    old_audit = _audit_rows(baseline)
    new_audit = _audit_rows(candidate)
    old_overlap = _overlap_rows(baseline)
    new_overlap = _overlap_rows(candidate)
    old_loss = _final_losses(baseline)
    new_loss = _final_losses(candidate)
    keys = sorted(old_audit)
    if keys != sorted(new_audit) or keys != sorted(old_overlap) or keys != sorted(new_overlap):
        raise RuntimeError("baseline and candidate seed/species panels do not match")

    paired = []
    for seed, species in keys:
        old_mmd = float(old_audit[(seed, species)]["target_shape_mmd2"])
        new_mmd = float(new_audit[(seed, species)]["target_shape_mmd2"])
        paired.append(
            {
                "reference_seed": seed,
                "species": species,
                "baseline_target_mmd2": old_mmd,
                "candidate_target_mmd2": new_mmd,
                "target_mmd2_fractional_reduction": 1.0 - new_mmd / old_mmd,
                "baseline_target_spatial_overlap": float(
                    old_overlap[(seed, species)]["target"]["spatial_overlap"]
                ),
                "candidate_target_spatial_overlap": float(
                    new_overlap[(seed, species)]["target"]["spatial_overlap"]
                ),
                "baseline_target_orientation_overlap": float(
                    old_overlap[(seed, species)]["target"]["orientation_overlap"]
                ),
                "candidate_target_orientation_overlap": float(
                    new_overlap[(seed, species)]["target"]["orientation_overlap"]
                ),
                "baseline_final_cfm_loss": old_loss[(seed, species)],
                "candidate_final_cfm_loss": new_loss[(seed, species)],
            }
        )

    summaries = {}
    for species in ("plus", "minus", "combined"):
        rows = paired if species == "combined" else [
            row for row in paired if row["species"] == species
        ]
        summaries[species] = {
            metric: _summary([float(row[metric]) for row in rows])
            for metric in (
                "baseline_target_mmd2",
                "candidate_target_mmd2",
                "target_mmd2_fractional_reduction",
                "baseline_target_spatial_overlap",
                "candidate_target_spatial_overlap",
                "baseline_target_orientation_overlap",
                "candidate_target_orientation_overlap",
                "baseline_final_cfm_loss",
                "candidate_final_cfm_loss",
            )
        }

    initial_identity = _initial_nodes_identical(baseline, candidate, keys)
    training_sample_identity = _training_endpoint_sample_identity(
        baseline, candidate
    )
    bank_files = ("physical_bank.npz", "two_species_defect_bank.npz")
    bank_identity = {
        name: {
            "baseline_sha256": file_sha256(baseline / name),
            "candidate_sha256": file_sha256(candidate / name),
        }
        for name in bank_files
    }
    for row in bank_identity.values():
        row["identical"] = row["baseline_sha256"] == row["candidate_sha256"]

    return {
        "schema_version": 1,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "scientific_change": {
            "baseline_training_initial_density": "empirical",
            "candidate_training_initial_density": "periodic_kde",
            "rollout_initial_density": "periodic_kde in both arms",
            "target_endpoint_samples": "same declared endpoint sampler and seed",
            "architecture_optimizer_loss_seeds": "unchanged",
            "validation_accessed": False,
        },
        "bank_identity": bank_identity,
        "rollout_initial_nodes_bitwise_equal": initial_identity,
        "all_rollout_initial_nodes_bitwise_equal": all(initial_identity.values()),
        "training_endpoint_sample_identity": training_sample_identity,
        "all_training_target_samples_bitwise_equal": all(
            row["target_samples_bitwise_equal"]
            for row in training_sample_identity.values()
        ),
        "all_pairs_improve_target_mmd2": all(
            row["candidate_target_mmd2"] < row["baseline_target_mmd2"]
            for row in paired
        ),
        "paired_rows": paired,
        "summaries": summaries,
        "metric_notes": {
            "mmd2": "production periodic 3-D multiscale histogram MMD squared; lower is better",
            "overlap": "smoothed marginal histogram intersection; higher is better",
            "cfm_loss": (
                "raw final training objective; not directly comparable across arms "
                "because KDE jitter changes the bridge-displacement distribution"
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    baseline = args.baseline.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else candidate / "initial_density_ablation_comparison.json"
    )
    payload = compare(baseline, candidate)
    write_json(output, payload)
    combined = payload["summaries"]["combined"]
    print(f"saved {output}")
    print(
        "target MMD2 mean: "
        f"{combined['baseline_target_mmd2']['mean']:.9f} -> "
        f"{combined['candidate_target_mmd2']['mean']:.9f}"
    )
    print(
        "mean paired target MMD2 reduction: "
        f"{100.0 * combined['target_mmd2_fractional_reduction']['mean']:.2f}%"
    )
    print(
        "all six pairs improve: "
        f"{payload['all_pairs_improve_target_mmd2']}"
    )
    print(
        "all rollout initial particles bitwise equal: "
        f"{payload['all_rollout_initial_nodes_bitwise_equal']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
