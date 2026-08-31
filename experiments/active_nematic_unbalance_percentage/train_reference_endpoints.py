"""Train an isolated endpoint-only active-nematic reference ensemble.

The script derives a prospective configuration from an existing frozen setup,
changes only the physical endpoint interval and experiment name, generates the
complete physical/defect banks, verifies endpoint support, and trains every
declared plus/minus reference pair.  Completed stage artifacts and individual
flow checkpoints are reused on restart.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.cache import file_sha256
from mfsi.config import load_config
from mfsi.io import write_json

from domain import PhysicalBank, make_run_split
from risk import (
    PeriodicHistogramGrid,
    histogram_mass,
    multiscale_periodic_kernel_fft,
    periodic_grid_mmd2,
)
from run import (
    audit_defect_bank,
    build_defect_bank,
    build_physical_bank,
    build_references,
    physics_config,
    reference_seeds,
    split_config,
)
from unbalanced_state import TwoSpeciesDefectBank


DEFAULT_BASE_CONFIG = SCRIPT_DIR / "config_more_training_v2.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "reference_t5_t25_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=float, default=5.0)
    parser.add_argument("--end", type=float, default=25.0)
    parser.add_argument(
        "--time-step",
        type=float,
        help=(
            "retain an inclusive intermediate-time grid for rollout/evaluation; "
            "training still samples only the first and last endpoints"
        ),
    )
    parser.add_argument(
        "--initial-training-density",
        choices=("empirical", "periodic_kde"),
        default="empirical",
        help="initial endpoint law sampled during flow matching",
    )
    parser.add_argument(
        "--reuse-bank-from",
        type=Path,
        help="reuse compatible physical/defect bank files from this output directory",
    )
    return parser.parse_args()


def _time_grid(start: float, end: float, time_step: float | None) -> list[float]:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("physical endpoints must be finite and strictly ordered")
    if time_step is None:
        return [float(start), float(end)]
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time-step must be finite and positive")
    interval_count = int(round((end - start) / time_step))
    if interval_count < 1 or not math.isclose(
        start + interval_count * time_step,
        end,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("time-step must divide the physical interval exactly")
    return np.linspace(start, end, interval_count + 1, dtype=np.float64).tolist()


def _derived_config(
    base_path: Path,
    start: float,
    end: float,
    initial_training_density: str,
    time_step: float | None = None,
) -> dict[str, Any]:
    times = _time_grid(start, end, time_step)
    cfg = copy.deepcopy(load_config(base_path.expanduser().resolve(), smoke=False))
    density_suffix = "_matched_kde" if initial_training_density == "periodic_kde" else ""
    cfg["name"] = (
        f"{cfg['name']}_reference_t{start:g}_t{end:g}{density_suffix}_v1".replace(".", "p")
    )
    cfg["physical_bank"]["save_times"] = times
    cfg["physical_bank"]["population_times"] = times
    if initial_training_density == "periodic_kde":
        cfg["reference_training"]["initial_endpoint_density_model"] = "periodic_kde"
    return cfg


def _require_or_write_config(
    output: Path,
    base_path: Path,
    cfg: dict[str, Any],
    reuse_bank_from: Path | None,
) -> None:
    effective_path = output / "effective_config.json"
    if effective_path.is_file():
        existing = json.loads(effective_path.read_text(encoding="utf-8"))
        if existing != cfg:
            raise RuntimeError(
                "the requested configuration differs from the frozen endpoint study"
            )
    else:
        write_json(effective_path, cfg)
    changed_fields = [
        "name",
        "physical_bank.save_times",
        "physical_bank.population_times",
    ]
    if cfg["reference_training"].get(
        "initial_endpoint_density_model", "empirical"
    ) != "empirical":
        changed_fields.append("reference_training.initial_endpoint_density_model")
    write_json(
        output / "provenance.json",
        {
            "schema_version": 1,
            "base_config": str(base_path.resolve()),
            "base_config_sha256": file_sha256(base_path.resolve()),
            "changed_scientific_fields": changed_fields,
            "reused_bank_from": (
                str(reuse_bank_from.resolve()) if reuse_bank_from is not None else None
            ),
            "reference_training_uses_endpoints_only": True,
            "intermediate_truth_used": False,
            "reference_rollout_time_grid": cfg["physical_bank"]["save_times"],
        },
    )


def _expected_seeds(cfg: dict[str, Any]) -> np.ndarray:
    split = split_config(cfg)
    return (
        int(cfg["seed"])
        + int(cfg["physical_bank"].get("seed_offset", 1001))
        + np.arange(split.total_runs, dtype=np.int64)
    )


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate_physical_bank(cfg: dict[str, Any], path: Path) -> None:
    bank = PhysicalBank.load(path)
    if bank.params != physics_config(cfg):
        raise RuntimeError("saved physical bank has incompatible physics")
    if not np.array_equal(bank.times, cfg["physical_bank"]["save_times"]):
        raise RuntimeError("saved physical bank has incompatible endpoints")
    if not np.array_equal(bank.seeds, _expected_seeds(cfg)):
        raise RuntimeError("saved physical bank has incompatible run seeds")


def _ensure_physical_bank(
    cfg: dict[str, Any], output: Path, reuse_bank_from: Path | None
) -> Path:
    path = output / "physical_bank.npz"
    if not path.is_file():
        source = reuse_bank_from / path.name if reuse_bank_from is not None else None
        if source is not None and source.is_file():
            _validate_physical_bank(cfg, source)
            _link_or_copy(source, path)
        else:
            return build_physical_bank(cfg, output)
    _validate_physical_bank(cfg, path)
    return path


def _validate_defect_bank(cfg: dict[str, Any], path: Path) -> None:
    bank = TwoSpeciesDefectBank.load(path)
    expected_times = np.asarray(cfg["physical_bank"]["population_times"])
    if not np.array_equal(bank.times, expected_times):
        raise RuntimeError("saved defect bank has incompatible endpoints")
    if bank.run_count != len(_expected_seeds(cfg)):
        raise RuntimeError("saved defect bank has incompatible run count")


def _ensure_defect_bank(
    cfg: dict[str, Any], output: Path, reuse_bank_from: Path | None
) -> Path:
    path = output / "two_species_defect_bank.npz"
    if not path.is_file():
        source = reuse_bank_from / path.name if reuse_bank_from is not None else None
        if source is not None and source.is_file():
            _validate_defect_bank(cfg, source)
            _link_or_copy(source, path)
        else:
            build_defect_bank(cfg, output)
    _validate_defect_bank(cfg, path)
    return path


def _support_receipt(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    bank = TwoSpeciesDefectBank.load(output / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "physical_interval": [float(bank.times[0]), float(bank.times[-1])],
        "train_runs": split.train.tolist(),
        "minimum_mass": minimum_mass,
        "species": {},
    }
    for species in ("plus", "minus"):
        counts = np.asarray(getattr(bank, f"{species}_counts"), dtype=np.int64)
        train_counts = counts[split.train]
        masses = np.mean(train_counts, axis=0)
        row = {
            "train_mass": masses.tolist(),
            "train_total_defects": np.sum(train_counts, axis=0).tolist(),
            "train_runs_with_nonzero_mass": np.count_nonzero(
                train_counts, axis=0
            ).tolist(),
            "all_run_mass": np.mean(counts, axis=0).tolist(),
        }
        receipt["species"][species] = row
        if np.any(masses < minimum_mass):
            raise RuntimeError(
                f"{species} endpoint mass {masses.tolist()} is below "
                f"minimum_mass={minimum_mass:g}"
            )
    write_json(output / "endpoint_support.json", receipt)
    write_json(
        output / "view_manifest.json",
        {
            "schema_version": 1,
            "train_runs": split.train.tolist(),
            "design_runs": split.design.tolist(),
            "validation_runs": split.validation.tolist(),
        },
    )
    return receipt


def _endpoint_audit(cfg: dict[str, Any], output: Path) -> Path:
    bank = TwoSpeciesDefectBank.load(output / "two_species_defect_bank.npz")
    split = make_run_split(split_config(cfg))
    periods = (
        float(cfg["physics"]["box_size"]),
        float(cfg["physics"]["box_size"]),
        2.0 * np.pi,
    )
    grid = PeriodicHistogramGrid(
        periods,
        tuple(int(value) for value in cfg["law"]["grid_shape"]),
    )
    kernel = multiscale_periodic_kernel_fft(
        grid, jnp.asarray(cfg["law"]["mmd_bandwidths"], dtype=jnp.float64)
    )
    truth = {}
    minimum_mass = float(cfg["unbalanced"]["minimum_mass"])
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
    for seed in reference_seeds(cfg):
        directory = output / f"reference_seed_{seed}"
        schedule = json.loads(
            (directory / "reference_mass_schedule.json").read_text(encoding="utf-8")
        )
        for species in ("plus", "minus"):
            with np.load(
                directory / f"{species}_reference_bank.npz", allow_pickle=False
            ) as saved:
                nodes = np.asarray(saved["nodes"], dtype=np.float64)
                weights = np.asarray(saved["weights"], dtype=np.float64)
            reference_histograms = [
                histogram_mass(jnp.asarray(nodes[index]), jnp.asarray(weights[index]), grid)
                for index in (0, len(nodes) - 1)
            ]
            initial_mmd2 = float(
                periodic_grid_mmd2(reference_histograms[0], truth[(species, 0)], kernel)
            )
            target_mmd2 = float(
                periodic_grid_mmd2(reference_histograms[1], truth[(species, 1)], kernel)
            )
            no_transport_mmd2 = float(
                periodic_grid_mmd2(reference_histograms[0], truth[(species, 1)], kernel)
            )
            rows.append(
                {
                    "reference_seed": int(seed),
                    "species": species,
                    "initial_shape_mmd2": initial_mmd2,
                    "target_shape_mmd2": target_mmd2,
                    "no_transport_target_mmd2": no_transport_mmd2,
                    "target_mmd2_reduction_vs_no_transport": (
                        1.0 - target_mmd2 / no_transport_mmd2
                        if no_transport_mmd2 > 0.0
                        else None
                    ),
                    "reference_mass_initial": float(schedule[f"mass_{species}"][0]),
                    "reference_mass_target": float(schedule[f"mass_{species}"][-1]),
                }
            )
    target = output / "reference_endpoint_audit.json"
    write_json(
        target,
        {
            "schema_version": 1,
            "physical_interval": [float(bank.times[0]), float(bank.times[-1])],
            "metric": "production periodic 3-D multiscale histogram MMD squared",
            "rows": rows,
        },
    )
    return target


def main() -> int:
    args = _parse_args()
    base_path = args.base_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reuse_bank_from = (
        args.reuse_bank_from.expanduser().resolve()
        if args.reuse_bank_from is not None
        else None
    )
    cfg = _derived_config(
        base_path,
        float(args.start),
        float(args.end),
        args.initial_training_density,
        args.time_step,
    )
    _require_or_write_config(output, base_path, cfg, reuse_bank_from)
    print(
        f"endpoint reference stage=physical-bank "
        f"output={_ensure_physical_bank(cfg, output, reuse_bank_from)}",
        flush=True,
    )
    print(
        f"endpoint reference stage=defects "
        f"output={_ensure_defect_bank(cfg, output, reuse_bank_from)}",
        flush=True,
    )
    print(f"endpoint reference stage=defect-audit output={audit_defect_bank(cfg, output)}", flush=True)
    support = _support_receipt(cfg, output)
    print(f"endpoint support={support['species']}", flush=True)
    print(f"endpoint reference stage=reference output={build_references(cfg, output, output)}", flush=True)
    print(f"endpoint reference stage=endpoint-audit output={_endpoint_audit(cfg, output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
