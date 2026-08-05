"""Orchestrate calibration ablations in isolated JAX worker processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODES = ("base", "post_hoc", "relax_e2e", "full_e2e")
TRAINED_MODES = ("base", "relax_e2e", "full_e2e")


def _worker_paths(directory: Path, mode: str) -> dict[str, Path]:
    return {
        "report": directory / f"{mode}.json",
        "trace": directory / f"{mode}_trace.csv",
        "arrays": directory / f"{mode}_outputs.npz",
        "parameters": directory / f"{mode}_parameters.npz",
    }


def _run_worker(
    mode: str,
    config: Path,
    paths: dict[str, Path],
    *,
    base_paths: dict[str, Path] | None = None,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "calibration" / "run_calibration_mode.py"),
        "--mode",
        mode,
        "--config",
        str(config),
        "--output",
        str(paths["report"]),
        "--trace-output",
        str(paths["trace"]),
        "--arrays-output",
        str(paths["arrays"]),
        "--parameters-output",
        str(paths["parameters"]),
    ]
    if mode == "post_hoc":
        if base_paths is None:
            raise ValueError("base_paths are required for Post-hoc")
        command.extend(
            [
                "--parameters-input",
                str(base_paths["parameters"]),
                "--source-report",
                str(base_paths["report"]),
            ]
        )
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)



def _run_gradient_worker(config: Path, output: Path) -> None:
    command = [
        sys.executable,
        str(
            REPO_ROOT
            / "experiments"
            / "calibration"
            / "run_calibration_gradient_check.py"
        ),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

def _parameter_distance(left_path: Path, right_path: Path) -> float:
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        if set(left.files) != set(right.files):
            raise ValueError("parameter archives have different keys")
        squared = sum(
            np.sum((left[name] - right[name]) ** 2, dtype=np.float64)
            for name in left.files
        )
    return float(np.sqrt(squared))


def _combine_npz(
    destination: Path,
    worker_paths: dict[str, dict[str, Path]],
    field: str,
) -> None:
    payload: dict[str, np.ndarray] = {}
    for mode, paths in worker_paths.items():
        with np.load(paths[field], allow_pickle=False) as archive:
            for name in archive.files:
                payload[f"{mode}.{name}"] = archive[name]
    np.savez_compressed(destination, **payload)


def _combine_traces(
    destination: Path,
    worker_paths: dict[str, dict[str, Path]],
) -> None:
    base_trace = worker_paths["base"]["trace"]
    rows_by_mode: dict[str, list[dict[str, str]]] = {}
    fieldnames: list[str] | None = None
    for mode in MODES:
        source = base_trace if mode == "post_hoc" else worker_paths[mode]["trace"]
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"trace has no header: {source}")
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError("worker trace schemas do not match")
            rows_by_mode[mode] = list(reader)
    assert fieldnames is not None
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", *fieldnames])
        writer.writeheader()
        for mode in MODES:
            for row in rows_by_mode[mode]:
                writer.writerow({"mode": mode, **row})


def _validate_report(report: dict[str, Any], acceptance: dict[str, Any]) -> None:
    failures: list[str] = []
    if report["base_post_hoc_parameter_distance"] != 0.0:
        failures.append("Base and Post-hoc parameters differ")
    for mode in MODES:
        mode_report = report["modes"][mode]
        if not np.isfinite(mode_report["training_final_loss"]):
            failures.append(f"{mode}: non-finite training loss")
        for split_name in ("train", "validation"):
            split = mode_report[split_name]
            pipeline = split["pipeline_metrics"]
            if pipeline["relaxation_converged"] < acceptance[
                "minimum_relaxation_convergence_rate"
            ]:
                failures.append(f"{mode}/{split_name}: relaxation convergence")
            if pipeline["projection_converged"] < acceptance[
                "minimum_projection_convergence_rate"
            ]:
                failures.append(f"{mode}/{split_name}: projection convergence")
            if pipeline["projection_rank_deficient"] > acceptance[
                "maximum_rank_deficient_rate"
            ]:
                failures.append(f"{mode}/{split_name}: projection rank deficiency")
            if not np.isfinite(split["serving_angular"]["angular_mmd2"]):
                failures.append(f"{mode}/{split_name}: non-finite angular MMD")
    full_validation = report["modes"]["full_e2e"]["validation"]["pipeline_metrics"]
    if full_validation["moment_error_projected"] > acceptance[
        "maximum_full_validation_projected_moment_error"
    ]:
        failures.append("Full-E2E validation projected moment error is too large")
    for mode, check in report["finite_difference"].items():
        if check["best_relative_error"] > acceptance["maximum_gradient_relative_error"]:
            failures.append(f"{mode}: parameter gradient failed finite differences")
    if failures:
        raise SystemExit("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "calibration_ablations.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "calibration_ablations.json",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "calibration_ablations_trace.csv",
    )
    parser.add_argument(
        "--arrays-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "calibration_ablations_outputs.npz",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "calibration_ablations_parameters.npz",
    )
    parser.add_argument(
        "--skip-workers",
        action="store_true",
        help="aggregate already completed worker artifacts",
    )
    parser.add_argument(
        "--worker-directory",
        type=Path,
        default=REPO_ROOT / "artifacts" / ".calibration_ablation_workers",
    )
    args = parser.parse_args()
    args.worker_directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    workers = {mode: _worker_paths(args.worker_directory, mode) for mode in MODES}
    gradient_report_path = args.worker_directory / "gradient_checks.json"
    if not args.skip_workers:
        for mode in TRAINED_MODES:
            _run_worker(mode, args.config, workers[mode])
        _run_worker(
            "post_hoc", args.config, workers["post_hoc"], base_paths=workers["base"]
        )
        _run_gradient_worker(args.config, gradient_report_path)
    missing: list[str] = []
    for mode, paths in workers.items():
        required_fields = ("report", "arrays", "parameters")
        if mode != "post_hoc":
            required_fields = (*required_fields, "trace")
        for field in required_fields:
            if not paths[field].exists():
                missing.append(str(paths[field]))
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if configuration.get("finite_difference", {}).get("modes"):
        if not gradient_report_path.exists():
            missing.append(str(gradient_report_path))
    if missing:
        raise FileNotFoundError(f"missing worker artifacts: {missing}")

    mode_reports = {
        mode: json.loads(workers[mode]["report"].read_text(encoding="utf-8"))
        for mode in MODES
    }
    shared = mode_reports["base"].pop("shared")
    for mode in MODES[1:]:
        candidate = mode_reports[mode].pop("shared")
        if candidate != shared:
            raise ValueError(f"worker metadata differs for mode {mode}")
    finite_difference = (
        json.loads(gradient_report_path.read_text(encoding="utf-8"))
        if gradient_report_path.exists()
        else {}
    )
    for value in mode_reports.values():
        value.pop("finite_difference", None)
        value.pop("schema_version", None)
        value.pop("mode", None)

    base_post_distance = _parameter_distance(
        workers["base"]["parameters"], workers["post_hoc"]["parameters"]
    )
    post_correction = mode_reports["post_hoc"]["validation"]["pipeline_metrics"][
        "total_correction_rms"
    ]
    full_correction = mode_reports["full_e2e"]["validation"]["pipeline_metrics"][
        "total_correction_rms"
    ]
    report = {
        "schema_version": 1,
        "backend": "local-jax-isolated-workers",
        **shared,
        "modes": mode_reports,
        "finite_difference": finite_difference,
        "base_post_hoc_parameter_distance": base_post_distance,
        "full_vs_post_hoc_validation_correction_ratio": full_correction
        / max(post_correction, 1e-15),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _combine_traces(args.trace_output, workers)
    _combine_npz(args.arrays_output, workers, "arrays")
    _combine_npz(args.parameters_output, workers, "parameters")

    _validate_report(report, configuration["acceptance"])
    console = {
        "dataset_shape": report["dataset_shape"],
        "train_size": len(report["train_indices"]),
        "validation_size": len(report["validation_indices"]),
        "num_updates": len(report["minibatch_indices"]),
        "full_vs_post_hoc_validation_correction_ratio": report[
            "full_vs_post_hoc_validation_correction_ratio"
        ],
        "validation": {
            mode: {
                "moment_error_initial": value["validation"]["pipeline_metrics"][
                    "moment_error_initial"
                ],
                "moment_error_projected": value["validation"]["pipeline_metrics"][
                    "moment_error_projected"
                ],
                "total_correction_rms": value["validation"]["pipeline_metrics"][
                    "total_correction_rms"
                ],
                "serving_angular_mmd2": value["validation"]["serving_angular"][
                    "angular_mmd2"
                ],
                "serving_regime_separation_ratio": value["validation"][
                    "serving_angular"
                ]["regime_separation_ratio"],
            }
            for mode, value in report["modes"].items()
        },
        "report": str(args.output),
    }
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
