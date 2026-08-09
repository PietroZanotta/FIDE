"""Internal command-line utilities used by the repository shell workflows.

Users should invoke the corresponding scripts/*.sh wrappers rather than calling
this module directly. Keeping the Python implementations in the package avoids
exposing a second, parallel set of user-facing entrypoints.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import jax
import numpy as np
import yaml

from .adaptive_components import (
    ProposalArchitecture,
    WarmStartArchitecture,
    initialize_proposal_model,
    initialize_warm_start_model,
)
from .config import load_config
from .energy import conditioned_from_reference, conditioned_probabilities, distribution_summaries, prior_probabilities
from .flow import (
    FlowArchitecture,
    flow_diffpop_objective_and_gradient,
    flow_gradient_directional_check,
    initialize_flow_model,
)
from .homometric import build_population_support
from .metrics import energy_score_discrete, total_variation
from .network import PriorParameters
from .seed_study import aggregate_reports
from .solvers import calibrate_dual_from_probabilities, tilted_ensemble_from_probabilities
from .tesseract_backend import serialize_calibration, serialize_tilted
from .uq import summarize_higher_order


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_norm(tree) -> float:
    return float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(leaf)))
                for leaf in jax.tree_util.tree_leaves(tree)
            )
        )
    )


def _tree_difference_norm(left, right) -> float:
    differences = jax.tree_util.tree_map(lambda a, b: a - b, left, right)
    return _tree_norm(differences)


def _load_apply(path: Path, module_name: str) -> Callable[[dict], dict]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply


def command_seed_sweep(args: argparse.Namespace) -> int:
    sweep_path = Path(args.config).resolve()
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    base_path = Path(sweep["base_config"])
    if not base_path.is_absolute():
        base_path = (_repository_root() / base_path).resolve()
    load_config(base_path)

    output_root = Path(args.output or sweep["output_root"])
    if not output_root.is_absolute():
        output_root = _repository_root() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    report_paths: list[str] = []
    environment = dict(os.environ)
    source_path = str(_repository_root() / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else source_path + os.pathsep + environment["PYTHONPATH"]
    )

    for seed in sweep["seeds"]:
        run_dir = output_root / f"seed_{int(seed)}"
        report_path = run_dir / "scientific_comparison_report.json"
        if not args.aggregate_only:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "manybody_completion.comparison_cli",
                    "--config",
                    str(base_path),
                    "--seed",
                    str(int(seed)),
                    "--output",
                    str(run_dir),
                ],
                cwd=_repository_root(),
                env=environment,
                check=True,
            )
        if not report_path.is_file():
            raise SystemExit(f"missing {report_path}")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        try:
            report_paths.append(str(report_path.relative_to(_repository_root())))
        except ValueError:
            report_paths.append(str(report_path))

    aggregate = aggregate_reports(reports, report_paths)
    aggregate_path = output_root / "multi_seed_scientific_comparison.json"
    _write_json(aggregate_path, aggregate)
    print(f"wrote {aggregate_path}")
    return 0



def command_particle_correction_study(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    support = build_population_support(int(config["system"]["n_spins"]))
    sampler_options = dict(config["sampler"])
    calibration_options = dict(config["calibration"])
    variant_specs = {
        "Flow-DiffPOP-PostHoc": ("population_flow_probabilities", "posthoc", 400),
        "Flow-DiffPOP-StopGrad": ("stopgrad_flow_probabilities", "stopgrad", 500),
        "Flow-DiffPOP-FullE2E": ("full_e2e_flow_probabilities", "full_e2e", 600),
    }
    seed_results = []
    for seed_dir in sorted(input_root.glob("seed_*")):
        report_path = seed_dir / "scientific_comparison_report.json"
        arrays_path = seed_dir / "scientific_comparison_arrays.npz"
        if not report_path.is_file() or not arrays_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        seed = int(report["metadata"]["seed"])
        with np.load(arrays_path) as arrays:
            reference = np.asarray(arrays["reference_probabilities"], dtype=np.float64)
            target = float(np.sum(reference * support.pair))
            per_seed = {
                "seed": seed,
                "direct_conditional_flow": {
                    key: float(report["methods"]["Direct-Conditional-Flow"][key])
                    for key in (
                        "moment_error",
                        "mode_probability_error",
                        "joint_total_variation",
                        "hidden_energy_score",
                    )
                },
                "variants": {},
            }
            for method_name, (prior_key, array_prefix, seed_offset) in variant_specs.items():
                prior = np.asarray(arrays[prior_key], dtype=np.float64)
                original = np.asarray(
                    arrays[f"{array_prefix}_conditioned_probabilities"], dtype=np.float64
                )
                exact, exact_dual = conditioned_from_reference(prior, support, target)
                corrected = calibrate_dual_from_probabilities(
                    prior,
                    support,
                    target,
                    sampler_options=sampler_options,
                    calibration_options=calibration_options,
                    seed=seed + seed_offset,
                )

                def metrics(probabilities: np.ndarray) -> dict[str, float]:
                    probabilities = probabilities / probabilities.sum()
                    summary = distribution_summaries(probabilities, support)
                    reference_summary = distribution_summaries(reference, support)
                    return {
                        "moment_error": abs(summary["pair_mean"] - target),
                        "mode_probability_error": abs(
                            summary["mode_plus_probability"]
                            - reference_summary["mode_plus_probability"]
                        ),
                        "joint_total_variation": total_variation(probabilities, reference),
                        "hidden_energy_score": energy_score_discrete(
                            support.triplet, probabilities, support.triplet, reference
                        ),
                    }

                uq = summarize_higher_order(
                    support.triplet,
                    support.labels,
                    corrected.final_ensemble.atom_probabilities,
                    reference_probabilities=reference,
                    effective_sample_size=corrected.final_ensemble.effective_sample_size,
                )
                per_seed["variants"][method_name] = {
                    "original_500_particle": metrics(original),
                    "exact_tilt_model": metrics(exact),
                    "corrected_final_ensemble": metrics(
                        corrected.final_ensemble.atom_probabilities
                    ),
                    "exact_dual": float(exact_dual),
                    "fitted_dual": float(corrected.dual),
                    "fit_particles": int(sampler_options["particles"]),
                    "final_particles": int(corrected.final_ensemble.indices.size),
                    "effective_sample_size": float(
                        corrected.final_ensemble.effective_sample_size
                    ),
                    "particle_to_exact_tilt_tv": total_variation(
                        corrected.final_ensemble.atom_probabilities, exact
                    ),
                    "higher_order_uq": uq,
                }
            seed_results.append(per_seed)

    if not seed_results:
        raise SystemExit(f"no completed seed runs found under {input_root}")

    aggregate = {"seed_count": len(seed_results), "seeds": seed_results, "method_means": {}}
    for method_name in variant_specs:
        aggregate["method_means"][method_name] = {}
        for stage in ("original_500_particle", "exact_tilt_model", "corrected_final_ensemble"):
            aggregate["method_means"][method_name][stage] = {
                endpoint: float(
                    np.mean(
                        [
                            seed_result["variants"][method_name][stage][endpoint]
                            for seed_result in seed_results
                        ]
                    )
                )
                for endpoint in (
                    "moment_error",
                    "mode_probability_error",
                    "joint_total_variation",
                    "hidden_energy_score",
                )
            }
    direct_means = {
        endpoint: float(
            np.mean(
                [seed_result["direct_conditional_flow"][endpoint] for seed_result in seed_results]
            )
        )
        for endpoint in (
            "moment_error",
            "mode_probability_error",
            "joint_total_variation",
            "hidden_energy_score",
        )
    }
    aggregate["direct_conditional_flow_means"] = direct_means
    aggregate["interpretation"] = {
        "primary_change": "independent final ensemble increased from 500 to 8000 particles",
        "training_changed": False,
        "model_distribution_metric": "exact finite-support exponential tilt",
        "particle_metric": "fresh independent SMC ensemble",
    }
    json_path = output_root / "particle_correction_study.json"
    _write_json(json_path, aggregate)

    lines = [
        "# DiffPOP particle-correction study",
        "",
        f"Seeds: `{len(seed_results)}`",
        "",
        "The learned flow models are unchanged. Only the independent final ensemble was increased from 500 to 8,000 particles.",
        "",
        "| Method | Stage | Moment error | Mode error | Joint TV | Hidden energy score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method_name in variant_specs:
        for stage in ("original_500_particle", "corrected_final_ensemble", "exact_tilt_model"):
            values = aggregate["method_means"][method_name][stage]
            lines.append(
                f"| {method_name} | {stage} | {values['moment_error']:.5f} | "
                f"{values['mode_probability_error']:.5f} | {values['joint_total_variation']:.5f} | "
                f"{values['hidden_energy_score']:.5f} |"
            )
    lines.extend(
        [
            "",
            "Direct conditional flow mean: "
            f"moment `{direct_means['moment_error']:.5f}`, mode `{direct_means['mode_probability_error']:.5f}`, "
            f"TV `{direct_means['joint_total_variation']:.5f}`, hidden score `{direct_means['hidden_energy_score']:.5f}`.",
            "",
        ]
    )
    markdown_path = output_root / "PARTICLE_CORRECTION_STUDY.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0

def command_gradient_probe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    support = build_population_support(int(config["system"]["n_spins"]))
    true_params = PriorParameters.from_mapping(config["true_prior"])
    reference = conditioned_probabilities(true_params, support, float(config["target"]["true_tilt"]))
    target = distribution_summaries(reference, support)["pair_mean"]
    rng = np.random.default_rng(int(config["seed"]) + 991)
    score_ids = rng.choice(support.size, size=64, p=reference)
    state_dim = support.n_spins + 1
    base = rng.normal(size=(48, state_dim))
    model = initialize_flow_model(
        FlowArchitecture(
            state_dim=state_dim,
            hidden_width=min(int(config["flow"]["hidden_width"]), 24),
            hidden_layers=1,
        ),
        seed=int(config["seed"]) + 992,
        label_scale=float(config["flow"]["label_scale"]),
    )
    common = {
        "target_moment": target,
        "sample_indices_for_score": score_ids,
        "base_samples": base,
        "sampling_steps": 3,
        "assignment_temperature": max(float(config["flow"]["assignment_temperature"]), 0.25),
        "dual_iterations": 10,
    }
    check = flow_gradient_directional_check(
        model,
        support,
        epsilon=2e-5,
        seed=int(config["seed"]) + 993,
        **common,
    )
    full_value, full_gradient = flow_diffpop_objective_and_gradient(
        model, support, differentiate_dual=True, **common
    )
    stop_value, stop_gradient = flow_diffpop_objective_and_gradient(
        model, support, differentiate_dual=False, **common
    )
    payload = {
        **check,
        "full_objective": full_value,
        "stopgrad_objective": stop_value,
        "full_gradient_norm": _tree_norm(full_gradient),
        "stopgrad_gradient_norm": _tree_norm(stop_gradient),
        "full_minus_stopgrad_gradient_norm": _tree_difference_norm(full_gradient, stop_gradient),
        "passed": bool(
            check["relative_error"] < 2e-3
            and np.isfinite(check["autodiff_directional_derivative"])
        ),
    }
    if args.experiment_report:
        payload["experiment_report"] = args.experiment_report
    output = Path(args.output)
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def command_backend_smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    support = build_population_support(int(config["system"]["n_spins"]))
    params = PriorParameters.from_mapping(config["true_prior"])
    sampler = dict(config["sampler"])
    calibration = dict(config["calibration"])
    dual = float(config["target"]["true_tilt"])
    target_distribution = conditioned_probabilities(params, support, dual)
    target = float(np.sum(target_distribution * support.pair))

    learned_prior = prior_probabilities(params, support)
    learned_prior = learned_prior * np.exp(0.08 * support.triplet)
    learned_prior /= learned_prior.sum()

    root = _repository_root()
    tilted_apply = _load_apply(
        root / "tesseracts/scientific_tilted_ensemble/tesseract_api.py", "tilted_api"
    )
    dual_apply = _load_apply(
        root / "tesseracts/scientific_dual_calibration/tesseract_api.py", "dual_api"
    )
    proposal_model = initialize_proposal_model(
        ProposalArchitecture(hidden_width=10, hidden_layers=1),
        seed=41,
        defensive_mixture=0.12,
    )
    warm_start_model = initialize_warm_start_model(
        WarmStartArchitecture(hidden_width=10, hidden_layers=1),
        seed=43,
        max_abs_dual=float(calibration["max_dual_norm"]),
    )
    local_tilt = serialize_tilted(
        tilted_ensemble_from_probabilities(
            learned_prior,
            support,
            dual,
            seed=123,
            proposal_model=proposal_model,
            **sampler,
        )
    )
    api_tilt = tilted_apply(
        {
            "n_spins": support.n_spins,
            "prior_probabilities": learned_prior.tolist(),
            "dual": dual,
            "sampler_options": sampler,
            "proposal_model": proposal_model.to_mapping(),
            "seed": 123,
        }
    )
    local_dual = serialize_calibration(
        calibrate_dual_from_probabilities(
            learned_prior,
            support,
            target,
            sampler_options=sampler,
            calibration_options=calibration,
            seed=456,
            proposal_model=proposal_model,
            warm_start_model=warm_start_model,
        )
    )
    api_dual = dual_apply(
        {
            "n_spins": support.n_spins,
            "prior_probabilities": learned_prior.tolist(),
            "target_moment": target,
            "sampler_options": sampler,
            "calibration_options": calibration,
            "proposal_model": proposal_model.to_mapping(),
            "warm_start_model": warm_start_model.to_mapping(),
            "seed": 456,
        }
    )
    tilt_error = float(
        np.max(
            np.abs(
                np.asarray(local_tilt["atom_probabilities"])
                - np.asarray(api_tilt["atom_probabilities"])
            )
        )
    )
    dual_error = abs(float(local_dual["dual"]) - float(api_dual["dual"]))
    result = {
        "prior_interface": "prior_probabilities",
        "adaptive_components": True,
        "tilted_atom_probability_max_error": tilt_error,
        "dual_error": dual_error,
        "tilted_status_equal": local_tilt["moment_mean"] == api_tilt["moment_mean"],
        "calibration_status_equal": local_dual["status"] == api_dual["status"],
        "passed": bool(tilt_error == 0.0 and dual_error == 0.0),
    }
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


def command_validate(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    required = [
        directory / "scientific_comparison_report.json",
        directory / "scientific_comparison_summary.csv",
        directory / "scientific_comparison_arrays.npz",
        directory / "SCIENTIFIC_COMPARISON_SUMMARY.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")

    report = json.loads(required[0].read_text(encoding="utf-8"))
    for name, method in report["methods"].items():
        for key in ("moment_error", "ess_fraction", "mode_probability_error", "hidden_energy_score"):
            value = float(method[key])
            if not math.isfinite(value):
                raise SystemExit(f"{name}.{key} is non-finite")
        if method["moment_error"] < 0 or method["mode_probability_error"] < 0:
            raise SystemExit(f"{name} has a negative error")

    with required[1].open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(report["methods"]):
        raise SystemExit("CSV method count does not match JSON report")

    with np.load(required[2]) as arrays:
        if not arrays.files:
            raise SystemExit("NPZ contains no arrays")
        for key in arrays.files:
            if not np.all(np.isfinite(arrays[key])):
                raise SystemExit(f"array {key} contains non-finite values")
    print(f"validated {directory}")
    return 0


def _canonical_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    report.get("metadata", {}).pop("created_utc", None)
    return report


def command_reproducibility(args: argparse.Namespace) -> int:
    left = Path(args.left)
    right = Path(args.right)
    report_equal = _canonical_report(left / "scientific_comparison_report.json") == _canonical_report(
        right / "scientific_comparison_report.json"
    )
    with np.load(left / "scientific_comparison_arrays.npz") as left_arrays, np.load(
        right / "scientific_comparison_arrays.npz"
    ) as right_arrays:
        arrays_equal = set(left_arrays.files) == set(right_arrays.files) and all(
            np.array_equal(left_arrays[key], right_arrays[key]) for key in left_arrays.files
        )
    payload = {
        "report_equal": report_equal,
        "arrays_equal": arrays_equal,
        "passed": report_equal and arrays_equal,
    }
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-sweep")
    seed.add_argument("--config", default="configs/scientific_seed_sweep.yaml")
    seed.add_argument("--output")
    seed.add_argument("--aggregate-only", action="store_true")
    seed.set_defaults(handler=command_seed_sweep)


    particle = subparsers.add_parser("particle-correction-study")
    particle.add_argument("--config", default="configs/diffpop_micro.yaml")
    particle.add_argument("--input", default="artifacts/flow_micro_three_seed")
    particle.add_argument(
        "--output", default="artifacts/particle_correction_study"
    )
    particle.set_defaults(handler=command_particle_correction_study)

    gradient = subparsers.add_parser("gradient-probe")
    gradient.add_argument("--config", default="configs/diffpop_micro.yaml")
    gradient.add_argument("--output", required=True)
    gradient.add_argument("--experiment-report")
    gradient.set_defaults(handler=command_gradient_probe)

    backend = subparsers.add_parser("backend-smoke")
    backend.add_argument("--config", default="configs/diffpop_micro.yaml")
    backend.add_argument("--output", required=True)
    backend.set_defaults(handler=command_backend_smoke)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--directory", required=True)
    validate.set_defaults(handler=command_validate)

    reproducibility = subparsers.add_parser("reproducibility")
    reproducibility.add_argument("--left", required=True)
    reproducibility.add_argument("--right", required=True)
    reproducibility.add_argument("--output", required=True)
    reproducibility.set_defaults(handler=command_reproducibility)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
