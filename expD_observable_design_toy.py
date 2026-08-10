#!/usr/bin/env python3
"""CLI for Experiment D: learned observables on the Experiment-B toy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import example_b as exb
import observable_design_toy as od


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "observable_design_toy.yaml"
DEFAULT_OUT = ROOT / "results" / "observable_design_toy"


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(f"{path} is not JSON-compatible YAML and PyYAML is not installed") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("configuration root must be a mapping")
        return data


def _key(seed: int, stream: int) -> jax.Array:
    return jax.random.fold_in(jax.random.PRNGKey(seed), stream)


def _load_reference(path: Path):
    if path.resolve() == exb.MODEL_PATH.resolve():
        return exb.load_model()[0]
    data = np.load(path)
    return exb.unflatten(jnp.asarray(data["reference_params"]), exb.REFERENCE_HIDDEN, exb.STATE_DIM)


def _load_observable(path: Path, standardization: od.Standardization) -> od.ObservableModel:
    data = np.load(path)
    return od.ObservableModel(jnp.asarray(data["A"]), standardization)


def _train_objective(
    objective: str,
    key: jax.Array,
    standardization: od.Standardization,
    B0: jax.Array,
    reference_params,
    budget: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    common = dict(steps=int(budget["observable_steps"]))
    if objective == "info":
        return od.train_info(key, standardization, B0, n_train=int(budget["objective_train_samples"]),
                             n_validation=int(budget["objective_validation_samples"]), **common)
    if objective == "cv":
        return od.train_cv(key, standardization, B0, reference_params,
                           n_train=int(budget["objective_train_samples"]),
                           n_validation=int(budget["objective_validation_samples"]), **common)
    if objective == "fiber":
        return od.train_fiber(key, standardization, B0, reference_params,
                              n_times=int(budget["fiber_times"]),
                              n_particles=int(budget["fiber_particles"]),
                              delta_t=float(budget["local_dt"]), **common)
    raise ValueError(objective)


def _save_ritz(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, potential_params=np.asarray(exb.core.flatten_mlp(params)))


def _load_ritz(path: Path):
    return exb.unflatten(jnp.asarray(np.load(path)["potential_params"]), exb.RITZ_HIDDEN, 1)


def _summary_row(label: str, model: od.ObservableModel, result: dict[str, Any]) -> dict[str, Any]:
    projection = result["projection"]
    endpoint = result["endpoint"]
    downstream = result["downstream"]
    training = result["training"]
    robust = result["robustness"]
    robust_downstream = result.get("robustness_downstream", [])
    if robust_downstream:
        nominal = downstream["summary"]["mfsi_learned_safe"]["mean_interior_mmd"]
        robustness_degradation = max(
            row["downstream"]["summary"]["mfsi_learned_safe"]["mean_interior_mmd"] - nominal
            for row in robust_downstream
        )
    else:
        # Feasibility-only fallback, explicitly based on ESS rather than rollout.
        robustness_degradation = min(r["ess_fraction"] for r in projection) - min(r["min_endpoint_ess"] for r in robust)
    return {
        "objective": label,
        "R": model.R,
        "coefficients_path": result["checkpoint"],
        "endpoint_expectation_gap": endpoint["expectation_gap_norm"],
        "endpoint_classifier_auroc": result["endpoint_classifier"]["auroc"],
        "endpoint_phi_mmd": endpoint["phi_space_mmd"],
        "hidden_endpoint_angular_gap": endpoint["hidden_angular_gap_norm"],
        "min_ess": min(r["ess_fraction"] for r in projection),
        "mean_ess": float(np.mean([r["ess_fraction"] for r in projection])),
        "max_covariance_condition": max(r["covariance_condition"] for r in projection),
        "mean_projection_distortion": float(np.mean([r["projection_distortion"] for r in projection])),
        "reduced_flow_closure_R2": result["reduced_flow_closure"]["closure_R2"],
        "local_tangent_law_closure_mmd": downstream["local_summary"]["mean_tangent_next_mmd"],
        "tangent_mfsi_velocity_gap": downstream["local_summary"]["mean_velocity_gap_mse"],
        "tangent_interior_rollout_mmd": downstream["summary"]["moment_tangent"]["mean_interior_mmd"],
        "mfsi_safe_interior_rollout_mmd": downstream["summary"]["mfsi_learned_safe"]["mean_interior_mmd"],
        "max_moment_error": downstream["summary"]["mfsi_learned_safe"]["max_moment_error"],
        "hidden_angular_reconstruction_error": downstream["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"],
        "robustness_degradation": robustness_degradation,
    }


def _crossed_bootstrap(records: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    metrics = ("tangent_local_mmd", "tangent_rollout_mmd", "mfsi_rollout_mmd", "velocity_gap", "min_ess")
    objectives = sorted({r["objective"] for r in records})
    model_seeds = sorted({int(r["model_seed"]) for r in records})
    eval_seeds = sorted({int(r["evaluation_seed"]) for r in records})
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for objective in objectives:
        for metric in metrics:
            table = {(int(r["model_seed"]), int(r["evaluation_seed"])): float(r[metric])
                     for r in records if r["objective"] == objective}
            if any((m, e) not in table for m in model_seeds for e in eval_seeds):
                continue
            matrices[(objective, metric)] = np.asarray([[table[(m, e)] for e in eval_seeds] for m in model_seeds])
    rng = np.random.default_rng(seed)
    md = rng.integers(0, len(model_seeds), (replicates, len(model_seeds)))
    ed = rng.integers(0, len(eval_seeds), (replicates, len(eval_seeds)))
    output: dict[str, Any] = {"method": "independent model-row/evaluation-column percentile bootstrap",
                              "replicates": replicates, "contrasts": {}}
    for left, right in (("fiber", "info"), ("fiber", "cv")):
        if left not in objectives or right not in objectives:
            continue
        contrast = {}
        for metric in metrics:
            if (left, metric) not in matrices or (right, metric) not in matrices:
                continue
            diff = matrices[(left, metric)] - matrices[(right, metric)]
            draws = diff[md[:, :, None], ed[:, None, :]].mean(axis=(1, 2))
            lo, hi = np.quantile(draws, [0.025, 0.975])
            contrast[metric] = {"mean": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}
        output["contrasts"][f"{left}_minus_{right}"] = contrast
    return output


def _write_report(path: Path, run: dict[str, Any], phase: str) -> None:
    names = list(run["objectives"])
    pairwise = run["subspace_comparison"]
    learned = [name for name in ("info", "cv", "fiber") if name in run["objectives"]]
    entries = run["objectives"]
    question_lines: list[str] = []
    if len(learned) >= 2:
        offdiag = [pairwise["distances"][a][b] for i, a in enumerate(learned) for b in learned[i + 1:]]
        question_lines += [
            "### Question 1: do the objectives select different subspaces?",
            "",
            f"In this run the learned-subspace distances range from `{min(offdiag):.4g}` to `{max(offdiag):.4g}`. "
            "For a smoke run this is only a collapse/separation diagnostic, not a stable scientific conclusion.",
            "",
        ]
    if "info" in entries:
        info = entries["info"]
        other_aurocs = [entries[n]["endpoint_classifier"]["auroc"] for n in learned if n != "info"]
        other_ess = [min(r["ess_fraction"] for r in entries[n]["projection"]) for n in learned if n != "info"]
        question_lines += [
            "### Question 2: information versus fiber conditioning",
            "",
            f"INFO endpoint AUROC is `{info['endpoint_classifier']['auroc']:.4g}`"
            + (f" versus `{np.mean(other_aurocs):.4g}` averaged over the other learned maps" if other_aurocs else "")
            + f"; its minimum path ESS is `{min(r['ess_fraction'] for r in info['projection']):.4g}`"
            + (f" versus `{np.mean(other_ess):.4g}` for the others" if other_ess else "") + ".",
            "",
        ]
    if learned:
        closure = {n: entries[n]["reduced_flow_closure"]["closure_R2"] for n in learned}
        question_lines += [
            "### Question 3: reduced-flow closure",
            "",
            f"Fresh frozen-observable closure R2 values are `{json.dumps(closure)}`. These are reported separately from law-level MMD.",
            "",
        ]
    if "fiber" in entries and "info" in entries:
        fiber_local = entries["fiber"]["downstream"]["local_summary"]["mean_tangent_next_mmd"]
        info_local = entries["info"]["downstream"]["local_summary"]["mean_tangent_next_mmd"]
        question_lines += [
            "### Question 4: tangent-to-next-law closure",
            "",
            f"Mean local tangent MMD is `{fiber_local:.4g}` for FIBER and `{info_local:.4g}` for INFO (FIBER minus INFO `{fiber_local-info_local:+.4g}`).",
            "",
            "### Question 5: local-to-global translation",
            "",
            f"FIBER's mean tangent-versus-MFSI velocity-gap MSE is `{entries['fiber']['downstream']['local_summary']['mean_velocity_gap_mse']:.4g}`; "
            f"its tangent and safe-MFSI interior rollout MMDs are `{entries['fiber']['downstream']['summary']['moment_tangent']['mean_interior_mmd']:.4g}` and "
            f"`{entries['fiber']['downstream']['summary']['mfsi_learned_safe']['mean_interior_mmd']:.4g}`. No direction is declared beneficial from this smoke cell.",
            "",
        ]
    if learned:
        robust_ess = {n: min(r["min_endpoint_ess"] for r in entries[n]["robustness"]) for n in learned}
        robust_law = {
            n: [r["downstream"]["summary"]["mfsi_learned_safe"]["mean_interior_mmd"]
                for r in entries[n].get("robustness_downstream", [])]
            for n in learned
        }
        has_robust_law = all(robust_law[n] for n in learned)
        question_lines += [
            "### Question 6: held-out rotations",
            "",
            f"Frozen-A minimum endpoint ESS across the two rotations is `{json.dumps(robust_ess)}`."
            + (f" Matched-retraining safe-MFSI interior MMDs by rotation are `{json.dumps(robust_law)}`."
               if has_robust_law else " This panel establishes feasibility only; it is not a frozen-network rollout claim."),
            "",
        ]
    lines = [
        "# Observable-design toy report",
        "",
        f"Phase: **{phase}**. " + ("This is a plumbing/debug run; no scientific claims are made." if phase == "smoke" else "Confirmatory protocol."),
        "",
        "## Reused implementation",
        "",
        "The experiment imports the Experiment-B endpoint samplers, linear stochastic interpolant and derivative, frozen flow-matched reference velocity, implicit empirical I-projection, stable covariance solve, Deep-Ritz potential/integrand, Heun rollout, RBF-MMD bandwidth convention, and held-out angular Fourier features. Experiment B itself was not modified.",
        "",
        "## Observable family and objectives",
        "",
        "The raw dictionary is `b=[x1,x2,x1^2,x1*x2,x2^2]`. A design-only pooled bank fixes `c_b` and an invertible symmetric whitening `W`; every learned map is `Phi_A=A W (b-c_b)` with `A A^T=I_R` and target zero. INFO minimizes supervised endpoint-label cross-entropy, CV minimizes variance-normalized reduced-flow closure error, and FIBER minimizes delta-t-normalized weighted MMD from a calibrated tangent pushforward to an independently calibrated next law, subject to hard calibration/rank/ESS gates.",
        "",
        "INFO is a supervised state-information proxy, not a mutual-information estimator. A discriminative distribution of `Phi_A(X)` is compatible with matched endpoint expectations.",
        "",
        "## Run status",
        "",
        f"Completed observable families: {', '.join(names)}.",
        f"Design raw endpoint discrepancy: `{run['design']['raw_endpoint_discrepancy']}`.",
        "",
        "## Learned subspaces",
        "",
        f"Pairwise projection distances: `{json.dumps(pairwise['distances'])}`.",
        f"Principal angles (radians): `{json.dumps(pairwise['principal_angles'])}`.",
        "",
        "See `summary.csv` for endpoint ambiguity, fiber conditioning, local closure, velocity-gap, rollout, hidden-law, and robustness diagnostics. The seven prespecified figure families are emitted as PNGs.",
        "",
        "## Prespecified scientific questions",
        "",
        *question_lines,
        "## Gradient validation",
        "",
        "Gradient-validation results are produced by `tests/test_observable_design_toy.py`; they are intentionally not fabricated into this run artifact. Run `python3 -m pytest tests/test_observable_design_toy.py -q` in the validated environment.",
        "",
        "## Interpretation",
        "",
        "No positive outcome is assumed. Inspect the pairwise angles and untouched-bank contrasts. A smoke run is only evidence that the protocol executes; confirmatory conclusions require the prespecified crossed model-seed/evaluation-bank run.",
        "",
        "## Current scope note",
        "",
        ("The rotation panel includes condition-specific matched-compute downstream retraining with A frozen; it is distinct from a strict zero-shot frozen-network test."
         if all(entries[n].get("robustness_downstream") for n in learned) else
         "The rotation panel reports frozen-A endpoint feasibility with the common rotated target recomputed. It is not labeled as a zero-shot downstream-network result. Full matched-compute downstream retraining per rotation remains required before a confirmatory robustness claim."),
    ]
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    phase_cfg = dict(config["phases"][args.phase])
    R = args.R if args.R is not None else int(config["headline_R"])
    if R == 5 and args.objective in {"all", *od.OBJECTIVES}:
        raise ValueError("R=5 is an invariance control, not a learned-observable headline; choose R=2,3,4")
    selected = list(od.OBJECTIVES) if args.objective == "all" else [args.objective]
    base_seed = int(args.seed if args.seed is not None else config["base_seed"])
    if args.seed is not None:
        model_seeds = [base_seed]
        eval_seeds = [base_seed + int(config["seed_offsets"]["evaluation"])]
    else:
        model_seeds = [base_seed + i for i in range(int(phase_cfg["model_seeds"]))]
        eval_seeds = [base_seed + int(config["seed_offsets"]["evaluation"]) + i
                      for i in range(int(phase_cfg["evaluation_seeds"]))]
    out = (args.out / args.phase / f"R{R}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[expD] phase={args.phase} R={R} objectives={selected} models={len(model_seeds)} evaluations={len(eval_seeds)}", flush=True)

    reference_path = Path(config["reference_checkpoint"])
    if not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    if not reference_path.exists():
        raise FileNotFoundError(f"frozen Experiment-B reference checkpoint not found: {reference_path}")
    reference_params = _load_reference(reference_path)

    # A fixed, design-only standardization shared by every objective/seed.
    design_seed = base_seed + int(config["seed_offsets"]["design"])
    kd0, kd1 = jax.random.split(jax.random.PRNGKey(design_seed))
    design_n = int(phase_cfg["design_samples_per_endpoint"])
    design_minus, design_plus = exb.sample_ring(kd0, design_n), exb.sample_four_lobes(kd1, design_n)
    standardization = od.fit_standardization(design_minus, design_plus)
    raw_minus = jnp.mean(od.raw_dictionary(design_minus), axis=0)
    raw_plus = jnp.mean(od.raw_dictionary(design_plus), axis=0)
    z_minus = jnp.mean(od.standardized_dictionary(design_minus, standardization), axis=0)
    z_plus = jnp.mean(od.standardized_dictionary(design_plus, standardization), axis=0)
    design = {
        "seed": design_seed, "samples_per_endpoint": design_n,
        "raw_minus": np.asarray(raw_minus).tolist(), "raw_plus": np.asarray(raw_plus).tolist(),
        "raw_endpoint_discrepancy": np.asarray(raw_minus - raw_plus).tolist(),
        "center": np.asarray(standardization.center).tolist(),
        "whitening": np.asarray(standardization.whitening).tolist(),
        "covariance_eigenvalues": np.asarray(standardization.covariance_eigenvalues).tolist(),
        "standardized_minus_mean": np.asarray(z_minus).tolist(),
        "standardized_plus_mean": np.asarray(z_plus).tolist(),
    }
    np.savez(out / "design_standardization.npz", center=np.asarray(standardization.center),
             whitening=np.asarray(standardization.whitening),
             covariance_eigenvalues=np.asarray(standardization.covariance_eigenvalues))

    labels = selected.copy()
    if not args.no_controls:
        labels += ["random", "full_phi5"]
    objective_results: dict[str, Any] = {}
    seed_records: list[dict[str, Any]] = []
    robustness_seed_records: list[dict[str, Any]] = []
    learned_for_comparison: dict[str, jax.Array] = {}
    cells_dir = out / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    for model_index, model_seed in enumerate(model_seeds):
        print(f"[expD] model seed {model_seed} ({model_index + 1}/{len(model_seeds)})", flush=True)
        init_key = _key(model_seed, int(config["seed_offsets"]["observable_initialization"]))
        B0, A0 = od.initialize_stiefel(init_key, R)
        for label_index, label in enumerate(labels):
            print(f"[expD]   {label}: observable/diagnostics", flush=True)
            label_R = 5 if label == "full_phi5" else R
            if label == "full_phi5":
                A = jnp.eye(5, dtype=jnp.float64)
                train_meta = {"control": "original full Phi-5 in the common standardized coordinates",
                              "equal_capacity": False}
            elif label == "random":
                A = A0
                train_meta = {"control": "random row-orthonormal subspace"}
            else:
                checkpoint = out / "checkpoints" / f"observable_{label}_modelseed_{model_seed}.npz"
                if checkpoint.exists() and not args.force:
                    cached = np.load(checkpoint)
                    A = jnp.asarray(cached["A"])
                    train_meta = json.loads(str(cached["metadata_json"]))
                    train_meta["loaded"] = True
                else:
                    print(f"[expD]   {label}: training observable", flush=True)
                    A, train_meta = _train_objective(
                        label, _key(model_seed, 100 + label_index), standardization, B0,
                        reference_params, phase_cfg,
                    )
                learned_for_comparison.setdefault(label, A)
            model = od.ObservableModel(A, standardization)
            checkpoint = out / "checkpoints" / f"observable_{label}_modelseed_{model_seed}.npz"
            od.save_observable(checkpoint, label, model, train_meta)

            # Diagnostics and downstream seeds are matched across objectives.
            kdiag = _key(model_seed, 300)
            ke, kc, kp, kr, kcvdiag = jax.random.split(kdiag, 5)
            endpoint = od.endpoint_equivalence(ke, model, int(phase_cfg["diagnostic_particles"]))
            endpoint_classifier = od.fit_frozen_representation_classifier(
                kc, model, n_train=int(phase_cfg["classifier_train_samples"]),
                n_evaluation=int(phase_cfg["classifier_evaluation_samples"]),
                steps=int(phase_cfg["classifier_steps"]),
            )
            times = list(map(float, config["evaluation_times"]))
            projection = od.projection_diagnostics(kp, model, reference_params, times,
                                                   int(phase_cfg["diagnostic_particles"]))
            reduced_flow = od.reduced_flow_closure_diagnostic(
                kcvdiag, model, reference_params,
                steps=int(phase_cfg["closure_diagnostic_steps"]),
                n_train=int(phase_cfg["objective_train_samples"]),
                n_validation=int(phase_cfg["objective_validation_samples"]),
            )
            robustness = od.rotated_endpoint_diagnostics(
                kr, model, [float(x) for x in config["robustness_rotations"]],
                int(phase_cfg["diagnostic_particles"]),
            )

            ritz_path = out / "checkpoints" / f"ritz_{label}_modelseed_{model_seed}.npz"
            if ritz_path.exists() and not args.force:
                potential = _load_ritz(ritz_path)
                ritz_meta = {"loaded": True}
            else:
                print(f"[expD]   {label}: training nominal Deep-Ritz", flush=True)
                potential, ritz_meta = od.train_downstream_ritz(
                    _key(model_seed, 500), model, reference_params,
                    steps=int(phase_cfg["ritz_steps"]), n_times=int(phase_cfg["ritz_times"]),
                    n_particles=int(phase_cfg["ritz_particles"]),
                )
                _save_ritz(ritz_path, potential)

            first_downstream = None
            for eval_seed in eval_seeds:
                cell_path = cells_dir / f"nominal_{label}_model_{model_seed}_eval_{eval_seed}.json"
                if cell_path.exists() and not args.force:
                    downstream = json.loads(cell_path.read_text())["downstream"]
                else:
                    print(f"[expD]   {label}: nominal evaluation bank {eval_seed}", flush=True)
                    downstream = od.evaluate_downstream(
                        _key(eval_seed, 700), model, reference_params, potential,
                        times=times, n_particles=int(phase_cfg["rollout_particles"]),
                        target_particles=int(phase_cfg["target_particles"]),
                        flow_steps=int(phase_cfg["flow_steps"]), local_dt=float(phase_cfg["local_dt"]),
                    )
                    cell_path.write_text(json.dumps({"downstream": od.json_ready(downstream)}, indent=2))
                first_downstream = first_downstream or downstream
                seed_records.append({
                    "objective": label, "model_seed": model_seed, "evaluation_seed": eval_seed,
                    "tangent_local_mmd": downstream["local_summary"]["mean_tangent_next_mmd"],
                    "tangent_rollout_mmd": downstream["summary"]["moment_tangent"]["mean_interior_mmd"],
                    "mfsi_rollout_mmd": downstream["summary"]["mfsi_learned_safe"]["mean_interior_mmd"],
                    "velocity_gap": downstream["local_summary"]["mean_velocity_gap_mse"],
                    "min_ess": min(r["ess_fraction"] for r in projection),
                    "max_moment_error": downstream["summary"]["mfsi_learned_safe"]["max_moment_error"],
                    "angular_error": downstream["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"],
                })

            first_robustness_downstream = []
            if bool(phase_cfg.get("robustness_downstream", False)) and label in selected:
                for rotation_index, robust_row in enumerate(robustness):
                    angle = float(robust_row["angle"])
                    rotated_target = jnp.asarray(robust_row["target"], dtype=jnp.float64)
                    robust_ritz_path = out / "checkpoints" / (
                        f"ritz_{label}_modelseed_{model_seed}_rotation_{rotation_index}.npz"
                    )
                    if robust_ritz_path.exists() and not args.force:
                        robust_potential = _load_ritz(robust_ritz_path)
                        robust_ritz_meta = {"loaded": True}
                    else:
                        print(f"[expD]   {label}: training rotation {rotation_index} Deep-Ritz", flush=True)
                        robust_potential, robust_ritz_meta = od.train_downstream_ritz(
                            _key(model_seed, 500), model, reference_params,
                            steps=int(phase_cfg["ritz_steps"]), n_times=int(phase_cfg["ritz_times"]),
                            n_particles=int(phase_cfg["ritz_particles"]),
                            target=rotated_target, angle=angle,
                        )
                        _save_ritz(robust_ritz_path, robust_potential)
                    first_condition = None
                    for eval_seed in eval_seeds:
                        cell_path = cells_dir / (
                            f"rotation_{rotation_index}_{label}_model_{model_seed}_eval_{eval_seed}.json"
                        )
                        if cell_path.exists() and not args.force:
                            condition_downstream = json.loads(cell_path.read_text())["downstream"]
                        else:
                            print(f"[expD]   {label}: rotation {rotation_index} evaluation bank {eval_seed}", flush=True)
                            condition_downstream = od.evaluate_downstream(
                                _key(eval_seed, 700), model, reference_params, robust_potential,
                                times=times, n_particles=int(phase_cfg["rollout_particles"]),
                                target_particles=int(phase_cfg["target_particles"]),
                                flow_steps=int(phase_cfg["flow_steps"]), local_dt=float(phase_cfg["local_dt"]),
                                target=rotated_target, angle=angle,
                            )
                            cell_path.write_text(json.dumps({"downstream": od.json_ready(condition_downstream)}, indent=2))
                        first_condition = first_condition or condition_downstream
                        robustness_seed_records.append({
                            "objective": label, "rotation_index": rotation_index, "angle": angle,
                            "model_seed": model_seed, "evaluation_seed": eval_seed,
                            "tangent_local_mmd": condition_downstream["local_summary"]["mean_tangent_next_mmd"],
                            "tangent_rollout_mmd": condition_downstream["summary"]["moment_tangent"]["mean_interior_mmd"],
                            "mfsi_rollout_mmd": condition_downstream["summary"]["mfsi_learned_safe"]["mean_interior_mmd"],
                            "velocity_gap": condition_downstream["local_summary"]["mean_velocity_gap_mse"],
                            "min_ess": min(r["ess_fraction"] for r in condition_downstream["target"]),
                            "max_moment_error": condition_downstream["summary"]["mfsi_learned_safe"]["max_moment_error"],
                            "angular_error": condition_downstream["summary"]["mfsi_learned_safe"]["mean_interior_angular_error"],
                        })
                    first_robustness_downstream.append({
                        "rotation_index": rotation_index, "angle": angle,
                        "target": np.asarray(rotated_target).tolist(),
                        "ritz_training": robust_ritz_meta,
                        "downstream": first_condition,
                    })
            if model_index == 0:
                objective_results[label] = {
                    "A": np.asarray(A).tolist(), "raw_coefficients": np.asarray(model.raw_coefficients).tolist(),
                    "raw_intercept": np.asarray(model.raw_intercept).tolist(),
                    "singular_values": np.asarray(jnp.linalg.svd(A, compute_uv=False)).tolist(),
                    "training": {**train_meta, "downstream_ritz": ritz_meta},
                    "endpoint": endpoint, "endpoint_classifier": endpoint_classifier,
                    "projection": projection, "reduced_flow_closure": reduced_flow,
                    "robustness": robustness,
                    "robustness_downstream": first_robustness_downstream,
                    "downstream": first_downstream, "checkpoint": str(checkpoint),
                }
        # Release per-seed training closures while retaining compiled rollout
        # kernels across all equal-R objectives within the crossed seed block.
        jax.clear_caches()

    compare_names = [n for n in selected if n in learned_for_comparison]
    distances, angles = {}, {}
    for left in compare_names:
        distances[left], angles[left] = {}, {}
        for right in compare_names:
            distances[left][right] = float(od.subspace_distance(learned_for_comparison[left], learned_for_comparison[right]))
            angles[left][right] = np.asarray(od.principal_angles(learned_for_comparison[left], learned_for_comparison[right])).tolist()

    robust_bootstrap = {}
    for rotation_index in sorted({r["rotation_index"] for r in robustness_seed_records}):
        subset = [r for r in robustness_seed_records if r["rotation_index"] == rotation_index]
        robust_bootstrap[str(rotation_index)] = _crossed_bootstrap(
            subset, int(phase_cfg["bootstrap_replicates"]),
            base_seed + int(config["seed_offsets"]["bootstrap"]) + rotation_index + 1,
        )
    result = {
        "protocol": {"phase": args.phase, "R": R, "config": str(args.config.resolve()),
                     "reference_checkpoint": str(reference_path),
                     "warning": "smoke results are debugging-only" if args.phase == "smoke" else None},
        "seed_manifest": {"base_seed": base_seed, "design_seed": design_seed,
                          "model_seeds": model_seeds, "evaluation_seeds": eval_seeds,
                          "offsets": config["seed_offsets"]},
        "design": design, "objectives": objective_results,
        "subspace_comparison": {"distances": distances, "principal_angles": angles},
        "seed_level_records": seed_records,
        "robustness_seed_level_records": robustness_seed_records,
        "crossed_bootstrap": _crossed_bootstrap(seed_records, int(phase_cfg["bootstrap_replicates"]),
                                                 base_seed + int(config["seed_offsets"]["bootstrap"])),
        "robustness_crossed_bootstrap": robust_bootstrap,
    }
    result = od.json_ready(result)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    (out / "seed_manifest.json").write_text(json.dumps(result["seed_manifest"], indent=2))
    (out / "resolved_config.json").write_text(json.dumps(config, indent=2))

    rows = []
    for label, entry in objective_results.items():
        model = od.ObservableModel(jnp.asarray(entry["A"]), standardization)
        rows.append(_summary_row(label, model, entry))
    if rows:
        with (out / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (out / "seed_level_results.csv").open("w", newline="") as f:
        if seed_records:
            writer = csv.DictWriter(f, fieldnames=list(seed_records[0]))
            writer.writeheader(); writer.writerows(seed_records)
    with (out / "robustness_seed_level_results.csv").open("w", newline="") as f:
        if robustness_seed_records:
            writer = csv.DictWriter(f, fieldnames=list(robustness_seed_records[0]))
            writer.writeheader(); writer.writerows(robustness_seed_records)
    if not args.no_plots:
        od.make_figures(out, result)
    _write_report(out / "OBSERVABLE_DESIGN_TOY_REPORT.md", result, args.phase)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "confirmatory"), default="smoke")
    parser.add_argument("--objective", choices=(*od.OBJECTIVES, "all"), default="all")
    parser.add_argument("--R", type=int, choices=(2, 3, 4), default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="run one model seed x one evaluation seed (useful for debugging)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="replace this experiment's cached checkpoints")
    parser.add_argument("--no-controls", action="store_true", help="skip random and higher-dimensional Phi-5 controls")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run(args)
    resolved_R = args.R or result["protocol"]["R"]
    print(json.dumps({"output": str((args.out / args.phase / f"R{resolved_R}").resolve()),
                      "objectives": list(result["objectives"]),
                      "seed_cells": len(result["seed_level_records"])}, indent=2))


if __name__ == "__main__":
    main()
