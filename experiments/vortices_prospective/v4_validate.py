from __future__ import annotations

"""Fresh post-freeze acquisition and paired validation for prospective v4."""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aggregate_qois import qoi_features
from common import artifact_dirs, config_hash, load_config, write_json_atomic
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from frozen_diagnostic_core import paired_statistics
from mfsi.cache import file_sha256, fingerprint
from physical import truth_from_config
from prospective_data import TargetProspectiveData
from v4_objective import distribution

jax.config.update("jax_enable_x64", True)


def _ensure_fresh_hidden_bank(
    cfg: dict[str, Any], hidden_dir: Path, manifest_sha: str
) -> tuple[Path, np.ndarray]:
    hidden_dir.mkdir(parents=True, exist_ok=True)
    path = hidden_dir / "v4_hidden_state_bank.npz"
    seed = int(cfg["seeds"]["validation_physical"])
    signature = fingerprint(
        {
            "schema": 4,
            "config_hash": config_hash(cfg),
            "manifest_sha256_at_creation": manifest_sha,
            "particles": int(cfg["truth"]["hidden_validation_particles"]),
            "seed": seed,
        }
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (
                str(np.asarray(data["signature"]).item()) == signature
                and str(np.asarray(data["manifest_sha256_at_creation"]).item()) == manifest_sha
            ):
                print("[v4-validation] reusing sealed fresh hidden bank", flush=True)
                return path, np.asarray(data["states"], dtype=np.float64)
        raise RuntimeError("existing v4 hidden bank is not tied to this frozen receipt")
    truth = truth_from_config(cfg)
    times = jnp.linspace(
        0.0, 1.0, int(cfg["time"]["scientific_nodes"]), dtype=jnp.float64
    )
    bank = truth.make_bank(
        seed=seed,
        n=int(cfg["truth"]["hidden_validation_particles"]),
        times=times,
        substeps_per_interval=int(cfg["truth"]["rk4_substeps_per_interval"]),
    )
    states = np.asarray(bank.particles, dtype=np.float64)
    np.savez_compressed(
        path,
        role=np.asarray("v4_fresh_hidden_validation_created_only_after_freeze"),
        signature=np.asarray(signature),
        manifest_sha256_at_creation=np.asarray(manifest_sha),
        physical_seed=np.asarray(seed),
        times=np.asarray(times),
        states=states,
    )
    return path, states


def _ensure_validation_randomness(
    cfg: dict[str, Any], hidden_dir: Path, particle_n: int, manifest_sha: str
):
    path = hidden_dir / "v4_hidden_observation_randomness.npz"
    trials = int(cfg["v4"]["validation_trials"])
    shape = (
        trials,
        int(cfg["time"]["acquisition_nodes"]),
        int(cfg["measurement"]["finite_n"]),
    )
    detector_shape = (
        trials,
        int(cfg["time"]["acquisition_nodes"]),
        int(cfg["measurement"]["n_sensors"]),
    )
    sampling_seed = int(cfg["seeds"]["validation_sampling"])
    detector_seed = int(cfg["seeds"]["validation_detector"])
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (
                tuple(data["sample_indices"].shape) == shape
                and tuple(data["detector_z"].shape) == detector_shape
                and int(data["particle_n"]) == int(particle_n)
                and str(np.asarray(data["manifest_sha256_at_creation"]).item()) == manifest_sha
                and int(data["sampling_seed"]) == sampling_seed
                and int(data["detector_seed"]) == detector_seed
            ):
                return path, np.asarray(data["sample_indices"]), np.asarray(data["detector_z"])
        raise RuntimeError("existing v4 validation randomness is incompatible with the freeze")
    sampling_rng = np.random.default_rng(sampling_seed)
    detector_rng = np.random.default_rng(detector_seed)
    indices = sampling_rng.integers(0, particle_n, size=shape, dtype=np.int32)
    detector_z = detector_rng.standard_normal(detector_shape)
    np.savez_compressed(
        path,
        role=np.asarray("v4_fresh_paired_validation_randomness"),
        sample_indices=indices,
        detector_z=detector_z,
        particle_n=np.asarray(particle_n),
        manifest_sha256_at_creation=np.asarray(manifest_sha),
        sampling_seed=np.asarray(sampling_seed),
        detector_seed=np.asarray(detector_seed),
    )
    return path, indices, detector_z


def _realized_bank_and_moments(evaluator, eta, states, sample_indices, detector_z):
    phi = np.asarray(
        evaluator.sensors.features(jnp.asarray(states), jnp.asarray(eta)),
        dtype=np.float64,
    )
    response_mean = np.mean(phi, axis=1)
    response_second = np.mean(phi * phi, axis=1)
    acq_idx = np.asarray(evaluator.acq_idx, dtype=np.int32)
    phi_acq = phi[acq_idx]
    sampled = np.empty(detector_z.shape, dtype=np.float64)
    for trial in range(len(sample_indices)):
        for acquisition in range(len(acq_idx)):
            sampled[trial, acquisition] = np.mean(
                phi_acq[acquisition, sample_indices[trial, acquisition]], axis=0
            )
    acq_mean = response_mean[acq_idx]
    variance = np.maximum(response_second[acq_idx] - acq_mean * acq_mean, 0.0)
    finite_se = np.sqrt(variance / float(evaluator.cfg["measurement"]["finite_n"]))
    effective_sampling_z = np.divide(
        sampled - acq_mean[None, :, :],
        finite_se[None, :, :],
        out=np.zeros_like(sampled),
        where=finite_se[None, :, :] > 1.0e-15,
    )
    qoi_targets = np.asarray(
        jnp.mean(qoi_features(jnp.asarray(states)), axis=1), dtype=np.float64
    )
    return (
        AggregateObservationBank(effective_sampling_z, detector_z),
        response_mean,
        response_second,
        qoi_targets,
    )


def _trial_values(result: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        [row[key] for row in result["trials"] if row["valid"] and row[key] is not None],
        dtype=np.float64,
    )


def _mean(result: dict[str, Any], key: str) -> float:
    value = result[key]["mean"]
    return float(value) if value is not None else float("nan")


def _certification(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["trials"]
    return {
        "valid_fraction": float(result["valid_fraction"]),
        "invalid_trial_count": int(sum(not row["valid"] for row in rows)),
        "max_projection_residual": float(max(row["max_projection_residual"] for row in rows)),
        "min_ess_fraction": float(min(row["min_ess_fraction"] for row in rows)),
        "min_covariance_eigenvalue": float(min(row["min_covariance_eigenvalue"] for row in rows)),
        "max_poisson_relative_residual": float(max(row["max_poisson_relative_residual"] for row in rows)),
        "max_component_compatibility_residual": float(max(row["max_component_compatibility_residual"] for row in rows)),
        "max_full_moment_rate_residual": float(max(row["max_full_moment_rate_residual"] for row in rows)),
        "all_full_solvers_converged": bool(all(row["full_solver_converged"] for row in rows)),
        "nan_or_inf_count": int(
            sum(
                not np.isfinite(row[key])
                for row in rows
                for key in ("scientific_risk", "full_action")
                if row[key] is not None
            )
        ),
        "boundary_domain": [0.0, 2.0, 0.0, 1.0],
        "boundary_treatment": "no-flux physical-direct rectangular-grid solve",
        "periodicity": "not applicable; double-gyre box is nonperiodic",
    }


def _write_outputs(results_dir: Path, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    write_json_atomic(results_dir / "validation_result.json", result)
    rows = []
    for method in ("Law", "Full"):
        for row in result["methods"][method]["realized"]["trials"]:
            rows.append({"method": method, **row})
    with (results_dir / "validation_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    law = result["methods"]["Law"]
    full = result["methods"]["Full"]
    paired = result["paired_full_minus_law"]
    diff = paired["difference_full_minus_law"]
    selection = manifest["selection_metrics"]
    lines = [
        "# Prospective v4 robust-Full result",
        "",
        "This is a genuinely new preregistered replicate. Selection was frozen before the fresh hidden "
        "microscopic bank was generated; the previous hidden validation bank was never used for v4 design.",
        "",
        "## Gradient-based selection",
        "",
        f"Law used `{selection['law_gradient_starts']}` gradient starts. Full used "
        f"`{selection['full_gradient_starts']}` independent Full-specific starts, producing "
        f"`{selection['genuinely_full_refined_risk_feasible_candidates']}` distinct exact-risk-feasible "
        f"Full-refined candidates and `{selection['authoritative_full_finalists']}` authoritative Full finalists.",
        "",
        "Full was differentiated through the aggregate response, frozen finite-sampling and detector-noise "
        "reparameterizations, production spline reconstruction, implicit I-projection, multiplier derivative, "
        "forcing, rasterization, and implicit Poisson solve. Leading candidates received Full-objective "
        "Adam/L-BFGS polishing; this was not proxy-only optimization followed by incumbent rescoring.",
        "",
        "## Frozen selection and held-out acquisition",
        "",
        "| Method | Centers | Predicted Full mean | Predicted SD | Realized Full mean | Realized risk |",
        "|:--|:--|--:|--:|--:|--:|",
    ]
    for name, block in (("Law", law), ("Full", full)):
        centers = " ".join(f"({x:.4f},{y:.4f})" for x, y in block["centers"])
        lines.append(
            f"| {name} | {centers} | {block['predicted_full_distribution']['mean']:.6g} | "
            f"{block['predicted_full_distribution']['sd']:.6g} | {block['realized_full_action']:.6g} | "
            f"{block['realized_risk']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Predicted Full reduction: `{100*result['claims']['predicted_full_reduction_vs_law']:.3f}%`. "
            f"Realized Full reduction: `{100*result['claims']['realized_full_reduction_vs_law']:.3f}%`.",
            f"Realized scientific-risk constraint: **{'PASS' if result['claims']['full_realized_risk_within_allowance'] else 'FAIL'}**.",
            "",
            "## Paired validation",
            "",
            f"For `{paired['valid_pair_count']}` aligned trials, `Full - Law` has mean `{diff['mean']:.6g}`, "
            f"median `{diff['median']:.6g}`, SD `{diff['std']:.6g}`, and SE `{diff['se']:.6g}`. "
            f"The paired t 95% CI is `[{paired['paired_t_95_ci'][0]:.6g}, "
            f"{paired['paired_t_95_ci'][1]:.6g}]`; the bootstrap CI is "
            f"`[{paired['paired_bootstrap_95_ci'][0]:.6g}, {paired['paired_bootstrap_95_ci'][1]:.6g}]`. "
            f"Full is lower in `{100*paired['fraction_full_lower']:.1f}%` of trials.",
            "",
            "## Prediction transfer and certification",
            "",
            f"Law realization-minus-prediction gap: `{law['prediction_to_realization_gap']['full_action']:.6g}`. "
            f"Full gap: `{full['prediction_to_realization_gap']['full_action']:.6g}`.",
            "",
            f"Law invalid trials: `{law['certification']['invalid_trial_count']}`; Full invalid trials: "
            f"`{full['certification']['invalid_trial_count']}`. The maximum projection residual is "
            f"`{max(law['certification']['max_projection_residual'], full['certification']['max_projection_residual']):.3e}`; "
            f"minimum ESS `{min(law['certification']['min_ess_fraction'], full['certification']['min_ess_fraction']):.4f}`; "
            f"maximum Poisson residual "
            f"`{max(law['certification']['max_poisson_relative_residual'], full['certification']['max_poisson_relative_residual']):.3e}`.",
            "",
            f"Frozen receipt: `{result['frozen_manifest_path']}`.",
        ]
    )
    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 3.7))
    for name, color, marker in (("Law", "#3366cc", "o"), ("Full", "#dc3912", "^")):
        centers = np.asarray(result["methods"][name]["centers"])
        ax.scatter(centers[:, 0], centers[:, 1], c=color, marker=marker, s=75, label=name)
    ax.set(xlim=(0, 2), ylim=(0, 1), xlabel="x", ylabel="y", title="Frozen v4 sensor geometries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "sensor_geometries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.7))
    difference = np.asarray(result["paired_trial_difference_full_minus_law"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.scatter(np.arange(len(difference)), difference, s=18)
    ax.set(xlabel="paired validation trial", ylabel="Full - Law action")
    fig.tight_layout()
    fig.savefig(results_dir / "paired_validation_differences.png", dpi=150)
    plt.close(fig)


def validate_v4(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dirs = artifact_dirs(output_dir)
    manifest_path = dirs["results"] / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("v4 validation requires the frozen pre-validation manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("status") != "frozen_before_hidden_validation"
        or manifest.get("hidden_validation_loaded") is not False
        or manifest.get("previous_hidden_validation_used_for_v4_design") is not False
    ):
        raise RuntimeError("manifest is not a valid sealed v4 preregistration")
    if manifest.get("config_hash") != config_hash(cfg):
        raise RuntimeError("v4 validation configuration differs from the frozen selection")
    result_path = dirs["results"] / "validation_result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("frozen_manifest_sha256") == manifest_sha:
            print("[v4-validation] reusing compatible completed validation", flush=True)
            return cached
        raise RuntimeError("existing v4 validation result belongs to another freeze")

    started = time.perf_counter()
    hidden_path, states = _ensure_fresh_hidden_bank(cfg, dirs["hidden"], manifest_sha)
    randomness_path, indices, detector_z = _ensure_validation_randomness(
        cfg, dirs["hidden"], states.shape[1], manifest_sha
    )
    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(
        cfg, data, dirs["endpoint"] / "reference_rollout.npz"
    )
    methods = {}
    actions = {}
    for name in ("Law", "Full"):
        eta = np.asarray(manifest["selected"][name]["eta"], dtype=np.float64)
        bank, mean, second, qoi = _realized_bank_and_moments(
            evaluator, eta, states, indices, detector_z
        )
        realized = evaluator.evaluate_population(
            eta, mean, second, qoi, bank, compute_full=True
        )
        predicted = manifest["selected"][name]["predicted"]
        action_values = _trial_values(realized, "full_action")
        actions[name] = action_values
        methods[name] = {
            "eta": eta.tolist(),
            "centers": eta.reshape((-1, 2)).tolist(),
            "predicted_risk": float(predicted["risk"]),
            "predicted_full_distribution": predicted["full_distribution"],
            "predicted_robust_score": float(predicted["robust_score"]),
            "realized_risk": _mean(realized, "risk"),
            "realized_full_action": _mean(realized, "full_action"),
            "realized_full_distribution": distribution(action_values),
            "prediction_to_realization_gap": {
                "risk": _mean(realized, "risk") - float(predicted["risk"]),
                "full_action": _mean(realized, "full_action") - float(predicted["mean_full"]),
            },
            "certification": _certification(realized),
            "realized": realized,
        }
    if len(actions["Law"]) != len(actions["Full"]):
        raise RuntimeError("paired v4 validation trials are not aligned")
    paired = paired_statistics(
        actions["Law"],
        actions["Full"],
        bootstrap_seed=int(cfg["seeds"]["validation_bootstrap"]),
    )
    law, full = methods["Law"], methods["Full"]
    result = {
        "schema_version": 4,
        "experiment": cfg["name"],
        "mode": cfg["mode"],
        "frozen_manifest_path": str(manifest_path.resolve()),
        "frozen_manifest_sha256": manifest_sha,
        "fresh_hidden_validation": {
            "state_bank_path": str(hidden_path.resolve()),
            "state_bank_sha256": file_sha256(hidden_path),
            "observation_randomness_path": str(randomness_path.resolve()),
            "observation_randomness_sha256": file_sha256(randomness_path),
            "physical_seed": int(cfg["seeds"]["validation_physical"]),
            "sampling_seed": int(cfg["seeds"]["validation_sampling"]),
            "detector_seed": int(cfg["seeds"]["validation_detector"]),
            "created_after_freeze": True,
            "previous_hidden_bank_reused": False,
            "common_randomness_across_law_and_full": True,
            "selection_geometry_changed": False,
        },
        "methods": methods,
        "paired_full_minus_law": paired,
        "paired_trial_difference_full_minus_law": (actions["Full"] - actions["Law"]).tolist(),
        "claims": {
            "full_realized_risk_within_allowance": bool(
                full["realized_risk"]
                <= (1.0 + float(cfg["risk_allowance"])) * law["realized_risk"]
            ),
            "predicted_full_reduction_vs_law": 1.0
            - full["predicted_full_distribution"]["mean"]
            / law["predicted_full_distribution"]["mean"],
            "realized_full_reduction_vs_law": 1.0
            - full["realized_full_action"] / law["realized_full_action"],
            "realized_full_lower_than_law": bool(
                full["realized_full_action"] < law["realized_full_action"]
            ),
            "paired_ci_excludes_zero": bool(
                paired["paired_t_95_ci"][1] < 0.0
                or paired["paired_t_95_ci"][0] > 0.0
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("frozen v4 manifest changed during hidden validation")
    _write_outputs(dirs["results"], manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate_v4(load_config(args.config), args.output_dir)
    print(json.dumps(result["claims"], indent=2))


if __name__ == "__main__":
    main()
