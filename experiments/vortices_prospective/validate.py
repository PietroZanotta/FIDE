from __future__ import annotations

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
from common import artifact_dirs, config_hash, load_config, nested_indices, write_json_atomic
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from mfsi.cache import file_sha256, fingerprint
from physical import truth_from_config
from prospective_data import TargetProspectiveData

jax.config.update("jax_enable_x64", True)


def _ensure_hidden_bank(cfg: dict[str, Any], hidden_dir: Path, frozen_manifest_sha: str) -> tuple[Path, np.ndarray]:
    hidden_dir.mkdir(parents=True, exist_ok=True)
    path = hidden_dir / "hidden_state_bank.npz"
    signature = fingerprint({
        "schema": 1,
        "config_hash": config_hash(cfg),
        "manifest_sha256_at_creation": frozen_manifest_sha,
        "particles": cfg["truth"]["hidden_validation_particles"],
        "seed": int(cfg["seed"]) + int(cfg["truth"]["hidden_validation_seed_offset"]),
    })
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["signature"]).item()) == signature:
                print("[validation] reusing sealed hidden validation state bank", flush=True)
                return path, np.asarray(data["states"], dtype=np.float64)
    truth = truth_from_config(cfg)
    times = jnp.linspace(0.0, 1.0, int(cfg["time"]["scientific_nodes"]), dtype=jnp.float64)
    bank = truth.make_bank(
        seed=int(cfg["seed"]) + int(cfg["truth"]["hidden_validation_seed_offset"]),
        n=int(cfg["truth"]["hidden_validation_particles"]),
        times=times,
        substeps_per_interval=int(cfg["truth"]["rk4_substeps_per_interval"]),
    )
    states = np.asarray(bank.particles, dtype=np.float64)
    np.savez_compressed(
        path,
        role=np.asarray("hidden_validation_microscopic_states_post_freeze_only"),
        signature=np.asarray(signature),
        manifest_sha256_at_creation=np.asarray(frozen_manifest_sha),
        times=np.asarray(times),
        states=states,
    )
    return path, states


def _ensure_hidden_randomness(cfg: dict[str, Any], hidden_dir: Path, particle_n: int):
    path = hidden_dir / "hidden_observation_randomness.npz"
    trials = int(cfg["search"]["validation_trials"])
    acq_n = int(cfg["time"]["acquisition_nodes"])
    finite_n = int(cfg["measurement"]["finite_n"])
    sensors = int(cfg["measurement"]["n_sensors"])
    expected = (trials, acq_n, finite_n)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["sample_indices"].shape) == expected and int(data["particle_n"]) == particle_n:
                return path, np.asarray(data["sample_indices"]), np.asarray(data["detector_z"])
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 7201]))
    indices = rng.integers(0, particle_n, size=expected, dtype=np.int32)
    detector = rng.standard_normal((trials, acq_n, sensors))
    np.savez_compressed(path, sample_indices=indices, detector_z=detector, particle_n=np.asarray(particle_n))
    return path, indices, detector


def _realized_bank_and_moments(evaluator, eta, states, sample_indices, detector_z):
    phi = np.asarray(evaluator.sensors.features(jnp.asarray(states), jnp.asarray(eta)), dtype=np.float64)
    response_mean = np.mean(phi, axis=1)
    response_second = np.mean(phi * phi, axis=1)
    acq_idx = np.asarray(evaluator.acq_idx, dtype=np.int32)
    phi_acq = phi[acq_idx]
    sampled = np.empty(detector_z.shape, dtype=np.float64)
    for trial in range(len(sample_indices)):
        for acq in range(len(acq_idx)):
            sampled[trial, acq] = np.mean(phi_acq[acq, sample_indices[trial, acq]], axis=0)
    acq_mean = response_mean[acq_idx]
    variance = np.maximum(response_second[acq_idx] - acq_mean * acq_mean, 0.0)
    finite_se = np.sqrt(variance / float(evaluator.cfg["measurement"]["finite_n"]))
    effective_z = np.divide(
        sampled - acq_mean[None, :, :],
        finite_se[None, :, :],
        out=np.zeros_like(sampled),
        where=finite_se[None, :, :] > 1.0e-15,
    )
    bank = AggregateObservationBank(effective_z, detector_z)
    qoi_targets = np.asarray(jnp.mean(qoi_features(jnp.asarray(states)), axis=1), dtype=np.float64)
    return bank, response_mean, response_second, qoi_targets


def _mean(block: dict[str, Any], key: str) -> float:
    value = block[key]["mean"]
    return float(value) if value is not None else float("nan")


def _write_outputs(output_dir: Path, manifest: dict, result: dict) -> None:
    write_json_atomic(output_dir / "validation_result.json", result)
    rows = []
    for method in ("Law", "Tangent", "Full"):
        for row in result["methods"][method]["realized"]["trials"]:
            rows.append({"method": method, **row})
    with (output_dir / "validation_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)

    law = result["methods"]["Law"]
    tangent = result["methods"]["Tangent"]
    full = result["methods"]["Full"]
    lines = [
        "# Prospective vortices smoke/production result",
        "",
        "Sensors were frozen before the hidden microscopic validation bank was generated or loaded.",
        "Selection used endpoints plus aggregate response/QoI predictions only.",
        "",
        "| Method | Centers | Predicted risk | Realized risk | Predicted Full | Realized Full | Valid trials |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for name, block in (("Law", law), ("Tangent", tangent), ("Full", full)):
        centers = " ".join(f"({x:.4f},{y:.4f})" for x, y in block["centers"])
        lines.append(
            f"| {name} | {centers} | {block['predicted_risk']:.6g} | {block['realized_risk']:.6g} | "
            f"{block['predicted_full_action']:.6g} | {block['realized_full_action']:.6g} | "
            f"{100.0 * block['realized']['valid_fraction']:.1f}% |"
        )
    mode_note = (
        "These are development results and are not paper-authoritative."
        if result.get("mode") == "smoke"
        else "These results use the declared production configuration."
    )
    lines.extend([
        "",
        f"Tangent realized risk constraint: **{'PASS' if result['claims']['tangent_realized_risk_within_allowance'] else 'FAIL'}**.",
        f"Tangent realized Full-action reduction versus Law: **{100.0 * result['claims']['tangent_full_action_reduction_vs_law']:.3f}%**.",
        f"Full realized risk constraint: **{'PASS' if result['claims']['full_realized_risk_within_allowance'] else 'FAIL'}**.",
        f"Realized Full-action reduction versus Law: **{100.0 * result['claims']['full_action_reduction_vs_law']:.3f}%**.",
        "",
        mode_note,
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        for name, color, marker in (
            ("Law", "#3366cc", "o"),
            ("Tangent", "#109618", "s"),
            ("Full", "#dc3912", "^"),
        ):
            centers = np.asarray(result["methods"][name]["centers"])
            ax.scatter(centers[:, 0], centers[:, 1], label=name, c=color, marker=marker, s=70)
        ax.set(xlim=(0, 2), ylim=(0, 1), xlabel="x", ylabel="y", title="Frozen sensor geometries")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "sensor_geometries.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        x = np.arange(3)
        width = 0.34
        blocks = [law, tangent, full]
        ax.bar(x - width / 2, [b["predicted_full_action"] for b in blocks], width, label="predicted")
        ax.bar(x + width / 2, [b["realized_full_action"] for b in blocks], width, label="realized")
        ax.set_xticks(x, ["Law", "Tangent", "Full"])
        ax.set_ylabel("Full action")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "predicted_vs_realized_action.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.scatter(
            [law["predicted_risk"], tangent["predicted_risk"], full["predicted_risk"]],
            [law["predicted_full_action"], tangent["predicted_full_action"], full["predicted_full_action"]],
        )
        for name, block in (("Law", law), ("Tangent", tangent), ("Full", full)):
            ax.annotate(name, (block["predicted_risk"], block["predicted_full_action"]))
        ax.set(xlabel="Predicted aggregate scientific risk", ylabel="Predicted Full action")
        fig.tight_layout()
        fig.savefig(output_dir / "risk_vs_full_action.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def validate(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dirs = artifact_dirs(output_dir)
    manifest_path = dirs["results"] / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("selection must write a frozen manifest before hidden validation")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = file_sha256(manifest_path)
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "frozen_before_hidden_validation" or manifest.get("hidden_validation_loaded") is not False:
        raise RuntimeError("manifest is not a sealed pre-validation selection")
    if manifest.get("config_hash") != config_hash(cfg):
        raise RuntimeError("validation configuration differs from the frozen selection configuration")

    result_path = dirs["results"] / "validation_result.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("frozen_manifest_sha256") == manifest_sha:
            print("[validation] reusing compatible validation result", flush=True)
            return cached

    started = time.perf_counter()
    hidden_path, states = _ensure_hidden_bank(cfg, dirs["hidden"], manifest_sha)
    randomness_path, sample_indices, detector_z = _ensure_hidden_randomness(cfg, dirs["hidden"], states.shape[1])
    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(cfg, data, dirs["endpoint"] / "reference_rollout.npz")
    methods = {}
    realized_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for name in ("Law", "Tangent", "Full"):
        eta = np.asarray(manifest["selected"][name]["eta"], dtype=np.float64)
        geometry_key = tuple(np.round(eta, 12))
        if geometry_key in realized_cache:
            realized = realized_cache[geometry_key]
        else:
            bank, response_mean, response_second, qoi_targets = _realized_bank_and_moments(
                evaluator, eta, states, sample_indices, detector_z
            )
            realized = evaluator.evaluate_population(
                eta, response_mean, response_second, qoi_targets, bank, compute_full=True
            )
            realized_cache[geometry_key] = realized
        predicted = manifest["selected"][name]["predicted"]
        methods[name] = {
            "eta": eta.tolist(),
            "centers": eta.reshape((-1, 2)).tolist(),
            "predicted_risk": _mean(predicted, "risk"),
            "predicted_full_action": _mean(predicted, "full_action"),
            "realized_risk": _mean(realized, "risk"),
            "realized_full_action": _mean(realized, "full_action"),
            "prediction_to_realization_gap": {
                "risk": _mean(realized, "risk") - _mean(predicted, "risk"),
                "full_action": _mean(realized, "full_action") - _mean(predicted, "full_action"),
            },
            "realized": realized,
        }
    law, tangent, full = methods["Law"], methods["Tangent"], methods["Full"]
    allowance = float(manifest["risk_allowance"])
    result = {
        "schema_version": 1,
        "experiment": cfg["name"],
        "mode": cfg["mode"],
        "frozen_manifest_sha256": manifest_sha,
        "hidden_validation": {
            "state_bank_sha256": file_sha256(hidden_path),
            "observation_randomness_sha256": file_sha256(randomness_path),
            "seed_namespace": 7201,
            "selection_geometry_changed": False,
        },
        "methods": methods,
        "claims": {
            "full_realized_risk_within_allowance": bool(
                full["realized_risk"] <= (1.0 + allowance) * law["realized_risk"]
            ),
            "full_action_reduction_vs_law": 1.0 - full["realized_full_action"] / law["realized_full_action"],
            "full_action_lower_than_law": bool(full["realized_full_action"] < law["realized_full_action"]),
            "tangent_realized_risk_within_allowance": bool(
                tangent["realized_risk"] <= (1.0 + allowance) * law["realized_risk"]
            ),
            "tangent_full_action_reduction_vs_law": (
                1.0 - tangent["realized_full_action"] / law["realized_full_action"]
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError("frozen manifest changed during validation")
    _write_outputs(dirs["results"], manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(load_config(args.config), args.output_dir)
    print(json.dumps(result["claims"], indent=2))


if __name__ == "__main__":
    main()
