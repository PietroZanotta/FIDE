from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.cache import file_sha256, fingerprint, load_npz_cache, save_npz_cache
from mfsi.io import write_csv, write_json
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import IProjectionConfig

from .deep_ritz import (
    CertificateConfig,
    DeepRitzConfig,
    audit_deep_ritz,
    load_ritz_checkpoint,
    save_ritz_checkpoint,
    solve_deep_ritz,
)
from .domain import ConfigurationBank, SkyrmionConfig, SkyrmionTruth
from .forcing import ForcingConfig, continuity_forcing, strict_project_trajectory
from .measurements import LocalDensitySensors, local_sensor_designs, random_sensor_designs
from .reference import (
    EquivariantReferenceFlow,
    ReferenceTrainingConfig,
    load_reference,
    save_reference,
    train_endpoint_reference,
)
from .risk import integrated_risk, many_body_features, whitening_from_truth
from .selection import BankArtifact, BankRegistry

Array = jax.Array


def _physics_config(cfg: dict[str, Any]) -> SkyrmionConfig:
    values = dict(cfg["physics"])
    for key in ("time_nodes", "truth_substeps"):
        values.pop(key, None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _time_weights(times: Array) -> Array:
    delta = jnp.diff(times)
    weights = jnp.concatenate([delta[:1] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1:] / 2])
    return weights / jnp.sum(weights)


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


_FROZEN_REFERENCE_FILES = (
    "reference.npz",
    "reference_manifest.json",
    "reference_bank_projection.npz",
    "reference_bank_ritz_train.npz",
    "reference_bank_ritz_audit.npz",
    "reference_bank_validation_fit.npz",
    "reference_bank_validation_audit.npz",
)


def _load_frozen_truth_banks(
    source_dir: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    times: Array,
) -> tuple[ConfigurationBank, ConfigurationBank, ConfigurationBank]:
    source = source_dir / "truth_banks.npz"
    if not source.exists():
        raise RuntimeError(f"frozen truth-bank artifact is missing: {source}")
    with np.load(source, allow_pickle=False) as data:
        frozen_times = np.asarray(data["times"], dtype=np.float64)
        design = np.asarray(data["design"], dtype=np.float64)
        validation = np.asarray(data["validation"], dtype=np.float64)
        endpoint0 = np.asarray(data["endpoint0"], dtype=np.float64)
        endpoint1 = np.asarray(data["endpoint1"], dtype=np.float64)
    expected_times = np.asarray(times, dtype=np.float64)
    expected_particles = int(cfg["physics"]["n_particles"])
    expected_shapes = (
        (len(expected_times), int(cfg["banks"]["truth_design_samples"]), expected_particles, 2),
        (len(expected_times), int(cfg["banks"]["truth_validation_samples"]), expected_particles, 2),
        (int(cfg["banks"]["endpoint_samples"]), expected_particles, 2),
    )
    if not np.array_equal(frozen_times, expected_times):
        raise RuntimeError("frozen truth-bank time grid is incompatible with this Pareto point")
    if design.shape != expected_shapes[0] or validation.shape != expected_shapes[1]:
        raise RuntimeError("frozen truth-bank sample shape is incompatible with this Pareto point")
    if endpoint0.shape != expected_shapes[2] or endpoint1.shape != expected_shapes[2]:
        raise RuntimeError("frozen endpoint-bank shape is incompatible with this Pareto point")
    destination = output_dir / "truth_banks.npz"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return (
        ConfigurationBank(jnp.asarray([0.0, 1.0]), jnp.asarray(np.stack([endpoint0, endpoint1]))),
        ConfigurationBank(jnp.asarray(frozen_times), jnp.asarray(design)),
        ConfigurationBank(jnp.asarray(frozen_times), jnp.asarray(validation)),
    )


def _install_frozen_reference_artifacts(source_dir: Path, output_dir: Path) -> dict[str, str]:
    expected_hashes: dict[str, str] = {}
    for name in _FROZEN_REFERENCE_FILES:
        source = source_dir / name
        if not source.exists():
            raise RuntimeError(f"frozen reference artifact is missing: {source}")
        expected_hashes[name] = file_sha256(source)
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    return expected_hashes


def _ensure_reference(
    output_dir: Path,
    endpoints: ConfigurationBank,
    cfg: dict[str, Any],
    box: tuple[float, float],
) -> tuple[EquivariantReferenceFlow, list[dict[str, float]]]:
    training = ReferenceTrainingConfig(**cfg["reference_training"])
    signature = fingerprint({
        "training": asdict(training), "box": box,
        "endpoint_shape": list(endpoints.configurations.shape),
        "endpoint_digest": fingerprint(np.asarray(endpoints.configurations[:, : min(16, endpoints.configurations.shape[1])])),
    })
    checkpoint = output_dir / "reference.npz"
    manifest_path = output_dir / "reference_manifest.json"
    if checkpoint.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            return load_reference(checkpoint), manifest.get("history", [])
    flow, history = train_endpoint_reference(
        endpoints.configurations[0], endpoints.configurations[-1], training, box=box
    )
    save_reference(checkpoint, flow)
    write_json(manifest_path, {
        "signature": signature,
        "endpoint_only": True,
        "frozen_for_all_designs": True,
        "history": history,
    })
    return flow, history


def _reference_bank(
    flow: EquivariantReferenceFlow,
    endpoint0: Array,
    times: Array,
    *,
    seed: int,
    samples: int,
    substeps: int,
) -> dict[str, Array]:
    key = jax.random.PRNGKey(int(seed))
    indices = jax.random.randint(key, (int(samples),), 0, endpoint0.shape[0])
    configurations = flow.rollout(endpoint0[indices], times, substeps_per_interval=int(substeps))
    velocity = flow.velocity(configurations, times)
    weights = jnp.full((len(times), int(samples)), 1.0 / float(samples), dtype=jnp.float64)
    return {"configurations": configurations, "velocity": velocity, "base_weights": weights}


def _ensure_reference_bank(
    output_dir: Path,
    name: str,
    flow: EquivariantReferenceFlow,
    endpoint0: Array,
    times: Array,
    *,
    seed: int,
    samples: int,
    substeps: int,
    reference_hash: str,
) -> dict[str, Array]:
    signature = fingerprint({
        "schema": 1, "role": name, "seed": int(seed), "samples": int(samples),
        "substeps": int(substeps), "times": np.asarray(times),
        "reference_hash": reference_hash,
        "endpoint0_digest": fingerprint(np.asarray(endpoint0[: min(16, endpoint0.shape[0])])),
    })
    path = output_dir / f"reference_bank_{name}.npz"
    cached = load_npz_cache(path, signature=signature)
    if cached is not None:
        arrays, _ = cached
        return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in arrays.items()}
    bank = _reference_bank(
        flow, endpoint0, times, seed=seed, samples=samples, substeps=substeps
    )
    save_npz_cache(
        path, bank, signature=signature,
        metadata={"role": name, "seed": int(seed), "frozen_reference_hash": reference_hash},
    )
    return bank


def _moment_reconstruction(
    eta: Array,
    truth: ConfigurationBank,
    family: LocalDensitySensors,
    cfg: dict[str, Any],
    *,
    noise_seed: int,
) -> tuple[Array, Array, dict[str, Any]]:
    measurement = cfg["measurement"]
    count = int(measurement["acquisition_count"])
    indices = np.unique(np.rint(np.linspace(0, len(truth.times) - 1, count)).astype(int))
    if len(indices) < 2:
        raise ValueError("acquisition grid must contain at least two distinct times")
    phi = family.features(truth.configurations, eta)
    finite_n = min(int(measurement["finite_configurations"]), int(phi.shape[1]))
    observations = jnp.mean(phi[jnp.asarray(indices), :finite_n, :], axis=1)
    noise = float(measurement["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(int(noise_seed)), observations.shape, dtype=jnp.float64
    )
    observations = observations + noise
    endpoint0 = jnp.mean(phi[0], axis=0)
    endpoint1 = jnp.mean(phi[-1], axis=0)
    recon_cfg = AnchoredCubicSplineConfig(**cfg["moment_reconstruction"])
    reconstructor = AnchoredCubicSplineReconstructor(
        np.asarray(truth.times)[indices], np.asarray(truth.times), recon_cfg
    )
    result = reconstructor.reconstruct(observations, endpoint0, endpoint1)
    return result.c, result.c_dot, {
        "acquisition_indices": indices.tolist(),
        "finite_configurations": finite_n,
        "residual_sum_squares": float(result.residual_sum_squares),
        "roughness": float(result.roughness),
    }


def _projection_config(cfg: dict[str, Any]) -> IProjectionConfig:
    values = dict(cfg["projection"])
    values.pop("trajectory_backend", None)
    return IProjectionConfig(**values)


def _projection_backend(cfg: dict[str, Any]) -> str:
    return str(cfg["projection"].get("trajectory_backend", "jax"))


def _selection_risk(
    eta: Array,
    truth: ConfigurationBank,
    reference: dict[str, Array],
    family: LocalDensitySensors,
    reference_features: Array,
    truth_feature_means: Array,
    whitening: Array,
    time_weights: Array,
    cfg: dict[str, Any],
    *,
    noise_seed: int,
) -> dict[str, Any]:
    try:
        targets, derivatives, reconstruction = _moment_reconstruction(
            eta, truth, family, cfg, noise_seed=noise_seed
        )
        phi = family.features(reference["configurations"], eta)
        projection = strict_project_trajectory(
            phi,
            reference["base_weights"],
            targets,
            projection_cfg=_projection_config(cfg),
            tolerance=float(cfg["forcing"]["projection_tolerance"]),
            trajectory_backend=_projection_backend(cfg),
        )
        risk = float(integrated_risk(
            projection.weights, reference_features, truth_feature_means, whitening, time_weights
        ))
        max_residual = float(jnp.max(jnp.linalg.norm(projection.residual, axis=-1)))
        min_ess = float(jnp.min(projection.ess_fraction))
        valid = (
            np.isfinite(risk)
            and min_ess >= float(cfg["forcing"]["minimum_ess_fraction"])
        )
        return {
            "risk": risk, "valid": valid, "targets": targets, "derivatives": derivatives,
            "max_projection_residual": max_residual, "minimum_ess_fraction": min_ess,
            "reconstruction": reconstruction,
        }
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return {"risk": float("inf"), "valid": False, "failure_reason": str(exc)}


def _forcing_for_bank(
    eta: Array,
    targets: Array,
    derivatives: Array,
    bank: dict[str, Array],
    family: LocalDensitySensors,
    cfg: dict[str, Any],
):
    return continuity_forcing(
        bank["configurations"], bank["velocity"], bank["base_weights"],
        targets, derivatives, eta, family,
        projection_cfg=_projection_config(cfg),
        cfg=ForcingConfig(**cfg["forcing"]),
        fail_loudly=True,
        projection_backend=_projection_backend(cfg),
    )


def _authoritative_candidate(
    eta: Array,
    risk_row: dict[str, Any],
    train_bank: dict[str, Array],
    audit_bank: dict[str, Array],
    family: LocalDensitySensors,
    times: Array,
    time_weights: Array,
    cfg: dict[str, Any],
    *,
    seed_offset: int = 0,
    initial_params=None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        targets, derivatives = risk_row["targets"], risk_row["derivatives"]
        train_forcing = _forcing_for_bank(eta, targets, derivatives, train_bank, family, cfg)
        audit_forcing = _forcing_for_bank(eta, targets, derivatives, audit_bank, family, cfg)
        deep_cfg_values = dict(cfg["deep_ritz"])
        max_restarts = int(deep_cfg_values.pop("authoritative_restarts", 1))
        polish_iterations = int(deep_cfg_values.pop("certification_polish_iterations", 0))
        base_seed = int(deep_cfg_values["seed"]) + int(seed_offset)
        deep_cfg_values["box"] = tuple(family.box)
        certificate_cfg = CertificateConfig(**cfg["certificates"])
        attempts: list[dict[str, Any]] = []
        selected_solve = None
        selected_certificate = None
        selected_score = float("inf")
        for restart in range(max(1, max_restarts)):
            restart_values = dict(deep_cfg_values)
            restart_values["seed"] = base_seed + 10007 * restart
            deep_cfg = DeepRitzConfig(**restart_values)
            solve = solve_deep_ritz(
                train_bank["configurations"], train_forcing.projection.weights,
                train_forcing.forcing, times, time_weights, deep_cfg,
                initial_params=initial_params if restart == 0 else None,
            )
            certificate = audit_deep_ritz(
                solve.params,
                audit_bank["configurations"], audit_forcing.projection.weights,
                audit_forcing.forcing, times, time_weights,
                family=family, eta=eta, reference_velocity=audit_bank["velocity"],
                target_derivatives=derivatives,
                cfg=certificate_cfg, box=tuple(family.box),
            )
            threshold = certificate["thresholds"]
            score = max(
                certificate["maximum_weak_residual"] / threshold["maximum_weak_residual"],
                certificate["maximum_energy_residual"] / threshold["maximum_energy_residual"],
                certificate["maximum_gauge_residual"] / threshold["maximum_gauge_residual"],
                certificate["maximum_moment_rate_residual"] / threshold["maximum_moment_rate_residual"],
            )
            optimization = {
                "restart": restart,
                "seed": restart_values["seed"],
                "adam_final_objective": solve.adam_final_objective,
                "lbfgs_final_objective": solve.lbfgs_final_objective,
                "adam_seconds": solve.adam_seconds,
                "lbfgs_seconds": solve.lbfgs_seconds,
                "lbfgs_converged": solve.lbfgs_converged,
                "finite": solve.finite,
                "history": solve.history,
                "certificate": certificate,
            }
            attempts.append(optimization)
            if score < selected_score:
                selected_solve, selected_certificate, selected_score = solve, certificate, score
            if solve.finite and certificate["valid"]:
                selected_solve, selected_certificate = solve, certificate
                break
        # A near-certified solve is continued deterministically on the same
        # full training objective.  The audit bank is never used by L-BFGS;
        # it only decides whether the continued iterate is acceptable.  This
        # implements the experiment's fail-closed "increase inner fidelity"
        # policy without spending another Adam basin search.
        if (
            selected_solve is not None
            and selected_certificate is not None
            and not selected_certificate["valid"]
            and polish_iterations > 0
        ):
            restart_values = dict(deep_cfg_values)
            restart_values["seed"] = base_seed + 10007 * max(1, max_restarts)
            restart_values["adam_steps"] = 0
            restart_values["lbfgs_iterations"] = polish_iterations
            deep_cfg = DeepRitzConfig(**restart_values)
            solve = solve_deep_ritz(
                train_bank["configurations"], train_forcing.projection.weights,
                train_forcing.forcing, times, time_weights, deep_cfg,
                initial_params=selected_solve.params,
            )
            certificate = audit_deep_ritz(
                solve.params,
                audit_bank["configurations"], audit_forcing.projection.weights,
                audit_forcing.forcing, times, time_weights,
                family=family, eta=eta, reference_velocity=audit_bank["velocity"],
                target_derivatives=derivatives,
                cfg=certificate_cfg, box=tuple(family.box),
            )
            threshold = certificate["thresholds"]
            score = max(
                certificate["maximum_weak_residual"] / threshold["maximum_weak_residual"],
                certificate["maximum_energy_residual"] / threshold["maximum_energy_residual"],
                certificate["maximum_gauge_residual"] / threshold["maximum_gauge_residual"],
                certificate["maximum_moment_rate_residual"] / threshold["maximum_moment_rate_residual"],
            )
            optimization = {
                "phase": "certification_polish",
                "restart": max(1, max_restarts),
                "seed": restart_values["seed"],
                "adam_final_objective": solve.adam_final_objective,
                "lbfgs_final_objective": solve.lbfgs_final_objective,
                "adam_seconds": solve.adam_seconds,
                "lbfgs_seconds": solve.lbfgs_seconds,
                "lbfgs_converged": solve.lbfgs_converged,
                "finite": solve.finite,
                "history": solve.history,
                "certificate": certificate,
            }
            attempts.append(optimization)
            if score < selected_score or (solve.finite and certificate["valid"]):
                selected_solve, selected_certificate, selected_score = solve, certificate, score
        assert selected_solve is not None and selected_certificate is not None
        solve, certificate = selected_solve, selected_certificate
        maximum_projection = max(
            float(jnp.max(jnp.linalg.norm(train_forcing.projection.residual, axis=-1))),
            float(jnp.max(jnp.linalg.norm(audit_forcing.projection.residual, axis=-1))),
        )
        minimum_ess = min(
            float(jnp.min(train_forcing.projection.ess_fraction)),
            float(jnp.min(audit_forcing.projection.ess_fraction)),
        )
        maximum_forcing_mean = max(
            float(jnp.max(jnp.abs(train_forcing.forcing_mean_before_centering))),
            float(jnp.max(jnp.abs(audit_forcing.forcing_mean_before_centering))),
        )
        valid = bool(solve.finite and certificate["valid"])
        return {
            "eta": np.asarray(eta).tolist(), "risk": float(risk_row["risk"]),
            "action": float(certificate["action"]), "valid": valid,
            "maximum_projection_residual": maximum_projection,
            "minimum_ess_fraction": minimum_ess,
            "maximum_forcing_mean_residual": maximum_forcing_mean,
            "certificate": certificate,
            "optimization": {
                "adam_final_objective": solve.adam_final_objective,
                "lbfgs_final_objective": solve.lbfgs_final_objective,
                "adam_seconds": solve.adam_seconds,
                "lbfgs_seconds": solve.lbfgs_seconds,
                "lbfgs_converged": solve.lbfgs_converged,
                "finite": solve.finite,
                "history": solve.history,
                "attempts": attempts,
            },
            "params": solve.params,
            "total_seconds": time.perf_counter() - started,
        }
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return {
            "eta": np.asarray(eta).tolist(), "risk": float(risk_row.get("risk", float("inf"))),
            "action": float("inf"), "valid": False, "failure_reason": str(exc),
            "total_seconds": time.perf_counter() - started,
        }


def _reaudit_cached_candidate(
    cached_result: dict[str, Any],
    params: Any,
    eta: Array,
    risk_row: dict[str, Any],
    train_bank: dict[str, Array],
    audit_bank: dict[str, Array],
    family: LocalDensitySensors,
    times: Array,
    time_weights: Array,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Re-certify a compatible cached state against the current frozen data.

    The outer search can reproduce an otherwise identical candidate with
    last-bit differences in reconstructed targets. In that case the exact
    content signature changes even though the cached network remains a useful
    solution. We only reuse it after rebuilding both forcings and passing the
    full independent audit for the current candidate.
    """

    started = time.perf_counter()
    targets, derivatives = risk_row["targets"], risk_row["derivatives"]
    train_forcing = _forcing_for_bank(eta, targets, derivatives, train_bank, family, cfg)
    audit_forcing = _forcing_for_bank(eta, targets, derivatives, audit_bank, family, cfg)
    certificate = audit_deep_ritz(
        params,
        audit_bank["configurations"],
        audit_forcing.projection.weights,
        audit_forcing.forcing,
        times,
        time_weights,
        family=family,
        eta=eta,
        reference_velocity=audit_bank["velocity"],
        target_derivatives=derivatives,
        cfg=CertificateConfig(**cfg["certificates"]),
        box=tuple(family.box),
    )
    result = dict(cached_result)
    result.update({
        "eta": np.asarray(eta).tolist(),
        "risk": float(risk_row["risk"]),
        "action": float(certificate["action"]),
        "valid": bool(certificate["valid"]),
        "maximum_projection_residual": max(
            float(jnp.max(jnp.linalg.norm(train_forcing.projection.residual, axis=-1))),
            float(jnp.max(jnp.linalg.norm(audit_forcing.projection.residual, axis=-1))),
        ),
        "minimum_ess_fraction": min(
            float(jnp.min(train_forcing.projection.ess_fraction)),
            float(jnp.min(audit_forcing.projection.ess_fraction)),
        ),
        "maximum_forcing_mean_residual": max(
            float(jnp.max(jnp.abs(train_forcing.forcing_mean_before_centering))),
            float(jnp.max(jnp.abs(audit_forcing.forcing_mean_before_centering))),
        ),
        "certificate": certificate,
        "params": params,
        "total_seconds": (
            float(cached_result.get("total_seconds", 0.0))
            + time.perf_counter() - started
        ),
        "cache_reaudited": True,
    })
    return result


def _serializable_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "params"}


def _serializable_search_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items()
        if key not in {"targets", "derivatives"}
    }


def _normalized_certificate_score(row: dict[str, Any]) -> float:
    certificate = row.get("certificate", {})
    thresholds = certificate.get("thresholds", {})
    ratios = []
    for name in (
        "maximum_weak_residual",
        "maximum_energy_residual",
        "maximum_gauge_residual",
        "maximum_moment_rate_residual",
    ):
        value = float(certificate.get(name, float("inf")))
        threshold = float(thresholds.get(name, 0.0))
        ratios.append(value / threshold if threshold > 0.0 else float("inf"))
    return max(ratios)


def _selection_signature_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return only configuration that can change an authoritative action solve.

    Validation-bank sizes and outer-search budgets do not alter a candidate's
    frozen targets, forcing, Ritz training bank, or audit bank.  Excluding them
    lets a validation-only fidelity increase reuse expensive selection solves,
    while changes to the actual solver or scientific banks invalidate caches.
    """

    signature_cfg = deepcopy(cfg)
    signature_cfg.pop("execution_profile", None)
    signature_cfg.pop("preflight", None)
    signature_cfg.pop("smoke", None)
    signature_cfg.pop("runtime", None)
    signature_cfg.pop("search", None)
    banks = signature_cfg.get("banks", {})
    for key in (
        "truth_validation_samples",
        "validation_fit_samples",
        "validation_audit_samples",
    ):
        banks.pop(key, None)
    return signature_cfg


def _validation_signature_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return configuration that determines a disjoint validation solve."""

    signature_cfg = deepcopy(cfg)
    signature_cfg.pop("execution_profile", None)
    signature_cfg.pop("preflight", None)
    signature_cfg.pop("smoke", None)
    signature_cfg.pop("runtime", None)
    signature_cfg.pop("search", None)
    return signature_cfg


def run_experiment(
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    smoke: bool = False,
    shared_candidate_cache: Path | None = None,
    shared_validation_cache: Path | None = None,
    frozen_artifact_source: Path | None = None,
) -> dict[str, Any]:
    """Run the Law-vs-certified-Full 3% milestone (never a Tangent decomposition)."""

    output_dir = Path(output_dir).resolve()
    isolated_root = (Path(__file__).resolve().parent / "outputs").resolve()
    if output_dir != isolated_root and isolated_root not in output_dir.parents:
        raise ValueError(
            f"isolated experiment output must be beneath {isolated_root}, got {output_dir}"
        )
    for label, cache_path in (
        ("shared_candidate_cache", shared_candidate_cache),
        ("shared_validation_cache", shared_validation_cache),
    ):
        if cache_path is None:
            continue
        resolved_cache = Path(cache_path).resolve()
        if resolved_cache != isolated_root and isolated_root not in resolved_cache.parents:
            raise ValueError(
                f"{label} must be beneath {isolated_root}, got {resolved_cache}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()
    stage_times: dict[str, float] = {}
    seed = int(cfg["seed"])
    offsets = cfg["banks"]["seed_offsets"]
    physics = _physics_config(cfg)
    truth_model = SkyrmionTruth(physics)
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    time_weights = _time_weights(times)
    family = LocalDensitySensors(
        int(cfg["measurement"]["n_sensors"]), float(cfg["measurement"]["sensor_width"]),
        tuple(physics.box), float(cfg["measurement"]["min_separation"]),
    )

    started = time.perf_counter()
    if frozen_artifact_source is not None:
        endpoints, truth_design, truth_validation = _load_frozen_truth_banks(
            Path(frozen_artifact_source), output_dir, cfg, times
        )
    else:
        endpoints = truth_model.make_bank(
            seed=seed + int(offsets["endpoints"]), samples=int(cfg["banks"]["endpoint_samples"]),
            times=jnp.asarray([0.0, 1.0]),
            substeps_per_interval=int(cfg["physics"]["truth_substeps"]) * max(len(times) - 1, 1),
        )
        truth_design = truth_model.make_bank(
            seed=seed + int(offsets["truth_design"]), samples=int(cfg["banks"]["truth_design_samples"]),
            times=times, substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
        )
        truth_validation = truth_model.make_bank(
            seed=seed + int(offsets["truth_validation"]), samples=int(cfg["banks"]["truth_validation_samples"]),
            times=times, substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
        )
        np.savez_compressed(
            output_dir / "truth_banks.npz", times=np.asarray(times),
            design=np.asarray(truth_design.configurations), validation=np.asarray(truth_validation.configurations),
            endpoint0=np.asarray(endpoints.configurations[0]), endpoint1=np.asarray(endpoints.configurations[-1]),
        )
    stage_times["truth_generation"] = time.perf_counter() - started

    started = time.perf_counter()
    frozen_hashes = None
    if frozen_artifact_source is not None:
        frozen_hashes = _install_frozen_reference_artifacts(
            Path(frozen_artifact_source), output_dir
        )
    reference, reference_history = _ensure_reference(output_dir, endpoints, cfg, tuple(physics.box))
    stage_times["reference_training"] = time.perf_counter() - started
    bank_cfg = cfg["banks"]
    substeps = int(bank_cfg["reference_substeps"])
    reference_hash = file_sha256(output_dir / "reference.npz")
    reference_banks = {
        name: _ensure_reference_bank(
            output_dir, name,
            reference, endpoints.configurations[0], times,
            seed=seed + int(offsets[name]), samples=int(bank_cfg[f"{name}_samples"]),
            substeps=substeps, reference_hash=reference_hash,
        )
        for name in ("projection", "ritz_train", "ritz_audit", "validation_fit", "validation_audit")
    }
    if frozen_hashes is not None:
        mismatches = [
            name for name, expected in frozen_hashes.items()
            if file_sha256(output_dir / name) != expected
        ]
        if mismatches:
            raise RuntimeError(
                "frozen Pareto reference artifacts were incompatible and would have been regenerated: "
                + ", ".join(mismatches)
            )
    registry = BankRegistry({
        "truth_design": BankArtifact("design", f"truth-design-{seed + int(offsets['truth_design'])}", truth_design),
        "projection": BankArtifact("projection", f"projection-{seed + int(offsets['projection'])}", reference_banks["projection"]),
        "ritz_train": BankArtifact("ritz_optimization", f"ritz-train-{seed + int(offsets['ritz_train'])}", reference_banks["ritz_train"]),
        "ritz_audit": BankArtifact("ritz_audit", f"ritz-audit-{seed + int(offsets['ritz_audit'])}", reference_banks["ritz_audit"]),
        "truth_validation": BankArtifact("validation", f"truth-validation-{seed + int(offsets['truth_validation'])}", truth_validation),
        "validation_fit": BankArtifact("validation", f"validation-fit-{seed + int(offsets['validation_fit'])}", reference_banks["validation_fit"]),
        "validation_audit": BankArtifact("validation", f"validation-audit-{seed + int(offsets['validation_audit'])}", reference_banks["validation_audit"]),
    })
    stage_times["reference_rollout"] = time.perf_counter() - started

    truth_features = many_body_features(truth_design.configurations, tuple(physics.box))
    truth_means = jnp.mean(truth_features, axis=1)
    whitening = whitening_from_truth(truth_features)
    selection_reference = registry.get("projection", consumer="selection")
    selection_ref_features = many_body_features(selection_reference["configurations"], tuple(physics.box))

    started = time.perf_counter()
    candidates = random_sensor_designs(
        jax.random.PRNGKey(seed + int(offsets["candidate_pool"])),
        count=int(cfg["search"]["candidate_count"]), family=family,
    )
    mandatory = cfg.get("search", {}).get("mandatory_etas", [])
    if mandatory:
        mandatory_a = jnp.asarray(mandatory, dtype=jnp.float64).reshape(-1, 2 * family.n_sensors)
        candidates = jnp.concatenate([mandatory_a, candidates], axis=0)
    risk_rows: list[dict[str, Any]] = []
    seen_designs: set[tuple[float, ...]] = set()

    # Cross-bank support screening uses the projection and Ritz-optimization
    # banks only.  Screen before a design may become a Law-refinement center,
    # so the convergence rule tracks the same supported anchor reported later.
    train_bank = registry.get("ritz_train", consumer="selection")

    def screen_support(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            try:
                projection_forcing = _forcing_for_bank(
                    jnp.asarray(row["eta"]), row["targets"], row["derivatives"],
                    selection_reference, family, cfg,
                )
                train_forcing = _forcing_for_bank(
                    jnp.asarray(row["eta"]), row["targets"], row["derivatives"],
                    train_bank, family, cfg,
                )
                row["search_action_proxy"] = float(jnp.sum(
                    time_weights * jnp.einsum(
                        "tn,tn,tn->t",
                        projection_forcing.projection.weights,
                        projection_forcing.forcing,
                        projection_forcing.forcing,
                    )
                ))
                row["support_minimum_ess_fraction"] = min(
                    float(jnp.min(projection_forcing.projection.ess_fraction)),
                    float(jnp.min(train_forcing.projection.ess_fraction)),
                )
                row["support_valid"] = True
            except RuntimeError as exc:
                row["search_action_proxy"] = float("inf")
                row["support_valid"] = False
                row["support_failure_reason"] = str(exc)

    def evaluate_designs(designs: Array) -> None:
        for eta in designs:
            key = tuple(np.round(np.asarray(eta, dtype=np.float64), 12))
            if key in seen_designs:
                continue
            seen_designs.add(key)
            index = len(risk_rows)
            row = _selection_risk(
                eta, registry.get("truth_design", consumer="selection"), selection_reference,
                family, selection_ref_features, truth_means, whitening, time_weights, cfg,
                noise_seed=seed + int(offsets["observation"]),
            )
            row.update({"id": f"candidate-{index:04d}", "eta": np.asarray(eta).tolist()})
            risk_rows.append(row)
            if row["valid"] and np.isfinite(row["risk"]):
                screen_support([row])

    evaluate_designs(candidates)
    search_cfg = cfg["search"]
    minimum_law_rounds = int(search_cfg.get("law_refinement_rounds", 2))
    maximum_law_rounds = max(
        minimum_law_rounds,
        int(search_cfg.get("law_refinement_max_rounds", minimum_law_rounds)),
    )
    stable_law_rounds_required = int(search_cfg.get("law_refinement_stable_rounds", 1))
    law_improvement_tolerance = float(search_cfg.get("law_refinement_relative_tolerance", 0.0))
    stable_law_rounds = 0
    law_refinement_trace: list[dict[str, Any]] = []
    for round_index in range(maximum_law_rounds):
        current = sorted(
            (
                row for row in risk_rows
                if row["valid"] and row.get("support_valid") and np.isfinite(row["risk"])
            ),
            key=lambda row: (row["risk"], row["id"]),
        )
        if not current:
            break
        risk_before = float(current[0]["risk"])
        center_count = min(int(search_cfg.get("law_refinement_centers", 6)), len(current))
        centers = jnp.asarray([row["eta"] for row in current[:center_count]], dtype=jnp.float64)
        local = local_sensor_designs(
            jax.random.fold_in(
                jax.random.PRNGKey(seed + int(offsets["candidate_pool"])), round_index + 1
            ),
            centers,
            count_per_center=int(search_cfg.get("law_local_count_per_center", 8)),
            scale=float(search_cfg.get("law_local_scale", 0.08)) / float(round_index + 1),
            family=family,
        )
        evaluate_designs(local)
        current_after = sorted(
            (
                row for row in risk_rows
                if row["valid"] and row.get("support_valid") and np.isfinite(row["risk"])
            ),
            key=lambda row: (row["risk"], row["id"]),
        )
        risk_after = float(current_after[0]["risk"])
        relative_improvement = max(0.0, (risk_before - risk_after) / max(risk_before, 1.0e-12))
        stable_law_rounds = (
            stable_law_rounds + 1
            if relative_improvement <= law_improvement_tolerance else 0
        )
        law_refinement_trace.append({
            "round": round_index + 1,
            "risk_before": risk_before,
            "risk_after": risk_after,
            "relative_improvement": relative_improvement,
            "stable_rounds": stable_law_rounds,
        })
        if (
            round_index + 1 >= minimum_law_rounds
            and stable_law_rounds >= stable_law_rounds_required
        ):
            break

    if (
        maximum_law_rounds > 0
        and stable_law_rounds < stable_law_rounds_required
    ):
        write_json(
            output_dir / "search_diagnostics.json",
            {
                "stage": "law_refinement_not_converged",
                "law_refinement_trace": law_refinement_trace,
                "rows": [_serializable_search_row(row) for row in risk_rows],
            },
        )
        raise RuntimeError(
            "Law-risk refinement did not reach its predeclared stability rule "
            f"within {maximum_law_rounds} rounds"
        )

    valid_risks = [row for row in risk_rows if row["valid"] and np.isfinite(row["risk"])]
    if not valid_risks:
        write_json(
            output_dir / "search_diagnostics.json",
            {"stage": "risk", "rows": [_serializable_search_row(row) for row in risk_rows]},
        )
        raise RuntimeError(
            "no design passed hard I-projection/risk screening: "
            + repr([row.get("failure_reason", {
                "risk": row.get("risk"), "minimum_ess_fraction": row.get("minimum_ess_fraction")
            }) for row in risk_rows])
        )

    supported_risks = [row for row in valid_risks if row.get("support_valid")]
    if not supported_risks:
        write_json(
            output_dir / "search_diagnostics.json",
            {"stage": "support", "rows": [_serializable_search_row(row) for row in risk_rows]},
        )
        raise RuntimeError("no design passed cross-bank projection/ESS/forcing support screening")
    # Risk refinement locates the Law anchor.  This second, distinct refinement
    # explores low-action proxy geometries inside/near its 3% band.
    allowance = float(cfg["search"]["risk_allowance_percent"])
    for round_index in range(int(search_cfg.get("full_refinement_rounds", 3))):
        law_now = min(supported_risks, key=lambda row: (row["risk"], row["id"]))
        exploration_limit = (
            1.0
            + float(search_cfg.get("full_refinement_risk_multiplier", 2.0))
            * allowance / 100.0
        ) * float(law_now["risk"])
        near_band = [row for row in supported_risks if row["risk"] <= exploration_limit]
        near_band.sort(key=lambda row: (row["search_action_proxy"], row["risk"], row["id"]))
        if not near_band:
            break
        center_count = min(int(search_cfg.get("full_refinement_centers", 6)), len(near_band))
        centers = jnp.asarray([row["eta"] for row in near_band[:center_count]], dtype=jnp.float64)
        before = len(risk_rows)
        local = local_sensor_designs(
            jax.random.fold_in(
                jax.random.PRNGKey(seed + int(offsets["candidate_pool"])),
                1000 + round_index,
            ),
            centers,
            count_per_center=int(search_cfg.get("full_local_count_per_center", 8)),
            scale=float(search_cfg.get("full_local_scale", 0.06)) / float(round_index + 1),
            family=family,
        )
        evaluate_designs(local)
        new_valid = [
            row for row in risk_rows[before:]
            if row["valid"] and np.isfinite(row["risk"])
        ]
        supported_risks.extend(row for row in new_valid if row.get("support_valid"))

    write_json(
        output_dir / "search_diagnostics.json",
        {
            "stage": "support",
            "law_refinement_trace": law_refinement_trace,
            "rows": [_serializable_search_row(row) for row in risk_rows],
        },
    )
    fixed_law_eta = search_cfg.get("fixed_law_eta")
    fixed_law_risk = search_cfg.get("fixed_law_risk")
    if (fixed_law_eta is None) != (fixed_law_risk is None):
        raise RuntimeError("fixed_law_eta and fixed_law_risk must be provided together")
    if fixed_law_eta is not None:
        expected_eta = np.asarray(fixed_law_eta, dtype=np.float64)
        fixed_matches = [
            row for row in supported_risks
            if np.allclose(
                np.asarray(row["eta"], dtype=np.float64), expected_eta,
                rtol=0.0, atol=1.0e-13,
            )
        ]
        if not fixed_matches:
            raise RuntimeError("frozen Law design did not pass current support screening")
        law_risk_row = fixed_matches[0]
        anchor = float(fixed_law_risk)
        if not np.isclose(
            float(law_risk_row["risk"]), anchor, rtol=0.0, atol=1.0e-9
        ):
            raise RuntimeError(
                "frozen Law risk does not reproduce on the frozen scientific banks"
            )
    else:
        law_risk_row = min(supported_risks, key=lambda row: (row["risk"], row["id"]))
        anchor = float(law_risk_row["risk"])
    if anchor <= 0.0:
        raise RuntimeError("Law anchor risk must be strictly positive")
    risk_limit = (1.0 + allowance / 100.0) * anchor
    feasible = [row for row in supported_risks if row["risk"] <= risk_limit]
    stage_times["risk_search"] = time.perf_counter() - started

    # The precomputed forcing-norm proxy only ranks the feasible shortlist.  It
    # is never reported as Full action and cannot win without a fresh solve.
    started = time.perf_counter()
    feasible.sort(key=lambda row: (row["search_action_proxy"], row["risk"], row["id"]))
    limit = int(cfg["search"]["authoritative_candidates"])
    shortlisted = feasible[:limit]
    mandatory_keys = {
        tuple(np.round(np.asarray(eta, dtype=np.float64), 12)) for eta in mandatory
    }
    for row in feasible:
        if tuple(np.round(np.asarray(row["eta"], dtype=np.float64), 12)) in mandatory_keys and row not in shortlisted:
            shortlisted.append(row)
    if law_risk_row not in shortlisted:
        shortlisted.append(law_risk_row)
    stage_times["action_proxy"] = time.perf_counter() - started

    audit_bank = registry.get("ritz_audit", consumer="selection")
    authoritative: list[dict[str, Any]] = []
    started = time.perf_counter()
    candidate_cache = output_dir / "authoritative_candidates"
    candidate_cache.mkdir(parents=True, exist_ok=True)
    if shared_candidate_cache is not None:
        shared_candidate_cache.mkdir(parents=True, exist_ok=True)
    selection_signature_cfg = _selection_signature_config(cfg)
    for index, row in enumerate(shortlisted):
        selection_polish_iterations = int(
            cfg["search"].get("selection_polish_iterations", 0)
        )
        signature = fingerprint({
            "schema": 3,
            "config": selection_signature_cfg,
            "eta": row["eta"],
            "risk": row["risk"],
            "search_action_proxy": row["search_action_proxy"],
            "targets": np.asarray(row["targets"]),
            "derivatives": np.asarray(row["derivatives"]),
            "train_shape": list(train_bank["configurations"].shape),
            "audit_shape": list(audit_bank["configurations"].shape),
        })
        local_json_path = candidate_cache / f"{row['id']}.json"
        local_params_path = candidate_cache / f"{row['id']}.npz"
        cache_locations = [(local_json_path, local_params_path)]
        if shared_candidate_cache is not None:
            cache_locations.append((
                shared_candidate_cache / f"{signature}.json",
                shared_candidate_cache / f"{signature}.npz",
            ))
            cache_locations.extend(
                (path, path.with_suffix(".npz"))
                for path in sorted(shared_candidate_cache.glob("*.json"))
                if path.name != f"{signature}.json"
            )
        result = None
        loaded_from: tuple[Path, Path] | None = None
        for json_path, params_path in cache_locations:
            if not json_path.exists() or not params_path.exists():
                continue
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            cached_result = cached.get("result", {})
            exact_scientific_match = (
                np.allclose(
                    np.asarray(cached_result.get("eta", []), dtype=np.float64),
                    np.asarray(row["eta"], dtype=np.float64),
                    rtol=0.0,
                    atol=1.0e-13,
                )
                and np.isclose(
                    float(cached_result.get("risk", np.nan)), float(row["risk"]),
                    rtol=0.0, atol=1.0e-9,
                )
                and np.isclose(
                    float(cached_result.get("search_action_proxy", np.nan)),
                    float(row["search_action_proxy"]),
                    rtol=0.0, atol=1.0e-9,
                )
            )
            if (
                json_path == local_json_path
                and cached.get("signature_schema") == 3
                and not exact_scientific_match
            ):
                cached_eta = np.asarray(cached_result.get("eta", []), dtype=np.float64)
                current_eta = np.asarray(row["eta"], dtype=np.float64)
                eta_delta = (
                    float(np.max(np.abs(cached_eta - current_eta)))
                    if cached_eta.shape == current_eta.shape else float("inf")
                )
                print(
                    f"[authoritative] {row['id']} cache mismatch "
                    f"eta_delta={eta_delta:.3e} "
                    f"risk_delta={abs(float(cached_result.get('risk', np.nan)) - float(row['risk'])):.3e} "
                    f"proxy_delta={abs(float(cached_result.get('search_action_proxy', np.nan)) - float(row['search_action_proxy'])):.3e}",
                    flush=True,
                )
            legacy_scientific_match = (
                exact_scientific_match and cached.get("signature_schema") in (None, 2)
            )
            invalid_warm_start_match = (
                exact_scientific_match
                and cached.get("signature_schema") == 3
                and not cached_result.get("valid", False)
                and selection_polish_iterations
                > int(cached_result.get("selection_polish_iterations_completed", 0))
            )
            valid_reaudit_match = (
                exact_scientific_match
                and cached.get("signature_schema") == 3
                and cached_result.get("valid", False)
            )
            if (
                cached.get("signature") == signature
                or legacy_scientific_match
                or invalid_warm_start_match
                or valid_reaudit_match
            ):
                params, _ = load_ritz_checkpoint(params_path)
                if valid_reaudit_match and cached.get("signature") != signature:
                    try:
                        result = _reaudit_cached_candidate(
                            cached_result, params, jnp.asarray(row["eta"]), row,
                            train_bank, audit_bank, family, times, time_weights, cfg,
                        )
                    except (RuntimeError, ValueError, FloatingPointError):
                        # A stale state that cannot pass the current audit is
                        # not a cache hit; fall through to a fresh solve.
                        result = None
                    if result is None or not result.get("valid", False):
                        result = None
                        continue
                else:
                    result = dict(cached_result)
                    result["params"] = params
                result["id"] = row["id"]
                result["risk"] = float(row["risk"])
                result["search_action_proxy"] = float(row["search_action_proxy"])
                loaded_from = (json_path, params_path)
                cache_kind = (
                    "exact" if cached.get("signature") == signature
                    else "fresh-reaudit" if valid_reaudit_match
                    else "invalid-warm-start" if invalid_warm_start_match
                    else "scientific-match"
                )
                print(
                    f"[authoritative] {index + 1}/{len(shortlisted)} {row['id']} "
                    f"cache hit ({cache_kind})",
                    flush=True,
                )
                break
        if result is None:
            print(f"[authoritative] {index + 1}/{len(shortlisted)} {row['id']} solving", flush=True)
            result = _authoritative_candidate(
                jnp.asarray(row["eta"]), row, train_bank, audit_bank, family, times, time_weights, cfg,
                seed_offset=100 * index,
            )
            result["id"] = row["id"]
            result["search_action_proxy"] = row["search_action_proxy"]
            print(
                f"[authoritative] {row['id']} valid={result['valid']} "
                f"action={result.get('action')} elapsed={result.get('total_seconds'):.1f}s",
                flush=True,
            )
        completed_polish_iterations = int(
            result.get("selection_polish_iterations_completed", 0)
        )
        if (
            not result.get("valid", False)
            and "params" in result
            and selection_polish_iterations > completed_polish_iterations
        ):
            print(
                f"[authoritative] {row['id']} certification polish "
                f"({selection_polish_iterations} exact L-BFGS iterations)",
                flush=True,
            )
            polish_cfg = deepcopy(cfg)
            polish_cfg["deep_ritz"]["authoritative_restarts"] = 1
            polish_cfg["deep_ritz"]["adam_steps"] = 0
            polish_cfg["deep_ritz"]["lbfgs_iterations"] = selection_polish_iterations
            polish_cfg["deep_ritz"]["certification_polish_iterations"] = 0
            polished = _authoritative_candidate(
                jnp.asarray(row["eta"]), row, train_bank, audit_bank,
                family, times, time_weights, polish_cfg,
                seed_offset=100 * index,
                initial_params=result["params"],
            )
            previous_attempts = list(result.get("optimization", {}).get("attempts", []))
            polish_attempts = list(polished.get("optimization", {}).get("attempts", []))
            for attempt in polish_attempts:
                attempt["phase"] = "selection_certification_polish"
            use_polished = (
                polished.get("valid", False)
                or _normalized_certificate_score(polished) < _normalized_certificate_score(result)
            )
            selected_result = polished if use_polished else result
            selected_result["id"] = row["id"]
            selected_result["search_action_proxy"] = row["search_action_proxy"]
            selected_result["selection_polish_iterations_completed"] = selection_polish_iterations
            selected_result["total_seconds"] = (
                float(result.get("total_seconds", 0.0))
                + float(polished.get("total_seconds", 0.0))
            )
            if "optimization" in selected_result:
                selected_result["optimization"]["attempts"] = previous_attempts + polish_attempts
            result = selected_result
            print(
                f"[authoritative] {row['id']} post-polish valid={result['valid']} "
                f"action={result.get('action')}",
                flush=True,
            )
        if "params" in result:
            payload = {
                "signature_schema": 3,
                "signature": signature,
                "result": _serializable_candidate(result),
            }
            # Always materialize a candidate-specific local artifact, even
            # when the numerical state came from the shared Pareto cache.
            save_ritz_checkpoint(
                local_params_path,
                result["params"],
                metadata={"signature": signature, "candidate_id": row["id"]},
            )
            write_json(local_json_path, payload)
            if shared_candidate_cache is not None:
                shared_json_path = shared_candidate_cache / f"{signature}.json"
                shared_params_path = shared_candidate_cache / f"{signature}.npz"
                if loaded_from != (shared_json_path, shared_params_path):
                    save_ritz_checkpoint(
                        shared_params_path,
                        result["params"],
                        metadata={"signature": signature, "candidate_id": row["id"]},
                    )
                    write_json(shared_json_path, payload)
        authoritative.append(result)
    stage_times["authoritative_selection"] = time.perf_counter() - started
    valid_authoritative = [row for row in authoritative if row["valid"] and row["risk"] <= risk_limit]
    if not valid_authoritative:
        failures = [row.get("failure_reason", row.get("certificate", {})) for row in authoritative]
        write_json(
            output_dir / "authoritative_failure.json",
            {"rows": [_serializable_candidate(row) for row in authoritative], "failures": failures},
        )
        raise RuntimeError(f"no authoritative Deep Ritz candidate certified: {failures}")
    full = min(valid_authoritative, key=lambda row: (row["action"], row["risk"], row["id"]))
    law_matches = [row for row in authoritative if row["id"] == law_risk_row["id"] and row["valid"]]
    if not law_matches:
        raise RuntimeError("Law design failed authoritative Deep Ritz certification")
    law = law_matches[0]
    save_ritz_checkpoint(
        output_dir / "ritz_law.npz", law["params"],
        metadata={"role": "Law", "eta": law["eta"], "config_hash": fingerprint(cfg)},
    )
    save_ritz_checkpoint(
        output_dir / "ritz_full.npz", full["params"],
        metadata={"role": "Full Deep Ritz", "eta": full["eta"], "config_hash": fingerprint(cfg)},
    )

    # A Full candidate that materially beats the Law risk invalidates the claimed anchor.
    consistency_tol = float(cfg["search"]["risk_anchor_consistency_tolerance"])
    if fixed_law_eta is None and full["risk"] < anchor - consistency_tol:
        raise RuntimeError("Full search found a materially better Law risk; refine the Law anchor")

    started = time.perf_counter()
    validation_truth = registry.get("truth_validation", consumer="validation")
    validation_fit = registry.get("validation_fit", consumer="validation")
    validation_audit = registry.get("validation_audit", consumer="validation")
    validation_truth_features = many_body_features(validation_truth.configurations, tuple(physics.box))
    validation_truth_means = jnp.mean(validation_truth_features, axis=1)
    validation_ref_features = many_body_features(validation_fit["configurations"], tuple(physics.box))
    validation_rows: dict[str, Any] = {}
    validation_cfg = deepcopy(cfg)
    validation_cfg["deep_ritz"]["authoritative_restarts"] = int(
        cfg["search"].get("validation_authoritative_restarts", 4)
    )
    validation_cfg["deep_ritz"]["certification_polish_iterations"] = int(
        cfg["search"].get("validation_polish_iterations", 0)
    )
    validation_cfg["deep_ritz"]["independent_time_nodes"] = int(
        cfg["search"].get("validation_independent_time_nodes", 0)
    )
    validation_cache = output_dir / "validation_candidates"
    validation_cache.mkdir(parents=True, exist_ok=True)
    validation_failure_path = output_dir / "validation_failure.json"
    validation_failure_path.unlink(missing_ok=True)
    if shared_validation_cache is not None:
        shared_validation_cache.mkdir(parents=True, exist_ok=True)
    validation_signature_cfg = _validation_signature_config(validation_cfg)
    for label, selected in (("law", law), ("full", full)):
        eta = jnp.asarray(selected["eta"])
        validation_risk_row = _selection_risk(
            eta, validation_truth, validation_fit, family, validation_ref_features,
            validation_truth_means, whitening, time_weights, cfg,
            noise_seed=seed + int(offsets["observation"]) + 10000,
        )
        validation_signature = fingerprint({
            "schema": 2,
            "config": validation_signature_cfg,
            "eta": selected["eta"],
            "risk": validation_risk_row["risk"],
            "targets": np.asarray(validation_risk_row["targets"]),
            "derivatives": np.asarray(validation_risk_row["derivatives"]),
            "fit_shape": list(validation_fit["configurations"].shape),
            "audit_shape": list(validation_audit["configurations"].shape),
        })
        local_json_path = validation_cache / f"{label}.json"
        local_params_path = validation_cache / f"{label}.npz"
        cache_locations = [(local_json_path, local_params_path)]
        if shared_validation_cache is not None:
            cache_locations.append((
                shared_validation_cache / f"{validation_signature}.json",
                shared_validation_cache / f"{validation_signature}.npz",
            ))
            cache_locations.extend(
                (path, path.with_suffix(".npz"))
                for path in sorted(shared_validation_cache.glob("*.json"))
                if path.name != f"{validation_signature}.json"
            )
        validation_result = None
        cached_initial_params = None
        loaded_from: tuple[Path, Path] | None = None
        for json_path, params_path in cache_locations:
            if not json_path.exists() or not params_path.exists():
                continue
            cached = json.loads(json_path.read_text(encoding="utf-8"))
            cached_result = cached.get("result", {})
            exact_scientific_match = (
                np.allclose(
                    np.asarray(cached_result.get("eta", []), dtype=np.float64),
                    np.asarray(selected["eta"], dtype=np.float64),
                    rtol=0.0, atol=1.0e-13,
                )
                and np.isclose(
                    float(cached_result.get("risk", np.nan)),
                    float(validation_risk_row["risk"]),
                    rtol=0.0, atol=1.0e-12,
                )
            )
            legacy_scientific_match = (
                exact_scientific_match and cached.get("signature_schema") is None
            )
            if cached.get("signature") == validation_signature or legacy_scientific_match:
                params, _ = load_ritz_checkpoint(params_path)
                validation_result = dict(cached_result)
                validation_result["params"] = params
                validation_result["risk"] = float(validation_risk_row["risk"])
                loaded_from = (json_path, params_path)
                cache_kind = "exact" if cached.get("signature") == validation_signature else "scientific-match"
                print(f"[validation] {label} cache hit ({cache_kind})", flush=True)
                break
            if exact_scientific_match and cached.get("signature_schema") == 2:
                params, _ = load_ritz_checkpoint(params_path)
                try:
                    reaudited = _reaudit_cached_candidate(
                        cached_result, params, eta, validation_risk_row,
                        validation_fit, validation_audit, family, times,
                        time_weights, validation_cfg,
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    reaudited = None
                if reaudited is not None and reaudited.get("valid", False):
                    validation_result = reaudited
                    loaded_from = (json_path, params_path)
                    print(
                        f"[validation] {label} cache hit (fresh-reaudit)",
                        flush=True,
                    )
                    break
                # Never accept an invalid stale result, but retain its state as
                # a warm start for a complete solve on the current fit/audit pair.
                cached_initial_params = params
        if validation_result is None:
            print(f"[validation] {label} solving", flush=True)
            validation_result = _authoritative_candidate(
                eta, validation_risk_row, validation_fit, validation_audit,
                family, times, time_weights, validation_cfg,
                seed_offset=5000 + (0 if label == "law" else 1000),
                initial_params=(
                    cached_initial_params
                    if cached_initial_params is not None else selected.get("params")
                ),
            )
            print(
                f"[validation] {label} valid={validation_result['valid']} "
                f"action={validation_result.get('action')} "
                f"elapsed={validation_result.get('total_seconds'):.1f}s",
                flush=True,
            )
        if "params" in validation_result:
            payload = {
                "signature_schema": 2,
                "signature": validation_signature,
                "result": _serializable_candidate(validation_result),
            }
            save_ritz_checkpoint(
                local_params_path,
                validation_result["params"],
                metadata={"signature": validation_signature, "role": f"validation-{label}"},
            )
            write_json(local_json_path, payload)
            if shared_validation_cache is not None:
                shared_json_path = shared_validation_cache / f"{validation_signature}.json"
                shared_params_path = shared_validation_cache / f"{validation_signature}.npz"
                if loaded_from != (shared_json_path, shared_params_path):
                    save_ritz_checkpoint(
                        shared_params_path,
                        validation_result["params"],
                        metadata={"signature": validation_signature, "role": f"validation-{label}"},
                    )
                    write_json(shared_json_path, payload)
        checkpoint = f"ritz_validation_{label}.npz"
        if "params" in validation_result:
            save_ritz_checkpoint(
                output_dir / checkpoint,
                validation_result["params"],
                metadata={"role": f"validation-{label}", "eta": selected["eta"]},
            )
        validation_rows[label] = _serializable_candidate(validation_result)
        validation_rows[label]["checkpoint"] = checkpoint if "params" in validation_result else None
        if label == "law" and not validation_result.get("valid", False):
            write_json(
                validation_failure_path,
                {
                    "reason": "independent Law validation failed certification",
                    "law": validation_rows[label],
                },
            )
            raise RuntimeError(
                "independent Law validation failed certification; Pareto remains locked"
            )
    stage_times["independent_validation"] = time.perf_counter() - started

    extra_risk_percent = 100.0 * (float(full["risk"]) / anchor - 1.0)
    action_reduction = 1.0 - float(full["action"]) / float(law["action"]) if law["action"] > 0 else float("nan")
    validation_action_reduction = (
        1.0 - float(validation_rows["full"]["action"]) / float(validation_rows["law"]["action"])
        if float(validation_rows["law"]["action"]) > 0.0 else float("nan")
    )
    validation_neighborhood = float(validation_rows["full"]["risk"]) <= (
        1.0 + allowance / 100.0 + float(cfg["search"].get("validation_relative_slack", 0.05))
    ) * max(float(validation_rows["law"]["risk"]), 1.0e-12)
    minimum_reduction = float(cfg["search"].get("minimum_action_reduction_fraction", 0.01))
    milestone_success = bool(
        law["valid"] and full["valid"] and full["risk"] <= risk_limit
        and validation_rows["law"]["valid"] and validation_rows["full"]["valid"]
        and validation_neighborhood and np.isfinite(action_reduction)
        and action_reduction >= minimum_reduction
        and np.isfinite(validation_action_reduction)
        and validation_action_reduction >= minimum_reduction
    )
    stage_times["total"] = time.perf_counter() - overall_started

    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema_version": 1,
        "experiment": cfg["name"],
        "execution_profile": cfg.get("execution_profile", "authoritative"),
        "method": "FIDE full-law action solved by time-conditioned permutation-invariant JAX Deep Ritz",
        "smoke": bool(smoke),
        "config": cfg,
        "config_hash": fingerprint(cfg),
        "git_commit": _git_commit(repo_root),
        "physics": asdict(physics),
        "reference": {
            "checkpoint": "reference.npz", "endpoint_only": True,
            "frozen_for_all_designs": True, "training_history": reference_history,
            "pareto_frozen_source": (
                str(Path(frozen_artifact_source).resolve())
                if frozen_artifact_source is not None else None
            ),
            "checkpoint_sha256": reference_hash,
        },
        "bank_manifest": registry.manifest(),
        "bank_access_log": registry.access_log,
        "law_anchor": {
            "id": law["id"], "eta": law["eta"], "risk": anchor,
            "action": law["action"], "checkpoint": "ritz_law.npz",
            "frozen_for_pareto": fixed_law_eta is not None,
        },
        "full_3_percent": {
            "id": full["id"], "eta": full["eta"], "selection_risk": full["risk"],
            "risk_limit": risk_limit, "extra_risk_percent": extra_risk_percent,
            "selection_action": full["action"], "action_reduction_vs_law": action_reduction,
            "minimum_ess_fraction": full["minimum_ess_fraction"],
            "maximum_projection_residual": full["maximum_projection_residual"],
            "maximum_forcing_mean_residual": full["maximum_forcing_mean_residual"],
            "certificate": full["certificate"], "optimization": full["optimization"],
            "valid": full["valid"],
            "checkpoint": "ritz_full.npz",
        },
        "authoritative_candidates": [_serializable_candidate(row) for row in authoritative],
        "validation": validation_rows,
        "validation_contrast": {
            "full_vs_law_action_reduction": validation_action_reduction,
            "minimum_required_reduction": minimum_reduction,
            "scientific_risk_neighborhood_pass": validation_neighborhood,
        },
        "timings_seconds": stage_times,
        "milestone_success": milestone_success,
        "pareto_unlocked": (
            milestone_success
            and cfg.get("execution_profile", "authoritative") == "authoritative"
        ),
        "comparisons": ["Law", "Full Deep Ritz"],
        "forbidden_decompositions_computed": False,
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "timings.json", stage_times)
    write_json(output_dir / "bank_manifest.json", {"banks": registry.manifest(), "access_log": registry.access_log})
    write_csv(output_dir / "result.candidate_summary.csv", [
        {
            "id": row["id"], "risk": row["risk"], "action": row["action"],
            "valid": row["valid"], "minimum_ess_fraction": row.get("minimum_ess_fraction"),
            "maximum_projection_residual": row.get("maximum_projection_residual"),
            "maximum_weak_residual": row.get("certificate", {}).get("maximum_weak_residual"),
            "maximum_energy_residual": row.get("certificate", {}).get("maximum_energy_residual"),
            "total_seconds": row.get("total_seconds"),
        }
        for row in authoritative
    ])
    return result
