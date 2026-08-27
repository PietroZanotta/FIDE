"""Experiment-local data preparation, Ritz tracking, refinement, and audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from mfsi.cache import fingerprint, load_npz_cache, save_npz_cache
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import IProjectionConfig

from .deep_ritz import (
    CertificateConfig,
    DeepRitzConfig,
    DeepRitzResult,
    RitzParams,
    audit_deep_ritz,
    save_ritz_checkpoint,
    solve_deep_ritz,
)
from .domain import ConfigurationBank, SkyrmionTruth
from .experiment import (
    _ensure_reference,
    _ensure_reference_bank,
    _install_frozen_reference_artifacts,
    _load_frozen_truth_banks,
    _physics_config,
    _time_weights,
)
from .forcing import ForcingConfig
from .full_gradient import (
    EnvelopeDiagnostics,
    FrozenEtaProblem,
    ReferenceBank,
    envelope_diagnostics,
    envelope_full_value_and_grad,
    forcing_state,
    full_energy,
    minimum_sensor_separation,
    periodic_branch_distance,
    projected_law_risk,
    reconstruct_moments,
    ritz_objective_eta,
    smooth_separation_penalty,
    wrap_periodic,
)
from .measurements import LocalDensitySensors
from .risk import many_body_features, whitening_from_truth

Array = jax.Array
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


@dataclass(frozen=True)
class PreparedExperiment:
    selection_problem: FrozenEtaProblem
    validation_problem: FrozenEtaProblem
    projection_bank: ReferenceBank
    ritz_train_bank: ReferenceBank
    ritz_audit_bank: ReferenceBank
    validation_fit_bank: ReferenceBank
    validation_audit_bank: ReferenceBank
    selection_reference_features: Array
    selection_truth_means: Array
    validation_reference_features: Array
    validation_truth_means: Array
    whitening: Array


@dataclass(frozen=True)
class InnerSolution:
    params: RitzParams
    result: DeepRitzResult
    diagnostics: EnvelopeDiagnostics


@dataclass(frozen=True)
class CandidateEvaluation:
    params: RitzParams | None
    payload: dict[str, Any]


def require_output_path(path: Path) -> Path:
    """Reject every write target outside this experiment's output directory."""

    resolved = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"output path must be beneath {root}, got {resolved}")
    return resolved


def _reference_bank(values: dict[str, Array]) -> ReferenceBank:
    return ReferenceBank(
        configurations=jnp.asarray(values["configurations"], dtype=jnp.float64),
        velocity=jnp.asarray(values["velocity"], dtype=jnp.float64),
        base_weights=jnp.asarray(values["base_weights"], dtype=jnp.float64),
    )


def _acquisition_indices(time_count: int, acquisition_count: int) -> Array:
    if acquisition_count < 2 or acquisition_count > time_count:
        raise ValueError("acquisition_count must be in [2, time_count]")
    values = tuple(
        round(index * (time_count - 1) / (acquisition_count - 1))
        for index in range(acquisition_count)
    )
    if len(set(values)) != acquisition_count:
        raise ValueError("acquisition grid contains duplicate time indices")
    return jnp.asarray(values, dtype=jnp.int32)


def _projection_config(cfg: dict[str, Any]) -> tuple[IProjectionConfig, str]:
    values = dict(cfg["projection"])
    backend = str(values.pop("trajectory_backend", "jax"))
    allowed = {item.name for item in fields(IProjectionConfig)}
    return IProjectionConfig(**{key: value for key, value in values.items() if key in allowed}), backend


def _forcing_config(cfg: dict[str, Any]) -> ForcingConfig:
    allowed = {item.name for item in fields(ForcingConfig)}
    return ForcingConfig(**{
        key: value for key, value in cfg["forcing"].items() if key in allowed
    })


def _make_problem(
    cfg: dict[str, Any],
    truth: ConfigurationBank,
    family: LocalDensitySensors,
    times: Array,
    time_weights: Array,
    *,
    noise_seed: int,
) -> FrozenEtaProblem:
    acquisition = _acquisition_indices(
        len(times), int(cfg["measurement"]["acquisition_count"])
    )
    reconstruction_cfg = AnchoredCubicSplineConfig(**cfg["moment_reconstruction"])
    reconstructor = AnchoredCubicSplineReconstructor(
        jax.device_get(times[acquisition]),
        jax.device_get(times),
        reconstruction_cfg,
    )
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(int(noise_seed)),
        (len(acquisition), family.n_sensors),
        dtype=jnp.float64,
    )
    projection_cfg, backend = _projection_config(cfg)
    return FrozenEtaProblem(
        truth_configurations=jnp.asarray(truth.configurations, dtype=jnp.float64),
        times=jnp.asarray(times, dtype=jnp.float64),
        time_weights=jnp.asarray(time_weights, dtype=jnp.float64),
        acquisition_indices=acquisition,
        finite_configuration_count=min(
            int(cfg["measurement"]["finite_configurations"]),
            int(truth.configurations.shape[1]),
        ),
        detector_noise=noise,
        family=family,
        reconstructor=reconstructor,
        projection_config=projection_cfg,
        forcing_config=_forcing_config(cfg),
        projection_backend=backend,
        box=tuple(cfg["physics"]["box"]),
    )


def _truth_banks(
    cfg: dict[str, Any], artifact_dir: Path, times: Array, frozen_source: Path | None
) -> tuple[ConfigurationBank, ConfigurationBank, ConfigurationBank]:
    if frozen_source is not None:
        return _load_frozen_truth_banks(frozen_source, artifact_dir, cfg, times)

    signature = fingerprint({
        "schema": 1,
        "physics": cfg["physics"],
        "banks": {
            key: cfg["banks"][key]
            for key in ("endpoint_samples", "truth_design_samples", "truth_validation_samples")
        },
        "seed": cfg["seed"],
        "times": jax.device_get(times).tolist(),
    })
    cache_path = artifact_dir / "truth_banks.npz"
    cached = load_npz_cache(cache_path, signature=signature)
    if cached is not None:
        arrays, _ = cached
        endpoints = ConfigurationBank(
            jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            jnp.stack([jnp.asarray(arrays["endpoint0"]), jnp.asarray(arrays["endpoint1"])]),
        )
        return (
            endpoints,
            ConfigurationBank(times, jnp.asarray(arrays["design"], dtype=jnp.float64)),
            ConfigurationBank(times, jnp.asarray(arrays["validation"], dtype=jnp.float64)),
        )

    seed = int(cfg["seed"])
    offsets = cfg["banks"]["seed_offsets"]
    model = SkyrmionTruth(_physics_config(cfg))
    endpoints = model.make_bank(
        seed=seed + int(offsets["endpoints"]),
        samples=int(cfg["banks"]["endpoint_samples"]),
        times=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        substeps_per_interval=int(cfg["physics"]["truth_substeps"])
        * max(len(times) - 1, 1),
    )
    design = model.make_bank(
        seed=seed + int(offsets["truth_design"]),
        samples=int(cfg["banks"]["truth_design_samples"]),
        times=times,
        substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
    )
    validation = model.make_bank(
        seed=seed + int(offsets["truth_validation"]),
        samples=int(cfg["banks"]["truth_validation_samples"]),
        times=times,
        substeps_per_interval=int(cfg["physics"]["truth_substeps"]),
    )
    save_npz_cache(
        cache_path,
        {
            "times": times,
            "design": design.configurations,
            "validation": validation.configurations,
            "endpoint0": endpoints.configurations[0],
            "endpoint1": endpoints.configurations[-1],
        },
        signature=signature,
        metadata={"role": "new_experiment_local_frozen_truth"},
    )
    return endpoints, design, validation


def prepare_experiment(
    cfg: dict[str, Any],
    artifact_dir: Path,
    *,
    frozen_source: Path | None = None,
) -> PreparedExperiment:
    """Create or load deterministic banks, writing only below the new output root."""

    artifact_dir = require_output_path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    times = jnp.linspace(
        0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64
    )
    time_weights = _time_weights(times)
    physics = _physics_config(cfg)
    family = LocalDensitySensors(
        n_sensors=int(cfg["measurement"]["n_sensors"]),
        width=float(cfg["measurement"]["sensor_width"]),
        box=tuple(physics.box),
        min_separation=float(cfg["measurement"]["min_separation"]),
    )
    frozen_source = Path(frozen_source).resolve() if frozen_source is not None else None
    endpoints, truth_design, truth_validation = _truth_banks(
        cfg, artifact_dir, times, frozen_source
    )

    if frozen_source is not None:
        _install_frozen_reference_artifacts(frozen_source, artifact_dir)
    reference, _ = _ensure_reference(artifact_dir, endpoints, cfg, tuple(physics.box))
    reference_hash = fingerprint({
        "params": jax.tree_util.tree_map(lambda value: jax.device_get(value), reference.params),
        "box": tuple(physics.box),
    })
    offsets = cfg["banks"]["seed_offsets"]
    bank_cfg = cfg["banks"]
    banks = {
        name: _reference_bank(_ensure_reference_bank(
            artifact_dir,
            name,
            reference,
            endpoints.configurations[0],
            times,
            seed=int(cfg["seed"]) + int(offsets[name]),
            samples=int(bank_cfg[f"{name}_samples"]),
            substeps=int(bank_cfg["reference_substeps"]),
            reference_hash=reference_hash,
        ))
        for name in (
            "projection", "ritz_train", "ritz_audit",
            "validation_fit", "validation_audit",
        )
    }
    selection_problem = _make_problem(
        cfg,
        truth_design,
        family,
        times,
        time_weights,
        noise_seed=int(cfg["seed"]) + int(offsets["observation"]),
    )
    validation_problem = _make_problem(
        cfg,
        truth_validation,
        family,
        times,
        time_weights,
        noise_seed=int(cfg["seed"]) + int(offsets["observation"]) + 10000,
    )
    selection_truth_features = many_body_features(
        truth_design.configurations, tuple(physics.box)
    )
    validation_truth_features = many_body_features(
        truth_validation.configurations, tuple(physics.box)
    )
    whitening = whitening_from_truth(selection_truth_features)
    return PreparedExperiment(
        selection_problem=selection_problem,
        validation_problem=validation_problem,
        projection_bank=banks["projection"],
        ritz_train_bank=banks["ritz_train"],
        ritz_audit_bank=banks["ritz_audit"],
        validation_fit_bank=banks["validation_fit"],
        validation_audit_bank=banks["validation_audit"],
        selection_reference_features=many_body_features(
            banks["projection"].configurations, tuple(physics.box)
        ),
        selection_truth_means=jnp.mean(selection_truth_features, axis=1),
        validation_reference_features=many_body_features(
            banks["validation_fit"].configurations, tuple(physics.box)
        ),
        validation_truth_means=jnp.mean(validation_truth_features, axis=1),
        whitening=whitening,
    )


def inner_config(cfg: dict[str, Any], mode: str, *, sample_count: int) -> DeepRitzConfig:
    if mode not in {
        "full", "track", "smoke",
        "gradient_validation_center", "gradient_validation_polish",
    }:
        raise ValueError(f"unknown inner mode {mode!r}")
    values = dict(cfg["deep_ritz"])
    values.update(cfg.get("envelope", {}).get("inner", {}).get(mode, {}))
    allowed = {item.name for item in fields(DeepRitzConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    values["box"] = tuple(cfg["physics"]["box"])
    # Deterministic full-bank Adam is required for envelope/FD comparisons.
    values["adam_batch_size"] = max(int(values.get("adam_batch_size", sample_count)), sample_count)
    values["lbfgs_batch_size"] = min(
        max(int(values.get("lbfgs_batch_size", sample_count)), 1), sample_count
    )
    return DeepRitzConfig(**values)


def solve_inner(
    eta: Array,
    cfg: dict[str, Any],
    problem: FrozenEtaProblem,
    bank: ReferenceBank,
    *,
    mode: str,
    initial_params: RitzParams | None = None,
) -> InnerSolution:
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    solve = solve_deep_ritz(
        bank.configurations,
        state.projection.weights,
        state.forcing,
        problem.times,
        problem.time_weights,
        inner_config(cfg, mode, sample_count=int(bank.configurations.shape[1])),
        initial_params=initial_params,
    )
    diagnostics = envelope_diagnostics(solve.params, eta, problem, bank)
    return InnerSolution(solve.params, solve, diagnostics)


def selection_risk(eta: Array, data: PreparedExperiment) -> Array:
    return projected_law_risk(
        eta,
        data.selection_problem,
        data.projection_bank,
        data.selection_reference_features,
        data.selection_truth_means,
        data.whitening,
    )


def validation_risk(eta: Array, data: PreparedExperiment) -> Array:
    return projected_law_risk(
        eta,
        data.validation_problem,
        data.validation_fit_bank,
        data.validation_reference_features,
        data.validation_truth_means,
        data.whitening,
    )


def law_risk_anchor(
    cfg: dict[str, Any], data: PreparedExperiment, *, validation: bool = False
) -> float:
    """Evaluate the fixed law design on the same bank used by the candidate.

    The number stored in ``config.json`` documents the production anchor, but
    smoke and validation banks are deliberately different.  Re-evaluating the
    fixed law eta prevents a bank change from silently changing the risk gate.
    """

    risk_fn = validation_risk if validation else selection_risk
    return float(risk_fn(jnp.asarray(cfg["envelope"]["law_eta"]), data))


def forcing_proxy(eta: Array, problem: FrozenEtaProblem, bank: ReferenceBank) -> Array:
    state = forcing_state(eta, problem, bank)
    rows = jnp.einsum(
        "tn,tn,tn->t", state.projection.weights, state.forcing, state.forcing
    )
    return jnp.sum(problem.time_weights * rows)


def _finite_scalar(value: Array) -> bool:
    return bool(jax.device_get(jnp.isfinite(value)))


def hard_forcing_audit(eta: Array, problem: FrozenEtaProblem, bank: ReferenceBank) -> dict[str, Any]:
    state = forcing_state(eta, problem, bank)
    maximum_projection = jnp.max(jnp.linalg.norm(state.projection.residual, axis=-1))
    minimum_ess = jnp.min(state.projection.ess_fraction)
    maximum_mean = jnp.max(jnp.abs(state.forcing_mean_before_centering))
    maximum_condition = jnp.max(state.covariance_condition)
    cfg = problem.forcing_config
    valid = bool(
        _finite_scalar(maximum_projection)
        and _finite_scalar(minimum_ess)
        and _finite_scalar(maximum_mean)
        and _finite_scalar(maximum_condition)
        and float(maximum_projection) <= float(cfg.projection_tolerance)
        and float(minimum_ess) >= float(cfg.minimum_ess_fraction)
        and float(maximum_mean) <= float(cfg.forcing_mean_tolerance)
        and float(maximum_condition) <= float(cfg.max_covariance_condition)
    )
    return {
        "valid": valid,
        "maximum_projection_residual": float(maximum_projection),
        "minimum_ess_fraction": float(minimum_ess),
        "maximum_forcing_mean": float(maximum_mean),
        "maximum_covariance_condition": float(maximum_condition),
    }


def _envelope_payload(diagnostics: EnvelopeDiagnostics) -> dict[str, float]:
    return {name: float(value) for name, value in zip(
        diagnostics._fields, diagnostics, strict=True
    )}


def _inner_solver_payload(inner: InnerSolution) -> dict[str, Any]:
    last_gradient = next(
        (
            float(row["gradient_norm"])
            for row in reversed(inner.result.history)
            if row.get("phase") == "lbfgs" and "gradient_norm" in row
        ),
        None,
    )
    return {
        "finite": bool(inner.result.finite),
        "lbfgs_converged": bool(inner.result.lbfgs_converged),
        "adam_final_objective": inner.result.adam_final_objective,
        "lbfgs_final_objective": inner.result.lbfgs_final_objective,
        "last_lbfgs_gradient_norm": last_gradient,
        "energy_identity_relerr": float(inner.diagnostics.energy_identity_relerr),
    }


def authoritative_evaluate(
    eta: Array,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    *,
    allowance_percent: float,
    initial_params: RitzParams | None = None,
    validation: bool = False,
) -> CandidateEvaluation:
    """High-accuracy solve plus unchanged hard Deep Ritz certificate thresholds."""

    if validation:
        problem = data.validation_problem
        train_bank = data.validation_fit_bank
        audit_bank = data.validation_audit_bank
        risk_fn = lambda design: validation_risk(design, data)
    else:
        problem = data.selection_problem
        train_bank = data.ritz_train_bank
        audit_bank = data.ritz_audit_bank
        risk_fn = lambda design: selection_risk(design, data)

    eta = wrap_periodic(eta, problem.family)
    geometry_valid = bool(jax.device_get(problem.family.geometry_valid(eta)))
    risk = float(risk_fn(eta))
    anchor = law_risk_anchor(cfg, data, validation=validation)
    risk_limit = (1.0 + float(allowance_percent) / 100.0) * anchor
    try:
        inner = solve_inner(
            eta, cfg, problem, train_bank,
            mode="full", initial_params=initial_params,
        )
        train_forcing_audit = hard_forcing_audit(eta, problem, train_bank)
        audit_forcing_audit = hard_forcing_audit(eta, problem, audit_bank)
        reconstruction = reconstruct_moments(eta, problem)
        audit_state = forcing_state(eta, problem, audit_bank, reconstruction)
        certificate = audit_deep_ritz(
            inner.params,
            audit_bank.configurations,
            audit_state.projection.weights,
            audit_state.forcing,
            problem.times,
            problem.time_weights,
            family=problem.family,
            eta=eta,
            reference_velocity=audit_bank.velocity,
            target_derivatives=reconstruction.derivatives,
            cfg=CertificateConfig(**cfg["certificates"]),
            box=problem.box,
            chunk_size=min(1024, int(audit_bank.configurations.shape[1])),
        )
        valid = bool(
            geometry_valid
            and risk <= risk_limit
            and inner.result.finite
            and train_forcing_audit["valid"]
            and audit_forcing_audit["valid"]
            and certificate["valid"]
        )
        return CandidateEvaluation(
            inner.params,
            {
                "eta": jax.device_get(eta).tolist(),
                "risk": risk,
                "law_risk_anchor": anchor,
                "risk_limit": risk_limit,
                "action": float(certificate["action"]),
                "valid": valid,
                "geometry_valid": geometry_valid,
                "minimum_separation": float(minimum_sensor_separation(eta, problem.family)),
                "periodic_branch_distance": float(periodic_branch_distance(eta, problem.family)),
                "inner_finite": bool(inner.result.finite),
                "inner_lbfgs_converged": bool(inner.result.lbfgs_converged),
                "inner_solver": _inner_solver_payload(inner),
                "envelope_diagnostics": _envelope_payload(inner.diagnostics),
                "train_forcing_audit": train_forcing_audit,
                "audit_forcing_audit": audit_forcing_audit,
                "certificate": certificate,
                "validation": validation,
            },
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return CandidateEvaluation(None, {
            "eta": jax.device_get(eta).tolist(),
            "risk": risk,
            "law_risk_anchor": anchor,
            "risk_limit": risk_limit,
            "action": float("inf"),
            "valid": False,
            "geometry_valid": geometry_valid,
            "failure_reason": str(exc),
            "validation": validation,
        })


def deterministic_direction(
    eta: Array,
    family: LocalDensitySensors,
    *,
    seed: int,
    maximum_epsilon: float,
) -> Array:
    """Choose a deterministic normalized direction feasible at both endpoints."""

    eta = wrap_periodic(eta, family)
    for attempt in range(64):
        direction = jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(int(seed)), attempt),
            eta.shape,
            dtype=jnp.float64,
        )
        direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-30)
        plus = wrap_periodic(eta + float(maximum_epsilon) * direction, family)
        minus = wrap_periodic(eta - float(maximum_epsilon) * direction, family)
        if bool(jax.device_get(family.geometry_valid(plus) & family.geometry_valid(minus))):
            return direction
    raise RuntimeError("could not construct a feasible deterministic gradient-check direction")


def run_gradient_check(
    cfg: dict[str, Any],
    data: PreparedExperiment,
    *,
    eta0: Array | None = None,
    inner_mode: str = "full",
) -> tuple[dict[str, Any], RitzParams]:
    """Compare the envelope VJP with independently reoptimized centered FDs."""

    eta0 = jnp.asarray(
        cfg["envelope"]["eta0"] if eta0 is None else eta0,
        dtype=jnp.float64,
    )
    eta0 = wrap_periodic(eta0, data.selection_problem.family)
    epsilons = tuple(float(value) for value in cfg["envelope"]["gradient_check_eps"])
    direction = deterministic_direction(
        eta0,
        data.selection_problem.family,
        seed=int(cfg["envelope"]["gradient_check_direction_seed"]),
        maximum_epsilon=max(epsilons),
    )
    center = solve_inner(
        eta0, cfg, data.selection_problem, data.ritz_train_bank, mode=inner_mode
    )
    envelope_value, gradient, diagnostics = envelope_full_value_and_grad(
        eta0, center.params, data.selection_problem, data.ritz_train_bank
    )
    directional_ad = jnp.vdot(gradient, direction)
    rows: list[dict[str, Any]] = []
    for epsilon in epsilons:
        plus_eta = wrap_periodic(
            eta0 + epsilon * direction, data.selection_problem.family
        )
        minus_eta = wrap_periodic(
            eta0 - epsilon * direction, data.selection_problem.family
        )
        plus = solve_inner(
            plus_eta, cfg, data.selection_problem, data.ritz_train_bank,
            mode=inner_mode, initial_params=center.params,
        )
        minus = solve_inner(
            minus_eta, cfg, data.selection_problem, data.ritz_train_bank,
            mode=inner_mode, initial_params=center.params,
        )
        action_plus = full_energy(
            plus.params, plus_eta, data.selection_problem, data.ritz_train_bank
        )
        action_minus = full_energy(
            minus.params, minus_eta, data.selection_problem, data.ritz_train_bank
        )
        directional_fd = (action_plus - action_minus) / (2.0 * epsilon)
        absolute = jnp.abs(directional_fd - directional_ad)
        relative = absolute / jnp.maximum(
            jnp.maximum(jnp.abs(directional_fd), jnp.abs(directional_ad)), 1.0e-12
        )
        rows.append({
            "epsilon": epsilon,
            "directional_ad": float(directional_ad),
            "directional_fd": float(directional_fd),
            "absolute_discrepancy": float(absolute),
            "relative_discrepancy": float(relative),
            "action_plus": float(action_plus),
            "action_minus": float(action_minus),
            "plus_envelope_diagnostics": _envelope_payload(plus.diagnostics),
            "minus_envelope_diagnostics": _envelope_payload(minus.diagnostics),
            "plus_ritz_solve": _inner_solver_payload(plus),
            "minus_ritz_solve": _inner_solver_payload(minus),
        })
    tolerance = float(cfg["envelope"]["gradient_check_relative_tolerance"])
    identity_tolerance = float(cfg["envelope"]["maximum_energy_identity_relerr"])
    best_relative = min(row["relative_discrepancy"] for row in rows)
    passed = bool(
        jnp.all(jnp.isfinite(gradient))
        and gradient.shape == eta0.shape
        and best_relative <= tolerance
        and float(diagnostics.energy_identity_relerr) <= identity_tolerance
    )
    return ({
        "eta0": jax.device_get(eta0).tolist(),
        "direction": jax.device_get(direction).tolist(),
        "envelope_value": float(envelope_value),
        "gradient": jax.device_get(gradient).tolist(),
        "gradient_shape": list(gradient.shape),
        "directional_ad": float(directional_ad),
        "center_envelope_diagnostics": _envelope_payload(diagnostics),
        "center_inner": _inner_solver_payload(center),
        "rows": rows,
        "best_relative_discrepancy": best_relative,
        "relative_tolerance": tolerance,
        "energy_identity_tolerance": identity_tolerance,
        "passed": passed,
    }, center.params)


def _outer_objective(
    eta: Array,
    theta: RitzParams,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    risk_limit: float,
) -> Array:
    action = -2.0 * ritz_objective_eta(
        theta, eta, data.selection_problem, data.ritz_train_bank
    )
    risk = selection_risk(eta, data)
    risk_violation = jax.nn.relu(risk / float(risk_limit) - 1.0)
    return (
        action
        + float(cfg["envelope"]["risk_penalty"]) * risk_violation**2
        + float(cfg["envelope"]["separation_penalty"])
        * smooth_separation_penalty(eta, data.selection_problem.family)
    )


def refine_design(
    eta0: Array,
    theta0: RitzParams,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    *,
    allowance_percent: float,
) -> tuple[Array, RitzParams, list[dict[str, Any]]]:
    """Alternating Ritz tracking and Adam refinement with feasible backtracking."""

    eta = wrap_periodic(eta0, data.selection_problem.family)
    theta = theta0
    moment1 = jnp.zeros_like(eta)
    moment2 = jnp.zeros_like(eta)
    risk_limit = (1.0 + float(allowance_percent) / 100.0) * law_risk_anchor(
        cfg, data
    )
    trace: list[dict[str, Any]] = []
    outer_steps = int(cfg["envelope"]["outer_steps"])
    polish_every = max(int(cfg["envelope"].get("polish_every", 0)), 0)
    for step in range(1, outer_steps + 1):
        mode = "full" if polish_every and step % polish_every == 0 else "track"
        inner = solve_inner(
            eta, cfg, data.selection_problem, data.ritz_train_bank,
            mode=mode, initial_params=theta,
        )
        theta = inner.params
        if (
            float(inner.diagnostics.energy_identity_relerr)
            > float(cfg["envelope"]["maximum_energy_identity_relerr"])
            and mode != "full"
        ):
            inner = solve_inner(
                eta, cfg, data.selection_problem, data.ritz_train_bank,
                mode="full", initial_params=theta,
            )
            theta = inner.params

        value, gradient = jax.value_and_grad(
            lambda design: _outer_objective(
                design, theta, cfg, data, risk_limit
            )
        )(eta)
        moment1 = 0.9 * moment1 + 0.1 * gradient
        moment2 = 0.999 * moment2 + 0.001 * gradient * gradient
        mhat = moment1 / (1.0 - 0.9**step)
        vhat = moment2 / (1.0 - 0.999**step)
        update = float(cfg["envelope"]["outer_learning_rate"]) * mhat / (
            jnp.sqrt(vhat) + 1.0e-8
        )
        accepted = False
        candidate = eta
        for backtrack in range(int(cfg["envelope"]["outer_backtracking_steps"]) + 1):
            scale = 0.5**backtrack
            proposal = wrap_periodic(
                eta - scale * update, data.selection_problem.family
            )
            if bool(jax.device_get(data.selection_problem.family.geometry_valid(proposal))):
                candidate = proposal
                accepted = True
                break
        envelope_value, envelope_gradient, envelope_diag = envelope_full_value_and_grad(
            eta, theta, data.selection_problem, data.ritz_train_bank
        )
        trace.append({
            "step": step,
            "inner_mode": mode,
            "eta": jax.device_get(eta).tolist(),
            "objective": float(value),
            "envelope_value": float(envelope_value),
            "envelope_gradient_norm": float(jnp.linalg.norm(envelope_gradient)),
            "total_gradient_norm": float(jnp.linalg.norm(gradient)),
            "risk": float(selection_risk(eta, data)),
            "minimum_separation": float(minimum_sensor_separation(eta, data.selection_problem.family)),
            "periodic_branch_distance": float(periodic_branch_distance(eta, data.selection_problem.family)),
            "energy_identity_relerr": float(envelope_diag.energy_identity_relerr),
            "accepted": accepted,
        })
        if not accepted:
            break
        eta = candidate
    return eta, theta, trace


def random_global_candidates(
    cfg: dict[str, Any], family: LocalDensitySensors, count: int
) -> Array:
    """Fixed-shape, JAX-only feasible global candidate generation."""

    oversample = max(32 * int(count), int(count))
    box = jnp.asarray(family.box, dtype=jnp.float64)
    draws = jax.random.uniform(
        jax.random.PRNGKey(int(cfg["seed"]) + 29011),
        (oversample, family.n_sensors, 2),
        dtype=jnp.float64,
    ) * box
    flat = draws.reshape((oversample, -1))
    valid = jax.vmap(family.geometry_valid)(flat)
    order = jnp.argsort(~valid)
    if int(jnp.sum(valid)) < int(count):
        raise RuntimeError("not enough feasible global sensor candidates")
    return flat[order[: int(count)]]


def global_initial_starts(
    cfg: dict[str, Any], data: PreparedExperiment, *, allowance_percent: float
) -> list[dict[str, Any]]:
    family = data.selection_problem.family
    incumbent = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    random = random_global_candidates(
        cfg, family, int(cfg["envelope"]["global_candidate_count"])
    )
    designs = jnp.concatenate([incumbent[None, :], random], axis=0)
    risk_limit = (1.0 + float(allowance_percent) / 100.0) * law_risk_anchor(
        cfg, data
    )
    rows: list[dict[str, Any]] = []
    for index, eta in enumerate(designs):
        risk = float(selection_risk(eta, data))
        proxy = float(forcing_proxy(eta, data.selection_problem, data.ritz_train_bank))
        valid = bool(jax.device_get(family.geometry_valid(eta))) and risk <= risk_limit
        rows.append({
            "id": "incumbent" if index == 0 else f"random-{index:03d}",
            "eta": jax.device_get(wrap_periodic(eta, family)).tolist(),
            "risk": risk,
            "forcing_proxy": proxy,
            "valid": valid,
        })
    feasible = [row for row in rows if row["valid"]]
    feasible.sort(key=lambda row: (row["forcing_proxy"], row["risk"], row["id"]))
    incumbent_rows = [row for row in rows if row["id"] == "incumbent" and row["valid"]]
    selected = feasible[: int(cfg["envelope"]["global_start_count"])]
    if incumbent_rows and incumbent_rows[0] not in selected:
        selected.append(incumbent_rows[0])
    if not selected:
        raise RuntimeError("no global candidate passed geometry and risk prescreen")
    return selected


def save_candidate_checkpoint(
    path: Path, evaluation: CandidateEvaluation, *, role: str
) -> None:
    path = require_output_path(path)
    if evaluation.params is None:
        raise ValueError("cannot checkpoint an evaluation without Ritz parameters")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ritz_checkpoint(path, evaluation.params, metadata={
        "role": role,
        "eta": evaluation.payload["eta"],
        "risk": evaluation.payload["risk"],
        "action": evaluation.payload["action"],
    })


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": cfg["name"],
        "execution_profile": cfg.get("execution_profile"),
        "physics": cfg["physics"],
        "measurement": cfg["measurement"],
        "projection": cfg["projection"],
        "forcing": cfg["forcing"],
        "deep_ritz": cfg["deep_ritz"],
        "certificates": cfg["certificates"],
        "envelope": cfg["envelope"],
    }


def optimize_continuously(
    cfg: dict[str, Any],
    data: PreparedExperiment,
    output_dir: Path,
    *,
    allowance_percent: float,
    inner_mode: str,
) -> dict[str, Any]:
    """Run the guarded multistart refinement and authoritative comparison.

    The reoptimized directional check is deliberately executed in this
    function, immediately before any outer optimizer call.  A failed check is
    a hard stop rather than a warning.
    """

    output_dir = require_output_path(output_dir)
    gradient_check, center_theta = run_gradient_check(
        cfg, data, inner_mode=inner_mode
    )
    write_json(output_dir / "gradient_check.json", gradient_check)
    if not gradient_check["passed"]:
        raise RuntimeError(
            "reoptimized directional gradient check failed; outer optimization refused"
        )

    incumbent_eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    incumbent = authoritative_evaluate(
        incumbent_eta,
        cfg,
        data,
        allowance_percent=allowance_percent,
        initial_params=center_theta,
    )
    starts = global_initial_starts(
        cfg, data, allowance_percent=allowance_percent
    )
    candidates: list[tuple[dict[str, Any], CandidateEvaluation]] = []
    for index, start in enumerate(starts):
        eta_start = jnp.asarray(start["eta"], dtype=jnp.float64)
        initialized = solve_inner(
            eta_start,
            cfg,
            data.selection_problem,
            data.ritz_train_bank,
            mode=inner_mode,
            initial_params=center_theta if start["id"] == "incumbent" else None,
        )
        eta_final, theta_final, trace = refine_design(
            eta_start,
            initialized.params,
            cfg,
            data,
            allowance_percent=allowance_percent,
        )
        evaluation = authoritative_evaluate(
            eta_final,
            cfg,
            data,
            allowance_percent=allowance_percent,
            initial_params=theta_final,
        )
        row = {
            "id": start["id"],
            "start": start,
            "ending_eta": jax.device_get(eta_final).tolist(),
            "trace": trace,
            "authoritative": evaluation.payload,
        }
        candidates.append((row, evaluation))
        write_json(output_dir / f"candidate_{index:02d}.json", row)
        if evaluation.params is not None:
            save_candidate_checkpoint(
                output_dir / "checkpoints" / f"candidate_{index:02d}.npz",
                evaluation,
                role=f"selection_candidate_{index:02d}",
            )

    minimum_improvement = float(cfg["envelope"]["minimum_improvement"])
    selected = incumbent
    selected_id = "incumbent"
    if incumbent.payload["valid"]:
        incumbent_action = float(incumbent.payload["action"])
        eligible = [
            (row, evaluation)
            for row, evaluation in candidates
            if evaluation.payload["valid"]
            and float(evaluation.payload["action"])
            < incumbent_action - minimum_improvement
        ]
        if eligible:
            best_row, selected = min(
                eligible, key=lambda item: float(item[1].payload["action"])
            )
            selected_id = str(best_row["id"])
    else:
        eligible = [item for item in candidates if item[1].payload["valid"]]
        if eligible:
            best_row, selected = min(
                eligible, key=lambda item: float(item[1].payload["action"])
            )
            selected_id = str(best_row["id"])

    validation = authoritative_evaluate(
        jnp.asarray(selected.payload["eta"]),
        cfg,
        data,
        allowance_percent=allowance_percent,
        initial_params=selected.params,
        validation=True,
    )
    # Independent validation is a hard final gate.  It never participates in
    # candidate ranking and therefore cannot leak into selection.
    accepted = bool(selected.payload["valid"] and validation.payload["valid"])
    result = {
        "mode": "optimize",
        "allowance_percent": float(allowance_percent),
        "config": config_snapshot(cfg),
        "gradient_check": gradient_check,
        "law_risk_anchor": law_risk_anchor(cfg, data),
        "incumbent": incumbent.payload,
        "global_starts": starts,
        "candidates": [row for row, _ in candidates],
        "selected_id": selected_id,
        "selected": selected.payload,
        "validation": validation.payload,
        "accepted": accepted,
        "authoritative_improvement": (
            float(incumbent.payload["action"]) - float(selected.payload["action"])
            if incumbent.payload["valid"] and selected.payload["valid"] else None
        ),
    }
    write_json(output_dir / "result.json", result)
    if selected.params is not None:
        save_candidate_checkpoint(
            output_dir / "checkpoints" / "selected_selection.npz",
            selected,
            role="selected_selection",
        )
    if validation.params is not None:
        save_candidate_checkpoint(
            output_dir / "checkpoints" / "selected_validation.npz",
            validation,
            role="selected_validation",
        )
    return result


def certify_designs(
    cfg: dict[str, Any],
    data: PreparedExperiment,
    output_dir: Path,
    etas: list[Array],
    *,
    allowance_percent: float,
) -> dict[str, Any]:
    """Fresh selection and disjoint-validation audits for explicit designs."""

    output_dir = require_output_path(output_dir)
    rows: list[dict[str, Any]] = []
    for index, eta in enumerate(etas):
        selection = authoritative_evaluate(
            eta, cfg, data, allowance_percent=allowance_percent
        )
        validation = authoritative_evaluate(
            eta,
            cfg,
            data,
            allowance_percent=allowance_percent,
            initial_params=selection.params,
            validation=True,
        )
        row = {
            "id": f"design-{index:02d}",
            "selection": selection.payload,
            "validation": validation.payload,
            "valid": bool(selection.payload["valid"] and validation.payload["valid"]),
        }
        rows.append(row)
        if selection.params is not None:
            save_candidate_checkpoint(
                output_dir / "checkpoints" / f"design_{index:02d}_selection.npz",
                selection,
                role=f"certify_design_{index:02d}_selection",
            )
        if validation.params is not None:
            save_candidate_checkpoint(
                output_dir / "checkpoints" / f"design_{index:02d}_validation.npz",
                validation,
                role=f"certify_design_{index:02d}_validation",
            )
    result = {
        "mode": "certify",
        "allowance_percent": float(allowance_percent),
        "config": config_snapshot(cfg),
        "designs": rows,
        "all_valid": all(row["valid"] for row in rows),
    }
    write_json(output_dir / "result.json", result)
    return result
