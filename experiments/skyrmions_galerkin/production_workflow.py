"""Strict production-artifact reproduction and gated Galerkin workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .deep_ritz import CertificateConfig, audit_deep_ritz, load_ritz_checkpoint
from .domain import ConfigurationBank
from .experiment import _physics_config, _time_weights
from .full_gradient import envelope_diagnostics, forcing_state, reconstruct_moments
from .measurements import LocalDensitySensors
from .production_artifacts import PRODUCTION_ROOT, require_production_output_path
from .risk import many_body_features, whitening_from_truth
from .workflow import (
    PreparedExperiment,
    _make_problem,
    _reference_bank,
    hard_forcing_audit,
    selection_risk,
    write_json,
)

Array = jax.Array

PUBLISHED_TARGETS = {
    "law_risk": 5.186549474478041,
    "risk_limit": 5.342145958712383,
    "selected_risk": 5.340106050966004,
    "selected_action": 0.20345379368395117,
    "maximum_projection_residual": 8.027942557792514e-11,
    "minimum_ess_fraction": 0.06916265902448057,
    "maximum_forcing_mean": 9.75849490103542e-09,
    "maximum_weak_residual": 0.08776182994308217,
    "maximum_energy_residual": 0.06131636504009692,
    "maximum_gauge_residual": 1.375039405254741e-16,
    "maximum_moment_rate_residual": 0.024428864519093547,
}


def _load_truth(path: Path) -> tuple[Array, Array, Array]:
    with np.load(path, allow_pickle=False) as arrays:
        return (
            jnp.asarray(arrays["times"], dtype=jnp.float64),
            jnp.asarray(arrays["design"], dtype=jnp.float64),
            jnp.asarray(arrays["validation"], dtype=jnp.float64),
        )


def _load_reference_bank(path: Path):
    with np.load(path, allow_pickle=False) as arrays:
        return _reference_bank({
            "configurations": arrays["configurations"],
            "velocity": arrays["velocity"],
            "base_weights": arrays["base_weights"],
        })


def load_production_data(cfg: dict[str, Any], artifact_dir: Path) -> PreparedExperiment:
    """Load the already-copied banks directly; never train, generate, or write."""

    artifact_dir = require_production_output_path(artifact_dir)
    times, truth_design_values, truth_validation_values = _load_truth(
        artifact_dir / "truth_banks.npz"
    )
    expected_times = jnp.linspace(
        0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64
    )
    if not bool(jnp.array_equal(times, expected_times)):
        raise RuntimeError("copied production time grid does not match authoritative config")
    physics = _physics_config(cfg)
    family = LocalDensitySensors(
        n_sensors=int(cfg["measurement"]["n_sensors"]),
        width=float(cfg["measurement"]["sensor_width"]),
        box=tuple(physics.box),
        min_separation=float(cfg["measurement"]["min_separation"]),
    )
    design = ConfigurationBank(times, truth_design_values)
    validation = ConfigurationBank(times, truth_validation_values)
    offsets = cfg["banks"]["seed_offsets"]
    time_weights = _time_weights(times)
    selection_problem = _make_problem(
        cfg, design, family, times, time_weights,
        noise_seed=int(cfg["seed"]) + int(offsets["observation"]),
    )
    validation_problem = _make_problem(
        cfg, validation, family, times, time_weights,
        noise_seed=int(cfg["seed"]) + int(offsets["observation"]) + 10000,
    )
    banks = {
        name: _load_reference_bank(artifact_dir / f"reference_bank_{name}.npz")
        for name in (
            "projection", "ritz_train", "ritz_audit",
            "validation_fit", "validation_audit",
        )
    }
    box = tuple(physics.box)
    selection_truth_features = many_body_features(truth_design_values, box)
    validation_truth_features = many_body_features(truth_validation_values, box)
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
            banks["projection"].configurations, box
        ),
        selection_truth_means=jnp.mean(selection_truth_features, axis=1),
        validation_reference_features=many_body_features(
            banks["validation_fit"].configurations, box
        ),
        validation_truth_means=jnp.mean(validation_truth_features, axis=1),
        whitening=whitening,
    )


def _forcing_payload(eta: Array, problem, bank, reconstruction) -> dict[str, Any]:
    state = forcing_state(eta, problem, bank, reconstruction)
    hard = hard_forcing_audit(eta, problem, bank)
    post_mean = jnp.einsum("tn,tn->t", state.projection.weights, state.forcing)
    return {
        **hard,
        "maximum_post_centering_forcing_mean": float(jnp.max(jnp.abs(post_mean))),
        "projection_residual_by_time": jax.device_get(
            jnp.linalg.norm(state.projection.residual, axis=-1)
        ).tolist(),
        "ess_fraction_by_time": jax.device_get(state.projection.ess_fraction).tolist(),
        "covariance_condition_by_time": jax.device_get(state.covariance_condition).tolist(),
        "lambda": jax.device_get(state.projection.lam).tolist(),
        "lambda_norm_by_time": jax.device_get(
            jnp.linalg.norm(state.projection.lam, axis=-1)
        ).tolist(),
        "lambda_dot": jax.device_get(state.lambda_dot).tolist(),
        "lambda_dot_norm_by_time": jax.device_get(
            jnp.linalg.norm(state.lambda_dot, axis=-1)
        ).tolist(),
    }


def _discrepancy(actual: float, target: float) -> dict[str, float]:
    return {
        "actual": float(actual),
        "target": float(target),
        "absolute": abs(float(actual) - float(target)),
        "relative": abs(float(actual) - float(target)) / max(abs(float(target)), 1.0e-30),
    }


def _deep_ritz_reproduction(
    cfg: dict[str, Any], data: PreparedExperiment, artifact_dir: Path,
    eta: Array, reconstruction, audit_forcing,
) -> dict[str, Any]:
    params, metadata = load_ritz_checkpoint(artifact_dir / "ritz_full.npz")
    diagnostics = envelope_diagnostics(
        params, eta, data.selection_problem, data.ritz_train_bank
    )
    audit_state = forcing_state(
        eta, data.selection_problem, data.ritz_audit_bank, reconstruction
    )
    certificate = audit_deep_ritz(
        params,
        data.ritz_audit_bank.configurations,
        audit_state.projection.weights,
        audit_state.forcing,
        data.selection_problem.times,
        data.selection_problem.time_weights,
        family=data.selection_problem.family,
        eta=eta,
        reference_velocity=data.ritz_audit_bank.velocity,
        target_derivatives=reconstruction.derivatives,
        cfg=CertificateConfig(**cfg["certificates"]),
        box=data.selection_problem.box,
        chunk_size=1024,
    )
    return {
        "checkpoint": str(artifact_dir / "ritz_full.npz"),
        "checkpoint_metadata": metadata,
        "train_ritz_objective": float(diagnostics.ritz_objective),
        "train_kinetic_action": float(diagnostics.full_energy),
        "train_energy_identity_relerr": float(diagnostics.energy_identity_relerr),
        "held_out_certificate": certificate,
        "held_out_forcing_audit": audit_forcing,
        "action_discrepancy": _discrepancy(
            certificate["action"], PUBLISHED_TARGETS["selected_action"]
        ),
    }


def run_production_reproduction(
    cfg: dict[str, Any], artifact_dir: Path, output_dir: Path
) -> tuple[dict[str, Any], PreparedExperiment]:
    output_dir = require_production_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_production_data(cfg, artifact_dir)
    eta = jnp.asarray(cfg["envelope"]["eta0"], dtype=jnp.float64)
    problem = data.selection_problem
    reconstruction = reconstruct_moments(eta, problem)
    reconstruction_repeat = reconstruct_moments(eta, problem)
    deterministic_error = max(
        float(jnp.max(jnp.abs(reconstruction.values - reconstruction_repeat.values))),
        float(jnp.max(jnp.abs(
            reconstruction.derivatives - reconstruction_repeat.derivatives
        ))),
    )
    audits = {
        "projection": _forcing_payload(
            eta, problem, data.projection_bank, reconstruction
        ),
        "ritz_train": _forcing_payload(
            eta, problem, data.ritz_train_bank, reconstruction
        ),
        "ritz_audit": _forcing_payload(
            eta, problem, data.ritz_audit_bank, reconstruction
        ),
    }
    risk = float(selection_risk(eta, data))
    law_eta = jnp.asarray(cfg["envelope"]["law_eta"], dtype=jnp.float64)
    law_risk = float(selection_risk(law_eta, data))
    risk_limit = 1.03 * law_risk
    gate_a = bool(
        deterministic_error == 0.0
        and all(audit["valid"] for audit in audits.values())
        and audits["ritz_train"]["maximum_projection_residual"] <= 2.0e-6
        and audits["ritz_train"]["minimum_ess_fraction"] >= 0.05
        and audits["ritz_train"]["maximum_forcing_mean"] <= 2.0e-7
        and audits["ritz_train"]["maximum_covariance_condition"] <= 1.0e10
    )
    result: dict[str, Any] = {
        "eta0": jax.device_get(eta).tolist(),
        "artifact_dir": str(artifact_dir),
        "scientific_risk": risk,
        "law_risk": law_risk,
        "risk_ceiling_3_percent": risk_limit,
        "reconstruction_determinism_max_abs": deterministic_error,
        "acquisition_indices": jax.device_get(problem.acquisition_indices).tolist(),
        "finite_configuration_count": problem.finite_configuration_count,
        "time_nodes": jax.device_get(problem.times).tolist(),
        "time_weights": jax.device_get(problem.time_weights).tolist(),
        "c_eta": jax.device_get(reconstruction.values).tolist(),
        "cdot_eta": jax.device_get(reconstruction.derivatives).tolist(),
        "reconstruction_residual_sum_squares": float(reconstruction.residual_sum_squares),
        "reconstruction_roughness": float(reconstruction.roughness),
        "forcing_audits": audits,
        "target_discrepancies": {
            "law_risk": _discrepancy(law_risk, PUBLISHED_TARGETS["law_risk"]),
            "risk_ceiling": _discrepancy(risk_limit, PUBLISHED_TARGETS["risk_limit"]),
            "selected_risk": _discrepancy(risk, PUBLISHED_TARGETS["selected_risk"]),
            "train_projection_residual": _discrepancy(
                audits["ritz_train"]["maximum_projection_residual"],
                PUBLISHED_TARGETS["maximum_projection_residual"],
            ),
            "train_minimum_ess": _discrepancy(
                audits["ritz_train"]["minimum_ess_fraction"],
                PUBLISHED_TARGETS["minimum_ess_fraction"],
            ),
            "train_forcing_mean": _discrepancy(
                audits["ritz_train"]["maximum_forcing_mean"],
                PUBLISHED_TARGETS["maximum_forcing_mean"],
            ),
        },
        "gate_a_passed": gate_a,
    }
    if gate_a:
        result["deep_ritz_reproduction"] = _deep_ritz_reproduction(
            cfg, data, artifact_dir, eta, reconstruction, audits["ritz_audit"]
        )
        result["outcome_classification"] = (
            "B. PRODUCTION GALERKIN SOLVER VALID, ETA GRADIENT NOT YET VALIDATED"
        )
    else:
        result["deep_ritz_reproduction"] = {"ran": False, "reason": "Gate A failed"}
        result["outcome_classification"] = "D. PRODUCTION PROJECTED-LAW REPRODUCTION FAILED"
    write_json(output_dir / "result.json", result)
    return result, data


__all__ = [
    "PUBLISHED_TARGETS", "load_production_data", "run_production_reproduction",
]
