"""Fixed-checkpoint platform equivalence for authoritative Deep Ritz results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .deep_ritz import CertificateConfig, audit_deep_ritz, load_ritz_checkpoint
from .full_gradient import envelope_diagnostics, forcing_state, reconstruct_moments
from .production_workflow import load_production_data
from .workflow import hard_forcing_audit, selection_risk, validation_risk


def evaluate_fixed_authoritative_checkpoint(
    cfg: dict[str, Any], artifact_dir: Path, checkpoint: Path,
    eta: jax.Array, *, validation: bool = False,
) -> dict[str, Any]:
    """Evaluate a frozen network without optimization or checkpoint mutation."""

    data = load_production_data(cfg, artifact_dir)
    params, metadata = load_ritz_checkpoint(checkpoint)
    if validation:
        problem, train, audit = (
            data.validation_problem, data.validation_fit_bank, data.validation_audit_bank
        )
        risk = validation_risk(eta, data)
    else:
        problem, train, audit = (
            data.selection_problem, data.ritz_train_bank, data.ritz_audit_bank
        )
        risk = selection_risk(eta, data)
    reconstruction = reconstruct_moments(eta, problem)
    diagnostics = envelope_diagnostics(params, eta, problem, train)
    audit_state = forcing_state(eta, problem, audit, reconstruction)
    certificate = audit_deep_ritz(
        params, audit.configurations, audit_state.projection.weights,
        audit_state.forcing, problem.times, problem.time_weights,
        family=problem.family, eta=eta, reference_velocity=audit.velocity,
        target_derivatives=reconstruction.derivatives,
        cfg=CertificateConfig(**cfg["certificates"]), box=problem.box,
        chunk_size=min(1024, int(audit.configurations.shape[1])),
    )
    return {
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "eta": jax.device_get(eta).tolist(),
        "risk": float(risk),
        "train_ritz_objective": float(diagnostics.ritz_objective),
        "train_kinetic_action": float(diagnostics.full_energy),
        "train_energy_identity_relerr": float(diagnostics.energy_identity_relerr),
        "train_forcing": hard_forcing_audit(eta, problem, train),
        "audit_forcing": hard_forcing_audit(eta, problem, audit),
        "certificate": certificate,
    }


__all__ = ["evaluate_fixed_authoritative_checkpoint"]
