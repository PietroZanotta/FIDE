from __future__ import annotations

"""Differentiable equal-weight multi-reference objectives for v6."""

from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp

from prospective_data import TargetProspectiveData
from reflected_raster import common_reference_scott_bandwidth
from v4_objective import V4DifferentiableObjective

jax.config.update("jax_enable_x64", True)


class V6MultiReferenceObjective:
    def __init__(
        self,
        cfg: dict[str, Any],
        data: TargetProspectiveData,
        rollout_paths: Iterable[str | Path],
    ):
        paths = [Path(path) for path in rollout_paths]
        if not paths:
            raise ValueError("v6 requires at least one design reference")
        self.cfg = cfg
        (
            self.common_raster_bandwidth,
            self.per_reference_raster_bandwidths,
        ) = common_reference_scott_bandwidth(paths)
        self.objectives = [
            V4DifferentiableObjective(
                cfg,
                data,
                path,
                raster_bandwidth=self.common_raster_bandwidth,
            )
            for path in paths
        ]

    @property
    def reference_count(self) -> int:
        return len(self.objectives)

    def fidelity(self, name: str):
        return self.objectives[0].fidelity(name)

    def risk_trials_by_reference(self, eta, sampling_z, detector_z):
        return jnp.stack(
            [obj.risk_trials(eta, sampling_z, detector_z) for obj in self.objectives],
            axis=0,
        )

    def risk_mean(self, eta, sampling_z, detector_z):
        return jnp.mean(self.risk_trials_by_reference(eta, sampling_z, detector_z))

    def full_trials_by_reference(self, eta, sampling_z, detector_z, fidelity_name: str):
        rows = [
            obj.full_trials(eta, sampling_z, detector_z, fidelity_name)
            for obj in self.objectives
        ]
        return tuple(jnp.stack([row[index] for row in rows], axis=0) for index in range(5))

    @staticmethod
    def robust_score(action_by_reference, beta: float):
        flat = jnp.reshape(action_by_reference, (-1,))
        mean = jnp.mean(flat)
        sd = jnp.std(flat, ddof=1) if int(flat.shape[0]) > 1 else jnp.asarray(0.0)
        return mean + float(beta) * sd, mean, sd

    def full_score(self, eta, sampling_z, detector_z, fidelity_name: str, beta: float):
        actions = self.full_trials_by_reference(
            eta, sampling_z, detector_z, fidelity_name
        )[0]
        return self.robust_score(actions, beta)[0]

    def constrained_full_loss(
        self,
        eta,
        sampling_z,
        detector_z,
        fidelity_name: str,
        risk_limits,
        beta: float,
    ):
        actions, risks, residual, ess, poisson = self.full_trials_by_reference(
            eta, sampling_z, detector_z, fidelity_name
        )
        score, _, _ = self.robust_score(actions, beta)
        block = self.cfg["v4"]
        scale = float(block["constraint_softplus_scale"])
        positive = lambda x: scale * jax.nn.softplus(x / scale)
        mean_risk = jnp.mean(risks, axis=1)
        risk_violation = positive(mean_risk - jnp.asarray(risk_limits))
        numerical = (
            jnp.sum(positive(jnp.max(residual, axis=1) - float(self.cfg["validity"]["max_projection_residual"])) ** 2)
            + jnp.sum(positive(float(self.cfg["validity"]["min_ess_fraction"]) - jnp.min(ess, axis=1)) ** 2)
            + jnp.sum(positive(jnp.max(poisson, axis=1) - float(block["gradient_max_poisson_relative_residual"])) ** 2)
        )
        return (
            score
            + float(block["risk_penalty"]) * jnp.sum(risk_violation * risk_violation)
            + float(block["numerical_penalty"]) * numerical
        )

    def tangent_trials_by_reference(self, eta, sampling_z, detector_z):
        rows = []
        for objective in self.objectives:
            projection, _, _, tangent, _, risks = objective._project(
                eta, sampling_z, detector_z
            )
            action = jnp.sum(objective.evaluator.time_weights[None, :] * tangent, axis=1)
            residual = jnp.max(jnp.linalg.norm(projection.residual, axis=-1), axis=1)
            ess = jnp.min(projection.ess_fraction, axis=1)
            rows.append((action, risks, residual, ess))
        return tuple(jnp.stack([row[index] for row in rows], axis=0) for index in range(4))

    def tangent_score(self, eta, sampling_z, detector_z):
        return jnp.mean(self.tangent_trials_by_reference(eta, sampling_z, detector_z)[0])

    def constrained_tangent_loss(self, eta, sampling_z, detector_z, risk_limits):
        action, risks, residual, ess = self.tangent_trials_by_reference(
            eta, sampling_z, detector_z
        )
        block = self.cfg["v4"]
        scale = float(block["constraint_softplus_scale"])
        positive = lambda x: scale * jax.nn.softplus(x / scale)
        risk_violation = positive(jnp.mean(risks, axis=1) - jnp.asarray(risk_limits))
        numerical = (
            jnp.sum(positive(jnp.max(residual, axis=1) - float(self.cfg["validity"]["max_projection_residual"])) ** 2)
            + jnp.sum(positive(float(self.cfg["validity"]["min_ess_fraction"]) - jnp.min(ess, axis=1)) ** 2)
        )
        return (
            jnp.mean(action)
            + float(block["risk_penalty"]) * jnp.sum(risk_violation * risk_violation)
            + float(block["numerical_penalty"]) * numerical
        )


__all__ = ["V6MultiReferenceObjective"]
