from __future__ import annotations

"""Differentiable prospective Full objective for the preregistered v4 replicate.

The hidden microscopic validation system is deliberately absent from this module.
Every design-dependent quantity is built from the frozen aggregate-response table,
the endpoint-only reference rollout, and a fixed common-random-number observation
bank.  Authoritative host-side evaluation remains in :mod:`evaluator`.
"""

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common import nested_indices, trap_weights
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from mfsi.grid import RectangularGrid2D
from mfsi.poisson import PoissonConfig, solve_weighted_poisson
from prospective_data import TargetProspectiveData
from reflected_raster import (
    ReflectedRasterPlan,
    build_reflected_raster_plan,
    rasterize_reflected_with_plan,
)

jax.config.update("jax_enable_x64", True)


# All Pareto points share one frozen rollout and the same fidelity grids. Keep
# one resident plan per unique key instead of rebuilding/retaining another copy
# for Law, Tangent, Full, and every allowance.
_REFLECTED_PLAN_CACHE: dict[tuple[Any, ...], ReflectedRasterPlan] = {}


def _cached_reflected_plan(
    rollout_path: str | Path,
    nodes,
    time_indices: np.ndarray,
    grid: RectangularGrid2D,
    *,
    bandwidth: float,
    image_pairs: int,
) -> ReflectedRasterPlan:
    path = Path(rollout_path).resolve()
    key = (
        str(path),
        tuple(int(value) for value in time_indices),
        int(grid.nx),
        int(grid.ny),
        float(bandwidth),
        int(image_pairs),
    )
    plan = _REFLECTED_PLAN_CACHE.get(key)
    if plan is None:
        plan = build_reflected_raster_plan(
            nodes,
            grid,
            bandwidth=float(bandwidth),
            image_pairs=int(image_pairs),
        )
        _REFLECTED_PLAN_CACHE[key] = plan
    return plan


@dataclass(frozen=True)
class V4CRNBank:
    """Fixed reparameterized finite-sampling and detector-noise draws."""

    sampling_z: np.ndarray
    detector_z: np.ndarray

    @property
    def trials(self) -> int:
        return int(self.sampling_z.shape[0])

    def prefix(self, trials: int) -> "V4CRNBank":
        count = int(trials)
        if not 1 <= count <= self.trials:
            raise ValueError("CRN prefix must lie inside the frozen master bank")
        return V4CRNBank(self.sampling_z[:count], self.detector_z[:count])

    def subset(self, indices) -> "V4CRNBank":
        idx = np.asarray(indices, dtype=np.int32)
        return V4CRNBank(self.sampling_z[idx], self.detector_z[idx])

    def as_observation_bank(self) -> AggregateObservationBank:
        return AggregateObservationBank(self.sampling_z, self.detector_z)


def make_v4_crn_bank(cfg: dict[str, Any], trials: int) -> V4CRNBank:
    """Generate independent, explicitly seeded reparameterization draws."""
    shape = (
        int(trials),
        int(cfg["time"]["acquisition_nodes"]),
        int(cfg["measurement"]["n_sensors"]),
    )
    seeds = cfg["seeds"]
    sampling_rng = np.random.default_rng(int(seeds["selection_sampling_crn"]))
    detector_rng = np.random.default_rng(int(seeds["selection_detector_crn"]))
    return V4CRNBank(
        sampling_rng.standard_normal(shape),
        detector_rng.standard_normal(shape),
    )


def ensure_v4_crn_bank(path: str | Path, cfg: dict[str, Any], trials: int) -> V4CRNBank:
    path = Path(path)
    expected = (
        int(trials),
        int(cfg["time"]["acquisition_nodes"]),
        int(cfg["measurement"]["n_sensors"]),
    )
    seeds = cfg["seeds"]
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (
                tuple(data["sampling_z"].shape) == expected
                and int(data["sampling_seed"]) == int(seeds["selection_sampling_crn"])
                and int(data["detector_seed"]) == int(seeds["selection_detector_crn"])
            ):
                return V4CRNBank(
                    np.asarray(data["sampling_z"], dtype=np.float64),
                    np.asarray(data["detector_z"], dtype=np.float64),
                )
        raise RuntimeError("existing v4 selection CRN bank has incompatible preregistration")
    path.parent.mkdir(parents=True, exist_ok=True)
    bank = make_v4_crn_bank(cfg, trials)
    np.savez_compressed(
        path,
        role=np.asarray("v4_preregistered_selection_common_random_numbers"),
        sampling_z=bank.sampling_z,
        detector_z=bank.detector_z,
        sampling_seed=np.asarray(int(seeds["selection_sampling_crn"])),
        detector_seed=np.asarray(int(seeds["selection_detector_crn"])),
    )
    return bank


@dataclass(frozen=True)
class FullFidelity:
    name: str
    trials: int
    time_indices: np.ndarray
    time_weights: jax.Array
    grid: RectangularGrid2D
    poisson: PoissonConfig
    raster_plan: ReflectedRasterPlan


class V4DifferentiableObjective:
    """Expected operational Full action with fixed prospective observation CRNs."""

    def __init__(
        self,
        cfg: dict[str, Any],
        data: TargetProspectiveData,
        rollout_path: str | Path,
        *,
        raster_bandwidth: float | None = None,
    ):
        self.cfg = cfg
        # Native authoritative projection is retained for final evaluation.  The
        # gradient path uses the same equation with the established implicit JAX
        # VJP rather than differentiating through Newton iterations.
        gradient_cfg = copy.deepcopy(cfg)
        gradient_cfg["projection"]["backend"] = "jax"
        self.evaluator = ProspectiveEvaluator(
            gradient_cfg,
            data,
            rollout_path,
            raster_bandwidth=raster_bandwidth,
        )
        self.beta = float(cfg["v4"]["robustness_beta"])
        self._fidelities: dict[str, FullFidelity] = {}
        for name, block in cfg["v4"]["full_fidelities"].items():
            time_n = min(int(block["time_nodes"]), len(self.evaluator.times))
            time_idx = np.unique(
                np.rint(np.linspace(0, len(self.evaluator.times) - 1, time_n)).astype(np.int32)
            )
            grid = RectangularGrid2D(
                0.0, 2.0, 0.0, 1.0, int(block["grid_nx"]), int(block["grid_ny"])
            )
            poisson = PoissonConfig(
                dx=grid.require_isotropic_spacing(),
                operator_floor_rel=float(block["operator_floor_rel"]),
                cg_tol=float(block["cg_tol"]),
                cg_maxiter=int(block["cg_maxiter"]),
                gauge_strength=1.0,
            )
            weights = trap_weights(np.asarray(self.evaluator.times)[time_idx])
            raster_plan = _cached_reflected_plan(
                rollout_path,
                self.evaluator.nodes[jnp.asarray(time_idx, dtype=jnp.int32)],
                time_idx,
                grid,
                bandwidth=float(self.evaluator.authoritative_raster_bandwidth),
                image_pairs=int(self.evaluator.reflected_image_pairs),
            )
            self._fidelities[name] = FullFidelity(
                name=name,
                trials=int(block["trials"]),
                time_indices=time_idx,
                time_weights=jnp.asarray(weights, dtype=jnp.float64),
                grid=grid,
                poisson=poisson,
                raster_plan=raster_plan,
            )

    def fidelity(self, name: str) -> FullFidelity:
        try:
            return self._fidelities[name]
        except KeyError as exc:
            raise KeyError(f"unknown Full gradient fidelity: {name}") from exc

    def _project(self, eta, sampling_z, detector_z):
        mean, second = self.evaluator.prospective_population(eta)
        cross_second = self.evaluator.prospective_cross_second(eta)
        bank = AggregateObservationBank(sampling_z, detector_z)
        projection, weights, forcing, tangent, spline_rss = self.evaluator._project(
            eta, mean, second, bank, response_cross_second=cross_second
        )
        projected_qoi = jnp.einsum(
            "btn,tnk->btk", weights, self.evaluator.reference_qois
        )
        qoi_error = (
            projected_qoi
            - jnp.asarray(self.evaluator.data.scientific_qoi_predictions)[None, :, :]
        ) / jnp.asarray(self.evaluator.data.qoi_scales)[None, None, :]
        risk_trials = jnp.sum(
            self.evaluator.time_weights[None, :, None] * qoi_error * qoi_error,
            axis=(1, 2),
        )
        return projection, weights, forcing, tangent, spline_rss, risk_trials

    def risk_trials(self, eta, sampling_z, detector_z):
        return self._project(eta, sampling_z, detector_z)[-1]

    def risk_mean(self, eta, sampling_z, detector_z):
        return jnp.mean(self.risk_trials(eta, sampling_z, detector_z))

    def reconstruction(self, eta, sampling_z, detector_z):
        mean, second = self.evaluator.prospective_population(eta)
        return self.evaluator.reconstruct(
            mean,
            second,
            AggregateObservationBank(sampling_z, detector_z),
            response_cross_second=self.evaluator.prospective_cross_second(eta),
        )

    def full_trials(self, eta, sampling_z, detector_z, fidelity_name: str):
        fidelity = self.fidelity(fidelity_name)
        projection, weights, forcing, _, _, risk_trials = self._project(
            eta, sampling_z, detector_z
        )
        idx = jnp.asarray(fidelity.time_indices, dtype=jnp.int32)
        weights = weights[:, idx]
        forcing = forcing[:, idx]
        rasters = rasterize_reflected_with_plan(
            weights, forcing, fidelity.raster_plan
        )
        solved = jax.vmap(
            jax.vmap(lambda q, h: solve_weighted_poisson(q, h, fidelity.poisson))
        )(rasters.q, rasters.h)
        action_trials = jnp.sum(
            solved.action * fidelity.time_weights[None, :], axis=1
        )
        residual = jnp.max(
            jnp.linalg.norm(projection.residual, axis=-1), axis=1
        )
        ess = jnp.min(projection.ess_fraction, axis=1)
        poisson_residual = jnp.max(solved.relative_residual, axis=1)
        return action_trials, risk_trials, residual, ess, poisson_residual

    @staticmethod
    def robust_score(action_trials, beta: float):
        mean = jnp.mean(action_trials)
        if int(action_trials.shape[0]) > 1:
            sd = jnp.std(action_trials, ddof=1)
        else:
            sd = jnp.asarray(0.0, dtype=action_trials.dtype)
        return mean + float(beta) * sd, mean, sd

    def full_score(self, eta, sampling_z, detector_z, fidelity_name: str):
        action, _, _, _, _ = self.full_trials(
            eta, sampling_z, detector_z, fidelity_name
        )
        return self.robust_score(action, self.beta)[0]

    def constrained_full_loss(
        self,
        eta,
        sampling_z,
        detector_z,
        fidelity_name: str,
        risk_limit: float,
    ):
        action, risks, residual, ess, poisson = self.full_trials(
            eta, sampling_z, detector_z, fidelity_name
        )
        score, _, _ = self.robust_score(action, self.beta)
        v4 = self.cfg["v4"]
        scale = float(v4["constraint_softplus_scale"])

        def positive_part(x):
            return scale * jax.nn.softplus(x / scale)

        risk_violation = positive_part(jnp.mean(risks) - float(risk_limit))
        numerical = (
            positive_part(jnp.max(residual) - float(self.cfg["validity"]["max_projection_residual"])) ** 2
            + positive_part(float(self.cfg["validity"]["min_ess_fraction"]) - jnp.min(ess)) ** 2
            + positive_part(
                jnp.max(poisson)
                - float(v4["gradient_max_poisson_relative_residual"])
            ) ** 2
        )
        return (
            score
            + float(v4["risk_penalty"]) * risk_violation * risk_violation
            + float(v4["numerical_penalty"]) * numerical
        )


def geometry_penalty(eta, cfg: dict[str, Any]):
    """Smooth pairwise-separation barrier; box feasibility is projected by Adam."""
    centers = jnp.asarray(eta, dtype=jnp.float64).reshape(
        (int(cfg["measurement"]["n_sensors"]), 2)
    )
    count = centers.shape[0]
    if count < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)
    delta = centers[:, None, :] - centers[None, :, :]
    distance = jnp.sqrt(jnp.maximum(jnp.sum(delta * delta, axis=-1), 1.0e-16))
    upper_i, upper_j = np.triu_indices(count, k=1)
    pair_distance = distance[jnp.asarray(upper_i), jnp.asarray(upper_j)]
    scale = float(cfg["v4"]["geometry_softplus_scale"])
    violation = scale * jax.nn.softplus(
        (float(cfg["measurement"]["min_separation"]) - pair_distance) / scale
    )
    return float(cfg["v4"]["geometry_penalty"]) * jnp.sum(violation * violation)


def project_box(eta, cfg: dict[str, Any]):
    centers = jnp.asarray(eta, dtype=jnp.float64).reshape((-1, 2))
    margin = float(cfg["measurement"]["boundary_margin"])
    x = jnp.clip(centers[:, 0], margin, 2.0 - margin)
    y = jnp.clip(centers[:, 1], margin, 1.0 - margin)
    return jnp.stack([x, y], axis=-1).reshape((-1,))


def distribution(values) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "sd": None, "se": None}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "se": float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "max": float(np.max(x)),
    }


def canonical_geometry_key(eta, decimals: int = 7) -> tuple[float, ...]:
    # Point sensors are labelled throughout optimization and finite-CRN
    # evaluation.  Permuting centers is therefore not a cache-safe identity.
    return tuple(np.round(np.asarray(eta, dtype=np.float64).reshape(-1), int(decimals)))


def acquisition_reparameterization_formula(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "finite_sampling": "mean + chol(full_sensor_covariance/finite_n) * fixed_sampling_z",
        "detector_noise": "noise_std * fixed_detector_z",
        "endpoint_policy": "exact predicted population endpoints; stochastic perturbations disabled",
        "finite_n": int(cfg["measurement"]["finite_n"]),
        "detector_noise_std": float(cfg["measurement"]["noise_std"]),
        "acquisition_indices": nested_indices(
            int(cfg["time"]["scientific_nodes"]),
            int(cfg["time"]["acquisition_nodes"]),
        ).tolist(),
    }


__all__ = [
    "FullFidelity",
    "V4CRNBank",
    "V4DifferentiableObjective",
    "acquisition_reparameterization_formula",
    "canonical_geometry_key",
    "distribution",
    "ensure_v4_crn_bank",
    "geometry_penalty",
    "make_v4_crn_bank",
    "project_box",
]
