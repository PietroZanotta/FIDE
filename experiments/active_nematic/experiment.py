"""MFSI design layer for normalized active-nematic +1/2-defect laws.

The class mirrors the vortices experiment's geometry -> sparse observation ->
anchored moment reconstruction -> I-projection -> law/tangent/full evaluation
pipeline.  Shared MFSI abstractions are imported directly.  Only periodic
measurement geometry, periodic law risk, and the position-only periodic Poisson
discretization are experiment-local.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.design import OptimizerConfig, optimize_multistart_candidates
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.particles import ParticleMFSIConfig, particle_mfsi_state
from mfsi.projection import EmpiricalIProjector, IProjectionConfig

try:
    from .measurements import (
        PeriodicGaussianSensors,
        periodic_separation_violation,
        random_periodic_sensor_starts,
    )
    from .periodic_numerics import (
        PeriodicGrid2D,
        PeriodicPoissonConfig,
        rasterize_periodic_particles,
        solve_periodic_weighted_poisson,
    )
    from .risk import (
        PeriodicHistogramGrid,
        histogram_mass,
        multiscale_periodic_kernel_fft,
        periodic_grid_mmd2,
        trapezoid_weights,
    )
except ImportError:  # pragma: no cover - direct experiment-script convention.
    from measurements import PeriodicGaussianSensors, periodic_separation_violation, random_periodic_sensor_starts
    from periodic_numerics import PeriodicGrid2D, PeriodicPoissonConfig, rasterize_periodic_particles, solve_periodic_weighted_poisson
    from risk import PeriodicHistogramGrid, histogram_mass, multiscale_periodic_kernel_fft, periodic_grid_mmd2, trapezoid_weights

Array = jax.Array


class PolarityFullActionUnavailable(NotImplementedError):
    """Raised instead of silently replacing the requested 3-D full action."""


class ObservationTrialBank(NamedTuple):
    sample_indices: Array  # [trial,acquisition_time,finite_n]
    detector_z: Array      # [trial,acquisition_time,observable]


class Reconstruction(NamedTuple):
    c: Array
    c_dot: Array
    residual_sum_squares: Array
    roughness: Array


@dataclass(frozen=True)
class TrialMetrics:
    law_risk: float
    tangent_action: float
    full_action: float
    max_calibration_residual: float
    min_ess_fraction: float
    max_poisson_relative_residual: float
    valid: bool


@dataclass(frozen=True)
class DesignComparison:
    law_eta: Array
    tangent_eta: Array
    full_eta: Array | None
    risk_star: float
    risk_max: float
    candidates: dict[str, list[dict[str, Any]]]


def nested_acquisition_indices(time_n: int, count: int) -> np.ndarray:
    """Same endpoint-including, nested acquisition policy used by vortices."""
    if count < 2 or count > time_n:
        raise ValueError("acquisition count must satisfy 2 <= count <= time_n")
    index = np.unique(np.rint(np.linspace(0, time_n - 1, count)).astype(np.int32))
    if len(index) != count:
        interior = np.arange(1, time_n - 1, dtype=np.int32)
        index = np.concatenate(
            [[0], interior[np.rint(np.linspace(0, len(interior) - 1, count - 2)).astype(int)], [time_n - 1]]
        ).astype(np.int32)
    return index


def make_observation_bank(
    *,
    seed: int,
    namespace: int,
    trials: int,
    acquisition_count: int,
    finite_n: int,
    truth_particle_count: int,
    n_observables: int,
) -> ObservationTrialBank:
    """Freeze common random numbers independently of physical/reference seeds."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(namespace)]))
    indices = rng.integers(
        0,
        int(truth_particle_count),
        size=(int(trials), int(acquisition_count), int(finite_n)),
        dtype=np.int32,
    )
    z = rng.standard_normal((int(trials), int(acquisition_count), int(n_observables)))
    return ObservationTrialBank(jnp.asarray(indices), jnp.asarray(z, dtype=jnp.float64))


class ActiveNematicExperiment:
    """Controlled-oracle benchmark over frozen truth and reference particle banks."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        times: Array,
        truth_particles: Array,
        reference_nodes: Array,
        reference_velocity: Array,
        reference_weights: Array,
    ):
        self.cfg = cfg
        self.times = jnp.asarray(times, dtype=jnp.float64)
        self.truth_particles = jnp.asarray(truth_particles, dtype=jnp.float64)
        self.reference_nodes = jnp.asarray(reference_nodes, dtype=jnp.float64)
        self.reference_velocity = jnp.asarray(reference_velocity, dtype=jnp.float64)
        self.reference_weights = jnp.asarray(reference_weights, dtype=jnp.float64)
        if self.truth_particles.ndim != 3 or self.reference_nodes.ndim != 3:
            raise ValueError("truth and reference particles must have shape [time,sample,state_dim]")
        if self.reference_nodes.shape != self.reference_velocity.shape:
            raise ValueError("reference nodes and velocities must have identical shapes")
        if self.reference_weights.shape != self.reference_nodes.shape[:2]:
            raise ValueError("reference weights must have shape [time,sample]")
        if self.truth_particles.shape[0] != len(self.times) or self.reference_nodes.shape[0] != len(self.times):
            raise ValueError("all banks must share the configured time dimension")
        if self.truth_particles.shape[-1] != self.reference_nodes.shape[-1]:
            raise ValueError("truth and reference state dimensions differ")
        self.state_dim = int(self.truth_particles.shape[-1])
        if self.state_dim not in (2, 3):
            raise ValueError("active-nematic scientific state dimension must be 2 or 3")

        measurement = cfg["measurement"]
        channels = tuple(measurement.get("channels", ["occupancy"]))
        self.family = PeriodicGaussianSensors(
            box_size=float(cfg["physics"]["box_size"]),
            width=float(measurement["sensor_width"]),
            n_sensors=int(measurement["n_sensors"]),
            channels=channels,
        )
        if self.family.requires_polarity and self.state_dim != 3:
            raise ValueError("polarity sensor channels cannot be used in position-only mode")
        self.acq_idx = jnp.asarray(
            nested_acquisition_indices(len(self.times), int(measurement["acquisition_k"])),
            dtype=jnp.int32,
        )
        spline = cfg.get("moment_reconstruction", {})
        self.reconstructor = AnchoredCubicSplineReconstructor(
            self.times[self.acq_idx],
            self.times,
            AnchoredCubicSplineConfig(
                internal_knots=int(spline.get("internal_knots", 3)),
                smoothing=float(spline.get("smoothing", 1.0e-4)),
                ridge_rel=float(spline.get("ridge_rel", 1.0e-10)),
                roughness_quadrature_order=int(spline.get("roughness_quadrature_order", 8)),
            ),
        )
        projection = cfg.get("projection", {})
        self.projector = EmpiricalIProjector(IProjectionConfig(
            max_steps=int(projection.get("max_steps", 300)),
            residual_tol=float(projection.get("residual_tol", 1.0e-8)),
            newton_ridge=float(projection.get("newton_ridge", 1.0e-7)),
            step_cap=float(projection.get("step_cap", 20.0)),
            lambda_clip=float(projection.get("lambda_clip", 1000.0)),
            line_search_steps=int(projection.get("line_search_steps", 8)),
        ))
        particle = cfg.get("particle_mfsi", {})
        self.particle_cfg = ParticleMFSIConfig(
            covariance_ridge=float(particle.get("covariance_ridge", 1.0e-7)),
            tangent_ridge=float(particle.get("tangent_ridge", 1.0e-7)),
        )
        action = cfg.get("full_action", {})
        self.grid = PeriodicGrid2D(float(cfg["physics"]["box_size"]), int(action.get("grid_n", 64)))
        self.poisson_cfg = PeriodicPoissonConfig(
            operator_floor_rel=float(action.get("operator_floor_rel", 2.0e-5)),
            cg_tol=float(action.get("cg_tol", 1.0e-7)),
            cg_maxiter=int(action.get("cg_maxiter", 520)),
            gauge_strength=float(action.get("gauge_strength", 1.0)),
        )
        self.raster_bandwidth = float(action.get("raster_bandwidth", 0.0))
        self.time_weights = trapezoid_weights(self.times)
        self.periods = jnp.asarray(
            [self.grid.box_size, self.grid.box_size] + ([2.0 * np.pi] if self.state_dim == 3 else []),
            dtype=jnp.float64,
        )
        self.mmd_bandwidths = jnp.asarray(cfg["law"].get("mmd_bandwidths", [0.5, 1.0, 2.0]))
        default_shape = [64, 64] if self.state_dim == 2 else [48, 48, 24]
        law_shape = tuple(int(value) for value in cfg["law"].get("grid_shape", default_shape))
        self.law_grid = PeriodicHistogramGrid(tuple(float(value) for value in self.periods), law_shape)
        self.law_kernel_fft = multiscale_periodic_kernel_fft(self.law_grid, self.mmd_bandwidths)
        hidden_weights = jnp.full(
            (self.truth_particles.shape[1],), 1.0 / self.truth_particles.shape[1], dtype=jnp.float64
        )
        self.truth_masses = jax.vmap(
            lambda samples: histogram_mass(samples, hidden_weights, self.law_grid)
        )(self.truth_particles)

    @property
    def full_action_supported(self) -> bool:
        return self.state_dim == 2

    def _geometry(self, eta: Array) -> tuple[Array, Array, Array]:
        eta = self.family.canonicalize(eta)
        return (
            self.family.features(self.truth_particles, eta),
            self.family.features(self.reference_nodes, eta),
            self.family.feature_gradients(self.reference_nodes, eta),
        )

    def _reconstruct(self, phi_truth: Array, bank: ObservationTrialBank, trial: int | Array) -> Reconstruction:
        phi_acq = phi_truth[self.acq_idx]
        sampled = jax.vmap(lambda features, index: jnp.mean(features[index], axis=0))(
            phi_acq, bank.sample_indices[trial]
        )
        exact = jnp.mean(phi_acq, axis=1)
        observed = sampled + float(self.cfg["measurement"].get("obs_noise_std", 0.0)) * bank.detector_z[trial]
        endpoint = (self.acq_idx == 0) | (self.acq_idx == len(self.times) - 1)
        observed = jnp.where(endpoint[:, None], exact, observed)
        fit = self.reconstructor.reconstruct(observed, exact[0], exact[-1])
        return Reconstruction(fit.c, fit.c_dot, fit.residual_sum_squares, fit.roughness)

    def _trial_values(self, eta: Array, bank: ObservationTrialBank, trial: int | Array, *, full: bool):
        if full and not self.full_action_supported:
            raise PolarityFullActionUnavailable(
                "full action for (x,y,polarity) requires a 3-D periodic raster and weighted Poisson solver; "
                "use state.mode='position' for the explicit fallback"
            )
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        reconstruction = self._reconstruct(phi_truth, bank, trial)
        law_rows, tangent_rows, full_rows = [], [], []
        max_residual = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        max_poisson = jnp.asarray(0.0)
        for time_index in range(len(self.times)):
            state = particle_mfsi_state(
                phi=phi_ref[time_index],
                grad_phi=grad_ref[time_index],
                velocity=self.reference_velocity[time_index],
                base_weights=self.reference_weights[time_index],
                target=reconstruction.c[time_index],
                target_dot=reconstruction.c_dot[time_index],
                projector=self.projector,
                cfg=self.particle_cfg,
            )
            max_residual = jnp.maximum(max_residual, jnp.linalg.norm(state.projection.residual))
            min_ess = jnp.minimum(min_ess, state.projection.ess_fraction)
            projected_mass = histogram_mass(
                self.reference_nodes[time_index], state.projection.weights, self.law_grid
            )
            law_rows.append(periodic_grid_mmd2(
                projected_mass, self.truth_masses[time_index], self.law_kernel_fft
            ))
            tangent_rows.append(state.tangent_action)
            if full:
                raster = rasterize_periodic_particles(
                    self.reference_nodes[time_index], state.projection.weights, state.forcing,
                    self.grid, bandwidth=self.raster_bandwidth,
                )
                poisson = solve_periodic_weighted_poisson(raster.q, raster.h, self.grid, self.poisson_cfg)
                full_rows.append(poisson.action)
                max_poisson = jnp.maximum(max_poisson, poisson.relative_residual)
        validity = self.cfg.get("validity", {})
        valid = (
            max_residual <= float(validity.get("max_calibration_residual", 1.0e-3))
        ) & (min_ess >= float(validity.get("min_ess_fraction", 0.03)))
        if full:
            valid = valid & (
                max_poisson <= float(validity.get("max_poisson_relative_residual", 1.0e-5))
            )
        integrate = lambda rows: jnp.sum(self.time_weights * jnp.stack(rows))
        return (
            integrate(law_rows),
            integrate(tangent_rows),
            integrate(full_rows) if full else jnp.asarray(jnp.nan),
            max_residual,
            min_ess,
            max_poisson if full else jnp.asarray(jnp.nan),
            valid,
        )

    def trial_metrics(self, eta: Array, bank: ObservationTrialBank, trial: int, *, full: bool) -> TrialMetrics:
        values = self._trial_values(eta, bank, trial, full=full)
        return TrialMetrics(
            law_risk=float(values[0]),
            tangent_action=float(values[1]),
            full_action=float(values[2]),
            max_calibration_residual=float(values[3]),
            min_ess_fraction=float(values[4]),
            max_poisson_relative_residual=float(values[5]),
            valid=bool(values[6]),
        )

    def mean_metric(self, eta: Array, bank: ObservationTrialBank, name: str) -> Array:
        if name not in ("law_risk", "tangent_action", "full_action"):
            raise ValueError("unknown experiment metric")
        full = name == "full_action"
        metric_index = {"law_risk": 0, "tangent_action": 1, "full_action": 2}[name]
        rows = [
            self._trial_values(eta, bank, trial, full=full)
            for trial in range(bank.sample_indices.shape[0])
        ]
        penalty = float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3))
        values = [jnp.where(row[6], row[metric_index], row[metric_index] + penalty) for row in rows]
        return jnp.mean(jnp.stack(values))

    def optimize_designs(self, bank: ObservationTrialBank) -> DesignComparison:
        """Compute law optimum, then tangent/full minima in the risk-feasible set."""
        opt = self.cfg.get("optimization", {})
        optimizer = OptimizerConfig(
            steps=int(opt.get("steps", 60)),
            learning_rate=float(opt.get("learning_rate", 0.01)),
            constraint_penalty=float(opt.get("constraint_penalty", 1.0e4)),
            feasibility_tol=float(opt.get("feasibility_tol", 1.0e-6)),
        )
        starts = random_periodic_sensor_starts(
            jax.random.PRNGKey(int(self.cfg["seed"]) + 17),
            int(opt.get("start_count", 16)),
            n_sensors=self.family.n_sensors,
            box_size=self.family.box_size,
            min_separation=float(self.cfg["measurement"].get("min_sep", 0.0)),
            oversample=int(opt.get("start_oversample", 64)),
        )
        geometry = ((periodic_separation_violation(
            float(self.cfg["measurement"].get("min_sep", 0.0)),
            n_sensors=self.family.n_sensors,
            box_size=self.family.box_size,
        ), 0.0),)
        law_fn = lambda eta: self.mean_metric(eta, bank, "law_risk")
        law_candidates = optimize_multistart_candidates(
            law_fn, starts, optimizer, constraints=geometry, canonicalize=self.family.canonicalize
        )
        feasible_law = [candidate for candidate in law_candidates if candidate.feasible]
        law = min(feasible_law, key=lambda candidate: candidate.value)
        risk_star = float(law.value)
        risk_max = risk_star + float(self.cfg["law"].get("epsilon_r", 0.0))
        conditioned_constraints = geometry + ((law_fn, risk_max),)
        conditioned_starts = jnp.concatenate([law.eta[None, :], starts], axis=0)

        tangent_candidates = optimize_multistart_candidates(
            lambda eta: self.mean_metric(eta, bank, "tangent_action"),
            conditioned_starts,
            optimizer,
            constraints=conditioned_constraints,
            canonicalize=self.family.canonicalize,
        )
        tangent = min((row for row in tangent_candidates if row.feasible), key=lambda row: row.value)
        full = None
        full_candidates = []
        if self.full_action_supported:
            full_candidates = optimize_multistart_candidates(
                lambda eta: self.mean_metric(eta, bank, "full_action"),
                conditioned_starts,
                optimizer,
                constraints=conditioned_constraints,
                canonicalize=self.family.canonicalize,
                vectorize_starts=False,
            )
            full = min((row for row in full_candidates if row.feasible), key=lambda row: row.value)

        serialize = lambda rows: [
            {"eta": np.asarray(row.eta).tolist(), "value": row.value, "feasible": row.feasible,
             "violations": list(row.violations)} for row in rows
        ]
        return DesignComparison(
            law.eta,
            tangent.eta,
            None if full is None else full.eta,
            risk_star,
            risk_max,
            {"law": serialize(law_candidates), "tangent": serialize(tangent_candidates), "full": serialize(full_candidates)},
        )
