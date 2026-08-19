"""MFSI design layer for normalized active-nematic +1/2-defect laws.

The class mirrors the vortices experiment's geometry -> sparse observation ->
anchored moment reconstruction -> I-projection -> law/tangent/full evaluation
pipeline. Shared MFSI abstractions are imported directly. Only periodic
measurement geometry, periodic law risk, and the experiment-local 2-D/3-D
periodic Poisson discretizations are specialized here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.design import OptimizerConfig, optimize_multistart_candidates
from mfsi.exact_feasibility import robust_empirical_tilt_exact
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
        PeriodicGrid3D,
        PeriodicPoissonConfig,
        rasterize_periodic_particles,
        rasterize_periodic_particles3d,
        solve_periodic_weighted_poisson3d_batch_jax,
        solve_periodic_weighted_poisson,
    )
    from .poisson3d_tesseract import (
        NATIVE_SOLVER_REVISION,
        solve_periodic_weighted_poisson3d_batch_tesseract,
    )
    from .risk import (
        PeriodicHistogramGrid,
        histogram_mass,
        multiscale_periodic_kernel_fft,
        periodic_grid_mmd2,
        trapezoid_weights,
    )
except ImportError:  # pragma: no cover - direct experiment-script convention.
    from measurements import (
        PeriodicGaussianSensors,
        periodic_separation_violation,
        random_periodic_sensor_starts,
    )
    from periodic_numerics import (
        PeriodicGrid2D,
        PeriodicGrid3D,
        PeriodicPoissonConfig,
        rasterize_periodic_particles,
        rasterize_periodic_particles3d,
        solve_periodic_weighted_poisson3d_batch_jax,
        solve_periodic_weighted_poisson,
    )
    from poisson3d_tesseract import (
        NATIVE_SOLVER_REVISION,
        solve_periodic_weighted_poisson3d_batch_tesseract,
    )
    from risk import (
        PeriodicHistogramGrid,
        histogram_mass,
        multiscale_periodic_kernel_fft,
        periodic_grid_mmd2,
        trapezoid_weights,
    )

Array = jax.Array


class PolarityFullActionUnavailable(NotImplementedError):
    """Legacy exception retained for callers of pre-3D experiment versions."""


class ObservationTrialBank(NamedTuple):
    sample_indices: Array  # [trial,acquisition_time,finite_n]
    detector_z: Array      # [trial,acquisition_time,observable]


class Reconstruction(NamedTuple):
    c: Array
    c_dot: Array
    residual_sum_squares: Array
    roughness: Array


class TrialAssembly(NamedTuple):
    law_rows: Array
    tangent_rows: Array
    full_rows: Array
    raster_q: Array
    raster_h: Array
    max_calibration_residual: Array
    min_ess_fraction: Array
    max_poisson_relative_residual: Array


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
    certified: bool


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


def _empirical_coordinate_support_gap(
    features: np.ndarray,
    base_weights: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Cheap exact certificate for targets outside the empirical moment box."""
    margins = []
    for time_index in range(features.shape[0]):
        active = base_weights[time_index] > 0.0
        if not np.any(active):
            raise ValueError(
                f"reference weights have empty support at time index {time_index}"
            )
        lower = np.min(features[time_index, active], axis=0)
        upper = np.max(features[time_index, active], axis=0)
        margins.append(
            np.minimum(
                targets[time_index] - lower,
                upper - targets[time_index],
            )
        )
    return float(np.min(np.asarray(margins)))


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
        box_size = float(cfg["physics"]["box_size"])
        self.polarity_metric_radius = float(action.get("polarity_metric_radius", 1.0))
        if self.state_dim == 2:
            self.grid: PeriodicGrid2D | PeriodicGrid3D = PeriodicGrid2D(
                box_size, int(action.get("grid_n", 64))
            )
        else:
            grid_shape = tuple(
                int(value)
                for value in action.get("grid_shape_polarity", [48, 48, 24])
            )
            self.grid = PeriodicGrid3D(
                box_size,
                grid_shape,
                polarity_metric_radius=self.polarity_metric_radius,
            )
        self.poisson3d_backend = str(action.get("backend_3d", "tesseract_cpp"))
        if self.poisson3d_backend not in ("jax", "tesseract_cpp"):
            raise ValueError("full_action.backend_3d must be 'jax' or 'tesseract_cpp'")
        self.poisson_cfg = PeriodicPoissonConfig(
            operator_floor_rel=float(action.get("operator_floor_rel", 2.0e-5)),
            cg_tol=float(action.get("cg_tol", 1.0e-7)),
            cg_maxiter=int(action.get("cg_maxiter", 520)),
            gauge_strength=float(action.get("gauge_strength", 1.0)),
        )
        self.raster_bandwidth = float(action.get("raster_bandwidth", 0.0))
        self.time_guard_points = int(
            cfg.get("evaluation", {}).get("time_guard_points", 0)
        )
        if self.time_guard_points < 0:
            raise ValueError("evaluation.time_guard_points must be nonnegative")
        if self.time_guard_points == 0:
            self.time_weights = trapezoid_weights(self.times)
        else:
            guard = self.time_guard_points
            if 2 * guard >= len(self.times) - 1:
                raise ValueError(
                    "evaluation.time_guard_points leaves fewer than two action times"
                )
            interior = trapezoid_weights(self.times[guard:-guard])
            self.time_weights = jnp.pad(interior, (guard, guard))
        self.periods = jnp.asarray(
            [box_size, box_size] + ([2.0 * np.pi] if self.state_dim == 3 else []),
            dtype=jnp.float64,
        )
        self.mmd_bandwidths = jnp.asarray(cfg["law"].get("mmd_bandwidths", [0.5, 1.0, 2.0]))
        shape_key = "grid_shape_position" if self.state_dim == 2 else "grid_shape_polarity"
        default_shape = [64, 64] if self.state_dim == 2 else [48, 48, 24]
        law_shape = tuple(int(value) for value in cfg["law"].get(shape_key, default_shape))
        self.law_grid = PeriodicHistogramGrid(tuple(float(value) for value in self.periods), law_shape)
        self.law_kernel_fft = multiscale_periodic_kernel_fft(self.law_grid, self.mmd_bandwidths)
        hidden_weights = jnp.full(
            (self.truth_particles.shape[1],), 1.0 / self.truth_particles.shape[1], dtype=jnp.float64
        )
        self.truth_masses = jax.vmap(
            lambda samples: histogram_mass(samples, hidden_weights, self.law_grid)
        )(self.truth_particles)
        self._exact_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    @property
    def full_action_supported(self) -> bool:
        return True

    def full_action_provenance(self) -> dict[str, Any]:
        grid_shape = (
            [self.grid.n, self.grid.n]
            if self.state_dim == 2
            else list(self.grid.shape)
        )
        return {
            "state_dimension": self.state_dim,
            "backend": (
                "jax_2d" if self.state_dim == 2 else self.poisson3d_backend
            ),
            "native_solver_revision": (
                NATIVE_SOLVER_REVISION
                if self.state_dim == 3 and self.poisson3d_backend == "tesseract_cpp"
                else None
            ),
            "grid_shape": grid_shape,
            "polarity_metric_radius": (
                self.polarity_metric_radius if self.state_dim == 3 else None
            ),
            "raster_bandwidth": self.raster_bandwidth,
            "operator_floor_rel": self.poisson_cfg.operator_floor_rel,
            "cg_tol": self.poisson_cfg.cg_tol,
            "cg_maxiter": self.poisson_cfg.cg_maxiter,
            "gauge_strength": self.poisson_cfg.gauge_strength,
            "trial_time_batching": self.state_dim == 3,
            "time_guard_points": self.time_guard_points,
            "time_weights": np.asarray(self.time_weights).tolist(),
        }

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

    def _metric_geometry(self, eta: Array) -> tuple[Array, Array, Array, Array]:
        phi_truth, phi_ref, grad_ref = self._geometry(eta)
        velocity_ref = self.reference_velocity
        if self.state_dim == 3:
            # The particle tangent norm and the Poisson gradient must use the
            # same product metric. This coordinate transform leaves J Phi . u
            # invariant while scaling angular covectors/vectors reciprocally.
            grad_ref = grad_ref.at[..., 2].divide(self.polarity_metric_radius)
            velocity_ref = velocity_ref.at[..., 2].multiply(self.polarity_metric_radius)
        return phi_truth, phi_ref, grad_ref, velocity_ref

    def _assemble_trial(
        self,
        geometry: tuple[Array, Array, Array, Array],
        bank: ObservationTrialBank,
        trial: int | Array,
        *,
        full: bool,
    ) -> TrialAssembly:
        phi_truth, phi_ref, grad_ref, velocity_ref = geometry
        reconstruction = self._reconstruct(phi_truth, bank, trial)
        law_rows, tangent_rows, full_rows = [], [], []
        raster_q, raster_h = [], []
        max_residual = jnp.asarray(0.0)
        min_ess = jnp.asarray(jnp.inf)
        max_poisson = jnp.asarray(0.0)
        for time_index in range(len(self.times)):
            state = particle_mfsi_state(
                phi=phi_ref[time_index],
                grad_phi=grad_ref[time_index],
                velocity=velocity_ref[time_index],
                base_weights=self.reference_weights[time_index],
                target=reconstruction.c[time_index],
                target_dot=reconstruction.c_dot[time_index],
                projector=self.projector,
                cfg=self.particle_cfg,
            )
            residual_norm = jnp.sqrt(
                jnp.sum(state.projection.residual**2) + 1.0e-30
            )
            max_residual = jnp.maximum(max_residual, residual_norm)
            min_ess = jnp.minimum(min_ess, state.projection.ess_fraction)
            projected_mass = histogram_mass(
                self.reference_nodes[time_index], state.projection.weights, self.law_grid
            )
            law_rows.append(periodic_grid_mmd2(
                projected_mass, self.truth_masses[time_index], self.law_kernel_fft
            ))
            tangent_rows.append(state.tangent_action)
            if full:
                if self.state_dim == 2:
                    raster = rasterize_periodic_particles(
                        self.reference_nodes[time_index],
                        state.projection.weights,
                        state.forcing,
                        self.grid,
                        bandwidth=self.raster_bandwidth,
                    )
                    poisson = solve_periodic_weighted_poisson(
                        raster.q, raster.h, self.grid, self.poisson_cfg
                    )
                    full_rows.append(poisson.action)
                    max_poisson = jnp.maximum(
                        max_poisson, poisson.relative_residual
                    )
                else:
                    raster = rasterize_periodic_particles3d(
                        self.reference_nodes[time_index],
                        state.projection.weights,
                        state.forcing,
                        self.grid,
                        bandwidth=self.raster_bandwidth,
                    )
                    raster_q.append(raster.q)
                    raster_h.append(raster.h)
        empty = jnp.empty((0,), dtype=jnp.float64)
        return TrialAssembly(
            law_rows=jnp.stack(law_rows),
            tangent_rows=jnp.stack(tangent_rows),
            full_rows=jnp.stack(full_rows) if full_rows else empty,
            raster_q=jnp.stack(raster_q) if raster_q else empty,
            raster_h=jnp.stack(raster_h) if raster_h else empty,
            max_calibration_residual=max_residual,
            min_ess_fraction=min_ess,
            max_poisson_relative_residual=max_poisson,
        )

    def _solve_poisson3d_batch(self, q: Array, h: Array):
        solve_batch = (
            solve_periodic_weighted_poisson3d_batch_tesseract
            if self.poisson3d_backend == "tesseract_cpp"
            else solve_periodic_weighted_poisson3d_batch_jax
        )
        return solve_batch(q, h, self.grid, self.poisson_cfg)

    def _assembly_values(
        self,
        assembly: TrialAssembly,
        *,
        full: bool,
        poisson_actions: Array | None = None,
        poisson_residuals: Array | None = None,
    ):
        full_value = jnp.asarray(jnp.nan)
        max_poisson = jnp.asarray(jnp.nan)
        if full and self.state_dim == 2:
            full_value = jnp.sum(self.time_weights * assembly.full_rows)
            max_poisson = assembly.max_poisson_relative_residual
        elif full:
            if poisson_actions is None or poisson_residuals is None:
                raise ValueError("3-D full action requires solved Poisson rows")
            full_value = jnp.sum(self.time_weights * poisson_actions)
            max_poisson = jnp.max(poisson_residuals)
        validity = self.cfg.get("validity", {})
        valid = (
            assembly.max_calibration_residual
            <= float(validity.get("max_calibration_residual", 1.0e-3))
        ) & (
            assembly.min_ess_fraction
            >= float(validity.get("min_ess_fraction", 0.03))
        )
        if full:
            valid = valid & (
                max_poisson <= float(validity.get("max_poisson_relative_residual", 1.0e-5))
            )
        return (
            jnp.sum(self.time_weights * assembly.law_rows),
            jnp.sum(self.time_weights * assembly.tangent_rows),
            full_value,
            assembly.max_calibration_residual,
            assembly.min_ess_fraction,
            max_poisson,
            valid,
        )

    def _trial_values(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        trial: int | Array,
        *,
        full: bool,
    ):
        assembly = self._assemble_trial(
            self._metric_geometry(eta), bank, trial, full=full
        )
        if full and self.state_dim == 3:
            poisson = self._solve_poisson3d_batch(
                assembly.raster_q, assembly.raster_h
            )
            return self._assembly_values(
                assembly,
                full=True,
                poisson_actions=poisson.action,
                poisson_residuals=poisson.relative_residual,
            )
        return self._assembly_values(assembly, full=full)

    def _batch_trial_values(
        self, eta: Array, bank: ObservationTrialBank, *, full: bool
    ) -> list[tuple[Array, ...]]:
        """Evaluate all CRN trials, using one 3-D solve over trial and time."""
        geometry = self._metric_geometry(eta)
        trial_count = int(bank.sample_indices.shape[0])
        assemblies = [
            self._assemble_trial(geometry, bank, trial, full=full)
            for trial in range(trial_count)
        ]
        if full and self.state_dim == 3:
            q = jnp.concatenate([assembly.raster_q for assembly in assemblies])
            h = jnp.concatenate([assembly.raster_h for assembly in assemblies])
            poisson = self._solve_poisson3d_batch(q, h)
            action = poisson.action.reshape((trial_count, len(self.times)))
            residual = poisson.relative_residual.reshape(
                (trial_count, len(self.times))
            )
            return [
                self._assembly_values(
                    assembly,
                    full=True,
                    poisson_actions=action[trial],
                    poisson_residuals=residual[trial],
                )
                for trial, assembly in enumerate(assemblies)
            ]
        return [
            self._assembly_values(assembly, full=full)
            for assembly in assemblies
        ]

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
        rows = self._batch_trial_values(eta, bank, full=full)
        penalty = float(self.cfg.get("optimization", {}).get("invalid_penalty", 1.0e3))
        validity = self.cfg.get("validity", {})
        calibration_limit = float(
            validity.get("max_calibration_residual", 1.0e-3)
        )
        ess_limit = float(validity.get("min_ess_fraction", 0.03))
        poisson_limit = float(
            validity.get("max_poisson_relative_residual", 1.0e-5)
        )

        def penalized(row):
            calibration_violation = jax.nn.relu(
                row[3] / max(calibration_limit, 1.0e-14) - 1.0
            )
            ess_violation = jax.nn.relu(
                1.0 - row[4] / max(ess_limit, 1.0e-14)
            )
            poisson_violation = (
                jax.nn.relu(row[5] / max(poisson_limit, 1.0e-14) - 1.0)
                if full
                else jnp.asarray(0.0)
            )
            violation = (
                calibration_violation**2
                + ess_violation**2
                + poisson_violation**2
            )
            return row[metric_index] + penalty * violation

        values = [penalized(row) for row in rows]
        return jnp.mean(jnp.stack(values))

    def _batch_trial_array_values(
        self, eta: Array, bank: ObservationTrialBank, *, full: bool
    ) -> tuple[Array, ...]:
        rows = self._batch_trial_values(eta, bank, full=full)
        return tuple(jnp.stack([row[index] for row in rows]) for index in range(7))

    @staticmethod
    def _audit_from_arrays(
        arrays: tuple[Array, ...], *, metric_index: int, full: bool
    ) -> dict[str, Any]:
        return {
            "value": float(jnp.mean(arrays[metric_index])),
            "valid": bool(jnp.all(arrays[6])),
            "trials": int(arrays[0].shape[0]),
            "max_calibration_residual": float(jnp.max(arrays[3])),
            "min_ess_fraction": float(jnp.min(arrays[4])),
            "max_poisson_relative_residual": (
                float(jnp.max(arrays[5])) if full else None
            ),
        }

    def _exact_tilt(
        self,
        phi: np.ndarray,
        base_weights: np.ndarray,
        target: np.ndarray,
        lam0: np.ndarray,
    ):
        projection = self.cfg.get("projection", {})
        return robust_empirical_tilt_exact(
            phi,
            base_weights,
            target,
            lam0=lam0,
            newton_steps=int(projection.get("max_steps", 300)),
            newton_ridge=float(projection.get("newton_ridge", 1.0e-7)),
            step_cap=float(projection.get("step_cap", 20.0)),
            lambda_clip=float(projection.get("lambda_clip", 1000.0)),
            accept_tol=float(projection.get("solver_accept_tol", 2.0e-6)),
            lbfgs_maxiter=int(projection.get("lbfgs_maxiter", 800)),
            retry_multiplier=float(projection.get("retry_clip_multiplier", 2.0)),
            retries=int(projection.get("max_retries", 2)),
        )

    def _exact_key(self, eta: Array) -> tuple[float, ...]:
        canonical = np.asarray(self.family.canonicalize(eta), dtype=np.float64)
        return tuple(np.round(canonical, 12))

    def _exact_rows(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        *,
        full: bool,
    ) -> list[dict[str, Any]]:
        """Robust, non-differentiated final scores for a frozen trial bank."""
        cache_key = (self._exact_key(eta), id(bank), bool(full))
        if cache_key in self._exact_cache:
            return [dict(row) for row in self._exact_cache[cache_key]]

        phi_truth, phi_ref, grad_ref, velocity_ref = (
            np.asarray(value, dtype=np.float64)
            for value in self._metric_geometry(eta)
        )
        base_weights = np.asarray(self.reference_weights, dtype=np.float64)
        reference_nodes = np.asarray(self.reference_nodes, dtype=np.float64)
        time_weights = np.asarray(self.time_weights, dtype=np.float64)
        validity = self.cfg.get("validity", {})
        projection_cfg = self.cfg.get("projection", {})
        support_tol = float(projection_cfg.get("support_certificate_tol", 1.0e-10))
        accept_tol = float(projection_cfg.get("solver_accept_tol", 2.0e-6))
        calibration_limit = min(
            float(validity.get("max_calibration_residual", 1.0e-3)),
            accept_tol,
        )
        ess_limit = float(validity.get("min_ess_fraction", 0.03))
        poisson_limit = float(
            validity.get("max_poisson_relative_residual", 1.0e-5)
        )
        covariance_ridge = float(self.particle_cfg.covariance_ridge)
        tangent_ridge = float(self.particle_cfg.tangent_ridge)
        observable_count = int(phi_ref.shape[-1])
        eye = np.eye(observable_count, dtype=np.float64)

        rows: list[dict[str, Any]] = []
        q_batches: list[Array] = []
        h_batches: list[Array] = []
        trial_count = int(bank.sample_indices.shape[0])
        for trial in range(trial_count):
            reconstruction = self._reconstruct(
                jnp.asarray(phi_truth), bank, trial
            )
            targets = np.asarray(reconstruction.c, dtype=np.float64)
            target_dot = np.asarray(reconstruction.c_dot, dtype=np.float64)
            support_gap = _empirical_coordinate_support_gap(
                phi_ref, base_weights, targets
            )
            lam = np.zeros(observable_count, dtype=np.float64)
            law_values: list[float] = []
            tangent_values: list[float] = []
            full_values: list[float] = []
            trial_q: list[Array] = []
            trial_h: list[Array] = []
            max_residual = 0.0
            min_ess = np.inf
            projection_success = True

            for time_index in range(len(self.times)):
                projection = self._exact_tilt(
                    phi_ref[time_index],
                    base_weights[time_index],
                    targets[time_index],
                    lam,
                )
                lam = projection.lam
                projection_success &= bool(projection.success)
                max_residual = max(max_residual, projection.residual_norm)
                min_ess = min(min_ess, projection.ess_fraction)
                weights = projection.weights
                projected_mass = histogram_mass(
                    jnp.asarray(reference_nodes[time_index]),
                    jnp.asarray(weights),
                    self.law_grid,
                )
                law_values.append(
                    float(
                        periodic_grid_mmd2(
                            projected_mass,
                            self.truth_masses[time_index],
                            self.law_kernel_fft,
                        )
                    )
                )

                advective = np.einsum(
                    "nmd,nd->nm",
                    grad_ref[time_index],
                    velocity_ref[time_index],
                )
                mean_advective = weights @ advective
                tangent_residual = mean_advective - target_dot[time_index]
                tangent_gram = np.einsum(
                    "n,nmd,nkd->mk",
                    weights,
                    grad_ref[time_index],
                    grad_ref[time_index],
                )
                tangent_coefficient = np.linalg.solve(
                    tangent_gram + tangent_ridge * eye,
                    tangent_residual,
                )
                tangent_values.append(
                    float(tangent_residual @ tangent_coefficient)
                )

                if full:
                    g = advective @ projection.lam
                    mean_g = float(weights @ g)
                    centered = phi_ref[time_index] - projection.moments[None, :]
                    covariance_phi_g = np.sum(
                        weights[:, None]
                        * centered
                        * (g - mean_g)[:, None],
                        axis=0,
                    )
                    lambda_dot = np.linalg.solve(
                        projection.covariance + covariance_ridge * eye,
                        target_dot[time_index]
                        - mean_advective
                        - covariance_phi_g,
                    )
                    forcing = centered @ lambda_dot + g - mean_g
                    forcing -= float(weights @ forcing)
                    if self.state_dim == 2:
                        raster = rasterize_periodic_particles(
                            jnp.asarray(reference_nodes[time_index]),
                            jnp.asarray(weights),
                            jnp.asarray(forcing),
                            self.grid,
                            bandwidth=self.raster_bandwidth,
                        )
                        poisson = solve_periodic_weighted_poisson(
                            raster.q, raster.h, self.grid, self.poisson_cfg
                        )
                        full_values.append(float(poisson.action))
                        trial_q.append(poisson.relative_residual)
                    else:
                        raster = rasterize_periodic_particles3d(
                            jnp.asarray(reference_nodes[time_index]),
                            jnp.asarray(weights),
                            jnp.asarray(forcing),
                            self.grid,
                            bandwidth=self.raster_bandwidth,
                        )
                        trial_q.append(raster.q)
                        trial_h.append(raster.h)

            row = {
                "trial": trial,
                "law_risk": float(np.sum(time_weights * law_values)),
                "law_risk_by_time": list(law_values),
                "tangent_action": float(np.sum(time_weights * tangent_values)),
                "tangent_action_by_time": list(tangent_values),
                "full_action": (
                    float(np.sum(time_weights * full_values))
                    if full and self.state_dim == 2
                    else float("nan")
                ),
                "full_action_by_time": (
                    list(full_values) if full and self.state_dim == 2 else None
                ),
                "max_calibration_residual": float(max_residual),
                "min_ess_fraction": float(min_ess),
                "max_poisson_relative_residual": (
                    float(max(np.asarray(trial_q, dtype=np.float64)))
                    if full and self.state_dim == 2
                    else float("nan")
                ),
                "projection_success": projection_success,
                "min_empirical_hull_support_gap": support_gap,
                "invalid_reason": None,
                "poisson_error": None,
                "_q_count": len(trial_q) if full and self.state_dim == 3 else 0,
            }
            rows.append(row)
            if full and self.state_dim == 3:
                q_batches.extend(trial_q)
                h_batches.extend(trial_h)

        if full and self.state_dim == 3 and q_batches:
            try:
                poisson = self._solve_poisson3d_batch(
                    jnp.stack(q_batches), jnp.stack(h_batches)
                )
                actions = np.asarray(poisson.action, dtype=np.float64)
                residuals = np.asarray(poisson.relative_residual, dtype=np.float64)
                offset = 0
                for row in rows:
                    count = int(row.pop("_q_count"))
                    row_actions = actions[offset : offset + count]
                    row["full_action"] = float(
                        np.sum(time_weights * row_actions)
                    )
                    row["full_action_by_time"] = row_actions.tolist()
                    row["max_poisson_relative_residual"] = float(
                        np.max(residuals[offset : offset + count])
                    )
                    offset += count
            except RuntimeError as error:
                for row in rows:
                    row.pop("_q_count", None)
                    row["full_action"] = float("nan")
                    row["full_action_by_time"] = None
                    row["max_poisson_relative_residual"] = float("inf")
                    row["poisson_error"] = str(error)
        else:
            for row in rows:
                row.pop("_q_count", None)

        for row in rows:
            valid = bool(
                row["projection_success"]
                and row["min_empirical_hull_support_gap"] >= -support_tol
                and row["max_calibration_residual"] <= calibration_limit
                and row["min_ess_fraction"] >= ess_limit
            )
            if full:
                valid &= bool(
                    np.isfinite(row["full_action"])
                    and row["max_poisson_relative_residual"] <= poisson_limit
                )
            row["valid"] = valid
            if not valid:
                row["invalid_reason"] = (
                    "target_outside_empirical_moment_hull"
                    if row["min_empirical_hull_support_gap"] < -support_tol
                    else "calibration_ess_or_numerical_gate"
                )
            row["projection_solver"] = "robust_empirical_tilt_exact"

        self._exact_cache[cache_key] = [dict(row) for row in rows]
        return [dict(row) for row in rows]

    @staticmethod
    def _audit_from_exact_rows(
        rows: list[dict[str, Any]], *, name: str, full: bool
    ) -> dict[str, Any]:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        return {
            "value": float(np.mean(values)),
            "valid": bool(all(row["valid"] for row in rows)),
            "trials": len(rows),
            "max_calibration_residual": float(
                max(row["max_calibration_residual"] for row in rows)
            ),
            "min_ess_fraction": float(
                min(row["min_ess_fraction"] for row in rows)
            ),
            "max_poisson_relative_residual": (
                float(max(row["max_poisson_relative_residual"] for row in rows))
                if full
                else None
            ),
            "min_empirical_hull_support_gap": float(
                min(row["min_empirical_hull_support_gap"] for row in rows)
            ),
            "projection_solver": "robust_empirical_tilt_exact",
            "poisson_errors": sorted(
                {
                    str(row["poisson_error"])
                    for row in rows
                    if row.get("poisson_error")
                }
            ),
        }

    def exact_trial_metrics(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        trial: int,
        *,
        full: bool,
    ) -> TrialMetrics:
        row = self._exact_rows(eta, bank, full=full)[int(trial)]
        return TrialMetrics(
            law_risk=float(row["law_risk"]),
            tangent_action=float(row["tangent_action"]),
            full_action=float(row["full_action"]),
            max_calibration_residual=float(row["max_calibration_residual"]),
            min_ess_fraction=float(row["min_ess_fraction"]),
            max_poisson_relative_residual=float(
                row["max_poisson_relative_residual"]
            ),
            valid=bool(row["valid"]),
        )

    def exact_trial_rows(
        self,
        eta: Array,
        bank: ObservationTrialBank,
        *,
        full: bool,
    ) -> list[dict[str, Any]]:
        """Return detailed authoritative rows, including failure provenance."""
        return self._exact_rows(eta, bank, full=full)

    def audit_metric(self, eta: Array, bank: ObservationTrialBank, name: str) -> dict[str, Any]:
        """Return an independent robust score and all-trials certificate."""
        full = name == "full_action"
        return self._audit_from_exact_rows(
            self._exact_rows(eta, bank, full=full),
            name=name,
            full=full,
        )

    def optimize_designs(self, bank: ObservationTrialBank) -> DesignComparison:
        """Proxy search followed by authoritative frozen-bank candidate audits."""
        opt = self.cfg.get("optimization", {})

        def report(phase: str, **values: Any) -> None:
            fields = " ".join(f"{key}={value}" for key, value in values.items())
            print(
                f"active_nematic optimization_phase={phase} {fields}".rstrip(),
                flush=True,
            )

        def optimizer(stage: str) -> OptimizerConfig:
            return OptimizerConfig(
                steps=int(opt.get(f"{stage}_steps", opt.get("steps", 60))),
                learning_rate=float(
                    opt.get(f"{stage}_learning_rate", opt.get("learning_rate", 0.01))
                ),
                constraint_penalty=float(opt.get("constraint_penalty", 1.0e4)),
                feasibility_tol=float(opt.get("feasibility_tol", 1.0e-6)),
            )

        def prefix(count: int) -> ObservationTrialBank:
            count = min(max(int(count), 1), int(bank.sample_indices.shape[0]))
            return ObservationTrialBank(
                bank.sample_indices[:count], bank.detector_z[:count]
            )

        def limited(candidates, limit: int, mandatory: list[Array] = ()):
            ordered = sorted(candidates, key=lambda row: row.value)
            chosen = []
            for row in ordered:
                if all(
                    not np.allclose(np.asarray(row.eta), np.asarray(other.eta))
                    for other in chosen
                ):
                    chosen.append(row)
                if len(chosen) >= max(int(limit), 1):
                    break
            for eta in mandatory:
                if not candidates:
                    break
                match = min(
                    candidates,
                    key=lambda row: float(
                        jnp.linalg.norm(
                            self.family.canonicalize(row.eta)
                            - self.family.canonicalize(eta)
                        )
                    ),
                )
                if all(
                    not np.allclose(np.asarray(match.eta), np.asarray(row.eta))
                    for row in chosen
                ):
                    chosen.append(match)
            return chosen

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
        law_gradient_bank = prefix(int(opt.get("law_gradient_trials", 2)))
        tangent_gradient_bank = prefix(int(opt.get("tangent_gradient_trials", 2)))
        full_gradient_bank = prefix(int(opt.get("full_gradient_trials", 2)))
        law_fn = lambda eta: self.mean_metric(eta, law_gradient_bank, "law_risk")
        law_starts = starts[: int(opt.get("law_start_count", len(starts)))]
        report(
            "law_proxy",
            starts=len(law_starts),
            trials=int(law_gradient_bank.sample_indices.shape[0]),
        )
        law_candidates = optimize_multistart_candidates(
            law_fn,
            law_starts,
            optimizer("law"),
            constraints=geometry,
            canonicalize=self.family.canonicalize,
        )
        feasible_law = [candidate for candidate in law_candidates if candidate.feasible]
        law_audits = []
        law_audit_candidates = limited(
            feasible_law, int(opt.get("law_exact_audit_candidates", 8))
        )
        report("law_exact", candidates=len(law_audit_candidates))
        for index, candidate in enumerate(law_audit_candidates):
            law_audits.append(
                (
                    candidate,
                    self.audit_metric(candidate.eta, bank, "law_risk"),
                )
            )
            report("law_exact_candidate", completed=index + 1, total=len(law_audit_candidates))
        if not law_audits:
            raise RuntimeError("no geometrically feasible law candidate was generated")
        valid_law = [(candidate, audit) for candidate, audit in law_audits if audit["valid"]]
        allow_invalid = bool(self.cfg.get("validity", {}).get("allow_invalid_selection", False))
        if not valid_law and not allow_invalid:
            raise RuntimeError("no calibration/ESS-valid law design survived selection audit")
        law, law_audit = min(
            valid_law or law_audits,
            key=lambda item: (
                item[1]["value"]
                if np.isfinite(item[1]["value"])
                else float("inf")
            ),
        )
        risk_star = float(law_audit["value"])
        risk_max = risk_star + float(self.cfg["law"].get("epsilon_r", 0.0))
        proxy_risk_max = float(law_fn(law.eta)) + float(
            self.cfg["law"].get("epsilon_r", 0.0)
        )
        conditioned_constraints = geometry + ((law_fn, proxy_risk_max),)
        tangent_starts = jnp.concatenate(
            [
                law.eta[None, :],
                starts[: int(opt.get("tangent_start_count", len(starts)))],
            ],
            axis=0,
        )

        report(
            "tangent_proxy",
            starts=len(tangent_starts),
            trials=int(tangent_gradient_bank.sample_indices.shape[0]),
        )
        tangent_candidates = optimize_multistart_candidates(
            lambda eta: self.mean_metric(
                eta, tangent_gradient_bank, "tangent_action"
            ),
            tangent_starts,
            optimizer("tangent"),
            constraints=conditioned_constraints,
            canonicalize=self.family.canonicalize,
        )
        feasible_tangent = [row for row in tangent_candidates if row.feasible]
        tangent_audits = []
        tangent_audit_candidates = limited(
            feasible_tangent,
            int(opt.get("tangent_exact_audit_candidates", 8)),
            mandatory=[law.eta],
        )
        report("tangent_exact", candidates=len(tangent_audit_candidates))
        for index, candidate in enumerate(tangent_audit_candidates):
            rows = self._exact_rows(candidate.eta, bank, full=False)
            tangent_audits.append(
                (
                    candidate,
                    self._audit_from_exact_rows(
                        rows, name="tangent_action", full=False
                    ),
                    self._audit_from_exact_rows(
                        rows, name="law_risk", full=False
                    ),
                )
            )
            report(
                "tangent_exact_candidate",
                completed=index + 1,
                total=len(tangent_audit_candidates),
            )
        if not tangent_audits:
            raise RuntimeError("no geometrically feasible tangent candidate was generated")
        valid_tangent = [
            (row, audit, law_screen)
            for row, audit, law_screen in tangent_audits
            if audit["valid"]
            and law_screen["valid"]
            and law_screen["value"] <= risk_max
        ]
        if not valid_tangent and not allow_invalid:
            raise RuntimeError("no calibration/ESS-valid tangent design survived selection audit")
        tangent, tangent_audit, tangent_law_audit = min(
            valid_tangent or tangent_audits,
            key=lambda item: (
                item[1]["value"]
                if np.isfinite(item[1]["value"])
                else float("inf")
            ),
        )
        full = None
        full_audit = {"valid": True}
        full_law_audit = {"valid": True, "value": float("nan")}
        full_candidates = []
        full_audits = []
        if self.full_action_supported:
            proxy_cfg = copy.deepcopy(self.cfg)
            if self.state_dim == 3:
                proxy_cfg["full_action"]["grid_shape_polarity"] = list(
                    opt.get("full_gradient_grid_shape", [24, 24, 12])
                )
                proxy_cfg["full_action"]["cg_tol"] = float(
                    opt.get("full_gradient_cg_tol", 1.0e-6)
                )
                proxy_cfg["full_action"]["cg_maxiter"] = int(
                    opt.get("full_gradient_cg_maxiter", 360)
                )
            proxy_experiment = ActiveNematicExperiment(
                proxy_cfg,
                times=self.times,
                truth_particles=self.truth_particles,
                reference_nodes=self.reference_nodes,
                reference_velocity=self.reference_velocity,
                reference_weights=self.reference_weights,
            )
            full_starts = jnp.concatenate(
                [
                    law.eta[None, :],
                    tangent.eta[None, :],
                    starts[: int(opt.get("full_start_count", 3))],
                ],
                axis=0,
            )
            report(
                "full_proxy",
                starts=len(full_starts),
                trials=int(full_gradient_bank.sample_indices.shape[0]),
                grid=(
                    "x".join(str(value) for value in proxy_experiment.grid.shape)
                    if self.state_dim == 3
                    else f"{proxy_experiment.grid.n}x{proxy_experiment.grid.n}"
                ),
            )
            full_candidates = optimize_multistart_candidates(
                lambda eta: proxy_experiment.mean_metric(
                    eta, full_gradient_bank, "full_action"
                ),
                full_starts,
                optimizer("full"),
                constraints=conditioned_constraints,
                canonicalize=self.family.canonicalize,
                vectorize_starts=False,
            )
            feasible_full = [row for row in full_candidates if row.feasible]
            full_audit_candidates = limited(
                feasible_full,
                int(opt.get("full_exact_audit_candidates", 8)),
                mandatory=[law.eta, tangent.eta],
            )
            report("full_exact", candidates=len(full_audit_candidates))
            for index, candidate in enumerate(full_audit_candidates):
                rows = self._exact_rows(candidate.eta, bank, full=True)
                full_audits.append(
                    (
                        candidate,
                        self._audit_from_exact_rows(
                            rows, name="full_action", full=True
                        ),
                        self._audit_from_exact_rows(
                            rows, name="law_risk", full=True
                        ),
                    )
                )
                report(
                    "full_exact_candidate",
                    completed=index + 1,
                    total=len(full_audit_candidates),
                )
            if not full_audits:
                raise RuntimeError("no geometrically feasible full-action candidate was generated")
            valid_full = [
                (row, audit, law_screen)
                for row, audit, law_screen in full_audits
                if audit["valid"]
                and law_screen["valid"]
                and law_screen["value"] <= risk_max
            ]
            if not valid_full and not allow_invalid:
                raise RuntimeError("no calibration/ESS/Poisson-valid full design survived selection audit")
            full, full_audit, full_law_audit = min(
                valid_full or full_audits,
                key=lambda item: (
                    item[1]["value"]
                    if np.isfinite(item[1]["value"])
                    else float("inf")
                ),
            )

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
            {
                "law": serialize(law_candidates),
                "tangent": serialize(tangent_candidates),
                "full": serialize(full_candidates),
                "law_exact": [
                    {"eta": np.asarray(row.eta).tolist(), **audit}
                    for row, audit in law_audits
                ],
                "tangent_exact": [
                    {
                        "eta": np.asarray(row.eta).tolist(),
                        "action": audit,
                        "law": law_screen,
                    }
                    for row, audit, law_screen in tangent_audits
                ],
                "full_exact": [
                    {
                        "eta": np.asarray(row.eta).tolist(),
                        "action": audit,
                        "law": law_screen,
                    }
                    for row, audit, law_screen in full_audits
                ],
                "full_proxy": [
                    {
                        "trials": int(full_gradient_bank.sample_indices.shape[0]),
                        "grid_shape": (
                            list(proxy_experiment.grid.shape)
                            if self.state_dim == 3
                            else [proxy_experiment.grid.n, proxy_experiment.grid.n]
                        ),
                        "cg_tol": proxy_experiment.poisson_cfg.cg_tol,
                        "cg_maxiter": proxy_experiment.poisson_cfg.cg_maxiter,
                    }
                ],
            },
            bool(
                law_audit["valid"]
                and tangent_audit["valid"]
                and tangent_law_audit["valid"]
                and tangent_law_audit["value"] <= risk_max
                and full_audit["valid"]
                and full_law_audit["valid"]
                and full_law_audit["value"] <= risk_max
            ),
        )
