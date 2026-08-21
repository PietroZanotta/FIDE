"""Two-species unbalanced active-nematic MFSI experiment.

This module composes the existing normalized I-projection/particle machinery
twice, once for each charge, and adds finite mass explicitly.  It does not alter
the balanced active-nematic experiment or shared MFSI defaults.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from mfsi.design import OptimizeResult, OptimizerConfig, optimize_multistart_candidates
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.particles import ParticleMFSIConfig, particle_mfsi_state
from mfsi.projection import EmpiricalIProjector, IProjectionConfig

try:
    from .measurements import periodic_separation_violation, random_periodic_sensor_starts
    from .periodic_numerics import PeriodicGrid3D, rasterize_periodic_particles3d
    from .risk import (
        PeriodicHistogramGrid,
        histogram_mass,
        multiscale_periodic_kernel_fft,
        trapezoid_weights,
    )
    from .unbalanced_correction import (
        UnbalancedCorrectionConfig,
        solve_unbalanced_screened_poisson_batch_jax,
        unbalanced_residual,
    )
    from .screened_poisson3d_tesseract import (
        solve_unbalanced_screened_poisson3d_batch_tesseract,
    )
    from .unbalanced_measurements import ChargeResolvedSensors
    from .unbalanced_risk import periodic_grid_finite_mmd2
    from .unbalanced_state import Species
    from .unbalanced_tangent import (
        append_global_mass_observable,
        unbalanced_tangent_action,
    )
except ImportError:  # pragma: no cover
    from measurements import periodic_separation_violation, random_periodic_sensor_starts
    from periodic_numerics import PeriodicGrid3D, rasterize_periodic_particles3d
    from risk import (
        PeriodicHistogramGrid,
        histogram_mass,
        multiscale_periodic_kernel_fft,
        trapezoid_weights,
    )
    from unbalanced_correction import (
        UnbalancedCorrectionConfig,
        solve_unbalanced_screened_poisson_batch_jax,
        unbalanced_residual,
    )
    from screened_poisson3d_tesseract import (
        solve_unbalanced_screened_poisson3d_batch_tesseract,
    )
    from unbalanced_measurements import ChargeResolvedSensors
    from unbalanced_risk import periodic_grid_finite_mmd2
    from unbalanced_state import Species
    from unbalanced_tangent import append_global_mass_observable, unbalanced_tangent_action


Array = jax.Array


def _quasi_newton_candidates(
    primary,
    starts: Array,
    *,
    constraints,
    canonicalize,
    penalty: float,
    feasibility_tol: float,
    maxiter: int,
    gtol: float,
    ftol: float,
    progress_callback=None,
    diagnostic_callback=None,
) -> list[OptimizeResult]:
    """Retain seeds and refine them with an exact-objective L-BFGS solve.

    This local helper is reserved for the low-dimensional, full-bank law
    reconciliation stage.  Seed retention makes the exact audited incumbent
    monotone even when a numerical refinement step is unhelpful.
    """
    starts = jnp.asarray(starts, dtype=jnp.float64)

    def penalized(eta):
        value = primary(eta)
        for fn, upper in constraints:
            violation = jax.nn.relu(fn(eta) - upper)
            value = value + float(penalty) * violation * violation
        return value

    value_and_grad = jax.jit(jax.value_and_grad(penalized))
    primary_eval = jax.jit(primary)
    out: list[OptimizeResult] = []

    def add_candidate(eta) -> None:
        eta = canonicalize(jnp.asarray(eta, dtype=jnp.float64))
        violations = tuple(float(fn(eta) - upper) for fn, upper in constraints)
        out.append(
            OptimizeResult(
                eta=eta,
                value=float(primary_eval(eta)),
                feasible=all(value <= feasibility_tol for value in violations),
                violations=violations,
            )
        )

    # Every seed is a law-stage candidate in its own right; an optimizer may
    # improve it, but can never erase its exact-risk result.
    for eta in starts:
        add_candidate(eta)

    total = int(starts.shape[0])
    for index, eta0 in enumerate(np.asarray(starts, dtype=np.float64)):
        def scipy_value_and_grad(eta_np):
            value, gradient = value_and_grad(jnp.asarray(eta_np, dtype=jnp.float64))
            return float(value), np.asarray(gradient, dtype=np.float64)

        result = minimize(
            scipy_value_and_grad,
            eta0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(maxiter),
                "gtol": float(gtol),
                "ftol": float(ftol),
                "maxls": 30,
            },
        )
        raw_eta = np.asarray(result.x, dtype=np.float64)
        raw_canonical = canonicalize(jnp.asarray(raw_eta, dtype=jnp.float64))
        raw_violations = tuple(
            float(fn(raw_canonical) - upper) for fn, upper in constraints
        )
        candidate_eta = raw_eta
        projected = False
        if any(value > feasibility_tol for value in raw_violations):
            seed_canonical = canonicalize(jnp.asarray(eta0, dtype=jnp.float64))
            seed_violations = tuple(
                float(fn(seed_canonical) - upper) for fn, upper in constraints
            )
            if all(value <= 0.0 for value in seed_violations):
                # Recover the largest certified step on the optimizer's path.
                # The action solve found the direction; this inexpensive exact
                # constraint bisection removes quadratic-penalty boundary error.
                lower, upper_fraction = 0.0, 1.0
                for _ in range(28):
                    fraction = 0.5 * (lower + upper_fraction)
                    trial_eta = eta0 + fraction * (raw_eta - eta0)
                    trial_canonical = canonicalize(
                        jnp.asarray(trial_eta, dtype=jnp.float64)
                    )
                    trial_violations = tuple(
                        float(fn(trial_canonical) - upper)
                        for fn, upper in constraints
                    )
                    if all(value <= 0.0 for value in trial_violations):
                        lower = fraction
                    else:
                        upper_fraction = fraction
                candidate_eta = eta0 + lower * (raw_eta - eta0)
                projected = True
            else:
                # The retained seed remains authoritative if it was accepted
                # only through feasibility tolerance rather than strict slack.
                candidate_eta = eta0
                projected = True
        add_candidate(candidate_eta)
        if diagnostic_callback is not None:
            diagnostic_callback(
                index + 1,
                result,
                out[-1],
                eta0,
                candidate_eta,
                raw_violations,
                projected,
            )
        if progress_callback is not None:
            progress_callback(index + 1, total)

    return out


class UnbalancedObservationBank(NamedTuple):
    plus_sample_indices: Array
    minus_sample_indices: Array
    plus_detector_z: Array
    minus_detector_z: Array


@dataclass(frozen=True)
class SpeciesExperimentData:
    truth_particles: Array
    truth_mass: Array
    reference_nodes: Array
    reference_velocity: Array
    reference_weights: Array
    reference_mass: Array
    reference_source_rate: Array
    target_mass: Array
    target_mass_dot: Array
    target_relative_mass_rate: Array


@dataclass(frozen=True)
class UnbalancedTrialMetrics:
    law_risk_total: float
    law_risk_plus: float
    law_risk_minus: float
    full_unbalanced_action_total: float
    full_unbalanced_action_plus: float
    full_unbalanced_action_minus: float
    move_action_plus: float
    reaction_action_plus: float
    move_action_minus: float
    reaction_action_minus: float
    tangent_unbalanced_total: float
    tangent_transport_total: float
    tangent_reaction_total: float
    max_calibration_residual: float
    min_ess_fraction: float
    max_pde_relative_residual: float
    valid: bool


@dataclass(frozen=True)
class UnbalancedDesignComparison:
    law_eta: Array
    tangent_eta: Array
    full_eta: Array
    risk_star: float
    risk_max: float
    candidates: dict[str, list[dict[str, Any]]]
    certified: bool


def nested_acquisition_indices(time_n: int, count: int) -> np.ndarray:
    if count < 2 or count > time_n:
        raise ValueError("acquisition count must satisfy 2 <= count <= time_n")
    index = np.unique(np.rint(np.linspace(0, time_n - 1, count)).astype(np.int32))
    if len(index) != count:
        interior = np.arange(1, time_n - 1, dtype=np.int32)
        middle = interior[
            np.rint(np.linspace(0, len(interior) - 1, count - 2)).astype(int)
        ]
        index = np.concatenate(([0], middle, [time_n - 1])).astype(np.int32)
    return index


def make_unbalanced_observation_bank(
    *,
    seed: int,
    namespace: int,
    trials: int,
    acquisition_count: int,
    finite_n: int,
    plus_truth_particle_count: int,
    minus_truth_particle_count: int,
    plus_observables: int,
    minus_observables: int,
) -> UnbalancedObservationBank:
    """Freeze paired random numbers independently of physical/reference seeds."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(namespace)]))
    shape = (int(trials), int(acquisition_count), int(finite_n))
    plus_indices = rng.integers(0, int(plus_truth_particle_count), size=shape, dtype=np.int32)
    minus_indices = rng.integers(0, int(minus_truth_particle_count), size=shape, dtype=np.int32)
    plus_z = rng.standard_normal((int(trials), int(acquisition_count), int(plus_observables)))
    minus_z = rng.standard_normal((int(trials), int(acquisition_count), int(minus_observables)))
    return UnbalancedObservationBank(
        jnp.asarray(plus_indices),
        jnp.asarray(minus_indices),
        jnp.asarray(plus_z, dtype=jnp.float64),
        jnp.asarray(minus_z, dtype=jnp.float64),
    )


class UnbalancedActiveNematicExperiment:
    """Information-first design over two finite positive defect measures."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        times: Array,
        plus: SpeciesExperimentData,
        minus: SpeciesExperimentData,
    ):
        self.cfg = cfg
        self.times = jnp.asarray(times, dtype=jnp.float64)
        self.data = {"plus": self._validated_data("plus", plus), "minus": self._validated_data("minus", minus)}
        measurement = cfg["measurement"]
        self.sensors = ChargeResolvedSensors(
            box_size=float(cfg["physics"]["box_size"]),
            width=float(measurement["sensor_width"]),
            n_sensors=int(measurement["n_sensors"]),
            use_plus_orientation=bool(measurement.get("use_plus_orientation", True)),
            use_minus_triatic_orientation=bool(
                measurement.get("use_minus_triatic_orientation", True)
            ),
            mass_is_global_observation=bool(
                measurement.get("mass_is_global_observation", True)
            ),
        )
        self.family = self.sensors.family("plus")
        self.acq_idx = jnp.asarray(
            nested_acquisition_indices(len(self.times), int(measurement["acquisition_k"])),
            dtype=jnp.int32,
        )
        spline = cfg.get("moment_reconstruction", {})
        spline_cfg = AnchoredCubicSplineConfig(
            internal_knots=int(spline.get("internal_knots", 3)),
            smoothing=float(spline.get("smoothing", 1.0e-4)),
            ridge_rel=float(spline.get("ridge_rel", 1.0e-10)),
            roughness_quadrature_order=int(spline.get("roughness_quadrature_order", 8)),
        )
        self.reconstructor = AnchoredCubicSplineReconstructor(
            self.times[self.acq_idx], self.times, spline_cfg
        )
        projection = cfg.get("projection", {})
        self.projector = EmpiricalIProjector(
            IProjectionConfig(
                max_steps=int(projection.get("max_steps", 300)),
                residual_tol=float(projection.get("residual_tol", 1.0e-8)),
                newton_ridge=float(projection.get("newton_ridge", 1.0e-7)),
                step_cap=float(projection.get("step_cap", 20.0)),
                lambda_clip=float(projection.get("lambda_clip", 1000.0)),
                line_search_steps=int(projection.get("line_search_steps", 8)),
            )
        )
        particle = cfg.get("particle_mfsi", {})
        self.particle_cfg = ParticleMFSIConfig(
            covariance_ridge=float(particle.get("covariance_ridge", 1.0e-7)),
            tangent_ridge=float(particle.get("tangent_ridge", 1.0e-7)),
        )
        unbalanced = cfg["unbalanced"]
        self.reaction_kappa = float(unbalanced["reaction_kappa"])
        self.species_weights = {
            "plus": float(unbalanced.get("species_weight_plus", 1.0)),
            "minus": float(unbalanced.get("species_weight_minus", 1.0)),
        }
        self.risk_weights = {
            "plus": float(unbalanced.get("risk_weight_plus", 1.0)),
            "minus": float(unbalanced.get("risk_weight_minus", 1.0)),
        }
        action = cfg["full_action"]
        self.full_action_backend = str(action.get("backend", "tesseract_cpp"))
        if self.full_action_backend not in {"tesseract_cpp", "jax_screened"}:
            raise ValueError(
                "full_action.backend must be 'tesseract_cpp' or 'jax_screened'"
            )
        self.polarity_metric_radius = float(action.get("polarity_metric_radius", 1.0))
        self.grid = PeriodicGrid3D(
            float(cfg["physics"]["box_size"]),
            tuple(int(x) for x in action.get("grid_shape", [48, 48, 24])),
            polarity_metric_radius=self.polarity_metric_radius,
        )
        self.raster_bandwidth = float(action.get("raster_bandwidth", 1.2))
        self.correction_cfg = UnbalancedCorrectionConfig(
            reaction_kappa=self.reaction_kappa,
            operator_floor_rel=float(action.get("operator_floor_rel", 0.0)),
            cg_tol=float(action.get("cg_tol", 1.0e-7)),
            cg_maxiter=int(action.get("cg_maxiter", 800)),
        )
        self.time_weights = trapezoid_weights(self.times)
        self.periods = jnp.asarray(
            [float(cfg["physics"]["box_size"])] * 2 + [2.0 * np.pi],
            dtype=jnp.float64,
        )
        law = cfg["law"]
        self.law_grid = PeriodicHistogramGrid(
            tuple(float(x) for x in self.periods),
            tuple(int(x) for x in law.get("grid_shape", [48, 48, 24])),
        )
        self.law_kernel_fft = multiscale_periodic_kernel_fft(
            self.law_grid, jnp.asarray(law.get("mmd_bandwidths", [0.5, 1.0, 2.0, 4.0]))
        )
        self.truth_finite_histogram = {}
        for species in ("plus", "minus"):
            data = self.data[species]
            normalized = jax.vmap(
                lambda samples: histogram_mass(
                    samples,
                    jnp.full((samples.shape[0],), 1.0 / samples.shape[0]),
                    self.law_grid,
                )
            )(data.truth_particles)
            self.truth_finite_histogram[species] = normalized * data.truth_mass[:, None, None, None]

    def _validated_data(self, species: Species, data: SpeciesExperimentData) -> SpeciesExperimentData:
        values = SpeciesExperimentData(
            **{name: jnp.asarray(getattr(data, name), dtype=jnp.float64) for name in data.__dataclass_fields__}
        )
        if values.truth_particles.ndim != 3 or values.truth_particles.shape[-1] != 3:
            raise ValueError(f"{species} truth particles must have shape [time,n,3]")
        if values.reference_nodes.shape != values.reference_velocity.shape:
            raise ValueError(f"{species} reference nodes/velocity shapes differ")
        if values.reference_nodes.ndim != 3 or values.reference_nodes.shape[-1] != 3:
            raise ValueError(f"{species} reference bank must have shape [time,n,3]")
        if values.reference_weights.shape != values.reference_nodes.shape[:2]:
            raise ValueError(f"{species} reference weights have wrong shape")
        for name in (
            "truth_mass", "reference_mass", "reference_source_rate", "target_mass",
            "target_mass_dot", "target_relative_mass_rate",
        ):
            if getattr(values, name).shape != (len(self.times),):
                raise ValueError(f"{species} {name} must have one value per time")
        return values

    def _geometry(self, species: Species, eta: Array):
        data = self.data[species]
        family = self.sensors.family(species)
        phi_truth = family.features(data.truth_particles, eta)
        phi_ref = family.features(data.reference_nodes, eta)
        grad_ref = family.feature_gradients(data.reference_nodes, eta)
        velocity = data.reference_velocity
        grad_ref = grad_ref.at[..., 2].divide(self.polarity_metric_radius)
        velocity = velocity.at[..., 2].multiply(self.polarity_metric_radius)
        return phi_truth, phi_ref, grad_ref, velocity

    def _reconstruct(self, species: Species, phi_truth: Array, bank: UnbalancedObservationBank, trial: int):
        indices = getattr(bank, f"{species}_sample_indices")[trial]
        detector_z = getattr(bank, f"{species}_detector_z")[trial]
        phi_acq = phi_truth[self.acq_idx]
        sampled = jax.vmap(lambda features, index: jnp.mean(features[index], axis=0))(
            phi_acq, indices
        )
        exact = jnp.mean(phi_acq, axis=1)
        observed = sampled + float(self.cfg["measurement"].get("obs_noise_std", 0.0)) * detector_z
        endpoint = (self.acq_idx == 0) | (self.acq_idx == len(self.times) - 1)
        observed = jnp.where(endpoint[:, None], exact, observed)
        fit = self.reconstructor.reconstruct(observed, exact[0], exact[-1])
        return fit.c, fit.c_dot

    def _species_trial(
        self,
        species: Species,
        eta: Array,
        bank: UnbalancedObservationBank,
        trial: int,
        *,
        compute_tangent: bool,
        compute_full: bool,
    ):
        data = self.data[species]
        phi_truth, phi_ref, grad_ref, velocity = self._geometry(species, eta)
        target, target_dot = self._reconstruct(species, phi_truth, bank, trial)
        law_rows = []
        tangent_total_rows = []
        tangent_move_rows = []
        tangent_react_rows = []
        full_rows = []
        move_rows = []
        react_rows = []
        reaction_fraction_rows = []
        residual_rows = []
        raster_q_rows = []
        raster_h_rows = []
        calibration_rows = []
        ess_rows = []
        for time_index in range(len(self.times)):
            state = particle_mfsi_state(
                phi=phi_ref[time_index],
                grad_phi=grad_ref[time_index],
                velocity=velocity[time_index],
                base_weights=data.reference_weights[time_index],
                target=target[time_index],
                target_dot=target_dot[time_index],
                projector=self.projector,
                cfg=self.particle_cfg,
            )
            calibration_rows.append(jnp.linalg.norm(state.projection.residual))
            ess_rows.append(state.projection.ess_fraction)
            projected_shape = histogram_mass(
                data.reference_nodes[time_index], state.projection.weights, self.law_grid
            )
            risk = periodic_grid_finite_mmd2(
                data.target_mass[time_index] * projected_shape,
                self.truth_finite_histogram[species][time_index],
                self.law_kernel_fft,
            )
            law_rows.append(risk)

            if compute_tangent:
                tangent_phi, tangent_grad = phi_ref[time_index], grad_ref[time_index]
                raw_target = data.target_mass[time_index] * target[time_index]
                raw_target_dot = (
                    data.target_mass_dot[time_index] * target[time_index]
                    + data.target_mass[time_index] * target_dot[time_index]
                )
                if self.sensors.mass_is_global_observation:
                    tangent_phi, tangent_grad = append_global_mass_observable(
                        tangent_phi, tangent_grad
                    )
                    raw_target = jnp.concatenate([raw_target, data.target_mass[time_index, None]])
                    raw_target_dot = jnp.concatenate(
                        [raw_target_dot, data.target_mass_dot[time_index, None]]
                    )
                tangent = unbalanced_tangent_action(
                    phi=tangent_phi,
                    grad_phi=tangent_grad,
                    velocity=velocity[time_index],
                    normalized_weights=state.projection.weights,
                    mass=data.target_mass[time_index],
                    target_raw_moments=raw_target,
                    target_raw_moment_dot=raw_target_dot,
                    reference_source_rate=data.reference_source_rate[time_index],
                    reaction_kappa=self.reaction_kappa,
                    ridge=float(self.particle_cfg.tangent_ridge),
                )
                tangent_total_rows.append(tangent.total_action)
                tangent_move_rows.append(tangent.transport_action)
                tangent_react_rows.append(tangent.reaction_action)

            if compute_full:
                raster = rasterize_periodic_particles3d(
                    data.reference_nodes[time_index],
                    state.projection.weights,
                    state.forcing,
                    self.grid,
                    bandwidth=self.raster_bandwidth,
                )
                h_ub = unbalanced_residual(
                    raster.h,
                    data.target_relative_mass_rate[time_index],
                    data.reference_source_rate[time_index],
                )
                raster_q_rows.append(raster.q)
                raster_h_rows.append(h_ub)

        if compute_full:
            q_batch = jnp.stack(raster_q_rows)
            h_batch = jnp.stack(raster_h_rows)
            if self.full_action_backend == "tesseract_cpp":
                correction = solve_unbalanced_screened_poisson3d_batch_tesseract(
                    q_batch,
                    h_batch,
                    mass=data.target_mass,
                    grid=self.grid,
                    config=self.correction_cfg,
                )
            else:
                correction = solve_unbalanced_screened_poisson_batch_jax(
                    q_batch,
                    h_batch,
                    mass=data.target_mass,
                    grid=self.grid,
                    config=self.correction_cfg,
                )
            full_rows = list(correction.total_action)
            move_rows = list(correction.move_action)
            react_rows = list(correction.reaction_action)
            reaction_fraction_rows = list(correction.reaction_fraction)
            residual_rows = list(correction.relative_residual)

        integrate = lambda rows: jnp.sum(self.time_weights * jnp.stack(rows))
        finite_rows = jnp.stack([row.finite_measure_risk for row in law_rows])
        shape_rows = jnp.stack([row.shape_mmd for row in law_rows])
        mass_error_rows = jnp.stack([row.mass_error for row in law_rows])
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        full = integrate(full_rows) if compute_full else zero
        reaction = integrate(react_rows) if compute_full else zero
        return {
            "law_risk": jnp.sum(self.time_weights * finite_rows),
            "shape_mmd": jnp.sum(self.time_weights * shape_rows),
            "mass_error": jnp.sum(self.time_weights * mass_error_rows),
            "tangent_action": integrate(tangent_total_rows) if compute_tangent else zero,
            "tangent_transport": integrate(tangent_move_rows) if compute_tangent else zero,
            "tangent_reaction": integrate(tangent_react_rows) if compute_tangent else zero,
            "full_action": full,
            "move_action": integrate(move_rows) if compute_full else zero,
            "reaction_action": reaction,
            "reaction_fraction": jnp.where(full > 0.0, reaction / full, 0.0),
            "max_calibration_residual": jnp.max(jnp.stack(calibration_rows)),
            "min_ess_fraction": jnp.min(jnp.stack(ess_rows)),
            "max_pde_relative_residual": jnp.max(jnp.stack(residual_rows)) if compute_full else zero,
        }

    def _species_metric_batch(
        self,
        species: Species,
        eta: Array,
        bank: UnbalancedObservationBank,
        *,
        compute_tangent: bool,
        compute_full: bool,
    ):
        """Evaluate every frozen trial in one compact differentiable graph.

        This is the optimizer/audit path.  ``_species_trial`` remains the
        authoritative scalar implementation used by final serialization.
        Keeping both paths makes the speedup independently checkable rather
        than changing the definition of a reported trial.
        """
        data = self.data[species]
        phi_truth, phi_ref, grad_ref, velocity = self._geometry(species, eta)
        trial_count = int(getattr(bank, f"{species}_sample_indices").shape[0])
        targets, target_dots = jax.vmap(
            lambda trial: self._reconstruct(species, phi_truth, bank, trial)
        )(jnp.arange(trial_count, dtype=jnp.int32))
        projection = self.projector.project_trajectory(
            phi_ref, data.reference_weights, targets
        )

        def law_for_trial(weights):
            def law_at_time(nodes, projected_weights, mass, truth):
                projected_shape = histogram_mass(
                    nodes, projected_weights, self.law_grid
                )
                return periodic_grid_finite_mmd2(
                    mass * projected_shape, truth, self.law_kernel_fft
                )

            return jax.vmap(law_at_time)(
                data.reference_nodes,
                weights,
                data.target_mass,
                self.truth_finite_histogram[species],
            )

        law = jax.vmap(law_for_trial)(projection.weights)
        integrate = lambda values: jnp.sum(
            values * self.time_weights[None, :], axis=1
        )
        law_risk = integrate(law.finite_measure_risk)
        shape_mmd = integrate(law.shape_mmd)
        mass_error = integrate(law.mass_error)
        zero = jnp.zeros((trial_count,), dtype=jnp.float64)

        tangent_action = zero
        tangent_transport = zero
        tangent_reaction = zero
        if compute_tangent:
            tangent_phi, tangent_grad = phi_ref, grad_ref
            raw_targets = data.target_mass[None, :, None] * targets
            raw_target_dots = (
                data.target_mass_dot[None, :, None] * targets
                + data.target_mass[None, :, None] * target_dots
            )
            if self.sensors.mass_is_global_observation:
                tangent_phi, tangent_grad = append_global_mass_observable(
                    tangent_phi, tangent_grad
                )
                raw_targets = jnp.concatenate(
                    [
                        raw_targets,
                        jnp.broadcast_to(
                            data.target_mass[None, :, None],
                            (trial_count, len(self.times), 1),
                        ),
                    ],
                    axis=-1,
                )
                raw_target_dots = jnp.concatenate(
                    [
                        raw_target_dots,
                        jnp.broadcast_to(
                            data.target_mass_dot[None, :, None],
                            (trial_count, len(self.times), 1),
                        ),
                    ],
                    axis=-1,
                )

            def tangent_for_trial(weights, raw_target, raw_target_dot):
                return jax.vmap(
                    lambda phi, grad, vel, weight, mass, target, target_dot, source: unbalanced_tangent_action(
                        phi=phi,
                        grad_phi=grad,
                        velocity=vel,
                        normalized_weights=weight,
                        mass=mass,
                        target_raw_moments=target,
                        target_raw_moment_dot=target_dot,
                        reference_source_rate=source,
                        reaction_kappa=self.reaction_kappa,
                        ridge=float(self.particle_cfg.tangent_ridge),
                    )
                )(
                    tangent_phi,
                    tangent_grad,
                    velocity,
                    weights,
                    data.target_mass,
                    raw_target,
                    raw_target_dot,
                    data.reference_source_rate,
                )

            tangent = jax.vmap(tangent_for_trial)(
                projection.weights, raw_targets, raw_target_dots
            )
            tangent_action = integrate(tangent.total_action)
            tangent_transport = integrate(tangent.transport_action)
            tangent_reaction = integrate(tangent.reaction_action)

        full_action = zero
        move_action = zero
        reaction_action = zero
        max_pde = zero
        if compute_full:
            advective = jnp.einsum("tnmd,tnd->tnm", grad_ref, velocity)
            mean_advective = jnp.einsum(
                "btn,tnm->btm", projection.weights, advective
            )
            g = jnp.einsum("tnm,btm->btn", advective, projection.lam)
            mean_g = jnp.einsum("btn,btn->bt", projection.weights, g)
            centered_phi = (
                phi_ref[None, :, :, :] - projection.moments[:, :, None, :]
            )
            cov_phi_g = jnp.einsum(
                "btn,btnm,btn->btm",
                projection.weights,
                centered_phi,
                g - mean_g[:, :, None],
            )
            rhs = target_dots - mean_advective - cov_phi_g
            eye = jnp.eye(phi_ref.shape[-1], dtype=jnp.float64)
            lambda_dot = jnp.linalg.solve(
                projection.covariance
                + float(self.particle_cfg.covariance_ridge) * eye,
                rhs[..., None],
            ).squeeze(-1)
            forcing = (
                jnp.einsum("btnm,btm->btn", centered_phi, lambda_dot)
                + g
                - mean_g[:, :, None]
            )
            forcing = forcing - jnp.einsum(
                "btn,btn->bt", projection.weights, forcing
            )[:, :, None]

            def raster_for_trial(weights, source):
                return jax.vmap(
                    lambda nodes, weight, forcing_row: rasterize_periodic_particles3d(
                        nodes,
                        weight,
                        forcing_row,
                        self.grid,
                        bandwidth=self.raster_bandwidth,
                    )
                )(data.reference_nodes, weights, source)

            raster = jax.vmap(raster_for_trial)(projection.weights, forcing)
            h_ub = (
                raster.h
                + data.target_relative_mass_rate[None, :, None, None, None]
                - data.reference_source_rate[None, :, None, None, None]
            )
            flat_shape = (trial_count * len(self.times),) + self.grid.shape
            q_batch = raster.q.reshape(flat_shape)
            h_batch = h_ub.reshape(flat_shape)
            mass_batch = jnp.broadcast_to(
                data.target_mass[None, :], (trial_count, len(self.times))
            ).reshape(-1)
            if self.full_action_backend == "tesseract_cpp":
                correction = solve_unbalanced_screened_poisson3d_batch_tesseract(
                    q_batch,
                    h_batch,
                    mass=mass_batch,
                    grid=self.grid,
                    config=self.correction_cfg,
                )
            else:
                correction = solve_unbalanced_screened_poisson_batch_jax(
                    q_batch,
                    h_batch,
                    mass=mass_batch,
                    grid=self.grid,
                    config=self.correction_cfg,
                )
            correction_shape = (trial_count, len(self.times))
            full_action = integrate(correction.total_action.reshape(correction_shape))
            move_action = integrate(correction.move_action.reshape(correction_shape))
            reaction_action = integrate(
                correction.reaction_action.reshape(correction_shape)
            )
            max_pde = jnp.max(
                correction.relative_residual.reshape(correction_shape), axis=1
            )

        return {
            "law_risk": law_risk,
            "shape_mmd": shape_mmd,
            "mass_error": mass_error,
            "tangent_action": tangent_action,
            "tangent_transport": tangent_transport,
            "tangent_reaction": tangent_reaction,
            "full_action": full_action,
            "move_action": move_action,
            "reaction_action": reaction_action,
            "reaction_fraction": jnp.where(
                full_action > 0.0, reaction_action / full_action, 0.0
            ),
            "max_calibration_residual": jnp.max(
                jnp.linalg.norm(projection.residual, axis=-1), axis=1
            ),
            "min_ess_fraction": jnp.min(projection.ess_fraction, axis=1),
            "max_pde_relative_residual": max_pde,
        }

    def trial_values_batch(
        self,
        eta: Array,
        bank: UnbalancedObservationBank,
        *,
        compute_tangent: bool = True,
        compute_full: bool = True,
    ):
        """Vectorized counterpart of :meth:`trial_values`."""
        plus = self._species_metric_batch(
            "plus",
            eta,
            bank,
            compute_tangent=compute_tangent,
            compute_full=compute_full,
        )
        minus = self._species_metric_batch(
            "minus",
            eta,
            bank,
            compute_tangent=compute_tangent,
            compute_full=compute_full,
        )
        law = (
            self.risk_weights["plus"] * plus["law_risk"]
            + self.risk_weights["minus"] * minus["law_risk"]
        )
        tangent = (
            self.species_weights["plus"] * plus["tangent_action"]
            + self.species_weights["minus"] * minus["tangent_action"]
        )
        full = (
            self.species_weights["plus"] * plus["full_action"]
            + self.species_weights["minus"] * minus["full_action"]
        )
        return law, tangent, full, plus, minus

    def trial_values(
        self,
        eta: Array,
        bank: UnbalancedObservationBank,
        trial: int,
        *,
        compute_tangent: bool = True,
        compute_full: bool = True,
    ):
        plus = self._species_trial(
            "plus", eta, bank, trial,
            compute_tangent=compute_tangent, compute_full=compute_full,
        )
        minus = self._species_trial(
            "minus", eta, bank, trial,
            compute_tangent=compute_tangent, compute_full=compute_full,
        )
        law = self.risk_weights["plus"] * plus["law_risk"] + self.risk_weights["minus"] * minus["law_risk"]
        tangent = self.species_weights["plus"] * plus["tangent_action"] + self.species_weights["minus"] * minus["tangent_action"]
        full = self.species_weights["plus"] * plus["full_action"] + self.species_weights["minus"] * minus["full_action"]
        return law, tangent, full, plus, minus

    def mean_metric(self, eta: Array, bank: UnbalancedObservationBank, name: str) -> Array:
        index = {"law_risk": 0, "tangent_action": 1, "full_action": 2}.get(name)
        if index is None:
            raise ValueError("unknown unbalanced metric")
        values = self.trial_values_batch(
            eta,
            bank,
            compute_tangent=name != "law_risk",
            compute_full=name == "full_action",
        )[index]
        return jnp.mean(values)

    def exact_trial_rows(self, eta: Array, bank: UnbalancedObservationBank) -> list[dict[str, Any]]:
        validity = self.cfg["validity"]
        rows = []
        for trial in range(int(bank.plus_sample_indices.shape[0])):
            law, tangent, full, plus, minus = self.trial_values(eta, bank, trial)
            max_calibration = jnp.maximum(plus["max_calibration_residual"], minus["max_calibration_residual"])
            min_ess = jnp.minimum(plus["min_ess_fraction"], minus["min_ess_fraction"])
            max_pde = jnp.maximum(plus["max_pde_relative_residual"], minus["max_pde_relative_residual"])
            valid = bool(
                float(max_calibration) <= float(validity["max_calibration_residual"])
                and float(min_ess) >= float(validity["min_ess_fraction"])
                and float(max_pde) <= float(validity["max_screened_pde_relative_residual"])
            )
            rows.append({
                "trial": trial,
                "valid": valid,
                "law_risk_total": float(law),
                "law_risk_plus": float(plus["law_risk"]),
                "law_risk_minus": float(minus["law_risk"]),
                "shape_mmd_plus": float(plus["shape_mmd"]),
                "shape_mmd_minus": float(minus["shape_mmd"]),
                "mass_error_plus": float(plus["mass_error"]),
                "mass_error_minus": float(minus["mass_error"]),
                "tangent_unbalanced_total": float(tangent),
                "tangent_transport_plus": float(plus["tangent_transport"]),
                "tangent_reaction_plus": float(plus["tangent_reaction"]),
                "tangent_transport_minus": float(minus["tangent_transport"]),
                "tangent_reaction_minus": float(minus["tangent_reaction"]),
                "full_unbalanced_action_total": float(full),
                "full_unbalanced_action_plus": float(plus["full_action"]),
                "full_unbalanced_action_minus": float(minus["full_action"]),
                "move_action_plus": float(plus["move_action"]),
                "reaction_action_plus": float(plus["reaction_action"]),
                "move_action_minus": float(minus["move_action"]),
                "reaction_action_minus": float(minus["reaction_action"]),
                "reaction_fraction_plus": float(plus["reaction_fraction"]),
                "reaction_fraction_minus": float(minus["reaction_fraction"]),
                "reaction_fraction_total": float(
                    (self.species_weights["plus"] * plus["reaction_action"] + self.species_weights["minus"] * minus["reaction_action"])
                    / jnp.maximum(full, 1.0e-300)
                ),
                "max_calibration_residual": float(max_calibration),
                "min_ess_fraction": float(min_ess),
                "max_screened_pde_relative_residual": float(max_pde),
                "sensor_geometry": np.asarray(self.sensors.centers(eta)).tolist(),
            })
        return rows

    def certified_trial_rows(
        self,
        eta: Array,
        bank: UnbalancedObservationBank,
        *,
        scalar_spotcheck_trials: int = 2,
    ) -> list[dict[str, Any]]:
        """Serialize batched trials after checking a scalar authority subset."""
        values = self.trial_values_batch(
            eta, bank, compute_tangent=True, compute_full=True
        )
        law, tangent, full, plus, minus = values
        validity = self.cfg["validity"]
        max_calibration = jnp.maximum(
            plus["max_calibration_residual"], minus["max_calibration_residual"]
        )
        min_ess = jnp.minimum(
            plus["min_ess_fraction"], minus["min_ess_fraction"]
        )
        max_pde = jnp.maximum(
            plus["max_pde_relative_residual"],
            minus["max_pde_relative_residual"],
        )
        trial_count = int(bank.plus_sample_indices.shape[0])
        geometry = np.asarray(self.sensors.centers(eta)).tolist()
        rows = []
        for trial in range(trial_count):
            valid = bool(
                float(max_calibration[trial])
                <= float(validity["max_calibration_residual"])
                and float(min_ess[trial])
                >= float(validity["min_ess_fraction"])
                and float(max_pde[trial])
                <= float(validity["max_screened_pde_relative_residual"])
            )
            rows.append({
                "trial": trial,
                "valid": valid,
                "law_risk_total": float(law[trial]),
                "law_risk_plus": float(plus["law_risk"][trial]),
                "law_risk_minus": float(minus["law_risk"][trial]),
                "shape_mmd_plus": float(plus["shape_mmd"][trial]),
                "shape_mmd_minus": float(minus["shape_mmd"][trial]),
                "mass_error_plus": float(plus["mass_error"][trial]),
                "mass_error_minus": float(minus["mass_error"][trial]),
                "tangent_unbalanced_total": float(tangent[trial]),
                "tangent_transport_plus": float(
                    plus["tangent_transport"][trial]
                ),
                "tangent_reaction_plus": float(
                    plus["tangent_reaction"][trial]
                ),
                "tangent_transport_minus": float(
                    minus["tangent_transport"][trial]
                ),
                "tangent_reaction_minus": float(
                    minus["tangent_reaction"][trial]
                ),
                "full_unbalanced_action_total": float(full[trial]),
                "full_unbalanced_action_plus": float(plus["full_action"][trial]),
                "full_unbalanced_action_minus": float(minus["full_action"][trial]),
                "move_action_plus": float(plus["move_action"][trial]),
                "reaction_action_plus": float(plus["reaction_action"][trial]),
                "move_action_minus": float(minus["move_action"][trial]),
                "reaction_action_minus": float(minus["reaction_action"][trial]),
                "reaction_fraction_plus": float(
                    plus["reaction_fraction"][trial]
                ),
                "reaction_fraction_minus": float(
                    minus["reaction_fraction"][trial]
                ),
                "reaction_fraction_total": float(
                    (
                        self.species_weights["plus"]
                        * plus["reaction_action"][trial]
                        + self.species_weights["minus"]
                        * minus["reaction_action"][trial]
                    )
                    / jnp.maximum(full[trial], 1.0e-300)
                ),
                "max_calibration_residual": float(max_calibration[trial]),
                "min_ess_fraction": float(min_ess[trial]),
                "max_screened_pde_relative_residual": float(max_pde[trial]),
                "sensor_geometry": geometry,
            })

        checked = min(max(int(scalar_spotcheck_trials), 0), trial_count)
        for trial in range(checked):
            scalar = self.trial_values(eta, bank, trial)
            expected = np.asarray(
                [float(scalar[0]), float(scalar[1]), float(scalar[2])]
            )
            actual = np.asarray([
                rows[trial]["law_risk_total"],
                rows[trial]["tangent_unbalanced_total"],
                rows[trial]["full_unbalanced_action_total"],
            ])
            try:
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=2.0e-6,
                    atol=2.0e-7,
                    err_msg=f"batched/scalar trial mismatch at trial {trial}",
                )
            except AssertionError:
                print(
                    "unbalanced certification batch/scalar spot check failed; "
                    "falling back to authoritative scalar trial evaluation",
                    flush=True,
                )
                return self.exact_trial_rows(eta, bank)
        return rows

    def audit_metric(self, eta: Array, bank: UnbalancedObservationBank, name: str) -> dict[str, Any]:
        if name not in {"law_risk", "tangent_action", "full_action"}:
            raise ValueError("unknown unbalanced metric")
        compute_tangent = name != "law_risk"
        compute_full = name == "full_action"
        value_index = {"law_risk": 0, "tangent_action": 1, "full_action": 2}[name]
        validity = self.cfg["validity"]
        values = self.trial_values_batch(
            eta,
            bank,
            compute_tangent=compute_tangent,
            compute_full=compute_full,
        )
        plus, minus = values[3], values[4]
        max_calibration = jnp.maximum(
            plus["max_calibration_residual"], minus["max_calibration_residual"]
        )
        min_ess = jnp.minimum(
            plus["min_ess_fraction"], minus["min_ess_fraction"]
        )
        max_pde = jnp.maximum(
            plus["max_pde_relative_residual"],
            minus["max_pde_relative_residual"],
        )
        rows = []
        for trial in range(int(bank.plus_sample_indices.shape[0])):
            valid = (
                float(max_calibration[trial])
                <= float(validity["max_calibration_residual"])
                and float(min_ess[trial]) >= float(validity["min_ess_fraction"])
                and (
                    not compute_full
                    or float(max_pde[trial])
                    <= float(validity["max_screened_pde_relative_residual"])
                )
            )
            rows.append(
                {
                    "trial": trial,
                    "value": float(values[value_index][trial]),
                    "valid": valid,
                    "max_calibration_residual": float(max_calibration[trial]),
                    "min_ess_fraction": float(min_ess[trial]),
                    "max_screened_pde_relative_residual": (
                        float(max_pde[trial]) if compute_full else None
                    ),
                }
            )
        return {
            "value": float(np.mean([row["value"] for row in rows])),
            "valid": all(row["valid"] for row in rows),
            "trials": len(rows),
            "rows": rows,
        }

    def optimize_designs(self, bank: UnbalancedObservationBank) -> UnbalancedDesignComparison:
        """Law first, then tangent and Full within the declared risk tolerance."""
        try:
            from .percentage_selection import optimize_percentage_designs
        except ImportError:  # pragma: no cover - direct script execution
            from percentage_selection import optimize_percentage_designs

        result = optimize_percentage_designs(self, bank)
        return UnbalancedDesignComparison(
            law_eta=result["law_eta"],
            tangent_eta=result["tangent_eta"],
            full_eta=result["full_eta"],
            risk_star=float(result["risk_star"]),
            risk_max=float(result["risk_max"]),
            candidates=result["candidates"],
            certified=True,
        )

    def _legacy_refinement_heavy_optimize_designs(
        self, bank: UnbalancedObservationBank
    ) -> UnbalancedDesignComparison:
        """Archived source implementation; intentionally unused here."""
        opt = self.cfg["optimization"]

        def gradient_bank(key: str, default: int) -> UnbalancedObservationBank:
            available = int(bank.plus_sample_indices.shape[0])
            count = min(int(opt.get(key, default)), available)
            if count < 1:
                raise ValueError(f"optimization.{key} must be positive")
            return UnbalancedObservationBank(*(row[:count] for row in bank))

        law_gradient_bank = gradient_bank("law_gradient_trials", 8)
        tangent_gradient_bank = gradient_bank("tangent_gradient_trials", 8)
        full_gradient_bank = gradient_bank("full_gradient_trials", 2)
        tangent_refinement_bank = gradient_bank("tangent_refinement_trials", 32)
        full_refinement_bank = gradient_bank("full_refinement_trials", 16)
        starts = random_periodic_sensor_starts(
            jax.random.PRNGKey(int(self.cfg["seed"]) + 17),
            int(opt.get("start_count", 16)),
            n_sensors=self.sensors.n_sensors,
            box_size=self.sensors.box_size,
            min_separation=float(self.cfg["measurement"].get("min_sep", 0.0)),
            oversample=int(opt.get("start_oversample", 64)),
        )
        geometry = ((periodic_separation_violation(
            float(self.cfg["measurement"].get("min_sep", 0.0)),
            n_sensors=self.sensors.n_sensors,
            box_size=self.sensors.box_size,
        ), 0.0),)

        def optimizer(stage: str) -> OptimizerConfig:
            return OptimizerConfig(
                steps=int(opt.get(f"{stage}_steps", 30)),
                learning_rate=float(opt.get(f"{stage}_learning_rate", 0.006)),
                constraint_penalty=float(opt.get("constraint_penalty", 1.0e4)),
                feasibility_tol=float(opt.get("feasibility_tol", 1.0e-6)),
            )

        def select(candidates, metric: str, risk_max: float | None = None):
            audited = []
            limit = int(opt.get(f"{metric.split('_')[0]}_exact_audit_candidates", 8))
            ordered = sorted(candidates, key=lambda row: row.value)

            def audit(rows):
                for candidate in rows:
                    if not candidate.feasible:
                        continue
                    score = self.audit_metric(candidate.eta, bank, metric)
                    risk = score if metric == "law_risk" else self.audit_metric(candidate.eta, bank, "law_risk")
                    audited.append((candidate, score, risk))

            def certified():
                return [
                    row for row in audited
                    if row[1]["valid"]
                    and row[2]["valid"]
                    and (risk_max is None or row[2]["value"] <= risk_max)
                ]

            audit(ordered[:limit])
            valid = certified()
            # The audit count is a normal-case speed budget, not permission for
            # a noisy proxy ranking to discard retained feasible incumbents.
            if not valid:
                audit(ordered[limit:])
                valid = certified()
            if not valid:
                raise RuntimeError(f"no certified {metric} candidate survived")
            return min(valid, key=lambda row: row[1]["value"]), audited

        def progress(stage: str):
            return lambda completed, total: print(
                f"unbalanced design stage={stage} optimized={completed}/{total}",
                flush=True,
            )

        def best_audited(rows, *, risk_max: float | None = None):
            valid = [
                row for row in rows
                if row[1]["valid"]
                and row[2]["valid"]
                and (risk_max is None or row[2]["value"] <= risk_max)
            ]
            if not valid:
                raise RuntimeError("no certified refined candidate survived")
            return min(valid, key=lambda row: row[1]["value"])

        def refinement_starts(rows, *, risk_max: float | None = None):
            valid = [
                row for row in rows
                if row[1]["valid"]
                and row[2]["valid"]
                and (risk_max is None or row[2]["value"] <= risk_max)
            ]
            count = min(int(opt.get("refinement_start_count", 2)), len(valid))
            if count < 1:
                raise RuntimeError("no certified candidate available for refinement")
            ordered = sorted(valid, key=lambda row: row[1]["value"])
            return jnp.stack([row[0].eta for row in ordered[:count]])

        def quasi_newton(
            stage: str,
            primary,
            stage_starts,
            constraints,
            *,
            label: str | None = None,
        ):
            diagnostic_label = label or stage
            return _quasi_newton_candidates(
                primary,
                stage_starts,
                constraints=constraints,
                canonicalize=self.sensors.canonicalize,
                penalty=float(
                    opt.get(
                        "refinement_constraint_penalty",
                        opt.get("constraint_penalty", 1.0e4),
                    )
                ),
                feasibility_tol=float(opt.get("feasibility_tol", 1.0e-6)),
                maxiter=int(opt.get(f"{stage}_refinement_maxiter", 60)),
                gtol=float(opt.get("refinement_gtol", 1.0e-8)),
                ftol=float(opt.get("refinement_ftol", 1.0e-12)),
                progress_callback=progress(label or stage),
                diagnostic_callback=lambda index, result, candidate, eta0,
                accepted_eta, raw_violations, projected: print(
                    "unbalanced design stage="
                    f"{diagnostic_label} candidate={index} "
                    f"success={bool(result.success)} status={int(result.status)} "
                    f"iterations={int(result.nit)} "
                    f"gradient_inf={float(np.max(np.abs(result.jac))):.6g} "
                    "raw_step_norm="
                    f"{float(np.linalg.norm(result.x - eta0)):.6g} "
                    "accepted_step_norm="
                    f"{float(np.linalg.norm(accepted_eta - eta0)):.6g} "
                    f"projected={bool(projected)} "
                    "raw_max_violation="
                    f"{max(raw_violations, default=-np.inf):.6g} "
                    f"feasible={bool(candidate.feasible)} "
                    "max_violation="
                    f"{max(candidate.violations, default=-np.inf):.6g} "
                    f"message={str(result.message)}",
                    flush=True,
                ),
            )

        print(
            "unbalanced design stage=law compiling "
            f"gradient_trials={law_gradient_bank.plus_sample_indices.shape[0]}",
            flush=True,
        )
        law_candidates = optimize_multistart_candidates(
            lambda eta: self.mean_metric(eta, law_gradient_bank, "law_risk"),
            starts[: int(opt.get("law_start_count", len(starts)))],
            optimizer("law"), constraints=geometry,
            canonicalize=self.sensors.canonicalize,
            vectorize_starts=False,
            progress_callback=progress("law"),
        )
        (law, law_audit, _), law_rows = select(law_candidates, "law_risk")
        law_refinement_label = "law_exact_refinement"
        print(
            f"unbalanced design stage={law_refinement_label} compiling "
            f"gradient_trials={bank.plus_sample_indices.shape[0]}",
            flush=True,
        )
        law_exact_candidates = quasi_newton(
            "law",
            lambda eta: self.mean_metric(eta, bank, "law_risk"),
            refinement_starts(law_rows),
            geometry,
            label=law_refinement_label,
        )
        _, law_exact_rows = select(law_exact_candidates, "law_risk")
        law_rows.extend(law_exact_rows)
        law, law_audit, _ = best_audited(law_rows)
        risk_star = float(law_audit["value"])
        proxy_cfg = copy.deepcopy(self.cfg)
        proxy_cfg["full_action"]["grid_shape"] = list(
            opt.get("full_gradient_grid_shape", [24, 24, 12])
        )
        proxy_cfg["full_action"]["cg_tol"] = float(
            opt.get("full_gradient_cg_tol", 1.0e-6)
        )
        proxy_cfg["full_action"]["cg_maxiter"] = int(
            opt.get("full_gradient_cg_maxiter", 360)
        )
        proxy_experiment = UnbalancedActiveNematicExperiment(
            proxy_cfg,
            times=self.times,
            plus=self.data["plus"],
            minus=self.data["minus"],
        )
        refinement_cfg = copy.deepcopy(self.cfg)
        refinement_cfg["full_action"]["grid_shape"] = list(
            opt.get("full_refinement_grid_shape", [36, 36, 18])
        )
        refinement_cfg["full_action"]["cg_tol"] = float(
            opt.get("full_refinement_cg_tol", 3.0e-7)
        )
        refinement_cfg["full_action"]["cg_maxiter"] = int(
            opt.get("full_refinement_cg_maxiter", 720)
        )
        refinement_experiment = UnbalancedActiveNematicExperiment(
            refinement_cfg,
            times=self.times,
            plus=self.data["plus"],
            minus=self.data["minus"],
        )
        epsilon_r = float(self.cfg["law"]["epsilon_r"])

        def run_action_stages(law_candidate, exact_risk_star, pass_index):
            risk_ceiling = exact_risk_star + epsilon_r
            refinement_risk_margin = float(
                opt.get("refinement_risk_margin", 1.0e-5)
            )
            if not 0.0 <= refinement_risk_margin < epsilon_r:
                raise ValueError(
                    "optimization.refinement_risk_margin must be nonnegative "
                    "and smaller than law.epsilon_r"
                )
            tangent_proxy_ceiling = float(
                self.mean_metric(
                    law_candidate.eta, tangent_gradient_bank, "law_risk"
                )
            ) + epsilon_r
            tangent_constraints = geometry + ((
                lambda eta: self.mean_metric(
                    eta, tangent_gradient_bank, "law_risk"
                ),
                tangent_proxy_ceiling,
            ),)
            tangent_label = f"tangent_broad_pass_{pass_index}"
            print(
                f"unbalanced design stage={tangent_label} compiling "
                "gradient_trials="
                f"{tangent_gradient_bank.plus_sample_indices.shape[0]}",
                flush=True,
            )
            tangent_candidates = optimize_multistart_candidates(
                lambda eta: self.mean_metric(
                    eta, tangent_gradient_bank, "tangent_action"
                ),
                jnp.concatenate([
                    law_candidate.eta[None],
                    starts[: int(opt.get("tangent_start_count", len(starts)))],
                ], axis=0),
                optimizer("tangent"),
                constraints=tangent_constraints,
                canonicalize=self.sensors.canonicalize,
                vectorize_starts=False,
                progress_callback=progress(tangent_label),
            )
            (tangent_candidate, _, _), tangent_audits = select(
                tangent_candidates, "tangent_action", risk_ceiling
            )
            exact_risk_constraints = geometry + ((
                lambda eta: self.mean_metric(eta, bank, "law_risk"),
                risk_ceiling - refinement_risk_margin,
            ),)
            tangent_refinement_label = f"tangent_refinement_pass_{pass_index}"
            print(
                f"unbalanced design stage={tangent_refinement_label} compiling "
                "gradient_trials="
                f"{tangent_refinement_bank.plus_sample_indices.shape[0]}",
                flush=True,
            )
            tangent_refined_candidates = quasi_newton(
                "tangent",
                lambda eta: self.mean_metric(
                    eta, tangent_refinement_bank, "tangent_action"
                ),
                refinement_starts(
                    tangent_audits,
                    risk_max=risk_ceiling - refinement_risk_margin,
                ),
                exact_risk_constraints,
                label=tangent_refinement_label,
            )
            _, tangent_refined_audits = select(
                tangent_refined_candidates, "tangent_action", risk_ceiling
            )
            tangent_audits.extend(tangent_refined_audits)
            tangent_candidate, _, _ = best_audited(
                tangent_audits, risk_max=risk_ceiling
            )

            full_proxy_ceiling = float(
                self.mean_metric(
                    law_candidate.eta, full_gradient_bank, "law_risk"
                )
            ) + epsilon_r
            full_constraints = geometry + ((
                lambda eta: self.mean_metric(
                    eta, full_gradient_bank, "law_risk"
                ),
                full_proxy_ceiling,
            ),)
            full_label = f"full_proxy_pass_{pass_index}"
            print(
                f"unbalanced design stage={full_label} compiling "
                f"gradient_trials={full_gradient_bank.plus_sample_indices.shape[0]}",
                flush=True,
            )
            full_candidates = optimize_multistart_candidates(
                lambda eta: proxy_experiment.mean_metric(
                    eta, full_gradient_bank, "full_action"
                ),
                jnp.concatenate([
                    law_candidate.eta[None],
                    tangent_candidate.eta[None],
                    starts[: int(opt.get("full_start_count", len(starts)))],
                ], axis=0),
                optimizer("full"),
                constraints=full_constraints,
                canonicalize=self.sensors.canonicalize,
                vectorize_starts=False,
                progress_callback=progress(full_label),
            )
            (full_candidate, _, _), full_audits = select(
                full_candidates, "full_action", risk_ceiling
            )
            full_refinement_label = f"full_refinement_pass_{pass_index}"
            print(
                f"unbalanced design stage={full_refinement_label} compiling "
                f"gradient_trials={full_refinement_bank.plus_sample_indices.shape[0]} "
                "grid_shape="
                f"{refinement_cfg['full_action']['grid_shape']}",
                flush=True,
            )
            full_refined_candidates = quasi_newton(
                "full",
                lambda eta: refinement_experiment.mean_metric(
                    eta, full_refinement_bank, "full_action"
                ),
                refinement_starts(
                    full_audits,
                    risk_max=risk_ceiling - refinement_risk_margin,
                ),
                exact_risk_constraints,
                label=full_refinement_label,
            )
            _, full_refined_audits = select(
                full_refined_candidates, "full_action", risk_ceiling
            )
            full_audits.extend(full_refined_audits)
            full_candidate, _, _ = best_audited(
                full_audits, risk_max=risk_ceiling
            )
            return tangent_candidate, tangent_audits, full_candidate, full_audits

        maximum_refinements = int(opt.get("law_refinement_rounds", 2))
        refinement_tolerance = float(opt.get("law_refinement_tolerance", 1.0e-8))
        for pass_index in range(maximum_refinements + 1):
            tangent, tangent_rows, full, full_rows = run_action_stages(
                law, risk_star, pass_index + 1
            )
            later_best = min(
                (row for row in tangent_rows + full_rows if row[2]["valid"]),
                key=lambda row: row[2]["value"],
            )
            improvement = risk_star - float(later_best[2]["value"])
            if improvement <= refinement_tolerance:
                break
            if pass_index >= maximum_refinements:
                raise RuntimeError(
                    "a later action-stage geometry still improves exact law risk "
                    f"by {improvement:.6g} after {maximum_refinements} law "
                    "refinement rounds"
                )

            refinement_label = f"law_refinement_{pass_index + 1}"
            print(
                f"unbalanced design stage={refinement_label} trigger="
                f"later_exact_risk_improvement={improvement:.6g}",
                flush=True,
            )
            refinement_candidates = quasi_newton(
                "law",
                lambda eta: self.mean_metric(eta, bank, "law_risk"),
                jnp.stack([law.eta, later_best[0].eta]),
                geometry,
                label=refinement_label,
            )
            _, refinement_rows = select(
                refinement_candidates, "law_risk"
            )
            law_rows.extend(refinement_rows)
            law, law_audit, _ = min(
                (row for row in law_rows if row[1]["valid"]),
                key=lambda row: row[1]["value"],
            )
            risk_star = float(law_audit["value"])
        else:  # pragma: no cover - the loop always breaks or raises.
            raise AssertionError("unreachable law refinement state")

        risk_max = risk_star + epsilon_r

        def serial(rows):
            return [{
                "eta": np.asarray(candidate.eta).tolist(),
                "proxy_value": candidate.value,
                "audit": audit,
                "law_screen": risk,
            } for candidate, audit, risk in rows]

        return UnbalancedDesignComparison(
            law_eta=law.eta,
            tangent_eta=tangent.eta,
            full_eta=full.eta,
            risk_star=risk_star,
            risk_max=risk_max,
            candidates={"law": serial(law_rows), "tangent": serial(tangent_rows), "full": serial(full_rows)},
            certified=True,
        )
