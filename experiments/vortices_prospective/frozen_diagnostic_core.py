from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from common import trap_weights
from evaluator import AggregateObservationBank, ProspectiveEvaluator
from mfsi.decomposition import raster_tangent_projection
from mfsi.poisson import solve_weighted_poisson_physical_direct_batch
from mfsi.raster import rasterize_projected_particles_rect

jax.config.update("jax_enable_x64", True)


def summary(values) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": None, "se": None}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "se": float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
    }


def paired_statistics(law, full, *, bootstrap_seed: int = 8119) -> dict[str, Any]:
    law = np.asarray(law, dtype=np.float64)
    full = np.asarray(full, dtype=np.float64)
    valid = np.isfinite(law) & np.isfinite(full)
    law, full = law[valid], full[valid]
    difference = full - law
    relative_reduction = (law - full) / law
    n = len(difference)
    if n < 2:
        raise ValueError("paired inference requires at least two valid trials")
    se = float(np.std(difference, ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    rng = np.random.default_rng(int(bootstrap_seed))
    draw = rng.integers(0, n, size=(50000, n))
    boot = np.mean(difference[draw], axis=1)
    rel_boot = np.mean(relative_reduction[draw], axis=1)
    out = {
        "aligned_common_randomness": True,
        "valid_pair_count": int(n),
        "difference_full_minus_law": summary(difference),
        "paired_t_95_ci": [float(np.mean(difference) - critical * se), float(np.mean(difference) + critical * se)],
        "paired_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "fraction_full_lower": float(np.mean(difference < 0.0)),
        "fraction_full_higher": float(np.mean(difference > 0.0)),
        "fraction_equal": float(np.mean(difference == 0.0)),
        "relative_reduction_trialwise": summary(relative_reduction),
        "relative_reduction_bootstrap_95_ci": np.quantile(rel_boot, [0.025, 0.975]).tolist(),
        "ratio_of_means_reduction": float(1.0 - np.mean(full) / np.mean(law)),
    }
    return out


def curve_error_metrics(predicted, oracle, times) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(oracle, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError("curves must have matching [time, channel] shapes")
    w = trap_weights(times)
    rows = []
    for channel in range(pred.shape[1]):
        error = pred[:, channel] - true[:, channel]
        denom = np.sqrt(np.sum(w * true[:, channel] ** 2))
        correlation = (
            float(np.corrcoef(pred[:, channel], true[:, channel])[0, 1])
            if np.std(pred[:, channel]) > 0.0 and np.std(true[:, channel]) > 0.0
            else None
        )
        interior = error[1:-1]
        rows.append({
            "channel": channel,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "relative_rmse": float(np.sqrt(np.sum(w * error**2)) / denom) if denom > 1.0e-14 else None,
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "time_integrated_squared_error": float(np.sum(w * error**2)),
            "correlation": correlation,
            "endpoint_maximum_absolute_error": float(np.max(np.abs(error[[0, -1]]))),
            "interior_rmse": float(np.sqrt(np.mean(interior**2))) if len(interior) else 0.0,
            "error_by_time": error.tolist(),
        })
    return {
        "channels": rows,
        "aggregate_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "aggregate_maximum_absolute_error": float(np.max(np.abs(pred - true))),
    }


def reconstruct_exact_population(
    evaluator: ProspectiveEvaluator, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    response = jnp.asarray(response, dtype=jnp.float64)
    acquisition = response[evaluator.acq_idx]
    fit = evaluator.reconstructor.reconstruct(
        acquisition, acquisition[0], acquisition[-1]
    )
    margin = float(evaluator.cfg["moment_reconstruction"]["clip_margin"])
    inside = (fit.c > margin) & (fit.c < 1.0 - margin)
    return (
        np.asarray(jnp.clip(fit.c, margin, 1.0 - margin), dtype=np.float64),
        np.asarray(jnp.where(inside, fit.c_dot, 0.0), dtype=np.float64),
    )


def realized_observation_banks(
    evaluator: ProspectiveEvaluator,
    phi_hidden: np.ndarray,
    sample_indices: np.ndarray,
    detector_z: np.ndarray,
) -> tuple[AggregateObservationBank, AggregateObservationBank, np.ndarray, np.ndarray]:
    response_mean = np.mean(phi_hidden, axis=1)
    response_second = np.mean(phi_hidden * phi_hidden, axis=1)
    acq_idx = np.asarray(evaluator.acq_idx, dtype=np.int32)
    phi_acq = phi_hidden[acq_idx]
    sampled = np.empty(detector_z.shape, dtype=np.float64)
    for trial in range(len(sample_indices)):
        for acq in range(len(acq_idx)):
            sampled[trial, acq] = np.mean(
                phi_acq[acq, sample_indices[trial, acq]], axis=0
            )
    acq_mean = response_mean[acq_idx]
    variance = np.maximum(response_second[acq_idx] - acq_mean * acq_mean, 0.0)
    finite_se = np.sqrt(
        variance / float(evaluator.cfg["measurement"]["finite_n"])
    )
    effective_sampling_z = np.divide(
        sampled - acq_mean[None, :, :],
        finite_se[None, :, :],
        out=np.zeros_like(sampled),
        where=finite_se[None, :, :] > 1.0e-15,
    )
    sample_only = AggregateObservationBank(
        effective_sampling_z, np.zeros_like(detector_z)
    )
    observed = AggregateObservationBank(effective_sampling_z, detector_z)
    return sample_only, observed, response_mean, response_second


@dataclass
class ExplicitEvaluation:
    action_by_trial: np.ndarray
    action_by_time: np.ndarray
    lam: np.ndarray
    lambda_dot: np.ndarray
    covariance_condition: np.ndarray
    ess_fraction: np.ndarray
    calibration_residual: np.ndarray
    weights: np.ndarray | None
    forcing: np.ndarray | None
    time_term: np.ndarray | None
    advection_term: np.ndarray | None
    certification: dict[str, Any]

    @property
    def action_mean(self) -> float:
        return float(np.mean(self.action_by_trial))


def evaluate_explicit_moments(
    evaluator: ProspectiveEvaluator,
    eta,
    targets,
    target_dot,
    *,
    retain_particle_details: bool = False,
) -> ExplicitEvaluation:
    eta = jnp.asarray(eta, dtype=jnp.float64)
    targets = jnp.asarray(targets, dtype=jnp.float64)
    target_dot = jnp.asarray(target_dot, dtype=jnp.float64)
    if targets.ndim == 2:
        targets = targets[None, ...]
    if target_dot.ndim == 2:
        target_dot = target_dot[None, ...]
    phi = evaluator.sensors.features(evaluator.nodes, eta)
    grad = evaluator.sensors.feature_gradients(evaluator.nodes, eta)
    projection = evaluator.projector.project_trajectory(
        phi, evaluator.base_weights, targets
    )
    weights = projection.weights
    advective = jnp.einsum("tnmd,tnd->tnm", grad, evaluator.velocity)
    mean_advective = jnp.einsum("btn,tnm->btm", weights, advective)
    g = jnp.einsum("tnm,btm->btn", advective, projection.lam)
    mean_g = jnp.einsum("btn,btn->bt", weights, g)
    centered = phi[None, :, :, :] - projection.moments[:, :, None, :]
    cov_phi_g = jnp.einsum(
        "btn,btnm,btn->btm", weights, centered, g - mean_g[:, :, None]
    )
    ridge = float(evaluator.cfg["particle_mfsi"]["covariance_ridge"])
    eye = jnp.eye(phi.shape[-1], dtype=jnp.float64)
    lambda_dot = jnp.linalg.solve(
        projection.covariance + ridge * eye,
        (target_dot - mean_advective - cov_phi_g)[..., None],
    )[..., 0]
    time_term = jnp.einsum("btnm,btm->btn", centered, lambda_dot)
    advection_term = g - mean_g[:, :, None]
    forcing = time_term + advection_term
    forcing -= jnp.einsum("btn,btn->bt", weights, forcing)[:, :, None]

    def raster_trial(w_trial, f_trial):
        return jax.vmap(
            lambda x, w, f: rasterize_projected_particles_rect(
                x, w, f, evaluator.grid, evaluator.raster_cfg
            )
        )(evaluator.nodes, w_trial, f_trial)

    rasters = jax.vmap(raster_trial)(weights, forcing)
    leading = rasters.q.shape[:2]
    solved = solve_weighted_poisson_physical_direct_batch(
        np.asarray(rasters.q).reshape((-1,) + evaluator.grid.shape),
        np.asarray(rasters.h).reshape((-1,) + evaluator.grid.shape),
        evaluator.poisson_cfg,
    )
    action_by_time = np.asarray(solved.action, dtype=np.float64).reshape(leading)
    action_by_trial = np.sum(
        action_by_time * np.asarray(evaluator.time_weights)[None, :], axis=1
    )
    potential = jnp.asarray(solved.potential).reshape(rasters.q.shape)
    grid_features = evaluator.sensors.features(evaluator.grid.points(), eta)
    decomposition = raster_tangent_projection(
        potential,
        rasters.q,
        rasters.h,
        grid_features,
        dx=float(evaluator.poisson_cfg.dx),
        cell_area=float(evaluator.grid.cell_area),
        pinv_rcond=1.0e-10,
        operator_floor_rel=0.0,
        gauge_strength=0.0,
    )
    covariance = np.asarray(projection.covariance, dtype=np.float64)
    eig = np.linalg.eigvalsh(covariance)
    condition = np.max(eig, axis=-1) / np.maximum(np.min(eig, axis=-1), 1.0e-300)
    calibration = np.linalg.norm(np.asarray(projection.residual), axis=-1)
    ess = np.asarray(projection.ess_fraction, dtype=np.float64)
    poisson = np.asarray(solved.relative_residual).reshape(leading)
    compatibility = np.asarray(
        solved.maximum_component_compatibility_residual
    ).reshape(leading)
    converged = np.asarray(solved.solver_converged).reshape(leading)
    moment_residual = np.linalg.norm(
        np.asarray(decomposition.full_moment_residual), axis=-1
    )
    forcing_compatibility = np.abs(
        np.asarray(jnp.einsum("btn,btn->bt", weights, forcing))
    )
    mass = np.asarray(rasters.mass)
    arrays_for_finite = (
        action_by_time,
        np.asarray(projection.lam),
        np.asarray(lambda_dot),
        covariance,
        ess,
        calibration,
        poisson,
        compatibility,
        moment_residual,
    )
    certification = {
        "trial_count": int(targets.shape[0]),
        "invalid_trial_count": int(
            np.sum(
                (np.max(calibration, axis=1) > float(evaluator.cfg["validity"]["max_projection_residual"]))
                | (np.min(ess, axis=1) < float(evaluator.cfg["validity"]["min_ess_fraction"]))
                | (np.max(poisson, axis=1) > float(evaluator.cfg["validity"]["max_poisson_relative_residual"]))
                | (~np.all(converged, axis=1))
            )
        ),
        "max_projection_residual": float(np.max(calibration)),
        "min_ess_fraction": float(np.min(ess)),
        "max_covariance_condition_number": float(np.max(condition)),
        "min_covariance_eigenvalue": float(np.min(eig)),
        "max_forcing_compatibility_residual": float(np.max(forcing_compatibility)),
        "max_poisson_relative_residual": float(np.max(poisson)),
        "max_component_compatibility_residual": float(np.max(compatibility)),
        "max_full_moment_rate_residual": float(np.max(moment_residual)),
        "minimum_raster_mass": float(np.min(np.sum(mass, axis=(-2, -1)))),
        "maximum_raster_mass": float(np.max(np.sum(mass, axis=(-2, -1)))),
        "all_physical_solvers_converged": bool(np.all(converged)),
        "nan_or_inf_count": int(
            sum(np.size(a) - np.count_nonzero(np.isfinite(a)) for a in arrays_for_finite)
        ),
    }
    return ExplicitEvaluation(
        action_by_trial=action_by_trial,
        action_by_time=action_by_time,
        lam=np.asarray(projection.lam, dtype=np.float64),
        lambda_dot=np.asarray(lambda_dot, dtype=np.float64),
        covariance_condition=condition,
        ess_fraction=ess,
        calibration_residual=calibration,
        weights=np.asarray(weights, dtype=np.float64) if retain_particle_details else None,
        forcing=np.asarray(forcing, dtype=np.float64) if retain_particle_details else None,
        time_term=np.asarray(time_term, dtype=np.float64) if retain_particle_details else None,
        advection_term=np.asarray(advection_term, dtype=np.float64) if retain_particle_details else None,
        certification=certification,
    )


def projection_path_comparison(pred: ExplicitEvaluation, oracle: ExplicitEvaluation, times) -> dict[str, Any]:
    if pred.lam.shape[0] != 1 or oracle.lam.shape[0] != 1:
        raise ValueError("projection path comparison expects deterministic paths")
    lam_delta = np.linalg.norm(pred.lam[0] - oracle.lam[0], axis=-1)
    dot_delta = np.linalg.norm(pred.lambda_dot[0] - oracle.lambda_dot[0], axis=-1)
    return {
        "lambda_difference_norm_by_time": lam_delta.tolist(),
        "lambda_dot_difference_norm_by_time": dot_delta.tolist(),
        "lambda_difference": summary(lam_delta),
        "lambda_dot_difference": summary(dot_delta),
        "predicted_covariance_condition_by_time": pred.covariance_condition[0].tolist(),
        "oracle_covariance_condition_by_time": oracle.covariance_condition[0].tolist(),
        "predicted_ess_by_time": pred.ess_fraction[0].tolist(),
        "oracle_ess_by_time": oracle.ess_fraction[0].tolist(),
        "predicted_calibration_residual_by_time": pred.calibration_residual[0].tolist(),
        "oracle_calibration_residual_by_time": oracle.calibration_residual[0].tolist(),
    }


def forcing_path_comparison(pred: ExplicitEvaluation, oracle: ExplicitEvaluation, times) -> dict[str, Any]:
    for obj in (pred, oracle):
        if obj.weights is None or obj.forcing is None:
            raise ValueError("forcing comparison requires retained particle details")
    w = oracle.weights[0]
    total = np.sqrt(np.sum(w * (pred.forcing[0] - oracle.forcing[0]) ** 2, axis=-1))
    time_term = np.sqrt(np.sum(w * (pred.time_term[0] - oracle.time_term[0]) ** 2, axis=-1))
    advection = np.sqrt(
        np.sum(w * (pred.advection_term[0] - oracle.advection_term[0]) ** 2, axis=-1)
    )
    tw = trap_weights(np.asarray(times))
    return {
        "empirical_norm_under_oracle_q_by_time": total.tolist(),
        "time_calibration_term_error_by_time": time_term.tolist(),
        "reference_advection_term_error_by_time": advection.tolist(),
        "integrated_total_squared_error": float(np.sum(tw * total**2)),
        "integrated_time_calibration_squared_error": float(np.sum(tw * time_term**2)),
        "integrated_reference_advection_squared_error": float(np.sum(tw * advection**2)),
        "dominant_term": (
            "time_calibration"
            if np.sum(tw * time_term**2) >= np.sum(tw * advection**2)
            else "reference_advection"
        ),
    }
