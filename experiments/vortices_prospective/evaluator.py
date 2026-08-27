from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from aggregate_qois import qoi_features
from common import nested_indices, trap_weights
from mfsi.decomposition import raster_tangent_projection
from mfsi.grid import RectangularGrid2D
from mfsi.measurements import GaussianPointSensors2D
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.poisson import PoissonConfig, solve_weighted_poisson_physical_direct_batch
from mfsi.projection import EmpiricalIProjector, IProjectionConfig
from mfsi.raster import RasterConfig, rasterize_projected_particles_rect
from prospective_data import TargetProspectiveData

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AggregateObservationBank:
    sampling_z: np.ndarray
    detector_z: np.ndarray

    @property
    def trials(self) -> int:
        return int(self.sampling_z.shape[0])


def make_observation_bank(cfg: dict[str, Any], trials: int, namespace: int) -> AggregateObservationBank:
    measurement = cfg["measurement"]
    acq_n = int(cfg["time"]["acquisition_nodes"])
    sensors = int(measurement["n_sensors"])
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), int(namespace)]))
    shape = (int(trials), acq_n, sensors)
    return AggregateObservationBank(rng.standard_normal(shape), rng.standard_normal(shape))


def ensure_observation_bank(path: str | Path, cfg: dict[str, Any], trials: int, namespace: int):
    path = Path(path)
    expected = (int(trials), int(cfg["time"]["acquisition_nodes"]), int(cfg["measurement"]["n_sensors"]))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["sampling_z"].shape) == expected and int(data["namespace"]) == int(namespace):
                print(f"[randomness] reusing {path.name}", flush=True)
                return AggregateObservationBank(np.asarray(data["sampling_z"]), np.asarray(data["detector_z"]))
    bank = make_observation_bank(cfg, trials, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sampling_z=bank.sampling_z, detector_z=bank.detector_z, namespace=np.asarray(namespace))
    return bank


def _summarize(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "se": None, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "se": float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "n": int(len(finite)),
    }


class ProspectiveEvaluator:
    """Evaluate aggregate-implied laws without any target microscopic trajectory."""

    def __init__(self, cfg: dict[str, Any], data: TargetProspectiveData, rollout_path: str | Path):
        self.cfg = cfg
        self.data = data
        with np.load(rollout_path, allow_pickle=False) as ref:
            role = str(np.asarray(ref["role"]).item())
            if role != "frozen_endpoint_only_reference_rollout":
                raise ValueError("selection requires the frozen endpoint-only reference rollout")
            self.times = jnp.asarray(ref["times"], dtype=jnp.float64)
            self.nodes = jnp.asarray(ref["nodes"], dtype=jnp.float64)
            self.velocity = jnp.asarray(ref["velocity"], dtype=jnp.float64)
            self.base_weights = jnp.asarray(ref["weights"], dtype=jnp.float64)
        if not np.allclose(np.asarray(self.times), data.times):
            raise ValueError("prospective aggregate and reference time grids differ")
        self.time_weights = jnp.asarray(trap_weights(np.asarray(self.times)), dtype=jnp.float64)
        self.acq_idx = jnp.asarray(
            nested_indices(len(self.times), int(cfg["time"]["acquisition_nodes"])), dtype=jnp.int32
        )
        m = cfg["measurement"]
        self.sensors = GaussianPointSensors2D(width=float(m["sensor_width"]), n_sensors=int(m["n_sensors"]))
        r = cfg["moment_reconstruction"]
        self.reconstructor = AnchoredCubicSplineReconstructor(
            self.times[self.acq_idx], self.times,
            AnchoredCubicSplineConfig(
                internal_knots=int(r["internal_knots"]), smoothing=float(r["smoothing"]),
                ridge_rel=float(r["ridge_rel"]), roughness_quadrature_order=int(r["roughness_quadrature_order"]),
            ),
        )
        p = cfg["projection"]
        self.projector = EmpiricalIProjector(
            IProjectionConfig(
                max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
                newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
                lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
            ),
            trajectory_backend=str(p["backend"]),
        )
        po = cfg["poisson"]
        self.grid = RectangularGrid2D(0.0, 2.0, 0.0, 1.0, int(po["grid_nx"]), int(po["grid_ny"]))
        self.poisson_cfg = PoissonConfig(
            dx=self.grid.require_isotropic_spacing(), operator_floor_rel=0.0,
            cg_tol=float(po["cg_tol"]), cg_maxiter=int(po["cg_maxiter"]), gauge_strength=1.0,
        )
        self.raster_cfg = RasterConfig()
        self.reference_qois = qoi_features(self.nodes)

    def prospective_population(self, eta):
        centers = self.sensors.centers(jnp.asarray(eta, dtype=jnp.float64))
        return self.data.response(centers), self.data.response_second(centers)

    def reconstruct(self, response_mean, response_second, bank: AggregateObservationBank):
        mean = jnp.asarray(response_mean, dtype=jnp.float64)
        second = jnp.asarray(response_second, dtype=jnp.float64)
        acq_mean = mean[self.acq_idx]
        acq_second = second[self.acq_idx]
        variance = jnp.maximum(acq_second - acq_mean * acq_mean, 0.0)
        finite_se = jnp.sqrt(variance / float(self.cfg["measurement"]["finite_n"]))
        y = (
            acq_mean[None, :, :]
            + finite_se[None, :, :] * jnp.asarray(bank.sampling_z)
            + float(self.cfg["measurement"]["noise_std"]) * jnp.asarray(bank.detector_z)
        )
        endpoint = (self.acq_idx == 0) | (self.acq_idx == len(self.times) - 1)
        y = jnp.where(endpoint[None, :, None], acq_mean[None, :, :], y)
        margin = float(self.cfg["moment_reconstruction"]["clip_margin"])
        y = jnp.clip(y, margin, 1.0 - margin)

        def one(observations):
            fit = self.reconstructor.reconstruct(observations, acq_mean[0], acq_mean[-1])
            inside = (fit.c > margin) & (fit.c < 1.0 - margin)
            return jnp.clip(fit.c, margin, 1.0 - margin), jnp.where(inside, fit.c_dot, 0.0), fit.residual_sum_squares

        return jax.vmap(one)(y)

    def _project(self, eta, response_mean, response_second, bank):
        eta = jnp.asarray(eta, dtype=jnp.float64)
        targets, targets_dot, spline_rss = self.reconstruct(response_mean, response_second, bank)
        phi = self.sensors.features(self.nodes, eta)
        grad = self.sensors.feature_gradients(self.nodes, eta)
        projection = self.projector.project_trajectory(phi, self.base_weights, targets)
        weights = projection.weights
        advective = jnp.einsum("tnmd,tnd->tnm", grad, self.velocity)
        mean_advective = jnp.einsum("btn,tnm->btm", weights, advective)
        g = jnp.einsum("tnm,btm->btn", advective, projection.lam)
        mean_g = jnp.einsum("btn,btn->bt", weights, g)
        centered = phi[None, :, :, :] - projection.moments[:, :, None, :]
        cov_phi_g = jnp.einsum("btn,btnm,btn->btm", weights, centered, g - mean_g[:, :, None])
        covariance = projection.covariance
        ridge = float(self.cfg["particle_mfsi"]["covariance_ridge"])
        eye = jnp.eye(phi.shape[-1], dtype=jnp.float64)
        lambda_dot = jnp.linalg.solve(
            covariance + ridge * eye,
            (targets_dot - mean_advective - cov_phi_g)[..., None],
        )[..., 0]
        forcing = jnp.einsum("btnm,btm->btn", centered, lambda_dot) + g - mean_g[:, :, None]
        forcing -= jnp.einsum("btn,btn->bt", weights, forcing)[:, :, None]
        tangent_gram = jnp.einsum("btn,tnmd,tnkd->btmk", weights, grad, grad)
        tangent_residual = mean_advective - targets_dot
        tangent_coeff = jnp.linalg.solve(tangent_gram + ridge * eye, tangent_residual[..., None])[..., 0]
        tangent = jnp.einsum("btm,btm->bt", tangent_residual, tangent_coeff)
        return projection, weights, forcing, tangent, spline_rss

    def evaluate_population(
        self,
        eta,
        response_mean,
        response_second,
        qoi_targets,
        bank: AggregateObservationBank,
        *,
        compute_full: bool,
    ) -> dict[str, Any]:
        projection, weights, forcing, tangent, spline_rss = self._project(
            eta, response_mean, response_second, bank
        )
        projected_qoi = jnp.einsum("btn,tnk->btk", weights, self.reference_qois)
        qoi_error = (projected_qoi - jnp.asarray(qoi_targets)[None, :, :]) / jnp.asarray(self.data.qoi_scales)[None, None, :]
        risk_by_trial = jnp.sum(self.time_weights[None, :, None] * qoi_error * qoi_error, axis=(1, 2))
        tangent_by_trial = jnp.sum(self.time_weights[None, :] * tangent, axis=1)
        residual = jnp.linalg.norm(projection.residual, axis=-1)
        max_residual = np.asarray(jnp.max(residual, axis=1), dtype=np.float64)
        min_ess = np.asarray(jnp.min(projection.ess_fraction, axis=1), dtype=np.float64)
        covariance_eigs = np.linalg.eigvalsh(np.asarray(projection.covariance))
        min_covariance_eigenvalue = np.min(covariance_eigs, axis=(1, 2))
        valid = (
            np.isfinite(np.asarray(risk_by_trial))
            & (max_residual <= float(self.cfg["validity"]["max_projection_residual"]))
            & (min_ess >= float(self.cfg["validity"]["min_ess_fraction"]))
            & (min_covariance_eigenvalue > 0.0)
        )
        full_by_trial = np.full(bank.trials, np.nan, dtype=np.float64)
        poisson_residual = np.full(bank.trials, np.nan, dtype=np.float64)
        compatibility = np.full(bank.trials, np.nan, dtype=np.float64)
        full_moment_residual = np.full(bank.trials, np.nan, dtype=np.float64)
        solver_converged = np.zeros(bank.trials, dtype=bool)
        if compute_full:
            def raster_trial(w_trial, f_trial):
                return jax.vmap(
                    lambda x, w, f: rasterize_projected_particles_rect(x, w, f, self.grid, self.raster_cfg)
                )(self.nodes, w_trial, f_trial)

            rasters = jax.vmap(raster_trial)(weights, forcing)
            leading = rasters.q.shape[:2]
            solved = solve_weighted_poisson_physical_direct_batch(
                np.asarray(rasters.q).reshape((-1,) + self.grid.shape),
                np.asarray(rasters.h).reshape((-1,) + self.grid.shape),
                self.poisson_cfg,
            )
            actions = np.asarray(solved.action).reshape(leading)
            full_by_trial = np.sum(actions * np.asarray(self.time_weights)[None, :], axis=1)
            poisson_by_time = np.asarray(solved.relative_residual).reshape(leading)
            compatibility_by_time = np.asarray(solved.maximum_component_compatibility_residual).reshape(leading)
            converged_by_time = np.asarray(solved.solver_converged).reshape(leading)
            poisson_residual = np.max(poisson_by_time, axis=1)
            compatibility = np.max(compatibility_by_time, axis=1)
            solver_converged = np.all(converged_by_time, axis=1)
            potentials = jnp.asarray(solved.potential).reshape(rasters.q.shape)
            grid_features = self.sensors.features(self.grid.points(), jnp.asarray(eta))
            decomposition = raster_tangent_projection(
                potentials, rasters.q, rasters.h, grid_features,
                dx=float(self.poisson_cfg.dx), cell_area=float(self.grid.cell_area),
                pinv_rcond=1.0e-10, operator_floor_rel=0.0, gauge_strength=0.0,
            )
            full_moment_residual = np.max(
                np.linalg.norm(np.asarray(decomposition.full_moment_residual), axis=-1), axis=1
            )
            valid &= (
                np.isfinite(full_by_trial)
                & (poisson_residual <= float(self.cfg["validity"]["max_poisson_relative_residual"]))
                & (compatibility <= 1.0e-10)
                & solver_converged
                & (full_moment_residual <= 1.0e-5)
            )

        risk_np = np.where(valid, np.asarray(risk_by_trial), np.nan)
        tangent_np = np.where(valid, np.asarray(tangent_by_trial), np.nan)
        full_np = np.where(valid, full_by_trial, np.nan)
        rows = []
        for trial in range(bank.trials):
            rows.append({
                "trial": trial,
                "valid": bool(valid[trial]),
                "scientific_risk": float(risk_np[trial]) if np.isfinite(risk_np[trial]) else None,
                "tangent_proxy": float(tangent_np[trial]) if np.isfinite(tangent_np[trial]) else None,
                "full_action": float(full_np[trial]) if np.isfinite(full_np[trial]) else None,
                "max_projection_residual": float(max_residual[trial]),
                "min_ess_fraction": float(min_ess[trial]),
                "min_covariance_eigenvalue": float(min_covariance_eigenvalue[trial]),
                "max_poisson_relative_residual": float(poisson_residual[trial]) if compute_full else None,
                "max_component_compatibility_residual": float(compatibility[trial]) if compute_full else None,
                "max_full_moment_rate_residual": float(full_moment_residual[trial]) if compute_full else None,
                "full_solver_converged": bool(solver_converged[trial]) if compute_full else None,
                "spline_residual_sum_squares": float(np.asarray(spline_rss)[trial]),
            })
        return {
            "valid": bool(np.all(valid)),
            "valid_fraction": float(np.mean(valid)),
            "risk": _summarize(risk_np),
            "tangent_proxy": _summarize(tangent_np),
            "full_action": _summarize(full_np),
            "trials": rows,
        }

    def evaluate_prospective(self, eta, bank: AggregateObservationBank, *, compute_full: bool):
        mean, second = self.prospective_population(eta)
        return self.evaluate_population(
            eta, mean, second, self.data.scientific_qoi_predictions, bank, compute_full=compute_full
        )

    def evaluate_full_proxy(self, eta, bank: AggregateObservationBank) -> dict[str, Any]:
        """Reduced-trial/time/grid Full score used only to form a shortlist."""
        trial_n = min(int(self.cfg["search"]["full_proxy_trials"]), bank.trials)
        proxy_bank = AggregateObservationBank(
            np.asarray(bank.sampling_z[:trial_n]), np.asarray(bank.detector_z[:trial_n])
        )
        mean, second = self.prospective_population(eta)
        projection, weights, forcing, _, _ = self._project(eta, mean, second, proxy_bank)
        time_n = min(int(self.cfg["search"]["full_proxy_time_nodes"]), len(self.times))
        time_idx = np.unique(
            np.rint(np.linspace(0, len(self.times) - 1, time_n)).astype(np.int32)
        )
        grid = RectangularGrid2D(
            0.0, 2.0, 0.0, 1.0,
            int(self.cfg["search"]["full_proxy_grid_nx"]),
            int(self.cfg["search"]["full_proxy_grid_ny"]),
        )
        proxy_cfg = PoissonConfig(
            dx=grid.require_isotropic_spacing(), operator_floor_rel=0.0,
            cg_tol=max(float(self.poisson_cfg.cg_tol), 1.0e-6),
            cg_maxiter=int(self.poisson_cfg.cg_maxiter), gauge_strength=1.0,
        )
        nodes = self.nodes[jnp.asarray(time_idx)]

        def raster_trial(w_trial, f_trial):
            return jax.vmap(
                lambda x, w, f: rasterize_projected_particles_rect(x, w, f, grid, self.raster_cfg)
            )(nodes, w_trial[time_idx], f_trial[time_idx])

        rasters = jax.vmap(raster_trial)(weights, forcing)
        leading = rasters.q.shape[:2]
        solved = solve_weighted_poisson_physical_direct_batch(
            np.asarray(rasters.q).reshape((-1,) + grid.shape),
            np.asarray(rasters.h).reshape((-1,) + grid.shape),
            proxy_cfg,
        )
        actions = np.asarray(solved.action, dtype=np.float64).reshape(leading)
        weights_time = trap_weights(np.asarray(self.times)[time_idx])
        trial_values = np.sum(actions * weights_time[None, :], axis=1)
        max_projection = np.max(
            np.linalg.norm(np.asarray(projection.residual), axis=-1), axis=1
        )
        min_ess = np.min(np.asarray(projection.ess_fraction), axis=1)
        poisson = np.max(
            np.asarray(solved.relative_residual).reshape(leading), axis=1
        )
        converged = np.all(
            np.asarray(solved.solver_converged).reshape(leading), axis=1
        )
        valid = (
            np.isfinite(trial_values)
            & (max_projection <= float(self.cfg["validity"]["max_projection_residual"]))
            & (min_ess >= float(self.cfg["validity"]["min_ess_fraction"]))
            & (poisson <= max(float(self.cfg["validity"]["max_poisson_relative_residual"]), 1.0e-6))
            & converged
        )
        return {
            "valid": bool(np.all(valid)),
            "full_proxy": _summarize(np.where(valid, trial_values, np.nan)),
            "time_indices": time_idx.tolist(),
            "trials": trial_n,
            "grid": [grid.nx, grid.ny],
            "max_projection_residual": float(np.max(max_projection)),
            "min_ess_fraction": float(np.min(min_ess)),
            "max_poisson_relative_residual": float(np.max(poisson)),
        }


__all__ = [
    "AggregateObservationBank",
    "ProspectiveEvaluator",
    "ensure_observation_bank",
    "make_observation_bank",
]
