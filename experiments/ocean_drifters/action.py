"""Dense-time multiplier and tangent-action readiness for ocean drifters."""

from __future__ import annotations

import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial import ConvexHull
from scipy.special import logsumexp

from mfsi.cache import file_sha256, fingerprint, load_npz_cache, save_npz_cache, write_json_atomic
from mfsi.projection import IProjectionConfig
from mfsi.projection_tesseract import solve_i_projection_trajectory_tesseract_forward
from mfsi.reference_density import (
    backward_latent_with_log_density_correction,
    logistic_log_abs_det_jacobian,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _grid_points(bounds: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, float, float]:
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    dx = (xmax - xmin) / nx; dy = (ymax - ymin) / ny
    x = xmin + (np.arange(nx) + 0.5) * dx
    y = ymin + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel()), axis=-1), dx, dy


def _features(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    delta = points[:, None] - centers[None]
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2)


def _positive_kernel_reconstruct(
    raw: np.ndarray, observation_days: np.ndarray, evaluation_days: np.ndarray, bandwidth_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convex temporal reconstruction and its analytic time derivative."""
    delta = evaluation_days[:, None] - observation_days[None]
    logits = -0.5 * (delta / float(bandwidth_days)) ** 2
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits); weights /= weights.sum(axis=1, keepdims=True)
    weighted_observation_day = weights @ observation_days
    weight_dot = weights * (
        observation_days[None] - weighted_observation_day[:, None]
    ) / float(bandwidth_days) ** 2
    reconstructed = np.einsum("tk,dkm->dtm", weights, raw)
    derivative_per_day = np.einsum("tk,dkm->dtm", weight_dot, raw)
    return reconstructed, derivative_per_day, weights


def _minimum_simplex_residual(phi: np.ndarray, target: np.ndarray) -> float:
    particle_n, moment_n = phi.shape
    objective = np.r_[np.zeros(particle_n), 1.0]
    upper = np.vstack([
        np.c_[phi.T, -np.ones(moment_n)],
        np.c_[-phi.T, -np.ones(moment_n)],
    ])
    result = linprog(
        objective, A_ub=upper, b_ub=np.r_[target, -target],
        A_eq=np.c_[np.ones((1, particle_n)), np.zeros((1, 1))],
        b_eq=np.asarray([1.0]), bounds=[(0.0, None)] * particle_n + [(0.0, None)],
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"domain-grid simplex residual LP failed: {result.message}")
    achieved = result.x[:-1] @ phi
    return float(np.max(np.abs(achieved - target)))


def _log_kde(points: np.ndarray, atoms: np.ndarray, bandwidth: np.ndarray, chunk: int = 4096) -> np.ndarray:
    inverse = np.linalg.inv(bandwidth)
    log_norm = -math.log(2.0 * math.pi) - 0.5 * np.linalg.slogdet(bandwidth)[1] - math.log(len(atoms))
    output = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk):
        delta = points[start:start + chunk, None] - atoms[None]
        quadratic = np.einsum("nki,ij,nkj->nk", delta, inverse, delta)
        output[start:start + chunk] = logsumexp(-0.5 * quadratic, axis=1) + log_norm
    return output


def _summary(phi: np.ndarray, log_base: np.ndarray, target: np.ndarray, lam: np.ndarray) -> dict[str, Any]:
    logits = log_base + phi @ lam
    log_z = float(logsumexp(logits))
    weights = np.exp(logits - log_z)
    moments = weights @ phi
    residual = moments - target
    centered = phi - moments
    covariance = centered.T @ (weights[:, None] * centered)
    log_ratio = phi @ lam - log_z
    kl = float(weights @ log_ratio)
    log_ess = float(-logsumexp(2.0 * (np.log(np.maximum(weights, 1e-300)) - log_base)))
    eig = np.linalg.eigvalsh(covariance)
    return {
        "weights": weights,
        "moments": moments,
        "residual_norm": float(np.linalg.norm(residual)),
        "covariance": covariance,
        "covariance_eigenvalues": eig,
        "covariance_condition": float(eig[-1] / max(eig[0], 1e-300)),
        "kl": kl,
        "log10_intrinsic_ess": log_ess / math.log(10.0),
    }


def _trust_solve(
    phi: np.ndarray, log_base: np.ndarray, target: np.ndarray, initial: np.ndarray,
    acceptance: float, maximum_evaluations: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    last: dict[str, Any] = {}
    best_norm = math.inf
    best_lambda = np.asarray(initial, dtype=np.float64).copy()
    evaluations = 0

    class AcceptedResidual(Exception):
        pass

    def fun(lam: np.ndarray) -> np.ndarray:
        nonlocal last, best_norm, best_lambda, evaluations
        evaluations += 1
        last = _summary(phi, log_base, target, lam)
        residual = last["moments"] - target
        norm = float(np.linalg.norm(residual))
        if norm < best_norm:
            best_norm = norm; best_lambda = np.asarray(lam).copy()
        if norm <= acceptance:
            raise AcceptedResidual
        return residual

    def jac(lam: np.ndarray) -> np.ndarray:
        nonlocal last
        last = _summary(phi, log_base, target, lam)
        return last["covariance"]

    try:
        fit = least_squares(
            fun, np.asarray(initial, dtype=np.float64), jac=jac,
            bounds=(-1e12, 1e12), max_nfev=int(maximum_evaluations),
            x_scale="jac",
            xtol=1e-14, ftol=1e-14, gtol=1e-14,
        )
        chosen = np.asarray(fit.x) if float(np.linalg.norm(fun(fit.x))) < best_norm else best_lambda
    except AcceptedResidual:
        chosen = best_lambda
    result = _summary(phi, log_base, target, chosen)
    return chosen, result, evaluations


class OceanActionReadiness:
    def __init__(self, experiment, analysis_dir: Path, output_dir: Path):
        self.experiment = experiment
        self.cfg = experiment.cfg["action"]
        self.analysis = Path(analysis_dir)
        self.output = Path(output_dir)
        self.table_dir = self.analysis / "tables"
        self.figure_dir = self.analysis / "figures/action_readiness"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        with np.load(experiment.paths["dense_moments"], allow_pickle=False) as data:
            if bool(data["final_test_accessed"]):
                raise RuntimeError("dense moments report final-test access")
            self.designs = np.asarray(data["design_indices"], dtype=int)
            self.times = np.asarray(data["normalized_times"], dtype=np.float64)
            self.raw = np.asarray(data["raw_moments"], dtype=np.float64)
            self.failed_spline_targets = np.asarray(data["smoothed_moments"], dtype=np.float64)
            self.failed_spline_derivative = np.asarray(data["moment_derivative"], dtype=np.float64)
        freeze = json.loads(experiment.paths["risk_freeze"].read_text(encoding="utf-8"))
        expected = [int(value.split("_")[-1]) for value in freeze["near_optimal_design_ids"]]
        if self.designs.tolist() != expected:
            raise RuntimeError("dense moments do not match the frozen 68-layout risk set")
        expected_time_count = int(experiment.cfg["moment_reconstruction"]["action_time_grid_count"])
        if len(self.times) != expected_time_count:
            raise RuntimeError(
                f"dense moments contain {len(self.times)} times, expected {expected_time_count}"
            )
        self._build_positive_kernel_reconstruction()

    def _build_positive_kernel_reconstruction(self) -> None:
        cfg = self.experiment.cfg["moment_reconstruction"]
        observation_days = self.times * 45.0
        candidates = [float(value) for value in cfg["bandwidth_candidates_days"]]
        selectable = {float(value) for value in cfg["nominal_selection_candidates_days"]}
        exclusion = float(cfg["leave_time_window_out_days"])
        scale = np.maximum(np.var(self.raw, axis=1, ddof=1), 1e-12)
        cv_rows: list[dict[str, Any]] = []
        for bandwidth in candidates:
            delta = observation_days[:, None] - observation_days[None]
            allowed = np.abs(delta) > exclusion + 1e-12
            logits = -0.5 * (delta / bandwidth) ** 2
            logits = np.where(allowed, logits, -np.inf)
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits); weights /= weights.sum(axis=1, keepdims=True)
            prediction = np.einsum("tk,dkm->dtm", weights, self.raw)
            normalized_error = (prediction - self.raw) ** 2 / scale[:, None]
            cv_rows.append({
                "bandwidth_days": bandwidth,
                "eligible_for_nominal_selection": bandwidth in selectable,
                "blocked_window_half_width_days": exclusion,
                "aggregate_normalized_mse": float(np.mean(normalized_error)),
                "median_layout_normalized_mse": float(np.median(np.mean(normalized_error, axis=(1, 2)))),
                "maximum_layout_normalized_mse": float(np.max(np.mean(normalized_error, axis=(1, 2)))),
            })
        eligible_rows = [row for row in cv_rows if row["eligible_for_nominal_selection"]]
        selected = min(eligible_rows, key=lambda row: row["aggregate_normalized_mse"])
        nominal = float(selected["bandwidth_days"])
        position = candidates.index(nominal)
        if position == 0 or position == len(candidates) - 1:
            raise RuntimeError("selected positive-kernel bandwidth lacks two predeclared neighbors")
        low, high = candidates[position - 1], candidates[position + 1]
        self.bandwidths = {"low": low, "nominal": nominal, "high": high}
        self.sensitivity_targets: dict[str, np.ndarray] = {}
        self.sensitivity_target_dot: dict[str, np.ndarray] = {}
        for label, bandwidth in self.bandwidths.items():
            values, derivative_day, _ = _positive_kernel_reconstruct(
                self.raw, observation_days, observation_days, bandwidth
            )
            self.sensitivity_targets[label] = values
            self.sensitivity_target_dot[label] = derivative_day * 45.0
        self.targets = self.sensitivity_targets["nominal"]
        self.target_dot = self.sensitivity_target_dot["nominal"]
        refined_days = np.linspace(0.0, 45.0, int(cfg["refined_audit_time_grid_count"]))
        refined_values, refined_derivative_day, _ = _positive_kernel_reconstruct(
            self.raw, observation_days, refined_days, nominal
        )
        self.refined_days = refined_days
        self.refined_targets = refined_values
        self.refined_target_dot = refined_derivative_day * 45.0
        _write_csv(self.table_dir / "positive_kernel_bandwidth_selection.csv", cv_rows)
        spline_negative = self.failed_spline_targets < 0.0
        write_json_atomic(self.table_dir / "failed_spline_reconstruction_provenance.json", {
            "backend": "endpoint-anchored unconstrained cubic B-spline",
            "classification": "failed action-readiness reconstruction method",
            "negative_component_count": int(np.sum(spline_negative)),
            "affected_layout_count": int(len(np.unique(np.where(spline_negative)[0]))),
            "affected_time_count": int(len(np.unique(np.where(spline_negative)[1]))),
            "minimum_moment": float(self.failed_spline_targets.min()),
            "layouts_were_not_classified_action_invalid": True,
            "artifact_preserved": str(self.experiment.paths["dense_moments"]),
            "final_test_accessed": False,
        })
        signature = fingerprint({
            "schema": 1, "raw_source_sha256": file_sha256(self.experiment.paths["dense_moments"]),
            "config": cfg, "selected_bandwidth_days": nominal,
        })
        save_npz_cache(
            self.experiment._resolve("experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"),
            {
                "design_indices": self.designs, "normalized_times": self.times,
                "raw_moments": self.raw, "smoothed_moments": self.targets,
                "moment_derivative": self.target_dot,
                "low_moments": self.sensitivity_targets["low"],
                "low_moment_derivative": self.sensitivity_target_dot["low"],
                "high_moments": self.sensitivity_targets["high"],
                "high_moment_derivative": self.sensitivity_target_dot["high"],
            },
            signature=signature,
            metadata={"bandwidths_days": self.bandwidths, "final_test_accessed": False},
        )
        self._write_reconstruction_diagnostics()

    def _write_reconstruction_diagnostics(self) -> None:
        rows = []
        coarse_energy = np.trapezoid(np.sum((self.target_dot / 45.0) ** 2, axis=-1), self.times * 45.0, axis=1)
        refined_energy = np.trapezoid(
            np.sum((self.refined_target_dot / 45.0) ** 2, axis=-1), self.refined_days, axis=1
        )
        for local, design in enumerate(self.designs):
            error = self.targets[local] - self.raw[local]
            rows.append({
                "design_index": int(design), "design_id": self.experiment.sensor_bank.design_ids[design],
                "bandwidth_low_days": self.bandwidths["low"],
                "bandwidth_nominal_days": self.bandwidths["nominal"],
                "bandwidth_high_days": self.bandwidths["high"],
                "observation_time_rmse": float(np.sqrt(np.mean(error**2))),
                "observation_time_maximum_absolute_error": float(np.max(np.abs(error))),
                "minimum_reconstructed_moment": float(self.refined_targets[local].min()),
                "maximum_reconstructed_moment": float(self.refined_targets[local].max()),
                "maximum_absolute_derivative_per_normalized_time": float(np.max(np.abs(self.refined_target_dot[local]))),
                "derivative_all_finite": bool(np.isfinite(self.refined_target_dot[local]).all()),
                "coarse_derivative_energy_per_day": float(coarse_energy[local]),
                "refined_derivative_energy_per_day": float(refined_energy[local]),
                "temporal_grid_refinement_relative_energy_change": float(
                    abs(refined_energy[local] - coarse_energy[local]) / max(abs(refined_energy[local]), 1e-14)
                ),
            })
        _write_csv(self.table_dir / "positive_kernel_reconstruction_diagnostics.csv", rows)

    def target_preflight(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        tolerance = float(self.cfg["target_range_tolerance"])
        range_valid = np.all(
            (self.refined_targets >= -tolerance) & (self.refined_targets <= 1.0 + tolerance), axis=(1, 2)
        )
        feasibility = self._joint_feasibility_audit()
        rows = []
        for local, design in enumerate(self.designs):
            feasible = feasibility[int(design)]
            accepted = bool(range_valid[local] and feasible["joint_feasible"])
            rows.append({
                "design_index": int(design),
                "design_id": self.experiment.sensor_bank.design_ids[design],
                "dense_target_range_valid": bool(range_valid[local]),
                "dense_target_joint_feasible": feasible["joint_feasible"],
                "domain_grid_hull_maximum_violation": feasible["maximum_hull_violation"],
                "domain_grid_worst_target_lp_linf_residual": feasible["worst_target_lp_linf_residual"],
                "domain_grid_resolution": feasible["grid_resolution"],
                "minimum_smoothed_moment": float(self.refined_targets[local].min()),
                "maximum_smoothed_moment": float(self.refined_targets[local].max()),
                "minimum_raw_moment": float(self.raw[local].min()),
                "classification": "pending_dense_projection" if accepted else "reconstruction_feasibility_failure",
                "failure_reason": "" if accepted else "positive-kernel reconstruction failed its predeclared range/domain-grid feasibility audit",
            })
        _write_csv(self.table_dir / "dense_target_admissibility.csv", rows)
        self._plot_splines(range_valid)
        return np.asarray([
            row["dense_target_range_valid"] and row["dense_target_joint_feasible"] for row in rows
        ]), rows

    def _joint_feasibility_audit(self) -> dict[int, dict[str, Any]]:
        cfg = self.experiment.cfg["moment_reconstruction"]
        tolerance = float(cfg["joint_feasibility_tolerance"])
        bounds = np.asarray(self.experiment.cfg["scientific"]["domain_km"], dtype=np.float64)
        resolutions = [
            tuple(int(value) for value in cfg["joint_feasibility_grid_resolution"]),
            tuple(int(value) for value in cfg["joint_feasibility_refinement_resolution"]),
        ]
        output: dict[int, dict[str, Any]] = {}
        audit_rows: list[dict[str, Any]] = []
        for ordinal, design in enumerate(self.designs, start=1):
            local = ordinal - 1
            final: dict[str, Any] | None = None
            for resolution in resolutions:
                points, _, _ = _grid_points(bounds, *resolution)
                phi = _features(
                    points, self.experiment.sensor_bank.centers_km[design],
                    self.experiment.sensor_bank.sigma_km,
                )
                hull = ConvexHull(phi, qhull_options="QJ")
                violation_by_target = np.max(
                    self.refined_targets[local] @ hull.equations[:, :-1].T
                    + hull.equations[:, -1][None], axis=1,
                )
                worst_index = int(np.argmax(violation_by_target))
                lp_residual = _minimum_simplex_residual(phi, self.refined_targets[local, worst_index])
                final = {
                    "joint_feasible": bool(lp_residual <= tolerance),
                    "maximum_hull_violation": float(violation_by_target[worst_index]),
                    "worst_target_lp_linf_residual": lp_residual,
                    "worst_target_day": float(self.refined_days[worst_index]),
                    "grid_resolution": f"{resolution[0]}x{resolution[1]}",
                }
                audit_rows.append({
                    "design_index": int(design),
                    "design_id": self.experiment.sensor_bank.design_ids[design],
                    **final,
                })
                if final["joint_feasible"]:
                    break
            assert final is not None
            output[int(design)] = final
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(self.designs):
                print(f"[ocean action] moment-hull audit {ordinal}/{len(self.designs)}", flush=True)
        _write_csv(self.table_dir / "positive_kernel_joint_feasibility.csv", audit_rows)
        return output

    def _plot_splines(self, range_valid: np.ndarray) -> None:
        examples = [0, int(np.argmin(self.failed_spline_targets.min(axis=(1, 2))))]
        out = self.figure_dir / "moment_splines"; out.mkdir(parents=True, exist_ok=True)
        for local in examples:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
            for component, axis in enumerate(axes.ravel()):
                axis.plot(self.times * 45.0, self.raw[local, :, component], color="#9ecae1", linewidth=1, label="raw inference moment")
                axis.plot(self.times * 45.0, self.targets[local, :, component], color="#08519c", linewidth=1.4, label="positive-kernel reconstruction")
                axis.plot(self.times * 45.0, self.failed_spline_targets[local, :, component], color="#c53030", linewidth=0.8, linestyle="--", alpha=0.7, label="failed cubic spline")
                twin = axis.twinx(); twin.plot(self.times * 45.0, self.target_dot[local, :, component], color="#d95f02", alpha=0.55, linewidth=0.8, label="derivative")
                axis.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
                axis.set_title(f"sensor {component + 1}"); axis.grid(alpha=0.18)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.supxlabel("day"); fig.supylabel("Gaussian moment")
            design = int(self.designs[local]); fig.suptitle(f"{self.experiment.sensor_bank.design_ids[design]} range_valid={bool(range_valid[local])}")
            fig.savefig(out / f"{self.experiment.sensor_bank.design_ids[design]}.png", dpi=190); plt.close(fig)

    def dense_reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny = (int(value) for value in self.cfg["dense_grid_resolution"])
        bounds = np.asarray(self.experiment.cfg["scientific"]["domain_km"], dtype=np.float64)
        points, dx, dy = _grid_points(bounds, nx, ny)
        flow = self.experiment.reference()
        with np.load(self.experiment.paths["conditioned_endpoint_estimator"], allow_pickle=False) as data:
            atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
            bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
        normalization_rows = _read_csv(self.experiment.paths["conditioned_kde_normalization"])
        z0 = float(next(row["Z_hat"] for row in normalization_rows if row["endpoint"] == "day0"))
        signature_base = {
            "schema": 1, "grid": [nx, ny], "reference": file_sha256(self.experiment.paths["reference_checkpoint"]),
            "endpoint": file_sha256(self.experiment.paths["conditioned_endpoint_estimator"]),
            "normalization": z0,
        }
        cache_dir = self.experiment._resolve("experiments/ocean_drifters/cache/action_reference_256x136")
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_base_all = np.empty((len(self.times), len(points)), dtype=np.float64)
        velocity_all = np.empty((len(self.times), len(points), 2), dtype=np.float64)
        zt = flow.to_latent(jnp.asarray(points))
        velocity_fn = jax.jit(lambda value: flow.velocity(jnp.asarray(points), value))

        @lru_cache(maxsize=None)
        def backward_fn(steps: int):
            return jax.jit(lambda z, value: backward_latent_with_log_density_correction(
                flow.params, z, value, steps=steps
            ))

        for source, value in enumerate(self.times):
            signature = fingerprint({**signature_base, "source": source, "time": float(value)})
            cache = cache_dir / f"reference_t{source:03d}.npz"
            loaded = load_npz_cache(cache, signature=signature)
            if loaded is not None:
                arrays, _ = loaded
                log_base_all[source] = arrays["log_base_mass"]
                velocity_all[source] = arrays["velocity"]
                continue
            steps = max(1, int(math.ceil(source / 5.0)))
            backward = backward_fn(steps)
            log_density = np.empty(len(points), dtype=np.float64)
            for start in range(0, len(points), 4096):
                stop = min(start + 4096, len(points))
                local_z = zt[start:stop]
                initial_z, correction = backward(local_z, jnp.asarray(value))
                initial_x = np.asarray(flow.to_physical(initial_z))
                initial_log = _log_kde(initial_x, atoms, bandwidth) - math.log(z0)
                log_density[start:stop] = (
                    initial_log + np.asarray(logistic_log_abs_det_jacobian(initial_z, bounds))
                    + np.asarray(correction) - np.asarray(logistic_log_abs_det_jacobian(local_z, bounds))
                )
            log_mass = log_density + math.log(dx * dy)
            log_base = log_mass - logsumexp(log_mass)
            velocity = np.asarray(velocity_fn(jnp.asarray(value)))
            log_base_all[source] = log_base; velocity_all[source] = velocity
            save_npz_cache(
                cache, {"log_base_mass": log_base, "velocity": velocity}, signature=signature,
                metadata={"source_time_index": source, "final_test_accessed": False},
            )
            if source == 0 or (source + 1) % 10 == 0 or source == len(self.times) - 1:
                print(f"[ocean action] continuous reference {source + 1}/{len(self.times)}", flush=True)
        return points, log_base_all, velocity_all

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        range_valid, preflight_rows = self.target_preflight()
        valid_locals = np.flatnonzero(range_valid)
        if not len(valid_locals):
            return self._finish(preflight_rows, [], started)
        points, log_base_all, velocity_all = self.dense_reference()
        p = self.cfg["dense_projection_solver"]
        solver = IProjectionConfig(
            max_steps=int(p["max_steps"]), residual_tol=float(p["residual_tol"]),
            newton_ridge=float(p["newton_ridge"]), step_cap=float(p["step_cap"]),
            lambda_clip=float(p["lambda_clip"]), line_search_steps=int(p["line_search_steps"]),
            implicit_ridge=0.0,
        )
        time_rows: list[dict[str, Any]] = []
        design_results: dict[int, dict[str, Any]] = {}
        sensitivity_rows: list[dict[str, Any]] = []
        for ordinal, local in enumerate(valid_locals, start=1):
            design = int(self.designs[local])
            result, rows, nominal_arrays = self._one_design(
                design, self.targets[local], self.target_dot[local], points,
                log_base_all, velocity_all, solver, bandwidth_label="nominal",
            )
            if result["multiplier_dynamics_valid"] and result["tangent_action_valid"]:
                sensitivity_results: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
                for label in ("low", "high"):
                    alternative, _, alternative_arrays = self._one_design(
                        design, self.sensitivity_targets[label][local],
                        self.sensitivity_target_dot[label][local], points,
                        log_base_all, velocity_all, solver, bandwidth_label=label,
                    )
                    sensitivity_results[label] = (alternative, alternative_arrays)
                sensitivity = self._bandwidth_sensitivity(
                    design, result, nominal_arrays, sensitivity_results
                )
            else:
                sensitivity = {
                    "design_index": design,
                    "design_id": self.experiment.sensor_bank.design_ids[design],
                    "nominal_bandwidth_days": self.bandwidths["nominal"],
                    "low_bandwidth_days": self.bandwidths["low"],
                    "high_bandwidth_days": self.bandwidths["high"],
                    "nominal_tangent_action": result["tangent_action"],
                    "bandwidth_sensitivity_evaluable": False,
                    "bandwidth_sensitivity_valid": False,
                    "not_evaluable_reason": "nominal multiplier/tangent readiness failed",
                }
            sensitivity_rows.append(sensitivity)
            if sensitivity.get("bandwidth_sensitivity_evaluable", True) and not sensitivity["bandwidth_sensitivity_valid"]:
                result["multiplier_dynamics_valid"] = False
                result["tangent_action_valid"] = False
                result["failure_reason"] = (
                    (result["failure_reason"] + "; ") if result["failure_reason"] else ""
                ) + "low/nominal/high bandwidth sensitivity contract failure"
            design_results[design] = result; time_rows.extend(rows)
            print(
                f"[ocean action] {ordinal}/{len(valid_locals)} design={design} "
                f"projection={result['dense_projection_valid']} multiplier={result['multiplier_dynamics_valid']} "
                f"tangent={result['tangent_action_valid']}", flush=True,
            )
        _write_csv(self.table_dir / "action_bandwidth_sensitivity.csv", sensitivity_rows)
        return self._finish(
            preflight_rows, time_rows, started, design_results,
            sensitivity_rows=sensitivity_rows,
        )

    def _bandwidth_sensitivity(
        self, design: int, nominal: dict[str, Any], nominal_arrays: dict[str, np.ndarray],
        alternatives: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "design_index": design, "design_id": self.experiment.sensor_bank.design_ids[design],
            "nominal_bandwidth_days": self.bandwidths["nominal"],
            "nominal_tangent_action": nominal["tangent_action"],
            "bandwidth_sensitivity_evaluable": True,
        }
        valid = bool(nominal["multiplier_dynamics_valid"] and nominal["tangent_action_valid"])
        for label in ("low", "high"):
            result, arrays = alternatives[label]
            lambda_change = float(
                np.linalg.norm(arrays["lambda_value"] - nominal_arrays["lambda_value"])
                / max(np.linalg.norm(nominal_arrays["lambda_value"]), 1e-14)
            )
            finite_dot = np.isfinite(arrays["lambda_dot"]) & np.isfinite(nominal_arrays["lambda_dot"])
            if finite_dot.all():
                lambda_dot_change = float(
                    np.linalg.norm(arrays["lambda_dot"] - nominal_arrays["lambda_dot"])
                    / max(np.linalg.norm(nominal_arrays["lambda_dot"]), 1e-14)
                )
            else:
                lambda_dot_change = math.inf
            if np.isfinite(result["tangent_action"]) and np.isfinite(nominal["tangent_action"]):
                action_change = float(
                    abs(result["tangent_action"] - nominal["tangent_action"])
                    / max(abs(nominal["tangent_action"]), 1e-14)
                )
            else:
                action_change = math.inf
            row.update({
                f"{label}_bandwidth_days": self.bandwidths[label],
                f"{label}_multiplier_dynamics_valid": result["multiplier_dynamics_valid"],
                f"{label}_tangent_action_valid": result["tangent_action_valid"],
                f"{label}_tangent_action": result["tangent_action"],
                f"{label}_relative_lambda_change": lambda_change,
                f"{label}_relative_lambda_dot_change": lambda_dot_change,
                f"{label}_relative_tangent_action_change": action_change,
            })
            valid &= bool(
                result["multiplier_dynamics_valid"] and result["tangent_action_valid"]
                and lambda_dot_change <= float(self.cfg["maximum_relative_lambda_dot_change"])
                and action_change <= float(self.cfg["maximum_relative_tangent_action_change"])
            )
        row["bandwidth_sensitivity_valid"] = valid
        return row

    def _one_design(
        self, design: int, target: np.ndarray, target_dot: np.ndarray,
        points: np.ndarray, log_base: np.ndarray, velocity: np.ndarray,
        solver: IProjectionConfig, *, bandwidth_label: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
        centers = self.experiment.sensor_bank.centers_km[design]
        sigma = self.experiment.sensor_bank.sigma_km
        phi = _features(points, centers, sigma)
        delta = points[:, None] - centers[None]
        gradient = -(delta / sigma**2) * phi[:, :, None]
        phi_time = np.ascontiguousarray(np.broadcast_to(phi, (len(self.times), *phi.shape)))
        native = solve_i_projection_trajectory_tesseract_forward(phi_time, log_base, target[None], solver)
        native_lambda = np.asarray(native["lambda_values"][0], dtype=np.float64)
        lambdas = np.empty_like(native_lambda); lambda_dot = np.full_like(native_lambda, np.nan)
        tangent_density = np.full(len(self.times), np.nan)
        warm = np.zeros(4); rows: list[dict[str, Any]] = []
        projection_all = True; covariance_all = True; compatibility_all = True; ranks = []
        for source in range(len(self.times)):
            lam = native_lambda[source]
            summary = _summary(phi, log_base[source], target[source], lam)
            iterations = int(native["iterations"][0, source]); backend = "tesseract_cpp"
            if summary["residual_norm"] > float(self.cfg["moment_residual_tolerance"]):
                lam, summary, iterations = _trust_solve(
                    phi, log_base[source], target[source], np.zeros(4, dtype=np.float64),
                    float(self.cfg["moment_residual_tolerance"]),
                    int(self.cfg["dense_trust_fallback_maximum_function_evaluations"]),
                )
                backend = "scipy_exact_covariance_fallback"
            warm = lam; lambdas[source] = lam
            projection_ok = summary["residual_norm"] <= float(self.cfg["moment_residual_tolerance"])
            projection_all &= projection_ok
            weights = summary["weights"]
            m_nodes = np.einsum("nmd,nd->nm", gradient, velocity[source])
            expected_m = weights @ m_nodes
            gram = np.einsum("n,nmd,nkd->mk", weights, gradient, gradient)
            gram_eig, gram_vec = np.linalg.eigh(gram)
            threshold = float(self.cfg["gram_relative_rank_tolerance"]) * max(gram_eig[-1], 1e-300)
            retained = gram_eig > threshold; rank = int(retained.sum()); ranks.append(rank)
            r = expected_m - target_dot[source]
            projected_r = gram_vec[:, retained] @ (gram_vec[:, retained].T @ r) if rank else np.zeros_like(r)
            compatibility = float(np.linalg.norm(r - projected_r) / max(np.linalg.norm(r), 1e-14))
            compatible = compatibility <= float(self.cfg["gram_compatibility_relative_tolerance"])
            compatibility_all &= compatible
            if rank and compatible:
                tangent_density[source] = float(np.sum((gram_vec[:, retained].T @ r) ** 2 / gram_eig[retained]))
            covariance = summary["covariance"]; covariance_eig = summary["covariance_eigenvalues"]
            covariance_condition = summary["covariance_condition"]
            covariance_ready = bool(
                covariance_eig[0] >= float(self.cfg["covariance_minimum_eigenvalue"])
                and covariance_condition <= float(self.cfg["covariance_maximum_condition"])
            )
            covariance_all &= covariance_ready
            lambda_m = m_nodes @ lam
            centered_phi = phi - summary["moments"]
            cov_phi_lambda_m = np.einsum(
                "n,ni,n->i", weights, centered_phi, lambda_m - weights @ lambda_m
            )
            rhs = target_dot[source] - expected_m - cov_phi_lambda_m
            solve_residual = math.nan
            if covariance_ready:
                lambda_dot[source] = np.linalg.solve(covariance, rhs)
                solve_residual = float(
                    np.linalg.norm(covariance @ lambda_dot[source] - rhs) / max(np.linalg.norm(rhs), 1e-14)
                )
            rows.append({
                "design_index": design, "design_id": self.experiment.sensor_bank.design_ids[design],
                "bandwidth_label": bandwidth_label,
                "source_time_index": source, "day": float(self.times[source] * 45.0),
                "projection_valid": projection_ok, "projection_backend": backend,
                "moment_residual": summary["residual_norm"], "lambda_norm": float(np.linalg.norm(lam)),
                "KL": summary["kl"], "log10_intrinsic_ESS": summary["log10_intrinsic_ess"],
                "covariance_minimum_eigenvalue": float(covariance_eig[0]),
                "covariance_maximum_eigenvalue": float(covariance_eig[-1]),
                "covariance_condition": covariance_condition,
                "multiplier_coordinate_ready": covariance_ready,
                "lambda_dot_norm": float(np.linalg.norm(lambda_dot[source])) if covariance_ready else math.nan,
                "lambda_dot_solve_relative_residual": solve_residual,
                "gram_rank": rank, "gram_compatibility_relative_residual": compatibility,
                "gram_compatible": compatible, "tangent_action_density": tangent_density[source],
                "iterations": iterations,
            })
        rank_constant = len(set(ranks)) == 1
        tangent_finite = bool(np.isfinite(tangent_density).all())
        multiplier_valid = bool(projection_all and covariance_all)
        tangent_valid = bool(multiplier_valid and compatibility_all and rank_constant and tangent_finite)
        tangent_action = float(np.trapezoid(tangent_density, self.times)) if tangent_valid else math.nan
        cache_dir = self.experiment._resolve("experiments/ocean_drifters/cache/action_projected_laws")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_dir / f"design_{design:06d}_{bandwidth_label}.npz", lambda_value=lambdas,
            lambda_dot=lambda_dot, tangent_action_density=tangent_density,
            final_test_accessed=np.asarray(False),
        )
        design_result = {
            "design_index": design, "design_id": self.experiment.sensor_bank.design_ids[design],
            "dense_projection_valid": bool(projection_all),
            "multiplier_dynamics_valid": multiplier_valid,
            "tangent_action_valid": tangent_valid,
            "full_action_valid": False,
            "gram_rank_constant": rank_constant,
            "gram_rank_values": ";".join(str(value) for value in sorted(set(ranks))),
            "maximum_moment_residual": max(row["moment_residual"] for row in rows),
            "minimum_covariance_eigenvalue": min(row["covariance_minimum_eigenvalue"] for row in rows),
            "maximum_covariance_condition": max(row["covariance_condition"] for row in rows),
            "maximum_gram_compatibility_residual": max(row["gram_compatibility_relative_residual"] for row in rows),
            "tangent_action": tangent_action,
            "failure_reason": "" if tangent_valid else self._failure_reason(projection_all, covariance_all, compatibility_all, rank_constant, tangent_finite),
        }
        arrays = {
            "lambda_value": lambdas,
            "lambda_dot": lambda_dot,
            "tangent_action_density": tangent_density,
        }
        return design_result, rows, arrays

    @staticmethod
    def _failure_reason(projection: bool, covariance: bool, compatibility: bool, rank: bool, finite: bool) -> str:
        reasons = []
        if not projection: reasons.append("dense I-projection residual failure")
        if not covariance: reasons.append("covariance eigenvalue/condition contract failure")
        if not compatibility: reasons.append("tangent residual outside numerical Gram range")
        if not rank: reasons.append("Gram rank changes over time")
        if not finite: reasons.append("nonfinite tangent-action density")
        return "; ".join(reasons)

    def _finish(
        self, preflight_rows: list[dict[str, Any]], time_rows: list[dict[str, Any]],
        started: float, design_results: dict[int, dict[str, Any]] | None = None,
        sensitivity_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        design_results = design_results or {}
        sensitivity_rows = sensitivity_rows or []
        final_rows = []
        for row in preflight_rows:
            design = int(row["design_index"])
            if design in design_results:
                final_rows.append({**row, **design_results[design]})
            else:
                final_rows.append({
                    **row, "dense_projection_valid": False,
                    "multiplier_dynamics_valid": False, "tangent_action_valid": False,
                    "full_action_valid": False, "tangent_action": math.nan,
                })
        _write_csv(self.table_dir / "multiplier_dynamics_audit.csv", final_rows)
        _write_csv(self.table_dir / "multiplier_dynamics_time.csv", time_rows)
        _write_csv(self.table_dir / "tangent_action.csv", [{
            "design_index": row["design_index"], "design_id": row["design_id"],
            "tangent_action": row["tangent_action"], "valid": row["tangent_action_valid"],
        } for row in final_rows])
        self._plot_multiplier_results(final_rows, time_rows)
        summary = {
            "schema_version": 1,
            "frozen_layout_count": len(final_rows),
            "dense_target_range_valid_count": sum(row["dense_target_range_valid"] for row in final_rows),
            "dense_target_joint_feasible_count": sum(
                row["dense_target_joint_feasible"] for row in final_rows
            ),
            "dense_projection_valid_count": sum(row["dense_projection_valid"] for row in final_rows),
            "nominal_multiplier_tangent_valid_count": sum(
                row.get("bandwidth_sensitivity_evaluable", False)
                for row in sensitivity_rows
            ),
            "bandwidth_sensitivity_valid_count": sum(
                row.get("bandwidth_sensitivity_valid", False)
                for row in sensitivity_rows
            ),
            "multiplier_dynamics_valid_count": sum(row["multiplier_dynamics_valid"] for row in final_rows),
            "tangent_action_valid_count": sum(row["tangent_action_valid"] for row in final_rows),
            "elapsed_seconds": time.perf_counter() - started,
            "final_test_accessed": False,
        }
        write_json_atomic(self.table_dir / "action_readiness_summary.json", summary)
        return summary

    def _plot_multiplier_results(self, rows: list[dict[str, Any]], time_rows: list[dict[str, Any]]) -> None:
        out = self.figure_dir / "multiplier_dynamics"; out.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        valid = [row for row in rows if _as_bool(row.get("dense_target_range_valid"))]
        if valid:
            x = np.arange(len(valid))
            condition = np.log10(np.maximum(
                [float(row.get("maximum_covariance_condition", np.nan)) for row in valid],
                1e-300,
            ))
            color = [
                "#2f855a" if _as_bool(row.get("multiplier_dynamics_valid")) else "#c53030"
                for row in valid
            ]
            axis.scatter(x, condition, c=color, s=38)
            axis.set_xticks(x, [row["design_id"].replace("design_", "") for row in valid], rotation=60, fontsize=7)
        axis.axhline(
            math.log10(float(self.cfg["covariance_maximum_condition"])),
            color="black", linestyle="--", linewidth=1,
        )
        axis.set(ylabel=r"$\log_{10}$ maximum covariance condition", xlabel="range-valid frozen layout")
        axis.grid(alpha=0.18)
        fig.savefig(out / "covariance_condition.png", dpi=190); plt.close(fig)

        tangent = [row for row in rows if _as_bool(row.get("tangent_action_valid"))]
        out = self.figure_dir / "tangent_action"; out.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        if tangent:
            tangent = sorted(tangent, key=lambda row: float(row["tangent_action"]))
            axis.bar(
                np.arange(len(tangent)),
                [float(row["tangent_action"]) for row in tangent],
                color="#2b6cb0",
            )
            axis.set_xticks(np.arange(len(tangent)), [row["design_id"] for row in tangent], rotation=60, fontsize=7)
        else:
            axis.text(0.5, 0.5, "No layout passed the frozen tangent-readiness contract", ha="center", va="center", transform=axis.transAxes)
        axis.set(ylabel="tangent action", xlabel="layout"); axis.grid(alpha=0.18, axis="y")
        fig.savefig(out / "tangent_action_ranked.png", dpi=190); plt.close(fig)


def run_action_readiness(experiment, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    return OceanActionReadiness(experiment, analysis_dir, output_dir).run()
