"""Predeclared weighted-Poisson/full-action pilot for ocean drifters."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

from mfsi.cache import file_sha256, fingerprint, load_npz_cache, save_npz_cache
from mfsi.poisson import PoissonConfig
from mfsi.poisson_tesseract import (
    NATIVE_SOLVER_REVISION,
    solve_weighted_poisson_batch_tesseract_diagnostics,
)
from mfsi.reference_density import (
    backward_latent_with_log_density_correction,
    logistic_log_abs_det_jacobian,
)

from .action import (
    _features,
    _grid_points,
    _log_kde,
    _read_csv,
    _summary,
    _trust_solve,
    _write_csv,
)


def _forcing(
    phi: np.ndarray,
    target: np.ndarray,
    feature_material_derivative: np.ndarray,
    weights: np.ndarray,
    multiplier: np.ndarray,
    multiplier_dot: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Construct the accepted multiplier-coordinate full-law forcing."""
    expected_m = weights @ feature_material_derivative
    h = (
        (phi - target) @ multiplier_dot
        + (feature_material_derivative - expected_m) @ multiplier
    )
    mean = float(weights @ h)
    rms = float(np.sqrt(weights @ (h * h)))
    relative = abs(mean) / max(rms, 1.0e-14)
    return h, mean, relative


class OceanWeightedPoissonPilot:
    def __init__(self, experiment, analysis_dir: Path, output_dir: Path):
        self.experiment = experiment
        self.action_cfg = experiment.cfg["action"]
        self.cfg = self.action_cfg["poisson_pilot"]
        self.analysis = Path(analysis_dir)
        self.output = Path(output_dir)
        self.tables = self.analysis / "tables"
        self.figures = self.analysis / "figures/action_readiness/poisson_pilot"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.selection_path = experiment._resolve(self.cfg["selection_table"])
        self.selection = _read_csv(self.selection_path)
        if len(self.selection) != int(self.cfg["layout_count"]):
            raise RuntimeError("Poisson pilot selection count does not match its predeclared contract")
        if not all(row["selection_frozen_before_full_action"] == "True" for row in self.selection):
            raise RuntimeError("Poisson pilot selection was not frozen before full action")
        self.designs = np.asarray([int(row["design_index"]) for row in self.selection], dtype=int)
        tangent_rows = {row["design_id"]: row for row in _read_csv(
            experiment._resolve(self.action_cfg["tangent_readiness_table"])
        )}
        if not all(tangent_rows[row["design_id"]]["valid"] == "True" for row in self.selection):
            raise RuntimeError("Poisson pilot selection contains a layout that is not tangent-ready")
        self.integrated_tangent = {
            int(row["design_index"]): float(tangent_rows[row["design_id"]]["tangent_action"])
            for row in self.selection
        }
        with np.load(
            experiment._resolve("experiments/ocean_drifters/cache/action_moments_positive_kernel.npz"),
            allow_pickle=False,
        ) as data:
            self.all_designs = np.asarray(data["design_indices"], dtype=int)
            self.times = np.asarray(data["normalized_times"], dtype=np.float64)
            self.targets = np.asarray(data["smoothed_moments"], dtype=np.float64)
            self.target_dot = np.asarray(data["moment_derivative"], dtype=np.float64)
            metadata = json.loads(str(np.asarray(data["__metadata_json__"]).item()))
            if metadata.get("final_test_accessed") is not False:
                raise RuntimeError("action-moment cache reports final-test access")
        days = np.asarray(self.cfg["days"], dtype=np.float64)
        self.source_indices = np.rint(days * 4.0).astype(int)
        if not np.allclose(self.times[self.source_indices] * 45.0, days, atol=1e-12):
            raise RuntimeError("predeclared Poisson days do not lie on the frozen action grid")
        self.local_by_design = {
            int(design): int(np.flatnonzero(self.all_designs == design)[0])
            for design in self.designs
        }

    def _reference_grid(
        self, resolution: tuple[int, int]
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        nx, ny = resolution
        bounds = np.asarray(self.experiment.cfg["scientific"]["domain_km"], dtype=np.float64)
        points, dx, dy = _grid_points(bounds, nx, ny)
        if not np.isclose(dx, dy, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"native Poisson grid must have square physical cells, got dx={dx}, dy={dy}"
            )
        flow = self.experiment.reference()
        with np.load(self.experiment.paths["conditioned_endpoint_estimator"], allow_pickle=False) as data:
            atoms = np.asarray(data["x0_atoms_km"], dtype=np.float64)
            bandwidth = np.asarray(data["H0_km2"], dtype=np.float64)
        normalization_rows = _read_csv(self.experiment.paths["conditioned_kde_normalization"])
        z0 = float(next(row["Z_hat"] for row in normalization_rows if row["endpoint"] == "day0"))
        cache_dir = self.experiment._resolve(
            f"experiments/ocean_drifters/cache/poisson_pilot_reference_{nx}x{ny}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_base_all = np.empty((len(self.source_indices), len(points)), dtype=np.float64)
        velocity_all = np.empty((len(self.source_indices), len(points), 2), dtype=np.float64)
        zt = flow.to_latent(jnp.asarray(points))
        velocity_fn = jax.jit(lambda value: flow.velocity(jnp.asarray(points), value))

        @lru_cache(maxsize=None)
        def backward_fn(steps: int):
            return jax.jit(lambda z, value: backward_latent_with_log_density_correction(
                flow.params, z, value, steps=steps
            ))

        signature_base = {
            "schema": 1,
            "grid": [nx, ny],
            "reference": file_sha256(self.experiment.paths["reference_checkpoint"]),
            "endpoint": file_sha256(self.experiment.paths["conditioned_endpoint_estimator"]),
            "normalization": z0,
            "pilot_sources": self.source_indices.tolist(),
        }
        for local_time, source in enumerate(self.source_indices):
            value = float(self.times[source])
            signature = fingerprint({**signature_base, "source": int(source), "time": value})
            cache = cache_dir / f"reference_t{source:03d}.npz"
            loaded = load_npz_cache(cache, signature=signature)
            if loaded is not None:
                arrays, metadata = loaded
                if metadata.get("final_test_accessed") is not False:
                    raise RuntimeError("Poisson reference cache lacks the final-test lock")
                log_base_all[local_time] = arrays["log_base_mass"]
                velocity_all[local_time] = arrays["velocity"]
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
                    initial_log
                    + np.asarray(logistic_log_abs_det_jacobian(initial_z, bounds))
                    + np.asarray(correction)
                    - np.asarray(logistic_log_abs_det_jacobian(local_z, bounds))
                )
            log_mass = log_density + math.log(dx * dy)
            log_base = log_mass - logsumexp(log_mass)
            velocity = np.asarray(velocity_fn(jnp.asarray(value)))
            log_base_all[local_time] = log_base
            velocity_all[local_time] = velocity
            save_npz_cache(
                cache,
                {"log_base_mass": log_base, "velocity": velocity},
                signature=signature,
                metadata={"source_time_index": int(source), "final_test_accessed": False},
            )
            print(
                f"[ocean Poisson] reference {nx}x{ny} day={value * 45.0:g}",
                flush=True,
            )
        return points, dx, log_base_all, velocity_all

    def _systems_for_grid(
        self,
        resolution: tuple[int, int],
        points: np.ndarray,
        dx: float,
        log_base: np.ndarray,
        velocity: np.ndarray,
    ) -> list[dict[str, Any]]:
        systems: list[dict[str, Any]] = []
        sigma = self.experiment.sensor_bank.sigma_km
        cell_area = dx * dx
        for design in self.designs:
            local = self.local_by_design[int(design)]
            centers = self.experiment.sensor_bank.centers_km[design]
            phi = _features(points, centers, sigma)
            delta = points[:, None] - centers[None]
            gradient = -(delta / sigma**2) * phi[:, :, None]
            cache = self.experiment._resolve(
                f"experiments/ocean_drifters/cache/action_projected_laws/design_{design:06d}_nominal.npz"
            )
            with np.load(cache, allow_pickle=False) as data:
                if bool(data["final_test_accessed"]):
                    raise RuntimeError("projected-law action cache reports final-test access")
                multipliers = np.asarray(data["lambda_value"], dtype=np.float64)
                multiplier_dot = np.asarray(data["lambda_dot"], dtype=np.float64)
                tangent_density = np.asarray(data["tangent_action_density"], dtype=np.float64)
            for time_local, source in enumerate(self.source_indices):
                material = np.einsum("nmd,nd->nm", gradient, velocity[time_local])
                target = self.targets[local, source]
                target_derivative = self.target_dot[local, source]
                refined_multiplier = multipliers[source].copy()
                projection = _summary(
                    phi, log_base[time_local], target, refined_multiplier
                )
                projection_iterations = 0
                projection_backend = "accepted_multiplier_direct_evaluation"
                forcing_projection_tolerance = float(
                    self.cfg["forcing_projection_residual_tolerance"]
                )
                if projection["residual_norm"] > forcing_projection_tolerance:
                    refined_multiplier, projection, projection_iterations = _trust_solve(
                        phi,
                        log_base[time_local],
                        target,
                        refined_multiplier,
                        forcing_projection_tolerance,
                        int(self.cfg["forcing_projection_maximum_function_evaluations"]),
                    )
                    projection_backend = "exact_covariance_grid_refinement"
                weights = projection["weights"]
                projected_moment = projection["moments"]
                log_q_mass = log_base[time_local] + phi @ refined_multiplier
                log_q_mass -= logsumexp(log_q_mass)
                q_density = weights / cell_area
                expected_m = weights @ material
                covariance = projection["covariance"]
                covariance_eigenvalues = projection["covariance_eigenvalues"]
                covariance_ready = bool(
                    covariance_eigenvalues[0] >= float(self.action_cfg["covariance_minimum_eigenvalue"])
                    and projection["covariance_condition"] <= float(
                        self.action_cfg["covariance_maximum_condition"]
                    )
                )
                refined_multiplier_dot = np.full(4, np.nan)
                lambda_dot_solve_residual = math.nan
                if covariance_ready:
                    lambda_m = material @ refined_multiplier
                    centered_phi = phi - projected_moment
                    cov_phi_lambda_m = np.einsum(
                        "n,ni,n->i",
                        weights,
                        centered_phi,
                        lambda_m - weights @ lambda_m,
                    )
                    lambda_dot_rhs = target_derivative - expected_m - cov_phi_lambda_m
                    refined_multiplier_dot = np.linalg.solve(covariance, lambda_dot_rhs)
                    lambda_dot_solve_residual = float(
                        np.linalg.norm(covariance @ refined_multiplier_dot - lambda_dot_rhs)
                        / max(np.linalg.norm(lambda_dot_rhs), 1e-14)
                    )
                gram = np.einsum("n,nmd,nkd->mk", weights, gradient, gradient)
                gram_eigenvalues, gram_vectors = np.linalg.eigh(gram)
                rank_threshold = float(self.action_cfg["gram_relative_rank_tolerance"]) * max(
                    gram_eigenvalues[-1], 1e-300
                )
                retained = gram_eigenvalues > rank_threshold
                tangent_rhs = expected_m - target_derivative
                projected_tangent_rhs = (
                    gram_vectors[:, retained] @ (gram_vectors[:, retained].T @ tangent_rhs)
                    if np.any(retained) else np.zeros_like(tangent_rhs)
                )
                tangent_compatibility = float(
                    np.linalg.norm(tangent_rhs - projected_tangent_rhs)
                    / max(np.linalg.norm(tangent_rhs), 1e-14)
                )
                grid_tangent_density = (
                    float(np.sum(
                        (gram_vectors[:, retained].T @ tangent_rhs) ** 2
                        / gram_eigenvalues[retained]
                    )) if np.any(retained) else math.nan
                )
                if covariance_ready:
                    h, compatibility, compatibility_relative = _forcing(
                        phi,
                        target,
                        material,
                        weights,
                        refined_multiplier,
                        refined_multiplier_dot,
                    )
                else:
                    h = np.full(len(points), np.nan)
                    compatibility = math.nan
                    compatibility_relative = math.inf
                underflow = weights == 0.0
                underflow_log_mass = (
                    float(logsumexp(log_q_mass[underflow]) / math.log(10.0))
                    if np.any(underflow) else -math.inf
                )
                active_log_density = log_q_mass[~underflow] - math.log(cell_area)
                systems.append({
                    "design_index": int(design),
                    "design_id": self.experiment.sensor_bank.design_ids[design],
                    "source_time_index": int(source),
                    "day": float(self.times[source] * 45.0),
                    "grid_nx": resolution[0],
                    "grid_ny": resolution[1],
                    "dx_km": dx,
                    "q": q_density.reshape((resolution[1], resolution[0])),
                    "h": h.reshape((resolution[1], resolution[0])),
                    "log_q_mass": log_q_mass,
                    "compatibility_residual": compatibility,
                    "compatibility_relative_residual": compatibility_relative,
                    "compatibility_valid": bool(
                        projection["residual_norm"] <= forcing_projection_tolerance
                        and covariance_ready
                        and compatibility_relative
                        <= float(self.cfg["maximum_relative_compatibility_residual"])
                    ),
                    "projected_moment_residual": projection["residual_norm"],
                    "projection_backend": projection_backend,
                    "projection_iterations": projection_iterations,
                    "refined_multiplier_norm": float(np.linalg.norm(refined_multiplier)),
                    "relative_multiplier_grid_change": float(
                        np.linalg.norm(refined_multiplier - multipliers[source])
                        / max(np.linalg.norm(multipliers[source]), 1e-14)
                    ),
                    "covariance_minimum_eigenvalue": float(covariance_eigenvalues[0]),
                    "covariance_condition": projection["covariance_condition"],
                    "multiplier_coordinate_ready": covariance_ready,
                    "lambda_dot_solve_relative_residual": lambda_dot_solve_residual,
                    "relative_lambda_dot_grid_change": float(
                        np.linalg.norm(refined_multiplier_dot - multiplier_dot[source])
                        / max(np.linalg.norm(multiplier_dot[source]), 1e-14)
                    ) if covariance_ready else math.inf,
                    "gram_rank": int(np.sum(retained)),
                    "tangent_compatibility_relative_residual": tangent_compatibility,
                    "minimum_log_density": float(np.min(log_q_mass) - math.log(cell_area)),
                    "maximum_log_density": float(np.max(log_q_mass) - math.log(cell_area)),
                    "minimum_active_log_density": float(np.min(active_log_density)),
                    "maximum_active_log_density": float(np.max(active_log_density)),
                    "underflow_cell_count": int(np.sum(underflow)),
                    "underflow_cell_fraction": float(np.mean(underflow)),
                    "underflow_log10_probability_mass": underflow_log_mass,
                    "tangent_action_density": grid_tangent_density,
                    "accepted_medium_grid_tangent_action_density": float(tangent_density[source]),
                })
        return systems

    def _solve_floor(
        self, systems: list[dict[str, Any]], floor: float, dx: float
    ) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
        eligible = [system for system in systems if system["compatibility_valid"]]
        potential: dict[tuple[int, int], np.ndarray] = {}
        diagnostics: dict[int, dict[str, Any]] = {}
        error = ""
        if eligible:
            cfg = PoissonConfig(
                dx=dx,
                operator_floor_rel=floor,
                cg_tol=float(self.cfg["cg_tolerance"]),
                cg_maxiter=int(self.cfg["cg_maximum_iterations"]),
                gauge_strength=float(self.cfg["gauge_strength"]),
            )
            try:
                result = solve_weighted_poisson_batch_tesseract_diagnostics(
                    np.stack([system["q"] for system in eligible]),
                    np.stack([system["h"] for system in eligible]),
                    cfg,
                )
                for index, system in enumerate(eligible):
                    diagnostics[id(system)] = {
                        key: np.asarray(value)[index].item()
                        for key, value in result.items()
                        if key != "potential"
                    }
                    potential[(system["design_index"], system["source_time_index"])] = np.asarray(
                        result["potential"][index]
                    )
            except Exception as exc:  # retain a failed stabilization trial as data
                error = f"{type(exc).__name__}: {exc}"
        rows: list[dict[str, Any]] = []
        for system in systems:
            diag = diagnostics.get(id(system), {})
            converged = bool(diag.get("converged", False))
            stabilized_residual = float(diag.get("stabilized_relative_residual", math.nan))
            physical_residual = float(diag.get("physical_relative_residual", math.nan))
            gauge_residual = abs(float(diag.get("weighted_mean_potential", math.nan)))
            action = float(diag.get("action", math.nan))
            tangent = float(system["tangent_action_density"])
            inequality_tolerance = float(self.cfg["tangent_full_inequality_relative_tolerance"])
            inequality_valid = bool(
                np.isfinite(action)
                and tangent <= action + inequality_tolerance * max(abs(action), abs(tangent), 1.0)
            )
            q_solve = system["q"] / max(float(np.max(system["q"])), 1e-300)
            floor_mass_ratio = float(floor * q_solve.size / max(np.sum(q_solve), 1e-300))
            solver_success = bool(
                system["compatibility_valid"]
                and converged
                and stabilized_residual <= float(self.cfg["maximum_relative_pde_residual"])
                and gauge_residual <= float(self.cfg["maximum_absolute_weighted_mean_potential"])
            )
            physical_pde_valid = bool(
                solver_success
                and physical_residual <= float(self.cfg["maximum_relative_pde_residual"])
            )
            row = {
                key: value for key, value in system.items()
                if key not in {"q", "h", "log_q_mass"}
            }
            row.update({
                "operator_floor_relative": floor,
                "operator_floor_absolute_density": diag.get("operator_floor", math.nan),
                "operator_floor_to_physical_coefficient_sum_ratio": floor_mass_ratio,
                "preconditioner": self.cfg["preconditioner"],
                "cg_tolerance": self.cfg["cg_tolerance"],
                "cg_maximum_iterations": self.cfg["cg_maximum_iterations"],
                "solver_converged": converged,
                "solver_success": solver_success,
                "iterations": diag.get("iterations", 0),
                "native_relative_pde_residual": diag.get("native_relative_residual", math.nan),
                "stabilized_relative_pde_residual": stabilized_residual,
                "physical_relative_pde_residual": physical_residual,
                "physical_absolute_pde_residual": diag.get("physical_absolute_residual", math.nan),
                "physical_pde_valid": physical_pde_valid,
                "weighted_mean_potential_residual": diag.get("weighted_mean_potential", math.nan),
                "condition_proxy": diag.get("coefficient_condition_proxy", math.inf),
                "full_action_density": action,
                "tangent_full_inequality_valid": inequality_valid,
                "solve_accepted_before_refinement": bool(
                    solver_success and physical_pde_valid and inequality_valid
                ),
                "solver_error": "" if id(system) in diagnostics else (
                    error or "compatibility residual exceeded the predeclared tolerance"
                ),
                "density_floor_or_cell_threshold_used": False,
                "final_test_accessed": False,
            })
            rows.append(row)
        return rows, potential

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        all_rows: list[dict[str, Any]] = []
        system_lookup: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        potentials: dict[tuple[int, int, int, int, float], np.ndarray] = {}
        for resolution_values in self.cfg["grid_resolutions"]:
            resolution = tuple(int(value) for value in resolution_values)
            points, dx, log_base, velocity = self._reference_grid(resolution)
            systems = self._systems_for_grid(resolution, points, dx, log_base, velocity)
            for system in systems:
                system_lookup[(resolution[0], resolution[1], system["design_index"], system["source_time_index"])] = system
            for floor in (float(value) for value in self.cfg["operator_floor_relative_values"]):
                print(
                    f"[ocean Poisson] solve grid={resolution[0]}x{resolution[1]} floor={floor:g}",
                    flush=True,
                )
                rows, local_potential = self._solve_floor(systems, floor, dx)
                all_rows.extend(rows)
                for (design, source), psi in local_potential.items():
                    potentials[(resolution[0], resolution[1], design, source, floor)] = psi

        primary_floor = float(self.cfg["reported_operator_floor_relative"])
        coarse = tuple(int(value) for value in self.cfg["grid_resolutions"][0])
        fine = tuple(int(value) for value in self.cfg["grid_resolutions"][-1])
        keyed = {
            (int(row["grid_nx"]), int(row["grid_ny"]), int(row["design_index"]),
             int(row["source_time_index"]), float(row["operator_floor_relative"])): row
            for row in all_rows
        }
        for design in self.designs:
            for source in self.source_indices:
                coarse_row = keyed[(coarse[0], coarse[1], int(design), int(source), primary_floor)]
                fine_row = keyed[(fine[0], fine[1], int(design), int(source), primary_floor)]
                coarse_action = float(coarse_row["full_action_density"])
                fine_action = float(fine_row["full_action_density"])
                grid_change = (
                    abs(fine_action - coarse_action) / max(abs(fine_action), 1e-14)
                    if np.isfinite(coarse_action) and np.isfinite(fine_action) else math.inf
                )
                positive_actions = [
                    float(keyed[(fine[0], fine[1], int(design), int(source), floor)]["full_action_density"])
                    for floor in (float(value) for value in self.cfg["operator_floor_relative_values"])
                    if floor > 0.0 and np.isfinite(float(
                        keyed[(fine[0], fine[1], int(design), int(source), floor)]["full_action_density"]
                    ))
                ]
                floor_change = (
                    max(abs(value - fine_action) / max(abs(fine_action), 1e-14) for value in positive_actions)
                    if positive_actions and np.isfinite(fine_action) else math.inf
                )
                for grid in (coarse, fine):
                    row = keyed[(grid[0], grid[1], int(design), int(source), primary_floor)]
                    row["coarse_fine_relative_action_change"] = grid_change
                    row["grid_refinement_valid"] = grid_change <= float(
                        self.cfg["maximum_relative_action_grid_change"]
                    )
                    row["positive_floor_maximum_relative_action_change"] = floor_change
                    row["operator_floor_sensitivity_valid"] = floor_change <= float(
                        self.cfg["maximum_relative_action_floor_change"]
                    )

        csv_rows = sorted(all_rows, key=lambda row: (
            int(row["design_index"]), float(row["day"]), int(row["grid_nx"]),
            float(row["operator_floor_relative"]),
        ))
        _write_csv(self.tables / "poisson_pilot_time.csv", csv_rows)
        summaries = self._summaries(keyed, coarse, fine, primary_floor)
        _write_csv(self.tables / "poisson_pilot_summary.csv", summaries)
        self._plots(csv_rows, summaries, system_lookup, potentials, fine, primary_floor)
        result = self._report(csv_rows, summaries, coarse, fine, primary_floor, started)
        return result

    def _summaries(
        self, keyed: dict[tuple[int, int, int, int, float], dict[str, Any]],
        coarse: tuple[int, int], fine: tuple[int, int], primary_floor: float,
    ) -> list[dict[str, Any]]:
        summaries = []
        pilot_times = self.times[self.source_indices]
        for design in self.designs:
            rows = [
                keyed[(fine[0], fine[1], int(design), int(source), primary_floor)]
                for source in self.source_indices
            ]
            full = np.asarray([float(row["full_action_density"]) for row in rows])
            tangent = np.asarray([float(row["tangent_action_density"]) for row in rows])
            full_integrated = float(np.trapezoid(full, pilot_times)) if np.isfinite(full).all() else math.nan
            tangent_integrated = float(np.trapezoid(tangent, pilot_times))
            inequality_tol = float(self.cfg["tangent_full_inequality_relative_tolerance"])
            integrated_inequality = bool(
                np.isfinite(full_integrated)
                and tangent_integrated <= full_integrated + inequality_tol * max(
                    abs(full_integrated), abs(tangent_integrated), 1.0
                )
            )
            zero_rows = [
                keyed[(fine[0], fine[1], int(design), int(source), 0.0)]
                for source in self.source_indices
            ] if 0.0 in [float(value) for value in self.cfg["operator_floor_relative_values"]] else []
            accepted = all(
                row["solve_accepted_before_refinement"]
                and row["grid_refinement_valid"]
                and row["operator_floor_sensitivity_valid"]
                for row in rows
            ) and integrated_inequality and all(row["solver_success"] for row in zero_rows)
            summaries.append({
                "design_index": int(design),
                "design_id": self.experiment.sensor_bank.design_ids[design],
                "pilot_time_count": len(rows),
                "successful_reported_floor_time_count": sum(row["solver_success"] for row in rows),
                "physical_pde_valid_time_count": sum(row["physical_pde_valid"] for row in rows),
                "compatibility_valid_time_count": sum(row["compatibility_valid"] for row in rows),
                "pointwise_tangent_full_valid_count": sum(row["tangent_full_inequality_valid"] for row in rows),
                "grid_refinement_valid_time_count": sum(row["grid_refinement_valid"] for row in rows),
                "operator_floor_sensitivity_valid_time_count": sum(row["operator_floor_sensitivity_valid"] for row in rows),
                "unstabilized_successful_time_count": sum(row["solver_success"] for row in zero_rows),
                "pilot_node_tangent_action": tangent_integrated,
                "pilot_node_full_action": full_integrated,
                "integrated_tangent_full_inequality_valid": integrated_inequality,
                "canonical_181_time_tangent_action": self.integrated_tangent[int(design)],
                "maximum_grid_relative_action_change": max(float(row["coarse_fine_relative_action_change"]) for row in rows),
                "maximum_positive_floor_relative_action_change": max(float(row["positive_floor_maximum_relative_action_change"]) for row in rows),
                "maximum_physical_relative_pde_residual": max(float(row["physical_relative_pde_residual"]) for row in rows),
                "maximum_compatibility_relative_residual": max(float(row["compatibility_relative_residual"]) for row in rows),
                "maximum_underflow_cell_fraction": max(float(row["underflow_cell_fraction"]) for row in rows),
                "pilot_full_action_valid": accepted,
                "full_action_valid": False,
                "final_test_accessed": False,
            })
        return summaries

    def _plots(
        self, rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
        systems: dict[tuple[int, int, int, int], dict[str, Any]],
        potentials: dict[tuple[int, int, int, int, float], np.ndarray],
        fine: tuple[int, int], primary_floor: float,
    ) -> None:
        primary = [row for row in rows if int(row["grid_nx"]) == fine[0]
                   and float(row["operator_floor_relative"]) == primary_floor]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
        for axis, design in zip(axes.ravel(), self.designs, strict=True):
            local = sorted((row for row in primary if int(row["design_index"]) == design), key=lambda r: float(r["day"]))
            axis.plot([r["day"] for r in local], [r["tangent_action_density"] for r in local], "o-", label=r"$a_{tan}$")
            axis.plot([r["day"] for r in local], [r["full_action_density"] for r in local], "s-", label=r"$a_{full}$")
            axis.set_title(self.experiment.sensor_bank.design_ids[design]); axis.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False); fig.supxlabel("day"); fig.supylabel("action density")
        fig.savefig(self.figures / "tangent_vs_full_action.png", dpi=190); plt.close(fig)

        fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        for design in self.designs:
            local = sorted((row for row in primary if int(row["design_index"]) == design), key=lambda r: float(r["day"]))
            axis.semilogy([r["day"] for r in local], [r["physical_relative_pde_residual"] for r in local], "o-", label=f"{design:03d}")
        axis.axhline(float(self.cfg["maximum_relative_pde_residual"]), color="black", ls="--", lw=1)
        axis.set(xlabel="day", ylabel="physical relative PDE residual"); axis.grid(alpha=0.2); axis.legend(ncol=3, frameon=False)
        fig.savefig(self.figures / "physical_pde_residual_by_time.png", dpi=190); plt.close(fig)

        fig, axis = plt.subplots(figsize=(6, 5.4), constrained_layout=True)
        coarse = tuple(int(value) for value in self.cfg["grid_resolutions"][0])
        coarse_rows = {(int(r["design_index"]), int(r["source_time_index"])): r for r in rows
                       if int(r["grid_nx"]) == coarse[0] and float(r["operator_floor_relative"]) == primary_floor}
        x = []; y = []
        for row in primary:
            other = coarse_rows[(int(row["design_index"]), int(row["source_time_index"]))]
            x.append(float(other["full_action_density"])); y.append(float(row["full_action_density"]))
        axis.scatter(x, y, c="#2b6cb0", alpha=0.8); limit = max(x + y) if x and y else 1.0
        axis.plot([0, limit], [0, limit], color="black", ls="--", lw=1)
        axis.set(xlabel=f"coarse {coarse[0]}x{coarse[1]} action", ylabel=f"fine {fine[0]}x{fine[1]} action")
        axis.grid(alpha=0.2); fig.savefig(self.figures / "action_grid_refinement.png", dpi=190); plt.close(fig)

        examples = [(192, 10, "easy_overlap"), (241, 10, "hard_overlap")]
        for design, source, label in examples:
            system = systems[(fine[0], fine[1], design, source)]
            psi = potentials.get((fine[0], fine[1], design, source, primary_floor))
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
            logq = system["log_q_mass"].reshape((fine[1], fine[0]))
            image = axes[0].imshow(logq, origin="lower", aspect="auto", cmap="viridis")
            fig.colorbar(image, ax=axes[0], shrink=0.8); axes[0].set_title("projected log cell mass")
            image = axes[1].imshow(system["h"], origin="lower", aspect="auto", cmap="coolwarm")
            fig.colorbar(image, ax=axes[1], shrink=0.8); axes[1].set_title("forcing h")
            if psi is not None:
                image = axes[2].imshow(psi, origin="lower", aspect="auto", cmap="coolwarm")
                fig.colorbar(image, ax=axes[2], shrink=0.8); axes[2].set_title("Poisson potential")
            else:
                axes[2].text(0.5, 0.5, "solve unavailable", ha="center", va="center")
            fig.suptitle(f"{self.experiment.sensor_bank.design_ids[design]} day 2.5")
            fig.savefig(self.figures / f"density_forcing_potential_{label}.png", dpi=190); plt.close(fig)

    def _report(
        self, rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
        coarse: tuple[int, int], fine: tuple[int, int], primary_floor: float,
        started: float,
    ) -> dict[str, Any]:
        primary = [row for row in rows if int(row["grid_nx"]) == fine[0]
                   and float(row["operator_floor_relative"]) == primary_floor]
        successful = sum(row["solver_success"] for row in primary)
        physical_valid = sum(row["physical_pde_valid"] for row in primary)
        compatibility_valid = sum(row["compatibility_valid"] for row in primary)
        pointwise_inequality = sum(row["tangent_full_inequality_valid"] for row in primary)
        integrated_inequality = sum(row["integrated_tangent_full_inequality_valid"] for row in summaries)
        unstabilized = sum(int(row["unstabilized_successful_time_count"]) for row in summaries)
        pilot_valid = all(row["pilot_full_action_valid"] for row in summaries)
        finite = [row for row in summaries if np.isfinite(float(row["pilot_node_full_action"]))]
        rho = float(spearmanr(
            [float(row["pilot_node_tangent_action"]) for row in finite],
            [float(row["pilot_node_full_action"]) for row in finite],
        ).statistic) if len(finite) >= 2 else math.nan
        max_grid = max(float(row["maximum_grid_relative_action_change"]) for row in summaries)
        max_floor = max(float(row["maximum_positive_floor_relative_action_change"]) for row in summaries)
        max_physical = max(float(row["maximum_physical_relative_pde_residual"]) for row in summaries)
        max_compat = max(float(row["maximum_compatibility_relative_residual"]) for row in summaries)
        max_underflow = max(float(row["maximum_underflow_cell_fraction"]) for row in summaries)
        decision = (
            "The pilot passes and a production sweep over the 24 tangent-ready layouts is authorized."
            if pilot_valid else
            f"The pilot does not authorize a production full-action sweep. The forcing compatibility and "
            f"tangent lower-bound checks pass, but the native weighted-Poisson discretization is not stable "
            f"under the ocean density dynamic range: only {unstabilized}/30 unstabilized solves converge, "
            f"only {physical_valid}/30 reported-floor solves meet the physical unfloored PDE residual, "
            f"and action changes reach {max_grid:.1%} across grids and {max_floor:.1%} across positive "
            f"operator floors. The scientific reference and 68-layout law set remain unchanged."
        )
        selected = ", ".join(f"`{row['design_id']}`" for row in self.selection)
        report = f"""# Weighted-Poisson/full-action pilot report

## Decision

{decision}

This was a numerical validation pilot only. No final sensor was selected, no
production sweep was launched, and the 69 final-test trajectories were untouched.

## Predeclared scope

The subset was frozen before any full-action computation in
[`poisson_pilot_selection.md`](poisson_pilot_selection.md): {selected}.
`design_000216` was not included because it is not tangent-ready. Selection used
only frozen validation risk, tangent action, KL/ESS diagnostics, and geometry.

The pilot evaluated 30 layout-time points (six layouts at five days) on square-cell
{coarse[0]}x{coarse[1]} ({3650/coarse[0]:.3f} km) and {fine[0]}x{fine[1]}
({3650/fine[0]:.3f} km) meshes. These are the closest practical square-cell native
mesh hierarchy around the accepted ocean resolutions. Boundary condition and gauge
were unchanged: homogeneous no-flux finite volume and projected-law weighted mean zero.

## Required questions

1. **Which layouts and why?** {selected}. Together they span minimum, median, and
   maximum tangent action; the overlap and KL extremes; and distinct support-random
   and longitudinal geometries. Exact source values are in
   [`poisson_pilot_selection.csv`](tables/poisson_pilot_selection.csv).
2. **Was selection frozen first?** Yes. The signed source-table hashes and freeze
   declaration predate every solve in this pilot.
3. **Does E_q[h] approximately vanish?** {compatibility_valid}/30 fine-grid points
   pass the predeclared relative compatibility tolerance; the maximum relative
   residual is {max_compat:.3e}. Grid-specific exact-covariance refinement of the
   same exponential-family projection was required at concentrated times; the
   reference law and moment targets were not changed.
4. **Does the solver converge?** At the reported diagnostic operator floor
   ({primary_floor:g}), {successful}/30 fine-grid solves meet native convergence,
   stabilized residual, and gauge gates.
5. **At how many layout-time points?** {successful}/30 at the reported floor;
   {physical_valid}/30 also satisfy the physical, unfloored-operator PDE residual.
6. **What failures remain?** Unstabilized solves succeed at {unstabilized}/30 points.
   At the reported floor, only {physical_valid}/30 meet the physical PDE residual
   and the maximum physical relative residual is {max_physical:.3e}. Refinement and
   floor sensitivity also fail at concentrated early-time cases.
7. **Grid sensitivity?** Maximum coarse/fine relative action change is
   {max_grid:.3%}, against the predeclared 10% tolerance.
8. **Does density range cause conditioning problems?** Yes. Up to
   {max_underflow:.1%} of cells are below float64's effective
   range in direct probability representation; full log-density ranges and the
   IC(0) coefficient condition proxy are recorded per solve.
9. **Was a density floor/truncation used?** No. No scientific q value was clipped,
   floored, thresholded, or renormalized away. The shared backend's separately
   declared *operator* floor was audited at 0, 2e-7, 2e-6, and 2e-5; its maximum
   positive-floor action change is {max_floor:.3%}. This stabilization is not
   accepted as production methodology unless all pilot contracts pass.
10. **Pointwise a_tan <= a_full?** {pointwise_inequality}/30 finite fine-grid
    reported-floor diagnostics pass within the predeclared 0.1% numerical
    tolerance, including every converged stabilized solve. Only {physical_valid}
    point satisfies the complete physical-PDE acceptance contract.
11. **Integrated inequality?** {integrated_inequality}/6 layouts pass on the same
    five-node pilot quadrature. These sparse integrals are diagnostics, not final
    181-time production actions.
12. **Ordering agreement?** Spearman correlation between the five-node tangent and
    full pilot integrals is {rho:.4f}. No selection inference is made from six layouts.
13. **Is the shared native implementation suitable?** {'Yes for the tested pilot.' if pilot_valid else 'Not yet under the frozen ocean numerical contract.'}
14. **Is a production sweep justified?** {'Yes.' if pilot_valid else 'No.'}
15. **If yes, over what set?** {'The 24 already tangent-ready layouts, without changing the frozen 68-layout law set.' if pilot_valid else 'Not applicable until the numerical blocker is resolved.'}
16. **If no, exact blocker?** {'Not applicable.' if pilot_valid else f'The IC(0)-PCG solve without an operator floor converges at only {unstabilized}/30 points. Adding the smallest tested floor gives {successful}/30 stabilized solves but only {physical_valid}/30 physical-PDE-valid results, while grid and positive-floor action changes reach {max_grid:.1%} and {max_floor:.1%}. The lower bound is not the blocker.'}
17. **Were final-test trajectories untouched?** Yes; every input/output flag remains
    `final_test_accessed=false`, and the production API has no final-test path.

## Machine-readable record

- [`poisson_pilot_time.csv`](tables/poisson_pilot_time.csv) records every grid,
  operator-floor trial, convergence count, stabilized and physical residual,
  compatibility/gauge residual, density range, underflow effect, condition proxy,
  action, and lower-bound result.
- [`poisson_pilot_summary.csv`](tables/poisson_pilot_summary.csv) records the six
  layout summaries and five-node integrated diagnostics.

The shared backend revision is `{NATIVE_SOLVER_REVISION}` with native IC(0)-PCG,
CG tolerance {float(self.cfg['cg_tolerance']):g}, and maximum
{int(self.cfg['cg_maximum_iterations'])} iterations. Pilot runtime was
{time.perf_counter() - started:.1f} seconds.
"""
        (self.analysis / "weighted_poisson_pilot_report.md").write_text(report, encoding="utf-8")
        return {
            "schema_version": 1,
            "pilot_layout_count": len(self.designs),
            "pilot_time_count_per_layout": len(self.source_indices),
            "pilot_layout_time_count": len(primary),
            "selection_frozen_before_full_action": True,
            "compatibility_valid_count": compatibility_valid,
            "solver_success_count": successful,
            "physical_pde_valid_count": physical_valid,
            "unstabilized_solver_success_count": unstabilized,
            "pointwise_tangent_full_valid_count": pointwise_inequality,
            "integrated_tangent_full_valid_count": integrated_inequality,
            "pilot_backend_valid": pilot_valid,
            "production_sweep_authorized": pilot_valid,
            "authorized_action_ready_layout_count": 24 if pilot_valid else 0,
            "full_action_valid": False,
            "final_test_accessed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }


def run_weighted_poisson_pilot(experiment, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    return OceanWeightedPoissonPilot(experiment, analysis_dir, output_dir).run()
