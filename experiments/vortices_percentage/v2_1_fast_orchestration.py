"""V2.1-only deterministic orchestration accelerations for resumed selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from v2_1_parallel_exact_solver import solve_v2_parallel


def install_parallel_exact_solver(harness: Any) -> None:
    """Redirect only the loaded V2 selection harness's exact solver global."""
    harness.solve_v2 = solve_v2_parallel


def install_threaded_multistart(harness: Any, *, workers: int = 4) -> None:
    """Use the optimizer's existing ordered, compiled single-start thread path."""
    original = harness.optimize_multistart_candidates

    def threaded(*args, **kwargs):
        if kwargs.get("vectorize_starts", True) is False:
            kwargs = dict(kwargs)
            kwargs["start_workers"] = int(workers)
        return original(*args, **kwargs)

    harness.optimize_multistart_candidates = threaded


def parallel_fast_rank(
    harness: Any,
    pool: dict[tuple[float, ...], dict[str, Any]],
    objective: Any,
    constraints: tuple,
    label: str,
    *,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Evaluate the unchanged single-candidate graph concurrently, in order."""
    candidates = list(pool.values())
    if not candidates:
        return []
    fn = jax.jit(objective)
    first = jnp.asarray(candidates[0]["eta"], dtype=jnp.float64)
    compiled = fn.lower(first).compile()

    def evaluate(candidate: dict[str, Any]) -> dict[str, Any]:
        eta = jnp.asarray(candidate["eta"], dtype=jnp.float64)
        violation = max([0.0] + [float(cfn(eta) - upper) for cfn, upper in constraints])
        try:
            dispatched = compiled(eta)
            value = float(jax.block_until_ready(dispatched))
        except Exception as exc:
            value = 1.0e12
            candidate = dict(candidate, fast_error=repr(exc))
        return dict(candidate, fast_value=value, fast_violation=violation)

    print(f"[{label} threaded rank] 0/{len(candidates)}", flush=True)
    with ThreadPoolExecutor(max_workers=min(int(workers), len(candidates))) as executor:
        rows = list(executor.map(evaluate, candidates))
    rows.sort(
        key=lambda row: (
            not np.isfinite(row["fast_value"]),
            row["fast_violation"],
            row["fast_value"],
            row["candidate_id"],
        )
    )
    print(f"[{label} threaded rank] {len(candidates)}/{len(candidates)}", flush=True)
    return rows


def install_threaded_fast_rank(harness: Any, *, workers: int = 4) -> None:
    harness.fast_rank = lambda pool, objective, constraints, label: parallel_fast_rank(
        harness, pool, objective, constraints, label, workers=workers
    )


def install_batched_decomposition_evaluator(harness: Any) -> None:
    """Batch the four existing per-trial decomposition calls without changing them."""
    original = harness.Evaluator

    class BatchedDecompositionEvaluator(original):
        def full(self, eta: Any, trials: int, grid_shape: tuple[int, int], *, decomposition: bool):
            group = f"full_{grid_shape[0]}x{grid_shape[1]}_{trials}_{'decomp' if decomposition else 'basic'}"

            def compute():
                grid = harness.make_grid(*grid_shape)
                per_ref = []
                all_actions = []
                overall_valid = True
                for ref_index, context in enumerate(self.contexts):
                    fiber_static = self._fiber_static(context, eta)
                    actions = []
                    maxima = {
                        "maximum_calibration_residual": 0.0,
                        "minimum_ess_fraction": float("inf"),
                        "maximum_mass_error": 0.0,
                        "maximum_source_compatibility_absolute": 0.0,
                        "maximum_poisson_relative_residual": 0.0,
                        "maximum_component_compatibility_residual": 0.0,
                        "maximum_component_count": 0,
                        "maximum_full_moment_rate_residual": 0.0,
                        "maximum_tangent_moment_rate_residual": 0.0,
                        "maximum_hidden_nullspace_residual": 0.0,
                        "maximum_orthogonality_absolute": 0.0,
                        "maximum_pythagorean_absolute": 0.0,
                        "maximum_raw_hierarchy_violation": 0.0,
                    }
                    trial_rows = []
                    features = (
                        context.exp.family.features(
                            grid.points(), jnp.asarray(eta, dtype=jnp.float64)
                        ) if decomposition else None
                    )
                    chunk_size = 4
                    for begin in range(0, trials, chunk_size):
                        trial_ids = list(range(begin, min(begin + chunk_size, trials)))
                        states = [
                            self._hard_fiber_state_from_static(context, eta, trial, fiber_static)
                            for trial in trial_ids
                        ]
                        raster = self._raster_state_chunk(states, ref_index, context, grid)
                        batch_count = len(states)
                        solved = harness.solve_v2(
                            raster["q"].reshape((batch_count * 21, grid.ny, grid.nx)),
                            raster["source"].reshape((batch_count * 21, grid.ny, grid.nx)),
                            grid,
                        )
                        action_matrix = np.asarray(solved.action, dtype=np.float64).reshape((batch_count, 21))
                        potential = np.asarray(solved.potential, dtype=np.float64).reshape((batch_count, 21, grid.ny, grid.nx))
                        poisson_residual = np.asarray(solved.relative_residual, dtype=np.float64).reshape((batch_count, 21))
                        compatibility = np.asarray(solved.maximum_component_compatibility_residual, dtype=np.float64).reshape((batch_count, 21))
                        component_count = np.asarray(solved.component_count).reshape((batch_count, 21))
                        converged = np.asarray(solved.solver_converged).reshape((batch_count, 21))
                        compatible = np.asarray(solved.compatible).reshape((batch_count, 21))
                        decomp_arrays = None
                        if decomposition:
                            decomp = harness.raster_tangent_projection(
                                jnp.asarray(potential, dtype=jnp.float64),
                                jnp.asarray(raster["q"], dtype=jnp.float64),
                                -jnp.asarray(raster["source"], dtype=jnp.float64),
                                features,
                                dx=float(grid.dx), cell_area=float(grid.cell_area),
                                pinv_rcond=1e-10, operator_floor_rel=0.0,
                                gauge_strength=0.0, source_is_density=True,
                            )
                            decomp_arrays = {
                                "full": np.asarray(decomp.full_moment_residual),
                                "tangent": np.asarray(decomp.tangent_moment_residual),
                                "hidden": np.asarray(decomp.hidden_moment_residual),
                                "orthogonality": np.asarray(decomp.tangent_hidden_inner_product),
                                "pythagorean": np.asarray(decomp.pythagorean_residual),
                                "hierarchy": np.asarray(decomp.hierarchy_raw_violation),
                            }
                        for local_index, (trial, state) in enumerate(zip(trial_ids, states)):
                            action_by_time = action_matrix[local_index]
                            action = float(np.sum(self.weights * action_by_time))
                            diag = {
                                "trial": trial,
                                "action": action,
                                "maximum_calibration_residual": float(np.max(state.calibration_residual)),
                                "minimum_ess_fraction": float(np.min(state.ess_fraction)),
                                "maximum_mass_error": float(np.max(np.abs(np.sum(raster["mass"][local_index], axis=(-2, -1)) - 1.0))),
                                "maximum_source_compatibility_absolute": float(np.max(np.abs(np.sum(raster["source"][local_index], axis=(-2, -1)) * grid.cell_area))),
                                "maximum_poisson_relative_residual": float(np.max(poisson_residual[local_index])),
                                "maximum_component_compatibility_residual": float(np.max(compatibility[local_index])),
                                "maximum_component_count": int(np.max(component_count[local_index])),
                                "solver_converged": bool(np.all(converged[local_index])),
                                "component_compatible": bool(np.all(compatible[local_index])),
                                "strictly_positive_q": bool(np.all(raster["q"][local_index] > 0.0)),
                            }
                            if decomp_arrays is not None:
                                diag.update({
                                    "maximum_full_moment_rate_residual": float(np.max(np.linalg.norm(decomp_arrays["full"][local_index], axis=-1))),
                                    "maximum_tangent_moment_rate_residual": float(np.max(np.linalg.norm(decomp_arrays["tangent"][local_index], axis=-1))),
                                    "maximum_hidden_nullspace_residual": float(np.max(np.linalg.norm(decomp_arrays["hidden"][local_index], axis=-1))),
                                    "maximum_orthogonality_absolute": float(np.max(np.abs(decomp_arrays["orthogonality"][local_index]))),
                                    "maximum_pythagorean_absolute": float(np.max(np.abs(decomp_arrays["pythagorean"][local_index]))),
                                    "maximum_raw_hierarchy_violation": float(np.max(decomp_arrays["hierarchy"][local_index])),
                                })
                            actions.append(action)
                            trial_rows.append(diag)
                            for key in maxima:
                                if key == "minimum_ess_fraction":
                                    maxima[key] = min(maxima[key], diag[key])
                                elif key in diag:
                                    maxima[key] = max(maxima[key], diag[key])
                            valid = (
                                np.isfinite(action)
                                and diag["maximum_calibration_residual"] <= float(self.gates["maximum_finite_calibration_residual"])
                                and diag["minimum_ess_fraction"] >= float(self.gates["minimum_ess_fraction"])
                                and diag["maximum_mass_error"] <= float(self.gates["maximum_mass_absolute_error"])
                                and diag["maximum_source_compatibility_absolute"] <= float(self.gates["maximum_source_compatibility_absolute"])
                                and diag["maximum_poisson_relative_residual"] <= float(self.gates["maximum_poisson_relative_residual"])
                                and diag["maximum_component_count"] == int(self.gates["required_conductive_component_count"])
                                and diag["solver_converged"] and diag["component_compatible"] and diag["strictly_positive_q"]
                            )
                            if decomposition:
                                valid = valid and (
                                    diag["maximum_full_moment_rate_residual"] <= float(self.gates["maximum_full_moment_rate_residual"])
                                    and diag["maximum_tangent_moment_rate_residual"] <= float(self.gates["maximum_tangent_moment_rate_residual"])
                                    and diag["maximum_hidden_nullspace_residual"] <= float(self.gates["maximum_hidden_nullspace_residual"])
                                    and diag["maximum_orthogonality_absolute"] <= float(self.gates["maximum_orthogonality_absolute"])
                                    and diag["maximum_pythagorean_absolute"] <= float(self.gates["maximum_pythagorean_absolute"])
                                    and diag["maximum_raw_hierarchy_violation"] <= float(self.gates["maximum_raw_hierarchy_violation"])
                                )
                            overall_valid = overall_valid and bool(valid)
                    reference_valid = bool(all(
                        np.isfinite(row["action"])
                        and row["maximum_calibration_residual"] <= float(self.gates["maximum_finite_calibration_residual"])
                        and row["minimum_ess_fraction"] >= float(self.gates["minimum_ess_fraction"])
                        and row["maximum_mass_error"] <= float(self.gates["maximum_mass_absolute_error"])
                        and row["maximum_source_compatibility_absolute"] <= float(self.gates["maximum_source_compatibility_absolute"])
                        and row["maximum_poisson_relative_residual"] <= float(self.gates["maximum_poisson_relative_residual"])
                        and row["maximum_component_count"] == int(self.gates["required_conductive_component_count"])
                        and row["solver_converged"] and row["component_compatible"] and row["strictly_positive_q"]
                        for row in trial_rows
                    ))
                    ref_value = float(np.mean(actions))
                    all_actions.append(ref_value)
                    per_ref.append({"reference_index": ref_index, "valid": reference_valid, "value": ref_value, "diagnostics": maxima, "trials": trial_rows})
                return {
                    "kind": "reflected_v2_full_action",
                    "grid": list(grid_shape), "trials": trials, "decomposition": decomposition,
                    "valid": bool(overall_valid),
                    "value": float(np.mean(all_actions)) if overall_valid else None,
                    "per_reference": per_ref,
                }
            return self._cached(group, eta, compute)

    harness.Evaluator = BatchedDecompositionEvaluator


def install_all(harness: Any, *, workers: int = 4) -> None:
    install_parallel_exact_solver(harness)
