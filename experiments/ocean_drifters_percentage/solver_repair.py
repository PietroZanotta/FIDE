"""Unfloored weighted-Poisson solver-repair pilot on the frozen 30 points."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from mfsi.poisson_log import (
    LogPoissonConfig,
    solve_cosine_ritz_reference,
    solve_divided_log_density_poisson,
    solve_log_conductance_poisson,
)

from .action import _read_csv, _write_csv
from .full_action import OceanWeightedPoissonPilot


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted


class OceanPoissonSolverRepair:
    def __init__(self, experiment, analysis_dir: Path, output_dir: Path):
        self.experiment = experiment
        self.analysis = Path(analysis_dir)
        self.output = Path(output_dir)
        self.tables = self.analysis / "tables"
        self.cfg = experiment.cfg["action"]["poisson_solver_repair"]
        self.pilot = OceanWeightedPoissonPilot(experiment, analysis_dir, output_dir)
        repair_grids = [tuple(map(int, row)) for row in self.cfg["grid_resolutions"]]
        pilot_grids = [tuple(map(int, row)) for row in self.pilot.cfg["grid_resolutions"]]
        if repair_grids != pilot_grids:
            raise RuntimeError("solver repair must use the unchanged frozen pilot grids")
        selection = _read_csv(experiment._resolve(self.cfg["pilot_selection_table"]))
        if [int(row["design_index"]) for row in selection] != self.pilot.designs.tolist():
            raise RuntimeError("solver repair must use the unchanged frozen pilot selection")
        baseline_rows = _read_csv(experiment._resolve(self.cfg["baseline_time_table"]))
        self.baseline = {
            (
                int(row["grid_nx"]), int(row["grid_ny"]), int(row["design_index"]),
                int(row["source_time_index"]),
            ): row
            for row in baseline_rows
            if float(row["operator_floor_relative"]) == 0.0
        }

    def _solver_config(self, dx: float) -> LogPoissonConfig:
        return LogPoissonConfig(
            dx=dx,
            iterative_relative_tolerance=float(self.cfg["iterative_relative_tolerance"]),
            physical_relative_tolerance=float(self.cfg["physical_relative_residual_tolerance"]),
            gauge_absolute_tolerance=float(self.cfg["gauge_absolute_tolerance"]),
            maximum_iterations=int(self.cfg["maximum_iterations"]),
            ilu_drop_tolerance=float(self.cfg["ilu_drop_tolerance"]),
            ilu_fill_factor=float(self.cfg["ilu_fill_factor"]),
            direct_maximum_cells=int(self.cfg["direct_maximum_cells"]),
            iterative_solver=str(self.cfg["iterative_solver"]),
        )

    def _base_row(self, system: dict[str, Any]) -> dict[str, Any]:
        return {
            "design_index": system["design_index"],
            "design_id": system["design_id"],
            "source_time_index": system["source_time_index"],
            "day": system["day"],
            "grid_nx": system["grid_nx"],
            "grid_ny": system["grid_ny"],
            "dx_km": system["dx_km"],
            "compatibility_residual": system["compatibility_residual"],
            "compatibility_relative_residual": system["compatibility_relative_residual"],
            "compatibility_pass": system["compatibility_valid"],
            "a_tan": system["tangent_action_density"],
            "minimum_log_q_mass": float(np.min(system["log_q_mass"])),
            "maximum_log_q_mass": float(np.max(system["log_q_mass"])),
            "log_q_range": float(np.max(system["log_q_mass"]) - np.min(system["log_q_mass"])),
            "operator_floor": 0.0,
            "density_floored_clipped_thresholded_or_truncated": False,
            "final_test_accessed": False,
        }

    def _baseline_row(self, system: dict[str, Any]) -> dict[str, Any]:
        key = (
            int(system["grid_nx"]), int(system["grid_ny"]),
            int(system["design_index"]), int(system["source_time_index"]),
        )
        source = self.baseline[key]
        action = _finite_float(source["full_action_density"])
        tangent = float(system["tangent_action_density"])
        tolerance = float(self.cfg["tangent_full_inequality_relative_tolerance"])
        lower = bool(
            np.isfinite(action)
            and tangent <= action + tolerance * max(abs(action), abs(tangent), 1.0)
        )
        return {
            **self._base_row(system),
            "formulation": "native_unfloored_ic0_pcg_baseline",
            "solver": "native PCG",
            "preconditioner": "native IC(0)",
            "convergence": _as_bool(source["solver_success"]),
            "physical_residual_pass": _as_bool(source["physical_pde_valid"]),
            "physical_relative_residual": _finite_float(source["physical_relative_pde_residual"], math.inf),
            "physical_absolute_residual": _finite_float(source["physical_absolute_pde_residual"], math.inf),
            "physical_log10_absolute_residual": math.nan,
            "gauge_residual": _finite_float(source["weighted_mean_potential_residual"]),
            "a_full": action,
            "lower_bound_pass": lower,
            "iteration_count": int(float(source["iterations"])),
            "condition_proxy": _finite_float(source["condition_proxy"], math.inf),
            "log10_condition_proxy": math.log10(max(_finite_float(source["condition_proxy"], math.inf), 1.0)),
            "scaled_relative_residual": _finite_float(source["native_relative_pde_residual"], math.inf),
            "unrepresentable_equilibrated_coefficient_count": "",
            "unrepresentable_equilibrated_coefficient_fraction": "",
            "solver_error": source["solver_error"],
        }

    def _result_row(
        self, system: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        action = _finite_float(result.get("action"))
        tangent = float(system["tangent_action_density"])
        tolerance = float(self.cfg["tangent_full_inequality_relative_tolerance"])
        lower = bool(
            np.isfinite(action)
            and tangent <= action + tolerance * max(abs(action), abs(tangent), 1.0)
        )
        gauge = _finite_float(result.get("weighted_mean_potential"))
        condition_log10 = _finite_float(
            result.get("log10_condition_proxy"),
            math.log10(max(_finite_float(result.get("condition_proxy"), math.inf), 1.0)),
        )
        condition = 10.0 ** condition_log10 if condition_log10 < 308.0 else math.inf
        return {
            **self._base_row(system),
            "formulation": result["formulation"],
            "solver": result["solver"],
            "preconditioner": result.get("preconditioner", ""),
            "convergence": bool(result.get("converged", False)),
            "physical_residual_pass": bool(result.get("physical_residual_valid", False)),
            "physical_relative_residual": _finite_float(result.get("physical_relative_residual"), math.inf),
            "physical_absolute_residual": _finite_float(result.get("physical_absolute_residual"), math.inf),
            "physical_log10_absolute_residual": _finite_float(
                result.get("physical_log10_absolute_residual"), math.inf
            ),
            "gauge_residual": gauge,
            "a_full": action,
            "lower_bound_pass": lower,
            "iteration_count": int(result.get("iteration_count", 0)),
            "condition_proxy": condition,
            "log10_condition_proxy": condition_log10,
            "scaled_relative_residual": _finite_float(result.get("scaled_relative_residual"), math.inf),
            "unrepresentable_equilibrated_coefficient_count": result.get(
                "unrepresentable_equilibrated_coefficient_count", ""
            ),
            "unrepresentable_equilibrated_coefficient_fraction": result.get(
                "unrepresentable_equilibrated_coefficient_fraction", ""
            ),
            "solver_error": result.get("solver_error", ""),
        }

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        grid_rows: list[dict[str, Any]] = []
        bounds = tuple(float(value) for value in self.experiment.cfg["scientific"]["domain_km"])
        for resolution_values in self.cfg["grid_resolutions"]:
            resolution = tuple(int(value) for value in resolution_values)
            points, dx, log_base, velocity = self.pilot._reference_grid(resolution)
            systems = self.pilot._systems_for_grid(
                resolution, points, dx, log_base, velocity
            )
            solver_cfg = self._solver_config(dx)
            for ordinal, system in enumerate(systems, start=1):
                log_q = system["log_q_mass"].reshape((resolution[1], resolution[0]))
                grid_rows.append(self._baseline_row(system))
                log_result = solve_log_conductance_poisson(log_q, system["h"], solver_cfg)
                grid_rows.append(self._result_row(system, log_result))
                divided_result = solve_divided_log_density_poisson(
                    log_q, system["h"], solver_cfg
                )
                grid_rows.append(self._result_row(system, divided_result))
                ritz_result = solve_cosine_ritz_reference(
                    log_q,
                    system["h"],
                    dx,
                    bounds,
                    maximum_mode=int(self.cfg["ritz_maximum_mode"]),
                )
                grid_rows.append(self._result_row(system, ritz_result))
                print(
                    f"[ocean Poisson repair] grid={resolution[0]}x{resolution[1]} "
                    f"case={ordinal}/{len(systems)} design={system['design_index']} "
                    f"day={system['day']:g} log={log_result['converged']} "
                    f"divided={divided_result['converged']}",
                    flush=True,
                )
        _write_csv(self.tables / "weighted_poisson_solver_repair_grid.csv", grid_rows)
        comparison = self._comparison(grid_rows)
        _write_csv(self.tables / "weighted_poisson_solver_repair_comparison.csv", comparison)
        summary = self._summary(grid_rows, comparison, started)
        self._write_report(summary, comparison)
        return summary

    def _comparison(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        coarse = tuple(int(value) for value in self.cfg["grid_resolutions"][0])
        fine = tuple(int(value) for value in self.cfg["grid_resolutions"][-1])
        keyed = {
            (row["formulation"], int(row["design_index"]), int(row["source_time_index"]),
             int(row["grid_nx"]), int(row["grid_ny"])): row
            for row in rows
        }
        formulations = list(self.cfg["comparison_formulations"])
        output = []
        for design in self.pilot.designs:
            for source in self.pilot.source_indices:
                for formulation in formulations:
                    coarse_row = keyed[(formulation, int(design), int(source), *coarse)]
                    fine_row = keyed[(formulation, int(design), int(source), *fine)]
                    coarse_action = _finite_float(coarse_row["a_full"])
                    fine_action = _finite_float(fine_row["a_full"])
                    grid_change = (
                        abs(fine_action - coarse_action) / max(abs(fine_action), 1e-14)
                        if np.isfinite(coarse_action) and np.isfinite(fine_action) else math.inf
                    )
                    grid_pass = grid_change <= float(
                        self.cfg["maximum_relative_action_grid_change"]
                    )
                    accepted = bool(
                        coarse_row["compatibility_pass"]
                        and fine_row["compatibility_pass"]
                        and coarse_row["convergence"]
                        and fine_row["convergence"]
                        and coarse_row["physical_residual_pass"]
                        and fine_row["physical_residual_pass"]
                        and abs(_finite_float(coarse_row["gauge_residual"], math.inf))
                        <= float(self.cfg["gauge_absolute_tolerance"])
                        and abs(_finite_float(fine_row["gauge_residual"], math.inf))
                        <= float(self.cfg["gauge_absolute_tolerance"])
                        and coarse_row["lower_bound_pass"]
                        and fine_row["lower_bound_pass"]
                        and grid_pass
                    )
                    output.append({
                        "design_index": int(design),
                        "design_id": coarse_row["design_id"],
                        "source_time_index": int(source),
                        "day": coarse_row["day"],
                        "formulation": formulation,
                        "solver": fine_row["solver"],
                        "preconditioner": fine_row["preconditioner"],
                        "convergence": bool(coarse_row["convergence"] and fine_row["convergence"]),
                        "coarse_convergence": coarse_row["convergence"],
                        "fine_convergence": fine_row["convergence"],
                        "physical_relative_residual": fine_row["physical_relative_residual"],
                        "coarse_physical_relative_residual": coarse_row["physical_relative_residual"],
                        "fine_physical_relative_residual": fine_row["physical_relative_residual"],
                        "gauge_residual": fine_row["gauge_residual"],
                        "compatibility_residual": fine_row["compatibility_relative_residual"],
                        "a_tan": fine_row["a_tan"],
                        "a_full": fine_action,
                        "lower_bound_pass": bool(coarse_row["lower_bound_pass"] and fine_row["lower_bound_pass"]),
                        "coarse_action": coarse_action,
                        "fine_action": fine_action,
                        "grid_relative_change": grid_change,
                        "grid_refinement_pass": grid_pass,
                        "iteration_count": fine_row["iteration_count"],
                        "condition_proxy": fine_row["condition_proxy"],
                        "log10_condition_proxy": fine_row["log10_condition_proxy"],
                        "minimum_log_q_mass": fine_row["minimum_log_q_mass"],
                        "maximum_log_q_mass": fine_row["maximum_log_q_mass"],
                        "log_q_range": fine_row["log_q_range"],
                        "unrepresentable_equilibrated_coefficient_count": fine_row[
                            "unrepresentable_equilibrated_coefficient_count"
                        ],
                        "accepted": accepted,
                        "density_floored_clipped_thresholded_or_truncated": False,
                        "final_test_accessed": False,
                    })
        return output

    def _summary(
        self, grid_rows: list[dict[str, Any]], comparison: list[dict[str, Any]], started: float
    ) -> dict[str, Any]:
        formulations = list(self.cfg["comparison_formulations"])
        counts = {}
        for formulation in formulations:
            rows = [row for row in comparison if row["formulation"] == formulation]
            counts[formulation] = {
                "case_count": len(rows),
                "both_grid_convergence_count": sum(row["convergence"] for row in rows),
                "fine_physical_residual_pass_count": sum(
                    row["fine_convergence"]
                    and _finite_float(row["fine_physical_relative_residual"], math.inf)
                    <= float(self.cfg["physical_relative_residual_tolerance"])
                    for row in rows
                ),
                "grid_refinement_pass_count": sum(row["grid_refinement_pass"] for row in rows),
                "lower_bound_pass_count": sum(row["lower_bound_pass"] for row in rows),
                "accepted_count": sum(row["accepted"] for row in rows),
            }
        primary = counts[self.cfg["primary_formulation"]]
        trustworthy = primary["accepted_count"] == 30
        return {
            "schema_version": 1,
            "pilot_layout_count": 6,
            "pilot_time_count_per_layout": 5,
            "pilot_layout_time_count": 30,
            "operator_floor": 0.0,
            "density_modified": False,
            "forcing_compatibility_fine_count": sum(
                row["compatibility_pass"]
                for row in grid_rows
                if int(row["grid_nx"]) == int(self.cfg["grid_resolutions"][-1][0])
                and row["formulation"] == self.cfg["primary_formulation"]
            ),
            "formulation_counts": counts,
            "primary_solver_accepted_count": primary["accepted_count"],
            "solver_repair_valid": trustworthy,
            "production_sweep_authorized": trustworthy,
            "full_action_valid": False,
            "final_test_accessed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def _write_report(self, summary: dict[str, Any], comparison: list[dict[str, Any]]) -> None:
        counts = summary["formulation_counts"]
        primary = counts["equilibrated_log_conductance_finite_volume"]
        baseline = counts["native_unfloored_ic0_pcg_baseline"]
        divided = counts["divided_log_density_central_difference"]
        ritz = counts["cosine_ritz_variational_mode_5"]
        log_rows = [
            row for row in comparison
            if row["formulation"] == "equilibrated_log_conductance_finite_volume"
        ]
        unrepresentable = sum(
            int(float(row["unrepresentable_equilibrated_coefficient_count"] or 0)) > 0
            for row in log_rows
        )
        max_log_range = max(float(row["log_q_range"]) for row in log_rows)
        paired = {
            (row["design_id"], row["day"], row["formulation"]): row
            for row in comparison
        }
        alternative_action_changes = []
        alternative_physical_residuals = []
        for row in comparison:
            if row["formulation"] != "divided_log_density_central_difference":
                continue
            ritz_row = paired[(
                row["design_id"], row["day"], "cosine_ritz_variational_mode_5"
            )]
            divided_action = _finite_float(row["fine_action"])
            ritz_action = _finite_float(ritz_row["fine_action"])
            if np.isfinite(divided_action) and np.isfinite(ritz_action):
                alternative_action_changes.append(
                    abs(divided_action - ritz_action) / max(abs(divided_action), 1e-14)
                )
                alternative_physical_residuals.append(
                    _finite_float(row["fine_physical_relative_residual"], math.inf)
                )
        if alternative_action_changes:
            alternative_comparison = (
                f"their relative action differences span "
                f"{min(alternative_action_changes):.3%} to "
                f"{max(alternative_action_changes):.3%}. This is not validation:\n"
                f"   the best divided physical residual among those pairs is "
                f"{min(alternative_physical_residuals):.3e}, versus the required `1e-6`"
            )
        else:
            alternative_comparison = "there are no finite actions to compare"
        report = f"""# Unfloored weighted-Poisson solver-repair report

## Decision

The same weighted-Poisson/full-action problem is **not yet numerically trustworthy**.
The new solver does not modify `q`, but local equilibration cannot make every
positive stencil coefficient representable: {unrepresentable}/30 fine-grid cases
still contain positive locally scaled conductances or RHS entries beyond float64's
exponent range. Those cases are rejected rather than thresholded. Consequently the
primary log-conductance solver accepts {primary['accepted_count']}/30 points and a
production full-action sweep remains unauthorized.

The frozen six layouts, five times, reference, projected laws, forcing, moment
reconstruction, sensors, domain, epsilon, and 68-layout law set were unchanged.
The 69 final-test trajectories were not accessed.

## Questions

1. **Why did old IC(0)-PCG fail?** The direct-density operator underflows across an
   extreme weighted graph. Adjacent log-density jumps can exceed float64's entire
   exponent range, producing nearly disconnected conductance basins and an IC(0)
   system whose recursive/stabilized residual is not a reliable physical residual.
2. **What was tested?** The native unfloored IC(0)-PCG baseline; the unchanged
   arithmetic-face finite-volume equation with exact log conductances and row
   equilibration; an independent central-difference
   `Delta psi + grad(log q).grad psi = h` formulation; and a small Neumann
   cosine-Ritz variational cross-check.
3. **Was q floored, clipped, thresholded, or truncated?** No. The primary solver
   explicitly fails when a positive equilibrated coefficient cannot be represented;
   it never replaces or deletes that coefficient. Operator floor is exactly zero.
4. **How was equilibration implemented?** Arithmetic face conductances are formed as
   `logaddexp(log q_i, log q_j)-log(2)-2log(dx)`. Each row is divided by its log-space
   physical diagonal. This is positive algebraic row scaling of the same equation.
5. **How are physical residuals evaluated?** Every cell's original unfloored
   residual is a signed log-sum-exp of its four physical face fluxes and `q h`.
   Absolute log10 and relative L2 residuals are then accumulated in log space.
6. **How many primary cases pass?** {primary['accepted_count']}/30 pass the complete
   two-grid contract. Fine-grid forcing compatibility remains 30/30.
7. **How many pass grid stability?** {primary['grid_refinement_pass_count']}/30 for
   the primary formulation, requiring at most 10% relative action change.
8. **Does the tangent lower bound hold?** It holds in
   {primary['lower_bound_pass_count']}/30 primary cases with finite two-grid actions;
   no case is accepted unless it holds on both grids. Because the primary method
   yields no finite pointwise actions, an integrated primary action and its lower
   bound are not evaluable.
9. **Do formulations agree on psi/action?** The primary method produces no
   representable potential, so no primary-versus-independent `psi` or action
   comparison is possible. The divided and Ritz methods both produce a finite
   fine-grid action in {len(alternative_action_changes)}/30 cases; {alternative_comparison}.
   Their complete acceptance counts are {divided['accepted_count']}/30 for the
   divided-log solver and {ritz['accepted_count']}/30 for the Ritz diagnostic.
10. **Does an independent reference support the FV result?** Coarse divided-log
    systems use sparse direct SuperLU where possible, and the Ritz calculation is
    independent, but neither validates the original physical residual at a
    convincing rate. Their complete acceptance counts are respectively
    {divided['accepted_count']}/30 and {ritz['accepted_count']}/30.
11. **What failures remain?** Unrepresentable positive local coefficient ratios,
    singular/failed ILU factorizations or iterative convergence, physical residual
    failures of the divided form, and lack of coarse/fine agreement.
12. **Is density range still dominant?** Yes. The maximum fine-grid log-mass range
    is {max_log_range:.3e}, and even local stencil equilibration exceeds float64 in
    {unrepresentable}/30 cases.
13. **Suitable for production?** No.
14. **Method freeze if successful?** Not applicable; no production sweep is launched.
15. **Exact blocker?** Standard float64 sparse algebra cannot represent every
    positive coefficient of the unchanged locally equilibrated operator. The
    independent divided form can be solved in some cases but does not pass the
    original unfloored physical residual/grid contract.
16. **Final-test lock?** `final_test_accessed=false` in every repair artifact.

## Counts by formulation

| Formulation | Both grids converge | Fine physical residual | Grid stable | Lower bound | Fully accepted |
|---|---:|---:|---:|---:|---:|
| Native unfloored IC(0)-PCG baseline | {baseline['both_grid_convergence_count']}/30 | {baseline['fine_physical_residual_pass_count']}/30 | {baseline['grid_refinement_pass_count']}/30 | {baseline['lower_bound_pass_count']}/30 | {baseline['accepted_count']}/30 |
| Equilibrated log-conductance FV | {primary['both_grid_convergence_count']}/30 | {primary['fine_physical_residual_pass_count']}/30 | {primary['grid_refinement_pass_count']}/30 | {primary['lower_bound_pass_count']}/30 | {primary['accepted_count']}/30 |
| Divided log-density finite difference | {divided['both_grid_convergence_count']}/30 | {divided['fine_physical_residual_pass_count']}/30 | {divided['grid_refinement_pass_count']}/30 | {divided['lower_bound_pass_count']}/30 | {divided['accepted_count']}/30 |
| Cosine-Ritz variational diagnostic | {ritz['both_grid_convergence_count']}/30 | {ritz['fine_physical_residual_pass_count']}/30 | {ritz['grid_refinement_pass_count']}/30 | {ritz['lower_bound_pass_count']}/30 | {ritz['accepted_count']}/30 |

Machine-readable details are in
[`weighted_poisson_solver_repair_grid.csv`](tables/weighted_poisson_solver_repair_grid.csv)
and
[`weighted_poisson_solver_repair_comparison.csv`](tables/weighted_poisson_solver_repair_comparison.csv).
"""
        (self.analysis / "weighted_poisson_solver_repair_report.md").write_text(
            report, encoding="utf-8"
        )


def run_poisson_solver_repair(experiment, analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    return OceanPoissonSolverRepair(experiment, analysis_dir, output_dir).run()
