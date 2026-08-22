"""Ocean-local conductivity regularization diagnostics.

The regularized problem is deliberately a different estimand from the
unregularized MFSI Poisson equation.  On a cell grid it is

    -div((q + epsilon * max(q)) grad(psi_epsilon)) = -q h,

with the original q-weighted gauge.  The reported physical action remains
``E_q[|grad psi_epsilon|^2]``; the added uniform-conductivity energy is
reported separately.  This is the same regularization already implemented by
the repository's finite-volume ``operator_floor_rel`` option.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg

from .direct_qr_ritz import PreparedDirectRitzBasis


@dataclass(frozen=True)
class ConductivityRegularizedRitzResult:
    """Small-system Ritz audit of the explicitly regularized equation."""

    physical_action: float
    regularization_action: float
    operator_action: float
    dual_action: float
    action_identity_relative_error: float
    spectral_operator_action: float
    cholesky_spectral_relative_difference: float
    coefficient_relative_difference: float
    condition_number: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    coefficients: np.ndarray
    spectral_coefficients: np.ndarray
    success: bool


def normalized_trapezoid_weights(times: np.ndarray) -> np.ndarray:
    """Match the normalized trapezoid convention used by vortices."""
    values = np.asarray(times, dtype=np.float64).ravel()
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("times must contain at least two finite values")
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("times must be strictly increasing")
    weights = np.zeros_like(values)
    weights[0] = 0.5 * differences[0]
    weights[-1] = 0.5 * differences[-1]
    if len(values) > 2:
        weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    weights /= np.sum(weights)
    return weights


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


class OceanPostDispersionAction:
    """Vortices-shaped read API for the frozen ocean action audit.

    The ocean benchmark has fixed, predeclared layouts rather than continuous
    layout parameters and repeated observation banks.  Consequently the
    vortex-style ``exact_*_result`` methods take a frozen ``design_index`` and
    return ``{valid, value, rows}`` over the deterministic action window.
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        details_path: Path,
        design_ids: Sequence[str],
        temporal_details_path: Path | None = None,
        temporal_summary_path: Path | None = None,
    ) -> None:
        self.cfg = dict(cfg)
        self.details_path = Path(details_path)
        self.design_ids = tuple(str(value) for value in design_ids)
        self.start_day = float(self.cfg["window_start_day_inclusive"])
        self.end_day = float(self.cfg["window_end_day_inclusive"])
        self.audit_days = np.asarray(self.cfg["audit_days"], dtype=np.float64)
        self.days = self.audit_days.copy()
        self.primary_floor = float(
            self.cfg["primary_operator_floor_relative"]
        )
        horizon = float(self.cfg["source_horizon_days"])
        self.time_scale = (self.end_day - self.start_day) / horizon
        self.normalized_times = (self.days - self.start_day) / (
            self.end_day - self.start_day
        )
        self.time_weights = normalized_trapezoid_weights(self.normalized_times)
        self.action_density_scale = self.time_scale * self.time_scale
        with self.details_path.open(newline="", encoding="utf-8") as handle:
            self._rows = list(csv.DictReader(handle))
        self._validate()
        self._audit_primary_by_design = self._primary_by_design
        self._audit_weights = self.time_weights.copy()
        if temporal_details_path is not None or temporal_summary_path is not None:
            if temporal_details_path is None or temporal_summary_path is None:
                raise ValueError("both temporal-refinement artifacts are required")
            self._load_temporal_refinement(
                Path(temporal_details_path), Path(temporal_summary_path)
            )
        else:
            self._temporal_summary = None

    def _validate(self) -> None:
        if self.start_day != 12.0 or self.end_day != 45.0:
            raise ValueError("the frozen ocean action window must be [12,45] days")
        if float(self.cfg["source_horizon_days"]) != 45.0:
            raise ValueError("the source ocean path must retain its 45-day horizon")
        if self.cfg.get("time_parameterization") != "s=(day-12)/(45-12)":
            raise ValueError("unknown post-dispersion time parameterization")
        if self.cfg.get("integration_rule") != "normalized_trapezoid_over_s":
            raise ValueError("unknown post-dispersion integration rule")
        if self.cfg.get("temporal_quadrature_status") != (
            "seven_node_audit_not_refinement_certified"
        ):
            raise ValueError("unknown post-dispersion temporal quadrature status")
        if self.cfg.get("primary_action") != "unregularized_direct_qr_ritz":
            raise ValueError("the primary post-dispersion action changed")
        if not np.array_equal(
            self.audit_days,
            np.asarray([12.0, 12.5, 15.0, 22.5, 30.0, 37.5, 45.0]),
        ):
            raise ValueError("the frozen post-dispersion audit nodes changed")
        expected_row_count = (
            int(self.cfg["layout_count"])
            * len(self.audit_days)
            * len(self.cfg["grid_resolutions"])
            * len(self.cfg["operator_floor_relative_values"])
        )
        if len(self._rows) != expected_row_count:
            raise ValueError(
                "the frozen post-dispersion audit row count changed: "
                f"expected {expected_row_count}, got {len(self._rows)}"
            )
        if any(
            _as_bool(row.get("production_run"))
            or _as_bool(row.get("scientific_ranking_performed"))
            or _as_bool(row.get("final_test_accessed"))
            for row in self._rows
        ):
            raise ValueError("post-dispersion audit violates its safety locks")
        designs = sorted({int(row["design_index"]) for row in self._rows})
        if len(designs) != int(self.cfg["layout_count"]):
            raise ValueError("post-dispersion audit layout count changed")
        expected_days = set(self.audit_days.tolist())
        if {float(row["day"]) for row in self._rows} != expected_days:
            raise ValueError("post-dispersion audit days changed")
        expected_grids = {
            tuple(int(value) for value in resolution)
            for resolution in self.cfg["grid_resolutions"]
        }
        if {
            (int(row["grid_nx"]), int(row["grid_ny"])) for row in self._rows
        } != expected_grids:
            raise ValueError("post-dispersion audit grids changed")
        expected_floors = {
            float(value) for value in self.cfg["operator_floor_relative_values"]
        }
        if {
            float(row["operator_floor_relative"]) for row in self._rows
        } != expected_floors:
            raise ValueError("post-dispersion regularization ladder changed")
        if not all(_as_bool(row["combined_floor_case_valid"]) for row in self._rows):
            raise ValueError("post-dispersion audit contains an invalid floor case")
        for design in designs:
            if not (0 <= design < len(self.design_ids)):
                raise ValueError("post-dispersion audit contains an unknown design")
            expected_id = self.design_ids[design]
            if any(
                row["design_id"] != expected_id
                for row in self._rows
                if int(row["design_index"]) == design
            ):
                raise ValueError("post-dispersion design ID/index mismatch")
        self.design_indices = tuple(designs)

        fine_nx = int(self.cfg["grid_resolutions"][-1][0])
        primary = [
            row for row in self._rows
            if int(row["grid_nx"]) == fine_nx
            and float(row["operator_floor_relative"]) == self.primary_floor
        ]
        self._primary_by_design: dict[int, list[dict[str, str]]] = {}
        for design in designs:
            local = sorted(
                (row for row in primary if int(row["design_index"]) == design),
                key=lambda row: float(row["day"]),
            )
            if len(local) != len(self.audit_days) or not np.array_equal(
                np.asarray([float(row["day"]) for row in local]), self.audit_days
            ):
                raise ValueError("post-dispersion primary series is incomplete")
            self._primary_by_design[design] = local

    def _load_temporal_refinement(
        self, details_path: Path, summary_path: Path
    ) -> None:
        with details_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("temporal_quadrature_refinement_certified") is not True
            or summary.get("production_authorized") is not True
            or summary.get("final_test_accessed") is not False
            or summary.get("scientific_ranking_performed") is not False
            or int(summary.get("local_case_count", -1)) != 133
            or int(summary.get("local_valid_count", -1)) != 133
        ):
            raise ValueError("temporal-refinement summary is not production-ready")
        designs = sorted({int(row["design_index"]) for row in rows})
        if tuple(designs) != self.design_indices or len(rows) != 133:
            raise ValueError("temporal-refinement design panel changed")
        ordered = sorted(rows, key=lambda row: int(row["source_time_index"]))
        if not np.array_equal(
            np.asarray([int(row["source_time_index"]) for row in ordered]),
            np.arange(48, 181, dtype=int),
        ):
            raise ValueError("temporal-refinement source grid changed")
        if not all(
            _as_bool(row["local_valid"])
            and not _as_bool(row["production_run"])
            and not _as_bool(row["scientific_ranking_performed"])
            and not _as_bool(row["final_test_accessed"])
            for row in ordered
        ):
            raise ValueError("temporal-refinement local validity changed")
        self.days = np.asarray([float(row["day"]) for row in ordered])
        self.normalized_times = (self.days - self.start_day) / (
            self.end_day - self.start_day
        )
        self.time_weights = normalized_trapezoid_weights(self.normalized_times)
        self._primary_by_design = {designs[0]: ordered}
        levels = summary.get("levels", [])
        if [int(level["node_count"]) for level in levels] != [12, 23, 45, 133]:
            raise ValueError("temporal-refinement level ladder changed")
        finest = levels[-1]
        for field, summary_field in (
            ("tangent_action", "tangent_action"),
            ("direct_action_qr", "full_action"),
        ):
            density = self.action_density_scale * np.asarray(
                [float(row[field]) for row in ordered], dtype=np.float64
            )
            integrated = float(self.time_weights @ density)
            if not np.isclose(
                integrated,
                float(finest[summary_field]),
                rtol=1e-13,
                atol=1e-9,
            ):
                raise ValueError("temporal-refinement summary disagrees with details")
        self._temporal_summary = summary

    def _design(self, design_index: int) -> tuple[int, list[dict[str, str]]]:
        design = int(design_index)
        try:
            return design, self._primary_by_design[design]
        except KeyError as exc:
            raise ValueError(
                f"design {design} is not in the frozen post-dispersion panel"
            ) from exc

    def _scaled_series(
        self, design_index: int, field: str
    ) -> tuple[int, np.ndarray, list[dict[str, Any]]]:
        design, source = self._design(design_index)
        density = self.action_density_scale * np.asarray(
            [float(row[field]) for row in source], dtype=np.float64
        )
        rows = [
            {
                "design_index": design,
                "design_id": self.design_ids[design],
                "source_time_index": int(row["source_time_index"]),
                "day": float(row["day"]),
                "window_normalized_time": float(self.normalized_times[index]),
                "time_weight": float(self.time_weights[index]),
                "action_density": float(density[index]),
            }
            for index, row in enumerate(source)
        ]
        return design, density, rows

    def exact_tangent_result(self, design_index: int) -> dict[str, Any]:
        """Return the integrated tangent action in the window parameterization."""
        design, density, rows = self._scaled_series(design_index, "tangent_action")
        value = float(self.time_weights @ density)
        source = self._primary_by_design[design]
        valid = bool(
            np.isfinite(density).all()
            and value >= 0.0
            and all(_as_bool(row.get("local_valid", True)) for row in source)
        )
        return {
            "valid": valid,
            "value": value if valid else math.inf,
            "rows": rows,
            "design_index": design,
            "design_id": self.design_ids[design],
            "window_days": [self.start_day, self.end_day],
        }

    def exact_full_result(self, design_index: int) -> dict[str, Any]:
        """Return the primary unregularized direct-QR window action."""
        design, density, rows = self._scaled_series(design_index, "direct_action_qr")
        value = float(self.time_weights @ density)
        source = self._primary_by_design[design]
        valid = bool(
            np.isfinite(density).all()
            and value >= 0.0
            and all(
                _as_bool(row.get("local_valid", row.get("direct_reference_valid")))
                for row in source
            )
        )
        return {
            "valid": valid,
            "value": value if valid else math.inf,
            "rows": rows,
            "design_index": design,
            "design_id": self.design_ids[design],
            "window_days": [self.start_day, self.end_day],
            "formulation": "unregularized_direct_qr_ritz",
        }

    def evaluate_layout_exact(self, design_index: int) -> dict[str, Any]:
        """Evaluate primary and diagnostic actions for one frozen layout."""
        design, source = self._design(design_index)
        diagnostic_source = self._audit_primary_by_design[design]
        tangent = self.exact_tangent_result(design)
        full = self.exact_full_result(design)
        audit_direct_density = self.action_density_scale * np.asarray([
            float(row["direct_action_qr"]) for row in diagnostic_source
        ])
        regularized_ritz_density = self.action_density_scale * np.asarray([
            float(row["regularized_ritz_physical_action"])
            for row in diagnostic_source
        ])
        regularized_fv_density = self.action_density_scale * np.asarray([
            float(row["fv_physical_action"]) for row in diagnostic_source
        ])
        audit_direct = float(self._audit_weights @ audit_direct_density)
        regularized_ritz = float(self._audit_weights @ regularized_ritz_density)
        regularized_fv = float(self._audit_weights @ regularized_fv_density)
        scale = max(abs(full["value"]), abs(tangent["value"]), 1.0)
        lower_bound_valid = bool(
            tangent["value"]
            <= full["value"]
            + float(self.cfg["tangent_full_relative_tolerance"]) * scale
        )
        valid = bool(
            tangent["valid"]
            and full["valid"]
            and lower_bound_valid
            and all(_as_bool(row["local_valid"]) for row in source)
            and all(
                _as_bool(row["combined_floor_case_valid"])
                for row in diagnostic_source
            )
        )
        return {
            "design_index": design,
            "design_id": self.design_ids[design],
            "valid": valid,
            "tangent_action": tangent["value"],
            "full_action": full["value"],
            "seven_node_unregularized_action_diagnostic": audit_direct,
            "regularized_ritz_action_diagnostic": regularized_ritz,
            "regularized_fv_action_diagnostic": regularized_fv,
            "integrated_regularization_bias": symmetric_relative_difference(
                audit_direct, regularized_ritz
            ),
            "integrated_fv_ritz_difference": symmetric_relative_difference(
                regularized_fv, regularized_ritz
            ),
            "tangent_lower_bound_valid": lower_bound_valid,
            "maximum_node_regularization_bias": max(
                float(row[
                    "regularization_bias_relative_to_unregularized_ritz"
                ]) for row in diagnostic_source
            ),
            "maximum_node_fv_ritz_difference": max(
                float(row["fv_regularized_ritz_relative_action_difference"])
                for row in diagnostic_source
            ),
            "window_days": [self.start_day, self.end_day],
            "action_time_parameterization": self.cfg["time_parameterization"],
            "primary_formulation": "unregularized_direct_qr_ritz",
            "regularization_diagnostic_time_node_count": len(diagnostic_source),
        }

    def evaluate_layouts_exact(self) -> list[dict[str, Any]]:
        """Evaluate every frozen layout without ranking or selection."""
        return [self.evaluate_layout_exact(design) for design in self.design_indices]

    def result_payload(self) -> dict[str, Any]:
        """Return a saved-result block analogous to the vortices validation block."""
        layouts = self.evaluate_layouts_exact()
        valid_count = sum(bool(row["valid"]) for row in layouts)
        return {
            "schema_version": 1,
            "window_days": [self.start_day, self.end_day],
            "source_horizon_days": float(self.cfg["source_horizon_days"]),
            "time_parameterization": self.cfg["time_parameterization"],
            "time_scale": self.time_scale,
            "action_density_scale": self.action_density_scale,
            "integration_rule": self.cfg["integration_rule"],
            "temporal_quadrature_status": (
                "nested_12_23_45_133_certified"
                if self._temporal_summary is not None
                else self.cfg["temporal_quadrature_status"]
            ),
            "temporal_quadrature_refinement_certified": bool(
                self._temporal_summary is not None
            ),
            "audit_days": self.audit_days.tolist(),
            "production_time_grid_days": self.days.tolist(),
            "time_weights": self.time_weights.tolist(),
            "primary_formulation": "unregularized_direct_qr_ritz",
            "regularized_equation_diagnostic": self.cfg["regularized_equation"],
            "primary_operator_floor_relative_diagnostic": self.primary_floor,
            "layout_count": len(layouts),
            "valid_layout_count": valid_count,
            "all_layouts_valid": valid_count == len(layouts),
            "full_action_production_valid": bool(
                self._temporal_summary is not None
                and valid_count == len(layouts)
            ),
            "temporal_refinement": self._temporal_summary,
            "layouts": layouts,
            "production_run": bool(self._temporal_summary is not None),
            "scientific_ranking_performed": False,
            "final_test_accessed": False,
        }


def symmetric_relative_difference(left: float, right: float) -> float:
    """Symmetric relative difference used by the frozen ocean audits."""
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), np.finfo(np.float64).tiny
    )


def post_dispersion_source_indices(
    normalized_times: np.ndarray,
    *,
    start_day_inclusive: float,
    end_day_inclusive: float,
) -> np.ndarray:
    """Return frozen-grid indices in ``[start_day, end_day]``."""
    days = np.asarray(normalized_times, dtype=np.float64).ravel() * 45.0
    if not np.isfinite(days).all() or np.any(np.diff(days) <= 0.0):
        raise ValueError("normalized_times must be finite and strictly increasing")
    if not (math.isfinite(start_day_inclusive) and math.isfinite(end_day_inclusive)):
        raise ValueError("window bounds must be finite")
    if start_day_inclusive >= end_day_inclusive:
        raise ValueError("the post-dispersion window must have positive length")
    return np.flatnonzero(
        (days >= float(start_day_inclusive))
        & (days <= float(end_day_inclusive))
    )


def solve_conductivity_regularized_ritz(
    prepared: PreparedDirectRitzBasis,
    physical_weights: np.ndarray,
    forcing: np.ndarray,
    operator_floor_relative: float,
) -> ConductivityRegularizedRitzResult:
    """Solve and independently audit the regularized Ritz system.

    The projected-law masses are normalized to one.  Adding
    ``epsilon * max(q)`` to a density at every equal-area cell is therefore
    represented exactly by adding ``epsilon * max(weights)`` to each cell's
    stiffness weight.  The RHS, gauge measure, and reported physical action
    continue to use the unmodified projected law.
    """
    weights = np.asarray(physical_weights, dtype=np.float64).ravel()
    h = np.asarray(forcing, dtype=np.float64).ravel()
    epsilon = float(operator_floor_relative)
    if len(weights) != len(prepared.values) or h.shape != weights.shape:
        raise ValueError("prepared basis, weights, and forcing disagree")
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isfinite(h).all()
    ):
        raise ValueError("weights and forcing must be finite; weights nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("operator_floor_relative must be finite and positive")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("physical weights must have finite positive mass")
    weights = weights / total

    physical_mean = weights @ prepared.values
    raw_load = -(prepared.values - physical_mean).T @ (weights * h)
    load = prepared.raw_to_whitened.T @ raw_load

    gx = prepared.gradient_x_whitened
    gy = prepared.gradient_y_whitened
    floor_mass = epsilon * float(np.max(weights))
    operator_weights = weights + floor_mass
    physical_stiffness = (
        gx.T @ (weights[:, None] * gx)
        + gy.T @ (weights[:, None] * gy)
    )
    regularization_stiffness = floor_mass * (gx.T @ gx + gy.T @ gy)
    operator = physical_stiffness + regularization_stiffness
    operator = 0.5 * (operator + operator.T)

    eigenvalues, eigenvectors = linalg.eigh(operator, check_finite=True)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    positive = bool(minimum > 0.0 and np.isfinite(eigenvalues).all())
    if not positive:
        size = len(load)
        nan = np.full(size, np.nan)
        return ConductivityRegularizedRitzResult(
            physical_action=math.nan,
            regularization_action=math.nan,
            operator_action=math.nan,
            dual_action=math.nan,
            action_identity_relative_error=math.inf,
            spectral_operator_action=math.nan,
            cholesky_spectral_relative_difference=math.inf,
            coefficient_relative_difference=math.inf,
            condition_number=math.inf,
            minimum_eigenvalue=minimum,
            maximum_eigenvalue=maximum,
            coefficients=nan,
            spectral_coefficients=nan.copy(),
            success=False,
        )

    factor = linalg.cho_factor(operator, lower=True, check_finite=True)
    coefficients = linalg.cho_solve(factor, load, check_finite=True)
    spectral_load = eigenvectors.T @ load
    spectral_coordinates = spectral_load / eigenvalues
    spectral_coefficients = eigenvectors @ spectral_coordinates
    spectral_action = float(np.sum(spectral_load * spectral_load / eigenvalues))

    physical_action = float(coefficients @ physical_stiffness @ coefficients)
    regularization_action = float(
        coefficients @ regularization_stiffness @ coefficients
    )
    operator_action = physical_action + regularization_action
    dual_action = float(coefficients @ load)
    identity_error = symmetric_relative_difference(operator_action, dual_action)
    coefficient_difference = float(
        np.linalg.norm(coefficients - spectral_coefficients)
        / max(np.linalg.norm(coefficients), np.finfo(np.float64).tiny)
    )
    cholesky_spectral = symmetric_relative_difference(
        operator_action, spectral_action
    )
    # Certification targets the action and the weak energy identity.  The
    # coefficient comparison is retained as a conditioning diagnostic, but is
    # gauge/basis-coordinate dependent and is not an action-validity gate.
    success = bool(
        np.isfinite(
            [physical_action, regularization_action, operator_action, dual_action]
        ).all()
        and physical_action >= 0.0
        and regularization_action >= 0.0
        and identity_error <= 1.0e-10
        and cholesky_spectral <= 1.0e-10
    )
    return ConductivityRegularizedRitzResult(
        physical_action=physical_action,
        regularization_action=regularization_action,
        operator_action=operator_action,
        dual_action=dual_action,
        action_identity_relative_error=identity_error,
        spectral_operator_action=spectral_action,
        cholesky_spectral_relative_difference=cholesky_spectral,
        coefficient_relative_difference=coefficient_difference,
        condition_number=maximum / minimum,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        coefficients=coefficients,
        spectral_coefficients=spectral_coefficients,
        success=success,
    )


__all__ = [
    "ConductivityRegularizedRitzResult",
    "OceanPostDispersionAction",
    "normalized_trapezoid_weights",
    "post_dispersion_source_indices",
    "solve_conductivity_regularized_ritz",
    "symmetric_relative_difference",
]
