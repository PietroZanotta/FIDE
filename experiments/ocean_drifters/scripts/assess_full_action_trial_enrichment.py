"""Assess sensor-feature enrichment of the frozen ocean Ritz trial space."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mfsi.config import load_config
from experiments.ocean_drifters.action import _features, _read_csv, _write_csv
from experiments.ocean_drifters.experiment import OceanDriftersExperiment
from experiments.ocean_drifters.full_action_production import (
    OceanFullActionProduction,
)
from experiments.ocean_drifters.scripts.assess_full_action_rank_repair import (
    _representatives,
)


def _cosine_basis(
    points: np.ndarray, bounds: np.ndarray, maximum_mode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    phase_x = np.pi * (points[:, 0] - xmin) / (xmax - xmin)
    phase_y = np.pi * (points[:, 1] - ymin) / (ymax - ymin)
    values = []
    gradient_x = []
    gradient_y = []
    for y_mode in range(maximum_mode + 1):
        for x_mode in range(maximum_mode + 1):
            if x_mode == 0 and y_mode == 0:
                continue
            x_phase = x_mode * phase_x
            y_phase = y_mode * phase_y
            values.append(np.cos(x_phase) * np.cos(y_phase))
            gradient_x.append(
                -(x_mode * np.pi / (xmax - xmin))
                * np.sin(x_phase) * np.cos(y_phase)
            )
            gradient_y.append(
                -(y_mode * np.pi / (ymax - ymin))
                * np.cos(x_phase) * np.sin(y_phase)
            )
    return (
        np.column_stack(values),
        np.column_stack(gradient_x),
        np.column_stack(gradient_y),
    )


def _sensor_basis(
    experiment: OceanDriftersExperiment,
    points: np.ndarray,
    design: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = experiment.sensor_bank.centers_km[design]
    sigma = experiment.sensor_bank.sigma_km
    values = _features(points, centers, sigma)
    delta = points[:, None] - centers[None]
    gradient = -(delta / sigma**2) * values[:, :, None]
    return values, gradient[:, :, 0], gradient[:, :, 1]


def _spectral_actions(
    values: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    weights: np.ndarray,
    forcing: np.ndarray,
    tolerances: tuple[float, ...],
) -> dict[float, dict[str, float | int]]:
    mean = weights @ values
    load = (values - mean).T @ (weights * forcing)
    gram = (
        gradient_x.T @ (weights[:, None] * gradient_x)
        + gradient_y.T @ (weights[:, None] * gradient_y)
    )
    diagonal = np.diag(gram)
    scale = np.zeros_like(diagonal)
    scale[diagonal > 0.0] = 1.0 / np.sqrt(diagonal[diagonal > 0.0])
    scaled_gram = scale[:, None] * gram * scale[None]
    scaled_load = scale * load
    eigenvalues, eigenvectors = np.linalg.eigh(scaled_gram)
    maximum = float(eigenvalues[-1])
    output = {}
    for tolerance in tolerances:
        retained = eigenvalues > tolerance * max(
            maximum, np.finfo(np.float64).tiny
        )
        projection = eigenvectors[:, retained].T @ scaled_load
        action = float(np.sum(projection**2 / eigenvalues[retained]))
        output[tolerance] = {
            "action": action,
            "retained_rank": int(np.sum(retained)),
            "condition_proxy": (
                maximum / float(eigenvalues[retained][0])
                if np.any(retained) else math.inf
            ),
        }
    return output


def main() -> None:
    cfg = load_config(EXPERIMENT_DIR / "config.json")
    experiment = OceanDriftersExperiment(cfg)
    production = OceanFullActionProduction(
        experiment,
        EXPERIMENT_DIR / "analysis",
        EXPERIMENT_DIR / "outputs/full_action_production",
    )
    production_rows = _read_csv(
        EXPERIMENT_DIR / "analysis/tables/full_action_production_time.csv"
    )
    cases = _representatives(production_rows)
    bounds = np.asarray(cfg["scientific"]["domain_km"], dtype=np.float64)
    tolerances = tuple(float(value) for value in cfg["action"][
        "variational_poisson"
    ]["rank_sensitivity_relative_tolerances"])
    primary = float(cfg["action"]["variational_poisson"][
        "rank_relative_tolerance"
    ])
    output = []
    for source in sorted({int(case["source_time_index"]) for case in cases}):
        local_cases = [
            case for case in cases if int(case["source_time_index"]) == source
        ]
        designs = np.asarray([
            int(case["design_index"]) for case in local_cases
        ], dtype=int)
        production.runner.source_indices = np.asarray([source], dtype=int)
        production.runner.designs = designs
        points, dx, log_base, velocity = production.runner._reference_grid(
            production.resolution,
            source_indices=production.runner.source_indices,
            cache_namespace="full_action_production_reference",
        )
        systems = production.runner._systems_for_grid(
            production.resolution, points, dx, log_base, velocity
        )
        labels = {
            int(case["design_index"]): str(case["case_label"])
            for case in local_cases
        }
        cosine = _cosine_basis(points, bounds, maximum_mode=5)
        for system in systems:
            design = int(system["design_index"])
            sensor = _sensor_basis(experiment, points, design)
            families = {
                "sensor_features_only": sensor,
                "cosine_mode_5_only": cosine,
                "cosine_mode_5_plus_sensor_features": tuple(
                    np.column_stack((cosine[index], sensor[index]))
                    for index in range(3)
                ),
            }
            weights = np.exp(
                system["log_q_mass"] - logsumexp(system["log_q_mass"])
            )
            for family, (values, grad_x, grad_y) in families.items():
                actions = _spectral_actions(
                    values,
                    grad_x,
                    grad_y,
                    weights,
                    system["h"].ravel(),
                    tolerances,
                )
                primary_action = float(actions[primary]["action"])
                rank_change = max(
                    abs(float(entry["action"]) - primary_action)
                    / max(abs(float(entry["action"])), abs(primary_action), 1e-14)
                    for tolerance, entry in actions.items()
                    if tolerance != primary
                )
                tangent = float(system["tangent_action_density"])
                output.append({
                    "case_label": labels[design],
                    "design_index": design,
                    "design_id": system["design_id"],
                    "source_time_index": source,
                    "day": float(system["day"]),
                    "trial_family": family,
                    "basis_size": values.shape[1],
                    "retained_rank_primary": int(
                        actions[primary]["retained_rank"]
                    ),
                    "condition_proxy_primary": float(
                        actions[primary]["condition_proxy"]
                    ),
                    "tangent_action_density": tangent,
                    "full_action_density_primary": primary_action,
                    "full_to_tangent_ratio": primary_action / tangent,
                    "maximum_relative_rank_action_change": rank_change,
                    "rank_sensitivity_valid": rank_change <= 0.05,
                    "tangent_full_inequality_valid": tangent <= primary_action,
                    **{
                        f"rank_tolerance_{tolerance:.0e}_action": float(
                            entry["action"]
                        )
                        for tolerance, entry in actions.items()
                    },
                    "density_modified": False,
                    "operator_floor": 0.0,
                    "final_test_accessed": False,
                })
    _write_csv(
        EXPERIMENT_DIR
        / "analysis/tables/full_action_rank_repair_trial_enrichment.csv",
        output,
    )
    print(f"wrote {len(output)} frozen trial-space enrichment audits")


if __name__ == "__main__":
    main()
