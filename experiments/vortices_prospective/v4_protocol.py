from __future__ import annotations

"""Preregistration, immutable-input preparation, and time-unit audits for v4."""

import copy
import json
from pathlib import Path
import shutil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from build_prospective_data import build
from common import SCRIPT_DIR, artifact_dirs, fingerprint, write_json_atomic
from evaluator import ProspectiveEvaluator
from mfsi.cache import file_sha256
from mfsi.measurements import GaussianPointSensors2D
from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from physical import truth_from_config
from prospective_data import TargetProspectiveData
from train_reference import train_and_rollout

jax.config.update("jax_enable_x64", True)


V4_SOURCE_FILES = (
    "v4_objective.py",
    "v4_protocol.py",
    "v4_select.py",
    "v4_validate.py",
    "run_v4.py",
)


def v4_source_hash() -> str:
    return fingerprint(
        {
            name: file_sha256(SCRIPT_DIR / name)
            for name in V4_SOURCE_FILES
            if (SCRIPT_DIR / name).exists()
        }
    )


def _artifact_compatibility_view(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(cfg[key])
        for key in (
            "seed",
            "truth",
            "time",
            "measurement",
            "aggregate_predictor",
            "qoi",
            "moment_reconstruction",
            "reference_training",
            "reference",
        )
    }


def prepare_v4_inputs(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Materialize immutable aggregate/reference inputs without touching v3."""
    dirs = artifact_dirs(output_dir)
    dirs["endpoint"].mkdir(parents=True, exist_ok=True)
    dirs["prospective"].mkdir(parents=True, exist_ok=True)
    source_value = cfg["v4"].get("artifact_source_run")
    if not source_value:
        build_receipt = build(cfg, output_dir)
        reference_receipt = train_and_rollout(cfg, output_dir)
        receipt = {
            "schema_version": 1,
            "mode": "new_smoke_artifacts",
            "build_receipt": build_receipt,
            "reference_receipt": reference_receipt,
        }
        write_json_atomic(dirs["prospective"] / "v4_input_receipt.json", receipt)
        return receipt

    source_root = (SCRIPT_DIR / str(source_value)).resolve()
    source_manifest_path = source_root / "results" / "frozen_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"declared immutable source manifest is missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "frozen_before_hidden_validation":
        raise RuntimeError("artifact source is not a sealed pre-validation manifest")
    if _artifact_compatibility_view(source_manifest["config"]) != _artifact_compatibility_view(cfg):
        raise RuntimeError("v4 scientific inputs differ from the declared immutable artifact source")

    copies = {
        source_root / "endpoint_reference" / "endpoint_data.npz": dirs["endpoint"] / "endpoint_data.npz",
        source_root / "endpoint_reference" / "reference_checkpoint.npz": dirs["endpoint"] / "reference_checkpoint.npz",
        source_root / "endpoint_reference" / "reference_rollout.npz": dirs["endpoint"] / "reference_rollout.npz",
        source_root / "endpoint_reference" / "reference_receipt.json": dirs["endpoint"] / "reference_receipt.json",
        source_root / "prospective" / "aggregate_predictions.npz": dirs["prospective"] / "aggregate_predictions.npz",
        source_root / "prospective" / "build_receipt.json": dirs["prospective"] / "source_build_receipt.json",
    }
    copied = {}
    for source, destination in copies.items():
        if not source.exists():
            raise FileNotFoundError(f"declared immutable source artifact is missing: {source}")
        source_sha = file_sha256(source)
        if destination.exists():
            if file_sha256(destination) != source_sha:
                raise RuntimeError(f"v4 input copy differs from immutable source: {destination}")
        else:
            shutil.copy2(source, destination)
        copied[destination.name] = {
            "source": str(source),
            "source_sha256": source_sha,
            "copied_sha256": file_sha256(destination),
        }
    receipt = {
        "schema_version": 1,
        "mode": "immutable_copy_from_prior_prevalidation_artifacts",
        "source_run": str(source_root),
        "source_frozen_manifest": str(source_manifest_path),
        "source_frozen_manifest_sha256": file_sha256(source_manifest_path),
        "old_hidden_validation_accessed": False,
        "copied": copied,
    }
    write_json_atomic(dirs["prospective"] / "v4_input_receipt.json", receipt)
    return receipt


def run_time_derivative_audit(
    cfg: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Audit normalized-time, observable-Jacobian, and spline derivative units."""
    dirs = artifact_dirs(output_dir)
    dirs["results"].mkdir(parents=True, exist_ok=True)
    data = TargetProspectiveData.load(
        dirs["endpoint"] / "endpoint_data.npz",
        dirs["prospective"] / "aggregate_predictions.npz",
    )
    evaluator = ProspectiveEvaluator(
        {**cfg, "projection": {**cfg["projection"], "backend": "jax"}},
        data,
        dirs["endpoint"] / "reference_rollout.npz",
    )
    truth = truth_from_config(cfg)
    points = jnp.asarray(
        [[0.35, 0.25], [0.78, 0.62], [1.25, 0.31], [1.67, 0.74]],
        dtype=jnp.float64,
    )
    t = jnp.asarray(0.37, dtype=jnp.float64)
    normalized_velocity = truth.velocity(points, t)
    block = cfg["truth"]
    tau = float(block["horizon"]) * t
    omega = 2.0 * jnp.pi / float(block["period"])
    a = float(block["epsilon"]) * jnp.sin(omega * tau)
    b = 1.0 - 2.0 * a
    xx, yy = points[:, 0], points[:, 1]
    f = a * xx * xx + b * xx
    dfdx = 2.0 * a * xx + b
    vx_physical = -jnp.pi * float(block["amplitude"]) * jnp.sin(jnp.pi * f) * jnp.cos(jnp.pi * yy)
    vy_physical = jnp.pi * float(block["amplitude"]) * jnp.cos(jnp.pi * f) * jnp.sin(jnp.pi * yy) * dfdx
    expected_normalized = float(block["horizon"]) * jnp.stack(
        [vx_physical, vy_physical], axis=-1
    )
    velocity_error = float(jnp.max(jnp.abs(normalized_velocity - expected_normalized)))

    sensors = GaussianPointSensors2D(
        width=float(cfg["measurement"]["sensor_width"]),
        n_sensors=int(cfg["measurement"]["n_sensors"]),
    )
    eta = jnp.asarray(
        [0.38, 0.30, 0.82, 0.68, 1.23, 0.32, 1.62, 0.69],
        dtype=jnp.float64,
    )
    jacobian_rate = jnp.einsum(
        "nmd,nd->nm", sensors.feature_gradients(points, eta), normalized_velocity
    )
    h = 1.0e-6
    directional_fd = (
        sensors.features(points + h * normalized_velocity, eta)
        - sensors.features(points - h * normalized_velocity, eta)
    ) / (2.0 * h)
    observable_rate_error = float(jnp.max(jnp.abs(jacobian_rate - directional_fd)))

    t_obs = np.asarray(evaluator.times)[np.asarray(evaluator.acq_idx)]
    reconstructor = AnchoredCubicSplineReconstructor(
        t_obs,
        np.asarray(evaluator.times),
        AnchoredCubicSplineConfig(
            internal_knots=int(cfg["moment_reconstruction"]["internal_knots"]),
            smoothing=0.0,
            ridge_rel=float(cfg["moment_reconstruction"]["ridge_rel"]),
            roughness_quadrature_order=int(cfg["moment_reconstruction"]["roughness_quadrature_order"]),
        ),
    )
    intercept = jnp.asarray([0.2, 0.3, 0.4, 0.5], dtype=jnp.float64)
    slope = jnp.asarray([0.11, -0.07, 0.03, -0.09], dtype=jnp.float64)
    y = intercept[None, :] + jnp.asarray(t_obs)[:, None] * slope[None, :]
    fit = reconstructor.reconstruct(y, intercept, intercept + slope)
    spline_value_error = float(
        jnp.max(
            jnp.abs(
                fit.c
                - (intercept[None, :] + evaluator.times[:, None] * slope[None, :])
            )
        )
    )
    spline_derivative_error = float(jnp.max(jnp.abs(fit.c_dot - slope[None, :])))
    tolerance = float(cfg["v4"]["time_audit_tolerance"])
    result = {
        "schema_version": 1,
        "normalized_time_interval": [float(evaluator.times[0]), float(evaluator.times[-1])],
        "physical_time_horizon": float(block["horizon"]),
        "velocity_scaling_rule": "dX/dt_normalized = horizon * dX/dtau_physical",
        "max_normalized_velocity_scaling_error": velocity_error,
        "max_observable_jacobian_velocity_error": observable_rate_error,
        "max_linear_spline_value_error": spline_value_error,
        "max_linear_spline_derivative_error": spline_derivative_error,
        "endpoint_derivative_convention": "anchored cubic derivative evaluated analytically at both endpoints",
        "one_sided_finite_difference_used": False,
        "reference_velocity_units": "state units per normalized scientific time",
        "spline_derivative_units": "moment units per normalized scientific time",
        "tolerance": tolerance,
    }
    result["passed"] = bool(
        max(
            velocity_error,
            observable_rate_error,
            spline_value_error,
            spline_derivative_error,
        )
        <= tolerance
    )
    if not result["passed"]:
        raise RuntimeError(f"v4 time/derivative audit failed: {result}")
    write_json_atomic(dirs["results"] / "time_derivative_audit.json", result)
    return result


__all__ = [
    "V4_SOURCE_FILES",
    "prepare_v4_inputs",
    "run_time_derivative_audit",
    "v4_source_hash",
]
