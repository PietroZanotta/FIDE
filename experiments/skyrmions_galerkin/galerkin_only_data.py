"""Selection-sealed production artifact loaders for the Galerkin-only study."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import IProjectionConfig

from .domain import ConfigurationBank, SkyrmionConfig
from .forcing import ForcingConfig
from .full_gradient import FrozenEtaProblem, projected_law_risk
from .measurements import LocalDensitySensors
from .production_artifacts import require_production_output_path
from .risk import many_body_features, whitening_from_truth

Array = jax.Array


class GalerkinReferenceBank(NamedTuple):
    configurations: Array
    velocity: Array
    base_weights: Array


@dataclass(frozen=True)
class SelectionGalerkinData:
    selection_problem: FrozenEtaProblem
    projection_bank: GalerkinReferenceBank
    train_bank: GalerkinReferenceBank
    audit_bank: GalerkinReferenceBank
    reference_features: Array
    truth_means: Array
    whitening: Array


@dataclass(frozen=True)
class ValidationGalerkinData:
    validation_problem: FrozenEtaProblem
    fit_bank: GalerkinReferenceBank
    audit_bank: GalerkinReferenceBank
    reference_features: Array
    truth_means: Array
    whitening: Array


def _physics_config(cfg: dict[str, Any]) -> SkyrmionConfig:
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(
        tuple(row) for row in values["pinning_centers"]
    )
    return SkyrmionConfig(**values)


def _time_weights(times: Array) -> Array:
    delta = jnp.diff(times)
    weights = jnp.concatenate([
        delta[:1] / 2.0,
        (delta[:-1] + delta[1:]) / 2.0,
        delta[-1:] / 2.0,
    ])
    return weights / jnp.sum(weights)


def _reference_bank(path: Path) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def _projection_config(cfg: dict[str, Any]) -> tuple[IProjectionConfig, str]:
    values = dict(cfg["projection"])
    backend = str(values.pop("trajectory_backend", "jax"))
    allowed = {item.name for item in fields(IProjectionConfig)}
    return IProjectionConfig(**{
        key: value for key, value in values.items() if key in allowed
    }), backend


def _forcing_config(cfg: dict[str, Any]) -> ForcingConfig:
    allowed = {item.name for item in fields(ForcingConfig)}
    return ForcingConfig(**{
        key: value for key, value in cfg["forcing"].items() if key in allowed
    })


def _acquisition_indices(time_count: int, acquisition_count: int) -> Array:
    values = tuple(
        round(index * (time_count - 1) / (acquisition_count - 1))
        for index in range(acquisition_count)
    )
    if acquisition_count < 2 or acquisition_count > time_count:
        raise ValueError("acquisition_count must be in [2, time_count]")
    if len(set(values)) != acquisition_count:
        raise ValueError("acquisition grid contains duplicate time indices")
    return jnp.asarray(values, dtype=jnp.int32)


def _make_problem(
    cfg: dict[str, Any], truth: Array, times: Array,
    family: LocalDensitySensors, *, noise_seed: int,
) -> FrozenEtaProblem:
    acquisition = _acquisition_indices(
        len(times), int(cfg["measurement"]["acquisition_count"])
    )
    reconstructor = AnchoredCubicSplineReconstructor(
        jax.device_get(times[acquisition]),
        jax.device_get(times),
        AnchoredCubicSplineConfig(**cfg["moment_reconstruction"]),
    )
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(int(noise_seed)),
        (len(acquisition), family.n_sensors),
        dtype=jnp.float64,
    )
    projection, backend = _projection_config(cfg)
    return FrozenEtaProblem(
        truth_configurations=jnp.asarray(truth, dtype=jnp.float64),
        times=jnp.asarray(times, dtype=jnp.float64),
        time_weights=_time_weights(times),
        acquisition_indices=acquisition,
        finite_configuration_count=min(
            int(cfg["measurement"]["finite_configurations"]), int(truth.shape[1])
        ),
        detector_noise=noise,
        family=family,
        reconstructor=reconstructor,
        projection_config=projection,
        forcing_config=_forcing_config(cfg),
        projection_backend=backend,
        box=tuple(cfg["physics"]["box"]),
    )


def _times_and_truth(path: Path, key: str) -> tuple[Array, Array]:
    # Selection callers request only ``design``; the sealed ``validation`` key
    # in this shared archive is never read by the selection loader.
    with np.load(path, allow_pickle=False) as arrays:
        return (
            jnp.asarray(arrays["times"], dtype=jnp.float64),
            jnp.asarray(arrays[key], dtype=jnp.float64),
        )


def _family(cfg: dict[str, Any]) -> LocalDensitySensors:
    physics = _physics_config(cfg)
    return LocalDensitySensors(
        n_sensors=int(cfg["measurement"]["n_sensors"]),
        width=float(cfg["measurement"]["sensor_width"]),
        box=tuple(physics.box),
        min_separation=float(cfg["measurement"]["min_separation"]),
    )


def _verify_times(cfg: dict[str, Any], times: Array) -> None:
    expected = jnp.linspace(
        0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64
    )
    if not bool(jnp.array_equal(times, expected)):
        raise RuntimeError("frozen production time grid does not match config")


def load_selection_galerkin_data(
    cfg: dict[str, Any], artifact_dir: Path,
) -> SelectionGalerkinData:
    """Load selection truth and selection banks without validation arrays."""

    artifact_dir = require_production_output_path(artifact_dir)
    times, truth = _times_and_truth(artifact_dir / "truth_banks.npz", "design")
    _verify_times(cfg, times)
    family = _family(cfg)
    noise_seed = int(cfg["seed"]) + int(cfg["banks"]["seed_offsets"]["observation"])
    problem = _make_problem(cfg, truth, times, family, noise_seed=noise_seed)
    projection = _reference_bank(artifact_dir / "reference_bank_projection.npz")
    train = _reference_bank(artifact_dir / "reference_bank_ritz_train.npz")
    audit = _reference_bank(artifact_dir / "reference_bank_ritz_audit.npz")
    truth_features = many_body_features(truth, tuple(cfg["physics"]["box"]))
    return SelectionGalerkinData(
        selection_problem=problem,
        projection_bank=projection,
        train_bank=train,
        audit_bank=audit,
        reference_features=many_body_features(
            projection.configurations, tuple(cfg["physics"]["box"])
        ),
        truth_means=jnp.mean(truth_features, axis=1),
        whitening=whitening_from_truth(truth_features),
    )


def load_validation_galerkin_data(
    cfg: dict[str, Any], artifact_dir: Path,
) -> ValidationGalerkinData:
    """Open validation artifacts only after the caller verifies a frozen winner."""

    artifact_dir = require_production_output_path(artifact_dir)
    truth_path = artifact_dir / "truth_banks.npz"
    times, validation_truth = _times_and_truth(truth_path, "validation")
    _, selection_truth = _times_and_truth(truth_path, "design")
    _verify_times(cfg, times)
    family = _family(cfg)
    noise_seed = (
        int(cfg["seed"]) + int(cfg["banks"]["seed_offsets"]["observation"]) + 10000
    )
    problem = _make_problem(
        cfg, validation_truth, times, family, noise_seed=noise_seed
    )
    fit = _reference_bank(artifact_dir / "reference_bank_validation_fit.npz")
    audit = _reference_bank(artifact_dir / "reference_bank_validation_audit.npz")
    box = tuple(cfg["physics"]["box"])
    return ValidationGalerkinData(
        validation_problem=problem,
        fit_bank=fit,
        audit_bank=audit,
        reference_features=many_body_features(fit.configurations, box),
        truth_means=jnp.mean(many_body_features(validation_truth, box), axis=1),
        whitening=whitening_from_truth(many_body_features(selection_truth, box)),
    )


def selection_risk(eta: Array, data: SelectionGalerkinData) -> Array:
    return projected_law_risk(
        eta, data.selection_problem, data.projection_bank,
        data.reference_features, data.truth_means, data.whitening,
    )


def validation_risk(eta: Array, data: ValidationGalerkinData) -> Array:
    return projected_law_risk(
        eta, data.validation_problem, data.fit_bank,
        data.reference_features, data.truth_means, data.whitening,
    )


__all__ = [
    "GalerkinReferenceBank", "SelectionGalerkinData", "ValidationGalerkinData",
    "load_selection_galerkin_data", "load_validation_galerkin_data",
    "selection_risk", "validation_risk",
]
