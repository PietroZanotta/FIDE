"""Fresh-bank generation and sealed validation for the official Pareto sweep."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.moments import AnchoredCubicSplineConfig, AnchoredCubicSplineReconstructor
from mfsi.projection import IProjectionConfig

from .domain import SkyrmionTruth
from .forcing import ForcingConfig
from .full_gradient import FrozenEtaProblem, forcing_state, projected_law_risk, reconstruct_moments, wrap_periodic
from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
from .galerkin_only import GalerkinCertificateThresholds, _forcing_state_payload, device_payload
from .galerkin_only_data import GalerkinReferenceBank, ValidationGalerkinData
from .measurements import LocalDensitySensors
from .official_pareto_common import (
    ALLOWANCES, ARTIFACT_DIR, DICTIONARY_PATH, OFFICIAL_K, OUTPUT_ROOT,
    allowance_slug, read_json, require_frozen_protocol,
    common_solver_reduction, strict_validation_ceiling, validation_ceiling,
    validation_classification, write_json,
)
from .production_artifacts import file_sha256
from .production_basis import load_dictionary
from .production_galerkin import _normalized_chunk, assemble_hybrid_system, audit_hybrid_solutions
from .reference import load_reference
from .risk import many_body_features, whitening_from_truth


FRESH_ROOT = OUTPUT_ROOT / "fresh_validation"
FRESH_TRUTH = FRESH_ROOT / "truth_validation.npz"
FRESH_FIT = FRESH_ROOT / "reference_validation_fit.npz"
FRESH_AUDIT = FRESH_ROOT / "reference_validation_audit.npz"
FRESH_NOISE = FRESH_ROOT / "measurement_noise.npz"
FRESH_MANIFEST = FRESH_ROOT / "artifact_manifest.json"


def _physics_config(cfg: dict[str, Any]):
    from .domain import SkyrmionConfig
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def _time_weights(times: jax.Array) -> jax.Array:
    delta = jnp.diff(times)
    values = jnp.concatenate([delta[:1] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1:] / 2])
    return values / jnp.sum(values)


def _selection_seal() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    selection_path = OUTPUT_ROOT / "selection" / "pareto_selection.json"
    manifest_path = OUTPUT_ROOT / "selection" / "manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("fresh validation requires a frozen selection manifest")
    selection, manifest = read_json(selection_path), read_json(manifest_path)
    selection_hash, manifest_hash = file_sha256(selection_path), file_sha256(manifest_path)
    if not selection.get("selection_frozen") or selection.get("validation_accessed", True):
        raise RuntimeError("selection is not sealed against validation access")
    if manifest.get("pareto_selection_sha256") != selection_hash:
        raise RuntimeError("selection manifest hash mismatch")
    if manifest.get("winning_etas") != [row["winner"]["eta"] for row in selection["allowances"]]:
        raise RuntimeError("selection winner manifest mismatch")
    return selection, manifest, selection_hash, manifest_hash


def _save_npz_once(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite fresh validation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in arrays.items()})


def _rollout_reference_bank(flow: Any, truth_model: SkyrmionTruth, times: jax.Array,
                            seed: int, samples: int, substeps: int,
                            *, chunk_size: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = truth_model.sample_initial(jax.random.PRNGKey(int(seed)), int(samples))
    configurations, velocities = [], []
    for start in range(0, int(samples), int(chunk_size)):
        stop = min(start + int(chunk_size), int(samples))
        rows = flow.rollout(initial[start:stop], times, substeps_per_interval=int(substeps))
        velocity = flow.velocity(rows, times)
        configurations.append(np.asarray(rows))
        velocities.append(np.asarray(velocity))
    configurations_np = np.concatenate(configurations, axis=1)
    velocities_np = np.concatenate(velocities, axis=1)
    weights = np.full((len(times), int(samples)), 1.0 / float(samples), dtype=np.float64)
    return configurations_np, velocities_np, weights


def _row_hashes(rows: np.ndarray) -> set[str]:
    flat = np.ascontiguousarray(rows).reshape((rows.shape[0], -1))
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in flat}


def _npz_initial(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        values = np.asarray(arrays[key])
        return values[0] if values.ndim == 4 else values


def _disjointness_audit() -> dict[str, Any]:
    fresh = {
        "truth": _row_hashes(_npz_initial(FRESH_TRUTH, "configurations")),
        "reference_fit": _row_hashes(_npz_initial(FRESH_FIT, "configurations")),
        "reference_audit": _row_hashes(_npz_initial(FRESH_AUDIT, "configurations")),
    }
    old = {
        "truth_selection": _row_hashes(_npz_initial(ARTIFACT_DIR / "truth_banks.npz", "design")),
        "truth_old_validation": _row_hashes(_npz_initial(ARTIFACT_DIR / "truth_banks.npz", "validation")),
    }
    for name in ("projection", "ritz_train", "ritz_audit", "validation_fit", "validation_audit"):
        old[f"reference_{name}"] = _row_hashes(
            _npz_initial(ARTIFACT_DIR / f"reference_bank_{name}.npz", "configurations")
        )
    comparisons = []
    for fresh_name, fresh_hashes in fresh.items():
        for old_name, old_hashes in old.items():
            overlap = fresh_hashes & old_hashes
            comparisons.append({
                "fresh": fresh_name, "existing": old_name,
                "exact_initial_row_overlap_count": len(overlap), "disjoint": not overlap,
            })
    for left, right in (("truth", "reference_fit"), ("truth", "reference_audit"), ("reference_fit", "reference_audit")):
        overlap = fresh[left] & fresh[right]
        comparisons.append({
            "fresh": left, "existing": f"fresh_{right}",
            "exact_initial_row_overlap_count": len(overlap), "disjoint": not overlap,
        })
    return {"passed": all(row["disjoint"] for row in comparisons), "comparisons": comparisons}


def generate_fresh_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    selection, selection_manifest, selection_hash, manifest_hash = _selection_seal()
    if FRESH_MANIFEST.is_file():
        previous = read_json(FRESH_MANIFEST)
        if (previous.get("selection_sha256") == selection_hash
                and previous.get("protocol_sha256") == protocol["protocol_sha256"]
                and previous.get("passed")):
            for row in previous["artifacts"]:
                path = OUTPUT_ROOT / row["relative_path"]
                if file_sha256(path) != row["sha256"]:
                    raise RuntimeError("fresh validation artifact changed after sealing")
            return {**previous, "cache_hit": True}
        raise RuntimeError("incompatible fresh validation manifest exists")
    FRESH_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(FRESH_ROOT / "generation_seal.json", {
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": selection_hash, "selection_manifest_sha256": manifest_hash,
        "all_six_winners_frozen_before_generation": True,
        "winning_etas": selection_manifest["winning_etas"],
    }, overwrite=False)
    seed_rows = {row["label"]: row for row in protocol["fresh_validation"]["seeds"]}
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth_model = SkyrmionTruth(_physics_config(cfg))
    truth = truth_model.make_bank(
        seed=int(seed_rows["truth"]["seed"]),
        samples=int(protocol["fresh_validation"]["truth_samples"]), times=times,
        substeps_per_interval=int(protocol["fresh_validation"]["truth_substeps"]),
    )
    _save_npz_once(FRESH_TRUTH, times=times, configurations=truth.configurations)
    family = LocalDensitySensors(
        n_sensors=int(cfg["measurement"]["n_sensors"]),
        width=float(cfg["measurement"]["sensor_width"]),
        box=tuple(cfg["physics"]["box"]),
        min_separation=float(cfg["measurement"]["min_separation"]),
    )
    acquisition_count = int(cfg["measurement"]["acquisition_count"])
    noise = float(cfg["measurement"]["observation_noise_std"]) * jax.random.normal(
        jax.random.PRNGKey(int(seed_rows["measurement_noise"]["seed"])),
        (acquisition_count, family.n_sensors), dtype=jnp.float64,
    )
    _save_npz_once(FRESH_NOISE, detector_noise=noise)
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    for label, samples, path in (
        ("reference_fit", int(protocol["fresh_validation"]["reference_fit_samples"]), FRESH_FIT),
        ("reference_audit", int(protocol["fresh_validation"]["reference_audit_samples"]), FRESH_AUDIT),
    ):
        configurations, velocity, weights = _rollout_reference_bank(
            flow, truth_model, times, int(seed_rows[label]["seed"]), samples,
            int(protocol["fresh_validation"]["reference_substeps"]),
        )
        _save_npz_once(path, configurations=configurations, velocity=velocity, base_weights=weights)
    disjointness = _disjointness_audit()
    artifact_paths = (FRESH_TRUTH, FRESH_NOISE, FRESH_FIT, FRESH_AUDIT)
    result = {
        "schema_version": 1, "ran": True, "passed": bool(disjointness["passed"]),
        "cache_hit": False, "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": selection_hash, "selection_manifest_sha256": manifest_hash,
        "all_six_winners_frozen_before_generation": True,
        "validation_generated_after_selection_freeze": True,
        "reference_checkpoint_sha256": file_sha256(ARTIFACT_DIR / "reference.npz"),
        "reference_retrained": False, "seed_records": list(seed_rows.values()),
        "disjointness": disjointness,
        "artifacts": [{
            "relative_path": str(path.relative_to(OUTPUT_ROOT)),
            "bytes": path.stat().st_size, "sha256": file_sha256(path),
        } for path in artifact_paths],
    }
    write_json(FRESH_MANIFEST, result, overwrite=False)
    if not result["passed"]:
        raise RuntimeError("fresh validation bank failed exact disjointness audit")
    return result


def _load_bank(path: Path) -> GalerkinReferenceBank:
    with np.load(path, allow_pickle=False) as arrays:
        return GalerkinReferenceBank(
            jnp.asarray(arrays["configurations"], dtype=jnp.float64),
            jnp.asarray(arrays["velocity"], dtype=jnp.float64),
            jnp.asarray(arrays["base_weights"], dtype=jnp.float64),
        )


def load_fresh_validation(cfg: dict[str, Any]) -> ValidationGalerkinData:
    manifest = read_json(FRESH_MANIFEST)
    if not manifest.get("passed"):
        raise RuntimeError("fresh validation artifact manifest did not pass")
    for row in manifest["artifacts"]:
        if file_sha256(OUTPUT_ROOT / row["relative_path"]) != row["sha256"]:
            raise RuntimeError("fresh validation artifact hash mismatch")
    with np.load(FRESH_TRUTH, allow_pickle=False) as arrays:
        times = jnp.asarray(arrays["times"], dtype=jnp.float64)
        truth = jnp.asarray(arrays["configurations"], dtype=jnp.float64)
    with np.load(FRESH_NOISE, allow_pickle=False) as arrays:
        noise = jnp.asarray(arrays["detector_noise"], dtype=jnp.float64)
    acquisition_count = int(cfg["measurement"]["acquisition_count"])
    acquisition = jnp.asarray(tuple(
        round(index * (len(times) - 1) / (acquisition_count - 1))
        for index in range(acquisition_count)
    ), dtype=jnp.int32)
    family = LocalDensitySensors(
        int(cfg["measurement"]["n_sensors"]), float(cfg["measurement"]["sensor_width"]),
        tuple(cfg["physics"]["box"]), float(cfg["measurement"]["min_separation"]),
    )
    projection_values = dict(cfg["projection"])
    backend = str(projection_values.pop("trajectory_backend", "jax"))
    projection_allowed = {item.name for item in fields(IProjectionConfig)}
    projection = IProjectionConfig(**{
        key: value for key, value in projection_values.items() if key in projection_allowed
    })
    forcing_allowed = {item.name for item in fields(ForcingConfig)}
    forcing_cfg = ForcingConfig(**{
        key: value for key, value in cfg["forcing"].items() if key in forcing_allowed
    })
    reconstructor = AnchoredCubicSplineReconstructor(
        jax.device_get(times[acquisition]), jax.device_get(times),
        AnchoredCubicSplineConfig(**cfg["moment_reconstruction"]),
    )
    problem = FrozenEtaProblem(
        truth_configurations=truth, times=times, time_weights=_time_weights(times),
        acquisition_indices=acquisition,
        finite_configuration_count=min(int(cfg["measurement"]["finite_configurations"]), int(truth.shape[1])),
        detector_noise=noise, family=family, reconstructor=reconstructor,
        projection_config=projection, forcing_config=forcing_cfg,
        projection_backend=backend, box=tuple(cfg["physics"]["box"]),
    )
    fit, audit = _load_bank(FRESH_FIT), _load_bank(FRESH_AUDIT)
    with np.load(ARTIFACT_DIR / "truth_banks.npz", allow_pickle=False) as arrays:
        selection_truth = jnp.asarray(arrays["design"], dtype=jnp.float64)
    box = tuple(cfg["physics"]["box"])
    return ValidationGalerkinData(
        validation_problem=problem, fit_bank=fit, audit_bank=audit,
        reference_features=many_body_features(fit.configurations, box),
        truth_means=jnp.mean(many_body_features(truth, box), axis=1),
        whitening=whitening_from_truth(many_body_features(selection_truth, box)),
    )


def _validation_risk(eta: jax.Array, data: ValidationGalerkinData) -> jax.Array:
    return projected_law_risk(
        eta, data.validation_problem, data.fit_bank,
        data.reference_features, data.truth_means, data.whitening,
    )


def _action_uncertainty(dictionary: Any, coefficients: jax.Array, bank: Any,
                        weights: jax.Array, time_weights: jax.Array,
                        chunk_size: int) -> dict[str, Any]:
    kinetic, kinetic_second = [], []
    sample_count = int(bank.configurations.shape[1])
    evaluators = [
        jax.jit(lambda rows, t=t: _normalized_chunk(dictionary, rows, t))
        for t in range(int(bank.configurations.shape[0]))
    ]
    for time_index in range(int(bank.configurations.shape[0])):
        first = jnp.asarray(0.0, dtype=jnp.float64)
        second = jnp.asarray(0.0, dtype=jnp.float64)
        for start in range(0, sample_count, int(chunk_size)):
            stop = min(start + int(chunk_size), sample_count)
            _, gradients = evaluators[time_index](bank.configurations[time_index, start:stop])
            potential_gradient = jnp.einsum("k,nkpd->npd", coefficients[time_index], gradients)
            rows = jnp.sum(potential_gradient * potential_gradient, axis=(-2, -1))
            chunk_weights = weights[time_index, start:stop]
            first = first + jnp.einsum("n,n->", chunk_weights, rows)
            second = second + jnp.einsum("n,n->", chunk_weights, rows * rows)
        kinetic.append(first)
        kinetic_second.append(second)
    kinetic, kinetic_second = jnp.stack(kinetic), jnp.stack(kinetic_second)
    variance = jnp.maximum(kinetic_second - kinetic * kinetic, 0.0)
    effective_samples = 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=-1), 1e-300)
    standard_error = jnp.sqrt(jnp.sum((time_weights * jnp.sqrt(
        variance / jnp.maximum(effective_samples, 1.0)
    )) ** 2))
    return {
        "action": float(jnp.sum(time_weights * kinetic)),
        "action_standard_error": float(standard_error),
        "kinetic_by_time": np.asarray(kinetic).tolist(),
        "uncertainty_convention": "production weighted empirical audit-sample standard error; no pseudo-blocks introduced",
    }


def _evaluate(cfg: dict[str, Any], data: ValidationGalerkinData,
              dictionary: Any, eta: Any) -> dict[str, Any]:
    problem = data.validation_problem
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    reconstruction = reconstruct_moments(eta, problem)
    fit_state = forcing_state(eta, problem, data.fit_bank, reconstruction)
    chunk_size = int(cfg["production_galerkin"]["chunk_size"])
    started = time.perf_counter()
    system = assemble_hybrid_system(
        dictionary, data.fit_bank, fit_state.projection.weights, fit_state.forcing,
        chunk_size=chunk_size,
    )
    solve = rank_aware_quadratic_solve(
        system.gram, system.load,
        relative_rank_tolerance=float(cfg["production_galerkin"]["relative_rank_tolerance"]),
    )
    aggregate = aggregate_quadratic_values(solve, problem.time_weights)
    fit_seconds = time.perf_counter() - started
    audit_state = forcing_state(eta, problem, data.audit_bank, reconstruction)
    adapter = SimpleNamespace(ritz_audit_bank=data.audit_bank, selection_problem=problem)
    thresholds = GalerkinCertificateThresholds(**cfg["production_galerkin"]["certificate_thresholds"])
    started = time.perf_counter()
    certificate = audit_hybrid_solutions(
        dictionary, solve.coefficients[None], adapter, eta, reconstruction,
        audit_state, thresholds, chunk_size=chunk_size,
    )[0]
    audit_seconds = time.perf_counter() - started
    uncertainty = _action_uncertainty(
        dictionary, solve.coefficients, data.audit_bank, audit_state.projection.weights,
        problem.time_weights, chunk_size,
    )
    if not np.isclose(uncertainty["action"], certificate["action"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("fresh validation audit action/uncertainty mismatch")
    settings = cfg["production_galerkin"]
    rank_fraction = solve.numerical_rank / float(dictionary.size)
    algebra_valid = bool(
        float(aggregate["identity_relerr"]) <= float(settings["maximum_identity_relerr"])
        and float(jnp.max(solve.range_residual)) <= float(settings["maximum_range_residual"])
        and float(jnp.max(solve.stationarity_residual)) <= float(settings["maximum_stationarity_residual"])
        and float(jnp.max(system.raw_symmetry_residual)) <= float(settings["maximum_symmetry_residual"])
        and float(jnp.max(solve.condition_number)) <= float(settings["maximum_retained_condition"])
        and float(jnp.min(rank_fraction)) >= float(settings["minimum_rank_fraction"])
    )
    fit_forcing = _forcing_state_payload(fit_state, problem)
    audit_forcing = _forcing_state_payload(audit_state, problem)
    numerical_valid = bool(
        problem.family.geometry_valid(eta) and fit_forcing["valid"]
        and audit_forcing["valid"] and algebra_valid and certificate["valid"]
    )
    return {
        "eta": np.asarray(eta).tolist(), "risk": float(_validation_risk(eta, data)),
        "validation_fit_action": float(aggregate["action"]),
        "validation_audit_action": float(certificate["action"]),
        "action_standard_error": uncertainty["action_standard_error"],
        "uncertainty_convention": uncertainty["uncertainty_convention"],
        "geometry_valid": bool(problem.family.geometry_valid(eta)),
        "fit_forcing_audit": fit_forcing, "audit_forcing_audit": audit_forcing,
        "identity_relerr": float(aggregate["identity_relerr"]),
        "rank_by_time": np.asarray(solve.numerical_rank).tolist(),
        "minimum_rank_fraction": float(jnp.min(rank_fraction)),
        "worst_retained_condition": float(jnp.max(solve.condition_number)),
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
        "algebra_valid": algebra_valid, "heldout_certificate": certificate,
        "numerically_certified": numerical_valid,
        "timings": {"fit_assembly_solve_seconds": fit_seconds, "audit_certificate_seconds": audit_seconds},
    }


def run_fresh_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_frozen_protocol(cfg)
    selection, selection_manifest, selection_hash, manifest_hash = _selection_seal()
    fresh_manifest = read_json(FRESH_MANIFEST)
    if fresh_manifest.get("selection_sha256") != selection_hash or not fresh_manifest.get("passed"):
        raise RuntimeError("fresh validation manifest is not tied to frozen selection")
    output_path = FRESH_ROOT / "pareto_validation.json"
    seal = {
        "protocol_sha256": protocol["protocol_sha256"], "selection_sha256": selection_hash,
        "selection_manifest_sha256": manifest_hash, "fresh_manifest_sha256": file_sha256(FRESH_MANIFEST),
        "winning_etas": selection_manifest["winning_etas"],
    }
    if output_path.is_file():
        old = read_json(output_path)
        if old.get("seal") == seal:
            return {**old, "cache_hit": True}
        raise RuntimeError("sealed fresh validation output exists with different prerequisites")
    write_json(FRESH_ROOT / "validation_seal.json", seal, overwrite=False)
    data = load_fresh_validation(cfg)
    dictionary = load_dictionary(DICTIONARY_PATH, box=tuple(cfg["physics"]["box"]))
    if dictionary.size != OFFICIAL_K:
        raise RuntimeError("official validation dictionary is not K=280")
    law = _evaluate(cfg, data, dictionary, cfg["envelope"]["law_eta"])
    unique: dict[str, dict[str, Any]] = {}
    for row in selection["allowances"]:
        key = hashlib.sha256(np.asarray(row["winner"]["eta"], dtype=np.float64).tobytes()).hexdigest()
        if key not in unique:
            unique[key] = _evaluate(cfg, data, dictionary, row["winner"]["eta"])
    rows = []
    for selection_row in selection["allowances"]:
        allowance = float(selection_row["allowance_percent"])
        key = hashlib.sha256(np.asarray(selection_row["winner"]["eta"], dtype=np.float64).tobytes()).hexdigest()
        candidate = unique[key]
        actual_ratio = candidate["risk"] / law["risk"] - 1.0
        strict_pass = candidate["risk"] <= strict_validation_ceiling(law["risk"], allowance)
        declared_pass = candidate["risk"] <= validation_ceiling(law["risk"], allowance)
        classification = validation_classification(
            numerical_valid=candidate["numerically_certified"],
            declared_risk_pass=declared_pass,
        )
        rows.append({
            "allowance_percent": allowance, "eta": candidate["eta"],
            "law_risk": law["risk"], "full_risk": candidate["risk"],
            "actual_validation_risk_increase_percent": 100.0 * actual_ratio,
            "strict_risk_ceiling": strict_validation_ceiling(law["risk"], allowance),
            "declared_risk_ceiling": validation_ceiling(law["risk"], allowance),
            "strict_p_percent_pass": strict_pass,
            "declared_p_plus_5pp_pass": declared_pass,
            "validation_fit_action": candidate["validation_fit_action"],
            "validation_audit_action": candidate["validation_audit_action"],
            "action_standard_error": candidate["action_standard_error"],
            "fit_reduction_vs_law": common_solver_reduction(law["validation_fit_action"], candidate["validation_fit_action"]),
            "audit_reduction_vs_law": common_solver_reduction(law["validation_audit_action"], candidate["validation_audit_action"]),
            "numerically_certified": candidate["numerically_certified"],
            "classification": classification, "diagnostics": candidate,
        })
    immutable = selection_manifest["winning_etas"] == [row["eta"] for row in rows]
    result = {
        "schema_version": 1, "ran": True, "passed": immutable,
        "cache_hit": False, "seal": seal, "device": device_payload(),
        "basis_size": OFFICIAL_K, "dictionary_sha256": file_sha256(DICTIONARY_PATH),
        "selection_winners_unchanged": immutable,
        "selection_reopened": False, "optimization_run": False, "deep_ritz_used": False,
        "law": law, "unique_design_evaluations": unique, "allowances": rows,
        "fresh_validation_manifest_sha256": file_sha256(FRESH_MANIFEST),
    }
    write_json(output_path, result, overwrite=False)
    if not immutable:
        raise RuntimeError("selection winner changed at validation boundary")
    return result


__all__ = [
    "FRESH_AUDIT", "FRESH_FIT", "FRESH_MANIFEST", "FRESH_NOISE", "FRESH_ROOT",
    "FRESH_TRUTH", "generate_fresh_validation", "load_fresh_validation",
    "run_fresh_validation",
]
