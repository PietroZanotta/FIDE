"""Selection-only ESS qualification and K=280 performance diagnostics."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from mfsi.projection import EmpiricalIProjector, IProjectionConfig

from .domain import SkyrmionTruth
from .full_gradient import reconstruct_moments, wrap_periodic
from .galerkin_only_data import GalerkinReferenceBank, load_selection_galerkin_data, selection_risk
from .measurements import local_sensor_designs, random_sensor_designs
from .production_artifacts import PRODUCTION_ROOT, file_sha256
from .reference import load_reference
from .resolution_study import FIXED_GEOMETRIES


PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "ess_qualification"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PROTOCOL_HASH_PATH = OUTPUT_ROOT / "protocol_hash.txt"
REPORT_PATH = PACKAGE_ROOT / "ESS_QUALIFICATION_AND_PERFORMANCE.md"
ARTIFACT_DIR = PRODUCTION_ROOT / "artifacts"
DICTIONARY_PATH = PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "dictionaries" / "dictionary_K280.npz"
EXPECTED_DICTIONARY_SHA256 = "37e9b60fcb92c4e5a0ee7ec1651fb7f8889f7ac6bdb02d3bd314e9ef40833326"
VERSION = "skyrmion_ess_qualification_v1"
LADDER = {8192: 4, 16384: 4, 32768: 3}
ALLOWANCES = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
RESS_THRESHOLD = 0.05
ENERGY_THRESHOLD = 0.08
STAGE_B_FLOOR = 0.04
STAGE_B_TOP_M = 32
STAGE_C_FLOOR = 0.045
CANDIDATE_BATCH_SIZE = 8
T95 = {2: 4.302652729696142, 3: 3.182446305284263, 4: 2.7764451051977987}


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def derive_seed(global_seed: int, size: int, replicate: int) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:ess_qualification:v1:N{int(size)}:rep{int(replicate)}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "text": text, "sha256": digest, "N": int(size), "replicate": int(replicate),
        "seed": int(digest[:16], 16) % (2**31 - 1),
    }


def require_output_path(path: Path) -> Path:
    resolved, root = Path(path).resolve(), OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"ESS-study output must be beneath {root}, got {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = require_output_path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite ESS-study output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _physics_config(cfg: dict[str, Any]):
    from .domain import SkyrmionConfig
    values = dict(cfg["physics"])
    values.pop("time_nodes", None)
    values.pop("truth_substeps", None)
    values["box"] = tuple(values["box"])
    values["pinning_centers"] = tuple(tuple(row) for row in values["pinning_centers"])
    return SkyrmionConfig(**values)


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    if file_sha256(DICTIONARY_PATH) != EXPECTED_DICTIONARY_SHA256:
        raise RuntimeError("fixed K=280 dictionary changed")
    if float(cfg["forcing"]["minimum_ess_fraction"]) != RESS_THRESHOLD:
        raise RuntimeError("relative ESS threshold changed")
    if float(cfg["production_galerkin"]["certificate_thresholds"]["maximum_energy_residual"]) != ENERGY_THRESHOLD:
        raise RuntimeError("energy threshold changed")
    if float(cfg["production_galerkin"]["relative_rank_tolerance"]) != 1e-12:
        raise RuntimeError("rank tolerance changed")
    seeds = [derive_seed(cfg["seed"], n, rep) for n, count in LADDER.items() for rep in range(count)]
    immutable = {
        name: file_sha256(PACKAGE_ROOT / name) for name in (
            "FINAL_3PCT_GALERKIN_CROSSCHECK.md", "GALERKIN_RESOLUTION_STUDY.md",
            "GALERKIN_K280_QUADRATURE_QUALIFICATION.md",
            "OFFICIAL_GALERKIN_PARETO_EVALUATION.md", "OFFICIAL_GALERKIN_PARETO_PROTOCOL.md",
        )
    }
    return {
        "schema_version": 1, "version": VERSION,
        "purpose": "cheap selection-development-only ESS qualification and performance audit",
        "validation_access_permitted": False, "eta_full_optimization_permitted": False,
        "pareto_sweep_permitted": False, "deep_ritz_permitted": False,
        "constants": {
            "K": 280, "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
            "relative_rank_tolerance": 1e-12, "minimum_ress": RESS_THRESHOLD,
            "maximum_energy_residual": ENERGY_THRESHOLD, "allowances_percent": list(ALLOWANCES),
        },
        "anchors": [{"id": name, "provenance": provenance, "eta": eta}
                    for name, provenance, eta in FIXED_GEOMETRIES],
        "banks": {"ladder": {str(k): v for k, v in LADDER.items()}, "seed_records": seeds,
                  "replicate_zero_reused_for_staged_screening": True,
                  "reference_retrained": False, "selection_development_only": True},
        "anchor_classification": {
            "interval": "two-sided 95% Student-t interval over replicate min-rESS",
            "clearly_above": "lower bound > 0.05", "clearly_below": "upper bound < 0.05",
            "borderline": "interval crosses 0.05 and abs(N32768 mean-0.05)<=0.005",
            "unresolved": "otherwise",
        },
        "candidate_pool": {
            "source": "frozen resolution-study future-v2 deterministic construction",
            "interpolation_points_per_segment": 17, "local_count_per_center": 16,
            "local_scale": 0.01, "risk_tangent_direction_count": 16,
            "risk_tangent_radii": [0.0001, 0.0005, 0.001, 0.005],
            "global_count": 32, "global_oversample": 16,
            "seed": _future_pool_seed(cfg["seed"]),
        },
        "stages": {"A_N": 8192, "B_N": 16384, "B_ress_floor": STAGE_B_FLOOR,
                   "B_top_M_per_allowance": STAGE_B_TOP_M, "C_N": 32768,
                   "C_ress_floor": STAGE_C_FLOOR, "candidate_batch_size": CANDIDATE_BATCH_SIZE},
        "historical_immutable_sha256": immutable,
        "source_sha256": {"ess_study.py": file_sha256(PACKAGE_ROOT / "ess_study.py"),
                          "config.json": file_sha256(PACKAGE_ROOT / "config.json")},
    }


def freeze_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(cfg)
    digest = payload_sha256(payload)
    result = {**payload, "protocol_sha256": digest, "protocol_frozen": True}
    if PROTOCOL_PATH.exists():
        if read_json(PROTOCOL_PATH) != result:
            raise RuntimeError("different ESS protocol already exists")
        return {**result, "cache_hit": True}
    write_json(PROTOCOL_PATH, result)
    PROTOCOL_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return result


def require_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_HASH_PATH.is_file():
        raise RuntimeError("freeze-protocol must run first")
    saved = read_json(PROTOCOL_PATH)
    body = dict(saved)
    digest = body.pop("protocol_sha256", None)
    frozen = body.pop("protocol_frozen", None)
    if not frozen or payload_sha256(body) != digest or PROTOCOL_HASH_PATH.read_text().strip() != digest:
        raise RuntimeError("ESS protocol seal mismatch")
    if protocol_payload(cfg) != body:
        raise RuntimeError("current implementation/config differs from frozen ESS protocol")
    return saved


def _future_pool_seed(global_seed: int) -> dict[str, Any]:
    text = f"{int(global_seed)}:skyrmion:galerkin_resolution:v1:future_starts:1"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {"text": text, "sha256": digest, "seed": int(digest[:16], 16) % (2**31 - 1)}


def _bank_path(size: int) -> Path:
    return OUTPUT_ROOT / "banks" / f"selection_development_only_N{size}_rep0.npz"


def _generate_bank(cfg: dict[str, Any], seed: int, samples: int) -> tuple[GalerkinReferenceBank, float]:
    times = jnp.linspace(0.0, 1.0, int(cfg["physics"]["time_nodes"]), dtype=jnp.float64)
    truth = SkyrmionTruth(_physics_config(cfg))
    flow = load_reference(ARTIFACT_DIR / "reference.npz")
    started = time.perf_counter()
    initial = truth.sample_initial(jax.random.PRNGKey(int(seed)), int(samples))
    configs, velocities = [], []
    for start in range(0, int(samples), 2048):
        stop = min(start + 2048, int(samples))
        rows = flow.rollout(initial[start:stop], times,
                            substeps_per_interval=int(cfg["banks"]["reference_substeps"]))
        configs.append(np.asarray(rows))
        velocities.append(np.asarray(flow.velocity(rows, times)))
    x, v = np.concatenate(configs, axis=1), np.concatenate(velocities, axis=1)
    w = np.full(x.shape[:2], 1.0 / float(samples), dtype=np.float64)
    return GalerkinReferenceBank(jnp.asarray(x), jnp.asarray(v), jnp.asarray(w)), time.perf_counter() - started


def _save_bank_once(path: Path, bank: GalerkinReferenceBank, seed_record: dict[str, Any]) -> None:
    path = require_output_path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, configurations=np.asarray(bank.configurations), velocity=np.asarray(bank.velocity),
             base_weights=np.asarray(bank.base_weights), selection_development_only=np.asarray(True),
             seed=np.asarray(seed_record["seed"]), seed_sha256=np.asarray(seed_record["sha256"]))


def load_screen_bank(size: int) -> GalerkinReferenceBank:
    path = _bank_path(size)
    if not path.is_file():
        raise RuntimeError(f"missing staged-screening bank: {path}; run anchors first")
    with np.load(path, allow_pickle=False) as arrays:
        if not bool(np.asarray(arrays["selection_development_only"]).item()):
            raise RuntimeError("bank selection-only marker missing")
        x, v, w = arrays["configurations"], arrays["velocity"], arrays["base_weights"]
        return GalerkinReferenceBank(jnp.asarray(x), jnp.asarray(v), jnp.asarray(w))


def exact_ess(weights: np.ndarray, base_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    base = np.asarray(base_weights, dtype=np.float64)
    absolute = 1.0 / np.maximum(np.sum(weights * weights, axis=-1), 1e-300)
    base_ess = 1.0 / np.maximum(np.sum(base * base, axis=-1), 1e-300)
    return absolute, absolute / base_ess, base_ess


def _projection_cfg(cfg: dict[str, Any]) -> tuple[IProjectionConfig, str]:
    values = dict(cfg["projection"])
    backend = str(values.pop("trajectory_backend", "jax"))
    allowed = {item.name for item in fields(IProjectionConfig)}
    return IProjectionConfig(**{k: v for k, v in values.items() if k in allowed}), backend


def _project(problem: Any, bank: GalerkinReferenceBank, eta: jax.Array,
             values: jax.Array | None = None, features: jax.Array | None = None) -> Any:
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    values = reconstruct_moments(eta, problem).values if values is None else values
    features = problem.family.features(bank.configurations, eta) if features is None else features
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    projected = projector.project_trajectory(features, bank.base_weights, values[None, ...])
    return jax.tree_util.tree_map(lambda x: x[0], projected)


def projection_payload(state: Any, bank: GalerkinReferenceBank) -> dict[str, Any]:
    w = np.asarray(state.weights)
    absolute, relative, base_ess = exact_ess(w, np.asarray(bank.base_weights))
    residuals = np.linalg.norm(np.asarray(state.residual), axis=-1)
    eig = np.linalg.eigvalsh(np.asarray(state.covariance))
    conditions = eig[:, -1] / np.maximum(eig[:, 0], 1e-300)
    lam = np.asarray(state.lam)
    controlling = int(np.argmin(relative))
    ratio = w / np.asarray(bank.base_weights)
    population_relation = np.mean(ratio, axis=-1) ** 2 / np.mean(ratio * ratio, axis=-1)
    return {
        "samples": int(w.shape[-1]), "ess_fraction_by_time": relative.tolist(),
        "absolute_ess_by_time": absolute.tolist(), "base_ess_by_time": base_ess.tolist(),
        "minimum_ess_fraction": float(relative[controlling]),
        "minimum_absolute_ess": float(absolute[controlling]),
        "controlling_time_index": controlling,
        "maximum_projection_residual": float(np.max(residuals)),
        "projection_residual_by_time": residuals.tolist(),
        "maximum_covariance_condition": float(np.max(conditions)),
        "covariance_condition_by_time": conditions.tolist(),
        "lambda_norm_by_time": np.linalg.norm(lam, axis=-1).tolist(),
        "maximum_lambda_norm": float(np.max(np.linalg.norm(lam, axis=-1))),
        "maximum_absolute_lambda_component": float(np.max(np.abs(lam))),
        "population_ratio_relation_by_time": population_relation.tolist(),
        "maximum_ess_relation_discrepancy": float(np.max(np.abs(population_relation-relative))),
        "uniform_base_weights": bool(np.allclose(np.asarray(bank.base_weights), 1.0/w.shape[-1], rtol=0, atol=1e-16)),
    }


def _anchor_payload(cfg: dict[str, Any], data: Any, bank: GalerkinReferenceBank,
                    geometry_id: str, eta: Iterable[float], seed_record: dict[str, Any],
                    generation_seconds: float) -> dict[str, Any]:
    problem = data.selection_problem
    eta_array = jnp.asarray(eta, dtype=jnp.float64)
    reconstruction = reconstruct_moments(eta_array, problem)
    started = time.perf_counter()
    state = _project(problem, bank, eta_array, reconstruction.values)
    state.weights.block_until_ready()
    elapsed = time.perf_counter() - started
    payload = projection_payload(state, bank)
    # Forcing compatibility is cheap relative to a Full solve; construct only the
    # moment-rate terms here, never a Galerkin basis/system.
    advective = problem.family.jvp(bank.configurations, bank.velocity, eta_array)
    moment_m = jnp.einsum("tn,tnr->tr", state.weights, advective)
    scalar_m = jnp.einsum("tnr,tr->tn", advective, state.lam)
    centered_phi = problem.family.features(bank.configurations, eta_array) - state.moments[:, None, :]
    centered_g = scalar_m - jnp.einsum("tn,tn->t", state.weights, scalar_m)[:, None]
    covariance_phi_g = jnp.einsum("tn,tnr,tn->tr", state.weights, centered_phi, centered_g)
    rhs = reconstruction.derivatives - moment_m - covariance_phi_g
    ridge = float(problem.forcing_config.covariance_ridge)
    lambda_dot = jax.vmap(jnp.linalg.solve)(state.covariance + ridge*jnp.eye(4), rhs)
    forcing = (jnp.einsum("tr,tnr->tn", lambda_dot, centered_phi)
               + jnp.einsum("tr,tnr->tn", state.lam, advective-moment_m[:, None, :]))
    force_mean = np.asarray(jnp.einsum("tn,tn->t", state.weights, forcing))
    payload.update({
        "geometry_id": geometry_id, "eta": list(map(float, eta)), "seed_record": seed_record,
        "scientific_selection_risk": float(selection_risk(eta_array, data)),
        "geometry_valid": bool(problem.family.geometry_valid(eta_array)),
        "forcing_mean_before_centering_by_time": force_mean.tolist(),
        "maximum_forcing_mean": float(np.max(np.abs(force_mean))),
        "generation_seconds": generation_seconds, "projection_seconds": elapsed,
        "galerkin_constructed": False, "validation_accessed": False,
    })
    return payload


def _mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values))
    half = T95[len(values)-1] * se if len(values) > 1 else 0.0
    return {"mean": mean, "standard_deviation": sd, "standard_error": se,
            "minimum": min(values), "maximum": max(values),
            "ci95_lower": mean-half, "ci95_upper": mean+half}


def _extrapolate(rows: list[dict[str, Any]], power: float) -> float:
    x, y = [], []
    for row in rows:
        x.append(row["N"] ** (-power)); y.append(row["minimum_ress"]["mean"])
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), np.asarray(y), rcond=None)[0][0])


def run_anchors(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    summary_path = OUTPUT_ROOT / "fixed_anchor_ess" / "summary.json"
    if summary_path.is_file():
        return {**read_json(summary_path), "cache_hit": True}
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    records = []
    seeds = {(row["N"], row["replicate"]): row for row in protocol["banks"]["seed_records"]}
    for n, count in LADDER.items():
        for rep in range(count):
            seed_record = seeds[(n, rep)]
            bank, generation_seconds = _generate_bank(cfg, seed_record["seed"], n)
            if rep == 0:
                _save_bank_once(_bank_path(n), bank, seed_record)
            for geometry_id, _, eta in FIXED_GEOMETRIES:
                row = _anchor_payload(cfg, data, bank, geometry_id, eta, seed_record, generation_seconds)
                path = OUTPUT_ROOT / "fixed_anchor_ess" / f"N{n}" / f"rep{rep}" / f"{geometry_id}.json"
                write_json(path, row)
                records.append(row)
            del bank
    aggregate = []
    for geometry_id, _, _ in FIXED_GEOMETRIES:
        ladder_rows = []
        for n in LADDER:
            group = [r for r in records if r["geometry_id"] == geometry_id and r["samples"] == n]
            minima = [r["minimum_ess_fraction"] for r in group]
            absolute = [r["minimum_absolute_ess"] for r in group]
            controls = [r["controlling_time_index"] for r in group]
            ladder_rows.append({"N": n, "replicates": len(group), "minimum_ress": _mean_ci(minima),
                                "minimum_absolute_ess": _mean_ci(absolute),
                                "controlling_time_indices": controls})
        final = ladder_rows[-1]["minimum_ress"]
        if final["ci95_lower"] > RESS_THRESHOLD:
            classification = "CLEARLY ABOVE 0.05"
        elif final["ci95_upper"] < RESS_THRESHOLD:
            classification = "CLEARLY BELOW 0.05"
        elif abs(final["mean"] - RESS_THRESHOLD) <= 0.005:
            classification = "BORDERLINE"
        else:
            classification = "UNRESOLVED"
        aggregate.append({"geometry_id": geometry_id, "ladder": ladder_rows,
                          "r_inf_1_over_N": _extrapolate(ladder_rows, 1.0),
                          "r_inf_1_over_sqrt_N": _extrapolate(ladder_rows, 0.5),
                          "classification": classification})
    result = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
              "passed": True, "validation_accessed": False, "galerkin_constructed": False,
              "anchors": aggregate}
    write_json(summary_path, result)
    return result


def build_candidate_pool(cfg: dict[str, Any], data: Any) -> dict[str, Any]:
    path = OUTPUT_ROOT / "candidate_pool" / "manifest.json"
    if path.is_file():
        return read_json(path)
    protocol = require_protocol(cfg)
    settings = protocol["candidate_pool"]
    family = data.selection_problem.family
    fixed = jnp.asarray([row[2] for row in FIXED_GEOMETRIES], dtype=jnp.float64)
    law, historical = fixed[0], fixed[1:]
    seed = int(settings["seed"]["seed"])
    alphas = jnp.linspace(0.0, 1.0, int(settings["interpolation_points_per_segment"]), dtype=jnp.float64)
    interpolated = jnp.concatenate([jax.vmap(lambda a: wrap_periodic(law+a*(center-law), family))(alphas)
                                    for center in historical])
    local = local_sensor_designs(jax.random.PRNGKey(seed), fixed,
                                 count_per_center=int(settings["local_count_per_center"]),
                                 scale=float(settings["local_scale"]), family=family)
    _, risk_gradient = jax.value_and_grad(lambda eta: selection_risk(eta, data))(law)
    tangent = []
    key = jax.random.PRNGKey(seed+1)
    for index in range(int(settings["risk_tangent_direction_count"])):
        direction = jax.random.normal(jax.random.fold_in(key, index), law.shape, dtype=jnp.float64)
        direction -= jnp.dot(direction, risk_gradient)/jnp.maximum(jnp.dot(risk_gradient, risk_gradient), 1e-30)*risk_gradient
        direction /= jnp.maximum(jnp.linalg.norm(direction), 1e-30)
        for radius in settings["risk_tangent_radii"]:
            tangent.extend([wrap_periodic(law+radius*direction, family), wrap_periodic(law-radius*direction, family)])
    global_rows = random_sensor_designs(jax.random.PRNGKey(seed+2), count=int(settings["global_count"]),
                                        family=family, oversample=int(settings["global_oversample"]))
    raw = np.asarray(jnp.concatenate([law[None], historical, interpolated, local, jnp.asarray(tangent), global_rows]))
    unique: list[np.ndarray] = []
    for eta in raw:
        if bool(family.geometry_valid(jnp.asarray(eta))) and not any(np.linalg.norm(eta-old) <= 1e-12 for old in unique):
            unique.append(eta)
    risks = [float(selection_risk(jnp.asarray(eta), data)) for eta in unique]
    law_risk = risks[0]
    candidates = [{"candidate_id": f"candidate_{i:03d}", "eta": eta.tolist(),
                   "scientific_selection_risk": risk, "law_relative_risk_increase": risk/law_risk-1.0,
                   "geometry_valid": True} for i, (eta, risk) in enumerate(zip(unique, risks, strict=True))]
    result = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
              "pool_count": len(candidates), "law_risk": law_risk, "candidates": candidates,
              "pool_sha256": payload_sha256(candidates), "selection_only": True,
              "validation_accessed": False, "eta_optimization_run": False}
    write_json(path, result)
    return result


def _batched_preprocess(problem: Any, bank: GalerkinReferenceBank, etas: np.ndarray,
                        batch_size: int = CANDIDATE_BATCH_SIZE):
    fn = jax.jit(jax.vmap(lambda eta: (
        reconstruct_moments(eta, problem).values,
        problem.family.features(bank.configurations, eta),
    )))
    for start in range(0, len(etas), batch_size):
        stop = min(start+batch_size, len(etas)); actual = stop-start
        block = etas[start:stop]
        if actual < batch_size:
            block = np.concatenate([block, np.repeat(block[-1:], batch_size-actual, axis=0)])
        values, features = fn(jnp.asarray(block, dtype=jnp.float64))
        features.block_until_ready()
        yield start, stop, values[:actual], features[:actual]


def score_candidates(cfg: dict[str, Any], data: Any, bank: GalerkinReferenceBank,
                     candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problem = data.selection_problem
    etas = np.asarray([row["eta"] for row in candidates], dtype=np.float64)
    output = []
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    for start, stop, values, features in _batched_preprocess(problem, bank, etas):
        for local, candidate in enumerate(candidates[start:stop]):
            projected = projector.project_trajectory(features[local], bank.base_weights, values[local][None, ...])
            state = jax.tree_util.tree_map(lambda x: x[0], projected)
            diag = projection_payload(state, bank)
            projection_valid = bool(diag["maximum_projection_residual"] <= float(cfg["forcing"]["projection_tolerance"])
                                    and diag["maximum_covariance_condition"] <= float(cfg["forcing"]["max_covariance_condition"]))
            output.append({**candidate, **diag, "projection_valid": projection_valid,
                           "galerkin_constructed": False, "validation_accessed": False})
    return output


def _allowance_tables(rows: list[dict[str, Any]], law_risk: float) -> list[dict[str, Any]]:
    tables = []
    for allowance in ALLOWANCES:
        ceiling = (1+allowance/100)*law_risk
        feasible = [r for r in rows if r["scientific_selection_risk"] <= ceiling and r["geometry_valid"]]
        projection = [r for r in feasible if r["projection_valid"]]
        both = [r for r in projection if r["minimum_ess_fraction"] >= RESS_THRESHOLD]
        values = np.asarray([r["minimum_ess_fraction"] for r in feasible])
        quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1]).tolist()
        tables.append({"allowance_percent": allowance, "risk_ceiling": ceiling,
                       "total_candidates": len(rows), "risk_feasible_candidates": len(feasible),
                       "risk_and_ress_candidates": sum(r["minimum_ess_fraction"] >= RESS_THRESHOLD for r in feasible),
                       "risk_and_projection_valid_candidates": len(projection),
                       "risk_projection_and_ress_candidates": len(both),
                       "risk_feasible_ress_quantiles": dict(zip(("min","p05","p25","median","p75","p95","max"), quantiles, strict=True))})
    return tables


def run_candidate_screen(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    path = OUTPUT_ROOT / "candidate_pool" / "stage_A_N8192.json"
    if path.is_file():
        return {**read_json(path), "cache_hit": True}
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    pool = build_candidate_pool(cfg, data)
    rows = score_candidates(cfg, data, load_screen_bank(8192), pool["candidates"])
    result = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
              "stage": "A", "N": 8192, "rows": rows,
              "allowance_tables": _allowance_tables(rows, pool["law_risk"]),
              "galerkin_constructed": False, "validation_accessed": False}
    write_json(path, result)
    return result


def _candidate_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["candidate_id"]: r for r in rows}


def _stage_b_ids(stage_a: dict[str, Any], law_risk: float) -> list[str]:
    selected: set[str] = set()
    for allowance in ALLOWANCES:
        ceiling = (1+allowance/100)*law_risk
        feasible = [r for r in stage_a["rows"] if r["scientific_selection_risk"] <= ceiling
                    and r["geometry_valid"] and r["projection_valid"]]
        selected.update(r["candidate_id"] for r in feasible if r["minimum_ess_fraction"] >= STAGE_B_FLOOR)
        selected.update(r["candidate_id"] for r in sorted(feasible, key=lambda r: r["minimum_ess_fraction"], reverse=True)[:STAGE_B_TOP_M])
    return sorted(selected)


def run_staged_rescore(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    out = OUTPUT_ROOT / "staged_rescoring" / "summary.json"
    if out.is_file():
        return {**read_json(out), "cache_hit": True}
    stage_a = run_candidate_screen(cfg)
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    pool = build_candidate_pool(cfg, data); pool_map = _candidate_map(pool["candidates"])
    ids_b = _stage_b_ids(stage_a, pool["law_risk"])
    rows_b = score_candidates(cfg, data, load_screen_bank(16384), [pool_map[i] for i in ids_b])
    ids_c = sorted(r["candidate_id"] for r in rows_b if r["geometry_valid"] and r["projection_valid"]
                   and r["minimum_ess_fraction"] >= STAGE_C_FLOOR)
    rows_c = score_candidates(cfg, data, load_screen_bank(32768), [pool_map[i] for i in ids_c])
    map_c = _candidate_map(rows_c)
    feasibility = []
    for allowance in ALLOWANCES:
        ceiling = (1+allowance/100)*pool["law_risk"]
        eligible = [r for r in rows_c if r["scientific_selection_risk"] <= ceiling
                    and r["geometry_valid"] and r["projection_valid"]]
        witnesses = [r for r in eligible if r["minimum_ess_fraction"] >= RESS_THRESHOLD]
        best = max(eligible, key=lambda r: r["minimum_ess_fraction"], default=None)
        feasibility.append({"allowance_percent": allowance, "answer": "YES" if witnesses else "UNRESOLVED",
                            "N32768_evaluated": len(eligible), "witness_count": len(witnesses),
                            "best": None if best is None else {k: best[k] for k in (
                                "candidate_id", "eta", "scientific_selection_risk", "law_relative_risk_increase",
                                "minimum_ess_fraction", "minimum_absolute_ess", "controlling_time_index")}})
    result = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
              "stage_B": {"N": 16384, "selected_count": len(ids_b), "rows": rows_b},
              "stage_C": {"N": 32768, "selected_count": len(ids_c), "rows": rows_c},
              "feasibility": feasibility, "galerkin_constructed": False,
              "validation_accessed": False, "eta_optimization_run": False}
    write_json(out, result)
    run_error_vs_ess(cfg, data, pool, stage_a, map_c)
    return result


def run_error_vs_ess(cfg: dict[str, Any], data: Any, pool: dict[str, Any],
                     stage_a: dict[str, Any], existing_c: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = OUTPUT_ROOT / "error_vs_ess" / "summary.json"
    if out.is_file():
        return read_json(out)
    targets = (.03, .04, .05, .07, .10)
    chosen, used = [], set()
    for target in targets:
        row = min((r for r in stage_a["rows"] if r["candidate_id"] not in used),
                  key=lambda r: abs(r["minimum_ess_fraction"]-target))
        used.add(row["candidate_id"]); chosen.append(row)
    missing = [r for r in chosen if r["candidate_id"] not in existing_c]
    rescored = score_candidates(cfg, data, load_screen_bank(32768),
                                [next(c for c in pool["candidates"] if c["candidate_id"] == r["candidate_id"]) for r in missing])
    high = {**existing_c, **_candidate_map(rescored)}
    rows = []
    for low in chosen:
        hi = high[low["candidate_id"]]
        rows.append({"candidate_id": low["candidate_id"], "target_ress": targets[len(rows)],
                     "N8192_ress": low["minimum_ess_fraction"], "N32768_ress": hi["minimum_ess_fraction"],
                     "absolute_ress_difference": abs(hi["minimum_ess_fraction"]-low["minimum_ess_fraction"]),
                     "maximum_lambda_norm_relative_difference": abs(hi["maximum_lambda_norm"]-low["maximum_lambda_norm"])/max(abs(hi["maximum_lambda_norm"]),1e-30),
                     "risk_same_frozen_exact_value": low["scientific_selection_risk"] == hi["scientific_selection_risk"]})
    result = {"schema_version": 1, "rows": rows, "full_action_evaluated": False,
              "validation_accessed": False,
              "interpretation": "descriptive modest subset; not a threshold-changing analysis"}
    write_json(out, result)
    return result


def _timed(call, repeats: int = 3) -> dict[str, Any]:
    values = []
    for _ in range(repeats):
        start = time.perf_counter(); result = call()
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
        values.append(time.perf_counter()-start)
    return {"first_call_seconds": values[0], "steady_median_seconds": statistics.median(values[1:] or values),
            "all_seconds": values}


def run_performance_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    protocol = require_protocol(cfg)
    out = OUTPUT_ROOT / "performance" / "benchmark.json"
    if out.is_file():
        return {**read_json(out), "cache_hit": True}
    data = load_selection_galerkin_data(cfg, ARTIFACT_DIR)
    bank = load_screen_bank(8192); problem = data.selection_problem
    pool = build_candidate_pool(cfg, data); etas = jnp.asarray([r["eta"] for r in pool["candidates"][:8]])
    scalar_pre = jax.jit(lambda eta: (reconstruct_moments(eta, problem).values,
                                      problem.family.features(bank.configurations, eta)))
    batch_pre = jax.jit(jax.vmap(lambda eta: (reconstruct_moments(eta, problem).values,
                                              problem.family.features(bank.configurations, eta))))
    scalar_timing = _timed(lambda: [scalar_pre(eta) for eta in etas], 3)
    batch_timing = _timed(lambda: batch_pre(etas), 3)
    scalar_values = [scalar_pre(eta) for eta in etas]
    batched_values = batch_pre(etas)
    max_pre_diff = max(float(np.max(np.abs(np.asarray(scalar_values[i][j])-np.asarray(batched_values[j][i]))))
                       for i in range(8) for j in range(2))
    scalar_ess, batched_ess = [], []
    projector_equiv = EmpiricalIProjector(problem.projection_config,
                                           trajectory_backend=problem.projection_backend)
    for index in range(2):
        scalar_state = projector_equiv.project_trajectory(
            scalar_values[index][1], bank.base_weights, scalar_values[index][0][None, ...])
        batched_state = projector_equiv.project_trajectory(
            batched_values[1][index], bank.base_weights, batched_values[0][index][None, ...])
        scalar_ess.append(np.asarray(scalar_state.ess_fraction))
        batched_ess.append(np.asarray(batched_state.ess_fraction))
    max_ess_diff = float(np.max(np.abs(np.asarray(scalar_ess)-np.asarray(batched_ess))))
    eta = etas[0]
    reconstruction = reconstruct_moments(eta, problem)
    features = problem.family.features(bank.configurations, eta)
    projector = EmpiricalIProjector(problem.projection_config, trajectory_backend=problem.projection_backend)
    projection_timing = _timed(lambda: projector.project_trajectory(features, bank.base_weights,
                                                                     reconstruction.values[None, ...]), 3)
    from .full_gradient import forcing_state
    forcing_timing = _timed(lambda: forcing_state(eta, problem, bank, reconstruction), 3)
    # Reuse the immutable, exact K=280 fixed-basis cache to profile current Full
    # operations without constructing or optimizing a new candidate.
    from .galerkin_only import GalerkinOnlyContext
    from .galerkin import aggregate_quadratic_values, rank_aware_quadratic_solve
    dictionary = DICTIONARY_PATH
    cache = PACKAGE_ROOT / "outputs" / "galerkin_only_3pct" / "cache" / "K280"
    started = time.perf_counter()
    context = GalerkinOnlyContext(cfg, ARTIFACT_DIR, data, dictionary, cache_dir=cache)
    context_seconds = time.perf_counter()-started
    historical_bank = data.train_bank
    rec_hist = reconstruct_moments(eta, problem)
    state_hist = forcing_state(eta, problem, historical_bank, rec_hist)
    assembly_timing = _timed(lambda: context.assemble(state_hist.projection.weights, state_hist.forcing, 280), 3)
    system = context.assemble(state_hist.projection.weights, state_hist.forcing, 280)
    solve_timing = _timed(lambda: rank_aware_quadratic_solve(system.gram, system.load,
                         relative_rank_tolerance=1e-12), 3)
    solve = rank_aware_quadratic_solve(system.gram, system.load, relative_rank_tolerance=1e-12)
    potential, kinetic = context.potential_rows(solve.coefficients, 280)
    gradient_timing = _timed(lambda: context._envelope_value_grad(eta, potential, kinetic), 3)
    complete_timing = _timed(lambda: context.evaluate(eta, basis_size=280, with_gradient=True), 3)
    eval_result = context.evaluate(eta, basis_size=280, with_gradient=True)
    aggregate = aggregate_quadratic_values(eval_result.solve, problem.time_weights)
    historical_profile_path = PACKAGE_ROOT / "outputs" / "fast_production_3pct" / "profiling" / "result.json"
    old_profile = read_json(historical_profile_path)
    result = {"schema_version": 1, "protocol_sha256": protocol["protocol_sha256"],
              "device": {"platform": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
                         "float64": bool(jax.config.jax_enable_x64)},
              "candidate_preprocessing": {"candidate_count": 8, "scalar_loop": scalar_timing,
                  "batched": batch_timing, "steady_speedup": scalar_timing["steady_median_seconds"]/batch_timing["steady_median_seconds"],
                  "max_absolute_discrepancy": max_pre_diff,
                  "max_ess_discrepancy_after_projection": max_ess_diff,
                  "estimated_peak_feature_bytes_scalar": int(np.asarray(scalar_values[0][1]).nbytes),
                  "estimated_peak_feature_bytes_batch": int(np.asarray(batched_values[1]).nbytes),
                  "batch_size": 8},
              "current_N8192": {"information_projection": projection_timing, "forcing": forcing_timing},
              "current_K280_cached": {"context_load_seconds": context_seconds, "K_f_assembly": assembly_timing,
                  "coefficient_eigensolve": solve_timing, "fixed_coefficient_value_gradient": gradient_timing,
                  "complete_value_gradient": complete_timing, "action": float(eval_result.action),
                  "gradient": np.asarray(eval_result.gradient).tolist(), "identity_relerr": float(aggregate["identity_relerr"]),
                  "cache_gib": sum(p.stat().st_size for p in cache.glob("*.npy"))/2**30},
              "historical_before_after_equivalence": old_profile["equivalence"],
              "historical_K160_speedup": old_profile["steady_value_gradient_speedup"],
              "implemented_in_this_task": ["fixed-size batched observation/reconstruction preprocessing for ESS screening"],
              "mathematical_semantics_changed": False, "validation_accessed": False,
              "eta_optimization_run": False, "heldout_audit": {
                  "measured_here": False,
                  "reason": "immutable prior K=280 held-out certificate is reused; no redundant Full audit is needed for an ESS diagnostic",
                  "prior_eta0_audit_action": 0.2966927692122766,
                  "prior_eta0_max_energy_residual": 0.07986682440563893}}
    write_json(out, result)
    return result


def verify_summary_consistency(summary: dict[str, Any]) -> bool:
    return bool(summary["thresholds"]["minimum_ress"] == RESS_THRESHOLD
                and summary["thresholds"]["maximum_energy_residual"] == ENERGY_THRESHOLD
                and len(summary["allowance_feasibility"]) == len(ALLOWANCES)
                and summary["validation_accessed"] is False
                and summary["eta_optimization_run"] is False)


__all__ = [
    "ALLOWANCES", "CANDIDATE_BATCH_SIZE", "DICTIONARY_PATH", "ENERGY_THRESHOLD",
    "EXPECTED_DICTIONARY_SHA256", "FIXED_GEOMETRIES", "LADDER", "OUTPUT_ROOT",
    "PROTOCOL_PATH", "REPORT_PATH", "RESS_THRESHOLD", "STAGE_B_FLOOR", "STAGE_B_TOP_M",
    "STAGE_C_FLOOR", "build_candidate_pool", "derive_seed", "exact_ess", "freeze_protocol",
    "load_screen_bank", "payload_sha256", "projection_payload", "require_output_path",
    "require_protocol", "run_anchors", "run_candidate_screen", "run_performance_audit",
    "run_staged_rescore", "score_candidates", "verify_summary_consistency",
]
