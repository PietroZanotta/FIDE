"""Fixed permutation-invariant Galerkin approximation for the Full correction.

The feature trunks in this module are frozen before eta evaluation.  The only
inner solve is a rank-aware symmetric quadratic solve performed outside the
differentiated eta closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .deep_ritz import (
    CertificateConfig,
    RitzParams,
    _audit_features,
    init_ritz_params,
    load_ritz_checkpoint,
    save_ritz_checkpoint,
)
from .full_gradient import (
    FrozenEtaProblem,
    ReferenceBank,
    forcing_state,
    periodic_branch_distance,
    reconstruct_moments,
    wrap_periodic,
)
from .workflow import (
    OUTPUT_ROOT,
    PreparedExperiment,
    hard_forcing_audit,
    require_output_path,
)

Array = jax.Array


@dataclass(frozen=True)
class FrozenDeepSetsBasis:
    params: RitzParams
    name: str
    source: str
    source_sha256: str | None
    box: tuple[float, float]

    @property
    def width(self) -> int:
        return int(self.params["head"][0]["W"].shape[-1])


class BasisEvaluation(NamedTuple):
    values: Array
    state_gradients: Array


class GalerkinSystem(NamedTuple):
    gram: Array
    load: Array
    basis_means: Array
    centered_basis: Array
    weights: Array
    forcing: Array
    raw_symmetry_residual: Array
    forcing_mean: Array


class GalerkinSolve(NamedTuple):
    coefficients: Array
    eigenvalues: Array
    retained: Array
    numerical_rank: Array
    range_residual: Array
    stationarity_residual: Array
    condition_number: Array
    action_by_time: Array
    objective_by_time: Array
    identity_relerr_by_time: Array


def _layers_at_time(layers: tuple[dict[str, Array], ...], t: Array):
    if layers[0]["W"].ndim == 2:
        return layers
    nodes = int(layers[0]["W"].shape[0])
    index = jnp.clip(jnp.rint(t * (nodes - 1)).astype(jnp.int32), 0, nodes - 1)
    return tuple({"W": layer["W"][index], "b": layer["b"][index]} for layer in layers)


def _silu_mlp(layers: tuple[dict[str, Array], ...], value: Array) -> Array:
    hidden = value
    for layer in layers:
        hidden = jax.nn.silu(hidden @ layer["W"] + layer["b"])
    return hidden


def frozen_deepsets_latent(
    basis: FrozenDeepSetsBasis,
    configuration: Array,
    time: Array,
) -> Array:
    """Return the frozen invariant representation immediately before scalar output."""

    x = jnp.asarray(configuration, dtype=jnp.float64)
    t = jnp.asarray(time, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * x / jnp.asarray(basis.box, dtype=x.dtype)
    time_features = jnp.asarray([
        t,
        jnp.sin(jnp.pi * t),
        jnp.cos(jnp.pi * t),
        jnp.sin(2.0 * jnp.pi * t),
        jnp.cos(2.0 * jnp.pi * t),
    ])
    local_time = jnp.broadcast_to(time_features, x.shape[:-1] + (5,))
    local = jnp.concatenate([jnp.sin(phase), jnp.cos(phase), local_time], axis=-1)
    embedded = _silu_mlp(_layers_at_time(basis.params["embed"], t), local)
    pooled = jnp.mean(embedded, axis=-2)
    head_input = jnp.concatenate([pooled, time_features], axis=-1)
    first_head = _layers_at_time(basis.params["head"], t)[0]
    return jax.nn.silu(head_input @ first_head["W"] + first_head["b"])


def evaluate_basis(
    basis: FrozenDeepSetsBasis,
    configurations: Array,
    times: Array,
    basis_size: int,
) -> BasisEvaluation:
    """Evaluate fixed basis coordinates and exact state Jacobians."""

    if basis_size < 1 or basis_size > basis.width:
        raise ValueError(f"basis_size must be in [1, {basis.width}]")
    x = jnp.asarray(configurations, dtype=jnp.float64)
    times = jnp.asarray(times, dtype=jnp.float64)

    def one(configuration: Array, time: Array) -> tuple[Array, Array]:
        function = lambda state: frozen_deepsets_latent(basis, state, time)[:basis_size]
        return function(configuration), jax.jacrev(function)(configuration)

    per_time = jax.vmap(lambda rows, time: jax.vmap(lambda row: one(row, time))(rows))
    values, gradients = per_time(x, times)
    return BasisEvaluation(values=values, state_gradients=gradients)


def build_galerkin_system(
    basis_evaluation: BasisEvaluation,
    weights: Array,
    forcing: Array,
) -> GalerkinSystem:
    values = jnp.asarray(basis_evaluation.values, dtype=jnp.float64)
    gradients = jnp.asarray(basis_evaluation.state_gradients, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    forcing = jnp.asarray(forcing, dtype=jnp.float64)
    means = jnp.einsum("tn,tnk->tk", weights, values)
    centered = values - means[:, None, :]
    raw_gram = jnp.einsum(
        "tn,tnjpd,tnkpd->tjk", weights, gradients, gradients
    )
    transpose = jnp.swapaxes(raw_gram, -1, -2)
    symmetry = jnp.linalg.norm(raw_gram - transpose, axis=(-2, -1)) / jnp.maximum(
        jnp.linalg.norm(raw_gram, axis=(-2, -1)), 1.0e-30
    )
    gram = 0.5 * (raw_gram + transpose)
    load = jnp.einsum("tn,tn,tnk->tk", weights, forcing, centered)
    forcing_mean = jnp.einsum("tn,tn->t", weights, forcing)
    return GalerkinSystem(
        gram=gram,
        load=load,
        basis_means=means,
        centered_basis=centered,
        weights=weights,
        forcing=forcing,
        raw_symmetry_residual=symmetry,
        forcing_mean=forcing_mean,
    )


def rank_aware_quadratic_solve(
    gram: Array,
    load: Array,
    *,
    relative_rank_tolerance: float,
) -> GalerkinSolve:
    """Minimum-norm pseudoinverse solution on the supported Gram range."""

    gram = 0.5 * (jnp.asarray(gram) + jnp.swapaxes(jnp.asarray(gram), -1, -2))
    load = jnp.asarray(load)

    def solve_one(matrix: Array, rhs: Array):
        eigenvalues, vectors = jnp.linalg.eigh(matrix)
        maximum = jnp.maximum(jnp.max(eigenvalues), 0.0)
        threshold = float(relative_rank_tolerance) * maximum
        retained = (eigenvalues > threshold) & (eigenvalues > 0.0)
        inverse = jnp.where(retained, 1.0 / jnp.maximum(eigenvalues, 1.0e-300), 0.0)
        coordinates = vectors.T @ rhs
        coefficients = -(vectors @ (inverse * coordinates))
        projected_rhs = vectors @ (retained.astype(rhs.dtype) * coordinates)
        range_residual = jnp.linalg.norm(rhs - projected_rhs) / jnp.maximum(
            jnp.linalg.norm(rhs), 1.0e-30
        )
        residual = matrix @ coefficients + rhs
        stationarity = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), 1.0e-30)
        retained_minimum = jnp.min(jnp.where(retained, eigenvalues, jnp.inf))
        condition = jnp.where(
            jnp.any(retained), maximum / jnp.maximum(retained_minimum, 1.0e-300), jnp.inf
        )
        action = coefficients @ matrix @ coefficients
        objective = 0.5 * action + rhs @ coefficients
        identity = jnp.abs(action + 2.0 * objective) / jnp.maximum(jnp.abs(action), 1.0e-30)
        return (
            coefficients,
            eigenvalues,
            retained,
            jnp.sum(retained),
            range_residual,
            stationarity,
            condition,
            action,
            objective,
            identity,
        )

    values = jax.vmap(solve_one)(gram, load)
    return GalerkinSolve(*values)


def aggregate_quadratic_values(
    solve: GalerkinSolve, time_weights: Array
) -> dict[str, Array]:
    weights = jnp.asarray(time_weights, dtype=jnp.float64)
    action = jnp.sum(weights * solve.action_by_time)
    objective = jnp.sum(weights * solve.objective_by_time)
    identity = jnp.abs(action + 2.0 * objective) / jnp.maximum(jnp.abs(action), 1.0e-30)
    return {
        "action": action,
        "objective": objective,
        "identity_relerr": identity,
    }


def system_at_eta(
    eta: Array,
    problem: FrozenEtaProblem,
    bank: ReferenceBank,
    basis_evaluation: BasisEvaluation,
) -> tuple[GalerkinSystem, Any]:
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    return build_galerkin_system(
        basis_evaluation, state.projection.weights, state.forcing
    ), state


def galerkin_envelope_value_and_grad(
    eta: Array,
    coefficients_fixed: Array,
    problem: FrozenEtaProblem,
    bank: ReferenceBank,
    basis_evaluation: BasisEvaluation,
) -> tuple[Array, Array]:
    """Return ``-2 J(a_fixed, eta)`` without differentiating the eigensolve."""

    coefficients_fixed = jnp.asarray(coefficients_fixed, dtype=jnp.float64)

    def value(design: Array) -> Array:
        system, _ = system_at_eta(design, problem, bank, basis_evaluation)
        kinetic = jnp.einsum(
            "ti,tij,tj->t", coefficients_fixed, system.gram, coefficients_fixed
        )
        linear = jnp.einsum("ti,ti->t", system.load, coefficients_fixed)
        objective = 0.5 * kinetic + linear
        return -2.0 * jnp.sum(problem.time_weights * objective)

    return jax.value_and_grad(value)(jnp.asarray(eta, dtype=jnp.float64))


def _certificate_payload(
    coefficients: Array,
    basis_evaluation: BasisEvaluation,
    problem: FrozenEtaProblem,
    bank: ReferenceBank,
    weights: Array,
    forcing: Array,
    eta: Array,
    target_derivatives: Array,
    thresholds: CertificateConfig,
) -> dict[str, Any]:
    """Held-out weak and physical diagnostics equivalent to Deep Ritz audit."""

    values = jnp.einsum("tk,tnk->tn", coefficients, basis_evaluation.values)
    gradients = jnp.einsum(
        "tk,tnkpd->tnpd", coefficients, basis_evaluation.state_gradients
    )
    value_means = jnp.einsum("tn,tn->t", weights, values)
    centered_values = values - value_means[:, None]
    feature_function = lambda state: _audit_features(state, problem.box)
    per_sample = jax.vmap(
        lambda state: (feature_function(state), jax.jacrev(feature_function)(state))
    )
    tests, test_gradients = jax.vmap(per_sample)(bank.configurations)
    test_means = jnp.einsum("tn,tnk->tk", weights, tests)
    centered_tests = tests - test_means[:, None, :]
    kinetic_rows = jnp.sum(gradients * gradients, axis=(-2, -1))
    kinetic = jnp.einsum("tn,tn->t", weights, kinetic_rows)
    linear = jnp.einsum("tn,tn,tn->t", weights, forcing, centered_values)
    gauge = jnp.einsum("tn,tn->t", weights, centered_values)
    weak_left = jnp.einsum(
        "tn,tnpd,tnkpd->tk", weights, gradients, test_gradients
    )
    weak_right = jnp.einsum("tn,tn,tnk->tk", weights, forcing, centered_tests)
    grad_scale = jnp.einsum(
        "tn,tnkpd,tnkpd->tk", weights, test_gradients, test_gradients
    )
    forcing_scale = jnp.einsum("tn,tn,tn->t", weights, forcing, forcing)
    test_scale = jnp.einsum("tn,tnk,tnk->tk", weights, centered_tests, centered_tests)
    weak = jnp.abs(weak_left + weak_right) / jnp.maximum(
        jnp.sqrt(kinetic)[:, None] * jnp.sqrt(grad_scale)
        + jnp.sqrt(forcing_scale)[:, None] * jnp.sqrt(test_scale),
        1.0e-12,
    )
    energy = jnp.abs(kinetic + linear) / jnp.maximum(
        kinetic + jnp.abs(linear), 1.0e-12
    )
    corrected = problem.family.jvp(bank.configurations, -gradients, eta)
    corrected_rate = jnp.einsum("tn,tnr->tr", weights, corrected)
    advective = problem.family.jvp(bank.configurations, bank.velocity, eta)
    advective_rate = jnp.einsum("tn,tnr->tr", weights, advective)
    rhs = target_derivatives - advective_rate
    moment_rate = jnp.max(
        jnp.linalg.norm(corrected_rate - rhs, axis=-1)
        / jnp.maximum(1.0, jnp.linalg.norm(rhs, axis=-1))
    )
    action = jnp.sum(problem.time_weights * kinetic)
    maximum_weak = float(jnp.max(weak))
    maximum_energy = float(jnp.max(energy))
    maximum_gauge = float(jnp.max(jnp.abs(gauge)))
    maximum_moment = float(moment_rate)
    finite = all(np.isfinite(value) for value in (
        maximum_weak, maximum_energy, maximum_gauge, maximum_moment, float(action)
    ))
    valid = bool(
        finite
        and maximum_weak <= thresholds.maximum_weak_residual
        and maximum_energy <= thresholds.maximum_energy_residual
        and maximum_gauge <= thresholds.maximum_gauge_residual
        and maximum_moment <= thresholds.maximum_moment_rate_residual
    )
    return {
        "action": float(action),
        "maximum_weak_residual": maximum_weak,
        "weak_residual_by_time_and_feature": jax.device_get(weak).tolist(),
        "maximum_energy_residual": maximum_energy,
        "energy_residual_by_time": jax.device_get(energy).tolist(),
        "maximum_gauge_residual": maximum_gauge,
        "maximum_moment_rate_residual": maximum_moment,
        "valid": valid,
        "thresholds": {
            "maximum_weak_residual": thresholds.maximum_weak_residual,
            "maximum_energy_residual": thresholds.maximum_energy_residual,
            "maximum_gauge_residual": thresholds.maximum_gauge_residual,
            "maximum_moment_rate_residual": thresholds.maximum_moment_rate_residual,
        },
    }


def evaluate_fixed_eta(
    eta: Array,
    cfg: dict[str, Any],
    data: PreparedExperiment,
    basis: FrozenDeepSetsBasis,
    basis_size: int,
    *,
    train_basis: BasisEvaluation | None = None,
    audit_basis: BasisEvaluation | None = None,
) -> tuple[dict[str, Any], GalerkinSolve, BasisEvaluation]:
    problem = data.selection_problem
    eta = wrap_periodic(eta, problem.family)
    if train_basis is None:
        train_basis = evaluate_basis(
            basis, data.ritz_train_bank.configurations, problem.times, basis_size
        )
    if audit_basis is None:
        audit_basis = evaluate_basis(
            basis, data.ritz_audit_bank.configurations, problem.times, basis_size
        )
    reconstruction = reconstruct_moments(eta, problem)
    train_state = forcing_state(eta, problem, data.ritz_train_bank, reconstruction)
    system = build_galerkin_system(
        train_basis, train_state.projection.weights, train_state.forcing
    )
    galerkin_cfg = cfg["galerkin"]
    solve = rank_aware_quadratic_solve(
        system.gram,
        system.load,
        relative_rank_tolerance=float(galerkin_cfg["relative_rank_tolerance"]),
    )
    aggregate = aggregate_quadratic_values(solve, problem.time_weights)
    audit_state = forcing_state(eta, problem, data.ritz_audit_bank, reconstruction)
    certificates = CertificateConfig(**galerkin_cfg["certificate_thresholds"])
    certificate = _certificate_payload(
        solve.coefficients,
        audit_basis,
        problem,
        data.ritz_audit_bank,
        audit_state.projection.weights,
        audit_state.forcing,
        eta,
        reconstruction.derivatives,
        certificates,
    )
    train_forcing = hard_forcing_audit(eta, problem, data.ritz_train_bank)
    audit_forcing = hard_forcing_audit(eta, problem, data.ritz_audit_bank)
    algebra_valid = bool(
        float(aggregate["identity_relerr"])
        <= float(galerkin_cfg["maximum_identity_relerr"])
        and float(jnp.max(solve.stationarity_residual))
        <= float(galerkin_cfg["maximum_stationarity_residual"])
        and float(jnp.max(solve.range_residual))
        <= float(galerkin_cfg["maximum_range_residual"])
        and float(jnp.max(system.raw_symmetry_residual))
        <= float(galerkin_cfg["maximum_symmetry_residual"])
    )
    physical_valid = bool(
        algebra_valid
        and certificate["valid"]
        and train_forcing["valid"]
        and audit_forcing["valid"]
        and bool(jax.device_get(problem.family.geometry_valid(eta)))
    )
    minimum_eigenvalue = jnp.min(solve.eigenvalues, axis=-1)
    maximum_eigenvalue = jnp.max(solve.eigenvalues, axis=-1)
    payload = {
        "basis_family": basis.name,
        "basis_source": basis.source,
        "basis_source_sha256": basis.source_sha256,
        "basis_width": basis.width,
        "basis_size": int(basis_size),
        "eta": jax.device_get(eta).tolist(),
        "galerkin_action": float(aggregate["action"]),
        "galerkin_objective": float(aggregate["objective"]),
        "aggregate_identity_relerr": float(aggregate["identity_relerr"]),
        "action_by_time": jax.device_get(solve.action_by_time).tolist(),
        "objective_by_time": jax.device_get(solve.objective_by_time).tolist(),
        "identity_relerr_by_time": jax.device_get(solve.identity_relerr_by_time).tolist(),
        "stationarity_residual_by_time": jax.device_get(solve.stationarity_residual).tolist(),
        "range_residual_by_time": jax.device_get(solve.range_residual).tolist(),
        "numerical_rank_by_time": jax.device_get(solve.numerical_rank).tolist(),
        "condition_number_by_time": jax.device_get(solve.condition_number).tolist(),
        "minimum_eigenvalue_by_time": jax.device_get(minimum_eigenvalue).tolist(),
        "maximum_eigenvalue_by_time": jax.device_get(maximum_eigenvalue).tolist(),
        "symmetry_residual_by_time": jax.device_get(system.raw_symmetry_residual).tolist(),
        "forcing_mean_by_time": jax.device_get(system.forcing_mean).tolist(),
        "worst_stationarity_residual": float(jnp.max(solve.stationarity_residual)),
        "mean_stationarity_residual": float(jnp.mean(solve.stationarity_residual)),
        "worst_range_residual": float(jnp.max(solve.range_residual)),
        "mean_range_residual": float(jnp.mean(solve.range_residual)),
        "worst_condition_number": float(jnp.max(solve.condition_number)),
        "worst_symmetry_residual": float(jnp.max(system.raw_symmetry_residual)),
        "algebra_valid": algebra_valid,
        "physical_valid": physical_valid,
        "held_out_certificate": certificate,
        "train_forcing_audit": train_forcing,
        "audit_forcing_audit": audit_forcing,
        "geometry_valid": bool(jax.device_get(problem.family.geometry_valid(eta))),
        "periodic_branch_distance": float(periodic_branch_distance(eta, problem.family)),
    }
    return payload, solve, train_basis


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_basis_families(cfg: dict[str, Any]) -> list[FrozenDeepSetsBasis]:
    """Load the incumbent trunk when available plus a same-shape random control."""

    checkpoint = OUTPUT_ROOT / "gradient_checks" / "smoke" / "theta_center.npz"
    families: list[FrozenDeepSetsBasis] = []
    if checkpoint.is_file():
        params, _ = load_ritz_checkpoint(checkpoint)
        families.append(FrozenDeepSetsBasis(
            params=params,
            name="incumbent_trained_latent",
            source=str(checkpoint),
            source_sha256=_sha256(checkpoint),
            box=tuple(cfg["physics"]["box"]),
        ))
        width = int(params["head"][0]["W"].shape[-1])
        hidden_layers = len(params["embed"])
        independent_time_nodes = (
            int(params["embed"][0]["W"].shape[0])
            if params["embed"][0]["W"].ndim == 3 else 0
        )
    else:
        width = int(cfg["deep_ritz"]["hidden_width"])
        hidden_layers = int(cfg["deep_ritz"]["hidden_layers"])
        independent_time_nodes = 0
    random_params = init_ritz_params(
        jax.random.PRNGKey(int(cfg["galerkin"]["random_basis_seed"])),
        hidden_width=width,
        hidden_layers=hidden_layers,
        independent_time_nodes=independent_time_nodes,
    )
    families.append(FrozenDeepSetsBasis(
        params=random_params,
        name="deterministic_random_latent",
        source=f"jax.random.PRNGKey({int(cfg['galerkin']['random_basis_seed'])})",
        source_sha256=None,
        box=tuple(cfg["physics"]["box"]),
    ))
    return families


def basis_size_ladder(basis: FrozenDeepSetsBasis, cfg: dict[str, Any]) -> list[int]:
    requested = [int(value) for value in cfg["galerkin"]["basis_size_ladder"]]
    ladder = sorted(set(value for value in requested if 0 < value <= basis.width))
    if basis.width not in ladder:
        ladder.append(basis.width)
    return ladder


def save_basis_checkpoint(path: Path, basis: FrozenDeepSetsBasis) -> None:
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ritz_checkpoint(path, basis.params, metadata={
        "role": "frozen_galerkin_basis",
        "basis_family": basis.name,
        "source": basis.source,
        "source_sha256": basis.source_sha256,
        "basis_width": basis.width,
    })


def save_galerkin_arrays(
    path: Path,
    solve: GalerkinSolve,
    basis_evaluation: BasisEvaluation,
) -> None:
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coefficients=np.asarray(jax.device_get(solve.coefficients)),
        eigenvalues=np.asarray(jax.device_get(solve.eigenvalues)),
        retained=np.asarray(jax.device_get(solve.retained)),
        basis_values=np.asarray(jax.device_get(basis_evaluation.values)),
        basis_state_gradients=np.asarray(jax.device_get(basis_evaluation.state_gradients)),
    )


__all__ = [
    "BasisEvaluation",
    "FrozenDeepSetsBasis",
    "GalerkinSolve",
    "GalerkinSystem",
    "aggregate_quadratic_values",
    "basis_size_ladder",
    "build_galerkin_system",
    "evaluate_basis",
    "evaluate_fixed_eta",
    "frozen_deepsets_latent",
    "galerkin_envelope_value_and_grad",
    "load_basis_families",
    "rank_aware_quadratic_solve",
    "save_basis_checkpoint",
    "save_galerkin_arrays",
    "system_at_eta",
]
