"""Pure-JAX K=280 Galerkin and Tangent kernels for the prospective B1 V2 study.

This module intentionally has no native/Tesseract Galerkin branch.  The only
Galerkin assembly and solve operations are JAX float64 kernels.  Candidate-
independent dictionary evaluation is fused with chunk sufficient-statistic
accumulation so full state Jacobians never cross the device boundary or remain
materialized for an entire bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .domain import minimum_image
from .full_gradient import forcing_state, reconstruct_moments, wrap_periodic
from .galerkin import (
    GalerkinSystem,
    aggregate_quadratic_values,
    rank_aware_quadratic_solve,
)
from .galerkin_only import GalerkinCertificateThresholds
from .galerkin_only_data import selection_risk
from .production_basis import HybridInvariantDictionary, load_dictionary
from .production_galerkin import _normalized_chunk, audit_hybrid_solutions


Array = jax.Array
K = 280


def _synchronize(value: Any) -> Any:
    """Block on the first array leaf and return ``value`` unchanged."""

    leaves = jax.tree.leaves(value)
    if leaves:
        leaves[0].block_until_ready()
    return value


def forcing_payload(state: Any, problem: Any) -> dict[str, Any]:
    residual = float(jnp.max(jnp.linalg.norm(state.projection.residual, axis=-1)))
    minimum_ress = float(jnp.min(state.projection.ess_fraction))
    forcing_mean = float(jnp.max(jnp.abs(state.forcing_mean_before_centering)))
    post_mean = float(jnp.max(jnp.abs(jnp.einsum(
        "tn,tn->t", state.projection.weights, state.forcing
    ))))
    covariance_condition = float(jnp.max(state.covariance_condition))
    cfg = problem.forcing_config
    finite = all(np.isfinite(value) for value in (
        residual, minimum_ress, forcing_mean, post_mean, covariance_condition
    ))
    return {
        "valid": bool(
            finite
            and residual <= cfg.projection_tolerance
            and minimum_ress >= cfg.minimum_ess_fraction
            and forcing_mean <= cfg.forcing_mean_tolerance
            and covariance_condition <= cfg.max_covariance_condition
        ),
        "maximum_projection_residual": residual,
        "minimum_ess_fraction": minimum_ress,
        "maximum_forcing_mean": forcing_mean,
        "maximum_post_centering_forcing_mean": post_mean,
        "maximum_covariance_condition": covariance_condition,
    }


def _algebra_valid(cfg: dict[str, Any], payload: dict[str, Any]) -> bool:
    settings = cfg["production_galerkin"]
    return bool(
        payload["identity_relerr"] <= float(settings["maximum_identity_relerr"])
        and payload["worst_range_residual"] <= float(settings["maximum_range_residual"])
        and payload["worst_stationarity_residual"]
        <= float(settings["maximum_stationarity_residual"])
        and payload["worst_symmetry_residual"]
        <= float(settings["maximum_symmetry_residual"])
        and payload["worst_retained_condition"]
        <= float(settings["maximum_retained_condition"])
        and payload["minimum_rank_fraction"]
        >= float(settings["minimum_rank_fraction"])
    )


@dataclass(frozen=True)
class TimedEvaluation:
    payload: dict[str, Any]
    timings: dict[str, float]


class JaxGalerkinContext:
    """Compile-once, chunked JAX evaluator for one pair of scientific banks."""

    def __init__(
        self,
        cfg: dict[str, Any],
        data: Any,
        dictionary_path: Any,
        *,
        chunk_size: int | None = None,
    ) -> None:
        if not bool(jax.config.jax_enable_x64):
            raise RuntimeError("authoritative V2 requires JAX_ENABLE_X64=1")
        if str(data.selection_problem.projection_backend) != "jax":
            raise RuntimeError("V2 projection backend must be JAX")
        self.cfg = cfg
        self.data = data
        self.dictionary = load_dictionary(
            dictionary_path, box=tuple(cfg["physics"]["box"])
        )
        if self.dictionary.size != K:
            raise RuntimeError(f"V2 dictionary size {self.dictionary.size} != {K}")
        configured = int(cfg["production_galerkin"]["chunk_size"])
        self.chunk_size = int(chunk_size or configured)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._stats = [
            self._make_statistics_kernel(time_index)
            for time_index in range(int(data.train_bank.configurations.shape[0]))
        ]
        self._potential = [
            self._make_potential_kernel(time_index)
            for time_index in range(int(data.train_bank.configurations.shape[0]))
        ]
        problem, bank = data.selection_problem, data.train_bank

        def envelope(eta: Array, potentials: Array, kinetics: Array) -> Array:
            reconstruction = reconstruct_moments(eta, problem)
            state = forcing_state(eta, problem, bank, reconstruction)
            weights, source = state.projection.weights, state.forcing
            kinetic = jnp.einsum("tn,tn->t", weights, kinetics)
            potential_mean = jnp.einsum("tn,tn->t", weights, potentials)
            source_mean = jnp.einsum("tn,tn->t", weights, source)
            linear = (
                jnp.einsum("tn,tn,tn->t", weights, source, potentials)
                - source_mean * potential_mean
            )
            return -2.0 * jnp.sum(
                problem.time_weights * (0.5 * kinetic + linear)
            )

        self._envelope = jax.jit(jax.value_and_grad(envelope, argnums=0))
        self._risk = jax.jit(lambda eta: selection_risk(eta, data))
        self._risk_value_grad = jax.jit(
            jax.value_and_grad(lambda eta: selection_risk(eta, data))
        )

    def _make_statistics_kernel(self, time_index: int):
        dictionary = self.dictionary

        @jax.jit
        def kernel(rows: Array, weights: Array, source: Array):
            values, gradients = _normalized_chunk(dictionary, rows, time_index)
            return (
                jnp.einsum("n,njpd,nkpd->jk", weights, gradients, gradients),
                jnp.einsum("n,n,nk->k", weights, source, values),
                jnp.einsum("n,nk->k", weights, values),
                jnp.einsum("n,n->", weights, source),
            )

        return kernel

    def _make_potential_kernel(self, time_index: int):
        dictionary = self.dictionary

        @jax.jit
        def kernel(rows: Array, coefficients: Array):
            values, gradients = _normalized_chunk(dictionary, rows, time_index)
            potential = jnp.einsum("k,nk->n", coefficients, values)
            gradient = jnp.einsum("k,nkpd->npd", coefficients, gradients)
            return potential, jnp.sum(gradient * gradient, axis=(-2, -1))

        return kernel

    def assemble(self, weights: Array, source: Array) -> GalerkinSystem:
        bank = self.data.train_bank
        sample_count = int(bank.configurations.shape[1])
        if sample_count % self.chunk_size:
            raise RuntimeError(
                "V2 static chunking requires bank size divisible by chunk_size"
            )
        grams, loads, means, symmetries, source_means = [], [], [], [], []
        for time_index, kernel in enumerate(self._stats):
            gram = jnp.zeros((K, K), dtype=jnp.float64)
            load = jnp.zeros((K,), dtype=jnp.float64)
            mean = jnp.zeros((K,), dtype=jnp.float64)
            source_mean = jnp.asarray(0.0, dtype=jnp.float64)
            for start in range(0, sample_count, self.chunk_size):
                stop = start + self.chunk_size
                chunk = kernel(
                    bank.configurations[time_index, start:stop],
                    weights[time_index, start:stop],
                    source[time_index, start:stop],
                )
                gram = gram + chunk[0]
                load = load + chunk[1]
                mean = mean + chunk[2]
                source_mean = source_mean + chunk[3]
            load = load - source_mean * mean
            symmetry = jnp.linalg.norm(gram - gram.T) / jnp.maximum(
                jnp.linalg.norm(gram), 1.0e-30
            )
            grams.append(0.5 * (gram + gram.T))
            loads.append(load)
            means.append(mean)
            symmetries.append(symmetry)
            source_means.append(source_mean)
        empty = jnp.zeros((0,), dtype=jnp.float64)
        return GalerkinSystem(
            jnp.stack(grams),
            jnp.stack(loads),
            jnp.stack(means),
            empty,
            empty,
            empty,
            jnp.stack(symmetries),
            jnp.stack(source_means),
        )

    def potential_rows(self, coefficients: Array) -> tuple[Array, Array]:
        bank = self.data.train_bank
        sample_count = int(bank.configurations.shape[1])
        potentials, kinetics = [], []
        for time_index, kernel in enumerate(self._potential):
            potential_chunks, kinetic_chunks = [], []
            for start in range(0, sample_count, self.chunk_size):
                stop = start + self.chunk_size
                potential, kinetic = kernel(
                    bank.configurations[time_index, start:stop],
                    coefficients[time_index],
                )
                potential_chunks.append(potential)
                kinetic_chunks.append(kinetic)
            potentials.append(jnp.concatenate(potential_chunks))
            kinetics.append(jnp.concatenate(kinetic_chunks))
        return jnp.stack(potentials), jnp.stack(kinetics)

    def exact_risk(self, eta: Any, *, gradient: bool = False):
        eta = wrap_periodic(
            jnp.asarray(eta, dtype=jnp.float64),
            self.data.selection_problem.family,
        )
        return self._risk_value_grad(eta) if gradient else self._risk(eta)

    def evaluate(self, eta: Any, *, gradient: bool = False) -> TimedEvaluation:
        problem, bank = self.data.selection_problem, self.data.train_bank
        eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
        timings: dict[str, float] = {}

        started = time.perf_counter()
        reconstruction = _synchronize(reconstruct_moments(eta, problem))
        timings["observation_reconstruction"] = time.perf_counter() - started

        started = time.perf_counter()
        state = _synchronize(forcing_state(eta, problem, bank, reconstruction))
        timings["information_projection_and_forcing"] = time.perf_counter() - started

        started = time.perf_counter()
        system = _synchronize(self.assemble(state.projection.weights, state.forcing))
        timings["basis_and_gram_load"] = time.perf_counter() - started

        started = time.perf_counter()
        solve = _synchronize(rank_aware_quadratic_solve(
            system.gram,
            system.load,
            relative_rank_tolerance=float(
                self.cfg["production_galerkin"]["relative_rank_tolerance"]
            ),
        ))
        aggregate = _synchronize(
            aggregate_quadratic_values(solve, problem.time_weights)
        )
        timings["k280_solve"] = time.perf_counter() - started

        action = aggregate["action"]
        derivative = None
        if gradient:
            started = time.perf_counter()
            potentials, kinetics = _synchronize(self.potential_rows(solve.coefficients))
            action, derivative = _synchronize(
                self._envelope(eta, potentials, kinetics)
            )
            timings["action_gradient"] = time.perf_counter() - started

        started = time.perf_counter()
        risk = _synchronize(self._risk(eta))
        timings["risk"] = time.perf_counter() - started

        ranks = solve.numerical_rank
        payload = {
            "eta": np.asarray(eta).tolist(),
            "action": float(action),
            "action_by_time": np.asarray(solve.action_by_time).tolist(),
            "risk": float(risk),
            "gradient": None if derivative is None else np.asarray(derivative).tolist(),
            "gradient_norm": (
                None if derivative is None else float(jnp.linalg.norm(derivative))
            ),
            "identity_relerr": float(aggregate["identity_relerr"]),
            "rank_by_time": np.asarray(ranks).tolist(),
            "minimum_rank_fraction": float(jnp.min(ranks / float(K))),
            "worst_range_residual": float(jnp.max(solve.range_residual)),
            "worst_stationarity_residual": float(
                jnp.max(solve.stationarity_residual)
            ),
            "worst_retained_condition": float(jnp.max(solve.condition_number)),
            "worst_symmetry_residual": float(
                jnp.max(system.raw_symmetry_residual)
            ),
            "galerkin_backend": "jax",
            "dtype": "float64",
            "train_forcing_audit": forcing_payload(state, problem),
            "geometry_valid": bool(problem.family.geometry_valid(eta)),
            "_eta": eta,
            "_reconstruction": reconstruction,
            "_solve": solve,
        }
        payload["algebra_valid"] = _algebra_valid(self.cfg, payload)
        payload["search_valid"] = bool(
            payload["algebra_valid"]
            and payload["geometry_valid"]
            and payload["train_forcing_audit"]["valid"]
        )
        return TimedEvaluation(payload, timings)

    def audit(self, evaluation: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        eta = evaluation["_eta"]
        reconstruction = evaluation["_reconstruction"]
        state = forcing_state(
            eta, self.data.selection_problem, self.data.audit_bank, reconstruction
        )
        adapter = SimpleNamespace(
            ritz_audit_bank=self.data.audit_bank,
            selection_problem=self.data.selection_problem,
        )
        certificate = audit_hybrid_solutions(
            self.dictionary,
            evaluation["_solve"].coefficients[None],
            adapter,
            eta,
            reconstruction,
            state,
            GalerkinCertificateThresholds(
                **self.cfg["production_galerkin"]["certificate_thresholds"]
            ),
            chunk_size=self.chunk_size,
        )[0]
        forcing = forcing_payload(state, self.data.selection_problem)
        result = {
            "audit_forcing": forcing,
            "heldout_certificate": certificate,
            "valid": bool(
                evaluation["search_valid"] and forcing["valid"] and certificate["valid"]
            ),
        }
        return result, time.perf_counter() - started


def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def local_density_gradients(
    configurations: Array, eta: Array, family: Any
) -> Array:
    x = jnp.asarray(configurations, dtype=jnp.float64)
    centers = family.centers(eta)
    delta = minimum_image(
        x[..., :, None, :] - centers,
        jnp.asarray(family.box, dtype=x.dtype),
    )
    kernel = jnp.exp(
        -0.5 * jnp.sum(delta * delta, axis=-1) / float(family.width) ** 2
    )
    return -kernel[..., None] * delta / (
        float(x.shape[-2]) * float(family.width) ** 2
    )


def tangent_evaluate(data: Any, eta: Any, *, gradient: bool) -> dict[str, Any]:
    problem, bank = data.selection_problem, data.train_bank
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)

    def value(point: Array) -> Array:
        reconstruction = reconstruct_moments(point, problem)
        state = forcing_state(point, problem, bank, reconstruction)
        gradients = local_density_gradients(
            bank.configurations, point, problem.family
        )
        advective = problem.family.jvp(bank.configurations, bank.velocity, point)
        gram = jnp.einsum(
            "tn,tnpjd,tnpkd->tjk",
            state.projection.weights,
            gradients,
            gradients,
        )
        rate = reconstruction.derivatives - jnp.einsum(
            "tn,tnr->tr", state.projection.weights, advective
        )
        coefficients = jax.vmap(jnp.linalg.solve)(gram, rate)
        return jnp.sum(
            problem.time_weights * jnp.einsum("tr,tr->t", coefficients, rate)
        )

    action, derivative = (
        jax.value_and_grad(value)(eta) if gradient else (value(eta), None)
    )
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    forcing = forcing_payload(state, problem)
    return {
        "eta": np.asarray(eta).tolist(),
        "action": float(action),
        "gradient": None if derivative is None else np.asarray(derivative).tolist(),
        "gradient_norm": (
            None if derivative is None else float(jnp.linalg.norm(derivative))
        ),
        "risk": float(selection_risk(eta, data)),
        "forcing": forcing,
        "geometry_valid": bool(problem.family.geometry_valid(eta)),
        "valid": bool(forcing["valid"] and problem.family.geometry_valid(eta)),
    }


def tangent_audit(data: Any, eta: Any, *, use_train: bool = False) -> dict[str, Any]:
    problem = data.selection_problem
    bank = data.train_bank if use_train else data.audit_bank
    eta = wrap_periodic(jnp.asarray(eta, dtype=jnp.float64), problem.family)
    reconstruction = reconstruct_moments(eta, problem)
    state = forcing_state(eta, problem, bank, reconstruction)
    weights = state.projection.weights
    gradients = local_density_gradients(bank.configurations, eta, problem.family)
    advective = problem.family.jvp(bank.configurations, bank.velocity, eta)
    gram = jnp.einsum("tn,tnpjd,tnpkd->tjk", weights, gradients, gradients)
    advective_rate = jnp.einsum("tn,tnr->tr", weights, advective)
    required_rate = reconstruction.derivatives - advective_rate
    coefficients = jax.vmap(jnp.linalg.solve)(gram, required_rate)
    tangent_velocity = jnp.einsum("tr,tnprd->tnpd", coefficients, gradients)
    energy_rows = jnp.sum(tangent_velocity * tangent_velocity, axis=(-2, -1))
    kinetic = jnp.einsum("tn,tn->t", weights, energy_rows)
    kinetic_second = jnp.einsum("tn,tn->t", weights, energy_rows * energy_rows)
    action = jnp.sum(problem.time_weights * kinetic)
    corrected_rate = advective_rate + jnp.einsum("tjk,tk->tj", gram, coefficients)
    normalized_residual = jnp.linalg.norm(
        corrected_rate - reconstruction.derivatives, axis=-1
    ) / jnp.maximum(1.0, jnp.linalg.norm(required_rate, axis=-1))
    eigenvalues = jnp.linalg.eigvalsh(gram)
    condition = eigenvalues[:, -1] / jnp.maximum(eigenvalues[:, 0], 1.0e-300)
    ess = 1.0 / jnp.maximum(jnp.sum(weights * weights, axis=-1), 1.0e-300)
    variance = jnp.maximum(kinetic_second - kinetic * kinetic, 0.0)
    standard_error = jnp.sqrt(jnp.sum(
        (
            problem.time_weights
            * jnp.sqrt(variance / jnp.maximum(ess, 1.0))
        ) ** 2
    ))
    forcing = forcing_payload(state, problem)
    maximum_condition = float(jnp.max(condition))
    maximum_residual = float(jnp.max(normalized_residual))
    finite = all(np.isfinite(value) for value in (
        float(action), float(standard_error), maximum_condition, maximum_residual
    ))
    valid = bool(
        finite
        and forcing["valid"]
        and maximum_condition <= 1.0e8
        and maximum_residual <= 1.0e-10
        and float(jnp.min(eigenvalues)) > 0.0
    )
    return {
        "action": float(action),
        "action_standard_error": float(standard_error),
        "action_by_time": np.asarray(kinetic).tolist(),
        "maximum_gram_condition": maximum_condition,
        "minimum_gram_eigenvalue": float(jnp.min(eigenvalues)),
        "maximum_moment_rate_residual": maximum_residual,
        "moment_rate_residual_by_time": np.asarray(normalized_residual).tolist(),
        "forcing": forcing,
        "valid": valid,
        "thresholds": {
            "maximum_gram_condition": 1.0e8,
            "maximum_moment_rate_residual": 1.0e-10,
        },
    }


__all__ = [
    "JaxGalerkinContext",
    "K",
    "TimedEvaluation",
    "forcing_payload",
    "local_density_gradients",
    "public_payload",
    "tangent_audit",
    "tangent_evaluate",
]
