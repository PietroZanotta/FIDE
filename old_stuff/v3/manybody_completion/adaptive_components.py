"""Learned finite-budget components shared by the two DiffPOP Tesseracts.

The models in this module improve *how* the exact exponential tilt is reached:

* ``ProposalModel`` defines a defensive, tilt-conditioned proposal used by the
  tilted-ensemble Tesseract.  Importance weights and Metropolis corrections
  preserve the requested target distribution.
* ``WarmStartModel`` predicts the first dual iterate used by the dual-calibration
  Tesseract.  Covariance-Newton updates remain responsible for final moment
  calibration.

The finite-support benchmark lets us pretrain and differentiate these components
with exact sums.  The operational evaluation still uses stochastic particles,
so improvements in ESS, sampler calls, and fresh-sample calibration are measured
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .homometric import PopulationSupport

jax.config.update("jax_enable_x64", True)

MLPParameters = tuple[tuple[jax.Array, jax.Array], ...]


@dataclass(frozen=True)
class ProposalArchitecture:
    hidden_width: int
    hidden_layers: int
    input_dim: int = 7

    @property
    def parameter_count(self) -> int:
        dimensions = [self.input_dim]
        dimensions.extend([self.hidden_width] * self.hidden_layers)
        dimensions.append(1)
        return int(
            sum((left * right) + right for left, right in zip(dimensions[:-1], dimensions[1:]))
        )


@dataclass(frozen=True)
class ProposalModel:
    architecture: ProposalArchitecture
    parameters: MLPParameters
    defensive_mixture: float = 0.10
    max_logit_correction: float = 8.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "hidden_width": self.architecture.hidden_width,
            "hidden_layers": self.architecture.hidden_layers,
            "input_dim": self.architecture.input_dim,
            "defensive_mixture": self.defensive_mixture,
            "max_logit_correction": self.max_logit_correction,
            "parameters": [
                {"weight": np.asarray(weight).tolist(), "bias": np.asarray(bias).tolist()}
                for weight, bias in self.parameters
            ],
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ProposalModel":
        architecture = ProposalArchitecture(
            hidden_width=int(mapping["hidden_width"]),
            hidden_layers=int(mapping["hidden_layers"]),
            input_dim=int(mapping.get("input_dim", 7)),
        )
        parameters = tuple(
            (
                jnp.asarray(layer["weight"], dtype=jnp.float64),
                jnp.asarray(layer["bias"], dtype=jnp.float64),
            )
            for layer in mapping["parameters"]
        )
        return cls(
            architecture=architecture,
            parameters=parameters,
            defensive_mixture=float(mapping.get("defensive_mixture", 0.10)),
            max_logit_correction=float(mapping.get("max_logit_correction", 8.0)),
        )


@dataclass(frozen=True)
class WarmStartArchitecture:
    hidden_width: int
    hidden_layers: int
    input_dim: int = 7

    @property
    def parameter_count(self) -> int:
        dimensions = [self.input_dim]
        dimensions.extend([self.hidden_width] * self.hidden_layers)
        dimensions.append(1)
        return int(
            sum((left * right) + right for left, right in zip(dimensions[:-1], dimensions[1:]))
        )


@dataclass(frozen=True)
class WarmStartModel:
    architecture: WarmStartArchitecture
    parameters: MLPParameters
    max_abs_dual: float = 20.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "hidden_width": self.architecture.hidden_width,
            "hidden_layers": self.architecture.hidden_layers,
            "input_dim": self.architecture.input_dim,
            "max_abs_dual": self.max_abs_dual,
            "parameters": [
                {"weight": np.asarray(weight).tolist(), "bias": np.asarray(bias).tolist()}
                for weight, bias in self.parameters
            ],
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "WarmStartModel":
        architecture = WarmStartArchitecture(
            hidden_width=int(mapping["hidden_width"]),
            hidden_layers=int(mapping["hidden_layers"]),
            input_dim=int(mapping.get("input_dim", 7)),
        )
        parameters = tuple(
            (
                jnp.asarray(layer["weight"], dtype=jnp.float64),
                jnp.asarray(layer["bias"], dtype=jnp.float64),
            )
            for layer in mapping["parameters"]
        )
        return cls(
            architecture=architecture,
            parameters=parameters,
            max_abs_dual=float(mapping.get("max_abs_dual", 20.0)),
        )


def _initialize_mlp(
    input_dim: int,
    hidden_width: int,
    hidden_layers: int,
    output_dim: int,
    *,
    seed: int,
    final_zero: bool,
) -> MLPParameters:
    key = jax.random.PRNGKey(int(seed))
    dimensions = [input_dim]
    dimensions.extend([hidden_width] * hidden_layers)
    dimensions.append(output_dim)
    layers: list[tuple[jax.Array, jax.Array]] = []
    for index, (fan_in, fan_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        key, weight_key = jax.random.split(key)
        if final_zero and index + 1 == len(dimensions) - 1:
            weight = jnp.zeros((fan_in, fan_out), dtype=jnp.float64)
        else:
            scale = np.sqrt(2.0 / float(fan_in + fan_out))
            weight = scale * jax.random.normal(
                weight_key, (fan_in, fan_out), dtype=jnp.float64
            )
        bias = jnp.zeros((fan_out,), dtype=jnp.float64)
        layers.append((weight, bias))
    return tuple(layers)


def initialize_proposal_model(
    architecture: ProposalArchitecture,
    *,
    seed: int,
    defensive_mixture: float = 0.10,
    max_logit_correction: float = 8.0,
) -> ProposalModel:
    if not 0.0 < defensive_mixture <= 1.0:
        raise ValueError("defensive_mixture must lie in (0, 1]")
    return ProposalModel(
        architecture=architecture,
        parameters=_initialize_mlp(
            architecture.input_dim,
            architecture.hidden_width,
            architecture.hidden_layers,
            1,
            seed=seed,
            final_zero=True,
        ),
        defensive_mixture=float(defensive_mixture),
        max_logit_correction=float(max_logit_correction),
    )


def initialize_warm_start_model(
    architecture: WarmStartArchitecture,
    *,
    seed: int,
    max_abs_dual: float = 20.0,
) -> WarmStartModel:
    return WarmStartModel(
        architecture=architecture,
        parameters=_initialize_mlp(
            architecture.input_dim,
            architecture.hidden_width,
            architecture.hidden_layers,
            1,
            seed=seed,
            final_zero=False,
        ),
        max_abs_dual=float(max_abs_dual),
    )


def _mlp(parameters: MLPParameters, values: jax.Array) -> jax.Array:
    hidden = values
    for layer_index, (weight, bias) in enumerate(parameters):
        hidden = hidden @ weight + bias
        if layer_index + 1 < len(parameters):
            hidden = jax.nn.silu(hidden)
    return hidden


def normalized_probabilities_jax(probabilities: jax.Array) -> jax.Array:
    values = jnp.maximum(jnp.asarray(probabilities, dtype=jnp.float64), 1e-15)
    return values / jnp.sum(values)


def exact_tilt_probabilities_jax(
    prior: jax.Array, pair_values: jax.Array, dual: jax.Array | float
) -> jax.Array:
    prior = normalized_probabilities_jax(prior)
    logits = jnp.log(prior) + jnp.asarray(dual, dtype=prior.dtype) * pair_values
    return jax.nn.softmax(logits)


def distribution_summary_features_jax(
    prior: jax.Array,
    pair_values: jax.Array,
    triplet_values: jax.Array,
    labels: jax.Array,
    target_moment: jax.Array | float,
) -> jax.Array:
    prior = normalized_probabilities_jax(prior)
    _ = triplet_values, labels  # held-out descriptors are deliberately not used
    pair_mean = jnp.sum(prior * pair_values)
    centered = pair_values - pair_mean
    pair_variance = jnp.sum(prior * jnp.square(centered))
    pair_third_central = jnp.sum(prior * jnp.power(centered, 3))
    pair_fourth_central = jnp.sum(prior * jnp.power(centered, 4))
    target = jnp.asarray(target_moment, dtype=prior.dtype)
    return jnp.asarray(
        [
            pair_mean,
            pair_variance,
            pair_third_central,
            pair_fourth_central,
            target,
            target - pair_mean,
            jnp.abs(target - pair_mean),
        ],
        dtype=prior.dtype,
    )


def proposal_probabilities_jax(
    parameters: MLPParameters,
    architecture: ProposalArchitecture,
    prior: jax.Array,
    pair_values: jax.Array,
    triplet_values: jax.Array,
    labels: jax.Array,
    dual: jax.Array | float,
    *,
    defensive_mixture: float,
    max_logit_correction: float,
) -> jax.Array:
    """Return a defensive proposal with full prior support.

    The network predicts an additive correction to ``log prior``.  The final
    proposal is mixed with the original prior, ensuring that any atom supported
    by the flow prior remains reachable even if the learned proposal is poor.
    """
    if architecture.input_dim != 7:
        raise ValueError("the current proposal feature map requires input_dim=7")
    prior = normalized_probabilities_jax(prior)
    dual_value = jnp.asarray(dual, dtype=prior.dtype)
    _ = triplet_values, labels  # no held-out or latent-label feature leakage
    pair_mean = jnp.sum(prior * pair_values)
    centered = pair_values - pair_mean
    pair_variance = jnp.sum(prior * jnp.square(centered))
    features = jnp.stack(
        [
            pair_values,
            jnp.log(prior),
            jnp.full_like(pair_values, dual_value),
            dual_value * pair_values,
            centered,
            jnp.square(centered),
            jnp.full_like(pair_values, pair_variance),
        ],
        axis=1,
    )
    raw_correction = _mlp(parameters, features)[:, 0]
    correction = max_logit_correction * jnp.tanh(
        raw_correction / max(max_logit_correction, 1e-12)
    )
    learned = jax.nn.softmax(jnp.log(prior) + correction)
    proposal = (1.0 - defensive_mixture) * learned + defensive_mixture * prior
    return normalized_probabilities_jax(proposal)


def proposal_probabilities(
    model: ProposalModel,
    prior: np.ndarray,
    support: PopulationSupport,
    dual: float,
) -> np.ndarray:
    return np.asarray(
        proposal_probabilities_jax(
            model.parameters,
            model.architecture,
            jnp.asarray(prior, dtype=jnp.float64),
            jnp.asarray(support.pair, dtype=jnp.float64),
            jnp.asarray(support.triplet, dtype=jnp.float64),
            jnp.asarray(support.labels, dtype=jnp.float64),
            jnp.asarray(dual, dtype=jnp.float64),
            defensive_mixture=model.defensive_mixture,
            max_logit_correction=model.max_logit_correction,
        ),
        dtype=np.float64,
    )


def warm_start_dual_jax(
    parameters: MLPParameters,
    architecture: WarmStartArchitecture,
    prior: jax.Array,
    pair_values: jax.Array,
    triplet_values: jax.Array,
    labels: jax.Array,
    target_moment: jax.Array | float,
    *,
    max_abs_dual: float,
) -> jax.Array:
    if architecture.input_dim != 7:
        raise ValueError("the current warm-start feature map requires input_dim=7")
    features = distribution_summary_features_jax(
        prior, pair_values, triplet_values, labels, target_moment
    )
    raw = _mlp(parameters, features[None, :])[0, 0]
    return max_abs_dual * jnp.tanh(raw / max(max_abs_dual, 1e-12))


def warm_start_dual(
    model: WarmStartModel,
    prior: np.ndarray,
    support: PopulationSupport,
    target_moment: float,
) -> float:
    return float(
        warm_start_dual_jax(
            model.parameters,
            model.architecture,
            jnp.asarray(prior, dtype=jnp.float64),
            jnp.asarray(support.pair, dtype=jnp.float64),
            jnp.asarray(support.triplet, dtype=jnp.float64),
            jnp.asarray(support.labels, dtype=jnp.float64),
            jnp.asarray(target_moment, dtype=jnp.float64),
            max_abs_dual=model.max_abs_dual,
        )
    )


def importance_ess_fraction_jax(target: jax.Array, proposal: jax.Array) -> jax.Array:
    """Asymptotic ESS fraction for normalized target/proposal importance weights."""
    target = normalized_probabilities_jax(target)
    proposal = normalized_probabilities_jax(proposal)
    return 1.0 / jnp.sum(jnp.square(target) / jnp.maximum(proposal, 1e-15))


def importance_ess_fraction(target: np.ndarray, proposal: np.ndarray) -> float:
    return float(
        importance_ess_fraction_jax(
            jnp.asarray(target, dtype=jnp.float64),
            jnp.asarray(proposal, dtype=jnp.float64),
        )
    )


def flatten_adaptive_parameters(
    proposal_model: ProposalModel, warm_start_model: WarmStartModel
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix, parameters in (
        ("proposal", proposal_model.parameters),
        ("warm_start", warm_start_model.parameters),
    ):
        for index, (weight, bias) in enumerate(parameters):
            arrays[f"{prefix}_layer_{index}_weight"] = np.asarray(weight, dtype=np.float64)
            arrays[f"{prefix}_layer_{index}_bias"] = np.asarray(bias, dtype=np.float64)
    return arrays
