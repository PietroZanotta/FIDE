"""Continuous flow matching for the finite many-body benchmark.

The learned model transports a standard Gaussian in a continuous embedding of
(spins, latent regime) toward the empirical population.  Generated continuous
states are softly assigned to the exact finite support, which supplies a
strictly positive discrete prior for DiffPOP's exponential tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .energy import conditioned_probabilities, prior_probabilities, sample_indices
from .homometric import PopulationSupport
from .network import PriorParameters

jax.config.update("jax_enable_x64", True)

FlowParameters = tuple[tuple[jax.Array, jax.Array], ...]


@dataclass(frozen=True)
class FlowArchitecture:
    state_dim: int
    hidden_width: int
    hidden_layers: int
    condition_dim: int = 0

    @property
    def input_dim(self) -> int:
        return self.state_dim + 1 + self.condition_dim

    @property
    def parameter_count(self) -> int:
        dimensions = [self.input_dim]
        dimensions.extend([self.hidden_width] * self.hidden_layers)
        dimensions.append(self.state_dim)
        return int(
            sum((left * right) + right for left, right in zip(dimensions[:-1], dimensions[1:]))
        )


@dataclass(frozen=True)
class FlowModel:
    architecture: FlowArchitecture
    parameters: FlowParameters
    label_scale: float = 1.0


@dataclass(frozen=True)
class FlowDistribution:
    probabilities: np.ndarray
    samples: np.ndarray
    hard_indices: np.ndarray
    quantization_rmse: float
    mean_assignment_entropy: float
    sampling_steps: int
    assignment_temperature: float


def support_embeddings(support: PopulationSupport, label_scale: float = 1.0) -> np.ndarray:
    """Embed each exact atom as spins followed by its latent regime coordinate."""
    return np.concatenate(
        [support.spins.astype(np.float64), label_scale * support.labels[:, None]], axis=1
    )


def initialize_flow_model(
    architecture: FlowArchitecture,
    *,
    seed: int,
    label_scale: float = 1.0,
) -> FlowModel:
    key = jax.random.PRNGKey(int(seed))
    dimensions = [architecture.input_dim]
    dimensions.extend([architecture.hidden_width] * architecture.hidden_layers)
    dimensions.append(architecture.state_dim)
    parameters: list[tuple[jax.Array, jax.Array]] = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:]):
        key, weight_key = jax.random.split(key)
        scale = np.sqrt(2.0 / float(fan_in + fan_out))
        weight = scale * jax.random.normal(weight_key, (fan_in, fan_out), dtype=jnp.float64)
        bias = jnp.zeros((fan_out,), dtype=jnp.float64)
        parameters.append((weight, bias))
    return FlowModel(architecture, tuple(parameters), float(label_scale))


def _as_batch_condition(
    condition: jax.Array | float | None,
    batch_size: int,
    condition_dim: int,
    dtype,
) -> jax.Array | None:
    if condition_dim == 0:
        return None
    if condition is None:
        raise ValueError("conditional flow requires a condition")
    values = jnp.asarray(condition, dtype=dtype)
    if values.ndim == 0:
        values = jnp.full((batch_size, condition_dim), values, dtype=dtype)
    elif values.ndim == 1:
        if condition_dim == 1 and values.shape[0] == batch_size:
            values = values[:, None]
        elif values.shape[0] == condition_dim:
            values = jnp.broadcast_to(values[None, :], (batch_size, condition_dim))
        else:
            raise ValueError("condition shape does not match batch or condition dimension")
    elif values.ndim == 2:
        if values.shape != (batch_size, condition_dim):
            raise ValueError("condition matrix has an incompatible shape")
    else:
        raise ValueError("condition must be scalar, vector, or matrix")
    return values


def vector_field(
    parameters: FlowParameters,
    architecture: FlowArchitecture,
    state: jax.Array,
    time: jax.Array | float,
    condition: jax.Array | float | None = None,
) -> jax.Array:
    """Evaluate the neural velocity field v_theta(x, t, c)."""
    values = jnp.asarray(state, dtype=jnp.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != architecture.state_dim:
        raise ValueError("state has an incompatible shape")
    batch_size = values.shape[0]
    times = jnp.asarray(time, dtype=values.dtype)
    if times.ndim == 0:
        times = jnp.full((batch_size, 1), times, dtype=values.dtype)
    elif times.ndim == 1:
        if times.shape[0] == batch_size:
            times = times[:, None]
        elif times.shape[0] == 1:
            times = jnp.broadcast_to(times, (batch_size, 1))
        else:
            raise ValueError("time vector has an incompatible shape")
    elif times.ndim == 2 and times.shape == (batch_size, 1):
        pass
    else:
        raise ValueError("time must be scalar or one value per state")
    cond = _as_batch_condition(
        condition, batch_size, architecture.condition_dim, values.dtype
    )
    pieces = [values, times]
    if cond is not None:
        pieces.append(cond)
    hidden = jnp.concatenate(pieces, axis=-1)
    for layer_index, (weight, bias) in enumerate(parameters):
        hidden = hidden @ weight + bias
        if layer_index + 1 < len(parameters):
            hidden = jax.nn.silu(hidden)
    return hidden[0] if squeeze else hidden


def integrate_flow(
    parameters: FlowParameters,
    architecture: FlowArchitecture,
    base_samples: jax.Array,
    *,
    condition: jax.Array | float | None = None,
    steps: int,
    method: str = "heun",
) -> jax.Array:
    """Integrate the learned ODE from t=0 to t=1 with fixed steps."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    state = jnp.asarray(base_samples, dtype=jnp.float64)
    dt = jnp.asarray(1.0 / steps, dtype=state.dtype)
    for step in range(int(steps)):
        time = jnp.asarray(step / steps, dtype=state.dtype)
        first = vector_field(parameters, architecture, state, time, condition)
        if method == "euler":
            state = state + dt * first
        else:
            proposal = state + dt * first
            second = vector_field(parameters, architecture, proposal, time + dt, condition)
            state = state + 0.5 * dt * (first + second)
    return state


def soft_atom_probabilities_jax(
    samples: jax.Array,
    atom_embeddings: jax.Array,
    assignment_temperature: float,
) -> tuple[jax.Array, jax.Array]:
    if assignment_temperature <= 0:
        raise ValueError("assignment_temperature must be positive")
    differences = samples[:, None, :] - atom_embeddings[None, :, :]
    squared_distance = jnp.sum(jnp.square(differences), axis=-1)
    assignments = jax.nn.softmax(-squared_distance / assignment_temperature, axis=1)
    probabilities = jnp.mean(assignments, axis=0)
    probabilities = jnp.maximum(probabilities, 1e-15)
    probabilities = probabilities / jnp.sum(probabilities)
    return probabilities, assignments


def flow_distribution_from_base(
    model: FlowModel,
    support: PopulationSupport,
    base_samples: np.ndarray,
    *,
    condition: float | np.ndarray | None,
    sampling_steps: int,
    assignment_temperature: float,
    integration_method: str = "heun",
) -> FlowDistribution:
    embeddings = jnp.asarray(support_embeddings(support, model.label_scale))
    generated = integrate_flow(
        model.parameters,
        model.architecture,
        jnp.asarray(base_samples, dtype=jnp.float64),
        condition=condition,
        steps=sampling_steps,
        method=integration_method,
    )
    probabilities, assignments = soft_atom_probabilities_jax(
        generated, embeddings, assignment_temperature
    )
    generated_np = np.asarray(generated, dtype=np.float64)
    assignments_np = np.asarray(assignments, dtype=np.float64)
    distances = np.sum(
        np.square(generated_np[:, None, :] - np.asarray(embeddings)[None, :, :]), axis=-1
    )
    hard_indices = np.argmin(distances, axis=1).astype(np.int64)
    quantization_rmse = float(np.sqrt(np.mean(np.min(distances, axis=1))))
    entropy = -np.sum(
        assignments_np * np.log(np.maximum(assignments_np, 1e-300)), axis=1
    )
    return FlowDistribution(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        samples=generated_np,
        hard_indices=hard_indices,
        quantization_rmse=quantization_rmse,
        mean_assignment_entropy=float(np.mean(entropy)),
        sampling_steps=int(sampling_steps),
        assignment_temperature=float(assignment_temperature),
    )


def sample_flow_distribution(
    model: FlowModel,
    support: PopulationSupport,
    *,
    sample_count: int,
    seed: int,
    condition: float | np.ndarray | None = None,
    sampling_steps: int,
    assignment_temperature: float,
    integration_method: str = "heun",
) -> FlowDistribution:
    rng = np.random.default_rng(int(seed))
    base = rng.normal(size=(int(sample_count), model.architecture.state_dim))
    return flow_distribution_from_base(
        model,
        support,
        base,
        condition=condition,
        sampling_steps=sampling_steps,
        assignment_temperature=assignment_temperature,
        integration_method=integration_method,
    )


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _adam_step(
    parameters,
    gradient,
    first_moment,
    second_moment,
    iteration: int,
    learning_rate: float,
    gradient_clip: float,
):
    norm = _tree_global_norm(gradient)
    scale = jnp.minimum(1.0, gradient_clip / jnp.maximum(norm, 1e-12))
    gradient = jax.tree_util.tree_map(lambda value: value * scale, gradient)
    first_moment = jax.tree_util.tree_map(
        lambda old, value: 0.9 * old + 0.1 * value, first_moment, gradient
    )
    second_moment = jax.tree_util.tree_map(
        lambda old, value: 0.999 * old + 0.001 * jnp.square(value), second_moment, gradient
    )
    first_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - 0.9 ** iteration), first_moment
    )
    second_hat = jax.tree_util.tree_map(
        lambda value: value / (1.0 - 0.999 ** iteration), second_moment
    )
    parameters = jax.tree_util.tree_map(
        lambda param, first, second: param
        - learning_rate * first / (jnp.sqrt(second) + 1e-8),
        parameters,
        first_hat,
        second_hat,
    )
    return parameters, first_moment, second_moment, float(norm)


def fit_flow_matching(
    initial_model: FlowModel,
    target_embeddings: np.ndarray,
    *,
    conditions: np.ndarray | None,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    gradient_clip: float = 10.0,
) -> tuple[FlowModel, list[float], list[float]]:
    """Fit the conditional flow-matching objective on linear Gaussian paths."""
    targets = np.asarray(target_embeddings, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != initial_model.architecture.state_dim:
        raise ValueError("target embeddings have an incompatible shape")
    if initial_model.architecture.condition_dim:
        if conditions is None:
            raise ValueError("conditional training requires conditions")
        condition_values = np.asarray(conditions, dtype=np.float64)
        if condition_values.ndim == 1:
            condition_values = condition_values[:, None]
        if condition_values.shape != (
            targets.shape[0],
            initial_model.architecture.condition_dim,
        ):
            raise ValueError("conditions have an incompatible shape")
    else:
        condition_values = np.empty((targets.shape[0], 0), dtype=np.float64)

    architecture = initial_model.architecture

    def loss_fn(parameters, key, batch_targets, batch_conditions):
        noise_key, time_key = jax.random.split(key)
        base = jax.random.normal(noise_key, batch_targets.shape, dtype=jnp.float64)
        times = jax.random.uniform(
            time_key, (batch_targets.shape[0], 1), minval=0.0, maxval=1.0, dtype=jnp.float64
        )
        interpolated = (1.0 - times) * base + times * batch_targets
        target_velocity = batch_targets - base
        condition_arg = batch_conditions if architecture.condition_dim else None
        predicted = vector_field(
            parameters, architecture, interpolated, times[:, 0], condition_arg
        )
        return jnp.mean(jnp.square(predicted - target_velocity))

    loss_and_gradient = jax.jit(jax.value_and_grad(loss_fn))
    parameters = initial_model.parameters
    first_moment = _tree_zeros_like(parameters)
    second_moment = _tree_zeros_like(parameters)
    rng = np.random.default_rng(int(seed))
    key = jax.random.PRNGKey(int(seed) + 17)
    trace: list[float] = []
    gradient_norms: list[float] = []
    effective_batch = min(int(batch_size), targets.shape[0])
    for iteration in range(1, int(steps) + 1):
        ids = rng.choice(targets.shape[0], size=effective_batch, replace=False)
        batch_targets = jnp.asarray(targets[ids])
        batch_conditions = jnp.asarray(condition_values[ids])
        key, step_key = jax.random.split(key)
        value, gradient = loss_and_gradient(
            parameters, step_key, batch_targets, batch_conditions
        )
        parameters, first_moment, second_moment, norm = _adam_step(
            parameters,
            gradient,
            first_moment,
            second_moment,
            iteration,
            float(learning_rate),
            float(gradient_clip),
        )
        trace.append(float(value))
        gradient_norms.append(norm)
    return (
        FlowModel(architecture, parameters, initial_model.label_scale),
        trace,
        gradient_norms,
    )


def _calibrate_distribution_jax(
    prior: jax.Array,
    pair_values: jax.Array,
    target: jax.Array,
    iterations: int,
) -> jax.Array:
    log_prior = jnp.log(jnp.maximum(prior, 1e-15))
    dual = jnp.asarray(0.0, dtype=prior.dtype)
    for _ in range(int(iterations)):
        probabilities = jax.nn.softmax(log_prior + dual * pair_values)
        mean = jnp.sum(probabilities * pair_values)
        covariance = jnp.sum(probabilities * jnp.square(pair_values - mean))
        step = jnp.clip((mean - target) / (covariance + 1e-8), -3.0, 3.0)
        dual = dual - step
    return dual


def fine_tune_diffpop_flow(
    initial_model: FlowModel,
    support: PopulationSupport,
    task_targets: Sequence[float],
    task_sample_indices: Sequence[np.ndarray],
    *,
    steps: int,
    learning_rate: float,
    flow_particles: int,
    sampling_steps: int,
    assignment_temperature: float,
    dual_iterations: int,
    ess_weight: float,
    anchor_weight: float,
    differentiate_dual: bool,
    seed: int,
    gradient_clip: float = 10.0,
) -> tuple[FlowModel, list[float], list[float]]:
    """Fine-tune a population flow through DiffPOP's exact finite-support solve."""
    if initial_model.architecture.condition_dim != 0:
        raise ValueError("DiffPOP population prior must not be directly conditioned on the moment")
    if len(task_targets) != len(task_sample_indices) or not task_targets:
        raise ValueError("tasks must contain targets and sample indices")
    sample_lengths = {int(np.asarray(ids).size) for ids in task_sample_indices}
    if len(sample_lengths) != 1:
        raise ValueError("all task sample arrays must have equal length")

    rng = np.random.default_rng(int(seed))
    base_samples = rng.normal(
        size=(len(task_targets), int(flow_particles), initial_model.architecture.state_dim)
    )
    embeddings = jnp.asarray(support_embeddings(support, initial_model.label_scale))
    pair_values = jnp.asarray(support.pair, dtype=jnp.float64)
    targets = jnp.asarray(task_targets, dtype=jnp.float64)
    ids = jnp.asarray(np.stack(task_sample_indices), dtype=jnp.int32)
    base = jnp.asarray(base_samples, dtype=jnp.float64)

    def distributions(parameters):
        output = []
        for task_index in range(len(task_targets)):
            generated = integrate_flow(
                parameters,
                initial_model.architecture,
                base[task_index],
                steps=sampling_steps,
                method="heun",
            )
            probabilities, _ = soft_atom_probabilities_jax(
                generated, embeddings, assignment_temperature
            )
            output.append(probabilities)
        return jnp.stack(output)

    anchor_distributions = jax.lax.stop_gradient(distributions(initial_model.parameters))

    def loss_fn(parameters):
        priors = distributions(parameters)
        losses = []
        for task_index in range(len(task_targets)):
            prior = priors[task_index]
            dual = _calibrate_distribution_jax(
                prior, pair_values, targets[task_index], dual_iterations
            )
            if not differentiate_dual:
                dual = jax.lax.stop_gradient(dual)
            log_conditioned = jnp.log(jnp.maximum(prior, 1e-15)) + dual * pair_values
            log_conditioned = log_conditioned - jax.scipy.special.logsumexp(log_conditioned)
            conditioned = jnp.exp(log_conditioned)
            negative_log_score = -jnp.mean(log_conditioned[ids[task_index]])
            ess_fraction = 1.0 / jnp.sum(
                jnp.square(conditioned) / jnp.maximum(prior, 1e-15)
            )
            anchor = anchor_distributions[task_index]
            anchor_kl = jnp.sum(
                anchor
                * (
                    jnp.log(jnp.maximum(anchor, 1e-15))
                    - jnp.log(jnp.maximum(prior, 1e-15))
                )
            )
            losses.append(
                negative_log_score
                - ess_weight * jnp.log(jnp.maximum(ess_fraction, 1e-12))
                + anchor_weight * anchor_kl
            )
        return jnp.mean(jnp.stack(losses))

    loss_and_gradient = jax.jit(jax.value_and_grad(loss_fn))
    parameters = initial_model.parameters
    first_moment = _tree_zeros_like(parameters)
    second_moment = _tree_zeros_like(parameters)
    trace: list[float] = []
    gradient_norms: list[float] = []
    for iteration in range(1, int(steps) + 1):
        value, gradient = loss_and_gradient(parameters)
        parameters, first_moment, second_moment, norm = _adam_step(
            parameters,
            gradient,
            first_moment,
            second_moment,
            iteration,
            float(learning_rate),
            float(gradient_clip),
        )
        trace.append(float(value))
        gradient_norms.append(norm)
    return (
        FlowModel(initial_model.architecture, parameters, initial_model.label_scale),
        trace,
        gradient_norms,
    )


def flatten_flow_parameters(model: FlowModel) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for index, (weight, bias) in enumerate(model.parameters):
        arrays[f"layer_{index}_weight"] = np.asarray(weight, dtype=np.float64)
        arrays[f"layer_{index}_bias"] = np.asarray(bias, dtype=np.float64)
    return arrays


# Compatibility helpers for the exact parametric oracle used by numerical tests.
def sample_prior(
    params: PriorParameters,
    support: PopulationSupport,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_indices(prior_probabilities(params, support), size, rng)


def sample_conditioned_exact(
    params: PriorParameters,
    support: PopulationSupport,
    dual: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_indices(conditioned_probabilities(params, support, dual), size, rng)


def flow_diffpop_objective_and_gradient(
    model: FlowModel,
    support: PopulationSupport,
    *,
    target_moment: float,
    sample_indices_for_score: np.ndarray,
    base_samples: np.ndarray,
    sampling_steps: int,
    assignment_temperature: float,
    dual_iterations: int,
    ess_weight: float = 0.05,
    differentiate_dual: bool = True,
) -> tuple[float, FlowParameters]:
    """Evaluate one differentiable flow→tilt objective and its parameter gradient."""
    embeddings = jnp.asarray(support_embeddings(support, model.label_scale))
    pair_values = jnp.asarray(support.pair, dtype=jnp.float64)
    base = jnp.asarray(base_samples, dtype=jnp.float64)
    ids = jnp.asarray(sample_indices_for_score, dtype=jnp.int32)
    target = jnp.asarray(target_moment, dtype=jnp.float64)

    def objective(parameters):
        generated = integrate_flow(
            parameters,
            model.architecture,
            base,
            steps=sampling_steps,
            method="heun",
        )
        prior, _ = soft_atom_probabilities_jax(
            generated, embeddings, assignment_temperature
        )
        dual = _calibrate_distribution_jax(prior, pair_values, target, dual_iterations)
        if not differentiate_dual:
            dual = jax.lax.stop_gradient(dual)
        log_conditioned = jnp.log(jnp.maximum(prior, 1e-15)) + dual * pair_values
        log_conditioned -= jax.scipy.special.logsumexp(log_conditioned)
        conditioned = jnp.exp(log_conditioned)
        ess_fraction = 1.0 / jnp.sum(
            jnp.square(conditioned) / jnp.maximum(prior, 1e-15)
        )
        return -jnp.mean(log_conditioned[ids]) - ess_weight * jnp.log(
            jnp.maximum(ess_fraction, 1e-12)
        )

    value, gradient = jax.value_and_grad(objective)(model.parameters)
    return float(value), gradient


def flow_gradient_directional_check(
    model: FlowModel,
    support: PopulationSupport,
    *,
    target_moment: float,
    sample_indices_for_score: np.ndarray,
    base_samples: np.ndarray,
    sampling_steps: int,
    assignment_temperature: float,
    dual_iterations: int,
    epsilon: float = 1e-5,
    seed: int = 0,
) -> dict[str, float]:
    """Compare the full composed gradient with a central directional difference."""
    value, gradient = flow_diffpop_objective_and_gradient(
        model,
        support,
        target_moment=target_moment,
        sample_indices_for_score=sample_indices_for_score,
        base_samples=base_samples,
        sampling_steps=sampling_steps,
        assignment_temperature=assignment_temperature,
        dual_iterations=dual_iterations,
        differentiate_dual=True,
    )
    rng = np.random.default_rng(int(seed))
    direction = jax.tree_util.tree_map(
        lambda parameter: jnp.asarray(rng.normal(size=parameter.shape), dtype=parameter.dtype),
        model.parameters,
    )
    direction_norm = float(_tree_global_norm(direction))
    direction = jax.tree_util.tree_map(lambda x: x / direction_norm, direction)
    autodiff = float(
        sum(
            jnp.sum(left * right)
            for left, right in zip(
                jax.tree_util.tree_leaves(gradient),
                jax.tree_util.tree_leaves(direction),
            )
        )
    )

    def shifted(sign: float) -> FlowModel:
        parameters = jax.tree_util.tree_map(
            lambda parameter, delta: parameter + sign * epsilon * delta,
            model.parameters,
            direction,
        )
        return FlowModel(model.architecture, parameters, model.label_scale)

    plus, _ = flow_diffpop_objective_and_gradient(
        shifted(1.0),
        support,
        target_moment=target_moment,
        sample_indices_for_score=sample_indices_for_score,
        base_samples=base_samples,
        sampling_steps=sampling_steps,
        assignment_temperature=assignment_temperature,
        dual_iterations=dual_iterations,
        differentiate_dual=True,
    )
    minus, _ = flow_diffpop_objective_and_gradient(
        shifted(-1.0),
        support,
        target_moment=target_moment,
        sample_indices_for_score=sample_indices_for_score,
        base_samples=base_samples,
        sampling_steps=sampling_steps,
        assignment_temperature=assignment_temperature,
        dual_iterations=dual_iterations,
        differentiate_dual=True,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative_error = abs(autodiff - finite_difference) / max(
        abs(finite_difference), abs(autodiff), 1e-12
    )
    return {
        "objective": value,
        "autodiff_directional_derivative": autodiff,
        "finite_difference_directional_derivative": finite_difference,
        "relative_error": relative_error,
    }
