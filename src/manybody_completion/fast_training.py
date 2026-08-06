"""High-throughput JAX training utilities.

This module is intentionally model-agnostic.  It fuses loss evaluation,
reverse-mode differentiation, gradient clipping, and Adam into one compiled
function, then executes several optimizer updates per host dispatch with
``jax.lax.scan``.

The loss callback must have the signature::

    loss_fn(parameters, batch, key) -> (scalar_loss, metrics_pytree)

Every leaf of ``batches`` must have the optimizer-step axis in position zero.
The same is true for ``keys``.  Metrics remain on device until a chunk finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

PyTree: TypeAlias = Any
LossFunction: TypeAlias = Callable[[PyTree, PyTree, Array], tuple[Array, PyTree]]


@dataclass(frozen=True)
class FastAdamOptions:
    """Settings for a compiled chunked Adam training loop."""

    learning_rate: float
    num_steps: int
    chunk_size: int = 32
    gradient_clip_norm: float = 5.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


class AdamState(NamedTuple):
    step: Array
    first_moment: PyTree
    second_moment: PyTree


class FastTrainingResult(NamedTuple):
    parameters: PyTree
    optimizer_state: AdamState
    history: PyTree


def _zeros_like_tree(tree: PyTree) -> PyTree:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def initialize_adam(parameters: PyTree) -> AdamState:
    """Initialize device-resident Adam state."""
    return AdamState(
        step=jnp.asarray(0, dtype=jnp.int32),
        first_moment=_zeros_like_tree(parameters),
        second_moment=_zeros_like_tree(parameters),
    )


def tree_global_norm(tree: PyTree) -> Array:
    """Compute one stable global L2 norm over a pytree."""
    squared = sum(
        jnp.sum(jnp.asarray(leaf, dtype=jnp.float32) ** 2)
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return jnp.sqrt(squared + jnp.asarray(1e-24, dtype=jnp.float32))


def clip_by_global_norm(tree: PyTree, maximum_norm: float) -> tuple[PyTree, Array]:
    """Clip a gradient pytree and return its unclipped norm."""
    norm = tree_global_norm(tree)
    scale = jnp.minimum(1.0, jnp.asarray(maximum_norm, norm.dtype) / norm)
    return jax.tree_util.tree_map(lambda value: value * scale, tree), norm


def adam_update(
    parameters: PyTree,
    gradients: PyTree,
    state: AdamState,
    options: FastAdamOptions,
) -> tuple[PyTree, AdamState]:
    """Pure JAX Adam update suitable for use inside ``lax.scan``."""
    step = state.step + jnp.asarray(1, dtype=state.step.dtype)
    first = jax.tree_util.tree_map(
        lambda moment, gradient: options.beta1 * moment
        + (1.0 - options.beta1) * gradient,
        state.first_moment,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda moment, gradient: options.beta2 * moment
        + (1.0 - options.beta2) * gradient * gradient,
        state.second_moment,
        gradients,
    )
    bias_one = 1.0 - options.beta1**step.astype(jnp.float32)
    bias_two = 1.0 - options.beta2**step.astype(jnp.float32)
    updated = jax.tree_util.tree_map(
        lambda parameter, one, two: parameter
        - options.learning_rate
        * (one / bias_one)
        / (jnp.sqrt(two / bias_two) + options.epsilon),
        parameters,
        first,
        second,
    )
    return updated, AdamState(step=step, first_moment=first, second_moment=second)


def _leading_size(tree: PyTree) -> int:
    sizes = {int(leaf.shape[0]) for leaf in jax.tree_util.tree_leaves(tree)}
    if len(sizes) != 1:
        raise ValueError(f"all scheduled leaves need the same leading size; got {sizes}")
    return sizes.pop()


def _slice_tree(tree: PyTree, start: int, size: int) -> PyTree:
    return jax.tree_util.tree_map(lambda value: value[start : start + size], tree)


def make_chunk_trainer(
    loss_fn: LossFunction,
    options: FastAdamOptions,
) -> Callable[[PyTree, AdamState, PyTree, Array], tuple[PyTree, AdamState, PyTree]]:
    """Build one compiled multi-update function.

    Construct this function once, outside the Python training loop.  Its input
    chunk shapes must remain fixed to avoid recompilation.
    """
    options.validate()

    def one_step(
        carry: tuple[PyTree, AdamState],
        inputs: tuple[PyTree, Array],
    ) -> tuple[tuple[PyTree, AdamState], PyTree]:
        parameters, optimizer_state = carry
        batch, key = inputs
        (loss, metrics), gradients = jax.value_and_grad(
            loss_fn,
            has_aux=True,
        )(parameters, batch, key)
        gradients, gradient_norm = clip_by_global_norm(
            gradients,
            options.gradient_clip_norm,
        )
        parameters, optimizer_state = adam_update(
            parameters,
            gradients,
            optimizer_state,
            options,
        )
        output_metrics = {
            **metrics,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "optimizer_step": optimizer_state.step,
        }
        return (parameters, optimizer_state), output_metrics

    @jax.jit(donate_argnums=(0, 1))
    def train_chunk(
        parameters: PyTree,
        optimizer_state: AdamState,
        batches: PyTree,
        keys: Array,
    ) -> tuple[PyTree, AdamState, PyTree]:
        (parameters, optimizer_state), metrics = jax.lax.scan(
            one_step,
            (parameters, optimizer_state),
            (batches, keys),
        )
        return parameters, optimizer_state, metrics

    return train_chunk


def train_in_chunks(
    parameters: PyTree,
    batches: PyTree,
    keys: Array,
    loss_fn: LossFunction,
    options: FastAdamOptions,
    *,
    optimizer_state: AdamState | None = None,
    synchronize_each_chunk: bool = False,
) -> FastTrainingResult:
    """Run a device-resident schedule with one host dispatch per chunk.

    ``num_steps`` must be exactly represented in ``batches`` and ``keys``.
    Choose a chunk size that divides the schedule.  For typical training jobs,
    16--64 updates per chunk gives low host overhead without creating an
    unwieldy compilation.
    """
    options.validate()
    batch_steps = _leading_size(batches)
    if keys.ndim != 2 or keys.shape[-1] != 2:
        raise ValueError("keys must have shape (num_steps, 2)")
    if batch_steps != options.num_steps or keys.shape[0] != options.num_steps:
        raise ValueError(
            "scheduled batches/keys must match options.num_steps; "
            f"got {batch_steps}, {keys.shape[0]}, and {options.num_steps}"
        )
    if options.num_steps % options.chunk_size:
        raise ValueError(
            "num_steps must be divisible by chunk_size for a fixed-shape fast path"
        )

    parameters = jax.device_put(parameters)
    batches = jax.device_put(batches)
    keys = jax.device_put(keys)
    state = initialize_adam(parameters) if optimizer_state is None else optimizer_state
    train_chunk = make_chunk_trainer(loss_fn, options)
    history_chunks: list[PyTree] = []

    for start in range(0, options.num_steps, options.chunk_size):
        chunk_batches = _slice_tree(batches, start, options.chunk_size)
        chunk_keys = keys[start : start + options.chunk_size]
        parameters, state, metrics = train_chunk(
            parameters,
            state,
            chunk_batches,
            chunk_keys,
        )
        history_chunks.append(metrics)
        if synchronize_each_chunk:
            jax.block_until_ready(parameters)

    history = jax.tree_util.tree_map(
        lambda *values: jnp.concatenate(values, axis=0),
        *history_chunks,
    )
    return FastTrainingResult(parameters, state, history)


def stack_step_batches(batches: list[PyTree]) -> PyTree:
    """Stack a list of equal-shaped minibatch pytrees once on the host."""
    if not batches:
        raise ValueError("batches must be nonempty")
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *batches)
