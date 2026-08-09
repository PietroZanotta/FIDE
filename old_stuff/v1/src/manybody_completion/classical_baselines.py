"""Accelerated observation-only baselines for the homometric benchmark.

The public API is unchanged. Reverse Monte Carlo (RMC) and iterative Boltzmann
inversion (IBI) execute their dependent Markov updates inside JAX ``lax.scan``
loops while vectorizing independent ensembles. Metropolis proposals use exact
incremental energy/statistic updates: moving one particle recomputes only its
``N - 1`` affected pairs, rather than rebuilding every pair in every replica.
This removes Python work from the hot path, keeps state on one device, and reuses
compiled executables across training seeds with identical shapes and options.

RMC receives only the configured smooth pair moments. IBI receives a
Gaussian-smoothed radial pair histogram estimated from microscopic training
configurations; this richer pair-level information budget remains explicit in
the returned diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class RMCOptions:
    """Metropolis simulated-annealing settings for reverse Monte Carlo."""

    num_steps: int = 400
    proposal_scale: float = 0.08
    initial_temperature: float = 0.05
    final_temperature: float = 0.002
    physical_weight: float = 0.0
    physical_r0: float = 0.22
    physical_kappa: float = 30.0
    trace_stride: int = 20

    def validate(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.proposal_scale <= 0:
            raise ValueError("proposal_scale must be positive")
        if self.initial_temperature <= 0 or self.final_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.physical_weight < 0:
            raise ValueError("physical_weight must be nonnegative")
        if self.trace_stride < 1:
            raise ValueError("trace_stride must be positive")


@dataclass(frozen=True)
class IBIOptions:
    """Histogram IBI and Metropolis sampling settings."""

    num_iterations: int = 5
    metropolis_steps_per_iteration: int = 240
    proposal_scale: float = 0.07
    inverse_temperature: float = 1.0
    update_rate: float = 0.35
    num_bins: int = 24
    radial_max: float = 0.48
    kernel_width: float = 0.025
    density_floor: float = 1e-4
    potential_clip: float = 12.0
    initial_repulsion: float = 1.0
    convergence_window: int = 2
    convergence_relative_tolerance: float = 0.10
    density_rmse_tolerance: float = 0.01

    def validate(self) -> None:
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.metropolis_steps_per_iteration < 1:
            raise ValueError("metropolis_steps_per_iteration must be positive")
        if self.proposal_scale <= 0 or self.inverse_temperature <= 0:
            raise ValueError(
                "proposal_scale and inverse_temperature must be positive"
            )
        if self.update_rate <= 0:
            raise ValueError("update_rate must be positive")
        if self.num_bins < 4 or self.radial_max <= 0 or self.kernel_width <= 0:
            raise ValueError("invalid radial discretization")
        if self.density_floor <= 0 or self.potential_clip <= 0:
            raise ValueError(
                "density_floor and potential_clip must be positive"
            )
        if self.convergence_window < 1:
            raise ValueError("convergence_window must be positive")
        if self.convergence_window > self.num_iterations:
            raise ValueError("convergence_window cannot exceed num_iterations")
        if self.convergence_relative_tolerance <= 0:
            raise ValueError("convergence_relative_tolerance must be positive")
        if self.density_rmse_tolerance <= 0:
            raise ValueError("density_rmse_tolerance must be positive")


def _working_dtype(*values: Any) -> jnp.dtype:
    """Use the pipeline dtype, while avoiding accidental integer arithmetic."""
    dtype = jnp.result_type(*[jnp.asarray(value).dtype for value in values])
    if not jnp.issubdtype(dtype, jnp.floating):
        return jnp.float32
    return dtype


def _chord_distances_jax(coordinates: Array, box: Array) -> Array:
    delta = coordinates[..., :, None, :] - coordinates[..., None, :, :]
    displacement = (box / jnp.pi) * jnp.sin(jnp.pi * delta / box)
    return jnp.sqrt(jnp.sum(displacement * displacement, axis=-1) + 1e-24)


def _unordered_distances_jax(coordinates: Array, box: Array) -> Array:
    distances = _chord_distances_jax(coordinates, box)
    num_particles = coordinates.shape[-2]
    row, column = jnp.triu_indices(num_particles, k=1)
    return distances[..., row, column]


def _pair_moments_from_distances(
    distances: Array,
    centers: Array,
    widths: Array,
    dtype: jnp.dtype,
) -> Array:
    num_particles = distances.shape[-1]
    mask = 1.0 - jnp.eye(num_particles, dtype=dtype)
    standardized = (distances[..., None] - centers) / widths
    values = jnp.exp(-0.5 * standardized * standardized) * mask[..., None]
    per_replica = jnp.sum(values, axis=(-3, -2)) / (
        num_particles * (num_particles - 1)
    )
    return jnp.mean(per_replica, axis=1)


def _ensemble_pair_moments_batch(
    coordinates: Array,
    box: Array,
    centers: Array,
    widths: Array,
) -> Array:
    """Pair moments for a leading batch of ensembles ``(E, M, N, 2)``."""
    distances = _chord_distances_jax(coordinates, box)
    return _pair_moments_from_distances(
        distances,
        centers,
        widths,
        coordinates.dtype,
    )


def _physical_energy_from_distances(
    distances: Array,
    r0: Array,
    kappa: Array,
    dtype: jnp.dtype,
) -> Array:
    num_particles = distances.shape[-1]
    mask = 1.0 - jnp.eye(num_particles, dtype=dtype)
    penalty = jax.nn.softplus(kappa * (r0 - distances)) ** 2 * mask
    per_replica = jnp.sum(penalty, axis=(-2, -1)) / (
        num_particles * (num_particles - 1)
    )
    return jnp.mean(per_replica, axis=1)


def _rmc_objective_batch(
    coordinates: Array,
    target_moments: Array,
    moment_scales: Array,
    box: Array,
    centers: Array,
    widths: Array,
    physical_weight: Array,
    physical_r0: Array,
    physical_kappa: Array,
) -> Array:
    distances = _chord_distances_jax(coordinates, box)
    moments = _pair_moments_from_distances(
        distances,
        centers,
        widths,
        coordinates.dtype,
    )
    residual = (moments - target_moments) / jnp.maximum(moment_scales, 1e-12)
    pair_objective = jnp.sum(residual * residual, axis=-1)

    def add_physical(_: None) -> Array:
        physical = _physical_energy_from_distances(
            distances,
            physical_r0,
            physical_kappa,
            coordinates.dtype,
        )
        return pair_objective + physical_weight * physical

    return jax.lax.cond(
        physical_weight > 0.0,
        add_physical,
        lambda _: pair_objective,
        operand=None,
    )


def _pair_basis_row(distances: Array, centers: Array, widths: Array) -> Array:
    """RBF values for one moved particle against all particles."""
    standardized = (distances[..., None] - centers) / widths
    return jnp.exp(-0.5 * standardized * standardized)


def _repulsive_row(
    distances: Array,
    r0: Array,
    kappa: Array,
) -> Array:
    """Repulsive penalties for one moved particle against all particles."""
    return jax.nn.softplus(kappa * (r0 - distances)) ** 2


@partial(jax.jit, static_argnames=("num_steps",))
def _run_rmc_device(
    coordinates: Array,
    target_moments: Array,
    moment_scales: Array,
    box: Array,
    centers: Array,
    widths: Array,
    seed_key: Array,
    proposal_scale: Array,
    initial_temperature: Array,
    final_temperature: Array,
    physical_weight: Array,
    physical_r0: Array,
    physical_kappa: Array,
    *,
    num_steps: int,
) -> tuple[Array, Array, Array, Array]:
    """Run vectorized RMC with exact local statistic updates."""
    num_ensembles, num_replicas, num_particles = coordinates.shape[:3]
    ensemble_indices = jnp.arange(num_ensembles, dtype=jnp.int32)

    distances = _chord_distances_jax(coordinates, box)
    ensemble_moments = _pair_moments_from_distances(
        distances,
        centers,
        widths,
        coordinates.dtype,
    )
    residual = (ensemble_moments - target_moments) / jnp.maximum(
        moment_scales,
        1e-12,
    )
    pair_objective = jnp.sum(residual * residual, axis=-1)
    physical_energy = _physical_energy_from_distances(
        distances,
        physical_r0,
        physical_kappa,
        coordinates.dtype,
    )
    current_objective = pair_objective + physical_weight * physical_energy
    best_coordinates = coordinates
    best_objective = current_objective
    accepted = jnp.zeros((num_ensembles,), dtype=jnp.int32)

    fractions = jnp.arange(num_steps, dtype=coordinates.dtype) / jnp.maximum(
        num_steps - 1,
        1,
    )
    temperatures = initial_temperature * (
        final_temperature / initial_temperature
    ) ** fractions
    step_keys = jax.random.split(seed_key, num_steps)
    normalizer = jnp.asarray(
        num_particles * (num_particles - 1),
        dtype=coordinates.dtype,
    )
    replica_normalizer = jnp.asarray(num_replicas, dtype=coordinates.dtype)

    def metropolis_step(carry, inputs):
        (
            current,
            current_moments,
            current_physical,
            current_value,
            best,
            best_value,
            accepted_count,
        ) = carry
        step_key, temperature = inputs
        replica_key, particle_key, delta_key, accept_key = jax.random.split(
            step_key,
            4,
        )
        replica_indices = jax.random.randint(
            replica_key,
            (num_ensembles,),
            0,
            num_replicas,
            dtype=jnp.int32,
        )
        particle_indices = jax.random.randint(
            particle_key,
            (num_ensembles,),
            0,
            num_particles,
            dtype=jnp.int32,
        )
        selected_replicas = current[ensemble_indices, replica_indices]
        old_position = selected_replicas[
            ensemble_indices,
            particle_indices,
        ]
        new_position = jnp.mod(
            old_position
            + proposal_scale
            * jax.random.normal(
                delta_key,
                (num_ensembles, 2),
                dtype=current.dtype,
            ),
            box,
        )

        old_delta = old_position[:, None, :] - selected_replicas
        new_delta = new_position[:, None, :] - selected_replicas
        old_disp = (box / jnp.pi) * jnp.sin(jnp.pi * old_delta / box)
        new_disp = (box / jnp.pi) * jnp.sin(jnp.pi * new_delta / box)
        old_distances = jnp.sqrt(jnp.sum(old_disp * old_disp, axis=-1) + 1e-24)
        new_distances = jnp.sqrt(jnp.sum(new_disp * new_disp, axis=-1) + 1e-24)
        mask = 1.0 - jax.nn.one_hot(
            particle_indices,
            num_particles,
            dtype=current.dtype,
        )

        old_basis = _pair_basis_row(old_distances, centers, widths)
        new_basis = _pair_basis_row(new_distances, centers, widths)
        replica_moment_delta = (
            2.0
            * jnp.sum((new_basis - old_basis) * mask[..., None], axis=1)
            / normalizer
        )
        moment_delta = replica_moment_delta / replica_normalizer
        proposed_moments = current_moments + moment_delta
        proposed_residual = (proposed_moments - target_moments) / jnp.maximum(
            moment_scales,
            1e-12,
        )
        proposed_pair_objective = jnp.sum(
            proposed_residual * proposed_residual,
            axis=-1,
        )

        old_penalty = _repulsive_row(
            old_distances,
            physical_r0,
            physical_kappa,
        )
        new_penalty = _repulsive_row(
            new_distances,
            physical_r0,
            physical_kappa,
        )
        physical_delta = (
            2.0
            * jnp.sum((new_penalty - old_penalty) * mask, axis=1)
            / normalizer
            / replica_normalizer
        )
        proposed_physical = current_physical + physical_delta
        proposed_value = (
            proposed_pair_objective + physical_weight * proposed_physical
        )

        objective_delta = proposed_value - current_value
        log_uniform = jnp.log(
            jax.random.uniform(
                accept_key,
                (num_ensembles,),
                minval=jnp.finfo(current.dtype).tiny,
                maxval=1.0,
                dtype=current.dtype,
            )
        )
        accept = (objective_delta <= 0.0) | (
            log_uniform < -objective_delta / temperature
        )
        accepted_count = accepted_count + accept.astype(jnp.int32)
        updated_positions = jnp.where(accept[:, None], new_position, old_position)
        current = current.at[
            ensemble_indices,
            replica_indices,
            particle_indices,
        ].set(updated_positions)
        current_moments = jnp.where(
            accept[:, None],
            proposed_moments,
            current_moments,
        )
        current_physical = jnp.where(
            accept,
            proposed_physical,
            current_physical,
        )
        current_value = jnp.where(accept, proposed_value, current_value)

        improved = accept & (proposed_value < best_value)
        best = jnp.where(improved[:, None, None, None], current, best)
        best_value = jnp.where(improved, proposed_value, best_value)
        return (
            current,
            current_moments,
            current_physical,
            current_value,
            best,
            best_value,
            accepted_count,
        ), jnp.mean(best_value)

    initial_mean = jnp.mean(best_objective)
    (
        _,
        _,
        _,
        _,
        best_coordinates,
        best_objective,
        accepted,
    ), trace = jax.lax.scan(
        metropolis_step,
        (
            coordinates,
            ensemble_moments,
            physical_energy,
            current_objective,
            best_coordinates,
            best_objective,
            accepted,
        ),
        (step_keys, temperatures),
    )
    trace = jnp.concatenate((initial_mean[None], trace), axis=0)
    return best_coordinates, best_objective, accepted, trace


def _radial_density_jax(
    coordinates: Array,
    box: Array,
    bin_centers: Array,
    kernel_width: Array,
) -> Array:
    distances = _unordered_distances_jax(coordinates, box).reshape(-1)
    standardized = (distances[:, None] - bin_centers[None, :]) / kernel_width
    density = jnp.mean(jnp.exp(-0.5 * standardized * standardized), axis=0)
    return density / jnp.maximum(jnp.sum(density), 1e-15)


def _ibi_update_potential_jax(
    potential: Array,
    current_density: Array,
    target_density: Array,
    update_rate: Array,
    density_floor: Array,
    potential_clip: Array,
) -> Array:
    ratio = (current_density + density_floor) / (
        target_density + density_floor
    )
    updated = potential + update_rate * jnp.log(ratio)
    padded = jnp.pad(updated, (1, 1), mode="edge")
    updated = (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )
    updated = updated - jnp.min(updated)
    return jnp.clip(updated, 0.0, potential_clip)


def _pair_potential_energy_batch(
    coordinates: Array,
    box: Array,
    bin_centers: Array,
    potential: Array,
) -> Array:
    distances = _unordered_distances_jax(coordinates, box)
    values = jnp.interp(
        distances,
        bin_centers,
        potential,
        left=potential[0],
        right=potential[-1],
    )
    # Replicas are independent samples from the same canonical distribution.
    # Their joint energy is therefore the *sum* of the replica energies.  An
    # earlier implementation averaged here and divided a local Metropolis
    # energy difference by ``num_replicas`` below.  That silently changed the
    # effective inverse temperature from beta to beta / num_replicas.
    return jnp.sum(jnp.sum(values, axis=-1), axis=1)


@partial(
    jax.jit,
    static_argnames=("num_iterations", "metropolis_steps_per_iteration"),
)
def _run_ibi_device(
    coordinates: Array,
    reference_coordinates: Array,
    box: Array,
    bin_centers: Array,
    seed_key: Array,
    proposal_scale: Array,
    inverse_temperature: Array,
    update_rate: Array,
    kernel_width: Array,
    density_floor: Array,
    potential_clip: Array,
    initial_repulsion: Array,
    radial_max: Array,
    *,
    num_iterations: int,
    metropolis_steps_per_iteration: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Fit and sample IBI using exact local pair-energy updates."""
    num_ensembles, num_replicas, num_particles = coordinates.shape[:3]
    ensemble_indices = jnp.arange(num_ensembles, dtype=jnp.int32)
    target_density = _radial_density_jax(
        reference_coordinates,
        box,
        bin_centers,
        kernel_width,
    )
    potential = initial_repulsion * jnp.exp(
        -0.5 * (bin_centers / jnp.maximum(0.15 * radial_max, 1e-3)) ** 2
    )
    iteration_keys = jax.random.split(seed_key, num_iterations)

    def ibi_iteration(carry, iteration_key):
        current, current_potential = carry
        energies = _pair_potential_energy_batch(
            current,
            box,
            bin_centers,
            current_potential,
        )
        accepted = jnp.asarray(0, dtype=jnp.int32)
        step_keys = jax.random.split(
            iteration_key,
            metropolis_steps_per_iteration,
        )

        def metropolis_step(inner_carry, step_key):
            inner_coordinates, inner_energies, accepted_count = inner_carry
            replica_key, particle_key, delta_key, accept_key = jax.random.split(
                step_key,
                4,
            )
            replica_indices = jax.random.randint(
                replica_key,
                (num_ensembles,),
                0,
                num_replicas,
                dtype=jnp.int32,
            )
            particle_indices = jax.random.randint(
                particle_key,
                (num_ensembles,),
                0,
                num_particles,
                dtype=jnp.int32,
            )
            selected_replicas = inner_coordinates[
                ensemble_indices,
                replica_indices,
            ]
            old_position = selected_replicas[
                ensemble_indices,
                particle_indices,
            ]
            new_position = jnp.mod(
                old_position
                + proposal_scale
                * jax.random.normal(
                    delta_key,
                    (num_ensembles, 2),
                    dtype=inner_coordinates.dtype,
                ),
                box,
            )
            old_delta = old_position[:, None, :] - selected_replicas
            new_delta = new_position[:, None, :] - selected_replicas
            old_disp = (box / jnp.pi) * jnp.sin(jnp.pi * old_delta / box)
            new_disp = (box / jnp.pi) * jnp.sin(jnp.pi * new_delta / box)
            old_distances = jnp.sqrt(
                jnp.sum(old_disp * old_disp, axis=-1) + 1e-24
            )
            new_distances = jnp.sqrt(
                jnp.sum(new_disp * new_disp, axis=-1) + 1e-24
            )
            mask = 1.0 - jax.nn.one_hot(
                particle_indices,
                num_particles,
                dtype=inner_coordinates.dtype,
            )
            old_values = jnp.interp(
                old_distances,
                bin_centers,
                current_potential,
                left=current_potential[0],
                right=current_potential[-1],
            )
            new_values = jnp.interp(
                new_distances,
                bin_centers,
                current_potential,
                left=current_potential[0],
                right=current_potential[-1],
            )
            # Moving one particle changes exactly the N - 1 pairs in its
            # selected replica.  Do not normalize this local canonical-energy
            # difference by the number of independent replicas.
            energy_delta = jnp.sum(
                (new_values - old_values) * mask,
                axis=1,
            )
            proposed_energies = inner_energies + energy_delta
            log_uniform = jnp.log(
                jax.random.uniform(
                    accept_key,
                    (num_ensembles,),
                    minval=jnp.finfo(inner_coordinates.dtype).tiny,
                    maxval=1.0,
                    dtype=inner_coordinates.dtype,
                )
            )
            accept = (energy_delta <= 0.0) | (
                log_uniform < -inverse_temperature * energy_delta
            )
            accepted_count = accepted_count + jnp.sum(
                accept.astype(jnp.int32),
                dtype=jnp.int32,
            )
            updated_positions = jnp.where(
                accept[:, None],
                new_position,
                old_position,
            )
            inner_coordinates = inner_coordinates.at[
                ensemble_indices,
                replica_indices,
                particle_indices,
            ].set(updated_positions)
            inner_energies = jnp.where(
                accept,
                proposed_energies,
                inner_energies,
            )
            return (
                inner_coordinates,
                inner_energies,
                accepted_count,
            ), None

        (current, _, accepted), _ = jax.lax.scan(
            metropolis_step,
            (current, energies, accepted),
            step_keys,
        )
        current_density = _radial_density_jax(
            current,
            box,
            bin_centers,
            kernel_width,
        )
        density_rmse = jnp.sqrt(
            jnp.mean((current_density - target_density) ** 2)
        )
        acceptance_rate = accepted.astype(current.dtype) / (
            metropolis_steps_per_iteration * num_ensembles
        )
        updated_potential = _ibi_update_potential_jax(
            current_potential,
            current_density,
            target_density,
            update_rate,
            density_floor,
            potential_clip,
        )
        return (current, updated_potential), (
            density_rmse,
            acceptance_rate,
        )

    (coordinates, potential), history = jax.lax.scan(
        ibi_iteration,
        (coordinates, potential),
        iteration_keys,
    )
    final_density = _radial_density_jax(
        coordinates,
        box,
        bin_centers,
        kernel_width,
    )
    density_rmse, acceptance_rate = history
    return (
        coordinates,
        target_density,
        final_density,
        potential,
        density_rmse,
        acceptance_rate,
    )


def radial_density(
    coordinates: np.ndarray,
    box: np.ndarray,
    bin_centers: np.ndarray,
    kernel_width: float,
) -> np.ndarray:
    """Gaussian-smoothed normalized density of unordered chord distances."""
    dtype = _working_dtype(coordinates, box, bin_centers)
    result = _radial_density_jax(
        jnp.asarray(coordinates, dtype=dtype),
        jnp.asarray(box, dtype=dtype),
        jnp.asarray(bin_centers, dtype=dtype),
        jnp.asarray(kernel_width, dtype=dtype),
    )
    return np.asarray(jax.device_get(result))


def ibi_update_potential(
    potential: np.ndarray,
    current_density: np.ndarray,
    target_density: np.ndarray,
    options: IBIOptions,
) -> np.ndarray:
    """One damped IBI update, with smoothing and a fixed additive gauge."""
    options.validate()
    dtype = _working_dtype(potential, current_density, target_density)
    result = _ibi_update_potential_jax(
        jnp.asarray(potential, dtype=dtype),
        jnp.asarray(current_density, dtype=dtype),
        jnp.asarray(target_density, dtype=dtype),
        jnp.asarray(options.update_rate, dtype=dtype),
        jnp.asarray(options.density_floor, dtype=dtype),
        jnp.asarray(options.potential_clip, dtype=dtype),
    )
    return np.asarray(jax.device_get(result))


def run_reverse_monte_carlo(
    initial_coordinates: np.ndarray,
    target_moments: np.ndarray,
    moment_scales: np.ndarray,
    box: np.ndarray,
    basis_centers: np.ndarray,
    basis_widths: np.ndarray,
    options: RMCOptions,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run independent, vectorized RMC chains for a batch of ensembles."""
    options.validate()
    coordinates_np = np.asarray(initial_coordinates)
    if coordinates_np.ndim != 4 or coordinates_np.shape[-1] != 2:
        raise ValueError("initial_coordinates must have shape (E, M, N, 2)")
    dtype = _working_dtype(
        coordinates_np,
        box,
        basis_centers,
        basis_widths,
    )
    coordinates = jax.device_put(jnp.asarray(coordinates_np, dtype=dtype))
    box_array = jax.device_put(jnp.asarray(box, dtype=dtype))
    centers = jax.device_put(jnp.asarray(basis_centers, dtype=dtype))
    widths = jax.device_put(jnp.asarray(basis_widths, dtype=dtype))
    scales = jax.device_put(jnp.asarray(moment_scales, dtype=dtype))
    if box_array.shape != (2,):
        raise ValueError("box must have shape (2,)")
    if centers.ndim != 1 or widths.shape != centers.shape:
        raise ValueError("basis centers and widths must be matching vectors")
    if bool(np.any(np.asarray(widths) <= 0)):
        raise ValueError("basis widths must be positive")
    if scales.shape != centers.shape:
        raise ValueError("moment_scales must match the basis dimension")

    target = jnp.asarray(target_moments, dtype=dtype)
    if target.shape == (centers.size,):
        target = jnp.broadcast_to(
            target,
            (coordinates.shape[0], centers.size),
        )
    if target.shape != (coordinates.shape[0], centers.size):
        raise ValueError("target_moments has an incompatible shape")
    target = jax.device_put(target)

    best, best_objective, accepted, full_trace = _run_rmc_device(
        coordinates,
        target,
        scales,
        box_array,
        centers,
        widths,
        jax.random.PRNGKey(seed),
        jnp.asarray(options.proposal_scale, dtype=dtype),
        jnp.asarray(options.initial_temperature, dtype=dtype),
        jnp.asarray(options.final_temperature, dtype=dtype),
        jnp.asarray(options.physical_weight, dtype=dtype),
        jnp.asarray(options.physical_r0, dtype=dtype),
        jnp.asarray(options.physical_kappa, dtype=dtype),
        num_steps=options.num_steps,
    )
    best, best_objective, accepted, full_trace = jax.device_get(
        (best, best_objective, accepted, full_trace)
    )

    trace_steps = np.arange(
        0,
        options.num_steps + 1,
        options.trace_stride,
        dtype=np.int32,
    )
    if trace_steps[-1] != options.num_steps:
        trace_steps = np.concatenate(
            (trace_steps, np.asarray([options.num_steps], dtype=np.int32))
        )
    trace_objective = np.asarray(full_trace)[trace_steps]
    return np.asarray(best), {
        "objective_before": float(full_trace[0]),
        "objective_after": float(np.mean(best_objective)),
        "acceptance_rate": float(
            np.mean(np.asarray(accepted, dtype=np.float64) / options.num_steps)
        ),
        "per_ensemble_objective": np.asarray(best_objective),
        "trace_steps": trace_steps,
        "trace_objective": trace_objective,
        "execution_backend": jax.default_backend(),
        "execution_device": str(jax.devices()[0]),
        "execution_strategy": (
            "vectorized ensembles + compiled lax.scan + incremental pair updates"
        ),
        "information_budget": "smooth pair-moment condition only",
    }


def run_iterative_boltzmann_inversion(
    initial_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    box: np.ndarray,
    options: IBIOptions,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an isotropic pair potential by vectorized, device-resident IBI."""
    options.validate()
    coordinates_np = np.asarray(initial_coordinates)
    reference_np = np.asarray(reference_coordinates)
    if coordinates_np.ndim != 4 or reference_np.ndim != 4:
        raise ValueError("coordinates must have shapes (E, M, N, 2)")
    if coordinates_np.shape[-1] != 2 or reference_np.shape[-1] != 2:
        raise ValueError("the final coordinate dimension must be two")
    if coordinates_np.shape[1:] != reference_np.shape[1:]:
        raise ValueError(
            "initial and reference ensembles must share (M, N, 2) shape"
        )

    dtype = _working_dtype(coordinates_np, reference_np, box)
    coordinates = jax.device_put(jnp.asarray(coordinates_np, dtype=dtype))
    reference = jax.device_put(jnp.asarray(reference_np, dtype=dtype))
    box_array = jax.device_put(jnp.asarray(box, dtype=dtype))
    if box_array.shape != (2,):
        raise ValueError("box must have shape (2,)")

    bin_centers = jnp.linspace(
        options.radial_max / (2 * options.num_bins),
        options.radial_max * (1 - 1 / (2 * options.num_bins)),
        options.num_bins,
        dtype=dtype,
    )
    result = _run_ibi_device(
        coordinates,
        reference,
        box_array,
        bin_centers,
        jax.random.PRNGKey(seed),
        jnp.asarray(options.proposal_scale, dtype=dtype),
        jnp.asarray(options.inverse_temperature, dtype=dtype),
        jnp.asarray(options.update_rate, dtype=dtype),
        jnp.asarray(options.kernel_width, dtype=dtype),
        jnp.asarray(options.density_floor, dtype=dtype),
        jnp.asarray(options.potential_clip, dtype=dtype),
        jnp.asarray(options.initial_repulsion, dtype=dtype),
        jnp.asarray(options.radial_max, dtype=dtype),
        num_iterations=options.num_iterations,
        metropolis_steps_per_iteration=options.metropolis_steps_per_iteration,
    )
    (
        coordinates,
        target_density,
        final_density,
        potential,
        density_rmse_history,
        acceptance_history,
    ) = jax.device_get(result)
    history = [
        {
            "iteration": float(index),
            "density_rmse": float(density_rmse_history[index]),
            "acceptance_rate": float(acceptance_history[index]),
        }
        for index in range(options.num_iterations)
    ]
    convergence_tail = np.asarray(density_rmse_history)[
        -options.convergence_window :
    ]
    convergence_relative_range = float(
        np.ptp(convergence_tail)
        / max(float(np.mean(convergence_tail)), np.finfo(np.float64).tiny)
    )
    return np.asarray(coordinates), {
        "bin_centers": np.asarray(jax.device_get(bin_centers)),
        "target_density": np.asarray(target_density),
        "final_density": np.asarray(final_density),
        "potential": np.asarray(potential),
        "density_rmse": float(
            np.sqrt(np.mean((final_density - target_density) ** 2))
        ),
        "history": history,
        "converged": bool(
            density_rmse_history[-1] <= options.density_rmse_tolerance
            and convergence_relative_range
            <= options.convergence_relative_tolerance
        ),
        "convergence_window": options.convergence_window,
        "convergence_relative_range": convergence_relative_range,
        "convergence_relative_tolerance": (
            options.convergence_relative_tolerance
        ),
        "density_rmse_tolerance": options.density_rmse_tolerance,
        "execution_backend": jax.default_backend(),
        "execution_device": str(jax.devices()[0]),
        "execution_strategy": (
            "vectorized ensembles + nested compiled lax.scan + "
            "incremental pair-energy updates"
        ),
        "information_budget": (
            "Gaussian-smoothed radial pair histogram estimated from "
            "microscopic training configurations; richer than the RBF condition"
        ),
    }
