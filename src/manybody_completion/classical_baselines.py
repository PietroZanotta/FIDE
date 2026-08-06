"""Observation-only classical baselines for the homometric benchmark.

Reverse Monte Carlo (RMC) matches the configured smooth pair moments directly.
The iterative Boltzmann inversion (IBI) implementation uses a Gaussian-smoothed
radial pair histogram estimated from microscopic training configurations.  That
is still pair-level information, but it is richer than the RBF condition used
by RMC and the learned models; the comparison report records this explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    def validate(self) -> None:
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if self.metropolis_steps_per_iteration < 1:
            raise ValueError("metropolis_steps_per_iteration must be positive")
        if self.proposal_scale <= 0 or self.inverse_temperature <= 0:
            raise ValueError("proposal_scale and inverse_temperature must be positive")
        if self.update_rate <= 0:
            raise ValueError("update_rate must be positive")
        if self.num_bins < 4 or self.radial_max <= 0 or self.kernel_width <= 0:
            raise ValueError("invalid radial discretization")
        if self.density_floor <= 0 or self.potential_clip <= 0:
            raise ValueError("density_floor and potential_clip must be positive")


def _chord_distances(coordinates: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = coordinates[..., :, None, :] - coordinates[..., None, :, :]
    displacement = (box / np.pi) * np.sin(np.pi * delta / box)
    return np.sqrt(np.sum(displacement * displacement, axis=-1) + 1e-24)


def _unordered_distances(coordinates: np.ndarray, box: np.ndarray) -> np.ndarray:
    distances = _chord_distances(coordinates, box)
    num_particles = coordinates.shape[-2]
    row, column = np.triu_indices(num_particles, k=1)
    return distances[..., row, column]


def _ensemble_pair_moments(
    ensemble: np.ndarray,
    box: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    distances = _chord_distances(ensemble, box)
    num_particles = ensemble.shape[-2]
    mask = 1.0 - np.eye(num_particles, dtype=ensemble.dtype)
    standardized = (distances[..., None] - centers) / widths
    values = np.exp(-0.5 * standardized * standardized) * mask[..., None]
    per_replica = np.sum(values, axis=(-3, -2)) / (
        num_particles * (num_particles - 1)
    )
    return np.mean(per_replica, axis=0)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(values, 0.0)


def _physical_energy(
    ensemble: np.ndarray,
    box: np.ndarray,
    r0: float,
    kappa: float,
) -> float:
    distances = _chord_distances(ensemble, box)
    num_particles = ensemble.shape[-2]
    mask = 1.0 - np.eye(num_particles, dtype=ensemble.dtype)
    penalty = _softplus(kappa * (r0 - distances)) ** 2 * mask
    return float(np.mean(np.sum(penalty, axis=(-2, -1)) / (num_particles * (num_particles - 1))))


def _rmc_objective(
    ensemble: np.ndarray,
    target_moments: np.ndarray,
    moment_scales: np.ndarray,
    box: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    options: RMCOptions,
) -> float:
    moments = _ensemble_pair_moments(ensemble, box, centers, widths)
    residual = (moments - target_moments) / np.maximum(moment_scales, 1e-12)
    objective = float(np.sum(residual * residual))
    if options.physical_weight:
        objective += options.physical_weight * _physical_energy(
            ensemble,
            box,
            options.physical_r0,
            options.physical_kappa,
        )
    return objective


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
    """Run independent RMC chains for a batch of ensembles.

    The evaluation ensemble is the independent statistical unit.  Each chain
    starts from the same matched-prior coordinates used by the learned models.
    """
    options.validate()
    coordinates = np.asarray(initial_coordinates, dtype=np.float64).copy()
    target = np.asarray(target_moments, dtype=np.float64)
    scales = np.asarray(moment_scales, dtype=np.float64)
    box_array = np.asarray(box, dtype=np.float64)
    centers = np.asarray(basis_centers, dtype=np.float64)
    widths = np.asarray(basis_widths, dtype=np.float64)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("initial_coordinates must have shape (E, M, N, 2)")
    if target.shape == (centers.size,):
        target = np.broadcast_to(target, (coordinates.shape[0], centers.size))
    if target.shape != (coordinates.shape[0], centers.size):
        raise ValueError("target_moments has an incompatible shape")

    rng = np.random.default_rng(seed)
    accepted = np.zeros(coordinates.shape[0], dtype=np.int64)
    current = np.asarray(
        [
            _rmc_objective(
                coordinates[index],
                target[index],
                scales,
                box_array,
                centers,
                widths,
                options,
            )
            for index in range(coordinates.shape[0])
        ]
    )
    best = coordinates.copy()
    best_objective = current.copy()
    trace_steps: list[int] = [0]
    trace_objective: list[float] = [float(np.mean(current))]

    for step in range(options.num_steps):
        fraction = step / max(options.num_steps - 1, 1)
        temperature = options.initial_temperature * (
            options.final_temperature / options.initial_temperature
        ) ** fraction
        for ensemble_index in range(coordinates.shape[0]):
            replica_index = int(rng.integers(coordinates.shape[1]))
            particle_index = int(rng.integers(coordinates.shape[2]))
            proposal = coordinates[ensemble_index].copy()
            proposal[replica_index, particle_index] = np.mod(
                proposal[replica_index, particle_index]
                + rng.normal(scale=options.proposal_scale, size=2),
                box_array,
            )
            proposal_objective = _rmc_objective(
                proposal,
                target[ensemble_index],
                scales,
                box_array,
                centers,
                widths,
                options,
            )
            delta = proposal_objective - current[ensemble_index]
            if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
                coordinates[ensemble_index] = proposal
                current[ensemble_index] = proposal_objective
                accepted[ensemble_index] += 1
                if proposal_objective < best_objective[ensemble_index]:
                    best[ensemble_index] = proposal
                    best_objective[ensemble_index] = proposal_objective
        if (step + 1) % options.trace_stride == 0 or step + 1 == options.num_steps:
            trace_steps.append(step + 1)
            trace_objective.append(float(np.mean(best_objective)))

    return best, {
        "objective_before": float(trace_objective[0]),
        "objective_after": float(np.mean(best_objective)),
        "acceptance_rate": float(np.mean(accepted / options.num_steps)),
        "per_ensemble_objective": best_objective,
        "trace_steps": np.asarray(trace_steps, dtype=np.int32),
        "trace_objective": np.asarray(trace_objective, dtype=np.float64),
        "information_budget": "smooth pair-moment condition only",
    }


def radial_density(
    coordinates: np.ndarray,
    box: np.ndarray,
    bin_centers: np.ndarray,
    kernel_width: float,
) -> np.ndarray:
    """Gaussian-smoothed normalized density of unordered chord distances."""
    distances = _unordered_distances(coordinates, box).reshape(-1)
    standardized = (distances[:, None] - bin_centers[None, :]) / kernel_width
    density = np.mean(np.exp(-0.5 * standardized * standardized), axis=0)
    return density / np.maximum(np.sum(density), 1e-15)


def ibi_update_potential(
    potential: np.ndarray,
    current_density: np.ndarray,
    target_density: np.ndarray,
    options: IBIOptions,
) -> np.ndarray:
    """One damped IBI update, with smoothing and a fixed additive gauge."""
    ratio = (current_density + options.density_floor) / (
        target_density + options.density_floor
    )
    updated = potential + options.update_rate * np.log(ratio)
    if updated.size >= 3:
        padded = np.pad(updated, (1, 1), mode="edge")
        updated = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    updated = updated - np.min(updated)
    return np.clip(updated, 0.0, options.potential_clip)


def _interpolated_potential(
    distances: np.ndarray,
    bin_centers: np.ndarray,
    potential: np.ndarray,
) -> np.ndarray:
    return np.interp(
        distances,
        bin_centers,
        potential,
        left=potential[0],
        right=potential[-1],
    )


def _pair_potential_energy(
    ensemble: np.ndarray,
    box: np.ndarray,
    bin_centers: np.ndarray,
    potential: np.ndarray,
) -> float:
    distances = _unordered_distances(ensemble, box)
    values = _interpolated_potential(distances, bin_centers, potential)
    return float(np.mean(np.sum(values, axis=-1)))


def run_iterative_boltzmann_inversion(
    initial_coordinates: np.ndarray,
    reference_coordinates: np.ndarray,
    box: np.ndarray,
    options: IBIOptions,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an isotropic pair potential by histogram IBI and return its samples."""
    options.validate()
    coordinates = np.asarray(initial_coordinates, dtype=np.float64).copy()
    reference = np.asarray(reference_coordinates, dtype=np.float64)
    box_array = np.asarray(box, dtype=np.float64)
    if coordinates.ndim != 4 or reference.ndim != 4:
        raise ValueError("coordinates must have shapes (E, M, N, 2)")
    bin_centers = np.linspace(
        options.radial_max / (2 * options.num_bins),
        options.radial_max * (1 - 1 / (2 * options.num_bins)),
        options.num_bins,
    )
    target_density = radial_density(
        reference,
        box_array,
        bin_centers,
        options.kernel_width,
    )
    potential = options.initial_repulsion * np.exp(
        -0.5 * (bin_centers / max(0.15 * options.radial_max, 1e-3)) ** 2
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    for iteration in range(options.num_iterations):
        accepted = 0
        attempted = 0
        energies = np.asarray(
            [
                _pair_potential_energy(item, box_array, bin_centers, potential)
                for item in coordinates
            ]
        )
        for _ in range(options.metropolis_steps_per_iteration):
            for ensemble_index in range(coordinates.shape[0]):
                replica_index = int(rng.integers(coordinates.shape[1]))
                particle_index = int(rng.integers(coordinates.shape[2]))
                proposal = coordinates[ensemble_index].copy()
                proposal[replica_index, particle_index] = np.mod(
                    proposal[replica_index, particle_index]
                    + rng.normal(scale=options.proposal_scale, size=2),
                    box_array,
                )
                proposal_energy = _pair_potential_energy(
                    proposal,
                    box_array,
                    bin_centers,
                    potential,
                )
                delta = proposal_energy - energies[ensemble_index]
                attempted += 1
                if delta <= 0.0 or rng.random() < np.exp(
                    -options.inverse_temperature * delta
                ):
                    coordinates[ensemble_index] = proposal
                    energies[ensemble_index] = proposal_energy
                    accepted += 1
        current_density = radial_density(
            coordinates,
            box_array,
            bin_centers,
            options.kernel_width,
        )
        density_rmse = float(np.sqrt(np.mean((current_density - target_density) ** 2)))
        history.append(
            {
                "iteration": float(iteration),
                "density_rmse": density_rmse,
                "acceptance_rate": accepted / max(attempted, 1),
            }
        )
        potential = ibi_update_potential(
            potential,
            current_density,
            target_density,
            options,
        )

    final_density = radial_density(
        coordinates,
        box_array,
        bin_centers,
        options.kernel_width,
    )
    return coordinates, {
        "bin_centers": bin_centers,
        "target_density": target_density,
        "final_density": final_density,
        "potential": potential,
        "density_rmse": float(np.sqrt(np.mean((final_density - target_density) ** 2))),
        "history": history,
        "information_budget": (
            "Gaussian-smoothed radial pair histogram estimated from microscopic "
            "training configurations; richer than the RBF condition"
        ),
    }
