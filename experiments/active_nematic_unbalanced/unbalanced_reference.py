"""Factorized two-species endpoint reference over the single t=21 -> 31 interval.

Each species owns an existing normalized ``PeriodicReferenceFlow`` for shape.
No intermediate physical marginal enters training.  Finite mass is represented
separately by one analytic, charge-coupled Fisher--Rao pair-mass schedule, so no
neural reaction-rate head is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .domain import EmpiricalEndpointSource
    from .periodic_reference import PeriodicReferenceFlow
    from .unbalanced_state import Species, TwoSpeciesDefectBank
except ImportError:  # pragma: no cover
    from domain import EmpiricalEndpointSource
    from periodic_reference import PeriodicReferenceFlow
    from unbalanced_state import Species, TwoSpeciesDefectBank


Array = jax.Array


def sample_periodic_kde_bank(
    states: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    periods: np.ndarray,
    position_std: float,
    beta_std: float,
) -> np.ndarray:
    """Sample a reproducible periodic KDE quadrature from endpoint defects.

    Resampling without jitter only duplicates the finite empirical atoms and
    can impose an artificial convex-hull barrier on held-out local moments.
    These declared bandwidths represent a continuous endpoint law; hidden
    intermediate physical marginals remain absent from the construction.
    """
    states = np.asarray(states, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 3 or periods.shape != (3,):
        raise ValueError("states and periods must have shapes [n,3] and [3]")
    if probabilities.shape != (len(states),) or len(states) == 0:
        raise ValueError("probabilities must align with a nonempty endpoint bank")
    if sample_count < 1 or position_std < 0.0 or beta_std < 0.0:
        raise ValueError("sample count must be positive and KDE widths nonnegative")
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(int(seed))
    index = rng.choice(len(states), size=int(sample_count), p=probabilities)
    samples = states[index].copy()
    bandwidth = np.asarray([position_std, position_std, beta_std])
    if np.any(bandwidth > 0.0):
        samples += rng.normal(size=samples.shape) * bandwidth
    return np.mod(samples, periods)


@dataclass(frozen=True)
class FisherRaoPairMassSchedule:
    """Square-root interpolation of pair mass with fixed charge imbalance."""

    pair_mass_start: float
    pair_mass_end: float
    charge_imbalance: float = 0.0
    minimum_mass: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.pair_mass_start, self.pair_mass_end) <= 0.0:
            raise ValueError("Fisher--Rao endpoint pair masses must be positive")
        endpoint_species = (
            self.pair_mass_start + 0.5 * self.charge_imbalance,
            self.pair_mass_start - 0.5 * self.charge_imbalance,
            self.pair_mass_end + 0.5 * self.charge_imbalance,
            self.pair_mass_end - 0.5 * self.charge_imbalance,
        )
        if min(endpoint_species) < self.minimum_mass:
            raise ValueError("Fisher--Rao endpoint species mass is below minimum_mass")

    def pair_mass_and_derivative(self, tau: Array) -> tuple[Array, Array]:
        tau = jnp.asarray(tau, dtype=jnp.float64)
        if bool(jnp.any((tau < 0.0) | (tau > 1.0))):
            raise ValueError("normalized reference time must lie in [0,1]")
        root0 = jnp.sqrt(float(self.pair_mass_start))
        root1 = jnp.sqrt(float(self.pair_mass_end))
        root = (1.0 - tau) * root0 + tau * root1
        derivative = 2.0 * root * (root1 - root0)
        return root**2, derivative

    def species_mass(self, species: Species, tau: Array) -> Array:
        pair, _ = self.pair_mass_and_derivative(tau)
        sign = 1.0 if species == "plus" else -1.0
        mass = pair + sign * 0.5 * float(self.charge_imbalance)
        if bool(jnp.any(mass < self.minimum_mass)):
            raise ValueError(f"{species} reference mass is below minimum_mass")
        return mass

    def species_source_rate(self, species: Species, tau: Array) -> Array:
        _, derivative = self.pair_mass_and_derivative(tau)
        return derivative / self.species_mass(species, tau)

    def to_dict(self, times: Array) -> dict[str, object]:
        times = jnp.asarray(times, dtype=jnp.float64)
        return {
            "interpolation": "fisher_rao",
            "pair_mass_start": self.pair_mass_start,
            "pair_mass_end": self.pair_mass_end,
            "charge_imbalance": self.charge_imbalance,
            "mass_plus": np.asarray(self.species_mass("plus", times)).tolist(),
            "mass_minus": np.asarray(self.species_mass("minus", times)).tolist(),
            "source_plus": np.asarray(
                self.species_source_rate("plus", times)
            ).tolist(),
            "source_minus": np.asarray(
                self.species_source_rate("minus", times)
            ).tolist(),
        }


@dataclass(frozen=True)
class TwoSpeciesReference:
    plus: PeriodicReferenceFlow
    minus: PeriodicReferenceFlow
    mass_schedule: FisherRaoPairMassSchedule
    plus_seed: int
    minus_seed: int
    metadata: Mapping[str, object] | None = None

    def flow(self, species: Species) -> PeriodicReferenceFlow:
        return self.plus if species == "plus" else self.minus

    def mass(self, species: Species, tau: Array) -> Array:
        return self.mass_schedule.species_mass(species, tau)

    def source_rate(self, species: Species, tau: Array) -> Array:
        return self.mass_schedule.species_source_rate(species, tau)


def endpoint_source_for_species(
    bank: TwoSpeciesDefectBank,
    species: Species,
    *,
    run_indices: np.ndarray,
    sample_count: int,
    seed: int,
    minimum_mass: float,
) -> EmpiricalEndpointSource:
    """Draw normalized endpoint samples without assigning them physical mass."""
    first = bank.measure(species, 0, run_indices)
    last = bank.measure(species, len(bank.times) - 1, run_indices)
    p0 = first.normalized_probabilities(minimum_mass=minimum_mass)
    p1 = last.normalized_probabilities(minimum_mass=minimum_mass)
    rng = np.random.default_rng(int(seed))
    x0 = first.states[rng.choice(len(first.states), size=int(sample_count), p=p0)]
    x1 = last.states[rng.choice(len(last.states), size=int(sample_count), p=p1)]
    return EmpiricalEndpointSource(jnp.asarray(x0), jnp.asarray(x1))


def endpoint_pair_mass_schedule(
    bank: TwoSpeciesDefectBank,
    *,
    run_indices: np.ndarray,
    minimum_mass: float,
) -> FisherRaoPairMassSchedule:
    plus = bank.mean_mass("plus", run_indices)
    minus = bank.mean_mass("minus", run_indices)
    # Reference construction is endpoint-only: even the constant topological
    # sector is inferred from t=21 and t=31, never from hidden marginals.
    imbalance = float(0.5 * ((plus[0] - minus[0]) + (plus[-1] - minus[-1])))
    pair = 0.5 * (plus + minus)
    return FisherRaoPairMassSchedule(
        pair_mass_start=float(pair[0]),
        pair_mass_end=float(pair[-1]),
        charge_imbalance=imbalance,
        minimum_mass=float(minimum_mass),
    )
