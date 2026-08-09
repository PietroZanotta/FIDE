"""Observed and held-out statistics for the spin-ring example."""

from __future__ import annotations

import numpy as np

from .geometry import periodic_roll, validate_spin_array


def magnetization(spins: np.ndarray) -> np.ndarray:
    validate_spin_array(spins)
    return np.mean(spins, axis=-1)


def pair_correlation(spins: np.ndarray) -> np.ndarray:
    """Mean nearest-neighbour product; this is the reduced observation."""
    validate_spin_array(spins)
    return np.mean(spins * periodic_roll(spins, -1), axis=-1)


def triplet_correlation(spins: np.ndarray) -> np.ndarray:
    """Mean cyclic three-spin product; held out from conditioning."""
    validate_spin_array(spins)
    return np.mean(
        spins * periodic_roll(spins, -1) * periodic_roll(spins, -2), axis=-1
    )


def domain_wall_fraction(spins: np.ndarray) -> np.ndarray:
    validate_spin_array(spins)
    return np.mean(spins != periodic_roll(spins, -1), axis=-1)
