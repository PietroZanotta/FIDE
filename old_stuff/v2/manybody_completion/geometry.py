"""Geometry and state enumeration for a periodic Ising ring."""

from __future__ import annotations

import itertools
import numpy as np


def enumerate_spin_states(n_spins: int) -> np.ndarray:
    """Return every {-1,+1} spin configuration in lexicographic order."""
    if n_spins < 3:
        raise ValueError("n_spins must be at least 3")
    return np.asarray(list(itertools.product((-1.0, 1.0), repeat=n_spins)), dtype=np.float64)


def periodic_roll(spins: np.ndarray, shift: int) -> np.ndarray:
    """Roll the final spin axis with periodic boundary conditions."""
    return np.roll(spins, shift=shift, axis=-1)


def global_spin_flip(spins: np.ndarray) -> np.ndarray:
    """Apply the global Z2 symmetry."""
    return -np.asarray(spins)


def validate_spin_array(spins: np.ndarray) -> None:
    values = np.unique(np.asarray(spins))
    if not set(values.tolist()).issubset({-1.0, 1.0}):
        raise ValueError(f"spin array contains values outside {{-1,+1}}: {values}")
