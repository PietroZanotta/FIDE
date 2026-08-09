"""Construct the finite population support and certify pair-level ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import enumerate_spin_states
from .observables import pair_correlation, triplet_correlation


@dataclass(frozen=True)
class PopulationSupport:
    spins: np.ndarray
    labels: np.ndarray
    pair: np.ndarray
    triplet: np.ndarray
    state_index: np.ndarray
    n_spins: int

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])


def build_population_support(n_spins: int = 8) -> PopulationSupport:
    states = enumerate_spin_states(n_spins)
    n_states = states.shape[0]
    spins = np.concatenate([states, states], axis=0)
    labels = np.concatenate(
        [-np.ones(n_states, dtype=np.float64), np.ones(n_states, dtype=np.float64)]
    )
    state_index = np.concatenate([np.arange(n_states), np.arange(n_states)])
    return PopulationSupport(
        spins=spins,
        labels=labels,
        pair=pair_correlation(spins).astype(np.float64),
        triplet=triplet_correlation(spins).astype(np.float64),
        state_index=state_index.astype(np.int64),
        n_spins=n_spins,
    )


def certify_pair_ambiguity(support: PopulationSupport, probabilities: np.ndarray) -> dict[str, float]:
    """Report pair and triplet differences between latent regimes."""
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum()
    out: dict[str, float] = {}
    for label, name in [(-1.0, "minus"), (1.0, "plus")]:
        mask = support.labels == label
        mass = float(p[mask].sum())
        cond = p[mask] / max(mass, 1e-300)
        out[f"mode_{name}_mass"] = mass
        out[f"mode_{name}_pair_mean"] = float(np.sum(cond * support.pair[mask]))
        out[f"mode_{name}_triplet_mean"] = float(np.sum(cond * support.triplet[mask]))
    out["pair_mean_gap"] = abs(out["mode_plus_pair_mean"] - out["mode_minus_pair_mean"])
    out["triplet_mean_gap"] = abs(
        out["mode_plus_triplet_mean"] - out["mode_minus_triplet_mean"]
    )
    return out
