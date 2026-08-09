"""Small parameter container standing in for a learned population network."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PriorParameters:
    """Parameters of the joint latent-regime population prior.

    `mode_logit` controls P(S=+1), `regime_strength` controls the opposite
    triplet preferences, and the final two parameters control the shared pair
    marginal.  In a larger implementation these values would be emitted by a
    context encoder or equivariant energy network.
    """

    mode_logit: float
    regime_strength: float
    pair_center: float
    log_pair_penalty: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.mode_logit,
                self.regime_strength,
                self.pair_center,
                self.log_pair_penalty,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "PriorParameters":
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"expected shape (4,), got {arr.shape}")
        return cls(*(float(x) for x in arr))

    @classmethod
    def from_mapping(cls, values: dict[str, float]) -> "PriorParameters":
        return cls(
            mode_logit=float(values["mode_logit"]),
            regime_strength=float(values["regime_strength"]),
            pair_center=float(values["pair_center"]),
            log_pair_penalty=float(values["log_pair_penalty"]),
        )

    def to_mapping(self) -> dict[str, float]:
        return {
            "mode_logit": self.mode_logit,
            "regime_strength": self.regime_strength,
            "pair_center": self.pair_center,
            "log_pair_penalty": self.log_pair_penalty,
        }
