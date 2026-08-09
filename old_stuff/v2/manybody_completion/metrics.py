"""Distributional and calibration metrics."""

from __future__ import annotations

import math
import numpy as np


def normalized(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probabilities have zero mass")
    return p / total


def weighted_mean(values: np.ndarray, probabilities: np.ndarray) -> float:
    p = normalized(probabilities)
    return float(np.sum(p * np.asarray(values, dtype=np.float64)))


def weighted_variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mean = weighted_mean(values, probabilities)
    return weighted_mean(np.square(values - mean), probabilities)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(normalized(p) - normalized(q))))


def smoothed_kl(reference: np.ndarray, candidate: np.ndarray, epsilon: float = 1e-12) -> float:
    p = normalized(reference)
    q = normalized(candidate)
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def energy_score_discrete(
    predictive_values: np.ndarray,
    predictive_probabilities: np.ndarray,
    reference_values: np.ndarray,
    reference_probabilities: np.ndarray,
) -> float:
    """Expected scalar energy score, evaluated exactly on finite supports."""
    x = np.asarray(predictive_values, dtype=np.float64)
    y = np.asarray(reference_values, dtype=np.float64)
    p = normalized(predictive_probabilities)
    q = normalized(reference_probabilities)
    cross = float(np.sum(p[:, None] * q[None, :] * np.abs(x[:, None] - y[None, :])))
    self_term = float(
        np.sum(p[:, None] * p[None, :] * np.abs(x[:, None] - x[None, :]))
    )
    return cross - 0.5 * self_term


def energy_distance_discrete(
    values: np.ndarray, p: np.ndarray, q: np.ndarray
) -> float:
    values = np.asarray(values, dtype=np.float64)
    p = normalized(p)
    q = normalized(q)
    distance = np.abs(values[:, None] - values[None, :])
    cross = 2.0 * float(np.sum(p[:, None] * q[None, :] * distance))
    pp = float(np.sum(p[:, None] * p[None, :] * distance))
    qq = float(np.sum(q[:, None] * q[None, :] * distance))
    return max(cross - pp - qq, 0.0)


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0))
