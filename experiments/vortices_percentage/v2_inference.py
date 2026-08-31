"""Frozen cross-method and cross-reference Vortices V2 bootstrap inference."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def validate_action_arrays(
    law_actions: Any, full_actions: Any
) -> tuple[np.ndarray, np.ndarray]:
    law = np.asarray(law_actions, dtype=np.float64)
    full = np.asarray(full_actions, dtype=np.float64)
    if law.ndim != 2 or law.shape[0] != 3:
        raise ValueError("law_actions must have shape [3 references, trials]")
    if full.ndim != 3 or full.shape[:2] != (3, 6):
        raise ValueError("full_actions must have shape [3 references, 6 allowances, trials]")
    if full.shape[2] != law.shape[1]:
        raise ValueError("Law and Full must use the same trial count")
    if law.shape[1] != 1024:
        raise ValueError("the frozen validation analysis requires exactly 1,024 trials")
    if not np.all(np.isfinite(law)) or not np.all(np.isfinite(full)):
        raise ValueError("all retained actions must be finite")
    return law, full


def ratio_of_means_effects(law: np.ndarray, full: np.ndarray) -> np.ndarray:
    law_mean = np.mean(law, axis=1)
    if np.any(law_mean == 0.0):
        raise ValueError("Law arithmetic mean is zero")
    return 1.0 - np.mean(full, axis=2) / law_mean[:, None]


def effects_for_common_indices(
    law_actions: Any, full_actions: Any, indices: Any
) -> np.ndarray:
    """Evaluate effects using one explicit index matrix shared by all 18 cells."""
    law = np.asarray(law_actions, dtype=np.float64)
    full = np.asarray(full_actions, dtype=np.float64)
    take = np.asarray(indices, dtype=np.int64)
    if law.ndim != 2 or law.shape[0] != 3:
        raise ValueError("law_actions must have shape [3, trials]")
    if full.ndim != 3 or full.shape[:2] != (3, 6) or full.shape[2] != law.shape[1]:
        raise ValueError("full_actions must have shape [3, 6, trials]")
    if take.ndim != 2 or np.any(take < 0) or np.any(take >= law.shape[1]):
        raise ValueError("indices must have shape [resamples, draws] and be in range")
    law_means = np.mean(law[:, take], axis=-1).T
    full_means = np.mean(full[:, :, take], axis=-1).transpose(2, 0, 1)
    return 1.0 - full_means / law_means[:, :, None]


def common_index_bootstrap(
    law_actions: Any,
    full_actions: Any,
    *,
    resamples: int = 100000,
    seed: int = 821775,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Use one index vector for every reference, method, and allowance.

    The returned digest commits to the complete ordered stream of common index
    vectors without retaining an approximately 800 MB integer array.
    """
    law, full = validate_action_arrays(law_actions, full_actions)
    if int(resamples) != 100000 or int(seed) != 821775:
        raise ValueError("the frozen analysis requires 100,000 resamples and seed 821775")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    observed = ratio_of_means_effects(law, full)
    bootstrap = np.empty((int(resamples), 3, 6), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    digest = hashlib.sha256()
    n = law.shape[1]
    for begin in range(0, int(resamples), int(chunk_size)):
        take = min(int(chunk_size), int(resamples) - begin)
        indices = rng.integers(0, n, size=(take, n), dtype=np.int32)
        digest.update(indices.tobytes(order="C"))
        bootstrap[begin : begin + take] = effects_for_common_indices(
            law, full, indices
        )
    maximum_deviation = np.max(np.abs(bootstrap - observed[None]), axis=(1, 2))
    critical = float(np.quantile(maximum_deviation, 0.95))
    pointwise = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    law_mean = np.mean(law, axis=1)
    full_mean = np.mean(full, axis=2)
    law_se = np.std(law, axis=1, ddof=1) / np.sqrt(n)
    full_se = np.std(full, axis=2, ddof=1) / np.sqrt(n)
    return {
        "effects": observed,
        "bootstrap_effects": bootstrap,
        "pointwise_lower": pointwise[0],
        "pointwise_upper": pointwise[1],
        "simultaneous_critical_half_width": critical,
        "simultaneous_lower": observed - critical,
        "simultaneous_upper": observed + critical,
        "law_mean": law_mean,
        "law_se": law_se,
        "law_relative_se": law_se / np.abs(law_mean),
        "full_mean": full_mean,
        "full_se": full_se,
        "full_relative_se": full_se / np.abs(full_mean),
        "shared_index_stream_sha256": digest.hexdigest(),
        "pairing": "one common index vector for all 3 references, Law, and 6 Full allowances",
        "resamples": int(resamples),
        "seed": int(seed),
    }
