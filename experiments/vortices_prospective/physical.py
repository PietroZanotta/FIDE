from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from aggregate_qois import qoi_features
from common import VORTICES_DIR  # noqa: F401  (ensures local vortices import path)
from domain import DoubleGyreConfig, DoubleGyreTruth, InitialLawConfig


def truth_from_config(cfg: dict[str, Any]) -> DoubleGyreTruth:
    block = cfg["truth"]
    initial = block["initial"]
    return DoubleGyreTruth(
        flow=DoubleGyreConfig(
            amplitude=float(block["amplitude"]),
            epsilon=float(block["epsilon"]),
            horizon=float(block["horizon"]),
            period=float(block["period"]),
        ),
        initial=InitialLawConfig(
            background_weight=float(initial["background_weight"]),
            mixture_weights=tuple(float(v) for v in initial["mixture_weights"]),
            centers=tuple(tuple(float(x) for x in center) for center in initial["centers"]),
            std_x=float(initial["std_x"]),
            std_y=float(initial["std_y"]),
        ),
    )


def gaussian_response_direct(states, centers, width: float):
    states = jnp.asarray(states, dtype=jnp.float64)
    centers = jnp.asarray(centers, dtype=jnp.float64)
    delta = states[..., None, :] - centers
    return jnp.mean(
        jnp.exp(-0.5 * jnp.sum(delta * delta, axis=-1) / float(width) ** 2),
        axis=-2,
    )


def numpy_qoi_means(states: np.ndarray) -> np.ndarray:
    return np.asarray(jnp.mean(qoi_features(states), axis=-2), dtype=np.float64)
