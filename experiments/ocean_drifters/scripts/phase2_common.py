"""Shared helpers for the frozen NOAA drifter MFSI Phase-2 benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_phase2_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else root() / "experiments/ocean_drifters/configs/mfsi_phase2.json"
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root() / path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmpiricalEndpointSource:
    """The same fixed-array EndpointSource interface used by vortices."""

    x0: jax.Array
    x1: jax.Array

    def __post_init__(self) -> None:
        if self.x0.shape != self.x1.shape or self.x0.ndim != 2 or self.x0.shape[-1] != 2:
            raise ValueError("x0 and x1 must both have shape [N,2]")

    def sample(self, key: jax.Array, n: int, endpoint: int) -> jax.Array:
        if endpoint not in (0, 1):
            raise ValueError("endpoint must be 0 or 1")
        bank = self.x0 if endpoint == 0 else self.x1
        indices = jax.random.randint(key, (int(n),), 0, int(bank.shape[0]))
        return bank[indices]


def gaussian_features_numpy(points: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    delta = points[..., None, :] - centers
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / float(sigma) ** 2)


def rff_parameters(seed: int, features: int, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(2, int(features))) / float(bandwidth)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(int(features),))
    return omega, phase


def rff_map(points: np.ndarray, omega: np.ndarray, phase: np.ndarray, dtype=np.float32) -> np.ndarray:
    scale = np.sqrt(2.0 / omega.shape[1])
    return np.asarray(scale * np.cos(np.asarray(points) @ omega + phase), dtype=dtype)


def seasons_from_month(month: np.ndarray) -> np.ndarray:
    labels = np.empty(len(month), dtype="U3")
    labels[np.isin(month, [12, 1, 2])] = "DJF"
    labels[np.isin(month, [3, 4, 5])] = "MAM"
    labels[np.isin(month, [6, 7, 8])] = "JJA"
    labels[np.isin(month, [9, 10, 11])] = "SON"
    return labels
