from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from common import fingerprint
from mfsi.cache import file_sha256


def _bilinear_table(table, x_grid, y_grid, centers):
    """Differentiable nonperiodic bilinear interpolation on the physical box."""
    values = jnp.asarray(table, dtype=jnp.float64)
    xg = jnp.asarray(x_grid, dtype=jnp.float64)
    yg = jnp.asarray(y_grid, dtype=jnp.float64)
    centers = jnp.asarray(centers, dtype=jnp.float64)
    x = jnp.clip(centers[..., 0], xg[0], xg[-1])
    y = jnp.clip(centers[..., 1], yg[0], yg[-1])
    ix = jnp.clip(jnp.searchsorted(xg, x, side="right") - 1, 0, len(xg) - 2)
    iy = jnp.clip(jnp.searchsorted(yg, y, side="right") - 1, 0, len(yg) - 2)
    x0, x1 = xg[ix], xg[ix + 1]
    y0, y1 = yg[iy], yg[iy + 1]
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)
    # table is [time, y, x]; advanced indices retain [time, sensor].
    v00 = values[:, iy, ix]
    v10 = values[:, iy, ix + 1]
    v01 = values[:, iy + 1, ix]
    v11 = values[:, iy + 1, ix + 1]
    return (
        v00 * (1.0 - fx)[None, :] * (1.0 - fy)[None, :]
        + v10 * fx[None, :] * (1.0 - fy)[None, :]
        + v01 * (1.0 - fx)[None, :] * fy[None, :]
        + v11 * fx[None, :] * fy[None, :]
    )


@dataclass(frozen=True)
class TargetProspectiveData:
    """Aggregate-only target interface used by selection.

    Deliberately absent: a hidden-validation path, intermediate particle array,
    simulator instance, or any method capable of loading microscopic truth.
    """

    endpoint_path: Path
    aggregate_path: Path
    endpoint_ensemble_0: np.ndarray
    endpoint_ensemble_1: np.ndarray
    times: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    response_mean_field: np.ndarray
    response_second_field: np.ndarray
    scientific_qoi_predictions: np.ndarray
    qoi_scales: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, endpoint_path: str | Path, aggregate_path: str | Path):
        endpoint_path = Path(endpoint_path).resolve()
        aggregate_path = Path(aggregate_path).resolve()
        forbidden = "hidden_validation"
        if forbidden in endpoint_path.parts or forbidden in aggregate_path.parts:
            raise PermissionError("prospective selection cannot load hidden-validation artifacts")
        with np.load(endpoint_path, allow_pickle=False) as endpoints:
            x0 = np.asarray(endpoints["x0"], dtype=np.float64)
            x1 = np.asarray(endpoints["x1"], dtype=np.float64)
            endpoint_role = str(np.asarray(endpoints["role"]).item())
        if endpoint_role != "endpoint_only_reference_training":
            raise ValueError("endpoint artifact has an invalid scientific role")
        with np.load(aggregate_path, allow_pickle=False) as data:
            role = str(np.asarray(data["role"]).item())
            if role != "prospective_aggregate_only":
                raise ValueError("aggregate artifact has an invalid scientific role")
            arrays = {key: np.asarray(data[key]) for key in data.files if key != "role"}
        metadata = {
            "role": role,
            "endpoint_sha256": file_sha256(endpoint_path),
            "aggregate_sha256": file_sha256(aggregate_path),
        }
        return cls(
            endpoint_path=endpoint_path,
            aggregate_path=aggregate_path,
            endpoint_ensemble_0=x0,
            endpoint_ensemble_1=x1,
            times=np.asarray(arrays["times"], dtype=np.float64),
            x_grid=np.asarray(arrays["x_grid"], dtype=np.float64),
            y_grid=np.asarray(arrays["y_grid"], dtype=np.float64),
            response_mean_field=np.asarray(arrays["response_mean_field"], dtype=np.float64),
            response_second_field=np.asarray(arrays["response_second_field"], dtype=np.float64),
            scientific_qoi_predictions=np.asarray(arrays["scientific_qoi_predictions"], dtype=np.float64),
            qoi_scales=np.asarray(arrays["qoi_scales"], dtype=np.float64),
            metadata=metadata,
        )

    @property
    def artifact_id(self) -> str:
        return fingerprint(self.metadata)

    def response(self, centers):
        return _bilinear_table(
            self.response_mean_field, self.x_grid, self.y_grid, centers
        )

    def response_second(self, centers):
        return _bilinear_table(
            self.response_second_field, self.x_grid, self.y_grid, centers
        )


__all__ = ["TargetProspectiveData", "_bilinear_table"]

