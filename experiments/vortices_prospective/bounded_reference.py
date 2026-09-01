from __future__ import annotations

"""Boundary-compatible endpoint reference for the double-gyre experiment.

The generic MLP reference flow lives on R^2.  For the double-gyre benchmark the
physical state must remain in Omega=[0,2]x[0,1].  We therefore train the same
flow-matching model in box-logit coordinates z in R^2 and map its rollout and
velocity back to physical coordinates by the logistic diffeomorphism.

This module is experiment-local on purpose: no generic mfsi reference or flow-
matching behavior is changed, which preserves the toy benchmark exactly.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from mfsi.flow_matching import FlowMatchingConfig, train_reference_flow as _train_reference_flow
from mfsi.reference import load_npz_checkpoint, rk4_rollout, velocity_mlp

Array = jax.Array

# Fixed scientific domain of the double-gyre benchmark.
BOX_LOW = (0.0, 0.0)
BOX_HIGH = (2.0, 1.0)
TRANSFORM_EPS = 1.0e-6
TRANSFORM_VERSION = "box_logit_reference_v1"


def _box_arrays(dtype=jnp.float64) -> tuple[Array, Array, Array]:
    lo = jnp.asarray(BOX_LOW, dtype=dtype)
    hi = jnp.asarray(BOX_HIGH, dtype=dtype)
    return lo, hi, hi - lo


def physical_to_latent(x: Array, *, eps: float = TRANSFORM_EPS) -> Array:
    """Map physical x in the rectangular domain to unconstrained logit coordinates."""
    x = jnp.asarray(x, dtype=jnp.float64)
    lo, _, span = _box_arrays(x.dtype)
    s = (x - lo) / span
    s = jnp.clip(s, float(eps), 1.0 - float(eps))
    return jnp.log(s) - jnp.log1p(-s)


def latent_to_physical(z: Array) -> Array:
    """Map unconstrained latent coordinates to the open physical rectangle."""
    z = jnp.asarray(z, dtype=jnp.float64)
    lo, _, span = _box_arrays(z.dtype)
    s = jax.nn.sigmoid(z)
    return lo + span * s


def latent_velocity_to_physical(x: Array, vz: Array) -> Array:
    """Chain-rule pushforward of dz/dt to dx/dt.

    If x = lo + span * sigmoid(z), then
        dx/dt = span * s * (1-s) * dz/dt.
    The Jacobian factor tends to zero at each physical boundary.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    vz = jnp.asarray(vz, dtype=jnp.float64)
    lo, _, span = _box_arrays(x.dtype)
    # Use the physical coordinate for the Jacobian factor.  Clipping only guards
    # roundoff; at an exact boundary the normal factor is exactly zero.
    s = jnp.clip((x - lo) / span, 0.0, 1.0)
    return span * s * (1.0 - s) * vz


@dataclass(frozen=True)
class _LatentEndpointSource:
    """Transform an existing physical EndpointSource into latent coordinates."""

    source: Any
    eps: float = TRANSFORM_EPS

    def sample(self, key: Array, n: int, endpoint: int) -> Array:
        return physical_to_latent(self.source.sample(key, n, endpoint), eps=self.eps)


@dataclass(frozen=True)
class BoxTransformedReferenceFlow:
    """Frozen endpoint-trained reference flow constrained to the double-gyre box."""

    params: Any
    substeps_per_interval: int = 16
    metadata: Mapping[str, Any] | None = None
    transform_eps: float = TRANSFORM_EPS

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        substeps_per_interval: int = 16,
    ) -> "BoxTransformedReferenceFlow":
        params, metadata = load_npz_checkpoint(path)
        marker = dict(metadata or {}).get("vortex_reference_transform")
        if marker != TRANSFORM_VERSION:
            raise RuntimeError(
                "incompatible vortex reference checkpoint: expected "
                f"vortex_reference_transform={TRANSFORM_VERSION!r}, got {marker!r}. "
                "The pre-fix unconstrained reference checkpoint must be retrained once."
            )
        eps = float(dict(metadata or {}).get("vortex_reference_transform_eps", TRANSFORM_EPS))
        return cls(
            params=params,
            substeps_per_interval=int(substeps_per_interval),
            metadata=metadata,
            transform_eps=eps,
        )

    def velocity(self, x: Array, t: Array) -> Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        z = physical_to_latent(x, eps=self.transform_eps)
        vz = velocity_mlp(self.params, t, z)
        return latent_velocity_to_physical(x, vz)

    def rollout(self, x0: Array, times: Array) -> Array:
        # Integrating in latent coordinates is preferable to integrating the
        # physical chain-rule field numerically: every returned particle lies
        # strictly inside the box by construction.
        z0 = physical_to_latent(x0, eps=self.transform_eps)
        z_nodes = rk4_rollout(
            self.params,
            z0,
            times,
            substeps_per_interval=self.substeps_per_interval,
        )
        return latent_to_physical(z_nodes)


def train_box_reference_flow(
    source,
    cfg: FlowMatchingConfig,
    *,
    substeps_per_interval: int,
):
    """Same public signature as mfsi.flow_matching.train_reference_flow.

    Training occurs in latent logit coordinates.  The returned reference exposes
    the physical-coordinate velocity and rollout expected by the MFSI pipeline.
    """
    latent_source = _LatentEndpointSource(source)
    raw, history = _train_reference_flow(
        latent_source,
        cfg,
        substeps_per_interval=int(substeps_per_interval),
    )
    metadata = dict(raw.metadata or {})
    metadata.update({
        "vortex_reference_transform": TRANSFORM_VERSION,
        "vortex_reference_transform_eps": float(TRANSFORM_EPS),
        "vortex_reference_box_low": list(BOX_LOW),
        "vortex_reference_box_high": list(BOX_HIGH),
        "coordinate_system": "box-logit latent flow; physical velocity via chain rule",
    })
    return (
        BoxTransformedReferenceFlow(
            params=raw.params,
            substeps_per_interval=int(substeps_per_interval),
            metadata=metadata,
            transform_eps=float(TRANSFORM_EPS),
        ),
        history,
    )
