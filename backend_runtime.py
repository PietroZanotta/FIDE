"""Runtime backend adapters for MFSI scientific components.

``jax`` executes the shared kernels in-process.
``tesseract`` talks to *served Pasteur/ISI Labs Tesseract Core containers* over
HTTP.  No Tesseract Python SDK is used here.

Training stays in native JAX because it is ordinary neural-network optimization;
the backend switch controls execution of the two scientific component maps used
for generation/evaluation:

  ReferenceTransport      : (x,t,theta) -> u_theta(t,x)
  MomentFiberRealizer     : projected fiber state + Deep-Ritz correction
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


def _jsonable(x: Any):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "__array__"):
        return np.asarray(x).tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _decode_tesseract_json(x: Any):
    """Decode Tesseract Core's typed JSON array representation."""
    if isinstance(x, dict) and x.get("object_type") == "array":
        data = x["data"]
        if data.get("encoding") != "json":
            raise ValueError(f"expected JSON array encoding, got {data.get('encoding')!r}")
        return np.asarray(data["buffer"], dtype=x["dtype"]).reshape(x["shape"])
    if isinstance(x, dict):
        return {k: _decode_tesseract_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode_tesseract_json(v) for v in x]
    return x


def _post(url: str, endpoint: str, payload: dict, timeout: float = 300.0) -> dict:
    target = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
    body = json.dumps(_jsonable(payload)).encode("utf-8")
    req = urllib.request.Request(
        target,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _decode_tesseract_json(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tesseract request failed: {target}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Tesseract at {target}. Build/start it with the shell scripts "
            "or choose --backend jax."
        ) from exc


@dataclass(frozen=True)
class TesseractRESTBackend:
    """Client for two already-served Tesseract Core containers."""

    reference_url: str
    fiber_url: str
    timeout: float = 300.0

    @classmethod
    def from_env(cls) -> "TesseractRESTBackend":
        ref = os.environ.get("MFSI_REFERENCE_TESSERACT_URL")
        fib = os.environ.get("MFSI_FIBER_TESSERACT_URL")
        if not ref or not fib:
            raise RuntimeError(
                "Tesseract backend selected but served-component URLs are missing. "
                "Invoke experiments through scripts/run_*.sh --backend tesseract so "
                "the containers are started automatically."
            )
        return cls(ref, fib, float(os.environ.get("MFSI_TESSERACT_TIMEOUT", "300")))

    def health(self) -> dict[str, Any]:
        # /health is a GET endpoint, but a lightweight apply call is the most
        # portable end-to-end check because it also verifies the component API.
        return {"reference_url": self.reference_url, "fiber_url": self.fiber_url}

    def reference_velocity(self, velocity_params, t: float, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        out = _post(
            self.reference_url,
            "apply",
            {"inputs": {
                "x": x,
                "t": float(t),
                "velocity_params": np.asarray(velocity_params, dtype=np.float64),
            }},
            self.timeout,
        )
        return np.asarray(out["velocity"], dtype=np.float64)

    def fiber_apply(
        self,
        *,
        x,
        t: float,
        velocity,
        phi_values,
        jphi_u,
        target,
        log_base_weights,
        potential_params,
    ) -> dict[str, np.ndarray | float | int]:
        x = np.asarray(x, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if velocity.ndim == 1:
            velocity = velocity[:, None]
        out = _post(
            self.fiber_url,
            "apply",
            {"inputs": {
                "x": x,
                "t": float(t),
                "velocity": velocity,
                "phi_values": np.asarray(phi_values, dtype=np.float64),
                "jphi_u": np.asarray(jphi_u, dtype=np.float64),
                "target": np.asarray(target, dtype=np.float64),
                "log_base_weights": np.asarray(log_base_weights, dtype=np.float64),
                "potential_params": np.asarray(potential_params, dtype=np.float64),
            }},
            self.timeout,
        )
        arrays = {
            "lambda_value", "projected_weights", "moments", "covariance",
            "lambda_dot", "forcing", "correction", "velocity",
        }
        return {
            k: (np.asarray(v, dtype=np.float64) if k in arrays else v)
            for k, v in out.items()
        }


def normalize_backend(name: str | None) -> str:
    name = (name or os.environ.get("MFSI_BACKEND") or "tesseract").lower()
    if name not in {"jax", "tesseract"}:
        raise ValueError(f"unknown backend {name!r}; choose 'tesseract' or 'jax'")
    return name
