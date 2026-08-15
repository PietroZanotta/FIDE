from __future__ import annotations

from typing import Any, Protocol

import jax

Array = jax.Array


class ReferenceFlow(Protocol):
    """Frozen reference path: velocity plus rollout."""

    def rollout(self, x0: Array, times: Array) -> Array: ...
    def velocity(self, x: Array, t: Array) -> Array: ...


class EndpointSource(Protocol):
    """Experiment-owned endpoint data source."""

    def sample(self, key: Array, n: int, endpoint: int) -> Array: ...


class MeasurementFamily(Protocol):
    """Differentiable observable family parameterized by eta."""

    def features(self, x: Array, eta: Array) -> Array: ...
    def feature_gradients(self, x: Array, eta: Array) -> Array: ...
    def canonicalize(self, eta: Array) -> Array: ...


class MomentReconstructor(Protocol):
    def reconstruct(self, *args: Any, **kwargs: Any) -> Any: ...


class IProjector(Protocol):
    def project(self, *args: Any, **kwargs: Any) -> Any: ...


class LawMetric(Protocol):
    def __call__(self, eta: Array) -> Array: ...


class ActionEvaluator(Protocol):
    def __call__(self, eta: Array) -> Array: ...
