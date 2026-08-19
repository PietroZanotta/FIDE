"""Shared sensor geometry with charge-resolved beta-phase channels."""

from __future__ import annotations

from dataclasses import dataclass

import jax

try:
    from .measurements import PeriodicGaussianSensors
    from .unbalanced_state import Species
except ImportError:  # pragma: no cover
    from measurements import PeriodicGaussianSensors
    from unbalanced_state import Species


Array = jax.Array


@dataclass(frozen=True)
class ChargeResolvedSensors:
    """One set of movable positions applied independently to both populations.

    Both signs use ``(occupancy, cos(beta), sin(beta))`` numerically.  For the
    minus population these angular channels encode triatic phase, never vector
    polarity.  Global masses are auxiliary observations and do not increase
    ``n_sensors``.
    """

    box_size: float
    width: float
    n_sensors: int
    use_plus_orientation: bool = True
    use_minus_triatic_orientation: bool = True
    mass_is_global_observation: bool = True

    def family(self, species: Species) -> PeriodicGaussianSensors:
        angular = (
            self.use_plus_orientation
            if species == "plus"
            else self.use_minus_triatic_orientation
        )
        channels = (
            ("occupancy", "polarity_cos", "polarity_sin")
            if angular
            else ("occupancy",)
        )
        return PeriodicGaussianSensors(
            box_size=self.box_size,
            width=self.width,
            n_sensors=self.n_sensors,
            channels=channels,
        )

    @property
    def plus_observables(self) -> int:
        return self.family("plus").n_observables

    @property
    def minus_observables(self) -> int:
        return self.family("minus").n_observables

    def canonicalize(self, eta: Array) -> Array:
        return self.family("plus").canonicalize(eta)

    def centers(self, eta: Array) -> Array:
        return self.family("plus").centers(eta)
