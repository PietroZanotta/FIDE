"""Ablation definitions and stage-selection rules for generator training.

The four modes follow the methodology's comparison table while avoiding an
ambiguous straight-through estimator:

* ``base`` trains and serves the native generator output.
* ``post_hoc`` trains the same native objective as ``base`` and applies both
  scientific solvers only for evaluation/serving.
* ``relax_e2e`` trains through physical relaxation; projection is evaluation-only.
* ``full_e2e`` trains through both scientific solvers.

This means ``base`` and ``post_hoc`` have identical optimization trajectories
under identical initialization and data.  Their reported outputs differ because
``post_hoc`` serves the fully repaired ensemble.  Keeping this equivalence
explicit is preferable to silently replacing stopped solver gradients with an
identity straight-through gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompletionStage(str, Enum):
    """Named coordinate/moment stages in the completion pipeline."""

    INITIAL = "initial"
    RELAXED = "relaxed"
    PROJECTED = "projected"


class AblationMode(str, Enum):
    """Supported generator-training and serving variants."""

    BASE = "base"
    POST_HOC = "post_hoc"
    RELAX_E2E = "relax_e2e"
    FULL_E2E = "full_e2e"

    @classmethod
    def parse(cls, value: "AblationMode | str") -> "AblationMode":
        """Normalize a user/config value into a validated mode."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "posthoc": cls.POST_HOC,
            "relax": cls.RELAX_E2E,
            "full": cls.FULL_E2E,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"unknown ablation mode {value!r}; choose one of: {choices}") from exc


@dataclass(frozen=True)
class AblationSpec:
    """Static routing contract for one ablation mode."""

    mode: AblationMode
    training_stage: CompletionStage
    serving_stage: CompletionStage
    differentiates_relaxation: bool
    differentiates_projection: bool

    @property
    def requires_training_relaxation(self) -> bool:
        return self.training_stage in (CompletionStage.RELAXED, CompletionStage.PROJECTED)

    @property
    def requires_training_projection(self) -> bool:
        return self.training_stage is CompletionStage.PROJECTED


_ABLATION_SPECS = {
    AblationMode.BASE: AblationSpec(
        mode=AblationMode.BASE,
        training_stage=CompletionStage.INITIAL,
        serving_stage=CompletionStage.INITIAL,
        differentiates_relaxation=False,
        differentiates_projection=False,
    ),
    AblationMode.POST_HOC: AblationSpec(
        mode=AblationMode.POST_HOC,
        training_stage=CompletionStage.INITIAL,
        serving_stage=CompletionStage.PROJECTED,
        differentiates_relaxation=False,
        differentiates_projection=False,
    ),
    AblationMode.RELAX_E2E: AblationSpec(
        mode=AblationMode.RELAX_E2E,
        training_stage=CompletionStage.RELAXED,
        serving_stage=CompletionStage.PROJECTED,
        differentiates_relaxation=True,
        differentiates_projection=False,
    ),
    AblationMode.FULL_E2E: AblationSpec(
        mode=AblationMode.FULL_E2E,
        training_stage=CompletionStage.PROJECTED,
        serving_stage=CompletionStage.PROJECTED,
        differentiates_relaxation=True,
        differentiates_projection=True,
    ),
}


def get_ablation_spec(mode: AblationMode | str) -> AblationSpec:
    """Return the immutable routing specification for ``mode``."""
    return _ABLATION_SPECS[AblationMode.parse(mode)]


def stage_key(prefix: str, stage: CompletionStage) -> str:
    """Build a key used by :func:`run_local_completion`."""
    if prefix == "coordinates":
        return f"{stage.value}_coordinates"
    return f"{prefix}_{stage.value}"
