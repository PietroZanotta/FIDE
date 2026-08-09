"""Names for the conditional gradient-routing ablation."""

from enum import StrEnum


class GradientRoute(StrEnum):
    STOP_GRAD = "calibrated_stopgrad"
    FULL_E2E = "full_e2e"
