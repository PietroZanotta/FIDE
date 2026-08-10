"""Gray–Scott morphogenesis benchmark construction for Experiment C."""

from .simulator import GrayScottParameters, generate_initial_conditions, simulate
from .observables import ShellDefinition, field_observables

__all__ = [
    "GrayScottParameters",
    "ShellDefinition",
    "field_observables",
    "generate_initial_conditions",
    "simulate",
]
