"""Scientific comparison package for periodic many-body completion."""

from .classical_baselines import (
    IBIOptions,
    RMCOptions,
    run_iterative_boltzmann_inversion,
    run_reverse_monte_carlo,
)
from .experiment import run_experiment
from .homometric import build_homometric_dataset, validate_homometric_pair
from .routing import AblationMode
from .scientific_comparison import run_scientific_comparison
from .uq import aggregate_seed_higher_order_uq, higher_order_conditional_uq

__all__ = [
    "AblationMode",
    "IBIOptions",
    "RMCOptions",
    "aggregate_seed_higher_order_uq",
    "build_homometric_dataset",
    "higher_order_conditional_uq",
    "run_experiment",
    "run_iterative_boltzmann_inversion",
    "run_reverse_monte_carlo",
    "run_scientific_comparison",
    "validate_homometric_pair",
]
