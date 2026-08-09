"""Co-adaptive flow matching and differentiable population completion."""

from .adaptive_components import (
    ProposalArchitecture,
    ProposalModel,
    WarmStartArchitecture,
    WarmStartModel,
)
from .flow import FlowArchitecture, FlowModel, sample_flow_distribution
from .homometric import PopulationSupport, build_population_support
from .network import PriorParameters
from .solvers import (
    calibrate_dual,
    calibrate_dual_from_probabilities,
    tilted_ensemble,
    tilted_ensemble_from_probabilities,
)

__all__ = [
    "FlowArchitecture",
    "FlowModel",
    "ProposalArchitecture",
    "ProposalModel",
    "WarmStartArchitecture",
    "WarmStartModel",
    "PriorParameters",
    "PopulationSupport",
    "build_population_support",
    "sample_flow_distribution",
    "calibrate_dual",
    "calibrate_dual_from_probabilities",
    "tilted_ensemble",
    "tilted_ensemble_from_probabilities",
]
