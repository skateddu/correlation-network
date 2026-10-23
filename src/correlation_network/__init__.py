"""correlation-network: Build and analyze association networks from DataFrames."""

from importlib.metadata import version

from correlation_network._types import (
    AssociationMethod,
    BinningMethod,
    CentralityMetric,
    LayoutAlgorithm,
)
from correlation_network.association import (
    AssociationStrategy,
    AutoAssociationStrategy,
    CorrelationRatioStrategy,
    CramersVStrategy,
    KendallStrategy,
    MutualInformationStrategy,
    PearsonStrategy,
    PhiKStrategy,
    SpearmanStrategy,
)
from correlation_network.centrality import CentralityAnalyzer
from correlation_network.exceptions import (
    CorrelationNetworkError,
    InsufficientColumnsError,
    NoCompleteObservationsError,
    NotFittedError,
)
from correlation_network.network import CorrelationNetwork
from correlation_network.visualization import NetworkVisualizer

# Read from the installed metadata so pyproject.toml stays the single source
__version__ = version("correlation-network")

__all__ = [
    "AssociationMethod",
    "AssociationStrategy",
    "AutoAssociationStrategy",
    "BinningMethod",
    "CentralityAnalyzer",
    "CentralityMetric",
    "CorrelationNetwork",
    "CorrelationNetworkError",
    "CorrelationRatioStrategy",
    "CramersVStrategy",
    "InsufficientColumnsError",
    "KendallStrategy",
    "LayoutAlgorithm",
    "MutualInformationStrategy",
    "NetworkVisualizer",
    "NoCompleteObservationsError",
    "NotFittedError",
    "PearsonStrategy",
    "PhiKStrategy",
    "SpearmanStrategy",
]
