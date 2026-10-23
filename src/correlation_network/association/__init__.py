"""Association strategies for computing pairwise relationships."""

from correlation_network.association._auto import AutoAssociationStrategy
from correlation_network.association._base import AssociationStrategy
from correlation_network.association._categorical import (
    CorrelationRatioStrategy,
    CramersVStrategy,
)
from correlation_network.association._correlation import (
    KendallStrategy,
    PearsonStrategy,
    SpearmanStrategy,
)
from correlation_network.association._universal import (
    MutualInformationStrategy,
    PhiKStrategy,
)

__all__ = [
    "AssociationStrategy",
    "AutoAssociationStrategy",
    "CorrelationRatioStrategy",
    "CramersVStrategy",
    "KendallStrategy",
    "MutualInformationStrategy",
    "PearsonStrategy",
    "PhiKStrategy",
    "SpearmanStrategy",
]
