"""Type aliases and literal types."""

from typing import Literal

CorrelationMethod = Literal["pearson", "spearman", "kendall"]

AssociationMethod = Literal[
    "pearson",
    "spearman",
    "kendall",
    "cramers_v",
    "correlation_ratio",
    "phik",
    "mutual_information",
    "auto",
]

CentralityMetric = Literal[
    "degree",
    "strength",
    "betweenness",
    "eigenvector",
    "closeness",
]

LayoutAlgorithm = Literal[
    "kamada_kawai",
    "spring",
    "circular",
    "shell",
    "spectral",
]
