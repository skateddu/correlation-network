"""Centrality analysis for association networks."""

import functools

import networkx as nx
import pandas as pd

from correlation_network._graph_utils import abs_weight_graph, distance_graph


class CentralityAnalyzer:
    """Compute centrality metrics for a NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        A weighted undirected graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    @functools.cached_property
    def _abs_graph(self) -> nx.Graph:
        """Graph copy with absolute weights, for magnitude-based metrics."""
        return abs_weight_graph(self.graph)

    @functools.cached_property
    def _distance_graph(self) -> nx.Graph:
        """Graph copy with weights as distances, for path-based metrics (Dijkstra)."""
        return distance_graph(self.graph)

    def degree(self) -> pd.Series:
        """Normalized degree centrality.

        Returns
        -------
        pd.Series
            Fraction of possible connections for each node, sorted descending.
        """
        return pd.Series(nx.degree_centrality(self.graph), name="degree").sort_values(
            ascending=False
        )

    def strength(self) -> pd.Series:
        """Weighted degree (sum of absolute edge weights).

        Returns
        -------
        pd.Series
            Weighted degree for each node, sorted descending.
        """
        weighted_degree = dict(self._abs_graph.degree(weight="weight"))
        return pd.Series(weighted_degree, name="strength").sort_values(ascending=False)

    def betweenness(self) -> pd.Series:
        """Betweenness centrality.

        Association values are converted to distances (``1 - |weight|``) so
        that shortest paths run through strongly associated pairs.

        Returns
        -------
        pd.Series
            Betweenness centrality for each node, sorted descending.
        """
        return pd.Series(
            nx.betweenness_centrality(self._distance_graph, weight="weight"),
            name="betweenness",
        ).sort_values(ascending=False)

    def eigenvector(self) -> pd.Series:
        """Eigenvector centrality.

        Falls back to NumPy solver if the power iteration does not converge.

        Returns
        -------
        pd.Series
            Eigenvector centrality for each node, sorted descending.
        """
        try:
            values = nx.eigenvector_centrality(self.graph, weight="weight")
        except nx.PowerIterationFailedConvergence:
            values = nx.eigenvector_centrality_numpy(self.graph, weight="weight")
        return pd.Series(values, name="eigenvector").sort_values(ascending=False)

    def closeness(self) -> pd.Series:
        """Closeness centrality.

        Association values are converted to distances (``1 - |weight|``) so
        that strongly associated nodes are treated as close together.

        Returns
        -------
        pd.Series
            Closeness centrality for each node, sorted descending.
        """
        return pd.Series(
            nx.closeness_centrality(self._distance_graph, distance="weight"),
            name="closeness",
        ).sort_values(ascending=False)

    def summary(self) -> pd.DataFrame:
        """Compute all centrality metrics and return as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: degree, strength, betweenness,
            eigenvector, closeness. Indexed by node name.
        """
        return pd.DataFrame(
            {
                "degree": self.degree(),
                "strength": self.strength(),
                "betweenness": self.betweenness(),
                "eigenvector": self.eigenvector(),
                "closeness": self.closeness(),
            }
        )
