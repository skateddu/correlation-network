"""Tests for CentralityAnalyzer."""

import networkx as nx
import pandas as pd
import pytest

from correlation_network.centrality import CentralityAnalyzer


@pytest.fixture()
def sample_graph() -> nx.Graph:
    """Create a simple weighted graph for testing."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=0.9)
    graph.add_edge("A", "C", weight=0.7)
    graph.add_edge("B", "C", weight=0.5)
    graph.add_edge("C", "D", weight=0.8)
    return graph


class TestCentralityAnalyzer:
    def test_degree_returns_series(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.degree()
        assert isinstance(result, pd.Series)
        assert result.name == "degree"

    def test_strength_returns_series(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.strength()
        assert isinstance(result, pd.Series)
        assert result.name == "strength"

    def test_strength_values_are_correct(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.strength()
        # A is connected to B(0.9) and C(0.7)
        assert result["A"] == pytest.approx(1.6)

    def test_betweenness_returns_series(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.betweenness()
        assert isinstance(result, pd.Series)
        assert result.name == "betweenness"

    def test_eigenvector_returns_series(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.eigenvector()
        assert isinstance(result, pd.Series)
        assert result.name == "eigenvector"

    def test_closeness_returns_series(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.closeness()
        assert isinstance(result, pd.Series)
        assert result.name == "closeness"

    def test_summary_returns_dataframe(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.summary()
        assert isinstance(result, pd.DataFrame)
        expected_columns = {
            "degree",
            "strength",
            "betweenness",
            "eigenvector",
            "closeness",
        }
        assert set(result.columns) == expected_columns

    def test_summary_contains_all_nodes(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.summary()
        assert set(result.index) == {"A", "B", "C", "D"}

    def test_degree_is_sorted_descending(self, sample_graph: nx.Graph):
        analyzer = CentralityAnalyzer(sample_graph)
        result = analyzer.degree()
        values = result.values.tolist()
        assert values == sorted(values, reverse=True)


@pytest.fixture()
def hub_graph() -> nx.Graph:
    """Path graph where B is the hub joined by strong edges.

    A --0.9-- B --0.9-- C --0.1-- D --0.1-- A

    Every shortest path between A and C should run through B (the strong
    route), not around via D (the weak route).
    """
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=0.9)
    graph.add_edge("B", "C", weight=0.9)
    graph.add_edge("C", "D", weight=0.1)
    graph.add_edge("D", "A", weight=0.1)
    return graph


class TestPathMetricsUseAssociationAsCloseness:
    """Strong associations must read as short distances, not long ones.

    NetworkX interprets the edge attribute passed to its shortest-path
    routines as a *cost*. Feeding raw association values in inverts the
    meaning, so these assert on values rather than types.
    """

    def test_betweenness_ranks_strong_hub_above_weak_detour(self, hub_graph: nx.Graph):
        result = CentralityAnalyzer(hub_graph).betweenness()
        assert result["B"] > result["D"]

    def test_betweenness_hub_is_strictly_positive(self, hub_graph: nx.Graph):
        result = CentralityAnalyzer(hub_graph).betweenness()
        assert result["B"] > 0

    def test_closeness_ranks_strong_hub_above_weak_detour(self, hub_graph: nx.Graph):
        result = CentralityAnalyzer(hub_graph).closeness()
        assert result["B"] > result["D"]

    def test_strength_still_uses_raw_magnitudes(self, hub_graph: nx.Graph):
        # strength is a magnitude sum, not a path metric: B = 0.9 + 0.9
        result = CentralityAnalyzer(hub_graph).strength()
        assert result["B"] == pytest.approx(1.8)

    def test_negative_weights_are_treated_by_magnitude(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=-0.9)
        graph.add_edge("B", "C", weight=0.9)
        graph.add_edge("C", "D", weight=0.1)
        graph.add_edge("D", "A", weight=0.1)
        result = CentralityAnalyzer(graph).betweenness()
        # a strong negative association is still a strong association
        assert result["B"] > result["D"]

    def test_perfect_association_does_not_produce_zero_distance(self):
        graph = nx.Graph()
        graph.add_edge("A", "B", weight=1.0)
        graph.add_edge("B", "C", weight=1.0)
        # a zero-length edge would make closeness blow up or divide by zero
        result = CentralityAnalyzer(graph).closeness()
        assert result.notna().all()
        assert (result > 0).all()
