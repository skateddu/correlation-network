"""End-to-end integration tests."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from correlation_network import (
    CentralityAnalyzer,
    CorrelationNetwork,
    NetworkVisualizer,
)


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """Create a realistic mixed-type DataFrame."""
    rng = np.random.default_rng(123)
    n = 500
    category = rng.choice(["low", "medium", "high"], n)
    price = np.where(
        np.array(category) == "low",
        rng.normal(loc=100, scale=10, size=n),
        np.where(
            np.array(category) == "medium",
            rng.normal(loc=200, scale=15, size=n),
            rng.normal(loc=300, scale=20, size=n),
        ),
    )
    sqft = price * 2 + rng.normal(loc=0, scale=30, size=n)
    bedrooms = rng.integers(low=1, high=6, size=n).astype(float)
    region = rng.choice(["north", "south", "east", "west"], n)
    return pd.DataFrame(
        {
            "category": category,
            "region": region,
            "price": price,
            "sqft": sqft,
            "bedrooms": bedrooms,
        }
    )


@pytest.mark.integration
class TestEndToEnd:
    def test_numeric_pipeline(self, numeric_dataframe: pd.DataFrame):
        """Full pipeline: fit -> transform -> centrality -> visualize."""
        net = CorrelationNetwork(method="pearson", threshold=0.3)
        graph = net.fit_transform(numeric_dataframe)

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0

        analyzer = CentralityAnalyzer(graph)
        summary = analyzer.summary()
        assert not summary.empty
        assert len(summary.columns) == 5

        viz = NetworkVisualizer(graph)
        fig = viz.plot(centrality_metric="degree")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_auto_method_with_mixed_data(self, mixed_df: pd.DataFrame):
        """Auto-detection with mixed numeric/categorical data."""
        net = CorrelationNetwork(method="auto", threshold=0.3)
        graph = net.fit_transform(mixed_df)

        assert isinstance(graph, nx.Graph)
        assert "category" in graph.nodes()
        assert "price" in graph.nodes()

        # Strongly associated columns should have edges
        if graph.has_edge("price", "sqft"):
            weight = graph.edges["price", "sqft"]["weight"]
            assert abs(weight) > 0.3

    def test_mutual_information_pipeline(self, numeric_dataframe: pd.DataFrame):
        """Pipeline using mutual information strategy."""
        net = CorrelationNetwork(method="mutual_information", threshold=0.1)
        graph = net.fit_transform(numeric_dataframe)
        assert isinstance(graph, nx.Graph)

    def test_column_selection(self, numeric_dataframe: pd.DataFrame):
        """Selecting a subset of columns works."""
        net = CorrelationNetwork(threshold=0.3)
        graph = net.fit_transform(numeric_dataframe, columns=["x", "y", "z"])
        assert set(graph.nodes()) == {"x", "y", "z"}

    def test_pipeline_with_datetime_and_id_columns(self):
        """Full pipeline with datetime, ID, numeric, and categorical columns."""
        rng = np.random.default_rng(99)
        n = 200
        x = rng.normal(loc=0, scale=1, size=n)
        y = x * 0.9 + rng.normal(loc=0, scale=0.2, size=n)
        df = pd.DataFrame(
            {
                "row_id": range(n),
                "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="h"),
                "category": rng.choice(["A", "B", "C"], size=n),
                "x": x,
                "y": y,
            }
        )

        net = CorrelationNetwork(method="auto", threshold=0.3)
        graph = net.fit_transform(df)

        assert isinstance(graph, nx.Graph)
        # ID and datetime columns should have been excluded
        assert "row_id" not in graph.nodes()
        assert "timestamp" not in graph.nodes()
        # Valid columns should remain
        assert "x" in graph.nodes()
        assert "y" in graph.nodes()
        assert "category" in graph.nodes()

    def test_from_matrix_pipeline(self, numeric_dataframe: pd.DataFrame):
        """Create network from pre-computed matrix, then analyze."""
        corr_matrix = numeric_dataframe.corr()
        net = CorrelationNetwork.from_matrix(corr_matrix, threshold=0.3)
        graph = net.transform()

        assert isinstance(graph, nx.Graph)
        assert set(graph.nodes()) == set(numeric_dataframe.columns)

        analyzer = CentralityAnalyzer(graph)
        summary = analyzer.summary()
        assert not summary.empty
