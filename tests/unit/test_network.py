"""Tests for CorrelationNetwork."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from correlation_network.exceptions import NotFittedError
from correlation_network.network import CorrelationNetwork


class TestCorrelationNetworkInit:
    def test_default_parameters(self):
        net = CorrelationNetwork()
        assert net.method == "pearson"
        assert net.threshold == 0.5
        assert net.cardinality_threshold == 0.95

    def test_custom_parameters(self):
        net = CorrelationNetwork(method="spearman", threshold=0.3)
        assert net.method == "spearman"
        assert net.threshold == 0.3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            CorrelationNetwork(method="invalid")

    def test_threshold_too_low_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CorrelationNetwork(threshold=0.0)

    def test_threshold_too_high_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CorrelationNetwork(threshold=1.0)

    def test_all_methods_accepted(self):
        methods = [
            "pearson",
            "spearman",
            "kendall",
            "cramers_v",
            "correlation_ratio",
            "mutual_information",
            "auto",
        ]
        for method in methods:
            net = CorrelationNetwork(method=method)
            assert net.method == method


class TestCorrelationNetworkFit:
    def test_fit_returns_self(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork()
        result = net.fit(numeric_dataframe)
        assert result is net

    def test_fit_creates_association_matrix(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork()
        net.fit(numeric_dataframe)
        assert hasattr(net, "association_matrix_")
        assert isinstance(net.association_matrix_, pd.DataFrame)

    def test_fit_creates_adjacency_matrix(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork()
        net.fit(numeric_dataframe)
        assert hasattr(net, "adjacency_matrix_")
        assert isinstance(net.adjacency_matrix_, pd.DataFrame)

    def test_fit_with_column_subset(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork()
        net.fit(numeric_dataframe, columns=["x", "y"])
        assert list(net.association_matrix_.columns) == ["x", "y"]

    def test_threshold_zeroes_weak_values(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.9)
        net.fit(numeric_dataframe)
        adj = net.adjacency_matrix_
        # All non-zero values should exceed threshold in absolute value
        non_zero = adj.values[adj.values != 0]
        assert all(abs(v) > 0.9 for v in non_zero)


class TestCorrelationNetworkTransform:
    def test_transform_before_fit_raises(self):
        net = CorrelationNetwork()
        with pytest.raises(NotFittedError):
            net.transform()

    def test_transform_returns_graph(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        net.fit(numeric_dataframe)
        graph = net.transform()
        assert isinstance(graph, nx.Graph)

    def test_graph_has_no_self_loops(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        net.fit(numeric_dataframe)
        graph = net.transform()
        assert len(list(nx.selfloop_edges(graph))) == 0

    def test_graph_nodes_match_columns(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        net.fit(numeric_dataframe)
        graph = net.transform()
        assert set(graph.nodes()) == set(numeric_dataframe.columns)


class TestCorrelationNetworkFitTransform:
    def test_fit_transform_returns_graph(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        graph = net.fit_transform(numeric_dataframe)
        assert isinstance(graph, nx.Graph)

    def test_fit_transform_equivalent_to_fit_then_transform(
        self, numeric_dataframe: pd.DataFrame
    ):
        net1 = CorrelationNetwork(threshold=0.3)
        graph1 = net1.fit_transform(numeric_dataframe)

        net2 = CorrelationNetwork(threshold=0.3)
        net2.fit(numeric_dataframe)
        graph2 = net2.transform()

        assert set(graph1.edges()) == set(graph2.edges())


class TestPreprocessing:
    def test_datetime_columns_excluded(self, dataframe_with_datetime: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        with pytest.warns(UserWarning, match="Datetime columns excluded"):
            net.fit(dataframe_with_datetime)
        assert "timestamp" not in net.association_matrix_.columns
        assert "timestamp" in net.excluded_columns_

    def test_high_cardinality_column_excluded(self, dataframe_with_id: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        with pytest.warns(UserWarning, match="High-cardinality columns excluded"):
            net.fit(dataframe_with_id)
        assert "row_id" not in net.association_matrix_.columns
        assert "uuid" not in net.association_matrix_.columns
        assert "row_id" in net.excluded_columns_
        assert "uuid" in net.excluded_columns_

    def test_normal_columns_not_excluded(self, numeric_dataframe: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        net.fit(numeric_dataframe)
        assert set(net.association_matrix_.columns) == {"x", "y", "z", "w"}
        assert net.excluded_columns_ == []

    def test_excluded_columns_attribute_stored(self, dataframe_with_id: pd.DataFrame):
        net = CorrelationNetwork(threshold=0.3)
        net.fit(dataframe_with_id)
        assert isinstance(net.excluded_columns_, list)
        assert len(net.excluded_columns_) > 0

    def test_cardinality_filter_disabled_with_none(
        self, dataframe_with_id: pd.DataFrame
    ):
        net = CorrelationNetwork(threshold=0.3, cardinality_threshold=None)
        net.fit(dataframe_with_id)
        # ID columns should still be in the matrix
        assert "row_id" in net.association_matrix_.columns

    def test_empty_dataframe_raises(self):
        net = CorrelationNetwork()
        empty_df = pd.DataFrame(
            {"x": pd.Series(dtype=float), "y": pd.Series(dtype=float)}
        )
        with pytest.raises(ValueError, match="empty DataFrame"):
            net.fit(empty_df)

    def test_all_columns_excluded_raises(self):
        """If preprocessing removes all columns, raise ValueError."""
        df = pd.DataFrame({"id": range(100), "uuid": [f"u-{i}" for i in range(100)]})
        net = CorrelationNetwork(threshold=0.3, cardinality_threshold=0.5)
        with pytest.raises(ValueError, match="Fewer than 2 columns remain"):
            net.fit(df)


class TestFromMatrix:
    @pytest.fixture()
    def sample_matrix(self) -> pd.DataFrame:
        """A simple 3x3 symmetric association matrix."""
        data = np.array(
            [
                [1.0, 0.8, 0.2],
                [0.8, 1.0, 0.4],
                [0.2, 0.4, 1.0],
            ]
        )
        return pd.DataFrame(data=data, index=["a", "b", "c"], columns=["a", "b", "c"])

    def test_creates_fitted_instance(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix, threshold=0.3)
        graph = net.transform()
        assert isinstance(graph, nx.Graph)

    def test_stores_association_matrix(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix, threshold=0.3)
        pd.testing.assert_frame_equal(net.association_matrix_, sample_matrix)

    def test_applies_threshold(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix, threshold=0.5)
        # 0.2 and 0.4 are below threshold, should be zeroed
        assert net.adjacency_matrix_.loc["a", "c"] == 0.0
        assert net.adjacency_matrix_.loc["b", "c"] == 0.0
        # 0.8 is above threshold, should be kept
        assert net.adjacency_matrix_.loc["a", "b"] == 0.8

    def test_no_threshold_passes_through(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix)
        pd.testing.assert_frame_equal(net.adjacency_matrix_, sample_matrix)

    def test_graph_has_correct_nodes(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix, threshold=0.3)
        graph = net.transform()
        assert set(graph.nodes()) == {"a", "b", "c"}

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="Expected a pandas or Polars"):
            CorrelationNetwork.from_matrix(np.eye(3))

    def test_rejects_non_square(self):
        df = pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="square"):
            CorrelationNetwork.from_matrix(df)

    def test_rejects_mismatched_labels(self):
        data = np.eye(2)
        df = pd.DataFrame(data=data, index=["a", "b"], columns=["x", "y"])
        with pytest.raises(ValueError, match="index must match"):
            CorrelationNetwork.from_matrix(df)

    def test_rejects_asymmetric(self):
        data = np.array([[1.0, 0.5], [0.3, 1.0]])
        df = pd.DataFrame(data=data, index=["a", "b"], columns=["a", "b"])
        with pytest.raises(ValueError, match="symmetric"):
            CorrelationNetwork.from_matrix(df)

    def test_repr_shows_precomputed(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix, threshold=0.3)
        assert "precomputed" in repr(net)

    def test_excluded_columns_is_empty(self, sample_matrix: pd.DataFrame):
        net = CorrelationNetwork.from_matrix(sample_matrix)
        assert net.excluded_columns_ == []

    def test_reapplying_threshold_when_none_is_a_noop(
        self, sample_matrix: pd.DataFrame
    ):
        # a None threshold must not blow up on the unary minus
        net = CorrelationNetwork.from_matrix(sample_matrix)
        pd.testing.assert_frame_equal(
            net._apply_threshold(sample_matrix), sample_matrix
        )

    def test_value_equal_to_threshold_is_kept(self):
        data = np.array([[1.0, 0.5], [0.5, 1.0]])
        df = pd.DataFrame(data=data, index=["a", "b"], columns=["a", "b"])
        net = CorrelationNetwork.from_matrix(df, threshold=0.5)
        # documented contract: threshold is the *minimum* value keeping an edge
        assert net.adjacency_matrix_.loc["a", "b"] == 0.5
        assert net.transform().number_of_edges() == 1

    def test_negative_value_equal_to_threshold_is_kept(self):
        data = np.array([[1.0, -0.5], [-0.5, 1.0]])
        df = pd.DataFrame(data=data, index=["a", "b"], columns=["a", "b"])
        net = CorrelationNetwork.from_matrix(df, threshold=0.5)
        assert net.adjacency_matrix_.loc["a", "b"] == -0.5


pl = pytest.importorskip("polars")


class TestPolarsSupport:
    def test_fit_accepts_polars_dataframe(self, numeric_dataframe: pd.DataFrame):
        polars_df = pl.from_pandas(numeric_dataframe)
        net = CorrelationNetwork(threshold=0.3)
        net.fit(polars_df)
        assert isinstance(net.association_matrix_, pd.DataFrame)
        assert set(net.association_matrix_.columns) == {"x", "y", "z", "w"}

    def test_fit_transform_accepts_polars(self, numeric_dataframe: pd.DataFrame):
        polars_df = pl.from_pandas(numeric_dataframe)
        net = CorrelationNetwork(threshold=0.3)
        graph = net.fit_transform(polars_df)
        assert isinstance(graph, nx.Graph)
        assert set(graph.nodes()) == {"x", "y", "z", "w"}

    def test_fit_with_column_subset(self, numeric_dataframe: pd.DataFrame):
        polars_df = pl.from_pandas(numeric_dataframe)
        net = CorrelationNetwork(threshold=0.3)
        net.fit(polars_df, columns=["x", "y"])
        assert list(net.association_matrix_.columns) == ["x", "y"]

    def test_fit_accepts_polars_lazy_frame(self, numeric_dataframe: pd.DataFrame):
        lazy_df = pl.from_pandas(numeric_dataframe).lazy()
        net = CorrelationNetwork(threshold=0.3)
        net.fit(lazy_df)
        assert isinstance(net.association_matrix_, pd.DataFrame)

    def test_results_match_pandas(self, numeric_dataframe: pd.DataFrame):
        polars_df = pl.from_pandas(numeric_dataframe)

        net_pd = CorrelationNetwork(threshold=0.3)
        net_pd.fit(numeric_dataframe)

        net_pl = CorrelationNetwork(threshold=0.3)
        net_pl.fit(polars_df)

        pd.testing.assert_frame_equal(
            net_pd.association_matrix_, net_pl.association_matrix_
        )

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError, match="Expected a pandas or Polars"):
            net = CorrelationNetwork(threshold=0.3)
            net.fit({"not": "a dataframe"})


class TestRepr:
    def test_repr(self):
        net = CorrelationNetwork(method="spearman", threshold=0.4)
        assert repr(net) == "CorrelationNetwork(method='spearman', threshold=0.4)"
