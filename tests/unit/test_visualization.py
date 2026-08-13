"""Tests for NetworkVisualizer."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pytest
from matplotlib.figure import Figure

from correlation_network.visualization import NetworkVisualizer


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figure a test leaves open, so Agg does not accumulate them."""
    yield
    plt.close("all")


@pytest.fixture()
def sample_graph() -> nx.Graph:
    """Create a graph with positive and negative edges."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=0.9)
    graph.add_edge("A", "C", weight=-0.6)
    graph.add_edge("B", "C", weight=0.5)
    return graph


@pytest.fixture()
def positive_graph() -> nx.Graph:
    """Graph whose weights are all positive (single-slope colormap branch)."""
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=0.9)
    graph.add_edge("B", "C", weight=0.5)
    graph.add_edge("C", "D", weight=0.7)
    return graph


class TestNetworkVisualizer:
    def test_plot_returns_figure(self, sample_graph: nx.Graph):
        fig = NetworkVisualizer(sample_graph).plot()
        assert isinstance(fig, Figure)

    def test_plot_with_centrality_metric(self, sample_graph: nx.Graph):
        fig = NetworkVisualizer(sample_graph).plot(centrality_metric="degree")
        assert isinstance(fig, Figure)

    @pytest.mark.parametrize(
        "layout",
        ["kamada_kawai", "spring", "circular", "shell", "spectral"],
    )
    def test_plot_with_each_layout(self, sample_graph: nx.Graph, layout: str):
        fig = NetworkVisualizer(sample_graph).plot(layout=layout)
        assert isinstance(fig, Figure)

    def test_plot_without_edge_labels(self, sample_graph: nx.Graph):
        fig = NetworkVisualizer(sample_graph).plot(show_edge_labels=False)
        assert isinstance(fig, Figure)


class TestEdgeColouringModes:
    def test_two_color_mode_draws_legend(self, sample_graph: nx.Graph):
        # edge_cmap=None switches to the positive/negative two-colour path
        fig = NetworkVisualizer(sample_graph).plot(edge_cmap=None)
        ax = fig.axes[0]
        labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert labels == {"Positive", "Negative"}

    def test_two_color_legend_omits_absent_sign(self, positive_graph: nx.Graph):
        fig = NetworkVisualizer(positive_graph).plot(edge_cmap=None)
        ax = fig.axes[0]
        labels = {t.get_text() for t in ax.get_legend().get_texts()}
        assert labels == {"Positive"}

    def test_colormap_mode_adds_colorbar(self, sample_graph: nx.Graph):
        fig = NetworkVisualizer(sample_graph).plot(edge_cmap="RdYlGn")
        # the colorbar lives on a second axes appended to the figure
        assert len(fig.axes) == 2

    def test_all_positive_weights_use_single_slope_norm(self, positive_graph: nx.Graph):
        fig = NetworkVisualizer(positive_graph).plot(edge_cmap="RdYlGn")
        assert isinstance(fig, Figure)

    def test_graph_without_edges_still_plots(self):
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B"])
        fig = NetworkVisualizer(graph).plot()
        assert isinstance(fig, Figure)


class TestCommunityColors:
    def test_community_colors_assigns_one_color_per_node(
        self, positive_graph: nx.Graph
    ):
        viz = NetworkVisualizer(positive_graph)
        colors = viz._community_colors
        assert len(colors) == positive_graph.number_of_nodes()

    def test_plot_with_community_colors(self, positive_graph: nx.Graph):
        fig = NetworkVisualizer(positive_graph).plot(community_colors=True)
        assert isinstance(fig, Figure)


class TestTitleAndSave:
    def test_title_is_rendered(self, sample_graph: nx.Graph):
        fig = NetworkVisualizer(sample_graph).plot(title="My Network")
        assert fig.axes[0].get_title() == "My Network"

    def test_save_writes_a_file(self, sample_graph: nx.Graph, tmp_path):
        target = tmp_path / "network.png"
        NetworkVisualizer(sample_graph).save(str(target), dpi=50)
        assert target.exists()
        assert target.stat().st_size > 0

    def test_save_forwards_plot_kwargs(self, sample_graph: nx.Graph, tmp_path):
        target = tmp_path / "titled.png"
        NetworkVisualizer(sample_graph).save(
            str(target), dpi=50, title="Forwarded", community_colors=True
        )
        assert target.exists()


class TestNodeSizes:
    def test_uniform_size_without_metric(self, sample_graph: nx.Graph):
        viz = NetworkVisualizer(sample_graph)
        assert viz._compute_node_sizes(metric=None) == 300.0

    def test_sizes_scale_with_metric(self, positive_graph: nx.Graph):
        viz = NetworkVisualizer(positive_graph)
        sizes = viz._compute_node_sizes(metric="degree")
        assert len(sizes) == positive_graph.number_of_nodes()
        assert min(sizes) >= 100
        assert max(sizes) <= 1500

    def test_flat_metric_falls_back_to_uniform_size(self):
        # a cycle gives every node identical degree -> no spread to scale
        graph = nx.cycle_graph(["A", "B", "C"])
        nx.set_edge_attributes(graph, 0.5, "weight")
        sizes = NetworkVisualizer(graph)._compute_node_sizes(metric="degree")
        assert sizes == [300.0, 300.0, 300.0]
