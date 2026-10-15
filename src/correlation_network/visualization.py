"""Network visualization utilities."""

import functools
import os
from collections.abc import Callable
from itertools import compress
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from correlation_network._graph_utils import abs_weight_graph, distance_graph
from correlation_network._types import CentralityMetric, LayoutAlgorithm
from correlation_network._validation import check_option
from correlation_network.centrality import CentralityAnalyzer

# Each layout is paired with the weight transform it needs. Kamada-Kawai reads
# weights as target *distances*, so association strength becomes
# ``1 - |weight|``; raw values would draw the strongest associations as the
# longest edges. Spring and spectral read weights as attraction and only need
# them non-negative. Circular and shell ignore weights.
_LAYOUTS: dict[
    str, tuple[Callable[..., dict[Any, np.ndarray]], Callable[[nx.Graph], nx.Graph]]
] = {
    "kamada_kawai": (nx.kamada_kawai_layout, distance_graph),
    "spring": (nx.spring_layout, abs_weight_graph),
    "circular": (nx.circular_layout, lambda graph: graph),
    "shell": (nx.shell_layout, lambda graph: graph),
    "spectral": (nx.spectral_layout, abs_weight_graph),
}

_CENTRALITY_METRICS: dict[str, Callable[[CentralityAnalyzer], pd.Series]] = {
    "degree": CentralityAnalyzer.degree,
    "strength": CentralityAnalyzer.strength,
    "betweenness": CentralityAnalyzer.betweenness,
    "eigenvector": CentralityAnalyzer.eigenvector,
    "closeness": CentralityAnalyzer.closeness,
}


class NetworkVisualizer:
    """Visualize an association network graph.

    Parameters
    ----------
    graph : nx.Graph
        A weighted undirected graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    @functools.cached_property
    def _centrality_analyzer(self) -> CentralityAnalyzer:
        return CentralityAnalyzer(self.graph)

    @functools.cached_property
    def _community_colors(self) -> list[tuple[float, float, float, float]]:
        communities = nx.community.greedy_modularity_communities(self.graph)
        palette = plt.get_cmap("tab10")
        node_to_color: dict[str, tuple[float, float, float, float]] = {}
        for idx, community in enumerate(communities):
            color = palette(idx % 10)
            for node in community:
                node_to_color[node] = color
        return [node_to_color[n] for n in self.graph.nodes()]

    def plot(
        self,
        layout: LayoutAlgorithm = "kamada_kawai",
        centrality_metric: CentralityMetric | None = None,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
        node_color: str = "tab:blue",
        community_colors: bool = False,
        edge_cmap: str | None = "RdYlGn",
        positive_edge_color: str = "darkgreen",
        negative_edge_color: str = "tab:red",
        font_size: int = 10,
        edge_label_font_size: int = 8,
        show_edge_labels: bool = True,
        title: str | None = None,
    ) -> Figure:
        """Plot the network graph.

        Parameters
        ----------
        layout : LayoutAlgorithm
            Graph layout algorithm. Default ``"kamada_kawai"``.
        centrality_metric : CentralityMetric | None
            If provided, node sizes are scaled by this centrality measure.
        figsize : tuple[float, float]
            Figure size in inches.
        dpi : int
            Figure resolution.
        node_color : str
            Color for nodes (ignored when ``community_colors`` is ``True``).
        community_colors : bool
            If ``True``, detect communities via greedy modularity and color
            each node by its community. Overrides ``node_color``.
        edge_cmap : str | None
            Matplotlib colormap for continuous edge coloring based on weight,
            read as diverging around 0 (lower half = negative). When set, a
            colorbar is shown. When ``None``, falls back to
            the two-color mode using ``positive_edge_color`` / ``negative_edge_color``.
        positive_edge_color : str
            Color for positive edges (only used when ``edge_cmap`` is ``None``).
        negative_edge_color : str
            Color for negative edges (only used when ``edge_cmap`` is ``None``).
        font_size : int
            Font size for node labels.
        edge_label_font_size : int
            Font size for edge weight labels.
        show_edge_labels : bool
            Whether to display edge weight labels.
        title : str | None
            Optional title displayed above the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object (not shown automatically).
        """
        node_sizes = self._compute_node_sizes(metric=centrality_metric)
        pos = self._layout_positions(layout=layout)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        resolved_node_color = self._community_colors if community_colors else node_color

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=node_sizes,
            node_color=resolved_node_color,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            ax=ax,
        )

        edge_data = list(self.graph.edges(data="weight", default=1.0))
        edges = [(u, v) for u, v, _ in edge_data]
        weights = np.array([w for _, _, w in edge_data])
        abs_w = np.abs(weights)
        widths = np.maximum(0.5, abs_w * 3)

        # Alpha proportional to |weight|: stronger edges are more opaque
        if len(abs_w) > 0 and abs_w.max() > 0:
            alphas = 0.3 + 0.6 * (abs_w / abs_w.max())
        else:
            alphas = np.full_like(abs_w, 0.6)

        if edge_cmap is not None and len(edges) > 0:
            self._draw_edges_cmap(
                pos=pos,
                edges=edges,
                weights=weights,
                widths=widths,
                alphas=alphas,
                cmap_name=edge_cmap,
                fig=fig,
                ax=ax,
            )
        elif len(edges) > 0:
            self._draw_edges_two_color(
                pos=pos,
                edges=edges,
                weights=weights,
                widths=widths,
                alphas=alphas,
                positive_color=positive_edge_color,
                negative_color=negative_edge_color,
                ax=ax,
            )
            self._add_edge_legend(
                weights=weights,
                positive_color=positive_edge_color,
                negative_color=negative_edge_color,
                font_size=font_size,
                ax=ax,
            )

        nx.draw_networkx_labels(self.graph, pos, font_size=font_size, ax=ax)

        if show_edge_labels:
            edge_labels = {
                (u, v): round(d.get("weight", 0), 2) for u, v, d in self.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=edge_label_font_size,
                ax=ax,
            )

        if title is not None:
            ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

        ax.set_axis_off()
        fig.tight_layout()
        return fig

    def save(self, path: str | os.PathLike[str], *, dpi: int = 150, **plot_kwargs: Any) -> None:
        """Plot the graph and save it to a file.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Output file path (e.g. ``"network.png"``). Format is inferred
            from the extension.
        dpi : int
            Resolution for the saved image.
        **plot_kwargs
            Additional keyword arguments forwarded to :meth:`plot`.
        """
        fig = self.plot(**plot_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _layout_positions(self, layout: LayoutAlgorithm) -> dict[Any, np.ndarray]:
        """Compute node positions, adapting weights to what the layout expects.

        Raises
        ------
        ValueError
            If ``layout`` is not a supported algorithm.
        """
        check_option(name="layout", value=layout, options=_LAYOUTS)
        layout_fn, prepare = _LAYOUTS[layout]
        return layout_fn(prepare(self.graph))

    def _draw_edges_cmap(
        self,
        *,
        pos: dict,
        edges: list[tuple],
        weights: np.ndarray,
        widths: np.ndarray,
        alphas: np.ndarray,
        cmap_name: str,
        fig: Figure,
        ax: Axes,
    ) -> None:
        """Draw edges colored by a continuous colormap with a colorbar."""
        cmap = plt.get_cmap(cmap_name)
        norm = self._edge_norm(weights)
        edge_colors = cmap(norm(weights))

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=edges,
            width=widths.tolist(),
            edge_color=edge_colors,
            style="dashed",
            alpha=alphas.tolist(),
            ax=ax,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Association strength", shrink=0.8)

    @staticmethod
    def _edge_norm(weights: np.ndarray) -> Normalize:
        """Map edge weights to colormap positions with 0 at the centre.

        The edge colormap is read as diverging: its lower half means negative
        association. Stretching single-signed weights over the full range
        would paint the weakest positive edge in the "negative" colour, so
        such weights keep a symmetric range around 0 instead.
        """
        w_min, w_max = float(weights.min()), float(weights.max())
        if w_min < 0 < w_max:
            return TwoSlopeNorm(vcenter=0, vmin=w_min, vmax=w_max)
        limit = max(abs(w_min), abs(w_max)) or 1.0
        return Normalize(vmin=-limit, vmax=limit)

    def _draw_edges_two_color(
        self,
        *,
        pos: dict,
        edges: list[tuple],
        weights: np.ndarray,
        widths: np.ndarray,
        alphas: np.ndarray,
        positive_color: str,
        negative_color: str,
        ax: Axes,
    ) -> None:
        """Draw edges with two fixed colors and per-edge alpha."""
        positive_mask = weights >= 0
        negative_mask = ~positive_mask

        positive_edges = list(compress(edges, positive_mask))
        negative_edges = list(compress(edges, negative_mask))

        if positive_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=positive_edges,
                width=widths[positive_mask].tolist(),
                edge_color=positive_color,
                style="dashed",
                alpha=alphas[positive_mask].tolist(),
                ax=ax,
            )

        if negative_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=negative_edges,
                width=widths[negative_mask].tolist(),
                edge_color=negative_color,
                style="dashed",
                alpha=alphas[negative_mask].tolist(),
                ax=ax,
            )

    @staticmethod
    def _add_edge_legend(
        *,
        weights: np.ndarray,
        positive_color: str,
        negative_color: str,
        font_size: int,
        ax: Axes,
    ) -> None:
        """Add a positive/negative edge legend to the axes."""
        has_positive = bool((weights >= 0).any())
        has_negative = bool((weights < 0).any())

        legend_handles = []
        if has_positive:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=positive_color,
                    linestyle="dashed",
                    linewidth=2,
                    label="Positive",
                )
            )
        if has_negative:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=negative_color,
                    linestyle="dashed",
                    linewidth=2,
                    label="Negative",
                )
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="best",
                fontsize=font_size,
                framealpha=0.8,
            )

    def _compute_node_sizes(self, metric: CentralityMetric | None) -> list[float] | float:
        """Compute node sizes scaled by centrality.

        Raises
        ------
        ValueError
            If ``metric`` is not a supported centrality measure.
        """
        if metric is None:
            return 300.0

        check_option(name="centrality_metric", value=metric, options=_CENTRALITY_METRICS)
        values = _CENTRALITY_METRICS[metric](self._centrality_analyzer)

        # Scale to a reasonable range [100, 1500]
        nodes = list(self.graph.nodes())
        raw = values.reindex(nodes).fillna(0).to_numpy(dtype=float)
        if raw.max() == raw.min():
            return [300.0] * len(nodes)
        normalized = (raw - raw.min()) / (raw.max() - raw.min())
        return (100 + normalized * 1400).tolist()
