"""Shared graph utility functions."""

import networkx as nx

# Floor for converted distances. A |weight| of 1.0 would otherwise map to a
# zero-length edge, collapsing two nodes into the same point for path-based
# metrics and making closeness degenerate.
_MIN_DISTANCE = 1e-6


def distance_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy with edge weights converted to distances.

    Association strength and graph distance run in opposite directions: a
    strong association (``|w|`` near 1) means two nodes are *close*, while
    NetworkX shortest-path algorithms read the edge attribute as a *cost*.
    Passing raw association values to Dijkstra therefore routes paths through
    the weakest edges. This helper applies ``distance = 1 - |weight|`` so that
    strong associations become short distances.

    Parameters
    ----------
    graph : nx.Graph
        A weighted undirected graph whose weights are association values.

    Returns
    -------
    nx.Graph
        Copy with edge weights replaced by distances, floored at a small
        positive value so no edge has zero or negative length.
    """
    g = graph.copy()
    nx.set_edge_attributes(
        g,
        values={
            (u, v): max(_MIN_DISTANCE, 1.0 - abs(d.get("weight", 1.0)))
            for u, v, d in g.edges(data=True)
        },
        name="weight",
    )
    return g


def abs_weight_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy of the graph with absolute edge weights.

    Useful for algorithms (Dijkstra, layout) that require non-negative weights.

    Parameters
    ----------
    graph : nx.Graph
        A weighted undirected graph.

    Returns
    -------
    nx.Graph
        Copy with all edge weights replaced by their absolute values.
    """
    g = graph.copy()
    nx.set_edge_attributes(
        g,
        values={(u, v): abs(d.get("weight", 1)) for u, v, d in g.edges(data=True)},
        name="weight",
    )
    return g
