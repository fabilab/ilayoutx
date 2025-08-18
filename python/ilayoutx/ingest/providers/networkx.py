from typing import (
    Sequence,
    Optional,
)
from collections.abc import (
    Hashable,
)
import importlib
import numpy as np
import pandas as pd

from ..typing import (
    NetworkDataProvider,
)


class NetworkXDataProvider(NetworkDataProvider):
    @staticmethod
    def check_dependencies() -> bool:
        return importlib.util.find_spec("networkx") is not None

    @staticmethod
    def graph_type():
        from networkx import Graph

        return Graph

    def number_of_vertices(self):
        """The number of vertices/nodes in the network."""
        return self.network.number_of_nodes()

    def number_of_edges(self):
        """The number of edges in the network."""
        return self.network.number_of_edges()

    def get_shortest_distance(self, weight=None) -> pd.Series:
        """Get shortest distances between nodes."""
        import networkx as nx

        n = self.number_of_vertices()
        index = self.vertices()
        tmp = pd.Series(np.arange(n), index=index)
        matrix = np.zeros((n, n), np.float64)
        for id_source, distd in nx.shortest_path_length(self.network, weight=weight):
            idx_source = tmp[id_source]
            for id_tgt, d in distd.items():
                idx_tgt = tmp[id_tgt]
                matrix[idx_source, idx_tgt] = d

        return dict(
            matrix=matrix,
            index=index,
        )

    def vertices(self) -> Sequence:
        """Get a list of vertices."""
        return list(self.network.nodes())

    def edges(self) -> Sequence:
        """Get a list of edges."""
        return list(self.network.edges())

    def adjacency_matrix(self, weights=None) -> np.ndarray:
        """Get the adjacency matrix as a numpy array."""
        import networkx as nx

        return nx.to_numpy_array(self.network, weight=weights)

    def bipartite(self) -> tuple[set]:
        """Get a bipartite split from a bipartite graph."""
        import networkx as nx

        return nx.bipartite.sets(self.network)

    def degrees(self) -> pd.Series:
        """Get the degrees of all vertices."""
        return pd.Series(dict(self.network.degree()), name="degree")

    def bfs(
        self,
        root_idx: Optional[int] = None,
        root: Optional[Hashable] = None,
    ) -> dict[str, np.ndarray[int]]:
        """Get a breadth-first search spanning tree of the network.

        Parameters:
            root_idx: The index of the root node to start the spanning tree from.
            root: The root node to start the spanning tree from. Either this or the "root_idx" parameter must be provided.

        Returns:
            A dictionary with three keys:
                - layer_switch: A list of indices where the layer changes.
                - vertices: A list of vertex indices - as per self.vertices() - in the order they were visited.
                - parents: A list of parent vertex indices - as per self.vertices() - for each vertex in the order they were visited.

        """
        import networkx as nx

        vertex_series = np.Series(
            np.arange(self.number_of_vertices()),
            index=self.vertices(),
        )

        if root is None:
            root = vertex_series.index[root_idx]

        vertices = []
        layer_switch = []
        parents = []
        for i, (parent, child) in enumerate(nx.generic_bfs_edges(self.network, root)):
            if len(vertices) == 0:
                layer_switch.append(0)
                vertices.append(vertex_series[parent])
                parents.append(-1)
            if parent not in parents:
                layer_switch.append(i + 1)
            vertices.append(vertex_series[child])
            parents.append(vertex_series[parent])

        return {
            "vertices": np.array(vertices),
            "parents": np.array(parents),
            "layer_switch": np.array(layer_switch),
        }
