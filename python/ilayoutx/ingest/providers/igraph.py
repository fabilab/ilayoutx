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


class IGraphDataProvider(NetworkDataProvider):
    @staticmethod
    def check_dependencies() -> bool:
        return importlib.util.find_spec("igraph") is not None

    @staticmethod
    def graph_type():
        import igraph as ig

        return ig.Graph

    def number_of_vertices(self):
        """The number of vertices/nodes in the network."""
        return self.network.vcount()

    def number_of_edges(self):
        """The number of edges in the network."""
        return self.network.ecount()

    def get_shortest_distance(self) -> pd.Series:
        """Get shortest distances between nodes."""
        import igraph as ig

        matrix = self.network.shortest_paths_dijkstra()
        matrix = np.asarray(matrix, dtype=np.float64)
        return dict(
            matrix=matrix,
            index=self.vertices(),
        )

    def vertices(self) -> Sequence:
        """Get a list of vertices."""
        return self.network.vs.indices

    def edges(self) -> Sequence:
        """Get a list of edges."""
        return self.network.get_edgelist()

    def adjacency_matrix(self, weights=None) -> np.ndarray:
        """Get the adjacency matrix as a numpy array."""
        import igraph as ig

        matrix = np.asarray(self.network.get_adjacency())
        if weights is not None:
            edge_indices = np.array(self.edges())
            matrix[edge_indices[:, 0], edge_indices[:, 1]] = weights

        return matrix

    def bipartite(self) -> tuple[set]:
        """Get a bipartit split from a bipartite graph."""
        is_bipartite, vertex_types = self.network.is_bipartite(return_types=True)
        if not is_bipartite:
            raise ValueError("The graph is not bipartite.")
        vertex_types = np.array(vertex_types, bool)
        first = np.flatnonzero(~vertex_types)
        second = np.flatnonzero(vertex_types)
        return first, second

    def degrees(self) -> pd.Series:
        """Get the degrees of all vertices."""
        return pd.Series(self.network.degree())

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

        # For igraph, vertices are integers from 0 upwards anyway
        if root_idx is None:
            root_idx = root

        vertices, layer_switch, parents_unordered = self.network.bfs(root_idx)

        # Reorder the parents in the order of the vertices
        parents = [parents_unordered[v] for v in vertices]

        return {
            "vertices": np.array(vertices),
            "parents": np.array(parents),
            "layer_switch": np.array(layer_switch),
        }
