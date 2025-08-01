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

    def vertices(self) -> list:
        """Get a list of vertices."""
        return self.network.vs.indices

    def bipartite(self) -> tuple[set]:
        """Get a bipartite split from a bipartite graph."""
        is_bipartite, vertex_types = self.network.is_bipartite(return_types=True)
        if not is_bipartite:
            raise ValueError("The graph is not bipartite.")
        vertex_types = np.array(vertex_types, bool)
        first = np.flatnonzero(~vertex_types)
        second = np.flatnonzero(vertex_types)
        return first, second
