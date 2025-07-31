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
            index=self.network.vs.indices,
        )
