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

    def vertices(self) -> list:
        """Get a list of vertices."""
        return list(self.network.nodes())

    def bipartite(self) -> tuple[set]:
        """Get a bipartite split from a bipartite graph."""
        import networkx as nx

        return nx.bipartite.sets(self.network)
