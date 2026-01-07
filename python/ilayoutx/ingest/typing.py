from typing import (
    Optional,
    Sequence,
    Protocol,
)
from collections.abc import Hashable
import numpy as np
import pandas as pd


class NetworkDataProvider(Protocol):
    """A protocol for a network object."""

    def __init__(
        self,
        network,
    ) -> None:
        """Initialise network data provider.

        Parameters:
            network: The network to ingest.
        """
        self.network = network

    @staticmethod
    def check_dependencies():
        """Check whether the dependencies for this provider are installed."""
        raise NotImplementedError("Network data providers must implement this method.")

    @staticmethod
    def graph_type():
        """Return the graph type from this provider to check for instances."""
        raise NotImplementedError("Network data providers must implement this method.")

    def number_of_vertices(self) -> int:
        """Get the number of nodes in the network."""
        ...

    def number_of_edges(self) -> int:
        """Get the number of edges in the network."""
        ...

    def get_shortest_distance(self) -> pd.Series:
        """Get shortest distances between nodes."""
        ...

    def get_vertices(self) -> list:
        """Get a list of vertices."""
        ...

    def degrees(self) -> pd.Series:
        """Get the degrees of all vertices."""
        ...

    def bfs(
        self,
        root_idx: Optional[int] = None,
        root: Optional[Hashable] = None,
    ) -> dict[str, Sequence[Hashable]]:
        """Get a minimum spanning of the graph."""
        ...

    def component_memberships(self, mode) -> np.ndarray:
        """Get the component memberships of all vertices."""
        ...

    def is_connected(self) -> bool:
        """Check whether the graph is connected."""
        memberships_unique = np.unique(self.component_memberships(mode="weak"))
        return len(memberships_unique) <= 1
