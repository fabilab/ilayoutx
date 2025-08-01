from typing import (
    Protocol,
)
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
