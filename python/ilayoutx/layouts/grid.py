import numpy as np

from ilayoutx._ilayoutx import grid as grid_rust
from ..ingest import data_providers, network_library


def grid(
    network,
    width: int,
) -> np.ndarray:
    """A grid layout with specified width.

    Parameters:
        network: The network to layout.
        width: The width of the grid.
    """

    # 0. Find what we have
    nl = network_library(network)
    provider = data_providers[nl](network)

    # 1. Get number of vertices
    nv = provider.number_of_vertices()

    # 2. Make the coordinates
    coords = grid_rust(nv, width)

    return coords
