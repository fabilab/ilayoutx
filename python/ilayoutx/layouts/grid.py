import numpy as np

from ilayoutx._ilayoutx import (
    grid_square as grid_square_rust,
    grid_triangle as grid_triangle_rust,
)
from ..ingest import data_providers, network_library


def grid(
    network,
    width: int,
    shape: str = "square",
) -> np.ndarray:
    """A grid layout with specified width.

    Parameters:
        network: The network to layout.
        width: The width of the grid.
        shape: The shape of the grid, either 'square' or 'triangle'.
    """

    # 0. Find what we have
    nl = network_library(network)
    provider = data_providers[nl](network)

    # 1. Get number of vertices
    nv = provider.number_of_vertices()

    # 2. Make the coordinates
    if shape == "triangle":
        coords = grid_triangle_rust(nv, width)
    elif shape == "square":
        coords = grid_square_rust(nv, width)
    else:
        raise ValueError(
            f"Grid shape must be 'square' or 'triangular', not '{shape}'.",
        )

    return coords
