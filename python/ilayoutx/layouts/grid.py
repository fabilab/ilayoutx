import numpy as np
import pandas as pd

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
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if shape == "triangle":
        coords = grid_triangle_rust(nv, width)
    elif shape == "square":
        coords = grid_square_rust(nv, width)
    else:
        raise ValueError(
            f"Grid shape must be 'square' or 'triangular', not '{shape}'.",
        )

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout
