"""Grid layouts."""

from typing import (
    Optional,
)
import numpy as np
import pandas as pd

from .._ilayoutx import (
    grid_square as grid_square_rust,
    grid_triangle as grid_triangle_rust,
)
from ..utils import (
    _recenter_layout,
)
from ..ingest import (
    data_providers,
    network_library,
)


def grid(
    network,
    width: int,
    shape: str = "square",
    trim_even_rows: bool = False,
    center: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """A rectangular or triangular grid layout with specified width.

    Parameters:
        network: The network to layout.
        width: The width of the grid.
        shape: The shape of the grid, either 'square' or 'triangle'.
        trim_even_rows: Only used for triangular lattices. If True, trim the even rows by one
            vertex to fit the width.
        center: If not None, recenter the final layout at this point as a tuple (x, y).
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    index = provider.vertices()
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    if shape == "triangle":
        coords = grid_triangle_rust(nv, width, equal_rows=not trim_even_rows)
    elif shape == "square":
        coords = grid_square_rust(nv, width)
    else:
        raise ValueError(
            f"Grid shape must be 'square' or 'triangular', not '{shape}'.",
        )

    if center is not None:
        _recenter_layout(coords, center)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])

    return layout
