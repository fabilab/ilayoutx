from typing import (
    Optional,
    Sequence,
    Mapping,
)
from collections.abc import (
    Hashable,
)
import numpy as np
from scipy.optimize import (
    curve_fit,
    root_scalar,
)
import pandas as pd

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import _format_initial_coords
from ilayoutx._ilayoutx import (
    random as random_rust,
)


class SugiyamaResult(tuple):
    """Pair-like class with a result of the Sugiyama layout algorithm.

    In addition to obj[0] and obj[1], it provides two properties:
        - obj.layout: Returns the layout DataFrame.
        - obj.waypoints: Returns the waypoints DataFrame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 2:
            raise ValueError(
                "SugiyamaResult must contain exactly two DataFrames: layout and waypoints.",
            )

    @property
    def layout(self) -> pd.DataFrame:
        """Returns the layout DataFrame."""
        return self[0]

    @property
    def waypoints(self) -> pd.DataFrame:
        """Returns the waypoints DataFrame."""
        return self[1]


def sugiyama(
    network,
    layers: Sequence[int] | Mapping[Hashable, int],
    root_coords: Optional[tuple[float, float]] = (0, 0),
    hgap: float = 0.1,
    vgap: float = 1.0,
    max_iter: int = 1000,
) -> SugiyamaResult[pd.DataFrame, pd.DataFrame]:
    """Sugiyama-style or layered layout for tree-like networks.

    Parameters:
        network: The network to layout.
        layers: A sequence or mapping indicating the layer of each node.
        root_coords: The coordinates of the root node. Subsequent layers will have a
            posiive y coordinate equal to (1 + vgap) * layer_index.
        hgap: The horizontal gap between nodes in the same layer.
        vgap: The vertical gap between layers.
        max_iter: The maximum number of iterations for the layout algorithm.
    Returns:
        An instance of `SugiyamaResult`, which behaves like a tuple, containing two DataFrames:
            layout: A DataFrame with the layout of the vertices.
            waypoints: A DataFrame with the waypoints (ghost nodes) of the edges.
    """

    layout = None
    waypoints = None

    return SugiyamaResult((layout, waypoints))
