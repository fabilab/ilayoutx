from typing import (
    Optional,
)
import numpy as np
import pandas as pd

from ilayoutx._ilayoutx import (
    bipartite as bipartite_rust,
)
from ..ingest import data_providers, network_library


def bipartite(
    network,
    first: Optional[set | list | tuple | frozenset | np.ndarray | pd.Index] = None,
    distance: float = 1.0,
    theta: float = 0.0,
):
    """Line layout.

    Parameters:
        network: The network to layout.
        theta: The angle of the line in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if first is None:
        first, second = provider.bipartite()
        index = list(first) + list(second)
    else:
        index = provider.vertices()
        first = list(first)
        second = [x for x in index if x not in first]

    n1 = len(first)
    n2 = len(second)
    coords = bipartite_rust(n1, n2, distance, theta)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
