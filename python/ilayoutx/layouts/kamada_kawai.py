from typing import (
    Optional,
)
import numpy as np
import pandas as pd

from ilayoutx._ilayoutx import (
    circle,
)
from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import _format_initial_coords
from ..external.networkx.kamada_kawai import (
    _kamada_kawai_solve,
)


def kamada_kawai(
    network,
    radius: Optional[float] = 1.0,
    center: Optional[tuple[float, float]] = (0, 0),
    initial_coords: Optional[np.ndarray | dict | pd.DataFrame] = None,
) -> pd.DataFrame:
    """Kamada-Kawai layout algorithm.

    Parameters:
        network: The network to layout.
        seed: A random seed to use.
    Returns:
        The layout of the network.
    """

    nl = network_library(network)
    provider = data_providers[nl](network)

    # Compute the distance matrix.
    tmp = provider.get_shortest_distance()
    dist = tmp["matrix"]
    index = tmp["index"]
    nv = len(index)

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    # Get and set largest finite distance.
    # Infinite distance stems from non-connected components.
    dist[np.isinf(dist)] = -1
    # In case they are all singletons, there is no max finite distance.
    dist[dist < 0] = max(dist.max(), 0)

    initial_coords = _format_initial_coords(
        initial_coords,
        index=index,
        fallback=lambda: circle(nv, radius=0.5 * np.sqrt(nv)),
    )

    # Solve the kk optimization problem
    coords = _kamada_kawai_solve(dist, initial_coords, 2)

    # Rescale and center the coordinates
    coords *= radius / np.abs(coords).max()
    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
