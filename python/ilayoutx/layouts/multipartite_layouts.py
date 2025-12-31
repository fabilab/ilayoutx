from typing import (
    Sequence,
)
from collections.abc import (
    Hashable,
)

import numpy as np
import pandas as pd

from ..ingest import data_providers, network_library


def multipartite(
    network,
    nlist: Sequence[Sequence[Hashable]],
    distance: float = 1.0,
    theta: float = 0.0,
    ycenter: bool = False,
) -> pd.DataFrame:
    """Multipartite layout.

    Parameters:
        network: The network to layout.
        nlist: List of lists of nodes in each layer.
        theta: Rotation angle in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """

    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    nodes = []
    i = 0
    coords = np.zeros((nv, 2), dtype=float)
    for ilayer, nodes_layer in enumerate(nlist):
        nlayer = len(nodes_layer)
        if nlayer == 0:
            continue
        nodes.extend(list(nodes_layer))
        if ycenter:
            offset = -0.5 * (nlayer - 1)
        else:
            offset = 0.0

        coords[i : i + nlayer, 0] = ilayer * distance
        coords[i : i + nlayer, 1] = np.arange(nlayer, dtype=float) + offset
        i += nlayer

    # Rotate coordinates
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    coords = coords @ rotation_matrix

    layout = pd.DataFrame(coords, index=nodes, columns=["x", "y"])
    return layout
