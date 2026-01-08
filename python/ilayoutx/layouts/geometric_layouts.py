"""Geometric layout when some edge lengths are known.

The idea behind this layout is taken with permission from netgraph:

https://github.com/paulbrodersen/netgraph/blob/8e4da50408a84fca8bc21dad4a8fb933b7d6907c/netgraph/_node_layout.py#L1680

The algorithm itself is similar but not identical.
"""

from typing import (
    Optional,
)
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import pandas as pd


from ilayoutx.ingest import data_providers, network_library
from ilayoutx.experimental.utils import get_debug_bool
from ilayoutx.layouts import spring
from ilayoutx.external.netgraph.geometric import get_geometric_layout


DEBUG_GEOM = get_debug_bool("ILAYOUTX_DEBUG_GEOM", False)


def geometric(
    network,
    edge_lengths: dict[tuple | int, float],
    tol: float = 1e-7,
    center: Optional[tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Geometric layout.

    Parameters:
        network: The network to layout.
        edge_lengths: A dictionary with edge lengths. The keys can be
            either edge tuples (u, v) or edge IDs (int).
        tol: Tolerance for the optimization.
        center: If not None, recenter the layout around this point.
        seed: A random seed to use.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()
    nodes_ser = pd.Series(np.arange(nv), index=index)

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=float)
    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        edges = [tuple(e) for e in provider.edges()]
        initial_coords = spring(network, seed=seed).values

        # NOTE: The layout is idempotent, i.e. double application
        # is the same as a single pass. Therefore it gets only one
        # shot at getting it right really. How well that works
        # seems to depend quite strongly on the initial layout, i.e.
        # on the random seed. This must be true for netgraph as well.
        # This does not mean the layout is useless, but definitely
        # a little unstable.
        result = get_geometric_layout(
            edges,
            edge_lengths,
            initial_coords,
            tol=tol,
        )
        coords = np.zeros_like(initial_coords, dtype=np.float64)
        for nodeid, pos in result.items():
            coords[nodes_ser[nodeid]] = pos

        if DEBUG_GEOM:
            import matplotlib.pyplot as plt
            import iplotx as ipx

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            ipx.network(network, initial_coords, ax=axs[0], title="Initial", node_labels=True)
            ipx.network(network, coords, ax=axs[1], title="Geometric", node_labels=True)
            plt.ion()
            plt.show()

    if center is not None:
        coords += np.array(center, dtype=np.float64) - coords.mean(axis=0)

    layout = pd.DataFrame(coords, index=nodes_ser.index, columns=["x", "y"])

    return layout
