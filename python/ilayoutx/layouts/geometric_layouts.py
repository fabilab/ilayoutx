"""Geometric layout when some edge lengths are known.

The idea behind this layout is taken with permission from netgraph:

https://github.com/paulbrodersen/netgraph/blob/8e4da50408a84fca8bc21dad4a8fb933b7d6907c/netgraph/_node_layout.py#L1680

The algorithm itself is similar but not identical.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import pandas as pd


from ..ingest import data_providers, network_library


DEBUG_GEOM = True


def geometric(
    network,
    edge_lengths: dict[tuple | int, float],
    tol: float = 1e-7,
    spread_strength: float = 0.01,
    regulariser_strength: float = 0.0,
) -> pd.DataFrame:
    """Geometric layout.

    Parameters:
        network: The network to layout.
        edge_lengths: A dictionary with edge lengths. The keys can be
            either edge tuples (u, v) or edge IDs (int).
        tol: Tolerance for the optimization.
        spread_strength: Multiplicative factor for the penalty term if
            nodes are too close together.
        regulariser_strength: Multiplicative factor for the regulariser
            that keeps nodes close to the origin.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=float)

    # Set of fixed distances
    nodes_ser = pd.Series(np.arange(nv), index=provider.vertices())
    esource = []
    etarget = []
    edist = []
    for (source, target), dist in edge_lengths.items():
        esource.append(nodes_ser.at[source])
        etarget.append(nodes_ser.at[target])
        edist.append(dist)
    esource = np.array(esource, dtype=int)
    etarget = np.array(etarget, dtype=int)
    edist = np.array(edist, dtype=float)

    coords = np.random.rand(nv, 2)

    # For an extended discussion of this cost function and alternatives see:
    # https://stackoverflow.com/q/75137677/2912349

    tracks = {
        "dist_nodes": [],
        "diff": [],
        "reg": [],
    }

    def _cost_function(positions):
        positions = positions.reshape((nv, 2))

        # Distance between all pairs of nodes
        dist_nodes = 2 * pdist(positions).sum()
        # Distances that are already fixed
        dist_set = squareform(pdist(positions))
        dist_set = dist_set[esource, etarget]
        # FIXME: Debugging output
        if DEBUG_GEOM:
            print(dist_set)
            print(edist)
        diff = (np.abs(dist_set - edist)).sum()
        # Regulariser (scaled to work with dist_nodes)
        reg = 0.5 * np.abs(positions).sum() * (nv - 1)
        # print(dist_nodes, diff, reg)
        tracks["dist_nodes"].append(dist_nodes)
        tracks["diff"].append(diff)
        tracks["reg"].append(reg)
        return diff - spread_strength * dist_nodes + regulariser_strength * reg

    result = minimize(
        _cost_function,
        coords.ravel(),
        tol=tol,
    )

    if DEBUG_GEOM:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(tracks["dist_nodes"], label="dist_nodes")
        ax.plot(tracks["diff"], label="diff")
        ax.plot(tracks["reg"], label="reg")
        ax.legend()
        fig.tight_layout()
        plt.ion()
        plt.show()

    coords_new = result.x.reshape((nv, 2))

    layout = pd.DataFrame(coords_new, index=nodes_ser.index, columns=["x", "y"])

    return layout
