from typing import (
    Optional,
)
from collections.abc import (
    Hashable,
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
from ..external.networkx.spring import (
    _fruchterman_reingold as fruchterman_reingold_networkx,
)
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def spring(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    optimal_distance: Optional[float] = None,
    radius: float = 1.0,
    center: Optional[tuple[float, float]] = (0, 0),
    gravity: float = 1.0,
    method="force",
    etol: float = 1e-4,
    max_iter: int = 50,
    seed: Optional[int] = None,
):
    """ForceAtlas2 algorithm from Gephi.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes.
        jitter_tolerance: Controls the tolerance for adjusting the speed of layout generation.
        scaling_ratio: Determines the scaling of attraction and repulsion forces.
        gravity: Determines the amount of attraction on nodes to the center. Prevents islands
            (i.e. weakly connected or disconnected parts of the graph) from drifting away.
        distributed_action: Distributes the attraction force evenly among nodes.
        strong_gravity: Applies a strong gravitational pull towards the center.
        mass: Maps nodes to their masses, influencing the attraction to other nodes.
        size: Maps nodes to their sizes, preventing crowding by creating a halo effect.
        dissuade_hubs: Prevents the clustering of hub nodes.
        linlog: Uses logarithmic attraction instead of linear.
        etol: Gradient sum of spring forces must be larger than etol before successful termination.
        max_iter: Max iterations before termination of the algorithm.
        seed: A random seed to use.
    Returns:
        The layout of the network.

    NOTE: This layout computed all mutual distances between nodes, which scales with O(n^2). On a
    laptop as an example, this works until around 1,000 nodes, after which numpy.linalg starts
    throwing overflow errors.
    """

    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: random_rust(nv, seed=seed),
        )

        # TODO: allow weights
        adjacency = provider.adjacency_matrix()

        if optimal_distance is None:
            optimal_distance = np.sqrt(1.0 / nv)

        # NOTE: the output is inserted in place into initial_coords
        fruchterman_reingold_networkx(
            A=adjacency,
            k=optimal_distance,
            pos=initial_coords,
            threshold=etol,
            max_iter=max_iter,
            seed=seed,
        )
        coords = initial_coords
        rmax = np.linalg.norm(coords, axis=1).max()
        coords *= radius / rmax

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
