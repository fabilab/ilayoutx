from typing import (
    Optional,
)
from collections.abc import (
    Hashable,
)
import numpy as np
import pandas as pd

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import _format_initial_coords
from ..external.networkx.arf import (
    arf_networkx,
)
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def arf(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    scaling: Optional[float] = 1.0,
    center: Optional[tuple[float, float]] = None,
    spring_strength: float = 1.1,
    etol: float = 1e-6,
    dt: float = 1e-3,
    max_iter: int = 1000,
    seed: Optional[int] = None,
):
    """Attractive-repulsive forces layout algorithm.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes.
        scaling: Strength of the repulsive forces. Larger values spread the nodes further apart.
        center: If not None, recenter the layout around this point.
        spring_strength: Strength of springs between connected nodes. Should be larger than 1.
        etol: Gradient sum of spring forces must be larger than etol before successful termination.
        dt: Time step for force differential equation simulations.
        max_iter: Max iterations before termination of the algorithm.
        seed: A random seed to use.
    Returns:
        The layout of the network.

    NOTE: This layout computed all mutual distances between nodes, which scales with O(n^2). On a
    laptop as an example, this works until around 1,000 nodes, after which numpy.linalg starts
    throwing overflow errors.
    """

    if spring_strength <= 1:
        raise ValueError("spring_strength should be larger than 1.")

    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: random_rust(nv, seed=seed),
        )

        edges = provider.edges()

        # NOTE: the output is inserted in place into initial_coords
        arf_networkx(
            nv,
            index,
            edges,
            pos=initial_coords,
            scaling=scaling,
            a=spring_strength,
            etol=etol,
            dt=dt,
            max_iter=max_iter,
        )
        coords = initial_coords

    if center is not None:
        coords += np.array(center, dtype=np.float64) - coords.mean(axis=0)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
