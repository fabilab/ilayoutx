from typing import (
    Optional,
    Sequence,
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
    fixed: Optional[Sequence[Hashable]] = None,
    center: tuple[float, float] = (0, 0),
    scale: Optional[float] = 1.0,
    gravity: float = 1.0,
    exponent_attraction: float = 1.0,
    exponent_repulsion: float = -2.0,
    method="force",
    etol: float = 1e-4,
    max_iter: int = 50,
    seed: Optional[int] = None,
):
    """Spring layout (Fruchterman-Reingold).

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes.
        optimal_distance: Optimal distance between nodes. If None, set to sqrt(1/n).
        fixed: Nodes to keep fixed during layout. Can be a list of node IDs of a dict of bools
            where True means to keep the node fixed. If this argument is not None and not empty,
            the following arguments "center" and "scale" are ignored.
        center: Recenter the layout around this point.
        scale: Scaling factor for the layout. The larger of x- and y-ranges will be equal to scale.
        gravity: Gravity force scaling to apply towards the center.
        exponent_attraction: Exponent for the attraction force (1.0 means spring-like attraction).
        exponent_repulsion: Exponent for the repulsion force (-2.0 means gravity-like repulsion).
        method: The method to use. Currently only "force" is supported.
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

    if fixed is not None:
        if len(fixed) == 0:
            fixed = None
        else:
            fixed_bool = pd.Series(np.zeros(nv, dtype=bool), index=index)
            if isinstance(fixed, dict):
                for key, val in fixed.items():
                    if val:
                        fixed_bool.at[key] = True
            else:
                fixed_bool[fixed] = True
            fixed = fixed_bool.values
            del fixed_bool

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    initial_coords = _format_initial_coords(
        initial_coords,
        index=index,
        fallback=lambda: random_rust(nv, seed=seed),
    )
    if ((fixed is not None) and fixed.all()) or (nv == 1):
        coords = initial_coords
    else:
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
            exponent_attraction=exponent_attraction,
            exponent_repulsion=exponent_repulsion,
            fixed=fixed,
        )
        coords = initial_coords

    if fixed is None:
        if center is not None:
            coords += np.array(center, dtype=np.float64) - coords.mean(axis=0)

        if scale is not None:
            current_scale = (coords.max(axis=0) - coords.min(axis=0)).max()
            if current_scale > 0:
                coords *= scale / current_scale

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
