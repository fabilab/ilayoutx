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
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def large_graph_layout(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    center: Optional[tuple[float, float]] = (0, 0),
    etol: float = 1e-10,
    max_iter: int = 1000,
    root: Hashable = None,
    seed: Optional[int] = None,
    inplace: bool = True,
):
    """Large graph layout (adapted) for connected networks.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes. See also the "inplace" parameter.
        center: The center of the layout.
        etol: Gradient sum of spring forces must be larger than etol before successful termination.
        max_iter: Max iterations before termination of the algorithm.
        seed: A random seed to use.
        inplace: If True and the initial coordinates are a numpy array of dtype np.float64,
            that array will be recycled for the output and will be changed in place.
    Returns:
        The layout of the network.

    NOTE: This algorithm is not currently working on graphs that are not connected.

    NOTE: The algorithm used herein is an adaptation of the original LGL algorithm as conceived
    by the igraph team in igraph 0.2.
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
        # Parameters
        area = nv**2
        cell_size = np.sqrt(nv)
        max_delta = 1.0 * nv

        if root is None:
            root_idx = np.random.randint(nv)
        else:
            root_idx = index.index(root)

        # Compute minimum spanning tree. For now (as igraph), we ignore weights
        # and therefore any spanning tree is fine.
        # FIXME: ensure we get indices back
        tree_dict = provider.network.bfs(root_idx)
        vertices_bfs = tree_dict["vertices"]
        parents = tree_dict["parents"]
        layer_switch = tree_dict["layer_switch"]

        nlayers = len(layer_switch) - 1
        harmonic_sum = (1.0 / np.linspace(1, nlayers - 1, nlayers - 1)).sum()

        # Default is a random layout scaled by the area of the grid
        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: np.sqrt(area) / np.pi * random_rust(nv, seed=seed),
            inplace=inplace,
        )

        coords = initial_coords

        force = np.zeros((nv, 2), dtype=np.float64)

        # TODO: create grid

        # Place root
        coords[root_idx] = 0.0

        eps = 1e-6
        scon = np.sqrt(area / np.pi) / harmonic_sum

        # Iterate over layers
        for ilayer in range(1, nlayers):
            max_delta = 1.0 + eps

            # 1. Place nodes in layer in a circle
            jprev, jcurr = layer_switch[ilayer - 1], layer_switch[ilayer]
            vertices_layer = vertices_bfs[jprev:jcurr]
            parents_layer = parents[jprev:jcurr]

            impulse = coords[vertices_layer] - coords[parents_layer]
            impulse /= np.linalg.norm(impulse + 1e-10, axis=1)[:, np.newaxis]

            # Add to the layout the next layer (i.e. starting from the children of root when ilayer == 1)
            for vertex_idx, imp in zip(vertices_layer, impulse):
                grid_ctr = coords.mean(axis=0)
                children_idx = vertices_bfs[parents == vertex_idx]
                # Children of the root are spread evenly in a circle,
                # for the later layers use rotationally symmetric random sampling via Gaussians
                if ilayer == 1:
                    thetas = np.linspace(
                        0, 2 * np.pi, len(children_idx), endpoint=False
                    )
                    r = np.zeros((len(children_idx), 2), dtype=np.float64)
                    r[:, 0] = np.cos(thetas)
                    r[:, 1] = np.sin(thetas)
                    del thetas
                else:
                    # NOTE: in igraph, this is uniform. That's wrong as it weighes the diagonals more heavily
                    # (they have longer arms)
                    r = np.random.normal(
                        loc=0.0, scale=1.0, size=(len(children_idx), 2)
                    )
                    r /= np.linalg.norm(r + 1e-10, axis=1)[:, np.newaxis]
                r *= scon / ilayer
                # TODO: add to grid the following
                # grid_ctr + imp + coords[vertex_idx] + r

            # 2. Determine which edges between vertices in this layer and the next are within the same cell
            #    Those are the ones that have an interaction, the rest is too far away
            #    NOTE: This obviously makes a mess when two adjacent nodes are right across a cell boundary,
            #    but the algorithm seems to ignore this problem. I guess it's quantisation, baby.
            edges_samecell = ...

            # 3. Compute forces along those vetted edges
            niter = 1
            maxchange = eps + 1
            # Iterate forces for this one layer
            while (niter <= max_iter) and (maxchange > eps):
                # Learning rate (decreasing 1 -> 0, decelerating)
                eta = ((max_iter - niter) / max_iter) ** 1.5
                t = max_delta * eta

                force[:] = 0
                maxchange = 0.0

                # Attract along vetted, cross-layer edges
                for vertex_idx, child_idx in edges_samecell:
                    ... attract

                # Repel anything within the cell
                for vertex_idx in vertices_layer:
                    ... find nodes nearby
                    ... repel

                # Move the nodes
                coords[...] += force

                # Record max change
                # NOTE: the igraph implementation does not include an absolute value here,
                # that looks kind of sus
                maxchange = max(maxchange, force.max())

            # Housekeeping

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
