"""Graph Embedder (GEM) layout algorithm.

Although the code in this module was written from scratch, the algorithm was learned from the igraph implementation.
Kind thanks to the igraph developers for their efforts.
"""

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


def _graph_embedder_optimisation(
    coords: np.ndarray,
    adjacency: np.ndarray,
    degrees: np.ndarray,
    etol: float,
    max_iter: Optional[int] = None,
    temp_max: Optional[float] = None,
    temp_min: float = 0.1,
) -> None:
    """Optimissation routine for GEM layout.

    Parameters:
        adjacency: Adjacency matrix of the graph.
        coords: Initial coordinates of the nodes. Will be modified in place.
        etol: Gradient sum of spring forces must be larger than etol before successful termination.
        max_iter: Max iterations before termination of the algorithm.
    Returns:
        None. The coords array is modified in place.
    """

    nv = len(coords)

    if max_iter is None:
        # Each iteration we move a single node, hence unless the user knows better, we should
        # ensure that max_iter >> nv.
        max_iter = nv * nv * 40
    if temp_max is None:
        temp_max = 1.0 * nv

    # Track dynamic center of layout
    layout_center = coords.mean(axis=0)

    # Attractor strength towards the center
    gamma_attr = 1.0 / 16.0

    # Repulsive strength from other nodes
    gamma_rep = 128 * 128

    # Impulse
    impulses_last = np.zeros((nv, 2), dtype=np.float64)
    imp = np.zeros(2)

    # Node degrees and Φ function
    phi = degrees.astype(np.float64)
    phi *= 1 + 0.5 * phi

    # Per-node temperature
    temp = np.sqrt(nv) * np.ones(nv)
    skew_gauge = np.zeros(nv)

    # Global temperature
    temp_global_nv = nv**1.5

    # Heuristic parameters
    sin_threshold = np.sin(2 * np.pi / 3)
    sigma_r = 0.5 / nv
    sigma_o = 1.0 / 3

    # Iterations
    node_order = np.arange(nv)
    np.random.shuffle(node_order)
    node_order_idx = 0
    while (temp_min * nv < temp_global_nv) and (max_iter > 0):
        print("inner loop, max_iter =", max_iter)
        # We move only one vertex at each iteration at random
        # Each time we run out of nodes, we reshuffle into a
        # different order / permutation.
        #
        # NOTE: This only really works if max_iter >> nv, which
        # is why the default max_iter is O(nv^2).
        if node_order_idx >= nv:
            np.random.shuffle(node_order)
            node_order_idx = 0

        # Choose the node
        node_idx = node_order[node_order_idx]

        # Node impulse
        # 1. Attraction to the center
        imp[:] = (layout_center - coords[node_idx]) * gamma_attr * phi[node_idx]
        # 2. Add some noise
        imp += 32 * (2 * np.random.rand(2) - 1)
        # 3. Repulsion from all other nodes
        delta_coords = coords - coords[node_idx]
        delta_dist2 = (delta_coords**2).sum(axis=1)
        # NOTE: Exclude self-repulsion and any other nodes that are straight on top of the node
        idx_nonoverlap = delta_dist2 > 0
        delta_coords = delta_coords[idx_nonoverlap]
        delta_dist2 = delta_dist2[idx_nonoverlap]
        imp -= gamma_rep * (delta_coords / delta_dist2[:, None]).sum(axis=0)
        # 4. Attraction to adjacent nodes (with an edge)
        adj_vector = adjacency[node_idx, idx_nonoverlap]
        imp += (
            (adj_vector[:, None] * delta_coords * delta_dist2[:, None]).sum(axis=0)
            / gamma_rep
            * phi[node_idx]
        )

        # Move node
        if imp.any():
            # Normalize the impulse vector to local temperature
            imp *= temp[node_idx] / np.linalg.norm(imp)
            coords[node_idx] += imp
            layout_center += imp / nv

        # Update node temperature (max displacement)
        imp_last = impulses_last[node_idx]
        if imp_last.any():
            # Check whether the node keeps moving in a straight line or not
            beta = np.arctan2(imp_last[1] - imp[1], imp_last[0] - imp[0])
            sinb, cosb = np.sin(beta), np.cos(beta)
            temp_old = temp[node_idx]

            # FIXME: this looks broken?
            if sinb > sin_threshold:
                skew_gauge[node_idx] += np.sign(sinb) * sigma_r

            # FIXME: this looks broken?
            if abs(cosb) >= 0:
                temp[node_idx] *= sigma_o * cosb

            temp[node_idx] *= 1 - abs(skew_gauge[node_idx])
            if temp[node_idx] > temp_max:
                temp[node_idx] = temp_max

            impulses_last[node_idx] = imp
            temp_global_nv += temp[node_idx] - temp_old

        # Housekeeping
        node_order_idx += 1
        max_iter -= 1


def graph_embedder(
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
    seed: Optional[int] = None,
    inplace: bool = True,
):
    """Graph embedder (GEM).

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
    """

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
            inplace=inplace,
        )

        # TODO: allow weights
        adjacency = provider.adjacency_matrix()
        degrees = provider.degrees()

        coords = initial_coords

        _graph_embedder_optimisation(
            coords,
            adjacency,
            degrees,
            etol,
            max_iter,
        )

    if center is not None:
        coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
