"""Layered or Sugiyama layout algorithm for directed graphs.

The algorithm has four key steps:
1. Remove cycles to make the graph acyclic.
2. Assign nodes to layers based on their dependencies.
3. Reduce edge crossings by reordering nodes within layers.
4. Minimize edge lengths by adjusting node coordinate within layers.

"""

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


def feedback_arc_set_eades(network, matrix):
    """Find a layering of the directed acyclic graph using Eades' algorithm.

    Parameters:
        network: The network to layout.
        provider: The data provider for the network.
    Returns:
        A pandas Series mapping each vertex to its layer.
    """
    from collections import deque

    # Use the adjacency matrix to find sinks and sources
    nv = len(matrix)

    # Initial sources and sinks
    outdeg = matrix.sum(axis=1)
    indeg = matrix.sum(axis=0)
    # FIXME: use actual weights if available
    out_strength = outdeg.astype(float)
    in_strength = indeg.astype(float)
    src_deq = deque(np.flatnonzero(indeg == 0))
    sink_deq = deque(np.flatnonzero(outdeg == 0))

    # The ranks of the nodes, filled in from left (sources) and right (sinks)
    ranks = np.empty(nv, dtype=np.int64)
    rank_left = 0
    rank_right = nv - 1

    # Process nodes from the safe ones (sources and sinks)
    # towards the center of the graph iteratively. Each iteration we
    # scrape off the last layers of source/sink until there are no nodes left
    nodes_left = nv
    while nodes_left:
        # Process sources (safe from left)
        while len(src_deq) > 0:
            idx = src_deq.popleft()
            ranks[idx] = rank_left
            rank_left += 1
            indeg[idx] = outdeg[idx] = -1  # Mark as processed
            nodes_left -= 1

            # Modify out-neighbors and possibly add to the queue
            neis = np.flatnonzero(matrix[idx] != 0)
            for nei in neis:
                if indeg[nei] < 0:
                    # Already processed
                    continue
                indeg[nei] -= 1
                in_strength[nei] -= 1.0
                if indeg[nei] == 0:
                    src_deq.append(nei)

        # Process sinks (safe from right)
        while len(sink_deq) > 0:
            idx = sink_deq.popleft()
            # Already processed
            if indeg[idx] < 0:
                continue
            ranks[idx] = rank_right
            rank_right -= 1
            indeg[idx] = outdeg[idx] = -1  # Mark as processed
            nodes_left -= 1

            # Modify in-neighbors and possibly add to the queue
            neis = np.flatnonzero(matrix[:, idx] != 0)
            for nei in neis:
                if outdeg[nei] < 0:
                    # Already processed
                    continue
                outdeg[nei] -= 1
                out_strength[nei] -= 1.0
                if outdeg[nei] == 0:
                    src_deq.append(nei)

        # At this stage no sources or sinks are left
        # Choose the node the looks most like a source and add it to the
        # left, then build the sources and sinks around it.
        strength_diff = out_strength - in_strength
        strength_diff[indeg < 0] = -np.inf
        idx = np.argmax(strength_diff)
        # Maybe they are all procssed already
        if strength_diff[idx] > -np.inf:
            ranks[idx] = rank_left
            rank_left += 1
            indeg[idx] = outdeg[idx] = -1  # Mark as processed
            nodes_left -= 1

            # Modify/add both outgoing and incoming neighbors since it's
            # neither a pure source nor a pure sink
            neis = np.flatnonzero(matrix[idx] != 0)
            for nei in neis:
                if indeg[nei] < 0:
                    # Already processed
                    continue
                indeg[nei] -= 1
                in_strength[nei] -= 1.0
                if indeg[nei] == 0:
                    src_deq.append(nei)

            # Modify in-neighbors and possibly add to the queue
            neis = np.flatnonzero(matrix[:, idx] != 0)
            for nei in neis:
                if outdeg[nei] < 0:
                    # Already processed
                    continue
                outdeg[nei] -= 1
                out_strength[nei] -= 1.0
                if outdeg[nei] == 0:
                    src_deq.append(nei)

    # Now we have an ranks, assign layers
    # Reverse index mapping to ranks: ranks[0] is the index of the first source
    idx_ordered = np.argsort(ranks)
    layers = np.zeros(nv, dtype=np.int64)
    for idx in idx_ordered:
        neis = np.flatnonzero(matrix[idx] != 0)
        for nei in neis:
            # Skip looops
            if nei == idx:
                continue
            # Skip back edges
            if ranks[idx] > ranks[nei]:
                continue
            # Take the forward edge
            layers[nei] = max(layers[nei], layers[idx] + 1)

    return layers


def _set_random_x_values(matrix, yvalues):
    from collections import Counter, defaultdict

    tmp = Counter()
    xvalues = np.zeros_like(yvalues)
    for i, y in enumerate(yvalues):
        xvalues[i] = tmp[y]
        tmp[y] += 1

    waypoints = defaultdict(list)
    edge_sources, edge_targets = np.nonzero(matrix)
    dys = yvalues[edge_targets] - yvalues[edge_sources]

    # Skip-layer edges (waypoints)
    skip_pos = np.flatnonzero((dys > 1) | (dys < -1))
    for src, tgt in zip(edge_sources[skip_pos], edge_targets[skip_pos]):
        y_src = yvalues[src]
        y_tgt = yvalues[tgt]
        # Iterator changes based on direction (we need space for back edges too)
        yrange = range(y_src + 1, y_tgt) if y_tgt > y_src else range(y_src - 1, y_tgt, -1)
        for y in yrange:
            x = tmp[y]
            tmp[y] += 1
            waypoints[(src, tgt)].append((x, y))

    return xvalues, waypoints


def _compute_barycenters(coords, matrix, idx_layer, direction):
    """Compute barycenters for nodes in a given layer.

    Parameters:
        coords: The coordinates of the nodes.
        matrix: The adjacency matrix of the graph.
        idx_layer: The indices of the nodes in the layer.
        direction: "in" to compute based on incoming edges, "out" for outgoing.
    Returns:
        The barycenters of the nodes in the layer.

    NOTE: Barycenters are the average x-coordinate of neighboring nodes in the
    chosen direction. If no neighbors of that direction exist, the node's own x-coordinate.
    """
    barys = np.zeros(idx_layer.shape[0], dtype=np.float64)
    for iwl, i in enumerate(idx_layer):
        if direction == "in":
            neis = np.flatnonzero(matrix[:, i] != 0)
        else:
            neis = np.flatnonzero(matrix[i, :] != 0)
        if len(neis) == 0:
            barys[iwl] = coords[i, 0]
        else:
            barys[iwl] = coords[neis, 0].mean()
    return barys


def _minimise_edge_crossings(coords, matrix, maxiter=10):
    """Swap x coordinates within layers to reduce edge crossings.


    NOTE: All edges are between consecutive layers due to ghost nodes for waypoints.
    """

    nlayers = coords[:, 1].max() + 1
    if nlayers < 2:
        # No need to do anything for 1 layer
        return coords[:, 0]

    for niter in range(maxiter):
        print("Minimising crossings, iteration", niter + 1)
        changed = False

        # Sort by uppper barycenter, from second layer to last
        for i in range(1, nlayers):
            idx_layer = np.flatnonzero(coords[:, 1] == i)
            nlayer = len(idx_layer)
            barys = _compute_barycenters(coords, matrix, idx_layer, direction="in")
            idx_sorted = np.argsort(barys)
            idx_sorted = idx_layer[idx_sorted]
            if (coords[idx_sorted, 0] != np.arange(nlayer)).any():
                print(coords[idx_sorted, 0], np.arange(nlayer))
                print("Changed upper barys")
                changed = True
            coords[idx_sorted, 0] = np.arange(nlayer)

        # Sort by lower barycenter, from second last layer to first
        for i in range(nlayers - 2, -1, -1):
            idx_layer = np.flatnonzero(coords[:, 1] == i)
            nlayer = len(idx_layer)
            barys = _compute_barycenters(coords, matrix, idx_layer, direction="out")
            idx_sorted = np.argsort(barys)
            idx_sorted = idx_layer[idx_sorted]
            if (coords[idx_sorted, 0] != np.arange(nlayer)).any():
                print(coords[idx_sorted, 0], np.arange(nlayer))
                print("Changed lower barys")
                changed = True
            coords[idx_sorted, 0] = np.arange(nlayer)

        if not changed:
            break

    return coords[:, 0]


def _to_extended_graph(coords, matrix, waypoints):
    """Create an extended graph with ghost nodes for waypoints.

    Parameters:
        coords: The coordinates of the original nodes.
        matrix: The adjacency matrix of the original graph.
        waypoints: The waypoints for skip-layer edges.
    Returns:
        A pair with the extended coordinates and extended adjacency matrix.
    """
    ncoords = len(coords)
    nwaypoints = sum(len(x) for x in waypoints.values())
    coords_ext = np.zeros((ncoords + nwaypoints, 2), dtype=coords.dtype)
    coords_ext[:ncoords, :] = coords
    matrix_ext = np.zeros((ncoords + nwaypoints, ncoords + nwaypoints), dtype=matrix.dtype)
    matrix_ext[:ncoords, :ncoords] = matrix
    iw = ncoords
    for (src, tgt), wp_list in waypoints.items():
        # Eliminate multi-layer edge
        matrix_ext[src, tgt] -= 1
        # Set single-layer edgges via waypoints
        matrix_ext[src, iw] = 1
        for i, (x, y) in enumerate(wp_list):
            coords_ext[iw, 0] = x
            coords_ext[iw, 1] = y
            if i == len(wp_list) - 1:
                itgt = tgt
            else:
                itgt = iw + 1
            matrix_ext[iw, itgt] = 1
            iw += 1

    return coords_ext, matrix_ext


def _from_extended_graph(coords_ext, matrix_ext, ncoords):
    """Create original coordinates and waypoints from extended graph."""
    coords = coords_ext[:ncoords, :]
    waypoints = {}
    src, tgt = None, None
    nlist = []
    for iw in range(ncoords, len(coords_ext)):
        # For the first waypoint of each list, also set the source
        if src is None:
            src_tmp = np.flatnonzero(matrix_ext[:, iw] != 0)
            if len(src_tmp) != 1:
                raise ValueError("Extended graph malformed, waypoint has multiple sources.")
            src = src_tmp[0]
        tgt_tmp = np.flatnonzero(matrix_ext[iw, :] != 0)
        if len(tgt_tmp) != 1:
            raise ValueError("Extended graph malformed, waypoint has multiple targets.")
        tgt_tmp = tgt_tmp[0]
        nlist.append((coords_ext[iw, 0], coords_ext[iw, 1]))
        # Target found, close off waypoint list
        if tgt_tmp < ncoords:
            tgt = tgt_tmp
            waypoints[(src, tgt)] = nlist
            nlist = []
            src, tgt = None, None

    return coords, waypoints


def sugiyama(
    network,
    theta: float = 0.0,
    center: Optional[tuple[float, float]] = (0, 0),
    maxiter_crossing: int = 100,
):
    """Sugiyama or layered layout for directed graphs.

    Parameters:
        network: The network to layout.
        theta: Angle in radians to rotate the layout.
        center: The center of the layout.
    Returns:
        The layout of the network.
    """

    nl = network_library(network)
    provider = data_providers[nl](network)
    if provider.is_directed() is False:
        # NOTE: If one wanted, one could just make a mimnmum spanning tree here.
        # For now, stick to the bare minimum and require directed graphs.
        raise ValueError("Sugiyama layout requires a directed graph.")

    index = provider.vertices()
    nv = provider.number_of_vertices()

    coords = np.zeros((nv, 2), dtype=np.int64)

    # 1. Remove cycles via minimum feedback arc set
    matrix = provider.adjacency_matrix()
    # Ignore loops for computing the layout
    matrix[np.arange(nv), np.arange(nv)] = 0

    # TODO: check that this is correct, seems to work on a few examples for now
    coords[:, 1] = feedback_arc_set_eades(network, matrix)
    coords[:, 0], waypoints = _set_random_x_values(matrix, coords[:, 1])

    coords_ext, matrix_ext = _to_extended_graph(coords, matrix, waypoints)

    coords_ext[:, 0] = _minimise_edge_crossings(coords_ext, matrix_ext, maxiter=maxiter_crossing)

    coords, waypoints = _from_extended_graph(coords_ext, matrix_ext, nv)

    coords = coords.astype(np.float64)

    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    coords = coords @ rotation_matrix

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout, waypoints
