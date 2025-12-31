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


# Debug flag, this is a pretty long algorithm
DEBUG_SUGI = __import__("os").getenv("ILAYOUTX_DEBUG_SUGI", "True") in (
    "True",
    "true",
    "y",
    "1",
    "yes",
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
            barys[iwl] = coords["x"][i]
        else:
            barys[iwl] = coords["x"][neis].mean()
    return barys


def _minimise_edge_crossings(coords, matrix, maxiter=10):
    """Swap x coordinates within layers to reduce edge crossings.


    NOTE: All edges are between consecutive layers due to ghost nodes for waypoints.
    """

    nlayers = coords["y"].max() + 1
    if nlayers < 2:
        # No need to do anything for 1 layer
        return coords["x"]

    for niter in range(maxiter):
        if DEBUG_SUGI:
            print("Minimising crossings, iteration", niter + 1)
        changed = False

        # Sort by uppper barycenter, from second layer to last
        for i in range(1, nlayers):
            idx_layer = np.flatnonzero(coords["y"] == i)
            nlayer = len(idx_layer)
            barys = _compute_barycenters(coords, matrix, idx_layer, direction="in")
            idx_sorted = np.argsort(barys)
            idx_sorted = idx_layer[idx_sorted]
            if (coords["x"][idx_sorted] != np.arange(nlayer)).any():
                if DEBUG_SUGI:
                    print(coords["x"][idx_sorted], np.arange(nlayer))
                    print("Changed upper barys")
                changed = True
            coords["x"][idx_sorted] = np.arange(nlayer)

        # Sort by lower barycenter, from second last layer to first
        for i in range(nlayers - 2, -1, -1):
            idx_layer = np.flatnonzero(coords["y"] == i)
            nlayer = len(idx_layer)
            barys = _compute_barycenters(coords, matrix, idx_layer, direction="out")
            idx_sorted = np.argsort(barys)
            idx_sorted = idx_layer[idx_sorted]
            if (coords["x"][idx_sorted] != np.arange(nlayer)).any():
                if DEBUG_SUGI:
                    print(coords["x"][idx_sorted], np.arange(nlayer))
                    print("Changed lower barys")
                changed = True
            coords["x"][idx_sorted] = np.arange(nlayer)

        if not changed:
            break

    return coords["x"]


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
    coords_ext = np.zeros(ncoords + nwaypoints, dtype=coords.dtype)
    coords_ext[:ncoords] = coords
    matrix_ext = np.zeros((ncoords + nwaypoints, ncoords + nwaypoints), dtype=matrix.dtype)
    matrix_ext[:ncoords, :ncoords] = matrix
    iw = ncoords
    for (src, tgt), wp_list in waypoints.items():
        # Eliminate multi-layer edge
        matrix_ext[src, tgt] -= 1
        # Set single-layer edgges via waypoints
        matrix_ext[src, iw] = 1
        for i, (x, y) in enumerate(wp_list):
            coords_ext["x"][iw] = x
            coords_ext["y"][iw] = y
            if i == len(wp_list) - 1:
                itgt = tgt
            else:
                itgt = iw + 1
            matrix_ext[iw, itgt] = 1
            iw += 1

    return coords_ext, matrix_ext


def _from_extended_graph(coords_ext, matrix_ext, ncoords):
    """Create original coordinates and waypoints from extended graph."""
    coords = coords_ext[:ncoords]
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
        nlist.append((coords_ext["x"][iw], coords_ext["y"][iw]))
        # Target found, close off waypoint list
        if tgt_tmp < ncoords:
            tgt = tgt_tmp
            waypoints[(src, tgt)] = nlist
            nlist = []
            src, tgt = None, None

    return coords, waypoints


def _make_one_alignment(coords_ext, matrix_ext, ignored_edges, align_left, align_top, nlayers):
    """Find roots and alignments in one of the four corners.

    Parameters:
        coords_ext: The coordinates of the extended graph.
        matrix_ext: The adjacency matrix of the extended graph.
        ignored_edges: The edges to ignore due to type 1 conflicts.
        align_left: Whether to align to the left (True) or right (False).
        align_top: Whether to align to the top (True) or bottom (False).
        nlayers: The number of layers in the graph (for convenience).
    Returns:
        Two arrays with the root and aligns indices for each node.
    """
    nv = len(coords_ext)
    roots = np.arange(nv)
    align = np.arange(nv)

    # I think we want to iterate over all nodes from the alignment corner outwards.
    # For top left as an example, we want to go from layer 0 (roots) to layer 1
    # by setting those roots, then down to the bottom. Within each layer, we
    # go left to right, setting alignments as we go.
    if align_left and align_top:
        idx_sorted = np.lexsort((coords_ext["x"], coords_ext["y"]))
    elif align_left:
        idx_sorted = np.lexsort((coords_ext["x"], -coords_ext["y"]))
    elif align_top:
        idx_sorted = np.lexsort((-coords_ext["x"], coords_ext["y"]))
    else:
        idx_sorted = np.lexsort((-coords_ext["x"], -coords_ext["y"]))

    yold = -1
    for idx in idx_sorted:
        y = coords_ext["y"][idx]

        # New layer
        if y != yold:
            yold = y
            # Store the extreme x coordinate we've aligned to so far
            r = None

        # Propagate information from previous layer
        if align_top:
            idx_neis = np.flatnonzero(matrix_ext[:, idx] != 0)
        else:
            idx_neis = np.flatnonzero(matrix_ext[idx, :] != 0)

        nneis = len(idx_neis)

        # Nothing to propagate (e.g. first node in first layer)
        if nneis == 0:
            continue

        # Trickle down the median(s) of the upper layer neighbors to this node
        if nneis == 1:
            idx_medians = [idx_neis[0]]
        else:
            idx_neis_sorted = np.argsort(coords_ext["x"][idx_neis])
            # Odd number of neighbors, take the central one
            if nneis % 2 == 1:
                idx_medians = [idx_neis[idx_neis_sorted[nneis // 2]]]
            # Even number of neighbors, take both central ones and
            # then there's logic to deal with the mess depending on what kind of
            # conflicts we have
            idx_medians = idx_neis[idx_neis_sorted[nneis // 2 - 1 : nneis // 2 + 1]]
            if not align_left:
                idx_medians = idx_medians[::-1]

        if DEBUG_SUGI:
            print("Layer", y, "node", idx, "neighbors:", idx_neis, "medians:", idx_medians)

        # If two medians found, choose which one to take (or a mixture) based
        # on inner edge preference and left/right alignment (left alignments
        # always try the left one first, right alignments the right one first)
        for idx_median in idx_medians:
            # FIXME: The idea is that if an edge was ignored due to type 1 conflict,
            # we can ignore this vertex. I'm actually not 100% sure this covers all cases
            if (align_top and ((idx_median, idx) in ignored_edges)) or (
                (not align_top) and ((idx, idx_median) in ignored_edges)
            ):
                if DEBUG_SUGI:
                    print("Ignored median: type 1 conflict")
                continue

            xmed = coords_ext["x"][idx_median]
            if (r is None) or (align_left and xmed > r) or (not align_left and xmed < r):
                # Anchor the upper layer alignment to the evolving front
                align[idx_median] = idx
                # Anchor the evolving front's roots to the top of the DAG
                roots[idx] = roots[idx_median]
                # FIXME: Back-anchor the alignment of the front to the root, unless
                # this is used as a feeding "median" neighbor in the next layer?
                align[idx] = roots[idx_median]
                r = xmed
                break

    return roots, align


# NOTE: This function is recursive and changes all kinds of things (e.g. the sinks)
# in place, no return. So be very careful before trying to optimise it out.
def _place_block(idx, idx_vertex_left, roots, aligns, sinks, shifts, dx, hgap):
    """Place a block, which correspond to an alignment root, recursively.

    Parameters:
        idx: The index of the root of the block to place.
        idx_vertex_left: The index of the vertex to the left in the same layer. For
            leftmost vertices, it's themselves.
        roots: The roots array from alignment. For nodes that are not used in this
            alignment (they are not along a chain to the root), it's themselves.
            For root nodes, it's themselves too. For everyone else, it's not themselves.
        aligns: The aligns array from alignment. For nodes that are not used in this
            alignment, it's themselves. For aligned nodes, it's the index of the node
            below them in the alignment. For the last node in an alignment, it's the root
            (it is basically a cycle around the aligned nodes only).
        hgap: The horizontal gap between nodes.
    Returns:
        None. The function changes dx, sinks, and shifts in place.
    """
    # Only place each root once, even though you might get there through multiple
    # paths within a block
    if idx in dx:
        return

    # The function is recursive up to the root of the block, which is initially placed at 0.0
    dx[idx] = 0.0

    # Start from the root (_place_block is ALWAYS called on roots only) and flow around the
    # alignment cycle exactly once, until you are back to the root. We visit each node in
    # the block exactly once that way.
    idx_align = None
    while (idx_align is None) or (idx_align != idx):
        if idx_align is None:
            idx_align = idx

        # This bit is a little awkward. The leftmost chain needs no shift per se, but gets
        # pushed left by a new chain incoming from the right. So the pushing happens when
        # we visit the right chain, not the left one.
        idx_left = idx_vertex_left[idx_align]
        if idx_left != idx_align:
            idx_left_root = roots[idx_left]
            # Align the left chain first. Inside this call, it checks itself whether there's
            # an even more left chain to align first, so this is recursive. Bottom line is aligning
            # left chains first, and then move right. That could be rewritten iteratively that way.
            _place_block(idx_left_root, idx_vertex_left, roots, aligns, sinks, shifts, dx, hgap)

            # Ok there was one (or more) left chains and they are all placed (but not necessarily
            # left pushed).
            sink_left_root = sinks[idx_left_root]
            sink_idx = sinks[idx]

            # NOTE: This is the only place that sets sinks. The default is that each node is its
            # own sink. If the sink on this note is unset, or if it's an actual sink of a chain
            # that is shorter than the one on the left, the left sink becomes the sink of this
            # chain because it goes deeper.
            if sink_idx == idx:
                sinks[idx] = sink_idx = sink_left_root

            # There is a chain on the left, same sink, we need to space them apart
            if sink_left_root == sink_idx:
                dx[idx] = max(dx[idx], dx[idx_left_root] + hgap)
            # There is a chain on the left but we have already assigned a different sink (e.g.
            # one even more left), we need to make sure the left chain's sink does not overlap
            else:
                shifts[sink_left_root] = min(
                    shifts[sink_left_root], dx[idx] - dx[idx_left_root] - hgap
                )

        # Follow the alignment, which cycles down and eventually back to the root
        idx_align = aligns[idx_align]


def _compact_horizontal(coords_ext, idx_vertex_left, roots, aligns, hgap=1.0):
    """Perform horizontal compaction given roots and aligns.

    Parameters:
        coords_ext: The coordinates of the extended graph.
        idx_vertex_left: The index of the vertex to the left in the same layer. For
            leftmost vertices, it's themselves.
        roots: The roots array from alignment. For nodes that are not used in this
            alignment (they are not along a chain to the root), it's themselves.
            For root nodes, it's themselves too. For everyone else, it's not themselves.
        aligns: The aligns array from alignment. For nodes that are not used in this
            alignment, it's themselves. For aligned nodes, it's the index of the node
            below them in the alignment. For the last node in an alignment, it's the root
            (it is basically a cycle around the aligned nodes only).
        hgap: The horizontal gap between nodes.
    """

    # NOTE: We compute two separate x coordinates and combine them at the end:
    # - dx: the position of each node within its block (aligned set of nodes).
    # - shifts: the shift to apply to each block's sink to avoid overlaps with other blocks.
    #     shifts are not defined for non-sink nodes (set to inf).
    # To help with this, we use the array "sinks", which maps each root to its sink node.
    # Non-root nodes *do not* have a sink (set to -1).
    # In pure Python these would be dicts of sorts, but this is ok for now.
    from collections import defaultdict

    dx = defaultdict(lambda: -np.inf)
    sinks = np.arange(len(roots))
    sink_shifts = np.inf * np.ones_like(roots, dtype=np.float64)
    roots_idx = np.flatnonzero(roots == np.arange(roots.shape[0]))
    if DEBUG_SUGI:
        print("Roots found:", roots_idx)
    for i in roots_idx:
        if DEBUG_SUGI:
            print("Placing block for root:", i)
        _place_block(i, idx_vertex_left, roots, aligns, sinks, sink_shifts, dx, hgap)

        if DEBUG_SUGI:
            print("Within for loops of _compact_horizontal")
            print(dx)
            print(sinks)
            print(sink_shifts)

    # Adjust each vertex coordinate based on its sink shift plus its dx from the sink
    x = np.array([dx[root] for root in roots], dtype=np.float64)
    xshift = sink_shifts[sinks[roots]]
    # Chains that did not get shifted can stay where they are
    good_shifts = xshift < np.inf
    x[good_shifts] += xshift[good_shifts]

    return x


def _make_and_compact_four_alignments(
    coords_ext, matrix_ext, idx_vertex_left, ignored_edges, nlayers
):
    """Build four extreme alignments to be median-ed for the final layout.

    Parameters:
        coords_ext: The coordinates of the extended graph.
        matrix_ext: The adjacency matrix of the extended graph.
        idx_vertex_left: The index of the vertex to the left in the same layer. For
            leftmost vertices, it's themselves.
        ignored_edges: The edges to ignore due to type 1 conflicts.
        nlayers: The number of layers in the graph (for convenience).
    Returns:
        A Nx4 array with the four extreme x coordinate layouts.

    NOTE: That this "compute four times and median" approach works seems
        surprising to many but hey they do usually look good.
    """
    xs = np.zeros((len(coords_ext), 4), dtype=np.float64)
    for i in range(4):
        # top left, top right, bottom left, bottom right
        align_left = i % 2 == 0
        align_top = i < 2

        if DEBUG_SUGI:
            print(f"Making alignment {i}:")
            if align_left and align_top:
                print("  Align top left")
            elif align_left:
                print("  Align bottom left")
            elif align_top:
                print("  Align top right")
            else:
                print("  Align bottom right")

        roots, aligns = _make_one_alignment(
            coords_ext,
            matrix_ext,
            ignored_edges,
            align_left,
            align_top,
            nlayers,
        )

        if DEBUG_SUGI:
            print("  Roots and aligns after alignment:")
            print(roots)
            print(aligns)
            print("  Compacting horizontally...")

        xs[:, i] = _compact_horizontal(coords_ext, idx_vertex_left, roots, aligns)
        if DEBUG_SUGI:
            print(xs[:, i])

    return xs


def _brandes_and_koepf(coords_ext, matrix_ext, ncoords):
    """Tweak the x coordinate to minimise edge lengths / maximise straight inner edges.

    Parameters:
        coords_ext: The coordinates of the extended graph.
        matrix_ext: The adjacency matrix of the extended graph.
        ncoords: The number of original coordinates (non-ghost nodes).
    Returns:
        Array with the adjusted x coordinates.

    # NOTE: This algo aims to put each node at the median of its neighbors.
    """
    # TODO: implement the Brandes and Koepf algorithm

    # The idea is to do three things:
    # 1. Identify "type 1 conflicts", i.e. when an inner edge crosses an outer edge.
    #    These are special because we would really like inner edges to be straight.
    # 2. Compute four "extreme" layouts aligned to top left, top right, etc.
    # 3. Compute the median of these 4 layouts.

    # Identify mixed-edge crossings
    ignored_edges = []
    nlayers = coords_ext["y"].max() + 1
    for il in range(nlayers - 1):
        # Find all edges from this layer
        srcs = np.flatnonzero(coords_ext["y"] == il)
        srcs, tgts = matrix_ext[srcs].nonzero()
        # Filter targets that are in next layer
        # NOTE: Should the not all be by now?
        idx_next_layer = coords_ext["y"][tgts] == il + 1
        srcs = srcs[idx_next_layer]
        tgts = tgts[idx_next_layer]

        # TODO: vectorise for optimisation
        # Find type 1 edge pairs
        for j1, (src1, tgt1) in enumerate(zip(srcs, tgts)):
            is_inner1 = (src1 >= ncoords) or (tgt1 >= ncoords)
            for j2, (src2, tgt2) in enumerate(zip(srcs[:j1], tgts[:j1])):
                is_inner2 = (src2 >= ncoords) or (tgt2 >= ncoords)
                # Not mixed, skip
                if is_inner1 == is_inner2:
                    continue
                # It's a mixed pair, which one is inner?
                jinner = j1 if is_inner1 else j2

                # Touching at one end is considered crossing, in that case
                # prioritise the inner edge also
                crossing = (src1 == src2) or (tgt1 == tgt2)
                # Of course, there is also true crossing
                # NOTE: for true crossing, the x coordinates should always be different
                # anyway (previous bits of the algo only assign one x coord per node on
                # each layer)
                crossing |= (coords_ext["x"][src1] - coords_ext["x"][src2]) * (
                    coords_ext["x"][tgt1] - coords_ext["x"][tgt2]
                ) < 0
                ignored_edges.append((srcs[jinner], tgts[jinner]))

    if DEBUG_SUGI:
        print("Ignored edges (type 1 conflicts):")
        print(ignored_edges)

    # Prepare an array with the vertex to the left in the same layer
    # For leftmost vertices, it's themselves
    idx_vertex_left = np.zeros(len(coords_ext), dtype=np.int64)
    for il in range(nlayers):
        idx_layer = np.flatnonzero(coords_ext["y"] == il)
        if len(idx_layer) == 0:
            continue
        idx_sorted = np.argsort(coords_ext["x"][idx_layer])
        idx_layer_sorted = idx_layer[idx_sorted]
        idx_vertex_left[idx_layer_sorted[0]] = idx_layer_sorted[0]
        if len(idx_layer_sorted) > 1:
            idx_vertex_left[idx_layer_sorted[1:]] = idx_layer_sorted[:-1]

    if DEBUG_SUGI:
        print("Idx vertex left:")
        print(idx_vertex_left)

    # Compute the four extreme layouts
    xs = _make_and_compact_four_alignments(
        coords_ext, matrix_ext, idx_vertex_left, ignored_edges, nlayers
    )

    # Find the smallest width alignment
    xs_max = xs.max(axis=0)
    xs_min = xs.min(axis=0)
    jmin = np.argmin(xs_max - xs_min)

    # Align the other 3 so they are vertically in line with the narrow one
    for j in range(4):
        if j == jmin:
            continue
        # j = 1 and 3 are right alighments, 0 and 2 are left alignments
        if j % 2 == 0:
            xs[:, j] += xs_min[jmin] - xs_min[j]
        else:
            xs[:, j] += xs_max[jmin] - xs_max[j]

    # Compute the median x coordinate
    xs.sort(axis=1)
    xmed = 0.5 * (xs[:, 1] + xs[:, 2])

    return xmed


def sugiyama(
    network,
    theta: float = 0.0,
    shift: Optional[tuple[float, float]] = (0, 0),
    maxiter_crossing: int = 100,
    hgap: float = 1.0,
    return_waypoints: bool = True,
):
    """Sugiyama or layered layout for directed graphs.

    Parameters:
        network: The network to layout.
        theta: Angle in radians to rotate the layout.
        shift: Shift the layout by this vector after rotation.
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

    if nv == 0:
        layout = pd.DataFrame(columns=["x", "y"], dtype=np.float64)
        waypoints = {}
    elif nv == 1:
        layout = pd.DataFrame([[0.0, 0.0]], index=index, columns=["x", "y"], dtype=np.float64)
        waypoints = {}
    else:
        # We use record arrays to keep y an integer while x is a float
        coords = np.zeros((nv), dtype=[("x", np.float64), ("y", np.int64)])

        # 1. Remove cycles via minimum feedback arc set
        matrix = provider.adjacency_matrix()
        # Ignore loops for computing the layout
        matrix[np.arange(nv), np.arange(nv)] = 0

        # TODO: check that this is correct, seems to work on a few examples for now
        coords["y"] = feedback_arc_set_eades(network, matrix)
        coords["x"], waypoints = _set_random_x_values(matrix, coords["y"])

        coords_ext, matrix_ext = _to_extended_graph(coords, matrix, waypoints)

        # NOTE: deal with multiple weakly connected components
        component_memberships = provider.component_memberships(mode="weak")
        n_components = component_memberships.max() + 1

        comp_xshift = 0.0
        for icomp in range(n_components):
            # Find the nodes in this connected component
            idx_comp = np.flatnonzero(component_memberships == icomp)

            coords_ext["x"][idx_comp] = _minimise_edge_crossings(
                coords_ext[idx_comp],
                matrix_ext[np.ix_(idx_comp, idx_comp)],
                maxiter=maxiter_crossing,
            )

            if DEBUG_SUGI:
                print("Coords[:, 0] after crossing minimisation:")
                print(coords_ext["x"][idx_comp])

            coords_ext["x"][idx_comp] = _brandes_and_koepf(
                coords_ext[idx_comp], matrix_ext[np.ix_(idx_comp, idx_comp)], nv
            )

            if DEBUG_SUGI:
                print("Coords[:, 0] after Brandes and Koepf:")
                print(coords_ext["x"][idx_comp])

            # Shift this component's coordinates to the right of the previous one
            # We will never see these nodes again, so modifying in place is fine
            coords_ext["x"][idx_comp] += comp_xshift
            comp_xshift = coords_ext["x"][idx_comp].max() + hgap

        coords, waypoints = _from_extended_graph(coords_ext, matrix_ext, nv)

        # Convert y values to float
        coords_float = np.zeros((nv, 2), dtype=np.float64)
        coords_float[:, 0] = coords["x"]
        coords_float[:, 1] = coords["y"].astype(np.float64)
        coords = coords_float

        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        coords = coords @ rotation_matrix

        coords += np.array(shift, dtype=np.float64)

        layout = pd.DataFrame(coords, index=index, columns=["x", "y"])

    if return_waypoints:
        return layout, waypoints
    else:
        return layout
