"""Geometric layout function from netgraph, used with permission.

See:

https://github.com/paulbrodersen/netgraph/issues/104

Here re-released under MIT LICENSE.
"""


# Copyright 2026 Paul Brodersen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, NonlinearConstraint


def get_geometric_layout(
    edges,
    edge_length,
    initial_positions,
    node_size=0.0,
    tol=1e-3,
):
    """Node layout for defined edge lengths but unknown node positions.

    Node positions are determined through non-linear optimisation: the
    total distance between nodes is maximised subject to the constraint
    imposed by the edge lengths, which are used as upper bounds.
    If provided, node sizes are used to set lower bounds to minimise collisions.

    ..note:: This implementation is slow.

    Parameters
    ----------
    edges : list
        The edges of the graph, with each edge being represented by a (source node ID, target node ID) tuple.
    edge_length : dict
        Mapping of edges to their lengths.
    node_size : scalar or dict, default 0.
        Size (radius) of nodes.
        Providing the correct node size minimises the overlap of nodes in the graph,
        which can otherwise occur if there are many nodes, or if the nodes differ considerably in size.
    tol : float, default 1e-3
        The tolerance of the cost function. Small values increase the accuracy, large values improve the computation time.

    Returns
    -------
    node_positions : dict
        Dictionary mapping each node ID to (float x, float y) tuple, the node position.

    """

    # TODO: assert triangle inequality is not violated.
    # HOLD: probably not necessary, as minimisation can still proceed when triangle inequality is violated.

    ## assert that the edges fit within the canvas dimensions
    # width, height = scale
    # max_length = np.sqrt(width**2 + height**2)
    # too_long = dict()
    # for edge, length in edge_length.items():
    #    if length > max_length:
    #        too_long[edge] = length
    # if too_long:
    #    msg = f"The following edges exceed the dimensions of the canvas (`scale={scale}`):"
    #    for edge, length in too_long.items():
    #        msg += f"\n\t{edge} : {length}"
    #    msg += "\nEither increase the `scale` parameter, or decrease the edge lengths."
    #    raise ValueError(msg)

    # ensure that graph is bi-directional
    edges = edges + [(target, source) for (source, target) in edges]  # forces copy
    edges = list(set(edges))
    # sync the lengths with the (bi-directional) edges
    lengths = []
    for source, target in edges:
        if (source, target) in edge_length:
            lengths.append(edge_length[(source, target)])
        else:
            lengths.append(edge_length[(target, source)])

    # upper bound: pairwise distance matrix with unknown distances set to the maximum possible distance given the canvas dimensions
    max_length = np.max(lengths) * 2.0

    sources, targets = zip(*edges)
    nodes = sources + targets
    unique_nodes = set(nodes)
    indices = range(len(unique_nodes))
    node_to_idx = dict(zip(unique_nodes, indices))
    source_indices = [node_to_idx[source] for source in sources]
    target_indices = [node_to_idx[target] for target in targets]

    total_nodes = len(unique_nodes)
    distance_matrix = np.full((total_nodes, total_nodes), max_length)
    distance_matrix[source_indices, target_indices] = lengths
    distance_matrix[np.diag_indices(total_nodes)] = 0
    upper_bounds = squareform(distance_matrix)

    # lower bound: sum of node sizes
    if isinstance(node_size, (int, float)):
        sizes = node_size * np.ones((total_nodes))
    elif isinstance(node_size, dict):
        sizes = np.array([node_size[node] if node in node_size else 0.0 for node in unique_nodes])

    sum_of_node_sizes = sizes[np.newaxis, :] + sizes[:, np.newaxis]
    sum_of_node_sizes -= np.diag(
        np.diag(sum_of_node_sizes)
    )  # squareform requires zeros on diagonal
    lower_bounds = squareform(sum_of_node_sizes)
    invalid = lower_bounds > upper_bounds
    lower_bounds[invalid] = upper_bounds[invalid] - 1e-8

    # For an extended discussion of this cost function and alternatives see:
    # https://stackoverflow.com/q/75137677/2912349
    def cost_function(positions):
        return 1 / np.sum(np.log(pdist(positions.reshape((-1, 2))) + 1))

    def constraint_function(positions):
        positions = np.reshape(positions, (-1, 2))
        return pdist(positions)

    nonlinear_constraint = NonlinearConstraint(
        constraint_function, lb=lower_bounds, ub=upper_bounds, jac="2-point"
    )
    result = minimize(
        cost_function,
        initial_positions.flatten(),
        method="SLSQP",
        jac="2-point",
        constraints=[nonlinear_constraint],
        options=dict(ftol=tol),
    )

    if not result.success:
        print("Warning: could not compute valid node positions for the given edge lengths.")
        print(f"scipy.optimize.minimize: {result.message}.")

    node_positions_as_array = result.x.reshape((-1, 2))
    # node_positions_as_array = _fit_to_frame(
    #    node_positions_as_array, np.array(origin), np.array(scale), pad_by
    # )
    node_positions = dict(zip(unique_nodes, node_positions_as_array))
    return node_positions
