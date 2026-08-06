from typing import (
    Optional,
    Sequence,
)
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_array
from scipy.sparse.linalg import cg

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import (
    _format_initial_coords,
    _recenter_layout,
)
from .._ilayoutx import (
    random as random_rust,
)
from ilayoutx.experimental.utils import get_debug_bool


DEBUG_NEATO = get_debug_bool("ILAYOUTX_DEBUG_NEATO", default=False)


def _stress(
    de_edges,
    d_edges,
    w_edges,
) -> float:
    """Compute the stress of the current layout.

    Parameters:
        de_edges: The current distances between connected nodes.
        d_edges: The graph-theoretical distances between connected nodes (unweighted edges means 1.0).
        w_edges: The weights of the edges (derived from d_edges).

    Returns:
        The stress value.
    """
    # NOTE: edges are not duplicated, therefore this already works as an i<j condition
    return (w_edges * ((de_edges - d_edges) ** 2)).sum()


def _compute_demb_edges(
    X: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Compute distances between connected nodes in embedding space.

    Parameters:
        X: The current coordinates of the nodes.
        edges: The edges of the network.
    Returns:
        An array of distances between connected nodes in embedding space.
    """
    return np.sqrt(((X[edges[:, 0]] - X[edges[:, 1]]) ** 2).sum(axis=1))


def _compute_LX(
    delta_nonsym: coo_matrix,
    de_edges: np.ndarray,
):
    """Compute the L^Z matrix for the current X.

    Parameters:
        delta_nonsym: Non-symmetrised version of the delta_ij matix.
        de_edges: The current distances between connected nodes in embedding space.
    Returns:
        The symmetric, zero-row/col-sum sparse L^Z matrix for the current X.
    """
    lx = -delta_nonsym
    # The order of the sparse stuff is anyway the same as the edges
    lx.data *= 1.0 / de_edges
    # symmetrise
    lx += lx.T
    # add diagonal (it makes the sum of each row/column zero)
    row_off_diag_sum = np.array(lx.sum(axis=1)).flatten()
    lx[np.arange(lx.shape[0]), np.arange(lx.shape[0])] = -row_off_diag_sum
    return lx


def _majorise_stress(
    X: np.ndarray,
    edges: Sequence[tuple[Hashable, Hashable]],
    degrees: Sequence[int],
    etol: float = 1e-4,
    max_iter: int = 50,
) -> None:
    """Majorise stress function via iterative conjugate gradient.

    NOTE: For details and notation see: https://www.graphviz.org/documentation/GKN04.pdf
    """
    nv = len(X)

    if DEBUG_NEATO:
        print("Edges (nonduplicated):")
        print(edges)

    # Convert edge data structure into a *nonduplicated* Mx2 numpy array of ints
    edges = np.array(edges, dtype=np.int64)

    if DEBUG_NEATO:
        print("Edges after casting to 2D array (nonduplicated):")
        print(edges)
        print("Initial coordinates:")
        print(X)

    # Filter out loops, they are irrelevant anyway
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Weights are always one for now
    d_edges = np.ones(edges.shape[0], dtype=np.float64)
    w_edges = 1.0 / d_edges / d_edges

    # Construct the graph Laplacian matrix (constant throughout, acts as a North star)
    # Start with one off-diagonal half
    Lw = coo_matrix(
        (-w_edges, (edges[:, 0], edges[:, 1])),
        shape=(nv, nv),
        dtype=np.float64,
    )
    # The other off-diagonal half since edges are not duplicated
    Lw += Lw.T
    # Add the diagonal entries
    Lw += coo_matrix(
        (degrees, (np.arange(nv), np.arange(nv))),
        shape=(nv, nv),
        dtype=np.float64,
    )
    # Convert to whatever scipy's optimiser likes to eat later on
    Lw = csc_array(Lw)

    # Compute the half delta matrix delta_ij (awkward notation)
    delta_nonsym = coo_matrix(
        (1.0 / w_edges, (edges[:, 0], edges[:, 1])),
        shape=(nv, nv),
        dtype=np.float64,
    )

    # Compute the initial stress for the first round of iteration
    de_edges = _compute_demb_edges(X, edges)
    stress_X = _stress(de_edges, d_edges, w_edges)

    # Each iteration we have:
    # b = L^X(t) * X(t)
    # and we solve for the next iter X(t+1)
    LX = _compute_LX(delta_nonsym, de_edges)
    b = np.array(LX @ X)

    if DEBUG_NEATO:
        print(f"Initial stress = {stress_X}")
        print(f"Number of vertices: {nv}, number of edges (nonduplicated): {edges.shape[0]}")
        print(f"Lw shape:\n{Lw.shape}")
        print(f"LX shape:\n{LX.shape}")
        print(f"X shape:\n{X.shape}")
        print(f"b = LX @ X shape:\n{b.shape}")

    for t in range(max_iter):
        # FIXME: remove degeneracy by fixing position of a vertex (translation/rotation)

        Xnext_0, exit_code = cg(Lw, b[:, 0], atol=1e-5)
        Xnext_1, exit_code = cg(Lw, b[:, 1], atol=1e-5)
        Xnext = np.column_stack((Xnext_0, Xnext_1))

        # FIXME: explain better what's up
        if exit_code != 0:
            print(
                f"Conjugate gradient solver did not converge at iteration {t}. Exit code: {exit_code}"
            )

            if DEBUG_NEATO:
                print(f"Lw:\n{Lw.toarray()}")
                print(f"LX:\n{LX.toarray()}")
                print(f"X:\n{X}")
                print(f"LX @ X:\n{b}")

            break

        if t == max_iter - 1:
            if DEBUG_NEATO:
                de_edges = _compute_demb_edges(X, edges)
                stress_Xnext = _stress(de_edges, d_edges, w_edges)
                relative_stress_change = abs(stress_Xnext - stress_X) / stress_X
                print(f"Iteration {t}:")
                print("  distance_embedding_edges:")
                for tmp in de_edges:
                    print(f"    {tmp}")
                print(f" stress = {stress_X}, relative stress change = {relative_stress_change}")

            break

        X = Xnext
        de_edges = _compute_demb_edges(X, edges)
        stress_Xnext = _stress(de_edges, d_edges, w_edges)
        relative_stress_change = abs(stress_Xnext - stress_X) / stress_X
        if DEBUG_NEATO:
            print(f"Iteration {t}:")
            print("  distance_embedding_edges:")
            for tmp in de_edges:
                print(f"    {tmp}")
            print(f" stress = {stress_X}, relative stress change = {relative_stress_change}")

        if relative_stress_change < etol:
            break

        # Prepare variables for the next round
        stress_X = stress_Xnext
        LX = _compute_LX(delta_nonsym, de_edges)
        b = np.array(LX @ X)


def neato(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    center: tuple[float, float] = (0, 0),
    scale: Optional[float] = 1.0,
    etol: float = 1e-4,
    max_iter: int = 50,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Neato layout algorithm.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes.
        center: Recenter the layout around this point.
        scale: Scaling factor for the layout. The larger of x- and y-ranges will be equal to scale.
        max_iter: Max iterations before termination of the algorithm.
    Returns:
        The layout of the network.


    References:
        Inspiration for this layout comes from the Graphviz neato layout algorithm:
            Gansner, E.R., Koren, Y., North, S. (2005). Graph Drawing by Stress Majorization.
            In: Pach, J. (eds) Graph Drawing. GD 2004. Lecture Notes in Computer Science, vol 3383. Springer, Berlin, Heidelberg.
            https://www.graphviz.org/documentation/GKN04.pdf
    """

    nl = network_library(network)
    provider = data_providers[nl](network)

    # If the graph is not undirected, fail
    if provider.is_directed():
        raise ValueError("Neato layout only works for undirected graphs.")

    # TODO: Check for multiedges?? maybe ok. Loops are excluded later on

    # Compute the distance matrix.
    tmp = provider.get_shortest_distance()
    dist = tmp["matrix"]
    index = tmp["index"]
    nv = len(index)

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        # Get and set largest finite distance.
        # Infinite distance stems from non-connected components.
        dist[np.isinf(dist)] = -1
        # In case they are all singletons, there is no max finite distance.
        dist[dist < 0] = max(dist.max(), 0)

        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: random_rust(nv, seed=seed),
        )
        initial_coords.setflags(write=True)

        if max_iter > 0:
            edges = provider.edges()

            _majorise_stress(
                X=initial_coords,
                edges=edges,
                degrees=provider.degrees(),
                etol=etol,
                max_iter=max_iter,
            )
        coords = initial_coords

    if center is not None:
        _recenter_layout(coords, center)

    if scale is not None:
        coords *= scale

    return pd.DataFrame(coords, index=index, columns=["x", "y"])
