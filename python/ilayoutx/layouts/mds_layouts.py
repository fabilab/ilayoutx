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
from ..external.networkx.forceatlas2 import (
    forceatlas2_layout as fa2_networkx,
)
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def _normalize_distance_matrix(
    mat: np.ndarray | list[list[float]] | dict[tuple[Hashable, Hashable], float],
    index: list[Hashable],
    nv: int,
    inplace: bool = True,
) -> np.ndarray:
    """Normalize the distance matrix to a square numpy array."""
    if isinstance(mat, dict):
        tmp = pd.Series(np.arange(nv), index=index)
        mat = np.zeros((nv, nv), dtype=np.float64)
        for (v1, v2), d in mat.items():
            i1, i2 = tmp[v1], tmp[v2]
            mat[i1, i2] = d
        del tmp

    elif isinstance(mat, (list, tuple)):
        # Convert list of lists to numpy array
        mat = np.array(mat, dtype=np.float64)

    elif isinstance(mat, np.ndarray):
        if not np.issubdtype(mat.dtype, np.float64):
            mat = np.array(mat, dtype=np.float64)
        elif not inplace:
            mat = mat.copy()

    return mat


def multidimensional_scaling(
    network,
    distance_matrix: Optional[
        np.ndarray | list[list[float]] | dict[tuple[Hashable, Hashable], float]
    ] = None,
    center: Optional[tuple[float, float]] = (0, 0),
    inplace: bool = True,
    check_connectedness: bool = False,
):
    """Classic multidimensional_scaling for connected networks.

    Parameters:
        network: The connected network to layout. This function does not check for connectedness
            unless "check_connectedness" is True.
        distance_matrix: A symmetric distance matrix, either as a numpy array, a list of lists,
            or a dictionary. This function does NOT check for symmetry: if you input a
            non-symmetric matrix, the results will be incorrect. See also the "inplace" parameter.
            If None, the shortest path distance matrix of the network is used (but not returned).
        center: The center of the layout.
        inplace: If True and the distance matrix is a (symmetric) numpy array of dtype np.flota64,
            the matrix isinstance modified in place to save memory. Otherwise, a copy is made. If
            the distance matrix is not a numpy array to start with, a copy is always made.
        check_connectedness: If True, the function checks whether the network is connected and
            throws an exception if it is not. If False, the function *assumes* the network is
            connected (which saves time) and returns undefined results if the network is
            disconnected.
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
        if check_connectedness and (not provider.is_connected()):
            raise ValueError("The input network is not connected.")

        if distance_matrix is None:
            mat = provider.distance_matrix()
        else:
            mat = _normalize_distance_matrix(
                distance_matrix,
                index,
                nv,
                inplace,
            )

        # Get square distance matrix
        mat *= mat

        # "Double centering"
        mat_C = np.identity(nv) - np.ones((nv, nv)) / nv
        mat = -0.5 * mat_C @ mat @ mat_C

        # Get the top 2 eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        # NOTE: the absolute value is only for numerical precision issues.
        # np.abs takes the Euclidean norm of complex numbers so the
        # imaginary part is NOT ignored.
        eigenvalues = np.abs(eigenvalues)
        eigv_idx = np.argsort(eigenvalues)[-2:][::-1]
        eigval = eigenvalues[eigv_idx]
        eigvec = eigenvectors[:, eigv_idx]

        # Result: so coords[0] is the first point across
        # both eigenvectors. The element-wise product
        # in numpy broadcasts over the last dimension
        # so it's ok
        coords = np.real(eigvec) * np.sqrt(eigval)

    if center is not None:
        coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
