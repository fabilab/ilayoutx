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
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def _find_ab_params(spread, min_dist):
    """Function taken from UMAP-learn : https://github.com/lmcinnes/umap
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def umap(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    center: Optional[tuple[float, float]] = (0, 0),
    nepochs: int = 100,
    seed: Optional[int] = None,
    inplace: bool = True,
):
    """Uniform Manifold Approximation and Projection (UMAP) layout.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes. See also the "inplace" parameter.
        center: The center of the layout.
        nepochs: The number of epochs to run the optimization.
        seed: A random seed to use.
        inplace: If True and the initial coordinates are a numpy array of dtype np.float64,
            that array will be recycled for the output and will be changed in place.
    Returns:
        The layout of the network.

    NOTE: This function assumes that the a KNN-like graph is used as input, directed from each
    node to its neighbors.
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
            inplace=inplace,
        )

        coords = initial_coords

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
