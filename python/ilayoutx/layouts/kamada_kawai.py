from typing import (
    Optional,
)
import numpy as np
import scipy as sp
import pandas as pd

from ilayoutx._ilayoutx import (
    circle,
)
from ..ingest import (
    network_library,
    data_providers,
)


def kamada_kawai(
    network,
    radius: Optional[float] = 1.0,
    center: Optional[tuple[float, float]] = (0, 0),
    initial_coords: Optional[np.ndarray | dict | pd.DataFrame] = None,
):
    """Kamada-Kawai layout algorithm.

    Parameters:
        network: The network to layout.
        seed: A random seed to use.
    Returns:
        The layout of the network.
    """

    # 0. Find what we have
    nl = network_library(network)
    provider = data_providers[nl](network)

    # 1. Compute the distance matrix.
    tmp = provider.get_shortest_distance()
    dist = tmp["matrix"]
    index = tmp["index"]
    nv = len(index)

    # 2. Get and set largest finite distance.
    # Infinite distance stems from non-connected components.
    dist[np.isinf(dist)] = -1
    # In case they are all singletons, there is no max finite distance.
    dist[dist < 0] = max(dist.max(), 0)

    # 3. Initial layout
    if initial_coords is None:
        # This should be what the paper suggested. Note that
        # igraph uses 0.36 * np.sqrt(nv) as the radius to
        # asymptotically converge for actual circular graphs.
        initial_coords = circle(nv, radius=0.5 * np.sqrt(nv))
    else:
        if isinstance(initial_coords, dict):
            initial_coords = pd.DataFrame(initial_coords).T.loc[index].values
        elif isinstance(initial_coords, np.ndarray):
            pass
        elif isinstance(initial_coords, pd.DataFrame):
            initial_coords = initial_coords.loc[index].values
        else:
            raise TypeError(
                "Initial coordinates must be a numpy array, pandas DataFrame, or dict.",
            )

    # 4. Solve the kk optimization problem
    coords = _kamada_kawai_solve(dist, initial_coords, 2)

    # 5. Rescale and center the coordinates
    coords *= radius / np.abs(coords).max()
    coords += np.array(center, dtype=np.float64)

    return coords


# The following is almost verbatim from the NetworkX source code, hence the license.


# NetworkX is distributed with the 3-clause BSD license.
#
# ::
#
#    Copyright (c) 2004-2025, NetworkX Developers
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#      * Neither the name of the NetworkX Developers nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim)

    optresult = sp.optimize.minimize(
        _kamada_kawai_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, dim))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    grad = np.einsum(
        "ij,ij,ijk->ik",
        invdist,
        offset,
        direction,
    ) - np.einsum(
        "ij,ij,ijk->jk",
        invdist,
        offset,
        direction,
    )

    # Regularisation to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())
