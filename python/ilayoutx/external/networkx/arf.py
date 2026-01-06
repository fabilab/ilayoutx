"""Networkx-derived support code for the arf layout."""
# Much of the following code is adapted from NetworkX

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

from typing import (
    Sequence,
)
from collections.abc import (
    Hashable,
)
import warnings
import numpy as np
import pandas as pd


def arf_networkx(
    N,
    index: list,
    edges: Sequence[tuple[Hashable, Hashable]],
    pos: np.ndarray,
    scaling: float = 1.0,
    a: float = 1.1,
    etol: float = 1e-6,
    dt: float = 1e-3,
    max_iter: int = 1000,
) -> None:
    """Arf layout for networkx

    The attractive and repulsive forces (arf) layout [1] improves the spring
    layout in three ways. First, it prevents congestion of highly connected nodes
    due to strong forcing between nodes. Second, it utilizes the layout space
    more effectively by preventing large gaps that spring layout tends to create.
    Lastly, the arf layout represents symmetries in the layout better than the
    default spring layout.

    Parameters
    ----------
    pos : np.ndarray
        Initial  position of  the nodes. The output will be stored in this variable.
    scaling : float
        Scales the radius of the circular layout space.
    a : float
        Strength of springs between connected nodes. Should be larger than 1.
        The greater a, the clearer the separation of unconnected sub clusters.
    etol : float
        Gradient sum of spring forces must be larger than `etol` before successful
        termination.
    dt : float
        Time step for force differential equation simulations.
    max_iter : int
        Max iterations before termination of the algorithm.

    Returns
    -------
        None: positions are changed in place in pos.

    References
    ----------
    .. [1] "Self-Organization Applied to Dynamic Network Layout", M. Geipel,
            International Journal of Modern Physics C, 2007, Vol 18, No 10,
            pp. 1537-1549.
            https://doi.org/10.1142/S0129183107011558 https://arxiv.org/abs/0704.1748
    """

    # attractive force of springs, ignoring loops
    tmp = pd.Series(np.arange(N), index=index)
    idx_attraction = np.array(
        [[tmp[x], tmp[y]] for x, y in edges if x != y],
    )
    K = np.ones((N, N)) - np.eye(N)
    K[idx_attraction[:, 0], idx_attraction[:, 1]] = a

    # global repulsive force, equation 10 in [1]
    rho = scaling * np.sqrt(N)

    # looping variables
    error = etol + 1
    n_iter = 0
    while error > etol:
        # A is the symmetric matrix of distances between all node pairs
        # NOTE: This is O(N^2) in memory and time. Moreover, we are not
        # recycling the memory properly, so it's allocated each iteration.
        # We could optimize by doing the latter
        diff = pos[:, np.newaxis] - pos[np.newaxis]
        A = np.linalg.norm(diff, axis=-1)[..., np.newaxis]

        # attraction_force - repulsions force
        # suppress nans due to division; caused by diagonal set to zero.
        # Does not affect the computation due to nansum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta_pos = K[..., np.newaxis] * diff - rho / A * diff
        delta_pos = np.nansum(delta_pos, axis=0)

        pos += delta_pos * dt

        error = np.linalg.norm(delta_pos, axis=-1).sum()
        if n_iter > max_iter:
            break
        n_iter += 1
