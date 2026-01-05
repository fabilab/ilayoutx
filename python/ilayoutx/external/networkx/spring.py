"""Networkx-derived support code for spring layout."""

# The code was modified from NetworkX library:

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

import numpy as np


def _fruchterman_reingold(
    A,
    k=None,
    pos=None,
    fixed=None,
    max_iter=50,
    threshold=1e-4,
    seed=None,
    exponent_attraction: float = 1.0,
    exponent_repulsion: float = -2.0,
) -> None:
    """Fruchterman-Reingold force-directed layout algorithm.

    NOTE: this function writes the output in place in the pos variable.
    """
    nnodes = len(A)

    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1

    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / (max_iter + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)

    # the inscrutable (but fast) version - this is still O(V^2)
    for iteration in range(max_iter):
        # pairwise distance
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)

        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)

        # "forces"
        ratio = distance / k
        repulsion = ratio**exponent_repulsion
        attraction = A * (ratio**exponent_attraction)
        force = repulsion - attraction

        # displacement as a result
        displacement = np.einsum("ijk,ij->ik", delta, force)

        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        np.clip(length, a_min=0.01, a_max=None, out=length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)

        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0

        pos += delta_pos

        # cool temperature
        t -= dt

        # check for convergence
        if (np.linalg.norm(delta_pos) / nnodes) < threshold:
            break
