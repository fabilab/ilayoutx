from typing import (
    Optional,
)
from collections.abc import (
    Hashable,
)
import numpy as np
from scipy.optimize import (
    curve_fit,
    root_scalar,
)
import pandas as pd

from ilayoutx.ingest import (
    network_library,
    data_providers,
)
from ilayoutx.utils import _format_initial_coords
from ilayoutx._ilayoutx import (
    random as random_rust,
    _umap_apply_forces as _apply_forces_rust,
)
from ilayoutx.utils import _recenter_layout
from ilayoutx.experimental.utils import get_debug_bool

# NOTE: This is only here for future tests in terms of edge
# parallelisation in the stochastic gradient descent. The
# layout function works already.
DEBUG_UMAP = get_debug_bool("ILAYOUTX_DEBUG_UMAP", default=False)


# pandas supports JIT groupby operations (agg and transform) via
# cython or numba. The default (None) is cython if found.
# This parameter is user-writable at runtime if they so wish, but
# it's for advanced users only, therefore we do not build any
# safety around it and users who misuse it will get the stack trace
# back from pandas directly.
gropby_ops_engine = None


# Some of the code below is originally from UMAP-learn, see LICENSE below:
# BSD 3-Clause License
#
# Copyright (c) 2017, Leland McInnes
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def _find_ab_params(spread: float = 1.0, min_dist: float = 0.1):
    """Fit smoothing parameters a and b from arbitrary spread and min_dist.

    Parameters:
        spread: How long after the nearest neighbor the connectivity
            should linger (scale of the exponential). At 2 * spread
            the connection is already only worth 14% of a closest neighbor.
        min_dist: The minimum distance below and at which the connectivity
            is always 1.

    This is supposed to mimick an offset exponential decay from 1 to 0,
    i.e. in probability or fuzzy space.

    NOTES:
        - Any fine texture at scales below min_dist is lost.
        - Any fine texture at scales much larger than spread is lost too.
    As a consequence, these two parmeters really bracket the type of
    distances/weights this one instance of UMAP can tease apart.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def _find_sigma_rho(
    distances: pd.Series,
    bandwidth: float = 1.0,
    local_connectivity: float = 1.0,
    maxiter: int = 64,
) -> float:
    """Find scale of exponential decay for a given set of distances.

    Parameters:
        distances: A pandas Series of distances, sorted with the shortest being zero.
    Returns:
        A pair with (sigma, rho) where sigma is the scale used for the computation of the smooth
        distance (equivalent to spread) and rho is the distance to the closest neighbor
        (equivalent to min_dist).
    """
    if not isinstance(distances, np.ndarray):
        distances = distances.values

    n = len(distances)
    target = np.log2(n) * bandwidth

    # Lifted straight from UMAP-learn:
    lo = 0.0
    hi = np.inf
    mid = 1.0

    # TODO: This is very inefficient, but will do for now. FIXME
    ith_distances = distances
    non_zero_dists = ith_distances[ith_distances > 0.0]
    if non_zero_dists.shape[0] >= local_connectivity:
        index = int(np.floor(local_connectivity))
        interpolation = local_connectivity - index
        if index > 0:
            rho = non_zero_dists[index - 1]
            if interpolation > 1e-5:
                rho += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
        else:
            rho = interpolation * non_zero_dists[0]
    elif non_zero_dists.shape[0] > 0:
        rho = np.max(non_zero_dists)

    for n in range(maxiter):
        psum = 0.0
        for j in range(1, distances.shape[0]):
            d = distances[j] - rho
            if d > 0:
                psum += np.exp(-(d / mid))
            else:
                psum += 1.0

        if np.fabs(psum - target) < 1e-5:
            break

        if psum > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == np.inf:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    sigma = mid

    # UMAP calls this MIN_K_DIST_SCALE
    sigma = np.maximum(sigma, 1e-3 * distances.mean())
    return sigma, rho


def _compute_connectivity_probability(
    distances: np.ndarray,
    sigmas: np.ndarray,
    rhos: np.ndarray,
) -> np.ndarray:
    """Compute connectivity probabilities from distances.

    Parameters:
        distances: Distances along edges.
        sigmas: Local scales for the distances.
        rhos: Local minimum distances.
    Reruns:
        The connectivity probabilities.
    """
    vals = np.zeros_like(distances)
    # If there is no scale or they are fully connected, we know for certain
    idx_fully_connected = (distances <= rhos) | (sigmas == 0)
    vals[idx_fully_connected] = 1.0
    vals[~idx_fully_connected] = np.exp(
        -(
            (distances[~idx_fully_connected] - rhos[~idx_fully_connected])
            / (sigmas[~idx_fully_connected])
        )
    )
    return vals


def _compute_sigma_rho_and_connectivity_probability(
    distances: pd.Series,
    bandwidth: float = 1.0,
    local_connectivity: float = 1.0,
    maxiter: int = 64,
) -> np.ndarray:
    """Bundle function for the sigma, rho and computation of the connectivity probabilities.

    Parameters:
        distances: A pandas Series of distances, sorted with the shortest being zero.
        bandwidth: The bandwidth parameter for the smoothing.
        local_connectivity: TODO
        maxiter: Maximum number of iterations for the sigma/rho finder.
    Returns:
        Array of connectivity probabilities.

    NOTE: This function exists because pd.GroupBy.transform() needs not propagate sigmas and
        rhos across the whole edge list.
    TODO: Check whether this even makes sense with a profiler.
    """
    sigma, rho = _find_sigma_rho(
        distances.values,
        bandwidth=bandwidth,
        local_connectivity=local_connectivity,
        maxiter=maxiter,
    )
    # Reimplementation of the computation for each group
    vals = np.ones_like(distances)
    if sigma > 0:
        idx_fully_connected = distances <= rho
        vals[~idx_fully_connected] = np.exp(-(distances[~idx_fully_connected] - rho) / sigma)

    return vals


def _fuzzy_symmetrisation(
    edge_df: pd.DataFrame,
    weight_col: str = "weight",
    operation: str = "union",
):
    """Symmetrise the weights via fuzzy operations.

    Parameters:
        edge_df: DataFrame with edges and weights.
        weight_col: The column name with the weights to symmetrise.
        operation: The operation to use for symmetrisation. The default
            in UMAP is "union": s(w, w') = w + w' - w * w'. This number
            is always larger than both w and w' and 0 < x <= 1 - Note
            that at least one of w, w' is nonzero otherwise we don't bother.
            "max" is also supported, which is the maximum of the two weights:
            this is quite close to the union in most cases.

    Returns:
        A DataFrame with the now undirected source and target vertices and
        symmetrised weights.
    """
    # Split in target > or < source. No loops so they are never equal.
    idx = edge_df["source"] > edge_df["target"]
    src = edge_df.loc[idx, "source"]
    tgt = edge_df.loc[idx, "target"]
    edge_df.loc[idx, "source"] = tgt
    edge_df.loc[idx, "target"] = src

    # union makes the manifold more connected because it believes
    # both fuzzy signals
    if operation == "union":

        def reduce(xs):
            """Reduce the weights to the union."""
            return xs.sum() - (xs.shape[0] - 1) * xs.prod()

        res = edge_df.groupby(["source", "target"])[weight_col].agg(
            reduce,
            engine=gropby_ops_engine,
        )

    # intersection makes the manifold more grainy because it only
    # believes the shared parts of the fuzzy signals
    elif operation == "intersection":
        res = edge_df.groupby(["source", "target"])[weight_col].prod()

    # max trusts the most optimistic signal only. Notice that this
    # is not as high as trusting the most optimisic signal and then
    # also complementing its message with bits and pieces from other
    # sources (which is what "union", the default in UMAP, does).
    elif operation == "max":
        res = edge_df.groupby(["source", "target"])[weight_col].max()

    # min trusts the most pessimistic signal only. Notice that this
    # is still better than only trusting the shared signal across
    # multiple fuzzy emitters.
    elif operation == "min":
        res = edge_df.groupby(["source", "target"])[weight_col].min()

    # mean trusts the average of the emitters' confidence. Does not
    # make a whole lot of sense but if you want to use it as a
    # relatively dumb control against other operators, you might as well.
    elif operation == "mean":
        res = edge_df.groupby(["source", "target"])[weight_col].mean()

    else:
        raise ValueError(
            f"Fuzzy symmetrisation operation '{operation}' is not supported.",
        )

    res = res.reset_index()
    return res


def _apply_forces(
    sym_edge_df: pd.DataFrame,
    coords: np.ndarray,
    a: float,
    b: float,
    n_epoch: int,
    n_epochs: int,
    avoid_neighbors_repulsion: bool,
    negative_sampling_rate: float,
    next_sampling_epoch: np.ndarray,
    initial_alpha: float,
    max_displacement: float = 4.0,
) -> None:
    """Apply stochastic forces to a single epoch suring stochastic gradient descent.

    The vague analogue of this function in UMAP-learn is _optimize_layout_euclidean_single_epoch
    in umap.layouts.py. That function is polluted by densMAP code but is still somewhat readable.

    Parameters:
        sym_edge_df: DataFrame with edges and weights. Weights should usually be np.float32.
        coords: Current coordinates of the nodes. Should also usually be np.float32.
        a: The UMAP a parameter.
        b: The UMAP b parameter.
        n_epoch: The current epoch number.
        n_epochs: The total number of epochs.
        avoid_neighbors_repulsion: If True, avoid repulsion between neighboring nodes.
        negative_sampling_rate: How many negative samples to take per positive sample.
        next_sampling_epoch: Array with the next epoch number when each edge should be sampled.
        initial_alpha: The initial learning rate.
        max_displacement: The maximum displacement per epoch (before multiplying by the current
            learning rate).

    Returns:
        None. The coords array is modified in place.
    """

    # The learning rate, which decays linearly over time
    alpha = 1 - n_epoch / n_epochs

    # Figure out what edges are sampled in this epoch: tenuous edges are sampled rarely if at all
    idx_edges = next_sampling_epoch <= n_epoch

    # Decide when the *next* sampling epoch will be... these are exponentially
    # decaying weights, so it can get far enough that it's basically "never"
    # pretty quickly. More or less because of the definition of sigma, after
    # the first log2(k) sinks, their weight is likely to be < ~0.3 so they
    # come up every 3 epochs. After 3 * log2(k) sinks, the edge is only sampled
    # every 20 epochs or so. Examples:
    # - For k=10, log2(k) = 3.32, so no edges are sampled less than every 20 epochs.
    # - For k=100, log2(k) = 6.64, so 80/100 edges are sampled less than every 20
    #   epochs, in fact 60% is sampled only every 400 epochs!
    # This ignores the symmetrisation (sinks "hanging on" to other sinks), but is
    # a useful guide nonetheless about the nonlinearity involved in the log2(k)
    # choice above. Of course, ignoring most edges most of the time is exactly
    # what makes UMAP fast :-P
    next_sampling_epoch[idx_edges] += 1.0 / sym_edge_df["weight"].values[idx_edges]

    # NOTE: unlike in igraph, where we have a binaty swap call for whether
    # source or sink are feeling the force, we follow the original UMAP code
    # and move both (towards one another, and away from the evil world). Templated
    # embedding only moves the source node. The argument in the original UMAP code
    # is called "move_other" and is True for normal embedding runs.

    # NOTE: UMAP has an explicit for loop for each edge. Here we try to vectorise
    # this operation as much as possible.

    _apply_forces_rust(
        sym_edge_df[["source", "target"]].values[idx_edges].astype(np.int64),
        coords,
        (a, b),
        alpha,
        max_displacement,
        negative_sampling_rate,
    )


def _stochastic_gradient_descent(
    sym_edge_df: pd.DataFrame,
    nv: int,
    initial_coords: np.ndarray,
    a: float,
    b: float,
    initial_alpha: float = 1.0,
    n_epochs: int = 50,
    negative_sampling_rate: int = 5,
    normalize_initial_coords: bool = True,
    avoid_neighbors_repulsion: Optional[bool] = None,
    max_displacement: float = 4.0,
    record: bool = False,
) -> Optional[np.ndarray]:
    """Compute the UMAP layout using stochastic gradient descent.

    Parameters:
        sym_edge_df: DataFrame with edges and weights.
        initial_coords: Initial coordinates for the nodes.
        a: Parameter a for the UMAP curve.
        b: Parameter b for the UMAP curve.
        initial_alpha: Initial learning rate.
        n_epochs: Number of epochs to run the optimization.
        negative_sampling_rate: How many negative samples to take per positive sample.
        normalize_initial_coords: If True, normalize the initial coordinates between 0 and 10,
            which is what the original UMAP does.
        avoid_neighbors_repulsion: If True, avoid repulsion between neighboring nodes. If None,
            only use this for networks with <= 100 nodes (it is computationally expensive to
            check if a random negative sample is a neighbor).
        record: If True, record the coordinates at each epoch.
    """

    coords = initial_coords
    ne = len(sym_edge_df)
    next_sampling_epoch = np.zeros(ne)

    # For small graphs, explicit avoidance of repulsion between neighbors
    # is not that costly and more accurate than blind negative sampling.
    # For large graphs, one might spend a lot of time checking whether
    # the negative sample includes neighbors, so we avoid that.
    if avoid_neighbors_repulsion is None:
        avoid_neighbors_repulsion = nv <= 100

    if normalize_initial_coords:
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        coords -= cmin
        coords *= 10.0 / (cmax - cmin)

    if record:
        coords_history = np.zeros((n_epochs + 1, nv, 2), dtype=np.float64)
        coords_history[0] = coords

    for n_epoch in range(n_epochs):
        _apply_forces(
            sym_edge_df,
            coords,
            a,
            b,
            n_epoch,
            n_epochs,
            avoid_neighbors_repulsion,
            negative_sampling_rate,
            next_sampling_epoch,
            initial_alpha,
            max_displacement,
        )
        if record:
            coords_history[n_epoch + 1] = coords

    if record:
        return coords_history


def _get_edge_distance_df(
    provider,
    distances: np.ndarray | pd.Series | dict[(Hashable, Hashable), float] | None,
    vertices: list[Hashable],
):
    """Get a DataFrame of edges and their associated distances.

    Parameters:
        provider: The data provider for the network (initialised).
        distances: Distances between nodes.

    """
    if isinstance(distances, pd.Series):
        if not isinstance(distances.index, pd.MultiIndex):
            distances.index = pd.MultiIndex.from_tuples(distances.index)
        edge_df = distances.reset_index()
        edge_df.columns = ["source", "target", "distance"]
    elif isinstance(distances, np.ndarray):
        edges = provider.edges()
        edge_df = pd.DataFrame(edges, columns=["source", "target"])
        edge_df["distance"] = distances
    elif distances is None:
        edges = provider.edges()
        edge_df = pd.DataFrame(edges, columns=["source", "target"])
        edge_df["distance"] = 1.0
    elif isinstance(distances, dict):
        ne = len(distances)
        nv = len(vertices)
        vertex_series = pd.Series(np.arange(nv), index=vertices)
        sources = np.zeros(ne, dtype=np.int64)
        targets = np.zeros(ne, dtype=np.int64)
        dists = np.zeros(ne, dtype=np.float64)
        for i, ((source, target), dist) in enumerate(distances.items()):
            sources[i] = vertex_series[source]
            targets[i] = vertex_series[target]
            dists[i] = dist
        edge_df = pd.DataFrame(
            {
                "source": sources,
                "target": targets,
                "distance": dists,
            }
        )
    else:
        raise TypeError(
            "distances/weights must be a pd.Series indexed by tuples, np.ndarray, dict keyed by tuples, or None.",
        )

    return edge_df


def umap(
    network,
    edge_distances: Optional[np.ndarray | pd.Series | dict[(Hashable, Hashable), float]] = None,
    edge_weights: Optional[np.ndarray | pd.Series | dict[(Hashable, Hashable), float]] = None,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    min_dist: float = 0.1,
    spread: float = 1.0,
    negative_sampling_rate: Optional[int] = None,
    center: Optional[tuple[float, float]] = None,
    max_iter: int = 100,
    seed: Optional[int] = None,
    inplace: bool = True,
    record: bool = False,
):
    """Uniform Manifold Approximation and Projection (UMAP) layout.

    Parameters:
        network: The network to layout.
        edge_distances: Distances associated with the edges of the network. If None, all edges are
            assigned a distance of 1.0. This argument and "edge_weights" cannot both be provided.
        edge_weights: Weights associated with the edges of the network. If None, use "edge_distances"
            instead. This argument and "edge_distances" cannot both be provided. If provided,
            these weights must be positive and will be rescaled so the max weight is 1.0. This
            parameter is for advanced users who want to skip the sigma-rho distance-to-probability
            computation: use "edge_distances" otherwise.
        initial_coords: Initial coordinates for the nodes. See also the "inplace" parameter.
        min_dist: A fudge parameter that controls how tightly clustered the nodes will be.
            This should be considered in relationship with the following "spread" parameter.
            Smaller values will result in more tightly clustered points.
        spread: The overall scale of the embedded points. This is evaluated together with
            the previous "min_dist" parameter.
        negative_sampling_rate: How many negative samples to take per positive sample. If None,
            This is computed such that most nodes repel at least one edge source node each epoch.
        center: The center of the layout.
        max_iter: The number of epochs to run the optimization. Note that UMAP does not
            technically converge, so each time this exact number of iterations will be run.
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
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        try:
            from umap.umap_ import simplicial_set_embedding
        except ImportError:
            print(
                "The umap-learn package is needed to construct nontrivial UMAP layouts. Install it e.g. via pip install umap-learn."
            )
        try:
            from scipy.sparse import coo_matrix
        except ImportError:
            print(
                "The scipy package is needed to construct nontrivial UMAP layouts. Install it e.g. via pip install scipy."
            )

        if DEBUG_UMAP:
            import time

            t0 = time.time()
        # Fit smoothing based on fudge parameters
        a, b = _find_ab_params(spread, min_dist)

        if DEBUG_UMAP:
            t1 = time.time()
            print("ilx UMAP find_ab_params time:", t1 - t0)

            t0 = time.time()

        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: random_rust(nv, seed=seed),
            inplace=inplace,
        )
        coords = initial_coords

        if DEBUG_UMAP:
            t1 = time.time()
            print("ilx UMAP format_initial_coords time:", t1 - t0)

        # Extract the directed edges and distances
        # FIXME: remove this scipy requirement once we know everything works
        from scipy.sparse import coo_matrix

        if edge_distances is None and edge_weights is None:
            adjacency = coo_matrix(provider.adjacency_matrix(), dtype=np.float32)
            sym_edge_df = pd.DataFrame(
                {
                    "source": adjacency.row,
                    "target": adjacency.col,
                    "weight": adjacency.data,
                }
            )
            # Cut the redundancy (there are/should be no loops)
            sym_edge_df = sym_edge_df[sym_edge_df["source"] < sym_edge_df["target"]]
            # Convert back to sparse adjacency matrix (not symmetric, attractive forces move both anyway)
            adjacency = coo_matrix(
                (
                    sym_edge_df["weight"].values.astype(np.float32),
                    (sym_edge_df["source"].values, sym_edge_df["target"].values),
                ),
                shape=(nv, nv),
            )
        else:
            if edge_weights is not None:
                edge_df = _get_edge_distance_df(
                    provider,
                    distances=edge_weights,
                    vertices=index,
                )
                edge_df.rename(columns={"distance": "weight"}, inplace=True)
                # NOTE: This is not strictly necessary
                edge_df.sort_values(by=["source", "weight"], inplace=True, ascending=[True, False])
            else:
                edge_df = _get_edge_distance_df(
                    provider,
                    edge_distances,
                    vertices=index,
                )

                if DEBUG_UMAP:
                    t0 = time.time()

                # Sort by source and distance
                edge_df.sort_values(by=["source", "distance"], inplace=True)

                # Compute sigmas, rhos, and connectivity probabilities in a single transform step
                edge_df["weight"] = edge_df.groupby("source")["distance"].transform(
                    _compute_sigma_rho_and_connectivity_probability,
                    engine=gropby_ops_engine,
                )

                if DEBUG_UMAP:
                    t1 = time.time()
                    print("ilx UMAP compute connectivity probabilities time:", t1 - t0)

            if DEBUG_UMAP:
                t0 = time.time()

            # Symmetrise by fuzzy set operators (default is union)
            sym_edge_df = _fuzzy_symmetrisation(edge_df, "weight")

            if DEBUG_UMAP:
                t1 = time.time()
                print("ilx UMAP fuzzy symmetrisation time:", t1 - t0)

            # Convert to sparse adjacency matrix (not symmetric, attractive forces move both anyway)
            adjacency = coo_matrix(
                (
                    sym_edge_df["weight"].values.astype(np.float32),
                    (sym_edge_df["source"].values, sym_edge_df["target"].values),
                ),
                shape=(nv, nv),
            )

        # Stochastic gradient descent optimization
        # The C order is not strictly needed, but since we copy anyway
        # it might help by providing assurances to the Rust layer
        coords = coords.astype(np.float32, copy=True, order="C")

        # Heuristic for the negative sampling rate
        if negative_sampling_rate is None:
            ne = len(sym_edge_df)
            negative_sampling_rate = max(2, int((nv * 5) / ne))
            print(f"Negative sampling rate set to {negative_sampling_rate}.")

        if DEBUG_UMAP:
            import time

            t0 = time.time()

        # NOTE: the history is only recorded if requested, otherwise it's None
        coords_history = _stochastic_gradient_descent(
            sym_edge_df,
            nv,
            initial_coords=coords,
            a=a,
            b=b,
            n_epochs=max_iter,
            record=record,
            negative_sampling_rate=negative_sampling_rate,
        )
        if DEBUG_UMAP:
            t1 = time.time()
            print("ilx SGD UMAP time:", t1 - t0)
        if record:
            coords = coords_history

    if center is not None:
        _recenter_layout(coords[:, :2], center)

    # If history was recorded
    if DEBUG_UMAP and (coords.ndim == 3):
        coords = coords.reshape(-1, 2)
        layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
        layout["epoch"] = np.repeat(np.arange(nv), max_iter)
    else:
        layout = pd.DataFrame(coords, index=index, columns=["x", "y"])

    if DEBUG_UMAP:
        import time

        t0 = time.time()
        coords_orig, aux_data = simplicial_set_embedding(
            None,  # We only use UMAP as a graph layout, no need for actual high-dimensional data
            adjacency,
            n_components=2,
            initial_alpha=1.0,
            a=a,
            b=b,
            gamma=1.0,
            negative_sample_rate=5,
            n_epochs=max_iter + 1,
            init=initial_coords,
            metric="euclidean",
            metric_kwds=None,
            random_state=np.random.RandomState(seed),
            # No need for any densMAP stuff here, it's barely used anyway
            densmap=False,
            densmap_kwds=None,
            output_dens=False,
            parallel=False,
        )
        t1 = time.time()
        print("original SGD UMAP time:", t1 - t0)

        import matplotlib.pyplot as plt
        import iplotx as ipx

        ncopy = __import__("networkx").empty_graph(n=len(network.nodes()))
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ipx.network(
            ncopy,
            layout=layout.copy(),
            ax=axs[0],
            node_facecolor=["blue"] * (nv // 3) + ["red"] * (nv // 3),
            node_edgecolor="black",
            edge_alpha=0.1,
            node_size=5,
            title="ilayoutx UMAP",
        )
        ipx.network(
            ncopy,
            layout=coords_orig,
            ax=axs[1],
            node_facecolor=["blue"] * (nv // 3) + ["red"] * (nv // 3),
            node_edgecolor="black",
            edge_alpha=0.1,
            node_size=5,
            title="original UMAP",
        )
        plt.ion()
        plt.show()

    return layout
