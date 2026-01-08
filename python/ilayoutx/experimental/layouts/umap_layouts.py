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
)


def _find_ab_params(spread, min_dist):
    """Function taken from UMAP-learn : https://github.com/lmcinnes/umap
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


def _find_sigma(
    distances: pd.Series,
    bandwidth: float = 1.0,
) -> float:
    """Find scale of exponential decay for a given set of distances.

    Parameters:
        distances: A pandas Series of distances, sorted with the shortest being zero.
    """
    n = len(distances)
    tgt = np.log2(n) * bandwidth

    def funmin(sigma):
        return tgt - (np.exp(-distances / sigma)).sum()

    return root_scalar(
        funmin,
        bracket=(0, 1000.0),
        method="brentq",
        maxiter=64,
    ).root


def _fuzzy_symmetrisation(
    edge_df: pd.DataFrame,
    weight_col: str,
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
    # Split in target > or < source. There must be no loops so they are never
    # equal.
    idx = edge_df["source"] > edge_df["target"]
    src = edge_df.loc[idx, "source"]
    tgt = edge_df.loc[idx, "target"]
    edge_df.loc[idx, "source"] = tgt
    edge_df.loc[idx, "target"] = src

    if operation == "max":
        res = edge_df.groupby(["source", "target"])[weight_col].max()

    # union makes the manifold more connected
    elif operation == "union":

        def reduce(xs):
            """Reduce the weights to the union."""
            return xs.sum() - (len(xs) - 1) * xs.prod()

        res = edge_df.groupby(["source", "target"])[weight_col].apply(
            reduce,
        )

    # intersection makes the manifold more grainy
    elif operation == "intersection":
        res = edge_df.groupby(["source", "target"])[weight_col].prod()

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
):
    dist2_min_attr = 1e-8
    dist2_min_rep = 1e-4
    nv = len(coords)

    # Linear simulated annealing
    learning_rate = 1 - n_epoch / n_epochs

    # Figure out what edges are sampled in this epoch
    idx_edges = next_sampling_epoch <= n_epoch
    idx_source = sym_edge_df["source"].values[idx_edges]
    idx_target = sym_edge_df["target"].values[idx_edges]

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
    # choice above. Of course, ignoring most edges most of the time makes the
    # algorithm fast :-P
    next_sampling_epoch[idx_edges] += 1.0 / sym_edge_df["weight"].values[idx_edges]

    # NOTE: unlike in igraph, where we have a binaty swap call for whether
    # source or sink are feeling the force, we follow the original UMAP code
    # and move both (towards one another, and away from the evil world). Templated
    # embedding would break this decision, we leave it for now.

    # Attractive force (cross-entropy)
    delta = coords[idx_source] - coords[idx_target]
    dist2 = (delta * delta).sum(axis=1)
    force_attr = -2 * a * b * dist2 ** (b - 1) / (1.0 + a * dist2**b)
    # Forfeit pairs that are already basically on top of one another
    force_attr[dist2 < dist2_min_attr] = 0
    coords[idx_source] += force_attr[:, None] * delta * learning_rate
    coords[idx_target] -= force_attr[:, None] * delta * learning_rate

    # Repulsive force via negative samlping (cross-entropy)
    # FIXME: improve this to the actual number
    n_negative_samples = negative_sampling_rate * np.ones(idx_edges.sum(), dtype=np.int64)
    # NOTE: This is nifty little trick by which we do not iterate directly
    # over the edges. To vectorise aggressively, we iterate over the observed
    # number of negative samples needed and push the same vertex multiple times.
    # This has somewhat fixed complexity that barely depends on the number of
    # edges - as far as Python is concerned. n_ns_max is basically the inverse
    # of the smallest weight in the sym_edge list *that was sampled this epoch*.
    # In other words, if you happen to sample a rare weight, here's where you
    # pay for it, but the cost is almost independent on the number of edges.
    # TODO: A more straightforward approach would be to do this in Rust, as a
    # straightup nested for loop (edges and negative samples for each edge).
    n_ns_max = max(n_negative_samples)
    for ineg in range(1, n_ns_max + 1):
        # Which of the edges require an additional negative sample.
        idx_negative_edges = n_negative_samples >= ineg

        # Sample a random vertex for each edge that requires a negative sample.
        # Repulsion for BOTH source and target from that vertex will be applied.
        idx_negative_vertices = np.random.randint(nv, size=idx_negative_edges.sum())
        coords_negative = coords[idx_negative_vertices]
        idxs_focal = {"source": idx_source, "target": idx_target}
        for name_focal, idx_focal in idxs_focal.items():
            idx_focal_neg = idx_focal[idx_negative_edges]
            coords_focal = coords[idx_focal_neg]
            delta_neg = coords_focal - coords_negative
            dist2_neg = (delta_neg * delta_neg).sum(axis=1)
            force_rep = 2 * b / (dist2_min_rep + dist2_neg) / (1.0 + a * dist2**b)
            # Shorten self-repulsion to zero
            force_rep[idx_focal == idx_negative_vertices] = 0
            # Shorten repulsion of neighbors to zero for small graphs
            if avoid_neighbors_repulsion:
                name_other = "target" if name_focal == "source" else "source"
                idx_other = idxs_focal[name_other]
                for ifr, (i, j) in enumerate(zip(idx_focal, idx_negative_vertices)):
                    if ((idx_focal == i) & (idx_other == j)).sum() > 0:
                        force_rep[ifr] = 0
            coords[idx_focal_neg] += force_rep[:, None] * delta_neg * learning_rate


def _stochastic_gradient_descent(
    sym_edge_df: pd.DataFrame,
    nv: int,
    initial_coords: np.ndarray,
    a: float,
    b: float,
    n_epochs: int = 50,
    # NOTE: this is actually computed depending on the weights in the original UMAP...
    # something like the sum of positive and negative samples across history is constant across edges...
    # so each time an edge is picked for sampling, we figure how long it's been waiting for (the epoch
    # delta is fixed) and we pick that number of negative samples. I don't think it matters much.
    # maybe do that?
    negative_sampling_rate: int = 5,
    record: bool = False,
) -> Optional[np.ndarray]:
    """Compute the UMAP layout using stochastic gradient descent.

    Parameters:
        sym_edge_df: DataFrame with edges and weights.
        initial_coords: Initial coordinates for the nodes.
        a: Parameter a for the UMAP curve.
        b: Parameter b for the UMAP curve.
        n_epochs: Number of epochs to run the optimization.
        negative_sampling_rate: How many negative samples to take per positive sample.
        record: If True, record the coordinates at each epoch.
    """

    ne = len(sym_edge_df)
    next_sampling_epoch = np.zeros(ne)

    # For small graphs, explicit avoidance of repulsion between neighbors
    # is not that costly and more accurate than blind negative sampling.
    # For large graphs, one might spend a lot of time checking whether
    # the negative sample includes neighbors, so we avoid that.
    avoid_neighbors_repulsion = nv <= 100

    learning_rate = 1.0
    coords = initial_coords
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
            "distances must be a pd.Series indexed by tuples, np.ndarray, dict keyed by tuples, or None.",
        )

    return edge_df


def umap(
    network,
    distances: Optional[np.ndarray | pd.Series | dict[(Hashable, Hashable), float]] = None,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    min_dist: float = 0.5,
    spread: float = 1.0,
    center: Optional[tuple[float, float]] = (0, 0),
    max_iter: int = 100,
    seed: Optional[int] = None,
    inplace: bool = True,
    record: bool = False,
):
    """Uniform Manifold Approximation and Projection (UMAP) layout.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes. See also the "inplace" parameter.
        min_dist: A fudge parameter that controls how tightly clustered the nodes will be.
            This should be considered in relationship with the following "spread" parameter.
            Smaller values will result in more tightly clustered points.
        spread: The overall scale of the embedded points. This is evaluated together with
            the previous "min_dist" parameter.
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

        # Fit smoothing based on fudge parameters
        a, b = _find_ab_params(spread, min_dist)

        # Extract the directed edges and distances
        edge_df = _get_edge_distance_df(
            provider,
            distances,
            vertices=index,
        )

        # Sort by source and distance
        edge_df.sort_values(by=["source", "distance"], inplace=True)

        # Subtract closest neighbour
        edge_df["distance"] -= edge_df.groupby("source")["distance"].transform("min")

        # Estimate sigma by scalar minimisation
        edge_df["sigma"] = edge_df.groupby("source")["distance"].transform(_find_sigma)

        # Compute weights
        edge_df["weight"] = np.exp(edge_df["distance"] / edge_df["sigma"])
        sym_edge_df = _fuzzy_symmetrisation(edge_df, "weight")

        # Stochastic gradient descent optimization
        # NOTE: the history is only recorded if requested, otherwise it's None
        coords_history = _stochastic_gradient_descent(
            sym_edge_df,
            nv,
            initial_coords=coords,
            a=a,
            b=b,
            n_epochs=max_iter,
            record=record,
        )
        if record:
            coords = coords_history

        __import__("ipdb").set_trace()

    coords += np.array(center, dtype=np.float64)

    # If history was recorded
    if coords.ndim == 3:
        coords = coords.reshape(-1, 2)
        layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
        layout["epoch"] = np.repeat(np.arange(nv), max_iter)
    else:
        layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
