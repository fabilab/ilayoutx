"""Test Uniform Manifold Approximation and Projection individual pieces."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

# umap currently buggy so we use an internal fix as control
from ilayoutx.external.umap.fastmath_fixes import smooth_knn_dist

nx = pytest.importorskip("networkx")
umap = pytest.importorskip("umap")


# The distance data needs to be pre-sorted
distancedata = [
    # Uniform distances (unweighted graph): corner but important case for us
    np.ones((5, 3), np.float64),
    # Nonuniform distances, no zeros
    [[1, 2, 3], [4, 5, 6], [2, 3, 4]],
]


@pytest.mark.filterwarnings("ignore:.*invalid.*:RuntimeWarning")
@pytest.mark.parametrize("distances", distancedata)
def test_sigma_rho(distances):
    """Test the local fuzziness calculations."""

    from ilayoutx.layouts.umap_layouts import _find_sigma_rho

    distances = np.asarray(distances, np.float64)

    res_orig = smooth_knn_dist(distances, distances.shape[1])
    res_ilx = np.array([_find_sigma_rho(distances[i]) for i in range(distances.shape[0])]).T

    np.testing.assert_allclose(res_ilx, res_orig, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("distances", distancedata)
def test_compute_connectivity_probability(distances):
    """Test the conversion from distances to probabilities."""

    from umap.umap_ import compute_membership_strengths
    from ilayoutx.layouts.umap_layouts import _compute_connectivity_probability

    distances = np.asarray(distances, np.float64)
    knn_idx = 100 * np.ones_like(distances, np.int64)
    sigmas, rhos = smooth_knn_dist(distances, distances.shape[1])

    rows, cols, vals_orig, _ = compute_membership_strengths(
        knn_idx,
        distances.astype(np.float32),
        sigmas.astype(np.float32),
        rhos.astype(np.float32),
    )
    vals_orig = vals_orig.astype(np.float64)

    sigmas_ilx = np.ones_like(distances)
    rhos_ilx = np.ones_like(distances)
    for i in range(len(distances)):
        sigmas_ilx[i], rhos_ilx[i] = sigmas[i], rhos[i]

    vals_ilx = _compute_connectivity_probability(
        distances.ravel(),
        sigmas_ilx.ravel(),
        rhos_ilx.ravel(),
    )

    np.testing.assert_allclose(vals_ilx, vals_orig, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("distances", distancedata)
def test_sigma_rho_compute_bundle(distances):
    """Test the conversion from distances to probabilities."""

    from umap.umap_ import compute_membership_strengths
    from ilayoutx.layouts.umap_layouts import (
        _compute_sigma_rho_and_connectivity_probability,
    )

    distances = np.asarray(distances, np.float64)
    knn_idx = 100 * np.ones_like(distances, np.int64)

    sigmas, rhos = smooth_knn_dist(distances, distances.shape[1])
    rows, cols, vals_orig, _ = compute_membership_strengths(
        knn_idx,
        distances.astype(np.float32),
        sigmas.astype(np.float32),
        rhos.astype(np.float32),
    )
    vals_orig = vals_orig.astype(np.float64)

    vals_ilx = np.concatenate(
        [
            _compute_sigma_rho_and_connectivity_probability(pd.Series(distancesi))
            for distancesi in distances
        ],
    )

    np.testing.assert_allclose(vals_ilx, vals_orig, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("distances", distancedata)
def test_sigma_rho_compute_bundle_as_df(distances):
    """Test the conversion from distances to probabilities."""

    from umap.umap_ import compute_membership_strengths
    from ilayoutx.layouts.umap_layouts import (
        _compute_sigma_rho_and_connectivity_probability,
    )

    distances = np.asarray(distances, np.float64)
    knn_idx = 100 * np.ones_like(distances, np.int64)

    sigmas, rhos = smooth_knn_dist(distances, distances.shape[1])
    rows, cols, vals_orig, _ = compute_membership_strengths(
        knn_idx,
        distances.astype(np.float32),
        sigmas.astype(np.float32),
        rhos.astype(np.float32),
    )
    vals_orig = vals_orig.astype(np.float64)

    # Create an edge dataframe like we designed to do in the main API
    edge_df = {"source": [], "target": [], "distance": []}
    for i, distancesi in enumerate(distances):
        for j, dist in enumerate(distancesi):
            edge_df["source"].append(i)
            edge_df["target"].append(j)
            edge_df["distance"].append(dist)
    edge_df = pd.DataFrame(edge_df)

    edge_df.sort_values(by=["source", "distance"], inplace=True)
    vals_ilx = edge_df.groupby("source")["distance"].transform(
        _compute_sigma_rho_and_connectivity_probability
    )

    np.testing.assert_allclose(vals_ilx, vals_orig, rtol=1e-5, atol=1e-8)


symmetrisation_data = [
    ("union", [0.98, 0.5, 0.4]),
    ("intersection", [0.72, 0.5, 0.4]),
    ("max", [0.9, 0.5, 0.4]),
    ("min", [0.8, 0.5, 0.4]),
    ("mean", [0.85, 0.5, 0.4]),
]


@pytest.mark.parametrize("operation,weights", symmetrisation_data)
def test_fuzzy_symmetrisation(operation, weights):
    from ilayoutx.layouts.umap_layouts import _fuzzy_symmetrisation

    edge_df = pd.DataFrame(
        [
            [0, 1, 0.9],
            [0, 2, 0.5],
            [1, 0, 0.8],
            [3, 0, 0.4],
        ],
        columns=["source", "target", "weight"],
    )

    sym = pd.DataFrame(
        [
            [0, 1],
            [0, 2],
            [0, 3],
        ],
        columns=["source", "target"],
    )
    sym["weight"] = weights

    # Test max
    res_ilx = _fuzzy_symmetrisation(
        edge_df,
        weight_col="weight",
        operation=operation,
    )
    pd.testing.assert_frame_equal(
        res_ilx,
        sym,
        atol=1e-8,
        rtol=1e-5,
    )


umap_sgd_data = [
    (4, 4, 0),
]


@pytest.mark.parametrize("n1,n2,n_epochs", umap_sgd_data)
def test_stochastic_gradient_descent(n1, n2, n_epochs):
    from scipy.sparse import coo_matrix
    from umap.umap_ import simplicial_set_embedding
    from ilayoutx.layouts.umap_layouts import (
        _stochastic_gradient_descent,
        _find_ab_params,
    )

    # Run-of-the-mill UMAP defaults
    a, b = _find_ab_params(spread=1.0, min_dist=0.1)
    seed = 42
    nsr = 5  # Negative sampling rate

    g1 = nx.complete_graph(n1)
    g2 = nx.complete_graph(n2)
    g = nx.disjoint_union(g1, g2)

    edge_df = nx.to_pandas_edgelist(g)
    # Assume there is no fuzziness in the edges, all is black and white
    edge_df["weight"] = np.float32(1.0)

    # This lets us also skip the symmetrisation step, since under "union"
    # certain signals cannot be questioned. The output must be nonredundant,
    # which this one is not (per networkx API).
    sym_edge_df = edge_df

    print(sym_edge_df)
    print(sym_edge_df.dtypes)

    initial_coords = (
        np.random.RandomState(seed)
        .uniform(
            low=-10.0,
            high=10.0,
            size=(g.number_of_nodes(), 2),
        )
        .astype(np.float32)
    )

    ## ORIGINAL UMAP IMPLEMENTATION ##
    # Convert to sparse adjacency matrix
    adjacency = coo_matrix(
        (
            sym_edge_df["weight"].values.astype(np.float32),
            (sym_edge_df["source"].values, sym_edge_df["target"].values),
        ),
        shape=(n1 + n2, n1 + n2),
    )
    # Make it actually symmetric and redundant
    # NOTE: UMAP is based on knn, so an undirected edge (a,b) for us means that
    # b is a neighbor of a and ALSO that a is a neighbor of b. We have to tell
    # UMAP about that explicitly. It will make the original implementation act
    # TWICE on this edge and, by virtue of the "move_other" parameter, it will
    # move BOTH vertices twice as well (because UMAP thinks they are bound by
    # a twin rope).
    # adjacency = adjacency + adjacency.T

    # UMAP automatically normalised the initial coordinates between -10 and 10
    coords_orig, aux_data = simplicial_set_embedding(
        None,  # We only use UMAP as a graph layout, no need for actual high-dimensional data
        adjacency,
        n_components=2,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=nsr,
        # FIXME: UMAP makes no changes in the first epoch (epoch 0)?? Awkward...
        n_epochs=n_epochs + 1,
        init=initial_coords,
        metric="euclidean",
        metric_kwds=None,
        random_state=np.random.RandomState(seed),
        # No need for any densMAP stuff here, it's barely used anyway
        densmap=False,
        densmap_kwds=None,
        output_dens=False,
        # Needed for reproducibility
        parallel=False,
    )

    ## ILX IMPLEMENTATION ##
    coords_ilx = initial_coords.copy()
    # NOTE: Some of the following parameters might or might not be optimal
    # in production, but they are set here because that's how the original UMAP
    # implementation does it.
    _stochastic_gradient_descent(
        sym_edge_df,
        g.number_of_nodes(),
        initial_coords=coords_ilx,
        a=a,
        b=b,
        n_epochs=n_epochs,
        negative_sampling_rate=nsr,
        normalize_initial_coords=True,
        avoid_neighbors_repulsion=False,
    )

    np.testing.assert_allclose(
        coords_ilx,
        coords_orig,
        atol=1e-5,
        rtol=1e-5,
    )
