"""Test Uniform Manifold Approximation and Projection individual pieces."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

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

    from umap.umap_ import smooth_knn_dist
    from ilayoutx.experimental.layouts.umap_layouts import _find_sigma_rho

    distances = np.asarray(distances, np.float64)

    res_orig = smooth_knn_dist(distances, distances.shape[1])
    res_ilx = np.array([_find_sigma_rho(distances[i]) for i in range(distances.shape[0])]).T

    np.testing.assert_allclose(res_ilx, res_orig, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("distances", distancedata)
def test_compute_connectivity_probability(distances):
    """Test the conversion from distances to probabilities."""

    from umap.umap_ import smooth_knn_dist, compute_membership_strengths
    from ilayoutx.experimental.layouts.umap_layouts import _compute_connectivity_probability

    distances = np.asarray(distances, np.float64)
    knn_idx = 100 * np.ones_like(distances, np.int64)
    sigmas, rhos = smooth_knn_dist(distances, distances.shape[1])

    print(sigmas, rhos)

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
