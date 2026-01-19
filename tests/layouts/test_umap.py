"""Test Uniform Manifold Approximation and Projection layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    g = nx.Graph()

    layout = ilx.layouts.umap(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = nx.DiGraph()
    g.add_node(0)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.umap(g, **kwargs)
    # Default center is (0, 0)
    if center is None:
        center = (0, 0)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [center],
        atol=1e-14,
    )


@pytest.mark.parametrize("n1,n2", [(20, 10), (5, 5)])
def test_two_clumps(helpers, n1, n2):
    """Test a pair of clumps."""

    g1 = nx.complete_graph(n1)
    g2 = nx.complete_graph(n2)
    g = nx.disjoint_union(g1, g2)

    layout = ilx.layouts.umap(g, min_dist=0.1, seed=42)

    for i in range(n1):
        d2i = ((layout.values - layout.values[i]) ** 2).sum(axis=1)
        assert d2i[:n1].max() < d2i[n1:].min()  # Clump 1 is separate
    for i in range(n1, n1 + n2):
        d2i = ((layout.values - layout.values[i]) ** 2).sum(axis=1)
        assert d2i[n1:].max() < d2i[:n1].min()  # Clump 2 is separate


@pytest.mark.parametrize("n1,n2", [(20, 10), (5, 5)])
def test_two_clumps_fixed(helpers, n1, n2):
    """Test a pair of clumps."""

    g1 = nx.complete_graph(n1)
    g2 = nx.complete_graph(n2)
    g = nx.disjoint_union(g1, g2)

    initial_coords = np.random.rand(n1 + n2, 2) * 30
    fixed = np.zeros(len(initial_coords), dtype=bool)
    fixed_idx = [2, 3]
    fixed[fixed_idx] = True

    layout = ilx.layouts.umap(
        g,
        initial_coords=initial_coords.copy(),
        fixed=fixed,
    )

    # NOTE: There is a np.float32 conversion in UMAP that makes this test less precise
    np.testing.assert_allclose(
        layout.values[fixed_idx],
        initial_coords[fixed_idx],
        atol=1e-5,
        rtol=1e-5,
    )

    others_nonclose = np.linalg.norm(initial_coords[~fixed] - layout.values[~fixed], axis=1) > 1e-4
    np.testing.assert_equal(others_nonclose, np.ones(others_nonclose.shape, dtype=bool))
