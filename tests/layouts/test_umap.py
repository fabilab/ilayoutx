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
