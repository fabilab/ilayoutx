"""Test dendrogram layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")
ig = pytest.importorskip("igraph")


@pytest.mark.parametrize("center", [None, (1, 1)])
def test_edgebundle_empty(helpers, center):
    """Test edge bundle layout on an empty graph."""
    g = nx.Graph()

    layout, waypoints = ilx.layouts.edgebundle(g, center=center)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)
    assert waypoints == {}


@pytest.mark.parametrize("center", [None, (1, 1)])
def test_edgebundle_singleton(helpers, center):
    """Test edge bundle layout on a singleton graph."""
    g = nx.Graph()
    g.add_node(0)

    layout, waypoints = ilx.layouts.edgebundle(g, center=center)

    expected_layout = np.array([[0, 0]], dtype=np.float64)
    if center is not None:
        expected_layout += np.array(center, dtype=np.float64)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected_layout,
        atol=1e-14,
    )
    assert waypoints == {}
