"""Test edge bundle layout."""

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


def test_edgebundle_two_nodes_no_edges_with_linkage_array(helpers):
    """Test edge bundle layout with nv>1 and explicit ndarray linkage."""
    g = nx.Graph()
    g.add_nodes_from([0, 1])

    linkage = np.array(
        [
            [1, 0, 1.0],
        ],
        dtype=np.float64,
    )

    layout, waypoints = ilx.layouts.edgebundle(g, linkage=linkage)

    helpers.check_generic_layout(layout)
    assert layout.shape == (2, 2)
    assert all(layout.index == list(g.nodes()))
    assert np.isfinite(layout.values).all()
    assert waypoints == {}


def test_edgebundle_two_nodes_no_edges_linkage_dataframe_matches_array():
    """Test DataFrame linkage gives same result as ndarray linkage."""
    g = nx.Graph()
    g.add_nodes_from([0, 1])

    linkage = np.array(
        [
            [1, 0, 1.0],
        ],
        dtype=np.float64,
    )

    layout_array, waypoints_array = ilx.layouts.edgebundle(g, linkage=linkage)
    layout_df, waypoints_df = ilx.layouts.edgebundle(g, linkage=pd.DataFrame(linkage))

    np.testing.assert_allclose(layout_df.values, layout_array.values, atol=1e-14)
    assert list(layout_df.index) == list(layout_array.index)
    assert waypoints_df == waypoints_array
