"""Test Sugiyama layout."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    g = nx.DiGraph()
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


def test_singleton(helpers):
    g = nx.DiGraph()
    g.add_node(0)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0, 0]],
        atol=1e-14,
    )
