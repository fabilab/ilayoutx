import pytest
import numpy as np

import ilayoutx as ilx

networkx = pytest.importorskip("networkx")


def test_line(helpers):
    """Test line layout."""

    g = networkx.path_graph(5)
    layout = ilx.layouts.line(g, theta=0.0)

    helpers.check_generic_layout(layout)
    assert layout.shape == (5, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        atol=1e-14,
    )


def test_circle(helpers):
    """Test circle layout."""
    g = networkx.path_graph(4)
    layout = ilx.layouts.circle(g, theta=0.0)

    helpers.check_generic_layout(layout)
    assert layout.shape == (4, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        atol=1e-14,
    )
