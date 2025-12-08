"""Test grid layouts."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_square_grid(helpers):
    """Test vanilla 2D grid layout."""
    g = nx.grid_2d_graph(3, 2)
    layout = ilx.layouts.grid(g, 3)

    helpers.check_generic_layout(layout)

    expected = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ]
    )
    assert layout.shape == (6, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )


def test_triangular_grid(helpers):
    """Test vanilla 2D grid layout."""
    g = nx.triangular_lattice_graph(3, 2, with_positions=True)

    expected = np.array([val["pos"] for key, val in g.nodes(data=True)])

    layout = ilx.layouts.grid(g, 2, shape="triangle")

    helpers.check_generic_layout(layout)

    print(expected)
    print(layout.values)
    assert layout.shape == (8, 2)
    assert all(layout.index == list(g.nodes()))

    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )
