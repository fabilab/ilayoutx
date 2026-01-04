"""Test multipartite layouts."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


@pytest.fixture(scope="module")
def network():
    """Create a bipartite network for testing."""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3, 4, 5])
    g.add_edges_from([(0, 3), (1, 4), (2, 5), (0, 4), (1, 5)])
    return g


first_data = [
    ([0, 1, 2], 1, 0, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
    ([0, 1], 1, 0, [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]),
    ([0, 1, 2], 3, 0, [[0, 0], [0, 1], [0, 2], [3, 0], [3, 1], [3, 2]]),
    ([0, 1], 4, 0, [[0, 0], [0, 1], [4, 0], [4, 1], [4, 2], [4, 3]]),
    ([0, 1, 2], 1, np.pi / 2, [[0, 0], [1, 0], [2, 0], [0, -1], [1, -1], [2, -1]]),
]


multi_data = [
    ([[0, 1, 2], [3, 4, 5]], 1, 0, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
    ([[0, 1, 2], [3, 4, 5]], 2, np.pi / 2, [[0, 0], [1, 0], [2, 0], [0, -2], [1, -2], [2, -2]]),
    ([[0, 1, 2], [3, 4], [5]], 2, np.pi / 2, [[0, 0], [1, 0], [2, 0], [0, -2], [1, -2], [0, -4]]),
]


@pytest.mark.parametrize("nlist,distance,theta,expected", multi_data)
def test_multipartite(helpers, network, nlist, distance, theta, expected):
    """Test multipartite layout."""

    expected = np.array(expected)

    layout = ilx.layouts.multipartite(network, nlist, distance=distance, theta=theta)
    helpers.check_generic_layout(layout)
    assert layout.shape == (6, 2)
    assert all(layout.index == list(network.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )
