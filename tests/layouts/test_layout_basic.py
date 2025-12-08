"""Test basic layouts."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


linedata = [
    (
        0,
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
    ),
    (
        np.pi / 2,
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
        ],
    ),
    (
        np.pi,
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
            [-3.0, 0.0],
            [-4.0, 0.0],
        ],
    ),
    (
        1.5 * np.pi,
        [
            [0.0, 0.0],
            [0.0, -1.0],
            [0.0, -2.0],
            [0.0, -3.0],
            [0.0, -4.0],
        ],
    ),
]


@pytest.mark.parametrize("theta,expected", linedata)
def test_line(helpers, theta, expected):
    """Test line layout."""

    g = nx.path_graph(5)
    layout = ilx.layouts.line(g, theta=theta)

    expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (5, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )


circledata = [
    (
        0,
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
    ),
    (
        np.pi / 2,
        [
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
        ],
    ),
]


@pytest.mark.parametrize("theta,expected", circledata)
def test_circle(helpers, theta, expected):
    """Test circle layout."""
    g = nx.path_graph(4)
    layout = ilx.layouts.circle(g, theta=theta)

    expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (4, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )


shelldata = [
    (
        0,
        [[0]],
        [[0, 0]],
    ),
    (
        np.pi / 2,
        [["hello", "world"]],
        [[0, 1], [0, -1]],
    ),
    (
        0,
        [
            [0],
            [1, 3, 4, 6],
            [2, 5],
        ],
        [
            [0, 0],
            [0.5, 0],
            [1, 0],
            [0, 0.5],
            [-0.5, 0],
            [-1, 0],
            [0, -0.5],
        ],
    ),
]


@pytest.mark.parametrize("theta,nodes_by_shell,expected", shelldata)
def test_shell(helpers, theta, nodes_by_shell, expected):
    nv = sum(len(x) for x in nodes_by_shell)
    nodes = sum(nodes_by_shell, [])
    # For numeric nodes, add them to the graph in order
    # to spice things up
    if (len(nodes) > 0) and (isinstance(nodes[0], int)):
        nodes.sort()
    g = nx.Graph()
    g.add_nodes_from(nodes)
    layout = ilx.layouts.shell(g, nodes_by_shell, theta=theta)
    print(nodes)
    print(layout)

    expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (nv, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )
