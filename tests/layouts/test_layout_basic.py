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

    if len(expected) == 0:
        expected = np.zeros((nv, 2), dtype=float)
    else:
        expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (nv, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )


spiraldata = [
    (
        0,
        0,
        [],
    ),
    (
        0,
        1,
        [[0, 0]],
    ),
    (
        0,
        0.3,
        [
            [-0.098167, -0.019057],
            [0.056732, -0.191785],
            [0.295570, -0.051368],
            [0.315688, 0.245644],
            [0.120245, 0.485326],
            [-0.184569, 0.570907],
            [-0.496693, 0.493251],
            [-0.747226, 0.285749],
            [-0.899988, -0.004605],
            [-0.943271, -0.332025],
        ],
    ),
]


@pytest.mark.parametrize("theta,slope,expected", spiraldata)
def test_spiral(helpers, theta, slope, expected):
    nv = len(expected)
    g = nx.path_graph(nv)

    layout = ilx.layouts.spiral(g, slope=slope, theta=theta)

    if len(expected) == 0:
        expected = np.zeros((nv, 2), dtype=float)
    else:
        expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (nv, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-4,
    )
