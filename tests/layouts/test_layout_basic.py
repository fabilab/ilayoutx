import pytest
import numpy as np

import ilayoutx as ilx

networkx = pytest.importorskip("networkx")


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

    g = networkx.path_graph(5)
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
    g = networkx.path_graph(4)
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
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
    )
]


@pytest.mark.parametrize("theta,expected", shelldata)
def test_shell(helpers, theta, expected):
    g = networkx.path_graph(4)
    layout = ilx.layouts.shell(g, theta=theta)

    expected = np.array(expected)

    helpers.check_generic_layout(layout)
    assert layout.shape == (4, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )
