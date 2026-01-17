"""Test LGL layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.large_graph_layout(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = ig.Graph(n=1)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.large_graph_layout(g, **kwargs)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == [0])
    np.testing.assert_allclose(
        layout.values,
        [center or (0, 0)],
        atol=1e-14,
    )


noforcedata = [
    (
        1,
        [
            [0, 0],
            [1.0, 0],
        ],
    ),
    (
        2,
        [
            [0, 0],
            [1, 0],
            [-1, 0],
        ],
    ),
    (
        4,
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ],
    ),
]


@pytest.mark.parametrize("nchildren,expected", noforcedata)
def test_noforce_onelayer(helpers, nchildren, expected):
    """Test basic LGL layout against igraph's internal implementation.

    NOTE: LGL places the nodes according to a tree first and relaxes a force-like
    system iteratively afterwards. Even with max_iter=0 (which is not currently
    supported in python-igraph due to a bug compared to the C core), the layout
    will not be untouched compared to the initial coordinates. Instead, it is the
    pure tree-based layout without any relaxation steps.
    """

    g = ig.Graph(edges=[(0, x + 1) for x in range(nchildren)], directed=True)

    initial_coords = np.zeros((g.vcount(), 2))

    # NOTE:This scale thing is kind of awkward, perhaps broken in networkx
    pos_ilx = ilx.layouts.large_graph_layout(
        g,
        initial_coords=initial_coords,
        max_iter=0,
        root=0,
    )

    helpers.check_generic_layout(pos_ilx)
    assert pos_ilx.shape == (g.vcount(), 2)

    pos_ig = (nchildren + 1) * np.array(expected)

    pos_ig = pd.DataFrame(pos_ig)
    pos_ig.columns = pos_ilx.columns

    np.testing.assert_allclose(
        pos_ilx.values,
        pos_ig.values,
        atol=1e-14,
    )


def test_noforce_multilayer(helpers):
    """Test LGL without forces, 2+ layers."""
    edges_1 = [(1, 3 + x) for x in range(50)]
    edges_2 = [(2, 3 + len(edges_1) + x) for x in range(50)]
    g = ig.Graph(edges=[(0, 1), (0, 2)] + edges_1 + edges_2, directed=True)

    initial_coords = np.zeros((g.vcount(), 2))

    # NOTE:This scale thing is kind of awkward, perhaps broken in networkx
    pos_ilx = ilx.layouts.large_graph_layout(
        g,
        initial_coords=initial_coords,
        max_iter=0,
        root=0,
    )

    helpers.check_generic_layout(pos_ilx)
    assert pos_ilx.shape == (g.vcount(), 2)

    np.testing.assert_allclose(
        pos_ilx.values[:3],
        68.6666666667 * np.array([[0, 0], [1, 0], [-1, 0]]),
        atol=1e-14,
    )

    # Check edges_1: the [1, 0] correction is the impulse of vertex 1
    dist_e1 = np.linalg.norm(
        pos_ilx.values[3 : 3 + len(edges_1)] - (pos_ilx.values[1] + np.array([1, 0])),
        axis=1,
    )
    np.testing.assert_allclose(
        dist_e1,
        34.32681154 * np.ones(len(edges_1)),
        rtol=1e-3,
        atol=1e-3,
    )

    # Check edges_1: the [-1, 0] correction is the impulse of vertex 2
    dist_e1 = np.linalg.norm(
        pos_ilx.values[3 + len(edges_1) :] - (pos_ilx.values[2] - np.array([1, 0])),
        axis=1,
    )
    np.testing.assert_allclose(
        dist_e1,
        34.32681154 * np.ones(len(edges_2)),
        rtol=1e-3,
        atol=1e-3,
    )


def test_large_layout(helpers):
    n = 1000
    g = ig.Graph.Erdos_Renyi(n, m=2 * n, directed=False)
    initial_coords = np.zeros((g.vcount(), 2))

    layout = ilx.layouts.large_graph_layout(
        g,
        initial_coords=initial_coords,
        max_iter=10,
        root=0,
        center=(0, 0),
        scaling=1.0,
    )

    helpers.check_generic_layout(layout)
    assert layout.shape == (g.vcount(), 2)

    # Make sure the distance between connected nodes is short
    edges = np.array(g.get_edgelist())
    nonedges = np.random.randint(0, n, size=(len(edges), 2))

    deltas = layout.values[edges[:, 1]] - layout.values[edges[:, 0]]
    dists = np.linalg.norm(deltas, axis=1)

    deltas_non = layout.values[nonedges[:, 1]] - layout.values[nonedges[:, 0]]
    dists_non = np.linalg.norm(deltas_non, axis=1)

    # Check that edges tend to be shorter than non-edges
    # This is clearer from the cumulative distrubutions, but for now an ok check
    assert (dists < dists_non).mean() > 0.7
