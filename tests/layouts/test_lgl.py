"""Test LGL layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


ilx.layouts.large_graph_layout = ilx.experimental.layouts.large_graph_layout


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
def test_noforce(helpers, nchildren, expected):
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

    pos_ig = (nchildren + 1) * np.array(expected)

    pos_ig = pd.DataFrame(pos_ig)
    pos_ig.columns = pos_ilx.columns

    # TODO: No way this actually works
    np.testing.assert_allclose(
        pos_ilx.values,
        pos_ig.values,
        atol=1e-14,
    )
