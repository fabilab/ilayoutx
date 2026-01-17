"""Test geometric layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.geometric(g, edge_lengths={})

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = ig.Graph(n=1)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.geometric(g, {}, **kwargs)
    # Default center is (0, 0)
    if center is None:
        center = (0, 0)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(range(g.vcount())))
    np.testing.assert_allclose(
        layout.values,
        [center],
        atol=1e-14,
    )


def test_ring(helpers):
    """Test geometric layout on a small ring."""
    g = ig.Graph.Ring(3)

    layout = ilx.layouts.geometric(
        g,
        edge_lengths={
            (0, 1): 10,
            (1, 2): 1,
            (2, 0): 10,
        },
        seed=42,
    )

    helpers.check_generic_layout(layout)
    assert layout.shape == (3, 2)
    assert all(layout.index == list(range(g.vcount())))

    expected = np.array(
        [
            [1.139334, 6.560126],
            [-0.077041, -3.365620],
            [-1.062293, -3.194506],
        ]
    )

    np.testing.assert_allclose(
        layout.values,
        expected,
        rtol=1e-4,
    )
