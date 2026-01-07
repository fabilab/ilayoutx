"""Test GEM layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_gem_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.graph_embedder(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_gem_singleton(helpers, center):
    g = ig.Graph(n=1)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.graph_embedder(g, **kwargs)
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
