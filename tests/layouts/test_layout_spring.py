"""Test spring layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_spring_empty(helpers):
    g = nx.Graph()

    layout = ilx.layouts.spring(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = nx.DiGraph()
    g.add_node(0)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.spring(g, **kwargs)
    # Default center is (0, 0)
    if center is None:
        center = (0, 0)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [center],
        atol=1e-14,
    )


def test_spring_basic(helpers):
    """Test basic spring layout from NetworkX."""
    g = nx.path_graph(4)

    pos_ilx = ilx.layouts.spring(g, seed=123)
    pos_nx = nx.spring_layout(g, seed=123)
    pos_nx = pd.DataFrame({key: val for key, val in pos_nx.items()}).T
    pos_nx.columns = pos_ilx.columns

    # NetworkX returns a dict of tuples, so just
    # check them one by one
    np.testing.assert_allclose(
        pos_ilx.values,
        pos_nx.values,
        atol=1e-6,
    )
