"""Test Attractive-repulsive forces layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    g = nx.Graph()

    layout = ilx.layouts.arf(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = nx.DiGraph()
    g.add_node(0)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.arf(g, **kwargs)
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


@pytest.mark.parametrize("max_iter", [0, 1, 10, 30, 100, 300, 1000])
def test_basic(helpers, max_iter):
    """Test basic ARF layout against NetworkX's internal implementation.

    NOTE: Numerical precision and random seeding (nx uses an old numpy rng) can cause
    small differences. We try to deal with that as well as possible here.
    """

    g = nx.path_graph(4)

    initial_coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (2.0, 1.0),
    }

    # NOTE:This scale thing is kind of awkward, perhaps broken in networkx
    pos_ilx = ilx.layouts.arf(g, initial_coords=initial_coords, max_iter=max_iter)
    pos_nx = nx.arf_layout(g, pos=initial_coords, max_iter=max_iter)
    pos_nx = pd.DataFrame({key: val for key, val in pos_nx.items()}).T
    pos_nx.columns = pos_ilx.columns

    # NOTE: For large max_iter, numerical precision can cause small differences
    np.testing.assert_allclose(
        pos_ilx.values,
        pos_nx.values,
        atol=1e-14,
    )
