"""Test ForceAtlas2 layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    g = nx.Graph()

    layout = ilx.layouts.forceatlas2(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
    g = nx.DiGraph()
    g.add_node(0)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.forceatlas2(g, **kwargs)
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
    """Test basic FA2 layout against NetworkX's internal implementation.

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

    pos_ilx = ilx.layouts.forceatlas2(g, initial_coords=initial_coords, max_iter=max_iter)

    # networkx bug https://github.com/networkx/networkx/pull/8451
    initial_coords = {key: np.array(val, dtype=np.float64) for key, val in initial_coords.items()}

    pos_nx = nx.forceatlas2_layout(g, pos=initial_coords, max_iter=max_iter)
    pos_nx = pd.DataFrame({key: val for key, val in pos_nx.items()}).T
    pos_nx.columns = pos_ilx.columns

    np.testing.assert_allclose(
        pos_ilx.values,
        pos_nx.values,
        atol=1e-14,
    )
