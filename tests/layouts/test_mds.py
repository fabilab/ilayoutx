"""Test GEM layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_mds_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.multidimensional_scaling(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_mds_singleton(helpers, center):
    g = ig.Graph(n=1)

    kwargs = {}
    if center is not None:
        kwargs["center"] = center
    layout = ilx.layouts.multidimensional_scaling(g, **kwargs)
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


def test_mds_disconnected(helpers):
    """Test exception for disconnected graphs."""

    ilx.layouts.multidimensional_scaling(ig.Graph(n=0), check_connectedness=True)
    ilx.layouts.multidimensional_scaling(ig.Graph(n=1), check_connectedness=True)
    with pytest.raises(ValueError):
        ilx.layouts.multidimensional_scaling(ig.Graph(n=2), check_connectedness=True)
