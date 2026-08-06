"""Test Neato (GraphViz) layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


# def test_empty(helpers):
#    g = nx.Graph()
#
#    layout = ilx.layouts.kamada_kawai(g)
#
#    helpers.check_generic_layout(layout)
#    assert layout.shape == (0, 2)
#
#
# @pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
# def test_singleton(helpers, center):
#    g = nx.Graph()
#    g.add_node(0)
#
#    kwargs = {}
#    if center is not None:
#        kwargs["center"] = center
#    layout = ilx.layouts.neato(g, **kwargs)
#    # Default center is (0, 0)
#    if center is None:
#        center = (0, 0)
#
#    helpers.check_generic_layout(layout)
#    assert layout.shape == (1, 2)
#    assert all(layout.index == list(g.nodes()))
#    np.testing.assert_allclose(
#        layout.values,
#        [center],
#        atol=1e-14,
#    )


def test_basic(helpers):
    """Test basic FA2 layout against NetworkX's internal implementation.

    NOTE: Numerical precision and random seeding (nx uses an old numpy rng) can cause
    small differences. We try to deal with that as well as possible here.
    """

    g = nx.path_graph(4)

    initial_coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 1.0),
        3: (3.0, -3.0),
    }

    pos_ilx = ilx.layouts.neato(g, initial_coords=initial_coords)

    # Check that the distances are really minimised
    edges = np.array(g.edges())
    distances_embedding = ((pos_ilx.values[edges[:, 0]] - pos_ilx.values[edges[:, 1]]) ** 2).sum(
        axis=1
    )
    # The graph is unweighted, so the expected distances are all 1.0
    distances_expected = np.ones(len(edges))

    np.testing.assert_allclose(
        distances_embedding,
        distances_expected,
        atol=1e-2,
    )
