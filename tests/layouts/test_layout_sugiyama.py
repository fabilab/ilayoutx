"""Test Sugiyama layout."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    g = nx.DiGraph()
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


def test_singleton(helpers):
    g = nx.DiGraph()
    g.add_node(0)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0, 0]],
        atol=1e-14,
    )


@pytest.mark.parametrize("ncomponents,hgap", [(2, 0.5), (2, 1.0), (3, 0.3), (7, 2.3)])
def test_two_singletons(helpers, ncomponents, hgap):
    g = nx.DiGraph()
    g.add_nodes_from(list(range(ncomponents)))
    layout, waypoints = ilx.layouts.sugiyama(g, hgap=hgap)

    helpers.check_generic_layout(layout)
    assert layout.shape == (ncomponents, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[hgap * i, 0] for i in range(ncomponents)],
        atol=1e-14,
    )


def test_two_node_chain(helpers):
    g = nx.from_edgelist([(0, 1)], create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (2, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0, 0], [0, 1]],
        atol=1e-14,
    )


@pytest.mark.parametrize("length", [3, 4, 5, 6, 7, 8, 9, 10])
def longer_chains(helpers, length):
    edgelist = [(i, i + 1) for i in range(length - 1)]
    g = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (length, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0, i] for i in range(length)],
        atol=1e-14,
    )
