"""Test sugiyama layout."""

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
def test_longer_chains(helpers, length):
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


def test_inverted_y(helpers):
    """Test a single inverted Y."""
    g = nx.from_edgelist([(0, 1), (0, 2)], create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (3, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0.5, 0], [0, 1], [1, 1]],
        atol=1e-14,
    )


def test_upright_y(helpers):
    """Test a single upright Y."""
    g = nx.from_edgelist([(0, 1), (2, 1)], create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (3, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        [[0, 0], [0.5, 1], [1, 0]],
        atol=1e-14,
    )


various_trees_data = [
    (
        [(0, 1), (0, 2), (1, 3), (1, 4)],
        [[0.5, 0], [0, 1], [1, 1], [-0.5, 2], [0.5, 2]],
    ),
    (
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        [[1.5, 0], [0.5, 1], [2.5, 1], [0, 2], [1, 2], [2, 2], [3, 2]],
    ),
]


@pytest.mark.parametrize("edgelist,expected", various_trees_data)
def test_double_branching(helpers, edgelist, expected):
    """Various branching trees."""
    g = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (len(g.nodes()), 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )


various_dags_data = [
    # Diamond
    (
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        [[0.5, 0], [0, 1], [1, 1], [0.5, 2]],
    ),
    # Double diamond
    (
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)],
        [[0.5, 0], [0, 1], [1, 1], [0.5, 2], [0, 3], [1, 3], [0.5, 4]],
    ),
]


@pytest.mark.parametrize("edgelist,expected", various_dags_data)
def test_dags(helpers, edgelist, expected):
    """Various branching trees."""
    g = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
    layout, waypoints = ilx.layouts.sugiyama(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (len(g.nodes()), 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected,
        atol=1e-14,
    )
