"""Test dendrogram layouts."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")
ig = pytest.importorskip("igraph")


layout_kinds = ["rectangular", "circular"]


@pytest.mark.parametrize("center", [None, (1, 1)])
@pytest.mark.parametrize("orientation", ["right", "left", "up", "down"])
@pytest.mark.parametrize("layout_kind", layout_kinds)
def test_dendrogram_empty(helpers, layout_kind, orientation, center):
    """Test dendrogram layouts on an empty graph."""
    g = nx.Graph()

    layout_fn = getattr(ilx.layouts, f"{layout_kind}_dendrogram")
    layout = layout_fn(g, root=0, orientation=orientation, center=center)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (1, 1)])
@pytest.mark.parametrize("orientation", ["right", "left", "up", "down"])
@pytest.mark.parametrize("layout_kind", layout_kinds)
def test_dendrogram_singleton(helpers, layout_kind, orientation, center):
    """Test dendrogram layouts on a singleton graph."""
    g = nx.Graph()
    g.add_node(0)

    layout_fn = getattr(ilx.layouts, f"{layout_kind}_dendrogram")
    layout = layout_fn(g, root=0, orientation=orientation, center=center)

    expected_layout = np.array([[0, 0]], dtype=np.float64)
    if center is not None:
        expected_layout += np.array(center, dtype=np.float64)

    helpers.check_generic_layout(layout)
    assert layout.shape == (1, 2)
    assert all(layout.index == list(g.nodes()))
    np.testing.assert_allclose(
        layout.values,
        expected_layout,
        atol=1e-14,
    )


@pytest.mark.parametrize("orientation", ["right", "left"])
@pytest.mark.parametrize("layout_kind", layout_kinds)
def test_dendrogram_tree(helpers, layout_kind, orientation):
    """Test dendrogram layouts on small path graphs of various sizes."""
    g = ig.Graph(
        edges=[
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 6),
            (3, 7),
            (3, 8),
        ]
    )

    layout_fn = getattr(ilx.layouts, f"{layout_kind}_dendrogram")
    layout = layout_fn(g, root=0, orientation=orientation)

    expected_layout = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0.5],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ],
        dtype=np.float64,
    )
    if layout_kind == "circular":
        radius = expected_layout[:, 0].copy()
        angles = -expected_layout[:, 1] * 2 * np.pi / (expected_layout[:, 1].max() + 1)
        if orientation == "left":
            angles *= -1
        expected_layout[:, 0] = radius * np.cos(angles)
        expected_layout[:, 1] = radius * np.sin(angles)
        del radius, angles
    elif orientation == "left":
        expected_layout[:, 0] *= -1

    expected_layout = pd.DataFrame(
        expected_layout,
        index=g.vs.indices,
        columns=pd.Index(["x", "y"]),
    )

    helpers.check_generic_layout(layout)
    assert layout.shape == (g.vcount(), 2)
    assert all(layout.index == list(range(g.vcount())))
    np.testing.assert_allclose(
        layout.values,
        expected_layout.values,
        atol=1e-14,
    )
