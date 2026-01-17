"""Test GEM layouts."""

import platform
import pytest
import numpy as np

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.multidimensional_scaling(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
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


def test_disconnected(helpers):
    """Test exception for disconnected graphs."""

    ilx.layouts.multidimensional_scaling(ig.Graph(n=0), check_connectedness=True)
    ilx.layouts.multidimensional_scaling(ig.Graph(n=1), check_connectedness=True)
    with pytest.raises(ValueError):
        ilx.layouts.multidimensional_scaling(ig.Graph(n=2), check_connectedness=True)


@pytest.mark.skipif(
    (platform.system() != "Linux") or (platform.machine() != "x86_64"),
    reason="MDS layout is OS-dependent, tests for Linux amd64 only",
)
@pytest.mark.parametrize("nv,radius", [(4, 1.0), (9, 2.0), (14, 3.177), (17, 3.815), (30, 6.765)])
def test_ring(helpers, nv, radius):
    """Test MDS on a circulant graph."""
    g = ig.Graph.Ring(nv)

    layout = ilx.layouts.multidimensional_scaling(g, center=(0, 0))

    helpers.check_generic_layout(layout)
    assert layout.shape == (nv, 2)
    assert all(layout.index == list(range(g.vcount())))

    # Check that nodes are on a circle
    radii = np.linalg.norm(layout.values, axis=1)
    np.testing.assert_allclose(
        radii,
        radius * np.ones(nv),
        rtol=2e-3 if nv > 9 else 1e-2,
    )
