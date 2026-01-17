"""Test GEM layouts."""

import pytest
import numpy as np

import ilayoutx as ilx

ig = pytest.importorskip("igraph")


def test_empty(helpers):
    g = ig.Graph()

    layout = ilx.layouts.graph_embedder(g)

    helpers.check_generic_layout(layout)
    assert layout.shape == (0, 2)


@pytest.mark.parametrize("center", [None, (0, 0), (1, 2.0)])
def test_singleton(helpers, center):
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


@pytest.mark.parametrize(
    "max_iter, n_matches_expected",
    [
        (0, 4),
        (1, 3),
        (2, 2),
        (3, 1),
        # Default is 40 * nv^2, which is much higher than nv and therefore sufficient
        # to move all nodes.
        (None, 0),
    ],
)
def test_losing_matches(helpers, max_iter, n_matches_expected):
    """Test basic ARF layout against NetworkX's internal implementation.

    NOTE: Numerical precision and random seeding (nx uses an old numpy rng) can cause
    small differences. We try to deal with that as well as possible here.
    """

    g = ig.Graph.Lattice([4])

    initial_coords = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (2.0, 1.0),
    }

    # NOTE:This scale thing is kind of awkward, perhaps broken in networkx
    pos_ilx = ilx.layouts.graph_embedder(
        g, initial_coords=initial_coords, max_iter=max_iter, center=None
    )

    expected = np.array([initial_coords[i] for i in range(4)])

    # Each GEM iteration moves a single vertex, and they are all different
    # until iterations reach nv, so we should lose matches from initial coords
    # as we slowly increase the number of iterations.
    n_matches_found = 0
    for i in range(len(initial_coords)):
        if np.allclose(pos_ilx.values[i], expected[i], atol=1e-14):
            n_matches_found += 1
    assert n_matches_found == n_matches_expected
