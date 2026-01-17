"""Test circular packing."""

import pytest
import numpy as np

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    """Test empty list of layouts."""
    empty_df = ilx.packing.circular([])
    helpers.check_generic_packing_concatenate(empty_df)

    empty_list = ilx.packing.circular([], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(empty_list)


def test_singleton(helpers):
    """Test singleton list of layouts."""
    g = nx.path_graph(1)
    layout = ilx.layouts.line(g)

    packing_df = ilx.packing.circular([layout])
    helpers.check_generic_packing_concatenate(packing_df)

    packing_list = ilx.packing.circular([layout], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(packing_list)


diamond_data = [
    (
        2,
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
            [-1.0, 1.0],
            [-2.0, 0.0],
            [-1.0, -1.0],
        ],
    ),
    (
        7,
        [
            [2.0, -1.73205081],
            [1.0, -0.73205081],
            [0.0, -1.73205081],
            [1.0, -2.73205081],
            [0.0, -1.73205081],
            [-1.0, -0.73205081],
            [-2.0, -1.73205081],
            [-1.0, -2.73205081],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-0.0, -1.0],
            [3.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, -1.0],
            [-1.0, 0.0],
            [-2.0, 1.0],
            [-3.0, 0.0],
            [-2.0, -1.0],
            [2.0, 1.73205081],
            [1.0, 2.73205081],
            [0.0, 1.73205081],
            [1.0, 0.73205081],
            [-0.0, 1.73205081],
            [-1.0, 2.73205081],
            [-2.0, 1.73205081],
            [-1.0, 0.73205081],
        ],
    ),
]


@pytest.mark.parametrize("ndiamonds,expected", diamond_data)
def test_basic(helpers, ndiamonds, expected):
    g = nx.circulant_graph(4, [1])
    layout = ilx.layouts.circle(g)

    packing_df = ilx.packing.circular([layout] * ndiamonds, concatenate=True, padding=0)
    helpers.check_generic_packing_concatenate(packing_df)

    assert packing_df.shape == (4 * ndiamonds, 4)

    np.testing.assert_allclose(
        packing_df[["x", "y"]].values,
        expected,
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("ndiamonds,expected", diamond_data)
def test_basic_nonconcat(helpers, ndiamonds, expected):
    g = nx.circulant_graph(4, [1])
    layout = ilx.layouts.circle(g)

    packing_list = ilx.packing.circular([layout] * ndiamonds, concatenate=False, padding=0)
    helpers.check_generic_packing_nonconcatenate(packing_list)
    i = 0
    for layout in packing_list:
        nv = len(layout)
        np.testing.assert_allclose(
            layout.values,
            expected[i : i + nv],
            rtol=1e-7,
            atol=1e-7,
        )
        i += nv
