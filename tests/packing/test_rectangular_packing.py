"""Test rectangular packing."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_empty(helpers):
    """Test empty list of layouts."""
    empty_df = ilx.packing.rectangular([])
    helpers.check_generic_packing_concatenate(empty_df)

    empty_list = ilx.packing.rectangular([], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(empty_list)


def test_singleton(helpers):
    """Test singleton list of layouts."""
    g = nx.path_graph(1)
    layout = ilx.layouts.line(g)

    packing_df = ilx.packing.rectangular([layout])
    helpers.check_generic_packing_concatenate(packing_df)

    packing_list = ilx.packing.rectangular([layout], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(packing_list)


diamond_data = [
    (
        9,
        6,
        6,
        [
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-3.0, -2.0],
            [-2.0, -3.0],
            [-1.0, 0.0],
            [-2.0, 1.0],
            [-3.0, 0.0],
            [-2.0, -1.0],
            [-1.0, 2.0],
            [-2.0, 3.0],
            [-3.0, 2.0],
            [-2.0, 1.0],
            [1.0, -2.0],
            [0.0, -1.0],
            [-1.0, -2.0],
            [0.0, -3.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 2.0],
            [0.0, 3.0],
            [-1.0, 2.0],
            [0.0, 1.0],
            [3.0, -2.0],
            [2.0, -1.0],
            [1.0, -2.0],
            [2.0, -3.0],
            [3.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, -1.0],
            [3.0, 2.0],
            [2.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ],
    )
]


@pytest.mark.parametrize("ndiamonds,max_width,max_height,expected", diamond_data)
def test_basic(helpers, ndiamonds, max_width, max_height, expected):
    g = nx.circulant_graph(4, [1])
    layout = ilx.layouts.circle(g)

    packing_df = ilx.packing.rectangular(
        [layout] * ndiamonds,
        concatenate=True,
        padding=0,
        max_width=max_width,
        max_height=max_height,
        center=(0, 0),
    )
    helpers.check_generic_packing_concatenate(packing_df)

    # NOTE: Depending on the platform, a mirror symmetry may occur. Check for it here
    # Both are ok answers since rectangular packing is itself invariant under rotation/reflection
    # Bottom line is if we change x, y should follow
    expected = np.array(expected)
    diff = packing_df["x"].values - expected[:, 0]
    idx_diff = np.abs(diff) > 1
    expected[idx_diff] = expected[idx_diff] + diff[idx_diff][:, np.newaxis]

    assert packing_df.shape == (4 * ndiamonds, 4)
    np.testing.assert_allclose(
        packing_df[["x", "y"]].values,
        expected,
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.mark.parametrize("ndiamonds,max_width,max_height,expected", diamond_data)
def test_basic_nonconcat(helpers, ndiamonds, max_width, max_height, expected):
    g = nx.circulant_graph(4, [1])
    layout = ilx.layouts.circle(g)

    packing_list = ilx.packing.rectangular(
        [layout] * ndiamonds,
        concatenate=False,
        padding=0,
        max_width=max_width,
        max_height=max_height,
        center=(0, 0),
    )
    helpers.check_generic_packing_nonconcatenate(packing_list)
    i = 0
    for layout in packing_list:
        nv = len(layout)

        # NOTE: Depending on the platform, a mirror symmetry may occur. Check for it here
        # Both are ok answers since rectangular packing is itself invariant under rotation/reflection
        # Bottom line is if we change x, y should follow
        expectedi = np.array(expected[i : i + nv])
        diff = layout["x"].values - expectedi[:, 0]
        idx_diff = np.abs(diff) > 1
        expectedi[idx_diff] = expectedi[idx_diff] + diff[idx_diff][:, np.newaxis]

        np.testing.assert_allclose(
            layout.values,
            expectedi,
            rtol=1e-7,
            atol=1e-7,
        )
        i += nv
