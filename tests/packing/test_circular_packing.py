"""Test circular packing."""

import pytest
import numpy as np
import pandas as pd

import ilayoutx as ilx

nx = pytest.importorskip("networkx")


def test_circular_packing_empty(helpers):
    """Test empty list of layouts."""
    empty_df = ilx.packing.circular([])
    helpers.check_generic_packing_concatenate(empty_df)

    empty_list = ilx.packing.circular([], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(empty_list)


def test_circular_packing_singleton(helpers):
    """Test singleton list of layouts."""
    g = nx.path_graph(1)
    layout = ilx.layouts.line(g)

    packing_df = ilx.packing.circular([layout])
    helpers.check_generic_packing_concatenate(packing_df)

    packing_list = ilx.packing.circular([layout], concatenate=False)
    helpers.check_generic_packing_nonconcatenate(packing_list)
