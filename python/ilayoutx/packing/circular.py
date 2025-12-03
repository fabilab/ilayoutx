"""Rectangular packing for disconnected graphs or sets of graphs."""

from typing import (
    Sequence,
    Optional,
)
import numpy as np
import pandas as pd
import circlify


def circular_packing(
    layouts: Sequence[pd.DataFrame],
    padding: Optional[float] = None,
    center: bool = True,
    concatenate: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Rectangular packing of multiple layouts.

    Parameters:
        layouts: Sequence of layouts to pack. Each layout is a pandas DataFrame with 'x' and 'y'
            columns.
        padding: White space between packed layouts. None uses 10% of the smallest nondegenerate
            layout.
        center: Whether to center the packed layout around the origin. Otherwise, the lower_left
            corner will be at (0, 0).
        concatenate: Whether to concatenate all layouts into a single DataFrame. If False, a list
            of layouts will be returned.
    Returns:
        DataFrame or list of DataFrames with the packed layout. If concatenate is True, the
        concatenated object has two additional columns: 'layout_id' to indicate which layout
        each node belongs to (indexed from 0), and 'id' which is the previous index.
    """
    if len(layouts) == 0:
        if concatenate:
            return pd.DataFrame(columns=["x", "y", "id", "layout_id"])
        else:
            return []

    centers = []
    areas = []
    index_map = {}
    j = 0
    for i, layout in enumerate(layouts):
        if len(layout) == 0:
            continue
        index_map[j] = i
        j += 1

        xmin, ymin = layout[["x", "y"]].values.min(axis=0)
        xmax, ymax = layout[["x", "y"]].values.max(axis=0)
        xctr = 0.5 * (xmin + xmax)
        yctr = 0.5 * (ymin + ymax)
        ctr = np.array([xctr, yctr])
        centers.append(ctr)
        r2max = ((layout[["x", "y"]].values - ctr) ** 2).sum(axis=1).max()
        areas.append(r2max)

    if padding is None:
        nondegenerate_areas = [a for a in areas if a > 0.0]
        if len(nondegenerate_areas) == 0:
            padding = 0.0
        else:
            min_r2 = min(nondegenerate_areas)
            padding = 0.16 * min_r2
    if padding > 0.0:
        areas = [a + padding for a in areas]

    print(areas)
    # NOTE: The resulting circles have areas *proportional* to the input areas,
    # we have to rescale them to the original areas.
    circles = circlify.circlify(areas, show_enclosure=False)

    scaling = areas[0] / circles[0].r ** 2

    new_layouts = []
    for j, (ctr, circ) in enumerate(zip(centers, circles)):
        layout_id = index_map[j]
        layout = layouts[layout_id]

        xctr = circ.x * scaling
        yctr = circ.y * scaling

        new_layout = layout.copy()
        new_layout["x"] += xctr - ctr[0]
        new_layout["y"] += yctr - ctr[1]
        if concatenate:
            new_layout["id"] = new_layout.index
            new_layout["layout_id"] = layout_id
        new_layouts.append(new_layout)

    if concatenate:
        return pd.concat(new_layouts, ignore_index=True)
    else:
        return new_layouts
