"""Rectangular packing for disconnected graphs or sets of graphs."""

from typing import (
    Sequence,
    Optional,
)
import numpy as np
import pandas as pd
import circlify


def _recenter_layouts(
    new_layouts: list[pd.DataFrame],
    center: tuple[float, float],
) -> None:
    """Recenter multiple layouts around a given center point.

    Parameters:
        new_layouts: List of layouts to recenter. Each layout is a pandas DataFrame with 'x' and 'y'
            columns (among others).
        center: The point to recenter the combined layout around.
    Returns:
        None. The input layouts are modified in place.
    """
    xymins = np.array([new_layout[["x", "y"]].values.min(axis=0) for new_layout in new_layouts])
    xymaxs = np.array([new_layout[["x", "y"]].values.max(axis=0) for new_layout in new_layouts])
    xmin, ymin = xymins.min(axis=0)
    xmax, ymax = xymaxs.max(axis=0)
    current_center = 0.5 * (np.array([xmin, ymin]) + np.array([xmax, ymax]))
    shift = np.array(center) - current_center
    for new_layout in new_layouts:
        new_layout["x"] += shift[0]
        new_layout["y"] += shift[1]


def _place_multiple_layouts(
    layouts: Sequence[pd.DataFrame],
    padding: float,
    add_ids: bool,
) -> list[pd.DataFrame]:
    """Place the layouts relative to one another.

    Parameters:
        layouts: Sequence of layouts to pack. Each layout is a pandas DataFrame with 'x' and 'y'
        padding: White space between packed layouts.
    Returns:
        List of pd.DataFrame with the packed layout.
    """
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
    areas = np.array(areas)

    if padding is None:
        nondegenerate_areas = areas[areas > 0.0]
        if len(nondegenerate_areas) == 0:
            padding = 0.0
        else:
            min_r2 = nondegenerate_areas.min()
            padding = 0.16 * min_r2

    areas = (np.sqrt(areas) + 0.5 * padding) ** 2

    # circlify both requests and spits out lists ordered by area (descending)
    # so we need to keep track of the original indices
    idx_descending = np.argsort(-areas)
    idx_ranks = np.zeros_like(idx_descending)
    idx_ranks[idx_descending] = np.arange(len(areas))

    # NOTE: The resulting circles have areas *proportional* to the input areas,
    # we have to rescale them to the original areas.
    circles = circlify.circlify(list(areas[idx_descending]), show_enclosure=False)

    # The circles are sorted by hierarchy and size, so we need to reorder them
    circles = [circles[idx_ranks[i]] for i in range(len(circles))]

    # NOTE: all ratios are the same so we just take the first one
    scaling = np.sqrt(areas[0]) / circles[0].r

    new_layouts = []
    for j, (ctr, circ) in enumerate(zip(centers, circles)):
        layout_id = index_map[j]
        layout = layouts[layout_id]

        xctr = circ.x * scaling
        yctr = circ.y * scaling

        new_layout = layout.copy()
        new_layout["x"] += xctr - ctr[0]
        new_layout["y"] += yctr - ctr[1]
        if add_ids:
            new_layout["id"] = new_layout.index
            new_layout["layout_id"] = layout_id
        new_layouts.append(new_layout)

    return new_layouts


def circular_packing(
    layouts: Sequence[pd.DataFrame],
    padding: Optional[float] = None,
    center: Optional[tuple[float, float]] = None,
    concatenate: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Rectangular packing of multiple layouts.

    Parameters:
        layouts: Sequence of layouts to pack. Each layout is a pandas DataFrame with 'x' and 'y'
            columns.
        padding: White space between packed layouts. None uses 10% of the smallest nondegenerate
            layout.
        center: If not None, recenter the combined layout around this point. If None, the lower
            left corner will be at (0, 0).
        concatenate: Whether to concatenate all layouts into a single DataFrame. If False, a list
            of layouts will be returned.
    Returns:
        DataFrame or list of DataFrames with the packed layout. If concatenate is True, the
        concatenated object has two additional columns: 'layout_id' to indicate which layout
        each node belongs to (indexed from 0), and 'id' which is the previous index.
    """
    if len(layouts) == 0:
        if concatenate:
            # DataFrames can only be initialised with a single dtype
            # but for consistency we want 'x' and 'y' to be float64
            df = pd.DataFrame(
                columns=["x", "y", "id", "layout_id"],
            )
            df["x"] = df["x"].astype(np.float64)
            df["y"] = df["y"].astype(np.float64)
            return df
        else:
            return []

    if len(layouts) == 1:
        new_layout = layouts[0].copy()
        if concatenate:
            new_layout["id"] = new_layout.index
            new_layout["layout_id"] = 0
        new_layouts = [new_layout]
    else:
        new_layouts = _place_multiple_layouts(
            layouts,
            padding,
            add_ids=concatenate,
        )

    if center is not None:
        _recenter_layouts(new_layouts, center)

    if concatenate:
        return pd.concat(new_layouts, ignore_index=True)
    else:
        return new_layouts
