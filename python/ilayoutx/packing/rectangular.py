"""Rectangular packing for disconnected graphs or sets of graphs."""

from typing import (
    Optional,
    Sequence,
)
import numpy as np
import pandas as pd
from rpack import pack as rectangular_pack_integers

from ilayoutx.utils import _recenter_layouts


def _place_multiple_layouts(
    layouts: list[pd.DataFrame],
    padding: float,
    add_ids: bool,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None,
):
    """
    Place the layouts relative to one another.

    Parameters:
        layouts: Sequence of layouts to pack. Each layout is a pandas DataFrame with 'x' and 'y'
        padding: White space between packed layouts.
    Returns:
        List of pd.DataFrame with the packed layout.
    """

    largest = 0
    dimensions = []
    xymins = []
    index_map = {}
    j = 0
    for i, layout in enumerate(layouts):
        if len(layout) == 0:
            continue
        index_map[j] = i
        j += 1

        xmin, ymin = layout[["x", "y"]].values.min(axis=0)
        xmax, ymax = layout[["x", "y"]].values.max(axis=0)
        width = xmax - xmin
        height = ymax - ymin
        largest = max(largest, width, height)
        dimensions.append((width, height))
        xymins.append((xmin, ymin))

    # rpack requires integers... scale to a reasonable default
    scaling = 1000.0 / largest
    dimensions = [
        (int((width + 0.5 * padding) * scaling), int((height + 0.5 * padding) * scaling))
        for width, height in dimensions
    ]

    if max_width is not None:
        max_width = int(max_width * scaling)
    if max_height is not None:
        max_height = int(max_height * scaling)
    lower_lefts = rectangular_pack_integers(
        dimensions,
        max_width=max_width,
        max_height=max_height,
    )

    new_layouts = []
    for j, ((llx, lly), (xmin, ymin)) in enumerate(zip(lower_lefts, xymins)):
        layout_id = index_map[j]
        layout = layouts[layout_id]

        llx = float(llx) / scaling
        lly = float(lly) / scaling
        new_layout = layout.copy()
        new_layout["x"] = new_layout["x"] - xmin + llx
        new_layout["y"] = new_layout["y"] - ymin + lly
        if add_ids:
            new_layout["id"] = new_layout.index
            new_layout["layout_id"] = layout_id
        new_layouts.append(new_layout)

    return new_layouts


def rectangular_packing(
    layouts: Sequence[pd.DataFrame],
    padding: float = 0.0,
    center: bool = True,
    concatenate: bool = True,
    max_width: Optional[float] = None,
    max_height: Optional[float] = None,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Rectangular packing of multiple layouts.

    Parameters:
        layouts: Sequence of layouts to pack. Each layout is a pandas DataFrame with 'x' and 'y'
            columns.
        padding: White space between packed layouts.
        center: Whether to center the packed layout around the origin. Otherwise, the lower_left
            corner will be at (0, 0).
        concatenate: Whether to concatenate all layouts into a single DataFrame. If False, a list
            of layouts will be returned.
        max_width: If not None, the maximum width of the packed layout.
        max_height: If not None, the maximum height of the packed layout.
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
            max_width=max_width,
            max_height=max_height,
        )

    if center is not None:
        _recenter_layouts(new_layouts, center)

    if concatenate:
        new_layouts = pd.concat(new_layouts, ignore_index=True)

    return new_layouts
