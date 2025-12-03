"""Rectangular packing for disconnected graphs or sets of graphs."""

from typing import Sequence
import pandas as pd
from rpack import pack


def rectangular_packing(
    layouts: Sequence[pd.DataFrame],
    padding: float = 10.0,
    center: bool = True,
    concatenate: bool = True,
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
    Returns:
        DataFrame or list of DataFrames with the packed layout. If concatenate is True, the
        concatenated object has two additional columns: 'layout_id' to indicate which layout
        each node belongs to (indexed from 0), and 'id' which is the previous index.
    """
    # TODO: deal with empty layouts

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
        width = (xmax - xmin) + 0.5 * padding
        height = (ymax - ymin) + 0.5 * padding
        largest = max(largest, width, height)
        dimensions.append((width, height))
        xymins.append((xmin, ymin))

    # rpack requires integers... scale to a reasonable default
    scaling = 1000.0 / largest
    dimensions = [(int(width * scaling), int(height * scaling)) for width, height in dimensions]

    print(dimensions)
    lower_lefts = pack(dimensions)
    new_layouts = []
    for j, ((llx, lly), (xmin, ymin)) in enumerate(zip(lower_lefts, xymins)):
        layout_id = index_map[j]
        layout = layouts[layout_id]

        llx = float(llx) / scaling
        lly = float(lly) / scaling
        new_layout = layout.copy()
        new_layout["x"] = new_layout["x"] - xmin + llx
        new_layout["y"] = new_layout["y"] - ymin + lly
        if concatenate:
            new_layout["id"] = new_layout.index
            new_layout["layout_id"] = layout_id
        new_layouts.append(new_layout)

    if concatenate:
        return pd.concat(new_layouts, ignore_index=True)
    else:
        return new_layouts
