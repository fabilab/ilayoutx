from typing import Optional
from collections.abc import Hashable
import numpy as np
import pandas as pd


def _format_initial_coords(
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ],
    index: list[Hashable],
    fallback: Optional[callable] = None,
    inplace: bool = False,
) -> np.ndarray:
    if initial_coords is None:
        # This should be what the paper suggested. Note that
        # igraph uses 0.36 * np.sqrt(nv) as the radius to
        # asymptotically converge for actual circular graphs.
        initial_coords = fallback() if fallback is not None else None
    else:
        if isinstance(initial_coords, dict):
            initial_coords = pd.DataFrame(initial_coords).T.loc[index].values
        elif isinstance(initial_coords, np.ndarray):
            if not inplace:
                initial_coords = initial_coords.copy()
        elif isinstance(initial_coords, pd.DataFrame):
            initial_coords = initial_coords.loc[index].values
        else:
            raise TypeError(
                "Initial coordinates must be a numpy array, pandas DataFrame, or dict.",
            )

    return initial_coords


def _recenter_layout(
    coords: np.ndarray,
    center: tuple[float, float],
) -> None:
    """Recenter a single layout in place around a given center point.

    Parameters:
        layout: The layout to recenter. A pandas DataFrame with 'x' and 'y' columns (among others).
        center: The point to recenter the layout around.
    Returns:
        None. The input layout is modified in place.

    NOTE: The layout is recentered based on extreme values, not barycentering.
    """
    xymin = coords.min()
    xymax = coords.min()
    xycenter = 0.5 * (xymin + xymax)

    shift = np.array(center) - xycenter
    coords += shift


def _rescale_layout(
    coords: np.ndarray,
    scaling: float,
) -> None:
    """Rescale a single layout in place by a given scaling factor.

    Parameters:
        layout: The layout to rescale. A pandas DataFrame with 'x' and 'y' columns (among others).
        scaling: The scaling factor to apply.
    Returns:
        None. The input layout is modified in place.
    """
    max_xy = coords.max(axis=0)
    min_xy = coords.min(axis=0)
    deltas = max_xy - min_xy
    max_delta = deltas.max()
    if max_delta > 0:
        coords *= scaling / max_delta


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
