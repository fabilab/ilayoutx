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
            pass
        elif isinstance(initial_coords, pd.DataFrame):
            initial_coords = initial_coords.loc[index].values
        else:
            raise TypeError(
                "Initial coordinates must be a numpy array, pandas DataFrame, or dict.",
            )

    return initial_coords
