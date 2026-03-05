from typing import (
    Optional,
)
from collections.abc import (
    Hashable,
)
import numpy as np
import pandas as pd

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import (
    _format_initial_coords,
    _recenter_layout,
)
from ..external.networkx.arf import (
    arf_networkx,
)
from ilayoutx._ilayoutx import (
    random as random_rust,
)


def rectangular_dendrogram(
    network,
    root: Hashable,
    center: Optional[tuple[float, float]] = None,
    orientation: str = "right",
) -> pd.DataFrame:
    """Rectangular dendrogram layout algorithm.

    parameters:
        network: the network to layout.
        root: the node to use as the root of the dendrogram.
        center: if not none, recenter the layout around this point.
        orientation: the orientation of the dendrogram. one of "right", "left", "up", or "down".
    returns:
        the layout of the network.

    """
    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=np.float64)

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        coords = np.zeros((nv, 2), dtype=np.float64)

        hierarchy = provider.bfs(root)
        hierarchy_df = pd.DataFrame(
            {
                "vertices": hierarchy["vertices"],
                "parents": hierarchy["parents"],
                "layer": -np.ones(nv, np.int64),
            }
        )
        for layer_switch in hierarchy["layer_switch"]:
            hierarchy_df.loc[hierarchy_df.index >= layer_switch, "layer"] += 1
        hierarchy_df["seen"] = False

        hierarchy_df = hierarchy_df.set_index("vertices", drop=False).loc[index]
        hierarchy_df["index"] = np.arange(nv)
        hierarchy_df["y"] = 0.0

        # Assume right orientation, then modify as needed
        coords[:, 0] = hierarchy_df["layer"].astype(np.float64)

        # Assign the y coordinate of the leaves
        leaves = list(set(hierarchy["vertices"]) - set(hierarchy["parents"]))
        print(leaves)
        print(hierarchy_df)
        hierarchy_df.loc[leaves, "seen"] = True
        hierarchy_df.loc[leaves, "y"] = np.arange(len(leaves))

        # Group by layer and parent, from high to low layers, and assign the
        # y coordinate as average of the children
        hierarchy_df["neglayer"] = -hierarchy_df["layer"]
        for layer, hierarchy_df_layer in hierarchy_df.groupby("neglayer"):
            # The root is always (0, 0)
            if layer == -1:
                break
            yparents = hierarchy_df_layer.groupby("parents")["y"].mean()
            parents = yparents.index
            hierarchy_df.loc[parents, "seen"] = True
            hierarchy_df.loc[parents, "y"] = yparents.values

        coords[:, 1] = hierarchy_df["y"].values

        if orientation == "left":
            coords[:, 0] *= -1
        elif orientation == "up":
            coords = coords[:, ::-1]
        elif orientation == "down":
            coords = coords[:, ::-1]
            coords[:, 1] *= -1

    if center is not None:
        _recenter_layout(coords, center)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout


def circular_dendrogram(
    network,
    root: Hashable,
    center: Optional[tuple[float, float]] = None,
    orientation: str = "right",
    theta: float = 0.0,
) -> pd.DataFrame:
    """Circular dendrogram layout algorithm.

    parameters:
        network: the network to layout.
        root: the node to use as the root of the dendrogram.
        center: if not none, recenter the layout around this point.
        orientation: the orientation of the dendrogram. one of "right" or "left".
        theta: the angle to rotate the layout by, in radians. theta=0 corresponds
            to the root being on the right, and for "left", theta=0 corresponds to the root being on the left.
    returns:
        the layout of the network.

    """
    layout = rectangular_dendrogram(network, root, center=None, orientation="right")
    if len(layout) == 0:
        return layout

    if len(layout) > 1:
        layout.rename(columns={"x": "r", "y": "angle"}, inplace=True)
        layout["angle"] *= -2 * np.pi / (layout["angle"].max() + 1)
        if orientation == "left":
            layout["angle"] *= -1
        layout["x"] = layout["r"] * np.cos(layout["angle"] + theta)
        layout["y"] = layout["r"] * np.sin(layout["angle"] + theta)
        del layout["r"]
        del layout["angle"]

    if center is not None:
        coords = np.array(layout[["x", "y"]], dtype=np.float64)
        _recenter_layout(coords, center)
        layout[["x", "y"]] = coords

    return layout
