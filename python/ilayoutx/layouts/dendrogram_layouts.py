from typing import (
    Sequence,
    Optional,
)
from collections.abc import (
    Hashable,
)
from collections import deque
import numpy as np
import pandas as pd

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import (
    _recenter_layout,
)


def _dendrogram_from_hierarchy(
    hierarchy_df: pd.DataFrame,
    coords: np.ndarray,
    index: Optional[np.ndarray | pd.Index] = None,
) -> None:
    """Build a right oriented rectangular dendrogram from a nondegenerate hierarchy dataframe.

    Parameters:
        hierarchy_df: a dataframe with columns "vertices", "parents", "layer", and "depth".
        coords: a numpy array of shape (nv, 2) to write the coordinates to.
    Returns:
        None. The coordinates are written to the coords array in place.
    """

    nv = len(hierarchy_df)
    hierarchy_df["seen"] = False

    hierarchy_df = hierarchy_df.set_index("vertices", drop=False).loc[index]
    hierarchy_df["index"] = np.arange(nv)
    hierarchy_df["y"] = 0.0

    # Assume right orientation, then modify as needed
    coords[:, 0] = hierarchy_df["depth"]

    # Assign the y coordinate of the leaves
    leaves = list(set(hierarchy_df["vertices"].values) - set(hierarchy_df["parents"].values))
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
    del hierarchy_df["seen"]
    del hierarchy_df["neglayer"]
    del hierarchy_df["index"]
    del hierarchy_df["y"]


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

        # NOTE: We should support arbitrar edge lengths, but then it's unclear a
        # simple bfs traversal would be sufficient (as opposed to some minimal
        # spanning tree).
        hierarchy_df["depth"] = hierarchy_df["layer"].astype(np.float64)

        _dendrogram_from_hierarchy(hierarchy_df, coords, index)

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


def _hierarchy_from_linkage(
    linkage: np.ndarray | pd.DataFrame,
) -> pd.DataFrame:
    """Build a hierarchy dataframe from a linkage matrix.

    Parameters:
        linkage: a linkage matrix in the format of scipy's hierarchical clustering.
    Returns:
        pd.DataFrame with columns "vertices", "parents", "layer", and "depth".
    """
    nv = len(linkage) + 1

    # NOTE: linkage starts from the leaves and ends with the root, keep it for now
    hierarchy_df = pd.DataFrame(
        {
            "vertices": np.arange(2 * nv - 1),
        }
    )
    hierarchy_df["parents"] = -1
    hierarchy_df["depth"] = 0.0
    childrend = {}
    for parent_idx, (vertex_idx1, vertex_idx2, distance) in enumerate(linkage[:, :3], nv):
        hierarchy_df.at[vertex_idx1, "parents"] = parent_idx
        hierarchy_df.at[vertex_idx2, "parents"] = parent_idx

        childrend[parent_idx] = (vertex_idx1, vertex_idx2)

        # Depth
        depth_max = max(
            hierarchy_df.at[vertex_idx1, "depth"], hierarchy_df.at[vertex_idx2, "depth"]
        )
        delta_vertices = abs(
            hierarchy_df.at[vertex_idx1, "depth"] - hierarchy_df.at[vertex_idx2, "depth"]
        )
        delta_from_max = 0.5 * (distance - delta_vertices)
        hierarchy_df.at[parent_idx, "depth"] = depth_max + delta_from_max

    # Measure depth from the root now
    hierarchy_df["depth"] = hierarchy_df["depth"].max() - hierarchy_df["depth"]

    # Set the layer from the root
    hierarchy_df["layer"] = 0
    node_idx_queue = deque([hierarchy_df.index[-1]])
    while node_idx_queue:
        node_idx = node_idx_queue.popleft()
        children = childrend.pop(node_idx, tuple())
        for child in children:
            hierarchy_df.at[child, "layer"] = hierarchy_df.at[node_idx, "layer"] + 1
            node_idx_queue.append(child)
    del childrend, node_idx_queue

    # Reverse and reindex to start from the root
    hierarchy_df = hierarchy_df.iloc[::-1].reset_index(drop=True)

    return hierarchy_df


def _compute_waypoints_for_edge(
    hierarchy_df: pd.DataFrame,
    edge: tuple[Hashable, Hashable],
    hierarchy_coords: np.ndarray,
):
    """Compute waypoints for a single edge based on the hierarchy.

    Parameters:
        hierarchy_df: a dataframe with columns "vertices", "parents", "layer", and "depth".
        edge: a pair of vertices (source, target) representing an edge in the network.
        hierarchy_coords: a numpy array containing the coordinates of the vertices, including the internal nodes.
    Returns:
        A list of waypoints along the path from the source to the target, excluding the source and target themselves.
    """
    source, target = edge
    if source == target:
        return []

    head = []
    tail = []
    while source != target:
        if hierarchy_df.at[source, "layer"] > hierarchy_df.at[target, "layer"]:
            head.append(source)
            source = hierarchy_df.at[source, "parents"]
        elif hierarchy_df.at[source, "layer"] < hierarchy_df.at[target, "layer"]:
            tail.append(target)
            target = hierarchy_df.at[target, "parents"]
        else:
            head.append(source)
            tail.append(target)
            source = hierarchy_df.at[source, "parents"]
            target = hierarchy_df.at[target, "parents"]
    waypoint_vertices = (head + [source] + tail[::-1])[1:-1]
    waypoints_coords = hierarchy_coords[waypoint_vertices]
    return waypoints_coords.tolist()


def _waypoints_from_hierarchy(
    hierarchy_df: pd.DataFrame,
    edges: Sequence[tuple[Hashable, Hashable]],
    hierarchy_coords: np.ndarray,
) -> tuple[np.ndarray, dict[Hashable, list[tuple[float, float]]]]:
    """Build a dictionary mapping each vertex to a list of waypoints along the path from the root to the vertex.

    Parameters:
        hierarchy_df: a dataframe with columns "vertices", "parents", "layer", and "depth".
        edges: a sequence of edges in the network, as tuples of (source, target).
        coords: a numpy array of shape (nv, 2) containing the coordinates of the vertices, including the .
    Returns:
        a dictionary mapping each vertex to a list of waypoints along the path from the root to the vertex.
    """
    # Create a head and a tail for each edge, we'll join them later

    hierarchy_df = hierarchy_df.copy()
    hierarchy_df["index"] = np.arange(len(hierarchy_df))
    hierarchy_df.set_index("vertices", drop=False, inplace=True)

    waypoints = {}
    for edge in edges:
        waypoints[edge] = _compute_waypoints_for_edge(hierarchy_df, edge, hierarchy_coords)

    nv = (len(hierarchy_df) + 1) // 2
    coords = hierarchy_coords[:nv]

    return coords, waypoints


def edgebundle(
    network,
    linkage: Optional[np.ndarray | pd.DataFrame] = None,
    center: Optional[tuple[float, float]] = None,
    orientation: str = "right",
    theta: float = 0.0,
):
    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()

    waypoints = {}

    if nv == 0:
        layout = pd.DataFrame(columns=["x", "y"], dtype=np.float64)
        return layout, waypoints

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        if linkage is None:
            from scipy.sparse import coo_matrix
            from scipy.cluster.hierarchy import linkage as linkage_fun
            from scipy.spatial.distance import squareform

            sources, targets = zip(*provider.edges())
            edge_weights = [1.0 for _ in sources]
            cdist = coo_matrix((edge_weights, (sources, targets)), shape=(nv, nv)).toarray()
            cdist += cdist.T
            cdist = 1.0 / (cdist + 1e-10)
            cdist[np.arange(len(cdist)), np.arange(len(cdist))] = 0
            pdist = squareform(cdist)
            linkage = linkage_fun(pdist)
        elif isinstance(linkage, pd.DataFrame):
            linkage = linkage.values

        # Prepare memory for additional vertices to be used as edge waypoints
        # The second column of a linkage is the index of the parent

        # Build hierarchy from linkage
        hierarchy_df = _hierarchy_from_linkage(linkage)
        hierarchy_index = np.arange(len(hierarchy_df))
        hierarchy_coords = np.zeros((len(hierarchy_df), 2), dtype=np.float64)

        # Assign coordintes to all vertices including the internal ones
        _dendrogram_from_hierarchy(hierarchy_df, hierarchy_coords, hierarchy_index)

        # Make circular with appropriate theta offset and orientation
        radius = hierarchy_coords[:, 0].copy()
        angle = -2 * hierarchy_coords[:, 1] * np.pi / (hierarchy_coords[:, 1].max() + 1)
        if orientation == "left":
            angle *= -1
        hierarchy_coords[:, 0] = radius * np.cos(angle + theta)
        hierarchy_coords[:, 1] = radius * np.sin(angle + theta)

        # Assign waypoints (internal nodes) to edges as waypoints
        coords, waypoints = _waypoints_from_hierarchy(
            hierarchy_df,
            provider.edges(),
            hierarchy_coords,
        )

    if center is not None:
        _recenter_layout(coords, center)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout, waypoints
