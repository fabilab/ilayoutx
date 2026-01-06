"""Basic layouts: line, circle, random, shell, spiral."""

from typing import (
    Optional,
    Sequence,
)
from collections.abc import (
    Hashable,
)
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd

from ilayoutx._ilayoutx import (
    line as line_rust,
    random as random_rust,
    shell as shell_rust,
    spiral as spiral_rust,
)
from ..ingest import data_providers, network_library


def line(
    network,
    theta: float = 0.0,
):
    """Line layout.

    Parameters:
        network: The network to layout.
        theta: The angle of the line in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    coords = line_rust(nv, np.degrees(theta))

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def circle(
    network,
    radius: float = 1.0,
    theta: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    sizes: Optional[Sequence[float]] = None,
):
    """Circular layout.

    Parameters:
        network: The network to layout.
        radius: The radius of the circle.
        theta: The angle of the line in radians.
        center: The center of the circle as a tuple (x, y).
        sizes: Relative sizes of the 360 angular space to be used for the vertices.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    if nv == 1:
        coords = np.zeros((1, 2), dtype=np.float64)
    else:
        if sizes is None:
            thetas = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        else:
            sizes = np.array(sizes, dtype=np.float64)
            if len(sizes) != nv:
                raise ValueError(
                    "sizes must be a sequence of length equal to the number of vertices.",
                )
            sizes /= sizes.sum()

            # Vertex 1 is at (radius, 0), then half its wedge and half of the next wedge, etc.
            sizes[:] = sizes.cumsum()
            thetas = np.zeros(nv, dtype=np.float64)
            thetas[1:] = 2 * np.pi * (sizes[:-1] + sizes[1:]) / 2

        thetas += theta

        coords = np.zeros((nv, 2), dtype=np.float64)
        coords[:, 0] = radius * np.cos(thetas)
        coords[:, 1] = radius * np.sin(thetas)

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def random(
    network,
    xmin: float = -1.0,
    xmax: float = 1.0,
    ymin: float = -1.0,
    ymax: float = 1.0,
    seed: Optional[float] = None,
    sizes: Optional[Sequence[float]] = None,
    size_maxtries: int = 10,
):
    """Random layout, uniform in a box.

    Parameters:
        network: The network to layout.
        xmin: Minimum x-coordinate.
        xmax: Maximum x-coordinate.
        ymin: Minimum y-coordinate.
        ymax: Maximum y-coordinate.
        seed: Optional random seed for reproducibility.
        sizes: Sizes of the vertices in data coordinates. If not None, this function will
            regenerate new random positions for vertices that are closer in Euclidean
            distance than the sum of their sizes. This is equivalent to assuming that the
            vertices are circles and their sizes are radii.
        size_maxtries: Maximum number of attempts to find a valid position for a vertex
            when sizes are not None.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    coords = random_rust(nv, xmin, xmax, ymin, ymax, seed)
    if sizes is not None:
        sizes = np.array(sizes, dtype=np.float64)
        pdis = squareform(pdist(coords, metric="euclidean"))
        radii_sum = sizes[:, None] + sizes[None, :]
        # Check the conflicts, except for self-conflicts.
        conflict_i, conflict_j = (pdis < radii_sum).nonzero()
        tmp_idx = conflict_i != conflict_j
        conflict_i = conflict_i[tmp_idx]
        conflict_j = conflict_j[tmp_idx]
        for i in conflict_i:
            coords_attempt = random_rust(size_maxtries, xmin, xmax, ymin, ymax, seed)
            for coords_try in coords_attempt:
                coords[i] = coords_try
                pdis[i] = pdis[:, i] = cdist(coords[i : i + 1], coords, metric="euclidean")[0]
                # Obviously, you are always in conflict with yourself.
                if (pdis[i] >= radii_sum[i]).sum() > 1:
                    break
            else:
                raise ValueError(
                    "Could not generate a layout compatible with the sizes.",
                )

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def shell(
    network,
    nlist: Sequence[Sequence[Hashable]],
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
):
    """Shell layout.

    Parameters:
        network: The network to layout.
        nlist: List of lists of nodes in each shell.
        radius: The radius of the shell.
        center: The center of the shell as a tuple (x, y).
        theta: The angle of the shell in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=float)

    nnodes_by_shell = [len(x) for x in nlist]
    coords = shell_rust(nnodes_by_shell, radius, center, np.degrees(theta))
    nodes = []
    for nodes_shell in nlist:
        nodes.extend(list(nodes_shell))

    layout = pd.DataFrame(coords, index=nodes, columns=["x", "y"])

    # NOTE: reorder to match initial nodes
    print(provider.vertices())
    print(layout.index)
    layout = layout.loc[provider.vertices()]

    return layout


def spiral(
    network,
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    slope: float = 0.25,
    exponent: float = 1.0,
    theta: float = 0.0,
):
    """Spiral layout.

    Parameters:
        network: The network to layout.
        radius: The radius of the shell.
        center: The center of the shell as a tuple (x, y).
        slope: The slope of the spiral.
        exponent: The exponent of the spiral.
        theta: The initial angle of the layout in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"], dtype=float)

    if nv == 1:
        coords = np.array([[center[0], center[1]]], dtype=np.float64)
    else:
        coords = spiral_rust(nv, slope, np.degrees(theta), exponent)
        rmax = np.linalg.norm(coords, axis=1).max()
        coords *= radius / rmax
        coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])

    return layout
