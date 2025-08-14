from typing import (
    Optional,
    Sequence,
)
import numpy as np
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

    coords = line_rust(nv, theta)

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
):
    """Random layout, uniform in a box.

    Parameters:
        network: The network to layout.
        xmin: Minimum x-coordinate.
        xmax: Maximum x-coordinate.
        ymin: Minimum y-coordinate.
        ymax: Maximum y-coordinate.
        seed: Optional random seed for reproducibility.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    coords = random_rust(nv, xmin, xmax, ymin, ymax, seed)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def shell(
    network,
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
):
    """Shell layout.

    Parameters:
        network: The network to layout.
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
        return pd.DataFrame(columns=["x", "y"])

    coords = shell_rust(nv, radius, center, theta)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def spiral(
    network,
    radius: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    slope: float = 1.0,
    theta: float = 0.0,
):
    """Spiral layout.

    Parameters:
        network: The network to layout.
        radius: The radius of the shell.
        center: The center of the shell as a tuple (x, y).
        slope: The slope of the spiral.
        theta: The angle of the shell in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    coords = spiral_rust(nv, radius, center, slope, theta)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])

    return layout
