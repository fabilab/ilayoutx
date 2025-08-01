from typing import Optional
import pandas as pd

from ilayoutx._ilayoutx import (
    line as line_rust,
    circle as circle_rust,
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

    coords = line_rust(nv, theta)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout


def circle(
    network,
    radius: float = 1.0,
    theta: float = 0.0,
):
    """Circular layout.

    Parameters:
        network: The network to layout.
        radius: The radius of the circle.
        theta: The angle of the line in radians.
    Returns:
        A pandas.DataFrame with the layout.
    """
    nl = network_library(network)
    provider = data_providers[nl](network)
    nv = provider.number_of_vertices()

    coords = circle_rust(nv, radius, theta)

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

    coords = spiral_rust(nv, radius, center, slope, theta)

    layout = pd.DataFrame(coords, index=provider.vertices(), columns=["x", "y"])
    return layout
