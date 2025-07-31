from typing import (
    Optional,
)

from ilayoutx._ilayoutx import kamada_kawai as kk_rust
from ..ingest import (
    network_library,
    data_providers,
)


def kamada_kawai(
    network,
    seed: Optional[int] = None,
):
    """Kamada-Kawai layout algorithm.

    Parameters:
        network: The network to layout.
        seed: A random seed to use.
    Returns:
        The layout of the network.
    """

    # 0. Find what we have
    nl = network_library(network)
    provider = data_providers[nl](network)

    # 1. Compute the distance matrix.
    dist = provider.get_shortest_distance()

    return dist

    # 2. Solve kk using that matrix
    # layout = kk_rust.kamada_kawai(
    #    dist,
    # )
