"""
This module focuses on how to ingest network/tree data into standard data structures no matter what library they come from.
"""

import pathlib
import pkgutil
import importlib
from typing import (
    Protocol,
)


# Internally supported data providers
data_providers: dict[str, Protocol] = {}
providers_path = pathlib.Path(__file__).parent.joinpath("providers")
for importer, module_name, _ in pkgutil.iter_modules([providers_path]):
    module = importlib.import_module(f"ilayoutx.ingest.providers.{module_name}")
    for key, val in module.__dict__.items():
        if key == "NetworkDataProvider":
            continue
        if key.endswith("DataProvider"):
            data_providers[module_name] = val
            break
del providers_path


def network_library(network) -> str:
    """Guess the network library used to create the network."""
    for name, provider in data_providers.items():
        if provider.check_dependencies():
            graph_type = provider.graph_type()
            if isinstance(network, graph_type):
                return name
    raise ValueError(
        f"Network {network} did not match any available network library.",
    )
