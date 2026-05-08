"""
This module focuses on how to ingest network/tree data into standard data structures no matter what library they come from.
"""

import pathlib
import pkgutil
import importlib
from typing import (
    Protocol,
)

# Accept as entry points both ilayoutx... and iplotx... to simplify downstream developers' life a bit.
# NOTE: ilayoutx MUST be first to ensure our own implementations are always there and preferred over
# any third-party ones with the EXACT same name.
_module_prefixes = (
    "ilayoutx",
    "iplotx",
)


# Internally supported data providers
data_providers: dict[str, Protocol] = {}
providers_path = pathlib.Path(__file__).parent.joinpath("providers")
for importer, module_name, _ in pkgutil.iter_modules([providers_path]):
    for module_prefix in _module_prefixes:
        try:
            module = importlib.import_module(f"{module_prefix}.ingest.providers.{module_name}")
        except ModuleNotFoundError:
            continue
        for key, val in module.__dict__.items():
            if key == "NetworkDataProvider":
                continue
            # If it's already there, do not overwrite it (see note above).
            if module_name in data_providers:
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
