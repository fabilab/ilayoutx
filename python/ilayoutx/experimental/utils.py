"""Utils for experimental API."""

import os


def get_debug_bool(var_name, default=True):
    """Check if a debug environment variable is set to true."""
    default = str(default).lower()

    return os.getenv(var_name, default).lower() not in ("false", "", "no", "n", "f")
