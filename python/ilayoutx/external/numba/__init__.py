"""Shim for optional numba dependency."""

import numpy as np

try:
    import numba

    maybe_numba = numba
    has_numba = True
except ImportError:
    maybe_numba = None
    has_numba = False


if maybe_numba is None:
    # Fallback no-op decorator
    def _dumb_decorator_with_args(*args, **kwargs):
        def _dumb_decorator(func):
            return func

        return _dumb_decorator

    maybe_numba = object()
    maybe_numba.njit = _dumb_decorator_with_args
    maybe_numba.types = object()
    types = ["float32", "float64", "int32", "int64", "uint8", "uint16"]
    for t in types:
        maybe_numba.types.__setattr__(t, getattr(np, t))

    maybe_numba.prange = range


__all__ = ("maybe_numba", "has_numba")
