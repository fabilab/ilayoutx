"""Shim for optional numba dependency."""

import numpy as np

try:
    import numba

    maybe_numba = numba
    has_numba = True
except ImportError:
    maybe_numba = None
    has_numba = False


# Fallback no-op decorator
def _dumb_decorator_with_args(*args, **kwargs):
    def _dumb_decorator(func):
        return func

    return _dumb_decorator


class NumbaTypes:
    pass


for t in ["float32", "float64", "int32", "int64", "uint8", "uint16"]:
    setattr(NumbaTypes, t, getattr(np, t))
NumbaTypes.intp = int


class NumbaMock:
    pass


NumbaMock.njit = _dumb_decorator_with_args
NumbaMock.prange = range
NumbaMock.types = NumbaTypes


if not has_numba:
    maybe_numba = NumbaMock


__all__ = ("maybe_numba", "has_numba")
