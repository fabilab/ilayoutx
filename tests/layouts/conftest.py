"""Utils for layout tests."""

from collections.abc import Callable
import pathlib
import pytest
import pandas as pd


class Helpers:
    @staticmethod
    def check_generic_layout(layout, dimension: int = 2):
        assert layout.shape[1] == dimension
        if dimension == 2:
            assert layout.columns.tolist() == ["x", "y"]
        else:
            assert layout.columns.tolist() == ["x", "y", "z"]

        assert layout.values.dtype == float

    @staticmethod
    def expected_layout_with_cache(
        cache_fn: str | pathlib.Path,
        compute_fun: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        """Wrap a certain expected layout computation with a cache.

        Parameters:
            cache_fn: The cache file name. If it exists, the expected layout will be read from this file.
            compute_fun: A function that computes the expected layout if the cache file does not exist.
                If called, also create the cache file after computing the expected layout.
        Returns:
            A pandas DataFrame containing the expected layout.
        """
        cache_fn = pathlib.Path(cache_fn)
        if cache_fn.exists():
            return pd.read_csv(cache_fn, sep=",", index_col=0)
        else:
            res = compute_fun()
            res.to_csv(cache_fn, sep=",", index=True)
            return res


@pytest.fixture
def helpers():
    return Helpers
