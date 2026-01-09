"""Utils for layout tests."""

import pytest


class Helpers:
    @staticmethod
    def check_generic_layout(layout, dimension: int = 2):
        assert layout.shape[1] == dimension
        if dimension == 2:
            assert layout.columns.tolist() == ["x", "y"]
        else:
            assert layout.columns.tolist() == ["x", "y", "z"]

        assert layout.values.dtype == float


@pytest.fixture
def helpers():
    return Helpers
