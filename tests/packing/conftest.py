"""Utils for packing tests."""

import pytest


class Helpers:
    @staticmethod
    def check_generic_packing_concatenate(packing, dimension: int = 2):
        assert packing.shape[1] == dimension + 2
        if dimension == 2:
            assert packing.columns.tolist() == ["x", "y", "id", "layout_id"]
        else:
            assert packing.columns.tolist() == ["x", "y", "z", "id", "layout_id"]

        assert packing.iloc[:, :-2].values.dtype == float

    @staticmethod
    def check_generic_packing_single(packing, dimension: int = 2):
        assert packing.shape[1] == dimension
        if dimension == 2:
            assert packing.columns.tolist() == ["x", "y"]
        else:
            assert packing.columns.tolist() == ["x", "y", "z"]

        assert packing.values.dtype == float

    @classmethod
    def check_generic_packing_nonconcatenate(cls, packing_sequence, dimension: int = 2):
        assert isinstance(packing_sequence, list)

        for packing in packing_sequence:
            cls.check_generic_packing_single(packing, dimension)


@pytest.fixture
def helpers():
    return Helpers
