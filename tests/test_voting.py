import pytest
from corgie.cli.vote import compute_softmin_temp

@pytest.mark.parametrize(
    "dist, weight, size, result",
    [
        (0, 0.4, 1, 0),
        (0, 0.6, 1, 0),
        (8, 0.99, 3, 8./6.), # SEAMLeSS params with MIP0 fields
        (1, 0.99, 3, 1./6.), # adapting SEAMLeSS params for mipless fields 
    ],
)
def test_compute_softmin_temp(dist, weight, size, result):
    assert pytest.approx(compute_softmin_temp(dist=dist, weight=weight, size=size), result)

@pytest.mark.parametrize(
    "dist, weight, size",
    [
        (-1, 0.4, 1),
        (0, -0.1, 1),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0.5, 0),
    ],
)
def test_compute_softmin_temp_errors(dist, weight, size):
    with pytest.raises(AssertionError):
        compute_softmin_temp(dist=dist, weight=weight, size=size)
