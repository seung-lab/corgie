import pytest

from corgie.helpers import BoolFn


@pytest.mark.parametrize(
    "exp, result",
    [
        (
            {
                "inputs": [
                    {"val": 1},
                    {"val": 1},
                ],
                "threshold": 3,
            },
            False,
        ),
        (
            {
                "inputs": [
                    {
                        "inputs": [
                            {"val": 1},
                            {"val": 1},
                            {"val": 1},
                        ],
                        "threshold": 2,
                    },
                    {
                        "inputs": [
                            {"val": 1},
                            {"val": 1},
                        ],
                        "threshold": 1,
                    },
                ],
                "threshold": 0,
            },
            True,
        ),
    ],
)
def test_bool_exp(exp, result):
    def eval(e):
        return e["val"]

    bf = BoolFn(exp)
    assert bf(eval) == result
