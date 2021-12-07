import pytest

from corgie.helpers import BoolFn
from corgie.cli.make_segmentation_masks import (
    DetectSlipMisalignmentsJob,
    DetectStepMisalignmentsJob,
    DetectThreeConescutiveMasks,
)


def misalignment_evaluation(stack, result, exp):
    bf = BoolFn(exp)
    mask = [False] * len(stack)

    def stack_similarity(a, b):
        """Is stack[a] == stack[b]? Stack outside of bounds is 1"""
        sa, sb = 1, 1
        if 0 < a < len(stack):
            sa = stack[a]
        if 0 < b < len(stack):
            sb = stack[b]
        return sa == sb

    for z in range(len(stack)):

        def eval(e):
            w, k, o = e["weight"], e["key"], e["offset"]
            return w * stack_similarity(z + o, z + o + k)

        mask[z] = bf(eval)
    assert mask == result


@pytest.mark.parametrize(
    "stack, result",
    [
        ([1, 1, 2, 1, 1], [False, False, True, False, False]),
        ([1, 1, 2, 2, 1], [False, False, True, True, False]),
        ([1, 1, 2, 2, 1, 3], [False, False, True, True, False, True]),
        # we accept not handling the following case
        ([1, 1, 2, 2, 1, 2], [False, False, False, False, False, False]),
    ],
)
def test_slip_misalignment_logic(stack, result):
    exp = DetectSlipMisalignmentsJob.get_exp()
    misalignment_evaluation(stack, result, exp)


@pytest.mark.parametrize(
    "stack, result",
    [
        ([1, 1, 2, 1, 1], [False, False, False, False, False]),
        ([1, 1, 2, 2, 1], [False, False, False, False, False]),
        ([1, 1, 2, 2, 2], [False, True, False, False, True]),
        ([1, 1, 2, 3, 3], [False, True, False, False, False]),
        ([1, 1, 2, 3, 4], [False, True, False, False, False]),
        # we accept not handling the following case
        ([1, 1, 2, 2, 1, 2], [False, False, False, False, False, True]),
    ],
)
def test_step_misalignment_logic(stack, result):
    exp = DetectStepMisalignmentsJob.get_exp()
    misalignment_evaluation(stack, result, exp)


@pytest.mark.parametrize(
    "stack, result",
    [
        ([False, False, False, False, False], [False, False, False, False, False]),
        ([False, True, False, False, False], [False, False, False, False, False]),
        ([False, True, False, True, False], [False, False, False, False, False]),
        ([False, True, True, False, False], [False, False, False, False, False]),
        ([False, True, True, True, False], [False, True, True, True, False]),
        ([False, True, True, True, True], [False, True, True, True, True]),
    ],
)
def test_three_consecutive_masks(stack, result):
    exp = DetectThreeConescutiveMasks.get_exp()
    bf = BoolFn(exp)
    mask = [False] * len(stack)

    def get_stack(a):
        """Return stack[a], or False if a outside bounds"""
        if 0 < a < len(stack):
            return stack[a]
        else:
            return False

    for z in range(len(stack)):

        def eval(e):
            w, k, o = e["weight"], e["key"], e["offset"]
            return w * get_stack(z + k + o)

        mask[z] = bf(eval)
    assert mask == result
