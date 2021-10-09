import pytest
from corgie.boundingcube import BoundingCube
from corgie.stack import Stack
from corgie.layers.base import BaseLayerType
from corgie.block import Block, partition, get_blocks


@pytest.mark.parametrize(
    "xrange, sz, overlap, skip, result",
    [
        (range(4), 2, 0, [], [range(0, 2), range(2, 4)]),
        (range(4), 2, 1, [], [range(0, 3), range(2, 4)]),
        (range(4), 2, 0, [2], [range(0, 3), range(3, 4)]),
        (range(4), 2, 0, [2, 3], [range(0, 4)]),
        (range(4), 2, 0, [2, 3, 4], [range(0, 4)]),
        (range(4), 2, 0, [0, 2, 3, 4], [range(1, 4)]),
        (range(0), 2, 0, [2], []),
        (range(1), 2, 0, [2], [range(1)]),
        (range(12), 4, 1, [4, 5], [range(0, 7), range(6, 9), range(8, 12)])
    ],
)
def test_partition(xrange, sz, overlap, skip, result):
    assert partition(xrange, sz, overlap, skip) == result


def test_block_get_bcube():
    bcube = BoundingCube(0, 1, 0, 1, 0, 1, mip=0)
    start, stop = 10, 11
    block = Block(
        start=start, stop=stop, src_stack=Stack(), dst_stack=Stack(), previous=None
    )
    adj_bcube = bcube.reset_coords(zs=start, ze=stop, in_place=False)
    assert adj_bcube == block.get_bcube(bcube)
    assert bcube.z != (start, stop)


def example_blocks():
    src = Stack("src")
    even = Stack("even")
    odd = Stack("odd")
    blocks = get_blocks(
        start=0,
        stop=10,
        block_size=2,
        block_overlap=0,
        skip_list=[],
        src_stack=src,
        even_stack=even,
        odd_stack=odd,
    )
    return blocks


def test_block_get_previous():
    blocks = example_blocks()
    assert blocks[0].previous is None
    for i in range(1, len(blocks)):
        assert blocks[i].previous == blocks[i - 1]


@pytest.mark.parametrize(
    "dist, src_id, neighbor_ids",
    [
        (2, 0, []),
        (2, 1, [0]),
        (3, 1, [0]),
        (2, 2, [1]),
        (3, 2, [0, 1]),
        (8, 4, [0, 1, 2, 3]),
    ],
)
def test_block_get_neighbors(dist, src_id, neighbor_ids):
    blocks = example_blocks()
    src = blocks[src_id]
    neighbors = src.get_neighbors(dist)
    assert [blocks[i] for i in neighbor_ids] == neighbors


@pytest.mark.parametrize(
    "block_overlap, i, stitch_size, start, stop",
    [
        (0, 0, 3, None, None),
        (0, 1, 3, 4, 7),
        (0, 1, 1, 4, 5),
        (1, 1, 1, 5, 6),
    ],
)
def test_block_overlap(block_overlap, i, stitch_size, start, stop):
    src = Stack("src")
    even = Stack("even")
    odd = Stack("odd")
    blocks = get_blocks(
        start=0,
        stop=8,
        block_size=4,
        block_overlap=block_overlap,
        skip_list=[],
        src_stack=src,
        even_stack=even,
        odd_stack=odd,
    )
    stitch_block = blocks[i].overlap(stitch_size)
    if i == 0:
        assert stitch_block is None
    else:
        assert stitch_block.dst_stack == blocks[i].previous.dst_stack
        assert stitch_block.start == start
        assert stitch_block.stop == stop