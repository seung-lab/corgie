from copy import deepcopy


class Block:
    def __init__(self, start, stop, src_stack, dst_stack, previous=None):
        self.start = start
        self.stop = stop
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.previous = previous

    def __str__(self):
        return f"Block {self.start}->{self.stop}, src_stack={self.src_stack.name}, dst_stack={self.dst_stack.name}"

    def __repr__(self):
        return self.__str__()

    def overlap(self, stitch_size):
        """Get block representing overlap between current and previous block

        Args:
            stitch_size (int): number of sections to use in stitching

        Returns:
            new Block
        """
        if self.previous is None:
            return None
        return Block(
            start=self.previous.stop,
            stop=self.previous.stop + stitch_size,
            src_stack=self.dst_stack,
            dst_stack=self.previous.dst_stack,
        )

    def broadcastable(self):
        """Get block representing the sections that are not block aligned by previous block

        Returns:
            new Block
        """
        if self.previous is None:
            return None
        return Block(
            start=self.previous.stop,
            stop=self.stop,
            src_stack=self.src_stack,
            dst_stack=self.dst_stack,
        )

    def get_neighbors(self, dist):
        """Get list of previous blocks where stop is within dist of current block's start.
        Order from furthest to nearest block (convention of FieldSet).

        Args:
            dist (int)

        Returns:
            list of blocks
        """
        neighbors = []
        prev = self.previous
        if prev is None:
            return neighbors
        while self.start - prev.stop < dist:
            neighbors.insert(0, prev)
            prev = prev.previous
            if prev is None:
                break
        return neighbors

    def get_bcube(self, bcube):
        return bcube.reset_coords(zs=self.start, ze=self.stop, in_place=False)


def get_blocks(
    start,
    stop,
    block_size,
    block_overlap,
    skip_list,
    src_stack,
    even_stack,
    odd_stack,
):
    partitions = partition(
        range(start, stop), sz=block_size, overlap=block_overlap, skip=skip_list
    )
    blocks = []
    previous = None
    for k, p in enumerate(partitions):
        dst_stack = even_stack if k % 2 == 0 else odd_stack
        block = Block(
            start=p.start,
            stop=p.stop,
            src_stack=src_stack,
            dst_stack=dst_stack,
            previous=previous,
        )
        blocks.append(block)
        previous = block
    return blocks


def partition(xrange, sz, overlap=0, skip=[]):
    """Divide range into subsets specified by size, overlap, and indices to skip.

    Args:
        xrange (range)
        sz (int): target size of subset
        overlap (int): amount of overlap between subsets
        skip ([int]): indices to skip in start/end of subset

    Returns:
        list of subset ranges
    """
    subsets = []
    xs_start = xrange.start
    xs_next = min(xs_start + sz, xrange.stop)
    while xs_start < xs_next:
        xs_stop = xs_next
        while xs_stop in skip and xs_stop < xrange.stop:
            xs_stop += 1
        while xs_start in skip and xs_start + overlap < xs_stop:
            xs_start += 1
        xs_stop = min(xrange.stop, xs_stop + overlap)
        if xs_start < xs_stop:
            subsets.append(range(xs_start, xs_stop))
        xs_start = xs_next
        xs_next = min(xs_next + sz, xrange.stop)
    return subsets
