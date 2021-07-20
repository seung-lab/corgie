import click
import torch
import procspec
import cachetools

from corgie import scheduling, argparsers, helpers

from corgie.log import logger as corgie_logger
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import (
    LAYER_HELP_STR,
    create_layer_from_spec,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


@cachetools.cached(cachetools.LRUCache(maxsize=4))
def get_gaussian_mask(shape, var):
    z = shape[0]
    channels = shape[1]
    kernel_size = shape[-2:]

    kernel = 1
    s = 150.0 * kernel_size[0] / 1024
    sigma = [s, s]
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(z, channels, 1, 1)
    return kernel


class BlendDivideTask(scheduling.Task):
    def __init__(self, dst_layer, accum_layer, weight_layer, count_layer, mip, bcube):
        self.dst_layer = dst_layer
        self.accum_layer = accum_layer
        self.weight_layer = weight_layer
        self.count_layer = count_layer
        self.mip = mip
        self.bcube = bcube
        super().__init__()

    def execute(self):
        accum_data = self.accum_layer.read(bcube=self.bcube, mip=self.mip)
        weight_data = self.weight_layer.read(bcube=self.bcube, mip=self.mip)
        count_data = self.count_layer.read(bcube=self.bcube, mip=self.mip)

        dst_data = accum_data
        many_writes_mask = (count_data > 1).squeeze()
        dst_data[..., many_writes_mask] /= weight_data[..., many_writes_mask]

        self.dst_layer.write(dst_data, bcube=self.bcube, mip=self.mip)


class BlendAccumulateTask(scheduling.Task):
    def __init__(
        self, src_layer, accum_layer, weight_layer, count_layer, blend_var, mip, bcube
    ):
        self.src_layer = src_layer
        self.accum_layer = accum_layer
        self.weight_layer = weight_layer
        self.count_layer = count_layer
        self.blend_var = blend_var
        self.mip = mip
        self.bcube = bcube
        super().__init__()

    def execute(self):
        src_data = self.src_layer.read(bcube=self.bcube, mip=self.mip)
        accum_data = self.accum_layer.read(bcube=self.bcube, mip=self.mip)
        weight_data = self.weight_layer.read(bcube=self.bcube, mip=self.mip)
        count_data = self.count_layer.read(bcube=self.bcube, mip=self.mip)

        weight_mask = get_gaussian_mask(src_data.shape, self.blend_var)

        first_write_mask = (count_data == 0).squeeze()
        second_write_mask = (count_data == 1).squeeze()

        accum_data[..., first_write_mask] = src_data[..., first_write_mask]
        accum_data[..., second_write_mask] *= weight_data[..., second_write_mask]

        accum_data[..., first_write_mask == False] += (
            weight_mask[..., first_write_mask == False]
            * src_data[..., first_write_mask == False]
        )

        # dst_layer may have many channels,
        # but our weights will have only one since it's the
        # same for channels
        weight_data += weight_mask[:, 0:1]

        count_data += 1

        self.accum_layer.write(accum_data, bcube=self.bcube, mip=self.mip)
        self.weight_layer.write(weight_data, bcube=self.bcube, mip=self.mip)
        self.count_layer.write(count_data, bcube=self.bcube, mip=self.mip)


class ChunkedJob(scheduling.Job):
    """Job that applies the given task to the bcube in chunk.
    Supports chunk blending"""

    def __init__(
        self,
        task_class,
        dst_layer,
        chunk_xy,
        chunk_z,
        bcube,
        mip,
        blend_xy=0,
        blend_var=1,
        suffix="",
    ):
        """blend_xy decides the blend region.
        it must be less than 50% of chunk_xy

        task_class is PartialSpecification of a task
        lacking only destination layer, bcube and MIP
        parameters"""
        self.suffix = suffix
        self.task_class = task_class

        self.dst_layer = dst_layer
        self.bcube = bcube
        self.mip = mip

        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

        if blend_xy < 0:
            raise exceptions.CorgieException(
                f"'blend_xy' must " f"be a positive integer. Received: {blend_xy}"
            )

        if blend_xy > chunk_xy // 2:
            raise exceptions.CorgieException(
                f"'blend_xy' must "
                f"be less than half of 'chunk_xy'. Given:"
                f"'blend_xy': {blend_xy}, 'chunk_xy': {chunk_xy}"
            )

        self.blend_xy = blend_xy
        self.blend_var = blend_var
        super().__init__()

    def task_generator(self):
        if self.blend_xy == 0:
            yield from self.noblend_task_generator()
        else:
            yield from self.blend_task_generator()

    def noblend_task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=self.chunk_z, mip=self.mip
        )

        tasks = [
            self.task_class(dst_layer=self.dst_layer, mip=self.mip, bcube=chunk)
            for chunk in chunks
        ]

        corgie_logger.debug(
            f"Yielding {type(tasks[0])} tasks "
            f"for bcube: {self.bcube}, MIP: {self.mip}"
        )
        yield tasks

    def blend_task_generator(self):
        assert self.chunk_xy % self.blend_xy == 0

        accum_layer = self.dst_layer.get_sublayer(
            name=f"accum_blend{self.suffix}",
            layer_type=self.dst_layer.get_layer_type(),
            force_chunk_xy=self.blend_xy,
            force_chunk_z=self.chunk_z,
            overwrite=True,
        )
        weight_layer = self.dst_layer.get_sublayer(
            name=f"accum_weight{self.suffix}",
            force_chunk_xy=self.blend_xy,
            force_chunk_z=self.chunk_z,
            layer_type="img",
            dtype="float32",
            overwrite=True,
        )
        count_layer = self.dst_layer.get_sublayer(
            name=f"accum_count{self.suffix}",
            force_chunk_xy=self.blend_xy,
            force_chunk_z=self.chunk_z,
            layer_type="img",
            dtype="uint8",
            overwrite=True,
        )

        blend_chunks = accum_layer.break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
            chunk_xy_step=self.chunk_xy - self.blend_xy,
            mip=self.mip,
            flatten=False,
        )

        checkerb_layers, checkerb_chunk_sets = self._create_checkerboard(blend_chunks)
        for i in range(len(checkerb_chunk_sets)):
            tasks = [
                self.task_class(dst_layer=checkerb_layers[i], mip=self.mip, bcube=chunk)
                for chunk in checkerb_chunk_sets[i]
            ]

            corgie_logger.debug(
                f"Yielding {type(tasks[0])} tasks "
                f"for bcube: {self.bcube}, MIP: {self.mip}, "
                f"checkerboard #{i}"
            )
            yield tasks

        yield scheduling.wait_until_done

        for i in range(len(checkerb_chunk_sets)):
            tasks = [
                BlendAccumulateTask(
                    src_layer=checkerb_layers[i],
                    accum_layer=accum_layer,
                    weight_layer=weight_layer,
                    count_layer=count_layer,
                    blend_var=self.blend_var,
                    mip=self.mip,
                    bcube=chunk,
                )
                for chunk in checkerb_chunk_sets[i]
            ]

            corgie_logger.debug(
                f"Yielding {type(tasks[0])} tasks "
                f"for bcube: {self.bcube}, MIP: {self.mip}, "
                f"checkerboard #{i}"
            )

            yield tasks
            yield scheduling.wait_until_done

        dst_chunks = self.dst_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=self.chunk_z, mip=self.mip
        )

        tasks = [
            BlendDivideTask(
                dst_layer=self.dst_layer,
                accum_layer=accum_layer,
                weight_layer=weight_layer,
                count_layer=count_layer,
                mip=self.mip,
                bcube=chunk,
            )
            for chunk in dst_chunks
        ]
        yield tasks

    def _create_checkerboard(self, blend_chunks):
        checkerb_layers = []
        checkerb_chunk_sets = []
        count = 0

        for x_chunk_offset in [0, 1]:
            for y_chunk_offset in [0, 1]:
                checkerb_layer = self.dst_layer.get_sublayer(
                    name=f"checkerboard_{count}{''}",
                    layer_type=self.dst_layer.get_layer_type(),
                    force_chunk_xy=self.blend_xy,
                    force_chunk_z=self.chunk_z,
                    overwrite=True,
                )

                checkerb_chunks = self._get_checkerboard_chunks(
                    xy_chunks=blend_chunks,
                    x_offset=x_chunk_offset,
                    y_offset=y_chunk_offset,
                )

                checkerb_layers.append(checkerb_layer)
                checkerb_chunk_sets.append(checkerb_chunks)
                count += 1

        return checkerb_layers, checkerb_chunk_sets

    def _get_checkerboard_chunks(self, xy_chunks, x_offset, y_offset):
        result = []
        for z in range(len(xy_chunks)):
            for x in range(x_offset, len(xy_chunks[0]), 2):
                for y in range(y_offset, len(xy_chunks[0][0]), 2):
                    result.append(xy_chunks[z][x][y])
        return result
