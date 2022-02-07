import click
import torch

from corgie import scheduling
from corgie.log import logger as corgie_logger
from corgie.data_backends import pass_data_backend
from corgie.layers import get_layer_types, DEFAULT_LAYER_TYPE, str_to_layer_type
from corgie.boundingcube import get_bcube_from_coords
from corgie import argparsers
from corgie.argparsers import (
    LAYER_HELP_STR,
    create_layer_from_spec,
    corgie_optgroup,
    corgie_option,
)


class DownsampleJob(scheduling.Job):
    def __init__(
        self,
        src_layer,
        mip_start,
        mip_end,
        bcube,
        chunk_xy,
        chunk_z,
        mips_per_task=3,
        dst_layer=None,
        preserve_zeros=False
    ):
        if dst_layer is None:
            dst_layer = src_layer
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mips_per_task = mips_per_task
        self.preserve_zeros = preserve_zeros

        super().__init__()

    def task_generator(self):
        for mip in range(self.mip_start, self.mip_end, self.mips_per_task):
            this_mip_start = mip
            this_mip_end = min(self.mip_end, mip + self.mips_per_task)
            chunks = self.dst_layer.break_bcube_into_chunks(
                bcube=self.bcube,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip=this_mip_end,
                return_generator=True,
            )

            tasks = (
                DownsampleTask(
                    self.src_layer,
                    self.dst_layer,
                    this_mip_start,
                    this_mip_end,
                    input_chunk,
                    preserve_zeros=self.preserve_zeros
                )
                for input_chunk in chunks
            )
            print(
                "Yielding downsample tasks for bcube: {}, MIPs: {}-{}".format(
                    self.bcube, this_mip_start, this_mip_end
                )
            )

            yield tasks

            if mip == self.mip_start:
                self.src_layer = self.dst_layer

            # if not the last iteration
            if mip + self.mips_per_task < self.mip_end:
                yield scheduling.wait_until_done


class DownsampleTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip_start, mip_end, bcube, preserve_zeros):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip_start = mip_start
        self.mip_end = mip_end
        self.bcube = bcube
        self.preserve_zeros = preserve_zeros

    def execute(self):
        src_data = self.src_layer.read(bcube=self.bcube, mip=self.mip_start)
        # How to downsample depends on layer type.
        # Images are avg pooled, masks are max pooled, segmentation is...
        downsampler = self.src_layer.get_downsampler()
        for mip in range(self.mip_start, self.mip_end):
            dst_data = downsampler(src_data)
            if self.preserve_zeros:
                zeros = src_data == 0
                zeros_d = torch.nn.functional.max_pool2d(zeros.float(), 2, 2)
                dst_data[zeros_d > 0] = 0
            self.dst_layer.write(dst_data, bcube=self.bcube, mip=mip + 1)
            src_data = dst_data


@click.command()
@corgie_optgroup("Layer Parameterd")
@corgie_option(
    "--src_layer_spec",
    "-s",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the source layer. " + LAYER_HELP_STR,
)
@corgie_option(
    "--dst_layer_spec",
    "-s",
    nargs=1,
    type=str,
    required=False,
    help="Specification for the destination layer. "
    + "Refer to 'src_layer_spec' for parameter format."
    + " DEFAULT: Same as src_layer_spec",
)
@corgie_optgroup("Downsample parameters")
@corgie_option("--mip_start", "-m", nargs=1, type=int, required=True)
@corgie_option("--mip_end", "-e", nargs=1, type=int, required=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=2048)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--mips_per_task", nargs=1, type=int, default=3)
@corgie_option("--preserve_zeros/--no_preserve_zeros", default=False)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def downsample(
    ctx,
    src_layer_spec,
    dst_layer_spec,
    mip_start,
    mip_end,
    chunk_xy,
    chunk_z,
    mips_per_task,
    start_coord,
    end_coord,
    coord_mip,
    preserve_zeros
):
    scheduler = ctx.obj["scheduler"]
    corgie_logger.debug("Setting up Source and Destination layers...")



    if dst_layer_spec is None:
        corgie_logger.info(
            "Destination layer not specified. Using Source layer " "as Destination."
        )
        src_layer = create_layer_from_spec(
            src_layer_spec, caller_name="src layer", readonly=False
        )
        dst_layer = src_layer
    else:
        src_layer = create_layer_from_spec(
            src_layer_spec, caller_name="src layer", readonly=True
        )
        dst_layer = create_layer_from_spec(
            dst_layer_spec,
            caller_name="dst_layer layer",
            readonly=False,
            reference=src_layer,
            overwrite=True,
        )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    downsample_job = DownsampleJob(
        src_layer=src_layer,
        dst_layer=dst_layer,
        mip_start=mip_start,
        mip_end=mip_end,
        bcube=bcube,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        mips_per_task=mips_per_task,
        preserve_zeros=preserve_zeros,
    )

    # create scheduler and execute the job
    scheduler.register_job(downsample_job, job_name="downsample")
    scheduler.execute_until_completion()
    result_report = (
        f"Downsampled {src_layer} from {mip_start} to {mip_end}. Result in {dst_layer}"
    )
    corgie_logger.info(result_report)
