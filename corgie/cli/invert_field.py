import click
import procspec
import torchfields

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

from corgie.cli.common import ChunkedJob


class InvertFieldJob(scheduling.Job):
    def __init__(
        self,
        src_layer,
        dst_layer,
        chunk_xy,
        chunk_z,
        mip,
        pad,
        bcube,
        blend_xy=0,
        crop=None,
    ):

        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.blend_xy = blend_xy
        self.pad = pad
        self.crop = pad
        if crop is not None:
            self.crop = crop
        self.bcube = bcube
        self.mip = mip

        super().__init__()

    def task_generator(self):
        this_task = helpers.PartialSpecification(
            InvertFieldTask,
            src_layer=self.src_layer,
            pad=self.pad,
            crop=self.crop,
        )

        chunked_job = ChunkedJob(
            task_class=this_task,
            dst_layer=self.dst_layer,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
            blend_xy=self.blend_xy,
            mip=self.mip,
            bcube=self.bcube,
        )

        yield from chunked_job.task_generator


class InvertFieldTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip, pad, crop, bcube):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.crop = crop
        self.bcube = bcube

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.mip)

        src_data = self.src_layer.read(src_bcube, mip=self.mip)

        result = (~src_data.field().from_pixels()).pixels()

        cropped_result = helpers.crop(result, self.crop)

        self.dst_layer.write(cropped_result, bcube=self.bcube, mip=self.mip)


@click.command()
@corgie_optgroup("Layer Parameters")
@corgie_option(
    "--src_layer_spec",
    "-s",
    nargs=1,
    type=str,
    required=True,
    multiple=False,
    help="Source layer spec. " + LAYER_HELP_STR,
)
#
@corgie_option(
    "--dst_layer_spec",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the destination layer. Must be a field type.",
)
@corgie_optgroup("Invert Field Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option("--pad", nargs=1, type=int, default=128)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--mip", nargs=1, type=int)
@corgie_option("--force_chunk_xy", is_flag=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def invert_field(
    ctx,
    src_layer_spec,
    dst_layer_spec,
    pad,
    crop,
    chunk_xy,
    start_coord,
    mip,
    end_coord,
    coord_mip,
    blend_xy,
    chunk_z,
    force_chunk_xy,
):
    scheduler = ctx.obj["scheduler"]

    if force_chunk_xy:
        force_chunk_xy = chunk_xy
    else:
        force_chunk_xy = None

    corgie_logger.debug("Setting up layers...")
    src_layer = create_layer_from_spec(
        src_layer_spec,
        allowed_types=["field"],
        default_type="field",
        readonly=True,
        caller_name="src_layer",
    )

    dst_layer = create_layer_from_spec(
        dst_layer_spec,
        allowed_types=["field"],
        default_type="field",
        readonly=False,
        caller_name="dst_layer",
        reference=src_layer,
        overwrite=True,
        force_chunk_xy=force_chunk_xy,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    invert_field_job = InvertFieldJob(
        src_layer=src_layer,
        dst_layer=dst_layer,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        blend_xy=blend_xy,
        pad=pad,
        mip=mip,
        crop=crop,
        bcube=bcube,
    )

    # create scheduler and execute the job
    scheduler.register_job(invert_field_job, job_name="Invert Field {}".format(bcube))
    scheduler.execute_until_completion()
