import click
import procspec

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
from corgie.cli.apply_processor import ApplyProcessorJob
import json


@click.command()
@corgie_optgroup("Layer Parameters")
@corgie_option(
    "--src_layer_spec",
    "-s",
    nargs=1,
    type=str,
    required=True,
    multiple=True,
    help="Source layer spec. Use multiple times to include all masks, fields, images. "
    + LAYER_HELP_STR,
)
@corgie_option(
    "--spec_path",
    nargs=1,
    type=str,
    required=True,
    help="JSON spec relating src stacks, src z to dst z",
)
@corgie_option(
    "--dst_layer_spec",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the destination layer.",
)
@corgie_option("--reference_key", nargs=1, type=str, default="img")
@corgie_optgroup("Apply Processor Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option(
    "--pad",
    nargs=1,
    type=int,
    default=512,
)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_spec", nargs=1, type=str, multiple=True, required=True)
@corgie_option("--processor_mip", nargs=1, type=int, multiple=True, required=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def apply_processor_by_spec(
    ctx,
    src_layer_spec,
    spec_path,
    dst_layer_spec,
    processor_spec,
    pad,
    crop,
    chunk_xy,
    start_coord,
    processor_mip,
    end_coord,
    coord_mip,
    blend_xy,
    chunk_z,
    reference_key,
):
    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    with open(spec_path, "r") as f:
        spec = set(json.load(f))

    reference_layer = None
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]

    dst_layer = create_layer_from_spec(
        dst_layer_spec,
        allowed_types=["img", "mask"],
        default_type="img",
        readonly=False,
        caller_name="dst_layer",
        reference=reference_layer,
        overwrite=True,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    for z in range(*bcube.z_range()):
        if z in spec:
            job_bcube = bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            apply_processor_job = ApplyProcessorJob(
                src_stack=src_stack,
                dst_layer=dst_layer,
                chunk_xy=chunk_xy,
                chunk_z=chunk_z,
                blend_xy=blend_xy,
                processor_spec=processor_spec,
                pad=pad,
                crop=crop,
                bcube=job_bcube,
                processor_mip=processor_mip,
            )

            # create scheduler and execute the job
            scheduler.register_job(
                apply_processor_job, job_name="Apply Processor {}".format(job_bcube)
            )
    scheduler.execute_until_completion()
