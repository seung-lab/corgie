import click

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
from corgie.cli.downsample import DownsampleJob
import json


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
    "--spec_path",
    nargs=1,
    type=str,
    required=True,
    help="JSON spec relating src stacks, src z to dst z",
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
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def downsample_by_spec(
    ctx,
    src_layer_spec,
    spec_path,
    dst_layer_spec,
    mip_start,
    mip_end,
    chunk_xy,
    chunk_z,
    mips_per_task,
    start_coord,
    end_coord,
    coord_mip,
):
    scheduler = ctx.obj["scheduler"]
    corgie_logger.debug("Setting up Source and Destination layers...")

    src_layer = create_layer_from_spec(
        src_layer_spec, caller_name="src layer", readonly=True
    )

    with open(spec_path, "r") as f:
        spec = set(json.load(f))

    if dst_layer_spec is None:
        corgie_logger.info(
            "Destination layer not specified. Using Source layer " "as Destination."
        )
        dst_layer = src_layer
        dst_layer.readonly = False
    else:
        dst_layer = create_layer_from_spec(
            dst_layer_spec,
            caller_name="dst_layer layer",
            readonly=False,
            reference=src_layer,
            chunk_z=chunk_z,
            overwrite=True,
        )
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    for z in range(*bcube.z_range()):
        if z in spec:
            job_bcube = bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            downsample_job = DownsampleJob(
                src_layer=src_layer,
                dst_layer=dst_layer,
                mip_start=mip_start,
                mip_end=mip_end,
                bcube=job_bcube,
                chunk_xy=chunk_xy,
                chunk_z=chunk_z,
                mips_per_task=mips_per_task,
            )

            # create scheduler and execute the job
            scheduler.register_job(downsample_job, job_name=f"Downsample {job_bcube}")
    scheduler.execute_until_completion()
    result_report = (
        f"Downsampled {src_layer} from {mip_start} to {mip_end}. Result in {dst_layer}"
    )
    corgie_logger.info(result_report)
