import click
from copy import deepcopy

from corgie import scheduling, argparsers, helpers, stack

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

from corgie.cli.render import RenderJob
from corgie.cli.copy import CopyJob
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.compare_sections import CompareSectionsJob

from corgie.cli.downsample import DownsampleJob
from corgie.cli.upsample import UpsampleJob


class SeethroughBlockJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_stack,
        render_method,
        bcube,
        seethrough_method=None,
        suffix=None,
    ):
        """Write out a block sequentially using seethrough method

        This is useful when a model requires a filled in image.
        """
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.bcube = bcube
        self.seethrough_method = seethrough_method
        self.render_method = render_method
        self.suffix = suffix
        super().__init__()

    def task_generator(self):
        seethrough_offset = -1
        seethrough_mask_layer = self.dst_stack.create_sublayer(
            f"seethrough_mask{self.suffix}",
            layer_type="mask",
            overwrite=True,
            force_chunk_z=1,
        )
        z_start, z_stop = self.bcube.z_range()
        for z in range(z_start, z_stop):
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            if z == z_start:
                corgie_logger.debug(f"Copy section {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=bcube,
                    mips=self.seethrough_method.mip,
                    blackout_masks=True,
                )

                yield from render_job.task_generator
                yield scheduling.wait_until_done
            else:
                # Now, we'll apply misalignment detection to produce a mask
                # this mask will be used in the final render step
                seethrough_mask_job = self.seethrough_method(
                    src_stack=self.src_stack,
                    tgt_stack=self.dst_stack,
                    bcube=bcube,
                    tgt_z_offset=seethrough_offset,
                    suffix=self.suffix,
                    dst_layer=seethrough_mask_layer,
                )

                yield from seethrough_mask_job.task_generator
                yield scheduling.wait_until_done
                corgie_logger.debug(f"Render {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=bcube,
                    blackout_masks=False,
                    seethrough_mask_layer=seethrough_mask_layer,
                    seethrough_offset=seethrough_offset,
                    mips=self.seethrough_method.mip,
                )
                yield from render_job.task_generator
                yield scheduling.wait_until_done


@click.command()
# Layers
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
    "--dst_folder",
    nargs=1,
    type=str,
    required=True,
    help="Folder where aligned stack will go",
)
@corgie_option("--suffix", nargs=1, type=str, default=None)
@corgie_optgroup("Render Method Specification")
@corgie_option(
    "--seethrough_spec",
    nargs=1,
    type=str,
    default=None,
    multiple=True,
    help="Seethrough method spec. Use multiple times to specify different methods (e.g. seethrough misalignments, seethrough black, etc.)",
)
@corgie_option(
    "--seethrough_limit",
    nargs=1,
    type=int,
    default=None,
    multiple=True,
    help="For each seethrough method, how many sections are allowed to be seenthrough. 0 or None means no limit.",
)
@corgie_option("--chunk_xy", nargs=1, type=int, default=1024)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def seethrough_block(
    ctx,
    src_layer_spec,
    dst_folder,
    chunk_xy,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
    seethrough_spec,
    seethrough_limit,
    seethrough_spec_mip,
    force_chunk_z=1,
):
    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_seethrough"
    else:
        suffix = f"_{suffix}"

    crop, pad = 0, 0
    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(
        src_layer_spec, name="src", readonly=True
    )
    src_stack.folder = dst_folder
    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        overwrite=True,
        force_chunk_z=force_chunk_z,
    )
    render_method = helpers.PartialSpecification(
        f=RenderJob,
        pad=pad,
        chunk_xy=chunk_xy,
        chunk_z=1,
        render_masks=False,
    )
    seethrough_method = helpers.PartialSpecification(
        f=CompareSectionsJob,
        mip=seethrough_spec_mip,
        processor_spec=seethrough_spec,
        chunk_xy=chunk_xy,
        pad=pad,
        crop=pad,
        seethrough_limit=seethrough_limit,
    )
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    seethrough_block_job = SeethroughBlockJob(
        src_stack=src_stack,
        dst_stack=dst_stack,
        bcube=bcube,
        render_method=render_method,
        seethrough_method=seethrough_method,
        suffix=suffix,
    )
    # create scheduler and execute the job
    scheduler.register_job(
        seethrough_block_job, job_name="Seethrough Block {}".format(bcube)
    )

    scheduler.execute_until_completion()
    result_report = (
        f"Rendered layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
