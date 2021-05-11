import click
import os
from corgie.block import get_blocks
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

from corgie.cli.align_block import AlignBlockJob
from corgie.cli.render import RenderJob
from corgie.cli.copy import CopyJob
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.compare_sections import CompareSectionsJob
from corgie.cli.vote import VoteOverZJob
from corgie.cli.broadcast import BroadcastJob


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
# @corgie_option('--seethrough_masks',    nargs=1, type=bool, default=False)
# @corgie_option('--seethrough_misalign', nargs=1, type=bool, default=False)
@corgie_option("--render_pad", nargs=1, type=int, default=512)
@corgie_option("--render_chunk_xy", nargs=1, type=int, default=1024)
@corgie_optgroup("Compute Field Method Specification")
@corgie_option("--processor_spec", nargs=1, type=str, required=True, multiple=True)
@corgie_option("--processor_mip", "-m", nargs=1, type=int, required=True, multiple=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--pad", nargs=1, type=int, default=256)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--copy_start/--no_copy_start", default=True)
@corgie_option("--seethrough_spec", nargs=1, type=str, default=None)
@corgie_option("--seethrough_spec_mip", nargs=1, type=int, default=None)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_optgroup("Block Alignment Specification")
@corgie_option("--bad_starter_path", nargs=1, type=str, default=None)
@corgie_option("--block_size", nargs=1, type=int, default=10)
@corgie_option(
    "--stitch_size",
    nargs=1,
    type=int,
    default=3,
    help="Number of sections involved in computing stitching fields.",
)
@corgie_option("--vote_dist", nargs=1, type=int, default=1)
@corgie_option("--decay_dist", nargs=1, type=int, default=50)
@corgie_option(
    "--restart_stage",
    nargs=1,
    type=int,
    default=0,
    help="0: block, 1: overlap, 2: stitch",
)
@click.pass_context
def align(
    ctx,
    src_layer_spec,
    dst_folder,
    render_pad,
    render_chunk_xy,
    processor_spec,
    pad,
    crop,
    processor_mip,
    chunk_xy,
    start_coord,
    end_coord,
    coord_mip,
    bad_starter_path,
    block_size,
    stitch_size,
    vote_dist,
    blend_xy,
    force_chunk_xy,
    suffix,
    copy_start,
    seethrough_spec,
    seethrough_spec_mip,
    decay_dist,
    restart_stage,
):

    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_aligned"
    else:
        suffix = f"_{suffix}"

    if crop is None:
        crop = pad

    corgie_logger.debug("Setting up layers...")

    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)
    src_stack.folder = dst_folder

    if force_chunk_xy is None:
        force_chunk_xy = chunk_xy

    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        overwrite=True,
    )

    even_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=os.path.join(dst_folder, "even"),
        name="even",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        overwrite=True,
    )

    odd_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=os.path.join(dst_folder, "odd"),
        name="odd",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        overwrite=True,
    )

    corgie_logger.debug("Done!")

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    field_name = f"field{suffix}"
    corgie_logger.debug("Calculating blocks...")
    # TODO: read in bad starter sections
    blocks = get_blocks(
        start=bcube.z_range()[0],
        stop=bcube.z_range()[1],
        block_size=block_size,
        block_overlap=0,
        skip_list=[],
        src_stack=src_stack,
        even_stack=even_stack,
        odd_stack=odd_stack,
    )
    corgie_logger.debug("blocks")
    for block in blocks:
        corgie_logger.debug(block)
        corgie_logger.debug("stitch")
        corgie_logger.debug(block.overlap(stitch_size, ""))

    render_method = helpers.PartialSpecification(
        f=RenderJob,
        pad=render_pad,
        chunk_xy=render_chunk_xy,
        chunk_z=1,
        render_masks=False,
    )

    cf_method = helpers.PartialSpecification(
        f=ComputeFieldJob,
        pad=pad,
        crop=crop,
        processor_mip=processor_mip,
        processor_spec=processor_spec,
        chunk_xy=chunk_xy,
        blend_xy=blend_xy,
        chunk_z=1,
    )
    if seethrough_spec is not None:
        assert seethrough_spec_mip is not None

        seethrough_method = helpers.PartialSpecification(
            f=CompareSectionsJob,
            mip=seethrough_spec_mip,
            processor_spec=seethrough_spec,
            chunk_xy=chunk_xy,
            pad=pad,
            crop=pad,
        )
    else:
        seethrough_method = None

    if restart_stage == 0:
        corgie_logger.debug("Aligning blocks...")
        for block in blocks:
            block_bcube = block.get_bcube(bcube)
            align_block_job_forv = AlignBlockJob(
                src_stack=block.src_stack,
                dst_stack=block.dst_stack,
                bcube=block_bcube,
                render_method=render_method,
                cf_method=cf_method,
                vote_dist=vote_dist,
                seethrough_method=seethrough_method,
                suffix=suffix,
                copy_start=copy_start,
                backward=False,
            )
            scheduler.register_job(
                align_block_job_forv, job_name=f"Forward Align {block} {block_bcube}"
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    stitching_field = None
    if restart_stage > 0:
        even_stack.create_sublayer(
            field_name,
            layer_type="field",
            overwrite=False,
        )
        odd_stack.create_sublayer(
            field_name,
            layer_type="field",
            overwrite=False,
        )

    if restart_stage <= 1:
        corgie_logger.debug("Creating stitching fields...")
        corgie_logger.debug("Aligning stitching blocks...")
        for block in blocks[1:]:
            stitch_block = block.overlap(stitch_size, field_name)
            block_bcube = stitch_block.get_bcube(bcube)
            align_block_job_forv = AlignBlockJob(
                src_stack=stitch_block.src_stack,
                dst_stack=stitch_block.dst_stack,
                bcube=block_bcube,
                render_method=render_method,
                cf_method=cf_method,
                vote_dist=vote_dist,
                seethrough_method=seethrough_method,
                suffix=suffix,
                copy_start=False,
                backward=False,
            )
            scheduler.register_job(
                align_block_job_forv, job_name=f"Stitch Align {block} {block_bcube}"
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

        if stitch_size > 1:
            corgie_logger.debug("Voting over stitching blocks")
            stitching_field = dst_stack.create_sublayer(
                "stitching_field", layer_type="field", overwrite=True
            )
            for block in blocks[1:]:
                stitch_block = block.overlap(stitch_size, field_name)
                block_bcube = bcube.reset_coords(
                    zs=block.start, ze=block.start + 1, in_place=False
                )
                align_block_job_forv = VoteOverZJob(
                    input_field=stitch_block.dst_stack[block_field],
                    output_field=stitching_field,
                    chunk_xy=chunk_xy,
                    bcube=block_bcube,
                    z_list=range(stitch_block.start, stitch_block.stop),
                    mip=processor_mip[0],
                )
                scheduler.register_job(
                    align_block_job_forv,
                    job_name=f"Stitching Vote {block} {block_bcube}",
                )

            scheduler.execute_until_completion()
            corgie_logger.debug("Done!")

    composed_field = dst_stack.create_sublayer(
        "composed_field", layer_type="field", overwrite=True
    )
    if restart_stage <= 2:
        corgie_logger.debug("Stitching blocks...")
        for block in blocks[1:]:
            block_bcube = block.get_bcube(bcube)
            block_list = block.get_neighbors(dist=decay_dist)
            corgie_logger.debug(f"src_block: {block}")
            corgie_logger.debug(f"influencing blocks: {block_list}")
            z_list = [b.stop for b in block_list]

            # stitching_field used if there is multi-section block overlap,
            # which requires voting to produce a corrected field.
            # If there is only single-section block overlap, then use fields
            # from each block.
            if stitching_field is not None:
                input_fields = [stitching_field]
            else:
                # Order with furthest block first (convention of FieldSet).
                input_fields = [
                    block.previous.dst_stack[field_name],
                    block.dst_stack[field_name],
                ]

            broadcast_job = BroadcastJob(
                input_fields=input_fields,
                output_field=composed_field,
                chunk_xy=chunk_xy,
                bcube=block_bcube,
                pad=pad,
                z_list=z_list,
                mip=processor_mip[0],
                decay_dist=decay_dist,
            )
            scheduler.register_job(
                broadcast_job, job_name=f"Broadcast {block} {block_bcube}"
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    if restart_stage <= 3:
        for block in blocks[:1]:
            block_bcube = block.get_bcube(bcube)
            first_src_stack = deepcopy(src_stack)
            first_src_stack.add_layer(even_stack[field_name])
            render_job = RenderJob(
                src_stack=first_src_stack,
                dst_stack=dst_stack,
                mips=processor_mip[0],
                pad=pad,
                bcube=block_bcube,
                chunk_xy=chunk_xy,
                chunk_z=1,
                render_masks=True,
                blackout_masks=False,
            )
            scheduler.register_job(
                render_job, job_name=f"Render first block {block_bcube}"
            )
        if len(blocks) > 1:
            block_bcube = bcube.reset_coords(
                zs=blocks[0].start, ze=blocks[-1].stop, in_place=False
            )
            src_stack.add_layer(composed_field)
            render_job = RenderJob(
                src_stack=src_stack,
                dst_stack=dst_stack,
                mips=processor_mip[0],
                pad=pad,
                bcube=block_bcube,
                chunk_xy=chunk_xy,
                chunk_z=1,
                render_masks=True,
                blackout_masks=False,
            )
            scheduler.register_job(
                render_job, job_name=f"Render remaining blocks {block_bcube}"
            )
        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    result_report = (
        f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
