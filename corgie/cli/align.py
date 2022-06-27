import click
import os
import math
from corgie.block import get_blocks
from copy import deepcopy

from corgie import exceptions, stack, helpers, scheduling, argparsers

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
from corgie.cli.copy import CopyLayerJob
from corgie.cli.downsample import DownsampleJob
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.seethrough import SeethroughCompareJob
from corgie.cli.vote import VoteJob
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
@corgie_option('--align_blocks/--no_align_blocks', default=True)
@corgie_option('--compute_stitch_fields/--no_compute_stitch_fields', default=True)
@corgie_option('--compose_fields/--no_compose_fields', default=True)
@corgie_option('--final_render/--no_final_render', default=True)
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
@corgie_option("--seethrough_spec_mip", nargs=1, type=int, default=None)
@corgie_optgroup("Voting Specification")
@corgie_option("--consensus_threshold", nargs=1, type=float, default=3.)
@corgie_option("--blur_sigma", nargs=1, type=float, default=15.0)
@corgie_option("--kernel_size", nargs=1, type=int, default=32)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_optgroup("Block Alignment Specification")
@corgie_option(
    "--bad_starter_path",
    nargs=1,
    type=str,
    default=None,
    help="Filepath to list of sections that should not be used as the first section of any block.",
)
@corgie_option("--block_size", nargs=1, type=int, default=10)
@corgie_option(
    "--stitch_size",
    nargs=1,
    type=int,
    default=3,
    help="The number of sections used to compute stitching fields. If >1, then multiple stitching field estimates will be made, then corrected by voting.",
)
@corgie_option(
    "--vote_dist",
    nargs=1,
    type=int,
    default=1,
    help="The number of previous sections for which fields will be estimated, then corrected by voting.",
)
@corgie_option(
    "--decay_dist",
    nargs=1,
    type=int,
    default=100,
    help="The maximum distance in sections over which a previous section may influence a later one.",
)
@corgie_option(
    "--blur_rate",
    nargs=1,
    type=float,
    default=0.2,
    help="The increase in the size of downsample factor based on distance used in broadcasting a stitching field.",
)
@corgie_option(
    "--restart_stage",
    nargs=1,
    type=int,
    default=0,
    help="0: align blocks, 1: align overlaps, 2: vote over overlaps, 3: broadcast stitch field, 4: render with composed field",
)
@corgie_option(
    "--restart_suffix",
    nargs=1,
    type=str,
    default=None,
    help="The suffix of a previous alignment run, which will be used as a starting point. New layers created after the restart will be labeld with --suffix",
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
    consensus_threshold,
    blur_sigma,
    kernel_size,
    blend_xy,
    force_chunk_xy,
    suffix,
    seethrough_spec,
    seethrough_limit,
    seethrough_spec_mip,
    decay_dist,
    blur_rate,
    restart_stage,
    restart_suffix,
    align_blocks,
    compute_stitch_fields,
    compose_fields,
    final_render,
):

    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_aligned"
    else:
        suffix = f"_{suffix}"
    if (restart_suffix is None) or (restart_stage == 0):
        restart_suffix = suffix

    if crop is None:
        crop = pad

    corgie_logger.debug("Setting up layers...")

    # TODO: store stitching images in layer other than even & odd
    if vote_dist + stitch_size - 2 >= block_size:
        raise exceptions.CorgieException(
            "block_size too small for stitching + voting requirements (stitch_size + vote_dist)"
        )

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
        suffix=restart_suffix,
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

    corgie_logger.debug("Calculating blocks...")
    skip_list = []
    if bad_starter_path is not None:
        with open(bad_starter_path) as f:
            line = f.readline()
            while line:
                skip_list.append(int(line))
                line = f.readline()
    blocks = get_blocks(
        start=bcube.z_range()[0],
        stop=bcube.z_range()[1],
        block_size=block_size,
        block_overlap=1,
        skip_list=skip_list,
        src_stack=src_stack,
        even_stack=even_stack,
        odd_stack=odd_stack,
    )
    stitch_blocks = [b.overlap(stitch_size) for b in blocks[1:]]
    corgie_logger.debug("All Blocks")
    for block, stitch_block in zip(blocks, [None] + stitch_blocks):
        corgie_logger.debug(block)
        corgie_logger.debug(f"Stitch {stitch_block}")
        corgie_logger.debug("\n")

    max_blur_mip = (
        math.ceil(math.log(decay_dist * blur_rate + 1, 2)) + processor_mip[-1]
    )
    corgie_logger.debug(f"Max blur mip for stitching field: {max_blur_mip}")

    # Set all field names, adjusting for restart suffix
    block_field_name = f"field{suffix}"
    stitch_estimated_suffix = f"_stitch_estimated{suffix}"
    stitch_estimated_name = f"field{stitch_estimated_suffix}"
    stitch_corrected_name = f"stitch_corrected{suffix}"
    stitch_corrected_field = None
    composed_name = f"composed{suffix}"
    if restart_stage <= 2:
        stitch_estimated_suffix = f"_stitch_estimated{restart_suffix}"
        stitch_estimated_name = f"field{stitch_estimated_suffix}"
        stitch_corrected_name = f"stitch_corrected{restart_suffix}"
    if restart_stage <= 3:
        composed_name = f"composed{restart_suffix}"

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
    if seethrough_spec != tuple():
        assert seethrough_spec_mip is not None
        seethrough_method = helpers.PartialSpecification(
            f=SeethroughCompareJob,
            mip=seethrough_spec_mip,
            processor_spec=seethrough_spec,
            chunk_xy=chunk_xy,
            pad=pad,
            crop=pad,
            seethrough_limit=seethrough_limit,
        )
    else:
        seethrough_method = None

    #restart_stage = 4
    #import pdb; pdb.set_trace()
    if restart_stage == 0 and align_blocks:
        corgie_logger.debug("Aligning blocks...")
        for block in blocks:
            block_bcube = block.get_bcube(bcube)
            # Use copies of src & dst so that aligning the stitching blocks
            # is not affected by these block fields.
            # Copying also allows local compute to not modify objects for other tasks
            align_block_job_forv = AlignBlockJob(
                src_stack=deepcopy(block.src_stack),
                dst_stack=deepcopy(block.dst_stack),
                bcube=block_bcube,
                render_method=render_method,
                cf_method=cf_method,
                vote_dist=vote_dist,
                seethrough_method=seethrough_method,
                suffix=suffix,
                copy_start=True,
                use_starters=True,
                backward=False,
                consensus_threshold=consensus_threshold,
                blur_sigma=blur_sigma,
                kernel_size=kernel_size,
            )
            scheduler.register_job(
                align_block_job_forv, job_name=f"Forward Align {block} {block_bcube}",
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    if restart_stage <= 1 and compute_stitch_fields:
        corgie_logger.debug("Aligning stitching blocks...")
        for stitch_block in stitch_blocks:
            block_bcube = stitch_block.get_bcube(bcube)
            # These blocks will have block-aligned images, but not
            # the block_fields that warped them.
            align_block_job_forv = AlignBlockJob(
                src_stack=deepcopy(stitch_block.src_stack),
                dst_stack=deepcopy(stitch_block.dst_stack),
                bcube=block_bcube,
                render_method=render_method,
                cf_method=cf_method,
                vote_dist=vote_dist,
                seethrough_method=seethrough_method,
                suffix=stitch_estimated_suffix,
                copy_start=False,
                use_starters=False,
                backward=False,
                consensus_threshold=consensus_threshold,
                blur_sigma=blur_sigma,
                kernel_size=kernel_size,
            )
            scheduler.register_job(
                align_block_job_forv,
                job_name=f"Stitch Align {stitch_block} {block_bcube}",
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    # Add in the stitch_estimated fields that were just created above
    even_stack.create_sublayer(
        stitch_estimated_name, layer_type="field", overwrite=False,
    )
    odd_stack.create_sublayer(
        stitch_estimated_name, layer_type="field", overwrite=False,
    )
    if restart_stage <= 2 and compose_fields:
        if stitch_size > 1:
            corgie_logger.debug("Voting over stitching blocks")
            stitch_corrected_field = dst_stack.create_sublayer(
                stitch_corrected_name, layer_type="field", overwrite=True
            )
            for stitch_block in stitch_blocks:
                stitch_estimated_field = stitch_block.dst_stack[stitch_estimated_name]
                block_bcube = bcube.reset_coords(
                    zs=stitch_block.start, ze=stitch_block.start + 1, in_place=False,
                )
                z_offsets = [z - block_bcube.z_range()[0] for z in range(stitch_block.start, stitch_block.stop)]
                vote_stitch_job = VoteJob(
                    input_fields=[stitch_estimated_field],
                    output_field=stitch_corrected_field,
                    chunk_xy=chunk_xy,
                    bcube=block_bcube,
                    z_offsets=z_offsets,
                    mip=processor_mip[-1],
                    consensus_threshold=consensus_threshold,
                    blur_sigma=blur_sigma,
                    kernel_size=kernel_size,
                )
                scheduler.register_job(
                    vote_stitch_job,
                    job_name=f"Stitching Vote {stitch_block} {block_bcube}",
                )

            scheduler.execute_until_completion()
            corgie_logger.debug("Done!")

        for stitch_block in stitch_blocks:
            block_bcube = bcube.reset_coords(
                zs=stitch_block.start, ze=stitch_block.start + 1, in_place=False
            )
            field_to_downsample = stitch_block.dst_stack[stitch_estimated_name]
            if stitch_corrected_field is not None:
                field_to_downsample = stitch_corrected_field
            # Hack for fafb
            field_info = field_to_downsample.get_info()
            for scale in field_info['scales']:
                scale['chunk_sizes'][-1][-1] = 1
                scale['encoding'] = 'raw'
            field_to_downsample.cv.store_info(field_info)
            field_to_downsample.cv.fetch_info()
            downsample_field_job = DownsampleJob(
                src_layer=field_to_downsample,
                mip_start=processor_mip[-1],
                mip_end=max_blur_mip,
                bcube=block_bcube,
                chunk_xy=chunk_xy,  # TODO: This probably needs to be modified at highest mips
                chunk_z=1,
                mips_per_task=2,
            )
            scheduler.register_job(
                downsample_field_job,
                job_name=f"Downsample stitching field {block_bcube}",
            )
        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    # Add in the block-align fields
    even_stack.create_sublayer(
        block_field_name, layer_type="field", overwrite=False,
    )
    odd_stack.create_sublayer(
        block_field_name, layer_type="field", overwrite=False,
    )
    composed_field = dst_stack.create_sublayer(
        composed_name, layer_type="field", overwrite=True
    )
    if (restart_stage > 2) and (stitch_size > 1):
        stitch_corrected_field = dst_stack.create_sublayer(
            stitch_corrected_name, layer_type="field", overwrite=False
        )
    if restart_stage <= 3 and compose_fields:
        corgie_logger.debug("Stitching blocks...")
        for block, stitch_block in zip(blocks[1:], stitch_blocks):
            block_bcube = block.broadcastable().get_bcube(bcube)
            block_list = block.get_neighbors(dist=decay_dist)
            corgie_logger.debug(f"src_block: {block}")
            corgie_logger.debug(f"influencing blocks: {block_list}")
            z_list = [b.stop for b in block_list]
            # stitch_corrected_field used if there is multi-section block overlap,
            # which requires voting to produce a corrected field.
            # If there is only single-section block overlap, then use
            # stitch_estimated_fields from each stitch_block
            if stitch_corrected_field is not None:
                stitching_fields = [stitch_corrected_field]
            else:
                # Order with furthest block first (convention of FieldSet).
                stitching_fields = [
                    stitch_block.dst_stack[stitch_estimated_name],
                    stitch_block.src_stack[stitch_estimated_name],
                ]

            broadcast_job = BroadcastJob(
                block_field=block.dst_stack[block_field_name],
                stitching_fields=stitching_fields,
                output_field=composed_field,
                chunk_xy=chunk_xy,
                bcube=block_bcube,
                pad=pad,
                z_list=z_list,
                mip=processor_mip[-1],
                decay_dist=decay_dist,
                blur_rate=blur_rate,
            )
            scheduler.register_job(
                broadcast_job, job_name=f"Broadcast {block} {block_bcube}"
            )

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

        if len(blocks) > 1:
            block_bcube = blocks[0].get_bcube(bcube)
            copy_job = CopyLayerJob(
                src_layer=even_stack[block_field_name],
                dst_layer=composed_field,
                mip=processor_mip[-1],
                bcube=block_bcube,
                chunk_xy=chunk_xy,
                chunk_z=1,
            )
            scheduler.register_job(
                copy_job, job_name=f"Copy first block_field to composed_field location"
            )
            scheduler.execute_until_completion()
            corgie_logger.debug("Done!")

    if restart_stage <= 4 and final_render:
        if len(blocks) == 1:
            block_bcube = blocks[0].get_bcube(bcube)
            render_job = RenderJob(
                src_stack=src_stack,
                dst_stack=dst_stack,
                mips=processor_mip[-1],
                pad=pad,
                bcube=block_bcube,
                chunk_xy=chunk_xy,
                chunk_z=1,
                render_masks=True,
                blackout_masks=False,
                additional_fields=[even_stack[block_field_name]],
            )
            scheduler.register_job(
                render_job, job_name=f"Render first block {block_bcube}"
            )
        else:
            block_bcube = bcube.reset_coords(
                zs=blocks[0].start, ze=blocks[-1].stop, in_place=False
            )
            render_job = RenderJob(
                src_stack=src_stack,
                dst_stack=dst_stack,
                mips=processor_mip[-1],
                pad=pad,
                bcube=block_bcube,
                chunk_xy=chunk_xy,
                chunk_z=1,
                render_masks=True,
                blackout_masks=True,
                additional_fields=[composed_field],
            )
            scheduler.register_job(
                render_job, job_name=f"Render all blocks {block_bcube}"
            )
        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    result_report = (
        f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
