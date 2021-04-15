import click
import os
from corgie.block import Block
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
#
@corgie_option(
    "--tgt_layer_spec",
    "-t",
    nargs=1,
    type=str,
    required=False,
    multiple=True,
    help="Target layer spec. Use multiple times to include all masks, fields, images. \n"
    "DEFAULT: Same as source layers",
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
@corgie_option("--block_overlap", nargs=1, type=int, default=3)
@corgie_option("--vote_dist", nargs=1, type=int, default=1)
@corgie_option('--decay_dist', nargs=1, type=int, default=50)

@click.pass_context
def align(ctx,
          src_layer_spec,
          tgt_layer_spec,
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
          block_overlap,
          vote_dist,
          blend_xy,
          force_chunk_xy,
          suffix,
          copy_start,
          seethrough_spec,
          seethrough_spec_mip,
          decay_dist):

    scheduler = ctx.obj['scheduler']

    if suffix is None:
        suffix = "_aligned"
    else:
        suffix = f"_{suffix}"

    if crop is None:
        crop = pad

    corgie_logger.debug("Setting up layers...")

    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)
    src_stack.folder = dst_folder

    tgt_stack = create_stack_from_spec(
        tgt_layer_spec, name="tgt", readonly=True, reference=src_stack
    )

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

    corgie_logger.debug("Calculating blocks...")
    # TODO: read in bad starter sections
    bad_starter_sections = []

    blocks = []
    overlap_blocks = []
    z = bcube.z_range()[0]
    while z < bcube.z_range()[-1]:
        block_start = z
        block_end = z + block_size
        while (
            block_end + block_overlap in bad_starter_sections
            and block_end + block_overlap < bcube.z_range()[-1]
        ):
            block_end += 1

        block = Block(block_start, block_end + block_overlap)
        blocks.append(block)
        overlap_block = Block(block_start, block_start + block_overlap)
        overlap_blocks.append(overlap_block)
        z = block_end
    overlap_blocks = overlap_blocks[1:] # num_overlaps = num_blocks - 1
    corgie_logger.debug("Done!")

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

    corgie_logger.debug("Aligning blocks...")
    for i in range(len(blocks)):
        block = blocks[i]

        block_bcube = bcube.copy()
        block_bcube.reset_coords(zs=block.z_start, ze=block.z_end)

        if i % 2 == 0:
            block_dst_stack = even_stack
        else:
            block_dst_stack = odd_stack

        align_block_job_forv = AlignBlockJob(
            src_stack=src_stack,
            tgt_stack=tgt_stack,
            dst_stack=block_dst_stack,
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

    corgie_logger.debug("Aligning overlaps...")
    for i in range(len(blocks)):
        block = overlap_blocks[i]

        block_bcube = bcube.copy()
        block_bcube.reset_coords(zs=block.z_start, ze=block.z_end)

        # align blockB to blockA
        block_field = f'field{suffix}'
        if i % 2 == 0:
            blockA = even_stack
            blockB = deepcopy(src_stack)
            fieldB = odd_stack[block_field]
        else:
            blockA = odd_stack
            blockB = deepcopy(src_stack)
            fieldB = even_stack[block_field]
        blockB.add_layer(fieldB)

        align_block_job_forv = AlignBlockJob(src_stack=blockB,
                                    tgt_stack=blockB,
                                    dst_stack=blockA,
                                    bcube=block_bcube,
                                    render_method=render_method,
                                    cf_method=cf_method,
                                    vote_dist=vote_dist,
                                    seethrough_method=seethrough_method,
                                    suffix=suffix,
                                    copy_start=False,
                                    backward=False)
        scheduler.register_job(align_block_job_forv, job_name=f"Overlap Align {block} {block_bcube}")

    scheduler.execute_until_completion()
    corgie_logger.debug("Done!")

    stitching_field = None
    if block_overlap > 1:
        corgie_logger.debug("Voting on overlaps")
        field_dir = 'stitching_field'
        stitching_field = dst_stack.create_sublayer(field_dir,
                                            layer_type='field',
                                            overwrite=True)
        for i in range(len(blocks)):
            block = overlap_blocks[i]

            block_bcube = bcube.copy()
            block_bcube.reset_coords(zs=block.z_start, ze=block.z_end)

            block_field = f'field{suffix}'
            if i % 2 == 0:
                fieldA = even_stack[block_field]
            else:
                fieldA = odd_stack[block_field]
            z_list = range(*block_bcube.z_range)

            align_block_job_forv = VoteOverZJob(
                                        input_field=fieldA,
                                        output_field=stitching_field,
                                        chunk_xy=chunk_xy,
                                        bcube=block_bcube,
                                        z_list=z_list,
                                        mip=processor_mip)
            scheduler.register_job(align_block_job_forv, job_name=f"Overlap Vote {block} {block_bcube}")

        scheduler.execute_until_completion()
        corgie_logger.debug("Done!")

    corgie_logger.debug("Stitching blocks...")
    composed_lookup = {z: [] for z in range(*bcube.z_range())}
    for i in range(len(blocks)):
        b_start = blocks[i].z_start
        for z in range(*bcube.z_range()):
            if z < bcube.z_range()[-1]:
                composed_lookup[z].append(b_start)

    field_dir = 'composed_field'
    composed_field = dst_stack.create_sublayer(field_dir,
                                        layer_type='field',
                                        overwrite=True)
    for z in range(*bcube.z_range()):
        section_bcube = bcube.copy()
        section_bcube.reset_coords(zs=z, ze=block.z_end, inplace=True)
        z_list = composed_lookup[z]

        if stitching_field is not None:
            input_fields = [stitching_field]
        else:
            i = 0
            while block[i].z_start != z_list[0]:
                i += 1
            block_field = f'field{suffix}'
            if i % 2 == 0:
                input_fields = [even_stack[block_field], odd_stack[block_field]]
            else:
                input_fields = [odd_stack[block_field], even_stack[block_field]]

        broadcast_job = BroadcastJob(input_fields=input_fields,
                                    output_field=composed_field,
                                    chunk_xy=chunk_xy,
                                    bcube=section_bcube,
                                    pad=pad,
                                    z_list=z_list,
                                    mip=processor_mip,
                                    decay_dist=decay_dist)
        scheduler.register_job(broadcast_job, job_name=f"Broadcast {block} {block_bcube}")

    scheduler.execute_until_completion()
    corgie_logger.debug("Done!")

    # Generate stitching fields

    result_report = (
        f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
