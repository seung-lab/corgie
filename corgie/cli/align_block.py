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
from corgie.cli.vote import VoteJob
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.seethrough import SeethroughCompareJob

from corgie.cli.downsample import DownsampleJob
from corgie.cli.upsample import UpsampleJob


class AlignBlockJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_stack,
        cf_method,
        render_method,
        bcube,
        seethrough_method=None,
        copy_start=True,
        backward=False,
        vote_dist=1,
        suffix=None,
        consensus_threshold=3.,
        blur_sigma=15,
        kernel_size=32,
        use_starters=True,
    ):
        """Align block with and without voting

        Args:
            use_starters (bool): whether to use starter sections or not
        """
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.bcube = bcube
        self.seethrough_method = seethrough_method

        self.cf_method = cf_method
        self.render_method = render_method

        self.copy_start = copy_start
        self.use_starters = use_starters
        self.backward = backward
        self.vote_dist = vote_dist
        self.suffix = suffix

        self.consensus_threshold = consensus_threshold
        self.blur_sigma = blur_sigma
        self.kernel_size = kernel_size

        self.tgt_stack = deepcopy(self.dst_stack)
        field_dir = f"field{self.suffix}"
        self.final_field = self.dst_stack.create_sublayer(
            field_dir, layer_type="field", overwrite=True
        )
        self.estimated_fields = {}
        self.z_offsets = range(1, self.vote_dist + 1)

        if self.backward:
            z_step = -1
        else:
            z_step = 1

        for k in self.z_offsets:
            offset = -k * z_step
            f = self.dst_stack.create_sublayer(
                f"{field_dir}/{offset}", layer_type="field", overwrite=True
            )
            self.estimated_fields[offset] = f

        if self.seethrough_method is not None:
            self.seethrough_mask_layer = self.dst_stack.create_sublayer(
                f"seethrough_mask{self.suffix}",
                layer_type="mask",
                overwrite=True,
            )
            self.pixel_offset_layer = self.tgt_stack.create_unattached_sublayer(
                f"pixel_offset{self.suffix}", layer_type="img", overwrite=True
            )
        else:
            self.seethrough_mask_layer = None
            self.pixel_offset_layer = None

        super().__init__()

    def task_generator(self):
        if not self.backward:
            z_step = 1
            z_start = self.bcube.z_range()[0]
            z_end = self.bcube.z_range()[1]
        else:
            z_step = -1
            z_start = self.bcube.z_range()[1]
            z_end = self.bcube.z_range()[0]
        seethrough_offset = -z_step
        if (self.vote_dist > 1) and self.use_starters:
            starter_section_start = z_start - (self.vote_dist - 1) * z_step
        else:
            starter_section_start = z_start
        z_range = range(starter_section_start, z_end, z_step)

        processor_mips = set(self.cf_method.processor_mip)
        processor_and_seethrough_mips = processor_mips.copy()
        if self.seethrough_method is not None:
            processor_and_seethrough_mips.add(self.seethrough_method.mip)

        corgie_logger.debug(
            f"Serial alignment, {z_start}->{z_end}, use_starters={self.use_starters}"
        )
        for z in z_range:
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            # COPY FIRST SECTION OF THE BLOCK
            if (z == z_start) and self.copy_start:
                corgie_logger.debug(f"Copy section {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.tgt_stack,
                    bcube=bcube,
                    mips=processor_and_seethrough_mips,
                    blackout_masks=True,
                )

                yield from render_job.task_generator
                yield scheduling.wait_until_done
            # CREATE STARTER SECTIONS FOR VOTING
            elif (z_step > 0 and z < z_start) or (z_step < 0 and z > z_start):
                corgie_logger.debug(f"Compute field vote starter {z_start}<{z}")
                offset = z_start - z
                compute_field_job = self.cf_method(
                    src_stack=self.src_stack,
                    tgt_stack=self.src_stack,
                    bcube=bcube,
                    tgt_z_offset=offset,
                    suffix=self.suffix,
                    dst_layer=self.final_field,
                )
                yield from compute_field_job.task_generator
                yield scheduling.wait_until_done

                corgie_logger.debug(f"Render vote starter {z_start}<{z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.tgt_stack,
                    bcube=bcube,
                    blackout_masks=True,
                    seethrough_mask_layer=None,  # No seethrough for starter sections
                    mips=self.cf_method.processor_mip,
                    additional_fields=[self.final_field],
                )
                yield from render_job.task_generator
                yield scheduling.wait_until_done

            # SERIAL ALIGNMENT (w/ or w/o voting)
            else:
                if self.vote_dist > 1:
                    for k in self.z_offsets:
                        offset = -k * z_step
                        corgie_logger.debug(f"Compute field {z+offset}<{z}")
                        compute_field_job = self.cf_method(
                            src_stack=self.src_stack,
                            tgt_stack=self.tgt_stack,
                            bcube=bcube,
                            tgt_z_offset=offset,
                            suffix=self.suffix,
                            dst_layer=self.estimated_fields[offset],
                        )

                        yield from compute_field_job.task_generator

                    yield scheduling.wait_until_done
                elif self.vote_dist == 1:
                    offset = -z_step
                    corgie_logger.debug(f"Compute final field field {z+offset}<{z}")
                    compute_field_job = self.cf_method(
                        src_stack=self.src_stack,
                        tgt_stack=self.tgt_stack,
                        bcube=bcube,
                        tgt_z_offset=offset,
                        suffix=self.suffix,
                        dst_layer=self.final_field,
                    )

                    yield from compute_field_job.task_generator
                    yield scheduling.wait_until_done
                else:
                    raise Exception()

                if self.vote_dist > 1:
                    corgie_logger.debug(f"Vote {z}")
                    chunk_xy = self.cf_method["chunk_xy"]
                    mip = self.cf_method["processor_mip"][-1]
                    ordered_fields = [self.estimated_fields[-k*z_step] for k in self.z_offsets]
                    vote_job = VoteJob(
                        input_fields=ordered_fields,
                        output_field=self.final_field,
                        chunk_xy=chunk_xy,
                        bcube=bcube,
                        z_offsets=[0],
                        mip=mip,
                        consensus_threshold=self.consensus_threshold,
                        blur_sigma=self.blur_sigma,
                        kernel_size=self.kernel_size,
                    )

                    yield from vote_job.task_generator
                    yield scheduling.wait_until_done

                if self.seethrough_method is not None:
                    # This sequence can be bundled into a "seethrough render" job
                    # First, adjust the field to the appropriate MIP for seethrough rendering
                    if self.cf_method["processor_mip"][-1] < self.seethrough_method.mip:
                        downsample_job = DownsampleJob(
                            src_layer=self.final_field,
                            chunk_xy=self.cf_method["chunk_xy"],
                            chunk_z=1,
                            mip_start=self.cf_method["processor_mip"][-1],
                            mip_end=self.seethrough_method.mip,
                            bcube=bcube,
                        )
                        yield from downsample_job.task_generator
                        yield scheduling.wait_until_done
                    elif (
                        self.cf_method["processor_mip"][-1] > self.seethrough_method.mip
                    ):
                        upsample_job = UpsampleJob(
                            src_layer=self.final_field,
                            chunk_xy=self.cf_method["chunk_xy"],
                            chunk_z=1,
                            mip_start=self.cf_method["processor_mip"][-1],
                            mip_end=self.seethrough_method.mip,
                            bcube=bcube,
                        )
                        yield from upsample_job.task_generator
                        yield scheduling.wait_until_done

                    # Next, render the images at the md mip level
                    # This could mean a reduntant render step, but that's fine
                    render_job = self.render_method(
                        src_stack=self.src_stack,
                        dst_stack=self.tgt_stack,
                        bcube=bcube,
                        blackout_masks=True,
                        preserve_zeros=True,
                        additional_fields=[self.final_field],
                        mips=self.seethrough_method.mip,
                    )

                    yield from render_job.task_generator
                    yield scheduling.wait_until_done

                    # Now, we'll apply misalignment detection to produce a mask
                    # this mask will be used in the final render step
                    seethrough_mask_job = self.seethrough_method(
                        src_stack=self.tgt_stack,  # we're looking for misalignments in the final stack
                        bcube=bcube,
                        tgt_z_offset=-z_step,
                        suffix=self.suffix,
                        dst_layer=self.seethrough_mask_layer,
                        pixel_offset_layer=self.pixel_offset_layer,
                    )

                    yield from seethrough_mask_job.task_generator
                    yield scheduling.wait_until_done

                    # We'll downsample the mask to be available at all mip levels
                    downsample_job = DownsampleJob(
                        src_layer=self.seethrough_mask_layer,
                        chunk_xy=self.seethrough_method.chunk_xy,
                        chunk_z=1,
                        mip_start=self.seethrough_method.mip,
                        mip_end=max(self.cf_method.processor_mip),
                        bcube=bcube,
                    )
                    upsample_job = UpsampleJob(
                        src_layer=self.seethrough_mask_layer,
                        chunk_xy=self.seethrough_method.chunk_xy,
                        chunk_z=1,
                        mip_start=self.seethrough_method.mip,
                        mip_end=min(self.cf_method.processor_mip),
                        bcube=bcube,
                    )

                    if (
                        min(self.cf_method.processor_mip) < self.seethrough_method.mip
                        or max(self.cf_method.processor_mip)
                        > self.seethrough_method.mip
                    ):
                        yield from downsample_job.task_generator
                        yield from upsample_job.task_generator
                        yield scheduling.wait_until_done

                corgie_logger.debug(f"Render {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.tgt_stack,
                    bcube=bcube,
                    blackout_masks=True,
                    seethrough_mask_layer=self.seethrough_mask_layer,
                    seethrough_offset=seethrough_offset,
                    mips=min(self.cf_method.processor_mip),
                    additional_fields=[self.final_field],
                )
                yield from render_job.task_generator
                yield scheduling.wait_until_done

                # We'll downsample the tgt_stack for one-pass
                if min(self.cf_method.processor_mip) != max(
                    self.cf_method.processor_mip
                ):
                    dst_layer = self.tgt_stack.get_layers_of_type("img")[0]
                    downsample_job = DownsampleJob(
                        src_layer=dst_layer,
                        chunk_xy=self.cf_method.chunk_xy,
                        chunk_z=1,
                        mip_start=min(self.cf_method.processor_mip),
                        mip_end=max(self.cf_method.processor_mip),
                        bcube=bcube,
                    )
                    yield from downsample_job.task_generator
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
@corgie_option("--seethrough_spec_mip", nargs=1, type=int, default=None)
@corgie_option("--render_pad", nargs=1, type=int, default=512)
@corgie_option("--render_chunk_xy", nargs=1, type=int, default=1024)
@corgie_optgroup("Compute Field Method Specification")
@corgie_option("--processor_spec", nargs=1, type=str, required=True, multiple=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--pad", nargs=1, type=int, default=256)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_mip", "-m", nargs=1, type=int, required=True, multiple=True)
@corgie_option("--copy_start/--no_copy_start", default=True)
@corgie_option("--use_starters/--no_starters", default=True)
@corgie_option(
    "--mode",
    type=click.Choice(["forward", "backward", "bidirectional"]),
    default="forward",
)
@corgie_option("--vote_dist", nargs=1, type=int, default=1)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def align_block(
    ctx,
    src_layer_spec,
    dst_folder,
    vote_dist,
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
    blend_xy,
    force_chunk_xy,
    suffix,
    copy_start,
    use_starters,
    seethrough_spec,
    seethrough_limit,
    seethrough_spec_mip,
    mode,
    chunk_z=1,
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

    render_method = helpers.PartialSpecification(
        f=RenderJob,
        pad=render_pad,
        chunk_xy=render_chunk_xy,
        chunk_z=1,
        render_masks=False,
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

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if mode == "bidirectional":
        z_mid = (bcube.z_range()[1] + bcube.z_range()[0]) // 2
        bcube_back = bcube.reset_coords(ze=z_mid, in_place=False)
        bcube_forv = bcube.reset_coords(zs=z_mid, in_place=False)

        align_block_job_back = AlignBlockJob(
            src_stack=src_stack,
            dst_stack=dst_stack,
            bcube=bcube_back,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=copy_start,
            backward=True,
            vote_dist=vote_dist,
            use_starters=use_starters,
        )
        scheduler.register_job(
            align_block_job_back,
            job_name="Backward Align Block {}".format(bcube),
        )

        align_block_job_forv = AlignBlockJob(
            src_stack=src_stack,
            dst_stack=deepcopy(dst_stack),
            bcube=bcube_forv,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=True,
            backward=False,
            vote_dist=vote_dist,
            use_starters=use_starters,
        )
        scheduler.register_job(
            align_block_job_forv,
            job_name="Forward Align Block {}".format(bcube),
        )
    else:
        align_block_job = AlignBlockJob(
            src_stack=src_stack,
            dst_stack=dst_stack,
            bcube=bcube,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=copy_start,
            backward=mode == "backward",
            vote_dist=vote_dist,
            use_starters=use_starters,
        )

        # create scheduler and execute the job
        scheduler.register_job(align_block_job, job_name="Align Block {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = (
        f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
