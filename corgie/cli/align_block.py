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
from corgie.cli.vote import VoteOverFieldsJob
from corgie.cli.compute_field import ComputeFieldJob
from corgie.cli.compare_sections import CompareSectionsJob

from corgie.cli.downsample import DownsampleJob
from corgie.cli.upsample import UpsampleJob


class AlignBlockJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        tgt_stack,
        dst_stack,
        cf_method,
        render_method,
        bcube,
        seethrough_method=None,
        copy_start=True,
        backward=False,
        vote_dist=1,
        suffix=None,
        softmin_temp=None,
        blur_sigma=1.0,
        resume_index=0,
        resume_stage=0,
    ):
        """Align block with and without voting

        Args:
            resume_index (int): indicates index in block
            resume_stage (int): indicates stage (compute field/vote/render)
        """
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_stack = dst_stack
        self.bcube = bcube
        self.seethrough_method = seethrough_method

        self.cf_method = cf_method
        self.render_method = render_method

        self.copy_start = copy_start
        self.backward = backward
        self.vote_dist = vote_dist
        self.suffix = suffix
        self.resume_index = resume_index
        self.resume_stage = resume_stage

        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma

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

        # Set up voting layers
        if self.vote_dist > 1:
            starter_section_start = z_start - self.vote_dist * z_step
        else:
            starter_section_start = z_start

        z_resume = list(range(starter_section_start, z_end, z_step))[self.resume_index]

        field_dir = f"field{self.suffix}"
        final_field = self.dst_stack.create_sublayer(
            field_dir, layer_type="field", overwrite=True
        )
        estimated_fields = {}
        for k in range(1, self.vote_dist + 1):
            offset = -k * z_step
            f = self.dst_stack.create_sublayer(
                f"{field_dir}/{offset}", layer_type="field", overwrite=True
            )
            estimated_fields[offset] = f

        # Set up seethrough layers
        if self.seethrough_method is not None:
            seethrough_mask_layer = self.dst_stack.create_sublayer(
                f"seethrough_mask{self.suffix}", layer_type="mask", overwrite=True
            )
        else:
            seethrough_mask_layer = None

        corgie_logger.debug(f"Serial alignment, {z_start}:{z_end}")
        if z_start != z_resume:
            corgie_logger.debug(f"Resume at {z_resume}")

        for z in range(z_resume, z_end, z_step):
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            # COPY FIRST SECTION OF THE BLOCK
            if (z == z_start) and self.copy_start:
                corgie_logger.debug(f"Copy section {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=bcube,
                    mips=self.cf_method.processor_mip,
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
                    dst_layer=final_field,
                )
                yield from compute_field_job.task_generator
                yield scheduling.wait_until_done

                if self.seethrough_method is not None:
                    # This sequence can be bundled into a "seethrough render" job

                    # First, render the images at the md mip level
                    # This could mean a reduntant render step, but that's fine
                    render_job = self.render_method(
                        src_stack=self.src_stack,
                        dst_stack=self.dst_stack,
                        bcube=bcube,
                        blackout_masks=False,
                        preserve_zeros=True,
                        additional_fields=[final_field],
                        mips=self.seethrough_method.mip,
                    )

                    yield from render_job.task_generator
                    yield scheduling.wait_until_done

                    # Now, we'll apply misalignment detection to produce a mask
                    # this mask will be used in the final render step
                    seethrough_mask_job = self.seethrough_method(
                        src_stack=self.dst_stack,  # we're looking for misalignments in the final stack
                        bcube=bcube,
                        tgt_z_offset=offset,
                        suffix=self.suffix,
                        dst_layer=seethrough_mask_layer,
                    )

                    yield from seethrough_mask_job.task_generator
                    yield scheduling.wait_until_done

                    # We'll downsample the mask to be available at all mip levels
                    downsample_job = DownsampleJob(
                        src_layer=seethrough_mask_layer,
                        chunk_xy=self.seethrough_method.chunk_xy,
                        chunk_z=1,
                        mip_start=self.seethrough_method.mip,
                        mip_end=max(self.cf_method.processor_mip),
                        bcube=self.bcube,
                    )
                    upsample_job = UpsampleJob(
                        src_layer=seethrough_mask_layer,
                        chunk_xy=self.seethrough_method.chunk_xy,
                        chunk_z=1,
                        mip_start=self.seethrough_method.mip,
                        mip_end=min(self.cf_method.processor_mip),
                        bcube=self.bcube,
                    )

                corgie_logger.debug(f"Render vote starter {z_start}<{z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=bcube,
                    blackout_masks=True,
                    seethrough_mask_layer=seethrough_mask_layer,
                    mips=self.cf_method.processor_mip,
                    additional_fields=[final_field],
                )
                yield from render_job.task_generator
                yield scheduling.wait_until_done

            # SERIAL ALIGNMENT (w/ or w/o voting)
            else:
                if (z_resume != z) or (self.resume_stage < 1):
                    if self.vote_dist > 1:
                        for k in range(1, self.vote_dist + 1):
                            offset = -k * z_step
                            corgie_logger.debug(f"Compute field {z+offset}<{z}")
                            compute_field_job = self.cf_method(
                                src_stack=self.src_stack,
                                tgt_stack=self.dst_stack,
                                bcube=bcube,
                                tgt_z_offset=offset,
                                suffix=self.suffix,
                                dst_layer=estimated_fields[offset],
                            )

                            yield from compute_field_job.task_generator

                        yield scheduling.wait_until_done
                    elif self.vote_dist == 1:
                        offset = -z_step
                        corgie_logger.debug(f"Compute final field field {z+offset}<{z}")
                        compute_field_job = self.cf_method(
                            src_stack=self.src_stack,
                            tgt_stack=self.dst_stack,
                            bcube=bcube,
                            tgt_z_offset=offset,
                            suffix=self.suffix,
                            dst_layer=final_field,
                        )

                        yield from compute_field_job.task_generator
                        yield scheduling.wait_until_done
                    else:
                        raise Exception()

                if (z_resume != z) or (self.resume_stage < 2):
                    if self.vote_dist > 1:
                        corgie_logger.debug(f"Vote {z}")
                        chunk_xy = self.cf_method["chunk_xy"]
                        chunk_z = self.cf_method["chunk_z"]
                        mip = self.cf_method["processor_mip"][0]
                        vote_job = VoteOverFieldsJob(
                            input_fields=estimated_fields,
                            output_field=final_field,
                            chunk_xy=chunk_xy,
                            bcube=bcube,
                            mip=mip,
                            softmin_temp=self.softmin_temp,
                            blur_sigma=self.blur_sigma,
                        )

                        yield from vote_job.task_generator
                        yield scheduling.wait_until_done

                    if self.seethrough_method is not None:
                        # This sequence can be bundled into a "seethrough render" job

                        # First, render the images at the md mip level
                        # This could mean a reduntant render step, but that's fine
                        render_job = self.render_method(
                            src_stack=self.src_stack,
                            dst_stack=self.dst_stack,
                            bcube=bcube,
                            blackout_masks=False,
                            preserve_zeros=True,
                            additional_fields=[final_field],
                            mips=self.seethrough_method.mip,
                        )

                        yield from render_job.task_generator
                        yield scheduling.wait_until_done

                        # Now, we'll apply misalignment detection to produce a mask
                        # this mask will be used in the final render step
                        seethrough_mask_job = self.seethrough_method(
                            src_stack=self.dst_stack,  # we're looking for misalignments in the final stack
                            bcube=bcube,
                            tgt_z_offset=-z_step,
                            suffix=self.suffix,
                            dst_layer=seethrough_mask_layer,
                        )

                        yield from seethrough_mask_job.task_generator
                        yield scheduling.wait_until_done

                        # We'll downsample the mask to be available at all mip levels
                        downsample_job = DownsampleJob(
                            src_layer=seethrough_mask_layer,
                            chunk_xy=self.seethrough_method.chunk_xy,
                            chunk_z=1,
                            mip_start=self.seethrough_method.mip,
                            mip_end=max(self.cf_method.processor_mip),
                            bcube=self.bcube,
                        )
                        upsample_job = UpsampleJob(
                            src_layer=seethrough_mask_layer,
                            chunk_xy=self.seethrough_method.chunk_xy,
                            chunk_z=1,
                            mip_start=self.seethrough_method.mip,
                            mip_end=min(self.cf_method.processor_mip),
                            bcube=self.bcube,
                        )

                        if (
                            min(self.cf_method.processor_mip)
                            < self.seethrough_method.mip
                            or max(self.cf_method.processor_mip)
                            > self.seethrough_method.mip
                        ):
                            yield from downsample_job.task_generator
                            yield from upsample_job.task_generator
                            yield scheduling.wait_until_done

                corgie_logger.debug(f"Render {z}")
                render_job = self.render_method(
                    src_stack=self.src_stack,
                    dst_stack=self.dst_stack,
                    bcube=bcube,
                    blackout_masks=False,
                    seethrough_mask_layer=seethrough_mask_layer,
                    seethrough_offset=seethrough_offset,
                    mips=self.cf_method.processor_mip,
                    additional_fields=[final_field],
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
@corgie_option("--seethrough_spec", nargs=1, type=str, default=None)
@corgie_option("--seethrough_spec_mip", nargs=1, type=int, default=None)
@corgie_option("--render_pad", nargs=1, type=int, default=512)
@corgie_option("--render_chunk_xy", nargs=1, type=int, default=1024)
@corgie_optgroup("Compute Field Method Specification")
@corgie_option("--processor_spec", nargs=1, type=str, required=True, multiple=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option("--force_chunk_xy", is_flag=True)
@corgie_option("--pad", nargs=1, type=int, default=256)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_mip", "-m", nargs=1, type=int, required=True, multiple=True)
@corgie_option("--copy_start/--no_copy_start", default=True)
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
@corgie_option("--resume_index", nargs=1, type=int, default=0)
@corgie_option("--resume_stage", nargs=1, type=int, default=0)
@click.pass_context
def align_block(
    ctx,
    src_layer_spec,
    tgt_layer_spec,
    dst_folder,
    vote_dist,
    resume_index,
    resume_stage,
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
    seethrough_spec,
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

    tgt_stack = create_stack_from_spec(
        tgt_layer_spec, name="tgt", readonly=True, reference=src_stack
    )

    force_chunk_xy = chunk_xy if force_chunk_xy else None
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
            tgt_stack=tgt_stack,
            dst_stack=dst_stack,
            bcube=bcube_back,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=copy_start,
            backward=True,
            vote_dist=vote_dist,
            resume_index=resume_index,
            resume_stage=resume_stage,
        )
        scheduler.register_job(
            align_block_job_back, job_name="Backward Align Block {}".format(bcube)
        )

        align_block_job_forv = AlignBlockJob(
            src_stack=src_stack,
            tgt_stack=tgt_stack,
            dst_stack=deepcopy(dst_stack),
            bcube=bcube_forv,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=True,
            backward=False,
            vote_dist=vote_dist,
            resume_index=resume_index,
            resume_stage=resume_stage,
        )
        scheduler.register_job(
            align_block_job_forv, job_name="Forward Align Block {}".format(bcube)
        )
    else:
        align_block_job = AlignBlockJob(
            src_stack=src_stack,
            tgt_stack=tgt_stack,
            dst_stack=dst_stack,
            bcube=bcube,
            render_method=render_method,
            cf_method=cf_method,
            seethrough_method=seethrough_method,
            suffix=suffix,
            copy_start=copy_start,
            backward=mode == "backward",
            vote_dist=vote_dist,
            resume_index=resume_index,
            resume_stage=resume_stage,
        )

        # create scheduler and execute the job
        scheduler.register_job(align_block_job, job_name="Align Block {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = (
        f"Aligned layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
