import click
import torch
import procspec

from corgie import scheduling, helpers, stack

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)

from corgie.cli.render import RenderJob
from corgie.cli.common import ChunkedJob


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


class SeethroughCompareJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_layer,
        chunk_xy,
        processor_spec,
        mip,
        pad,
        crop,
        bcube,
        tgt_z_offset,
        tgt_stack=None,
        suffix="",
        seethrough_limit=None,
        pixel_offset_layer=None,
    ):
        self.src_stack = src_stack
        if tgt_stack is None:
            tgt_stack = src_stack

        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.pad = pad
        self.crop = crop
        self.bcube = bcube
        self.tgt_z_offset = tgt_z_offset

        self.suffix = suffix

        self.processor_spec = processor_spec
        self.mip = mip
        self.pixel_offset_layer = pixel_offset_layer
        if seethrough_limit is None or seethrough_limit == tuple():
            # If limit not specified, no limit
            self.seethrough_limit = [0] * len(self.processor_spec)
        else:
            self.seethrough_limit = seethrough_limit

        self._validate_seethrough()

        super().__init__()

    def task_generator(self):
        for i in range(len(self.processor_spec)):
            cs_task = helpers.PartialSpecification(
                SeethroughCompareTask,
                processor_spec=self.processor_spec[i],
                tgt_z_offset=self.tgt_z_offset,
                src_stack=self.src_stack,
                pad=self.pad,
                crop=self.crop,
                tgt_stack=self.tgt_stack,
                seethrough_limit=self.seethrough_limit[i],
                pixel_offset_layer=self.pixel_offset_layer,
            )

            chunked_job = ChunkedJob(
                task_class=cs_task,
                dst_layer=self.dst_layer,
                chunk_xy=self.chunk_xy,
                chunk_z=1,
                mip=self.mip,
                bcube=self.bcube,
                suffix=self.suffix,
            )

            yield from chunked_job.task_generator

            # Each seethrough processor writes to the same mask layer, so we
            # wait for each processor to finish to avoid race conditions.
            if i < len(self.processor_spec) - 1:
                yield scheduling.wait_until_done

    def _validate_seethrough(self):
        num_ps = len(self.processor_spec)
        num_sl = len(self.seethrough_limit)
        if num_ps != num_sl:
            raise ValueError(
                f"{num_ps} processors and {num_sl} seethrough limits specified to a SeethroughCompareJob. These must be equal."
            )
        for sl in self.seethrough_limit:
            if type(sl) != int:
                raise ValueError(f"Specified seethrough limit {sl} is not an integer")
            if sl < 0:
                raise ValueError(
                    f"Seethrough limits to SeethroughCompareJobs must be non-negative."
                )


class SeethroughCompareTask(scheduling.Task):
    def __init__(
        self,
        processor_spec,
        src_stack,
        tgt_stack,
        dst_layer,
        mip,
        pad,
        crop,
        tgt_z_offset,
        bcube,
        seethrough_limit,
        pixel_offset_layer,
    ):
        super().__init__()
        self.processor_spec = processor_spec
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.crop = crop
        self.tgt_z_offset = tgt_z_offset
        self.bcube = bcube
        self.seethrough_limit = seethrough_limit
        self.pixel_offset_layer = pixel_offset_layer

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.mip)
        tgt_bcube = src_bcube.translate(z_offset=self.tgt_z_offset)

        processor = procspec.parse_proc(spec_str=self.processor_spec)

        _, tgt_data_dict = self.tgt_stack.read_data_dict(
            tgt_bcube, mip=self.mip, stack_name="tgt"
        )

        _, src_data_dict = self.src_stack.read_data_dict(
            src_bcube, mip=self.mip, stack_name="src"
        )

        processor_input = {**src_data_dict, **tgt_data_dict}

        result = processor(processor_input, output_key="result")

        tgt_pixel_data = self.pixel_offset_layer.read(
            bcube=self.bcube.translate(z_offset=self.tgt_z_offset), mip=self.mip
        )
        written_pixel_data = self.pixel_offset_layer.read(
            bcube=self.bcube, mip=self.mip
        )
        written_mask_data = self.dst_layer.read(bcube=self.bcube, mip=self.mip)
        result = result.to(device=written_mask_data.device)
        cropped_result = helpers.crop(result, self.crop)
        if self.seethrough_limit > 0:
            seethrough_mask = (cropped_result > 0) & (
                tgt_pixel_data < self.seethrough_limit
            )
        else:
            seethrough_mask = cropped_result > 0
        written_mask_data[seethrough_mask] = True
        written_pixel_data[seethrough_mask] = (
            torch.minimum(
                tgt_pixel_data[seethrough_mask],
                torch.ones_like(tgt_pixel_data[seethrough_mask]) * 254,
            )
            + 1
        )
        self.dst_layer.write(written_mask_data, bcube=self.bcube, mip=self.mip)
        self.pixel_offset_layer.write(
            written_pixel_data, bcube=self.bcube, mip=self.mip
        )


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
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)
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
        f=RenderJob, pad=pad, chunk_xy=chunk_xy, chunk_z=1, render_masks=False,
    )
    seethrough_method = helpers.PartialSpecification(
        f=SeethroughCompareJob,
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
