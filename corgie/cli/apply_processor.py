import click
import procspec
import json

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
from corgie.cli.downsample import DownsampleJob


class ApplyProcessorJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_layer,
        chunk_xy,
        chunk_z,
        processor_spec,
        processor_mip,
        processor_mip_out,
        pad,
        crop,
        bcube,
        blend_xy=0,
    ):

        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.blend_xy = blend_xy
        self.pad = pad
        self.crop = crop
        self.bcube = bcube
        self.processor_spec = processor_spec
        self.processor_mip = processor_mip
        self.processor_mip_out = processor_mip_out


        if isinstance(self.processor_spec, str):
            self.processor_spec = [self.processor_spec]
        if isinstance(self.processor_mip, int):
            self.processor_mip = [self.processor_mip]

        if self.processor_mip_out is None or len(self.processor_mip_out) == 0:
            self.processor_mip_out = self.processor_mip

        if len(self.processor_mip) != len(self.processor_spec):
            raise exceptions.CorgieException(
                "The number of processors doesn't "
                "match the number of specified processor MIPs"
            )

        super().__init__()

    def task_generator(self):
        for i in range(len(self.processor_spec)):
            this_proc = self.processor_spec[i]
            this_proc_mip = self.processor_mip[i]
            this_proc_mip_out = self.processor_mip_out[i]
            is_last_proc = i == len(self.processor_spec) - 1

            this_task = helpers.PartialSpecification(
                ApplyProcessorTask,
                src_stack=self.src_stack,
                processor_spec=this_proc,
                pad=self.pad,
                crop=self.crop,
                mip_in=this_proc_mip,
            )

            chunked_job = ChunkedJob(
                task_class=this_task,
                dst_layer=self.dst_layer,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                blend_xy=self.blend_xy,
                mip=this_proc_mip_out,
                bcube=self.bcube,
            )

            yield from chunked_job.task_generator

            if not is_last_proc:
                yield scheduling.wait_until_done
                next_proc_mip = self.processor_mip[i + 1]
                if this_proc_mip > next_proc_mip:
                    downsample_job = DownsampleJob(
                        src_layer=self.dst_layer,
                        chunk_xy=self.chunk_xy,
                        chunk_z=self.chunk_z,
                        mip_start=this_proc_mip,
                        mip_end=next_proc_mip,
                        bcube=self.bcube,
                    )
                    yield from downsample_job.task_generator
                    yield scheduling.wait_until_done

        if self.processor_mip[0] > self.processor_mip[-1]:
            # good manners
            # prepare the ground for the next you
            downsample_job = DownsampleJob(
                src_layer=self.dst_layer,
                chunk_xy=self.chunk_xy,
                chunk_z=self.chunk_z,
                mip_start=self.processor_mip[-1],
                mip_end=self.processor_mip[0],
                bcube=self.bcube,
            )


class ApplyProcessorTask(scheduling.Task):
    def __init__(self, processor_spec, src_stack, dst_layer, mip, pad, crop, bcube, mip_in):
        super().__init__()
        self.processor_spec = processor_spec
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.crop = crop
        self.bcube = bcube
        self.mip_in = mip_in

    def execute(self):
        src_bcube = self.bcube.uncrop(self.pad, self.mip_in)

        processor = procspec.parse_proc(spec_str=self.processor_spec)

        src_translation, src_data_dict = self.src_stack.read_data_dict(
            src_bcube, mip=self.mip_in, stack_name="src"
        )

        processor_input = {**src_data_dict}
        result = processor(processor_input, output_key="output")
        cropped_result = helpers.crop(result, self.crop // (2**(self.mip - self.mip_in)))
        self.dst_layer.write(cropped_result, bcube=self.bcube, mip=self.mip)


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
#
@corgie_option(
    "--dst_layer_spec",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the destination layer. Must be a field type.",
)
@corgie_option(
    "--spec_path",
    nargs=1,
    type=str,
    required=False,
    help="JSON spec relating src stacks, src z to dst z",
)
@corgie_option("--reference_key", nargs=1, type=str, default="img")
@corgie_optgroup("Apply Processor Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option(
    "--pad", nargs=1, type=int, default=512,
)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_spec", nargs=1, type=str, multiple=True, required=True)
@corgie_option("--processor_mip", nargs=1, type=int, multiple=True, required=True)
@corgie_option("--processor_mip_out", nargs=1, type=int, multiple=True, required=False)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def apply_processor(
    ctx,
    src_layer_spec,
    dst_layer_spec,
    spec_path,
    processor_spec,
    pad,
    crop,
    chunk_xy,
    start_coord,
    force_chunk_xy,
    processor_mip,
    processor_mip_out,
    end_coord,
    coord_mip,
    blend_xy,
    chunk_z,
    reference_key,
):
    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    reference_layer = None
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]

    if force_chunk_xy is None:
        force_chunk_xy = chunk_xy

    dst_layer = create_layer_from_spec(
        dst_layer_spec,
        allowed_types=["img", "mask", "section_value", "field"],
        default_type="img",
        readonly=False,
        caller_name="dst_layer",
        force_chunk_xy=force_chunk_xy,
        reference=reference_layer,
        overwrite=True,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    if spec_path:
        with open(spec_path, "r") as f:
            spec = json.load(f)

        for z in spec:
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
                processor_mip_out=processor_mip_out,
            )

            # create scheduler and execute the job
            scheduler.register_job(
                apply_processor_job, job_name="Apply Processor {}".format(job_bcube)
            )
    else:
        apply_processor_job = ApplyProcessorJob(
            src_stack=src_stack,
            dst_layer=dst_layer,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
            blend_xy=blend_xy,
            processor_spec=processor_spec,
            pad=pad,
            crop=crop,
            bcube=bcube,
            processor_mip=processor_mip,
            processor_mip_out=processor_mip_out,
        )

        # create scheduler and execute the job
        scheduler.register_job(
            apply_processor_job, job_name="Apply Processor {}".format(bcube)
        )
    scheduler.execute_until_completion()
