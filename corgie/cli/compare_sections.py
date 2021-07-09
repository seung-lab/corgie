import click
# import itertools
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


class CompareSectionsJob(scheduling.Job):
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
        seethrough_limit,
        tgt_stack=None,
        suffix="",
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
        self.seethrough_limit = seethrough_limit
        
        self._validate_seethrough()
        
        super().__init__()

    def task_generator(self):
        # chunked_jobs = []
        # pixel_offset_layers = []
        for i in range(len(self.processor_spec)):
            seethrough_limit = self.seethrough_limit[i]
            pixel_offset_layer = None
            pixel_offset_layer_name = None
            # seethrough_limit = 0 means no limit
            if seethrough_limit > 0:
                # Layer that keeps track of how many sections ago each pixel
                # of the tgt_stack comes from in the src_stack. Used to enforce
                # a seethrough limit.
                pixel_offset_layer_name = f"seethrough_{i}_pixel_offset{self.suffix}"
                # pixel_offset_layers.append(pixel_offset_layer_name)
                pixel_offset_layer = self.tgt_stack.create_sublayer(
                    pixel_offset_layer_name, layer_type="img", overwrite=True
                )

            cs_task = helpers.PartialSpecification(
                CompareSectionsTask,
                processor_spec=self.processor_spec[i],
                tgt_z_offset=self.tgt_z_offset,
                src_stack=self.src_stack,
                pad=self.pad,
                crop=self.crop,
                tgt_stack=self.tgt_stack,
                seethrough_limit=self.seethrough_limit[i],
                pixel_offset_layer=pixel_offset_layer
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

            if pixel_offset_layer_name is not None:
                self.tgt_stack.remove_layer(pixel_offset_layer_name)

            # chunked_jobs.append(chunked_job.task_generator)

        # yield from itertools.chain.from_iterable(chunked_jobs)

    def _validate_seethrough(self):
        num_ps = len(self.processor_spec)
        num_sl = len(self.seethrough_limit)
        if num_ps != num_sl:
            raise ValueError(f'{num_ps} processors and {num_sl} seethrough limits specified to a CompareSectionsJob. These must be equal.')
        for sl in self.seethrough_limit:
            if sl < 0:
                raise ValueError(f'Seethrough limits to CompareSectionsJobs must be non-negative.')

class CompareSectionsTask(scheduling.Task):
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

        # TODO:
        # 1. Read pixel_offset_layer from tgt_data_dict (might need to pass in name instead of layer?)
        # 2. Set seethrough mask layer (self.dst_layer) to be the union of itself
        # with (the intersection of cropped_result and area allowed by seethrough_limit)
        # 3. Write appropriate values to the pixel_offset_layer

        cropped_result = helpers.crop(result, self.crop)
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
@corgie_option(
    "--tgt_layer_spec",
    "-t",
    nargs=1,
    type=str,
    required=False,
    multiple=True,
    help="Target layer spec. Use multiple times to include all masks, fields, images. "
    "DEFAULT: Same as source layers",
)
@corgie_option(
    "--dst_layer_spec",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the destination layer. Must be an image or mask type.",
)
@corgie_option("--reference_key", nargs=1, type=str, default="img")
@corgie_optgroup("Compute Field Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option(
    "--pad",
    nargs=1,
    type=int,
    default=512,
)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_spec", nargs=1, type=str, multiple=False, required=True)
@corgie_option("--mip", nargs=1, type=int, multiple=False, required=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_option("--tgt_z_offset", nargs=1, type=str, default=-1)
@click.option("--suffix", nargs=1, type=str, default=None)
@click.pass_context
def compare_sections(
    ctx,
    src_layer_spec,
    tgt_layer_spec,
    dst_layer_spec,
    suffix,
    processor_spec,
    pad,
    crop,
    chunk_xy,
    start_coord,
    mip,
    end_coord,
    coord_mip,
    tgt_z_offset,
    reference_key,
):
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    tgt_stack = create_stack_from_spec(
        tgt_layer_spec, name="tgt", readonly=True, reference=src_stack
    )

    reference_layer = None
    if reference_key in src_stack.layers:
        reference_layer = src_stack.layers[reference_key]

    dst_layer = create_layer_from_spec(
        dst_layer_spec,
        allowed_types=["img", "mask"],
        default_type="field",
        readonly=False,
        caller_name="dst_layer",
        reference=reference_layer,
        overwrite=True,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if crop is None:
        crop = pad

    compare_job = CompareSectionsJob(
        src_stack=src_stack,
        tgt_stack=tgt_stack,
        dst_layer=dst_layer,
        chunk_xy=chunk_xy,
        processor_spec=processor_spec,
        pad=pad,
        crop=crop,
        bcube=bcube,
        tgt_z_offset=tgt_z_offset,
        suffix=suffix,
        mip=mip,
    )

    # create scheduler and execute the job
    scheduler.register_job(
        compare_job,
        job_name="Compare Job {}, tgt z offset {}".format(bcube, tgt_z_offset),
    )
    scheduler.execute_until_completion()
