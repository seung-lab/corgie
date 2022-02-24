import click
import procspec

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie import scheduling, helpers, stack
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)

class MultiSectionCompareJob(scheduling.Job):
    """Names of img layers in dst_stack should be z_offsets
    """
    def __init__(
        self,
        src_stack,
        dst_stack,
        chunk_xy,
        processor_spec,
        mip,
        dst_mip,
        pad,
        bcube,
        tgt_stack=None,
        suffix="",
    ):

        self.src_stack = src_stack
        if tgt_stack is None:
            tgt_stack = src_stack

        self.tgt_stack = tgt_stack
        self.dst_stack = dst_stack
        self.chunk_xy = chunk_xy
        self.pad = pad
        self.bcube = bcube

        self.suffix = suffix

        self.processor_spec = processor_spec
        self.mip = mip
        self.dst_mip = dst_mip
        super().__init__()

    def task_generator(self):
        chunks = self.dst_stack.get_layers()[0].break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=1,
            mip=self.dst_mip,
            return_generator=True,
        )

        tasks = (
            MultiSectionCompareTask(
                processor_spec=self.processor_spec,
                src_stack=self.src_stack,
                tgt_stack=self.tgt_stack,
                dst_stack=self.dst_stack,
                mip=self.mip,
                dst_mip=self.dst_mip,
                pad=self.pad,
                bcube=input_chunk,
            )
            for input_chunk in chunks
        )
        print(
            f"Yielding MultiSectionCompareTasks for bcube: {self.bcube}, MIP: {self.mip}"
        )

        yield tasks

class MultiSectionCompareTask(scheduling.Task):
    def __init__(
        self,
        processor_spec,
        src_stack,
        tgt_stack,
        dst_stack,
        mip,
        dst_mip,
        pad,
        bcube,
    ):
        super().__init__()
        self.processor_spec = processor_spec
        self.src_stack = src_stack
        self.tgt_stack = tgt_stack
        self.dst_stack = dst_stack
        self.mip = mip
        self.dst_mip = dst_mip
        self.pad = pad
        self.bcube = bcube

    def execute(self):
        tgt_bcube = self.bcube.uncrop(self.pad, self.mip)
        _, tgt_data_dict = self.tgt_stack.read_data_dict(
            tgt_bcube, mip=self.mip, stack_name="tgt"
        )

        dst_layers = self.dst_stack.get_layers_of_type("img")
        for dst_layer in dst_layers:
            z_offset = int(dst_layer.name)
            src_bcube = tgt_bcube.translate(z_offset=z_offset)
            processor = procspec.parse_proc(spec_str=self.processor_spec)

            _, src_data_dict = self.src_stack.read_data_dict(
                src_bcube, mip=self.mip, stack_name="src"
            )

            processor_input = {**src_data_dict, **tgt_data_dict}

            result = processor(processor_input, output_key="result")
            crop = self.pad // 2 ** (self.dst_mip - self.mip)
            cropped_result = helpers.crop(result, crop)
            dst_layer.write(cropped_result, bcube=self.bcube, mip=self.dst_mip)


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
@corgie_optgroup("Multi Section Compare Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--pad", nargs=1, type=int, default=256)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--z_offsets", multiple=True, type=int, default=[-1])
@corgie_option("--processor_spec", nargs=1, type=str, required=True, multiple=False)
@corgie_option("--processor_mip", "-m", nargs=1, type=int, required=True, multiple=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def multi_section_compare(
    ctx,
    src_layer_spec,
    dst_folder,
    chunk_xy,
    pad,
    crop,
    force_chunk_xy,
    z_offsets,
    processor_spec,
    processor_mip,
    start_coord,
    end_coord,
    coord_mip,
):

    scheduler = ctx.obj["scheduler"]

    if crop is None:
        crop = pad

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    if force_chunk_xy is None:
        force_chunk_xy = chunk_xy

    dst_stack = stack.Stack(name="dst", folder=dst_folder) 
    dst_stack.reference_layer = src_stack.get_layers()[0]

    for z_offset in z_offsets:
        dst_stack.create_sublayer(
            name=z_offset,
            layer_type="img",
            dtype="float32",
            force_chunk_xy=force_chunk_xy,
            overwrite=True,
        )

    multi_section_compare_job = MultiSectionCompareJob(
        src_stack=src_stack,
        dst_stack=dst_stack,
        chunk_xy=chunk_xy,
        bcube=bcube,
        pad=pad,
        processor_spec=processor_spec,
        mip=processor_mip[0],
        dst_mip=processor_mip[0],
    )
    scheduler.register_job(
        multi_section_compare_job, job_name=f"MultiSectionCompare {bcube}",
    )

    scheduler.execute_until_completion()
    corgie_logger.debug("Done!")

