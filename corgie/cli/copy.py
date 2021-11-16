import click

from corgie import scheduling, helpers, stack
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

class CopyLayerJob(scheduling.Job):
    def __init__(
        self,
        src_layer,
        dst_layer,
        mip,
        bcube,
        chunk_xy,
        chunk_z,
    ):
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
            mip=self.mip,
            return_generator=True,
        )

        tasks = (
            CopyLayerTask(
                self.src_layer,
                self.dst_layer,
                mip=self.mip,
                bcube=input_chunk,
            )
            for input_chunk in chunks
        )
        corgie_logger.info(
            f"Yielding copy layer tasks for bcube: {self.bcube}, MIP: {self.mip}"
        )

        yield tasks


class CopyLayerTask(scheduling.Task):
    def __init__(self, src_layer, dst_layer, mip, bcube):
        super().__init__(self)
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube

    def execute(self):
        src = self.src_layer.read(bcube=self.bcube, mip=self.mip)
        self.dst_layer.write(src, bcube=self.bcube, mip=self.mip)


class CopyJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_stack,
        mip,
        copy_masks,
        blackout_masks,
        bcube,
        chunk_xy,
        chunk_z,
    ):
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.copy_masks = copy_masks
        self.blackout_masks = blackout_masks

        super().__init__()

    def task_generator(self):
        chunks = self.dst_stack.get_layers()[0].break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
            mip=self.mip,
            return_generator=True,
        )

        tasks = (
            CopyTask(
                self.src_stack,
                self.dst_stack,
                blackout_masks=self.blackout_masks,
                copy_masks=self.copy_masks,
                mip=self.mip,
                bcube=input_chunk,
            )
            for input_chunk in chunks
        )
        corgie_logger.info(
            f"Yielding copy tasks for bcube: {self.bcube}, MIP: {self.mip}"
        )

        yield tasks


class CopyTask(scheduling.Task):
    def __init__(self, src_stack, dst_stack, copy_masks, blackout_masks, mip, bcube):
        super().__init__(self)
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.copy_masks = copy_masks
        self.blackout_masks = blackout_masks
        self.mip = mip
        self.bcube = bcube

    def execute(self):
        src_translation, src_data_dict = self.src_stack.read_data_dict(
            self.bcube, mip=self.mip, translation_adjuster=None, stack_name="src"
        )

        if self.blackout_masks:
            mask_layers = self.dst_stack.get_layers_of_type(["mask"])
            mask = helpers.read_mask_list(mask_layers, self.bcube, self.mip)
        else:
            mask = None
        if self.copy_masks:
            write_layers = self.dst_stack.get_layers_of_type(["img", "mask"])
        else:
            write_layers = self.dst_stack.get_layers_of_type("img")

        for l in write_layers:
            src = src_data_dict[f"src_{l.name}"]
            if mask is not None:
                src[mask] = 0
            l.write(src, bcube=self.bcube, mip=self.mip)

        # copy fields
        write_layers = self.dst_stack.get_layers_of_type("field")
        for l in write_layers:
            src = src_data_dict[f"src_{l.name}"]
            l.write(src, bcube=self.bcube, mip=self.mip)


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
    "--dst_folder",
    nargs=1,
    type=str,
    required=False,
    help="Destination folder for the copied stack",
)
@corgie_optgroup("Copy Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--mip", nargs=1, type=int, required=True)
@corgie_option("--blackout_masks/--no_blackout_masks", default=False)
@corgie_option("--copy_masks/--no_copy_masks", default=True)
@corgie_option("--force_chunk_xy", nargs=1, type=int,
   help="Will force the chunking of the underlying cloudvolume"
)
@corgie_option("--force_chunk_z", nargs=1, type=int,
   help="Will force the chunking of the underlying cloudvolume"
)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_option("--suffix", nargs=1, type=str, default=None)
@click.pass_context
def copy(
    ctx,
    src_layer_spec,
    dst_folder,
    copy_masks,
    blackout_masks,
    chunk_xy,
    chunk_z,
    start_coord,
    end_coord,
    coord_mip,
    mip,
    suffix,
    force_chunk_xy,
    force_chunk_z,
):

    scheduler = ctx.obj["scheduler"]
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    if not force_chunk_xy:
        force_chunk_xy = chunk_xy

    if not force_chunk_z:
        force_chunk_z = chunk_z

    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        force_chunk_z=force_chunk_z,
        overwrite=True,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    copy_job = CopyJob(
        src_stack=src_stack,
        dst_stack=dst_stack,
        mip=mip,
        bcube=bcube,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        copy_masks=copy_masks,
        blackout_masks=blackout_masks,
    )
    # create scheduler and execute the job
    scheduler.register_job(copy_job, job_name="Copy {}".format(bcube))
    scheduler.execute_until_completion()

    result_report = (
        f"Copied layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"to {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)
