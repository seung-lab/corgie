import click
import torch
from math import log
from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie import scheduling, stack
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


class VoteJob(scheduling.Job):
    def __init__(
        self,
        input_fields,
        output_field,
        chunk_xy,
        bcube,
        z_offsets,
        mip,
        consensus_threshold=3.,
        blur_sigma=15.0,
        kernel_size=32,
        weights_layer=None,
    ):
        self.input_fields = input_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.z_offsets = z_offsets
        self.mip = mip
        self.consensus_threshold = consensus_threshold
        self.blur_sigma = blur_sigma
        self.kernel_size = kernel_size
        self.weights_layer = weights_layer
        super().__init__()

    def task_generator(self):
        chunks = self.output_field.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = [
            VoteTask(
                input_fields=self.input_fields,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                z_offsets=self.z_offsets,
                consensus_threshold=self.consensus_threshold,
                blur_sigma=self.blur_sigma,
                kernel_size=self.kernel_size,
                weights_layer=self.weights_layer,
            )
            for chunk in chunks
        ]

        corgie_logger.debug(
            "Yielding VoteTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class VoteTask(scheduling.Task):
    def __init__(
        self,
        input_fields,
        output_field,
        mip,
        bcube,
        z_offsets,
        consensus_threshold=3.,
        blur_sigma=15,
        kernel_size=32,
        weights_layer=None,
    ):
        """Find median-like field with highest priority from a set of fields

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_fields ([Layer]): match length of z_offsets
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            z_offsets (range, [int]): offsets from z where fields should be accessed
            consensus_threshold (float): distance from winning subset that will still be 
                considered consensus
            blur_sigma (float): std of Gaussian that will blur weights of the fields
            weight_layer (Layer): if set, where intermediary weights will be written
        """
        super().__init__()
        self.input_fields = input_fields
        self.output_field = output_field
        self.mip = mip
        self.bcube = bcube
        self.z_offsets = z_offsets
        # match length of input_fields & z_offsets
        if len(self.input_fields) < len(self.z_offsets):
            self.input_fields = [self.input_fields[0]] * len(self.z_offsets)
        elif len(self.z_offsets) < len(self.input_fields):
            self.z_offsets = [self.z_offsets[0]] * len(self.input_fields)
        self.consensus_threshold = consensus_threshold
        self.blur_sigma = blur_sigma
        self.kernel_size = kernel_size
        self.weights_layer = weights_layer

    def execute(self):
        fields = []
        priorities = []
        for priority, (z_offset, field) in enumerate(zip(self.z_offsets, self.input_fields)):
            z = self.bcube.z_range()[0]
            bcube = self.bcube.reset_coords(
                zs=z + z_offset, ze=z + z_offset + 1, in_place=False
            )
            fields.append(field.read(bcube, mip=self.mip))
            priorities.append(torch.full_like(fields[-1][:,0,...], fill_value=len(self.input_fields) -priority))
        fields = torch.cat([f for f in fields]).field()
        priorities = torch.cat(priorities)
        weights = fields.get_priority_vote_weights(priorities=priorities, consensus_threshold=self.consensus_threshold)
        if self.weights_layer:
            self.weights_layer.write(data_tens=weights, bcube=self.bcube, mip=self.mip)
        voted_field = fields.smoothed_combination(weights=weights, blur_sigma=self.blur_sigma, kernel_size=self.kernel_size)
        self.output_field.write(data_tens=voted_field, bcube=self.bcube, mip=self.mip)


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
@corgie_optgroup("Voting Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--z_offsets", multiple=True, type=int, default=[0])
@corgie_option("--consensus_threshold", nargs=1, type=float, default=3.)
@corgie_option("--blur_sigma", nargs=1, type=float, default=15.0)
@corgie_option("--kernel_size", nargs=1, type=int, default=32)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--mip", nargs=1, type=int, default=0)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def vote(
    ctx,
    src_layer_spec,
    dst_folder,
    chunk_xy,
    mip,
    z_offsets,
    force_chunk_xy,
    consensus_threshold,
    blur_sigma,
    kernel_size,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
):

    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_voted"
    else:
        suffix = f"_{suffix}"

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)
    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    if force_chunk_xy is None:
        force_chunk_xy = chunk_xy

    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["field", "float_tensor"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        overwrite=True,
    )

    vote_weights = dst_stack.create_sublayer(
        name="vote_weights",
        layer_type="float_tensor",
        overwrite=True,
        num_channels=max(len(src_stack), len(z_offsets)),
    )
    voted_field = dst_stack.create_sublayer(
        name="voted_field", layer_type="field", overwrite=True
    )

    vote_stitch_job = VoteJob(
        input_fields=src_stack.get_layers(),
        output_field=voted_field,
        chunk_xy=chunk_xy,
        bcube=bcube,
        z_offsets=z_offsets,
        mip=mip,
        consensus_threshold=consensus_threshold,
        blur_sigma=blur_sigma,
        kernel_size=kernel_size,
        weights_layer=vote_weights,
    )
    scheduler.register_job(
        vote_stitch_job, job_name=f"Vote {bcube}",
    )

    scheduler.execute_until_completion()
    corgie_logger.debug("Done!")

