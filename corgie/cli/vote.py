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


def compute_softmin_temp(dist: float, weight: float, size: int):
    """Compute softmin temp given binary assumptions

    Assumes that voting subsets are either correct or incorrect.

    Args:
        dist: distance between the average differences of
            a correct and incorrect subset.
        weight: desired weight to be assigned for correct/incorrect
            distance.
        size: the number of subsets involved in voting. If the 
            number of vectors involved in voting is n, and k is the smallest
            number that represents a majority of n, then size should be 
            (n choose k). 

    Returns:
        float for softmin temperature that will achieve this weight
    """
    assert (weight > 0) and (weight < 1)
    assert size > 0
    assert dist >= 0
    return -dist / (log(1 - weight) - log(weight * size) + 1e-5)


class VoteOverFieldsJob(scheduling.Job):
    def __init__(
        self,
        input_fields,
        output_field,
        chunk_xy,
        bcube,
        mip,
        softmin_temp=None,
        blur_sigma=1.0,
    ):
        self.input_fields = input_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.mip = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        super().__init__()

    def task_generator(self):
        tmp_key = list(self.input_fields.keys())[0]
        tmp_layer = self.input_fields[tmp_key]
        chunks = tmp_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = [
            VoteOverFieldsTask(
                input_fields=self.input_fields,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                softmin_temp=self.softmin_temp,
                blur_sigma=self.blur_sigma,
            )
            for chunk in chunks
        ]

        corgie_logger.debug(
            "Yielding VoteTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class VoteOverFieldsTask(scheduling.Task):
    def __init__(
        self, input_fields, output_field, mip, bcube, softmin_temp, blur_sigma=1.0
    ):
        """Find median vector for single location over set of fields

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_fields (Stack): collection of fields
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
        """
        super().__init__()
        self.input_fields = input_fields
        self.output_field = output_field
        self.mip = mip
        self.bcube = bcube
        if softmin_temp is None:
            softmin_temp = compute_softmin_temp(
                dist=1, weight=0.99, size=len(input_fields)
            )
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma

    def execute(self):
        bcube = self.bcube
        mip = self.mip
        fields = [f.read(bcube, mip=mip) for f in self.input_fields.values()]
        fields = torch.cat([f for f in fields]).field()
        voted_field = fields.vote(
            softmin_temp=self.softmin_temp, blur_sigma=self.blur_sigma
        )
        self.output_field.write(data_tens=voted_field, bcube=bcube, mip=self.mip)


class VoteOverZJob(scheduling.Job):
    def __init__(
        self,
        input_field,
        output_field,
        chunk_xy,
        bcube,
        z_list,
        mip,
        softmin_temp=None,
        blur_sigma=1.0,
        weights_layer=None,
    ):
        self.input_field = input_field
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.z_list = z_list
        self.mip = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        self.weights_layer = weights_layer
        super().__init__()

    def task_generator(self):
        chunks = self.output_field.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=1, mip=self.mip
        )

        tasks = [
            VoteOverZTask(
                input_field=self.input_field,
                output_field=self.output_field,
                mip=self.mip,
                bcube=chunk,
                z_list=self.z_list,
                softmin_temp=self.softmin_temp,
                blur_sigma=self.blur_sigma,
                weights_layer=self.weights_layer,
            )
            for chunk in chunks
        ]

        corgie_logger.debug(
            "Yielding VoteOverZTask for bcube: {}, MIP: {}".format(self.bcube, self.mip)
        )
        yield tasks


class VoteOverZTask(scheduling.Task):
    def __init__(
        self,
        input_field,
        output_field,
        mip,
        bcube,
        z_list,
        softmin_temp,
        blur_sigma=1.0,
        weights_layer=None,
    ):
        """Find median vector for set of locations in a single field

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_field (Layer)
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            z_list (range, [int])
            weight_layer (Layer): if set, where intermediary weights will be written
        """
        super().__init__()
        self.input_field = input_field
        self.output_field = output_field
        self.mip = mip
        self.bcube = bcube
        self.z_list = z_list
        if softmin_temp is None:
            softmin_temp = compute_softmin_temp(dist=1, weight=0.99, size=len(z_list))
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        self.weights_layer = weights_layer

    def execute(self):
        fields = []
        for z in self.z_list:
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            fields.append(self.input_field.read(bcube, mip=self.mip))
        fields = torch.cat([f for f in fields]).field()
        voted_field = fields.vote(
            softmin_temp=self.softmin_temp, blur_sigma=self.blur_sigma
        )
        self.output_field.write(data_tens=voted_field, bcube=self.bcube, mip=self.mip)


class VoteJob(scheduling.Job):
    def __init__(
        self,
        input_fields,
        output_field,
        chunk_xy,
        bcube,
        z_offsets,
        mip,
        softmin_temp=None,
        blur_sigma=1.0,
        weights_layer=None,
    ):
        self.input_fields = input_fields
        self.output_field = output_field
        self.chunk_xy = chunk_xy
        self.bcube = bcube
        self.z_offsets = z_offsets
        self.mip = mip
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
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
                softmin_temp=self.softmin_temp,
                blur_sigma=self.blur_sigma,
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
        softmin_temp,
        blur_sigma=1.0,
        weights_layer=None,
    ):
        """Find median vector for set of locations in a set of fields

        Notes:
            Does not ignore identity fields.
            Padding & cropping is not necessary.

        Args:
            input_fields ([Layer]): match length of z_offsets
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            z_offsets (range, [int]): offsets from z where fields should be accessed
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
        if softmin_temp is None:
            softmin_temp = compute_softmin_temp(
                dist=1, weight=0.99, size=len(self.z_offsets)
            )
        self.softmin_temp = softmin_temp
        self.blur_sigma = blur_sigma
        self.weights_layer = weights_layer

    def execute(self):
        fields = []
        for z_offset, field in zip(self.z_offsets, self.input_fields):
            z = self.bcube.z_range()[0]
            bcube = self.bcube.reset_coords(
                zs=z + z_offset, ze=z + z_offset + 1, in_place=False
            )
            fields.append(field.read(bcube, mip=self.mip))
        fields = torch.cat([f for f in fields]).field()
        weights = fields.get_vote_weights(
            softmin_temp=self.softmin_temp, blur_sigma=self.blur_sigma
        )
        if self.weights_layer:
            self.weights_layer.write(data_tens=weights, bcube=self.bcube, mip=self.mip)
        voted_field = (fields * weights.unsqueeze(-3)).sum(dim=0, keepdim=True)
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
@corgie_option("--softmin_temp", nargs=1, type=float, default=None)
@corgie_option("--blur_sigma", nargs=1, type=float, default=1.0)
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
    softmin_temp,
    blur_sigma,
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
        softmin_temp=softmin_temp,
        blur_sigma=blur_sigma,
        weights_layer=vote_weights,
    )
    scheduler.register_job(
        vote_stitch_job, job_name=f"Vote {bcube}",
    )

    scheduler.execute_until_completion()
    corgie_logger.debug("Done!")

