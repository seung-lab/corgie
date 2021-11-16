import torch
import click
import json
from corgie.log import logger as corgie_logger
from corgie import scheduling, helpers, stack
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import (
    LAYER_HELP_STR,
    create_layer_from_spec,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


class CombineMasksJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        exp,
        dst_layer,
        mip,
        bcube,
        pad,
        chunk_xy,
        chunk_z=1,
        inner_threshold=0.01,
        outer_threshold=0.99,
    ):
        """
        Combine masks according to boolean expression

        Args:
            layers ({key: Layer})
            exp (list of lists)
            output_field (Layer)
            mip (int)
            bcube (BoundingCube)
            pad (int)
            inner_threshold (float)
            outer_threshold (float)
            chunk_xy (int)
            chunk_z (int)
        """
        super().__init__()
        self.src_stack = src_stack
        self.exp = exp
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
            bcube=self.bcube, chunk_xy=self.chunk_xy, chunk_z=self.chunk_z, mip=self.mip
        )

        tasks = []
        for chunk in chunks:
            task = CombineMasksTask(
                src_stack=self.src_stack,
                exp=self.exp,
                dst_layer=self.dst_layer,
                mip=self.mip,
                bcube=chunk,
                pad=self.pad,
                inner_threshold=self.inner_threshold,
                outer_threshold=self.outer_threshold,
            )
            tasks.append(task)

        corgie_logger.debug(
            "Yielding CombineMasksTask for bcube: {}, MIP: {}".format(
                self.bcube, self.mip
            )
        )
        yield tasks


class CombineMasksTask(scheduling.Task):
    def __init__(
        self,
        src_stack,
        exp,
        dst_layer,
        mip,
        bcube,
        pad,
        inner_threshold=0.01,
        outer_threshold=0.99,
    ):
        """Evaluate a Boolean function of masks that may be from different layers and
        with different z_offsets. The Boolean function is expressed in normative form
        (either conjunctive or disjunctive), e.g.
        ..math::
            (A | B) & (C | D)

        Args:
            layers ({key: Layer}): dict with values of mask layers indexed with key that's
                referenced in boolean expression
            exp (list of lists): nested lists of depth two. Inner lists must be lists of
                tuples with sign, layer key, and offset. Inner lists will be evaluated as
                disjunctions, while the outer list will evaluate as a conjunction.
                For example:
                ``[[(+1, "a", 0), (-1, "b", -1), (+1, "c", 2)], [(+1, "a", 1), (-1, "c", 0)]]```
                evaluates as
                `(a[0] | not b[-1] | c[2]) & (a[1] | not c[0])`.
            output_field (Layer)
            mip (int)
            bcube (BoundingCube): xy location of where to sample all fields;
                z locations of where to write output
            pad (int)
            inner_threshold (float): low for OR, high for AND
            outer_threshold (float): low for OR, high for AND
        """
        super().__init__()
        self.src_stack = src_stack
        self.exp = exp
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold

    def execute(self):
        corgie_logger.debug(f"CombineMaskTask, {self.bcube}")
        pbcube = self.bcube.uncrop(self.pad, self.mip)
        for z in range(*self.bcube.z_range()):
            w, h = pbcube.x_size(self.mip), pbcube.y_size(self.mip)
            outer = torch.zeros((1, 1, w, h), dtype=torch.uint8)
            for X in self.exp:
                inner = torch.zeros((1, 1, w, h), dtype=torch.uint8)
                for s, k, o in X:
                    bcube = pbcube.reset_coords(zs=z + o, ze=z + o + 1, in_place=False)
                    layer = self.src_stack.layers[k]
                    inner += s * layer.read(bcube=bcube, mip=self.mip)
                outer += inner > self.inner_threshold
            m = (outer > self.outer_threshold).to(torch.uint8)
            m = helpers.crop(m, self.pad)
            bcube = self.bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            self.dst_layer.write(m, bcube=bcube, mip=self.mip)
        for layer in self.src_stack.layers.values():
            layer.flush(self.mip)


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
    "--dst_layer_spec",
    nargs=1,
    type=str,
    required=True,
    help="Specification for the destination layer. Must be an image or mask type.",
)
@corgie_optgroup("Combine Mask Method Specification")
@corgie_option(
    "--exp",
    nargs=1,
    type=str,
    required=True,
    help="Boolean expression to be evaluated. "
    + "Must be a JSON-parseable string describing a list of lists."
    + "Inner list represents literals as three-element tuples: "
    + "1. int for sign of the literal, e.g. +/-1 "
    + "2. str of layer name "
    + "3. int of z offset "
    + "Expressions are evluated as linear threshold neurons.",
)
@corgie_option(
    "--inner_threshold",
    nargs=1,
    type=float,
    default=0.01,
    help="Threshold used for linear threshold neuron combining literals in inner lists of EXP. Set low to be disjunctive, high to be conjunctive.",
)
@corgie_option(
    "--outer_threshold",
    nargs=1,
    type=float,
    default=0.99,
    help="Threshold used for linear threshold neuron combining prositions within outer list of EXP. Set low to be disjunctive, high to be conjunctive.",
)
@corgie_option("--mip", nargs=1, type=int, required=True)
@corgie_option("--pad", nargs=1, type=int, default=0)
@corgie_option("--force_chunk_xy", nargs=1, type=int)
@corgie_option("--force_chunk_z", nargs=1, type=int, default=1)
@corgie_option("--chunk_xy", nargs=1, type=int, default=1024)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def combine_masks(
    ctx,
    src_layer_spec,
    dst_layer_spec,
    exp,
    inner_threshold,
    outer_threshold,
    chunk_xy,
    chunk_z,
    force_chunk_xy,
    force_chunk_z,
    start_coord,
    end_coord,
    coord_mip,
    mip,
    pad,
):
    scheduler = ctx.obj["scheduler"]

    if not force_chunk_xy:
        force_chunk_xy = chunk_xy

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)
    reference_layer = src_stack.reference_layer

    dst_layer = create_layer_from_spec(
        dst_layer_spec,
        allowed_types=["mask"],
        default_type="mask",
        readonly=False,
        caller_name="dst_layer",
        reference=reference_layer,
        force_chunk_xy=force_chunk_xy,
        force_chunk_z=force_chunk_z,
        overwrite=True,
    )
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    combine_masks_job = CombineMasksJob(
        src_stack=src_stack,
        exp=json.loads(exp),
        dst_layer=dst_layer,
        mip=mip,
        bcube=bcube,
        pad=pad,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        inner_threshold=inner_threshold,
        outer_threshold=outer_threshold,
    )
    # create scheduler and execute the job
    scheduler.register_job(combine_masks_job, job_name="Combine Masks {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = f"Results in {str(dst_layer)}"
    corgie_logger.info(result_report)
