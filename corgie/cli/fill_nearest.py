import click
import six
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


class FillNearestJob(scheduling.Job):
    def __init__(
        self, src_stack, dst_stack, bcube, mip, radius, chunk_xy,
    ):
        """Write image, filling in masked regions with nearest non-masked image

        This is useful when a model requires a filled in image.
        """
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.bcube = bcube
        self.mip = mip
        self.radius = radius
        self.chunk_xy = chunk_xy
        super().__init__()

    def task_generator(self):
        chunks = self.dst_stack.get_layers()[0].break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=1,
            mip=self.mip,
            return_generator=True,
        )

        tasks = (
            FillNearestTask(
                self.src_stack,
                self.dst_stack,
                mip=self.mip,
                bcube=chunk,
                radius=self.radius,
            )
            for chunk in chunks
        )
        corgie_logger.info(
            f"Yielding fill nearest tasks for bcube: {self.bcube}, MIP: {self.mip}"
        )

        yield tasks


class FillNearestTask(scheduling.Task):
    def __init__(
        self, src_stack, dst_stack, mip, bcube, radius=2,
    ):
        super().__init__(self)
        self.src_stack = src_stack
        self.dst_stack = dst_stack
        self.mip = mip
        self.bcube = bcube
        self.radius = radius
        self.count_threshold = 10

    def get_masks(self, data_dict, bcube):
        agg_mask = helpers.zeros(
            (bcube.x_size(self.mip), bcube.y_size(self.mip)), dtype=bool
        )
        mask_layers = self.dst_stack.get_layers_of_type(["mask"])
        mask_layer_names = [l.name for l in mask_layers]
        for n, d in six.iteritems(data_dict):
            if n in mask_layer_names:
                if agg_mask is None:
                    agg_mask = d
                else:
                    agg_mask = (agg_mask + d) > 0
        return agg_mask

    def execute(self):
        radii = [i for r in range(1, self.radius + 1) for i in [r, -r]]
        z = self.bcube.z_range()[0]
        _, src_data_dict = self.src_stack.read_data_dict(
            bcube=self.bcube, mip=self.mip, add_prefix=False, translation_adjuster=None,
        )
        layer = self.dst_stack.get_layers_of_type("img")[0]
        agg_img = src_data_dict[f"{layer.name}"]
        agg_mask = self.get_masks(data_dict=src_data_dict, bcube=self.bcube)
        mask_count = agg_mask.sum()
        k = 0
        while (mask_count > self.count_threshold) and (k < len(radii)):
            corgie_logger.debug(f"mask_count={mask_count}")
            corgie_logger.debug(f"radius={radii[k]}")
            bcube = self.bcube.reset_coords(
                zs=z + radii[k], ze=z + radii[k] + 1, in_place=False
            )
            _, src_data_dict = self.src_stack.read_data_dict(
                bcube=bcube, mip=self.mip, add_prefix=False, translation_adjuster=None,
            )
            img = src_data_dict[f"{layer.name}"]
            agg_img[agg_mask] = img[agg_mask]
            mask = self.get_masks(data_dict=src_data_dict, bcube=bcube)
            agg_mask = (agg_mask == 1) * (mask == 1)
            mask_count = agg_mask.sum()
            k += 1
        layer.write(agg_img, bcube=self.bcube, mip=self.mip)


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
@corgie_optgroup("Fill Nearest Method Specification")
@corgie_option("--mip", nargs=1, type=int, required=True)
@corgie_option("--radius", nargs=1, type=int, default=2)
@corgie_option("--chunk_xy", nargs=1, type=int, default=1024)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def fill_nearest(
    ctx,
    src_layer_spec,
    dst_folder,
    chunk_xy,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
    mip,
    radius,
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
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    fill_nearest_job = FillNearestJob(
        src_stack=src_stack,
        dst_stack=dst_stack,
        bcube=bcube,
        radius=radius,
        mip=mip,
        chunk_xy=chunk_xy,
    )
    # create scheduler and execute the job
    scheduler.register_job(
        fill_nearest_job, job_name="Fill Nearest Block {}".format(bcube)
    )

    scheduler.execute_until_completion()
    result_report = (
        f"Rendered layers {[str(l) for l in src_stack.get_layers_of_type('img')]}. "
        f"Results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
    )
    corgie_logger.info(result_report)

