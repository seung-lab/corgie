import os
import click

from corgie import scheduling
from corgie import helpers, stack
from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie.cli.compute_stats import ComputeStatsJob
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


class NormalizeJob(scheduling.Job):
    def __init__(
        self,
        src_layer,
        mask_layers,
        dst_layer,
        mean_layer,
        var_layer,
        stats_mip,
        mip,
        bcube,
        chunk_xy,
        chunk_z,
        mask_value,
    ):
        self.src_layer = src_layer
        self.mask_layers = mask_layers
        self.dst_layer = dst_layer
        self.var_layer = var_layer
        self.mean_layer = mean_layer
        self.stats_mip = stats_mip
        self.mip = mip
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mask_value = mask_value
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
            NormalizeTask(
                self.src_layer,
                self.mask_layers,
                self.dst_layer,
                self.mean_layer,
                self.var_layer,
                self.stats_mip,
                self.mip,
                self.mask_value,
                input_chunk,
            )
            for input_chunk in chunks
        )
        print(f"Yielding normalize tasks for bcube: {self.bcube}, MIP: {self.mip}")

        yield tasks


class NormalizeTask(scheduling.Task):
    def __init__(
        self,
        src_layer,
        mask_layers,
        dst_layer,
        mean_layer,
        var_layer,
        stats_mip,
        mip,
        mask_value,
        bcube,
    ):
        super().__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.mean_layer = mean_layer
        self.mask_layers = mask_layers
        self.var_layer = var_layer
        self.stats_mip = stats_mip
        self.mip = mip
        self.bcube = bcube
        self.mask_value = mask_value

    def execute(self):
        corgie_logger.info(
            f"Normalizing {self.src_layer} at MIP{self.mip}, region: {self.bcube}"
        )
        mean_data = self.mean_layer.read(self.bcube, mip=self.stats_mip)
        var_data = self.var_layer.read(self.bcube, mip=self.stats_mip)

        src_data = self.src_layer.read(self.bcube, mip=self.mip)
        mask_data = helpers.read_mask_list(
            mask_list=self.mask_layers, bcube=self.bcube, mip=self.mip
        )

        dst_data = (src_data - mean_data) / var_data.sqrt()
        if mask_data is not None:
            dst_data[mask_data] = self.mask_value
        self.dst_layer.write(dst_data, self.bcube, mip=self.mip)


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
    "--dst_folder",
    nargs=1,
    type=str,
    required=True,
    help="Folder where aligned stack will go",
)
@corgie_option("--suffix", "-s", nargs=1, type=str, default=None)
@corgie_optgroup("Normalize parameters")
@corgie_option("--recompute_stats/--no_recompute_stats", default=True)
@corgie_option("--stats_mip", "-m", nargs=1, type=int, required=None)
@corgie_option("--mip_start", "-m", nargs=1, type=int, required=True)
@corgie_option("--mip_end", "-e", nargs=1, type=int, required=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=2048)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--force_chunk_xy", nargs=1, type=int)
@corgie_option("--mask_value", nargs=1, type=float, default=0.0)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def normalize(
    ctx,
    src_layer_spec,
    dst_folder,
    stats_mip,
    mip_start,
    mip_end,
    chunk_xy,
    chunk_z,
    force_chunk_xy,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
    recompute_stats,
    mask_value,
):
    if chunk_z != 1:
        raise NotImplemented(
            "Compute Statistics command currently only \
                supports per-section statistics."
        )
    result_report = ""
    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_norm"
    else:
        suffix = f"_{suffix}"

    if stats_mip is None:
        stats_mip = mip_end

    if not force_chunk_xy:
        force_chunk_xy = chunk_xy

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["img"],
        readonly=False,
        suffix=suffix,
        overwrite=True,
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    img_layers = src_stack.get_layers_of_type("img")
    mask_layers = src_stack.get_layers_of_type("mask")
    field_layers = src_stack.get_layers_of_type("field")
    assert len(field_layers) == 0

    for l in img_layers:
        mean_layer = l.get_sublayer(
            name=f"mean_{l.name}{suffix}",
            path=os.path.join(dst_folder, f"mean_{l.name}{suffix}"),
            layer_type="section_value",
        )

        var_layer = l.get_sublayer(
            name=f"var_{l.name}{suffix}",
            path=os.path.join(dst_folder, f"var_{l.name}{suffix}"),
            layer_type="section_value",
        )

        if recompute_stats:
            compute_stats_job = ComputeStatsJob(
                src_layer=l,
                mask_layers=mask_layers,
                mean_layer=mean_layer,
                var_layer=var_layer,
                bcube=bcube,
                mip=stats_mip,
                chunk_xy=chunk_xy,
                chunk_z=chunk_z,
            )

            # create scheduler and execute the job
            scheduler.register_job(
                compute_stats_job, job_name=f"Compute Stats. Layer: {l}, Bcube: {bcube}"
            )
            scheduler.execute_until_completion()

        dst_layer = l.get_sublayer(
            name=f"{l.name}{suffix}",
            path=os.path.join(dst_folder, "img", f"{l.name}{suffix}"),
            layer_type=l.get_layer_type(),
            dtype="float32",
            force_chunk_xy=force_chunk_xy,
            overwrite=True,
        )

        result_report += f"Normalized {l} -> {dst_layer}\n"
        for mip in range(mip_start, mip_end + 1):
            normalize_job = NormalizeJob(
                src_layer=l,
                mask_layers=mask_layers,
                dst_layer=dst_layer,
                mean_layer=mean_layer,
                var_layer=var_layer,
                stats_mip=stats_mip,
                mip=mip,
                bcube=bcube,
                chunk_xy=chunk_xy,
                chunk_z=chunk_z,
                mask_value=mask_value,
            )

            # create scheduler and execute the job
            scheduler.register_job(
                normalize_job, job_name=f"Normalize {bcube}, MIP {mip}"
            )
    scheduler.execute_until_completion()
    corgie_logger.info(result_report)
