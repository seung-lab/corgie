import os
import click

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie.cli.compute_stats import ComputeStatsJob
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)
from corgie.cli.normalize import NormalizeJob
import json


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
    "--spec_path",
    nargs=1,
    type=str,
    required=True,
    help="JSON spec relating src stacks, src z to dst z",
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
@corgie_option("--mask_value", nargs=1, type=float, default=0.0)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def normalize_by_spec(
    ctx,
    src_layer_spec,
    spec_path,
    dst_folder,
    stats_mip,
    mip_start,
    mip_end,
    chunk_xy,
    chunk_z,
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

    with open(spec_path, "r") as f:
        spec = set(json.load(f))

    corgie_logger.debug("Setting up layers...")
    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    img_layers = src_stack.get_layers_of_type("img")
    mask_layers = src_stack.get_layers_of_type("mask")
    field_layers = src_stack.get_layers_of_type("field")
    assert len(field_layers) == 0

    for l in img_layers:
        mean_layer = l.get_sublayer(
            name=f"mean{suffix}",
            path=os.path.join(dst_folder, f"mean{suffix}"),
            layer_type="section_value",
        )

        var_layer = l.get_sublayer(
            name=f"var{suffix}",
            path=os.path.join(dst_folder, f"var{suffix}"),
            layer_type="section_value",
        )

        if recompute_stats:
            for z in range(*bcube.z_range()):
                if z in spec:
                    job_bcube = bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
                    compute_stats_job = ComputeStatsJob(
                        src_layer=l,
                        mask_layers=mask_layers,
                        mean_layer=mean_layer,
                        var_layer=var_layer,
                        bcube=job_bcube,
                        mip=stats_mip,
                        chunk_xy=chunk_xy,
                        chunk_z=chunk_z,
                    )

                    # create scheduler and execute the job
                    scheduler.register_job(
                        compute_stats_job,
                        job_name=f"Compute Stats. Layer: {l}, Bcube: {job_bcube}",
                    )
            scheduler.execute_until_completion()

        dst_layer = l.get_sublayer(
            name=f"{l.name}{suffix}",
            path=os.path.join(dst_folder, "img", f"{l.name}{suffix}"),
            layer_type=l.get_layer_type(),
            dtype="float32",
            overwrite=True,
        )

        for z in range(*bcube.z_range()):
            if z in spec:
                job_bcube = bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
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
                        bcube=job_bcube,
                        chunk_xy=chunk_xy,
                        chunk_z=chunk_z,
                        mask_value=mask_value,
                    )

                    # create scheduler and execute the job
                    scheduler.register_job(
                        normalize_job, job_name=f"Normalize {job_bcube}, MIP {mip}"
                    )
    scheduler.execute_until_completion()
    corgie_logger.info(result_report)
