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
)
from corgie.spec import (
    spec_to_stack,
    spec_to_layer_dict_readonly,
    spec_to_layer_dict_overwrite,
)

from corgie.cli.compute_field import ComputeFieldJob
import json
from copy import deepcopy
import numpy as np


@click.command()
@corgie_optgroup("Layer Parameters")
@corgie_option(
    "--spec_path",
    nargs=1,
    type=str,
    required=True,
    help="JSON spec relating src stacks, src z to dst z",
)
@corgie_optgroup("Copy Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--blend_xy", nargs=1, type=int, default=0)
@corgie_option("--pad", nargs=1, type=int, default=512)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_option("--processor_spec", nargs=1, type=str, multiple=True, required=True)
@corgie_option("--processor_mip", nargs=1, type=int, multiple=True, required=True)
@click.option("--clear_nontissue_field", type=str, default=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_option("--suffix", nargs=1, type=str, default=None)
@click.pass_context
def compute_field_by_spec(
    ctx,
    spec_path,
    chunk_xy,
    blend_xy,
    pad,
    crop,
    processor_spec,
    processor_mip,
    clear_nontissue_field,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
):

    scheduler = ctx.obj["scheduler"]
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    with open(spec_path, "r") as f:
        spec = json.load(f)

    src_layers = spec_to_layer_dict_readonly(spec["src"])
    tgt_layers = spec_to_layer_dict_readonly(spec["tgt"])

    # if force_chunk_xy:
    #     force_chunk_xy = chunk_xy
    # else:
    #     force_chunk_xy = None

    # if force_chunk_z:
    #     force_chunk_z = chunk_z
    # else:
    #     force_chunk_z = None
    if crop is None:
        crop = pad

    reference_layer = src_layers[list(src_layers.keys())[0]]
    dst_layers = spec_to_layer_dict_overwrite(
        spec["dst"], reference_layer=reference_layer, default_type="field"
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    for dst_z in range(*bcube.z_range()):
        spec_z = str(dst_z)
        if spec_z in spec["job_specs"].keys():
            for job_spec in spec["job_specs"][spec_z]:
                src_stack = spec_to_stack(job_spec, "src", src_layers)
                tgt_stack = spec_to_stack(job_spec, "tgt", tgt_layers)
                dst_layer = dst_layers[str(job_spec["dst_img"])]
                ps = json.loads(processor_spec[0])
                ps["ApplyModel"]["params"]["val"] = job_spec["mask_id"]
                ps["ApplyModel"]["params"]["scale"] = job_spec["scale"]
                processor_spec = (json.dumps(ps),)
                job_bcube = bcube.reset_coords(
                    zs=job_spec["src_z"], ze=job_spec["src_z"] + 1, in_place=False
                )
                tgt_z_offset = job_spec["tgt_z"] - job_spec["src_z"]
                compute_field_job = ComputeFieldJob(
                    src_stack=src_stack,
                    tgt_stack=tgt_stack,
                    dst_layer=dst_layer,
                    chunk_xy=chunk_xy,
                    chunk_z=1,
                    blend_xy=blend_xy,
                    processor_spec=processor_spec,
                    pad=pad,
                    crop=crop,
                    bcube=job_bcube,
                    tgt_z_offset=tgt_z_offset,
                    suffix=suffix,
                    processor_mip=processor_mip,
                    clear_nontissue_field=clear_nontissue_field,
                )
                scheduler.register_job(
                    compute_field_job,
                    job_name="ComputeField {},{}".format(
                        job_bcube, job_spec["mask_id"]
                    ),
                )
    scheduler.execute_until_completion()
