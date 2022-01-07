import json
import math

import click
import torch
from corgie import helpers, residuals, scheduling
from corgie.argparsers import (
    corgie_optgroup,
    corgie_option,
    create_layer_from_dict,
)
from corgie.boundingcube import get_bcube_from_coords
from corgie.log import logger as corgie_logger
from corgie.spec import spec_to_layer_dict_readonly
from corgie.stack import FieldSet


class MergeRenderJob(scheduling.Job):
    def __init__(self, src_layers, src_specs, dst_layer, mip, pad, bcube, chunk_xy):
        """Render multiple images to the same destination image

        Args:
            src_layers ({'img':{Layers},
            src_specs (json): list of dicts with img, field, mask, z, & mask_id per island
                ranked by layer priority (first image overwrites later images)
            dst_layer (Stack)
            mip (int)
            pad (int)
            bcube (BoundingCube)
        """
        self.src_layers = src_layers
        self.src_specs = src_specs
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        super().__init__()

    def task_generator(self):
        chunks = self.dst_layer.break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=1,
            mip=self.mip,
            return_generator=True,
        )

        if "src_img" in self.src_specs[0]:
            tasks = (
                MergeRenderImageTask(
                    src_layers=self.src_layers,
                    src_specs=self.src_specs,
                    dst_layer=self.dst_layer,
                    mip=self.mip,
                    pad=self.pad,
                    bcube=input_chunk,
                )
                for input_chunk in chunks
            )
        else:
            tasks = (
                MergeRenderMaskTask(
                    src_layers=self.src_layers,
                    src_specs=self.src_specs,
                    dst_layer=self.dst_layer,
                    mip=self.mip,
                    pad=self.pad,
                    bcube=input_chunk,
                )
                for input_chunk in chunks
            )
        corgie_logger.info(
            f"Yielding render tasks for bcube: {self.bcube}, MIP: {self.mip}"
        )

        yield tasks


class MergeRenderImageTask(scheduling.Task):
    def __init__(self, src_layers, src_specs, dst_layer, mip, pad, bcube):
        """Render multiple images to the same destination image

        Args:
            src_layers ({Layer})
            src_specs (json): list of dicts with img, field, mask, z, & mask_id per island
                ranked by layer priority (first image overwrites later images)
            dst_layer (Stack)
            mip (int)
            pad (int)
            bcube (BoundingCube)
        """
        super().__init__(self)
        self.src_layers = src_layers
        self.src_specs = src_specs
        self.dst_layer = dst_layer
        self.mip = mip
        self.pad = pad
        self.bcube = bcube

    def execute(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Field padding
        padded_bcube = self.bcube.uncrop(self.pad, self.mip)
        for k, specs in enumerate(self.src_specs[::-1]):
            src_z = specs["src_z"]
            dst_z = self.bcube.z_range()[0]

            corgie_logger.info(f"Load fields for {padded_bcube}")
            # backwards compatible
            if not isinstance(specs["src_field"], list):
                specs["src_field"] = [specs["src_field"]]
            mask_layer = self.src_layers[str(specs["src_mask"])]

            field_ids = list(map(str, specs["src_field"]))
            corgie_logger.info(f"field ids={field_ids}")
            z_list = specs.get("src_field_z", [src_z] * len(field_ids))
            fields = FieldSet([self.src_layers[n] for n in field_ids])
            field = fields.read(
                bcube=padded_bcube, z_list=z_list, mip=self.mip, device=device
            )
            bcube = padded_bcube.reset_coords(zs=src_z, ze=src_z + 1, in_place=False)

            # Extend image/mask cutout to account for field spread
            render_pad = int((field.max_vector() - field.min_vector()).max().ceil().tensor().item())
            snap_factor = 2 ** (max(self.mip, mask_layer.data_mip) - self.mip)
            render_pad = math.ceil(render_pad / snap_factor) * snap_factor
            render_pad = min(render_pad, 4096)  # Safety

            render_bcube = bcube.uncrop(render_pad, self.mip)
            corgie_logger.debug(f"render_pad: {render_pad}")

            # Move image/mask cutout to account for field drift
            img_trans = helpers.percentile_trans_adjuster(field)
            mask_trans = img_trans.round_to_mip(self.mip, mask_layer.data_mip)
            corgie_logger.debug(f"img_trans: {img_trans} | mask_trans: {mask_trans}")

            img_bcube = render_bcube.translate(
                x_offset=img_trans.y, y_offset=img_trans.x, mip=self.mip
            )
            mask_bcube = render_bcube.translate(
                x_offset=mask_trans.y, y_offset=mask_trans.x, mip=self.mip
            )

            if render_pad > 0:
                field = torch.nn.functional.pad(field, [render_pad, render_pad, render_pad, render_pad], mode='replicate')

            corgie_logger.info(f"Load masks for {mask_bcube}")
            mask_id = specs["mask_id"]
            mask_layer.binarizer = helpers.Binarizer(["eq", mask_id])
            mask = mask_layer.read(bcube=mask_bcube, mip=self.mip, device=device)
            mask = residuals.res_warp_img(
                mask.float(), field - mask_trans.to_tensor(device=field.device)
            ).tensor()
            mask = (mask > 0.4).bool()
            cropped_mask = helpers.crop(mask, self.pad + render_pad)

            corgie_logger.info(f"Load image for {img_bcube}")
            if cropped_mask.sum() == 0:
                cropped_img = torch.zeros_like(cropped_mask, dtype=torch.float)
            else:
                img_layer = self.src_layers[str(specs["src_img"])]
                img = img_layer.read(bcube=img_bcube, mip=self.mip, device=device)
                img = residuals.res_warp_img(
                    img.float(), field - img_trans.to_tensor(device=field.device)
                )
                cropped_img = helpers.crop(img, self.pad + render_pad)

            # write to composite image
            if k == 0:
                dst_img = cropped_img
                dst_img[~cropped_mask] = 0
            else:
                dst_img[cropped_mask] = cropped_img[cropped_mask]

        self.dst_layer.write(dst_img.cpu(), bcube=self.bcube, mip=self.mip)


class MergeRenderMaskTask(MergeRenderImageTask):
    """Render multiple masks into the same destination mask, with option to relabel

    Spec format:
        {
            'src_mask' (int)
            'src_field' (int)
            'mask_id' (int): the id of this mask in the source
            'relabel_id' (int): the id of this mask in the destination (optional)
            'src_z' (int)
        }
    """

    def execute(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        padded_bcube = self.bcube.uncrop(self.pad, self.mip)
        for k, specs in enumerate(self.src_specs[::-1]):
            src_z = specs["src_z"]
            dst_z = self.bcube.z_range()[0]

            corgie_logger.info(f"Load fields for {padded_bcube}")
            # backwards compatible
            if not isinstance(specs["src_field"], list):
                specs["src_field"] = [specs["src_field"]]
            mask_layer = self.src_layers[str(specs["src_mask"])]

            field_ids = list(map(str, specs["src_field"]))
            corgie_logger.info(f"field ids={field_ids}")
            z_list = specs.get("src_field_z", [src_z] * len(field_ids))
            fields = FieldSet([self.src_layers[n] for n in field_ids])
            field = fields.read(
                bcube=padded_bcube, z_list=z_list, mip=self.mip, device=device
            )
            bcube = padded_bcube.reset_coords(zs=src_z, ze=src_z + 1, in_place=False)

            mask_trans = helpers.percentile_trans_adjuster(field)
            mask_trans = mask_trans.round_to_mip(self.mip, mask_layer.data_mip)
            corgie_logger.debug(f"mask_trans: {mask_trans}")

            mask_bcube = bcube.translate(
                x_offset=mask_trans.y, y_offset=mask_trans.x, mip=self.mip
            )

            corgie_logger.info(f"Load masks for {mask_bcube}")
            mask_id = specs["mask_id"]
            mask_layer.binarizer = helpers.Binarizer(["eq", mask_id])
            mask = mask_layer.read(bcube=mask_bcube, mip=self.mip, device=device)
            mask = residuals.res_warp_img(
                mask.float(), field - mask_trans.to_tensor(device=field.device)
            ).tensor()
            mask = (mask > 0.4).bool()
            cropped_mask = helpers.crop(mask, self.pad)

            relabel_id = torch.as_tensor(specs.get("relabel_id", k + 1), dtype=torch.uint8)
            if k == 0:
                dst_img = cropped_mask * relabel_id
                dst_img[~cropped_mask] = 0
            else:
                dst_img[cropped_mask] = cropped_mask[cropped_mask] * relabel_id

        self.dst_layer.write(dst_img.cpu(), bcube=self.bcube, mip=self.mip)


@click.command()
@corgie_optgroup("Layer Parameters")
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
    required=False,
    help="Destination folder for the copied stack",
)
@corgie_optgroup("Render Method Specification")
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--force_chunk_xy", nargs=1, type=int)
@corgie_option("--pad", nargs=1, type=int, default=512)
@corgie_option("--mip", nargs=1, type=int, required=True)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@corgie_option("--suffix", nargs=1, type=str, default=None)
@click.pass_context
def merge_render(
    ctx,
    spec_path,
    dst_folder,
    chunk_xy,
    pad,
    start_coord,
    end_coord,
    coord_mip,
    force_chunk_xy,
    mip,
    suffix,
):

    scheduler = ctx.obj["scheduler"]
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    corgie_logger.debug("Setting up layers...")
    # create layers
    # collect image paths
    # collect mask paths

    if not force_chunk_xy:
        force_chunk_xy = chunk_xy

    with open(spec_path, "r") as f:
        spec = json.load(f)

    src_layers = spec_to_layer_dict_readonly(spec["src"])
    reference_layer = src_layers[list(src_layers.keys())[0]]
    dst_layer = create_layer_from_dict(
        {"path": dst_folder, "type": "img"},
        reference=reference_layer,
        force_chunk_xy=force_chunk_xy,
        overwrite=True
    )

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    for z in range(*bcube.z_range()):
        tgt_z = str(z)
        if tgt_z in spec["job_specs"].keys():
            job_bcube = bcube.reset_coords(zs=z, ze=z + 1, in_place=False)
            render_job = MergeRenderJob(
                src_layers=src_layers,
                src_specs=spec["job_specs"][tgt_z],
                dst_layer=dst_layer,
                mip=mip,
                pad=pad,
                bcube=job_bcube,
                chunk_xy=chunk_xy,
            )
            scheduler.register_job(
                render_job, job_name="MergeRender {}".format(job_bcube)
            )
    scheduler.execute_until_completion()
