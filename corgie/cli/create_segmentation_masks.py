import click

from corgie import exceptions, stack, helpers, scheduling, argparsers
import json

from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords

from corgie.argparsers import (
    LAYER_HELP_STR,
    create_layer_from_dict,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)

from corgie.cli.compare_sections import CompareSectionsJob
from corgie.cli.combine_masks import CombineMasksJob


class DetectSlipMisalignmentsJob(scheduling.Job):
    def __init__(self, src_stack, dst_layer, mip, bcube, pad, chunk_xy, chunk_z):
        super().__init__()
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube
        self.pad = pad
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

    @classmethod
    def get_exp(cls):
        return {
            "inputs": [
                {"weight": -1, "key": -3, "offset": 0},
                {"weight": -1, "key": -2, "offset": 0},
                {"weight": -1, "key": -1, "offset": 0},
                {"weight": -1, "key": -1, "offset": 1},
                {"weight": -1, "key": -2, "offset": 2},
                {"weight": -1, "key": -3, "offset": 3},
            ],
            "threshold": -2,
        }

    def task_generator(self):
        exp = DetectSlipMisalignmentsJob.get_exp()
        combine_masks_job = CombineMasksJob(
            src_stack=self.src_stack,
            exp=exp,
            dst_layer=self.dst_layer,
            mip=self.mip,
            bcube=self.bcube,
            pad=self.pad,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
        )
        yield from combine_masks_job.task_generator


class DetectStepMisalignmentsJob(scheduling.Job):
    def __init__(self, src_stack, dst_layer, mip, bcube, pad, chunk_xy, chunk_z):
        super().__init__()
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.mip = mip
        self.bcube = bcube
        self.pad = pad
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z

    @classmethod
    def get_exp(cls):
        return {
            "inputs": [
                {
                    "inputs": [
                        {"weight": 1, "key": -3, "offset": 0},
                        {"weight": 1, "key": -2, "offset": 0},
                        {"weight": 1, "key": -1, "offset": 0},
                    ],
                    "threshold": 1,
                },
                {
                    "inputs": [
                        {"weight": -1, "key": -1, "offset": 1},
                        {"weight": -1, "key": -2, "offset": 2},
                        {"weight": -1, "key": -3, "offset": 3},
                    ],
                    "threshold": -1,
                },
            ],
            "threshold": 1,
        }

    def task_generator(self):
        exp = DetectStepMisalignmentsJob.get_exp()
        combine_masks_job = CombineMasksJob(
            src_stack=self.src_stack,
            exp=exp,
            dst_layer=self.dst_layer,
            mip=self.mip,
            bcube=self.bcube,
            pad=self.pad,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
        )
        yield from combine_masks_job.task_generator


class DetectConsecutiveMasksJob(scheduling.Job):
    def __init__(
        self,
        src_stack,
        dst_layer,
        mip,
        bcube,
        pad,
        chunk_xy,
        chunk_z,
        num_consecutive=3,
        key="slip",
    ):
        super().__init__()
        self.src_stack = src_stack
        self.dst_layer = dst_layer
        self.num_consecutive = num_consecutive
        self.mip = mip
        self.bcube = bcube
        self.pad = pad
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.key = key

    @classmethod
    def get_exp(cls, n, key="slip"):
        """Return boolean expression for n consecutive sections.

        e.g. n=3 would return:
        {
            "inputs": [
                {
                    "inputs": [
                        {"weight": 1, "key": 0, "offset": -2},
                        {"weight": 1, "key": 0, "offset": -1},
                        {"weight": 1, "key": 0, "offset": 0},
                    ],
                    "threshold": 3,
                },
                {
                    "inputs": [
                        {"weight": 1, "key": 0, "offset": -1},
                        {"weight": 1, "key": 0, "offset": 0},
                        {"weight": 1, "key": 0, "offset": 1},
                    ],
                    "threshold": 3,
                },
                {
                    "inputs": [
                        {"weight": 1, "key": 0, "offset": 0},
                        {"weight": 1, "key": 0, "offset": 1},
                        {"weight": 1, "key": 0, "offset": 2},
                    ],
                    "threshold": 3,
                },
            ],
            "threshold": 1,
        }
        """
        o = {"inputs": [], "threshold": 0}
        for i in range(n):
            oi = {"inputs": [], "threshold": n - 1}
            for j in range(n):
                oij = {"weight": 1, "key": key, "offset": i + j - n + 1}
                oi["inputs"].append(oij)
            o["inputs"].append(oi)
        return o

    def task_generator(self):
        exp = DetectConsecutiveMasksJob.get_exp(n=self.num_consecutive, key=self.key)
        combine_masks_job = CombineMasksJob(
            src_stack=self.src_stack,
            exp=exp,
            dst_layer=self.dst_layer,
            mip=self.mip,
            bcube=self.bcube,
            pad=self.pad,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
        )
        yield from combine_masks_job.task_generator


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
@corgie_optgroup("Create Segmentation Masks Specification")
@corgie_option("--processor_spec", nargs=1, type=str, required=True, multiple=False)
@corgie_option(
    "--processor_mip",
    "-m",
    nargs=1,
    type=int,
    required=True,
    help="MIP of the input for creating masks",
)
@corgie_option(
    "--dst_mip",
    "-m",
    nargs=1,
    type=int,
    required=True,
    help="MIP of the final masks",
)
@corgie_option(
    "--compute_similarities/--skip_similarities",
    default=True,
    help="If not computing similarities, they must be defined as src_layer_specs. "
    + "Define similarities as masks with similar set to True, and False otherwise. "
    + "Label similarities with their z offset (e.g. -3, -2, -1).",
)
@corgie_option(
    "--similarity_threshold",
    nargs=1,
    type=float,
    default=0.1,
    help="If computing_similarities=True, similarity values above this threshold will be considered similar.",
)
@corgie_option(
    "--compute_slip_mask/--skip_slip_mask",
    default=True,
    help="If not computing slip mask, must be defined as src_layer_specs. "
    + "The mask should be named 'slip', and be True where there is a misalignment.",
)
@corgie_option(
    "--compute_step_mask/--skip_step_mask",
    default=True,
    help="If not computing step mask, it must be defined as src_layer_specs. "
    + "The mask should be named 'stpe', and be True where there is a misalignment.",
)
@corgie_option("--compute_affinity_mask/--skip_affinity_mask", default=True)
@corgie_option("--chunk_xy", "-c", nargs=1, type=int, default=1024)
@corgie_option("--chunk_z", nargs=1, type=int, default=1)
@corgie_option("--force_chunk_xy", nargs=1, type=int, default=None)
@corgie_option("--pad", nargs=1, type=int, default=256)
@corgie_option("--crop", nargs=1, type=int, default=None)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def create_segmentation_masks(
    ctx,
    src_layer_spec,
    dst_folder,
    processor_spec,
    pad,
    crop,
    processor_mip,
    dst_mip,
    chunk_xy,
    chunk_z,
    force_chunk_xy,
    start_coord,
    end_coord,
    coord_mip,
    suffix,
    similarity_threshold,
    compute_similarities,
    compute_slip_mask,
    compute_step_mask,
    compute_affinity_mask,
):

    scheduler = ctx.obj["scheduler"]

    if suffix is None:
        suffix = "_aligned"
    else:
        suffix = f"_{suffix}"

    if crop is None:
        crop = pad

    corgie_logger.debug("Setting up layers...")

    src_stack = create_stack_from_spec(src_layer_spec, name="src", readonly=True)
    src_stack.folder = dst_folder

    if force_chunk_xy is None:
        force_chunk_xy = chunk_xy

    dst_stack = stack.create_stack_from_reference(
        reference_stack=src_stack,
        folder=dst_folder,
        name="dst",
        types=["img", "mask"],
        readonly=False,
        suffix=suffix,
        force_chunk_xy=force_chunk_xy,
        overwrite=True,
    )

    corgie_logger.debug("Done!")

    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    if compute_similarities:
        z_offsets = range(-1, -4, -1)
        for tgt_z_offset in z_offsets:
            if tgt_z_offset not in src_stack:
                dst_layer = dst_stack.create_sublayer(
                    name=tgt_z_offset,
                    layer_type="img",
                    overwrite=True,
                    layer_args={"dtype": "uint8"},
                )

                proc_spec = json.loads(processor_spec)
                if isinstance(proc_spec, dict):
                    assert str(tgt_z_offset) in proc_spec
                    proc_spec = json.dumps(proc_spec[str(tgt_z_offset)])
                else:
                    proc_spec = processor_spec

                compare_job = CompareSectionsJob(
                    src_stack=src_stack,
                    tgt_stack=src_stack,
                    dst_layer=dst_layer,
                    chunk_xy=chunk_xy,
                    processor_spec=proc_spec,
                    pad=pad,
                    bcube=bcube,
                    tgt_z_offset=tgt_z_offset,
                    suffix=suffix,
                    mip=processor_mip,
                    dst_mip=dst_mip,
                )
                scheduler.register_job(
                    compare_job,
                    job_name="Compare Sections Job {}, tgt z offset {}".format(
                        bcube, tgt_z_offset
                    ),
                )
        scheduler.execute_until_completion()
        result_report = f"Similarity results in {[str(l) for l in dst_stack.get_layers_of_type('img')]}"
        corgie_logger.info(result_report)

        # If similarities were just computed, add them to the src_stack as masks
        # Otherwise, they need to be included with src_stack_specs as masks
        # See arg help above for similarity mask and misalignment mask definitions.
        for layer_name in z_offsets:
            img_layer = dst_stack[layer_name]
            binarizer = {
                "binarization": ["gt", similarity_threshold],
                # "cv_params": {"cache": True},
            }
            layer_dict = {
                "path": img_layer.path,
                "name": img_layer.name,
                "type": "mask",
                "args": binarizer,
            }
            mask_layer = create_layer_from_dict(layer_dict, reference=dst_stack)
            src_stack.add_layer(mask_layer)

    corgie_logger.info("Computing slip & step masks")
    if compute_slip_mask:
        slip_layer = dst_stack.create_sublayer(
            name="slip", layer_type="mask", overwrite=True
        )
        slip_bcube = bcube.reset_coords(
            zs=bcube.z[0] + 1, ze=bcube.z[1] - 1, in_place=False
        )
        slip_misalignments_job = DetectSlipMisalignmentsJob(
            src_stack=src_stack,
            dst_layer=slip_layer,
            mip=dst_mip,
            bcube=slip_bcube,
            pad=pad,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
        )
        scheduler.register_job(
            slip_misalignments_job,
            job_name="Detect Slip Misalignments {}".format(bcube),
        )
    if compute_step_mask:
        step_layer = dst_stack.create_sublayer(
            name="step", layer_type="mask", overwrite=True
        )
        step_bcube = bcube.reset_coords(
            zs=bcube.z[0] + 2, ze=bcube.z[1] - 2, in_place=False
        )
        step_misalignments_job = DetectStepMisalignmentsJob(
            src_stack=src_stack,
            dst_layer=step_layer,
            mip=dst_mip,
            bcube=step_bcube,
            pad=pad,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
        )
        scheduler.register_job(
            step_misalignments_job,
            job_name="Detect Step Misalignments {}".format(bcube),
        )
    if compute_slip_mask or compute_step_mask:
        # Execute slip & step masks at the same time
        scheduler.execute_until_completion()
        result_report = f"Slip & step masks in in {str(slip_layer), str(step_layer)}"
        corgie_logger.info(result_report)

    if compute_affinity_mask:
        corgie_logger.info("Creating affinity masks")
        affinity_layer = dst_stack.create_sublayer(
            name="affinity", layer_type="mask", overwrite=True
        )
        three_consecutive_exp = DetectConsecutiveMasksJob.get_exp(n=3, key="slip")
        exp = {
            "inputs": [
                three_consecutive_exp,
                {"weight": 1, "key": "step", "offset": 0},
            ],
            "threshold": 0,
        }
        affinity_masks_job = CombineMasksJob(
            src_stack=dst_stack,
            exp=exp,
            dst_layer=affinity_layer,
            mip=dst_mip,
            bcube=bcube,
            pad=pad,
            chunk_xy=chunk_xy,
            chunk_z=chunk_z,
        )
        scheduler.register_job(
            affinity_masks_job, job_name="Affinity Masks {}".format(bcube)
        )
        scheduler.execute_until_completion()
        result_report = f"Results in {str(affinity_layer)}"
        corgie_logger.info(result_report)