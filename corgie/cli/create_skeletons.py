import click
from cloudfiles import CloudFiles
from cloudvolume import PrecomputedSkeleton
from collections import defaultdict
from copy import deepcopy
import kimimaro
import numpy as np
import pickle

from corgie import scheduling, argparsers, helpers, stack
from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_coords
from corgie.argparsers import (
    LAYER_HELP_STR,
    create_layer_from_spec,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


class SkeletonJob(scheduling.Job):
    def __init__(
        self,
        seg_layer,
        dst_path,
        timestamp,
        bcube,
        chunk_xy,
        chunk_z,
        mip,
        teasar_params,
        object_ids=None,
        fix_branching=True,
        fix_borders=True,
        fix_avocados=False,
        dust_threshold=0,
        tick_threshold=1000,
        single_merge_mode=True,
    ):
        self.seg_layer = seg_layer
        self.dst_path = dst_path
        self.timestamp = timestamp
        self.bcube = bcube
        self.chunk_xy = chunk_xy
        self.chunk_z = chunk_z
        self.mip = mip
        self.teasar_params = teasar_params
        self.object_ids = object_ids
        self.fix_branching = fix_branching
        self.fix_borders = fix_borders
        self.fix_avocados = fix_avocados
        self.dust_threshold = dust_threshold
        self.tick_threshold = tick_threshold
        self.single_merge_mode = single_merge_mode
        super().__init__()

    def task_generator(self):
        chunks = self.seg_layer.break_bcube_into_chunks(
            bcube=self.bcube,
            chunk_xy=self.chunk_xy,
            chunk_z=self.chunk_z,
            mip=self.mip,
            readonly=True,
        )
        tasks = [
            SkeletonTask(
                self.seg_layer,
                self.dst_path,
                timestamp=self.timestamp,
                mip=self.mip,
                teasar_params=self.teasar_params,
                object_ids=self.object_ids,
                dust_threshold=self.dust_threshold,
                fix_branching=self.fix_branching,
                fix_borders=self.fix_borders,
                fix_avocados=self.fix_avocados,
                bcube=input_chunk,
            )
            for input_chunk in chunks
        ]
        corgie_logger.info(
            f"Yielding skeletonization tasks for bcube: {self.bcube}, MIP: {self.mip}"
        )
        yield tasks
        yield scheduling.wait_until_done
        if self.single_merge_mode:
            merge_tasks = [
                MergeSkeletonTask(
                    self.dst_path,
                    self.mip,
                    self.dust_threshold,
                    self.tick_threshold,
                    str(object_id),
                )
                for object_id in self.object_ids
            ]
            yield merge_tasks
        else:
            yield [
                MergeSkeletonTask(
                    self.dst_path, self.mip, self.dust_threshold, self.tick_threshold
                )
            ]


class SkeletonTask(scheduling.Task):
    def __init__(
        self,
        seg_layer,
        dst_path,
        timestamp,
        bcube,
        mip,
        teasar_params,
        object_ids,
        fix_branching,
        fix_borders,
        fix_avocados,
        dust_threshold,
    ):
        super().__init__(self)
        self.seg_layer = seg_layer
        self.dst_path = dst_path
        self.timestamp = timestamp
        self.bcube = bcube
        self.mip = mip
        self.teasar_params = teasar_params
        self.object_ids = object_ids
        self.fix_branching = fix_branching
        self.fix_borders = fix_borders
        self.fix_avocados = fix_avocados
        self.dust_threshold = dust_threshold

    def execute(self):
        corgie_logger.info(
            f"Skeletonizing {self.seg_layer} at MIP{self.mip}, region: {self.bcube}"
        )
        seg_data = self.seg_layer.read(
            bcube=self.bcube, mip=self.mip, timestamp=self.timestamp
        )
        resolution = self.seg_layer.cv[self.mip].resolution
        skeletons = kimimaro.skeletonize(
            seg_data,
            self.teasar_params,
            object_ids=self.object_ids,
            anisotropy=resolution,
            dust_threshold=self.dust_threshold,
            progress=False,
            fix_branching=self.fix_branching,
            fix_borders=self.fix_borders,
            fix_avocados=self.fix_avocados,
        ).values()

        minpt = self.bcube.minpt(self.mip)
        for skel in skeletons:
            skel.vertices[:] += minpt * resolution

        cf = CloudFiles(self.dst_path)
        for skel in skeletons:
            path = "{}:{}".format(skel.id, self.bcube.to_filename(self.mip))
            cf.put(
                path=path,
                content=pickle.dumps(skel),
                compress="gzip",
                content_type="application/python-pickle",
                cache_control=False,
            )


class MergeSkeletonTask(scheduling.Task):
    def __init__(self, dst_path, mip, dust_threshold, tick_threshold, prefix=""):
        super().__init__(self)
        self.dst_path = dst_path
        self.cf = CloudFiles(self.dst_path)
        self.mip = mip
        self.dust_threshold = dust_threshold
        self.tick_threshold = tick_threshold
        self.prefix = prefix

    def execute(self):
        corgie_logger.info(f"Merging skeletons at {self.dst_path}")
        fragment_filenames = self.cf.list(prefix=self.prefix, flat=True)
        skeleton_files = self.cf.get(fragment_filenames)
        skeletons = defaultdict(list)
        for skeleton_file in skeleton_files:
            try:
                colon_index = skeleton_file["path"].index(":")
            except ValueError:
                # File is full skeleton, not fragment
                continue
            seg_id = skeleton_file["path"][0:colon_index]
            skeleton_fragment = pickle.loads(skeleton_file["content"])
            if not skeleton_fragment.empty():
                skeletons[seg_id].append(skeleton_fragment)
        for seg_id, skeleton_fragments in skeletons.items():
            skeleton = PrecomputedSkeleton.simple_merge(
                skeleton_fragments
            ).consolidate()
            skeleton = kimimaro.postprocess(
                skeleton, self.dust_threshold, self.tick_threshold
            )
            skeleton.id = int(seg_id)
            self.cf.put(path=seg_id, content=skeleton.to_precomputed(), compress="gzip")
            corgie_logger.info(f"Finished skeleton {seg_id}")


@click.command()
@corgie_optgroup("Layer Parameters")
@corgie_option(
    "--seg_layer_spec",
    "--s",
    nargs=1,
    type=str,
    required=True,
    multiple=True,
    help="Seg layer from which to skeletonize segments.",
)
@corgie_option(
    "--dst_folder",
    nargs=1,
    type=str,
    required=True,
    help="Folder where to store the skeletons",
)
@corgie_option(
    "--timestamp",
    nargs=1,
    type=int,
    default=None,
    help="UNIX timestamp that specifies which segmentation to use. Only relevant for graphene segmentations",
)
@corgie_optgroup("Skeletonization Parameters")
@corgie_option("--mip", nargs=1, type=int, required=True)
@corgie_option("--teasar_scale", nargs=1, type=int, default=10)
@corgie_option("--teasar_const", nargs=1, type=int, default=10)
@corgie_option("--ids", multiple=True, type=int, help="Segmentation ids to skeletonize")
@corgie_option(
    "--ids_filepath", type=str, help="File containing segmentation ids to skeletonize"
)
@corgie_option("--tick_threshold", nargs=1, type=int, default=1000)
@corgie_option("--chunk_xy", nargs=1, type=int, default=256)
@corgie_option("--chunk_z", nargs=1, type=int, default=256)
@corgie_option(
    "--single_merge_mode",
    nargs=1,
    type=bool,
    default=True,
    help="Set to True to have 1 skeleton merge=1 task",
)
@corgie_optgroup("Data Region Specification")
@corgie_option("--start_coord", nargs=1, type=str, required=True)
@corgie_option("--end_coord", nargs=1, type=str, required=True)
@corgie_option("--coord_mip", nargs=1, type=int, default=0)
@click.pass_context
def create_skeletons(
    ctx,
    seg_layer_spec,
    dst_folder,
    timestamp,
    mip,
    teasar_scale,
    teasar_const,
    ids,
    ids_filepath,
    tick_threshold,
    chunk_xy,
    chunk_z,
    single_merge_mode,
    start_coord,
    end_coord,
    coord_mip,
):
    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")
    seg_stack = create_stack_from_spec(seg_layer_spec, name="src", readonly=True)
    bcube = get_bcube_from_coords(start_coord, end_coord, coord_mip)

    object_ids = ids
    if ids_filepath is not None:
        object_ids = []
        with open(ids_filepath, "r") as f:
            line = f.readline()
            while line:
                object_ids.append(int(line))
                line = f.readline()
    if object_ids is None or len(object_ids) == 0:
        raise ValueError("Must specify ids to skeletonize")
    object_ids = list(object_ids)
    teasar_params = {"scale": teasar_scale, "const": teasar_const}

    seg_layer = seg_stack.get_layers_of_type("segmentation")[0]
    skeleton_job = SkeletonJob(
        seg_layer=seg_layer,
        dst_path=dst_folder,
        timestamp=timestamp,
        bcube=bcube,
        chunk_xy=chunk_xy,
        chunk_z=chunk_z,
        mip=mip,
        teasar_params=teasar_params,
        object_ids=object_ids,
        tick_threshold=tick_threshold,
        single_merge_mode=single_merge_mode,
    )

    scheduler.register_job(skeleton_job, job_name="Skeletonize {}".format(bcube))

    scheduler.execute_until_completion()
    result_report = f"Skeletonized {str(seg_layer)}. "
    corgie_logger.info(result_report)
