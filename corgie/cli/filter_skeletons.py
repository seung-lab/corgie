import click
from cloudfiles import CloudFiles
from cloudvolume import PrecomputedSkeleton, Skeleton
import numpy as np
import os
import pickle

from corgie import scheduling, argparsers, stack
from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_vertices
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)
from cloudvolume.lib import Bbox


def get_skeleton(src_path, skeleton_id_str):
    cf = CloudFiles(src_path)
    return Skeleton.from_precomputed(cf.get(skeleton_id_str))


def get_skeleton_vert_neighbor_ids(sk, vert_id):
    vert_edges = np.where(
        np.isin(sk.edges[:, 0], vert_id) + np.isin(sk.edges[:, 1], vert_id)
    )[0]

    neighbors = sk.edges[vert_edges[-1]].flatten()

    # filter out self
    self_index = np.where(neighbors == vert_id)
    neighbors = np.delete(neighbors, self_index)

    return neighbors


def rip_out_verts(sk, vert_ids):
    vert_edges = np.where(
        np.isin(sk.edges[:, 0], vert_ids) + np.isin(sk.edges[:, 1], vert_ids)
    )[0]
    sk.edges = np.delete(sk.edges, vert_edges, axis=0)

    return sk.consolidate()


class FilterSkeletonsJob(scheduling.Job):
    def __init__(
        self,
        src_path,
        dst_path,
        skeleton_ids,
        bad_sections,
        z_start,
        z_end,
        skeleton_length_file=None,
    ):
        self.src_path = src_path
        self.dst_path = dst_path
        self.bad_sections = bad_sections
        self.z_start = z_start
        self.z_end = z_end

        if skeleton_ids is None:
            self.skeleton_ids = self.get_all_skeleton_ids()
        else:
            self.skeleton_ids = skeleton_ids
        self.skeleton_length_file = skeleton_length_file
        super().__init__()

    def task_generator(self):
        skeletons = self.get_skeletons(self.src_path)
        if self.z_start is not None and self.z_end is not None:
            bbox = Bbox((0, 0, self.z_start * 40), (10e8, 10e8, self.z_end * 40))
        else:
            bbox = None

        lengths = []
        for skeleton_id_str, sk in skeletons.items():
            deleted = 1
            if bbox is not None:
                sk = sk.crop(bbox)
            while deleted != 0:
                deleted = 0
                verts = sk.vertices
                vert_zs = verts[:, 2].copy()
                vert_zs /= 40  # hack
                vert_zs = vert_zs.astype(np.int32)

                bad_verts = np.where(np.isin(vert_zs, self.bad_sections))[0]

                # Find the next set of vertices to remove so that
                # we don't mess with indices, remove the in the next pass
                deleted = 0
                for bv in bad_verts:
                    neighbors = get_skeleton_vert_neighbor_ids(sk, bv)

                    # filter out other bad vertices
                    bad_index = np.where(np.isin(neighbors, bad_verts))
                    neighbors = np.delete(neighbors, bad_index)

                    assert np.isin(neighbors.flatten(), bad_verts).sum() == 0

                    # if there's a good neighbor, reasign all the endpoints to it
                    # else wait until the next round
                    if len(neighbors) != 0:
                        replacement = neighbors[0]
                        assert replacement not in bad_verts
                        ed = np.expand_dims(sk.edges, -1)
                        ed[np.where(ed == bv)] = replacement
                        sk.edges = ed.squeeze(-1)
                        # this leaves self-edges
                        deleted += 1
                # This removes self edges
                sk = sk.consolidate()
                new_v_count = sk.vertices.shape[0]

            verts = sk.vertices
            vert_zs = verts[:, 2].copy()
            vert_zs /= 40  # hack
            vert_zs = vert_zs.astype(np.int32)
            bad_verts = np.where(np.isin(vert_zs, self.bad_sections))[0]

            lengths.append((skeleton_id_str, sk.cable_length()))
            cf = CloudFiles(self.dst_path)
            cf.put(
                path=skeleton_id_str,
                content=sk.to_precomputed(),
                compress="gzip",
            )
        for n, l in sorted(lengths):
            print(l)
        import sys

        sys.exit(1)
        # print (len(bad_verts))
        # print (sk.vertices.shape)

    def get_skeletons(self, folder):
        skeleton_filenames = [str(skeleton_id) for skeleton_id in self.skeleton_ids]
        cf = CloudFiles(folder)
        skeleton_files = cf.get(skeleton_filenames)
        skeletons = {}
        for skeleton_file in skeleton_files:
            skeleton_id_str = skeleton_file["path"]
            skeleton = Skeleton.from_precomputed(skeleton_file["content"])
            skeletons[skeleton_id_str] = skeleton
        return skeletons

    def get_all_skeleton_ids(self):
        cf = CloudFiles(self.src_path)
        skeleton_filenames = cf.list(flat=True)
        skeleton_ids = []
        for skeleton_filename in skeleton_filenames:
            if ":" in skeleton_filename:
                # Fragment
                continue
            skeleton_ids.append(int(skeleton_filename))
        return skeleton_ids


@click.command()
@corgie_optgroup("Layer Parameters")
@corgie_option(
    "--src_folder",
    nargs=1,
    type=str,
    required=True,
    help="Folder where skeletons are stored",
)
@corgie_option(
    "--dst_folder",
    nargs=1,
    type=str,
    required=True,
    help="Folder where to store new skeletons",
)
@corgie_optgroup("Misc Parameters")
@corgie_option(
    "--bad_sections", "-b", multiple=True, type=int, help="Bad sections to filter out"
)
@corgie_option(
    "--bad_sections", "-b", multiple=True, type=int, help="Bad sections to filter out"
)
@corgie_option("--z_start", type=int, default=None)
@corgie_option("--z_end", type=int, default=None)
@corgie_option("--ids", multiple=True, type=int, help="Ids to transform")
@corgie_option("--ids_filepath", type=str, help="File containing ids to transform")
@click.pass_context
def filter_skeletons(
    ctx, src_folder, dst_folder, ids, bad_sections, ids_filepath, z_start, z_end
):
    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")

    skeleton_ids = ids
    if ids_filepath is not None:
        skeleton_ids = []
        with open(ids_filepath, "r") as f:
            line = f.readline()
            while line:
                skeleton_ids.append(int(line))
                line = f.readline()

    if len(skeleton_ids) == 0:
        skeleton_ids = None
    else:
        skeleton_ids = list(skeleton_ids)

    transform_skeletons_job = FilterSkeletonsJob(
        src_path=src_folder,
        dst_path=dst_folder,
        skeleton_ids=skeleton_ids,
        bad_sections=bad_sections,
        z_start=z_start,
        z_end=z_end,
    )

    scheduler.register_job(
        transform_skeletons_job,
        job_name="Filtering skeletons in {}".format(src_folder),
    )

    scheduler.execute_until_completion()
    result_report = f"Filtered skeletons stored at {dst_folder}. "
    corgie_logger.info(result_report)
