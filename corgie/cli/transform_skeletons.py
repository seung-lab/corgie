import click
from cloudfiles import CloudFiles
from cloudvolume import PrecomputedSkeleton, Skeleton
import numpy as np
import os
import pickle
import time

from corgie import scheduling, argparsers, stack
from corgie.log import logger as corgie_logger
from corgie.boundingcube import get_bcube_from_vertices
from corgie.argparsers import (
    LAYER_HELP_STR,
    corgie_optgroup,
    corgie_option,
    create_stack_from_spec,
)


def get_skeleton(src_path, skeleton_id_str):
    cf = CloudFiles(src_path)
    return Skeleton.from_precomputed(cf.get(skeleton_id_str))


class GenerateNewSkeletonTask(scheduling.Task):
    def __init__(
        self, skeleton_id_str, src_path, dst_path, task_vertex_size, vertex_sort=False
    ):
        self.src_path = src_path
        self.dst_path = dst_path
        self.skeleton_id_str = skeleton_id_str
        self.task_vertex_size = task_vertex_size
        self.vertex_sort = vertex_sort
        super().__init__()

    def execute(self):

        corgie_logger.info(
            f"Generate new skeleton vertices task for id {self.skeleton_id_str}"
        )
        skeleton = get_skeleton(self.src_path, self.skeleton_id_str)
        if self.vertex_sort:
            vertex_sort = skeleton.vertices[:, 2].argsort()
        else:
            vertex_sort = np.arange(0, len(skeleton.vertices))
        number_vertices = len(skeleton.vertices)
        index_points = list(range(0, number_vertices, self.task_vertex_size))
        cf = CloudFiles(f"{self.dst_path}")
        array_filenames = []
        for i in range(len(index_points)):
            start_index = index_points[i]
            if i + 1 == len(index_points):
                end_index = number_vertices
            else:
                end_index = index_points[i + 1]
            array_filenames.append(
                f"intermediary_arrays/{self.skeleton_id_str}:{start_index}-{end_index}"
            )
        array_files = cf.get(array_filenames)
        # Dict to make sure arrays are concatenated in correct order
        array_dict = {}
        for array_file in array_files:
            array_dict[array_file["path"]] = pickle.loads(array_file["content"])
        array_arrays = []
        for array_filename in array_filenames:
            array_arrays.append(array_dict[array_filename])
        array_arrays = np.concatenate(array_arrays)
        # Restore the correct order of the vertices
        restore_sort = vertex_sort.argsort()
        new_vertices = array_arrays[restore_sort]
        new_skeleton = Skeleton(
            vertices=new_vertices,
            edges=skeleton.edges,
            radii=skeleton.radius,
            vertex_types=skeleton.vertex_types,
            space=skeleton.space,
            transform=skeleton.transform,
        )
        cf.put(
            path=self.skeleton_id_str,
            content=new_skeleton.to_precomputed(),
            compress="gzip",
        )


class TransformSkeletonVerticesTask(scheduling.Task):
    def __init__(
        self,
        vector_field_layer,
        src_path,
        skeleton_id_str,
        dst_path,
        field_mip,
        start_vertex_index,
        end_vertex_index,
        vertex_sort=False,
        mip0_field=False,
    ):
        self.vector_field_layer = vector_field_layer
        self.skeleton_id_str = skeleton_id_str
        self.src_path = src_path
        self.dst_path = dst_path
        self.field_mip = field_mip
        self.start_vertex_index = start_vertex_index
        self.end_vertex_index = end_vertex_index
        self.vertex_sort = vertex_sort
        self.mip0_field = mip0_field
        super().__init__()

    def execute(self):
        corgie_logger.info(
            f"Starting transform skeleton vertices task for id {self.skeleton_id_str}"
        )

        skeleton = get_skeleton(self.src_path, self.skeleton_id_str)

        if self.vertex_sort:
            vertex_sort = skeleton.vertices[:, 2].argsort()
        else:
            vertex_sort = np.arange(0, len(skeleton.vertices))

        # How many vertices we will use at once to get a bcube to download from the vector field
        vertex_process_size = 50
        vertices_to_transform = skeleton.vertices[
            vertex_sort[self.start_vertex_index : self.end_vertex_index]
        ]
        index_vertices = list(range(0, self.number_vertices, vertex_process_size))
        new_vertices = []
        for i in range(len(index_vertices)):
            if i + 1 == len(index_vertices):
                current_batch_vertices = vertices_to_transform[index_vertices[i] :]
            else:
                current_batch_vertices = vertices_to_transform[
                    index_vertices[i] : index_vertices[i + 1]
                ]
            field_resolution = np.array(
                self.vector_field_layer.resolution(self.field_mip)
            )
            bcube = get_bcube_from_vertices(
                vertices=current_batch_vertices,
                resolution=field_resolution,
                mip=self.field_mip,
            )
            field_data = self.vector_field_layer.read(
                bcube=bcube, mip=self.field_mip
            ).permute(2, 3, 0, 1)
            current_batch_vertices_to_mip = current_batch_vertices / field_resolution
            bcube_minpt = bcube.minpt(self.field_mip)
            field_indices = current_batch_vertices_to_mip.astype(np.int) - bcube_minpt
            vector_resolution = (
                self.vector_field_layer.resolution(0)
                * np.array(
                    [
                        2 ** (self.field_mip - self.vector_field_layer.data_mip),
                        2 ** (self.field_mip - self.vector_field_layer.data_mip),
                        1,
                    ]
                )
                if self.mip0_field
                else self.vector_field_layer.resolution(self.field_mip)
            )
            vectors_to_add = []
            corgie_logger.info(f"{field_data.shape}, {field_indices.max(0)}")
            for i in range(len(field_data.shape)-1):
                if field_indices.max(0)[i] >= field_data.shape[i]:
                    import pdb; pdb.set_trace()
            for cur_field_index in field_indices:
                vector_at_point = field_data[
                    cur_field_index[0], cur_field_index[1], cur_field_index[2]
                ]
                # Each vector is stored in [Y,X] format
                vectors_to_add.append(
                    [
                        int(vector_resolution[0] * vector_at_point[1].item()),
                        int(vector_resolution[1] * vector_at_point[0].item()),
                        0,
                    ]
                )
            vectors_to_add = np.array(vectors_to_add)
            current_batch_warped_vertices = current_batch_vertices + vectors_to_add
            new_vertices.append(current_batch_warped_vertices)

        new_vertices = np.concatenate(new_vertices)
        cf = CloudFiles(f"{self.dst_path}/intermediary_arrays/")
        cf.put(
            path=f"{self.skeleton_id_str}:{self.start_vertex_index}-{self.end_vertex_index}",
            content=pickle.dumps(new_vertices),
        )

    @property
    def number_vertices(self):
        return self.end_vertex_index - self.start_vertex_index


class TransformSkeletonsJob(scheduling.Job):
    def __init__(
        self,
        vector_field_layer,
        src_path,
        dst_path,
        field_mip,
        skeleton_ids,
        task_vertex_size=200,
        skeleton_length_file=None,
        mip0_field=False,
    ):
        self.vector_field_layer = vector_field_layer
        self.src_path = src_path
        self.dst_path = dst_path
        self.field_mip = field_mip
        if skeleton_ids is None:
            self.skeleton_ids = self.get_all_skeleton_ids()
        else:
            self.skeleton_ids = skeleton_ids
        self.task_vertex_size = task_vertex_size
        self.skeleton_length_file = skeleton_length_file
        self.mip0_field = mip0_field
        super().__init__()

    def task_generator(self):
        skeletons = self.get_skeletons(self.src_path)
        transform_vertex_tasks = []
        generate_new_skeleton_tasks = []
        for skeleton_id_str, skeleton in skeletons.items():
            number_vertices = len(skeleton.vertices)
            # Vector field chunks are typically chunked by 1 in z, so we process
            # the skeleton's vertices in z-order for maximum download efficiency.
            index_points = list(range(0, number_vertices, self.task_vertex_size))
            for i in range(len(index_points)):
                start_vertex_index = index_points[i]
                if i + 1 == len(index_points):
                    end_vertex_index = number_vertices
                else:
                    end_vertex_index = index_points[i + 1]
                transform_vertex_tasks.append(
                    TransformSkeletonVerticesTask(
                        vector_field_layer=self.vector_field_layer,
                        skeleton_id_str=skeleton_id_str,
                        src_path=self.src_path,
                        dst_path=self.dst_path,
                        field_mip=self.field_mip,
                        start_vertex_index=start_vertex_index,
                        end_vertex_index=end_vertex_index,
                        vertex_sort=True,
                        mip0_field=self.mip0_field,
                    )
                )
            generate_new_skeleton_tasks.append(
                GenerateNewSkeletonTask(
                    skeleton_id_str=skeleton_id_str,
                    src_path=self.src_path,
                    dst_path=self.dst_path,
                    task_vertex_size=self.task_vertex_size,
                    vertex_sort=True,
                )
            )
        corgie_logger.info(
            f"Yielding transform skeleton vertex tasks for skeletons in {self.src_path}"
        )
        yield transform_vertex_tasks
        yield scheduling.wait_until_done
        corgie_logger.info(f"Generating skeletons to {self.dst_path}")
        yield generate_new_skeleton_tasks
        yield scheduling.wait_until_done
        # TODO: Delete intermediary vertex files

        if self.skeleton_length_file is not None:
            new_skeletons = self.get_skeletons(self.dst_path)
            corgie_logger.info(
                f"Calculating skeleton lengths to {self.skeleton_length_file}"
            )
            with open(self.skeleton_length_file, "w") as f:
                f.write(
                    "Skeleton id, Original Skeleton Length (nm), New Skeleton Length (nm)\n"
                )
                for skeleton_id_str in skeletons:
                    original_skeleton = skeletons[skeleton_id_str]
                    new_skeleton = new_skeletons[skeleton_id_str]
                    verts = new_skeleton.vertices
                    vl = [verts[i] for i in range(verts.shape[0])]
                    f.write(
                        f"{skeleton_id_str},{int(original_skeleton.cable_length())},{int(new_skeleton.cable_length())}\n"
                    )

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
    "--vector_field_spec",
    "--v",
    nargs=1,
    type=str,
    required=True,
    multiple=True,
    help="Vector field from origin to dst (in push notation, or from dst to origin in pull notation)"
    + LAYER_HELP_STR,
)
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
@corgie_option("--field_mip", nargs=1, type=int, help="MIP of vector field")
@corgie_option("--ids", multiple=True, type=int, help="Ids to transform")
@corgie_option("--ids_filepath", type=str, help="File containing ids to transform")
@corgie_option(
    "--task_vertex_size",
    type=int,
    default=1000,
    help="How many vertices are transformed by each TransformSkeletonVertices task. Less means more parallelizable, but more intermediary files",
)
@corgie_option(
    "--calculate_skeleton_lengths",
    type=bool,
    default=True,
    help="If True, write file containing lengths of each skeleton.",
)
@corgie_option(
    "--mip0_field",
    nargs=1,
    type=bool,
    default=False,
    help="Set to True if field values are stored in MIP 0 pixels even though the field is not itself MIP 0.",
)
@click.pass_context
def transform_skeletons(
    ctx,
    vector_field_spec,
    src_folder,
    dst_folder,
    field_mip,
    ids,
    ids_filepath,
    task_vertex_size,
    calculate_skeleton_lengths,
    mip0_field,
):
    scheduler = ctx.obj["scheduler"]

    corgie_logger.debug("Setting up layers...")
    vf_stack = create_stack_from_spec(vector_field_spec, name="src", readonly=True)

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

    skeleton_length_file = None
    if calculate_skeleton_lengths:
        if not os.path.exists("skeleton_lengths"):
            os.makedirs("skeleton_lengths")
        skeleton_length_file = f"skeleton_lengths/skeleton_lengths_{int(time.time())}"

    vf_layer = vf_stack.get_layers_of_type("field")[0]
    transform_skeletons_job = TransformSkeletonsJob(
        vector_field_layer=vf_layer,
        src_path=src_folder,
        dst_path=dst_folder,
        field_mip=field_mip,
        skeleton_ids=skeleton_ids,
        task_vertex_size=task_vertex_size,
        skeleton_length_file=skeleton_length_file,
        mip0_field=mip0_field,
    )

    scheduler.register_job(
        transform_skeletons_job,
        job_name="Transforming skeletons in {}".format(src_folder),
    )

    scheduler.execute_until_completion()
    result_report = f"Transformed skeletons stored at {dst_folder}. "
    corgie_logger.info(result_report)
